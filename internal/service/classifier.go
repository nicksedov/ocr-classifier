package service

import (
	"bytes"
	"image"

	"github.com/otiai10/gosseract/v2"
)

type BoundingBox struct {
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Word       string  `json:"word"`
	Confidence float64 `json:"confidence"`
}

type ClassifierResult struct {
	MeanConfidence     float64       `json:"mean_confidence"`
	WeightedConfidence float64       `json:"weighted_confidence"`
	TokenCount         int           `json:"token_count"`
	Boxes              []BoundingBox `json:"boxes"`
	Angle              int           `json:"angle"`
	ScaleFactor        float64       `json:"scale_factor"`
	IsTextDocument     bool          `json:"is_text_document"`
}

type Classifier struct{}

func NewClassifier() *Classifier {
	return &Classifier{}
}

// detectTextSingle performs OCR on a single image using English and Russian
func (c *Classifier) detectTextSingle(imageData []byte) (*ClassifierResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	if err := client.SetLanguage("eng+rus"); err != nil {
		return nil, err
	}

	if err := client.SetImageFromBytes(imageData); err != nil {
		return nil, err
	}

	// Constants for GetBoundingBoxes:
	// RIL_SYMBOL - Individual characters
	// RIL_WORD - Individual words
	// RIL_TEXTLINE - Text lines
	// RIL_PARA - Paragraphs
	// RIL_BLOCK - Text blocks (largest regions)
	boxes, err := client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		return nil, err
	}

	if len(boxes) == 0 {
		return &ClassifierResult{IsTextDocument : false}, nil
	}

	var totalConfidence float64
	var weightedConfidenceSum float64
	var totalTokens int
	resultBoxes := make([]BoundingBox, 0, len(boxes))

	for _, box := range boxes {
		if box.Confidence == 0 {
			continue
		}
		// Count tokens (Latin/Cyrillic letters and digits)
		tokens := countTokens(box.Word)
		// Filter: only include boxes with at least one token
		if tokens == 0 {
			continue
		}

		boxConfidence := float64(box.Confidence) / 100.0
		totalConfidence += float64(box.Confidence)
		weightedConfidenceSum += boxConfidence * float64(tokens)
		totalTokens += tokens

		resultBoxes = append(resultBoxes, BoundingBox{
			X:          box.Box.Min.X,
			Y:          box.Box.Min.Y,
			Width:      box.Box.Max.X - box.Box.Min.X,
			Height:     box.Box.Max.Y - box.Box.Min.Y,
			Word:       box.Word,
			Confidence: boxConfidence,
		})
	}

	// Handle case where no valid boxes found after filtering
	if len(resultBoxes) == 0 {
		return &ClassifierResult{IsTextDocument : false}, nil
	}

	// Calculate mean confidence
	meanConfidence := totalConfidence / float64(len(resultBoxes)) / 100.0
	if meanConfidence > 1.0 {
		meanConfidence = 1.0
	}
	if meanConfidence < 0 {
		meanConfidence = 0
	}

	// Calculate weighted confidence
	weightedConfidence := weightedConfidenceSum / float64(totalTokens)
	if weightedConfidence > 1.0 {
		weightedConfidence = 1.0
	}
	if weightedConfidence < 0 {
		weightedConfidence = 0
	}

	return &ClassifierResult{
		MeanConfidence:     meanConfidence,
		WeightedConfidence: weightedConfidence,
		TokenCount:         totalTokens,
		Boxes:              resultBoxes,
		Angle:              0,
	}, nil
}

func (c *Classifier) DetectText(imageData []byte, rule DecisionRule) (*ClassifierResult, error) {
	// Use default values if not specified or invalid
	if rule.MinConfidence <= 0 || rule.MinConfidence > 1 {
		rule.MinConfidence = GetDefaultDecisionRule().MinConfidence
	}
	if rule.MinTokenCount <= 0 {
		rule.MinTokenCount = GetDefaultDecisionRule().MinTokenCount
	}

	// Decode the original image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		// If we can't decode, try raw OCR
		result, ocrErr := c.detectTextSingle(imageData)
		if ocrErr != nil {
			return nil, ocrErr
		}
		result.Angle = 0
		result.ScaleFactor = 0
		result.IsTextDocument = EvaluateDecision(result.WeightedConfidence, result.TokenCount, rule)
		return result, nil
	}

	// Preprocess: dynamic scaling, grayscale, median blur
	preprocessed, scaleFactor := preprocessImage(img)

	// If image is too small, return empty result
	if preprocessed == nil {
		result := &ClassifierResult{}
		result.IsTextDocument = false
		return result, nil
	}

	// Encode preprocessed image for OCR
	preprocessedData, err := encodeImage(preprocessed, "png")
	if err != nil {
		return nil, err
	}

	// Phase 1: Try without rotation on preprocessed image
	result, err := c.detectTextSingle(preprocessedData)
	if err != nil {
		return nil, err
	}

	result.Angle = 0
	result.ScaleFactor = scaleFactor

	result.IsTextDocument = EvaluateDecision(result.WeightedConfidence, result.TokenCount, rule)
	if result.IsTextDocument {
		return result, nil
	}

	// Phase 2: Detect rotation angle using Canny edge detection and Hough Line Transform,
	// then try OCR at candidate orientations derived from the detected angle.
	return c.detectTextWithRotations(preprocessed, scaleFactor, result, rule)
}

func (c *Classifier) detectTextWithRotations(preprocessed *image.Gray, scaleFactor float64, phase1Result *ClassifierResult, rule DecisionRule) (*ClassifierResult, error) {
	// Detect candidate rotation angles using Canny edge detection and Hough Line Transform
	candidateAngles := detectSkewAngle(preprocessed)

	if len(candidateAngles) == 0 {
		phase1Result.IsTextDocument = EvaluateDecision(phase1Result.WeightedConfidence, phase1Result.TokenCount, rule)
		return phase1Result, nil
	}

	bestResult := phase1Result

	for _, angle := range candidateAngles {
		if angle == 0 {
			continue // Skip zero angle, already processed in phase 1
		}
		rotated := rotateImage(preprocessed, angle)
		data, err := encodeImage(rotated, "png")
		if err != nil {
			continue
		}

		res, err := c.detectTextSingle(data)
		if err != nil {
			continue
		}

		res.Angle = angle
		res.ScaleFactor = scaleFactor

		// Early exit if decision rule criteria are met
		if EvaluateDecision(res.WeightedConfidence, res.TokenCount, rule) {
			res.IsTextDocument = true
			return res, nil
		}
        // Otherwise preserve the best result so far, then go on with the next candidate
		if res.WeightedConfidence > bestResult.WeightedConfidence {
			bestResult = res
		}
	}

	// Final verdict evaluation for the best result
	bestResult.IsTextDocument = EvaluateDecision(bestResult.WeightedConfidence, bestResult.TokenCount, rule)
	return bestResult, nil
}
