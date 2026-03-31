package service

import (
	"bytes"
	"fmt"
	"image"

	"github.com/otiai10/gosseract/v2"
)

// BoundingBox represents a detected text region with its position and confidence.
type BoundingBox struct {
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Word       string  `json:"word"`
	Confidence float64 `json:"confidence"`
}

// ClassifierResult contains the results of text detection on an image.
type ClassifierResult struct {
	MeanConfidence     float64       `json:"mean_confidence"`
	WeightedConfidence float64       `json:"weighted_confidence"`
	TokenCount         int           `json:"token_count"`
	Boxes              []BoundingBox `json:"boxes"`
	Angle              int           `json:"angle"`
	ScaleFactor        float64       `json:"scale_factor"`
	IsTextDocument     bool          `json:"is_text_document"`
}

// Classifier performs OCR-based text detection on images.
type Classifier struct{}

// NewClassifier creates a new Classifier instance.
func NewClassifier() *Classifier {
	return &Classifier{}
}

// detectTextSingle performs OCR on a single image using English and Russian.
// It returns the detected text boxes with confidence scores and token counts.
func (c *Classifier) detectTextSingle(imageData []byte) (*ClassifierResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	if err := client.SetLanguage("eng+rus"); err != nil {
		return nil, fmt.Errorf("failed to set language: %w", err)
	}

	if err := client.SetImageFromBytes(imageData); err != nil {
		return nil, fmt.Errorf("failed to set image: %w", err)
	}

	boxes, err := client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		return nil, fmt.Errorf("failed to get bounding boxes: %w", err)
	}

	return c.processBoundingBoxes(boxes)
}

// processBoundingBoxes processes raw OCR bounding boxes and calculates confidence metrics.
func (c *Classifier) processBoundingBoxes(boxes []gosseract.BoundingBox) (*ClassifierResult, error) {
	if len(boxes) == 0 {
		return &ClassifierResult{IsTextDocument: false}, nil
	}

	resultBoxes, totalTokens := c.filterAndConvertBoxes(boxes)

	if len(resultBoxes) == 0 {
		return &ClassifierResult{IsTextDocument: false}, nil
	}

	meanConfidence, weightedConfidence := c.calculateConfidenceMetrics(resultBoxes, totalTokens)

	return &ClassifierResult{
		MeanConfidence:     meanConfidence,
		WeightedConfidence: weightedConfidence,
		TokenCount:         totalTokens,
		Boxes:              resultBoxes,
		Angle:              0,
	}, nil
}

// filterAndConvertBoxes filters valid boxes and converts them to BoundingBox format.
// Boxes are excluded if they have zero confidence, no valid tokens,
// or confidence below MinBoxConfidence (postprocessing threshold).
// Returns the filtered boxes and total token count.
func (c *Classifier) filterAndConvertBoxes(boxes []gosseract.BoundingBox) ([]BoundingBox, int) {
	resultBoxes := make([]BoundingBox, 0, len(boxes))
	totalTokens := 0

	for _, box := range boxes {
		boxConfidence := float64(box.Confidence) / 100.0

		if boxConfidence < minBoxConfidence {
			continue
		}

		tokens := countTokens(box.Word)
		if tokens == 0 {
			continue
		}

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

	return resultBoxes, totalTokens
}

// calculateConfidenceMetrics calculates mean and weighted confidence from boxes.
func (c *Classifier) calculateConfidenceMetrics(boxes []BoundingBox, totalTokens int) (meanConfidence, weightedConfidence float64) {
	var totalConfidence float64
	var weightedConfidenceSum float64

	for _, box := range boxes {
		tokens := countTokens(box.Word)
		totalConfidence += box.Confidence * 100.0
		weightedConfidenceSum += box.Confidence * float64(tokens)
	}

	meanConfidence = totalConfidence / float64(len(boxes)) / 100.0
	meanConfidence = clampFloat64(meanConfidence, 0.0, 1.0)

	weightedConfidence = weightedConfidenceSum / float64(totalTokens)
	weightedConfidence = clampFloat64(weightedConfidence, 0.0, 1.0)

	return meanConfidence, weightedConfidence
}

// clampFloat64 clamps a float64 value to the range [min, max].
func clampFloat64(value, min, max float64) float64 {
	if value > max {
		return max
	}
	if value < min {
		return min
	}
	return value
}

// DetectText performs text detection on the provided image data.
// It applies preprocessing, attempts OCR at multiple rotation angles if needed,
// and evaluates the result against the provided decision rule.
func (c *Classifier) DetectText(imageData []byte, rule DecisionRule) (*ClassifierResult, error) {
	rule = c.normalizeDecisionRule(rule)

	img, err := c.decodeImage(imageData)
	if err != nil {
		return c.detectWithoutPreprocessing(imageData, rule)
	}

	return c.detectWithPreprocessing(img, rule)
}

// normalizeDecisionRule ensures valid decision rule parameters.
func (c *Classifier) normalizeDecisionRule(rule DecisionRule) DecisionRule {
	if rule.MinConfidence <= 0 || rule.MinConfidence > 1 {
		rule.MinConfidence = GetDefaultDecisionRule().MinConfidence
	}
	if rule.MinTokenCount <= 0 {
		rule.MinTokenCount = GetDefaultDecisionRule().MinTokenCount
	}
	return rule
}

// decodeImage attempts to decode image data.
func (c *Classifier) decodeImage(imageData []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(imageData))
	return img, err
}

// detectWithoutPreprocessing performs OCR without image preprocessing.
// Used when image decoding fails.
func (c *Classifier) detectWithoutPreprocessing(imageData []byte, rule DecisionRule) (*ClassifierResult, error) {
	result, err := c.detectTextSingle(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to detect text: %w", err)
	}
	result.Angle = 0
	result.ScaleFactor = 0
	result.IsTextDocument = EvaluateDecision(result.WeightedConfidence, result.TokenCount, rule)
	return result, nil
}

// detectWithPreprocessing performs OCR with image preprocessing and rotation detection.
func (c *Classifier) detectWithPreprocessing(img image.Image, rule DecisionRule) (*ClassifierResult, error) {
	preprocessed, scaleFactor := preprocessImage(img)
	if preprocessed == nil {
		return &ClassifierResult{IsTextDocument: false}, nil
	}

	preprocessedData, err := encodeImage(preprocessed, "png")
	if err != nil {
		return nil, fmt.Errorf("failed to encode preprocessed image: %w", err)
	}

	result, err := c.detectTextOriginal(preprocessedData, scaleFactor, rule)
	if err != nil {
		return nil, err
	}

	if result.IsTextDocument {
		return result, nil
	}

	return c.detectTextWithRotations(preprocessed, scaleFactor, result, rule)
}

// detectTextOriginal performs the first phase of detection without rotation.
func (c *Classifier) detectTextOriginal(imageData []byte, scaleFactor float64, rule DecisionRule) (*ClassifierResult, error) {
	result, err := c.detectTextSingle(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to detect text in phase 1: %w", err)
	}

	result.Angle = 0
	result.ScaleFactor = scaleFactor
	result.IsTextDocument = EvaluateDecision(result.WeightedConfidence, result.TokenCount, rule)

	return result, nil
}

// detectTextWithRotations attempts OCR at multiple rotation angles to find the best text detection.
// It uses candidate angles detected via Canny edge detection and Hough Line Transform.
func (c *Classifier) detectTextWithRotations(preprocessed *image.Gray, scaleFactor float64, phase1Result *ClassifierResult, rule DecisionRule) (*ClassifierResult, error) {
	candidateAngles := detectSkewAngle(preprocessed)

	if len(candidateAngles) == 0 {
		phase1Result.IsTextDocument = EvaluateDecision(phase1Result.WeightedConfidence, phase1Result.TokenCount, rule)
		return phase1Result, nil
	}

	return c.tryRotationAngles(preprocessed, scaleFactor, phase1Result, rule, candidateAngles)
}

// tryRotationAngles attempts OCR at each candidate angle and returns the best result.
func (c *Classifier) tryRotationAngles(preprocessed *image.Gray, scaleFactor float64, currentBest *ClassifierResult, rule DecisionRule, angles []int) (*ClassifierResult, error) {
	bestResult := currentBest

	for _, angle := range angles {
		if angle == 0 {
			continue
		}

		result, shouldReturn := c.trySingleRotation(preprocessed, scaleFactor, rule, angle)
		if shouldReturn && result != nil {
			return result, nil
		}

		if result != nil && result.WeightedConfidence > bestResult.WeightedConfidence {
			bestResult = result
		}
	}

	bestResult.IsTextDocument = EvaluateDecision(bestResult.WeightedConfidence, bestResult.TokenCount, rule)
	return bestResult, nil
}

// trySingleRotation attempts OCR at a single rotation angle.
// Returns the result, and a boolean indicating if early exit should occur.
func (c *Classifier) trySingleRotation(preprocessed *image.Gray, scaleFactor float64, rule DecisionRule, angle int) (*ClassifierResult, bool) {
	rotated := rotateImage(preprocessed, angle)
	data, err := encodeImage(rotated, "png")
	if err != nil {
		return nil, false
	}

	res, err := c.detectTextSingle(data)
	if err != nil {
		return nil, false
	}

	res.Angle = angle
	res.ScaleFactor = scaleFactor

	if EvaluateDecision(res.WeightedConfidence, res.TokenCount, rule) {
		res.IsTextDocument = true
		return res, true
	}

	return res, false
}
