package service

import (
	"bytes"
	"context"
	"image"
	"sync"

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
}

type rotationResult struct {
	angle  int
	result *ClassifierResult
	err    error
}

type Classifier struct{}

func NewClassifier() *Classifier {
	return &Classifier{}
}

const (
	confidenceThreshold = 0.66 // Further OCR attempts will be stopped once the confidence threshold is reached
	numWorkers          = 4
)

// Phase 2 rotation angles: 90, 180, 270 and deviations of 5, 10 degrees from 0, 90, 180, 270
// Note: 0 is tested in Phase 1, so excluded here
var phase2Angles = []int{
	// Deviations from 0 (350, 355, 5, 10)
	350, 355, 5, 10,
	// 90 and deviations
	80, 85, 90, 95, 100,
	// 180 and deviations
	170, 175, 180, 185, 190,
	// 270 and deviations
	260, 265, 270, 275, 280,
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
	boxes, err := client.GetBoundingBoxes(gosseract.RIL_PARA)
	if err != nil {
		return nil, err
	}

	if len(boxes) == 0 {
		return &ClassifierResult{MeanConfidence: 0, WeightedConfidence: 0, TokenCount: 0, Boxes: []BoundingBox{}, Angle: 0}, nil
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
		return &ClassifierResult{MeanConfidence: 0, WeightedConfidence: 0, TokenCount: 0, Boxes: []BoundingBox{}, Angle: 0}, nil
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

func (c *Classifier) DetectText(imageData []byte) (*ClassifierResult, error) {
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
		return result, nil
	}

	// Preprocess: dynamic scaling, grayscale, median blur
	preprocessed, scaleFactor := preprocessImage(img)

	// If image is too small, return empty result
	if preprocessed == nil {
		return &ClassifierResult{}, nil
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
	if result.WeightedConfidence >= confidenceThreshold {
		return result, nil
	}

	// Phase 2: Try rotations at 90, 180, 270 and deviations (5, 10 degrees) from each main orientation
	// Rotations are applied to the preprocessed image
	return c.detectTextWithRotations(preprocessed, scaleFactor, result)
}

func (c *Classifier) detectTextWithRotations(preprocessed *image.Gray, scaleFactor float64, phase1Result *ClassifierResult) (*ClassifierResult, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	anglesChan := make(chan int, len(phase2Angles))
	resultsChan := make(chan rotationResult, len(phase2Angles))

	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.rotationWorker(ctx, anglesChan, resultsChan, preprocessed, scaleFactor)
		}()
	}

	// Send angles to workers
	go func() {
		defer close(anglesChan)
		for _, angle := range phase2Angles {
			select {
			case <-ctx.Done():
				return
			case anglesChan <- angle:
			}
		}
	}()

	// Wait for workers in a separate goroutine
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results and find the best one
	bestResult := phase1Result

	for rr := range resultsChan {
		if rr.err != nil {
			continue
		}
		if rr.result.WeightedConfidence > bestResult.WeightedConfidence {
			bestResult = rr.result
		}
		// Early exit if threshold reached
		if rr.result.WeightedConfidence >= confidenceThreshold {
			// Drain remaining results
			for range resultsChan {
			}
			return rr.result, nil
		}
	}

	return bestResult, nil
}

func (c *Classifier) rotationWorker(ctx context.Context, anglesChan <-chan int, resultsChan chan<- rotationResult, preprocessed *image.Gray, scaleFactor float64) {
	for {
		select {
		case <-ctx.Done():
			return
		case angle, ok := <-anglesChan:
			if !ok {
				return
			}

			rotated := rotateImage(preprocessed, angle)
			data, encErr := encodeImage(rotated, "png")
			if encErr != nil {
				resultsChan <- rotationResult{angle: angle, err: encErr}
				continue
			}

			res, detectErr := c.detectTextSingle(data)
			if detectErr != nil {
				resultsChan <- rotationResult{angle: angle, err: detectErr}
				continue
			}

			res.Angle = angle
			res.ScaleFactor = scaleFactor
			resultsChan <- rotationResult{angle: angle, result: res}

			// If confidence threshold reached, signal to stop
			if res.WeightedConfidence >= confidenceThreshold {
				return
			}
		}
	}
}
