package service

import (
	"github.com/otiai10/gosseract/v2"
)

// Supported languages for OCR
var SupportedLanguages = map[string]bool{
	"eng": true, // English
	"rus": true, // Russian
	"deu": true, // German
}

type ClassifierResult struct {
	Confidence float64 `json:"confidence"`
}

type Classifier struct{}

func NewClassifier() *Classifier {
	return &Classifier{}
}

func (c *Classifier) DetectText(imageData []byte, lang string) (*ClassifierResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	// Set language for OCR (default to English if not specified)
	if lang == "" {
		lang = "eng"
	}
	if err := client.SetLanguage(lang); err != nil {
		return nil, err
	}

	if err := client.SetImageFromBytes(imageData); err != nil {
		return nil, err
	}

	// Get text and boxes with confidence
	boxes, err := client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		return nil, err
	}

	if len(boxes) == 0 {
		return &ClassifierResult{Confidence: 0}, nil
	}

	// Calculate average confidence from all detected words
	var totalConfidence float64
	for _, box := range boxes {
		totalConfidence += float64(box.Confidence)
	}

	avgConfidence := totalConfidence / float64(len(boxes))
	// Normalize confidence to 0-1 range (Tesseract returns 0-100)
	normalizedConfidence := avgConfidence / 100.0

	if normalizedConfidence > 1.0 {
		normalizedConfidence = 1.0
	}
	if normalizedConfidence < 0 {
		normalizedConfidence = 0
	}

	return &ClassifierResult{Confidence: normalizedConfidence}, nil
}
