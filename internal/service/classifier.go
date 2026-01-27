package service

import (
	"bytes"
	"image"
	"image/jpeg"
	"image/png"
	"sync"

	"github.com/otiai10/gosseract/v2"
)

// Supported languages for OCR
var SupportedLanguages = map[string]bool{
	"eng": true, // English
	"rus": true, // Russian
}

type BoundingBox struct {
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Word       string  `json:"word"`
	Confidence float64 `json:"confidence"`
}

type ClassifierResult struct {
	Confidence float64       `json:"confidence"`
	Boxes      []BoundingBox `json:"boxes"`
	Angle      int           `json:"angle"`
}

type Classifier struct{}

func NewClassifier() *Classifier {
	return &Classifier{}
}

// rotateImage rotates an image by the specified angle (90, 180, or 270 degrees)
func rotateImage(img image.Image, angle int) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	var newImg *image.RGBA
	switch angle {
	case 90:
		newImg = image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(h-1-y, x, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
	case 180:
		newImg = image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(w-1-x, h-1-y, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
	case 270:
		newImg = image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(y, w-1-x, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
	default:
		// 0 degrees - return copy
		newImg = image.NewRGBA(bounds)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(x, y, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
	}
	return newImg
}

// encodeImage encodes an image to bytes in the specified format
func encodeImage(img image.Image, format string) ([]byte, error) {
	var buf bytes.Buffer
	var err error

	switch format {
	case "png":
		err = png.Encode(&buf, img)
	default:
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95})
	}

	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// detectTextSingle performs OCR on a single image
func (c *Classifier) detectTextSingle(imageData []byte, lang string) (*ClassifierResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	if err := client.SetLanguage(lang); err != nil {
		return nil, err
	}

	if err := client.SetImageFromBytes(imageData); err != nil {
		return nil, err
	}

	boxes, err := client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		return nil, err
	}

	if len(boxes) == 0 {
		return &ClassifierResult{Confidence: 0, Boxes: []BoundingBox{}, Angle: 0}, nil
	}

	var totalConfidence float64
	resultBoxes := make([]BoundingBox, 0, len(boxes))

	for _, box := range boxes {
		totalConfidence += float64(box.Confidence)
		resultBoxes = append(resultBoxes, BoundingBox{
			X:          box.Box.Min.X,
			Y:          box.Box.Min.Y,
			Width:      box.Box.Max.X - box.Box.Min.X,
			Height:     box.Box.Max.Y - box.Box.Min.Y,
			Word:       box.Word,
			Confidence: float64(box.Confidence) / 100.0,
		})
	}

	avgConfidence := totalConfidence / float64(len(boxes))
	normalizedConfidence := avgConfidence / 100.0

	if normalizedConfidence > 1.0 {
		normalizedConfidence = 1.0
	}
	if normalizedConfidence < 0 {
		normalizedConfidence = 0
	}

	return &ClassifierResult{Confidence: normalizedConfidence, Boxes: resultBoxes, Angle: 0}, nil
}

func (c *Classifier) DetectText(imageData []byte, lang string) (*ClassifierResult, error) {
	// Set default language
	if lang == "" {
		lang = "eng"
	}

	// Decode the image to determine format and rotate
	img, format, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		// If we can't decode, fall back to single detection
		return c.detectTextSingle(imageData, lang)
	}

	// Rotation angles to try
	angles := []int{0, 90, 180, 270}

	type rotationResult struct {
		angle  int
		result *ClassifierResult
		err    error
	}

	resultsChan := make(chan rotationResult, 4)
	var wg sync.WaitGroup

	// Process all rotations in parallel
	for _, angle := range angles {
		wg.Add(1)
		go func(a int) {
			defer wg.Done()

			var data []byte
			var encErr error

			if a == 0 {
				data = imageData
			} else {
				rotated := rotateImage(img, a)
				data, encErr = encodeImage(rotated, format)
				if encErr != nil {
					resultsChan <- rotationResult{angle: a, err: encErr}
					return
				}
			}

			res, detectErr := c.detectTextSingle(data, lang)
			resultsChan <- rotationResult{angle: a, result: res, err: detectErr}
		}(angle)
	}

	wg.Wait()
	close(resultsChan)

	// Find the result with highest confidence
	var bestResult *ClassifierResult
	var bestAngle int

	for rr := range resultsChan {
		if rr.err != nil {
			continue
		}
		if bestResult == nil || rr.result.Confidence > bestResult.Confidence {
			bestResult = rr.result
			bestAngle = rr.angle
		}
	}

	if bestResult == nil {
		return &ClassifierResult{Confidence: 0, Boxes: []BoundingBox{}, Angle: 0}, nil
	}

	bestResult.Angle = bestAngle

	return bestResult, nil
}
