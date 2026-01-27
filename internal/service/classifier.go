package service

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
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

const (
	confidenceThreshold = 0.65
	numWorkers          = 3
	angleStep           = 15
)

// rotateImage rotates an image by the specified angle in degrees
func rotateImage(img image.Image, angleDeg int) image.Image {
	// Normalize angle to 0-359
	angleDeg = ((angleDeg % 360) + 360) % 360

	if angleDeg == 0 {
		bounds := img.Bounds()
		newImg := image.NewRGBA(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				newImg.Set(x-bounds.Min.X, y-bounds.Min.Y, img.At(x, y))
			}
		}
		return newImg
	}

	// For 90, 180, 270 use optimized rotation
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	switch angleDeg {
	case 90:
		newImg := image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(h-1-y, x, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
		return newImg
	case 180:
		newImg := image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(w-1-x, h-1-y, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
		return newImg
	case 270:
		newImg := image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				newImg.Set(y, w-1-x, img.At(x+bounds.Min.X, y+bounds.Min.Y))
			}
		}
		return newImg
	}

	// General rotation for arbitrary angles
	angleRad := float64(angleDeg) * math.Pi / 180.0
	sin, cos := math.Sin(angleRad), math.Cos(angleRad)

	// Calculate new image dimensions
	newW := int(math.Ceil(math.Abs(float64(w)*cos) + math.Abs(float64(h)*sin)))
	newH := int(math.Ceil(math.Abs(float64(w)*sin) + math.Abs(float64(h)*cos)))

	newImg := image.NewRGBA(image.Rect(0, 0, newW, newH))

	// Fill with white background
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			newImg.Set(x, y, color.White)
		}
	}

	// Center points
	cx, cy := float64(w)/2, float64(h)/2
	ncx, ncy := float64(newW)/2, float64(newH)/2

	// Rotate each pixel
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			// Map back to original coordinates
			dx, dy := float64(x)-ncx, float64(y)-ncy
			srcX := cos*dx + sin*dy + cx
			srcY := -sin*dx + cos*dy + cy

			// Check bounds and copy pixel
			ix, iy := int(srcX), int(srcY)
			if ix >= 0 && ix < w && iy >= 0 && iy < h {
				newImg.Set(x, y, img.At(ix+bounds.Min.X, iy+bounds.Min.Y))
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

	// Phase 1: Try without rotation
	result, err := c.detectTextSingle(imageData, lang)
	if err != nil {
		return nil, err
	}

	result.Angle = 0
	if result.Confidence >= confidenceThreshold {
		return result, nil
	}

	// Decode the image for rotation in Phase 2
	img, format, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		// If we can't decode, return Phase 1 result
		return result, nil
	}

	// Phase 2: Try rotations with 15-degree increments using 3 workers
	// Generate angles: 15, 30, 45, ... 345 (excluding 0 which was done in Phase 1)
	var angles []int
	for a := angleStep; a < 360; a += angleStep {
		angles = append(angles, a)
	}

	type rotationResult struct {
		angle  int
		result *ClassifierResult
		err    error
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	anglesChan := make(chan int, len(angles))
	resultsChan := make(chan rotationResult, len(angles))

	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					return
				case angle, ok := <-anglesChan:
					if !ok {
						return
					}

					rotated := rotateImage(img, angle)
					data, encErr := encodeImage(rotated, format)
					if encErr != nil {
						resultsChan <- rotationResult{angle: angle, err: encErr}
						continue
					}

					res, detectErr := c.detectTextSingle(data, lang)
					if detectErr != nil {
						resultsChan <- rotationResult{angle: angle, err: detectErr}
						continue
					}

					res.Angle = angle
					resultsChan <- rotationResult{angle: angle, result: res}

					// If confidence threshold reached, signal to stop
					if res.Confidence >= confidenceThreshold {
						cancel()
						return
					}
				}
			}
		}()
	}

	// Send angles to workers
	go func() {
		for _, angle := range angles {
			select {
			case <-ctx.Done():
				break
			case anglesChan <- angle:
			}
		}
		close(anglesChan)
	}()

	// Wait for workers in a separate goroutine
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results and find the best one
	bestResult := result // Start with Phase 1 result

	for rr := range resultsChan {
		if rr.err != nil {
			continue
		}
		if rr.result.Confidence > bestResult.Confidence {
			bestResult = rr.result
		}
		// Early exit if threshold reached
		if rr.result.Confidence >= confidenceThreshold {
			// Drain remaining results
			for range resultsChan {
			}
			return rr.result, nil
		}
	}

	return bestResult, nil
}
