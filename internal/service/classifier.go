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
	"unicode"

	"github.com/anthonynsimon/bild/effect"
	"github.com/disintegration/imaging"
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
	Confidence  float64       `json:"confidence"`
	Boxes       []BoundingBox `json:"boxes"`
	Angle       int           `json:"angle"`
	ScaleFactor float64       `json:"scale_factor"`
}

type Classifier struct{}

func NewClassifier() *Classifier {
	return &Classifier{}
}

const (
	confidenceThreshold = 0.65
	numWorkers          = 4
	medianRadius        = 1.0 // Radius for median blur (kernel size 3 = radius 1)

	// Image size thresholds for dynamic scaling
	minDimension    = 32                // Minimum dimension in pixels (skip if smaller)
	halfMegapixel   = 524_288           // 0.5 MP
	oneMegapixel    = 2 * halfMegapixel // 1 MP
	twoMegapixels   = 2 * oneMegapixel  // 2 MP
	threeMegapixels = 3 * oneMegapixel  // 3 MP
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

// rotateImage rotates an image by the specified angle in degrees using imaging library
func rotateImage(img image.Image, angleDeg int) image.Image {
	// Normalize angle to 0-359
	angleDeg = ((angleDeg % 360) + 360) % 360

	if angleDeg == 0 {
		return imaging.Clone(img)
	}

	// Use optimized 90-degree rotations when possible
	switch angleDeg {
	case 90:
		return imaging.Rotate90(img)
	case 180:
		return imaging.Rotate180(img)
	case 270:
		return imaging.Rotate270(img)
	}

	// General rotation for arbitrary angles with white background
	return imaging.Rotate(img, float64(-angleDeg), color.White)
}

// encodeImage encodes an image to bytes in the specified format
func encodeImage(img image.Image, format string) ([]byte, error) {
	var buf bytes.Buffer
	var err error

	switch format {
	case "png":
		err = png.Encode(&buf, img)
	default:
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
	}

	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// otsuThreshold calculates optimal threshold using Otsu's method
func otsuThreshold(img *image.Gray) uint8 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	totalPixels := w * h

	// Calculate histogram
	histogram := make([]int, 256)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			histogram[img.GrayAt(x, y).Y]++
		}
	}

	// Calculate total mean
	var sumTotal float64
	for i := 0; i < 256; i++ {
		sumTotal += float64(i) * float64(histogram[i])
	}

	var sumB float64
	var wB, wF int
	var maxVariance float64
	var threshold uint8

	for t := 0; t < 256; t++ {
		wB += histogram[t]
		if wB == 0 {
			continue
		}

		wF = totalPixels - wB
		if wF == 0 {
			break
		}

		sumB += float64(t) * float64(histogram[t])

		mB := sumB / float64(wB)
		mF := (sumTotal - sumB) / float64(wF)

		// Between-class variance
		variance := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)

		if variance > maxVariance {
			maxVariance = variance
			threshold = uint8(t)
		}
	}

	return threshold
}

// applyThreshold applies binary threshold to a grayscale image
func applyThreshold(img *image.Gray, threshold uint8) *image.Gray {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	newImg := image.NewGray(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if img.GrayAt(x, y).Y > threshold {
				newImg.SetGray(x, y, color.Gray{Y: 255})
			} else {
				newImg.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}

	return newImg
}

// preprocessImage applies preprocessing pipeline: scale, grayscale, median blur
// Returns (nil, 0) if image is too small to process
// Returns (processedImage, scaleFactor) on success
func preprocessImage(img image.Image) (*image.Gray, float64) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	pixels := w * h

	// Skip images with any dimension too small to process
	if w <= minDimension || h <= minDimension {
		return nil, 0
	}

	// Calculate target dimensions and scale factor based on megapixels
	var newW, newH int
	var scaleFactor float64
	switch {
	case pixels < halfMegapixel:
		// Less than 0.5 MP: scale 4x
		scaleFactor = 4.0
		newW, newH = w*4, h*4
	case pixels < oneMegapixel:
		// Less than 1 MP: scale 3x
		scaleFactor = 3.0
		newW, newH = w*3, h*3
	case pixels < twoMegapixels:
		// Less than 2 MP: scale 1.5x
		scaleFactor = 1.5
		newW, newH = w*3/2, h*3/2
	case pixels <= threeMegapixels:
		// 2-3 MP: no scaling
		scaleFactor = 1.0
		newW, newH = w, h
	default:
		// More than 3 MP: scale down to 3 MP
		scaleFactor = math.Sqrt(float64(threeMegapixels) / float64(pixels))
		newW = int(float64(w) * scaleFactor)
		newH = int(float64(h) * scaleFactor)
	}

	// Step 1: Scale image using cubic interpolation (CatmullRom)
	scaled := imaging.Resize(img, newW, newH, imaging.CatmullRom)

	// Step 2: Convert to grayscale
	gray := imaging.Grayscale(scaled)

	// Step 3: Apply median blur to reduce noise
	blurred := effect.Median(gray, medianRadius)

	// Convert to *image.Gray for OTSU
	grayImg := image.NewGray(blurred.Bounds())
	for y := blurred.Bounds().Min.Y; y < blurred.Bounds().Max.Y; y++ {
		for x := blurred.Bounds().Min.X; x < blurred.Bounds().Max.X; x++ {
			grayImg.Set(x-blurred.Bounds().Min.X, y-blurred.Bounds().Min.Y, blurred.At(x, y))
		}
	}

	return grayImg, scaleFactor

	// Step 4: Apply OTSU thresholding for binary image
	//threshold := otsuThreshold(grayImg)
	//binary := applyThreshold(grayImg, threshold)

	//return binary
}

// containsLetters checks if a string contains Latin or Cyrillic letters
func containsLetters(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Latin, r) || unicode.Is(unicode.Cyrillic, r) {
			return true
		}
	}
	return false
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
		if box.Confidence == 0 {
			continue
		}
		// Filter: only include boxes with Latin or Cyrillic letters
		if !containsLetters(box.Word) {
			continue
		}
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

	// Handle case where no valid boxes found after filtering
	if len(resultBoxes) == 0 {
		return &ClassifierResult{Confidence: 0, Boxes: []BoundingBox{}, Angle: 0}, nil
	}

	avgConfidence := totalConfidence / float64(len(resultBoxes))
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

	// Decode the original image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		// If we can't decode, try raw OCR
		result, ocrErr := c.detectTextSingle(imageData, lang)
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
		return &ClassifierResult{Confidence: 0, Boxes: []BoundingBox{}, Angle: 0, ScaleFactor: 0}, nil
	}

	// Encode preprocessed image for OCR
	preprocessedData, err := encodeImage(preprocessed, "png")
	if err != nil {
		return nil, err
	}

	// Phase 1: Try without rotation on preprocessed image
	result, err := c.detectTextSingle(preprocessedData, lang)
	if err != nil {
		return nil, err
	}

	result.Angle = 0
	result.ScaleFactor = scaleFactor
	if result.Confidence >= confidenceThreshold {
		return result, nil
	}

	// Phase 2: Try rotations at 90, 180, 270 and deviations (5, 10 degrees) from each main orientation
	// Rotations are applied to the preprocessed image
	angles := phase2Angles

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

					rotated := rotateImage(preprocessed, angle)
					data, encErr := encodeImage(rotated, "png")
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
					res.ScaleFactor = scaleFactor
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
