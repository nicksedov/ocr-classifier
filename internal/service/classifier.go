package service

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"sort"
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
	scaleFactor         = 3 // Scale factor for preprocessing
	medianKernelSize    = 3 // Kernel size for median blur
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

// cubicWeight calculates the weight for bicubic interpolation
func cubicWeight(x float64) float64 {
	x = math.Abs(x)
	if x <= 1 {
		return 1.5*x*x*x - 2.5*x*x + 1
	} else if x < 2 {
		return -0.5*x*x*x + 2.5*x*x - 4*x + 2
	}
	return 0
}

// scaleImageBicubic scales an image using bicubic interpolation
func scaleImageBicubic(img image.Image, factor int) *image.Gray {
	bounds := img.Bounds()
	oldW, oldH := bounds.Dx(), bounds.Dy()
	newW, newH := oldW*factor, oldH*factor

	newImg := image.NewGray(image.Rect(0, 0, newW, newH))

	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			// Map to source coordinates
			srcX := float64(x) / float64(factor)
			srcY := float64(y) / float64(factor)

			// Get integer and fractional parts
			x0 := int(math.Floor(srcX))
			y0 := int(math.Floor(srcY))
			dx := srcX - float64(x0)
			dy := srcY - float64(y0)

			// Bicubic interpolation using 4x4 neighborhood
			var value float64
			var weightSum float64

			for j := -1; j <= 2; j++ {
				for i := -1; i <= 2; i++ {
					px := x0 + i
					py := y0 + j

					// Clamp to bounds
					if px < 0 {
						px = 0
					}
					if px >= oldW {
						px = oldW - 1
					}
					if py < 0 {
						py = 0
					}
					if py >= oldH {
						py = oldH - 1
					}

					// Get grayscale value from source
					r, g, b, _ := img.At(px+bounds.Min.X, py+bounds.Min.Y).RGBA()
					gray := float64(0.299*float64(r)+0.587*float64(g)+0.114*float64(b)) / 256.0

					// Calculate weight
					weight := cubicWeight(float64(i)-dx) * cubicWeight(float64(j)-dy)
					value += gray * weight
					weightSum += weight
				}
			}

			if weightSum > 0 {
				value /= weightSum
			}

			// Clamp to valid range
			if value < 0 {
				value = 0
			}
			if value > 255 {
				value = 255
			}

			newImg.SetGray(x, y, color.Gray{Y: uint8(value)})
		}
	}

	return newImg
}

// medianBlur applies median blur filter to a grayscale image
func medianBlur(img *image.Gray, kernelSize int) *image.Gray {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	newImg := image.NewGray(image.Rect(0, 0, w, h))

	radius := kernelSize / 2
	windowSize := kernelSize * kernelSize
	window := make([]uint8, windowSize)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Collect values in the kernel window
			idx := 0
			for ky := -radius; ky <= radius; ky++ {
				for kx := -radius; kx <= radius; kx++ {
					px, py := x+kx, y+ky

					// Clamp to bounds
					if px < 0 {
						px = 0
					}
					if px >= w {
						px = w - 1
					}
					if py < 0 {
						py = 0
					}
					if py >= h {
						py = h - 1
					}

					window[idx] = img.GrayAt(px, py).Y
					idx++
				}
			}

			// Sort and get median
			sort.Slice(window, func(i, j int) bool {
				return window[i] < window[j]
			})
			median := window[windowSize/2]

			newImg.SetGray(x, y, color.Gray{Y: median})
		}
	}

	return newImg
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

// preprocessImage applies preprocessing pipeline: scale, grayscale, median blur, OTSU threshold
func preprocessImage(img image.Image) *image.Gray {
	// Step 1: Scale image using bicubic interpolation (also converts to grayscale)
	scaled := scaleImageBicubic(img, scaleFactor)

	// Step 2: Apply median blur to reduce noise
	blurred := medianBlur(scaled, medianKernelSize)

	// Step 3: Apply OTSU thresholding for binary image
	threshold := otsuThreshold(blurred)
	binary := applyThreshold(blurred, threshold)

	return binary
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

	// Decode the original image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		// If we can't decode, try raw OCR
		result, ocrErr := c.detectTextSingle(imageData, lang)
		if ocrErr != nil {
			return nil, ocrErr
		}
		result.Angle = 0
		return result, nil
	}

	// Preprocess: scale 3x with bicubic interpolation, grayscale, median blur, OTSU threshold
	preprocessed := preprocessImage(img)

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
