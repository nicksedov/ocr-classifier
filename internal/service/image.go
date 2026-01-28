package service

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"

	"github.com/anthonynsimon/bild/effect"
	"github.com/disintegration/imaging"
)

const (
	medianRadius = 1.0 // Radius for median blur (kernel size 3 = radius 1)

	// Image size thresholds for dynamic scaling
	minDimension    = 32                // Minimum dimension in pixels (skip if smaller)
	halfMegapixel   = 524_288           // 0.5 MP
	oneMegapixel    = 2 * halfMegapixel // 1 MP
	twoMegapixels   = 2 * oneMegapixel  // 2 MP
	threeMegapixels = 3 * oneMegapixel  // 3 MP
)

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
	newW, newH, scaleFactor := calculateScaleDimensions(w, h, pixels)

	// Step 1: Scale image using cubic interpolation (CatmullRom)
	scaled := imaging.Resize(img, newW, newH, imaging.CatmullRom)

	// Step 2: Convert to grayscale
	gray := imaging.Grayscale(scaled)

	// Step 3: Apply median blur to reduce noise
	blurred := effect.Median(gray, medianRadius)

	// Convert to *image.Gray
	grayImg := convertToGray(blurred)

	return grayImg, scaleFactor
}

// calculateScaleDimensions determines target dimensions based on megapixel thresholds
func calculateScaleDimensions(w, h, pixels int) (newW, newH int, scaleFactor float64) {
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
	return
}

// convertToGray converts any image to *image.Gray
func convertToGray(img image.Image) *image.Gray {
	bounds := img.Bounds()
	grayImg := image.NewGray(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			grayImg.Set(x-bounds.Min.X, y-bounds.Min.Y, img.At(x, y))
		}
	}
	return grayImg
}
