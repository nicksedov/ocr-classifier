package service

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestBoundingBoxesEng(t *testing.T) {
	runBoundingBoxesTest(t, "eng", "2024318141.jpg")
}

func TestBoundingBoxesImage(t *testing.T) {
	runBoundingBoxesTest(t, "image", "clouded-sky.jpg")
}

func TestBoundingBoxesInscription(t *testing.T) {
	runBoundingBoxesTest(t, "inscription", "eng_althaus.jpg")
}

func TestBoundingBoxesRus(t *testing.T) {
	runBoundingBoxesTest(t, "rus", "passportscan02.png")
}

func runBoundingBoxesTest(t *testing.T, subfolder, filename string) {
	datasetPath := filepath.Join("..", "..", "test", "dataset", subfolder)
	imagePath := filepath.Join(datasetPath, filename)

	// Check if image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Fatalf("Test image does not exist: %s", imagePath)
	}

	// Read image
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("Failed to read image: %v", err)
	}

	// Decode image to get dimensions
	img, _, err := image.Decode(bytes.NewReader(imageData))
	var width, height int
	if err == nil {
		bounds := img.Bounds()
		width = bounds.Dx()
		height = bounds.Dy()
	}

	classifier := NewClassifier()
	result, err := classifier.DetectText(imageData)
	if err != nil {
		t.Fatalf("Failed to detect text: %v", err)
	}

	// Print bounding boxes info
	fmt.Println()
	fmt.Println(strings.Repeat("=", 90))
	fmt.Printf("Bounding Boxes for: %s/%s\n", subfolder, filename)
	fmt.Printf("Image Dimensions: %dx%d\n", width, height)
	fmt.Printf("Scale Factor: %.2f\n", result.ScaleFactor)
	fmt.Printf("Mean Confidence: %.4f\n", result.MeanConfidence)
	fmt.Printf("Weighted Confidence: %.4f\n", result.WeightedConfidence)
	fmt.Printf("Token Count: %d\n", result.TokenCount)
	fmt.Printf("Best Rotation Angle: %d\n", result.Angle)
	fmt.Printf("Total Boxes Found: %d\n", len(result.Boxes))
	fmt.Println(strings.Repeat("-", 90))
	fmt.Printf("%-5s | %-20s | %-8s | %-8s | %-8s | %-8s | %-10s\n",
		"#", "Word", "X", "Y", "Width", "Height", "Confidence")
	fmt.Println(strings.Repeat("-", 90))

	for i, box := range result.Boxes {
		fmt.Printf("%-5d | %-20s | %-8d | %-8d | %-8d | %-8d | %-10.4f\n",
			i+1, truncateString(box.Word, 20), box.X, box.Y, box.Width, box.Height, box.Confidence)
	}

	fmt.Println(strings.Repeat("=", 90))
	fmt.Println()
}

func truncateString(s string, maxLen int) string {
	if utf8.RuneCountInString(s) <= maxLen {
		return s
	}
	runes := []rune(s)
	return string(runes[:maxLen-3]) + "..."
}
