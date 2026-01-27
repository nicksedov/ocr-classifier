package service

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClassifierDataset(t *testing.T) {
	datasetPath := filepath.Join("..", "..", "test", "dataset")

	// Check if dataset directory exists
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		t.Fatalf("Dataset directory does not exist: %s", datasetPath)
	}

	classifier := NewClassifier()

	type result struct {
		file          string
		confidenceEng float64
		confidenceRus float64
		err           error
	}

	var results []result
	var processedCount int
	var errorCount int

	// Walk through all files in the dataset directory
	err := filepath.Walk(datasetPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Process only image files
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			return nil
		}

		// Read file
		imageData, err := os.ReadFile(path)
		if err != nil {
			results = append(results, result{file: path, err: err})
			errorCount++
			return nil
		}

		relPath, _ := filepath.Rel(datasetPath, path)

		// Classify with English
		resEng, errEng := classifier.DetectText(imageData, "eng")
		if errEng != nil {
			results = append(results, result{file: relPath, err: errEng})
			errorCount++
			return nil
		}

		// Classify with Russian
		resRus, errRus := classifier.DetectText(imageData, "rus")
		if errRus != nil {
			results = append(results, result{file: relPath, err: errRus})
			errorCount++
			return nil
		}

		results = append(results, result{
			file:          relPath,
			confidenceEng: resEng.Confidence,
			confidenceRus: resRus.Confidence,
		})
		processedCount++

		return nil
	})

	if err != nil {
		t.Fatalf("Error walking dataset directory: %v", err)
	}

	// Print results table
	fmt.Println()
	fmt.Println(strings.Repeat("=", 75))
	fmt.Printf("%-40s | %-12s | %-12s\n", "File", "English", "Russian")
	fmt.Println(strings.Repeat("-", 75))

	for _, r := range results {
		if r.err != nil {
			fmt.Printf("%-40s | ERROR: %v\n", r.file, r.err)
		} else {
			fmt.Printf("%-40s | %-12.4f | %-12.4f\n", r.file, r.confidenceEng, r.confidenceRus)
		}
	}

	fmt.Println(strings.Repeat("=", 75))
	fmt.Printf("Total files: %d, Processed: %d, Errors: %d\n", len(results), processedCount, errorCount)
	fmt.Println()

	// Test fails if there were any errors or no files were processed
	if len(results) == 0 {
		t.Fatal("No image files found in the dataset directory")
	}

	if errorCount > 0 {
		t.Fatalf("Failed to calculate confidence for %d file(s)", errorCount)
	}
}

func TestBoundingBoxesOutput(t *testing.T) {
	datasetPath := filepath.Join("..", "..", "test", "dataset")
	testImage := "image.jpg"
	imagePath := filepath.Join(datasetPath, testImage)

	// Check if image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Fatalf("Test image does not exist: %s", imagePath)
	}

	// Read image
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("Failed to read image: %v", err)
	}

	classifier := NewClassifier()
	result, err := classifier.DetectText(imageData, "eng")
	if err != nil {
		t.Fatalf("Failed to detect text: %v", err)
	}

	// Print bounding boxes info
	fmt.Println()
	fmt.Println(strings.Repeat("=", 90))
	fmt.Printf("Bounding Boxes for: %s\n", testImage)
	fmt.Printf("Overall Confidence: %.4f\n", result.Confidence)
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
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
