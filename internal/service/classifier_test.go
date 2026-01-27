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
		file       string
		confidence float64
		err        error
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

		// Classify
		res, err := classifier.DetectText(imageData, "eng")
		if err != nil {
			results = append(results, result{file: path, err: err})
			errorCount++
			return nil
		}

		relPath, _ := filepath.Rel(datasetPath, path)
		results = append(results, result{file: relPath, confidence: res.Confidence})
		processedCount++

		return nil
	})

	if err != nil {
		t.Fatalf("Error walking dataset directory: %v", err)
	}

	// Print results table
	fmt.Println()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("%-40s | %s\n", "File", "Confidence")
	fmt.Println(strings.Repeat("-", 60))

	for _, r := range results {
		if r.err != nil {
			fmt.Printf("%-40s | ERROR: %v\n", r.file, r.err)
		} else {
			fmt.Printf("%-40s | %.4f\n", r.file, r.confidence)
		}
	}

	fmt.Println(strings.Repeat("=", 60))
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
