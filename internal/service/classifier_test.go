package service

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

const defaultWorkers = 10

func TestClassifierDataset(t *testing.T) {
	datasetPath := filepath.Join("..", "..", "test", "dataset")

	// Check if dataset directory exists
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		t.Fatalf("Dataset directory does not exist: %s", datasetPath)
	}

	type job struct {
		path    string
		relPath string
	}

	type result struct {
		file          string
		confidenceEng float64
		confidenceRus float64
		err           error
	}

	// Collect all image files
	var jobs []job
	err := filepath.Walk(datasetPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			return nil
		}
		relPath, _ := filepath.Rel(datasetPath, path)
		jobs = append(jobs, job{path: path, relPath: relPath})
		return nil
	})

	if err != nil {
		t.Fatalf("Error walking dataset directory: %v", err)
	}

	if len(jobs) == 0 {
		t.Fatal("No image files found in the dataset directory")
	}

	// Channels for worker pool
	jobsChan := make(chan job, len(jobs))
	resultsChan := make(chan result, len(jobs))

	// Start workers
	var wg sync.WaitGroup
	numWorkers := defaultWorkers
	if len(jobs) < numWorkers {
		numWorkers = len(jobs)
	}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			classifier := NewClassifier()

			for j := range jobsChan {
				imageData, err := os.ReadFile(j.path)
				if err != nil {
					resultsChan <- result{file: j.relPath, err: err}
					continue
				}

				resEng, errEng := classifier.DetectText(imageData, "eng")
				if errEng != nil {
					resultsChan <- result{file: j.relPath, err: errEng}
					continue
				}

				resRus, errRus := classifier.DetectText(imageData, "rus")
				if errRus != nil {
					resultsChan <- result{file: j.relPath, err: errRus}
					continue
				}

				resultsChan <- result{
					file:          j.relPath,
					confidenceEng: resEng.Confidence,
					confidenceRus: resRus.Confidence,
				}
			}
		}()
	}

	// Send jobs to workers
	for _, j := range jobs {
		jobsChan <- j
	}
	close(jobsChan)

	// Wait for all workers to finish
	wg.Wait()
	close(resultsChan)

	// Collect results
	var results []result
	var processedCount, errorCount int
	for r := range resultsChan {
		results = append(results, r)
		if r.err != nil {
			errorCount++
		} else {
			processedCount++
		}
	}

	// Sort results by filename for consistent output
	sort.Slice(results, func(i, j int) bool {
		return results[i].file < results[j].file
	})

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
	fmt.Printf("Total files: %d, Processed: %d, Errors: %d, Workers: %d\n", len(results), processedCount, errorCount, numWorkers)
	fmt.Println()

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
