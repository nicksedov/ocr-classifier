package service

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

const defaultWorkers = 6

func TestClassifierDatasetEng(t *testing.T) {
	runDatasetTest(t, "eng")
}

func TestClassifierDatasetImage(t *testing.T) {
	runDatasetTest(t, "image")
}

func TestClassifierDatasetInscription(t *testing.T) {
	runDatasetTest(t, "inscription")
}

func TestClassifierDatasetRus(t *testing.T) {
	runDatasetTest(t, "rus")
}

func runDatasetTest(t *testing.T, subfolder string) {
	datasetPath := filepath.Join("..", "..", "test", "dataset", subfolder)

	// Check if dataset directory exists
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		t.Skipf("Dataset directory does not exist: %s", datasetPath)
	}

	type job struct {
		path    string
		relPath string
	}

	type result struct {
		file         string
		width        int
		height       int
		scaleFactor  float64
		meanConf     float64
		weightedConf float64
		tokenCount   int
		angle        int
		err          error
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
	timestamp := time.Now()
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

				// Decode image to get dimensions
				img, _, err := image.Decode(bytes.NewReader(imageData))
				var width, height int
				if err == nil {
					bounds := img.Bounds()
					width = bounds.Dx()
					height = bounds.Dy()
				}

				resEng, errEng := classifier.DetectText(imageData)
				if errEng != nil {
					resultsChan <- result{file: j.relPath, err: errEng}
					continue
				}

				resultsChan <- result{
					file:         j.relPath,
					width:        width,
					height:       height,
					scaleFactor:  resEng.ScaleFactor,
					meanConf:     resEng.MeanConfidence,
					weightedConf: resEng.WeightedConfidence,
					tokenCount:   resEng.TokenCount,
					angle:        resEng.Angle,
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
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Dataset: %s\n", subfolder)
	fmt.Println(strings.Repeat("-", 120))
	fmt.Printf("%-30s | %-12s | %-6s | %-10s | %-10s | %-8s | %-6s\n",
		"File", "Dimensions", "Scale", "MeanConf", "WeightConf", "Tokens", "Angle")
	fmt.Println(strings.Repeat("-", 120))

	for _, r := range results {
		if r.err != nil {
			fmt.Printf("%-30s | ERROR: %v\n", r.file, r.err)
		} else {
			dims := fmt.Sprintf("%dx%d", r.width, r.height)
			fmt.Printf("%-30s | %-12s | %-6.2f | %-10.4f | %-10.4f | %-8d | %-6d\n",
				r.file, dims, r.scaleFactor, r.meanConf, r.weightedConf, r.tokenCount, r.angle)
		}
	}

	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Total files: %d, Processed: %d, Errors: %d, Workers: %d\n", len(results), processedCount, errorCount, numWorkers)
	fmt.Printf("Processing time: %v\n", time.Since(timestamp))
	fmt.Println()

	if errorCount > 0 {
		t.Fatalf("Failed to calculate confidence for %d file(s)", errorCount)
	}
}

func truncateString(s string, maxLen int) string {
	if utf8.RuneCountInString(s) <= maxLen {
		return s
	}
	runes := []rune(s)
	return string(runes[:maxLen-3]) + "..."
}
