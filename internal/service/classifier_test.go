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

func TestClassifierDataset(t *testing.T) {
	datasetPath := filepath.Join("..", "..", "test", "dataset", "eng")

	// Check if dataset directory exists
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		t.Fatalf("Dataset directory does not exist: %s", datasetPath)
	}

	type job struct {
		path    string
		relPath string
	}

	type result struct {
		file           string
		width          int
		height         int
		confidenceEng  float64
		confidenceRus  float64
		angleEng       int
		angleRus       int
		scaleFactorEng float64
		scaleFactorRus float64
		err            error
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

				resEng, errEng := classifier.DetectText(imageData, "eng")
				if errEng != nil {
					resultsChan <- result{file: j.relPath, err: errEng}
					continue
				}

				resRus := &ClassifierResult{}
				if resEng.Confidence < confidenceThreshold {
					r, errRus := classifier.DetectText(imageData, "rus")
					if errRus != nil {
						resultsChan <- result{file: j.relPath, err: errRus}
						continue
					}
					resRus = r
				}

				resultsChan <- result{
					file:           j.relPath,
					width:          width,
					height:         height,
					confidenceEng:  resEng.Confidence,
					confidenceRus:  resRus.Confidence,
					angleEng:       resEng.Angle,
					angleRus:       resRus.Angle,
					scaleFactorEng: resEng.ScaleFactor,
					scaleFactorRus: resRus.ScaleFactor,
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
	fmt.Println(strings.Repeat("=", 135))
	fmt.Printf("%-30s | %-12s | %-6s | %-12s | %-6s | %-12s | %-6s\n",
		"File", "Dimensions", "Scale", "English", "Angle", "Russian", "Angle")
	fmt.Println(strings.Repeat("-", 135))

	for _, r := range results {
		if r.err != nil {
			fmt.Printf("%-30s | ERROR: %v\n", r.file, r.err)
		} else {
			dims := fmt.Sprintf("%dx%d", r.width, r.height)
			fmt.Printf("%-30s | %-12s | %-6.2f | %-12.4f | %-6d | %-12.4f | %-6d\n",
				r.file, dims, r.scaleFactorEng, r.confidenceEng, r.angleEng, r.confidenceRus, r.angleRus)
		}
	}

	fmt.Println(strings.Repeat("=", 135))
	fmt.Printf("Total files: %d, Processed: %d, Errors: %d, Workers: %d\n", len(results), processedCount, errorCount, numWorkers)
	fmt.Printf("Processing time: %v\n", time.Since(timestamp))
	fmt.Println()

	if errorCount > 0 {
		t.Fatalf("Failed to calculate confidence for %d file(s)", errorCount)
	}
}

func TestBoundingBoxesEng(t *testing.T) {
	runBoundingBoxesTest(t, "eng", "2024318141.jpg", "eng")
}

func TestBoundingBoxesImage(t *testing.T) {
	runBoundingBoxesTest(t, "image", "clouded-sky.jpg", "rus")
}

func TestBoundingBoxesInscription(t *testing.T) {
	runBoundingBoxesTest(t, "inscription", "eng_althaus.jpg", "eng")
}

func TestBoundingBoxesRus(t *testing.T) {
	runBoundingBoxesTest(t, "rus", "contract01.jpg", "rus")
}

func runBoundingBoxesTest(t *testing.T, subfolder, filename, lang string) {
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
	result, err := classifier.DetectText(imageData, lang)
	if err != nil {
		t.Fatalf("Failed to detect text: %v", err)
	}

	// Print bounding boxes info
	fmt.Println()
	fmt.Println(strings.Repeat("=", 90))
	fmt.Printf("Bounding Boxes for: %s/%s\n", subfolder, filename)
	fmt.Printf("Image Dimensions: %dx%d\n", width, height)
	fmt.Printf("Scale Factor: %.2f\n", result.ScaleFactor)
	fmt.Printf("Overall Confidence: %.4f\n", result.Confidence)
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
