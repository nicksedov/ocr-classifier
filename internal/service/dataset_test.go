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
)

const defaultWorkers = 6

type datasetResult struct {
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

func TestClassifierDatasetFull(t *testing.T) {
	runDatasetTest(t, "")
}

func runDatasetTest(t *testing.T, subfolder string) {
	datasetPath := filepath.Join("..", "..", "test", "dataset", subfolder)

	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		t.Skipf("Dataset directory does not exist: %s", datasetPath)
	}

	jobs := collectImageFiles(t, datasetPath)
	if len(jobs) == 0 {
		t.Fatal("No image files found in the dataset directory")
	}

	timestamp := time.Now()
	results := processDatasetJobs(jobs)

	sort.Slice(results, func(i, j int) bool {
		return results[i].file < results[j].file
	})

	errorCount := printDatasetReport(subfolder, results, len(jobs), timestamp)

	if errorCount > 0 {
		t.Fatalf("Failed to calculate confidence for %d file(s)", errorCount)
	}
}

type imageJob struct {
	path    string
	relPath string
}

func collectImageFiles(t *testing.T, datasetPath string) []imageJob {
	var jobs []imageJob
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
		jobs = append(jobs, imageJob{path: path, relPath: relPath})
		return nil
	})
	if err != nil {
		t.Fatalf("Error walking dataset directory: %v", err)
	}
	return jobs
}

func processDatasetJobs(jobs []imageJob) []datasetResult {
	jobsChan := make(chan imageJob, len(jobs))
	resultsChan := make(chan datasetResult, len(jobs))

	numWorkers := defaultWorkers
	if len(jobs) < numWorkers {
		numWorkers = len(jobs)
	}

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			classifier := NewClassifier()
			for j := range jobsChan {
				resultsChan <- processImage(classifier, j)
			}
		}()
	}

	for _, j := range jobs {
		jobsChan <- j
	}
	close(jobsChan)

	wg.Wait()
	close(resultsChan)

	var results []datasetResult
	for r := range resultsChan {
		results = append(results, r)
	}
	return results
}

func processImage(classifier *Classifier, j imageJob) datasetResult {
	imageData, err := os.ReadFile(j.path)
	if err != nil {
		return datasetResult{file: j.relPath, err: err}
	}

	width, height := getImageDimensions(imageData)

	res, err := classifier.DetectText(imageData)
	if err != nil {
		return datasetResult{file: j.relPath, err: err}
	}

	return datasetResult{
		file:         j.relPath,
		width:        width,
		height:       height,
		scaleFactor:  res.ScaleFactor,
		meanConf:     res.MeanConfidence,
		weightedConf: res.WeightedConfidence,
		tokenCount:   res.TokenCount,
		angle:        res.Angle,
	}
}

func getImageDimensions(imageData []byte) (int, int) {
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return 0, 0
	}
	bounds := img.Bounds()
	return bounds.Dx(), bounds.Dy()
}

func printDatasetReport(subfolder string, results []datasetResult, totalJobs int, startTime time.Time) int {
	fmt.Println()
	printDatasetHeader(subfolder)

	var processedCount, errorCount int
	for _, r := range results {
		printDatasetRow(r)
		if r.err != nil {
			errorCount++
		} else {
			processedCount++
		}
	}

	printDatasetSummary(totalJobs, processedCount, errorCount, startTime)
	return errorCount
}

func printDatasetHeader(subfolder string) {
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Dataset: %s\n", subfolder)
	fmt.Println(strings.Repeat("-", 120))
	fmt.Printf("%-30s | %-12s | %-6s | %-10s | %-10s | %-8s | %-6s\n",
		"File", "Dimensions", "Scale", "MeanConf", "WeightConf", "Tokens", "Angle")
	fmt.Println(strings.Repeat("-", 120))
}

func printDatasetRow(r datasetResult) {
	if r.err != nil {
		fmt.Printf("%-30s | ERROR: %v\n", r.file, r.err)
	} else {
		dims := fmt.Sprintf("%dx%d", r.width, r.height)
		fmt.Printf("%-30s | %-12s | %-6.2f | %-10.4f | %-10.4f | %-8d | %-6d\n",
			r.file, dims, r.scaleFactor, r.meanConf, r.weightedConf, r.tokenCount, r.angle)
	}
}

func printDatasetSummary(total, processed, errors int, startTime time.Time) {
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Total files: %d, Processed: %d, Errors: %d, Workers: %d\n", total, processed, errors, defaultWorkers)
	fmt.Printf("Processing time: %v\n", time.Since(startTime))
	fmt.Println()
}
