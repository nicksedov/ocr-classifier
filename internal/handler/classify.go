package handler

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"ocr-classifier/internal/service"
)

// ClassifyHandler handles image classification requests.
type ClassifyHandler struct {
	classifier *service.Classifier
}

// NewClassifyHandler creates a new ClassifyHandler instance.
func NewClassifyHandler() *ClassifyHandler {
	return &ClassifyHandler{
		classifier: service.NewClassifier(),
	}
}

// ErrorResponse represents an error response in JSON format.
type ErrorResponse struct {
	Error string `json:"error"`
}

// Classify processes image classification requests.
// It accepts POST requests with image/jpeg or image/png content type.
// Optional query parameters: confidence_threshold (0-1), min_token_count (positive integer).
func (h *ClassifyHandler) Classify(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		if err := json.NewEncoder(w).Encode(ErrorResponse{Error: "method not allowed, use POST"}); err != nil {
			fmt.Fprintf(w, `{"error":"method not allowed, use POST"}`)
		}
		return
	}

	// Check content type
	contentType := r.Header.Get("Content-Type")
	if contentType != "image/jpeg" && contentType != "image/png" {
		w.WriteHeader(http.StatusBadRequest)
		if err := json.NewEncoder(w).Encode(ErrorResponse{Error: "content-type must be image/jpeg or image/png"}); err != nil {
			fmt.Fprintf(w, `{"error":"content-type must be image/jpeg or image/png"}`)
		}
		return
	}

	// Read image data
	imageData, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		if err := json.NewEncoder(w).Encode(ErrorResponse{Error: "failed to read image data"}); err != nil {
			fmt.Fprintf(w, `{"error":"failed to read image data"}`)
		}
		return
	}
	defer r.Body.Close()

	if len(imageData) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		if err := json.NewEncoder(w).Encode(ErrorResponse{Error: "empty image data"}); err != nil {
			fmt.Fprintf(w, `{"error":"empty image data"}`)
		}
		return
	}

	// Parse query parameters with defaults
	decisionRule := service.GetDefaultDecisionRule()

	// Parse confidence_threshold from URL parameter
	if thresholdStr := r.URL.Query().Get("confidence_threshold"); thresholdStr != "" {
		if val, err := strconv.ParseFloat(thresholdStr, 64); err == nil && val > 0 && val <= 1 {
			decisionRule.MinConfidence = val
		}
	}

	// Parse min_token_count from URL parameter
	if tokenCountStr := r.URL.Query().Get("min_token_count"); tokenCountStr != "" {
		if val, err := strconv.Atoi(tokenCountStr); err == nil && val > 0 {
			decisionRule.MinTokenCount = val
		}
	}

	// Perform classification
	result, err := h.classifier.DetectText(imageData, decisionRule)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		if err := json.NewEncoder(w).Encode(ErrorResponse{Error: "failed to process image"}); err != nil {
			fmt.Fprintf(w, `{"error":"failed to process image"}`)
		}
		return
	}

	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(result); err != nil {
		fmt.Fprintf(w, `{"error":"failed to encode response"}`)
	}
}
