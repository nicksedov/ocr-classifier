package handler

import (
	"encoding/json"
	"io"
	"net/http"

	"ocr-classifier/internal/service"
)

type ClassifyHandler struct {
	classifier *service.Classifier
}

func NewClassifyHandler() *ClassifyHandler {
	return &ClassifyHandler{
		classifier: service.NewClassifier(),
	}
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func (h *ClassifyHandler) Classify(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "method not allowed, use POST"})
		return
	}

	// Check content type
	contentType := r.Header.Get("Content-Type")
	if contentType != "image/jpeg" && contentType != "image/png" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "content-type must be image/jpeg or image/png"})
		return
	}

	// Read image data
	imageData, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "failed to read image data"})
		return
	}
	defer r.Body.Close()

	if len(imageData) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "empty image data"})
		return
	}

	// Perform classification
	result, err := h.classifier.DetectText(imageData)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "failed to process image"})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(result)
}
