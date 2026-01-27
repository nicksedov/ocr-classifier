package main

import (
	"fmt"
	"log"
	"net/http"
	"ocr-classifier/internal/config"
	"ocr-classifier/internal/handler"
)

func main() {
	// 1. Load configuration
	cfg := config.Load()

	// 2. Initialize router
	mux := http.NewServeMux()

	// 3. Initialize handlers
	classifyHandler := handler.NewClassifyHandler()

	// 4. Register handlers
	mux.HandleFunc("/health", handler.HealthCheck)
	mux.HandleFunc("/classify", classifyHandler.Classify)

	// 5. Start HTTP server
	addr := ":" + cfg.Port
	fmt.Printf("Starting server on %s...\n", addr)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
