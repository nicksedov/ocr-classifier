package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	// Root API prefix: /ocr-classifier/api
	mux.HandleFunc("/ocr-classifier/api/health", handler.HealthCheck)
	mux.HandleFunc("/ocr-classifier/api/v1/classify", classifyHandler.Classify)

	// 5. Create HTTP server
	addr := ":" + cfg.Port
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	// 6. Start server in goroutine
	go func() {
		fmt.Printf("Starting server on %s...\n", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	// 7. Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	fmt.Println("\nShutting down server...")

	// 8. Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	fmt.Println("Server exited gracefully")
}
