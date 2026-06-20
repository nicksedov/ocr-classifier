package config

import (
	"os"
)

// Config holds application configuration.
type Config struct {
	Port string
}

// Load loads configuration from environment variables.
// Defaults to port 8080 if OCR_PORT is not set.
func Load() *Config {
	port := os.Getenv("OCR_PORT")
	if port == "" {
		port = "8080"
	}
	return &Config{
		Port: port,
	}
}
