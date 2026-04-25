package config

import (
	"os"
)

// Config holds application configuration.
type Config struct {
	Port string
}

// Load loads configuration from environment variables.
// Defaults to port 8080 if PORT is not set.
func Load() *Config {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	return &Config{
		Port: port,
	}
}
