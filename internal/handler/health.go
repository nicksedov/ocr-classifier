package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// HealthCheck handles health check requests.
// Returns a simple status response indicating the service is operational.
func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{"status": "ok"}); err != nil {
		fmt.Fprintf(w, `{"status":"ok"}`)
	}
}
