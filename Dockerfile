# ====== Build Stage ======
FROM golang:1.26.2-alpine3.23 AS builder

# Install build dependencies
RUN apk update && apk add --no-cache git gcc g++

# Install Tesseract OCR and Leptonica (needed for gosseract CGO bindings)
RUN apk update && apk add --no-cache tesseract-ocr-dev leptonica-dev

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-w -s" -o ocr-classifier ./cmd/server

# ====== Runtime Stage ======
FROM alpine:3.23

# Install runtime dependencies (Tesseract + Leptonica + language packs + C lib)
# Alpine 3.23 uses tesseract-ocr package with tesseract-ocr-lang-* for language packs
RUN apk update && apk add --no-cache tesseract-ocr tesseract-ocr-data-eng tesseract-ocr-data-rus libstdc++ ca-certificates

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/ocr-classifier .

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port (default 8080, configurable via OCR_PORT env)
ENV OCR_PORT=8080
EXPOSE ${OCR_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:${OCR_PORT}/ocr-classifier/api/health || exit 1

# Run the server
CMD ["./ocr-classifier"]
