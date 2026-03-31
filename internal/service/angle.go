package service

import (
	"image"
	"image/color"
	"math"
	"sort"
)

// detectSkewAngle detects the rotation angle of a document image using
// Canny edge detection and Hough Line Transform.
// Returns candidate rotation angles to try for OCR.
func detectSkewAngle(gray *image.Gray) []int {
	edges := cannyEdgeDetection(gray, 50, 150)

	// Adaptive Hough threshold proportional to image size
	bounds := gray.Bounds()
	minDim := bounds.Dx()
	if bounds.Dy() < minDim {
		minDim = bounds.Dy()
	}
	threshold := minDim / 16
	if threshold < 50 {
		threshold = 50
	}

	lines := houghLineTransform(edges, threshold)

	if len(lines) == 0 {
		return []int{90, 180, 270}
	}

	skew := findDominantSkew(lines)
	return generateCandidateAngles(skew)
}

// cannyEdgeDetection performs Canny edge detection on a grayscale image.
// Warning: it is required to apply smoothing (Gaussian/median blur) before passing the image to the method.
func cannyEdgeDetection(smoothedGray *image.Gray, lowThreshold, highThreshold float64) *image.Gray {
	bounds := smoothedGray.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	gx, gy := sobelGradients(smoothedGray)
	// Gradient magnitude and direction
	magnitude := make([][]float64, h)
	direction := make([][]float64, h)
	maxMag := 0.0
	for y := 0; y < h; y++ {
		magnitude[y] = make([]float64, w)
		direction[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			mag := math.Hypot(gx[y][x], gy[y][x])
			magnitude[y][x] = mag
			if mag > maxMag {
				maxMag = mag
			}
			direction[y][x] = math.Atan2(gy[y][x], gx[y][x])
		}
	}

	// Scale thresholds relative to max magnitude
	if maxMag > 0 {
		lowThreshold = lowThreshold / 255.0 * maxMag
		highThreshold = highThreshold / 255.0 * maxMag
	}

	suppressed := nonMaxSuppression(magnitude, direction, w, h)
	return hysteresisThresholding(suppressed, w, h, lowThreshold, highThreshold)
}

// sobelGradients computes Sobel gradients in X and Y directions.
func sobelGradients(gray *image.Gray) (gx, gy [][]float64) {
	bounds := gray.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	gx = make([][]float64, h)
	gy = make([][]float64, h)
	for y := 0; y < h; y++ {
		gx[y] = make([]float64, w)
		gy[y] = make([]float64, w)
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			p00 := float64(gray.GrayAt(x-1, y-1).Y)
			p10 := float64(gray.GrayAt(x, y-1).Y)
			p20 := float64(gray.GrayAt(x+1, y-1).Y)
			p01 := float64(gray.GrayAt(x-1, y).Y)
			p21 := float64(gray.GrayAt(x+1, y).Y)
			p02 := float64(gray.GrayAt(x-1, y+1).Y)
			p12 := float64(gray.GrayAt(x, y+1).Y)
			p22 := float64(gray.GrayAt(x+1, y+1).Y)

			gx[y][x] = -p00 + p20 - 2*p01 + 2*p21 - p02 + p22
			gy[y][x] = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22
		}
	}

	return
}

// nonMaxSuppression suppresses non-maximum gradient values along edge direction.
func nonMaxSuppression(magnitude, direction [][]float64, w, h int) [][]float64 {
	result := make([][]float64, h)
	for y := 0; y < h; y++ {
		result[y] = make([]float64, w)
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			mag := magnitude[y][x]
			angleDeg := direction[y][x] * 180.0 / math.Pi
			if angleDeg < 0 {
				angleDeg += 180
			}

			var q, r float64
			switch {
			case angleDeg < 22.5 || angleDeg >= 157.5:
				q = magnitude[y][x+1]
				r = magnitude[y][x-1]
			case angleDeg < 67.5:
				q = magnitude[y-1][x+1]
				r = magnitude[y+1][x-1]
			case angleDeg < 112.5:
				q = magnitude[y-1][x]
				r = magnitude[y+1][x]
			default:
				q = magnitude[y-1][x-1]
				r = magnitude[y+1][x+1]
			}

			if mag >= q && mag >= r {
				result[y][x] = mag
			}
		}
	}

	return result
}

// hysteresisThresholding applies double thresholding and edge tracking by hysteresis.
func hysteresisThresholding(suppressed [][]float64, w, h int, low, high float64) *image.Gray {
	edges := image.NewGray(image.Rect(0, 0, w, h))

	const (
		strong uint8 = 255
		weak   uint8 = 75
	)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := suppressed[y][x]
			if val >= high {
				edges.SetGray(x, y, color.Gray{Y: strong})
			} else if val >= low {
				edges.SetGray(x, y, color.Gray{Y: weak})
			}
		}
	}

	// Promote weak edges connected to strong edges
	changed := true
	for changed {
		changed = false
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if edges.GrayAt(x, y).Y != weak {
					continue
				}
				if edges.GrayAt(x-1, y-1).Y == strong ||
					edges.GrayAt(x, y-1).Y == strong ||
					edges.GrayAt(x+1, y-1).Y == strong ||
					edges.GrayAt(x-1, y).Y == strong ||
					edges.GrayAt(x+1, y).Y == strong ||
					edges.GrayAt(x-1, y+1).Y == strong ||
					edges.GrayAt(x, y+1).Y == strong ||
					edges.GrayAt(x+1, y+1).Y == strong {
					edges.SetGray(x, y, color.Gray{Y: strong})
					changed = true
				}
			}
		}
	}

	// Remove remaining weak edges
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if edges.GrayAt(x, y).Y == weak {
				edges.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}

	return edges
}

// houghLine represents a detected line in Hough parameter space.
type houghLine struct {
	rho   float64
	theta float64 // radians
	votes int
}

// houghLineTransform performs the standard Hough Line Transform on a binary edge image.
func houghLineTransform(edges *image.Gray, threshold int) []houghLine {
	bounds := edges.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	maxDist := math.Hypot(float64(w), float64(h))
	thetaRes := math.Pi / 180.0
	numTheta := 180
	numRho := int(2*maxDist) + 1

	cosThetas := make([]float64, numTheta)
	sinThetas := make([]float64, numTheta)
	for i := 0; i < numTheta; i++ {
		theta := float64(i) * thetaRes
		cosThetas[i] = math.Cos(theta)
		sinThetas[i] = math.Sin(theta)
	}

	accumulator := make([]int, numRho*numTheta)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if edges.GrayAt(x, y).Y == 0 {
				continue
			}
			fx, fy := float64(x), float64(y)
			for t := 0; t < numTheta; t++ {
				rho := fx*cosThetas[t] + fy*sinThetas[t]
				rhoIdx := int(math.Round(rho + maxDist))
				if rhoIdx >= 0 && rhoIdx < numRho {
					accumulator[rhoIdx*numTheta+t]++
				}
			}
		}
	}

	var lines []houghLine
	for r := 0; r < numRho; r++ {
		for t := 0; t < numTheta; t++ {
			votes := accumulator[r*numTheta+t]
			if votes >= threshold {
				lines = append(lines, houghLine{
					rho:   float64(r) - maxDist,
					theta: float64(t) * thetaRes,
					votes: votes,
				})
			}
		}
	}

	return lines
}

// findDominantSkew determines the dominant skew angle from detected Hough lines.
// Returns the skew in degrees (deviation from the nearest cardinal direction).
func findDominantSkew(lines []houghLine) float64 {
	sort.Slice(lines, func(i, j int) bool {
		return lines[i].votes > lines[j].votes
	})

	maxLines := 20
	if len(lines) < maxLines {
		maxLines = len(lines)
	}
	topLines := lines[:maxLines]

	var angles []float64
	var weights []float64
	for _, line := range topLines {
		// In Hough parameterization the normal angle is theta.
		// The line angle is theta - 90 degrees.
		lineAngleDeg := line.theta*180.0/math.Pi - 90.0

		// Normalize to [-45, 45) — deviation from nearest 90-degree axis
		skew := math.Mod(lineAngleDeg, 90.0)
		if skew > 45 {
			skew -= 90
		}
		if skew < -45 {
			skew += 90
		}

		angles = append(angles, skew)
		weights = append(weights, float64(line.votes))
	}

	return weightedMedian(angles, weights)
}

// weightedMedian returns the weighted median of a set of values.
func weightedMedian(values, weights []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	type pair struct {
		value  float64
		weight float64
	}
	pairs := make([]pair, len(values))
	totalWeight := 0.0
	for i := range values {
		pairs[i] = pair{values[i], weights[i]}
		totalWeight += weights[i]
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value < pairs[j].value
	})

	halfWeight := totalWeight / 2.0
	cumWeight := 0.0
	for _, p := range pairs {
		cumWeight += p.weight
		if cumWeight >= halfWeight {
			return p.value
		}
	}

	return pairs[len(pairs)-1].value
}

// generateCandidateAngles produces up to 4 candidate rotation angles
// from the detected skew, covering all cardinal orientations.
// Angle 0 is excluded because Phase 1 already tested it.
func generateCandidateAngles(skewDeg float64) []int {
	skew := int(math.Round(skewDeg))

	normalize := func(angle int) int {
		return ((angle % 360) + 360) % 360
	}

	raw := [4]int{
		normalize(skew),
		normalize(skew + 90),
		normalize(skew + 180),
		normalize(skew + 270),
	}

	seen := make(map[int]bool)
	var unique []int
	for _, c := range raw {
		if c != 0 && !seen[c] {
			seen[c] = true
			unique = append(unique, c)
		}
	}

	return unique
}
