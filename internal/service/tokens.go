package service

import "unicode"

// UnitSymbols defines Unicode range table for unit symbols used in token counting.
var UnitSymbols = &unicode.RangeTable{
	R16: []unicode.Range16{
		// Ratios and parts
		{Lo: 0x0025, Hi: 0x0025, Stride: 1}, // %
		{Lo: 0x2030, Hi: 0x2034, Stride: 1}, // ‰ ‱ ′ ″ ‴

		// Angles and degrees
		{Lo: 0x00B0, Hi: 0x00B0, Stride: 1}, // °
		{Lo: 0x210F, Hi: 0x210F, Stride: 1}, // ℏ (planck constant over 2pi, unit-like)

		// Temperature
		{Lo: 0x2103, Hi: 0x2103, Stride: 1}, // ℃
		{Lo: 0x2109, Hi: 0x2109, Stride: 1}, // ℉
		{Lo: 0x212A, Hi: 0x212A, Stride: 1}, // K
		{Lo: 0x212B, Hi: 0x212B, Stride: 1}, // Å (angstrom)

		// Length (common)
		{Lo: 0x00B5, Hi: 0x00B5, Stride: 1}, // µ
		{Lo: 0x3395, Hi: 0x339A, Stride: 1}, // ㎕ ㎖ ㎗ ㎘ ㎙ ㎚
		{Lo: 0x33A0, Hi: 0x33A1, Stride: 1}, // ㎠ ㎡
		{Lo: 0x33A9, Hi: 0x33AA, Stride: 1}, // ㎩ ㎪
		{Lo: 0x33AB, Hi: 0x33AF, Stride: 1}, // ㎫ ㎬ ㎭ ㎮ ㎯ ㎰

		// Area/volume (select)
		{Lo: 0x33BD, Hi: 0x33C4, Stride: 1}, // ㎡ ㎽ ㎾ ㎿ ㏀ ㏁
		{Lo: 0x33C6, Hi: 0x33CA, Stride: 1}, // ㏆ ㏇ ㏈ ㏉ ㏊
	},
}

// countTokens counts meaningful characters in a string:
//   - Latin and Cyrillic letters
//   - Digits
//   - Dot/comma between digits (e.g., "3.14", "1,000")
//   - Plus/minus before a number (e.g., "+5", "-10")
//   - Unit symbol (percent, promille, degree, minute, second etc.) after a number (e.g., "45°", "30′", "15″", "5‰")
//   - Currency symbols before/after a number (e.g., "$100", "50€")
//   - Quotation marks adjacent to letters or digits (e.g., "abc", '123', «text»)
func countTokens(s string) int {
	runes := []rune(s)
	count := 0

	for i, r := range runes {
		// Always count letters and digits
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			count++
			continue
		}

		// Check for dot or comma between digits
		if r == '.' || r == ',' || r == ':' || r == '/' {
			if i > 0 && i < len(runes)-1 && unicode.IsDigit(runes[i-1]) && unicode.IsDigit(runes[i+1]) {
				count++
			}
			continue
		}

		// Check for plus/minus before a digit
		if r == '+' || r == '-' || r == '−' { // including Unicode minus
			if i < len(runes)-1 && unicode.IsDigit(runes[i+1]) {
				count++
			}
			continue
		}

		// Check for unit symbol (percent, promille, degree, minute, second etc.) after a digit
		if unicode.Is(UnitSymbols, r) {
			if i > 0 && unicode.IsDigit(runes[i-1]) {
				count++
			}
			continue
		}

		// Check for currency symbols before or after a digit
		if unicode.Is(unicode.Sc, r) { // Sc = Symbol, currency
			hasPrevDigit := i > 0 && unicode.IsDigit(runes[i-1])
			hasNextDigit := i < len(runes)-1 && unicode.IsDigit(runes[i+1])
			if hasPrevDigit || hasNextDigit {
				count++
			}
			continue
		}

		// Check for quotation marks adjacent to letters or digits
		if unicode.Is(unicode.Quotation_Mark, r) {
			isAdjacent := false
			if i > 0 {
				prev := runes[i-1]
				if unicode.IsLetter(prev) || unicode.IsDigit(prev) {
					isAdjacent = true
				}
			}
			if i < len(runes)-1 {
				next := runes[i+1]
				if unicode.IsLetter(next) || unicode.IsDigit(next) {
					isAdjacent = true
				}
			}
			if isAdjacent {
				count++
			}
		}
	}

	return count
}
