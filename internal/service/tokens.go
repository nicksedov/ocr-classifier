package service

import "unicode"

// countTokens counts meaningful characters in a string:
// - Latin and Cyrillic letters
// - Digits
// - Dot/comma between digits (e.g., "3.14", "1,000")
// - Plus/minus before a number (e.g., "+5", "-10")
// - Degree symbol after a number (e.g., "45°")
// - Currency symbols before/after a number (e.g., "$100", "50€")
func countTokens(s string) int {
	runes := []rune(s)
	count := 0

	for i, r := range runes {
		// Always count letters and digits
		if unicode.Is(unicode.Latin, r) || unicode.Is(unicode.Cyrillic, r) || unicode.IsDigit(r) {
			count++
			continue
		}

		// Check for dot or comma between digits
		if r == '.' || r == ',' {
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

		// Check for degree symbol after a digit
		if r == '°' {
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
	}

	return count
}
