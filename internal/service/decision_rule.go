package service

// DecisionRule holds the criteria for determining if a document is a text document.
type DecisionRule struct {
	MinConfidence float64 // Minimum weighted confidence (0-1)
	MinTokenCount int     // Minimum token count
}

// GetDefaultDecisionRule returns the default decision criteria.
func GetDefaultDecisionRule() DecisionRule {
	return DecisionRule{
		MinConfidence: 0.66,
		MinTokenCount: 20,
	}
}

// EvaluateDecision determines if the document is a text document based on the criteria.
// Returns true if weighted confidence >= minConfidence AND token count >= minTokenCount.
func EvaluateDecision(weightedConfidence float64, tokenCount int, rule DecisionRule) bool {
	return weightedConfidence >= rule.MinConfidence && tokenCount >= rule.MinTokenCount
}

// EvaluateDecisionWithDefaults uses default criteria to evaluate the decision.
func EvaluateDecisionWithDefaults(weightedConfidence float64, tokenCount int) bool {
	criteria := GetDefaultDecisionRule()
	return EvaluateDecision(weightedConfidence, tokenCount, criteria)
}
