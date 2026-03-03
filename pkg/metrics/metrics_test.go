package metrics

import (
	"testing"
)

// TestRecordCompletedRequest tests request recording
func TestRecordCompletedRequest(t *testing.T) {
	Reset()

	data := MetricData{
		Model:            "llama2",
		Endpoint:         "/api/generate",
		Category:         "general",
		InputTokens:      100,
		OutputTokens:     50,
		Duration:         5000000000, // 5 seconds
		TimeToFirstToken: 100,
	}

	RecordCompletedRequest(data)

	// Verify metrics were recorded
	stats, exists := modelStats["llama2"]
	if !exists || stats.TotalInputTokens != 100 || stats.TotalOutputTokens != 50 {
		t.Errorf("Stats not recorded correctly")
	}
}

// TestGetModelStats tests stats retrieval
func TestGetModelStats(t *testing.T) {
	Reset()

	// Should return empty initially
	_, exists := modelStats["nonexistent"]
	if exists {
		t.Errorf("Unexpected stats for nonexistent model")
	}
}

// TestReset tests reset functionality
func TestReset(t *testing.T) {
	Reset()

	// Add some data
	data := MetricData{
		Model:        "test",
		InputTokens:  100,
		OutputTokens: 50,
	}
	RecordCompletedRequest(data)

	// Reset
	Reset()

	// Should be empty
	if len(modelStats) != 0 {
		t.Errorf("Expected empty modelStats after reset")
	}
}
