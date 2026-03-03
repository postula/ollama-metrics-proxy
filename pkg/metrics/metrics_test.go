package metrics

import (
	"testing"
	"time"
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

// TestRecordActiveRequests verifies the gauge update does not panic for
// both positive and negative deltas.
func TestRecordActiveRequests(t *testing.T) {
	Reset()

	// Positive delta — request starting
	RecordActiveRequests("llama2", 1)

	// Negative delta — request finishing
	RecordActiveRequests("llama2", -1)

	// Zero delta — no-op, must not panic either
	RecordActiveRequests("llama2", 0)
}

// TestRecordBackendHealth verifies that marking a backend healthy sets the
// gauge without panicking.
func TestRecordBackendHealth(t *testing.T) {
	Reset()

	RecordBackendHealth("http://localhost:11434")
}

// TestRecordBackendError verifies that marking a backend unhealthy sets the
// gauge without panicking.
func TestRecordBackendError(t *testing.T) {
	Reset()

	RecordBackendError("http://localhost:11434")
}

// TestGetModelStats_Found verifies that stats recorded via RecordCompletedRequest
// are returned correctly by GetModelStats.
func TestGetModelStats_Found(t *testing.T) {
	Reset()

	// Arrange
	data := MetricData{
		Model:        "mistral",
		Endpoint:     "/api/generate",
		Category:     "chat",
		InputTokens:  200,
		OutputTokens: 80,
		Duration:     3 * time.Second,
	}

	// Act
	RecordCompletedRequest(data)
	stats, ok := GetModelStats("mistral")

	// Assert
	if !ok {
		t.Fatal("expected ok=true for model mistral")
	}
	if stats.TotalInputTokens != 200 {
		t.Errorf("TotalInputTokens: got %d, want 200", stats.TotalInputTokens)
	}
	if stats.TotalOutputTokens != 80 {
		t.Errorf("TotalOutputTokens: got %d, want 80", stats.TotalOutputTokens)
	}
	if stats.TotalRequestCount != 1 {
		t.Errorf("TotalRequestCount: got %d, want 1", stats.TotalRequestCount)
	}
	if stats.TotalRequestDuration != 3*time.Second {
		t.Errorf("TotalRequestDuration: got %v, want 3s", stats.TotalRequestDuration)
	}
}

// TestGetModelStats_NotFound verifies that GetModelStats returns ok=false for
// a model that has never been recorded.
func TestGetModelStats_NotFound(t *testing.T) {
	Reset()

	_, ok := GetModelStats("unknown-model")

	if ok {
		t.Error("expected ok=false for unknown model")
	}
}

// TestGetAllModelStats verifies that all models recorded via RecordCompletedRequest
// are present in the map returned by GetAllModelStats.
func TestGetAllModelStats(t *testing.T) {
	Reset()

	// Arrange — two distinct models
	models := []string{"llama2", "mistral"}
	for _, model := range models {
		RecordCompletedRequest(MetricData{
			Model:        model,
			Endpoint:     "/api/generate",
			Category:     "chat",
			InputTokens:  10,
			OutputTokens: 5,
			Duration:     time.Second,
		})
	}

	// Act
	all := GetAllModelStats()

	// Assert
	for _, model := range models {
		if _, ok := all[model]; !ok {
			t.Errorf("model %q missing from GetAllModelStats result", model)
		}
	}
	if len(all) != len(models) {
		t.Errorf("expected %d models, got %d", len(models), len(all))
	}
}

// TestRecordCompletedRequest_WithTimeToFirstToken verifies that a non-zero
// TimeToFirstToken value is handled without errors.
func TestRecordCompletedRequest_WithTimeToFirstToken(t *testing.T) {
	Reset()

	data := MetricData{
		Model:            "llama2",
		Endpoint:         "/api/generate",
		Category:         "chat",
		InputTokens:      50,
		OutputTokens:     25,
		Duration:         2 * time.Second,
		TimeToFirstToken: 300,
	}

	// Should complete without panic
	RecordCompletedRequest(data)

	stats, ok := GetModelStats("llama2")
	if !ok {
		t.Fatal("expected stats to be recorded")
	}
	if stats.TotalRequestCount != 1 {
		t.Errorf("TotalRequestCount: got %d, want 1", stats.TotalRequestCount)
	}
}

// TestRecordCompletedRequest_ZeroTokens verifies that zero-token requests do
// not increment token counters but still update the request count.
func TestRecordCompletedRequest_ZeroTokens(t *testing.T) {
	Reset()

	data := MetricData{
		Model:        "llama2",
		Endpoint:     "/api/generate",
		Category:     "chat",
		InputTokens:  0,
		OutputTokens: 0,
		Duration:     time.Second,
	}

	RecordCompletedRequest(data)

	stats, ok := GetModelStats("llama2")
	if !ok {
		t.Fatal("expected stats entry even with zero tokens")
	}
	if stats.TotalInputTokens != 0 {
		t.Errorf("TotalInputTokens: got %d, want 0", stats.TotalInputTokens)
	}
	if stats.TotalOutputTokens != 0 {
		t.Errorf("TotalOutputTokens: got %d, want 0", stats.TotalOutputTokens)
	}
	if stats.TotalRequestCount != 1 {
		t.Errorf("TotalRequestCount: got %d, want 1", stats.TotalRequestCount)
	}
}
