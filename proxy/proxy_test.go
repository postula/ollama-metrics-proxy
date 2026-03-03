package proxy

import (
	"bytes"
	"net/http/httptest"
	"testing"
	"time"
)

// TestProxy handles proxy testing
func TestProxy(t *testing.T) {
	metricsCh := make(chan MetricData, 1)
	client := &PrometheusClient{MetricsCh: metricsCh}

	p := New("http://localhost:11434", client)
	if p == nil {
		t.Fatal("Expected proxy to be created")
	}
}

// TestExtractMetrics tests metrics extraction
func TestExtractMetrics(t *testing.T) {
	p := &Proxy{}

	// Simulate Ollama streaming response
	body := []byte(`{"model":"llama2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"}}
{"model":"llama2","created_at":"2024-01-01T00:00:00Z","prompt_eval_count":100,"prompt_eval_duration":5000000}
{"model":"llama","created_at":"2024-01-01T00:00:00Z","eval_count":50,"eval_duration":3000000,"total_duration":8000000,"done":true}`)

	metrics := p.extractMetrics(body)

	if metrics == nil {
		t.Fatal("Expected metrics, got nil")
	}

	if metrics.InputTokens != 100 {
		t.Errorf("Expected InputTokens=100, got %d", metrics.InputTokens)
	}

	if metrics.OutputTokens != 50 {
		t.Errorf("Expected OutputTokens=50, got %d", metrics.OutputTokens)
	}
}

// TestExtractModel tests model extraction from JSON request body
func TestExtractModel(t *testing.T) {
	p := &Proxy{}

	body := bytes.NewBufferString(`{"model":"mistral","prompt":"hello"}`)
	req := httptest.NewRequest("POST", "/api/generate", body)
	model := p.extractModel(req)

	if model != "mistral" {
		t.Errorf("Expected model 'mistral', got '%s'", model)
	}
}

// TestCategorizeRequest tests request categorization
func TestCategorizeRequest(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		expected string
	}{
		{"chat", "/api/chat", "chat"},
		{"generate", "/api/generate", "generate"},
		{"embedding", "/api/embeddings", "embedding"},
		{"general", "/unknown", "general"},
	}

	p := &Proxy{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.categorizeRequest(tt.path)
			if got != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, got)
			}
		})
	}
}

// TestGetAllModelStats tests retrieving all model statistics
func TestGetAllModelStats(t *testing.T) {
	p := New("http://localhost:11434", nil)

	// Initial state should be empty
	stats := p.GetAllModelStats()
	if len(stats) != 0 {
		t.Errorf("Expected empty stats, got %d models", len(stats))
	}

	// Add a model
	p.mu.Lock()
	p.modelStats["llama2"] = &ModelStats{
		Requests:      10,
		InputTokens:   1000,
		OutputTokens:  500,
		TotalDuration: 5000000000,
		LastActive:    time.Now(),
	}
	p.mu.Unlock()

	// Should now have models
	stats = p.GetAllModelStats()
	if len(stats) != 1 {
		t.Errorf("Expected 1 model, got %d", len(stats))
	}
}

// TestReset clears metrics
func TestReset(t *testing.T) {
	p := New("http://localhost:11434", nil)

	p.mu.Lock()
	p.modelStats["test"] = &ModelStats{Requests: 5}
	p.activeRequests = 5
	p.totalMetrics.TotalRequests = 10
	p.mu.Unlock()

	p.Reset()

	// Should be reset
	if p.activeRequests != 0 {
		t.Errorf("Expected 0 active requests, got %d", p.activeRequests)
	}

	stats := p.GetAllModelStats()
	if len(stats) != 0 {
		t.Errorf("Expected empty stats, got %d models", len(stats))
	}
}
