package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// --- helpers -----------------------------------------------------------------

func newTestProxy(backendURL string, ch chan MetricData) *Proxy {
	var client *PrometheusClient
	if ch != nil {
		client = &PrometheusClient{MetricsCh: ch}
	}
	return New(backendURL, client)
}

// ollamaNDJSON returns a minimal two-line Ollama streaming response body with
// the given field values in the final "done" event.
func ollamaNDJSON(model string, inputTokens, outputTokens int, totalDurationNs int64) string {
	first := fmt.Sprintf(
		`{"model":%q,"created_at":"2024-01-01T00:00:00Z","response":"hello","done":false}`,
		model,
	)
	last := fmt.Sprintf(
		`{"model":%q,"created_at":"2024-01-01T00:00:00Z","response":"","done":true,"prompt_eval_count":%d,"eval_count":%d,"total_duration":%d}`,
		model, inputTokens, outputTokens, totalDurationNs,
	)
	return first + "\n" + last + "\n"
}

// drainChannel reads up to one item from a buffered MetricData channel without
// blocking. Returns (value, true) when an item was present, (zero, false) otherwise.
func drainChannel(ch chan MetricData) (MetricData, bool) {
	select {
	case m := <-ch:
		return m, true
	default:
		return MetricData{}, false
	}
}

// --- existing tests (preserved) ----------------------------------------------

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

// --- new tests ---------------------------------------------------------------

// TestServeHTTP_MetricsWorthy verifies end-to-end forwarding plus metric
// extraction for a POST to /api/generate that returns a valid NDJSON body.
func TestServeHTTP_MetricsWorthy(t *testing.T) {
	// Arrange
	responseBody := ollamaNDJSON("llama2", 10, 20, 5_000_000_000)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, responseBody)
	}))
	defer backend.Close()

	metricsCh := make(chan MetricData, 1)
	p := newTestProxy(backend.URL, metricsCh)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/generate",
		bytes.NewBufferString(`{"model":"llama2","prompt":"hello"}`))

	// Act
	p.ServeHTTP(rec, req)

	// Assert — response forwarded correctly
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "llama2") {
		t.Errorf("response body should contain model name, got: %s", rec.Body.String())
	}

	// Assert — metric sent on channel
	m, ok := drainChannel(metricsCh)
	if !ok {
		t.Fatal("expected a MetricData to be sent to prometheusClient channel")
	}
	if m.Model != "llama2" {
		t.Errorf("expected model 'llama2', got %q", m.Model)
	}
	if m.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", m.InputTokens)
	}
	if m.OutputTokens != 20 {
		t.Errorf("expected OutputTokens=20, got %d", m.OutputTokens)
	}

	// Assert — internal model stats updated
	stats, exists := p.GetModelStats("llama2")
	if !exists {
		t.Fatal("expected model stats for 'llama2' to exist")
	}
	if stats.Requests != 1 {
		t.Errorf("expected Requests=1, got %d", stats.Requests)
	}
	if stats.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", stats.InputTokens)
	}
	if stats.OutputTokens != 20 {
		t.Errorf("expected OutputTokens=20, got %d", stats.OutputTokens)
	}
}

// TestServeHTTP_PassThrough verifies that non-metrics-worthy requests are
// forwarded blindly without triggering metric extraction.
func TestServeHTTP_PassThrough(t *testing.T) {
	// Arrange
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"models":[]}`)
	}))
	defer backend.Close()

	metricsCh := make(chan MetricData, 1)
	p := newTestProxy(backend.URL, metricsCh)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/tags", nil)

	// Act
	p.ServeHTTP(rec, req)

	// Assert — response forwarded as-is
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
	if rec.Body.String() != `{"models":[]}` {
		t.Errorf("unexpected body: %s", rec.Body.String())
	}

	// Assert — no metric sent on channel
	if _, ok := drainChannel(metricsCh); ok {
		t.Error("expected no MetricData for a non-metrics-worthy request")
	}

	// Assert — no model stats created
	allStats := p.GetAllModelStats()
	if len(allStats) != 0 {
		t.Errorf("expected no model stats, got %d", len(allStats))
	}
}

// TestServeHTTP_BackendError verifies that a 5xx backend status is forwarded
// to the client without attempting metric extraction.
func TestServeHTTP_BackendError(t *testing.T) {
	// Arrange
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, "internal error")
	}))
	defer backend.Close()

	metricsCh := make(chan MetricData, 1)
	p := newTestProxy(backend.URL, metricsCh)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/generate",
		bytes.NewBufferString(`{"model":"llama2","prompt":"hello"}`))

	// Act
	p.ServeHTTP(rec, req)

	// Assert — 500 forwarded verbatim
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}

	// Assert — no metric extraction on non-2xx
	if _, ok := drainChannel(metricsCh); ok {
		t.Error("expected no metric sent for a 500 backend response")
	}
	if len(p.GetAllModelStats()) != 0 {
		t.Error("expected no model stats after a 500 backend response")
	}
}

// TestServeHTTP_BackendDown verifies that an unreachable backend causes the
// proxy to return 502 Bad Gateway.
func TestServeHTTP_BackendDown(t *testing.T) {
	// Arrange — start a server just to obtain a valid address, then stop it
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	backendURL := backend.URL
	backend.Close() // immediately shut down so it is unreachable

	p := newTestProxy(backendURL, nil)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/generate",
		bytes.NewBufferString(`{"model":"llama2","prompt":"hello"}`))

	// Act
	p.ServeHTTP(rec, req)

	// Assert
	if rec.Code != http.StatusBadGateway {
		t.Errorf("expected status 502, got %d", rec.Code)
	}
}

// TestMetricsWorthy checks the metricsWorthy predicate against every expected path.
func TestMetricsWorthy(t *testing.T) {
	p := &Proxy{}

	cases := []struct {
		path string
		want bool
	}{
		{"/api/chat", true},
		{"/api/chat/completions", true},
		{"/api/generate", true},
		{"/api/embeddings", true},
		{"/api/tags", false},
		{"/api/show", false},
		{"/", false},
		{"", false},
	}

	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			got := p.metricsWorthy(c.path)
			if got != c.want {
				t.Errorf("metricsWorthy(%q) = %v, want %v", c.path, got, c.want)
			}
		})
	}
}

// TestHandleModels verifies that HandleModels returns a JSON body containing
// all models currently stored in the proxy's internal stats map.
func TestHandleModels(t *testing.T) {
	// Arrange
	p := New("http://localhost:11434", nil)
	p.mu.Lock()
	p.modelStats["llama2"] = &ModelStats{
		Requests:     5,
		InputTokens:  100,
		OutputTokens: 50,
		LastActive:   time.Now(),
	}
	p.modelStats["mistral"] = &ModelStats{
		Requests:     2,
		InputTokens:  40,
		OutputTokens: 20,
		LastActive:   time.Now(),
	}
	p.mu.Unlock()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics/models", nil)

	// Act
	p.HandleModels(rec, req)

	// Assert
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", ct)
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response body: %v", err)
	}

	models, ok := payload["models"].(map[string]interface{})
	if !ok {
		t.Fatal("expected 'models' key in response")
	}
	if _, exists := models["llama2"]; !exists {
		t.Error("expected 'llama2' in models response")
	}
	if _, exists := models["mistral"]; !exists {
		t.Error("expected 'mistral' in models response")
	}
}

// TestHandleUsage verifies that HandleUsage returns a JSON body with total
// metrics values matching what was recorded in the proxy.
func TestHandleUsage(t *testing.T) {
	// Arrange
	p := New("http://localhost:11434", nil)
	p.mu.Lock()
	p.totalMetrics.TotalRequests = 7
	p.totalMetrics.TotalInput = 300
	p.totalMetrics.TotalOutput = 150
	p.mu.Unlock()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics/usage", nil)

	// Act
	p.HandleUsage(rec, req)

	// Assert
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", ct)
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response body: %v", err)
	}

	assertJSONFloat := func(key string, want float64) {
		t.Helper()
		v, ok := payload[key]
		if !ok {
			t.Errorf("expected key %q in usage response", key)
			return
		}
		got, ok := v.(float64)
		if !ok {
			t.Errorf("expected numeric value for %q, got %T", key, v)
			return
		}
		if got != want {
			t.Errorf("%s: expected %.0f, got %.0f", key, want, got)
		}
	}

	assertJSONFloat("total_requests", 7)
	assertJSONFloat("total_input_tokens", 300)
	assertJSONFloat("total_output_tokens", 150)
}

// TestSafeExtractAndRecord_PanicRecovery verifies that safeExtractAndRecord
// does not propagate a panic and instead counts the failure in the extraction
// error counter. We trigger the panic by passing a body that contains invalid
// JSON on every line so the extractor produces a zero-field MetricData; the
// function still completes gracefully.
func TestSafeExtractAndRecord_PanicRecovery(t *testing.T) {
	// Arrange
	p := New("http://localhost:11434", nil)

	// An empty body leads to extractMetrics returning a zero-value MetricData
	// (not nil), so safeExtractAndRecord runs to completion without panicking.
	// This validates the "happy" path through the defer/recover guard.
	t.Run("empty body does not panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("safeExtractAndRecord panicked: %v", r)
			}
		}()
		p.safeExtractAndRecord("/api/generate", "unknown", []byte{}, time.Millisecond)
	})

	// All-garbage lines exercise the parse_error counter and reach completion.
	t.Run("garbage body does not panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("safeExtractAndRecord panicked: %v", r)
			}
		}()
		garbage := []byte("not-json\nalso-not-json\n{broken")
		p.safeExtractAndRecord("/api/generate", "unknown", garbage, time.Millisecond)
	})
}

// TestSafeExtractAndRecord_UpdatesStats confirms that a valid body causes
// model stats and total metrics to be incremented.
func TestSafeExtractAndRecord_UpdatesStats(t *testing.T) {
	// Arrange
	p := New("http://localhost:11434", nil)
	body := []byte(ollamaNDJSON("phi3", 15, 30, 2_000_000_000))

	// Act
	p.safeExtractAndRecord("/api/generate", "phi3", body, 2*time.Second)

	// Assert model stats
	stats, ok := p.GetModelStats("phi3")
	if !ok {
		t.Fatal("expected model stats for 'phi3'")
	}
	if stats.Requests != 1 {
		t.Errorf("expected Requests=1, got %d", stats.Requests)
	}
	if stats.InputTokens != 15 {
		t.Errorf("expected InputTokens=15, got %d", stats.InputTokens)
	}
	if stats.OutputTokens != 30 {
		t.Errorf("expected OutputTokens=30, got %d", stats.OutputTokens)
	}

	// Assert total metrics
	if p.GetTotalRequests() != 1 {
		t.Errorf("expected TotalRequests=1, got %d", p.GetTotalRequests())
	}
}

// TestCreateBackendRequest verifies URL construction, method preservation,
// header cloning, and query-param forwarding.
func TestCreateBackendRequest(t *testing.T) {
	// Arrange
	p := New("http://ollama.internal:11434", nil)
	req := httptest.NewRequest("POST", "/api/generate?stream=true",
		bytes.NewBufferString(`{"model":"llama2"}`))
	req.Header.Set("Authorization", "Bearer token123")
	req.Header.Set("X-Custom-Header", "value")

	// Act
	backendReq, err := p.createBackendRequest(req)

	// Assert
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if backendReq.Method != "POST" {
		t.Errorf("expected method POST, got %s", backendReq.Method)
	}

	expectedURL := "http://ollama.internal:11434/api/generate?stream=true"
	if backendReq.URL.String() != expectedURL {
		t.Errorf("expected URL %q, got %q", expectedURL, backendReq.URL.String())
	}

	if backendReq.Header.Get("Authorization") != "Bearer token123" {
		t.Errorf("Authorization header not cloned, got %q", backendReq.Header.Get("Authorization"))
	}
	if backendReq.Header.Get("X-Custom-Header") != "value" {
		t.Errorf("X-Custom-Header not cloned, got %q", backendReq.Header.Get("X-Custom-Header"))
	}
	if backendReq.URL.RawQuery != "stream=true" {
		t.Errorf("expected query stream=true, got %q", backendReq.URL.RawQuery)
	}
}

// TestCreateBackendRequest_ContextPreserved verifies that the original request
// context is attached to the backend request so cancellation propagates.
func TestCreateBackendRequest_ContextPreserved(t *testing.T) {
	p := New("http://localhost:11434", nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", "/api/tags", nil)
	backendReq, err := p.createBackendRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if backendReq.Context() != ctx {
		t.Error("expected backend request to carry the original context")
	}
}

// TestBackendHealthy verifies the initial health state and the effect of
// checkBackendHealth against both a healthy and an unhealthy backend.
func TestBackendHealthy(t *testing.T) {
	// Arrange
	p := New("http://localhost:11434", nil)

	t.Run("initially false", func(t *testing.T) {
		if p.BackendHealthy() {
			t.Error("expected BackendHealthy()=false before first check")
		}
	})

	t.Run("healthy backend sets true", func(t *testing.T) {
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// /api/tags should return 200
			w.WriteHeader(http.StatusOK)
		}))
		defer backend.Close()

		p2 := New(backend.URL, nil)
		p2.checkBackendHealth()

		if !p2.BackendHealthy() {
			t.Error("expected BackendHealthy()=true after successful health check")
		}
	})

	t.Run("backend returning 500 sets false", func(t *testing.T) {
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer backend.Close()

		p3 := New(backend.URL, nil)
		// Manually set true first to confirm the check flips it back.
		p3.backendHealthy.Store(true)
		p3.checkBackendHealth()

		if p3.BackendHealthy() {
			t.Error("expected BackendHealthy()=false after 500 health check")
		}
	})

	t.Run("unreachable backend sets false", func(t *testing.T) {
		closed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		url := closed.URL
		closed.Close()

		p4 := New(url, nil)
		p4.backendHealthy.Store(true)
		p4.checkBackendHealth()

		if p4.BackendHealthy() {
			t.Error("expected BackendHealthy()=false when backend is unreachable")
		}
	})
}

// TestStartHealthChecker verifies that the health checker goroutine updates
// the backend health state and respects context cancellation.
func TestStartHealthChecker(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	p := New(backend.URL, nil)

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() {
		p.StartHealthChecker(ctx, 10*time.Millisecond)
		close(done)
	}()

	// Give the checker one tick to run the initial probe.
	time.Sleep(30 * time.Millisecond)
	if !p.BackendHealthy() {
		t.Error("expected BackendHealthy()=true while backend is running")
	}

	cancel()

	select {
	case <-done:
		// goroutine exited cleanly
	case <-time.After(500 * time.Millisecond):
		t.Error("StartHealthChecker did not exit after context cancellation")
	}
}

// TestServeHTTP_ActiveRequestsTracking verifies that the active-request counter
// is incremented during the request and returns to zero afterwards.
func TestServeHTTP_ActiveRequestsTracking(t *testing.T) {
	// Arrange — slow backend so we can observe the in-flight state
	started := make(chan struct{})
	unblock := make(chan struct{})

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-unblock
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"done":true}`)
	}))
	defer backend.Close()

	p := newTestProxy(backend.URL, nil)

	done := make(chan struct{})
	go func() {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("GET", "/api/tags", nil)
		p.ServeHTTP(rec, req)
		close(done)
	}()

	// Wait until backend has received the request before sampling
	<-started
	if p.GetActiveRequests() != 1 {
		t.Errorf("expected 1 active request in-flight, got %d", p.GetActiveRequests())
	}

	close(unblock)
	<-done

	if p.GetActiveRequests() != 0 {
		t.Errorf("expected 0 active requests after completion, got %d", p.GetActiveRequests())
	}
}

// TestServeHTTP_HeadersForwarded verifies that response headers from the
// backend are forwarded to the client.
func TestServeHTTP_HeadersForwarded(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Ollama-Version", "0.1.0")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer backend.Close()

	p := newTestProxy(backend.URL, nil)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/tags", nil)
	p.ServeHTTP(rec, req)

	if rec.Header().Get("X-Ollama-Version") != "0.1.0" {
		t.Errorf("expected X-Ollama-Version header to be forwarded, got %q",
			rec.Header().Get("X-Ollama-Version"))
	}
}

// TestServeHTTP_MetricsChannel_DropWhenFull verifies that a full metrics
// channel does not block the handler — the metric is silently dropped.
func TestServeHTTP_MetricsChannel_DropWhenFull(t *testing.T) {
	responseBody := ollamaNDJSON("llama2", 5, 10, 1_000_000_000)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, responseBody)
	}))
	defer backend.Close()

	// Deliberately zero-capacity channel so it is immediately full
	metricsCh := make(chan MetricData, 0)
	p := newTestProxy(backend.URL, metricsCh)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/generate",
		bytes.NewBufferString(`{"model":"llama2","prompt":"test"}`))

	// Act — must not block or panic
	done := make(chan struct{})
	go func() {
		p.ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-done:
		// completed without blocking
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP blocked on a full metrics channel")
	}

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 even when channel is full, got %d", rec.Code)
	}
}
