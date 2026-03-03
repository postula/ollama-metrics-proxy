package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/postula/ollama-metrics-proxy/pkg/metrics"
)

// metricExtractionErrors counts failures during metric extraction (proxy self-health).
var metricExtractionErrors = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "ollama_proxy_metric_extraction_errors_total",
		Help: "Total metric extraction failures (proxy self-health)",
	},
	[]string{"endpoint", "reason"},
)

// OllamaEvent represents an Ollama API streaming event
type OllamaEvent struct {
	Model              string        `json:"model,omitempty"`
	Created            time.Time     `json:"created_at,omitempty"`
	Message            Message       `json:"message,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	Duration           time.Duration `json:"duration,omitempty"`
	Done               bool          `json:"done,omitempty"`
}

// Message represents a message in a chat response
type Message struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// MetricData holds collected metric information
type MetricData struct {
	Model        string        `json:"model"`
	Endpoint     string        `json:"endpoint"`
	Category     string        `json:"category"`
	InputTokens  int           `json:"input_tokens"`
	OutputTokens int           `json:"output_tokens"`
	Duration     time.Duration `json:"duration"`
	StatusCode   int           `json:"status_code"`
}

// RequestContext holds request tracking information
type RequestContext struct {
	StartTime time.Time
	Method    string
	Path      string
	Model     string
	Endpoint  string
}

// Proxy handles transparent forwarding of Ollama API requests
type Proxy struct {
	mu               sync.RWMutex
	backendURL       string
	httpClient       *http.Client
	modelStats       map[string]*ModelStats
	totalMetrics     *TotalMetrics
	activeRequests   int
	prometheusClient *PrometheusClient
	backendHealthy   atomic.Bool
}

type ModelStats struct {
	Requests      int
	InputTokens   int
	OutputTokens  int
	TotalDuration int64 // nanoseconds
	LastActive    time.Time
}

type TotalMetrics struct {
	Mu            sync.RWMutex
	TotalRequests int
	TotalInput    int
	TotalOutput   int
	StartTime     time.Time
	LastActive    time.Time
}

type PrometheusClient struct {
	MetricsCh chan MetricData
}

// New creates a new Proxy instance
func New(backendURL string, prometheusClient *PrometheusClient) *Proxy {
	return &Proxy{
		backendURL: backendURL,
		httpClient: &http.Client{},
		modelStats: make(map[string]*ModelStats),
		totalMetrics: &TotalMetrics{
			StartTime: time.Now(),
		},
		prometheusClient: prometheusClient,
	}
}

// categorizeRequest determines the request category from a URL path.
func (p *Proxy) categorizeRequest(path string) string {
	if strings.HasPrefix(path, "/api/chat") || strings.HasPrefix(path, "/v1/chat/completions") {
		return "chat"
	}
	if strings.HasPrefix(path, "/api/generate") {
		return "generate"
	}
	if strings.HasPrefix(path, "/api/embeddings") || strings.HasPrefix(path, "/v1/embeddings") {
		return "embedding"
	}
	return "general"
}

// metricsWorthy reports whether the path is an endpoint we track metrics for.
func (p *Proxy) metricsWorthy(path string) bool {
	return strings.HasPrefix(path, "/api/chat") ||
		strings.HasPrefix(path, "/api/generate") ||
		strings.HasPrefix(path, "/api/embeddings") ||
		strings.HasPrefix(path, "/v1/chat/completions") ||
		strings.HasPrefix(path, "/v1/embeddings")
}

// extractModel reads the model name from the JSON request body.
// The body is restored so downstream handlers can read it again.
func (p *Proxy) extractModel(r *http.Request) string {
	if r.Body == nil {
		return "unknown"
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return "unknown"
	}
	r.Body = io.NopCloser(bytes.NewReader(body))

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil || req.Model == "" {
		return "unknown"
	}
	return req.Model
}

// HandleModels returns list of available models
func (p *Proxy) HandleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	p.mu.RLock()
	defer p.mu.RUnlock()

	data := map[string]interface{}{
		"models": make(map[string]interface{}),
	}

	for model, stats := range p.modelStats {
		data["models"].(map[string]interface{})[model] = map[string]interface{}{
			"requests":       stats.Requests,
			"input_tokens":   stats.InputTokens,
			"output_tokens":  stats.OutputTokens,
			"total_duration": stats.TotalDuration,
			"last_active":    stats.LastActive,
		}
	}

	json.NewEncoder(w).Encode(data)
}

// HandleUsage returns current usage statistics
func (p *Proxy) HandleUsage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	p.mu.RLock()
	defer p.mu.RUnlock()

	data := map[string]interface{}{
		"total_requests":      p.totalMetrics.TotalRequests,
		"total_input_tokens":  p.totalMetrics.TotalInput,
		"total_output_tokens": p.totalMetrics.TotalOutput,
		"active_requests":     p.activeRequests,
		"uptime":              time.Since(p.totalMetrics.StartTime),
	}

	json.NewEncoder(w).Encode(data)
}

// GetActiveRequests returns current active request count
func (p *Proxy) GetActiveRequests() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.activeRequests
}

// GetTotalRequests returns total request count
func (p *Proxy) GetTotalRequests() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.totalMetrics.TotalRequests
}

// GetModelStats returns stats for a specific model
func (p *Proxy) GetModelStats(model string) (*ModelStats, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats, ok := p.modelStats[model]
	return stats, ok
}

// GetAllModelStats returns all model statistics
func (p *Proxy) GetAllModelStats() map[string]*ModelStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	result := make(map[string]*ModelStats)
	for k, v := range p.modelStats {
		result[k] = v
	}
	return result
}

// Reset clears all metrics
func (p *Proxy) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.modelStats = make(map[string]*ModelStats)
	p.activeRequests = 0
	p.totalMetrics = &TotalMetrics{
		StartTime: time.Now(),
	}
}

// createBackendRequest creates a new request to the Ollama backend.
func (p *Proxy) createBackendRequest(r *http.Request) (*http.Request, error) {
	targetURL, err := url.Parse(p.backendURL + r.URL.Path)
	if err != nil {
		return nil, err
	}
	targetURL.RawQuery = r.URL.RawQuery

	req, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL.String(), r.Body)
	if err != nil {
		return nil, err
	}
	req.Header = r.Header.Clone()
	return req, nil
}

// openAIResponse represents an OpenAI-compatible response from Ollama's /v1/ endpoints.
type openAIResponse struct {
	Model string `json:"model"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// extractMetrics extracts metrics from an Ollama response body.
// Supports both native NDJSON (/api/*) and OpenAI-compatible JSON (/v1/*).
func (p *Proxy) extractMetrics(body []byte) *MetricData {
	// Try OpenAI-compatible format first (single JSON object with "usage" field)
	var oai openAIResponse
	if err := json.Unmarshal(body, &oai); err == nil && oai.Usage.PromptTokens > 0 {
		return &MetricData{
			Model:        oai.Model,
			InputTokens:  oai.Usage.PromptTokens,
			OutputTokens: oai.Usage.CompletionTokens,
		}
	}

	// Fall back to Ollama native NDJSON format
	metrics := &MetricData{}
	lines := strings.Split(string(body), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}
		var event OllamaEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			metricExtractionErrors.WithLabelValues("", "parse_error").Inc()
			continue
		}
		if event.Model != "" {
			metrics.Model = event.Model
		}
		if event.PromptEvalCount > 0 {
			metrics.InputTokens = event.PromptEvalCount
		}
		if event.EvalCount > 0 {
			metrics.OutputTokens = event.EvalCount
		}
		if event.Done {
			metrics.Duration = time.Duration(event.TotalDuration)
		}
	}

	return metrics
}

// ServeHTTP forwards requests to the Ollama backend and records metrics.
// Metric extraction is best-effort: failures are counted but never affect the client response.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	worthy := p.metricsWorthy(r.URL.Path)

	// Only parse model from body for metrics-worthy requests
	var model string
	if worthy {
		model = p.extractModel(r)
	}

	// Track active requests
	p.mu.Lock()
	p.activeRequests++
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.activeRequests--
		p.mu.Unlock()
	}()

	// Forward to backend
	backendReq, err := p.createBackendRequest(r)
	if err != nil {
		log.Printf("ERROR: failed to create backend request: %v", err)
		http.Error(w, "Failed to create backend request", http.StatusInternalServerError)
		return
	}

	response, err := p.httpClient.Do(backendReq)
	if err != nil {
		log.Printf("ERROR: backend request failed: %v", err)
		http.Error(w, "Failed to forward request", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()

	// Read response body
	body, err := io.ReadAll(response.Body)
	if err != nil {
		log.Printf("ERROR: failed to read response body: %v", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}

	// ALWAYS forward response to client first — client is served after this block
	for key, values := range response.Header {
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
	w.WriteHeader(response.StatusCode)
	w.Write(body)

	// Best-effort metric extraction — only for known endpoints with success status
	if worthy && response.StatusCode >= 200 && response.StatusCode < 300 {
		p.safeExtractAndRecord(r.URL.Path, model, body, time.Since(startTime))
	}
}

// safeExtractAndRecord extracts and records metrics without ever panicking or blocking the caller.
// All failures are counted in metricExtractionErrors.
func (p *Proxy) safeExtractAndRecord(endpoint, model string, body []byte, duration time.Duration) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("ERROR: panic during metric extraction for %s: %v", endpoint, r)
			metricExtractionErrors.WithLabelValues(endpoint, "panic").Inc()
		}
	}()

	metrics := p.extractMetrics(body)
	if metrics == nil {
		log.Printf("WARN: no metrics extracted for %s model=%s", endpoint, model)
		metricExtractionErrors.WithLabelValues(endpoint, "empty_response").Inc()
		return
	}

	// Use model from response if request body parsing failed
	if model == "unknown" && metrics.Model != "" {
		model = metrics.Model
	}

	metrics.Model = model
	metrics.Endpoint = endpoint
	metrics.Category = p.categorizeRequest(endpoint)
	metrics.Duration = duration

	// Update internal stats
	p.mu.Lock()
	if _, ok := p.modelStats[model]; !ok {
		p.modelStats[model] = &ModelStats{}
	}
	p.modelStats[model].Requests++
	p.modelStats[model].InputTokens += metrics.InputTokens
	p.modelStats[model].OutputTokens += metrics.OutputTokens
	p.modelStats[model].LastActive = time.Now()
	p.totalMetrics.TotalRequests++
	p.totalMetrics.TotalInput += metrics.InputTokens
	p.totalMetrics.TotalOutput += metrics.OutputTokens
	p.totalMetrics.LastActive = time.Now()
	p.mu.Unlock()

	// Send to Prometheus channel — drop if full to avoid blocking
	if p.prometheusClient != nil && p.prometheusClient.MetricsCh != nil {
		select {
		case p.prometheusClient.MetricsCh <- *metrics:
		default:
			log.Printf("WARN: metrics channel full, dropping metric for %s", endpoint)
		}
	}
}

// StartHealthChecker probes the Ollama backend periodically and updates health state.
func (p *Proxy) StartHealthChecker(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	p.checkBackendHealth()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.checkBackendHealth()
		}
	}
}

func (p *Proxy) checkBackendHealth() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", p.backendURL+"/api/tags", nil)
	if err != nil {
		p.backendHealthy.Store(false)
		metrics.RecordBackendError(p.backendURL)
		return
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		p.backendHealthy.Store(false)
		metrics.RecordBackendError(p.backendURL)
		log.Printf("WARN: backend health check failed: %v", err)
		return
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		p.backendHealthy.Store(true)
		metrics.RecordBackendHealth(p.backendURL)
	} else {
		p.backendHealthy.Store(false)
		metrics.RecordBackendError(p.backendURL)
	}
}

// BackendHealthy returns the last known backend health state.
func (p *Proxy) BackendHealthy() bool {
	return p.backendHealthy.Load()
}
