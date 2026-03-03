package tests

import (
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// MockTransport is a mock HTTP transport for testing
type MockTransport struct {
	mu              sync.Mutex
	Calls           []MockCall
	DefaultResponse *MockResponse
}

type MockCall struct {
	Method string
	Path   string
	Body   []byte
}

type MockResponse struct {
	StatusCode int
	Body       string
	Latency    int // milliseconds delay before response
}

// NewMockTransport creates a new mock transport for testing
func NewMockTransport() *MockTransport {
	return &MockTransport{
		Calls: make([]MockCall, 0),
		DefaultResponse: &MockResponse{
			StatusCode: 200,
			Body:       `{"status": "ok"}`,
			Latency:    0,
		},
	}
}

// RoundTrip implements http.RoundTripper
func (m *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Record call
	var body []byte
	if req.Body != nil {
		body, _ = io.ReadAll(req.Body)
	}
	m.Calls = append(m.Calls, MockCall{
		Method: req.Method,
		Path:   req.URL.Path,
		Body:   body,
	})

	// Apply latency delay
	if m.DefaultResponse.Latency > 0 {
		time.Sleep(time.Duration(m.DefaultResponse.Latency) * time.Millisecond)
	}

	return &http.Response{
		StatusCode: m.DefaultResponse.StatusCode,
		Body:       io.NopCloser(strings.NewReader(m.DefaultResponse.Body)),
		Header:     make(http.Header),
		Request:    req,
	}, nil
}

// MockResponseBuilder provides a fluent interface for building mock responses
func MockResponseBuilder() *mockResponseBuilder {
	return &mockResponseBuilder{
		response: &MockResponse{
			StatusCode: 200,
			Body:       `{"status": "ok"}`,
			Latency:    0,
		},
	}
}

type mockResponseBuilder struct {
	response *MockResponse
}

func (b *mockResponseBuilder) WithStatus(code int) *mockResponseBuilder {
	b.response.StatusCode = code
	return b
}

func (b *mockResponseBuilder) WithBody(body string) *mockResponseBuilder {
	b.response.Body = body
	return b
}

func (b *mockResponseBuilder) WithLatency(ms int) *mockResponseBuilder {
	b.response.Latency = ms
	return b
}

func (b *mockResponseBuilder) Build() *MockResponse {
	return b.response
}

// MockResponseForEvent creates an Ollama streaming response for specific event types
func MockStreamingResponse(eventType string, body string) MockResponse {
	switch eventType {
	case "start":
		return MockResponse{
			StatusCode: 200,
			Body:       `{"model":"mamba-7b","created_at":"2024-02-28T12:00:00Z","message":{"role":"assistant","content":""}}\n\n`,
			Latency:    10,
		}
	case "prompt_start":
		return MockResponse{
			StatusCode: 200,
			Body: `{"model":"mamba-7b","created_at":"2024-02-28T12:00:00Z","prompt_eval_count":100,"prompt_eval_duration":5000000}

`,
			Latency: 5,
		}
	case "token":
		return MockResponse{
			StatusCode: 200,
			Body:       `{"model":"mamba-7b","created_at":"2024-02-28T12:00:00Z","message":{"role":"assistant","content":"Hello"}}\n\n`,
			Latency:    20,
		}
	case "done":
		return MockResponse{
			StatusCode: 200,
			Body: `{"model":"mamba-7b","created_at":"2024-02-28T12:00:00Z","eval_count":50,"eval_duration":1000000,"total_duration":6000000}

`,
			Latency: 10,
		}
	}
	return MockResponse{
		StatusCode: 200,
		Body:       `{"status":"ok"}`,
		Latency:    0,
	}
}
