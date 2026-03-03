# Ollama Metrics Proxy

A lightweight, transparent HTTP proxy for [Ollama](https://ollama.ai) that exposes Prometheus metrics for team usage monitoring.

Sits between your clients and Ollama, extracts token counts and latency from responses, and exposes them as Prometheus metrics. Zero client modifications required.

```
┌───────────┐     ┌──────────────┐     ┌───────────┐
│  Clients  │────▶│  Proxy       │────▶│  Ollama   │
│           │     │  :11434      │     │  :11435   │
└───────────┘     └──────────────┘     └───────────┘
                         │
                  ┌──────┴──────┐
                  │ Prometheus  │
                  │  /metrics   │
                  └─────────────┘
```

## Quick Start

```bash
# Build
make build

# Run (proxy on :11434, Ollama backend on another host/port)
./ollama-metrics --backend http://localhost:11435 --port 11434
```

## Metrics Exposed

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ollama_requests_total` | Counter | model, endpoint, category, status | Request count |
| `ollama_request_duration_seconds` | Histogram | model | Request latency |
| `ollama_tokens_generated` | Counter | model, token_type | Input/output token counts |
| `ollama_tokens_per_second` | Gauge | model | Token generation rate |
| `ollama_active_requests` | Gauge | model | In-flight requests |
| `ollama_backend_health` | Gauge | backend | Backend reachability (1/0) |
| `ollama_proxy_metric_extraction_errors_total` | Counter | endpoint, reason | Proxy self-health |

## Endpoints

| Path | Description |
|------|-------------|
| `/metrics` | Prometheus scrape endpoint |
| `/health` | Health check — 200 if backend reachable, 503 if not |
| `/models` | JSON summary of per-model stats |
| `/usage` | JSON summary of aggregate usage |
| `/*` | Everything else proxied to Ollama |

## CLI Flags

```
--backend   Ollama backend URL (default: http://localhost:11434)
--port      Proxy listen port (default: 8080)
--version   Print version and exit
```

## Design

- **Transparent first**: unknown paths are blindly forwarded — the proxy never breaks the client-Ollama contract
- **Graceful degradation**: metric extraction failures are logged and counted, never affect the response
- **Lightweight**: no SQLite, no OpenTelemetry, no CGO — single static binary
- **Health monitoring**: periodic backend probes update the `ollama_backend_health` gauge and `/health` endpoint
- **Graceful shutdown**: 10s drain on SIGINT/SIGTERM

## Building

```bash
make build        # Static binary (CGO_ENABLED=0)
make test         # Tests with race detector
make coverage     # HTML coverage report
make pre-commit   # fmt + vet + test
```

## License

MIT — see [LICENSE](LICENSE).
