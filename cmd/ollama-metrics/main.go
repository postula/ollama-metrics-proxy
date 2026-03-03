package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/postula/ollama-metrics-proxy/pkg/metrics"
	"github.com/postula/ollama-metrics-proxy/proxy"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	backendURL  = flag.String("backend", "http://localhost:11434", "Ollama backend URL")
	listenPort  = flag.Int("port", 8080, "Port to listen on")
	showVersion = flag.Bool("version", false, "Show version and exit")
)

var version = "dev"

func main() {
	flag.Parse()

	if *showVersion {
		fmt.Printf("ollama-metrics v%s\n", version)
		os.Exit(0)
	}

	metricsCh := make(chan proxy.MetricData, 100)
	p := proxy.New(*backendURL, &proxy.PrometheusClient{MetricsCh: metricsCh})

	// Start metrics processor
	go processMetrics(metricsCh)

	// Start server
	listenAddr := fmt.Sprintf(":%d", *listenPort)
	log.Printf("Starting Ollama Metrics Proxy on %s, backend: %s", listenAddr, *backendURL)

	// Set up HTTP handlers
	http.Handle("/metrics", promhttp.Handler())
	http.Handle("/health", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if p.BackendHealthy() {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status":"healthy","backend":"reachable"}`))
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"status":"degraded","backend":"unreachable"}`))
		}
	}))
	http.Handle("/models", http.HandlerFunc(p.HandleModels))
	http.Handle("/usage", http.HandlerFunc(p.HandleUsage))
	http.Handle("/", p)

	srv := &http.Server{
		Addr:    listenAddr,
		Handler: nil,
	}

	// Start health checker
	ctx, cancel := context.WithCancel(context.Background())
	go p.StartHealthChecker(ctx, 30*time.Second)

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down...")

		cancel()

		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()

		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("ERROR: forced shutdown: %v", err)
		}
	}()

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Failed to start server: %v", err)
	}

	log.Println("Server stopped")
}

func processMetrics(ch chan proxy.MetricData) {
	for m := range ch {
		metrics.RecordCompletedRequest(metrics.MetricData{
			Model:        m.Model,
			Endpoint:     m.Endpoint,
			Category:     m.Category,
			InputTokens:  m.InputTokens,
			OutputTokens: m.OutputTokens,
			Duration:     m.Duration,
		})
	}
}
