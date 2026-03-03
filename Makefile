.PHONY: all build test clean fmt vet coverage pre-commit

all: fmt vet test build

build:
	CGO_ENABLED=0 go build -ldflags="-w -s" -o ollama-metrics ./cmd/ollama-metrics

test:
	go test -v -race ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

clean:
	rm -f ollama-metrics coverage.out coverage.html

pre-commit: fmt vet test
