## Code of Conduct

This project has adopted the [Contributor Covenant Code of Conduct](./.github/CODE_OF_CONDUCT.md)

## Contributing

Contributions are welcome! Please read our [Contributing Guide](./.github/CONTRIBUTING.md) before submitting PRs.

### Development Setup

```bash
# Clone repository
git clone https://github.com/postula/ollama-metrics.git
cd ollama-metrics

# Install Go 1.22+
# Using system Go tooling or via Nix

# Run tests
make test
```

### Testing

All code must have unit tests. Target coverage >80%.

```bash
# Run all tests
make test

# Check coverage
make coverage

# Run integration tests (requires running Ollama server)
make integration
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make commits using Conventional Commits
4. Ensure all tests pass and coverage is >80%
5. Submit PR

## License

MIT - see LICENSE file for details.
