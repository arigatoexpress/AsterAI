# Contributing to AsterAI Trading Platform

Thank you for your interest in contributing to the AsterAI Trading Platform! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker (for testing)
- Google Cloud SDK (for deployment testing)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/AsterAI.git
   cd AsterAI
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Run tests before submitting a pull request:
  ```bash
  pytest tests/
  ```

### Documentation

- Update documentation for any new features
- Include examples in docstrings
- Update README.md if necessary

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

3. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request on GitHub

### Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Request review from maintainers

## Issue Reporting

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

## Feature Requests

For feature requests, please:

- Check existing issues first
- Provide a clear use case
- Explain the expected benefit
- Consider implementation complexity

## Security

- Do not commit API keys or secrets
- Report security vulnerabilities privately
- Follow responsible disclosure practices

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or start a discussion for any questions about contributing.
