# Contributing to Flood Forecasting ML

We appreciate your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/yourusername/flood-forecasting-ml.git
   cd flood-forecasting-ml
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

## Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Use type hints where appropriate

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (e.g., "Add", "Fix", "Update")
- Reference issues where applicable (e.g., "Fixes #123")

Example:
```
Add LSTM model implementation for streamflow prediction
- Implements bidirectional LSTM architecture
- Includes dropout regularization
- Fixes #45
```

### Pull Requests

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add a clear description of changes
4. Link related issues

## Testing

Before submitting a PR, ensure:
- Your code runs without errors
- Model training completes successfully
- Predictions are reasonable

## Documentation

- Update README.md if adding new features
- Add comments for complex logic
- Update docstrings when modifying functions

## Report Issues

Please report bugs and suggest features using GitHub Issues. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## Questions?

Feel free to open an issue or discussion for questions about the project.

---

Thank you for contributing to flood forecasting research!
