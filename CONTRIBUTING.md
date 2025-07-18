# Contributing to TAGT

We welcome contributions to the TAGT project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed descriptions
- Include reproduction steps

### Pull Requests
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where appropriate

### Testing
- Run existing tests before submitting
- Add tests for new features
- Ensure all tests pass

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/SLE-TAGT-Prediction.git
cd SLE-TAGT-Prediction

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run quality checks
python quality_assurance/cto_review_checklist.py
```

## Questions?

Feel free to open an issue for any questions about contributing.
