# Contributing to SLE Flare Prediction Project

Thank you for your interest in contributing to the SLE Flare Prediction Project! We welcome contributions from the community to help improve this medical AI tool for predicting systemic lupus erythematosus (SLE) flares.

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior to imaduddin.dev@gmail.com.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features
- Before creating a new issue, please search existing issues to avoid duplicates
- Provide detailed information including:
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Environment details (OS, Python version, etc.)
  - Any relevant error messages or logs

### Submitting Code Changes

1. **Fork the Repository**: Create your own fork of the repository by clicking the 'Fork' button on GitHub.

2. **Clone the Forked Repository**:
   ```bash
   git clone https://github.com/YourUsername/sle-flare-prediction.git
   cd sle-flare-prediction
   ```

3. **Set up Development Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Create a Branch**: Create a new branch for your improvements:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make Changes**: Implement your changes following our coding standards.

6. **Test Your Changes**: Run the tests to make sure everything is working:
   ```bash
   pytest tests/
   ```

7. **Commit Your Changes**: Commit your changes with a meaningful message:
   ```bash
   git commit -am 'Add: Description of changes'
   ```

8. **Push to Your Fork**: Push your branch to your GitHub fork:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request**: Go to the original repository on GitHub and click 'New Pull Request'.

## Development Guidelines

### Coding Standards

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Write docstrings for all public functions and classes
- Keep functions focused and reasonably sized
- Add type hints where appropriate
- Ensure code is well-commented, especially for complex algorithms

### Testing

- Write unit tests for new functionality
- Ensure all existing tests continue to pass
- Test coverage should be maintained or improved
- Include integration tests for end-to-end workflows

### Documentation

- Update README.md if your changes affect installation or usage
- Add docstrings to new functions and classes
- Update any relevant documentation files
- Consider adding examples for new features

### Medical Data and Privacy

This project deals with sensitive medical data. Please ensure:

- **No real patient data** is included in commits
- Use synthetic or anonymized data for testing
- Follow HIPAA and other relevant privacy guidelines
- Be mindful of data security in all contributions

### Clinical Validation

For changes affecting the prediction model or clinical outputs:

- Provide evidence or references supporting the changes
- Include validation metrics and comparisons
- Consider consulting with medical professionals
- Ensure changes maintain or improve clinical accuracy

## Types of Contributions We Welcome

- Bug fixes
- Feature enhancements
- Documentation improvements
- Test coverage improvements
- Performance optimizations
- UI/UX improvements
- Data preprocessing enhancements
- Model improvements (with proper validation)

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update version numbers following semantic versioning
3. Your pull request will be reviewed by maintainers
4. Address any feedback or requested changes
5. Once approved, your changes will be merged

## Getting Help

If you need help or have questions:

- Check the documentation first
- Search existing issues and discussions
- Create a new issue with the "question" label
- Reach out to maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to improving SLE patient care through technology!
