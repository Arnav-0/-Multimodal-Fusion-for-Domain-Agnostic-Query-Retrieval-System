# Contributing to Multimodal Document Q&A System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Screenshots** (if applicable)
- **Environment details**:
  - OS and version
  - Python version
  - CUDA version (if using GPU)
  - Relevant package versions

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear use case** - Why is this enhancement needed?
- **Proposed solution** - How should it work?
- **Alternatives considered** - What other approaches did you think about?
- **Additional context** - Screenshots, mockups, or examples

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Write clear commit messages**
6. **Submit a pull request**

## 📝 Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function parameters and return values
- Write **docstrings** for classes and functions
- Keep functions **focused and small**
- Use **meaningful variable names**

Example:
```python
def extract_text_from_pdf(pdf_path: str, max_chars: int = 3000) -> List[str]:
    """
    Extract text from PDF with character limit per page.
    
    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum characters per page
        
    Returns:
        List of text strings, one per page
    """
    # Implementation here
    pass
```

### Code Organization

- **Imports**: Group into stdlib, third-party, local
- **Constants**: Define at module top in UPPER_CASE
- **Classes**: Use CamelCase
- **Functions**: Use snake_case
- **Private**: Prefix with underscore (_private_function)

### Error Handling

- Use **specific exceptions** (not bare `except:`)
- **Log errors** with appropriate level
- **Provide context** in error messages
- **Clean up resources** (use context managers)

### Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for high code coverage
- Test edge cases

## 🔄 Development Workflow

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/multimodal-qa-system.git
cd multimodal-qa-system

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/multimodal-qa-system.git

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up .env
copy .env.example .env
# Edit .env with your API keys
```

### Making Changes

```bash
# Create a new branch
git checkout -b feature/my-new-feature

# Make your changes
# ... edit files ...

# Test your changes
python verify_setup.py
python evaluate_all_fusions.py  # If applicable

# Commit your changes
git add .
git commit -m "Add feature: description of changes"

# Push to your fork
git push origin feature/my-new-feature
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream main into your branch
git checkout main
git merge upstream/main

# Update your fork
git push origin main
```

## 📋 Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(fusion): add weighted score combination for hybrid mode

Implement adaptive weight calculation based on retrieval confidence.
This improves answer quality by 12% in benchmark tests.

Closes #123
```

```
fix(api): handle timeout errors in Gemini API calls

Add exponential backoff with jitter when rate limited.
Update error messages to be more user-friendly.

Fixes #456
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_fusion.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

```python
import pytest
from main_latefusion import extract_text_and_images

def test_extract_text_basic():
    """Test basic text extraction from PDF"""
    # Arrange
    pdf_path = "tests/fixtures/sample.pdf"
    
    # Act
    result = extract_text_and_images(pdf_path)
    
    # Assert
    assert len(result["pages_text"]) > 0
    assert all(isinstance(text, str) for text in result["pages_text"])

def test_extract_text_empty_pdf():
    """Test handling of empty PDF"""
    with pytest.raises(ValueError, match="No pages found"):
        extract_text_and_images("tests/fixtures/empty.pdf")
```

## 📚 Documentation Guidelines

### Code Documentation

- **Docstrings**: Use Google or NumPy style
- **Inline comments**: Explain "why", not "what"
- **Type hints**: Always include for public functions
- **Examples**: Provide usage examples for complex functions

### README Updates

When adding features, update:

- Features list
- Usage examples
- Configuration options
- Troubleshooting section

### Documentation Files

Update relevant docs when changing:

- `STARTUP_GUIDE.md` - Setup instructions
- `SETTINGS_FEATURE.md` - Configuration options
- `QUALITY_IMPROVEMENTS.md` - Performance enhancements

## 🎯 Areas Needing Contributions

### High Priority

- [ ] Add support for DOCX and PPT documents
- [ ] Implement conversation history
- [ ] Create Docker containerization
- [ ] Add streaming API responses
- [ ] Improve error messages

### Medium Priority

- [ ] Add unit tests for all modules
- [ ] Create API client library (Python)
- [ ] Support for local LLMs (Llama, Mistral)
- [ ] Multi-document comparison feature
- [ ] Batch processing API

### Documentation

- [ ] Video tutorials
- [ ] API usage examples
- [ ] Deployment guides (AWS, Azure, GCP)
- [ ] Performance tuning guide
- [ ] FAQ section

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**Import Errors**: Check virtual environment is activated
**GPU Errors**: Verify CUDA installation with `python -c "import torch; print(torch.cuda.is_available())"`
**API Errors**: Check `.env` file has correct API keys

### Useful Commands

```bash
# Check Python path
python -c "import sys; print(sys.executable)"

# List installed packages
pip list

# Check CUDA
nvidia-smi

# Test Gemini API
python -c "import google.generativeai as genai; print('OK')"
```

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create a GitHub Issue
- **Security**: Email security@example.com (do not open public issue)

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Any unprofessional conduct

## 🏆 Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project README

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make this project better! 🎉
