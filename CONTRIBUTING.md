# Contributing to AI Vision Pipeline

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU)

### Suggesting Features

We welcome feature suggestions! Please:
- Check existing issues first
- Clearly describe the use case
- Explain how it fits the project goals

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
4. **Run tests**:
   ```bash
   python -m unittest discover tests -v
   ```
5. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: your feature description"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/videos if applicable

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/modular-video-ai-pipeline.git
cd modular-video-ai-pipeline

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m unittest discover tests -v
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Documentation

- Update README.md if adding user-facing features
- Update ARCHITECTURE.md if changing system design
- Add inline comments for complex logic

## Questions?

Feel free to open an issue for discussion!

Thank you for contributing! 🎉
