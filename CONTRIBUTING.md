# Contributing to Addiction Risk Prediction

Thank you for your interest in contributing to the Addiction Risk Prediction project! This document provides guidelines for contributing to this machine learning project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

Before contributing, please:

1. Read through the project README to understand the project goals
2. Check existing issues and pull requests to avoid duplicates
3. Familiarize yourself with the codebase structure

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd AddictionRiskPredictionUsingPythonmain
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Verify installation by running the notebook or scripts

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature additions**: Add new functionality (e.g., new models, visualizations)
- **Documentation**: Improve README, add comments, create tutorials
- **Data preprocessing**: Enhance data cleaning and feature engineering
- **Model improvements**: Optimize hyperparameters, try new algorithms
- **Testing**: Add unit tests or validation scripts

### Contribution Workflow

1. **Fork the repository** to your GitHub account
2. **Create a new branch** for your feature:
   \`\`\`bash
   git checkout -b feature/your-feature-name
   \`\`\`
3. **Make your changes** following our code standards
4. **Test your changes** thoroughly
5. **Commit your changes** with clear messages:
   \`\`\`bash
   git commit -m "Add: Brief description of your changes"
   \`\`\`
6. **Push to your fork**:
   \`\`\`bash
   git push origin feature/your-feature-name
   \`\`\`
7. **Submit a Pull Request** with a detailed description

## Code Standards

### Python Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Example:

\`\`\`python
def calculate_risk_score(features: pd.DataFrame) -> np.ndarray:
    """
    Calculate addiction risk scores for given features.
    
    Args:
        features: DataFrame containing feature values
        
    Returns:
        Array of risk scores
    """
    # Implementation here
    pass
\`\`\`

### Jupyter Notebooks

- Clear markdown explanations before code cells
- Remove unnecessary output before committing
- Use descriptive cell titles
- Keep cells focused on single tasks

### Documentation

- Update README.md if adding new features
- Add inline comments for complex logic
- Include usage examples for new functions
- Document data requirements and formats

## Submitting Changes

### Pull Request Guidelines

Your pull request should:

1. **Have a clear title** describing the change
2. **Include a detailed description** explaining:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
3. **Reference related issues** (e.g., "Fixes #123")
4. **Include screenshots** for UI/visualization changes
5. **Pass all tests** (if applicable)

### Pull Request Template

\`\`\`markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Changes tested locally
- [ ] No breaking changes introduced
\`\`\`

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, etc.)
- **Error messages** or screenshots
- **Dataset information** (if relevant)

### Feature Requests

For feature requests, describe:

- **The problem** you're trying to solve
- **Proposed solution** or approach
- **Alternative solutions** considered
- **Use cases** and benefits

## Code Review Process

1. Maintainers will review your pull request
2. Feedback may be provided for improvements
3. Make requested changes and push updates
4. Once approved, your PR will be merged

## Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Reach out to project maintainers
- Check existing documentation and issues first

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Addiction Risk Prediction! Your efforts help improve mental health prediction and intervention strategies.
