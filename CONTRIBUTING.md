# Contributing to AETHER

Thank you for your interest in contributing to AETHER! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional. We welcome contributions from everyone.

## Getting Started

### Prerequisites

- **Node.js 18+** (20 recommended)
- **Python 3.9+** (3.11 recommended)
- **Git**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/chrbailey/aether.git
cd aether

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -e ".[dev]"

# Build TypeScript
npm run build

# Run all tests
npm run test:all
```

### Project Structure

```
aether/
├── core/                    # Python ML core
│   ├── encoder/            # Event encoding
│   ├── world_model/        # JEPA predictor
│   ├── critic/             # Uncertainty & calibration
│   ├── training/           # Training loops & losses
│   ├── inference/          # FastAPI server
│   └── tests/              # Python tests (303)
├── mcp-server/             # TypeScript MCP server
│   └── src/
│       ├── governance/     # Adaptive thresholds
│       ├── bridge/         # Python HTTP client
│       ├── tools/          # MCP tool definitions
│       ├── types/          # TypeScript types
│       └── __tests__/      # TypeScript tests (99)
├── scripts/                # Training & benchmark scripts
├── data/                   # Datasets & benchmarks
└── docs/                   # Documentation
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

Follow the coding standards below.

### 3. Run Tests

```bash
# TypeScript tests
npm test

# Python tests
npm run test:python

# Both
npm run test:all

# With coverage
npm run test:coverage
npm run test:python:coverage
```

### 4. Lint and Format

```bash
# TypeScript
npm run lint
npm run format

# Python
pip install ruff
ruff check core/ scripts/
ruff format core/ scripts/
```

### 5. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Features
git commit -m "feat(governance): add vocabulary-aware floor"

# Bug fixes
git commit -m "fix(bridge): handle connection timeout"

# Documentation
git commit -m "docs: update QUICKSTART guide"

# Tests
git commit -m "test(modulation): add vocab normalization tests"

# Chores
git commit -m "chore(deps): update vitest to 4.0.16"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### TypeScript

- **Strict mode**: All TypeScript uses strict mode
- **ES Modules**: Use `import/export`, not `require`
- **Type everything**: Avoid `any`, use proper types
- **Naming**: camelCase for variables/functions, PascalCase for types/classes
- **File structure**: One module per file, tests in `__tests__/`

```typescript
// Good
export function computeThreshold(value: number): number {
  return Math.max(0.5, Math.min(0.99, value));
}

// Avoid
export function computeThreshold(value: any) {
  return value > 0.99 ? 0.99 : value < 0.5 ? 0.5 : value;
}
```

### Python

- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Formatting**: Follow ruff/black style

```python
# Good
def compute_threshold(value: float, min_floor: float = 0.5) -> float:
    """Compute clamped threshold value.

    Args:
        value: Raw threshold value.
        min_floor: Minimum allowed value.

    Returns:
        Clamped threshold between min_floor and 0.99.
    """
    return max(min_floor, min(0.99, value))
```

### Tests

- **Descriptive names**: `it('returns base floor for small vocabularies')`
- **Arrange-Act-Assert**: Structure tests clearly
- **Edge cases**: Test boundary conditions
- **No flaky tests**: Use deterministic seeds for randomness

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally (`npm run test:all`)
- [ ] Linting passes (`npm run lint`)
- [ ] Types check (`npm run typecheck`)
- [ ] Documentation updated if needed
- [ ] No secrets or credentials in code

### PR Description

Include:
- **What**: Brief description of changes
- **Why**: Motivation/context
- **How**: High-level implementation approach
- **Testing**: How you tested the changes

### Review Process

1. Automated CI runs tests, linting, security scans
2. Maintainer reviews code
3. Address feedback
4. Squash and merge

## Adding Features

### New Governance Factor

1. Add configuration in `mcp-server/src/governance/aether.config.ts`
2. Implement computation in `mcp-server/src/governance/modulation.ts`
3. Add tests in `mcp-server/src/__tests__/modulation.test.ts`
4. Update documentation

### New Dataset Parser

1. Create `scripts/parse_<dataset>.py`
2. Follow existing parser patterns
3. Output to `data/external/<dataset>/`
4. Add to `scripts/generate_benchmark_report.py` DATASETS config

### New MCP Tool

1. Define in `mcp-server/src/tools/`
2. Add types in `mcp-server/src/types/`
3. Register in tool list
4. Add tests

## Reporting Issues

### Bug Reports

Include:
- AETHER version
- Node.js and Python versions
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Questions?

- Open a [Discussion](https://github.com/chrbailey/aether/discussions)
- Check existing [Issues](https://github.com/chrbailey/aether/issues)

Thank you for contributing!
