# Contributing to AETHER

Thanks for your interest in contributing to AETHER (Process-JEPA). This guide covers development setup, testing, and the PR process.

## Development Setup

AETHER has two codebases that work together:

### TypeScript (Governance + MCP Server)

```bash
git clone https://github.com/christopherbailey/aether.git
cd aether
npm install
npm run build
```

Requires Node.js >= 20.

### Python (ML Core)

```bash
pip install -e ".[dev]"    # Editable install with test dependencies (httpx, pytest, etc.)
```

Requires Python >= 3.11 and PyTorch >= 2.0.

## Running Tests

```bash
# TypeScript — 92 tests (governance, modulation, bridge, tools)
npm test

# Python — 303 tests (encoder, world model, critic, training, data)
python -m pytest core/tests/ -v

# Everything
npm run test:all
```

All tests must pass before submitting a PR. CI runs both suites automatically.

## Making Changes

1. **Fork** the repository
2. **Branch** from `main`: `git checkout -b your-feature`
3. **Make your changes** — see code style below
4. **Test** — run both TypeScript and Python test suites
5. **PR** — open a pull request against `main` with a clear description

## Code Style

### TypeScript
- Strict mode enabled (`strict: true` in tsconfig)
- ES modules (`"type": "module"` in package.json)
- Use the existing type system in `mcp-server/src/types/`

### Python
- Type hints on all function signatures
- f-strings for string formatting
- `pathlib` over `os.path`
- Follow existing patterns in `core/` modules

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a full walkthrough of the system — data flow, module responsibilities, key algorithms, and the theoretical foundations mapping to the JEPA framework.

## Questions?

Open an issue on GitHub. For questions about the JEPA-specific components, see the [Theoretical Foundations](docs/ARCHITECTURE.md#theoretical-foundations) section of the architecture doc.
