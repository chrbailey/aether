# AETHER

**Adaptive Epistemic Trust through Hierarchical Event Reasoning**

A system that learns adaptive governance thresholds from event sequences using LeCun-inspired world models with principled uncertainty decomposition.

## What It Does

AETHER predicts what happens next in business processes (SAP procurement, order-to-cash, etc.) and decomposes its uncertainty into **epistemic** (reducible with more data) vs. **aleatoric** (irreducible randomness). This decomposition drives governance: high epistemic uncertainty tightens oversight, while high aleatoric uncertainty does not.

### Key Components

- **Event Encoder** — Transforms business events into latent representations using activity/resource vocabularies and Time2Vec temporal encoding
- **World Model** — Predicts future latent states given current state + governance action, with path variant sampling for multiple futures
- **Critic** — Calibration tracking (ECE/MCE/Brier), epistemic/aleatoric decomposition via ensemble disagreement, and adaptive conformal prediction sets
- **MCP Server** — Exposes governance tools via the [Model Context Protocol](https://modelcontextprotocol.io) for integration with AI assistants

## Architecture

```
core/                   Python ML components
├── encoder/            Event encoding (activity vocab, Time2Vec)
├── world_model/        Transition, energy scoring, hierarchical prediction
├── critic/             Calibration, uncertainty decomposition, conformal prediction
├── training/           Data loading, loss functions, training loop
├── inference/          FastAPI inference server
├── data/               Data pipeline (SAP, BPI 2019, OCEL, CSV loaders)
└── tests/              Python test suite

mcp-server/             TypeScript MCP server
├── src/governance/     Modulation, autonomy controller, immutable safety floors
├── src/bridge/         Pinecone, PromptSpeak, Python inference bridges
├── src/tools/          MCP tool definitions (governance, calibration, predict)
├── src/types/          Type definitions
└── src/__tests__/      TypeScript test suite
```

## Quick Start

### TypeScript MCP Server

```bash
npm install
npm run build
npm test

# Start MCP server
npm start
```

### Python ML Core

```bash
pip install -r requirements.txt
cd core && python -m pytest tests/ -v

# Start inference server
python -m core.inference.server
```

## Governance Model

The governance modulation formula:

```
effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
```

Where:
- **mode_factor** scales by operational mode (flexible → standard → strict → forbidden)
- **uncertainty_factor** tightens governance when epistemic uncertainty is high
- **calibration_factor** adjusts based on historical prediction accuracy

### Safety Boundaries

- **Immutable floor**: `forbidden` mode, sensitive data handling, Dempster-Shafer conflict > 0.7, and circuit breaker floor cannot be overridden
- **Asymmetric trust**: Slow ascent (sustained calibration required), fast descent (single critical failure triggers lockdown)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AETHER_PYTHON_URL` | `http://localhost:8712` | Python inference server URL |

## Requirements

- **Node.js** ≥ 20.0.0
- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0.0

## License

[MIT](LICENSE)
