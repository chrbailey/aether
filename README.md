<p align="center">
  <strong>AETHER</strong><br>
  <em>Know when to automate vs. escalate</em>
</p>

<p align="center">
  <a href="https://github.com/chrbailey/aether/actions/workflows/ci.yml"><img src="https://github.com/chrbailey/aether/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/chrbailey/aether/security/code-scanning"><img src="https://img.shields.io/badge/CodeQL-enabled-brightgreen" alt="CodeQL"></a>
  <a href="https://github.com/chrbailey/aether/releases/latest"><img src="https://img.shields.io/github/v/release/chrbailey/aether" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  &nbsp;
  <a href="#the-problem">The Problem</a> &bull;
  <a href="#what-aether-does">What It Does</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#use-cases">Use Cases</a> &bull;
  <a href="docs/ARCHITECTURE.md">Architecture</a>
</p>

---

## The Confidence Layer for Process Automation

Every automation system asks: _"Should I handle this, or escalate to a human?"_

Most answer with **static thresholds** — if confidence < 90%, flag for review. This breaks in two ways:

1. **A well-calibrated model gets held back** by thresholds tuned for a bad one
2. **Not all uncertainty is equal** — sometimes more data would help, sometimes the process is just inherently random

AETHER solves this by decomposing uncertainty into what's *reducible* (more data helps) vs. what's *irreducible* (inherently random), then **dynamically adjusting governance** based on demonstrated model performance.

The result: fewer unnecessary escalations, fewer missed issues.

---

## The Problem

Your invoice processing bot flags 40% of invoices for human review because "confidence is below threshold."

But when you dig in:
- **Half those flags** are cases where the model just hasn't seen enough similar invoices yet — _training on more examples would help_
- **The other half** are vendors with legitimately unpredictable behavior — _no amount of human review changes a coin flip_

Static thresholds can't tell the difference. AETHER can.

---

## What AETHER Does

| Capability | What It Means For You |
|------------|----------------------|
| **Predicts next activities** | Know what happens next in your order-to-cash, purchase-to-pay, or procurement process |
| **Quantifies uncertainty** | Not just "80% confident" but "60% of that uncertainty is reducible with more data" |
| **Adapts governance thresholds** | Thresholds tighten when the model is uncertain, relax when it's proven accurate |
| **Provides prediction sets** | Instead of one answer, get a calibrated set of likely outcomes with coverage guarantees |
| **Tracks its own calibration** | Know when the model is degrading before it causes problems |

### The Core Formula

```
effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
```

- **Only reducible uncertainty tightens governance** — the system won't waste human attention on inherently random outcomes
- **Trust is earned slowly, lost quickly** — 10 good windows to level up, 1 critical miss to demote
- **Some constraints never relax** — sensitive data patterns, high-conflict decisions, and circuit breakers always trigger review

---

## Use Cases

### Invoice Processing
```
Invoice arrives → AETHER predicts: approve/reject/needs-info
                → Uncertainty: 15% total (12% epistemic, 3% aleatoric)
                → Decision: HIGH epistemic ratio → route to human
                           (more training data on this vendor type would help)
```

### Order Fulfillment
```
Order in progress → AETHER predicts: on-time (73%), late (27%)
                  → Uncertainty: 8% total (1% epistemic, 7% aleatoric)
                  → Decision: LOW epistemic ratio → trust the prediction
                             (this route is just variable, human review won't help)
                  → Action: Trigger proactive customer notification if >25% late probability
```

### Loan Applications
```
Application submitted → AETHER predicts: next step is "credit_check" (89%)
                      → Conformal set: {credit_check, document_request} (90% coverage)
                      → Governance: AUTO-APPROVE routing
                                   (model well-calibrated, low uncertainty)
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/chrbailey/aether.git
cd aether

# TypeScript (governance + MCP server)
npm install && npm run build

# Python (ML core)
pip install -e ".[dev]"
```

### Run

```bash
# Terminal 1: Python inference server
python -m core.inference.server          # Starts on localhost:8712

# Terminal 2: MCP server (connects to Claude, Cursor, etc.)
npm start
```

### Test

```bash
npm test                          # 99 TypeScript tests
python -m pytest core/tests/ -v   # 303 Python tests
```

---

## MCP Tools

AETHER exposes 7 tools via the [Model Context Protocol](https://modelcontextprotocol.io):

| Tool | What You Get |
|------|--------------|
| `predict_next_event` | Top-K next activities with probabilities + uncertainty decomposition + conformal set |
| `predict_outcome` | Will this case be on-time, late, or need rework? With confidence intervals |
| `get_calibration` | Is the model trustworthy right now? ECE, MCE, Brier scores |
| `get_autonomy_level` | Current trust level: SUPERVISED → GUIDED → COLLABORATIVE → AUTONOMOUS |
| `get_effective_thresholds` | All 6 adaptive thresholds with full breakdown of why |
| `evaluate_gate` | Should this specific decision be auto-approved, held, or blocked? |
| `get_production_metrics` | Latency, prediction counts, calibration drift alerts |

### Claude Desktop Integration

```json
{
  "mcpServers": {
    "aether": {
      "command": "node",
      "args": ["/path/to/aether/mcp-server/dist/index.js"]
    }
  }
}
```

---

## How It Works

AETHER uses a **JEPA-style architecture** (Joint Embedding Predictive Architecture) adapted for business event sequences.

> JEPA was developed by Yann LeCun's team for learning world models from images and video. AETHER asks: can the same approach model enterprise workflows, where the "world" is a structured sequence of business events?

### Architecture

```
                          MCP Tools (7)
                    predict_next_event
                    predict_outcome
                    get_calibration
                    get_autonomy_level
                    get_effective_thresholds
                    evaluate_gate
                    get_production_metrics
                              |
                    TypeScript MCP Server
                    ├── Governance Modulation    (adaptive thresholds)
                    ├── Autonomy Controller      (asymmetric trust)
                    ├── Immutable Constraints    (safety floor)
                    │         |
                    │    HTTP bridge (:8712)
                    │         |
                    Python FastAPI Server
                    ├── EventEncoder            (activity + time + context → 128D)
                    ├── TransitionModel         (JEPA predictor: z_t → z_{t+1})
                    ├── EnergyScorer            (energy-based anomaly scoring)
                    ├── HierarchicalPredictor   (activity / phase / outcome)
                    ├── UncertaintyDecomposer   (epistemic vs. aleatoric)
                    ├── CalibrationTracker      (ECE / MCE / Brier)
                    └── ConformalPredictor      (distribution-free prediction sets)
```

### Key Technical Concepts

| Concept | What It Means |
|---------|---------------|
| **Epistemic uncertainty** | Model doesn't know — more data would help |
| **Aleatoric uncertainty** | Process is random — more data won't help |
| **Conformal prediction** | Calibrated prediction sets with coverage guarantees (e.g., "90% of the time, the true answer is in this set") |
| **Adaptive thresholds** | Governance limits that tighten/loosen based on demonstrated model performance |

### Safety Floor (Never Relaxed)

Some constraints are immutable regardless of trust level:
- **Forbidden mode** → always block
- **Sensitive data patterns** (SSN, API keys) → always hold for human review
- **Dempster-Shafer conflict > 0.7** → always review (the model is confused)
- **Circuit breaker** (3+ consecutive failures) → block
- **Total uncertainty > 0.95** → hold

---

## Benchmark Results

Evaluated on **11 process mining datasets**:

| Dataset | Domain | Cases | Activity Accuracy | Notes |
|---------|--------|------:|:-----------------:|-------|
| **BPI 2017** | Finance (Loans) | 31,509 | **70.4%** | Primary benchmark — simple rules only achieve 49.6% |
| Road Traffic Fine | Government | 150,370 | 81.6% | High volume, deterministic outcomes |
| SAP Workflow | Enterprise | 2,896 | 68.2% | Best enterprise result |

See [`docs/BENCHMARK_COMPARISON.md`](docs/BENCHMARK_COMPARISON.md) for full analysis.

---

## Compared To Alternatives

| Approach | Limitation | AETHER Advantage |
|----------|-----------|------------------|
| **Celonis / UiPath** | Static thresholds, no uncertainty decomposition | Adaptive governance that learns |
| **Custom ML models** | Raw confidence scores without calibration | Calibrated predictions with coverage guarantees |
| **PM4Py** | Discovery only, no prediction layer | Full prediction + governance stack |
| **Rules engines** | Brittle, manual tuning required | Self-adjusting based on performance |

---

## Research Foundation

AETHER builds on established research:

- **JEPA** — LeCun, 2022. [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)
- **Conformal Prediction** — Gibbs & Candes, NeurIPS 2021. Distribution-free prediction sets
- **VICReg / SIGReg** — Bardes, Balestriero & LeCun. Latent collapse prevention
- **Energy-Based Models** — LeCun et al., 2006. Anomaly scoring framework

---

## License

[MIT](LICENSE) — Christopher Bailey, 2026
