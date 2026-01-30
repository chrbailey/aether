<p align="center">
  <strong>AETHER</strong><br>
  <em>Process-JEPA: Extending LeCun's Joint Embedding Architecture to Business Event Prediction</em>
</p>

<p align="center">
  <a href="https://github.com/christopherbailey/aether/actions/workflows/ci.yml"><img src="https://github.com/christopherbailey/aether/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="#jepa-for-processes">Why JEPA?</a> &bull;
  <a href="#the-problem">The Problem</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="docs/ARCHITECTURE.md">Architecture</a> &bull;
  <a href="docs/QUICKSTART.md">Quick Start</a>
</p>

---

## JEPA for Processes

If [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) can learn to predict the physical world from images and video, can it predict how business processes unfold?

**AETHER is JEPA implementation for discrete business event sequences.** It takes ideas from Yann LeCun's Joint Embedding Predictive Architecture and applies them to enterprise workflow prediction — purchase-to-pay, order-to-cash, and procurement processes.

The key ideas from the JEPA ecosystem that AETHER adapts:

| JEPA Concept | Original Domain | AETHER Application |
|---|---|---|
| Joint Embedding | Images (I-JEPA), Video (V-JEPA) | Business event sequences |
| Latent-space prediction | Pixel masking, frame prediction | Event transition: `f(z_t, action, variant) → z_{t+1}` |
| Energy-based scoring | LeCun's [EBM framework](https://proceedings.mlr.press/v1/lecun06a.html) (2006) | Process conformance anomaly detection |
| SIGReg loss | [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025) | Latent collapse prevention via eigenvalue regularization |
| VICReg loss | [VICReg](https://arxiv.org/abs/2105.04906) (Bardes, Ponce & LeCun, 2022) | Variance-Invariance-Covariance as alternative regularizer |

**The novel contribution:** AETHER combines these JEPA components with epistemic uncertainty decomposition and adaptive governance. The model decomposes uncertainty into what's _reducible_ (epistemic) vs. what's _inherently random_ (aleatoric), and uses that decomposition to dynamically tighten or relax governance thresholds — no static thresholds, no manual tuning. The system earns trust through demonstrated calibration.

> _"A possible path towards building a world model is to learn hierarchical representations of the world that capture both short-term and long-term dependencies."_ — LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) (2022)
>
> AETHER explores the complementary question: can JEPA model enterprise workflows, where the "world" is a structured sequence of business events?

---

## The Problem

Every AI governance system today uses **static thresholds**:
- Flag if confidence < 0.90
- Review if drift > 0.15
- Block if uncertainty > 0.80

These break immediately. A well-calibrated model gets held back by thresholds tuned for a bad one. A degrading model sails through gates set during its best day.

Worse: **not all uncertainty is equal**. A model that's uncertain because it hasn't seen enough data (epistemic) should trigger more review — human judgment helps. A model that's uncertain because the process is inherently random (aleatoric) should _not_ trigger more review — no amount of human oversight reduces coin-flip randomness.

No existing system makes this distinction.

---

## How It Works

### The Core Formula (v3)

```
effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
min_floor = 0.50 + 0.05 × log(vocab_size / 20) / log(4)   # v3: vocabulary-aware
```

Each factor is independently computed and composable:

| Factor | What it captures | Effect |
|--------|-----------------|--------|
| **Mode** | Operational context (flexible &rarr; strict &rarr; forbidden) | Symbolic governance from PromptSpeak modes |
| **Uncertainty** | Epistemic ratio of total uncertainty | Only _reducible_ uncertainty tightens governance |
| **Calibration** | Recent ECE/MCE/Brier score | Poorly calibrated models get tighter oversight |
| **Vocabulary** | Activity taxonomy complexity (v3) | High-vocab datasets get conservative floors |

The key insight: **aleatoric uncertainty is ignored in governance tightening.** This is the formal contribution. It means the system won't waste human attention on inherently random outcomes.

**v3 addition:** The vocabulary-aware minimum floor prevents regressions on high-activity datasets. At 80+ activities, the floor rises to match static thresholds, implementing a "do no harm" principle.

### Asymmetric Trust

Trust is earned slowly and lost quickly:

```
SUPERVISED ──[10 calibrated windows]──> GUIDED
GUIDED     ──[20 calibrated windows]──> COLLABORATIVE
COLLABORATIVE ──[50 calibrated windows]──> AUTONOMOUS

Any level ──[1 critical miss]──> immediate demotion
Any level ──[immutable violation]──> reset to SUPERVISED
```

This mirrors real-world trust: it takes months to build and seconds to destroy.

### Safety Floor

Some constraints never relax, regardless of trust level or calibration:
- **Forbidden mode** &rarr; always block
- **Sensitive data patterns** (SSN, API keys, private keys) &rarr; always hold
- **Dempster-Shafer conflict > 0.7** &rarr; always review
- **Circuit breaker floor** &rarr; 3+ consecutive failures = block
- **Uncertainty ceiling > 0.95** &rarr; always hold

---

## Quick Start

### TypeScript (Governance + MCP Server)

```bash
git clone https://github.com/christopherbailey/aether.git
cd aether
npm install
npm run build
npm test          # 99 tests — governance, modulation, bridge, tools, vocab-aware
```

### Python (ML Core)

```bash
pip install -e ".[dev]"              # Editable install with test dependencies
python -m pytest core/tests/ -v      # 303 tests — encoder, world model, critic, training, data
```

### Run Both Together

```bash
# Terminal 1: Python inference server
python -m core.inference.server          # Starts on localhost:8712

# Terminal 2: MCP server (connects to Claude, Cursor, etc.)
npm start
```

That's it. AETHER exposes 6 MCP tools that any AI assistant can call to get uncertainty-aware predictions and governance decisions.

---

## Architecture

```
                          MCP Tools (6)
                    predict_next_event
                    predict_outcome
                    get_calibration
                    get_autonomy_level
                    get_effective_thresholds
                    evaluate_gate
                              |
                    TypeScript MCP Server
                    ├── Governance Modulation ← aether.config.ts
                    ├── Autonomy Controller     (asymmetric trust)
                    ├── Immutable Constraints    (safety floor)
                    │         |
                    │    HTTP bridge (:8712)
                    │         |
                    Python FastAPI Server
                    ├── EventEncoder            (activity + time + context → 128D)
                    ├── TransitionModel          (JEPA predictor: z_t → z_{t+1})
                    ├── EnergyScorer            (energy-based anomaly scoring)
                    ├── HierarchicalPredictor    (activity / phase / outcome)
                    ├── LatentVariable          (Gumbel-Softmax path variants)
                    ├── UncertaintyDecomposer    (epistemic vs. aleatoric)
                    ├── CalibrationTracker       (ECE / MCE / Brier)
                    └── ConformalPredictor       (distribution-free prediction sets)
```

### Python Core (`core/`)

| Module | Purpose |
|--------|---------|
| `encoder/` | Event &rarr; 128D latent state via vocabularies + Time2Vec + causal transformer |
| `world_model/` | JEPA-style transition model with energy scoring and hierarchical predictions |
| `critic/` | Epistemic/aleatoric decomposition, calibration tracking, adaptive conformal inference |
| `training/` | VICReg + SIGReg loss functions, multi-loss training loop, checkpoints |
| `inference/` | FastAPI server with `/predict`, `/calibration`, `/health` endpoints |
| `data/` | Unified pipeline for SAP, BPI 2019, OCEL 2.0, and CSV event logs |

### TypeScript MCP Server (`mcp-server/`)

| Module | Purpose |
|--------|---------|
| `governance/` | Compositional modulation, autonomy state machine, immutable safety |
| `bridge/` | HTTP client to Python server with conservative fallbacks |
| `tools/` | 6 MCP tools for predictions, calibration, and governance decisions |
| `types/` | Full type system mirroring Python structures |

---

## Configuration

All governance tuning lives in one file: [`mcp-server/src/governance/aether.config.ts`](mcp-server/src/governance/aether.config.ts)

### Base Thresholds

```typescript
export const BASE_THRESHOLDS = {
  driftThreshold:       0.15,   // Concept drift detection
  reviewGateAutoPass:   0.92,   // Auto-pass confidence
  threatActivation:     0.60,   // Threat level activation
  conformanceDeviation: 0.05,   // Process conformance
  sayDoGap:             0.20,   // Say-Do consistency
  knowledgePromotion:   0.75,   // Knowledge promotion score
};
```

### Modulation Coefficients

```typescript
export const COEFFICIENTS = {
  modeStrength:          0.3,   // Governance mode sensitivity
  uncertaintyStrength:   0.5,   // Epistemic uncertainty sensitivity
  calibrationStrength:   0.4,   // Calibration quality sensitivity
};
```

### Clamp Bounds

Every threshold is bounded to prevent pathological behavior. See [`aether.config.ts`](mcp-server/src/governance/aether.config.ts) for the full configuration.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AETHER_PYTHON_URL` | `http://localhost:8712` | Python inference server URL |
| `AETHER_BPI2019_PATH` | — | Path to BPI 2019 dataset JSON file |

---

## MCP Tools

AETHER exposes 6 tools via the [Model Context Protocol](https://modelcontextprotocol.io):

| Tool | Description |
|------|-------------|
| `predict_next_event` | Next activity predictions with uncertainty decomposition and conformal sets |
| `predict_outcome` | Case outcome prediction (on-time, rework, remaining hours) |
| `get_calibration` | Current model calibration metrics (ECE, MCE, Brier) |
| `get_autonomy_level` | Trust state: SUPERVISED &rarr; GUIDED &rarr; COLLABORATIVE &rarr; AUTONOMOUS |
| `get_effective_thresholds` | All 6 adaptive thresholds with full modulation breakdown |
| `evaluate_gate` | Allow/hold/block decision with audit trail |

### Example: Claude Desktop Integration

Add to your `claude_desktop_config.json`:

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

## Key References

### JEPA Ecosystem (LeCun et al.)
- **JEPA** — LeCun, 2022. [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf). The foundational architecture.
- **LeJEPA** — Balestriero & LeCun, 2025. [Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544) (arXiv 2511.08544). SIGReg regularization via Epps-Pulley / random-projection. AETHER uses the eigenvalue formulation. ([Official repo](https://github.com/facebookresearch/LeJEPA))
- **I-JEPA** — Assran et al., CVPR 2023. Joint embedding for images.
- **V-JEPA** — Bardes et al., 2024. Joint embedding for video.
- **VICReg** — Bardes, Ponce & LeCun, ICLR 2022. [Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906).
- **Energy-Based Models** — LeCun et al., 2006. [A Tutorial on Energy-Based Learning](https://proceedings.mlr.press/v1/lecun06a.html). The theoretical framework for AETHER's anomaly scoring.

### Uncertainty & Calibration
- **Adaptive Conformal Inference** — Gibbs & Candes, NeurIPS 2021. Distribution-free prediction sets.
- **Law of Total Variance** — Classic. Epistemic/aleatoric uncertainty decomposition.

### Temporal Encoding
- **Time2Vec** — Kazemi et al., ICLR 2019. Continuous temporal encoding.

---

## Benchmark Results

AETHER has been evaluated on **10 process mining datasets** across 5 domains:

| Dataset | Domain | Cases | MCC Improvement | Notes |
|---------|--------|------:|:---------------:|-------|
| Road Traffic Fine | Government | 30,074 | **+266%** | Scale validation (150K total) |
| SAP Workflow | Enterprise | 2,896 | **+31.3%** | Best enterprise result |
| Wearable Tracker | Retail | 218 | **+17.8%** | O2C process |
| Sepsis | Healthcare | 210 | +2.3% | Clinical workflows |
| BPI 2019 | Finance | 500 | +0.6% | Procurement |
| BPIC 2012 | Finance | 500 | +0.4% | Loan applications |
| Judicial | Legal | 5 | 0.0% | Novel domain |
| BPI 2018 | Government | 2,000 | -2.2% | v3 floor applied |
| NetSuite 2025 | Finance | 274 | -3.3% | High class imbalance |
| SAP BSP669 | Enterprise | 767 | -24.0% | 77 activities (v3 candidate) |

**Key findings:**
- AETHER improves MCC on 7/10 datasets
- Largest improvement at scale: Road Traffic Fine (+266% on 150K cases)
- v3 vocabulary-aware floor reduces regressions on high-activity datasets

See [`docs/BENCHMARK_COMPARISON.md`](docs/BENCHMARK_COMPARISON.md) for detailed analysis.

---

## Testing

```bash
npm test                          # TypeScript: 99 tests
python -m pytest core/tests/ -v   # Python: 303 tests
npm run test:coverage             # TypeScript coverage report
npm run test:python:coverage      # Python coverage report
npm run test:all                  # Run everything
```

CI runs automatically on push to `main` and all PRs via GitHub Actions.

---

## License

[MIT](LICENSE) &mdash; Christopher Bailey, 2026
