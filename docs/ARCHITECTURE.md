# AETHER Architecture

## Overview

AETHER is a two-tier system: a **Python ML core** that encodes business events, predicts futures, and quantifies uncertainty, and a **TypeScript MCP server** that exposes governance decisions to AI assistants.

```
TypeScript MCP Server (6 tools)
    │ HTTP via python-bridge.ts
    ▼
Python FastAPI Inference Server (:8712)
    ├── EventEncoder        Structural + Temporal + Context → 128D latent
    ├── TransitionModel     Residual MLP: f(z_t, action, variant) → z_{t+1}
    ├── EnergyScorer        L2 latent distance → normalized anomaly score
    ├── HierarchicalPredictor  Activity / Phase / Outcome prediction heads
    ├── LatentVariable      Gumbel-Softmax categorical path variants
    ├── ConformalPredictor  Adaptive Prediction Sets with coverage guarantee
    ├── CalibrationTracker  ECE / MCE / Brier reliability metrics
    └── UncertaintyDecomposer  Ensemble variance → epistemic vs. aleatoric
```

---

## Data Flow

```
Raw Events (SAP, BPI 2019, CSV, OCEL)
    ↓
EventEncoder
  StructuralEncoder: activity/resource embeddings + numerical projection → 128D
  TemporalEncoder:   fuse structural encoding with Time2Vec temporal embedding → 128D
  ContextEncoder:    causal transformer with positional encoding → 128D latent states
    ↓
Latent States z_t ∈ R^128
    ↓
HierarchicalPredictor
  ActivityHead:  next activity probabilities + expected delta hours
  PhaseHead:     current/next phase + transition time
  OutcomeHead:   on-time probability, rework probability, remaining hours
    ↓
Ensemble (5 forward passes with noise)
  → decompose_from_ensemble() splits total variance into epistemic + aleatoric
    ↓
AdaptiveConformalPredictor
  → prediction set with coverage guarantee (Gibbs & Candes ACI)
    ↓
GovernanceModulation (TypeScript)
  effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
    ↓
GateDecision: allow / hold / block + audit trail
```

---

## Python Modules

### `core/encoder/`

| File | Class | Purpose |
|------|-------|---------|
| `event_encoder.py` | `StructuralEncoder` | Embeds categorical activity/resource, projects numerical attributes → 128D |
| `event_encoder.py` | `TemporalEncoder` | Fuses structural encoding with Time2Vec temporal embedding → 128D |
| `event_encoder.py` | `ContextEncoder` | Causal transformer over event sequence → 128D latent states |
| `time2vec.py` | `Time2Vec` | Continuous-time positional encoding: linear + periodic sine components |
| `vocabulary.py` | `ActivityVocabulary`, `ResourceVocabulary` | Token-to-index mapping with embedding layers |

### `core/world_model/`

| File | Class | Purpose |
|------|-------|---------|
| `transition.py` | `TransitionModel` | `f(z_t, a_t, c_t) → z_{t+1}` with residual MLP blocks. Supports `predict_multiple_futures()` |
| `hierarchical.py` | `HierarchicalPredictor` | Combines ActivityHead, PhaseHead, OutcomeHead for multi-timescale predictions |
| `energy.py` | `EnergyScorer` | L2 distance in latent space, normalized via sigmoid → [0, 1] anomaly score |
| `latent.py` | `LatentVariable` | Gumbel-Softmax for categorical path variants (standard, credit_hold, rework, expedited, exception, unknown) |

### `core/critic/`

| File | Class/Function | Purpose |
|------|----------------|---------|
| `decomposition.py` | `decompose_from_ensemble()` | Law of total variance: epistemic = Var(E[Y\|M]), aleatoric = E[Var(Y\|M)] |
| `calibration.py` | `CalibrationTracker` | Per-bucket reliability with ECE, MCE, Brier score |
| `adaptive_conformal.py` | `AdaptiveConformalPredictor` | ACI update rule: `α_{t+1} = α_t + γ·(α_target - err_t)` |

### `core/data/`

| File | Class | Purpose |
|------|-------|---------|
| `unified_pipeline.py` | `AetherDataPipeline` | Loads SAP SQLite, BPI 2019 JSON, CSV event logs, OCEL 2.0 SQLite. Builds vocabularies, splits train/val, creates PyTorch datasets |

### `core/training/`

| File | Purpose |
|------|---------|
| `train.py` | Training loop with validation |
| `losses.py` | Loss computation (activity CE, outcome BCE, energy regularization) |
| `data_loader.py` | `EventSequenceDataset` — PyTorch dataset with padding/batching |
| `run_training.py` | CLI entrypoint |

### `core/inference/`

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server on `:8712` with `/predict`, `/calibration`, `/health` endpoints |

---

## TypeScript MCP Server

### Tools (6)

| Tool | Input | Output |
|------|-------|--------|
| `predict_next_event` | `caseId`, `events[]` | Top-K activity predictions, uncertainty decomposition, conformal set |
| `predict_outcome` | `caseId`, `events[]` | On-time/rework probabilities, expected remaining hours |
| `get_calibration` | `windowSize?` | ECE, MCE, Brier, per-bucket reliability |
| `get_autonomy_level` | — | Current trust level: SUPERVISED → GUIDED → COLLABORATIVE → AUTONOMOUS |
| `get_effective_thresholds` | `mode?` | Base thresholds modulated by mode, uncertainty, calibration |
| `evaluate_gate` | `gateName`, `observedValue`, `mode?` | allow/hold/block decision with audit trail |

### Governance Module (`mcp-server/src/governance/`)

The governance modulation formula:

```
effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
```

- **mode_factor** — scales by operational mode (flexible → standard → strict → forbidden)
- **uncertainty_factor** — tightens when epistemic uncertainty is high (reducible uncertainty demands caution)
- **calibration_factor** — adjusts based on historical prediction accuracy (ECE, Brier)

### Safety Boundaries

- **Immutable floor**: `forbidden` mode, sensitive data, Dempster-Shafer conflict > 0.7, circuit breaker — these cannot be overridden
- **Asymmetric trust**: slow ascent (sustained calibration), fast descent (single critical failure triggers lockdown)

### Bridge (`mcp-server/src/bridge/`)

- `python-bridge.ts` — HTTP client to the Python inference server at `AETHER_PYTHON_URL`
- `pinecone-bridge.ts` — Vector search for historical case similarity
- `promptspeak-bridge.ts` — PromptSpeak symbol resolution

---

## Theoretical Foundations

AETHER's architecture maps directly onto the JEPA ecosystem and LeCun's broader research program. This section makes the connections explicit.

### Mapping to the JEPA Framework

| AETHER Component | JEPA Role | Reference |
|---|---|---|
| `EventEncoder` (StructuralEncoder → TemporalEncoder → ContextEncoder) | **Target encoder** — maps raw observations to latent representations `z_t ∈ R^128` | JEPA (LeCun, 2022): "An encoder that maps an observation to a representation" |
| `TransitionModel` (`f(z_t, a_t, c_t) → z_{t+1}`) | **JEPA predictor** — predicts in latent space, never reconstructs raw events | JEPA: "A predictor module that predicts the representation of y from the representation of x" |
| `EnergyScorer` (`‖z_hat − z‖² → sigmoid`) | **Energy function** — scores transition plausibility in latent space | [A Tutorial on Energy-Based Learning](https://proceedings.mlr.press/v1/lecun06a.html) (LeCun et al., 2006); [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) (LeCun, 2022) |
| `SIGRegLoss` | **Collapse prevention** — eigenvalue-based regularization of the covariance matrix | [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025): SIGReg via Epps-Pulley. We use the eigenvalue formulation |
| `VICRegLoss` | **Collapse prevention** (alternative) — Variance-Invariance-Covariance regularization | [VICReg](https://arxiv.org/abs/2105.04906) (Bardes, Ponce & LeCun, ICLR 2022) |
| `LatentVariable` (Gumbel-Softmax) | **Latent variable** — categorical path variants enabling multi-modal predictions | JEPA: latent variable `z` that captures information about `y` not present in `x` |

### What AETHER Adds

The components above are adapted from the JEPA ecosystem. AETHER's novel contribution is combining them into a **governance-aware pipeline** for process mining:

1. **Epistemic/Aleatoric Decomposition** (`UncertaintyDecomposer`) — Ensemble variance decomposition via the law of total variance. Separates reducible model uncertainty from irreducible process noise. Not present in I-JEPA, V-JEPA, or LeJEPA.

2. **Adaptive Governance Modulation** (`GovernanceModulation`) — The formula `effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor` uses the epistemic ratio to dynamically adjust governance. High epistemic uncertainty → tighten oversight. High aleatoric uncertainty → leave governance alone.

3. **Adaptive Conformal Prediction** (`AdaptiveConformalPredictor`) — Distribution-free prediction sets (Gibbs & Candes, NeurIPS 2021) wrapped around the JEPA predictor, providing coverage guarantees on non-exchangeable business event data.

4. **Asymmetric Trust State Machine** (`AutonomyController`) — Trust is earned slowly through sustained calibration and lost quickly through single failures. This operationalizes the energy-based scoring into real governance decisions.

### Energy-Based Model Connection

LeCun's 2006 tutorial on Energy-Based Learning describes a framework where compatibility between inputs and outputs is measured by an energy function — low energy means compatible, high energy means incompatible. AETHER's `EnergyScorer` is a direct implementation of this:

```
E(z_t, a_t, z_{t+1}) = ‖TransitionModel(z_t, a_t, c_t) − z_{t+1}‖²
```

- **Low energy**: the actual next state matches the world model's prediction → plausible transition
- **High energy**: surprise → anomalous event, process deviation, or conformance violation

The `EnergyContrastiveLoss` trains this scorer with a margin-based contrastive objective: actual transitions should have low energy, random/corrupted transitions should have energy above the margin.

---

## Key Algorithms

### Time2Vec (Kazemi et al., ICLR 2019)

Maps scalar inter-event time Δt to a d-dimensional embedding:

```
t2v[0] = ω₀·Δt + φ₀           (linear trend)
t2v[i] = sin(ω_i·Δt + φ_i)    (periodic components, i = 1..d-1)
```

Captures business rhythms (hourly, daily, weekly cycles) as learnable frequencies.

### Ensemble Uncertainty Decomposition

Applies the law of total variance across ensemble members:

```
Total     = Var(Y)
Epistemic = Var_M[E[Y|M]]    — disagreement between models (reducible)
Aleatoric = E_M[Var[Y|M]]    — average within-model variance (irreducible)
```

Governance response: high epistemic → tighten oversight (more data helps); high aleatoric → do not tighten (randomness is inherent).

### Adaptive Conformal Prediction (Gibbs & Candes, NeurIPS 2021)

Distribution-free prediction sets with coverage guarantees on non-exchangeable data:

1. Include classes in descending probability order until cumulative ≥ (1 - α)
2. Update α after each observation:
   - Miss: `α_{t+1} = α_t + γ·(α_target - 1)` → sets widen
   - Hit: `α_{t+1} = α_t + γ·α_target` → sets narrow

### Energy-Based Anomaly Scoring

```
E(z_pred, z_actual) = ||z_pred - z_actual||²
normalized_energy   = sigmoid(E / temperature)
```

Low energy = transition matches world model expectations. High energy = anomalous event sequence.

---

## Data Sources

| Source | Format | Content |
|--------|--------|---------|
| **BPI 2017** | **Parquet/JSON** | **Loan applications, 31K cases (primary benchmark)** |
| BPI 2020 | JSON | Travel expense declarations, 10.5K cases |
| Road Traffic Fine | XES/JSON | Traffic fine management, 150K cases |
| SAP SQLite | `sap.sqlite` | VBFA document flow, CDHDR change documents |
| BPI 2019 | JSON | Real purchase-to-pay, 251K cases |
| O2C / P2P CSV | CSV | Order-to-cash and purchase-to-pay event logs |
| OCEL 2.0 P2P | SQLite | Zenodo simulated SAP process (object-centric) |

### Dataset Parsing Scripts

| Script | Dataset | Source |
|--------|---------|--------|
| `scripts/parse_bpi2017.py` | BPI Challenge 2017 | Hugging Face (Modzo18/BPIC2017Iteration) |
| `scripts/parse_bpi2020.py` | BPI Challenge 2020 | 4TU.ResearchData |
| `scripts/parse_road_traffic.py` | Road Traffic Fine | 4TU.ResearchData |

The unified pipeline (`AetherDataPipeline`) normalizes all sources into a common event format with activity, resource, timestamp, and numerical attributes, then builds shared vocabularies and train/val splits.
