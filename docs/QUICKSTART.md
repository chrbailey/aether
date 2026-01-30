# AETHER v3 Quick Start Guide

Reproduce AETHER benchmark results in under 30 minutes.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) or Linux with CUDA
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **~5GB disk space** for datasets and models

## Step 1: Clone and Install (5 min)

```bash
# Clone repository
git clone https://github.com/christopherbailey/aether.git
cd aether

# Install Python dependencies
pip install -e ".[dev]"

# Install Node.js dependencies
npm install
npm run build

# Verify installation
npm test                          # Should pass 99 tests
python -m pytest core/tests/ -v   # Should pass 303 tests
```

## Step 2: Download a Dataset (2 min)

Choose one dataset to start:

### Option A: Sepsis (Small, Fast)
```bash
# Download from 4TU.ResearchData
curl -L -o data/external/sepsis/Sepsis.xes.gz \
  "https://data.4tu.nl/file/6b0c3d15-ef14-4c4f-9e4c-c2b7d4b3c5a1/Sepsis.xes.gz"
gunzip data/external/sepsis/Sepsis.xes.gz
```

### Option B: Road Traffic Fine (Large, Best Results)
```bash
mkdir -p data/external/road_traffic_fine
curl -L -o data/external/road_traffic_fine/Road_Traffic_Fine.xes.gz \
  "https://data.4tu.nl/file/[uuid]/Road_Traffic_Fine.xes.gz"
gunzip data/external/road_traffic_fine/Road_Traffic_Fine.xes.gz
```

### Option C: Use Existing BPI 2019 Dataset
If you have the BPI 2019 dataset, set the path:
```bash
export AETHER_BPI2019_PATH="/path/to/bpi2019.json"
```

## Step 3: Train a Model (10-20 min)

```bash
# For Sepsis dataset
python scripts/train_sepsis.py

# For Road Traffic Fine (takes longer)
python scripts/train_road_traffic.py

# Training output shows:
# - Epoch progress with accuracy and ECE
# - Model saved to data/external/<dataset>/models/best.pt
```

**Expected output:**
```
Epoch 50/50: accuracy=81.6%, ECE=0.0000
Model saved: data/external/road_traffic_fine/models/best.pt
```

## Step 4: Run Benchmark (5 min)

```bash
# Benchmark the trained model
python scripts/benchmark_road_traffic.py

# Output shows MCC improvement:
# Static:  MCC=0.0XXX
# AETHER:  MCC=0.XXXX
# MCC Improvement: +XX.X%
```

**Expected output for Road Traffic Fine:**
```
Static:  MCC=0.0056, Burden=7.5%
AETHER:  MCC=0.0205, Burden=2.9%
MCC Improvement: +266.2%
```

## Step 5: Generate Comparison Report

```bash
python scripts/generate_benchmark_report.py

# Creates:
# - docs/BENCHMARK_COMPARISON.md
# - data/benchmarks/aggregate_results.json
```

## Configuration

### v3 Vocabulary-Aware Floor

The v3 formula adds vocabulary-size normalization. Configure in `mcp-server/src/governance/aether.config.ts`:

```typescript
export const VOCAB_NORMALIZATION = {
  baseFloor: 0.50,        // Min floor for small vocabularies
  floorIncrement: 0.05,   // Increment per log-scale step
  referenceVocab: 20,     // Reference vocabulary size
  scaleFactor: 4,         // Log scale factor
  enabled: true,          // Toggle vocabulary normalization
};
```

### Training Parameters

Default training config (can be modified in training scripts):

```python
LATENT_DIM = 128      # Latent space dimension
N_EPOCHS = 50         # Training epochs
BATCH_SIZE = 32       # Batch size
LR = 3e-4             # Learning rate
LOSS_TYPE = "sigreg"  # Options: "sigreg", "vicreg"
```

## Running the MCP Server

```bash
# Terminal 1: Start Python inference server
python -m core.inference.server

# Terminal 2: Start MCP server
npm start

# The MCP server exposes 6 tools for AI assistants
```

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "MPS not available"
On Intel Mac or Linux without GPU:
```python
# In training script, change:
DEVICE = "cpu"  # Instead of "mps"
```

### "Out of memory"
Reduce batch size:
```python
BATCH_SIZE = 16  # Or 8 for very large datasets
```

## Dataset Sources

All datasets available from 4TU.ResearchData:

| Dataset | DOI | Size |
|---------|-----|------|
| Sepsis | [10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460) | 8 MB |
| BPI 2019 | [10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) | 1.2 GB |
| BPIC 2012 | [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f) | 74 MB |
| Road Traffic Fine | [10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5) | 183 MB |
| BPI 2018 | [10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972) | 2.1 GB |

## Next Steps

1. **Try different datasets** — Each domain shows different characteristics
2. **Tune the formula** — Modify coefficients in `aether.config.ts`
3. **Integrate with Claude** — Add to `claude_desktop_config.json`
4. **Read the analysis** — `docs/VOCABULARY_NORMALIZATION_ANALYSIS.md`

## Support

- Issues: https://github.com/christopherbailey/aether/issues
- Discussions: https://github.com/christopherbailey/aether/discussions
