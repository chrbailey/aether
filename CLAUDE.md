# AETHER: Adaptive Epistemic Trust through Hierarchical Event Reasoning

## PromptSpeak Symbol: `Ξ.SYSTEM.AETHER`

## Project Structure

- `mcp-server/` — TypeScript MCP server (governance, tools, bridges)
- `core/` — Python ML components (encoder, world model, critic, training)
- `data/` — Local data (gitignored): models, calibration history, cached events
- `docs/` — Architecture and research documentation

## Development

```bash
# TypeScript
npm install && npm run build && npm test

# Python
pip install -r requirements.txt
cd core && python -m pytest tests/ -v
```

## Key Concepts

- **Governance Modulation**: `effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor`
- **Epistemic vs Aleatoric**: Epistemic uncertainty tightens governance (more data helps). Aleatoric does NOT (irreducible).
- **Asymmetric Trust**: Slow ascent (sustained calibration), fast descent (single critical failure).
- **Immutable Floor**: forbidden mode, sensitive data, D-S conflict > 0.7, circuit breaker floor = 3.

## Integration Points

| System | What AETHER Provides |
|--------|---------------------|
| PromptSpeak | Adaptive drift thresholds replacing static 0.15 |
| EFC | Dynamic review gate thresholds |
| SAP Workflow Mining | Learned predictions replacing heuristic models |
| Belief Engine v6 | World model predictions as ClaimNode instances |
| VERITY | Calls decompose_from_ensemble() for epistemic/aleatoric split |

## Testing

```bash
npm test                          # TypeScript tests
cd core && python -m pytest -v    # Python tests
```
