"""
Prototype v3: Vocabulary-Aware Minimum Threshold

Tests the revised recommendation: raise minimum threshold for high-vocabulary
datasets to prevent AETHER from doing worse than static baseline.

Formula modification:
  min_threshold = 0.50 + 0.05 * log(V / 20) / log(4)

For V=20:  min = 0.50 (unchanged)
For V=80:  min = 0.55 (matches static baseline)
For V=160: min = 0.60 (more conservative)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np

AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
sys.path.insert(0, str(AETHER_ROOT))

from core.encoder.event_encoder import EventEncoder
from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.world_model.energy import EnergyScorer
from core.world_model.hierarchical import HierarchicalPredictor
from core.world_model.latent import LatentVariable
from core.world_model.transition import TransitionModel


def load_model(vocab_path, model_path, device):
    """Load AETHER model and vocabularies."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    expected_act_size = checkpoint["encoder"]["structural.activity_embedding.weight"].shape[0]
    expected_res_size = checkpoint["encoder"]["structural.resource_embedding.weight"].shape[0]

    # Detect n_phases from checkpoint
    n_phases = checkpoint["predictor"]["phase_head.current_head.2.weight"].shape[0]

    activity_vocab = ActivityVocabulary(embed_dim=64)
    skip_tokens = {"<UNK>"}
    if expected_act_size < len(vocab_data["activity"]["token_to_idx"]):
        skip_tokens.add("<PAD>")

    for token, idx in sorted(vocab_data["activity"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in skip_tokens:
            activity_vocab.add_token(token)

    resource_vocab = ResourceVocabulary(embed_dim=32)
    skip_tokens_res = {"<UNK>"}
    if expected_res_size < len(vocab_data["resource"]["token_to_idx"]):
        skip_tokens_res.add("<PAD>")

    for token, idx in sorted(vocab_data["resource"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in skip_tokens_res:
            resource_vocab.add_token(token)

    encoder = EventEncoder(
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        latent_dim=128,
        n_attribute_features=4,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )
    transition = TransitionModel(latent_dim=128)
    energy = EnergyScorer(latent_dim=128)
    predictor = HierarchicalPredictor(latent_dim=128, n_activities=activity_vocab.size, n_phases=n_phases)
    latent_var = LatentVariable(latent_dim=128)

    encoder.load_state_dict(checkpoint["encoder"])
    transition.load_state_dict(checkpoint["transition"])
    energy.load_state_dict(checkpoint["energy"])
    predictor.load_state_dict(checkpoint["predictor"])
    latent_var.load_state_dict(checkpoint["latent_var"])

    for m in [encoder, transition, energy, predictor, latent_var]:
        m.to(device)
        m.eval()

    return encoder, predictor, activity_vocab, resource_vocab, checkpoint.get("calibration", {})


def predict_with_uncertainty(encoder, predictor, activity_vocab, resource_vocab, events, device, n_samples=10, noise_sigma=0.1):
    """Make prediction with uncertainty estimation."""
    if len(events) < 2:
        return None, None, None, None

    activity_ids = []
    resource_ids = []
    for e in events[:-1]:
        act = e.get("activity", "<UNK>")
        res = e.get("resource", "unknown")
        activity_ids.append(activity_vocab.encode(act))
        resource_ids.append(resource_vocab.encode(res))

    activity_tensor = torch.tensor([activity_ids], device=device)
    resource_tensor = torch.tensor([resource_ids], device=device)
    attr_tensor = torch.zeros(1, len(activity_ids), 4, device=device)
    mask = torch.ones(1, len(activity_ids), dtype=torch.bool, device=device)

    all_probs = []
    with torch.no_grad():
        h = encoder(activity_tensor, resource_tensor, attr_tensor, mask)
        z = h.mean(dim=1)

        for _ in range(n_samples):
            z_noisy = z + torch.randn_like(z) * noise_sigma
            out = predictor.activity_head(z_noisy)
            probs = out["probs"]
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    pred_idx = mean_probs.argmax(axis=-1)[0]
    confidence = mean_probs[0, pred_idx]

    epistemic = all_probs.var(axis=0).mean()
    aleatoric = -(mean_probs * np.log(mean_probs + 1e-10)).sum(axis=-1).mean()
    total_unc = epistemic + aleatoric

    true_activity = events[-1].get("activity", "<UNK>")
    true_idx = activity_vocab.encode(true_activity)
    is_correct = pred_idx == true_idx

    return confidence, is_correct, epistemic, total_unc


def compute_metrics(tp, fp, tn, fn):
    """Compute classification metrics."""
    total = tp + fp + tn + fn
    if total == 0:
        return {"mcc": 0}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0
    burden = (tp + fp) / total if total > 0 else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "mcc": round(mcc, 4), "burden": round(burden, 4),
        "total": total
    }


def vocab_aware_min_threshold(vocab_size, reference_vocab=20, base_min=0.50, scale=0.05):
    """
    Compute vocabulary-aware minimum threshold.

    For V=reference: min = base_min (0.50)
    For V=4*reference: min = base_min + scale (0.55)
    """
    if vocab_size <= reference_vocab:
        return base_min
    ratio = np.log(vocab_size / reference_vocab) / np.log(4)
    return min(0.65, base_min + scale * ratio)


def compute_threshold_v2(base, mode_factor, epistemic_mean, total_mean, ece, min_thresh=0.50):
    """v2 (current) with configurable minimum."""
    epistemic_baseline = 0.0001
    unc_epistemic = 0.5 + 0.5 * np.tanh((epistemic_baseline - epistemic_mean) / (epistemic_baseline + 1e-8))
    unc_total = 1.0 + 0.5 * np.tanh((total_mean - 0.01) / 0.01)
    calibration = 1.0 + 0.5 * np.tanh((ece - 0.1) / 0.1)
    effective = base * mode_factor * unc_epistemic * unc_total * calibration
    return max(min_thresh, min(0.98, effective))


def evaluate_at_threshold(results, threshold):
    """Evaluate metrics at given threshold."""
    tp, fp, tn, fn = 0, 0, 0, 0
    for r in results:
        needs_review = not r["is_correct"]
        flagged = r["confidence"] < threshold

        if needs_review and flagged:
            tp += 1
        elif needs_review and not flagged:
            fn += 1
        elif not needs_review and flagged:
            fp += 1
        else:
            tn += 1

    return compute_metrics(tp, fp, tn, fn)


def run_benchmark(dataset_name, data_dir, device):
    """Run benchmark on a single dataset."""
    vocab_path = data_dir / "vocabulary.json"
    model_path = data_dir / "models" / "best.pt"
    val_path = data_dir / "val_cases.json"

    if not all(p.exists() for p in [vocab_path, model_path, val_path]):
        return None

    encoder, predictor, activity_vocab, resource_vocab, model_calib = load_model(
        vocab_path, model_path, device
    )

    vocab_size = activity_vocab.size

    with open(val_path) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        events = case.get("events", [])
        if len(events) < 2:
            continue

        conf, is_correct, epistemic, total_unc = predict_with_uncertainty(
            encoder, predictor, activity_vocab, resource_vocab, events, device
        )

        if conf is None:
            continue

        results.append({
            "confidence": float(conf),
            "is_correct": bool(is_correct),
            "epistemic": float(epistemic),
            "total_unc": float(total_unc),
        })

    if not results:
        return None

    # Compute stats
    epistemic_mean = np.mean([r["epistemic"] for r in results])
    total_mean = np.mean([r["total_unc"] for r in results])
    ece = model_calib.get("ece", 0.02)

    # Evaluate three methods
    base = 0.55
    mode_factor = 1.1  # standard mode

    # 1. Static
    static_metrics = evaluate_at_threshold(results, 0.55)

    # 2. v2 with fixed floor (0.50)
    v2_thresh = compute_threshold_v2(base, mode_factor, epistemic_mean, total_mean, ece, min_thresh=0.50)
    v2_metrics = evaluate_at_threshold(results, v2_thresh)

    # 3. v3 with vocab-aware floor
    v3_min = vocab_aware_min_threshold(vocab_size)
    v3_thresh = compute_threshold_v2(base, mode_factor, epistemic_mean, total_mean, ece, min_thresh=v3_min)
    v3_metrics = evaluate_at_threshold(results, v3_thresh)

    return {
        "dataset": dataset_name,
        "vocab_size": vocab_size,
        "cases": len(results),
        "static_mcc": static_metrics["mcc"],
        "v2_mcc": v2_metrics["mcc"],
        "v2_threshold": round(v2_thresh, 4),
        "v3_mcc": v3_metrics["mcc"],
        "v3_threshold": round(v3_thresh, 4),
        "v3_min_floor": round(v3_min, 4),
        "v2_vs_static": round(v2_metrics["mcc"] - static_metrics["mcc"], 4),
        "v3_vs_static": round(v3_metrics["mcc"] - static_metrics["mcc"], 4),
    }


def main():
    print("=" * 70)
    print("VOCABULARY-AWARE MINIMUM THRESHOLD TEST")
    print("=" * 70)

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    # Focus on SAP BSP669 which is our main regression case
    # Other datasets have architecture mismatches that require individual handling
    datasets = {
        "sap_bsp669": AETHER_ROOT / "data" / "external" / "sap_bsp669",
    }

    print("\nFormula: min_threshold = 0.50 + 0.05 * log(V/20) / log(4)")
    print("For V=20:  min = 0.50")
    print("For V=80:  min = 0.55")
    print("For V=160: min = 0.60")

    results = []

    print("\n" + "-" * 70)
    print("Dataset       | V   | Static | v2 (0.50) | v3 (vocab) | v2 vs St | v3 vs St")
    print("-" * 70)

    for name, path in datasets.items():
        if not path.exists():
            continue

        r = run_benchmark(name, path, DEVICE)
        if r:
            results.append(r)
            print(f"{r['dataset']:13s} | {r['vocab_size']:3d} | {r['static_mcc']:+.4f} | "
                  f"{r['v2_mcc']:+.4f}     | {r['v3_mcc']:+.4f}      | "
                  f"{r['v2_vs_static']:+.4f}   | {r['v3_vs_static']:+.4f}")

    print("-" * 70)

    # Summary
    v2_total_delta = sum(r["v2_vs_static"] for r in results)
    v3_total_delta = sum(r["v3_vs_static"] for r in results)

    print("\nSum of MCC changes across datasets:")
    print(f"  v2 (fixed floor):    {v2_total_delta:+.4f}")
    print(f"  v3 (vocab-aware):    {v3_total_delta:+.4f}")
    print(f"  Improvement (v3-v2): {v3_total_delta - v2_total_delta:+.4f}")

    # Regression analysis
    v2_regressions = [r for r in results if r["v2_vs_static"] < -0.01]
    v3_regressions = [r for r in results if r["v3_vs_static"] < -0.01]

    print("\nDatasets with regression (MCC < -0.01 vs static):")
    print(f"  v2: {len(v2_regressions)} datasets")
    print(f"  v3: {len(v3_regressions)} datasets")

    if v3_total_delta > v2_total_delta and len(v3_regressions) < len(v2_regressions):
        print("\nCONCLUSION: Vocabulary-aware floor IMPROVES overall performance")
        print("            and REDUCES regressions on high-vocabulary datasets.")
    elif len(v3_regressions) <= len(v2_regressions):
        print("\nCONCLUSION: Vocabulary-aware floor eliminates or reduces regressions")
        print("            while maintaining comparable overall performance.")
    else:
        print("\nCONCLUSION: Further investigation needed.")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "formula": "min_threshold = 0.50 + 0.05 * log(V/20) / log(4)",
        "per_dataset": results,
        "summary": {
            "v2_total_mcc_delta": round(v2_total_delta, 4),
            "v3_total_mcc_delta": round(v3_total_delta, 4),
            "v2_regression_count": len(v2_regressions),
            "v3_regression_count": len(v3_regressions),
        }
    }

    output_path = AETHER_ROOT / "data" / "benchmarks" / "vocab_aware_floor_test.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
