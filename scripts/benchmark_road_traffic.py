"""
Benchmark AETHER governance formula on Road Traffic Fine Management data.

Evaluates v2 bidirectional formula: effective_threshold = base * mode * uncertainty * calibration
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np

# Add AETHER to path
AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
sys.path.insert(0, str(AETHER_ROOT))

from core.encoder.event_encoder import EventEncoder
from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.world_model.energy import EnergyScorer
from core.world_model.hierarchical import HierarchicalPredictor
from core.world_model.latent import LatentVariable
from core.world_model.transition import TransitionModel

# Governance config
BASE_THRESHOLDS = {
    "reviewGateAutoPass": 0.55,
}
MODE_FACTORS = {"flexible": 1.0, "standard": 1.1, "strict": 1.2}

DATA_DIR = AETHER_ROOT / "data" / "external" / "road_traffic_fine"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(vocab_path, model_path, device):
    """Load AETHER model and vocabularies."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    expected_act_size = checkpoint["encoder"]["structural.activity_embedding.weight"].shape[0]
    expected_res_size = checkpoint["encoder"]["structural.resource_embedding.weight"].shape[0]

    # Build activity vocabulary
    activity_vocab = ActivityVocabulary(embed_dim=64)
    skip_tokens = {"<UNK>"}
    if expected_act_size < len(vocab_data["activity"]["token_to_idx"]):
        skip_tokens.add("<PAD>")

    for token, idx in sorted(vocab_data["activity"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in skip_tokens:
            activity_vocab.add_token(token)

    # Build resource vocabulary
    resource_vocab = ResourceVocabulary(embed_dim=32)
    skip_tokens_res = {"<UNK>"}
    if expected_res_size < len(vocab_data["resource"]["token_to_idx"]):
        skip_tokens_res.add("<PAD>")

    for token, idx in sorted(vocab_data["resource"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in skip_tokens_res:
            resource_vocab.add_token(token)

    # Build model
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
    predictor = HierarchicalPredictor(latent_dim=128, n_activities=activity_vocab.size, n_phases=4)
    latent_var = LatentVariable(latent_dim=128)

    encoder.load_state_dict(checkpoint["encoder"])
    transition.load_state_dict(checkpoint["transition"])
    energy.load_state_dict(checkpoint["energy"])
    predictor.load_state_dict(checkpoint["predictor"])
    latent_var.load_state_dict(checkpoint["latent_var"])

    for m in [encoder, transition, energy, predictor, latent_var]:
        m.to(device)
        m.eval()

    return encoder, transition, energy, predictor, latent_var, activity_vocab, resource_vocab, checkpoint.get("calibration", {})


def predict_with_uncertainty(encoder, transition, energy, predictor, latent_var, activity_vocab,
                             resource_vocab, events, device, n_samples=10, noise_sigma=0.1):
    """Make prediction with uncertainty estimation using MC dropout / noise injection.

    Matches training evaluation: predicts next activity from the LAST position state,
    comparing against the actual next activity in the sequence.
    """
    if len(events) < 2:
        return None, None, None, None

    # Encode ALL events (including the last one to predict what comes next)
    # But we evaluate prediction from second-to-last position -> last activity
    activity_ids = []
    resource_ids = []
    for e in events:
        act = e.get("activity", "<UNK>")
        res = e.get("resource", "unknown")
        activity_ids.append(activity_vocab.encode(act))
        resource_ids.append(resource_vocab.encode(res))

    activity_tensor = torch.tensor([activity_ids], device=device)
    resource_tensor = torch.tensor([resource_ids], device=device)
    attr_tensor = torch.zeros(1, len(activity_ids), 4, device=device)
    mask = torch.ones(1, len(activity_ids), dtype=torch.bool, device=device)

    # Collect predictions from multiple forward passes with noise
    all_probs = []
    with torch.no_grad():
        h = encoder(activity_tensor, resource_tensor, attr_tensor, mask)
        # Use the second-to-last position to predict the last activity
        # This matches training evaluation: z[:-1] predicts target_activities[:-1]
        z = h[:, -2, :]  # State at second-to-last position

        for _ in range(n_samples):
            z_noisy = z + torch.randn_like(z) * noise_sigma
            out = predictor.activity_head(z_noisy)
            probs = out["probs"]
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    pred_idx = mean_probs.argmax(axis=-1)[0]
    confidence = mean_probs[0, pred_idx]

    # Uncertainty decomposition
    epistemic = all_probs.var(axis=0).mean()
    aleatoric = -(mean_probs * np.log(mean_probs + 1e-10)).sum(axis=-1).mean()
    total_unc = epistemic + aleatoric

    # Ground truth: the last activity in the sequence
    true_activity = events[-1].get("activity", "<UNK>")
    true_idx = activity_vocab.encode(true_activity)
    is_correct = pred_idx == true_idx

    return confidence, is_correct, epistemic, total_unc


def compute_adaptive_threshold(base, mode_factor, epistemic_mean, total_mean, ece, epistemic_baseline=0.0001):
    """Compute v2 bidirectional adaptive threshold."""
    unc_epistemic = 0.5 + 0.5 * np.tanh((epistemic_baseline - epistemic_mean) / (epistemic_baseline + 1e-8))
    unc_total = 1.0 + 0.5 * np.tanh((total_mean - 0.01) / 0.01)
    calibration = 1.0 + 0.5 * np.tanh((ece - 0.1) / 0.1)
    effective = base * mode_factor * unc_epistemic * unc_total * calibration
    return max(0.5, min(0.98, effective)), {
        "mode": mode_factor, "unc_epistemic": unc_epistemic, "unc_total": unc_total, "calibration": calibration
    }


def compute_metrics(tp, fp, tn, fn):
    """Compute classification metrics."""
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0
    burden = (tp + fp) / total if total > 0 else 0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "fpr": round(fpr, 4), "specificity": round(specificity, 4), "mcc": round(mcc, 4),
        "burden": round(burden, 4), "total": total
    }


def main():
    print("=" * 60)
    print("ROAD TRAFFIC FINE GOVERNANCE BENCHMARK")
    print("=" * 60)

    start_time = time.time()

    # Load model
    print("Loading model...")
    encoder, transition, energy, predictor, latent_var, activity_vocab, resource_vocab, model_calib = load_model(
        DATA_DIR / "vocabulary.json",
        DATA_DIR / "models" / "best.pt",
        DEVICE
    )

    # Load validation cases
    with open(DATA_DIR / "val_cases.json") as f:
        cases = json.load(f)

    print(f"Evaluating {len(cases)} cases...")

    # Collect predictions
    results = []
    correct_count = 0
    wrong_count = 0

    for i, case in enumerate(cases):
        events = case.get("events", [])
        if len(events) < 2:
            continue

        conf, is_correct, epistemic, total_unc = predict_with_uncertainty(
            encoder, transition, energy, predictor, latent_var,
            activity_vocab, resource_vocab, events, DEVICE
        )

        if conf is None:
            continue

        results.append({
            "case_id": case.get("caseId"),
            "confidence": float(conf),
            "is_correct": bool(is_correct),
            "epistemic": float(epistemic),
            "total_unc": float(total_unc),
            "outcome": case.get("outcome", {}),
        })

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,} cases...")

    accuracy = correct_count / (correct_count + wrong_count)
    print(f"Model accuracy: {accuracy:.1%}")
    print(f"Cases needing review: {wrong_count} ({wrong_count/(correct_count+wrong_count):.1%})")

    # Compute uncertainty stats
    epistemic_mean = np.mean([r["epistemic"] for r in results])
    total_mean = np.mean([r["total_unc"] for r in results])
    ece = model_calib.get("ece", 0.02)

    print(f"\nUncertainty stats:")
    print(f"  Epistemic mean: {epistemic_mean:.6f}")
    print(f"  Total mean: {total_mean:.6f}")
    print(f"  Model ECE: {ece:.4f}")

    # Benchmark each mode
    benchmark_results = {
        "benchmark": "AETHER Road Traffic Fine Governance Benchmark (v2 bidirectional)",
        "dataset": {
            "label": "Road Traffic Fine Management Process",
            "data_dir": str(DATA_DIR),
            "total_cases": len(cases),
            "evaluated": len(results),
            "needs_review": wrong_count,
            "review_rate": round(wrong_count / len(results), 4),
            "accuracy": round(accuracy, 4),
            "source": "4TU.ResearchData Road Traffic Fine Management Process"
        },
        "model": {
            "checkpoint": str(DATA_DIR / "models" / "best.pt"),
            "calibration": model_calib,
            "ensemble_size": 10,
            "noise_sigma": 0.1
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_mode": {}
    }

    base = BASE_THRESHOLDS["reviewGateAutoPass"]

    for mode, factor in MODE_FACTORS.items():
        print(f"\n--- Mode: {mode} (factor={factor}) ---")

        # Compute adaptive threshold
        adaptive_thresh, factors = compute_adaptive_threshold(
            base, factor, epistemic_mean, total_mean, ece
        )

        # Static baseline
        static_thresh = base
        static_tp, static_fp, static_tn, static_fn = 0, 0, 0, 0

        # AETHER adaptive
        aether_tp, aether_fp, aether_tn, aether_fn = 0, 0, 0, 0
        aether_thresholds = []

        for r in results:
            needs_review = not r["is_correct"]
            conf = r["confidence"]

            # Static
            flagged_static = conf < static_thresh
            if needs_review and flagged_static:
                static_tp += 1
            elif needs_review and not flagged_static:
                static_fn += 1
            elif not needs_review and flagged_static:
                static_fp += 1
            else:
                static_tn += 1

            # AETHER
            flagged_aether = conf < adaptive_thresh
            aether_thresholds.append(adaptive_thresh)
            if needs_review and flagged_aether:
                aether_tp += 1
            elif needs_review and not flagged_aether:
                aether_fn += 1
            elif not needs_review and flagged_aether:
                aether_fp += 1
            else:
                aether_tn += 1

        static_metrics = compute_metrics(static_tp, static_fp, static_tn, static_fn)
        static_metrics["threshold"] = static_thresh
        aether_metrics = compute_metrics(aether_tp, aether_fp, aether_tn, aether_fn)
        aether_metrics["threshold_mean"] = round(np.mean(aether_thresholds), 6)
        aether_metrics["threshold_std"] = round(np.std(aether_thresholds), 6)

        mcc_improvement = aether_metrics["mcc"] - static_metrics["mcc"]

        print(f"Static:  MCC={static_metrics['mcc']:.4f}, F1={static_metrics['f1']:.4f}, Burden={static_metrics['burden']:.1%}")
        print(f"AETHER:  MCC={aether_metrics['mcc']:.4f}, F1={aether_metrics['f1']:.4f}, Burden={aether_metrics['burden']:.1%}, Thresh={aether_metrics['threshold_mean']:.4f}")
        print(f"MCC Improvement: {mcc_improvement:+.4f}")

        benchmark_results["per_mode"][mode] = {
            "mode_factor": factor,
            "static": static_metrics,
            "aether": aether_metrics,
            "factor_decomposition": factors,
            "mcc_improvement": round(mcc_improvement, 4)
        }

    # Save results
    elapsed = time.time() - start_time
    benchmark_results["elapsed_seconds"] = round(elapsed, 2)

    output_path = DATA_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Also copy to benchmarks directory
    benchmarks_dir = AETHER_ROOT / "data" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_path = benchmarks_dir / "road_traffic.json"
    with open(benchmarks_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Also saved to: {benchmarks_path}")

    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
