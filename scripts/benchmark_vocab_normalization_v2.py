"""
Prototype v2: Deep Analysis of Vocabulary-Size Regression

This script performs comprehensive analysis of the SAP BSP669 regression,
examining confidence distributions and optimal threshold ranges.

Key finding from v1 prototype: Both v2 and v3 hit the 0.5 floor, producing
identical results. The issue is not epistemic normalization but rather
the confidence distribution characteristics in high-vocabulary datasets.
"""

import sys
import json
import time
from pathlib import Path

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

DATA_DIR = AETHER_ROOT / "data" / "external" / "sap_bsp669"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(vocab_path, model_path, device):
    """Load AETHER model and vocabularies."""
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    expected_act_size = checkpoint["encoder"]["structural.activity_embedding.weight"].shape[0]
    expected_res_size = checkpoint["encoder"]["structural.resource_embedding.weight"].shape[0]

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
    predictor = HierarchicalPredictor(latent_dim=128, n_activities=activity_vocab.size, n_phases=6)
    latent_var = LatentVariable(latent_dim=128)

    encoder.load_state_dict(checkpoint["encoder"])
    transition.load_state_dict(checkpoint["transition"])
    energy.load_state_dict(checkpoint["energy"])
    predictor.load_state_dict(checkpoint["predictor"])
    latent_var.load_state_dict(checkpoint["latent_var"])

    for m in [encoder, transition, energy, predictor, latent_var]:
        m.to(device)
        m.eval()

    return encoder, transition, energy, predictor, latent_var, activity_vocab, resource_vocab


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
        return {"mcc": 0, "f1": 0, "burden": 0, "recall": 0, "precision": 0}

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


def threshold_sweep(results, thresholds):
    """Evaluate metrics at each threshold."""
    sweep_results = {}
    for thresh in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for r in results:
            needs_review = not r["is_correct"]
            flagged = r["confidence"] < thresh

            if needs_review and flagged:
                tp += 1
            elif needs_review and not flagged:
                fn += 1
            elif not needs_review and flagged:
                fp += 1
            else:
                tn += 1

        metrics = compute_metrics(tp, fp, tn, fn)
        metrics["threshold"] = thresh
        sweep_results[thresh] = metrics

    return sweep_results


def main():
    print("=" * 70)
    print("VOCABULARY REGRESSION DEEP ANALYSIS")
    print("Dataset: SAP BSP669 (80 activities)")
    print("=" * 70)

    start_time = time.time()

    # Load model
    print("\nLoading model...")
    encoder, _, _, predictor, _, activity_vocab, resource_vocab = load_model(
        DATA_DIR / "vocabulary.json",
        DATA_DIR / "models" / "best.pt",
        DEVICE
    )

    vocab_size = activity_vocab.size
    print(f"Activity vocabulary size: {vocab_size} tokens")

    # Load validation cases
    with open(DATA_DIR / "val_cases.json") as f:
        cases = json.load(f)

    print(f"Evaluating {len(cases)} cases...")

    # Collect predictions
    results = []
    for case in cases:
        events = case.get("events", [])
        if len(events) < 2:
            continue

        conf, is_correct, epistemic, total_unc = predict_with_uncertainty(
            encoder, predictor, activity_vocab, resource_vocab, events, DEVICE
        )

        if conf is None:
            continue

        results.append({
            "confidence": float(conf),
            "is_correct": bool(is_correct),
            "epistemic": float(epistemic),
            "total_unc": float(total_unc),
        })

    # Analyze confidence distributions
    correct = [r for r in results if r["is_correct"]]
    wrong = [r for r in results if not r["is_correct"]]

    correct_confs = [r["confidence"] for r in correct]
    wrong_confs = [r["confidence"] for r in wrong]

    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"\nCorrect predictions ({len(correct)} cases):")
    print(f"  Mean confidence: {np.mean(correct_confs):.4f}")
    print(f"  Median confidence: {np.median(correct_confs):.4f}")
    print(f"  Std confidence: {np.std(correct_confs):.4f}")
    print(f"  Min/Max: {np.min(correct_confs):.4f} / {np.max(correct_confs):.4f}")

    print(f"\nWrong predictions ({len(wrong)} cases):")
    print(f"  Mean confidence: {np.mean(wrong_confs):.4f}")
    print(f"  Median confidence: {np.median(wrong_confs):.4f}")
    print(f"  Std confidence: {np.std(wrong_confs):.4f}")
    print(f"  Min/Max: {np.min(wrong_confs):.4f} / {np.max(wrong_confs):.4f}")

    # KEY INSIGHT: Check overlap
    separation = np.mean(correct_confs) - np.mean(wrong_confs)
    print(f"\nSeparation (correct_mean - wrong_mean): {separation:.4f}")

    # Confidence distribution overlap analysis
    percentiles = [10, 25, 50, 75, 90]
    print("\nConfidence Percentiles:")
    print("  Percentile  | Correct | Wrong  | Overlap")
    print("  ------------|---------|--------|--------")
    for p in percentiles:
        c_val = np.percentile(correct_confs, p)
        w_val = np.percentile(wrong_confs, p)
        overlap = "YES" if c_val < np.percentile(wrong_confs, 100-p) else "no"
        print(f"  {p:>3}th       | {c_val:.4f}  | {w_val:.4f} | {overlap}")

    # Threshold sweep
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 70)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep = threshold_sweep(results, thresholds)

    print("\nThresh  | MCC    | F1     | Recall | Prec   | Burden | TP  | FP  | FN")
    print("--------|--------|--------|--------|--------|--------|-----|-----|----")
    for t in thresholds:
        m = sweep[t]
        print(f"{t:.2f}    | {m['mcc']:+.4f} | {m['f1']:.4f} | {m['recall']:.4f} | {m['precision']:.4f} | {m['burden']:.1%} | {m['tp']:3d} | {m['fp']:3d} | {m['fn']:3d}")

    # Find optimal threshold
    optimal = max(sweep.items(), key=lambda x: x[1]["mcc"])
    print(f"\nOPTIMAL THRESHOLD: {optimal[0]:.2f} (MCC = {optimal[1]['mcc']:.4f})")

    # Compare with static 0.55
    static_mcc = sweep[0.55]["mcc"]
    optimal_mcc = optimal[1]["mcc"]
    print(f"\nStatic (0.55): MCC = {static_mcc:.4f}")
    print(f"Optimal ({optimal[0]:.2f}): MCC = {optimal_mcc:.4f}")
    print(f"Potential improvement: {(optimal_mcc - static_mcc):.4f} ({(optimal_mcc - static_mcc)/abs(static_mcc)*100:+.1f}%)")

    # Root cause analysis
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # Check if model is overconfident on wrong predictions
    wrong_above_055 = len([r for r in wrong if r["confidence"] >= 0.55])
    wrong_below_055 = len([r for r in wrong if r["confidence"] < 0.55])
    correct_above_055 = len([r for r in correct if r["confidence"] >= 0.55])
    correct_below_055 = len([r for r in correct if r["confidence"] < 0.55])

    print("\nConfidence >= 0.55:")
    print(f"  Correct: {correct_above_055} ({correct_above_055/len(correct)*100:.1f}%)")
    print(f"  Wrong:   {wrong_above_055} ({wrong_above_055/len(wrong)*100:.1f}%)")
    print("\nConfidence < 0.55:")
    print(f"  Correct: {correct_below_055} ({correct_below_055/len(correct)*100:.1f}%)")
    print(f"  Wrong:   {wrong_below_055} ({wrong_below_055/len(wrong)*100:.1f}%)")

    # The problem: how many wrong predictions have HIGH confidence?
    high_conf_wrong = len([r for r in wrong if r["confidence"] >= 0.60])
    print(f"\nWrong predictions with confidence >= 0.60: {high_conf_wrong} ({high_conf_wrong/len(wrong)*100:.1f}%)")
    print("This represents overconfident errors that a higher threshold would catch.")

    # What AETHER is doing wrong
    print("\n" + "=" * 70)
    print("DIAGNOSIS: WHY AETHER FAILS ON THIS DATASET")
    print("=" * 70)

    # Current AETHER behavior (from earlier analysis)
    # - AETHER computes factors that reduce threshold to 0.5
    # - At 0.5, we flag conf < 0.5, missing overconfident errors

    # For this dataset, the model is OVERCONFIDENT on wrong predictions
    # We need a HIGHER threshold, not lower
    print("""
FINDINGS:

1. MODEL OVERCONFIDENCE: The model has many wrong predictions with
   confidence >= 0.55, meaning they escape review at standard threshold.

2. AETHER v2 DIRECTION ERROR: The formula LOWERS threshold to 0.5,
   which makes the problem WORSE by letting more overconfident errors through.

3. ROOT CAUSE: The current formula assumes that high epistemic uncertainty
   should tighten governance (raise threshold). But for this dataset,
   the epistemic signal is weak (0.0001), so the formula relaxes governance.

4. HIGH-VOCABULARY EFFECT: With 80 activities, the model can be confidently
   wrong because it has many similar activities to confuse. The softmax
   concentrates probability on the wrong activity.

PROPOSED FIX: For high-vocabulary datasets (V > 40), the formula should:
   a) Recognize that calibration is likely poor (ECE underestimated)
   b) Apply a vocabulary-based TIGHTENING factor, not relaxation
   c) Use a higher minimum threshold (0.55 not 0.50) for high-V datasets
""")

    # Save analysis
    output = {
        "dataset": "sap_bsp669",
        "vocab_size": vocab_size,
        "total_cases": len(results),
        "correct_cases": len(correct),
        "wrong_cases": len(wrong),
        "confidence_stats": {
            "correct_mean": round(np.mean(correct_confs), 4),
            "correct_median": round(np.median(correct_confs), 4),
            "wrong_mean": round(np.mean(wrong_confs), 4),
            "wrong_median": round(np.median(wrong_confs), 4),
            "separation": round(separation, 4),
        },
        "threshold_sweep": sweep,
        "optimal_threshold": optimal[0],
        "optimal_mcc": optimal[1]["mcc"],
        "static_mcc": static_mcc,
        "overconfident_wrong_rate": round(high_conf_wrong / len(wrong), 4),
        "diagnosis": "model_overconfidence_on_wrong_predictions",
        "recommendation": "raise_minimum_threshold_for_high_vocab_datasets",
        "elapsed_seconds": round(time.time() - start_time, 2)
    }

    output_path = AETHER_ROOT / "data" / "benchmarks" / "vocab_regression_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
