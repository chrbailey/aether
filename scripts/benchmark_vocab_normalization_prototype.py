"""
Prototype: Vocabulary-Size Normalized Adaptive Threshold (v3)

This script tests the vocabulary normalization hypothesis on SAP BSP669 data.
It compares:
- v2 (current): Fixed epistemic baseline
- v3 (proposed): Vocabulary-scaled epistemic baseline with entropy normalization

Expected outcome: v3 should reduce or eliminate the -24% MCC regression.
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
BASE_THRESHOLDS = {"reviewGateAutoPass": 0.55}
MODE_FACTORS = {"flexible": 1.0, "standard": 1.1, "strict": 1.2}

DATA_DIR = AETHER_ROOT / "data" / "external" / "sap_bsp669"
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

    return encoder, transition, energy, predictor, latent_var, activity_vocab, resource_vocab, checkpoint.get("calibration", {})


def predict_with_uncertainty(encoder, transition, energy, predictor, latent_var, activity_vocab,
                             resource_vocab, events, device, n_samples=10, noise_sigma=0.1):
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


# ============================================================================
# THRESHOLD FUNCTIONS - v2 (current) vs v3 (proposed)
# ============================================================================

def compute_adaptive_threshold_v2(base, mode_factor, epistemic_mean, total_mean, ece,
                                   epistemic_baseline=0.0001):
    """
    v2 (CURRENT): Fixed epistemic baseline - NO vocabulary normalization.
    This is what's currently causing the -24% regression on SAP BSP669.
    """
    unc_epistemic = 0.5 + 0.5 * np.tanh((epistemic_baseline - epistemic_mean) / (epistemic_baseline + 1e-8))
    unc_total = 1.0 + 0.5 * np.tanh((total_mean - 0.01) / 0.01)
    calibration = 1.0 + 0.5 * np.tanh((ece - 0.1) / 0.1)
    effective = base * mode_factor * unc_epistemic * unc_total * calibration
    return max(0.5, min(0.98, effective)), {
        "mode": mode_factor,
        "unc_epistemic": unc_epistemic,
        "unc_total": unc_total,
        "calibration": calibration,
        "version": "v2_fixed_baseline"
    }


def compute_adaptive_threshold_v3(base, mode_factor, epistemic_mean, total_mean, ece,
                                   vocab_size, reference_vocab=20, epistemic_baseline=0.0001):
    """
    v3 (PROPOSED): Vocabulary-normalized epistemic baseline.

    Key changes:
    1. Scale epistemic baseline by sqrt(vocab_size / reference_vocab)
    2. Normalize epistemic by entropy ratio: log(ref) / log(vocab)

    This accounts for the fact that larger vocabularies naturally produce
    higher variance in softmax outputs.
    """
    # Scale baseline by vocabulary ratio (sqrt for moderate scaling)
    scaled_baseline = epistemic_baseline * np.sqrt(vocab_size / reference_vocab)

    # Normalize epistemic by entropy ratio
    entropy_ratio = np.log(reference_vocab) / np.log(vocab_size)
    epistemic_normalized = epistemic_mean * entropy_ratio

    # Compute uncertainty factor with scaled baseline
    unc_epistemic = 0.5 + 0.5 * np.tanh(
        (scaled_baseline - epistemic_normalized) / (scaled_baseline + 1e-8)
    )

    unc_total = 1.0 + 0.5 * np.tanh((total_mean - 0.01) / 0.01)
    calibration = 1.0 + 0.5 * np.tanh((ece - 0.1) / 0.1)
    effective = base * mode_factor * unc_epistemic * unc_total * calibration
    return max(0.5, min(0.98, effective)), {
        "mode": mode_factor,
        "unc_epistemic": unc_epistemic,
        "unc_total": unc_total,
        "calibration": calibration,
        "scaled_baseline": scaled_baseline,
        "entropy_ratio": entropy_ratio,
        "epistemic_normalized": epistemic_normalized,
        "version": "v3_vocab_normalized"
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


def evaluate_method(results, threshold, method_name):
    """Evaluate a threshold method on all results."""
    tp, fp, tn, fn = 0, 0, 0, 0
    thresholds = []

    for r in results:
        needs_review = not r["is_correct"]
        conf = r["confidence"]
        flagged = conf < threshold
        thresholds.append(threshold)

        if needs_review and flagged:
            tp += 1
        elif needs_review and not flagged:
            fn += 1
        elif not needs_review and flagged:
            fp += 1
        else:
            tn += 1

    metrics = compute_metrics(tp, fp, tn, fn)
    metrics["threshold_mean"] = round(np.mean(thresholds), 6)
    metrics["method"] = method_name
    return metrics


def main():
    print("=" * 70)
    print("VOCABULARY NORMALIZATION PROTOTYPE TEST")
    print("Dataset: SAP BSP669 (77 activities)")
    print("=" * 70)

    start_time = time.time()

    # Load model
    print("\nLoading model...")
    encoder, transition, energy, predictor, latent_var, activity_vocab, resource_vocab, model_calib = load_model(
        DATA_DIR / "vocabulary.json",
        DATA_DIR / "models" / "best.pt",
        DEVICE
    )

    vocab_size = activity_vocab.size
    print(f"Activity vocabulary size: {vocab_size} tokens")
    print("Reference vocabulary: 20 tokens")
    print(f"Entropy ratio: log(20)/log({vocab_size}) = {np.log(20)/np.log(vocab_size):.4f}")

    # Load validation cases
    with open(DATA_DIR / "val_cases.json") as f:
        cases = json.load(f)

    print(f"\nEvaluating {len(cases)} cases...")

    # Collect predictions
    results = []
    correct_count = 0
    wrong_count = 0

    for case in cases:
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
        })

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

    accuracy = correct_count / (correct_count + wrong_count)
    print(f"Model accuracy: {accuracy:.1%}")
    print(f"Cases needing review: {wrong_count} ({wrong_count/(correct_count+wrong_count):.1%})")

    # Compute uncertainty stats
    epistemic_mean = np.mean([r["epistemic"] for r in results])
    total_mean = np.mean([r["total_unc"] for r in results])
    ece = model_calib.get("ece", 0.02)

    print("\nUncertainty stats:")
    print(f"  Raw epistemic mean: {epistemic_mean:.6f}")
    print(f"  Total mean: {total_mean:.6f}")
    print(f"  Model ECE: {ece:.4f}")

    # Test each method
    base = BASE_THRESHOLDS["reviewGateAutoPass"]
    mode_factor = MODE_FACTORS["standard"]  # Use standard mode for comparison

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON (Standard Mode)")
    print("=" * 70)

    # Static baseline
    static_thresh = 0.55
    static_metrics = evaluate_method(results, static_thresh, "static")
    print("\n1. STATIC (threshold=0.55):")
    print(f"   MCC={static_metrics['mcc']:.4f}, F1={static_metrics['f1']:.4f}, Burden={static_metrics['burden']:.1%}")

    # v2 (current)
    v2_thresh, v2_factors = compute_adaptive_threshold_v2(
        base, mode_factor, epistemic_mean, total_mean, ece
    )
    v2_metrics = evaluate_method(results, v2_thresh, "v2_current")
    v2_improvement = v2_metrics['mcc'] - static_metrics['mcc']
    print(f"\n2. v2 CURRENT (fixed baseline={0.0001}):")
    print(f"   Threshold: {v2_thresh:.4f}")
    print(f"   Factors: epistemic={v2_factors['unc_epistemic']:.4f}, total={v2_factors['unc_total']:.4f}, calib={v2_factors['calibration']:.4f}")
    print(f"   MCC={v2_metrics['mcc']:.4f}, F1={v2_metrics['f1']:.4f}, Burden={v2_metrics['burden']:.1%}")
    print(f"   MCC Improvement vs Static: {v2_improvement:+.4f} ({v2_improvement/abs(static_metrics['mcc'])*100:+.1f}%)")

    # v3 (proposed)
    v3_thresh, v3_factors = compute_adaptive_threshold_v3(
        base, mode_factor, epistemic_mean, total_mean, ece, vocab_size
    )
    v3_metrics = evaluate_method(results, v3_thresh, "v3_proposed")
    v3_improvement = v3_metrics['mcc'] - static_metrics['mcc']
    print(f"\n3. v3 PROPOSED (vocab-normalized, V={vocab_size}):")
    print(f"   Scaled baseline: {v3_factors['scaled_baseline']:.6f}")
    print(f"   Entropy ratio: {v3_factors['entropy_ratio']:.4f}")
    print(f"   Epistemic normalized: {v3_factors['epistemic_normalized']:.6f}")
    print(f"   Threshold: {v3_thresh:.4f}")
    print(f"   Factors: epistemic={v3_factors['unc_epistemic']:.4f}, total={v3_factors['unc_total']:.4f}, calib={v3_factors['calibration']:.4f}")
    print(f"   MCC={v3_metrics['mcc']:.4f}, F1={v3_metrics['f1']:.4f}, Burden={v3_metrics['burden']:.1%}")
    print(f"   MCC Improvement vs Static: {v3_improvement:+.4f} ({v3_improvement/abs(static_metrics['mcc'])*100:+.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nv2 (current):  {v2_improvement:+.4f} MCC change ({v2_improvement/abs(static_metrics['mcc'])*100:+.1f}%)")
    print(f"v3 (proposed): {v3_improvement:+.4f} MCC change ({v3_improvement/abs(static_metrics['mcc'])*100:+.1f}%)")

    delta = v3_improvement - v2_improvement
    if delta > 0:
        print(f"\nVocabulary normalization IMPROVES by {delta:+.4f} MCC")
        print("HYPOTHESIS CONFIRMED: v3 reduces regression on high-vocabulary datasets")
    else:
        print(f"\nVocabulary normalization does not improve ({delta:+.4f} MCC)")
        print("Further investigation needed")

    # Save results
    output = {
        "dataset": "sap_bsp669",
        "vocab_size": vocab_size,
        "reference_vocab": 20,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_evaluated": len(results),
        "accuracy": round(accuracy, 4),
        "epistemic_mean": epistemic_mean,
        "total_mean": total_mean,
        "results": {
            "static": static_metrics,
            "v2_current": {**v2_metrics, "threshold": v2_thresh, "factors": v2_factors},
            "v3_proposed": {**v3_metrics, "threshold": v3_thresh, "factors": v3_factors},
        },
        "mcc_improvements": {
            "v2_vs_static": round(v2_improvement, 4),
            "v3_vs_static": round(v3_improvement, 4),
            "v3_vs_v2": round(delta, 4),
        },
        "conclusion": "v3_improves" if delta > 0 else "v3_no_improvement",
        "elapsed_seconds": round(time.time() - start_time, 2)
    }

    output_path = AETHER_ROOT / "data" / "benchmarks" / "vocab_normalization_prototype.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print(f"\nElapsed: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
