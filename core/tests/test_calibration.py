"""Tests for calibration metrics and uncertainty decomposition.

Verifies:
1. ECE computation matches known benchmarks
2. MCE captures worst-bucket error
3. Brier score is mean squared error of probabilities
4. CalibrationTracker maintains rolling window correctly
5. Uncertainty decomposition: epistemic = model disagreement, aleatoric = inherent noise
6. Adaptive conformal inference widens sets on miscalibration
"""

from __future__ import annotations

import torch

from core.critic.calibration import (
    compute_ece,
    compute_mce,
    compute_brier,
    CalibrationTracker,
)
from core.critic.decomposition import (
    decompose_from_ensemble,
    decompose_from_mc_dropout,
)
from core.critic.adaptive_conformal import AdaptiveConformalPredictor


# --- ECE Tests ---


class TestECE:
    def test_perfect_calibration(self):
        """Perfect calibration: predicted probability matches actual frequency."""
        # 100 predictions at 0.7 confidence, 70% correct
        n = 1000
        preds = torch.full((n,), 0.7)
        actuals = torch.zeros(n)
        actuals[:700] = 1.0

        ece = compute_ece(preds, actuals, n_bins=10)
        assert ece < 0.05, f"Perfect calibration should have near-zero ECE, got {ece}"

    def test_poor_calibration(self):
        """Overconfident model: predicts 0.9 but only 50% correct."""
        n = 1000
        preds = torch.full((n,), 0.9)
        actuals = torch.zeros(n)
        actuals[:500] = 1.0

        ece = compute_ece(preds, actuals, n_bins=10)
        assert ece > 0.3, f"Poor calibration should have high ECE, got {ece}"

    def test_ece_bounds(self):
        """ECE should be in [0, 1]."""
        preds = torch.rand(200)
        actuals = (torch.rand(200) > 0.5).float()

        ece = compute_ece(preds, actuals)
        assert 0.0 <= ece <= 1.0

    def test_empty_predictions(self):
        """ECE of empty input should be 0."""
        ece = compute_ece(torch.tensor([]), torch.tensor([]))
        assert ece == 0.0


class TestMCE:
    def test_mce_geq_ece(self):
        """MCE (max bucket error) should be >= ECE (average)."""
        preds = torch.rand(500)
        actuals = (torch.rand(500) > 0.5).float()

        ece = compute_ece(preds, actuals)
        mce = compute_mce(preds, actuals)
        assert mce >= ece - 1e-6


class TestBrier:
    def test_perfect_predictions(self):
        """Perfect binary predictions should have Brier score near 0."""
        preds = torch.tensor([1.0, 0.0, 1.0, 0.0])
        actuals = torch.tensor([1.0, 0.0, 1.0, 0.0])

        brier = compute_brier(preds, actuals)
        assert brier < 1e-6

    def test_worst_predictions(self):
        """Completely wrong predictions should have Brier score near 1."""
        preds = torch.tensor([1.0, 0.0, 1.0, 0.0])
        actuals = torch.tensor([0.0, 1.0, 0.0, 1.0])

        brier = compute_brier(preds, actuals)
        assert brier > 0.9

    def test_brier_bounds(self):
        preds = torch.rand(100)
        actuals = (torch.rand(100) > 0.5).float()
        brier = compute_brier(preds, actuals)
        assert 0.0 <= brier <= 1.0


# --- CalibrationTracker Tests ---


class TestCalibrationTracker:
    def test_empty_tracker(self):
        tracker = CalibrationTracker(window_size=100)
        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == 0
        assert metrics["ece"] == 0.0

    def test_tracks_predictions(self):
        tracker = CalibrationTracker(window_size=100)
        for i in range(50):
            tracker.update(0.7, 1.0 if i < 35 else 0.0)

        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == 50
        assert metrics["ece"] >= 0

    def test_rolling_window(self):
        tracker = CalibrationTracker(window_size=10)
        for i in range(20):
            tracker.update(0.5, float(i % 2))

        assert tracker.count == 10  # Window capped at 10

    def test_batch_update(self):
        tracker = CalibrationTracker(window_size=100)
        preds = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        actuals = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        tracker.update_batch(preds, actuals)

        assert tracker.count == 5

    def test_buckets_returned(self):
        tracker = CalibrationTracker(window_size=100, n_bins=5)
        for _ in range(100):
            p = torch.rand(1).item()
            tracker.update(p, float(torch.rand(1).item() > 0.5))

        metrics = tracker.compute_metrics()
        assert len(metrics["buckets"]) == 5


# --- Uncertainty Decomposition Tests ---


class TestDecomposition:
    def test_ensemble_disagreement_is_epistemic(self):
        """When ensemble members disagree, epistemic uncertainty is high."""
        # Ensemble with high disagreement
        predictions = [
            torch.tensor([0.9, 0.1]),  # Model 1: confident class 0
            torch.tensor([0.1, 0.9]),  # Model 2: confident class 1
            torch.tensor([0.5, 0.5]),  # Model 3: uncertain
        ]

        result = decompose_from_ensemble(predictions)
        assert result["epistemic"] > result["aleatoric"]
        assert result["epistemicRatio"] > 0.5

    def test_ensemble_agreement_low_epistemic(self):
        """When ensemble members agree, epistemic uncertainty is low."""
        predictions = [
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.88, 0.12]),
            torch.tensor([0.91, 0.09]),
        ]

        result = decompose_from_ensemble(predictions)
        assert result["epistemic"] < result["aleatoric"]
        assert result["epistemicRatio"] < 0.5

    def test_total_equals_sum(self):
        """Total uncertainty = epistemic + aleatoric."""
        predictions = [torch.rand(5).softmax(dim=0) for _ in range(10)]
        result = decompose_from_ensemble(predictions)
        assert abs(result["total"] - result["epistemic"] - result["aleatoric"]) < 1e-4

    def test_epistemic_ratio_bounds(self):
        """Epistemic ratio should be in [0, 1]."""
        predictions = [torch.rand(3).softmax(dim=0) for _ in range(5)]
        result = decompose_from_ensemble(predictions)
        assert 0.0 <= result["epistemicRatio"] <= 1.0

    def test_empty_ensemble(self):
        result = decompose_from_ensemble([])
        assert result["total"] == 0.0
        assert result["epistemic"] == 0.0

    def test_mc_dropout_uses_same_math(self):
        predictions = [torch.rand(4).softmax(dim=0) for _ in range(5)]
        ens = decompose_from_ensemble(predictions)
        mc = decompose_from_mc_dropout(predictions)
        # Same math, different method label
        assert abs(ens["total"] - mc["total"]) < 1e-6
        assert mc["method"] == "mc_dropout"

    def test_the_key_insight(self):
        """THE NOVEL CONTRIBUTION: high aleatoric + low epistemic means
        don't tighten governance; high epistemic means do tighten.

        Scenario 1: Ensemble agrees on 50/50 → high aleatoric, low epistemic
        Scenario 2: Ensemble disagrees → high epistemic
        """
        # Scenario 1: All models agree the outcome is uncertain
        # (50/50 probability). This is ALEATORIC — irreducible.
        agree_uncertain = [
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
        ]
        result1 = decompose_from_ensemble(agree_uncertain)
        assert result1["epistemicRatio"] < 0.1, (
            f"Agreed uncertainty should be aleatoric, got epistemic_ratio={result1['epistemicRatio']}"
        )

        # Scenario 2: Models disagree about which class is correct.
        # This is EPISTEMIC — more data would resolve the disagreement.
        disagree = [
            torch.tensor([0.95, 0.05]),
            torch.tensor([0.05, 0.95]),
            torch.tensor([0.6, 0.4]),
        ]
        result2 = decompose_from_ensemble(disagree)
        assert result2["epistemicRatio"] > 0.3, (
            f"Model disagreement should be epistemic, got epistemic_ratio={result2['epistemicRatio']}"
        )


# --- Adaptive Conformal Inference Tests ---


class TestAdaptiveConformal:
    def test_initial_state(self):
        acp = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.01)
        assert abs(acp.alpha - 0.1) < 1e-6

    def test_alpha_decreases_on_miss(self):
        """When predictions miss, alpha decreases → threshold (1-α) rises → wider sets.

        ACI update: α_{t+1} = α_t + γ × (α_target - err_t)
        On miss: err_t = 1, so Δα = γ × (0.1 - 1) = -0.045 → α decreases.
        Lower α → threshold 1-α increases → prediction set must cover more.
        """
        acp = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.05)
        initial_alpha = acp.alpha

        # Feed a series of misses
        for _ in range(5):
            acp.update(prediction_set=[0], actual=5)  # miss

        assert acp.alpha < initial_alpha, "Alpha should decrease on misses (ACI: α_target - 1 < 0)"

    def test_alpha_increases_on_hit(self):
        """When predictions hit, alpha increases → threshold (1-α) falls → narrower sets.

        ACI update: α_{t+1} = α_t + γ × (α_target - err_t)
        On hit: err_t = 0, so Δα = γ × (0.1 - 0) = +0.005 → α increases.
        Higher α → threshold 1-α decreases → prediction set can be smaller.
        """
        acp = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.05)

        # Start with lowered alpha from misses
        for _ in range(10):
            acp.update(prediction_set=[0], actual=5)  # miss
        lowered_alpha = acp.alpha

        # Feed a series of hits
        for _ in range(10):
            acp.update(prediction_set=[0, 1, 2], actual=1)  # hit

        assert acp.alpha > lowered_alpha, "Alpha should increase on hits (ACI: α_target - 0 > 0)"

    def test_prediction_set_widens_with_higher_alpha(self):
        """Higher alpha (more uncertainty) → larger prediction sets."""
        acp = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.05)

        scores = torch.tensor([0.05, 0.10, 0.15, 0.30, 0.40])

        # Low alpha → need cumsum >= 0.95 → large set
        # High alpha → need cumsum >= 0.50 → small set
        # Note: alpha is miscoverage rate, so threshold = 1 - alpha

        acp.alpha = 0.05  # Need 95% coverage → large set
        large_set = acp.get_prediction_set(scores)

        acp.alpha = 0.5  # Need 50% coverage → small set
        small_set = acp.get_prediction_set(scores)

        assert len(large_set) >= len(small_set)

    def test_alpha_stays_bounded(self):
        """Alpha should stay in (0, 1) regardless of input."""
        acp = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.1)

        # Many misses
        for _ in range(1000):
            acp.update(prediction_set=[0], actual=5)
        assert 0 < acp.alpha < 1

        # Many hits
        for _ in range(1000):
            acp.update(prediction_set=[0, 1, 2, 3, 4], actual=2)
        assert 0 < acp.alpha < 1
