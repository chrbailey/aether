"""Tests for the World Model components.

Verifies:
1. TransitionModel produces correct-shape latent predictions
2. Multiple futures can be generated via variant sampling
3. EnergyScorer produces low energy for plausible transitions
4. HierarchicalPredictor produces activity/phase/outcome predictions
5. LatentVariable supports Gumbel-Softmax sampling and KL divergence
"""

from __future__ import annotations

import torch
import pytest

from core.world_model.latent import LatentVariable, NUM_VARIANTS
from core.world_model.transition import TransitionModel, NUM_ACTIONS
from core.world_model.energy import EnergyScorer
from core.world_model.hierarchical import (
    HierarchicalPredictor,
)


class TestLatentVariable:
    @pytest.fixture
    def lv(self):
        return LatentVariable(latent_dim=128, n_variants=NUM_VARIANTS)

    def test_forward_output_keys(self, lv):
        z = torch.randn(4, 128)
        result = lv(z)
        assert "logits" in result
        assert "probs" in result
        assert "sample" in result
        assert "hard_sample" in result

    def test_forward_output_shapes(self, lv):
        z = torch.randn(4, 128)
        result = lv(z)
        assert result["logits"].shape == (4, NUM_VARIANTS)
        assert result["probs"].shape == (4, NUM_VARIANTS)
        assert result["sample"].shape == (4, NUM_VARIANTS)

    def test_probs_sum_to_one(self, lv):
        z = torch.randn(4, 128)
        result = lv(z)
        row_sums = result["probs"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_training_uses_gumbel_softmax(self, lv):
        lv.train()
        z = torch.randn(4, 128)
        result = lv(z)
        # In training mode, sample uses Gumbel-Softmax
        assert result["sample"].shape == (4, NUM_VARIANTS)
        # Hard sample should be approximately one-hot
        hard_sums = result["hard_sample"].sum(dim=-1)
        assert torch.allclose(hard_sums, torch.ones(4), atol=1e-5)

    def test_entropy(self, lv):
        lv.eval()
        # Random z vectors will produce some entropy
        z = torch.randn(4, 128)
        entropy = lv.entropy(z)
        assert entropy.shape == (4,)
        assert (entropy >= 0).all()

    def test_kl_divergence(self, lv):
        z = torch.randn(4, 128)
        logits = lv.logit_head(z)
        kl = lv.kl_divergence(z, logits)
        # KL(q || q) = 0
        assert kl.abs().max().item() < 1e-4

    def test_log_prob(self, lv):
        z = torch.randn(4, 128)
        indices = torch.randint(0, NUM_VARIANTS, (4,))
        log_probs = lv.log_prob(z, indices)
        assert log_probs.shape == (4,)
        assert (log_probs <= 0).all()  # Log probs are non-positive

    def test_anneal_temperature(self, lv):
        lv.anneal_temperature(0.5)
        assert lv.temperature == 0.5
        lv.anneal_temperature(0.001)
        assert lv.temperature == 0.01  # Clamped minimum


class TestTransitionModel:
    @pytest.fixture
    def model(self):
        return TransitionModel(latent_dim=128, hidden_dim=256, n_blocks=2)

    def test_output_shape(self, model):
        batch = 4
        z_t = torch.randn(batch, 128)
        action = torch.zeros(batch, NUM_ACTIONS)
        action[:, 1] = 1.0  # standard mode
        variant = torch.zeros(batch, NUM_VARIANTS)
        variant[:, 0] = 1.0  # standard path

        z_next = model(z_t, action, variant)
        assert z_next.shape == (batch, 128)

    def test_residual_connection(self, model):
        """Output should be close to input when model is randomly initialized
        (the residual skip z_t + ... keeps output near input)."""
        z_t = torch.randn(1, 128)
        action = torch.zeros(1, NUM_ACTIONS)
        action[0, 0] = 1.0
        variant = torch.zeros(1, NUM_VARIANTS)
        variant[0, 0] = 1.0

        model.eval()
        with torch.no_grad():
            z_next = model(z_t, action, variant)

        # Due to residual connection, output should be correlated with input
        cosine_sim = torch.nn.functional.cosine_similarity(
            z_t, z_next, dim=-1
        ).item()
        assert cosine_sim > 0.3, f"Residual connection weak: cos_sim={cosine_sim}"

    def test_multiple_futures(self, model):
        batch = 2
        n_samples = 5
        z_t = torch.randn(batch, 128)
        action = torch.zeros(batch, NUM_ACTIONS)
        action[:, 1] = 1.0

        # Different path variants produce different futures
        variant_samples = torch.randn(batch, n_samples, NUM_VARIANTS).softmax(dim=-1)

        model.eval()
        with torch.no_grad():
            futures = model.predict_multiple_futures(z_t, action, variant_samples)

        assert futures.shape == (batch, n_samples, 128)

        # Different variants should produce different predictions
        diff = (futures[0, 0] - futures[0, 1]).abs().sum().item()
        assert diff > 0, "Different variants produced identical predictions"

    def test_gradients_flow(self, model):
        z_t = torch.randn(2, 128)
        action = torch.zeros(2, NUM_ACTIONS)
        action[:, 0] = 1.0
        variant = torch.randn(2, NUM_VARIANTS).softmax(dim=-1)

        z_next = model(z_t, action, variant)
        loss = z_next.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


class TestEnergyScorer:
    @pytest.fixture
    def scorer(self):
        return EnergyScorer(latent_dim=128)

    def test_output_shape(self, scorer):
        z_pred = torch.randn(4, 128)
        z_actual = torch.randn(4, 128)
        energy = scorer(z_pred, z_actual)
        assert energy.shape == (4,)

    def test_identical_states_low_energy(self, scorer):
        z = torch.randn(1, 128)
        energy = scorer(z, z)
        assert energy.item() < 1e-6, "Identical states should have near-zero energy"

    def test_different_states_higher_energy(self, scorer):
        z1 = torch.randn(1, 128)
        z2 = torch.randn(1, 128) * 10  # Very different
        energy_same = scorer(z1, z1)
        energy_diff = scorer(z1, z2)
        assert energy_diff.item() > energy_same.item()

    def test_energy_is_non_negative(self, scorer):
        z_pred = torch.randn(100, 128)
        z_actual = torch.randn(100, 128)
        energy = scorer(z_pred, z_actual)
        assert (energy >= -1e-6).all()

    def test_normalized_energy_bounded(self, scorer):
        z_pred = torch.randn(10, 128)
        z_actual = torch.randn(10, 128)
        norm = scorer.normalized_energy(z_pred, z_actual)
        assert (norm >= 0).all()
        assert (norm <= 1).all()


class TestHierarchicalPredictor:
    @pytest.fixture
    def predictor(self):
        return HierarchicalPredictor(
            latent_dim=128,
            n_activities=10,
            n_phases=5,
        )

    def test_activity_head_output(self, predictor):
        z = torch.randn(4, 128)
        result = predictor.activity_head(z)
        assert "logits" in result
        assert "probs" in result
        assert "delta_hours" in result
        assert result["logits"].shape == (4, 10)
        probs = result["probs"]
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_phase_head_output(self, predictor):
        z = torch.randn(4, 128)
        result = predictor.phase_head(z)
        assert "current_logits" in result
        assert "next_logits" in result
        assert "transition_hours" in result
        assert result["current_logits"].shape == (4, 5)

    def test_outcome_head_output(self, predictor):
        z = torch.randn(4, 128)
        result = predictor.outcome_head(z)
        assert "ontime_prob" in result
        assert "rework_prob" in result
        assert "remaining_hours" in result
        # Probabilities should be in [0, 1]
        assert (result["ontime_prob"] >= 0).all()
        assert (result["ontime_prob"] <= 1).all()
        # Hours should be positive (Softplus)
        assert (result["remaining_hours"] > 0).all()

    def test_full_predict(self, predictor):
        z = torch.randn(4, 128)
        result = predictor(z)
        assert "activity" in result
        assert "phase" in result
        assert "outcome" in result
        assert result["activity"]["logits"].shape == (4, 10)
        assert result["phase"]["current_logits"].shape == (4, 5)
        assert result["outcome"]["ontime_prob"].shape == (4, 1)
