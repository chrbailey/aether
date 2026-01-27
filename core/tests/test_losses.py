"""Tests for AETHER loss functions: VICReg, SIGReg, and EnergyContrastive."""

from __future__ import annotations

import pytest
import torch

from ..training.losses import EnergyContrastiveLoss, SIGRegLoss, VICRegLoss


# ---------------------------------------------------------------------------
# VICReg Loss Tests
# ---------------------------------------------------------------------------


class TestVICRegLoss:
    """Tests for VICReg loss components."""

    def test_output_keys(self) -> None:
        loss_fn = VICRegLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        out = loss_fn(z1, z2)
        assert set(out.keys()) == {"total", "invariance", "variance", "covariance"}

    def test_all_tensors(self) -> None:
        loss_fn = VICRegLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        out = loss_fn(z1, z2)
        for key, val in out.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
            assert val.ndim == 0, f"{key} is not scalar"

    def test_invariance_zero_when_identical(self) -> None:
        loss_fn = VICRegLoss()
        z = torch.randn(16, 64)
        out = loss_fn(z, z)
        assert out["invariance"].item() == pytest.approx(0.0, abs=1e-6)

    def test_variance_zero_when_spread(self) -> None:
        """Variance loss should be zero when std exceeds the target."""
        loss_fn = VICRegLoss(variance_target=0.1)
        z1 = torch.randn(32, 64) * 5.0  # Large std
        z2 = torch.randn(32, 64) * 5.0
        out = loss_fn(z1, z2)
        assert out["variance"].item() == pytest.approx(0.0, abs=0.01)

    def test_variance_high_when_collapsed(self) -> None:
        """Variance loss should be high when all embeddings are near-identical."""
        loss_fn = VICRegLoss(variance_target=1.0)
        z = torch.ones(32, 64) + torch.randn(32, 64) * 0.001  # Near-constant
        out = loss_fn(z, z)
        assert out["variance"].item() > 0.5

    def test_covariance_low_when_decorrelated(self) -> None:
        """Covariance loss should be low for uncorrelated dimensions."""
        loss_fn = VICRegLoss()
        z = torch.randn(100, 64)  # i.i.d. → low covariance
        out = loss_fn(z, z)
        # Random Gaussian dims should have low off-diagonal covariance
        assert out["covariance"].item() < 0.5

    def test_backward_pass(self) -> None:
        """Gradients should flow correctly."""
        loss_fn = VICRegLoss()
        z1 = torch.randn(16, 64, requires_grad=True)
        z2 = torch.randn(16, 64, requires_grad=True)
        out = loss_fn(z1, z2)
        out["total"].backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert not z1.grad.isnan().any()
        assert not z2.grad.isnan().any()

    def test_no_nan_with_normal_inputs(self) -> None:
        loss_fn = VICRegLoss()
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        out = loss_fn(z1, z2)
        for key, val in out.items():
            assert not val.isnan().any(), f"NaN in {key}"

    def test_clamp_prevents_overflow(self) -> None:
        """Large values should not cause NaN due to clamping in covariance."""
        loss_fn = VICRegLoss()
        z1 = torch.randn(16, 64) * 100.0  # Very large values
        z2 = torch.randn(16, 64) * 100.0
        out = loss_fn(z1, z2)
        # May be large but should NOT be NaN
        assert not out["total"].isnan(), "Total loss is NaN with large inputs"

    def test_weight_scaling(self) -> None:
        """Weights should scale the respective loss components."""
        z1 = torch.randn(32, 64)
        z2 = torch.randn(32, 64)

        loss_1x = VICRegLoss(invariance_weight=1.0, variance_weight=1.0, covariance_weight=1.0)
        loss_10x = VICRegLoss(invariance_weight=10.0, variance_weight=10.0, covariance_weight=10.0)

        out_1 = loss_1x(z1, z2)
        out_10 = loss_10x(z1, z2)

        # 10x weights → ~10x total (since components are the same)
        ratio = out_10["total"].item() / max(out_1["total"].item(), 1e-8)
        assert 9.0 < ratio < 11.0


# ---------------------------------------------------------------------------
# SIGReg Loss Tests
# ---------------------------------------------------------------------------


class TestSIGRegLoss:
    """Tests for SIGReg loss (Balestriero & LeCun, 2025)."""

    def test_output_keys(self) -> None:
        """Should return VICReg-compatible keys for seamless integration."""
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        out = loss_fn(z1, z2)
        assert set(out.keys()) == {"total", "invariance", "variance", "covariance"}

    def test_covariance_placeholder_is_zero(self) -> None:
        """Covariance should be zero (SIGReg subsumes it)."""
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        out = loss_fn(z1, z2)
        assert out["covariance"].item() == 0.0

    def test_all_tensors(self) -> None:
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        out = loss_fn(z1, z2)
        for key, val in out.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
            assert val.ndim == 0, f"{key} is not scalar"

    def test_invariance_zero_when_identical(self) -> None:
        loss_fn = SIGRegLoss()
        z = torch.randn(16, 64)
        out = loss_fn(z, z)
        assert out["invariance"].item() == pytest.approx(0.0, abs=1e-6)

    def test_sigreg_low_when_spread(self) -> None:
        """SIGReg should be low when eigenvalues are large (well-spread)."""
        loss_fn = SIGRegLoss(temperature=1.0)
        z1 = torch.randn(64, 32) * 5.0  # Large spread → big eigenvalues
        z2 = torch.randn(64, 32) * 5.0
        out = loss_fn(z1, z2)
        # sigmoid(large/1) ≈ 1 → -log(1) ≈ 0
        assert out["variance"].item() < 0.5

    def test_sigreg_high_when_collapsed(self) -> None:
        """SIGReg should be high when representations collapse (near-constant)."""
        loss_fn = SIGRegLoss(temperature=1.0)
        # Near-collapsed: all rows nearly the same → many near-zero eigenvalues
        base = torch.randn(1, 32)
        z = base.expand(64, -1) + torch.randn(64, 32) * 0.001
        out = loss_fn(z, z)
        # sigmoid(near-zero/1) ≈ 0.5 → -log(0.5) = 0.69 per eigenvalue
        assert out["variance"].item() > 0.5

    def test_temperature_effect(self) -> None:
        """Higher temperature → eigenvalues divided by more → sigmoid closer to 0.5."""
        z1 = torch.randn(32, 64)
        z2 = torch.randn(32, 64)

        loss_low_temp = SIGRegLoss(sigreg_weight=1.0, invariance_weight=0.0, temperature=0.1)
        loss_high_temp = SIGRegLoss(sigreg_weight=1.0, invariance_weight=0.0, temperature=100.0)

        out_low = loss_low_temp(z1, z2)
        out_high = loss_high_temp(z1, z2)

        # High temperature pushes all eigenvalues toward sigmoid(0)=0.5
        # → -log(0.5) = 0.693 per eigenvalue → higher loss
        assert out_high["total"].item() > out_low["total"].item()

    def test_backward_pass(self) -> None:
        """Gradients should flow through eigendecomposition."""
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64, requires_grad=True)
        z2 = torch.randn(16, 64, requires_grad=True)
        out = loss_fn(z1, z2)
        out["total"].backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert not z1.grad.isnan().any()
        assert not z2.grad.isnan().any()

    def test_no_nan_with_normal_inputs(self) -> None:
        loss_fn = SIGRegLoss()
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        out = loss_fn(z1, z2)
        for key, val in out.items():
            assert not val.isnan().any(), f"NaN in {key}"

    def test_no_nan_with_large_inputs(self) -> None:
        """SIGReg should handle large inputs without NaN (sigmoid saturates)."""
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64) * 100.0
        z2 = torch.randn(16, 64) * 100.0
        out = loss_fn(z1, z2)
        assert not out["total"].isnan(), "NaN with large inputs"

    def test_no_nan_with_small_inputs(self) -> None:
        """Near-zero inputs should not cause NaN."""
        loss_fn = SIGRegLoss()
        z1 = torch.randn(16, 64) * 0.001
        z2 = torch.randn(16, 64) * 0.001
        out = loss_fn(z1, z2)
        assert not out["total"].isnan(), "NaN with small inputs"

    def test_drop_in_replacement_for_vicreg(self) -> None:
        """SIGReg output shape and keys should match VICReg for trainer compat."""
        vicreg = VICRegLoss()
        sigreg = SIGRegLoss()

        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)

        out_v = vicreg(z1, z2)
        out_s = sigreg(z1, z2)

        assert set(out_v.keys()) == set(out_s.keys())
        for key in out_v:
            assert out_v[key].shape == out_s[key].shape

    def test_batch_size_invariance(self) -> None:
        """Loss should be roughly similar regardless of batch size."""
        loss_fn = SIGRegLoss(invariance_weight=0.0, sigreg_weight=1.0)
        torch.manual_seed(42)
        z_big = torch.randn(128, 64)
        z_small = z_big[:32]

        out_big = loss_fn(z_big, z_big)
        out_small = loss_fn(z_small, z_small)

        # Not exact but should be in same ballpark
        ratio = out_big["total"].item() / max(out_small["total"].item(), 1e-8)
        assert 0.3 < ratio < 3.0


# ---------------------------------------------------------------------------
# Energy Contrastive Loss Tests
# ---------------------------------------------------------------------------


class TestEnergyContrastiveLoss:
    """Tests for energy contrastive loss."""

    def test_output_keys(self) -> None:
        loss_fn = EnergyContrastiveLoss()
        pos = torch.randn(16)
        neg = torch.randn(16)
        out = loss_fn(pos, neg)
        assert set(out.keys()) == {"total", "positive", "negative"}

    def test_all_tensors(self) -> None:
        loss_fn = EnergyContrastiveLoss()
        pos = torch.randn(16)
        neg = torch.randn(16)
        out = loss_fn(pos, neg)
        for key, val in out.items():
            assert isinstance(val, torch.Tensor)
            assert val.ndim == 0

    def test_positive_is_mean_energy(self) -> None:
        """Positive loss should be the mean of positive energies."""
        loss_fn = EnergyContrastiveLoss()
        pos = torch.tensor([1.0, 2.0, 3.0])
        neg = torch.tensor([5.0, 5.0, 5.0])
        out = loss_fn(pos, neg)
        assert out["positive"].item() == pytest.approx(2.0, abs=1e-6)

    def test_negative_zero_when_above_margin(self) -> None:
        """Negative loss should be zero when all negative energies exceed margin."""
        loss_fn = EnergyContrastiveLoss(margin=1.0)
        pos = torch.tensor([0.5])
        neg = torch.tensor([2.0])  # > margin
        out = loss_fn(pos, neg)
        assert out["negative"].item() == pytest.approx(0.0, abs=1e-6)

    def test_negative_nonzero_when_below_margin(self) -> None:
        """Negative loss should penalize when negative energy is below margin."""
        loss_fn = EnergyContrastiveLoss(margin=5.0)
        pos = torch.tensor([0.5])
        neg = torch.tensor([2.0])  # < margin (5.0)
        out = loss_fn(pos, neg)
        assert out["negative"].item() > 0

    def test_backward_pass(self) -> None:
        loss_fn = EnergyContrastiveLoss()
        pos = torch.randn(16, requires_grad=True)
        neg = torch.randn(16, requires_grad=True)
        out = loss_fn(pos, neg)
        out["total"].backward()
        assert pos.grad is not None
        assert neg.grad is not None

    def test_margin_scaling(self) -> None:
        """Larger margin → more negative samples penalized → higher loss."""
        pos = torch.tensor([0.1, 0.2, 0.3])
        neg = torch.tensor([0.5, 0.8, 1.2])

        loss_small = EnergyContrastiveLoss(margin=0.3)
        loss_large = EnergyContrastiveLoss(margin=5.0)

        out_small = loss_small(pos, neg)
        out_large = loss_large(pos, neg)

        assert out_large["negative"].item() > out_small["negative"].item()
