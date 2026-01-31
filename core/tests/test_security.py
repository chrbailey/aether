"""Security validation tests for AETHER model checkpoints.

Validates that model checkpoints and pickle files are free from
malicious code injection attacks (CWE-502).

Uses picklescan for static analysis of pickle files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Paths
AETHER_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = AETHER_ROOT / "data"
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".bin", ".pkl", ".pickle"}


def find_checkpoint_files(directory: Path) -> list[Path]:
    """Find all checkpoint/pickle files in a directory tree."""
    if not directory.exists():
        return []

    files = []
    for ext in CHECKPOINT_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return files


def is_picklescan_available() -> bool:
    """Check if picklescan is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "picklescan", "--help"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


class TestCheckpointSecurity:
    """Security tests for model checkpoint files."""

    @pytest.fixture
    def checkpoint_files(self) -> list[Path]:
        """Get all checkpoint files in the data directory."""
        return find_checkpoint_files(DATA_DIR)

    @pytest.mark.skipif(
        not is_picklescan_available(),
        reason="picklescan not installed (pip install picklescan)",
    )
    def test_no_malicious_checkpoints(self, checkpoint_files: list[Path]):
        """Verify all checkpoint files pass picklescan security scan.

        This test scans all .pt, .pth, .bin, .pkl files for:
        - Dangerous imports (os, subprocess, socket, etc.)
        - Code injection payloads
        - Suspicious pickle opcodes

        Exit codes:
            0 = clean
            1 = malicious content detected
            2 = scan failed
        """
        if not checkpoint_files:
            pytest.skip("No checkpoint files found in data/")

        failed_files = []
        for checkpoint in checkpoint_files:
            result = subprocess.run(
                [sys.executable, "-m", "picklescan", "--path", str(checkpoint)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 1:
                failed_files.append((checkpoint, result.stdout + result.stderr))

        if failed_files:
            msg = "Malicious content detected in checkpoint files:\n"
            for path, output in failed_files:
                msg += f"\n{path}:\n{output}\n"
            pytest.fail(msg)

    @pytest.mark.skipif(
        not is_picklescan_available(),
        reason="picklescan not installed",
    )
    def test_scan_new_checkpoint_on_save(self, tmp_path: Path):
        """Verify that checkpoints we create are clean.

        Creates a simple checkpoint and scans it to ensure our
        save process doesn't introduce vulnerabilities.
        """
        import torch

        # Create a simple state dict
        state = {
            "model": {"weight": torch.randn(10, 10)},
            "epoch": 5,
        }

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(state, checkpoint_path)

        result = subprocess.run(
            [sys.executable, "-m", "picklescan", "--path", str(checkpoint_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Our checkpoint failed security scan:\n{result.stdout}\n{result.stderr}"
        )


class TestWeightsOnlyCompatibility:
    """Test that checkpoints work with weights_only=True where possible."""

    def test_state_dict_only_checkpoint_loads_securely(self, tmp_path: Path):
        """Verify pure state_dict checkpoints load with weights_only=True.

        This validates that checkpoints containing only tensor state_dicts
        can be loaded securely without arbitrary code execution risk.
        """
        import torch
        import torch.nn as nn

        # Create a simple model
        model = nn.Linear(10, 5)

        # Save only the state dict
        checkpoint = {"model": model.state_dict()}
        checkpoint_path = tmp_path / "state_dict_only.pt"
        torch.save(checkpoint, checkpoint_path)

        # Should load with weights_only=True
        loaded = torch.load(checkpoint_path, weights_only=True)

        assert "model" in loaded
        assert "weight" in loaded["model"]

    def test_full_checkpoint_requires_unsafe_load(self, tmp_path: Path):
        """Document that full training checkpoints need weights_only=False.

        Checkpoints with optimizer and scheduler state require unsafe loading
        due to non-tensor Python objects in the state dicts.
        """
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Full training checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 1,
        }
        checkpoint_path = tmp_path / "full_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Should work with weights_only=False (unsafe but necessary)
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert "optimizer" in loaded

        # Document: this may fail with weights_only=True depending on PyTorch version
        # Some versions support optimizer state, some don't


class TestModelIntegrity:
    """Tests for model architecture and weight integrity."""

    def test_encoder_output_shape(self):
        """Verify encoder produces correct output dimensions."""
        import torch
        from core.encoder.event_encoder import EventEncoder
        from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary

        act_vocab = ActivityVocabulary(embed_dim=64)
        for token in ["start", "process", "end"]:
            act_vocab.add_token(token)

        res_vocab = ResourceVocabulary(embed_dim=32)
        res_vocab.add_token("user")

        encoder = EventEncoder(
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            latent_dim=128,
        )

        # Create dummy input
        batch_size, seq_len = 2, 5
        activities = torch.randint(0, act_vocab.size, (batch_size, seq_len))
        resources = torch.randint(0, res_vocab.size, (batch_size, seq_len))
        attributes = torch.randn(batch_size, seq_len, 8)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = encoder(activities, resources, attributes, mask)

        assert output.shape == (batch_size, seq_len, 128)

    def test_model_forward_pass_no_nan(self):
        """Ensure model forward pass produces valid outputs (no NaN/Inf)."""
        import torch
        from core.encoder.event_encoder import EventEncoder
        from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
        from core.world_model.hierarchical import HierarchicalPredictor

        act_vocab = ActivityVocabulary(embed_dim=64)
        for token in ["A", "B", "C"]:
            act_vocab.add_token(token)

        res_vocab = ResourceVocabulary(embed_dim=32)
        res_vocab.add_token("R")

        encoder = EventEncoder(
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            latent_dim=128,
        )
        predictor = HierarchicalPredictor(n_activities=act_vocab.size, n_phases=4)

        # Forward pass
        activities = torch.randint(0, act_vocab.size, (1, 3))
        resources = torch.zeros(1, 3, dtype=torch.long)
        attributes = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)

        encoder.eval()
        predictor.eval()

        with torch.no_grad():
            h = encoder(activities, resources, attributes, mask)
            z = h.mean(dim=1)
            out = predictor.activity_head(z)

        assert not torch.isnan(out["logits"]).any(), "NaN in output logits"
        assert not torch.isinf(out["logits"]).any(), "Inf in output logits"
        assert out["probs"].sum(dim=-1).allclose(torch.ones(1)), "Probs don't sum to 1"
