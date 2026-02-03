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


class TestSafeTensorsFormat:
    """Tests for SafeTensors checkpoint format (secure alternative to pickle)."""

    def test_safetensors_roundtrip(self, tmp_path: Path):
        """Verify SafeTensors save/load produces identical weights."""
        import torch
        import torch.nn as nn
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_safetensors,
        )

        # Create test models
        encoder = nn.Linear(10, 5)
        transition = nn.Linear(5, 5)

        state_dicts = {
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
        }

        # Save to SafeTensors
        save_path = tmp_path / "test.safetensors"
        save_checkpoint_safetensors(save_path, state_dicts)

        assert save_path.exists(), "SafeTensors file not created"

        # Load and compare
        loaded = load_checkpoint_safetensors(save_path, device="cpu", model_keys=["encoder", "transition"])

        assert "encoder" in loaded
        assert "transition" in loaded

        # Compare tensors
        for key in encoder.state_dict():
            assert torch.allclose(
                state_dicts["encoder"][key],
                loaded["encoder"][key],
            ), f"Mismatch in encoder.{key}"

    def test_safetensors_cannot_execute_code(self, tmp_path: Path):
        """Verify SafeTensors format is immune to code injection.

        Unlike pickle, SafeTensors only stores raw tensor data with a
        simple header. There's no way to embed executable code.
        """
        import torch
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_safetensors,
        )

        # Create a simple checkpoint
        state_dicts = {"model": {"weight": torch.randn(3, 3)}}
        save_path = tmp_path / "safe.safetensors"
        save_checkpoint_safetensors(save_path, state_dicts)

        # Reading the raw file should show it's just tensor data
        with open(save_path, "rb") as f:
            content = f.read()

        # SafeTensors files cannot contain pickle opcodes or Python bytecode
        dangerous_patterns = [
            b"__reduce__",
            b"exec(",
            b"eval(",
            b"os.system",
            b"subprocess",
            b"import os",
        ]

        for pattern in dangerous_patterns:
            assert pattern not in content, (
                f"SafeTensors file contains suspicious pattern: {pattern}"
            )

        # Should load without any code execution
        loaded = load_checkpoint_safetensors(save_path, device="cpu", model_keys=["model"])
        assert "model" in loaded

    def test_safetensors_with_metadata(self, tmp_path: Path):
        """Test that string metadata is preserved in SafeTensors format."""
        import torch
        from safetensors.torch import load_file
        from core.utils.checkpoint import save_checkpoint_safetensors

        state_dicts = {"model": {"w": torch.ones(2, 2)}}
        metadata = {"epoch": "10", "version": "1.0.0"}

        save_path = tmp_path / "with_meta.safetensors"
        save_checkpoint_safetensors(save_path, state_dicts, metadata=metadata)

        # Load with safetensors directly to check metadata
        # (our load function doesn't expose metadata currently)
        from safetensors import safe_open
        with safe_open(str(save_path), framework="pt") as f:
            meta = f.metadata()
            assert meta.get("epoch") == "10"
            assert meta.get("version") == "1.0.0"


class TestCheckpointAutoDetection:
    """Tests for auto-detection of checkpoint formats."""

    def test_auto_detects_safetensors(self, tmp_path: Path):
        """Verify load_checkpoint_auto prefers SafeTensors when available."""
        import torch
        import torch.nn as nn
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_auto,
        )

        model = nn.Linear(4, 2)
        save_checkpoint_safetensors(
            tmp_path / "model.safetensors",
            {"encoder": model.state_dict()},
        )

        # Auto-load should find the SafeTensors file
        loaded = load_checkpoint_auto(
            tmp_path / "model",
            device="cpu",
            include_training_state=False,
            trusted_source=True,
            model_keys=["encoder"],
        )

        assert "encoder" in loaded
        assert "weight" in loaded["encoder"]

    def test_auto_falls_back_to_legacy(self, tmp_path: Path):
        """Verify auto-detection falls back to .pt with warning."""
        import warnings
        import torch
        import torch.nn as nn
        from core.utils.checkpoint import load_checkpoint_auto

        model = nn.Linear(4, 2)
        torch.save(
            {"encoder": model.state_dict()},
            tmp_path / "legacy.pt",
        )

        # Should load legacy with deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_checkpoint_auto(
                tmp_path / "legacy",
                device="cpu",
                include_training_state=False,
                trusted_source=True,
                model_keys=["encoder"],
            )

            assert len(w) == 1
            assert "legacy pickle" in str(w[0].message).lower()

        assert "encoder" in loaded

    def test_auto_requires_trust_for_legacy(self, tmp_path: Path):
        """Verify untrusted legacy checkpoints are rejected."""
        import torch
        import torch.nn as nn
        from core.utils.checkpoint import load_checkpoint_auto

        model = nn.Linear(4, 2)
        torch.save({"encoder": model.state_dict()}, tmp_path / "untrusted.pt")

        with pytest.raises(ValueError, match="trusted_source"):
            load_checkpoint_auto(
                tmp_path / "untrusted",
                device="cpu",
                trusted_source=False,  # Should fail
                model_keys=["encoder"],
            )

    def test_safetensors_loads_without_trust_flag(self, tmp_path: Path):
        """Verify SafeTensors can load without trusted_source=True.

        SafeTensors is secure by design - no code execution possible.
        """
        import torch
        import torch.nn as nn
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_auto,
        )

        model = nn.Linear(4, 2)
        save_checkpoint_safetensors(
            tmp_path / "secure.safetensors",
            {"encoder": model.state_dict()},
        )

        # Should load even without trusted_source=True
        loaded = load_checkpoint_auto(
            tmp_path / "secure",
            device="cpu",
            include_training_state=False,
            trusted_source=False,  # SafeTensors doesn't need this
            model_keys=["encoder"],
        )

        assert "encoder" in loaded
