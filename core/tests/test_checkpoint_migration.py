"""Tests for SafeTensors checkpoint migration.

Verifies:
- SafeTensors round-trip (save/load cycle)
- Training state round-trip (pickle)
- Legacy .pt to SafeTensors migration
- Format auto-detection
- Backward compatibility with existing checkpoints
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn


class TestSafeTensorsRoundTrip:
    """Test SafeTensors save/load produces identical model weights."""

    def test_simple_state_dict(self, tmp_path: Path):
        """Single model state dict round-trips correctly."""
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_safetensors,
        )

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        original_state = model.state_dict()
        save_checkpoint_safetensors(
            tmp_path / "single.safetensors",
            {"model": original_state},
        )

        loaded = load_checkpoint_safetensors(
            tmp_path / "single.safetensors",
            device="cpu",
            model_keys=["model"],
        )

        # Compare all tensors
        for key in original_state:
            assert key in loaded["model"], f"Missing key: {key}"
            assert torch.equal(original_state[key], loaded["model"][key]), (
                f"Mismatch at {key}"
            )

    def test_multiple_models(self, tmp_path: Path):
        """Multiple model state dicts round-trip correctly."""
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_safetensors,
        )

        encoder = nn.Linear(128, 64)
        transition = nn.Linear(64, 64)
        predictor = nn.Linear(64, 10)

        state_dicts = {
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
            "predictor": predictor.state_dict(),
        }

        save_checkpoint_safetensors(tmp_path / "multi.safetensors", state_dicts)

        loaded = load_checkpoint_safetensors(
            tmp_path / "multi.safetensors",
            device="cpu",
            model_keys=["encoder", "transition", "predictor"],
        )

        for model_name in state_dicts:
            assert model_name in loaded
            for key in state_dicts[model_name]:
                assert torch.equal(
                    state_dicts[model_name][key],
                    loaded[model_name][key],
                )

    def test_device_mapping(self, tmp_path: Path):
        """Tensors load to the specified device."""
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_safetensors,
        )

        model = nn.Linear(5, 3)
        save_checkpoint_safetensors(
            tmp_path / "device.safetensors",
            {"model": model.state_dict()},
        )

        loaded = load_checkpoint_safetensors(
            tmp_path / "device.safetensors",
            device="cpu",
            model_keys=["model"],
        )

        for tensor in loaded["model"].values():
            assert tensor.device == torch.device("cpu")

    def test_metadata_preserved(self, tmp_path: Path):
        """String metadata is stored and retrievable."""
        from core.utils.checkpoint import save_checkpoint_safetensors
        from safetensors import safe_open

        model = nn.Linear(2, 2)
        metadata = {"epoch": "42", "loss": "0.001", "version": "2.0"}

        save_checkpoint_safetensors(
            tmp_path / "meta.safetensors",
            {"model": model.state_dict()},
            metadata=metadata,
        )

        with safe_open(str(tmp_path / "meta.safetensors"), framework="pt") as f:
            stored_meta = f.metadata()
            assert stored_meta["epoch"] == "42"
            assert stored_meta["loss"] == "0.001"


class TestTrainingStateRoundTrip:
    """Test training state (optimizer, scheduler) save/load."""

    def test_optimizer_state(self, tmp_path: Path):
        """Optimizer state round-trips correctly."""
        from core.utils.checkpoint import (
            save_training_state,
            load_training_state,
        )

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do a fake step to create optimizer state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        save_training_state(
            tmp_path / "train.training.pt",
            optimizer=optimizer,
            epoch=10,
            best_val_loss=0.05,
        )

        loaded = load_training_state(
            tmp_path / "train.training.pt",
            trusted_source=True,
        )

        assert loaded["epoch"] == 10
        assert loaded["best_val_loss"] == 0.05
        assert "optimizer" in loaded

        # Verify optimizer can be loaded
        new_optimizer = torch.optim.Adam(model.parameters())
        new_optimizer.load_state_dict(loaded["optimizer"])

    def test_scheduler_state(self, tmp_path: Path):
        """Scheduler state round-trips correctly."""
        from core.utils.checkpoint import (
            save_training_state,
            load_training_state,
        )

        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Advance scheduler
        for _ in range(15):
            scheduler.step()

        save_training_state(
            tmp_path / "sched.training.pt",
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=15,
        )

        loaded = load_training_state(
            tmp_path / "sched.training.pt",
            trusted_source=True,
        )

        assert "scheduler" in loaded

        # Verify scheduler can be loaded
        new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        new_scheduler.load_state_dict(loaded["scheduler"])
        assert new_scheduler.last_epoch == 15

    def test_requires_trust(self, tmp_path: Path):
        """Training state load requires trusted_source=True."""
        from core.utils.checkpoint import (
            save_training_state,
            load_training_state,
        )

        model = nn.Linear(5, 3)
        optimizer = torch.optim.Adam(model.parameters())

        save_training_state(
            tmp_path / "untrusted.training.pt",
            optimizer=optimizer,
            epoch=1,
        )

        with pytest.raises(ValueError, match="trusted_source"):
            load_training_state(
                tmp_path / "untrusted.training.pt",
                trusted_source=False,
            )


class TestMigration:
    """Test legacy .pt to SafeTensors migration."""

    def test_migrate_full_checkpoint(self, tmp_path: Path):
        """Full legacy checkpoint migrates to two files."""
        from core.utils.checkpoint import migrate_checkpoint_to_safetensors

        # Create legacy checkpoint
        encoder = nn.Linear(128, 64)
        transition = nn.Linear(64, 64)
        optimizer = torch.optim.Adam(encoder.parameters())

        legacy_checkpoint = {
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": {"last_epoch": 5},
            "epoch": 10,
            "best_val_loss": 0.02,
        }

        legacy_path = tmp_path / "legacy.pt"
        torch.save(legacy_checkpoint, legacy_path)

        # Migrate
        st_path, train_path = migrate_checkpoint_to_safetensors(
            legacy_path,
            trusted_source=True,
            model_keys=["encoder", "transition"],
        )

        assert st_path.exists()
        assert train_path is not None and train_path.exists()
        assert st_path.suffix == ".safetensors"
        assert str(train_path).endswith(".training.pt")

        # Verify SafeTensors contains model weights
        from core.utils.checkpoint import load_checkpoint_safetensors
        loaded_models = load_checkpoint_safetensors(
            st_path, device="cpu", model_keys=["encoder", "transition"]
        )
        assert "encoder" in loaded_models
        assert "transition" in loaded_models

        # Verify training state file
        training_state = torch.load(train_path, weights_only=False)
        assert training_state["epoch"] == 10
        assert "optimizer" in training_state

    def test_migrate_models_only(self, tmp_path: Path):
        """Checkpoint with only models produces no training state file."""
        from core.utils.checkpoint import migrate_checkpoint_to_safetensors

        encoder = nn.Linear(10, 5)
        legacy_checkpoint = {"encoder": encoder.state_dict()}

        legacy_path = tmp_path / "models_only.pt"
        torch.save(legacy_checkpoint, legacy_path)

        st_path, train_path = migrate_checkpoint_to_safetensors(
            legacy_path,
            trusted_source=True,
            model_keys=["encoder"],
        )

        assert st_path.exists()
        assert train_path is None  # No training state to save

    def test_migrate_requires_trust(self, tmp_path: Path):
        """Migration requires trusted_source=True."""
        from core.utils.checkpoint import migrate_checkpoint_to_safetensors

        model = nn.Linear(5, 3)
        torch.save({"encoder": model.state_dict()}, tmp_path / "untrust.pt")

        with pytest.raises(ValueError, match="trusted_source"):
            migrate_checkpoint_to_safetensors(
                tmp_path / "untrust.pt",
                trusted_source=False,
            )

    def test_migrate_custom_output_dir(self, tmp_path: Path):
        """Migration can output to different directory."""
        from core.utils.checkpoint import migrate_checkpoint_to_safetensors

        # Create legacy in one dir
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        model = nn.Linear(10, 5)
        torch.save({"encoder": model.state_dict()}, src_dir / "model.pt")

        # Migrate to another dir
        dst_dir = tmp_path / "dest"
        dst_dir.mkdir()

        st_path, _ = migrate_checkpoint_to_safetensors(
            src_dir / "model.pt",
            output_dir=dst_dir,
            trusted_source=True,
            model_keys=["encoder"],
        )

        assert st_path.parent == dst_dir


class TestAutoDetection:
    """Test format auto-detection in load_checkpoint_auto."""

    def test_prefers_safetensors(self, tmp_path: Path):
        """When both formats exist, prefers SafeTensors."""
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            load_checkpoint_auto,
        )

        model = nn.Linear(10, 5)

        # Create both formats with different values
        torch.save(
            {"encoder": {"weight": torch.zeros(5, 10), "bias": torch.zeros(5)}},
            tmp_path / "model.pt",
        )
        save_checkpoint_safetensors(
            tmp_path / "model.safetensors",
            {"encoder": {"weight": torch.ones(5, 10), "bias": torch.ones(5)}},
        )

        loaded = load_checkpoint_auto(
            tmp_path / "model",
            device="cpu",
            trusted_source=True,
            model_keys=["encoder"],
        )

        # Should load ones (SafeTensors) not zeros (legacy)
        assert torch.all(loaded["encoder"]["weight"] == 1.0)

    def test_loads_training_state_when_requested(self, tmp_path: Path):
        """Include training state only when requested."""
        from core.utils.checkpoint import (
            save_checkpoint_safetensors,
            save_training_state,
            load_checkpoint_auto,
        )

        model = nn.Linear(5, 3)
        optimizer = torch.optim.Adam(model.parameters())

        save_checkpoint_safetensors(
            tmp_path / "model.safetensors",
            {"encoder": model.state_dict()},
        )
        save_training_state(
            tmp_path / "model.training.pt",
            optimizer=optimizer,
            epoch=5,
        )

        # Without training state
        loaded = load_checkpoint_auto(
            tmp_path / "model",
            device="cpu",
            include_training_state=False,
            trusted_source=True,
            model_keys=["encoder"],
        )
        assert "encoder" in loaded
        assert "optimizer" not in loaded

        # With training state
        loaded = load_checkpoint_auto(
            tmp_path / "model",
            device="cpu",
            include_training_state=True,
            trusted_source=True,
            model_keys=["encoder"],
        )
        assert "encoder" in loaded
        assert "optimizer" in loaded
        assert loaded["epoch"] == 5


class TestBackwardCompatibility:
    """Test that existing checkpoints continue to work."""

    def test_legacy_checkpoint_still_loads(self, tmp_path: Path):
        """Legacy .pt checkpoints load with trusted_source=True."""
        from core.utils.checkpoint import load_checkpoint_auto

        # Create AETHER-style legacy checkpoint
        encoder = nn.Linear(128, 64)
        transition = nn.Linear(64, 64)
        energy = nn.Linear(64, 1)
        predictor = nn.Linear(64, 10)
        latent_var = nn.Linear(64, 8)
        optimizer = torch.optim.AdamW(encoder.parameters())

        legacy = {
            "encoder": encoder.state_dict(),
            "transition": transition.state_dict(),
            "energy": energy.state_dict(),
            "predictor": predictor.state_dict(),
            "latent_var": latent_var.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": {"last_epoch": 0},
            "epoch": 50,
            "best_val_loss": 0.01,
        }

        torch.save(legacy, tmp_path / "best.pt")

        # Should load with deprecation warning but work
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            loaded = load_checkpoint_auto(
                tmp_path / "best.pt",
                device="cpu",
                include_training_state=True,
                trusted_source=True,
            )

        # All keys should be present
        assert "encoder" in loaded
        assert "transition" in loaded
        assert "energy" in loaded
        assert "predictor" in loaded
        assert "latent_var" in loaded
        assert "optimizer" in loaded
        assert loaded["epoch"] == 50

    def test_trainer_save_load_cycle(self, tmp_path: Path):
        """AetherTrainer save/load cycle works with new format."""
        # This is an integration test - skip if training module not available
        pytest.importorskip("core.training.train")

        from core.encoder.event_encoder import EventEncoder
        from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
        from core.world_model.energy import EnergyScorer
        from core.world_model.hierarchical import HierarchicalPredictor
        from core.world_model.latent import LatentVariable
        from core.world_model.transition import TransitionModel
        from core.training.train import AetherTrainer

        # Build minimal trainer
        act_vocab = ActivityVocabulary(embed_dim=64)
        for t in ["A", "B", "C"]:
            act_vocab.add_token(t)
        res_vocab = ResourceVocabulary(embed_dim=32)
        res_vocab.add_token("R")

        trainer = AetherTrainer(
            encoder=EventEncoder(act_vocab, res_vocab, latent_dim=32),
            transition=TransitionModel(latent_dim=32),
            energy=EnergyScorer(latent_dim=32),
            predictor=HierarchicalPredictor(latent_dim=32, n_activities=4, n_phases=2),
            latent_var=LatentVariable(latent_dim=32),
            activity_vocab=act_vocab,
            device="cpu",
            checkpoint_dir=tmp_path,
        )

        # Save with new format
        saved_path = trainer.save_checkpoint("test")

        # Should create both files
        assert (tmp_path / "test.safetensors").exists()
        assert (tmp_path / "test.training.pt").exists()

        # Load back
        trainer.load_checkpoint(tmp_path / "test")

        # Verify models work
        import torch
        with torch.no_grad():
            z = torch.randn(1, 32)
            preds = trainer.predictor(z)
            assert "activity" in preds
