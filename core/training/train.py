"""Training loop for the AETHER world model.

Trains the full pipeline: EventEncoder + TransitionModel + EnergyScorer
+ HierarchicalPredictor using VICReg + energy contrastive losses.
Logs calibration metrics after each epoch and saves checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..critic.calibration import CalibrationTracker
from ..encoder.event_encoder import EventEncoder
from ..encoder.vocabulary import ActivityVocabulary
from ..world_model.energy import EnergyScorer
from ..world_model.hierarchical import HierarchicalPredictor
from ..world_model.latent import LatentVariable
from ..world_model.transition import TransitionModel
from .losses import EnergyContrastiveLoss, SIGRegLoss, VICRegLoss

logger = logging.getLogger(__name__)


class AetherTrainer:
    """Training loop for the AETHER world model.

    Combines all model components and trains them end-to-end with:
        - VICReg loss on latent state predictions (anti-collapse)
        - Energy contrastive loss on transition plausibility
        - Cross-entropy loss on activity prediction
        - Binary cross-entropy on outcome predictions

    Args:
        encoder: EventEncoder producing latent states.
        transition: TransitionModel predicting next latent states.
        energy: EnergyScorer evaluating transition plausibility.
        predictor: HierarchicalPredictor for activity/phase/outcome heads.
        latent_var: LatentVariable for path variant sampling.
        activity_vocab: Activity vocabulary for label count.
        device: Torch device (cpu or cuda).
        lr: Learning rate.
        checkpoint_dir: Directory for saving model checkpoints.
        vicreg_weights: Tuple of (invariance, variance, covariance) weights.
        energy_margin: Margin for energy contrastive loss.
        temperature_schedule: Tuple of (start, end, decay_epochs) for
            Gumbel-Softmax temperature annealing.
        loss_type: 'vicreg' or 'sigreg'. SIGReg (Balestriero & LeCun, 2025)
            is recommended for MPS training due to better numerical stability.
    """

    def __init__(
        self,
        encoder: EventEncoder,
        transition: TransitionModel,
        energy: EnergyScorer,
        predictor: HierarchicalPredictor,
        latent_var: LatentVariable,
        activity_vocab: ActivityVocabulary,
        device: torch.device | str = "cpu",
        lr: float = 3e-4,
        checkpoint_dir: Path | str = Path("data/models"),
        vicreg_weights: tuple[float, float, float] = (25.0, 25.0, 1.0),
        energy_margin: float = 1.0,
        temperature_schedule: tuple[float, float, int] = (1.0, 0.1, 50),
        loss_type: str = "vicreg",
    ) -> None:
        self.device = torch.device(device)

        # Models
        self.encoder = encoder.to(self.device)
        self.transition = transition.to(self.device)
        self.energy = energy.to(self.device)
        self.predictor = predictor.to(self.device)
        self.latent_var = latent_var.to(self.device)
        self.activity_vocab = activity_vocab

        # Losses — SIGReg returns VICReg-compatible dict keys
        self.loss_type = loss_type
        if loss_type == "sigreg":
            self.vicreg = SIGRegLoss(
                invariance_weight=vicreg_weights[0],
                sigreg_weight=vicreg_weights[1],
            )
        else:
            self.vicreg = VICRegLoss(
                invariance_weight=vicreg_weights[0],
                variance_weight=vicreg_weights[1],
                covariance_weight=vicreg_weights[2],
            )
        self.energy_contrastive = EnergyContrastiveLoss(margin=energy_margin)

        # Optimizer (all parameters)
        all_params = (
            list(encoder.parameters())
            + list(transition.parameters())
            + list(energy.parameters())
            + list(predictor.parameters())
            + list(latent_var.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Temperature annealing for Gumbel-Softmax
        self.temp_start, self.temp_end, self.temp_epochs = temperature_schedule

        # Calibration tracking
        self.calibration_tracker = CalibrationTracker(window_size=5000)

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._epoch = 0
        self._best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dict of average loss values for the epoch.
        """
        self.encoder.train()
        self.transition.train()
        self.energy.train()
        self.predictor.train()
        self.latent_var.train()

        total_losses: dict[str, float] = {}
        loss_counts: dict[str, int] = {}

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            losses = self._train_step(batch)

            import math
            for key, value in losses.items():
                if not math.isnan(value) and not math.isinf(value):
                    total_losses[key] = total_losses.get(key, 0.0) + value
                    loss_counts[key] = loss_counts.get(key, 0) + 1

        # Average losses (skip NaN batches per-key)
        avg_losses = {
            k: v / max(loss_counts.get(k, 1), 1)
            for k, v in total_losses.items()
        }

        # Anneal Gumbel-Softmax temperature
        if self._epoch < self.temp_epochs:
            progress = self._epoch / self.temp_epochs
            new_temp = self.temp_start + (self.temp_end - self.temp_start) * progress
            self.latent_var.anneal_temperature(new_temp)

        self.scheduler.step()
        self._epoch += 1

        logger.info(
            f"Epoch {self._epoch}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
        )

        return avg_losses

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Returns dict of scalar loss values.
        """
        self.optimizer.zero_grad()

        # Encode event sequences → latent states
        z = self.encoder(
            activity_ids=batch["activity_ids"],
            resource_ids=batch["resource_ids"],
            attributes=batch["attributes"],
            time_deltas=batch["time_deltas"],
            padding_mask=batch["padding_mask"],
        )  # (batch, seq_len, latent_dim)

        batch_size, seq_len, latent_dim = z.shape
        seq_lens = batch["seq_lens"]

        # Get valid (non-padded) positions for transition learning
        # Use pairs of consecutive positions: z_t -> z_{t+1}
        loss_total = torch.tensor(0.0, device=self.device)
        loss_dict: dict[str, float] = {}

        # --- VICReg + Transition Loss on consecutive latent pairs ---
        if seq_len > 1:
            z_t = z[:, :-1, :]  # (batch, seq_len-1, dim)
            z_next = z[:, 1:, :]  # (batch, seq_len-1, dim)

            # Flatten for transition model
            z_t_flat = z_t.reshape(-1, latent_dim)
            z_next_flat = z_next.reshape(-1, latent_dim)

            # Sample path variants from current latent
            variant_out = self.latent_var(z_t_flat)
            variant_sample = variant_out["sample"]

            # Use "standard" governance action (one-hot index 1)
            from ..world_model.transition import NUM_ACTIONS
            action = F.one_hot(
                torch.ones(z_t_flat.shape[0], dtype=torch.long, device=self.device),
                num_classes=NUM_ACTIONS,
            ).float()

            # Predict next latent
            z_pred = self.transition(z_t_flat, action, variant_sample)

            # VICReg loss
            vicreg_out = self.vicreg(z_pred, z_next_flat)
            loss_total = loss_total + vicreg_out["total"]
            loss_dict["vicreg_total"] = vicreg_out["total"].item()
            loss_dict["vicreg_invariance"] = vicreg_out["invariance"].item()
            loss_dict["vicreg_variance"] = vicreg_out["variance"].item()
            loss_dict["vicreg_covariance"] = vicreg_out["covariance"].item()

            # Energy contrastive loss
            energy_pos = self.energy(z_pred, z_next_flat)
            # Negative: shuffle z_next to create random transitions
            perm = torch.randperm(z_next_flat.shape[0], device=self.device)
            z_neg = z_next_flat[perm]
            energy_neg = self.energy(z_pred, z_neg)

            energy_out = self.energy_contrastive(energy_pos, energy_neg)
            loss_total = loss_total + energy_out["total"]
            loss_dict["energy_total"] = energy_out["total"].item()

        # --- Activity Prediction Loss (all valid positions) ---
        # Predict next activity at every non-padded position.
        # target_activities[i] = activity_ids[i+1] (shifted), with 0 at last pos.
        # Use all positions except the last valid one (which has target=0).
        if seq_len > 1:
            # Reshape z for per-position activity prediction: (batch*seq, dim)
            z_flat = z[:, :-1, :].reshape(-1, latent_dim)  # exclude last pos
            act_preds = self.predictor.activity_head(z_flat)
            act_logits = act_preds["logits"]  # (batch*(seq-1), n_activities)

            # Targets: next activity at each position
            act_targets = batch["target_activities"][:, :-1].reshape(-1)

            # Mask out padded positions
            pos_mask = torch.arange(seq_len - 1, device=self.device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)
            pos_mask_flat = pos_mask.reshape(-1)

            if pos_mask_flat.any():
                activity_loss = F.cross_entropy(
                    act_logits[pos_mask_flat],
                    act_targets[pos_mask_flat],
                    ignore_index=0,
                )
                # Guard against NaN (all targets are 0/padding)
                if torch.isnan(activity_loss):
                    activity_loss = torch.tensor(0.0, device=self.device)
            else:
                activity_loss = torch.tensor(0.0, device=self.device)
        else:
            activity_loss = torch.tensor(0.0, device=self.device)

        loss_total = loss_total + activity_loss
        loss_dict["activity_ce"] = activity_loss.item()

        # --- Outcome Prediction Losses (last valid position) ---
        last_indices = (seq_lens - 1).clamp(min=0)
        z_last = z[torch.arange(batch_size, device=self.device), last_indices]

        predictions = self.predictor(z_last)

        # Clamp probabilities to avoid log(0) in BCE (causes NaN/Inf)
        eps = 1e-6
        ontime_pred = predictions["outcome"]["ontime_prob"].squeeze(-1).clamp(eps, 1 - eps)
        rework_pred = predictions["outcome"]["rework_prob"].squeeze(-1).clamp(eps, 1 - eps)

        ontime_loss = F.binary_cross_entropy(ontime_pred, batch["target_ontime"])
        rework_loss = F.binary_cross_entropy(rework_pred, batch["target_rework"])
        loss_total = loss_total + ontime_loss + rework_loss
        loss_dict["ontime_bce"] = ontime_loss.item()
        loss_dict["rework_bce"] = rework_loss.item()

        # Remaining hours regression (smooth L1)
        # Normalize target to log-scale to prevent gradient explosion from
        # large hour values (some cases have >1000 hours duration).
        target_remaining = batch["target_remaining"].clamp(min=0.01)
        pred_remaining = predictions["outcome"]["remaining_hours"].squeeze(-1)
        remaining_loss = F.smooth_l1_loss(
            torch.log1p(pred_remaining),
            torch.log1p(target_remaining),
        )
        loss_total = loss_total + remaining_loss
        loss_dict["remaining_l1"] = remaining_loss.item()

        loss_dict["total"] = loss_total.item()

        # Guard: skip step if loss is NaN
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            self.optimizer.zero_grad()
            logger.warning("NaN/Inf loss detected — skipping step")
            return loss_dict

        # Backward pass
        loss_total.backward()
        # Clip ALL model components (not just encoder/transition)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.transition.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.energy.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.latent_var.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Track calibration (on-time probability)
        with torch.no_grad():
            self.calibration_tracker.update_batch(
                predictions["outcome"]["ontime_prob"].squeeze(-1),
                batch["target_ontime"],
            )

        return loss_dict

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation and compute metrics.

        Computes activity accuracy and calibration metrics (ECE, MCE, Brier)
        freshly on validation data — NOT reusing the training tracker.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict of average validation metrics.
        """
        self.encoder.eval()
        self.transition.eval()
        self.energy.eval()
        self.predictor.eval()
        self.latent_var.eval()

        n_batches = 0
        correct = 0
        total = 0

        # Collect validation outcome predictions for fresh calibration
        val_ontime_preds: list[torch.Tensor] = []
        val_ontime_actuals: list[torch.Tensor] = []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            z = self.encoder(
                activity_ids=batch["activity_ids"],
                resource_ids=batch["resource_ids"],
                attributes=batch["attributes"],
                time_deltas=batch["time_deltas"],
                padding_mask=batch["padding_mask"],
            )

            batch_size, val_seq_len, latent_dim = z.shape
            seq_lens = batch["seq_lens"]

            # Activity accuracy over all valid positions
            if val_seq_len > 1:
                z_flat = z[:, :-1, :].reshape(-1, latent_dim)
                act_preds = self.predictor.activity_head(z_flat)
                pred_acts = act_preds["logits"].argmax(dim=-1)

                act_targets = batch["target_activities"][:, :-1].reshape(-1)
                pos_mask = torch.arange(val_seq_len - 1, device=self.device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)
                pos_mask_flat = pos_mask.reshape(-1)
                valid = pos_mask_flat & (act_targets != 0)

                correct += (pred_acts[valid] == act_targets[valid]).sum().item()
                total += valid.sum().item()

            # Outcome predictions for calibration
            last_indices = (seq_lens - 1).clamp(min=0)
            z_last = z[torch.arange(batch_size, device=self.device), last_indices]
            predictions = self.predictor(z_last)

            val_ontime_preds.append(
                predictions["outcome"]["ontime_prob"].squeeze(-1).cpu()
            )
            val_ontime_actuals.append(batch["target_ontime"].cpu())

            n_batches += 1

        # Compute calibration metrics on ALL validation predictions
        from ..critic.calibration import compute_ece, compute_mce, compute_brier

        all_preds = torch.cat(val_ontime_preds) if val_ontime_preds else torch.tensor([])
        all_actuals = torch.cat(val_ontime_actuals) if val_ontime_actuals else torch.tensor([])

        val_metrics: dict[str, float] = {}
        val_metrics["activity_accuracy"] = correct / max(total, 1)
        val_metrics["calibration_ece"] = compute_ece(all_preds, all_actuals)
        val_metrics["calibration_mce"] = compute_mce(all_preds, all_actuals)
        val_metrics["calibration_brier"] = compute_brier(all_preds, all_actuals)

        logger.info(
            "Validation: "
            + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        )

        return val_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        n_epochs: int = 100,
    ) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            n_epochs: Number of epochs to train.

        Returns:
            List of per-epoch loss dicts.
        """
        history: list[dict[str, float]] = []

        for epoch in range(n_epochs):
            train_losses = self.train_epoch(train_loader)
            history.append(train_losses)

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics.get("calibration_ece", float("inf"))

                # Guard: don't save "best" if metrics are NaN/degenerate
                import math
                if not math.isnan(val_loss) and val_loss > 0 and val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    logger.info(f"  New best model saved (ECE={val_loss:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final checkpoint
        self.save_checkpoint("final.pt")

        # Log final calibration from validation (if available)
        if val_loader is not None:
            final_val = self.validate(val_loader)
            logger.info(
                f"Final calibration: "
                f"ECE={final_val['calibration_ece']:.4f}, "
                f"MCE={final_val['calibration_mce']:.4f}, "
                f"Brier={final_val['calibration_brier']:.4f}"
            )
        else:
            # Fall back to training tracker
            cal_metrics = self.calibration_tracker.compute_metrics()
            logger.info(
                f"Final calibration (training): "
                f"ECE={cal_metrics['ece']:.4f}, "
                f"MCE={cal_metrics['mce']:.4f}, "
                f"Brier={cal_metrics['brierScore']:.4f}"
            )

        return history

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self._epoch,
                "encoder": self.encoder.state_dict(),
                "transition": self.transition.state_dict(),
                "energy": self.energy.state_dict(),
                "predictor": self.predictor.state_dict(),
                "latent_var": self.latent_var.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_val_loss": self._best_val_loss,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.transition.load_state_dict(checkpoint["transition"])
        self.energy.load_state_dict(checkpoint["energy"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.latent_var.load_state_dict(checkpoint["latent_var"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self._epoch = checkpoint["epoch"]
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Checkpoint loaded: {path} (epoch {self._epoch})")
