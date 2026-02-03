"""
Train AETHER model on BPI Challenge 2017 (Loan Application Process).

This is a REAL prediction task - simple rules achieve only 49.6% accuracy.
The model must learn from event sequences to predict loan acceptance.

Dataset:
- 31,509 cases, 1.2M events
- 26 activities, 149 resources
- 72% acceptance rate
- Mean 38 events per case

Source: BPI Challenge 2017 via Hugging Face
"""

import sys
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add AETHER to path
AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
sys.path.insert(0, str(AETHER_ROOT))

from core.encoder.event_encoder import EventEncoder
from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.world_model.energy import EnergyScorer
from core.world_model.hierarchical import HierarchicalPredictor
from core.world_model.latent import LatentVariable
from core.world_model.transition import TransitionModel
from core.training.data_loader import EventSequenceDataset, collate_fn
from core.critic.calibration import compute_ece, compute_mce, compute_brier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Config ---
DATA_DIR = AETHER_ROOT / "data" / "external" / "bpi2017"
MODEL_DIR = DATA_DIR / "models"
LATENT_DIM = 128
N_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def smooth_labels(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """Apply label smoothing to binary targets."""
    return targets * (1 - 2 * smoothing) + smoothing


def train_step(
    encoder: nn.Module,
    predictor: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    pos_weight: float,
) -> dict[str, float]:
    """Single training step."""
    encoder.train()
    predictor.train()

    optimizer.zero_grad()

    # Encode sequences
    z = encoder(
        activity_ids=batch["activity_ids"],
        resource_ids=batch["resource_ids"],
        attributes=batch["attributes"],
        time_deltas=batch["time_deltas"],
        padding_mask=batch["padding_mask"],
    )

    batch_size, seq_len, latent_dim = z.shape
    seq_lens = batch["seq_lens"]

    loss_total = torch.tensor(0.0, device=z.device)
    loss_dict = {}

    # --- Outcome Prediction (main focus) ---
    last_indices = (seq_lens - 1).clamp(min=0)
    z_last = z[torch.arange(batch_size, device=z.device), last_indices]

    # Get logits directly
    shared = predictor.outcome_head.shared(z_last)
    ontime_logits = predictor.outcome_head.ontime_head(shared)
    rework_logits = predictor.outcome_head.rework_head(shared)

    # Apply label smoothing
    target_ontime_smooth = smooth_labels(batch["target_ontime"], LABEL_SMOOTHING)
    target_rework_smooth = smooth_labels(batch["target_rework"], LABEL_SMOOTHING)

    # BCE with logits and class weighting
    ontime_loss = F.binary_cross_entropy_with_logits(
        ontime_logits.squeeze(-1),
        target_ontime_smooth,
        pos_weight=torch.tensor(pos_weight, device=z.device),
    )
    rework_loss = F.binary_cross_entropy_with_logits(
        rework_logits.squeeze(-1),
        target_rework_smooth,
    )

    loss_total = loss_total + ontime_loss + 0.5 * rework_loss
    loss_dict["ontime_bce"] = ontime_loss.item()
    loss_dict["rework_bce"] = rework_loss.item()

    # --- Activity Prediction ---
    if seq_len > 1:
        z_flat = z[:, :-1, :].reshape(-1, latent_dim)
        act_preds = predictor.activity_head(z_flat)
        act_logits = act_preds["logits"]
        act_targets = batch["target_activities"][:, :-1].reshape(-1)

        pos_mask = torch.arange(seq_len - 1, device=z.device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)
        pos_mask_flat = pos_mask.reshape(-1)

        if pos_mask_flat.any():
            activity_loss = F.cross_entropy(
                act_logits[pos_mask_flat],
                act_targets[pos_mask_flat],
                ignore_index=0,
            )
            if not torch.isnan(activity_loss):
                loss_total = loss_total + 0.5 * activity_loss
                loss_dict["activity_ce"] = activity_loss.item()

    loss_dict["total"] = loss_total.item()

    if torch.isnan(loss_total) or torch.isinf(loss_total):
        logger.warning("NaN/Inf loss - skipping step")
        return loss_dict

    loss_total.backward()

    nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

    optimizer.step()

    return loss_dict


@torch.no_grad()
def validate(
    encoder: nn.Module,
    predictor: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate and compute metrics."""
    encoder.eval()
    predictor.eval()

    all_preds = []
    all_targets = []
    correct_acts = 0
    total_acts = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        z = encoder(
            activity_ids=batch["activity_ids"],
            resource_ids=batch["resource_ids"],
            attributes=batch["attributes"],
            time_deltas=batch["time_deltas"],
            padding_mask=batch["padding_mask"],
        )

        batch_size, seq_len, latent_dim = z.shape
        seq_lens = batch["seq_lens"]

        # Outcome predictions
        last_indices = (seq_lens - 1).clamp(min=0)
        z_last = z[torch.arange(batch_size, device=device), last_indices]
        predictions = predictor(z_last)

        all_preds.append(predictions["outcome"]["ontime_prob"].squeeze(-1).cpu())
        all_targets.append(batch["target_ontime"].cpu())

        # Activity accuracy
        if seq_len > 1:
            z_flat = z[:, :-1, :].reshape(-1, latent_dim)
            act_preds = predictor.activity_head(z_flat)
            pred_acts = act_preds["logits"].argmax(dim=-1)
            act_targets = batch["target_activities"][:, :-1].reshape(-1)

            pos_mask = torch.arange(seq_len - 1, device=device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)
            pos_mask_flat = pos_mask.reshape(-1)
            valid = pos_mask_flat & (act_targets != 0)

            correct_acts += (pred_acts[valid] == act_targets[valid]).sum().item()
            total_acts += valid.sum().item()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calibration metrics
    ece = compute_ece(all_preds, all_targets)
    mce = compute_mce(all_preds, all_targets)
    brier = compute_brier(all_preds, all_targets)

    # Classification metrics
    pred_binary = (all_preds > 0.5).float()
    tp = ((pred_binary == 1) & (all_targets == 1)).sum().item()
    tn = ((pred_binary == 0) & (all_targets == 0)).sum().item()
    fp = ((pred_binary == 1) & (all_targets == 0)).sum().item()
    fn = ((pred_binary == 0) & (all_targets == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    # Prediction distribution
    pred_low = (all_preds < 0.1).float().mean().item()
    pred_mid = ((all_preds >= 0.1) & (all_preds <= 0.9)).float().mean().item()
    pred_high = (all_preds > 0.9).float().mean().item()

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "accuracy": accuracy,
        "mcc": mcc,
        "activity_accuracy": correct_acts / max(total_acts, 1),
        "pred_low": pred_low,
        "pred_mid": pred_mid,
        "pred_high": pred_high,
        "pred_mean": all_preds.mean().item(),
        "pred_std": all_preds.std().item(),
    }


def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info("BPI Challenge 2017: Loan Application Prediction")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    with open(DATA_DIR / "vocabulary.json") as f:
        vocab_data = json.load(f)

    activity_vocab = ActivityVocabulary(embed_dim=64)
    for token in sorted(vocab_data["activity"]["token_to_idx"], key=lambda x: vocab_data["activity"]["token_to_idx"][x]):
        if token not in ("<PAD>", "<UNK>"):
            activity_vocab.add_token(token)

    resource_vocab = ResourceVocabulary(embed_dim=32)
    for token in sorted(vocab_data["resource"]["token_to_idx"], key=lambda x: vocab_data["resource"]["token_to_idx"][x]):
        if token not in ("<PAD>", "<UNK>"):
            resource_vocab.add_token(token)

    logger.info(f"Activity vocab: {activity_vocab.size} tokens")
    logger.info(f"Resource vocab: {resource_vocab.size} tokens")

    # Create datasets
    train_dataset = EventSequenceDataset(
        events_path=DATA_DIR / "train_cases.json",
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        max_seq_len=64,  # Truncate long sequences
        n_attribute_features=4,
    )
    val_dataset = EventSequenceDataset(
        events_path=DATA_DIR / "val_cases.json",
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        max_seq_len=64,
        n_attribute_features=4,
    )

    logger.info(f"Train cases: {len(train_dataset)}")
    logger.info(f"Val cases: {len(val_dataset)}")

    # Compute class weights
    train_targets = [c.get("outcome", {}).get("onTime", False) for c in train_dataset.cases]
    n_pos = sum(train_targets)
    n_neg = len(train_targets) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"Class distribution: pos={n_pos} ({100*n_pos/len(train_targets):.1f}%), neg={n_neg}")
    logger.info(f"pos_weight: {pos_weight:.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize model
    encoder = EventEncoder(
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        latent_dim=LATENT_DIM,
        n_attribute_features=4,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    predictor = HierarchicalPredictor(
        latent_dim=LATENT_DIM,
        n_activities=activity_vocab.size,
        n_phases=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (N_EPOCHS - WARMUP_EPOCHS)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(f"Training for {N_EPOCHS} epochs")
    logger.info("=" * 70)

    best_mcc = -1.0
    patience = 10
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        # Training
        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            losses = train_step(encoder, predictor, batch, optimizer, pos_weight)
            for k, v in losses.items():
                if not math.isnan(v):
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        scheduler.step()

        # Validation
        val_metrics = validate(encoder, predictor, val_loader, DEVICE)

        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1:3d} | "
            f"loss={avg_losses.get('total', 0):.4f} | "
            f"lr={lr_current:.2e}"
        )
        logger.info(
            f"        | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"val_ece={val_metrics['ece']:.4f} | "
            f"act_acc={val_metrics['activity_accuracy']:.4f}"
        )

        # Save best model
        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            patience_counter = 0

            torch.save({
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mcc": best_mcc,
                "val_metrics": val_metrics,
            }, MODEL_DIR / "best.pt")

            logger.info(f"  → New best model saved (MCC={best_mcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best MCC: {best_mcc:.4f}")
    logger.info(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
