"""
Train AETHER model on Fitbit NetSuite Customer Journey dataset.

Uses the same architecture as Sepsis/BPI models:
  EventEncoder → TransitionModel → EnergyScorer → HierarchicalPredictor
"""

import sys
import json
import logging
from pathlib import Path

import torch
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
from core.training.train import AetherTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Config ---
DATA_DIR = AETHER_ROOT / "data" / "external" / "fitbit"
MODEL_DIR = DATA_DIR / "models"
LATENT_DIM = 128
N_EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data dir: {DATA_DIR}")

    # Load vocabulary from parsed data
    vocab_path = DATA_DIR / "vocabulary.json"
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    # Build activity vocabulary
    activity_vocab = ActivityVocabulary(embed_dim=64)
    for token, idx in sorted(vocab_data["activity"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in ("<PAD>", "<UNK>"):
            activity_vocab.add_token(token)

    # Build resource vocabulary
    resource_vocab = ResourceVocabulary(embed_dim=32)
    for token, idx in sorted(vocab_data["resource"]["token_to_idx"].items(), key=lambda x: x[1]):
        if token not in ("<PAD>", "<UNK>"):
            resource_vocab.add_token(token)

    logger.info(f"Activity vocab: {activity_vocab.size} tokens")
    logger.info(f"Resource vocab: {resource_vocab.size} tokens")

    # Create datasets
    train_dataset = EventSequenceDataset(
        events_path=DATA_DIR / "train_cases.json",
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        max_seq_len=64,  # Shorter sequences for faster training
        n_attribute_features=4,  # Fewer attributes than BPI data
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

    # Initialize model components
    encoder = EventEncoder(
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        latent_dim=LATENT_DIM,
        n_attribute_features=4,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )
    transition = TransitionModel(latent_dim=LATENT_DIM)
    energy = EnergyScorer(latent_dim=LATENT_DIM)
    predictor = HierarchicalPredictor(
        latent_dim=LATENT_DIM,
        n_activities=activity_vocab.size,
        n_phases=4,  # Fewer phases for simpler process
    )
    latent_var = LatentVariable(latent_dim=LATENT_DIM)

    # Count parameters
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [encoder, transition, energy, predictor, latent_var]
    )
    logger.info(f"Total parameters: {total_params:,}")

    # Create trainer — use SIGReg for MPS stability
    trainer = AetherTrainer(
        encoder=encoder,
        transition=transition,
        energy=energy,
        predictor=predictor,
        latent_var=latent_var,
        activity_vocab=activity_vocab,
        device=DEVICE,
        lr=LR,
        checkpoint_dir=MODEL_DIR,
        loss_type="sigreg",
    )

    logger.info(f"Training for {N_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})")
    logger.info("=" * 60)

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
    )

    # Print final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model saved to: {MODEL_DIR}")
    logger.info(f"Best val ECE: {trainer._best_val_loss:.4f}")

    # Print last epoch losses
    if history:
        last = history[-1]
        logger.info(f"Final epoch losses: " + ", ".join(f"{k}={v:.4f}" for k, v in last.items()))


if __name__ == "__main__":
    main()
