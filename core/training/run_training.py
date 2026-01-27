"""Resume training from checkpoint with real data.

Usage:
    python3 -m core.training.run_training [--epochs 50] [--resume data/models/final.pt]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aether.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="AETHER training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default="data/events")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="vicreg",
                        choices=["vicreg", "sigreg"],
                        help="Loss type: vicreg (original) or sigreg (LeJEPA, recommended for MPS)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load vocabulary
    vocab_path = data_dir / "vocabulary.json"
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
    activity_vocab = ActivityVocabulary(embed_dim=64)
    for token in vocab_data["activity"]["token_to_idx"]:
        if token != "<UNK>":
            activity_vocab.add_token(token)

    resource_vocab = ResourceVocabulary(embed_dim=32)
    for token in vocab_data["resource"]["token_to_idx"]:
        if token != "<UNK>":
            resource_vocab.add_token(token)

    logger.info(f"Activities: {activity_vocab.size}, Resources: {resource_vocab.size}")

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    max_seq_len = metadata.get("max_seq_len", 256)

    # Create datasets from JSON files
    from .data_loader import EventSequenceDataset, collate_fn

    train_path = data_dir / "train_cases.json"
    val_path = data_dir / "val_cases.json"
    logger.info(f"Loading training data from {train_path}...")

    train_ds = EventSequenceDataset(
        train_path, activity_vocab, resource_vocab, max_seq_len=max_seq_len
    )
    val_ds = EventSequenceDataset(
        val_path, activity_vocab, resource_vocab, max_seq_len=max_seq_len
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    logger.info(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    from ..encoder.event_encoder import EventEncoder
    from ..world_model.transition import TransitionModel
    from ..world_model.energy import EnergyScorer
    from ..world_model.hierarchical import HierarchicalPredictor
    from ..world_model.latent import LatentVariable
    from .train import AetherTrainer

    encoder = EventEncoder(activity_vocab=activity_vocab, resource_vocab=resource_vocab)
    transition = TransitionModel()
    energy = EnergyScorer()
    predictor = HierarchicalPredictor(
        n_activities=activity_vocab.size,
        n_phases=6,
    )
    latent_var = LatentVariable()

    trainer = AetherTrainer(
        encoder=encoder,
        transition=transition,
        energy=energy,
        predictor=predictor,
        latent_var=latent_var,
        activity_vocab=activity_vocab,
        device=device,
        lr=args.lr,
        loss_type=args.loss,
    )

    # Resume from checkpoint if available
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting fresh")

    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in transition.parameters())
    total_params += sum(p.numel() for p in energy.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    total_params += sum(p.numel() for p in latent_var.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(train_loader, val_loader, n_epochs=args.epochs)

    # Summary
    if history:
        last = history[-1]
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Final losses: {json.dumps({k: round(v, 4) for k, v in last.items()})}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
