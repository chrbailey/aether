"""Secure checkpoint loading utilities.

This module provides secure alternatives to torch.load() that mitigate
arbitrary code execution risks from pickle deserialization (CWE-502).

Security notes:
- SafeTensors format is preferred for model weights (no code execution possible)
- Training state (optimizer, scheduler) still requires pickle but is isolated
- weights_only=True prevents arbitrary code execution during legacy pickle load
- Checkpoints should only be loaded from trusted sources

Checkpoint format (v2 - SafeTensors):
- {name}.safetensors: Model state_dicts (secure, tensor-only)
- {name}.training.pt: Optimizer/scheduler state (pickle, trusted only)
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)

# Keys for model state dicts (must match what trainer saves)
MODEL_STATE_KEYS = ["encoder", "transition", "energy", "predictor", "latent_var"]


def load_checkpoint(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Securely load a checkpoint file.

    Uses weights_only=True to prevent arbitrary code execution during
    deserialization. This is the recommended method for loading checkpoints.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to (e.g., "cpu", "cuda", "mps").

    Returns:
        Dictionary containing checkpoint data (state_dicts, metadata).

    Raises:
        RuntimeError: If checkpoint contains non-tensor data that can't be
            loaded with weights_only=True. Use load_checkpoint_unsafe() for
            legacy checkpoints, but only from trusted sources.

    Example:
        >>> checkpoint = load_checkpoint("model.pt", map_location="cpu")
        >>> model.load_state_dict(checkpoint["encoder"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.debug(f"Loading checkpoint securely: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return checkpoint
    except Exception as e:
        if "weights_only" in str(e).lower():
            raise RuntimeError(
                f"Checkpoint {path} contains non-tensor data. "
                "If this is a trusted checkpoint, use load_checkpoint_unsafe(). "
                f"Original error: {e}"
            ) from e
        raise


def load_checkpoint_unsafe(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    trusted_source: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint file without security restrictions.

    WARNING: This function can execute arbitrary code if the checkpoint
    file is malicious. Only use with checkpoints from trusted sources.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to.
        trusted_source: Must be True to acknowledge the security risk.
            This is a safety gate to prevent accidental misuse.

    Returns:
        Dictionary containing checkpoint data.

    Raises:
        ValueError: If trusted_source is not explicitly set to True.
        FileNotFoundError: If checkpoint file doesn't exist.

    Example:
        >>> # Only for legacy checkpoints from TRUSTED sources
        >>> checkpoint = load_checkpoint_unsafe(
        ...     "legacy_model.pt",
        ...     map_location="cpu",
        ...     trusted_source=True
        ... )
    """
    if not trusted_source:
        raise ValueError(
            "load_checkpoint_unsafe() requires trusted_source=True. "
            "This function can execute arbitrary code - only use with "
            "checkpoints from trusted sources. Prefer load_checkpoint() "
            "for new checkpoints."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.warning(
        f"Loading checkpoint with weights_only=False: {path}. "
        "Ensure this file is from a trusted source."
    )
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# SafeTensors format (secure, tensor-only)
# ---------------------------------------------------------------------------


def _flatten_state_dict(state_dict: Dict[str, Any], prefix: str = "") -> Dict[str, torch.Tensor]:
    """Flatten a nested state dict for SafeTensors compatibility.

    SafeTensors doesn't support nested dicts, so we flatten keys like:
    {"encoder": {"layer.weight": tensor}} -> {"encoder.layer.weight": tensor}

    Args:
        state_dict: Nested state dict from model.state_dict().
        prefix: Key prefix for recursion.

    Returns:
        Flat dict with dot-separated keys.
    """
    flat = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_state_dict(value, full_key))
        elif isinstance(value, torch.Tensor):
            flat[full_key] = value
        else:
            # Skip non-tensor values (shouldn't happen in state_dicts)
            logger.debug(f"Skipping non-tensor value at {full_key}: {type(value)}")
    return flat


def _unflatten_state_dict(flat: Dict[str, torch.Tensor], keys: list[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Unflatten a SafeTensors dict back to nested state dicts.

    Args:
        flat: Flat dict from SafeTensors with dot-separated keys.
        keys: Top-level model keys (e.g., ["encoder", "transition"]).

    Returns:
        Nested dict: {"encoder": {...}, "transition": {...}, ...}
    """
    result: Dict[str, Dict[str, torch.Tensor]] = {k: {} for k in keys}

    for full_key, tensor in flat.items():
        # Split on first dot to get model name
        parts = full_key.split(".", 1)
        if len(parts) == 2 and parts[0] in keys:
            model_key, param_key = parts
            result[model_key][param_key] = tensor
        else:
            # Fallback: put in first matching prefix or skip
            matched = False
            for k in keys:
                if full_key.startswith(k + "."):
                    param_key = full_key[len(k) + 1:]
                    result[k][param_key] = tensor
                    matched = True
                    break
            if not matched:
                logger.warning(f"Unknown key in checkpoint: {full_key}")

    return result


def save_checkpoint_safetensors(
    path: Union[str, Path],
    state_dicts: Dict[str, Dict[str, torch.Tensor]],
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Save model state dicts to SafeTensors format (secure, no pickle).

    Args:
        path: Output path (should end in .safetensors).
        state_dicts: Dict of model name -> state_dict.
            E.g., {"encoder": encoder.state_dict(), "transition": ...}
        metadata: Optional string metadata (SafeTensors only supports str values).

    Returns:
        Path to saved file.

    Example:
        >>> save_checkpoint_safetensors(
        ...     "model.safetensors",
        ...     {
        ...         "encoder": encoder.state_dict(),
        ...         "transition": transition.state_dict(),
        ...     },
        ...     metadata={"epoch": "10", "version": "1.0"},
        ... )
    """
    path = Path(path)
    if not path.suffix == ".safetensors":
        path = path.with_suffix(".safetensors")

    # Flatten all state dicts
    flat = {}
    for model_name, state_dict in state_dicts.items():
        flat.update(_flatten_state_dict(state_dict, model_name))

    # SafeTensors metadata must be Dict[str, str]
    meta = metadata or {}

    safetensors_save(flat, str(path), metadata=meta)
    logger.info(f"SafeTensors checkpoint saved: {path}")
    return path


def load_checkpoint_safetensors(
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    model_keys: Optional[list[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load model state dicts from SafeTensors format (secure).

    This is the preferred method for loading model weights as SafeTensors
    cannot execute arbitrary code - only tensor data is stored.

    Args:
        path: Path to .safetensors file.
        device: Device to load tensors to (e.g., "cpu", "cuda", "mps").
        model_keys: List of model names to unflatten into.
            Defaults to MODEL_STATE_KEYS.

    Returns:
        Dict of model name -> state_dict.

    Raises:
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> checkpoint = load_checkpoint_safetensors("model.safetensors", device="cpu")
        >>> encoder.load_state_dict(checkpoint["encoder"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SafeTensors checkpoint not found: {path}")

    keys = model_keys or MODEL_STATE_KEYS
    device_str = str(device) if device else "cpu"

    flat = safetensors_load(str(path), device=device_str)
    logger.debug(f"Loaded SafeTensors checkpoint: {path} ({len(flat)} tensors)")

    return _unflatten_state_dict(flat, keys)


# ---------------------------------------------------------------------------
# Training state (pickle - requires trust)
# ---------------------------------------------------------------------------


def save_training_state(
    path: Union[str, Path],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
    **extra_state: Any,
) -> Path:
    """Save training state to pickle format.

    Training state (optimizer, scheduler) contains non-tensor Python objects
    that cannot be stored in SafeTensors. This file should only be loaded
    from trusted sources.

    Args:
        path: Output path (will be suffixed with .training.pt).
        optimizer: Optimizer with state to save.
        scheduler: Optional LR scheduler.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        **extra_state: Additional state to save.

    Returns:
        Path to saved file.

    Example:
        >>> save_training_state(
        ...     "best.training.pt",
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=10,
        ... )
    """
    path = Path(path)
    if not str(path).endswith(".training.pt"):
        path = path.with_suffix(".training.pt")

    state = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        **extra_state,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, path)
    logger.info(f"Training state saved: {path}")
    return path


def load_training_state(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    trusted_source: bool = False,
) -> Dict[str, Any]:
    """Load training state from pickle format.

    WARNING: This uses pickle deserialization. Only load from trusted sources.

    Args:
        path: Path to .training.pt file.
        map_location: Device to map tensors to.
        trusted_source: Must be True to acknowledge security risk.

    Returns:
        Dict with optimizer, scheduler state, epoch, etc.

    Raises:
        ValueError: If trusted_source is not True.
        FileNotFoundError: If file doesn't exist.
    """
    if not trusted_source:
        raise ValueError(
            "load_training_state() requires trusted_source=True. "
            "Training state files use pickle and can execute arbitrary code. "
            "Only load from trusted sources."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training state not found: {path}")

    logger.debug(f"Loading training state: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Auto-detection and backward compatibility
# ---------------------------------------------------------------------------


def load_checkpoint_auto(
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    include_training_state: bool = False,
    trusted_source: bool = False,
    model_keys: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Auto-detect checkpoint format and load appropriately.

    Supports both legacy .pt files and new SafeTensors format.
    Provides backward compatibility during migration.

    Format detection:
    - .safetensors -> load_checkpoint_safetensors()
    - .pt/.pth with companion .training.pt -> SafeTensors + training state
    - .pt/.pth alone -> legacy format (requires trusted_source=True)

    Args:
        path: Path to checkpoint (with or without extension).
        device: Device to load tensors to.
        include_training_state: Whether to load optimizer/scheduler state.
            Only applies if training state file exists.
        trusted_source: Required for loading pickle-based files.
        model_keys: Model names for SafeTensors unflattening.

    Returns:
        Dict with model state dicts and optionally training state.
        Format: {"encoder": {...}, "transition": {...}, ..., "epoch": N, ...}

    Example:
        >>> # Inference (secure, no training state)
        >>> checkpoint = load_checkpoint_auto("best", device="cpu")
        >>> encoder.load_state_dict(checkpoint["encoder"])
        >>>
        >>> # Training (resume)
        >>> checkpoint = load_checkpoint_auto(
        ...     "best",
        ...     include_training_state=True,
        ...     trusted_source=True,
        ... )
        >>> optimizer.load_state_dict(checkpoint["optimizer"])
    """
    path = Path(path)
    keys = model_keys or MODEL_STATE_KEYS

    # Try SafeTensors first (preferred)
    safetensors_path = path.with_suffix(".safetensors")
    if path.suffix == ".safetensors":
        safetensors_path = path

    training_path = path.with_suffix(".training.pt")
    if str(path).endswith(".training.pt"):
        # User passed training state path - derive safetensors path
        base = str(path).replace(".training.pt", "")
        safetensors_path = Path(base + ".safetensors")
        training_path = path

    if safetensors_path.exists():
        # New format: SafeTensors for models
        result = load_checkpoint_safetensors(safetensors_path, device=device, model_keys=keys)

        # Optionally load training state
        if include_training_state and training_path.exists():
            if not trusted_source:
                raise ValueError(
                    "Loading training state requires trusted_source=True. "
                    "Training state uses pickle which can execute arbitrary code."
                )
            training_state = load_training_state(training_path, map_location=device, trusted_source=True)
            result.update(training_state)
        elif include_training_state and not training_path.exists():
            logger.warning(f"Training state requested but not found: {training_path}")

        return result

    # Fall back to legacy .pt format
    legacy_path = path.with_suffix(".pt") if path.suffix not in {".pt", ".pth"} else path

    if legacy_path.exists():
        warnings.warn(
            f"Loading legacy pickle checkpoint: {legacy_path}. "
            "Consider migrating to SafeTensors format for security.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not trusted_source:
            raise ValueError(
                f"Legacy checkpoint {legacy_path} uses pickle format. "
                "Set trusted_source=True to load, or migrate to SafeTensors."
            )

        checkpoint = load_checkpoint_unsafe(legacy_path, map_location=device, trusted_source=True)

        # If not including training state, filter out non-model keys
        if not include_training_state:
            return {k: v for k, v in checkpoint.items() if k in keys}
        return checkpoint

    raise FileNotFoundError(
        f"No checkpoint found at {path}. "
        f"Tried: {safetensors_path}, {legacy_path}"
    )


# ---------------------------------------------------------------------------
# Migration utilities
# ---------------------------------------------------------------------------


def migrate_checkpoint_to_safetensors(
    legacy_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    trusted_source: bool = False,
    model_keys: Optional[list[str]] = None,
) -> Tuple[Path, Optional[Path]]:
    """Migrate a legacy .pt checkpoint to SafeTensors format.

    Splits a legacy checkpoint into:
    - {name}.safetensors: Model state dicts (secure)
    - {name}.training.pt: Training state (pickle, if present)

    Args:
        legacy_path: Path to legacy .pt checkpoint.
        output_dir: Directory for output files. Defaults to same as input.
        trusted_source: Must be True to load the legacy pickle file.
        model_keys: Model names to extract. Defaults to MODEL_STATE_KEYS.

    Returns:
        Tuple of (safetensors_path, training_path or None).

    Raises:
        ValueError: If trusted_source is not True.
        FileNotFoundError: If legacy checkpoint doesn't exist.

    Example:
        >>> st_path, train_path = migrate_checkpoint_to_safetensors(
        ...     "data/models/best.pt",
        ...     trusted_source=True,
        ... )
        >>> print(f"Migrated to: {st_path}, {train_path}")
    """
    legacy_path = Path(legacy_path)
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy checkpoint not found: {legacy_path}")

    if not trusted_source:
        raise ValueError(
            "migrate_checkpoint_to_safetensors() requires trusted_source=True. "
            "The legacy pickle file must be from a trusted source."
        )

    keys = model_keys or MODEL_STATE_KEYS
    out_dir = Path(output_dir) if output_dir else legacy_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = legacy_path.stem
    safetensors_path = out_dir / f"{base_name}.safetensors"
    training_path = out_dir / f"{base_name}.training.pt"

    # Load legacy checkpoint
    checkpoint = load_checkpoint_unsafe(legacy_path, trusted_source=True)

    # Extract model state dicts
    model_states = {}
    for key in keys:
        if key in checkpoint:
            model_states[key] = checkpoint[key]

    if not model_states:
        raise ValueError(f"No model state dicts found in checkpoint. Expected keys: {keys}")

    # Save SafeTensors
    metadata = {}
    if "epoch" in checkpoint:
        metadata["epoch"] = str(checkpoint["epoch"])

    save_checkpoint_safetensors(safetensors_path, model_states, metadata=metadata)

    # Save training state if present
    training_keys = {"optimizer", "scheduler", "epoch", "best_val_loss"}
    training_state = {k: v for k, v in checkpoint.items() if k in training_keys}

    training_saved = None
    if training_state:
        torch.save(training_state, training_path)
        training_saved = training_path
        logger.info(f"Training state saved: {training_path}")

    logger.info(
        f"Migration complete: {legacy_path} -> {safetensors_path}"
        + (f" + {training_path}" if training_saved else "")
    )

    return safetensors_path, training_saved
