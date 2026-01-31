"""Secure checkpoint loading utilities.

This module provides secure alternatives to torch.load() that mitigate
arbitrary code execution risks from pickle deserialization (CWE-502).

Security notes:
- weights_only=True prevents arbitrary code execution during load
- Checkpoints should only be loaded from trusted sources
- For production, implement signature verification of checkpoint files
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


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
