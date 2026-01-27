"""Vocabulary management for categorical event attributes.

Provides learned embeddings for activities and resources through
vocabulary-to-index mapping with reserved unknown token handling.
"""

from __future__ import annotations

from typing import List, Dict, Union

import torch.nn as nn

EventList = List[Dict[str, Union[str, int, float, bool]]]


class Vocabulary:
    """Base vocabulary mapping strings to indices with nn.Embedding.

    Index 0 is reserved for unknown/OOV tokens. Vocabulary is built
    from observed event data and remains fixed after construction.

    Args:
        embed_dim: Dimensionality of the learned embedding vectors.
        name: Human-readable name for this vocabulary (for logging).
    """

    UNK_INDEX: int = 0
    UNK_TOKEN: str = "<UNK>"

    def __init__(self, embed_dim: int = 64, name: str = "vocabulary") -> None:
        self._name = name
        self._embed_dim = embed_dim
        self._token_to_idx: dict[str, int] = {self.UNK_TOKEN: self.UNK_INDEX}
        self._idx_to_token: dict[int, str] = {self.UNK_INDEX: self.UNK_TOKEN}
        self._embedding: nn.Embedding | None = None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embedding vectors."""
        return self._embed_dim

    @property
    def size(self) -> int:
        """Number of tokens including UNK."""
        return len(self._token_to_idx)

    @property
    def embedding(self) -> nn.Embedding:
        """The nn.Embedding layer. Built lazily on first access."""
        if self._embedding is None:
            self._embedding = nn.Embedding(
                num_embeddings=self.size,
                embedding_dim=self._embed_dim,
                padding_idx=self.UNK_INDEX,
            )
        return self._embedding

    def encode(self, value: str) -> int:
        """Map a string token to its integer index.

        Returns UNK_INDEX (0) for unknown tokens.
        """
        return self._token_to_idx.get(value, self.UNK_INDEX)

    def decode(self, index: int) -> str:
        """Map an integer index back to its string token."""
        return self._idx_to_token.get(index, self.UNK_TOKEN)

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary. Returns its index."""
        if token not in self._token_to_idx:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
            # Invalidate cached embedding since vocab size changed
            self._embedding = None
        return self._token_to_idx[token]

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_idx

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"{self._name}(size={self.size}, embed_dim={self._embed_dim})"


class ActivityVocabulary(Vocabulary):
    """Vocabulary for activity/event type strings.

    Typical activities: 'create_order', 'approve_credit', 'ship_goods', etc.

    Args:
        embed_dim: Embedding dimension (positional). Also accepts
            ``embedding_dim`` as keyword for compatibility.
    """

    def __init__(self, embed_dim: int = 64, **kwargs: int) -> None:
        dim = kwargs.get("embedding_dim", embed_dim)
        super().__init__(embed_dim=dim, name="ActivityVocabulary")

    def build_from_events(self, events: list) -> None:
        """Build vocabulary from event data.

        Accepts either:
            - A list of event dicts (each with an 'activity' key)
            - A list of activity strings directly
        """
        activities: set[str] = set()
        for event in events:
            if isinstance(event, str):
                activities.add(event)
            elif isinstance(event, dict):
                activity = event.get("activity")
                if isinstance(activity, str):
                    activities.add(activity)

        for activity in sorted(activities):
            self.add_token(activity)


class ResourceVocabulary(Vocabulary):
    """Vocabulary for resource/actor strings.

    Typical resources: 'user_001', 'system_auto', 'manager_finance', etc.

    Args:
        embed_dim: Embedding dimension (positional). Also accepts
            ``embedding_dim`` as keyword for compatibility.
    """

    def __init__(self, embed_dim: int = 32, **kwargs: int) -> None:
        dim = kwargs.get("embedding_dim", embed_dim)
        super().__init__(embed_dim=dim, name="ResourceVocabulary")

    def build_from_events(self, events: list) -> None:
        """Build vocabulary from event data.

        Accepts either:
            - A list of event dicts (each with a 'resource' key)
            - A list of resource strings directly
        """
        resources: set[str] = set()
        for event in events:
            if isinstance(event, str):
                resources.add(event)
            elif isinstance(event, dict):
                resource = event.get("resource")
                if isinstance(resource, str):
                    resources.add(resource)

        for resource in sorted(resources):
            self.add_token(resource)
