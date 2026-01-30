"""Full Event Encoder pipeline for AETHER.

Transforms raw discrete business events into 128-dimensional latent
representations through a three-stage pipeline:

    Raw Event → StructuralEncoder → TemporalEncoder → ContextEncoder → z_t ∈ R^128

The encoder is JEPA-adapted: we predict in latent space, not in raw event space.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from .time2vec import Time2Vec
from .vocabulary import ActivityVocabulary, ResourceVocabulary


class StructuralEncoder(nn.Module):
    """Encodes categorical event attributes into a fixed-dimension vector.

    Combines activity embedding + resource embedding + attribute projection
    into a single structural representation per event.

    Args:
        activity_vocab: Vocabulary for activity strings.
        resource_vocab: Vocabulary for resource strings.
        attribute_dim: Dimension of the projected attribute features.
        output_dim: Dimension of the combined structural encoding.
        n_attribute_features: Number of raw numerical attribute features.
    """

    def __init__(
        self,
        activity_vocab: ActivityVocabulary,
        resource_vocab: ResourceVocabulary,
        attribute_dim: int = 32,
        output_dim: int = 128,
        n_attribute_features: int = 8,
    ) -> None:
        super().__init__()
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab

        self.activity_embedding = activity_vocab.embedding
        self.resource_embedding = resource_vocab.embedding

        # Project raw numerical attributes
        self.attribute_proj = nn.Linear(n_attribute_features, attribute_dim)

        # Combine all components
        combined_dim = (
            activity_vocab.embedding_dim
            + resource_vocab.embedding_dim
            + attribute_dim
        )
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        activity_ids: torch.Tensor,
        resource_ids: torch.Tensor,
        attributes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode structural event features.

        Args:
            activity_ids: (batch, seq_len) integer activity indices.
            resource_ids: (batch, seq_len) integer resource indices.
            attributes: (batch, seq_len, n_attribute_features) numerical attributes.

        Returns:
            Structural encodings of shape (batch, seq_len, output_dim).
        """
        act_emb = self.activity_embedding(activity_ids)
        res_emb = self.resource_embedding(resource_ids)
        attr_proj = self.attribute_proj(attributes)

        combined = torch.cat([act_emb, res_emb, attr_proj], dim=-1)
        return self.projection(combined)


class TemporalEncoder(nn.Module):
    """Applies Time2Vec encoding and fuses with structural representations.

    Takes structural event encodings and inter-event time deltas,
    producing time-aware event representations.

    Args:
        input_dim: Dimension of structural encodings.
        time_embed_dim: Dimension of Time2Vec output.
        output_dim: Dimension of fused output.
    """

    def __init__(
        self,
        input_dim: int = 128,
        time_embed_dim: int = 32,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time2vec = Time2Vec(embed_dim=time_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        structural: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse structural encodings with temporal embeddings.

        Args:
            structural: (batch, seq_len, input_dim) structural event encodings.
            time_deltas: (batch, seq_len) inter-event time deltas in hours.

        Returns:
            Time-aware encodings of shape (batch, seq_len, output_dim).
        """
        time_emb = self.time2vec(time_deltas)  # (batch, seq_len, time_embed_dim)
        fused = torch.cat([structural, time_emb], dim=-1)
        return self.fusion(fused)


class ContextEncoder(nn.Module):
    """Causal self-attention over the event sequence.

    Only attends to past events (causal masking) to produce
    context-aware latent states. Uses standard multi-head attention
    with pre-LayerNorm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Learnable positional encoding (supplementary to Time2Vec)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal self-attention over event sequence.

        Args:
            x: (batch, seq_len, d_model) time-aware event encodings.
            padding_mask: (batch, seq_len) boolean mask where True = padded.

        Returns:
            Context-aware latent states of shape (batch, seq_len, d_model).
        """
        seq_len = x.shape[1]

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Causal mask: prevent attending to future events
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device, dtype=x.dtype
        )

        return self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )


class EventEncoder(nn.Module):
    """Full event encoding pipeline: Raw Events → z_t in R^128.

    Chains StructuralEncoder → TemporalEncoder → ContextEncoder to produce
    JEPA-compatible latent representations of business event sequences.

    Args:
        activity_vocab: Vocabulary for activity strings.
        resource_vocab: Vocabulary for resource strings.
        latent_dim: Dimension of final latent representation (default: 128).
        n_attribute_features: Number of numerical attribute features per event.
        time_embed_dim: Dimension of Time2Vec encoding.
        n_heads: Number of attention heads in ContextEncoder.
        n_layers: Number of transformer layers in ContextEncoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        activity_vocab: ActivityVocabulary,
        resource_vocab: ResourceVocabulary,
        latent_dim: int = 128,
        n_attribute_features: int = 8,
        time_embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.structural = StructuralEncoder(
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            attribute_dim=32,
            output_dim=latent_dim,
            n_attribute_features=n_attribute_features,
        )
        self.temporal = TemporalEncoder(
            input_dim=latent_dim,
            time_embed_dim=time_embed_dim,
            output_dim=latent_dim,
        )
        self.context = ContextEncoder(
            d_model=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(
        self,
        activity_ids: torch.Tensor,
        resource_ids: torch.Tensor,
        attributes: torch.Tensor,
        time_deltas: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of event sequences into latent representations.

        Args:
            activity_ids: (batch, seq_len) integer activity indices.
            resource_ids: (batch, seq_len) integer resource indices.
            attributes: (batch, seq_len, n_attr) numerical attribute features.
            time_deltas: (batch, seq_len) inter-event time deltas in hours.
            padding_mask: (batch, seq_len) boolean mask, True = padded position.

        Returns:
            Latent states z_t of shape (batch, seq_len, latent_dim).
        """
        structural = self.structural(activity_ids, resource_ids, attributes)
        temporal = self.temporal(structural, time_deltas)
        latent = self.context(temporal, padding_mask)
        return latent
