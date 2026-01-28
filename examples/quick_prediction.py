#!/usr/bin/env python3
"""Quick prediction example for AETHER (Process-JEPA).

Demonstrates the core ML pipeline: load synthetic events, encode them into
128D latent states, predict the next event, and decompose uncertainty into
epistemic (reducible) vs. aleatoric (irreducible) components.

This example uses randomly initialized weights — in production, you'd load
a trained checkpoint. The point is to show the data flow and output format.

Usage:
    python examples/quick_prediction.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.encoder.event_encoder import StructuralEncoder, TemporalEncoder, ContextEncoder
from core.world_model.transition import TransitionModel
from core.world_model.energy import EnergyScorer
from core.world_model.latent import LatentVariable
from core.critic.decomposition import decompose_from_ensemble


def main() -> None:
    # Load synthetic events
    events_path = Path(__file__).parent / "sample_events.json"
    with open(events_path) as f:
        data = json.load(f)

    case = data["cases"][0]  # PO-2024-001: a clean purchase-to-pay flow
    print(f"Case: {case['caseId']}")
    print(f"Events: {len(case['events'])}")
    for e in case["events"]:
        print(f"  {e['activity']:<35} {e['timestamp']}")
    print()

    # Build vocabularies from the sample data
    all_activities: list[str] = []
    all_resources: list[str] = []
    for c in data["cases"]:
        for e in c["events"]:
            all_activities.append(e["activity"])
            all_resources.append(e["resource"])

    activity_vocab = ActivityVocabulary(sorted(set(all_activities)))
    resource_vocab = ResourceVocabulary(sorted(set(all_resources)))

    print(f"Activity vocabulary: {activity_vocab.size} tokens")
    print(f"Resource vocabulary: {resource_vocab.size} tokens")
    print()

    # Initialize the encoder pipeline (randomly initialized)
    structural = StructuralEncoder(activity_vocab, resource_vocab)
    temporal = TemporalEncoder()
    context = ContextEncoder()

    # Initialize world model components
    transition = TransitionModel()
    energy_scorer = EnergyScorer()
    latent_var = LatentVariable()

    # Encode the event sequence
    # In production, the data pipeline handles batching and feature extraction.
    # Here we manually construct tensors for illustration.
    events = case["events"]
    n_events = len(events)

    activity_indices = torch.tensor(
        [activity_vocab.token_to_index(e["activity"]) for e in events]
    ).unsqueeze(0)  # (1, seq_len)

    resource_indices = torch.tensor(
        [resource_vocab.token_to_index(e["resource"]) for e in events]
    ).unsqueeze(0)  # (1, seq_len)

    # Compute inter-event time deltas in hours
    from datetime import datetime, timezone

    timestamps = [
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        for e in events
    ]
    deltas_hours = [0.0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600.0
        deltas_hours.append(delta)
    delta_t = torch.tensor(deltas_hours).unsqueeze(0)  # (1, seq_len)

    # Numerical attributes (amount normalized)
    amounts = torch.tensor(
        [e["attributes"].get("amount", 0.0) / 100_000.0 for e in events]
    ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    attributes = torch.zeros(1, n_events, 8)
    attributes[:, :, 0:1] = amounts

    # Forward through encoder pipeline
    with torch.no_grad():
        structural_out = structural(activity_indices, resource_indices, attributes)
        temporal_out = temporal(structural_out, delta_t)
        latent_states = context(temporal_out)  # (1, seq_len, 128)

    print(f"Latent states shape: {latent_states.shape}")
    print(f"Latent state norm (last event): {latent_states[0, -1].norm():.3f}")
    print()

    # Predict next event using the transition model
    z_current = latent_states[0, -1:]  # (1, 128) — last event's state

    # Use "standard" governance action
    action_index = torch.tensor([1])  # standard mode

    # Sample a path variant
    with torch.no_grad():
        variant = latent_var.sample()  # (1, 6) Gumbel-Softmax
        z_predicted = transition(z_current, action_index, variant)  # (1, 128)

    print(f"Predicted next state norm: {z_predicted[0].norm():.3f}")
    print()

    # Ensemble uncertainty decomposition
    # Run 5 forward passes with different path variants to create an ensemble
    print("Ensemble uncertainty decomposition (5 members):")
    ensemble_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(5):
            variant_i = latent_var.sample()
            z_pred_i = transition(z_current, action_index, variant_i)
            # Use the predicted state as a proxy for activity probabilities
            # In production, the HierarchicalPredictor produces actual softmax outputs
            prob_proxy = torch.softmax(z_pred_i[0, :activity_vocab.size], dim=0)
            ensemble_predictions.append(prob_proxy)

    decomposition = decompose_from_ensemble(ensemble_predictions)

    print(f"  Total uncertainty:     {decomposition['total']:.6f}")
    print(f"  Epistemic (reducible): {decomposition['epistemic']:.6f}")
    print(f"  Aleatoric (irreducible): {decomposition['aleatoric']:.6f}")
    print(f"  Epistemic ratio:       {decomposition['epistemicRatio']:.3f}")
    print(f"  Method:                {decomposition['method']}")
    print()

    # Energy-based anomaly scoring
    # Compare predicted vs actual (using z_current as stand-in for actual next state)
    with torch.no_grad():
        energy = energy_scorer(z_predicted, z_current)

    print(f"Energy score: {energy['energy'].item():.4f}")
    print(f"Normalized (sigmoid): {energy['normalized_energy'].item():.4f}")
    print()

    if decomposition["epistemicRatio"] > 0.5:
        print(">> High epistemic ratio — governance should TIGHTEN (more data would help)")
    else:
        print(">> Low epistemic ratio — uncertainty is mostly aleatoric (inherent randomness)")
    print("   This distinction drives AETHER's adaptive governance modulation.")


if __name__ == "__main__":
    main()
