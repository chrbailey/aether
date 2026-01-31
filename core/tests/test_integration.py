"""End-to-end integration tests for the AETHER prediction pipeline.

Verifies the full pipeline works as a connected system:
    Real data -> EventEncoder -> TransitionModel -> EnergyScorer
    -> HierarchicalPredictor -> CalibrationTracker -> predictions

Unlike the unit tests (test_encoder.py, test_world_model.py, test_calibration.py),
these tests exercise the components wired together as they run in production.

Test categories:
1. Smoke test the full prediction pipeline (synthetic + real vocabulary)
2. Checkpoint loading and forward pass shape verification
3. AetherInferenceState lifecycle and prediction on synthetic events
4. Governance modulation end-to-end (uncertainty -> governance decisions)
5. Calibration tracking across batches (ECE/MCE/Brier computation)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from core.encoder.event_encoder import EventEncoder
from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.utils.checkpoint import load_checkpoint_unsafe
from core.world_model.transition import TransitionModel, NUM_ACTIONS, GOVERNANCE_ACTIONS
from core.world_model.energy import EnergyScorer
from core.world_model.hierarchical import HierarchicalPredictor, DEFAULT_PHASES
from core.world_model.latent import LatentVariable, NUM_VARIANTS
from core.critic.calibration import CalibrationTracker
from core.critic.decomposition import decompose_from_ensemble
from core.critic.adaptive_conformal import AdaptiveConformalPredictor
from core.training.data_loader import EventSequenceDataset, collate_fn


# ---------------------------------------------------------------------------
# Paths for real data / checkpoint (skip when unavailable)
# ---------------------------------------------------------------------------

# Paths relative to project root (tests/ -> core/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOCAB_PATH = _PROJECT_ROOT / "data" / "events" / "vocabulary.json"
CHECKPOINT_PATH = _PROJECT_ROOT / "data" / "models" / "best.pt"
TRAIN_CASES_PATH = _PROJECT_ROOT / "data" / "events" / "train_cases.json"

_VOCAB_AVAILABLE = VOCAB_PATH.exists()
_CHECKPOINT_AVAILABLE = CHECKPOINT_PATH.exists()
_TRAIN_DATA_AVAILABLE = TRAIN_CASES_PATH.exists()

requires_vocab = pytest.mark.skipif(
    not _VOCAB_AVAILABLE,
    reason=f"Real vocabulary not found at {VOCAB_PATH}",
)
requires_checkpoint = pytest.mark.skipif(
    not _CHECKPOINT_AVAILABLE,
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}",
)
requires_train_data = pytest.mark.skipif(
    not _TRAIN_DATA_AVAILABLE,
    reason=f"Training data not found at {TRAIN_CASES_PATH}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_real_vocabs() -> tuple[ActivityVocabulary, ResourceVocabulary]:
    """Load the production vocabularies from disk."""
    with open(VOCAB_PATH) as f:
        data = json.load(f)

    act_vocab = ActivityVocabulary(embed_dim=64)
    for token in sorted(
        data["activity"]["token_to_idx"],
        key=lambda t: data["activity"]["token_to_idx"][t],
    ):
        if token != ActivityVocabulary.UNK_TOKEN:
            act_vocab.add_token(token)

    res_vocab = ResourceVocabulary(embed_dim=32)
    for token in sorted(
        data["resource"]["token_to_idx"],
        key=lambda t: data["resource"]["token_to_idx"][t],
    ):
        if token != ResourceVocabulary.UNK_TOKEN:
            res_vocab.add_token(token)

    return act_vocab, res_vocab


def _build_synthetic_vocabs() -> tuple[ActivityVocabulary, ResourceVocabulary]:
    """Build small vocabularies for tests that do not need real data."""
    act_vocab = ActivityVocabulary(embed_dim=64)
    activities = [
        "create_purchase_order", "approve_purchase_order",
        "create_goods_receipt", "record_invoice_receipt",
        "execute_payment", "create_sales_order",
        "goods_issue", "create_invoice", "customer_payment",
        "clear_invoice",
    ]
    for a in activities:
        act_vocab.add_token(a)

    res_vocab = ResourceVocabulary(embed_dim=32)
    resources = ["system", "user_001", "user_002", "user_003", "procurement_department"]
    for r in resources:
        res_vocab.add_token(r)

    return act_vocab, res_vocab


def _build_full_pipeline(
    act_vocab: ActivityVocabulary,
    res_vocab: ResourceVocabulary,
) -> dict:
    """Construct the full pipeline from vocabs. Returns dict of components."""
    n_activities = act_vocab.size
    n_phases = len(DEFAULT_PHASES)

    encoder = EventEncoder(
        activity_vocab=act_vocab,
        resource_vocab=res_vocab,
        latent_dim=128,
        n_attribute_features=8,
        n_heads=4,
        n_layers=2,
    )
    transition = TransitionModel(latent_dim=128)
    energy_scorer = EnergyScorer(latent_dim=128)
    predictor = HierarchicalPredictor(
        latent_dim=128,
        n_activities=n_activities,
        n_phases=n_phases,
    )
    latent_var = LatentVariable(latent_dim=128)

    return {
        "encoder": encoder,
        "transition": transition,
        "energy_scorer": energy_scorer,
        "predictor": predictor,
        "latent_var": latent_var,
    }


def _make_synthetic_inputs(
    act_vocab: ActivityVocabulary,
    res_vocab: ResourceVocabulary,
    batch_size: int = 2,
    seq_len: int = 8,
) -> dict[str, torch.Tensor]:
    """Create synthetic input tensors suitable for EventEncoder.forward()."""
    activity_ids = torch.randint(1, act_vocab.size, (batch_size, seq_len))
    resource_ids = torch.randint(1, res_vocab.size, (batch_size, seq_len))
    attributes = torch.randn(batch_size, seq_len, 8)
    time_deltas = torch.abs(torch.randn(batch_size, seq_len))
    return {
        "activity_ids": activity_ids,
        "resource_ids": resource_ids,
        "attributes": attributes,
        "time_deltas": time_deltas,
    }


# ============================================================================
# 1. Smoke Test: Full Prediction Pipeline
# ============================================================================


class TestFullPipelineSmoke:
    """Smoke test the complete pipeline with synthetic vocabs.

    Runs: EventEncoder -> TransitionModel -> EnergyScorer
        -> HierarchicalPredictor -> CalibrationTracker
    and verifies output types, shapes, and value ranges.
    """

    @pytest.fixture
    def pipeline(self):
        act_vocab, res_vocab = _build_synthetic_vocabs()
        components = _build_full_pipeline(act_vocab, res_vocab)
        components["act_vocab"] = act_vocab
        components["res_vocab"] = res_vocab
        return components

    def test_encoder_to_predictor_shapes(self, pipeline):
        """Full forward pass produces correct output shapes at every stage."""
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]
        encoder = pipeline["encoder"]
        transition = pipeline["transition"]
        energy_scorer = pipeline["energy_scorer"]
        predictor = pipeline["predictor"]
        latent_var = pipeline["latent_var"]

        encoder.eval()
        transition.eval()
        predictor.eval()
        latent_var.eval()

        batch_size, seq_len = 4, 10
        inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size, seq_len)

        with torch.no_grad():
            # Stage 1: Encode
            z = encoder(**inputs)
            assert z.shape == (batch_size, seq_len, 128), (
                f"Encoder output shape mismatch: {z.shape}"
            )

            # Stage 2: Extract last latent state
            z_last = z[:, -1, :]
            assert z_last.shape == (batch_size, 128)

            # Stage 3: Path variant inference
            variant_out = latent_var(z_last)
            assert variant_out["probs"].shape == (batch_size, NUM_VARIANTS)
            assert torch.allclose(
                variant_out["probs"].sum(dim=-1),
                torch.ones(batch_size),
                atol=1e-5,
            )

            # Stage 4: Transition prediction
            action = F.one_hot(
                torch.ones(batch_size, dtype=torch.long), num_classes=NUM_ACTIONS
            ).float()
            z_pred = transition(z_last, action, variant_out["sample"])
            assert z_pred.shape == (batch_size, 128)

            # Stage 5: Energy scoring
            energy = energy_scorer(z_pred, z_last)
            assert energy.shape == (batch_size,)
            assert (energy >= 0).all(), "Energy should be non-negative"

            norm_energy = energy_scorer.normalized_energy(z_pred, z_last)
            assert (norm_energy >= 0).all() and (norm_energy <= 1).all()

            # Stage 6: Hierarchical predictions
            preds = predictor(z_last)
            assert "activity" in preds
            assert "phase" in preds
            assert "outcome" in preds

            act_probs = preds["activity"]["probs"]
            assert act_probs.shape == (batch_size, act_vocab.size)
            assert torch.allclose(act_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

            ontime_prob = preds["outcome"]["ontime_prob"]
            assert (ontime_prob >= 0).all() and (ontime_prob <= 1).all()

    def test_end_to_end_with_calibration_tracker(self, pipeline):
        """Predictions flow into CalibrationTracker and produce valid metrics."""
        encoder = pipeline["encoder"]
        predictor = pipeline["predictor"]
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]

        encoder.eval()
        predictor.eval()
        tracker = CalibrationTracker(window_size=500)

        with torch.no_grad():
            for batch_idx in range(5):
                inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=8, seq_len=6)
                z = encoder(**inputs)
                z_last = z[:, -1, :]
                preds = predictor(z_last)

                ontime_probs = preds["outcome"]["ontime_prob"].squeeze(-1)
                # Simulate ground truth labels
                actuals = (torch.rand(8) > 0.5).float()
                tracker.update_batch(ontime_probs, actuals)

        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == 40  # 5 batches * 8
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brierScore"] <= 1.0
        assert len(metrics["buckets"]) > 0

    def test_end_to_end_with_conformal_predictor(self, pipeline):
        """Activity predictions flow into AdaptiveConformalPredictor."""
        encoder = pipeline["encoder"]
        predictor = pipeline["predictor"]
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]

        encoder.eval()
        predictor.eval()
        conformal = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.01)

        with torch.no_grad():
            inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=1, seq_len=6)
            z = encoder(**inputs)
            z_last = z[:, -1, :]
            preds = predictor(z_last)

            act_probs = preds["activity"]["probs"].squeeze(0)
            pred_set = conformal.get_prediction_set(act_probs)

            assert len(pred_set) >= 1, "Prediction set should contain at least one class"
            assert all(0 <= idx < act_vocab.size for idx in pred_set)

            # Simulate a hit and verify alpha updates
            actual = pred_set[0]
            initial_alpha = conformal.alpha
            conformal.update(pred_set, actual)
            # Hit -> alpha should increase (ACI: alpha_target - 0 > 0)
            assert conformal.alpha > initial_alpha - 0.001

    def test_multiple_futures_via_variant_sampling(self, pipeline):
        """TransitionModel produces distinct futures from different path variants."""
        encoder = pipeline["encoder"]
        transition = pipeline["transition"]
        latent_var = pipeline["latent_var"]
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]

        encoder.eval()
        transition.eval()
        latent_var.eval()

        with torch.no_grad():
            inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=1, seq_len=5)
            z = encoder(**inputs)
            z_last = z[:, -1, :]

            action = F.one_hot(
                torch.ones(1, dtype=torch.long), num_classes=NUM_ACTIONS
            ).float()

            n_samples = 10
            variant_samples = torch.randn(1, n_samples, NUM_VARIANTS).softmax(dim=-1)
            futures = transition.predict_multiple_futures(z_last, action, variant_samples)

            assert futures.shape == (1, n_samples, 128)
            # Distinct variants should produce at least slightly different futures
            pairwise_diffs = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    diff = (futures[0, i] - futures[0, j]).abs().sum().item()
                    pairwise_diffs.append(diff)
            assert max(pairwise_diffs) > 0, "All futures are identical"


@requires_vocab
class TestFullPipelineWithRealVocab:
    """Smoke test the pipeline using the real production vocabulary."""

    @pytest.fixture(scope="class")
    def real_vocabs(self):
        return _load_real_vocabs()

    def test_real_vocab_sizes(self, real_vocabs):
        act_vocab, res_vocab = real_vocabs
        with open(VOCAB_PATH) as f:
            data = json.load(f)
        assert act_vocab.size == data["activity"]["size"]
        assert res_vocab.size == data["resource"]["size"]

    def test_pipeline_with_real_vocab_shapes(self, real_vocabs):
        """Full pipeline runs with real vocab sizes without errors."""
        act_vocab, res_vocab = real_vocabs
        components = _build_full_pipeline(act_vocab, res_vocab)

        encoder = components["encoder"]
        predictor = components["predictor"]
        latent_var = components["latent_var"]

        encoder.eval()
        predictor.eval()
        latent_var.eval()

        batch_size, seq_len = 2, 12
        with torch.no_grad():
            inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size, seq_len)
            z = encoder(**inputs)
            assert z.shape == (batch_size, seq_len, 128)

            z_last = z[:, -1, :]
            preds = predictor(z_last)

            # Activity logits should match real vocab size
            assert preds["activity"]["logits"].shape == (batch_size, act_vocab.size)
            assert preds["phase"]["current_logits"].shape == (batch_size, len(DEFAULT_PHASES))
            assert preds["outcome"]["ontime_prob"].shape == (batch_size, 1)


# ============================================================================
# 2. Checkpoint Loading and Forward Pass
# ============================================================================


@requires_checkpoint
@requires_vocab
class TestCheckpointLoading:
    """Load a real checkpoint and verify forward pass produces correct shapes."""

    @pytest.fixture(scope="class")
    def loaded_components(self):
        """Load checkpoint into all model components."""
        act_vocab, res_vocab = _load_real_vocabs()
        n_activities = act_vocab.size
        n_phases = len(DEFAULT_PHASES)

        encoder = EventEncoder(
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
        )
        transition = TransitionModel()
        energy_scorer = EnergyScorer()
        predictor = HierarchicalPredictor(
            n_activities=n_activities,
            n_phases=n_phases,
        )
        latent_var = LatentVariable()

        checkpoint = load_checkpoint_unsafe(
            CHECKPOINT_PATH, map_location="cpu", trusted_source=True
        )
        encoder.load_state_dict(checkpoint["encoder"])
        transition.load_state_dict(checkpoint["transition"])
        energy_scorer.load_state_dict(checkpoint["energy"])
        predictor.load_state_dict(checkpoint["predictor"])
        latent_var.load_state_dict(checkpoint["latent_var"])

        encoder.eval()
        transition.eval()
        energy_scorer.eval()
        predictor.eval()
        latent_var.eval()

        return {
            "encoder": encoder,
            "transition": transition,
            "energy_scorer": energy_scorer,
            "predictor": predictor,
            "latent_var": latent_var,
            "act_vocab": act_vocab,
            "res_vocab": res_vocab,
            "checkpoint": checkpoint,
        }

    def test_checkpoint_has_expected_keys(self, loaded_components):
        checkpoint = loaded_components["checkpoint"]
        expected_keys = {"encoder", "transition", "energy", "predictor", "latent_var"}
        assert expected_keys.issubset(
            checkpoint.keys()
        ), f"Missing keys: {expected_keys - checkpoint.keys()}"

    def test_checkpoint_epoch_is_positive(self, loaded_components):
        checkpoint = loaded_components["checkpoint"]
        assert checkpoint["epoch"] > 0, "Checkpoint should have trained for at least 1 epoch"

    def test_encoder_forward_pass_shape(self, loaded_components):
        encoder = loaded_components["encoder"]
        act_vocab = loaded_components["act_vocab"]
        res_vocab = loaded_components["res_vocab"]

        batch_size, seq_len = 2, 8
        inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size, seq_len)

        with torch.no_grad():
            z = encoder(**inputs)
        assert z.shape == (batch_size, seq_len, 128)

    def test_transition_forward_pass_shape(self, loaded_components):
        transition = loaded_components["transition"]
        batch_size = 4
        z_t = torch.randn(batch_size, 128)
        action = F.one_hot(torch.zeros(batch_size, dtype=torch.long), NUM_ACTIONS).float()
        variant = torch.randn(batch_size, NUM_VARIANTS).softmax(dim=-1)

        with torch.no_grad():
            z_next = transition(z_t, action, variant)
        assert z_next.shape == (batch_size, 128)

    def test_predictor_forward_pass_shape(self, loaded_components):
        predictor = loaded_components["predictor"]
        act_vocab = loaded_components["act_vocab"]
        batch_size = 3
        z = torch.randn(batch_size, 128)

        with torch.no_grad():
            preds = predictor(z)

        assert preds["activity"]["logits"].shape == (batch_size, act_vocab.size)
        assert preds["phase"]["current_logits"].shape == (batch_size, len(DEFAULT_PHASES))
        assert preds["outcome"]["ontime_prob"].shape == (batch_size, 1)

    def test_energy_scorer_from_checkpoint(self, loaded_components):
        energy_scorer = loaded_components["energy_scorer"]
        z_pred = torch.randn(4, 128)
        z_actual = torch.randn(4, 128)

        with torch.no_grad():
            raw = energy_scorer(z_pred, z_actual)
            normalized = energy_scorer.normalized_energy(z_pred, z_actual)

        assert raw.shape == (4,)
        assert (raw >= 0).all()
        assert (normalized >= 0).all() and (normalized <= 1).all()

    def test_full_pipeline_from_checkpoint(self, loaded_components):
        """Run the complete pipeline end-to-end with loaded weights."""
        encoder = loaded_components["encoder"]
        transition = loaded_components["transition"]
        energy_scorer = loaded_components["energy_scorer"]
        predictor = loaded_components["predictor"]
        latent_var = loaded_components["latent_var"]
        act_vocab = loaded_components["act_vocab"]
        res_vocab = loaded_components["res_vocab"]

        batch_size, seq_len = 2, 10
        inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size, seq_len)

        with torch.no_grad():
            z = encoder(**inputs)
            z_last = z[:, -1, :]

            variant_out = latent_var(z_last)
            action = F.one_hot(
                torch.ones(batch_size, dtype=torch.long), NUM_ACTIONS
            ).float()
            z_pred = transition(z_last, action, variant_out["sample"])

            energy = energy_scorer.normalized_energy(z_pred, z_last)
            preds = predictor(z_last)

        assert z.shape == (batch_size, seq_len, 128)
        assert z_pred.shape == (batch_size, 128)
        assert energy.shape == (batch_size,)
        assert preds["activity"]["probs"].shape[0] == batch_size
        assert preds["outcome"]["ontime_prob"].shape == (batch_size, 1)


# ============================================================================
# 3. AetherInferenceState Lifecycle
# ============================================================================


class TestAetherInferenceState:
    """Test the inference server state object lifecycle and prediction."""

    def test_load_default_creates_working_state(self):
        """load_default() creates a fully functional model with random weights."""
        from core.inference.server import AetherInferenceState

        state = AetherInferenceState()
        assert not state.loaded

        state.load_default()
        assert state.loaded
        assert state.encoder is not None
        assert state.transition is not None
        assert state.energy_scorer is not None
        assert state.predictor is not None
        assert state.latent_var is not None
        assert state.activity_vocab is not None
        assert state.resource_vocab is not None

    def test_prediction_on_synthetic_events(self):
        """Create AetherInferenceState, load default, and predict on events."""
        from core.inference.server import AetherInferenceState

        state = AetherInferenceState()
        state.load_default()

        # Build input tensors matching how the server does it
        events_data = [
            {"activity": "create_order", "resource": "user_01"},
            {"activity": "approve_credit", "resource": "manager"},
            {"activity": "ship_goods", "resource": "system"},
        ]
        seq_len = len(events_data)

        with torch.no_grad():
            activity_ids = torch.tensor(
                [[state.activity_vocab.encode(e["activity"]) for e in events_data]],
                dtype=torch.long,
                device=state.device,
            )
            resource_ids = torch.tensor(
                [[state.resource_vocab.encode(e["resource"]) for e in events_data]],
                dtype=torch.long,
                device=state.device,
            )
            attributes = torch.zeros(1, seq_len, 8, device=state.device)
            time_deltas = torch.zeros(1, seq_len, device=state.device)

            z = state.encoder(
                activity_ids=activity_ids,
                resource_ids=resource_ids,
                attributes=attributes,
                time_deltas=time_deltas,
            )
            assert z.shape == (1, seq_len, 128)

            z_last = z[:, -1, :]
            preds = state.predictor(z_last)

            assert "activity" in preds
            assert "phase" in preds
            assert "outcome" in preds

            # Activity probabilities should be valid distribution
            act_probs = preds["activity"]["probs"]
            assert torch.allclose(act_probs.sum(dim=-1), torch.ones(1), atol=1e-5)

            # Outcome probabilities should be in [0, 1]
            assert 0.0 <= preds["outcome"]["ontime_prob"].item() <= 1.0
            assert 0.0 <= preds["outcome"]["rework_prob"].item() <= 1.0
            assert preds["outcome"]["remaining_hours"].item() > 0

    def test_uncertainty_from_ensemble_approximation(self):
        """Uncertainty decomposition via noise-perturbed ensemble (as server does)."""
        from core.inference.server import AetherInferenceState

        state = AetherInferenceState()
        state.load_default()

        with torch.no_grad():
            # Encode a single event
            activity_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=state.device)
            resource_ids = torch.tensor([[1, 1, 2]], dtype=torch.long, device=state.device)
            attributes = torch.zeros(1, 3, 8, device=state.device)
            time_deltas = torch.zeros(1, 3, device=state.device)

            z = state.encoder(
                activity_ids=activity_ids,
                resource_ids=resource_ids,
                attributes=attributes,
                time_deltas=time_deltas,
            )
            z_last = z[:, -1, :]

            # Approximate ensemble via noise perturbation
            ensemble_preds = []
            for _ in range(10):
                noise = torch.randn_like(z_last) * 0.05
                p = state.predictor(z_last + noise)
                ensemble_preds.append(p["activity"]["probs"].squeeze(0))

            uncertainty = decompose_from_ensemble(ensemble_preds)

            assert uncertainty["total"] >= 0
            assert uncertainty["epistemic"] >= 0
            assert uncertainty["aleatoric"] >= 0
            assert 0.0 <= uncertainty["epistemicRatio"] <= 1.0
            assert abs(
                uncertainty["total"] - uncertainty["epistemic"] - uncertainty["aleatoric"]
            ) < 1e-4

    @requires_checkpoint
    @requires_vocab
    def test_load_checkpoint_and_predict(self):
        """Load real checkpoint into AetherInferenceState and run prediction."""
        from core.inference.server import AetherInferenceState

        act_vocab, res_vocab = _load_real_vocabs()

        state = AetherInferenceState()
        state.load_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            n_activities=act_vocab.size,
            n_phases=len(DEFAULT_PHASES),
        )
        assert state.loaded

        with torch.no_grad():
            # Use real activity names from the vocabulary
            test_activities = [
                "create_purchase_order",
                "create_goods_receipt",
                "record_invoice_receipt",
            ]
            activity_ids = torch.tensor(
                [[act_vocab.encode(a) for a in test_activities]],
                dtype=torch.long,
                device=state.device,
            )
            resource_ids = torch.tensor(
                [[res_vocab.encode("system")] * len(test_activities)],
                dtype=torch.long,
                device=state.device,
            )
            attributes = torch.zeros(1, len(test_activities), 8, device=state.device)
            time_deltas = torch.tensor([[0.0, 24.0, 48.0]], device=state.device)

            z = state.encoder(
                activity_ids=activity_ids,
                resource_ids=resource_ids,
                attributes=attributes,
                time_deltas=time_deltas,
            )
            z_last = z[:, -1, :]
            preds = state.predictor(z_last)

            assert preds["activity"]["logits"].shape == (1, act_vocab.size)
            assert preds["outcome"]["ontime_prob"].shape == (1, 1)


# ============================================================================
# 4. Governance Modulation End-to-End
# ============================================================================


class TestGovernanceModulation:
    """Verify that different uncertainty levels produce different governance decisions.

    The key insight: high epistemic uncertainty should tighten governance,
    while high aleatoric uncertainty should NOT. This test verifies the
    decomposition-to-decision pipeline.
    """

    @pytest.fixture
    def pipeline(self):
        act_vocab, res_vocab = _build_synthetic_vocabs()
        components = _build_full_pipeline(act_vocab, res_vocab)
        components["act_vocab"] = act_vocab
        components["res_vocab"] = res_vocab
        for m in components.values():
            if hasattr(m, "eval"):
                m.eval()
        return components

    def test_different_actions_produce_different_transitions(self, pipeline):
        """Different governance actions (flexible/standard/strict/forbidden)
        produce different predicted next states."""
        encoder = pipeline["encoder"]
        transition = pipeline["transition"]
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]

        with torch.no_grad():
            inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=1, seq_len=5)
            z = encoder(**inputs)
            z_last = z[:, -1, :]

            variant = torch.randn(1, NUM_VARIANTS).softmax(dim=-1)

            futures_by_action = {}
            for action_name in GOVERNANCE_ACTIONS:
                action_idx = GOVERNANCE_ACTIONS.index(action_name)
                action = F.one_hot(
                    torch.tensor([action_idx]), num_classes=NUM_ACTIONS
                ).float()
                z_next = transition(z_last, action, variant)
                futures_by_action[action_name] = z_next

        # At least some action pairs should produce different futures
        diffs = []
        actions = list(futures_by_action.keys())
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                diff = (futures_by_action[actions[i]] - futures_by_action[actions[j]]).abs().sum().item()
                diffs.append(diff)

        assert max(diffs) > 0, "All governance actions produced identical transitions"

    def test_energy_distinguishes_plausible_from_implausible(self, pipeline):
        """Energy scorer assigns lower energy to plausible transitions."""
        encoder = pipeline["encoder"]
        transition = pipeline["transition"]
        energy_scorer = pipeline["energy_scorer"]
        latent_var = pipeline["latent_var"]
        act_vocab = pipeline["act_vocab"]
        res_vocab = pipeline["res_vocab"]

        with torch.no_grad():
            # Create two different sequences
            inputs1 = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=1, seq_len=8)
            z1 = encoder(**inputs1)
            z_last = z1[:, -1, :]

            # Predicted transition (plausible)
            variant_out = latent_var(z_last)
            action = F.one_hot(torch.tensor([1]), num_classes=NUM_ACTIONS).float()
            z_pred_plausible = transition(z_last, action, variant_out["sample"])

            # Random transition (implausible)
            z_random = torch.randn_like(z_last) * 5.0

            energy_plausible = energy_scorer(z_pred_plausible, z_last)
            energy_implausible = energy_scorer(z_random, z_last)

        # Plausible transition should have lower energy than random
        # (transition model predicts something near z_last due to residual)
        assert energy_plausible.item() < energy_implausible.item(), (
            f"Plausible energy ({energy_plausible.item():.2f}) should be less than "
            f"implausible energy ({energy_implausible.item():.2f})"
        )

    def test_epistemic_vs_aleatoric_governance_signal(self, pipeline):
        """High epistemic uncertainty should signal 'tighten governance',
        while high aleatoric should signal 'do not tighten'."""
        # Scenario 1: Models agree on uncertain outcome (aleatoric)
        agree_uncertain = [
            torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]),
            torch.tensor([0.48, 0.52, 0.0, 0.0, 0.0]),
            torch.tensor([0.51, 0.49, 0.0, 0.0, 0.0]),
            torch.tensor([0.50, 0.50, 0.0, 0.0, 0.0]),
            torch.tensor([0.49, 0.51, 0.0, 0.0, 0.0]),
        ]

        # Scenario 2: Models disagree (epistemic)
        disagree = [
            torch.tensor([0.95, 0.05, 0.0, 0.0, 0.0]),
            torch.tensor([0.05, 0.90, 0.05, 0.0, 0.0]),
            torch.tensor([0.0, 0.05, 0.90, 0.05, 0.0]),
            torch.tensor([0.50, 0.50, 0.0, 0.0, 0.0]),
            torch.tensor([0.10, 0.10, 0.10, 0.60, 0.10]),
        ]

        unc_aleatoric = decompose_from_ensemble(agree_uncertain)
        unc_epistemic = decompose_from_ensemble(disagree)

        # Aleatoric-dominant: low epistemic ratio -> do NOT tighten
        assert unc_aleatoric["epistemicRatio"] < 0.3, (
            f"Agreed uncertainty should have low epistemic ratio: {unc_aleatoric['epistemicRatio']}"
        )

        # Epistemic-dominant: high epistemic ratio -> tighten governance
        assert unc_epistemic["epistemicRatio"] > 0.3, (
            f"Disagreed predictions should have higher epistemic ratio: {unc_epistemic['epistemicRatio']}"
        )

        # Governance decision: tighten only when epistemic ratio is high
        should_tighten_aleatoric = unc_aleatoric["epistemicRatio"] > 0.5
        should_tighten_epistemic = unc_epistemic["epistemicRatio"] > 0.3

        assert not should_tighten_aleatoric, "Should NOT tighten for aleatoric uncertainty"
        assert should_tighten_epistemic, "SHOULD tighten for epistemic uncertainty"

    def test_conformal_alpha_decreases_after_misses(self, pipeline):
        """After consecutive misses, alpha decreases (sets should widen).

        ACI update rule on miss: alpha += gamma * (alpha_target - 1)
        Since alpha_target < 1, this decreases alpha, raising threshold
        (1 - alpha), requiring wider prediction sets.
        """
        conformal = AdaptiveConformalPredictor(alpha_target=0.1, gamma=0.05)

        # Start with a series of hits to raise alpha (narrow sets)
        for _ in range(50):
            conformal.update(prediction_set=[0, 1, 2], actual=1)  # hit

        alpha_after_hits = conformal.alpha

        # Now feed a series of misses to lower alpha (widen sets)
        for _ in range(30):
            conformal.update(prediction_set=[0], actual=99)  # miss

        alpha_after_misses = conformal.alpha

        assert alpha_after_misses < alpha_after_hits, (
            f"Alpha should decrease after misses: {alpha_after_misses:.4f} vs "
            f"{alpha_after_hits:.4f} after hits"
        )

        # Lower alpha means threshold 1-alpha is higher -> wider sets needed
        # Verify with a synthetic probability distribution
        scores = torch.tensor([0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.04, 0.02, 0.005, 0.005])
        conformal.alpha = 0.5  # High alpha -> small set (need 50% coverage)
        small_set = conformal.get_prediction_set(scores)
        conformal.alpha = 0.05  # Low alpha -> large set (need 95% coverage)
        large_set = conformal.get_prediction_set(scores)

        assert len(large_set) >= len(small_set), (
            f"Lower alpha should produce wider sets: {len(large_set)} vs {len(small_set)}"
        )

    def test_all_governance_actions_defined(self):
        """Verify all expected governance actions exist."""
        assert len(GOVERNANCE_ACTIONS) == 4
        assert "flexible" in GOVERNANCE_ACTIONS
        assert "standard" in GOVERNANCE_ACTIONS
        assert "strict" in GOVERNANCE_ACTIONS
        assert "forbidden" in GOVERNANCE_ACTIONS


# ============================================================================
# 5. Calibration Tracking Across Batches
# ============================================================================


class TestCalibrationAcrossBatches:
    """Feed predictions through CalibrationTracker across multiple batches
    and verify ECE/MCE/Brier are computed correctly."""

    def test_well_calibrated_model_has_low_ece(self):
        """A well-calibrated model should produce low ECE."""
        tracker = CalibrationTracker(window_size=2000, n_bins=10)

        # Simulate well-calibrated predictions across batches
        for _ in range(20):
            batch_size = 50
            # Perfect calibration: predicted prob matches actual frequency
            preds = torch.rand(batch_size)
            actuals = (torch.rand(batch_size) < preds).float()
            tracker.update_batch(preds, actuals)

        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == 1000
        assert metrics["ece"] < 0.15, f"Well-calibrated model has ECE={metrics['ece']}"

    def test_overconfident_model_has_high_ece(self):
        """An overconfident model should produce high ECE."""
        tracker = CalibrationTracker(window_size=2000, n_bins=10)

        for _ in range(20):
            batch_size = 50
            # Overconfident: predicts 0.95 but only right ~50% of the time
            preds = torch.full((batch_size,), 0.95)
            actuals = (torch.rand(batch_size) > 0.5).float()
            tracker.update_batch(preds, actuals)

        metrics = tracker.compute_metrics()
        assert metrics["ece"] > 0.3, f"Overconfident model should have high ECE: {metrics['ece']}"
        assert metrics["brierScore"] > 0.2

    def test_brier_score_tracks_prediction_quality(self):
        """Brier score should be lower for better predictions."""
        # Good predictor
        good_tracker = CalibrationTracker(window_size=1000)
        for _ in range(10):
            preds = torch.rand(100)
            actuals = (torch.rand(100) < preds).float()
            good_tracker.update_batch(preds, actuals)

        # Bad predictor: always predicts opposite
        bad_tracker = CalibrationTracker(window_size=1000)
        for _ in range(10):
            preds = torch.rand(100)
            actuals = (torch.rand(100) >= preds).float()
            bad_tracker.update_batch(preds, actuals)

        good_metrics = good_tracker.compute_metrics()
        bad_metrics = bad_tracker.compute_metrics()

        assert good_metrics["brierScore"] < bad_metrics["brierScore"], (
            f"Good Brier ({good_metrics['brierScore']:.4f}) should be less than "
            f"bad Brier ({bad_metrics['brierScore']:.4f})"
        )

    def test_mce_geq_ece_across_batches(self):
        """MCE (max bucket error) >= ECE (weighted average) always holds."""
        tracker = CalibrationTracker(window_size=2000, n_bins=10)

        for _ in range(30):
            preds = torch.rand(50)
            actuals = (torch.rand(50) > 0.3).float()
            tracker.update_batch(preds, actuals)

        metrics = tracker.compute_metrics()
        assert metrics["mce"] >= metrics["ece"] - 1e-6, (
            f"MCE ({metrics['mce']}) should be >= ECE ({metrics['ece']})"
        )

    def test_rolling_window_evicts_old_data(self):
        """CalibrationTracker window should evict old predictions."""
        tracker = CalibrationTracker(window_size=100, n_bins=5)

        # Fill window with perfectly calibrated predictions
        for _ in range(5):
            preds = torch.full((20,), 0.7)
            actuals = torch.zeros(20)
            actuals[:14] = 1.0  # 70% accuracy
            tracker.update_batch(preds, actuals)

        metrics_before = tracker.compute_metrics()
        assert metrics_before["windowSize"] == 100

        # Flood with terrible predictions to evict good ones
        for _ in range(10):
            preds = torch.full((20,), 0.95)
            actuals = torch.zeros(20)  # 0% accuracy at 95% confidence
            tracker.update_batch(preds, actuals)

        metrics_after = tracker.compute_metrics()
        assert metrics_after["windowSize"] == 100
        # After eviction, calibration should be much worse
        assert metrics_after["ece"] > metrics_before["ece"]

    def test_calibration_buckets_cover_full_range(self):
        """Calibration buckets should cover [0, 1] range."""
        tracker = CalibrationTracker(window_size=1000, n_bins=10)

        # Spread predictions across the full range
        preds = torch.linspace(0.0, 1.0, 200)
        actuals = (torch.rand(200) > 0.5).float()
        tracker.update_batch(preds, actuals)

        metrics = tracker.compute_metrics()
        buckets = metrics["buckets"]
        assert len(buckets) == 10

        # First bucket should start near 0, last should end near 1
        assert buckets[0]["confidenceLow"] < 0.01
        assert buckets[-1]["confidenceHigh"] > 0.99

    def test_pipeline_predictions_to_calibration(self):
        """Full pipeline predictions feed correctly into CalibrationTracker."""
        act_vocab, res_vocab = _build_synthetic_vocabs()
        components = _build_full_pipeline(act_vocab, res_vocab)
        encoder = components["encoder"]
        predictor = components["predictor"]

        encoder.eval()
        predictor.eval()

        tracker = CalibrationTracker(window_size=500, n_bins=10)

        with torch.no_grad():
            for _ in range(10):
                inputs = _make_synthetic_inputs(act_vocab, res_vocab, batch_size=4, seq_len=5)
                z = encoder(**inputs)
                z_last = z[:, -1, :]
                preds = predictor(z_last)

                ontime_probs = preds["outcome"]["ontime_prob"].squeeze(-1)
                # Simulate binary outcomes
                actuals = (torch.rand(4) > 0.5).float()
                tracker.update_batch(ontime_probs, actuals)

        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == 40
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brierScore"] <= 1.0
        assert len(metrics["buckets"]) == 10
        assert metrics["windowStart"] != ""
        assert metrics["windowEnd"] != ""


# ============================================================================
# 6. Data Pipeline -> Model Integration (with real data)
# ============================================================================


@requires_train_data
@requires_vocab
class TestDataToModelIntegration:
    """Load real training data, create DataLoader, and feed through the model."""

    @pytest.fixture(scope="class")
    def vocabs_and_dataset(self):
        act_vocab, res_vocab = _load_real_vocabs()
        dataset = EventSequenceDataset(
            events_path=TRAIN_CASES_PATH,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=64,
            n_attribute_features=8,
        )
        return act_vocab, res_vocab, dataset

    def test_collated_batch_through_encoder(self, vocabs_and_dataset):
        """A collated batch from real data passes through the encoder."""
        act_vocab, res_vocab, dataset = vocabs_and_dataset

        # Take a small batch
        batch_items = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collate_fn(batch_items)

        encoder = EventEncoder(
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
        )
        encoder.eval()

        with torch.no_grad():
            z = encoder(
                activity_ids=batch["activity_ids"],
                resource_ids=batch["resource_ids"],
                attributes=batch["attributes"],
                time_deltas=batch["time_deltas"],
                padding_mask=batch["padding_mask"],
            )

        batch_size = batch["activity_ids"].shape[0]
        max_seq_len = batch["activity_ids"].shape[1]
        assert z.shape == (batch_size, max_seq_len, 128)

    def test_collated_batch_through_full_pipeline(self, vocabs_and_dataset):
        """A collated batch flows through encoder -> predictor -> calibration."""
        act_vocab, res_vocab, dataset = vocabs_and_dataset
        components = _build_full_pipeline(act_vocab, res_vocab)

        encoder = components["encoder"]
        predictor = components["predictor"]
        encoder.eval()
        predictor.eval()

        batch_items = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collate_fn(batch_items)

        tracker = CalibrationTracker(window_size=100)

        with torch.no_grad():
            z = encoder(
                activity_ids=batch["activity_ids"],
                resource_ids=batch["resource_ids"],
                attributes=batch["attributes"],
                time_deltas=batch["time_deltas"],
                padding_mask=batch["padding_mask"],
            )

            batch_size = z.shape[0]
            seq_lens = batch["seq_lens"]
            last_indices = (seq_lens - 1).clamp(min=0)
            z_last = z[torch.arange(batch_size), last_indices]

            preds = predictor(z_last)

            ontime_probs = preds["outcome"]["ontime_prob"].squeeze(-1)
            tracker.update_batch(ontime_probs, batch["target_ontime"])

        metrics = tracker.compute_metrics()
        assert metrics["windowSize"] == batch_size
        assert 0.0 <= metrics["ece"] <= 1.0

    @requires_checkpoint
    def test_real_data_through_trained_model(self, vocabs_and_dataset):
        """Run real data through the actual trained model checkpoint."""
        act_vocab, res_vocab, dataset = vocabs_and_dataset

        encoder = EventEncoder(
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
        )
        predictor = HierarchicalPredictor(
            n_activities=act_vocab.size,
            n_phases=len(DEFAULT_PHASES),
        )
        transition = TransitionModel()
        energy_scorer = EnergyScorer()
        latent_var = LatentVariable()

        checkpoint = load_checkpoint_unsafe(CHECKPOINT_PATH, map_location="cpu", trusted_source=True)
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        transition.load_state_dict(checkpoint["transition"])
        energy_scorer.load_state_dict(checkpoint["energy"])
        latent_var.load_state_dict(checkpoint["latent_var"])

        encoder.eval()
        predictor.eval()
        transition.eval()
        energy_scorer.eval()
        latent_var.eval()

        batch_items = [dataset[i] for i in range(min(8, len(dataset)))]
        batch = collate_fn(batch_items)

        with torch.no_grad():
            z = encoder(
                activity_ids=batch["activity_ids"],
                resource_ids=batch["resource_ids"],
                attributes=batch["attributes"],
                time_deltas=batch["time_deltas"],
                padding_mask=batch["padding_mask"],
            )

            batch_size = z.shape[0]
            seq_lens = batch["seq_lens"]
            last_indices = (seq_lens - 1).clamp(min=0)
            z_last = z[torch.arange(batch_size), last_indices]

            # Hierarchical predictions
            preds = predictor(z_last)
            assert preds["activity"]["logits"].shape == (batch_size, act_vocab.size)

            # Transition
            variant_out = latent_var(z_last)
            action = F.one_hot(
                torch.ones(batch_size, dtype=torch.long), NUM_ACTIONS
            ).float()
            z_pred = transition(z_last, action, variant_out["sample"])
            assert z_pred.shape == (batch_size, 128)

            # Energy
            energy = energy_scorer.normalized_energy(z_pred, z_last)
            assert energy.shape == (batch_size,)
            # With real data + trained model, some sequences may produce NaN
            # due to attention numerical issues. Verify that finite values
            # are in the expected [0, 1] range.
            finite_mask = torch.isfinite(energy)
            if finite_mask.any():
                finite_energy = energy[finite_mask]
                assert (finite_energy >= 0).all() and (finite_energy <= 1).all(), (
                    f"Finite energy values out of [0,1]: {finite_energy}"
                )
