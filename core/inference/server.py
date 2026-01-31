"""FastAPI inference server for AETHER world model.

Serves predictions to the TypeScript MCP bridge via HTTP. All responses
match the TypeScript prediction types exactly.

Endpoints:
    POST /predict  — predict from event sequence with uncertainty
    GET  /calibration — current calibration metrics
    GET  /health   — health check
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..critic.adaptive_conformal import AdaptiveConformalPredictor
from ..critic.calibration import CalibrationTracker
from ..utils.checkpoint import load_checkpoint_unsafe
from ..critic.decomposition import decompose_from_ensemble
from ..encoder.event_encoder import EventEncoder
from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from ..world_model.energy import EnergyScorer
from ..world_model.hierarchical import DEFAULT_PHASES, HierarchicalPredictor
from ..world_model.latent import INDEX_TO_VARIANT, LatentVariable
from ..world_model.transition import TransitionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models (matching TypeScript types)
# ---------------------------------------------------------------------------


class EventInput(BaseModel):
    """Single event in a prediction request."""
    activity: str
    resource: str
    timestamp: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    """Prediction request — a sequence of events for one case."""
    caseId: str
    events: list[EventInput]


class ActivityPredictionResponse(BaseModel):
    topK: list[dict[str, Any]]
    expectedDeltaHours: float


class PhasePredictionResponse(BaseModel):
    currentPhase: str
    nextPhase: str
    nextPhaseProbability: float
    expectedTransitionHours: float


class OutcomePredictionResponse(BaseModel):
    predictedStatus: str
    onTimeProbability: float
    reworkProbability: float
    expectedRemainingHours: float


class UncertaintyResponse(BaseModel):
    total: float
    epistemic: float
    aleatoric: float
    epistemicRatio: float
    method: str


class ConformalSetResponse(BaseModel):
    activitySet: list[str]
    outcomeSet: list[str]
    coverageTarget: float
    alpha: float
    empiricalCoverage: float
    setSize: int


class PredictResponse(BaseModel):
    predictionId: str
    caseId: str
    predictions: dict[str, Any]
    uncertainty: UncertaintyResponse
    energyScore: float
    conformalSet: ConformalSetResponse
    timestamp: str
    modelVersion: str


class CalibrationBucketResponse(BaseModel):
    confidenceLow: float
    confidenceHigh: float
    avgConfidence: float
    avgAccuracy: float
    count: int


class CalibrationResponse(BaseModel):
    ece: float
    mce: float
    brierScore: float
    windowSize: int
    windowStart: str
    windowEnd: str
    buckets: list[CalibrationBucketResponse]


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    modelVersion: str
    device: str


# ---------------------------------------------------------------------------
# Server state and model loading
# ---------------------------------------------------------------------------

MODEL_VERSION = "aether-v0.1.0"


class AetherInferenceState:
    """Holds loaded model components and inference state."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

        # Will be initialized on load
        self.activity_vocab: ActivityVocabulary | None = None
        self.resource_vocab: ResourceVocabulary | None = None
        self.encoder: EventEncoder | None = None
        self.transition: TransitionModel | None = None
        self.energy_scorer: EnergyScorer | None = None
        self.predictor: HierarchicalPredictor | None = None
        self.latent_var: LatentVariable | None = None

        # Inference-time components
        self.calibration_tracker = CalibrationTracker(window_size=5000)
        self.conformal = AdaptiveConformalPredictor(
            alpha_target=0.1, gamma=0.01
        )

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        activity_vocab: ActivityVocabulary,
        resource_vocab: ResourceVocabulary,
        n_activities: int = 20,
        n_phases: int = 6,
    ) -> None:
        """Load model from checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            activity_vocab: Pre-built activity vocabulary.
            resource_vocab: Pre-built resource vocabulary.
            n_activities: Number of activity classes.
            n_phases: Number of process phases.
        """
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab

        # Initialize model components
        self.encoder = EventEncoder(
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
        ).to(self.device)

        self.transition = TransitionModel().to(self.device)
        self.energy_scorer = EnergyScorer().to(self.device)
        self.predictor = HierarchicalPredictor(
            n_activities=n_activities,
            n_phases=n_phases,
        ).to(self.device)
        self.latent_var = LatentVariable().to(self.device)

        # Load weights
        checkpoint = load_checkpoint_unsafe(
            checkpoint_path, map_location=self.device, trusted_source=True
        )
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.transition.load_state_dict(checkpoint["transition"])
        self.energy_scorer.load_state_dict(checkpoint["energy"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.latent_var.load_state_dict(checkpoint["latent_var"])

        # Set to eval mode
        self.encoder.eval()
        self.transition.eval()
        self.energy_scorer.eval()
        self.predictor.eval()
        self.latent_var.eval()

        self.loaded = True

    def load_default(self) -> None:
        """Initialize with random weights for development/testing.

        Creates all model components with default hyperparameters but
        no trained weights. Useful for testing the inference pipeline.
        """
        self.activity_vocab = ActivityVocabulary(embed_dim=64)
        self.resource_vocab = ResourceVocabulary(embed_dim=32)

        # Add some default activities
        for act in [
            "create_order", "approve_credit", "check_inventory",
            "ship_goods", "invoice", "receive_payment", "close_order",
            "rework", "escalate", "cancel",
        ]:
            self.activity_vocab.add_token(act)

        for res in ["system", "user_01", "manager", "auto"]:
            self.resource_vocab.add_token(res)

        n_activities = self.activity_vocab.size
        n_phases = len(DEFAULT_PHASES)

        self.encoder = EventEncoder(
            activity_vocab=self.activity_vocab,
            resource_vocab=self.resource_vocab,
        ).to(self.device)

        self.transition = TransitionModel().to(self.device)
        self.energy_scorer = EnergyScorer().to(self.device)
        self.predictor = HierarchicalPredictor(
            n_activities=n_activities, n_phases=n_phases,
        ).to(self.device)
        self.latent_var = LatentVariable().to(self.device)

        self.encoder.eval()
        self.transition.eval()
        self.energy_scorer.eval()
        self.predictor.eval()
        self.latent_var.eval()

        self.loaded = True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

state = AetherInferenceState()


def _load_vocabularies(vocab_path: Path) -> tuple[ActivityVocabulary, ResourceVocabulary]:
    """Load activity and resource vocabularies from saved JSON."""
    import json

    with open(vocab_path) as f:
        data = json.load(f)

    activity_vocab = ActivityVocabulary(embed_dim=64)
    for token in data["activity"]["token_to_idx"]:
        if token != "<UNK>":
            activity_vocab.add_token(token)

    resource_vocab = ResourceVocabulary(embed_dim=32)
    for token in data["resource"]["token_to_idx"]:
        if token != "<UNK>":
            resource_vocab.add_token(token)

    return activity_vocab, resource_vocab


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on server startup (modern FastAPI lifespan pattern)."""
    checkpoint_path = Path("data/models/best.pt")
    vocab_path = Path("data/events/vocabulary.json")

    if checkpoint_path.exists() and vocab_path.exists():
        activity_vocab, resource_vocab = _load_vocabularies(vocab_path)
        state.load_checkpoint(
            checkpoint_path,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            n_activities=activity_vocab.size,
        )
        logger.info(
            f"Model loaded: {checkpoint_path} "
            f"({activity_vocab.size} activities, {resource_vocab.size} resources)"
        )
    elif checkpoint_path.exists():
        # Checkpoint exists but no vocabulary file — create minimal vocab
        activity_vocab = ActivityVocabulary(embed_dim=64)
        resource_vocab = ResourceVocabulary(embed_dim=32)
        state.load_checkpoint(
            checkpoint_path,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
        )
        logger.info(f"Model loaded (no vocab): {checkpoint_path}")
    else:
        # Development mode: random weights
        state.load_default()
        logger.info("Development mode: random weights loaded")

    yield  # Server runs here

    logger.info("Inference server shutting down")


app = FastAPI(
    title="AETHER Inference Server",
    description="JEPA-style world model for discrete business event prediction",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.loaded else "not_loaded",
        modelLoaded=state.loaded,
        modelVersion=MODEL_VERSION,
        device=str(state.device),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate prediction with uncertainty for an event sequence."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    assert state.activity_vocab is not None
    assert state.resource_vocab is not None
    assert state.encoder is not None
    assert state.transition is not None
    assert state.energy_scorer is not None
    assert state.predictor is not None
    assert state.latent_var is not None

    # Encode events
    events = request.events
    seq_len = len(events)
    if seq_len == 0:
        raise HTTPException(status_code=400, detail="No events provided")

    with torch.no_grad():
        # Build input tensors
        activity_ids = torch.tensor(
            [[state.activity_vocab.encode(e.activity) for e in events]],
            dtype=torch.long,
            device=state.device,
        )
        resource_ids = torch.tensor(
            [[state.resource_vocab.encode(e.resource) for e in events]],
            dtype=torch.long,
            device=state.device,
        )
        attributes = torch.zeros(
            1, seq_len, 8, device=state.device
        )
        time_deltas = torch.zeros(1, seq_len, device=state.device)

        # Encode to latent space
        z = state.encoder(
            activity_ids=activity_ids,
            resource_ids=resource_ids,
            attributes=attributes,
            time_deltas=time_deltas,
        )  # (1, seq_len, 128)

        z_last = z[:, -1, :]  # (1, 128)

        # Hierarchical predictions
        preds = state.predictor(z_last)

        # Path variant
        variant_out = state.latent_var(z_last)
        variant_probs = variant_out["probs"].squeeze(0)
        best_variant_idx = variant_probs.argmax().item()
        _best_variant_name = INDEX_TO_VARIANT.get(best_variant_idx, "unknown")  # noqa: F841

        # Transition prediction for energy score
        action_one_hot = F.one_hot(
            torch.tensor([1], device=state.device),  # "standard" action
            num_classes=4,
        ).float()
        z_pred = state.transition(z_last, action_one_hot, variant_out["sample"])
        # Energy score (use prediction vs itself as baseline)
        energy_raw = state.energy_scorer.normalized_energy(z_pred, z_last)
        energy_score = energy_raw.item()

        # --- Uncertainty decomposition via ensemble approximation ---
        # Use MC dropout-like approach: run prediction multiple times
        # with different noise (in production, use actual ensemble)
        ensemble_preds: list[torch.Tensor] = []
        for _ in range(5):
            noise = torch.randn_like(z_last) * 0.05
            p = state.predictor(z_last + noise)
            ensemble_preds.append(p["activity"]["probs"].squeeze(0))

        uncertainty = decompose_from_ensemble(ensemble_preds)

        # Activity prediction top-K
        act_probs = preds["activity"]["probs"].squeeze(0)
        top_k = min(5, act_probs.numel())
        top_probs, top_indices = act_probs.topk(top_k)

        top_k_activities = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            act_name = state.activity_vocab.decode(idx)
            top_k_activities.append({"activity": act_name, "probability": round(prob, 4)})

        # Conformal prediction set
        conf_set_indices = state.conformal.get_prediction_set(act_probs)
        conf_set_activities = [state.activity_vocab.decode(i) for i in conf_set_indices]
        coverage_stats = state.conformal.get_coverage_stats()

        # Phase prediction
        phase_probs = preds["phase"]["current_probs"].squeeze(0)
        current_phase_idx = phase_probs.argmax().item()
        next_phase_probs = preds["phase"]["next_probs"].squeeze(0)
        next_phase_idx = next_phase_probs.argmax().item()

        # Outcome prediction
        ontime_prob = preds["outcome"]["ontime_prob"].item()
        rework_prob = preds["outcome"]["rework_prob"].item()
        remaining_hours = preds["outcome"]["remaining_hours"].item()

        # Determine predicted status
        if rework_prob > 0.5:
            predicted_status = "rework"
        elif ontime_prob > 0.5:
            predicted_status = "completed"
        else:
            predicted_status = "late"

    now = datetime.now(timezone.utc).isoformat()

    return PredictResponse(
        predictionId=str(uuid.uuid4()),
        caseId=request.caseId,
        predictions={
            "activity": ActivityPredictionResponse(
                topK=top_k_activities,
                expectedDeltaHours=round(
                    preds["activity"]["delta_hours"].item(), 2
                ),
            ).model_dump(),
            "phase": PhasePredictionResponse(
                currentPhase=DEFAULT_PHASES[current_phase_idx % len(DEFAULT_PHASES)],
                nextPhase=DEFAULT_PHASES[next_phase_idx % len(DEFAULT_PHASES)],
                nextPhaseProbability=round(next_phase_probs[next_phase_idx].item(), 4),
                expectedTransitionHours=round(
                    preds["phase"]["transition_hours"].item(), 2
                ),
            ).model_dump(),
            "outcome": OutcomePredictionResponse(
                predictedStatus=predicted_status,
                onTimeProbability=round(ontime_prob, 4),
                reworkProbability=round(rework_prob, 4),
                expectedRemainingHours=round(remaining_hours, 2),
            ).model_dump(),
        },
        uncertainty=UncertaintyResponse(**uncertainty),
        energyScore=round(energy_score, 4),
        conformalSet=ConformalSetResponse(
            activitySet=conf_set_activities,
            outcomeSet=[predicted_status],
            coverageTarget=coverage_stats["coverageTarget"],
            alpha=coverage_stats["currentAlpha"],
            empiricalCoverage=coverage_stats["empiricalCoverage"],
            setSize=len(conf_set_activities),
        ),
        timestamp=now,
        modelVersion=MODEL_VERSION,
    )


@app.get("/calibration", response_model=CalibrationResponse)
async def calibration() -> CalibrationResponse:
    """Return current calibration metrics."""
    metrics = state.calibration_tracker.compute_metrics()
    return CalibrationResponse(
        ece=metrics["ece"],
        mce=metrics["mce"],
        brierScore=metrics["brierScore"],
        windowSize=metrics["windowSize"],
        windowStart=metrics["windowStart"],
        windowEnd=metrics["windowEnd"],
        buckets=[
            CalibrationBucketResponse(**b) for b in metrics["buckets"]
        ],
    )
