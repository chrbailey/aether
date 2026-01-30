"""Tests for the AETHER FastAPI inference server.

Verifies:
1. /health endpoint returns status, modelLoaded, device
2. /predict endpoint returns predictions, uncertainty, energy, conformal set
3. /calibration endpoint returns ECE, MCE, Brier metrics
4. Pydantic models validate input correctly
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.inference.server import (
    AetherInferenceState,
    EventInput,
    PredictRequest,
    app,
)

# FastAPI test client (requires httpx, which comes with fastapi[all])
try:
    from fastapi.testclient import TestClient
    HAS_TEST_CLIENT = True
except ImportError:
    HAS_TEST_CLIENT = False

requires_test_client = pytest.mark.skipif(
    not HAS_TEST_CLIENT,
    reason="fastapi.testclient not available (pip install httpx)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_state() -> AetherInferenceState:
    """Create a state with default (random) weights for testing."""
    state = AetherInferenceState()
    state.load_default()
    return state


@pytest.fixture
def client(mock_state: AetherInferenceState):
    """Create a TestClient with mocked inference state."""
    if not HAS_TEST_CLIENT:
        pytest.skip("TestClient not available")

    # Patch the module-level `state` used by the FastAPI endpoints
    with patch("core.inference.server.state", mock_state):
        with TestClient(app) as c:
            yield c


# ============================================================================
# TestHealth
# ============================================================================


@requires_test_client
class TestHealth:
    """Test the /health endpoint."""

    def test_returns_healthy_when_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["modelLoaded"] is True
        assert "device" in data
        assert data["modelVersion"].startswith("aether-")

    def test_returns_not_loaded_when_unloaded(self):
        unloaded_state = AetherInferenceState()
        # Also mock the lifespan so it doesn't call load_default()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            yield

        from fastapi import FastAPI
        test_app = FastAPI(lifespan=noop_lifespan)

        # Copy routes from the real app
        for route in app.routes:
            test_app.routes.append(route)

        with patch("core.inference.server.state", unloaded_state):
            with TestClient(test_app) as c:
                resp = c.get("/health")
                data = resp.json()
                assert data["status"] == "not_loaded"
                assert data["modelLoaded"] is False


# ============================================================================
# TestPredict
# ============================================================================


@requires_test_client
class TestPredict:
    """Test the /predict endpoint with mock models."""

    def test_predict_returns_full_response(self, client):
        payload = {
            "caseId": "test_001",
            "events": [
                {"activity": "create_order", "resource": "user_01",
                 "timestamp": "2024-06-15T09:00:00Z"},
                {"activity": "approve_credit", "resource": "manager",
                 "timestamp": "2024-06-15T10:00:00Z"},
            ],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "predictionId" in data
        assert data["caseId"] == "test_001"
        assert "predictions" in data
        assert "activity" in data["predictions"]
        assert "phase" in data["predictions"]
        assert "outcome" in data["predictions"]

    def test_predict_has_uncertainty(self, client):
        payload = {
            "caseId": "test_002",
            "events": [
                {"activity": "create_order", "resource": "system"},
            ],
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()

        unc = data["uncertainty"]
        assert "total" in unc
        assert "epistemic" in unc
        assert "aleatoric" in unc
        assert "epistemicRatio" in unc
        assert unc["method"] == "ensemble_variance"

    def test_predict_has_energy_score(self, client):
        payload = {
            "caseId": "test_003",
            "events": [
                {"activity": "create_order", "resource": "system"},
            ],
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert "energyScore" in data
        assert isinstance(data["energyScore"], float)

    def test_predict_has_conformal_set(self, client):
        payload = {
            "caseId": "test_004",
            "events": [
                {"activity": "create_order", "resource": "system"},
            ],
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()
        cs = data["conformalSet"]
        assert "activitySet" in cs
        assert isinstance(cs["activitySet"], list)
        assert "coverageTarget" in cs
        assert "alpha" in cs
        assert "setSize" in cs

    def test_predict_rejects_empty_events(self, client):
        payload = {"caseId": "empty", "events": []}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400

    def test_predict_rejects_not_loaded(self):
        unloaded = AetherInferenceState()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            yield

        from fastapi import FastAPI
        test_app = FastAPI(lifespan=noop_lifespan)
        for route in app.routes:
            test_app.routes.append(route)

        with patch("core.inference.server.state", unloaded):
            with TestClient(test_app) as c:
                payload = {
                    "caseId": "x",
                    "events": [{"activity": "a", "resource": "b"}],
                }
                resp = c.post("/predict", json=payload)
                assert resp.status_code == 503


# ============================================================================
# TestCalibration
# ============================================================================


@requires_test_client
class TestCalibration:
    """Test the /calibration endpoint."""

    def test_returns_calibration_metrics(self, client):
        resp = client.get("/calibration")
        assert resp.status_code == 200
        data = resp.json()

        assert "ece" in data
        assert "mce" in data
        assert "brierScore" in data
        assert "windowSize" in data
        assert isinstance(data["buckets"], list)


# ============================================================================
# TestPydanticModels
# ============================================================================


class TestPydanticModels:
    """Test Pydantic request/response validation."""

    def test_event_input_minimal(self):
        event = EventInput(activity="create_order", resource="system")
        assert event.activity == "create_order"
        assert event.timestamp == ""
        assert event.attributes == {}

    def test_event_input_full(self):
        event = EventInput(
            activity="ship_goods",
            resource="user_01",
            timestamp="2024-06-15T09:00:00Z",
            attributes={"weight": 5.0, "priority": "high"},
        )
        assert event.attributes["weight"] == 5.0

    def test_predict_request_validation(self):
        req = PredictRequest(
            caseId="case_001",
            events=[
                EventInput(activity="create_order", resource="system"),
            ],
        )
        assert req.caseId == "case_001"
        assert len(req.events) == 1

    def test_predict_request_rejects_missing_case_id(self):
        with pytest.raises(Exception):
            PredictRequest(events=[])  # type: ignore[call-arg]
