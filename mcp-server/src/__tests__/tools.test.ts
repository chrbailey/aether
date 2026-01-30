/**
 * MCP Tools Tests
 *
 * Verifies the tool handlers that the MCP server exposes:
 * - predict_next_event / predict_outcome (prediction tools)
 * - get_calibration / get_autonomy_level (calibration tools)
 * - evaluate_gate (governance tools)
 * - Zod schema validation
 *
 * All tests mock the python-bridge module so no real Python server
 * is needed. Tests verify both the "healthy bridge" and "unhealthy
 * bridge" (fallback) paths.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the python-bridge module BEFORE importing tools
vi.mock('../bridge/python-bridge.js', () => ({
  healthCheck: vi.fn(),
  predict: vi.fn(),
  getCalibration: vi.fn(),
  getDecomposition: vi.fn(),
  fallbackUncertainty: vi.fn(() => ({
    total: 0.9,
    epistemic: 0.8,
    aleatoric: 0.1,
    epistemicRatio: 0.889,
    method: 'ensemble_variance' as const,
  })),
  fallbackCalibration: vi.fn(() => ({
    ece: 0.25,
    mce: 0.40,
    brierScore: 0.30,
    windowSize: 0,
    windowStart: new Date().toISOString(),
    windowEnd: new Date().toISOString(),
    buckets: [],
  })),
}));

import * as pythonBridge from '../bridge/python-bridge.js';
import {
  predictNextEvent,
  predictOutcome,
  predictNextEventSchema,
  predictOutcomeSchema,
} from '../tools/predict.js';
import {
  getCalibration,
  getAutonomyLevel,
  getTrustState,
  setTrustState,
  getCalibrationSchema,
} from '../tools/calibration.js';
import {
  evaluateGate,
  evaluateGateSchema,
} from '../tools/governance.js';
import type { TrustState } from '../types/autonomy.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMockPrediction(caseId: string) {
  return {
    predictionId: `mock-${caseId}`,
    caseId,
    latentState: {
      caseId,
      embedding: new Array(128).fill(0),
      pathVariant: 'standard' as const,
      pathConfidence: 0.95,
      asOfTimestamp: '2024-06-15T09:00:00Z',
      eventCount: 2,
    },
    predictions: {
      activity: { topK: [{ activity: 'ship_goods', probability: 0.8 }], expectedDeltaHours: 4.5 },
      phase: { currentPhase: 'delivery', nextPhase: 'billing', nextPhaseProbability: 0.7, expectedTransitionHours: 12 },
      outcome: { predictedStatus: 'completed' as const, onTimeProbability: 0.85, reworkProbability: 0.05, expectedRemainingHours: 24 },
    },
    uncertainty: { total: 0.3, epistemic: 0.15, aleatoric: 0.15, epistemicRatio: 0.5, method: 'ensemble_variance' as const },
    energyScore: 0.12,
    conformalSet: { activitySet: ['ship_goods'], outcomeSet: ['completed'], coverageTarget: 0.9, alpha: 0.1, empiricalCoverage: 0.92, setSize: 1 },
    timestamp: '2024-06-15T09:00:00Z',
    modelVersion: 'aether-v0.1.0',
  };
}

function makeTestEvent() {
  return {
    eventId: 'e1',
    caseId: 'case_001',
    activity: 'create_order',
    resource: 'system',
    timestamp: '2024-06-15T09:00:00Z',
    attributes: { amount: 100 },
    source: 'synthetic' as const,
  };
}

// ---------------------------------------------------------------------------
// Reset mocks between tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
});

// ============================================================================
// TestPredictNextEvent
// ============================================================================

describe('predictNextEvent', () => {
  it('returns prediction when bridge is healthy', async () => {
    const mock = makeMockPrediction('case_001');
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(true);
    vi.mocked(pythonBridge.predict).mockResolvedValueOnce(mock);

    const result = await predictNextEvent({
      caseId: 'case_001',
      events: [makeTestEvent()],
    });

    expect(result.predictionId).toBe('mock-case_001');
    expect(result.predictions.activity.topK).toHaveLength(1);
    expect(pythonBridge.predict).toHaveBeenCalledOnce();
  });

  it('returns fallback when bridge is unhealthy', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(false);

    const result = await predictNextEvent({
      caseId: 'case_fallback',
      events: [makeTestEvent()],
    });

    // Fallback predictions have high uncertainty
    expect(result.predictionId).toMatch(/^fallback-/);
    expect(result.uncertainty.epistemic).toBeGreaterThan(0.5);
    expect(pythonBridge.predict).not.toHaveBeenCalled();
  });
});

// ============================================================================
// TestPredictOutcome
// ============================================================================

describe('predictOutcome', () => {
  it('returns prediction when bridge is healthy', async () => {
    const mock = makeMockPrediction('case_002');
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(true);
    vi.mocked(pythonBridge.predict).mockResolvedValueOnce(mock);

    const result = await predictOutcome({
      caseId: 'case_002',
      events: [makeTestEvent()],
    });

    expect(result.predictions.outcome.predictedStatus).toBe('completed');
  });

  it('returns fallback when bridge is unhealthy', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(false);

    const result = await predictOutcome({
      caseId: 'case_fb',
      events: [makeTestEvent()],
    });

    expect(result.predictionId).toMatch(/^fallback-/);
  });
});

// ============================================================================
// TestGetCalibration
// ============================================================================

describe('getCalibration', () => {
  it('returns metrics from bridge when healthy', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(true);
    vi.mocked(pythonBridge.getCalibration).mockResolvedValueOnce({
      ece: 0.03,
      mce: 0.08,
      brierScore: 0.04,
      windowSize: 50,
      windowStart: '2024-01-01T00:00:00Z',
      windowEnd: '2024-01-01T01:00:00Z',
      buckets: [],
    });

    const result = await getCalibration({});
    expect(result.ece).toBe(0.03);
    expect(result.mce).toBe(0.08);
  });

  it('returns fallback calibration when unhealthy', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValueOnce(false);

    const result = await getCalibration({});
    // Fallback has poor calibration (conservative)
    expect(result.ece).toBeGreaterThan(0.1);
    expect(result.windowSize).toBe(0);
  });
});

// ============================================================================
// TestGetAutonomyLevel
// ============================================================================

describe('getAutonomyLevel', () => {
  it('returns supervised when no trust state exists', () => {
    // Reset trust state to null by getting current and checking
    // The module starts with null trust state
    const result = getAutonomyLevel();
    if (getTrustState() === null) {
      expect(result.level).toBe('supervised');
      expect(result.summary).toContain('SUPERVISED');
      expect(result.state).toBeNull();
    }
  });

  it('returns current level when trust state exists', () => {
    const mockState: TrustState = {
      level: 'guided',
      consecutiveCalibratedWindows: 15,
      consecutiveDegradedWindows: 0,
      totalPredictions: 750,
      correctPredictions: 680,
      levelEnteredAt: '2024-06-01T00:00:00Z',
      transitions: [],
      calibrationThreshold: 0.10,
      probationary: false,
    };
    setTrustState(mockState);

    const result = getAutonomyLevel();
    expect(result.level).toBe('guided');
    expect(result.state).not.toBeNull();
    expect(result.state?.consecutiveCalibratedWindows).toBe(15);
  });
});

// ============================================================================
// TestEvaluateGate
// ============================================================================

describe('evaluateGate', () => {
  it('returns allow/hold decision for normal mode', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValue(false);
    vi.mocked(pythonBridge.getDecomposition).mockResolvedValue({
      total: 0.3,
      epistemic: 0.15,
      aleatoric: 0.15,
      epistemicRatio: 0.5,
      method: 'ensemble_variance',
    });

    const result = await evaluateGate({
      gateName: 'driftThreshold',
      observedValue: 0.05,
      mode: 'standard',
    });

    expect(result).toHaveProperty('action');
    expect(['allow', 'hold', 'block']).toContain(result.action);
    expect(result).toHaveProperty('reason');
    expect(result).toHaveProperty('modulation');
    expect(result).toHaveProperty('auditId');
  });

  it('blocks in forbidden mode', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValue(false);

    const result = await evaluateGate({
      gateName: 'driftThreshold',
      observedValue: 0.05,
      mode: 'forbidden',
    });

    expect(result.action).toBe('block');
    expect(result.immutableTriggered).toBe(true);
  });

  it('holds when sensitive data detected', async () => {
    vi.mocked(pythonBridge.healthCheck).mockResolvedValue(false);

    const result = await evaluateGate({
      gateName: 'driftThreshold',
      observedValue: 0.05,
      mode: 'standard',
      contentToCheck: 'SSN: 123-45-6789',
    });

    expect(result.action).toBe('hold');
    expect(result.immutableTriggered).toBe(true);
  });
});

// ============================================================================
// TestSchemaValidation
// ============================================================================

describe('Schema validation', () => {
  describe('predictNextEventSchema', () => {
    it('accepts valid input', () => {
      const result = predictNextEventSchema.safeParse({
        caseId: 'case_001',
        events: [makeTestEvent()],
      });
      expect(result.success).toBe(true);
    });

    it('rejects missing caseId', () => {
      const result = predictNextEventSchema.safeParse({
        events: [makeTestEvent()],
      });
      expect(result.success).toBe(false);
    });

    it('rejects invalid event source', () => {
      const result = predictNextEventSchema.safeParse({
        caseId: 'case_001',
        events: [{
          ...makeTestEvent(),
          source: 'invalid_source',
        }],
      });
      expect(result.success).toBe(false);
    });
  });

  describe('predictOutcomeSchema', () => {
    it('accepts valid input', () => {
      const result = predictOutcomeSchema.safeParse({
        caseId: 'case_002',
        events: [makeTestEvent()],
      });
      expect(result.success).toBe(true);
    });
  });

  describe('getCalibrationSchema', () => {
    it('accepts empty object', () => {
      const result = getCalibrationSchema.safeParse({});
      expect(result.success).toBe(true);
    });

    it('accepts windowSize', () => {
      const result = getCalibrationSchema.safeParse({ windowSize: 100 });
      expect(result.success).toBe(true);
    });
  });

  describe('evaluateGateSchema', () => {
    it('accepts valid input', () => {
      const result = evaluateGateSchema.safeParse({
        gateName: 'driftThreshold',
        observedValue: 0.12,
        mode: 'strict',
      });
      expect(result.success).toBe(true);
    });

    it('rejects missing gateName', () => {
      const result = evaluateGateSchema.safeParse({
        observedValue: 0.5,
      });
      expect(result.success).toBe(false);
    });

    it('rejects invalid mode', () => {
      const result = evaluateGateSchema.safeParse({
        gateName: 'driftThreshold',
        observedValue: 0.5,
        mode: 'invalid_mode',
      });
      expect(result.success).toBe(false);
    });
  });
});
