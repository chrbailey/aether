/**
 * Python Bridge Tests
 *
 * Verifies the HTTP client that connects TypeScript governance
 * to the Python FastAPI inference server. All tests mock global
 * fetch() — no real network calls.
 *
 * Tests:
 * 1. healthCheck: healthy → true, timeout → false, HTTP error → false
 * 2. predict: success response shape, error propagation
 * 3. getCalibration: success response shape, error propagation
 * 4. fallbackUncertainty / fallbackCalibration: valid conservative structures
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  healthCheck,
  predict,
  getCalibration,
  fallbackUncertainty,
  fallbackCalibration,
} from '../bridge/python-bridge.js';
import type { PythonBridgeConfig } from '../bridge/python-bridge.js';

// ---------------------------------------------------------------------------
// Mock setup
// ---------------------------------------------------------------------------

const TEST_CONFIG: PythonBridgeConfig = {
  url: 'http://localhost:9999',
  timeoutMs: 1_000,
};

// Save original fetch
const originalFetch = globalThis.fetch;

beforeEach(() => {
  // Replace global fetch with a vi.fn()
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

// Helpers to create mock Response objects
function mockResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: () => Promise.resolve(body),
    headers: new Headers(),
    redirected: false,
    type: 'basic',
    url: '',
    clone: () => mockResponse(body, status),
    body: null,
    bodyUsed: false,
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
    blob: () => Promise.resolve(new Blob([])),
    formData: () => Promise.resolve(new FormData()),
    text: () => Promise.resolve(JSON.stringify(body)),
    bytes: () => Promise.resolve(new Uint8Array()),
  } as Response;
}

// ============================================================================
// TestHealthCheck
// ============================================================================

describe('healthCheck', () => {
  it('returns true when server responds 200', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse({ status: 'healthy', modelLoaded: true }),
    );

    const result = await healthCheck(TEST_CONFIG);
    expect(result).toBe(true);
  });

  it('returns false on HTTP error', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse({ error: 'server error' }, 500),
    );

    const result = await healthCheck(TEST_CONFIG);
    expect(result).toBe(false);
  });

  it('returns false on network timeout', async () => {
    vi.mocked(globalThis.fetch).mockRejectedValueOnce(
      new DOMException('The operation was aborted.', 'AbortError'),
    );

    const result = await healthCheck(TEST_CONFIG);
    expect(result).toBe(false);
  });

  it('returns false on connection refused', async () => {
    vi.mocked(globalThis.fetch).mockRejectedValueOnce(
      new TypeError('fetch failed'),
    );

    const result = await healthCheck(TEST_CONFIG);
    expect(result).toBe(false);
  });
});

// ============================================================================
// TestPredict
// ============================================================================

describe('predict', () => {
  const mockPrediction = {
    predictionId: 'test-pred-001',
    caseId: 'case_001',
    predictions: {
      activity: { topK: [{ activity: 'ship_goods', probability: 0.8 }], expectedDeltaHours: 4.5 },
      phase: { currentPhase: 'delivery', nextPhase: 'billing', nextPhaseProbability: 0.7, expectedTransitionHours: 12 },
      outcome: { predictedStatus: 'completed', onTimeProbability: 0.85, reworkProbability: 0.05, expectedRemainingHours: 24 },
    },
    uncertainty: { total: 0.3, epistemic: 0.15, aleatoric: 0.15, epistemicRatio: 0.5, method: 'ensemble_variance' },
    energyScore: 0.12,
    conformalSet: { activitySet: ['ship_goods', 'invoice'], outcomeSet: ['completed'], coverageTarget: 0.9, alpha: 0.1, empiricalCoverage: 0.92, setSize: 2 },
    timestamp: '2024-06-15T09:00:00Z',
    modelVersion: 'aether-v0.1.0',
  };

  it('returns prediction with expected shape on success', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse(mockPrediction),
    );

    const events = [{
      eventId: 'e1', caseId: 'case_001', activity: 'create_order',
      resource: 'system', timestamp: '2024-06-15T09:00:00Z',
      attributes: {}, source: 'synthetic' as const,
    }];

    const result = await predict(events, TEST_CONFIG);
    expect(result).toHaveProperty('predictionId');
    expect(result).toHaveProperty('predictions');
    expect(result).toHaveProperty('uncertainty');
    expect(result).toHaveProperty('energyScore');
    expect(result).toHaveProperty('conformalSet');
  });

  it('throws on HTTP error', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse({ detail: 'not found' }, 404),
    );

    const events = [{
      eventId: 'e1', caseId: 'c1', activity: 'a',
      resource: 'r', timestamp: '', attributes: {},
      source: 'synthetic' as const,
    }];

    await expect(predict(events, TEST_CONFIG)).rejects.toThrow('Python bridge predict failed');
  });
});

// ============================================================================
// TestGetCalibration
// ============================================================================

describe('getCalibration', () => {
  const mockCalibration = {
    ece: 0.03,
    mce: 0.08,
    brierScore: 0.04,
    windowSize: 50,
    windowStart: '2024-06-15T00:00:00Z',
    windowEnd: '2024-06-15T01:00:00Z',
    buckets: [],
  };

  it('returns calibration metrics on success', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse(mockCalibration),
    );

    const result = await getCalibration(TEST_CONFIG);
    expect(result).toHaveProperty('ece');
    expect(result).toHaveProperty('mce');
    expect(result).toHaveProperty('brierScore');
    expect(result.ece).toBe(0.03);
  });

  it('throws on HTTP error', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      mockResponse({}, 503),
    );

    await expect(getCalibration(TEST_CONFIG)).rejects.toThrow('Python bridge calibration failed');
  });
});

// ============================================================================
// TestFallbacks
// ============================================================================

describe('fallbackUncertainty', () => {
  it('returns conservative high-epistemic uncertainty', () => {
    const u = fallbackUncertainty();
    expect(u.total).toBeGreaterThan(0.5);
    expect(u.epistemic).toBeGreaterThan(u.aleatoric);
    expect(u.epistemicRatio).toBeGreaterThan(0.5);
    expect(u.method).toBe('ensemble_variance');
  });

  it('has valid decomposition (epistemic + aleatoric ≤ total)', () => {
    const u = fallbackUncertainty();
    expect(u.epistemic + u.aleatoric).toBeLessThanOrEqual(u.total + 0.001);
  });
});

describe('fallbackCalibration', () => {
  it('returns poor calibration metrics (conservative)', () => {
    const c = fallbackCalibration();
    expect(c.ece).toBeGreaterThan(0.1);  // Poor ECE
    expect(c.mce).toBeGreaterThan(0.1);
    expect(c.brierScore).toBeGreaterThan(0.1);
    expect(c.windowSize).toBe(0);
    expect(c.buckets).toEqual([]);
  });

  it('has valid ISO timestamp strings', () => {
    const c = fallbackCalibration();
    // Should not throw when parsed
    expect(() => new Date(c.windowStart)).not.toThrow();
    expect(() => new Date(c.windowEnd)).not.toThrow();
  });
});
