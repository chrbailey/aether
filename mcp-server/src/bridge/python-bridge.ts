/**
 * Python Bridge
 *
 * Calls the Python FastAPI inference server from TypeScript.
 * The Python side runs the ML model (encoder, world model, critic);
 * the TypeScript side handles MCP protocol, governance logic, and
 * state management.
 *
 * Communication is via HTTP to localhost — the Python server
 * runs as a sidecar process.
 */

import type { AetherEvent } from '../types/events.js';
import type {
  PredictionWithUncertainty,
  CalibrationMetrics,
  UncertaintyDecomposition,
} from '../types/predictions.js';

/** Production metrics from the inference server */
export interface ProductionMetrics {
  predictions_total: number;
  predictions_last_hour: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  current_ece: number;
  current_mcc: number;
  calibration_drift_detected: boolean;
  uptime_seconds: number;
  model_version: string;
}

const DEFAULT_PYTHON_URL = 'http://localhost:8712';

export interface PythonBridgeConfig {
  /** URL of the Python inference server */
  url: string;
  /** Request timeout in ms */
  timeoutMs: number;
}

const DEFAULT_CONFIG: PythonBridgeConfig = {
  url: process.env['AETHER_PYTHON_URL'] ?? DEFAULT_PYTHON_URL,
  timeoutMs: 10_000,
};

/**
 * Call the Python inference server to get predictions.
 */
export async function predict(
  events: AetherEvent[],
  config: PythonBridgeConfig = DEFAULT_CONFIG,
): Promise<PredictionWithUncertainty> {
  // Extract caseId from first event (all events in a case share the same caseId)
  const caseId = events[0]?.caseId ?? 'unknown';

  // Transform events to only include fields the Python server expects
  // Python EventInput: activity, resource, timestamp, attributes
  const pythonEvents = events.map(e => ({
    activity: e.activity,
    resource: e.resource,
    timestamp: e.timestamp,
    attributes: e.attributes,
  }));

  const response = await fetchWithTimeout(`${config.url}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ caseId, events: pythonEvents }),
  }, config.timeoutMs);

  if (!response.ok) {
    throw new Error(`Python bridge predict failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<PredictionWithUncertainty>;
}

/**
 * Get current calibration metrics from the Python server.
 */
export async function getCalibration(
  config: PythonBridgeConfig = DEFAULT_CONFIG,
): Promise<CalibrationMetrics> {
  const response = await fetchWithTimeout(`${config.url}/calibration`, {
    method: 'GET',
  }, config.timeoutMs);

  if (!response.ok) {
    throw new Error(`Python bridge calibration failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<CalibrationMetrics>;
}

/**
 * Get uncertainty decomposition for a specific prediction.
 */
export async function getDecomposition(
  predictionId: string,
  config: PythonBridgeConfig = DEFAULT_CONFIG,
): Promise<UncertaintyDecomposition> {
  const response = await fetchWithTimeout(
    `${config.url}/decomposition/${predictionId}`,
    { method: 'GET' },
    config.timeoutMs,
  );

  if (!response.ok) {
    throw new Error(`Python bridge decomposition failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<UncertaintyDecomposition>;
}

/**
 * Check if the Python inference server is healthy.
 */
export async function healthCheck(
  config: PythonBridgeConfig = DEFAULT_CONFIG,
): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(`${config.url}/health`, {
      method: 'GET',
    }, 3_000);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get production metrics from the Python inference server.
 */
export async function getMetrics(
  config: PythonBridgeConfig = DEFAULT_CONFIG,
): Promise<ProductionMetrics> {
  const response = await fetchWithTimeout(`${config.url}/metrics`, {
    method: 'GET',
  }, config.timeoutMs);

  if (!response.ok) {
    throw new Error(`Python bridge metrics failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<ProductionMetrics>;
}

/**
 * Provide fallback metrics when Python server is unavailable.
 */
export function fallbackMetrics(): ProductionMetrics {
  return {
    predictions_total: 0,
    predictions_last_hour: 0,
    avg_latency_ms: 0,
    p50_latency_ms: 0,
    p95_latency_ms: 0,
    p99_latency_ms: 0,
    current_ece: 0.25,
    current_mcc: 0,
    calibration_drift_detected: false,
    uptime_seconds: 0,
    model_version: 'unavailable',
  };
}

/**
 * Provide fallback uncertainty when Python server is unavailable.
 * Uses maximum epistemic uncertainty (conservative — tightens governance).
 */
export function fallbackUncertainty(): UncertaintyDecomposition {
  return {
    total: 0.9,
    epistemic: 0.8,
    aleatoric: 0.1,
    epistemicRatio: 0.889,
    method: 'ensemble_variance',
  };
}

/**
 * Provide fallback calibration when Python server is unavailable.
 * Uses poor calibration (conservative — tightens governance).
 */
export function fallbackCalibration(): CalibrationMetrics {
  return {
    ece: 0.25,
    mce: 0.40,
    brierScore: 0.30,
    windowSize: 0,
    windowStart: new Date().toISOString(),
    windowEnd: new Date().toISOString(),
    buckets: [],
  };
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}
