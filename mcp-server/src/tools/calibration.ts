/**
 * AETHER Calibration Tools
 *
 * MCP tools for querying calibration state and autonomy level.
 */

import { z } from 'zod';
import type { CalibrationMetrics } from '../types/predictions.js';
import type { TrustState } from '../types/autonomy.js';
import * as pythonBridge from '../bridge/python-bridge.js';
import { summarizeTrustState } from '../governance/autonomy-controller.js';

/** Schema for get_calibration tool input */
export const getCalibrationSchema = z.object({
  windowSize: z
    .number()
    .optional()
    .describe('Number of recent predictions to include (default: 50)'),
});

/** Schema for get_autonomy_level tool input */
export const getAutonomyLevelSchema = z.object({});

export type GetCalibrationInput = z.infer<typeof getCalibrationSchema>;

/** In-memory trust state (production would persist to Pinecone) */
let trustState: TrustState | null = null;

export function getTrustState(): TrustState | null {
  return trustState;
}

export function setTrustState(state: TrustState): void {
  trustState = state;
}

/**
 * Get current calibration metrics from the inference server.
 */
export async function getCalibration(
  _input: GetCalibrationInput,
): Promise<CalibrationMetrics> {
  const healthy = await pythonBridge.healthCheck();

  if (!healthy) {
    return pythonBridge.fallbackCalibration();
  }

  return pythonBridge.getCalibration();
}

/**
 * Get current autonomy level and trust state summary.
 */
export function getAutonomyLevel(): {
  level: string;
  summary: string;
  state: TrustState | null;
} {
  const state = getTrustState();

  if (!state) {
    return {
      level: 'supervised',
      summary: 'No trust state initialized — defaulting to SUPERVISED',
      state: null,
    };
  }

  return {
    level: state.level,
    summary: summarizeTrustState(state),
    state,
  };
}
