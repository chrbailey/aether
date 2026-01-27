/**
 * AETHER Governance Tools
 *
 * MCP tools for querying effective thresholds and making
 * governance decisions with full uncertainty awareness.
 */

import { z } from 'zod';
import type { EffectiveThresholds, GateDecision } from '../types/governance.js';
import type { UncertaintyDecomposition, CalibrationMetrics } from '../types/predictions.js';
import { GOVERNANCE_MODES } from '../types/governance.js';
import {
  computeEffectiveThresholds,
  makeGateDecision,
  checkImmutableConstraints,
} from '../governance/index.js';
import * as pythonBridge from '../bridge/python-bridge.js';
import { getTrustState } from './calibration.js';

/** Schema for get_effective_thresholds tool input */
export const getEffectiveThresholdsSchema = z.object({
  mode: z
    .enum(['strict', 'standard', 'flexible', 'forbidden'])
    .optional()
    .default('standard')
    .describe('Governance mode (default: standard)'),
});

/** Schema for evaluate_gate tool input */
export const evaluateGateSchema = z.object({
  gateName: z.string().describe('Name of the gate to evaluate (e.g., "driftThreshold")'),
  observedValue: z.number().describe('The observed metric value'),
  mode: z
    .enum(['strict', 'standard', 'flexible', 'forbidden'])
    .optional()
    .default('standard'),
  contentToCheck: z.string().optional().describe('Optional content to scan for sensitive data'),
});

export type GetEffectiveThresholdsInput = z.infer<typeof getEffectiveThresholdsSchema>;
export type EvaluateGateInput = z.infer<typeof evaluateGateSchema>;

/** Gates where lower threshold = stricter */
const LOWER_IS_STRICTER: Set<string> = new Set([
  'driftThreshold',
  'conformanceDeviation',
  'sayDoGap',
]);

/**
 * Get current effective thresholds for all gates.
 * Incorporates governance mode, uncertainty, and calibration.
 */
export async function getEffectiveThresholds(
  input: GetEffectiveThresholdsInput,
): Promise<EffectiveThresholds> {
  const mode = GOVERNANCE_MODES[input.mode];

  // Get uncertainty and calibration from Python (or fallback)
  let uncertainty: UncertaintyDecomposition;
  let calibration: CalibrationMetrics;

  const healthy = await pythonBridge.healthCheck();
  if (healthy) {
    [uncertainty, calibration] = await Promise.all([
      pythonBridge.getDecomposition('current'),
      pythonBridge.getCalibration(),
    ]);
  } else {
    uncertainty = pythonBridge.fallbackUncertainty();
    calibration = pythonBridge.fallbackCalibration();
  }

  const trustState = getTrustState();
  const autonomyLevel = trustState?.level ?? 'supervised';

  return computeEffectiveThresholds(mode, uncertainty, calibration, autonomyLevel);
}

/**
 * Evaluate a specific gate with the current governance state.
 * Returns allow/hold/block decision with full explanation.
 */
export async function evaluateGate(
  input: EvaluateGateInput,
): Promise<GateDecision> {
  const mode = GOVERNANCE_MODES[input.mode];

  // Check immutable constraints first
  let uncertainty: UncertaintyDecomposition;
  const healthy = await pythonBridge.healthCheck();
  uncertainty = healthy
    ? await pythonBridge.getDecomposition('current')
    : pythonBridge.fallbackUncertainty();

  const immutableCheck = checkImmutableConstraints({
    mode,
    uncertainty,
    contentToCheck: input.contentToCheck,
  });

  if (!immutableCheck.passed) {
    return {
      action: immutableCheck.violatedConstraint === 'forbidden_mode' ? 'block' : 'hold',
      reason: immutableCheck.reason,
      immutableTriggered: true,
      threshold: 0,
      observedValue: input.observedValue,
      modulation: {
        baseThreshold: 0,
        modeFactor: 1,
        uncertaintyFactor: 1,
        calibrationFactor: 1,
        effectiveThreshold: 0,
        mode,
        uncertainty,
        autonomyLevel: getTrustState()?.level ?? 'supervised',
      },
      auditId: `AETHER-IMMUTABLE-${Date.now()}`,
    };
  }

  // Compute effective thresholds
  const thresholds = await getEffectiveThresholds({ mode: input.mode });
  const modulation = thresholds.modulations[input.gateName];

  if (!modulation) {
    return {
      action: 'hold',
      reason: `Unknown gate "${input.gateName}" — holding for review`,
      immutableTriggered: false,
      threshold: 0,
      observedValue: input.observedValue,
      modulation: {
        baseThreshold: 0,
        modeFactor: 1,
        uncertaintyFactor: 1,
        calibrationFactor: 1,
        effectiveThreshold: 0,
        mode,
        uncertainty,
        autonomyLevel: getTrustState()?.level ?? 'supervised',
      },
      auditId: `AETHER-UNKNOWN-GATE-${Date.now()}`,
    };
  }

  const lowerIsStricter = LOWER_IS_STRICTER.has(input.gateName);
  return makeGateDecision(input.gateName, input.observedValue, modulation, lowerIsStricter);
}
