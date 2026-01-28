/**
 * AETHER Governance Layer
 *
 * Unified entry point for the adaptive governance system.
 * Composes modulation, immutable constraints, and autonomy
 * into a single coherent decision pipeline.
 */

export {
  computeModeFactor,
  computeUncertaintyFactor,
  computeCalibrationFactor,
  computeModulation,
  computeEffectiveThresholds,
  makeGateDecision,
} from './modulation.js';

export {
  checkImmutableConstraints,
  containsSensitiveData,
} from './immutable.js';
export type { ImmutableCheckResult } from './immutable.js';

export {
  createInitialTrustState,
  processCalibrationWindow,
  summarizeTrustState,
  isActionPermitted,
} from './autonomy-controller.js';

export {
  BASE_THRESHOLDS,
  COEFFICIENTS,
  CLAMP_BOUNDS,
  GATE_DIRECTION,
} from './aether.config.js';
