/**
 * AETHER Governance Modulation — THE NOVEL PIECE
 *
 * Composes symbolic governance modes with continuous uncertainty
 * to produce effective thresholds for all gate decisions.
 *
 * The formal contribution:
 *   effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
 *
 * Key insight: Aleatoric uncertainty (irreducible randomness) should NOT
 * tighten governance — more human review won't reduce it. Epistemic
 * uncertainty (reducible with more data) SHOULD tighten governance —
 * human review or additional evidence genuinely helps.
 *
 * This decomposition bridges neurosymbolic governance (discrete modes)
 * with Bayesian uncertainty quantification (continuous measures),
 * producing adaptive thresholds that respond to what the model
 * actually knows vs. what is fundamentally unknowable.
 */

import type {
  GovernanceMode,
  GovernanceModulation,
  EffectiveThresholds,
  GateDecision,
} from '../types/governance.js';
import type { UncertaintyDecomposition } from '../types/predictions.js';
import type { CalibrationMetrics } from '../types/predictions.js';
import type { AutonomyLevel } from '../types/autonomy.js';
import { GOVERNANCE_MODES, IMMUTABLE_CONSTRAINTS } from '../types/governance.js';
import { AUTONOMY_RANK } from '../types/autonomy.js';
import {
  BASE_THRESHOLDS,
  COEFFICIENTS,
  CLAMP_BOUNDS,
} from './aether.config.js';

/**
 * Compute the mode factor from a symbolic governance mode.
 *
 * The mode factor represents how much to TIGHTEN governance.
 * Higher = more tightening.
 *
 * - forbidden (s=3) → 1.3 (maximum tightening)
 * - strict (s=2)    → 1.2 (strong tightening)
 * - standard (s=1)  → 1.1 (slight tightening)
 * - flexible (s=0)  → 1.0 (no tightening — baseline)
 *
 * Combined with the uncertainty and calibration factors, this produces
 * a single "tightening multiplier" that is uniformly applied:
 * - For lower-is-stricter gates: divide base by tightening (→ lower threshold)
 * - For higher-is-stricter gates: multiply base by tightening (→ higher threshold)
 */
export function computeModeFactor(mode: GovernanceMode): number {
  // s=0 (flexible) → 1.0, s=1 (standard) → 1.1, s=2 (strict) → 1.2, s=3 (forbidden) → 1.3
  return 1 + (mode.strictness / 3) * COEFFICIENTS.modeStrength;
}

/**
 * Compute the uncertainty factor from decomposed uncertainty.
 *
 * v2: Bidirectional around a baseline epistemic ratio.
 *
 *   uncertainty_factor = 1 + (epistemic_ratio − baseline) × uncertaintyStrength
 *
 * Only epistemic uncertainty contributes — because epistemic uncertainty
 * means "we don't have enough data/evidence" and human review or
 * additional evidence can reduce it. Aleatoric uncertainty is
 * irreducible — the process is inherently random at that point,
 * and no amount of human review will change the outcome.
 *
 * This is the key formal insight of the paper.
 *
 * Returns > 1.0 when epistemic ratio exceeds baseline (tighten — human review helps).
 * Returns < 1.0 when epistemic ratio is below baseline (relax — model is confident
 *   AND its uncertainty is mostly irreducible, so review adds no value).
 * Returns = 1.0 at the baseline (neutral).
 */
export function computeUncertaintyFactor(
  uncertainty: UncertaintyDecomposition,
): number {
  const deviation = uncertainty.epistemicRatio - COEFFICIENTS.baselineEpistemicRatio;
  return 1 + deviation * COEFFICIENTS.uncertaintyStrength;
}

/**
 * Compute the calibration factor from recent calibration metrics.
 *
 * v2: Bidirectional around a target ECE.
 *
 *   calibration_factor = 1 + (ece − target_ece) × calibrationStrength
 *
 * Poor calibration (ECE > target) → factor > 1 → tighter governance.
 * Good calibration (ECE < target) → factor < 1 → earned relaxation.
 * At target ECE                   → factor = 1 → neutral.
 *
 * The intuition: a well-calibrated model has EARNED lower oversight.
 * A poorly calibrated model needs MORE oversight because its
 * confidence estimates can't be trusted.
 */
export function computeCalibrationFactor(
  calibration: CalibrationMetrics,
): number {
  const ece = Math.min(calibration.ece, 1.0);
  return 1 + (ece - COEFFICIENTS.targetECE) * COEFFICIENTS.calibrationStrength;
}

/**
 * Compute the full governance modulation for a single threshold.
 *
 * This is the compositional function:
 *   effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
 *
 * For "lower-is-stricter" thresholds (drift, conformance, sayDoGap):
 *   We DIVIDE by the factors (tightening = lower threshold)
 *
 * For "higher-is-stricter" thresholds (autoPass, promotion):
 *   We MULTIPLY by the factors (tightening = higher threshold)
 */
export function computeModulation(
  baseThreshold: number,
  mode: GovernanceMode,
  uncertainty: UncertaintyDecomposition,
  calibration: CalibrationMetrics,
  autonomyLevel: AutonomyLevel,
): GovernanceModulation {
  const modeFactor = computeModeFactor(mode);
  const uncertaintyFactor = computeUncertaintyFactor(uncertainty);
  const calibrationFactor = computeCalibrationFactor(calibration);

  // Combined tightening factor
  const combinedFactor = modeFactor * uncertaintyFactor * calibrationFactor;

  // The effective threshold incorporates the combined factor.
  // Direction is handled by the caller (computeEffectiveThresholds).
  const effectiveThreshold = baseThreshold * combinedFactor;

  return {
    baseThreshold,
    modeFactor,
    uncertaintyFactor,
    calibrationFactor,
    effectiveThreshold,
    mode,
    uncertainty,
    autonomyLevel,
  };
}

/**
 * Compute effective thresholds for ALL configurable gates.
 *
 * This is the main entry point — takes the current system state
 * and produces adaptive thresholds for every gate across
 * PromptSpeak, EFC, SAP, SmallCap, and Knowledge Hooks.
 */
export function computeEffectiveThresholds(
  mode: GovernanceMode,
  uncertainty: UncertaintyDecomposition,
  calibration: CalibrationMetrics,
  autonomyLevel: AutonomyLevel,
): EffectiveThresholds {
  const modulations: Record<string, GovernanceModulation> = {};

  // Compute modulation for each threshold
  for (const key of Object.keys(BASE_THRESHOLDS) as Array<keyof typeof BASE_THRESHOLDS>) {
    modulations[key] = computeModulation(
      BASE_THRESHOLDS[key],
      mode,
      uncertainty,
      calibration,
      autonomyLevel,
    );
  }

  // Apply direction-aware thresholds:
  //
  // "Lower-is-stricter" thresholds: when we tighten, we LOWER the value.
  //   drift, conformance, sayDoGap → divide by combined factor
  //
  // "Higher-is-stricter" thresholds: when we tighten, we RAISE the value.
  //   autoPass, threatActivation, promotion → multiply by combined factor
  //   (but cap at reasonable bounds from config)

  const tighteningFactor = (m: GovernanceModulation) =>
    m.modeFactor * m.uncertaintyFactor * m.calibrationFactor;

  // Lower-is-stricter: tightening divides
  const driftThreshold = clamp(
    BASE_THRESHOLDS.driftThreshold / tighteningFactor(modulations['driftThreshold']),
    CLAMP_BOUNDS.driftThreshold.min,
    CLAMP_BOUNDS.driftThreshold.max,
  );

  const conformanceDeviation = clamp(
    BASE_THRESHOLDS.conformanceDeviation / tighteningFactor(modulations['conformanceDeviation']),
    CLAMP_BOUNDS.conformanceDeviation.min,
    CLAMP_BOUNDS.conformanceDeviation.max,
  );

  const sayDoGap = clamp(
    BASE_THRESHOLDS.sayDoGap / tighteningFactor(modulations['sayDoGap']),
    CLAMP_BOUNDS.sayDoGap.min,
    CLAMP_BOUNDS.sayDoGap.max,
  );

  // Higher-is-stricter: tightening multiplies (capped by config bounds)
  const reviewGateAutoPass = clamp(
    BASE_THRESHOLDS.reviewGateAutoPass * tighteningFactor(modulations['reviewGateAutoPass']),
    CLAMP_BOUNDS.reviewGateAutoPass.min,
    CLAMP_BOUNDS.reviewGateAutoPass.max,
  );

  const threatActivation = clamp(
    BASE_THRESHOLDS.threatActivation * tighteningFactor(modulations['threatActivation']),
    CLAMP_BOUNDS.threatActivation.min,
    CLAMP_BOUNDS.threatActivation.max,
  );

  const knowledgePromotion = clamp(
    BASE_THRESHOLDS.knowledgePromotion * tighteningFactor(modulations['knowledgePromotion']),
    CLAMP_BOUNDS.knowledgePromotion.min,
    CLAMP_BOUNDS.knowledgePromotion.max,
  );

  // Update modulations with direction-corrected effective thresholds
  modulations['driftThreshold'].effectiveThreshold = driftThreshold;
  modulations['reviewGateAutoPass'].effectiveThreshold = reviewGateAutoPass;
  modulations['threatActivation'].effectiveThreshold = threatActivation;
  modulations['conformanceDeviation'].effectiveThreshold = conformanceDeviation;
  modulations['sayDoGap'].effectiveThreshold = sayDoGap;
  modulations['knowledgePromotion'].effectiveThreshold = knowledgePromotion;

  return {
    driftThreshold,
    reviewGateAutoPass,
    threatActivation,
    conformanceDeviation,
    sayDoGap,
    knowledgePromotion,
    modulations,
    computedAt: new Date().toISOString(),
  };
}

/**
 * Make a gate decision by comparing an observed value against
 * the effective threshold, respecting immutable constraints.
 */
export function makeGateDecision(
  gateName: string,
  observedValue: number,
  modulation: GovernanceModulation,
  lowerIsStricter: boolean,
): GateDecision {
  const threshold = modulation.effectiveThreshold;

  // Check immutable constraints first
  if (modulation.mode.name === 'forbidden') {
    return {
      action: 'block',
      reason: `Forbidden mode active — immutable block on ${gateName}`,
      immutableTriggered: true,
      threshold,
      observedValue,
      modulation,
      auditId: generateAuditId(),
    };
  }

  if (modulation.uncertainty.total > IMMUTABLE_CONSTRAINTS.maxUncertaintyForAutoPass) {
    return {
      action: 'hold',
      reason: `Total uncertainty ${modulation.uncertainty.total.toFixed(3)} exceeds immutable max ${IMMUTABLE_CONSTRAINTS.maxUncertaintyForAutoPass}`,
      immutableTriggered: true,
      threshold,
      observedValue,
      modulation,
      auditId: generateAuditId(),
    };
  }

  // Direction-aware comparison
  const exceeds = lowerIsStricter
    ? observedValue > threshold   // e.g., drift 0.20 > threshold 0.12 → hold
    : observedValue < threshold;  // e.g., confidence 0.85 < threshold 0.94 → hold

  if (exceeds) {
    return {
      action: 'hold',
      reason: `${gateName}: observed ${observedValue.toFixed(3)} ${lowerIsStricter ? '>' : '<'} threshold ${threshold.toFixed(3)} (base=${modulation.baseThreshold}, mode=${modulation.mode.name}, epistemic_ratio=${modulation.uncertainty.epistemicRatio.toFixed(2)})`,
      immutableTriggered: false,
      threshold,
      observedValue,
      modulation,
      auditId: generateAuditId(),
    };
  }

  return {
    action: 'allow',
    reason: `${gateName}: observed ${observedValue.toFixed(3)} within threshold ${threshold.toFixed(3)}`,
    immutableTriggered: false,
    threshold,
    observedValue,
    modulation,
    auditId: generateAuditId(),
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

let auditCounter = 0;
function generateAuditId(): string {
  auditCounter++;
  return `AETHER-${Date.now()}-${auditCounter.toString().padStart(6, '0')}`;
}
