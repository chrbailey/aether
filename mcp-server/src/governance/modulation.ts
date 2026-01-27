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

/**
 * Base thresholds — the static values these systems currently use.
 * AETHER makes them dynamic by applying modulation factors.
 */
const BASE_THRESHOLDS = {
  driftThreshold: 0.15,         // PromptSpeak: driftThreshold
  reviewGateAutoPass: 0.92,     // EFC: auto_pass_threshold
  threatActivation: 0.60,       // Unified Belief System
  conformanceDeviation: 0.05,   // SAP: conformance > 5%
  sayDoGap: 0.20,               // SmallCap: Say-Do gap > 20%
  knowledgePromotion: 0.75,     // Knowledge Hooks: Wilson score
} as const;

/**
 * Modulation coefficients — control the sensitivity of each factor.
 * These are the "meta-parameters" of the governance modulation.
 */
const COEFFICIENTS = {
  /** How much the governance mode affects the threshold */
  modeStrength: 0.3,
  /** How much epistemic uncertainty affects the threshold */
  uncertaintyStrength: 0.5,
  /** How much calibration quality affects the threshold */
  calibrationStrength: 0.4,
} as const;

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
 * uncertainty_factor = 1 + (total × epistemic_ratio) × uncertaintyStrength
 *
 * Only epistemic uncertainty contributes — because epistemic uncertainty
 * means "we don't have enough data/evidence" and human review or
 * additional evidence can reduce it. Aleatoric uncertainty is
 * irreducible — the process is inherently random at that point,
 * and no amount of human review will change the outcome.
 *
 * This is the key formal insight of the paper.
 *
 * Returns > 1.0 when epistemic uncertainty is high (tighten governance).
 * Returns ≈ 1.0 when uncertainty is mostly aleatoric (don't tighten).
 */
export function computeUncertaintyFactor(
  uncertainty: UncertaintyDecomposition,
): number {
  const epistemicContribution = uncertainty.total * uncertainty.epistemicRatio;
  return 1 + epistemicContribution * COEFFICIENTS.uncertaintyStrength;
}

/**
 * Compute the calibration factor from recent calibration metrics.
 *
 * calibration_factor = 1 + (1 - calibration_score) × calibrationStrength
 *
 * Where calibration_score = 1 - ECE (so perfect calibration = 1.0).
 *
 * Poor calibration (high ECE) → factor > 1 → tighter governance.
 * Good calibration (low ECE)  → factor ≈ 1 → relaxes toward base.
 *
 * The intuition: if the model's confidence estimates are unreliable,
 * we should trust them less and require more oversight.
 */
export function computeCalibrationFactor(
  calibration: CalibrationMetrics,
): number {
  const calibrationScore = 1 - Math.min(calibration.ece, 1.0);
  return 1 + (1 - calibrationScore) * COEFFICIENTS.calibrationStrength;
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
  //   (but cap at reasonable bounds)

  const tighteningFactor = (m: GovernanceModulation) =>
    m.modeFactor * m.uncertaintyFactor * m.calibrationFactor;

  // Lower-is-stricter: tightening divides
  const driftThreshold = clamp(
    BASE_THRESHOLDS.driftThreshold / tighteningFactor(modulations['driftThreshold']),
    0.02,  // Never below 2% (would trigger on noise)
    0.30,  // Never above 30% (would miss real drift)
  );

  const conformanceDeviation = clamp(
    BASE_THRESHOLDS.conformanceDeviation / tighteningFactor(modulations['conformanceDeviation']),
    0.01,
    0.15,
  );

  const sayDoGap = clamp(
    BASE_THRESHOLDS.sayDoGap / tighteningFactor(modulations['sayDoGap']),
    0.05,
    0.40,
  );

  // Higher-is-stricter: tightening multiplies (but cap at 1.0 for probabilities)
  const reviewGateAutoPass = clamp(
    BASE_THRESHOLDS.reviewGateAutoPass * tighteningFactor(modulations['reviewGateAutoPass']),
    0.80,  // Never below 80% (would auto-pass too much)
    0.99,  // Never above 99% (would block almost everything)
  );

  const threatActivation = clamp(
    BASE_THRESHOLDS.threatActivation * tighteningFactor(modulations['threatActivation']),
    0.40,
    0.90,
  );

  const knowledgePromotion = clamp(
    BASE_THRESHOLDS.knowledgePromotion * tighteningFactor(modulations['knowledgePromotion']),
    0.60,
    0.95,
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
