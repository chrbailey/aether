/**
 * AETHER Governance Types
 *
 * The core formal contribution: how symbolic governance modes
 * compose with continuous uncertainty to produce effective thresholds.
 *
 * Key insight: aleatoric uncertainty should NOT tighten governance
 * (it's irreducible — more human review won't help). Epistemic
 * uncertainty SHOULD tighten governance (more evidence helps).
 */

import type { UncertaintyDecomposition } from './predictions.js';
import type { AutonomyLevel } from './autonomy.js';

/**
 * Symbolic governance mode — maps to PromptSpeak modes.
 * Each mode has a strictness level (0-3) that affects
 * the mode_factor in threshold computation.
 */
export interface GovernanceMode {
  /** Mode name matching PromptSpeak conventions */
  name: 'strict' | 'standard' | 'flexible' | 'forbidden';

  /** Strictness level: 0=flexible, 1=standard, 2=strict, 3=forbidden */
  strictness: 0 | 1 | 2 | 3;

  /** PromptSpeak symbol (e.g., "⊕" for strict) */
  symbol: string;
}

/** Pre-defined governance modes */
export const GOVERNANCE_MODES: Record<string, GovernanceMode> = {
  forbidden:  { name: 'forbidden', strictness: 3, symbol: '⊗' },
  strict:     { name: 'strict',    strictness: 2, symbol: '⊕' },
  standard:   { name: 'standard',  strictness: 1, symbol: '◈' },
  flexible:   { name: 'flexible',  strictness: 0, symbol: '◇' },
};

/**
 * The compositional governance modulation — THE NOVEL PIECE.
 *
 * effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
 *
 * Each factor is independently computed and their composition
 * produces the final threshold that governs gate decisions.
 */
export interface GovernanceModulation {
  /** Base threshold from static configuration */
  baseThreshold: number;

  /**
   * Mode factor: derived from symbolic governance mode.
   * mode_factor = 1 + (1 - s/3) × 0.3
   * strict (s=2) → 1.1, standard (s=1) → 1.2, flexible (s=0) → 1.3
   */
  modeFactor: number;

  /**
   * Uncertainty factor: derived from uncertainty decomposition.
   * uncertainty_factor = 1 + (total × epistemic_ratio) × 0.5
   * Only epistemic uncertainty tightens — aleatoric is irreducible.
   */
  uncertaintyFactor: number;

  /**
   * Calibration factor: derived from recent calibration quality.
   * calibration_factor = 1 + (1 - calibration_score) × 0.4
   * Poor calibration tightens; good calibration approaches 1.0.
   */
  calibrationFactor: number;

  /** Final computed threshold */
  effectiveThreshold: number;

  /** The governance mode used */
  mode: GovernanceMode;

  /** The uncertainty decomposition used */
  uncertainty: UncertaintyDecomposition;

  /** Current autonomy level */
  autonomyLevel: AutonomyLevel;
}

/**
 * Effective thresholds for all configurable gates.
 * These replace the static thresholds across all systems.
 */
export interface EffectiveThresholds {
  /** PromptSpeak drift detection threshold (replaces static 0.15) */
  driftThreshold: number;

  /** EFC review gate auto-pass threshold (replaces static 0.92) */
  reviewGateAutoPass: number;

  /** Belief system threat level activation threshold */
  threatActivation: number;

  /** SAP conformance deviation threshold (replaces static 0.05) */
  conformanceDeviation: number;

  /** SmallCap say-do gap threshold (replaces static 0.20) */
  sayDoGap: number;

  /** Knowledge hooks promotion threshold (replaces Wilson > 0.75) */
  knowledgePromotion: number;

  /** The modulation computation for each threshold */
  modulations: Record<string, GovernanceModulation>;

  /** Timestamp of computation */
  computedAt: string;
}

/**
 * Immutable safety constraints — the Intrinsic Cost module.
 * These NEVER adapt, regardless of calibration or trust level.
 */
export interface ImmutableConstraints {
  /** PromptSpeak forbidden mode always blocks */
  forbiddenModeBlocks: true;

  /** Sensitive data patterns always trigger hold */
  sensitiveDataHold: true;

  /** D-S conflict coefficient threshold for mandatory review */
  dsConflictThreshold: 0.7;

  /** Circuit breaker minimum floor (even in flexible mode) */
  circuitBreakerFloor: 3;

  /** Maximum allowed uncertainty before mandatory hold */
  maxUncertaintyForAutoPass: 0.95;
}

/** The immutable constraints — hardcoded, never changes */
export const IMMUTABLE_CONSTRAINTS: ImmutableConstraints = {
  forbiddenModeBlocks: true,
  sensitiveDataHold: true,
  dsConflictThreshold: 0.7,
  circuitBreakerFloor: 3,
  maxUncertaintyForAutoPass: 0.95,
};

/**
 * Gate decision — the output of the governance layer.
 * Similar to PromptSpeak's InterceptorDecision but
 * enriched with uncertainty-aware reasoning.
 */
export interface GateDecision {
  /** Allow, hold, or block */
  action: 'allow' | 'hold' | 'block';

  /** Why this decision was made */
  reason: string;

  /** Was this triggered by an immutable constraint? */
  immutableTriggered: boolean;

  /** The effective threshold used */
  threshold: number;

  /** The observed value compared against the threshold */
  observedValue: number;

  /** Full modulation details */
  modulation: GovernanceModulation;

  /** Audit identifier for traceability */
  auditId: string;
}
