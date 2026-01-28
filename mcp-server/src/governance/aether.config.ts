/**
 * AETHER Governance Configuration
 *
 * All tunable parameters for the governance modulation system.
 * Override these to customize AETHER for your domain.
 *
 * The governance formula:
 *   effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
 *
 * Each section below controls one aspect of the computation.
 */

// ---------------------------------------------------------------------------
// Base Thresholds — the static values before modulation
// ---------------------------------------------------------------------------

/** Base threshold values for each gate.
 *
 * These are the "unmodulated" thresholds. AETHER makes them dynamic
 * by applying mode, uncertainty, and calibration factors.
 */
export const BASE_THRESHOLDS = {
  /** PromptSpeak concept drift detection */
  driftThreshold: 0.15,
  /** EFC review gate auto-pass confidence */
  reviewGateAutoPass: 0.55,
  /** Unified Belief System threat activation */
  threatActivation: 0.60,
  /** SAP process conformance deviation */
  conformanceDeviation: 0.05,
  /** SmallCap Say-Do gap */
  sayDoGap: 0.20,
  /** Knowledge hooks Wilson score promotion */
  knowledgePromotion: 0.75,
} as const;

// ---------------------------------------------------------------------------
// Modulation Coefficients — sensitivity of each factor
// ---------------------------------------------------------------------------

/** How strongly each factor affects the effective threshold.
 *
 * Higher values = more sensitive to that factor.
 * Setting a coefficient to 0 disables that factor entirely.
 */
export const COEFFICIENTS = {
  /** How much the governance mode affects thresholds (0 = mode ignored) */
  modeStrength: 0.3,
  /** How much epistemic uncertainty modulates governance (0 = uncertainty ignored) */
  uncertaintyStrength: 0.5,
  /** How much calibration quality modulates governance (0 = calibration ignored) */
  calibrationStrength: 0.4,
  /** ECE below this target relaxes governance; above tightens.
   *  Typical well-calibrated models achieve ECE < 0.05. */
  targetECE: 0.05,
  /** Epistemic ratio baseline. Below → relax (model knows what it knows).
   *  Above → tighten (model uncertain in reducible way). */
  baselineEpistemicRatio: 0.3,
} as const;

// ---------------------------------------------------------------------------
// Clamp Bounds — safety limits for each threshold
// ---------------------------------------------------------------------------

/** Minimum and maximum effective values for each gate.
 *
 * Even with extreme modulation, thresholds stay within these bounds.
 * This prevents pathological behavior (e.g., threshold so low it
 * triggers on noise, or so high it never fires).
 */
export const CLAMP_BOUNDS = {
  driftThreshold:       { min: 0.02, max: 0.30 },
  reviewGateAutoPass:   { min: 0.50, max: 0.99 },
  threatActivation:     { min: 0.40, max: 0.90 },
  conformanceDeviation: { min: 0.01, max: 0.15 },
  sayDoGap:             { min: 0.05, max: 0.40 },
  knowledgePromotion:   { min: 0.60, max: 0.95 },
} as const;

// ---------------------------------------------------------------------------
// Gate Direction — which way "tighter" goes
// ---------------------------------------------------------------------------

/** Whether a gate uses lower-is-stricter or higher-is-stricter semantics.
 *
 * - lower-is-stricter: tightening LOWERS the threshold (e.g., drift detection)
 * - higher-is-stricter: tightening RAISES the threshold (e.g., auto-pass confidence)
 */
export const GATE_DIRECTION: Record<string, 'lower' | 'higher'> = {
  driftThreshold:       'lower',
  reviewGateAutoPass:   'higher',
  threatActivation:     'higher',
  conformanceDeviation: 'lower',
  sayDoGap:             'lower',
  knowledgePromotion:   'higher',
} as const;
