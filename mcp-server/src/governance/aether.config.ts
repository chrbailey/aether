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
// Vocabulary-Aware Minimum Floor (v3) — prevents regression on high-vocab datasets
// ---------------------------------------------------------------------------

/** Configuration for vocabulary-size normalization.
 *
 * High-vocabulary datasets (77+ activities) show confidence overlap between
 * correct and incorrect predictions. The adaptive formula drops to min=0.50,
 * but overconfident errors slip through. This log-scale floor prevents that.
 *
 * Formula: min_threshold = baseFloor + floorIncrement * log(V / referenceVocab) / log(scaleFactor)
 *
 * At referenceVocab (20), min = 0.50 (unchanged from v2)
 * At 80 activities, min = 0.55 (matches static baseline — "do no harm")
 * At 160 activities, min = 0.60 (more conservative for complex taxonomies)
 */
export const VOCAB_NORMALIZATION = {
  /** Base minimum floor (unchanged from v2 for small vocabularies) */
  baseFloor: 0.50,
  /** Increment per log-scale step */
  floorIncrement: 0.05,
  /** Reference vocabulary size where floor = baseFloor */
  referenceVocab: 20,
  /** Log scale factor (vocabulary doubles = one increment step) */
  scaleFactor: 4,
  /** Whether vocabulary normalization is enabled */
  enabled: true,
} as const;

/**
 * Compute the vocabulary-aware minimum threshold floor for reviewGateAutoPass.
 *
 * This implements the "do no harm" principle: when the adaptive formula would
 * perform worse than static (0.55), the floor rises to match static.
 *
 * @param vocabSize - Number of unique activities in the dataset
 * @returns The minimum threshold floor (>= 0.50)
 */
export function computeVocabAwareMinFloor(vocabSize: number): number {
  if (!VOCAB_NORMALIZATION.enabled || vocabSize <= VOCAB_NORMALIZATION.referenceVocab) {
    return VOCAB_NORMALIZATION.baseFloor;
  }

  const logRatio = Math.log(vocabSize / VOCAB_NORMALIZATION.referenceVocab);
  const logScale = Math.log(VOCAB_NORMALIZATION.scaleFactor);
  const adjustment = VOCAB_NORMALIZATION.floorIncrement * (logRatio / logScale);

  return Math.min(
    VOCAB_NORMALIZATION.baseFloor + adjustment,
    CLAMP_BOUNDS.reviewGateAutoPass.max - 0.05, // Never exceed max minus small buffer
  );
}

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
