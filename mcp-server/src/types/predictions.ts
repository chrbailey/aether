/**
 * AETHER Prediction Types
 *
 * Every prediction carries uncertainty decomposed into
 * epistemic (reducible with more data) and aleatoric (irreducible).
 * This decomposition drives the governance modulation.
 */

import type { HierarchicalPrediction, LatentEventState } from './events.js';

/** Uncertainty decomposed into epistemic and aleatoric components */
export interface UncertaintyDecomposition {
  /** Total uncertainty (entropy or variance) */
  total: number;

  /**
   * Epistemic uncertainty — reducible with more data/evidence.
   * HIGH epistemic → should tighten governance (human review helps).
   */
  epistemic: number;

  /**
   * Aleatoric uncertainty — irreducible randomness in the process.
   * HIGH aleatoric → should NOT tighten governance (review won't help).
   */
  aleatoric: number;

  /** Ratio of epistemic to total — drives governance tightening */
  epistemicRatio: number;

  /** Method used for decomposition */
  method: DecompositionMethod;
}

export type DecompositionMethod =
  | 'ensemble_variance'    // Variance across ensemble members
  | 'mc_dropout'           // Monte Carlo dropout
  | 'evidential'           // Evidential deep learning
  | 'verity_ds';           // VERITY Dempster-Shafer

/** A prediction with full uncertainty quantification */
export interface PredictionWithUncertainty {
  /** Unique prediction identifier */
  predictionId: string;

  /** Case this prediction is for */
  caseId: string;

  /** Current latent state used for prediction */
  latentState: LatentEventState;

  /** Hierarchical predictions at all three levels */
  predictions: HierarchicalPrediction;

  /** Uncertainty decomposition */
  uncertainty: UncertaintyDecomposition;

  /** Energy score — lower = more plausible transition */
  energyScore: number;

  /** Prediction set from conformal inference (coverage guarantee) */
  conformalSet: ConformalPredictionSet;

  /** When this prediction was made */
  timestamp: string;

  /** Model version that produced this prediction */
  modelVersion: string;
}

/**
 * Conformal prediction set — provides distribution-free
 * coverage guarantees via Adaptive Conformal Inference (ACI).
 *
 * The set automatically widens when the model is miscalibrated
 * (concept drift) and narrows when calibration improves.
 */
export interface ConformalPredictionSet {
  /** Set of plausible next activities (coverage-guaranteed) */
  activitySet: string[];

  /** Set of plausible outcomes */
  outcomeSet: string[];

  /** Current coverage target (adapts via ACI) */
  coverageTarget: number;

  /** Current α parameter (ACI learning rate adjusts this) */
  alpha: number;

  /** Empirical coverage over recent window */
  empiricalCoverage: number;

  /** Set size — larger set = more uncertain */
  setSize: number;
}

/** Calibration metrics for a window of predictions */
export interface CalibrationMetrics {
  /** Expected Calibration Error — primary metric */
  ece: number;

  /** Maximum Calibration Error — worst bucket */
  mce: number;

  /** Brier score — combines calibration + resolution */
  brierScore: number;

  /** Number of predictions in this window */
  windowSize: number;

  /** Start and end timestamps of the window */
  windowStart: string;
  windowEnd: string;

  /** Per-bucket calibration (for reliability diagrams) */
  buckets: CalibrationBucket[];
}

export interface CalibrationBucket {
  /** Bucket confidence range [low, high) */
  confidenceLow: number;
  confidenceHigh: number;
  /** Average predicted confidence in this bucket */
  avgConfidence: number;
  /** Actual accuracy in this bucket */
  avgAccuracy: number;
  /** Number of predictions in this bucket */
  count: number;
}

/** Prediction outcome — recorded after ground truth is known */
export interface PredictionOutcome {
  predictionId: string;
  /** What actually happened */
  actualActivity?: string;
  actualOutcome?: string;
  /** Was the prediction correct? */
  correct: boolean;
  /** Was the actual value in the conformal set? */
  inConformalSet: boolean;
  /** Timestamp of outcome observation */
  observedAt: string;
}
