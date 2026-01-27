/**
 * Pinecone Bridge
 *
 * Persists AETHER state (trust transitions, calibration history,
 * prediction outcomes) to Pinecone for cross-session continuity.
 *
 * Uses the claude-knowledge-base index with AETHER-prefixed IDs.
 */

import type { TrustState, TrustTransition } from '../types/autonomy.js';
import type { CalibrationMetrics } from '../types/predictions.js';

/** Record format for Pinecone upsert */
export interface AetherPineconeRecord {
  id: string;
  content: string;
  category: string;
  timestamp: string;
  [key: string]: string | number | boolean;
}

/**
 * Format a trust transition as a Pinecone record.
 */
export function trustTransitionToRecord(
  transition: TrustTransition,
): AetherPineconeRecord {
  return {
    id: `aether-trust-${Date.now()}`,
    content: `AETHER trust transition: ${transition.from} → ${transition.to} (${transition.direction}). Trigger: ${transition.trigger}. ECE at transition: ${transition.calibrationAtTransition.toFixed(4)}`,
    category: 'aether_trust',
    timestamp: transition.timestamp,
    from_level: transition.from,
    to_level: transition.to,
    direction: transition.direction,
    ece: transition.calibrationAtTransition,
  };
}

/**
 * Format calibration metrics as a Pinecone record.
 */
export function calibrationToRecord(
  metrics: CalibrationMetrics,
): AetherPineconeRecord {
  return {
    id: `aether-calibration-${Date.now()}`,
    content: `AETHER calibration window: ECE=${metrics.ece.toFixed(4)}, MCE=${metrics.mce.toFixed(4)}, Brier=${metrics.brierScore.toFixed(4)}, n=${metrics.windowSize}`,
    category: 'aether_calibration',
    timestamp: metrics.windowEnd,
    ece: metrics.ece,
    mce: metrics.mce,
    brier_score: metrics.brierScore,
    window_size: metrics.windowSize,
  };
}

/**
 * Format a trust state snapshot for persistence.
 */
export function trustStateToRecord(
  state: TrustState,
): AetherPineconeRecord {
  return {
    id: `aether-state-${Date.now()}`,
    content: `AETHER trust state: level=${state.level}, calibrated_windows=${state.consecutiveCalibratedWindows}, degraded_windows=${state.consecutiveDegradedWindows}, total_transitions=${state.transitions.length}, probationary=${state.probationary}`,
    category: 'aether_state',
    timestamp: new Date().toISOString(),
    level: state.level,
    calibrated_windows: state.consecutiveCalibratedWindows,
    degraded_windows: state.consecutiveDegradedWindows,
    total_predictions: state.totalPredictions,
    correct_predictions: state.correctPredictions,
    probationary: state.probationary,
  };
}
