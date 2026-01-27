/**
 * AETHER Autonomy Controller
 *
 * Implements progressive autonomy with asymmetric trust dynamics:
 * - Trust ASCENT is slow: requires sustained calibration over many windows
 * - Trust DESCENT is fast: a single critical failure triggers immediate demotion
 *
 * This solves the "governance descent problem" — a novel control-theoretic
 * treatment. The asymmetry is deliberate: it's much harder to earn trust
 * than to lose it, mirroring how real-world trust operates.
 *
 * State machine:
 *   SUPERVISED ──[10 calibrated]──→ GUIDED
 *   GUIDED     ──[20 calibrated]──→ COLLABORATIVE
 *   COLLABORATIVE ──[50 calibrated]──→ AUTONOMOUS
 *
 *   AUTONOMOUS     ──[1 critical miss]──→ COLLABORATIVE (immediate)
 *   Any level      ──[3 degraded windows]──→ one level down
 *   Any level      ──[immutable violation]──→ SUPERVISED (immediate)
 */

import type {
  AutonomyLevel,
  TrustState,
  TrustTransition,
} from '../types/autonomy.js';
import type { CalibrationMetrics } from '../types/predictions.js';
import {
  AUTONOMY_RANK,
  ASCENT_REQUIREMENTS,
  DESCENT_TRIGGERS,
} from '../types/autonomy.js';

/** The ordered list of autonomy levels for traversal */
const LEVEL_ORDER: AutonomyLevel[] = [
  'supervised',
  'guided',
  'collaborative',
  'autonomous',
];

/**
 * Create a fresh trust state at the supervised level.
 */
export function createInitialTrustState(): TrustState {
  return {
    level: 'supervised',
    consecutiveCalibratedWindows: 0,
    consecutiveDegradedWindows: 0,
    totalPredictions: 0,
    correctPredictions: 0,
    levelEnteredAt: new Date().toISOString(),
    transitions: [],
    calibrationThreshold: 0.10, // ECE < 0.10 = "calibrated"
    probationary: false,
  };
}

/**
 * Process a calibration window and update trust state.
 *
 * This is called after each evaluation window (e.g., every 50 predictions).
 * It evaluates whether the system should ascend, descend, or remain.
 */
export function processCalibrationWindow(
  state: TrustState,
  windowMetrics: CalibrationMetrics,
  criticalMiss: boolean,
  immutableViolation: boolean,
): TrustState {
  const newState = { ...state };
  newState.totalPredictions += windowMetrics.windowSize;
  newState.transitions = [...state.transitions];

  // --- DESCENT CHECKS (fast) ---

  // Immutable violation → immediate reset to supervised
  if (immutableViolation) {
    return transitionTo(newState, 'supervised', 'immutable_violation',
      `Immutable constraint violated — full trust reset`, windowMetrics.ece);
  }

  // Critical miss → immediate descent by one level
  if (criticalMiss) {
    const targetLevel = descendOneLevel(state.level);
    return transitionTo(newState, targetLevel, 'critical_miss',
      `Critical prediction failure — immediate demotion`, windowMetrics.ece);
  }

  // Check if this window is calibrated or degraded
  const isCalibrated = windowMetrics.ece < state.calibrationThreshold;

  if (!isCalibrated) {
    newState.consecutiveDegradedWindows++;
    newState.consecutiveCalibratedWindows = 0;

    // 3 consecutive degraded windows → descent by one level
    if (newState.consecutiveDegradedWindows >= 3) {
      const targetLevel = descendOneLevel(state.level);
      return transitionTo(newState, targetLevel, 'calibration_degradation',
        `3 consecutive degraded windows (ECE ≥ ${state.calibrationThreshold})`, windowMetrics.ece);
    }

    return newState;
  }

  // --- ASCENT CHECK (slow) ---
  newState.consecutiveCalibratedWindows++;
  newState.consecutiveDegradedWindows = 0;

  // Check if we've earned the next level
  const currentRank = AUTONOMY_RANK[state.level];
  if (currentRank < LEVEL_ORDER.length - 1) {
    const nextLevel = LEVEL_ORDER[currentRank + 1];
    const required = ASCENT_REQUIREMENTS[nextLevel];

    if (newState.consecutiveCalibratedWindows >= required) {
      return transitionTo(newState, nextLevel, 'sustained_calibration',
        `${required} consecutive calibrated windows achieved`, windowMetrics.ece);
    }
  }

  // Probationary period clears after 5 calibrated windows post-descent
  if (newState.probationary && newState.consecutiveCalibratedWindows >= 5) {
    newState.probationary = false;
  }

  return newState;
}

/**
 * Descend one level, or stay at supervised (can't go lower).
 */
function descendOneLevel(current: AutonomyLevel): AutonomyLevel {
  const rank = AUTONOMY_RANK[current];
  if (rank <= 0) return 'supervised';
  return LEVEL_ORDER[rank - 1];
}

/**
 * Transition to a new trust level, recording the transition.
 */
function transitionTo(
  state: TrustState,
  target: AutonomyLevel,
  trigger: string,
  description: string,
  eceAtTransition: number,
): TrustState {
  const direction = AUTONOMY_RANK[target] > AUTONOMY_RANK[state.level]
    ? 'ascent' : 'descent';

  const transition: TrustTransition = {
    from: state.level,
    to: target,
    direction,
    trigger: `${trigger}: ${description}`,
    calibrationAtTransition: eceAtTransition,
    timestamp: new Date().toISOString(),
  };

  return {
    ...state,
    level: target,
    consecutiveCalibratedWindows: 0, // Reset counter on any transition
    consecutiveDegradedWindows: 0,
    levelEnteredAt: new Date().toISOString(),
    transitions: [...state.transitions, transition],
    probationary: direction === 'descent', // Enter probation on descent
  };
}

/**
 * Get a human-readable summary of the current trust state.
 */
export function summarizeTrustState(state: TrustState): string {
  const rank = AUTONOMY_RANK[state.level];
  const nextLevel = rank < LEVEL_ORDER.length - 1
    ? LEVEL_ORDER[rank + 1] : null;

  let summary = `Level: ${state.level.toUpperCase()}`;

  if (state.probationary) {
    summary += ' (PROBATIONARY)';
  }

  if (nextLevel) {
    const required = ASCENT_REQUIREMENTS[nextLevel];
    const remaining = required - state.consecutiveCalibratedWindows;
    summary += ` | ${state.consecutiveCalibratedWindows}/${required} windows toward ${nextLevel}`;
    if (remaining > 0) {
      summary += ` (${remaining} more needed)`;
    }
  } else {
    summary += ' | Maximum autonomy achieved';
  }

  if (state.consecutiveDegradedWindows > 0) {
    summary += ` | ⚠ ${state.consecutiveDegradedWindows}/3 degraded windows`;
  }

  summary += ` | ${state.transitions.length} total transitions`;

  return summary;
}

/**
 * Check if the current trust level permits a specific action.
 * Higher-risk actions require higher trust levels.
 */
export function isActionPermitted(
  state: TrustState,
  requiredLevel: AutonomyLevel,
): boolean {
  return AUTONOMY_RANK[state.level] >= AUTONOMY_RANK[requiredLevel];
}
