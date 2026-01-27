/**
 * AETHER Autonomy Types
 *
 * Progressive autonomy — the system earns trust through
 * demonstrated calibration. Trust ascent is slow (requires
 * sustained calibration). Trust descent is fast (single
 * critical failure triggers immediate demotion).
 *
 * This asymmetric dynamic is a novel control-theoretic
 * treatment of the "governance descent problem."
 */

/**
 * Four levels of autonomy, each granting progressively
 * more freedom to the system.
 */
export type AutonomyLevel = 'supervised' | 'guided' | 'collaborative' | 'autonomous';

/** Numeric ordering for comparison */
export const AUTONOMY_RANK: Record<AutonomyLevel, number> = {
  supervised: 0,
  guided: 1,
  collaborative: 2,
  autonomous: 3,
};

/**
 * Ascent requirements — how many consecutive calibrated
 * windows are needed to advance to the next level.
 */
export const ASCENT_REQUIREMENTS: Record<AutonomyLevel, number> = {
  supervised: 0,     // Starting point — no requirement
  guided: 10,        // 10 calibrated windows → guided
  collaborative: 20, // 20 more calibrated windows → collaborative
  autonomous: 50,    // 50 more calibrated windows → autonomous
};

/**
 * Descent triggers — conditions that cause immediate
 * or gradual trust reduction.
 */
export interface DescentTrigger {
  /** Type of descent trigger */
  type: 'critical_miss' | 'calibration_degradation' | 'immutable_violation';

  /**
   * How many consecutive degraded windows trigger descent.
   * critical_miss = 1 (immediate), degradation = 3, immutable = 1
   */
  windowsRequired: number;

  /** How many levels to descend */
  levelsToDescend: number;

  /** Description for audit trail */
  description: string;
}

/** Pre-defined descent triggers */
export const DESCENT_TRIGGERS: DescentTrigger[] = [
  {
    type: 'critical_miss',
    windowsRequired: 1,
    levelsToDescend: 1,
    description: 'Single critical prediction failure',
  },
  {
    type: 'calibration_degradation',
    windowsRequired: 3,
    levelsToDescend: 1,
    description: 'Three consecutive windows with degraded calibration',
  },
  {
    type: 'immutable_violation',
    windowsRequired: 1,
    levelsToDescend: Infinity, // Reset to supervised
    description: 'Immutable safety constraint violated — full reset',
  },
];

/**
 * Trust state — the full state of the autonomy controller.
 * Tracks calibration history, current level, and transition log.
 */
export interface TrustState {
  /** Current autonomy level */
  level: AutonomyLevel;

  /** Consecutive calibrated windows at current level */
  consecutiveCalibratedWindows: number;

  /** Consecutive degraded windows (for descent detection) */
  consecutiveDegradedWindows: number;

  /** Total predictions made at this level */
  totalPredictions: number;

  /** Total correct predictions at this level */
  correctPredictions: number;

  /** When the current level was entered */
  levelEnteredAt: string;

  /** History of level transitions */
  transitions: TrustTransition[];

  /** Calibration threshold — ECE below this = "calibrated" */
  calibrationThreshold: number;

  /** Whether the system is in probationary period after descent */
  probationary: boolean;
}

/** A trust level transition event */
export interface TrustTransition {
  /** Previous level */
  from: AutonomyLevel;
  /** New level */
  to: AutonomyLevel;
  /** Direction of transition */
  direction: 'ascent' | 'descent';
  /** What triggered the transition */
  trigger: string;
  /** Calibration metrics at time of transition */
  calibrationAtTransition: number;
  /** Timestamp */
  timestamp: string;
}

/**
 * PromptSpeak autonomy symbols — registered in the symbol ontology.
 * Ξ.AUTONOMY.{SUPERVISED|GUIDED|COLLABORATIVE|AUTONOMOUS}
 */
export const AUTONOMY_SYMBOLS: Record<AutonomyLevel, string> = {
  supervised:    'Ξ.AUTONOMY.SUPERVISED',
  guided:        'Ξ.AUTONOMY.GUIDED',
  collaborative: 'Ξ.AUTONOMY.COLLABORATIVE',
  autonomous:    'Ξ.AUTONOMY.AUTONOMOUS',
};
