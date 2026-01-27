/**
 * AETHER Event Types
 *
 * Extends SAP ProcessEvent / EFC Event patterns into a unified
 * event representation for the world model. Events are the atomic
 * inputs to the JEPA-style encoder.
 */

/** Raw business event — the ground truth input layer */
export interface AetherEvent {
  /** Unique event identifier */
  eventId: string;

  /** Case/process instance this event belongs to */
  caseId: string;

  /** Activity performed (e.g., "create_order", "approve_credit") */
  activity: string;

  /** Resource/actor who performed the activity */
  resource: string;

  /** ISO timestamp of the event */
  timestamp: string;

  /** Event attributes — domain-specific key-value pairs */
  attributes: Record<string, string | number | boolean>;

  /** Source system identifier */
  source: EventSource;
}

/** Supported event source systems */
export type EventSource =
  | 'sap_o2c'        // SAP Order-to-Cash
  | 'sap_p2p'        // SAP Procure-to-Pay
  | 'efc'            // Epistemic Flow Control
  | 'promptspeak'    // PromptSpeak governance events
  | 'synthetic';     // Synthetic training data

/** A complete case (sequence of events for one process instance) */
export interface EventCase {
  caseId: string;
  events: AetherEvent[];
  /** Known outcome (for training/calibration) */
  outcome?: CaseOutcome;
}

/** Possible case outcomes for supervised training */
export interface CaseOutcome {
  /** Whether the case completed successfully */
  completed: boolean;
  /** Whether it completed on time */
  onTime: boolean;
  /** Whether rework was required */
  rework: boolean;
  /** Total case duration in hours */
  durationHours: number;
  /** Final status */
  status: 'completed' | 'late' | 'rework' | 'cancelled' | 'in_progress';
}

/**
 * Latent event representation — output of the Event Encoder.
 * This is the JEPA-adapted representation: we predict in this
 * space, not in raw event space.
 */
export interface LatentEventState {
  /** Case identifier */
  caseId: string;

  /** Latent vector (128-dimensional) */
  embedding: number[];

  /** Categorical latent variable — which process path variant */
  pathVariant: ProcessPathVariant;

  /** Confidence in path variant assignment */
  pathConfidence: number;

  /** Timestamp of the latest encoded event */
  asOfTimestamp: string;

  /** Number of events encoded so far */
  eventCount: number;
}

/**
 * Process path variants — the structured categorical latent variable.
 * Unlike continuous Gaussian latents, these represent discrete
 * business process paths that a case can follow.
 */
export type ProcessPathVariant =
  | 'standard'          // Normal happy path
  | 'credit_hold'       // Requires credit approval
  | 'rework'            // Needs correction/rework
  | 'expedited'         // Fast-tracked
  | 'exception'         // Exception handling path
  | 'unknown';          // Not yet determined

/**
 * Hierarchical prediction levels — the world model predicts
 * at three timescales simultaneously.
 */
export interface HierarchicalPrediction {
  /** Activity level: next event prediction */
  activity: ActivityPrediction;
  /** Phase level: next process phase */
  phase: PhasePrediction;
  /** Outcome level: case completion prediction */
  outcome: OutcomePrediction;
}

export interface ActivityPrediction {
  /** Top-K next activities with probabilities */
  topK: Array<{ activity: string; probability: number }>;
  /** Expected time to next event (hours) */
  expectedDeltaHours: number;
}

export interface PhasePrediction {
  /** Current phase */
  currentPhase: string;
  /** Next phase with probability */
  nextPhase: string;
  nextPhaseProbability: number;
  /** Expected time to phase transition (hours) */
  expectedTransitionHours: number;
}

export interface OutcomePrediction {
  /** Predicted outcome status */
  predictedStatus: CaseOutcome['status'];
  /** Probability of on-time completion */
  onTimeProbability: number;
  /** Probability of rework */
  reworkProbability: number;
  /** Expected remaining duration (hours) */
  expectedRemainingHours: number;
}
