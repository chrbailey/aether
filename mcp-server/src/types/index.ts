/**
 * AETHER Type System
 * Re-exports all type definitions for convenient imports.
 */

export type {
  AetherEvent,
  EventSource,
  EventCase,
  CaseOutcome,
  LatentEventState,
  ProcessPathVariant,
  HierarchicalPrediction,
  ActivityPrediction,
  PhasePrediction,
  OutcomePrediction,
} from './events.js';

export type {
  UncertaintyDecomposition,
  DecompositionMethod,
  PredictionWithUncertainty,
  ConformalPredictionSet,
  CalibrationMetrics,
  CalibrationBucket,
  PredictionOutcome,
} from './predictions.js';

export type {
  GovernanceMode,
  GovernanceModulation,
  EffectiveThresholds,
  ImmutableConstraints,
  GateDecision,
} from './governance.js';

export {
  GOVERNANCE_MODES,
  IMMUTABLE_CONSTRAINTS,
} from './governance.js';

export type {
  AutonomyLevel,
  DescentTrigger,
  TrustState,
  TrustTransition,
} from './autonomy.js';

export {
  AUTONOMY_RANK,
  ASCENT_REQUIREMENTS,
  DESCENT_TRIGGERS,
  AUTONOMY_SYMBOLS,
} from './autonomy.js';
