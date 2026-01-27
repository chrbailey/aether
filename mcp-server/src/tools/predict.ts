/**
 * AETHER Prediction Tools
 *
 * MCP tools for making predictions with the world model.
 * Each prediction carries full uncertainty quantification.
 */

import { z } from 'zod';
import type { AetherEvent } from '../types/events.js';
import type { PredictionWithUncertainty } from '../types/predictions.js';
import * as pythonBridge from '../bridge/python-bridge.js';

/** Schema for predict_next_event tool input */
export const predictNextEventSchema = z.object({
  caseId: z.string().describe('Case/process instance identifier'),
  events: z.array(z.object({
    eventId: z.string(),
    caseId: z.string(),
    activity: z.string(),
    resource: z.string(),
    timestamp: z.string(),
    attributes: z.record(z.union([z.string(), z.number(), z.boolean()])),
    source: z.enum(['sap_o2c', 'sap_p2p', 'efc', 'promptspeak', 'synthetic']),
  })).describe('Ordered sequence of events for this case'),
});

/** Schema for predict_outcome tool input */
export const predictOutcomeSchema = z.object({
  caseId: z.string().describe('Case/process instance identifier'),
  events: z.array(z.object({
    eventId: z.string(),
    caseId: z.string(),
    activity: z.string(),
    resource: z.string(),
    timestamp: z.string(),
    attributes: z.record(z.union([z.string(), z.number(), z.boolean()])),
    source: z.enum(['sap_o2c', 'sap_p2p', 'efc', 'promptspeak', 'synthetic']),
  })).describe('Ordered sequence of events for this case'),
});

export type PredictNextEventInput = z.infer<typeof predictNextEventSchema>;
export type PredictOutcomeInput = z.infer<typeof predictOutcomeSchema>;

/**
 * Predict the next event in a case sequence.
 * Returns top-K activities with probabilities and uncertainty.
 */
export async function predictNextEvent(
  input: PredictNextEventInput,
): Promise<PredictionWithUncertainty> {
  const healthy = await pythonBridge.healthCheck();

  if (!healthy) {
    // Return a high-uncertainty fallback prediction
    return createFallbackPrediction(input.caseId, input.events);
  }

  return pythonBridge.predict(input.events as AetherEvent[]);
}

/**
 * Predict the outcome of a case (on-time, late, rework).
 * Returns outcome probabilities with uncertainty decomposition.
 */
export async function predictOutcome(
  input: PredictOutcomeInput,
): Promise<PredictionWithUncertainty> {
  const healthy = await pythonBridge.healthCheck();

  if (!healthy) {
    return createFallbackPrediction(input.caseId, input.events);
  }

  return pythonBridge.predict(input.events as AetherEvent[]);
}

/**
 * Create a fallback prediction when the Python server is unavailable.
 * Uses maximum epistemic uncertainty (conservative — tightens governance).
 */
function createFallbackPrediction(
  caseId: string,
  events: Array<{ activity: string; timestamp: string }>,
): PredictionWithUncertainty {
  const lastEvent = events[events.length - 1];

  return {
    predictionId: `fallback-${Date.now()}`,
    caseId,
    latentState: {
      caseId,
      embedding: new Array(128).fill(0),
      pathVariant: 'unknown',
      pathConfidence: 0,
      asOfTimestamp: lastEvent?.timestamp ?? new Date().toISOString(),
      eventCount: events.length,
    },
    predictions: {
      activity: {
        topK: [{ activity: 'unknown', probability: 1.0 }],
        expectedDeltaHours: 24,
      },
      phase: {
        currentPhase: 'unknown',
        nextPhase: 'unknown',
        nextPhaseProbability: 0.5,
        expectedTransitionHours: 48,
      },
      outcome: {
        predictedStatus: 'in_progress',
        onTimeProbability: 0.5,
        reworkProbability: 0.2,
        expectedRemainingHours: 72,
      },
    },
    uncertainty: pythonBridge.fallbackUncertainty(),
    energyScore: 1.0, // Maximum energy = maximum implausibility
    conformalSet: {
      activitySet: ['unknown'],
      outcomeSet: ['completed', 'late', 'rework'],
      coverageTarget: 0.9,
      alpha: 0.1,
      empiricalCoverage: 0,
      setSize: 3,
    },
    timestamp: new Date().toISOString(),
    modelVersion: 'fallback-v0',
  };
}
