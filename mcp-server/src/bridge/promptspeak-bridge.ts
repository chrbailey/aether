/**
 * PromptSpeak Bridge
 *
 * Integration with PromptSpeak MCP server. AETHER provides
 * adaptive drift thresholds to replace the static 0.15.
 *
 * When PromptSpeak's gatekeeper checks drift, it can query
 * AETHER for the current effective threshold based on the
 * world model's uncertainty and calibration state.
 */

import type { EffectiveThresholds } from '../types/governance.js';
import type { AutonomyLevel } from '../types/autonomy.js';
import { AUTONOMY_SYMBOLS } from '../types/autonomy.js';

/**
 * Format AETHER state as a PromptSpeak frame annotation.
 * This can be appended to PromptSpeak frames for transparency.
 */
export function formatAsFrameAnnotation(
  thresholds: EffectiveThresholds,
  autonomyLevel: AutonomyLevel,
): string {
  const symbol = AUTONOMY_SYMBOLS[autonomyLevel];
  const driftMod = thresholds.modulations['driftThreshold'];

  return [
    `[${symbol}]`,
    `drift=${thresholds.driftThreshold.toFixed(3)}`,
    `(base=0.15`,
    `mode=${driftMod.modeFactor.toFixed(2)}`,
    `uncertainty=${driftMod.uncertaintyFactor.toFixed(2)}`,
    `calibration=${driftMod.calibrationFactor.toFixed(2)})`,
  ].join(' ');
}

/**
 * Convert AETHER's effective thresholds into PromptSpeak's
 * ConfidenceThresholds format for policy overlay integration.
 */
export function toPromptSpeakConfidenceThresholds(
  thresholds: EffectiveThresholds,
): Record<string, number> {
  return {
    driftThreshold: thresholds.driftThreshold,
    autoPassThreshold: thresholds.reviewGateAutoPass,
    holdThreshold: 1 - thresholds.driftThreshold, // Inverse relationship
  };
}

/**
 * Map an AETHER autonomy level to a PromptSpeak governance mode name.
 * This allows PromptSpeak to adjust its mode based on earned trust.
 */
export function autonomyToPromptSpeakMode(
  level: AutonomyLevel,
): 'strict' | 'standard' | 'flexible' {
  switch (level) {
    case 'supervised': return 'strict';
    case 'guided': return 'standard';
    case 'collaborative': return 'standard';
    case 'autonomous': return 'flexible';
  }
}
