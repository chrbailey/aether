/**
 * AETHER MCP Server
 *
 * Adaptive Epistemic Trust through Hierarchical Event Reasoning
 *
 * Exposes 6 MCP tools:
 * 1. predict_next_event  — Predict the next event in a process case
 * 2. predict_outcome     — Predict the outcome of a process case
 * 3. get_calibration     — Get current calibration metrics
 * 4. get_autonomy_level  — Get current trust/autonomy level
 * 5. get_effective_thresholds — Get adapted governance thresholds
 * 6. evaluate_gate       — Evaluate a gate decision with uncertainty
 *
 * PromptSpeak Symbol: Ξ.SYSTEM.AETHER
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

import {
  predictNextEvent,
  predictNextEventSchema,
  predictOutcome,
  predictOutcomeSchema,
} from './tools/predict.js';

import {
  getCalibration,
  getCalibrationSchema,
  getAutonomyLevel,
  getAutonomyLevelSchema,
  setTrustState,
} from './tools/calibration.js';

import {
  getEffectiveThresholds,
  getEffectiveThresholdsSchema,
  evaluateGate,
  evaluateGateSchema,
} from './tools/governance.js';

import { createInitialTrustState } from './governance/autonomy-controller.js';

const server = new McpServer({
  name: 'aether',
  version: '0.1.0',
});

// --- Tool Registration ---

server.tool(
  'predict_next_event',
  'Predict the next event in a process case. Returns top-K next activities with probabilities, uncertainty decomposition (epistemic vs aleatoric), and conformal prediction set.',
  predictNextEventSchema.shape,
  async ({ caseId, events }) => {
    const result = await predictNextEvent({ caseId, events });
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  'predict_outcome',
  'Predict the outcome of a process case (on-time, late, rework). Returns outcome probabilities with full uncertainty quantification.',
  predictOutcomeSchema.shape,
  async ({ caseId, events }) => {
    const result = await predictOutcome({ caseId, events });
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  'get_calibration',
  'Get current model calibration metrics: ECE (Expected Calibration Error), MCE, Brier score, and per-bucket reliability.',
  getCalibrationSchema.shape,
  async (input) => {
    const result = await getCalibration(input);
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  'get_autonomy_level',
  'Get the current autonomy/trust level (SUPERVISED → GUIDED → COLLABORATIVE → AUTONOMOUS) and how close the system is to the next level.',
  getAutonomyLevelSchema.shape,
  () => {
    const result = getAutonomyLevel();
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  'get_effective_thresholds',
  'Get adapted governance thresholds for all gates. Shows how base thresholds (e.g., drift=0.15) are modulated by governance mode, uncertainty, and calibration.',
  getEffectiveThresholdsSchema.shape,
  async (input) => {
    const result = await getEffectiveThresholds(input);
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  'evaluate_gate',
  'Evaluate a specific gate decision. Compares an observed value against the adaptive threshold, checking immutable constraints first. Returns allow/hold/block with full explanation.',
  evaluateGateSchema.shape,
  async (input) => {
    const result = await evaluateGate(input);
    return {
      content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }],
    };
  },
);

// --- Server Startup ---

async function main(): Promise<void> {
  // Initialize trust state at supervised level
  setTrustState(createInitialTrustState());

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error('AETHER server failed to start:', error);
  process.exit(1);
});
