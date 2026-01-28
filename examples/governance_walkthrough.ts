/**
 * Governance walkthrough example for AETHER (Process-JEPA).
 *
 * Demonstrates the core governance modulation formula:
 *   effective_threshold = base × mode_factor × uncertainty_factor × calibration_factor
 *
 * Shows how the same observed value can produce different gate decisions
 * depending on governance mode, uncertainty decomposition, and calibration.
 *
 * Usage:
 *   npx tsx examples/governance_walkthrough.ts
 */

import {
  computeModeFactor,
  computeUncertaintyFactor,
  computeCalibrationFactor,
  computeEffectiveThresholds,
  makeGateDecision,
} from '../mcp-server/src/governance/modulation.js';
import { GOVERNANCE_MODES } from '../mcp-server/src/types/governance.js';
import type { UncertaintyDecomposition } from '../mcp-server/src/types/predictions.js';
import type { CalibrationMetrics } from '../mcp-server/src/types/predictions.js';

// ---------------------------------------------------------------------------
// Scenario: A drift gate observes a drift score of 0.12.
// Is this within tolerance? Depends on the system's uncertainty state.
// ---------------------------------------------------------------------------

const observedDrift = 0.12;
const baseDriftThreshold = 0.15;

console.log('AETHER Governance Modulation Walkthrough');
console.log('========================================\n');
console.log(`Observed drift value: ${observedDrift}`);
console.log(`Base drift threshold: ${baseDriftThreshold}`);
console.log(`Static system would: ${observedDrift <= baseDriftThreshold ? 'ALLOW' : 'HOLD'}\n`);

// ---------------------------------------------------------------------------
// Scenario A: Low epistemic uncertainty, good calibration
// → System is confident and well-calibrated. Threshold stays near base.
// ---------------------------------------------------------------------------

console.log('--- Scenario A: Confident & Well-Calibrated ---');

const uncertaintyA: UncertaintyDecomposition = {
  total: 0.15,
  epistemic: 0.03,
  aleatoric: 0.12,
  epistemicRatio: 0.20,  // Only 20% of uncertainty is reducible
  method: 'ensemble_variance',
};

const calibrationA: CalibrationMetrics = {
  ece: 0.02,       // Excellent calibration
  mce: 0.08,
  brierScore: 0.05,
  reliability: [],
  windowSize: 100,
};

const modeFactorA = computeModeFactor(GOVERNANCE_MODES.standard);
const uncFactorA = computeUncertaintyFactor(uncertaintyA);
const calFactorA = computeCalibrationFactor(calibrationA);

console.log(`  Mode factor (standard):     ${modeFactorA.toFixed(4)}`);
console.log(`  Uncertainty factor:          ${uncFactorA.toFixed(4)}`);
console.log(`  Calibration factor:          ${calFactorA.toFixed(4)}`);
console.log(`  Combined tightening:         ${(modeFactorA * uncFactorA * calFactorA).toFixed(4)}`);

const effectiveA = baseDriftThreshold / (modeFactorA * uncFactorA * calFactorA);
console.log(`  Effective threshold:         ${effectiveA.toFixed(4)}`);
console.log(`  Decision: ${observedDrift <= effectiveA ? 'ALLOW ✓' : 'HOLD ✋'}`);
console.log(`  Reason: Low epistemic ratio means most uncertainty is irreducible.\n`);

// ---------------------------------------------------------------------------
// Scenario B: High epistemic uncertainty, same observation
// → Model is uncertain because it lacks data. Governance tightens.
// ---------------------------------------------------------------------------

console.log('--- Scenario B: High Epistemic Uncertainty ---');

const uncertaintyB: UncertaintyDecomposition = {
  total: 0.60,
  epistemic: 0.48,
  aleatoric: 0.12,
  epistemicRatio: 0.80,  // 80% of uncertainty is reducible!
  method: 'ensemble_variance',
};

const modeFactorB = computeModeFactor(GOVERNANCE_MODES.standard);
const uncFactorB = computeUncertaintyFactor(uncertaintyB);
const calFactorB = computeCalibrationFactor(calibrationA); // Same calibration

console.log(`  Mode factor (standard):     ${modeFactorB.toFixed(4)}`);
console.log(`  Uncertainty factor:          ${uncFactorB.toFixed(4)}`);
console.log(`  Calibration factor:          ${calFactorB.toFixed(4)}`);
console.log(`  Combined tightening:         ${(modeFactorB * uncFactorB * calFactorB).toFixed(4)}`);

const effectiveB = baseDriftThreshold / (modeFactorB * uncFactorB * calFactorB);
console.log(`  Effective threshold:         ${effectiveB.toFixed(4)}`);
console.log(`  Decision: ${observedDrift <= effectiveB ? 'ALLOW ✓' : 'HOLD ✋'}`);
console.log(`  Reason: High epistemic ratio means human review could help.\n`);

// ---------------------------------------------------------------------------
// Scenario C: Strict mode + poor calibration
// → Maximum tightening from governance mode + calibration quality.
// ---------------------------------------------------------------------------

console.log('--- Scenario C: Strict Mode + Poor Calibration ---');

const calibrationC: CalibrationMetrics = {
  ece: 0.25,       // Poor calibration
  mce: 0.45,
  brierScore: 0.35,
  reliability: [],
  windowSize: 100,
};

const modeFactorC = computeModeFactor(GOVERNANCE_MODES.strict);
const uncFactorC = computeUncertaintyFactor(uncertaintyA); // Low epistemic
const calFactorC = computeCalibrationFactor(calibrationC);

console.log(`  Mode factor (strict):       ${modeFactorC.toFixed(4)}`);
console.log(`  Uncertainty factor:          ${uncFactorC.toFixed(4)}`);
console.log(`  Calibration factor:          ${calFactorC.toFixed(4)}`);
console.log(`  Combined tightening:         ${(modeFactorC * uncFactorC * calFactorC).toFixed(4)}`);

const effectiveC = baseDriftThreshold / (modeFactorC * uncFactorC * calFactorC);
console.log(`  Effective threshold:         ${effectiveC.toFixed(4)}`);
console.log(`  Decision: ${observedDrift <= effectiveC ? 'ALLOW ✓' : 'HOLD ✋'}`);
console.log(`  Reason: Strict mode + poor calibration compounds tightening.\n`);

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log('========================================');
console.log('Summary: Same observation (0.12), three different decisions:');
console.log(`  A) Confident + calibrated  → threshold ${effectiveA.toFixed(3)} → ${observedDrift <= effectiveA ? 'ALLOW' : 'HOLD'}`);
console.log(`  B) High epistemic          → threshold ${effectiveB.toFixed(3)} → ${observedDrift <= effectiveB ? 'ALLOW' : 'HOLD'}`);
console.log(`  C) Strict + poor cal       → threshold ${effectiveC.toFixed(3)} → ${observedDrift <= effectiveC ? 'ALLOW' : 'HOLD'}`);
console.log('\nThis is the core of AETHER: governance that adapts to what the model knows.');
