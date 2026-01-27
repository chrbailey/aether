/**
 * Governance Modulation Tests
 *
 * Verifies the novel contribution: compositional uncertainty × governance.
 *
 * Key properties being tested:
 * 1. Epistemic uncertainty tightens governance (higher epistemic → stricter)
 * 2. Aleatoric uncertainty does NOT tighten governance (irreducible)
 * 3. Poor calibration tightens governance
 * 4. Strict mode is stricter than flexible mode
 * 5. Thresholds stay within reasonable bounds
 * 6. Direction-awareness: lower-is-stricter vs higher-is-stricter gates
 */

import { describe, it, expect } from 'vitest';
import {
  computeModeFactor,
  computeUncertaintyFactor,
  computeCalibrationFactor,
  computeEffectiveThresholds,
  makeGateDecision,
} from '../governance/modulation.js';
import { GOVERNANCE_MODES } from '../types/governance.js';
import type { UncertaintyDecomposition, CalibrationMetrics } from '../types/predictions.js';

// --- Test Fixtures ---

function makeUncertainty(overrides: Partial<UncertaintyDecomposition> = {}): UncertaintyDecomposition {
  return {
    total: 0.5,
    epistemic: 0.3,
    aleatoric: 0.2,
    epistemicRatio: 0.6,
    method: 'ensemble_variance',
    ...overrides,
  };
}

function makeCalibration(overrides: Partial<CalibrationMetrics> = {}): CalibrationMetrics {
  return {
    ece: 0.05,
    mce: 0.10,
    brierScore: 0.08,
    windowSize: 50,
    windowStart: '2025-01-01T00:00:00Z',
    windowEnd: '2025-01-01T01:00:00Z',
    buckets: [],
    ...overrides,
  };
}

// --- Mode Factor Tests ---

describe('computeModeFactor', () => {
  it('returns higher factor for strict mode (more tightening)', () => {
    const flexible = computeModeFactor(GOVERNANCE_MODES['flexible']);
    const strict = computeModeFactor(GOVERNANCE_MODES['strict']);
    expect(strict).toBeGreaterThan(flexible);
  });

  it('returns 1.0 for flexible mode (no tightening — baseline)', () => {
    const factor = computeModeFactor(GOVERNANCE_MODES['flexible']);
    expect(factor).toBeCloseTo(1.0);
  });

  it('produces correct values for each mode', () => {
    // flexible (s=0): 1 + (0/3) * 0.3 = 1.0
    expect(computeModeFactor(GOVERNANCE_MODES['flexible'])).toBeCloseTo(1.0);
    // standard (s=1): 1 + (1/3) * 0.3 = 1.1
    expect(computeModeFactor(GOVERNANCE_MODES['standard'])).toBeCloseTo(1.1);
    // strict (s=2): 1 + (2/3) * 0.3 = 1.2
    expect(computeModeFactor(GOVERNANCE_MODES['strict'])).toBeCloseTo(1.2);
    // forbidden (s=3): 1 + (3/3) * 0.3 = 1.3
    expect(computeModeFactor(GOVERNANCE_MODES['forbidden'])).toBeCloseTo(1.3);
  });
});

// --- Uncertainty Factor Tests ---

describe('computeUncertaintyFactor', () => {
  it('THE KEY INSIGHT: high epistemic uncertainty tightens governance', () => {
    const highEpistemic = makeUncertainty({
      total: 0.8,
      epistemic: 0.7,
      aleatoric: 0.1,
      epistemicRatio: 0.875,
    });

    const factor = computeUncertaintyFactor(highEpistemic);
    // 1 + (0.8 * 0.875) * 0.5 = 1 + 0.35 = 1.35
    expect(factor).toBeGreaterThan(1.3);
  });

  it('THE KEY INSIGHT: high aleatoric uncertainty does NOT tighten', () => {
    const highAleatoric = makeUncertainty({
      total: 0.8,
      epistemic: 0.1,
      aleatoric: 0.7,
      epistemicRatio: 0.125,
    });

    const factor = computeUncertaintyFactor(highAleatoric);
    // 1 + (0.8 * 0.125) * 0.5 = 1 + 0.05 = 1.05
    expect(factor).toBeLessThan(1.1);
  });

  it('same total uncertainty produces different factors based on decomposition', () => {
    const mostlyEpistemic = makeUncertainty({
      total: 0.6,
      epistemic: 0.5,
      aleatoric: 0.1,
      epistemicRatio: 0.833,
    });

    const mostlyAleatoric = makeUncertainty({
      total: 0.6,
      epistemic: 0.1,
      aleatoric: 0.5,
      epistemicRatio: 0.167,
    });

    const epistemicFactor = computeUncertaintyFactor(mostlyEpistemic);
    const aleatoricFactor = computeUncertaintyFactor(mostlyAleatoric);

    expect(epistemicFactor).toBeGreaterThan(aleatoricFactor);
    // The difference should be substantial
    expect(epistemicFactor - aleatoricFactor).toBeGreaterThan(0.15);
  });

  it('zero uncertainty produces factor of 1.0 (no tightening)', () => {
    const noUncertainty = makeUncertainty({
      total: 0,
      epistemic: 0,
      aleatoric: 0,
      epistemicRatio: 0,
    });

    expect(computeUncertaintyFactor(noUncertainty)).toBeCloseTo(1.0);
  });
});

// --- Calibration Factor Tests ---

describe('computeCalibrationFactor', () => {
  it('poor calibration (high ECE) tightens governance', () => {
    const poorCal = makeCalibration({ ece: 0.30 });
    const goodCal = makeCalibration({ ece: 0.02 });

    const poorFactor = computeCalibrationFactor(poorCal);
    const goodFactor = computeCalibrationFactor(goodCal);

    expect(poorFactor).toBeGreaterThan(goodFactor);
  });

  it('perfect calibration produces factor near 1.0', () => {
    const perfectCal = makeCalibration({ ece: 0.0 });
    expect(computeCalibrationFactor(perfectCal)).toBeCloseTo(1.0);
  });

  it('maximum calibration error (ECE=1.0) produces maximum factor', () => {
    const worstCal = makeCalibration({ ece: 1.0 });
    // 1 + (1 - 0) * 0.4 = 1.4
    expect(computeCalibrationFactor(worstCal)).toBeCloseTo(1.4);
  });
});

// --- Effective Thresholds Tests ---

describe('computeEffectiveThresholds', () => {
  it('produces valid thresholds with default inputs', () => {
    const thresholds = computeEffectiveThresholds(
      GOVERNANCE_MODES['standard'],
      makeUncertainty(),
      makeCalibration(),
      'supervised',
    );

    // All thresholds should be positive numbers
    expect(thresholds.driftThreshold).toBeGreaterThan(0);
    expect(thresholds.reviewGateAutoPass).toBeGreaterThan(0);
    expect(thresholds.threatActivation).toBeGreaterThan(0);
    expect(thresholds.conformanceDeviation).toBeGreaterThan(0);
    expect(thresholds.sayDoGap).toBeGreaterThan(0);
    expect(thresholds.knowledgePromotion).toBeGreaterThan(0);
  });

  it('strict mode produces tighter thresholds than flexible mode', () => {
    // Use minimal uncertainty and perfect calibration to isolate mode effect
    // and avoid hitting clamp boundaries
    const uncertainty = makeUncertainty({ total: 0.0, epistemicRatio: 0.0 });
    const calibration = makeCalibration({ ece: 0.0 });

    const strict = computeEffectiveThresholds(
      GOVERNANCE_MODES['strict'], uncertainty, calibration, 'supervised',
    );
    const flexible = computeEffectiveThresholds(
      GOVERNANCE_MODES['flexible'], uncertainty, calibration, 'supervised',
    );

    // Lower-is-stricter: strict should have LOWER drift threshold
    expect(strict.driftThreshold).toBeLessThan(flexible.driftThreshold);
    expect(strict.conformanceDeviation).toBeLessThan(flexible.conformanceDeviation);
    expect(strict.sayDoGap).toBeLessThan(flexible.sayDoGap);

    // Higher-is-stricter: strict should have HIGHER auto-pass threshold
    // With zero uncertainty and perfect calibration, no clamping occurs
    expect(strict.reviewGateAutoPass).toBeGreaterThan(flexible.reviewGateAutoPass);
    expect(strict.knowledgePromotion).toBeGreaterThan(flexible.knowledgePromotion);
  });

  it('high epistemic uncertainty tightens all thresholds', () => {
    // Use perfect calibration to isolate the uncertainty effect
    const calibration = makeCalibration({ ece: 0.0 });

    const lowEpistemic = computeEffectiveThresholds(
      GOVERNANCE_MODES['flexible'],  // Flexible to minimize mode tightening
      makeUncertainty({ total: 0.1, epistemicRatio: 0.1 }),
      calibration,
      'supervised',
    );

    const highEpistemic = computeEffectiveThresholds(
      GOVERNANCE_MODES['flexible'],
      makeUncertainty({ total: 0.8, epistemicRatio: 0.9 }),
      calibration,
      'supervised',
    );

    // Lower-is-stricter gates should be lower with high epistemic
    expect(highEpistemic.driftThreshold).toBeLessThan(lowEpistemic.driftThreshold);
    expect(highEpistemic.conformanceDeviation).toBeLessThan(lowEpistemic.conformanceDeviation);

    // Higher-is-stricter: check that thresholds are at least as tight
    // (may hit cap for reviewGateAutoPass due to 0.99 bound)
    expect(highEpistemic.reviewGateAutoPass).toBeGreaterThanOrEqual(lowEpistemic.reviewGateAutoPass);
    expect(highEpistemic.knowledgePromotion).toBeGreaterThanOrEqual(lowEpistemic.knowledgePromotion);
  });

  it('thresholds are bounded within reasonable ranges', () => {
    // Even with extreme inputs, thresholds should stay bounded
    const extreme = computeEffectiveThresholds(
      GOVERNANCE_MODES['strict'],
      makeUncertainty({ total: 1.0, epistemicRatio: 1.0 }),
      makeCalibration({ ece: 0.5 }),
      'supervised',
    );

    expect(extreme.driftThreshold).toBeGreaterThanOrEqual(0.02);
    expect(extreme.driftThreshold).toBeLessThanOrEqual(0.30);
    expect(extreme.reviewGateAutoPass).toBeGreaterThanOrEqual(0.80);
    expect(extreme.reviewGateAutoPass).toBeLessThanOrEqual(0.99);
    expect(extreme.conformanceDeviation).toBeGreaterThanOrEqual(0.01);
    expect(extreme.conformanceDeviation).toBeLessThanOrEqual(0.15);
  });

  it('includes modulation details for every threshold', () => {
    const thresholds = computeEffectiveThresholds(
      GOVERNANCE_MODES['standard'],
      makeUncertainty(),
      makeCalibration(),
      'supervised',
    );

    expect(thresholds.modulations['driftThreshold']).toBeDefined();
    expect(thresholds.modulations['reviewGateAutoPass']).toBeDefined();
    expect(thresholds.modulations['conformanceDeviation']).toBeDefined();
    expect(thresholds.computedAt).toBeDefined();
  });
});

// --- Gate Decision Tests ---

describe('makeGateDecision', () => {
  it('allows when observed value is within threshold (lower-is-stricter)', () => {
    const modulation = computeEffectiveThresholds(
      GOVERNANCE_MODES['standard'],
      makeUncertainty(),
      makeCalibration(),
      'supervised',
    ).modulations['driftThreshold'];

    // Observed drift below threshold → allow
    const decision = makeGateDecision(
      'driftThreshold',
      0.01, // Very low drift
      modulation,
      true, // lower-is-stricter
    );

    expect(decision.action).toBe('allow');
  });

  it('holds when observed value exceeds threshold (lower-is-stricter)', () => {
    const modulation = computeEffectiveThresholds(
      GOVERNANCE_MODES['strict'],
      makeUncertainty({ total: 0.8, epistemicRatio: 0.9 }),
      makeCalibration({ ece: 0.3 }),
      'supervised',
    ).modulations['driftThreshold'];

    // Observed drift above threshold → hold
    const decision = makeGateDecision(
      'driftThreshold',
      0.25, // High drift
      modulation,
      true,
    );

    expect(decision.action).toBe('hold');
    expect(decision.immutableTriggered).toBe(false);
  });

  it('blocks in forbidden mode regardless of values', () => {
    const modulation = computeEffectiveThresholds(
      GOVERNANCE_MODES['forbidden'],
      makeUncertainty(),
      makeCalibration(),
      'supervised',
    ).modulations['driftThreshold'];

    const decision = makeGateDecision('driftThreshold', 0.001, modulation, true);
    expect(decision.action).toBe('block');
    expect(decision.immutableTriggered).toBe(true);
  });

  it('generates unique audit IDs', () => {
    const modulation = computeEffectiveThresholds(
      GOVERNANCE_MODES['standard'],
      makeUncertainty(),
      makeCalibration(),
      'supervised',
    ).modulations['driftThreshold'];

    const d1 = makeGateDecision('test', 0.1, modulation, true);
    const d2 = makeGateDecision('test', 0.1, modulation, true);

    expect(d1.auditId).not.toBe(d2.auditId);
  });
});
