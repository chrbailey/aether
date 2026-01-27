/**
 * Governance Integration Tests
 *
 * Tests the immutable constraints and end-to-end governance flow:
 * - Immutable constraints never relax
 * - Sensitive data always triggers hold
 * - Forbidden mode always blocks
 * - D-S conflict above threshold always requires review
 * - Circuit breaker floor always holds
 */

import { describe, it, expect } from 'vitest';
import {
  checkImmutableConstraints,
  containsSensitiveData,
} from '../governance/immutable.js';
import { GOVERNANCE_MODES, IMMUTABLE_CONSTRAINTS } from '../types/governance.js';
import type { UncertaintyDecomposition } from '../types/predictions.js';

function makeUncertainty(overrides: Partial<UncertaintyDecomposition> = {}): UncertaintyDecomposition {
  return {
    total: 0.3,
    epistemic: 0.2,
    aleatoric: 0.1,
    epistemicRatio: 0.667,
    method: 'ensemble_variance',
    ...overrides,
  };
}

// --- Immutable Constraints ---

describe('checkImmutableConstraints', () => {
  it('blocks in forbidden mode', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['forbidden'],
      uncertainty: makeUncertainty(),
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('forbidden_mode');
    expect(result.severity).toBe('critical');
  });

  it('passes in standard mode with normal inputs', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
    });

    expect(result.passed).toBe(true);
  });

  it('catches SSN patterns', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      contentToCheck: 'The SSN is 123-45-6789 for this customer.',
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('sensitive_data');
  });

  it('catches password patterns', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      contentToCheck: 'password: mysecretpass123',
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('sensitive_data');
  });

  it('catches API key patterns', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      contentToCheck: 'Use key sk-1234567890abcdef for the API',
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('sensitive_data');
  });

  it('catches private key patterns', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      contentToCheck: '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIB...',
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('sensitive_data');
  });

  it('holds when D-S conflict exceeds threshold', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      dsConflictCoefficient: 0.75, // > 0.7 threshold
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('ds_conflict');
  });

  it('passes when D-S conflict is below threshold', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      dsConflictCoefficient: 0.5,
    });

    expect(result.passed).toBe(true);
  });

  it('holds when circuit breaker floor is hit', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      consecutiveFailures: 3,
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('circuit_breaker_floor');
  });

  it('passes when failures are below floor', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty(),
      consecutiveFailures: 2,
    });

    expect(result.passed).toBe(true);
  });

  it('holds when total uncertainty exceeds maximum', () => {
    const result = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['standard'],
      uncertainty: makeUncertainty({ total: 0.96 }),
    });

    expect(result.passed).toBe(false);
    expect(result.violatedConstraint).toBe('max_uncertainty');
  });

  it('immutable constraints cannot be disabled by any mode', () => {
    // Even flexible mode cannot bypass immutable constraints
    const flexibleWithSSN = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['flexible'],
      uncertainty: makeUncertainty(),
      contentToCheck: '123-45-6789',
    });

    expect(flexibleWithSSN.passed).toBe(false);

    const flexibleWithConflict = checkImmutableConstraints({
      mode: GOVERNANCE_MODES['flexible'],
      uncertainty: makeUncertainty(),
      dsConflictCoefficient: 0.8,
    });

    expect(flexibleWithConflict.passed).toBe(false);
  });
});

// --- Sensitive Data Detection ---

describe('containsSensitiveData', () => {
  it('detects SSN with dashes', () => {
    expect(containsSensitiveData('SSN: 123-45-6789')).toBe(true);
  });

  it('detects SSN without dashes', () => {
    expect(containsSensitiveData('SSN: 123456789')).toBe(true);
  });

  it('does not false-positive on normal text', () => {
    expect(containsSensitiveData('The order total is $1,234.56')).toBe(false);
  });

  it('detects API keys with common prefixes', () => {
    expect(containsSensitiveData('sk-abc123def456')).toBe(true);
    expect(containsSensitiveData('pk_live_abc123def456')).toBe(true);
  });
});

// --- Immutable Constants ---

describe('IMMUTABLE_CONSTRAINTS', () => {
  it('has expected constant values that never change', () => {
    expect(IMMUTABLE_CONSTRAINTS.forbiddenModeBlocks).toBe(true);
    expect(IMMUTABLE_CONSTRAINTS.sensitiveDataHold).toBe(true);
    expect(IMMUTABLE_CONSTRAINTS.dsConflictThreshold).toBe(0.7);
    expect(IMMUTABLE_CONSTRAINTS.circuitBreakerFloor).toBe(3);
    expect(IMMUTABLE_CONSTRAINTS.maxUncertaintyForAutoPass).toBe(0.95);
  });
});
