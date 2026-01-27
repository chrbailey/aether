/**
 * AETHER Immutable Safety Constraints
 *
 * The Intrinsic Cost module from LeCun's architecture — hardwired
 * constraints that NEVER relax, regardless of calibration quality,
 * trust level, or governance mode.
 *
 * These form the absolute safety floor. The adaptive governance
 * layer operates ABOVE this floor — it can tighten beyond these
 * constraints but can never loosen below them.
 */

import { IMMUTABLE_CONSTRAINTS } from '../types/governance.js';
import type { UncertaintyDecomposition } from '../types/predictions.js';
import type { GovernanceMode } from '../types/governance.js';

/** Result of an immutable constraint check */
export interface ImmutableCheckResult {
  /** Whether all immutable constraints passed */
  passed: boolean;

  /** Which constraint was violated (if any) */
  violatedConstraint: string | null;

  /** Human-readable explanation */
  reason: string;

  /** Severity — immutable violations are always critical */
  severity: 'critical';
}

/** Patterns that indicate sensitive data — always trigger hold */
const SENSITIVE_PATTERNS = [
  /\b\d{3}-\d{2}-\d{4}\b/,            // SSN
  /\b\d{9}\b/,                          // SSN without dashes
  /\bpassword\s*[:=]\s*\S+/i,          // Password in text
  /\b(?:sk-|pk_|sk_live_|pk_live_)\w+/, // API keys
  /-----BEGIN (?:RSA )?PRIVATE KEY-----/, // Private keys
] as const;

/**
 * Check all immutable constraints against the current context.
 * Returns immediately on first violation — these are non-negotiable.
 */
export function checkImmutableConstraints(context: {
  mode: GovernanceMode;
  uncertainty: UncertaintyDecomposition;
  dsConflictCoefficient?: number;
  consecutiveFailures?: number;
  contentToCheck?: string;
}): ImmutableCheckResult {
  // 1. Forbidden mode always blocks
  if (context.mode.name === 'forbidden') {
    return {
      passed: false,
      violatedConstraint: 'forbidden_mode',
      reason: 'Forbidden mode (⊗) active — immutable block',
      severity: 'critical',
    };
  }

  // 2. Sensitive data patterns always trigger hold
  if (context.contentToCheck) {
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(context.contentToCheck)) {
        return {
          passed: false,
          violatedConstraint: 'sensitive_data',
          reason: `Sensitive data pattern detected — immutable hold`,
          severity: 'critical',
        };
      }
    }
  }

  // 3. D-S conflict coefficient above threshold requires review
  if (
    context.dsConflictCoefficient !== undefined &&
    context.dsConflictCoefficient > IMMUTABLE_CONSTRAINTS.dsConflictThreshold
  ) {
    return {
      passed: false,
      violatedConstraint: 'ds_conflict',
      reason: `Dempster-Shafer conflict coefficient ${context.dsConflictCoefficient.toFixed(3)} > ${IMMUTABLE_CONSTRAINTS.dsConflictThreshold} — mandatory human review`,
      severity: 'critical',
    };
  }

  // 4. Circuit breaker floor — even flexible mode can't go below this
  if (
    context.consecutiveFailures !== undefined &&
    context.consecutiveFailures >= IMMUTABLE_CONSTRAINTS.circuitBreakerFloor
  ) {
    return {
      passed: false,
      violatedConstraint: 'circuit_breaker_floor',
      reason: `${context.consecutiveFailures} consecutive failures ≥ circuit breaker floor ${IMMUTABLE_CONSTRAINTS.circuitBreakerFloor} — immutable block`,
      severity: 'critical',
    };
  }

  // 5. Maximum uncertainty for auto-pass
  if (context.uncertainty.total > IMMUTABLE_CONSTRAINTS.maxUncertaintyForAutoPass) {
    return {
      passed: false,
      violatedConstraint: 'max_uncertainty',
      reason: `Total uncertainty ${context.uncertainty.total.toFixed(3)} > ${IMMUTABLE_CONSTRAINTS.maxUncertaintyForAutoPass} — immutable hold`,
      severity: 'critical',
    };
  }

  return {
    passed: true,
    violatedConstraint: null,
    reason: 'All immutable constraints passed',
    severity: 'critical',
  };
}

/**
 * Check if content contains sensitive data patterns.
 * Extracted for reuse in content-scanning pipelines.
 */
export function containsSensitiveData(content: string): boolean {
  return SENSITIVE_PATTERNS.some(pattern => pattern.test(content));
}
