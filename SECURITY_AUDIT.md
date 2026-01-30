# AETHER Security Audit Report

**Audit Date:** 2026-01-30
**Repository:** `/Volumes/OWC drive/Dev/aether`
**Auditor:** Automated Security Review

---

## Executive Summary

This security audit examined the AETHER codebase, which implements an Adaptive Epistemic Trust system with a JEPA-style world model for business event prediction. The review covered dependency vulnerabilities, code security issues, configuration security, authentication/authorization, and data handling practices.

**Overall Risk Assessment: MEDIUM**

The codebase demonstrates several security-conscious design patterns, including:
- Built-in sensitive data detection (SSN, passwords, API keys, private keys)
- Immutable safety constraints that cannot be bypassed
- CI pipeline with Trivy vulnerability scanning, CodeQL analysis, and dependency review
- Input validation via Zod schemas in the MCP server

However, several security concerns were identified that should be addressed before production deployment.

---

## Findings Summary

| Severity | Count | Description |
|----------|-------|-------------|
| Critical | 0 | None identified |
| High | 2 | Insecure deserialization, No authentication on inference server |
| Medium | 4 | SQL query patterns, Environment variable handling, Network exposure, Logging considerations |
| Low | 3 | Dependency versions, Debug mode, Error handling |
| Informational | 3 | CI/CD security, Design patterns, Missing configurations |

---

## Detailed Findings

### HIGH-1: Insecure Deserialization with `torch.load(weights_only=False)`

**Severity:** HIGH
**Location:** Multiple files
**CWE:** CWE-502 (Deserialization of Untrusted Data)

**Description:**
The codebase uses `torch.load()` with `weights_only=False` in 16+ locations. This setting allows arbitrary Python code execution via pickle deserialization, which is a known security risk if checkpoint files come from untrusted sources.

**Affected Files:**
```
core/training/train.py:511
core/inference/server.py:194
core/tests/test_training.py:236, 268
core/tests/test_integration.py:424, 1148
scripts/benchmark_bpi2018.py:44
scripts/benchmark_wearable_tracker.py:43
scripts/benchmark_sap_workflow.py:42
scripts/benchmark_vocab_normalization_prototype.py:45
scripts/benchmark_vocab_aware_floor.py:40
scripts/benchmark_sap_bsp669.py:47
scripts/benchmark_judicial.py:44
scripts/benchmark_netsuite_2025.py:45
scripts/benchmark_vocab_normalization_v2.py:41
scripts/benchmark_road_traffic.py:42
```

**Vulnerable Pattern:**
```python
# core/training/train.py:511
checkpoint = torch.load(path, map_location=self.device, weights_only=False)

# core/inference/server.py:194-195
checkpoint = torch.load(
    checkpoint_path, map_location=self.device, weights_only=False
)
```

**Risk:**
An attacker who can supply a malicious checkpoint file could achieve arbitrary code execution on the server or training machine.

**Remediation:**
1. Use `weights_only=True` (PyTorch 2.0+ default) where possible
2. Implement checkpoint signature verification
3. Only load checkpoints from trusted, controlled locations
4. Consider using SafeTensors format for model weights

---

### HIGH-2: No Authentication on Python Inference Server

**Severity:** HIGH
**Location:** `core/inference/server.py`
**CWE:** CWE-306 (Missing Authentication for Critical Function)

**Description:**
The FastAPI inference server exposes prediction endpoints without any authentication or authorization mechanisms. Anyone with network access to port 8712 can make predictions.

**Vulnerable Pattern:**
```python
# core/inference/server.py:340
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    # No authentication check
    ...
```

**Risk:**
- Unauthorized access to ML predictions
- Potential DoS via excessive prediction requests
- Information disclosure about model capabilities

**Remediation:**
1. Implement API key authentication
2. Add rate limiting
3. Restrict network binding to localhost (0.0.0.0 is not used, which is good)
4. Add request validation and size limits
5. Consider mutual TLS for production deployments

---

### MEDIUM-1: SQL Query Construction (Potential Injection)

**Severity:** MEDIUM
**Location:** `core/data/sap_extractor.py`, `core/data/unified_pipeline.py`
**CWE:** CWE-89 (SQL Injection)

**Description:**
The SAP data extractors use parameterized queries correctly in most places, but there are some patterns that construct queries using LIKE with user-derived values and f-string table name interpolation.

**Pattern of Concern:**
```python
# core/data/sap_extractor.py:472-476
awkey_check = conn.execute(
    "SELECT belnr FROM bkpf WHERE awtyp='VBRK' AND awkey LIKE ? AND belnr=?",
    (f"{inv_num}%", belnr)  # inv_num from document flow data
)

# core/data/unified_pipeline.py:1009
cursor.execute(
    f'SELECT ocel_id, ocel_time, resource FROM "{table_name}"'  # table_name from event type map
)
```

**Mitigating Factors:**
- Data comes from internal SAP database extractions, not user input
- Read-only database connections are used where possible
- The code uses parameterized queries for most values

**Risk:**
If the underlying SAP data is compromised or if these functions are used with different data sources, SQL injection could occur.

**Remediation:**
1. Validate table names against an allowlist
2. Use parameterized queries consistently
3. Add input sanitization for any user-controllable values

---

### MEDIUM-2: Environment Variable Handling

**Severity:** MEDIUM
**Location:** Multiple files
**CWE:** CWE-526 (Exposure of Sensitive Information Through Environmental Variables)

**Description:**
Environment variables are used for configuration, but there's no validation or secure default handling.

**Affected Locations:**
```python
# core/data/bpi2019_parser.py:34-37
DEFAULT_BPI2019_PATH: Path | None = (
    Path(os.environ["AETHER_BPI2019_PATH"])
    if os.environ.get("AETHER_BPI2019_PATH")
    else None
)

# mcp-server/src/bridge/python-bridge.ts:30
url: process.env['AETHER_PYTHON_URL'] ?? DEFAULT_PYTHON_URL,
```

**Risk:**
- Environment variables could be logged or exposed
- No validation that URLs are properly formatted
- Default fallback to localhost could mask configuration issues

**Remediation:**
1. Validate environment variable values (URL format, path existence)
2. Log when using default values (at DEBUG level only)
3. Consider using a configuration management library

---

### MEDIUM-3: Inference Server Network Exposure

**Severity:** MEDIUM
**Location:** `core/inference/server.py`, `mcp-server/src/bridge/python-bridge.ts`
**CWE:** CWE-200 (Exposure of Sensitive Information)

**Description:**
The inference server defaults to localhost:8712, but the configuration could expose it to the network if misconfigured.

**Current Configuration:**
```typescript
// mcp-server/src/bridge/python-bridge.ts:20
const DEFAULT_PYTHON_URL = 'http://localhost:8712';
```

**Positive Findings:**
- Server binds to localhost by default (not 0.0.0.0)
- Communication is local-only in the default configuration

**Risk:**
If deployed in a containerized environment or with incorrect network configuration, the server could be exposed.

**Remediation:**
1. Add HTTPS/TLS support for production
2. Implement request source validation
3. Document secure deployment configurations

---

### MEDIUM-4: Logging Considerations

**Severity:** MEDIUM
**Location:** Multiple files
**CWE:** CWE-532 (Insertion of Sensitive Information into Log File)

**Description:**
While no PII logging was identified, the logging patterns could potentially expose sensitive information in production.

**Patterns Observed:**
```python
# core/training/train.py:168
logger.info(
    f"Epoch {self._epoch}: "
    + ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
)

# core/inference/server.py:297-300
logger.info(
    f"Model loaded: {checkpoint_path} "
    f"({activity_vocab.size} activities, {resource_vocab.size} resources)"
)
```

**Positive Findings:**
- No customer data, case IDs, or PII are logged
- Logs focus on metrics and system state
- Debug-level logging uses `.debug()` appropriately

**Remediation:**
1. Review log output in production for any sensitive data
2. Implement log sanitization for any user-provided data
3. Use structured logging for better security monitoring

---

### LOW-1: Dependency Version Ranges

**Severity:** LOW
**Location:** `requirements.txt`, `pyproject.toml`, `package.json`

**Description:**
Dependencies use minimum version constraints (>=) rather than pinned versions, which could lead to unexpected behavior with new releases.

**Current Configuration:**
```python
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
fastapi>=0.100.0

# package.json
"@modelcontextprotocol/sdk": "^1.0.0",
"zod": "^3.22.4"
```

**Risk:**
- Minor/patch updates could introduce vulnerabilities
- Reproducibility issues in builds

**Remediation:**
1. Pin exact versions in production deployments
2. Use lock files (package-lock.json is present)
3. Implement automated dependency updates with security scanning

---

### LOW-2: Debug Mode Exposure Potential

**Severity:** LOW
**Location:** `core/data/prepare_training_data.py`

**Description:**
The training data preparation script accepts a `--verbose` flag that enables DEBUG logging.

```python
# core/data/prepare_training_data.py:176-182
parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Enable verbose (DEBUG) logging",
)
...
log_level = logging.DEBUG if args.verbose else logging.INFO
```

**Risk:**
Debug output could reveal sensitive information if used in production.

**Remediation:**
1. Ensure DEBUG mode is never enabled in production
2. Review debug output for sensitive information

---

### LOW-3: Error Handling Information Disclosure

**Severity:** LOW
**Location:** `mcp-server/src/index.ts`

**Description:**
Error messages are returned with stack traces to the console, which could reveal implementation details.

```typescript
// mcp-server/src/index.ts:133-136
main().catch((error) => {
  console.error('AETHER server failed to start:', error);
  process.exit(1);
});
```

**Remediation:**
1. Implement structured error handling
2. Sanitize error messages for production
3. Log full errors server-side, return generic messages to clients

---

### INFO-1: Strong CI/CD Security Practices (Positive Finding)

**Severity:** INFORMATIONAL (Positive)
**Location:** `.github/workflows/ci.yml`

**Description:**
The CI pipeline includes multiple security measures:

1. **Trivy Vulnerability Scanner:**
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    severity: 'CRITICAL,HIGH,MEDIUM'
```

2. **Dependency Review:**
```yaml
- name: Dependency Review
  uses: actions/dependency-review-action@v4
  with:
    fail-on-severity: high
```

3. **CodeQL Analysis:**
```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v3
  with:
    queries: +security-extended,security-and-quality
```

**Recommendation:**
Continue maintaining these security practices and consider adding:
- SAST for Python (Bandit)
- Secret scanning with GitLeaks or TruffleHog

---

### INFO-2: Built-in Sensitive Data Detection (Positive Finding)

**Severity:** INFORMATIONAL (Positive)
**Location:** `mcp-server/src/governance/immutable.ts`

**Description:**
The governance layer includes hardcoded patterns to detect and block sensitive data:

```typescript
const SENSITIVE_PATTERNS = [
  /\b\d{3}-\d{2}-\d{4}\b/,            // SSN
  /\b\d{9}\b/,                          // SSN without dashes
  /\bpassword\s*[:=]\s*\S+/i,          // Password in text
  /\b(?:sk-|pk_|sk_live_|pk_live_)\w+/, // API keys
  /-----BEGIN (?:RSA )?PRIVATE KEY-----/, // Private keys
];
```

These patterns are checked in `checkImmutableConstraints()` and trigger automatic holds, which is a strong security-by-design pattern.

---

### INFO-3: Missing HTTPS/CORS Configuration

**Severity:** INFORMATIONAL
**Location:** `core/inference/server.py`

**Description:**
The FastAPI server does not configure CORS or HTTPS:

```python
app = FastAPI(
    title="AETHER Inference Server",
    description="JEPA-style world model for discrete business event prediction",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
# No CORS middleware
# No HTTPS configuration
```

**Recommendation:**
For production deployments:
1. Add CORS configuration if web clients will access the API
2. Deploy behind a reverse proxy with TLS termination
3. Or add direct HTTPS support with valid certificates

---

## Security Architecture Assessment

### MCP Server Access Controls

The MCP server implements access control through the governance layer:

1. **Immutable Constraints:** Cannot be bypassed regardless of trust level
2. **Adaptive Thresholds:** Tighten based on uncertainty and calibration
3. **Autonomy Levels:** Progressive trust (SUPERVISED -> AUTONOMOUS)

**Assessment:** The governance model is well-designed with defense-in-depth. The "Intrinsic Cost" (immutable constraints) approach mirrors best practices in safety-critical systems.

### Python Inference Server Security

**Current State:**
- No authentication
- HTTP-only (no TLS)
- Local binding only (good)
- Pydantic input validation (good)

**Recommendations:**
1. Add API key or token authentication
2. Implement request rate limiting
3. Add input size limits
4. Consider sandboxing model inference

### Model Checkpoint Security

**Risk:** Pickle-based checkpoint loading allows arbitrary code execution

**Recommendations:**
1. Migrate to `weights_only=True` where possible
2. Implement checkpoint signing
3. Use SafeTensors format for production models
4. Verify checkpoint integrity before loading

---

## Recommendations Summary

### Priority 1 (High - Address Immediately)
1. [ ] Switch to `weights_only=True` for `torch.load()` or implement checkpoint verification
2. [ ] Add authentication to the inference server

### Priority 2 (Medium - Address Before Production)
3. [ ] Implement HTTPS for the inference server
4. [ ] Add rate limiting to API endpoints
5. [ ] Validate table names in SQL queries against allowlist
6. [ ] Review and sanitize log output for production

### Priority 3 (Low - Best Practice Improvements)
7. [ ] Pin dependency versions for production
8. [ ] Add Bandit (Python SAST) to CI pipeline
9. [ ] Implement structured logging
10. [ ] Document secure deployment configuration

---

## Compliance Notes

### Data Protection
- Sensitive data detection is built into the governance layer
- No PII logging was identified in the codebase
- Event data handling follows reasonable practices

### Security Testing
- Trivy vulnerability scanning in CI
- CodeQL static analysis
- Dependency review for PRs

---

## Conclusion

The AETHER codebase demonstrates thoughtful security-conscious design, particularly in its governance layer with immutable constraints and sensitive data detection. The CI/CD pipeline includes appropriate security tooling.

The primary concerns are:
1. Insecure deserialization in model checkpoint loading
2. Lack of authentication on the inference server

These should be addressed before any production deployment. The remaining findings are typical of research/development codebases and can be addressed as the project matures toward production use.

---

*This audit was conducted as a static analysis review. A comprehensive security assessment would include dynamic testing, penetration testing, and deployment configuration review.*
