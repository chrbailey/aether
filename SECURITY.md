# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.x.x   | :white_check_mark: |
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to the repository maintainer
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 7 days
  - Medium: 30 days
  - Low: Next release

### Security Measures

This project implements the following security measures:

#### Automated Scanning

- **CodeQL**: Static analysis for JavaScript/TypeScript and Python
- **Trivy**: Vulnerability scanning for dependencies and containers
- **Dependabot**: Automated dependency updates
- **Dependency Review**: PR checks for vulnerable dependencies

#### Code Security

- **No hardcoded secrets**: All sensitive data via environment variables
- **Input validation**: All external inputs are validated
- **Safe deserialization**: `torch.load` uses `weights_only=True` where possible
- **CORS protection**: Configurable origin restrictions

#### Infrastructure

- **HTTPS recommended**: For production deployments
- **Environment isolation**: Test mode available via `AETHER_TEST_MODE`
- **Audit logging**: All governance decisions are logged with audit IDs

### Known Security Considerations

#### Model Checkpoints

PyTorch model checkpoints (`.pt` files) can contain arbitrary code. Only load checkpoints from trusted sources. The training scripts in this repository generate safe checkpoints.

#### MCP Server

The MCP server is designed to run locally. If exposed to a network:
- Use a reverse proxy with authentication
- Enable HTTPS
- Restrict CORS origins

#### Event Log Data

Process mining event logs may contain PII. The system does not automatically anonymize data. Users are responsible for:
- Anonymizing event logs before processing
- Complying with data protection regulations (GDPR, CCPA, etc.)

### Security Best Practices for Users

1. **Keep dependencies updated**: Run `npm audit` and `pip audit` regularly
2. **Use environment variables**: Never hardcode API keys or secrets
3. **Review model sources**: Only use checkpoints from trusted sources
4. **Anonymize data**: Remove PII from event logs before processing
5. **Network security**: Don't expose the inference server to public networks

## Security Updates

Security updates are released as patch versions (e.g., 3.0.1) and announced via:
- GitHub Security Advisories
- Release notes
- README updates

## Compliance

This project follows:
- OWASP Top 10 guidelines
- CWE/SANS Top 25 awareness
- Secure coding practices for Python and TypeScript
