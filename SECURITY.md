# Security Policy

## Reporting a Vulnerability

The project team takes the security of this repository and research codebase seriously.

If you believe you have discovered a vulnerability, security bug, or exposed credential:

1. **Do Not Open a Public Issue**: Please refrain from publicly reporting the issue through GitHub Issues or Discussions.
2. **Private Notification**: Contact the repository owner or team leads directly:
   - Repository Owner: [jovialguard025](https://github.com/jovialguard025)
   - Contributor Team: [mayukh437](https://github.com/mayukh437), [KOUSTAV-04](https://github.com/KOUSTAV-04), [KousambiGhosh](https://github.com/KousambiGhosh)
3. **Details to Include**:
   - Description of the suspected vulnerability.
   - Steps to reproduce or proof-of-concept.
   - Any potential impact on data integrity, model confidentiality, or private API keys.

## Security Practices for Contributors
- **Never commit API keys or credentials**: Ensure `.env` or configuration secrets remain in your local environment and are ignored by `.gitignore`.
- **Sanitize experimental data**: Only upload sanitized outputs and public datasets to avoid leaking proprietary, private, or copyrighted assets.
- If a secret or sensitive token is accidentally pushed, immediately revoke the secret and notify the repository owner to scrub the git history.
