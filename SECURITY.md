# Security Policy

MatDaemon accepts security reports through GitHub issues for now. If a vulnerability involves private exploit details, open a minimal issue requesting a private coordination channel.

## Supported Versions

The latest `main` branch and the latest tagged release are supported.

## Scope

Relevant issues include:

- unsafe file handling in CLI or API surfaces
- denial-of-service risks in API payload handling
- dependency or packaging vulnerabilities
- CUDA backend crashes that can be triggered by malformed input

Do not post secrets, credentials, or private infrastructure details in public issues.
