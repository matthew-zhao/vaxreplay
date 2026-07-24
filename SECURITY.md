# Security policy

## Supported versions

VaxReplay has no supported production release. The planned `0.1.0-alpha.1` release is a research
technical preview. Its isolation and execution components have not received an independent
security audit and must not be treated as a qualified sandbox for hostile code or regulated data.

## Reporting a vulnerability

Use the repository's
[private vulnerability reporting](https://github.com/matthew-zhao/vaxreplay/security/advisories/new)
channel. Do not include credentials, private benchmark tasks, customer data, or working exploit
details in a public issue.

If private vulnerability reporting is unavailable, contact the repository owner through the
[GitHub profile](https://github.com/matthew-zhao) before sharing sensitive details. This
pre-alpha project provides best-effort response and does not yet offer a security-response SLA.

## Scope

Useful reports include:

- escape from an intended worker or provider boundary;
- exposure of private gold, organizer mappings, credentials, or customer outputs;
- bypass of one-attempt or authenticated-submission controls;
- artifact-integrity or signature-verification failures;
- unsafe archive extraction or path traversal;
- secret leakage through logs, receipts, or model-facing workspaces; and
- vulnerabilities in the public-preview export process.

Reports about a live customer pilot are governed by that pilot's separate incident-response and
notification terms.
