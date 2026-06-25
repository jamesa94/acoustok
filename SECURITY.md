# Security Policy

## Supported versions

`acoustok` is pre-1.0. Security fixes are applied to the latest released minor
version on a best-effort basis.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | ✅        |
| < 0.3   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private vulnerability reporting
([Security → Report a vulnerability](https://github.com/jamesa94/acoustok/security/advisories/new))
so the report stays confidential until a fix is available.

Include:

- a description of the issue and its impact,
- a minimal reproduction,
- the affected version(s).

You can expect an initial response within a few days. Thanks for helping keep the
project and its users safe.

## Scope notes

`acoustok` loads model checkpoints with `torch.load`. Only load checkpoints from
sources you trust — a malicious checkpoint can execute arbitrary code during
unpickling. This is a property of PyTorch serialization, not a bug in this
library, but it is worth repeating.
