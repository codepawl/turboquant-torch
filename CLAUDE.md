# TurboQuant-Torch

## Branching Strategy

- **main** — Production branch. Only merge stable, tested code here.
- **staging** — Prototype/pre-production branch. Use for experimental features and integration testing before promoting to main.

Feature branches should be created from `staging` and merged back into `staging` via PR. Once validated, `staging` is merged into `main`.

## Versioning

- Uses **setuptools-scm** — version is derived from git tags, no manual edits needed.
- Tag format: `v{MAJOR}.{MINOR}.{PATCH}` (SemVer)
  - PATCH: bug fixes, docs, CI improvements
  - MINOR: new features, no breaking changes
  - MAJOR: stable API or breaking changes
- Release flow: `git tag vX.Y.Z && git push origin vX.Y.Z` — CI handles build + TestPyPI + PyPI.

## Project

- PyTorch-based vector quantization library
- CI runs on GitHub Actions with uv (lint, test, security scan, build)
- Test matrix: Python 3.10–3.14
- Publish workflow triggers on `v*.*.*` tags
