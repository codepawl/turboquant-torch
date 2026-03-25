# TurboQuant-Torch

## Branching Strategy

- **main** — Production branch. Only merge stable, tested code here.
- **staging** — Prototype/pre-production branch. Use for experimental features and integration testing before promoting to main.

Feature branches should be created from `staging` and merged back into `staging` via PR. Once validated, `staging` is merged into `main`.

## Project

- PyTorch-based vector quantization library
- CI runs on GitHub Actions (lint, test, security scan, build)
- Test matrix: Python 3.10–3.14
