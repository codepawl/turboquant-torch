# Contributing to turboquant-torch

Thanks for your interest in contributing!

## Development Setup
```bash
git clone https://github.com/codepawl/turboquant-torch.git
cd turboquant-torch
git checkout staging
pip install -e ".[dev]"
```

## Branching Strategy

- `main` — stable releases only, tagged versions
- `staging` — integration branch, PRs go here
- Feature branches — `feat/your-feature` from staging

## Code Quality

All PRs must pass:
```bash
ruff check turboquant/ tests/ examples/
ruff format --check turboquant/ tests/ examples/
mypy turboquant/
pytest tests/ -v
```

## Adding a New Feature

1. Create a branch: `git checkout -b feat/my-feature staging`
2. Write code + tests
3. Update README if needed
4. Open PR against `staging`

## Running Benchmarks
```bash
python benchmarks/bench_downstream.py   # task accuracy
python benchmarks/bench_sliding_window.py  # sliding window
python benchmarks/bench_gqa.py          # GQA analysis
```

## Code Style

- Use `bit_width` (not `bits`) for quantization width parameters
- Type hints on all public methods
- Docstrings with Args/Returns
- Reference paper sections where relevant

## Reporting Issues

- Bug reports: include Python version, torch version, error traceback
- Feature requests: describe the use case
- Security issues: email security@codepawl.com
