# Changelog

All notable changes to turboquant-torch are documented here.

## [0.4.0] — 2026-03-31

### Added
- `turboquant.wrap(model)` — one-liner API for HF model compression
- `TurboQuantDynamicCache` — drop-in HF cache replacement
- Auto-detect model config (head_dim, GQA, hybrid layers)
- `hf` optional dependency group (`pip install turboquant-torch[hf]`)
- 29 new tests (229 total)

## [0.3.0] — 2026-03-30

### Added
- Outlier channel routing (`n_outlier_channels` param)
- Adaptive per-layer bit allocation (`AdaptiveKVCache`)
- Model compatibility detection (`turboquant.compat`)
- Compatibility table (Llama, Mistral, Qwen, DeepSeek, hybrid models)
- Property-based tests (Hypothesis)
- Regression tests with pinned outputs
- Benchmark regression tracker
- CONTRIBUTING.md, SECURITY.md, GitHub templates

### Fixed
- `TurboQuant.to()` now moves QJL to device
- Standardized `bits` → `bit_width` across all modules
- `LloydMaxCodebook` parameter order fixed to `(dim, bit_width)`
- Lazy-load normal codebooks (no scipy at import)
- QJL default seed standardized to 0
- Python version aligned to >=3.10

## [0.2.2] — 2026-03-30

### Added
- Pre-RoPE key quantization (`pre_rope=True`)
- Exact Beta codebooks for dim < 256 (precomputed)
- Downstream task benchmarks (HellaSwag 37.0%→38.5%, ARC-Easy 49.0%→49.5%)
- Live demo with Qwen3.5-4B (100% top-1 agreement, 10.2x compression)
- PyPI classifiers

### Fixed
- Codebook threshold raised 64→256
- HF demo residual_length fix

## [0.2.1] — 2026-03-30

### Added
- Sliding window (`residual_length` param)
- Separate `key_bit_width` / `value_bit_width`
- GQA-aware mode (`TurboQuantKVCache.for_gqa()`)
- Sliding window + GQA benchmark cards
- 95 tests (was 78)

## [0.2.0] — 2026-03-29

### Added
- Ruff + mypy linting
- GitHub Actions CI/CD (ci.yml, publish.yml, codeql.yml)
- PyPI Trusted Publishing (OIDC)
- Dependabot, pre-commit hooks
- HuggingFace demo with SmolLM2-135M
- Benchmarks on real model + synthetic Llama-7B/70B
- Visual benchmark cards (A through E)
- 78 tests

## [0.1.0] — 2026-03-29

### Added
- Two-stage online vector quantizer (Hadamard + Lloyd-Max + QJL)
- KV cache compression (`TurboQuantKVCache`)
- Vector search (`TurboQuantIndex`)
- 64 tests
