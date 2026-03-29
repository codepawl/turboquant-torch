"""Downstream task evaluation: fp16 vs TurboQuant 3-bit KV cache.

Evaluates actual task accuracy (not just MSE/perplexity) with and without
TurboQuant KV cache compression.  Runs on a single GPU with Qwen3.5-4B.

Tasks:
    HellaSwag  — commonsense reasoning, 4-way multiple choice
    ARC-Easy   — science QA, multiple choice

Outputs:
    benchmarks/downstream_results.json
    stdout summary table

Usage:
    pip install datasets transformers
    python benchmarks/bench_downstream.py
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit("Install: pip install transformers datasets") from exc

from turboquant import TurboQuantKVCache

# Models to try, in order of preference
MODEL_CHAIN = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-135M",
]

N_SAMPLES = 200  # per task — set to 50 for quick test runs


@dataclass
class TaskResult:
    task: str
    n_samples: int
    fp16_accuracy: float
    tq_accuracy: float
    diff: float
    fp16_time: float
    tq_time: float


# ── Model loading ────────────────────────────────────────────────────────


def load_model():
    """Load model with GPU auto-detection and fallback chain."""
    if torch.cuda.is_available():
        device_map: str = "auto"
        dtype = torch.float16
        print(f"  GPU: {torch.cuda.get_device_name()}")
    else:
        device_map = "cpu"
        dtype = torch.float32
        print("  Using CPU")

    for name in MODEL_CHAIN:
        try:
            print(f"  Loading {name}...")
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                trust_remote_code=True,
                dtype=dtype,
                device_map=device_map,
            )
            model.eval()
            device = next(model.parameters()).device
            print(f"  Loaded {name} on {device}")
            return model, tokenizer, name
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on GPU: {name}")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed: {exc}")

    raise RuntimeError("No model could be loaded")


def get_model_config(model):
    """Extract head_dim, KV heads, and layer count."""
    cfg = getattr(model.config, "text_config", model.config)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return head_dim, kv_heads, cfg.num_hidden_layers


# ── KV cache compression ────────────────────────────────────────────────


def _compress_kv_cache(past_kv, compressor):
    """Compress KV cache via TurboQuant, handling hybrid models (None layers).

    Uses deepcopy so the original cache is not modified.  Decompressed tensors
    are cast back to the original dtype (e.g. float16).
    """
    result = copy.deepcopy(past_kv)
    for i in range(len(result.key_cache)):
        k = result.key_cache[i]
        if k is None:
            continue
        v = result.value_cache[i]
        dtype = k.dtype
        compressed = compressor.compress(k, v)
        result.key_cache[i] = compressor.decompress_keys(compressed).to(dtype)
        result.value_cache[i] = compressor.decompress_values(compressed).to(dtype)
    return result


# ── Scoring ──────────────────────────────────────────────────────────────


def score_choices(model, tokenizer, prompt, choices, compressor=None):
    """Score multiple-choice options by length-normalised log-likelihood.

    For each choice, computes  log P(choice | prompt)  using teacher forcing.
    The prompt KV cache is computed once and optionally compressed; each choice
    continuation is scored against it.

    The first choice token is always scored from the prompt's last-position
    logits (identical for both fp16 and compressed paths) so the comparison
    is fair — compression effects are captured in the subsequent tokens where
    the model actually attends to the (possibly compressed) KV cache.

    Returns the index of the highest-scoring choice.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        prompt_out = model(prompt_ids, use_cache=True)
        base_kv = prompt_out.past_key_values

        if compressor is not None:
            base_kv = _compress_kv_cache(base_kv, compressor)

        # Logits for predicting the first choice token — same for both paths
        first_logits = torch.log_softmax(prompt_out.logits[0, -1], dim=-1)

        scores: list[float] = []
        for choice in choices:
            choice_ids = tokenizer.encode(choice, add_special_tokens=False, return_tensors="pt").to(
                device
            )
            n_tok = choice_ids.shape[1]

            if n_tok == 0:
                scores.append(float("-inf"))
                continue

            kv_copy = copy.deepcopy(base_kv)
            out = model(choice_ids, past_key_values=kv_copy, use_cache=False)

            # Token 0: scored from prompt's last-position logits
            lp = first_logits[choice_ids[0, 0]].item()

            # Tokens 1+: out.logits[0, j-1] predicts choice_ids[0, j]
            if n_tok > 1:
                log_probs = torch.log_softmax(out.logits[0, :-1], dim=-1)
                targets = choice_ids[0, 1:]
                lp += log_probs[torch.arange(len(targets)), targets].sum().item()

            scores.append(lp / n_tok)  # length-normalised

    return scores.index(max(scores))


# ── Task evaluators ──────────────────────────────────────────────────────


def eval_hellaswag(model, tokenizer, compressor=None, n_samples=N_SAMPLES):
    """Evaluate on HellaSwag (commonsense reasoning, 4-way MC)."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    total = min(n_samples, len(ds))
    correct = 0

    for i in range(total):
        s = ds[i]
        pred = score_choices(model, tokenizer, s["ctx"], s["endings"], compressor)
        if pred == int(s["label"]):
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"    HellaSwag: {i + 1}/{total} ({correct / (i + 1) * 100:.1f}%)")

    return correct / total


def eval_arc_easy(model, tokenizer, compressor=None, n_samples=N_SAMPLES):
    """Evaluate on ARC-Easy (science QA, multiple choice)."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    total = min(n_samples, len(ds))
    correct = 0

    for i in range(total):
        s = ds[i]
        label_key = s["answerKey"]
        label = int(label_key) - 1 if label_key.isdigit() else ord(label_key) - ord("A")
        prompt = f"Question: {s['question']}\nAnswer:"
        pred = score_choices(model, tokenizer, prompt, s["choices"]["text"], compressor)
        if pred == label:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"    ARC-Easy: {i + 1}/{total} ({correct / (i + 1) * 100:.1f}%)")

    return correct / total


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  TurboQuant Downstream Task Benchmark")
    print("  fp16 baseline vs 3-bit compressed KV cache")
    print("=" * 60)

    model, tokenizer, model_name = load_model()
    head_dim, kv_heads, n_layers = get_model_config(model)
    bit_width = 3 if head_dim >= 128 else 4

    print(f"\n  Model: {model_name}")
    print(f"  head_dim={head_dim}, KV heads={kv_heads}, layers={n_layers}")
    print(f"  Quantization: {bit_width}-bit, residual_length=0")

    device = next(model.parameters()).device
    cache = TurboQuantKVCache(head_dim=head_dim, bit_width=bit_width, residual_length=0).to(device)

    tasks = [
        ("HellaSwag", eval_hellaswag),
        ("ARC-Easy", eval_arc_easy),
    ]

    results: list[TaskResult] = []
    for task_name, eval_fn in tasks:
        print(f"\n  [{task_name}] fp16 baseline...")
        t0 = time.time()
        fp16_acc = eval_fn(model, tokenizer, compressor=None, n_samples=N_SAMPLES)
        fp16_time = time.time() - t0

        print(f"  [{task_name}] {bit_width}-bit TurboQuant...")
        t0 = time.time()
        tq_acc = eval_fn(model, tokenizer, compressor=cache, n_samples=N_SAMPLES)
        tq_time = time.time() - t0

        diff = tq_acc - fp16_acc
        results.append(
            TaskResult(
                task=task_name,
                n_samples=N_SAMPLES,
                fp16_accuracy=fp16_acc,
                tq_accuracy=tq_acc,
                diff=diff,
                fp16_time=fp16_time,
                tq_time=tq_time,
            )
        )
        print(
            f"  [{task_name}] fp16={fp16_acc * 100:.1f}% | "
            f"{bit_width}-bit={tq_acc * 100:.1f}% | diff={diff * 100:+.1f}%"
        )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Task':<15} {'fp16':>8} {f'{bit_width}-bit':>8} {'Diff':>8}")
    print(f"  {'─' * 41}")
    for r in results:
        print(
            f"  {r.task:<15} {r.fp16_accuracy * 100:>7.1f}% "
            f"{r.tq_accuracy * 100:>7.1f}% {r.diff * 100:>+7.1f}%"
        )
    avg_diff = sum(r.diff for r in results) / len(results)
    print(f"  {'─' * 41}")
    print(f"  {'Average':<15} {'':>8} {'':>8} {avg_diff * 100:>+7.1f}%")

    # ── Save ──
    output = {
        "model": model_name,
        "head_dim": head_dim,
        "bit_width": bit_width,
        "n_samples": N_SAMPLES,
        "results": [
            {
                "task": r.task,
                "fp16_accuracy": r.fp16_accuracy,
                "tq_accuracy": r.tq_accuracy,
                "diff_pct": r.diff * 100,
                "fp16_time_s": round(r.fp16_time, 1),
                "tq_time_s": round(r.tq_time, 1),
            }
            for r in results
        ],
    }
    out_path = Path("benchmarks/downstream_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
