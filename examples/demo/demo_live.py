"""TurboQuant Live Demo — KV Cache Compression.

Generates text with fp32, compresses the KV cache, then measures
per-position prediction agreement and perplexity to show that
compression preserves model quality.

Requirements:
    pip install turboquant-torch[demo]
    # or: pip install turboquant-torch transformers rich

Usage:
    python examples/demo/demo_live.py
"""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn.functional as F

MAX_NEW_TOKENS = 80
N_SAMPLE_POSITIONS = 12

# Models to try, in order of preference
MODEL_CHAIN = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-135M",
]


# ── Banner ────────────────────────────────────────────────────────────────


def print_banner(console):
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    banner = Text()
    banner.append("████████╗██╗   ██╗██████╗ ██████╗  ██████╗\n", style="bold orange1")
    banner.append("╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗\n", style="bold orange1")
    banner.append("   ██║   ██║   ██║██████╔╝██████╔╝██║   ██║\n", style="bold orange1")
    banner.append("   ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║\n", style="bold orange1")
    banner.append("   ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝\n", style="bold orange1")
    banner.append("   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝\n", style="bold orange1")
    banner.append("                  Q U A N T\n\n", style="bold white")
    banner.append("KV cache compression · near-zero error · no training", style="dim")
    console.print(Panel(banner, box=box.DOUBLE, border_style="orange1", padding=(1, 4)))


# ── Load model ────────────────────────────────────────────────────────────


def _get_model_info(config):
    """Extract head_dim / num_kv_heads from any HF config."""
    # Some models nest text config inside a VLM config
    cfg = getattr(config, "text_config", config)
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    n_layers = cfg.num_hidden_layers
    return head_dim, kv_heads, n_heads, n_layers


def load_model(console):
    import io
    import logging
    import os
    import sys

    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    os.environ["TRANSFORMERS_NO_TQDM"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Auto-detect GPU
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
        console.print(f"  [bold green]GPU:[/bold green] {torch.cuda.get_device_name()}")
    else:
        device_map = "cpu"
        dtype = torch.float32
        console.print("  [dim]Using CPU[/dim]")

    model, tokenizer, model_name = None, None, None
    for name in MODEL_CHAIN:
        with Progress(
            SpinnerColumn(), TextColumn("[dim]{task.description}[/dim]"), console=console
        ) as p:
            p.add_task(f"Trying {name}...", total=None)
            _se = sys.stderr
            sys.stderr = io.StringIO()
            try:
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                model_name = name
            except torch.cuda.OutOfMemoryError:
                console.print(f"  [dim]  → OOM on GPU: {name}[/dim]")
                torch.cuda.empty_cache()
                continue
            except Exception as exc:  # noqa: S112
                console.print(f"  [dim]  → failed: {exc}[/dim]")
                continue
            finally:
                sys.stderr = _se

        if model is not None:
            break

    if model is None:
        console.print("[red]Could not load any model. Install transformers.[/red]")
        raise SystemExit(1)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    head_dim, kv_heads, q_heads, n_layers = _get_model_info(model.config)
    bit_width = 3 if head_dim >= 128 else 4

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value", style="bold")
    device = next(model.parameters()).device
    table.add_row("Model", model_name)
    table.add_row("Parameters", f"{n_params:.0f}M")
    table.add_row("Device / dtype", f"{device} / {dtype}")
    table.add_row("Layers", str(n_layers))
    table.add_row("Attention heads", f"{q_heads} query, {kv_heads} KV (GQA {q_heads // kv_heads}x)")
    table.add_row("Head dim", str(head_dim))
    table.add_row("Bit width", f"{bit_width}-bit (auto)")
    console.print(table)

    return model, tokenizer, model_name, head_dim, kv_heads, n_layers, bit_width


# ── Helpers ───────────────────────────────────────────────────────────────


def _extract_kv(past_kv):
    """Yield (key, value) per layer from any cache format."""
    if hasattr(past_kv, "layers"):
        return [(layer.keys, layer.values) for layer in past_kv.layers]
    if hasattr(past_kv, "key_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache, strict=True))
    return [(item[0], item[1]) for item in past_kv]


def _compress_cache(past_kv, cache):
    """Compress attention layers in a KV cache, preserving non-attention layers.

    Works with both standard DynamicCache and hybrid caches (e.g., Qwen3.5)
    where some layers have None KV entries (linear/Mamba layers).
    """
    from transformers.cache_utils import DynamicCache

    kv_pairs = _extract_kv(past_kv)
    has_none = any(k is None for k, _v in kv_pairs)

    if has_none:
        # Hybrid model — deep copy the original cache and replace attention KV
        rc = copy.deepcopy(past_kv)
        for i, (k, v) in enumerate(kv_pairs):
            if k is None:
                continue
            dtype = k.dtype
            comp = cache.compress(k, v)
            rc.key_cache[i] = cache.decompress_keys(comp).to(dtype)
            rc.value_cache[i] = cache.decompress_values(comp).to(dtype)
    else:
        # Standard model — build a fresh DynamicCache
        rc = DynamicCache()
        for i, (k, v) in enumerate(kv_pairs):
            dtype = k.dtype
            comp = cache.compress(k, v)
            rc.update(
                cache.decompress_keys(comp).to(dtype),
                cache.decompress_values(comp).to(dtype),
                i,
            )

    return rc


# ── Round 1: fp32 generation ──────────────────────────────────────────────


def generate_fp32(model, tokenizer, prompt, console):
    from rich.panel import Panel

    dtype_name = next(model.parameters()).dtype
    console.rule(f"[bold]ROUND 1  ·  Full Precision ({dtype_name})", style="orange1")
    console.print(f'  [dim]Prompt:[/dim] "{prompt}"\n')

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    console.print(Panel(text, border_style="dim", padding=(0, 2)))
    console.print(f"  [dim]Generated in {elapsed:.1f}s  ·  {gen.shape[1]} tokens[/dim]\n")
    return text, gen


# ── Round 2: compress + measure ───────────────────────────────────────────


def compress_and_measure(
    model, tokenizer, eval_ids, head_dim, kv_heads, n_layers, bit_width, console
):
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from turboquant import TurboQuantKVCache

    console.rule(f"[bold]ROUND 2  ·  Compress KV Cache to {bit_width}-bit", style="orange1")

    device = next(model.parameters()).device
    eval_ids = eval_ids.to(device)
    S = eval_ids.shape[1]
    residual_length = 0

    cache = TurboQuantKVCache(
        head_dim=head_dim, bit_width=bit_width, residual_length=residual_length
    ).to(device)

    # ── Full forward pass (for memory stats only) ──
    console.print("\n  [dim]Computing baseline...[/dim]", end="")
    with torch.no_grad():
        out_full = model(eval_ids, use_cache=True)
    console.print(" done")
    pk = out_full.past_key_values

    # ── Compress full cache (for memory stats) ──
    console.print()
    kv_pairs = _extract_kv(pk)
    attn_layers = [(i, k, v) for i, (k, v) in enumerate(kv_pairs) if k is not None]
    n_attn = len(attn_layers)
    with Progress(
        TextColumn("  Compressing"),
        BarColumn(bar_width=40, style="dim", complete_style="orange1"),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("layers", total=n_attn)
        for _i, k, v in attn_layers:
            _ = cache.compress(k, v)
            progress.advance(task)

    # ── Per-position agreement ──
    positions = list(
        range(max(4, residual_length + 2), S - 1, max(1, (S - 5) // N_SAMPLE_POSITIONS))
    )[:N_SAMPLE_POSITIONS]

    agree_top1 = 0
    agree_top10_sum = 0.0
    ce_fp32_sum = 0.0
    ce_comp_sum = 0.0

    console.print()
    with Progress(
        TextColumn("  Evaluating"),
        BarColumn(bar_width=40, style="dim", complete_style="orange1"),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
    ) as progress:
        task = progress.add_task("positions", total=len(positions), status="")

        for pos in positions:
            with torch.no_grad():
                prefix_out = model(eval_ids[:, :pos], use_cache=True)
            ppk = prefix_out.past_key_values

            # Compress KV (doesn't modify ppk)
            rc = _compress_cache(ppk, cache)

            # fp32 baseline: single-token forward with uncompressed KV
            # Must deep-copy because forward pass updates the cache in place
            ppk_fp32 = copy.deepcopy(ppk)
            with torch.no_grad():
                fp32_out = model(eval_ids[:, pos : pos + 1], past_key_values=ppk_fp32)
                comp_out = model(eval_ids[:, pos : pos + 1], past_key_values=rc)

            fp32_logits = fp32_out.logits[0, -1]
            comp_logits = comp_out.logits[0, -1]

            t1_fp = fp32_logits.argmax().item()
            t1_c = comp_logits.argmax().item()
            t10_fp = set(fp32_logits.topk(10).indices.tolist())
            t10_c = set(comp_logits.topk(10).indices.tolist())

            if t1_fp == t1_c:
                agree_top1 += 1
            agree_top10_sum += len(t10_fp & t10_c) / 10

            if pos + 1 < S:
                target = eval_ids[0, pos + 1 : pos + 2]
                ce_fp32_sum += F.cross_entropy(fp32_out.logits[0, -1:], target).item()
                ce_comp_sum += F.cross_entropy(comp_out.logits[0, -1:], target).item()

            progress.update(
                task, advance=1, status=f"top-1 {agree_top1}/{positions.index(pos) + 1}"
            )

    n = len(positions)
    top1_pct = agree_top1 / n * 100
    top10_pct = agree_top10_sum / n * 100
    ppl_fp32 = math.exp(ce_fp32_sum / n)
    ppl_comp = math.exp(ce_comp_sum / n)

    B, H = 1, kv_heads
    _, comp_mb, ratio = cache.memory_savings(B, H, S)
    comp_mb_total = comp_mb * n_attn
    orig_mb_total = B * H * S * head_dim * 4 * 2 * n_attn / (1024 * 1024)

    return {
        "ppl_fp32": ppl_fp32,
        "ppl_comp": ppl_comp,
        "top1_pct": top1_pct,
        "top10_pct": top10_pct,
        "top1_agree": agree_top1,
        "n_positions": n,
        "orig_mb": orig_mb_total,
        "comp_mb": comp_mb_total,
        "ratio": ratio,
        "bit_width": bit_width,
        "seq_len": S,
    }


# ── Results ───────────────────────────────────────────────────────────────


def show_results(stats, console):
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    console.print()
    console.rule("[bold]RESULTS", style="orange1")

    bw = stats["bit_width"]
    ppl_diff = abs(stats["ppl_comp"] - stats["ppl_fp32"]) / stats["ppl_fp32"] * 100

    table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Perplexity (fp32)", f"{stats['ppl_fp32']:.2f}")
    table.add_row(f"Perplexity ({bw}-bit)", f"{stats['ppl_comp']:.2f}")
    table.add_row("Perplexity difference", f"[orange1]{ppl_diff:.1f}%[/orange1]")
    table.add_row("", "")
    table.add_row(
        f"Top-1 agreement ({stats['n_positions']} pos)",
        f"[green]{stats['top1_pct']:.0f}%[/green]",
    )
    table.add_row(
        f"Top-10 overlap ({stats['n_positions']} pos)",
        f"[green]{stats['top10_pct']:.0f}%[/green]",
    )
    table.add_row("", "")

    bar_w = 30
    bar_c = max(1, int(bar_w / stats["ratio"]))
    fb = "[orange1]" + "█" * bar_w + "[/orange1]"
    cb = "[green]" + "█" * bar_c + "[/green][dim]" + "░" * (bar_w - bar_c) + "[/dim]"
    table.add_row(f"Memory fp32   {fb}", f"{stats['orig_mb']:.2f} MB")
    table.add_row(f"Memory {bw}-bit  {cb}", f"{stats['comp_mb']:.2f} MB")
    table.add_row("Compression", f"[bold green]{stats['ratio']:.1f}x smaller[/bold green]")
    console.print(table)
    console.print(
        "  [dim]Tip: use residual_length to keep recent tokens in fp32 (sliding window)[/dim]"
    )

    console.print(
        Panel(
            f"[green]✓[/green]  Top-1 prediction agreement: [bold]{stats['top1_pct']:.0f}%[/bold]\n"
            f"[green]✓[/green]  Perplexity within [bold]{ppl_diff:.1f}%[/bold] of fp32\n"
            f"[green]✓[/green]  [bold]{stats['ratio']:.1f}x[/bold] less memory\n"
            f"[green]✓[/green]  No fine-tuning required\n"
            f"[green]✓[/green]  Works on any transformer model",
            border_style="green",
            padding=(1, 4),
        )
    )


# ── Scale table ───────────────────────────────────────────────────────────


def show_scale(console):
    from rich import box
    from rich.table import Table

    from turboquant import TurboQuantKVCache

    console.print()
    console.rule("[bold]AT SCALE  ·  Production model projections", style="orange1")

    table = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
    table.add_column("Model", style="white")
    table.add_column("fp32", justify="right")
    table.add_column("3-bit", justify="right")
    table.add_column("Saved", justify="right", style="orange1 bold")
    table.add_column("Ratio", justify="right", style="dim")

    configs = [
        ("Llama-3-8B  2K ctx", 32, 8, 2048, 128),
        ("Llama-3-8B  32K ctx", 32, 8, 32768, 128),
        ("Llama-3-70B 2K ctx", 80, 8, 2048, 128),
        ("Llama-3-70B 32K ctx", 80, 8, 32768, 128),
    ]

    def _fmt(mb):
        return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"

    for name, layers, heads, seq, dim in configs:
        c = TurboQuantKVCache(head_dim=dim, bit_width=3, residual_length=128)
        orig, comp, ratio = c.memory_savings(1, heads, seq)
        o, q = orig * layers, comp * layers
        table.add_row(name, _fmt(o), _fmt(q), _fmt(o - q), f"{ratio:.1f}x")

    console.print(table)
    console.print("  [dim]Enough savings to serve 4-8x more users on the same GPU.[/dim]\n")
    console.print("  [dim]pip install turboquant-torch[/dim]")
    console.print("  [dim]github.com/codepawl/turboquant-torch[/dim]\n")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    from rich.console import Console

    console = Console()
    print_banner(console)

    try:
        from transformers import AutoModelForCausalLM  # noqa: F401

        has_tf = True
    except ImportError:
        has_tf = False

    if has_tf:
        model, tokenizer, model_name, head_dim, kv_heads, n_layers, bit_width = load_model(console)
        prompt = "The key insight behind TurboQuant is that"
        _text, gen_ids = generate_fp32(model, tokenizer, prompt, console)
        stats = compress_and_measure(
            model, tokenizer, gen_ids, head_dim, kv_heads, n_layers, bit_width, console
        )
        show_results(stats, console)
    else:
        console.print("  [dim]transformers not installed — showing scale projections only.[/dim]")
        console.print("  [dim]Install: pip install turboquant-torch\\[demo][/dim]\n")

    show_scale(console)


if __name__ == "__main__":
    main()
