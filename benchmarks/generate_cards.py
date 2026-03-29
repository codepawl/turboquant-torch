"""Generate visual benchmark cards (PNG) for X posts.

Card D: Real model benchmark table
Card E: Scaling chart (memory at different context lengths)
Card F: Sliding window — attention accuracy vs residual length
Card G: GQA error amplification — why keys need more bits

Usage:
    python benchmarks/generate_cards.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Style constants ───────────────────────────────────────────────────────
BG_COLOR = "#101014"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#ff6e28"
ACCENT_DIM = "#cc5820"
GREY = "#3a3a40"
GREY_LIGHT = "#555560"
HIGHLIGHT_BG = "#1a1a22"
BLUE = "#4ea8de"
CARD_W, CARD_H = 12, 6.75  # 1200x675 at 100 dpi

plt.rcParams.update(
    {
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "font.family": "monospace",
        "font.size": 13,
    }
)


def load_results():
    results_path = Path(__file__).parent / "results.json"
    with results_path.open() as f:
        return json.load(f)


def generate_card_d(data, out_path):
    """Card D: Real Model Benchmark table."""
    model_info = data["model_info"]
    real_results = [r for r in data["results"] if "real" in r["config"]]

    fig, ax = plt.subplots(figsize=(CARD_W, CARD_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(
        5,
        6.3,
        "TurboQuant on Real LLM KV Cache",
        fontsize=24,
        fontweight="bold",
        color=ACCENT,
        ha="center",
        fontfamily="sans-serif",
    )
    ax.text(
        5,
        5.8,
        f"{model_info['name']}  |  {model_info['layers']} layers  |  "
        f"{model_info['kv_heads']} KV heads  |  head_dim={model_info['head_dim']}",
        fontsize=13,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    # Table header
    cols = ["Bit-width", "Key MSE", "Value MSE", "Attn MSE", "Ratio"]
    col_x = [1.0, 2.8, 4.6, 6.4, 8.5]
    header_y = 5.1

    for x, label in zip(col_x, cols, strict=True):
        ax.text(
            x,
            header_y,
            label,
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center",
            fontfamily="sans-serif",
        )

    # Divider line
    ax.plot([0.3, 9.7], [header_y - 0.25, header_y - 0.25], color=GREY, linewidth=1)

    # Table rows
    row_y = header_y - 0.7
    for r in real_results:
        is_3bit = r["bit_width"] == 3

        # Highlight 3-bit row
        if is_3bit:
            rect = mpatches.FancyBboxPatch(
                (0.3, row_y - 0.25),
                9.4,
                0.55,
                boxstyle="round,pad=0.05",
                facecolor=HIGHLIGHT_BG,
                edgecolor=ACCENT,
                linewidth=1.5,
            )
            ax.add_patch(rect)

        row_color = ACCENT if is_3bit else TEXT_COLOR
        weight = "bold" if is_3bit else "normal"

        values = [
            f"{r['bit_width']}-bit",
            f"{r['key_mse']:.4f}",
            f"{r['val_mse']:.4f}",
            f"{r['attn_mse']:.6f}",
            f"{r['ratio']:.1f}x",
        ]

        for x, val in zip(col_x, values, strict=True):
            ax.text(
                x,
                row_y,
                val,
                fontsize=14,
                color=row_color,
                ha="center",
                fontweight=weight,
                fontfamily="monospace",
            )

        row_y -= 0.7

    # Bottom note
    ax.text(
        5,
        1.2,
        "3-bit achieves near-zero attention distortion with ~10x memory savings",
        fontsize=14,
        color=ACCENT_DIM,
        ha="center",
        fontfamily="sans-serif",
        style="italic",
    )

    ax.text(
        5,
        0.6,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    fig.savefig(out_path, dpi=100, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_card_e(data, out_path):
    """Card E: KV Cache Memory at Scale (horizontal bars)."""
    # Filter 3-bit synthetic results
    scaling_configs = [
        ("Llama-7B 2K ctx", "Llama-7B\n2K ctx"),
        ("Llama-7B 8K ctx", "Llama-7B\n8K ctx"),
        ("Llama-7B 32K ctx", "Llama-7B\n32K ctx"),
        ("Llama-70B 2K ctx", "Llama-70B\n2K ctx"),
    ]

    configs_3bit = {}
    for r in data["results"]:
        if r["bit_width"] == 3 and "real" not in r["config"]:
            configs_3bit[r["config"]] = r

    fig, ax = plt.subplots(figsize=(CARD_W, CARD_H))

    # Title
    fig.text(
        0.5,
        0.93,
        "KV Cache Memory at Scale",
        fontsize=24,
        fontweight="bold",
        color=ACCENT,
        ha="center",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.88,
        "TurboQuant 3-bit vs fp32  |  batch=1",
        fontsize=13,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    labels = []
    orig_vals = []
    comp_vals = []

    for config_key, label in scaling_configs:
        if config_key in configs_3bit:
            r = configs_3bit[config_key]
            labels.append(label)
            orig_vals.append(r["orig_mb"])
            comp_vals.append(r["comp_mb"])

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    # Original (grey)
    bars_orig = ax.barh(
        y_pos + bar_height / 2,
        orig_vals,
        bar_height,
        color=GREY,
        label="fp32 (original)",
        edgecolor=GREY_LIGHT,
        linewidth=0.5,
    )

    # Compressed (orange)
    bars_comp = ax.barh(
        y_pos - bar_height / 2,
        comp_vals,
        bar_height,
        color=ACCENT,
        label="3-bit TurboQuant",
        edgecolor=ACCENT_DIM,
        linewidth=0.5,
    )

    # Value labels
    max_val = max(orig_vals)
    for bar, val in zip(bars_orig, orig_vals, strict=True):
        label_x = val + max_val * 0.01
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} MB",
            fontsize=11,
            color=GREY_LIGHT,
            va="center",
            fontfamily="monospace",
        )

    for i, (bar, val) in enumerate(zip(bars_comp, comp_vals, strict=True)):
        ratio = configs_3bit[scaling_configs[i][0]]["ratio"]
        label_x = val + max_val * 0.01
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} MB  ({ratio:.1f}x)",
            fontsize=11,
            color=ACCENT,
            va="center",
            fontweight="bold",
            fontfamily="monospace",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13, fontfamily="sans-serif")
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_xlim(0, max_val * 1.25)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=GREY, labelsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x / 1000:.0f} GB" if x >= 1000 else f"{x:.0f} MB")
    )

    # Legend
    ax.legend(
        loc="lower right",
        fontsize=12,
        framealpha=0.3,
        facecolor=BG_COLOR,
        edgecolor=GREY,
        labelcolor=TEXT_COLOR,
    )

    # Footer
    fig.text(
        0.5,
        0.03,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    fig.savefig(out_path, dpi=100, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_card_f(out_path):
    """Card F: Sliding Window — MSE reduction vs residual length."""
    data_path = Path(__file__).parent / "sliding_window_results.json"
    with data_path.open() as f:
        data = json.load(f)

    # Use synthetic results (meaningful trend, real model seq too short)
    synth = [r for r in data["results"] if "synthetic" in r["config"]]
    if not synth:
        print("No synthetic sliding window results found, skipping card F")
        return

    seq_len = synth[0]["seq_len"]
    res_lengths = [r["residual_length"] for r in synth]
    out_mses = [r["output_mse"] for r in synth]
    comp_mbs = [r["comp_mb"] for r in synth]

    baseline_mse = out_mses[0]
    reductions = [0.0 if baseline_mse == 0 else (1 - m / baseline_mse) * 100 for m in out_mses]
    baseline_mb = comp_mbs[0]

    fig, (ax_bar, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(CARD_W, CARD_H),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.05},
    )

    # ── Title ──
    fig.text(
        0.5,
        0.94,
        "Sliding Window: Attention Accuracy vs Residual Length",
        fontsize=22,
        fontweight="bold",
        color=ACCENT,
        ha="center",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.89,
        f"3-bit TurboQuant  |  Llama-3-8B-like synthetic  |  seq_len={seq_len}",
        fontsize=12,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    # ── Left: Bar chart of output MSE ──
    x = np.arange(len(res_lengths))
    colors = [GREY if i == 0 else ACCENT for i in range(len(res_lengths))]
    bars = ax_bar.bar(x, out_mses, color=colors, width=0.6, edgecolor="#1a1a22", linewidth=1)

    # Labels on bars
    for i, (bar, mse, red) in enumerate(zip(bars, out_mses, reductions, strict=True)):
        y_top = bar.get_height()
        if i == 0:
            label = f"{mse:.6f}\n(baseline)"
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                y_top * 1.02,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color=GREY_LIGHT,
                fontfamily="monospace",
            )
        else:
            label = f"{mse:.6f}\n\u2212{red:.0f}%"
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                y_top * 1.02,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color=ACCENT,
                fontweight="bold",
                fontfamily="monospace",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(r) for r in res_lengths], fontsize=12)
    ax_bar.set_xlabel("residual_length", fontsize=13, fontfamily="sans-serif")
    ax_bar.set_ylabel("Attention Output MSE", fontsize=13, fontfamily="sans-serif")
    ax_bar.set_ylim(0, max(out_mses) * 1.35)
    for spine in ax_bar.spines.values():
        spine.set_color(GREY)

    # ── Right: Memory & accuracy table ──
    attn_mses = [r["attn_mse"] for r in synth]

    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, 12)
    ax_tbl.set_ylim(0, 10)

    ax_tbl.text(
        6,
        9.2,
        "Memory & Accuracy",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="center",
        fontfamily="sans-serif",
    )

    col_x = [1.2, 4.2, 7.2, 10.5]
    headers = ["Residual", "Attn MSE", "MB", "\u0394 Mem"]
    y = 8.2
    for cx, h in zip(col_x, headers, strict=True):
        ax_tbl.text(
            cx,
            y,
            h,
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center",
            fontfamily="sans-serif",
        )

    ax_tbl.plot([0.1, 11.9], [y - 0.3, y - 0.3], color=GREY, linewidth=0.8)

    y = 7.4
    for i, (rl, amse, mb) in enumerate(zip(res_lengths, attn_mses, comp_mbs, strict=True)):
        is_highlight = rl == 128
        row_color = ACCENT if is_highlight else TEXT_COLOR
        weight = "bold" if is_highlight else "normal"

        if is_highlight:
            rect = mpatches.FancyBboxPatch(
                (0.05, y - 0.3),
                11.9,
                0.65,
                boxstyle="round,pad=0.05",
                facecolor=HIGHLIGHT_BG,
                edgecolor=ACCENT,
                linewidth=1.2,
            )
            ax_tbl.add_patch(rect)

        extra_mb = mb - baseline_mb
        extra_pct = (extra_mb / baseline_mb * 100) if baseline_mb > 0 else 0

        ax_tbl.text(
            col_x[0],
            y,
            str(rl),
            fontsize=11,
            color=row_color,
            ha="center",
            fontweight=weight,
            fontfamily="monospace",
        )
        ax_tbl.text(
            col_x[1],
            y,
            f"{amse:.2e}",
            fontsize=11,
            color=row_color,
            ha="center",
            fontweight=weight,
            fontfamily="monospace",
        )
        ax_tbl.text(
            col_x[2],
            y,
            f"{mb:.1f}",
            fontsize=11,
            color=row_color,
            ha="center",
            fontweight=weight,
            fontfamily="monospace",
        )
        delta_str = "\u2014" if i == 0 else f"+{extra_pct:.0f}%"
        ax_tbl.text(
            col_x[3],
            y,
            delta_str,
            fontsize=11,
            color=row_color,
            ha="center",
            fontweight=weight,
            fontfamily="monospace",
        )
        y -= 0.9

    # ── Key insight ──
    best_idx = 3  # residual_length=128
    if best_idx < len(reductions):
        extra_pct = (
            ((comp_mbs[best_idx] - baseline_mb) / baseline_mb * 100) if baseline_mb > 0 else 0
        )
        fig.text(
            0.5,
            0.04,
            f"residual_length=128 reduces attention error by {reductions[best_idx]:.0f}%"
            f" with only {extra_pct:.0f}% more memory",
            fontsize=14,
            color=ACCENT_DIM,
            ha="center",
            fontfamily="sans-serif",
            style="italic",
        )

    fig.text(
        0.5,
        -0.02,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=100, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_card_g(out_path):
    """Card G: GQA Error Amplification — bar chart + ratio sweep."""
    data_path = Path(__file__).parent / "gqa_results.json"
    with data_path.open() as f:
        data = json.load(f)

    results_r4 = [r for r in data["results"] if r["gqa_ratio"] == 4]
    if not results_r4:
        print("No GQA ratio=4 results found, skipping card G")
        return

    configs = [r["config"] for r in results_r4]
    attn_mses = [r["attn_mse"] for r in results_r4]
    out_mses = [r["output_mse"] for r in results_r4]

    baseline_configs = [r for r in data["results"] if r["key_bits"] == 3 and r["value_bits"] == 3]
    gqa_aware_configs = [r for r in data["results"] if r["key_bits"] == 4 and r["value_bits"] == 3]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(CARD_W, CARD_H),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
    )

    fig.text(
        0.5,
        0.94,
        "GQA Error Amplification: Why Keys Need More Bits",
        fontsize=22,
        fontweight="bold",
        color=ACCENT,
        ha="center",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.90,
        "Simulated Llama-3-style GQA (8 KV heads \u2192 32 query heads)"
        "  |  head_dim=128  |  seq_len=256",
        fontsize=11,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    # ── Left: Output MSE bars (Attn MSE ~1000x smaller, shown in annotation) ──
    short_labels = ["3-bit K/V", "4-bit K / 3-bit V\n(GQA-aware)", "4-bit K/V"]
    x = np.arange(len(configs))
    bar_colors = [GREY, ACCENT, ACCENT_DIM]

    bars_out = ax_left.bar(
        x,
        out_mses,
        0.55,
        color=bar_colors,
        edgecolor="#1a1a22",
        linewidth=1,
    )

    baseline_out = out_mses[0]
    for i, (bar, mse) in enumerate(zip(bars_out, out_mses, strict=True)):
        if i == 0:
            label = f"{mse:.6f}\n(baseline)"
            color = GREY_LIGHT
            weight = "normal"
        else:
            pct = (1 - mse / baseline_out) * 100
            label = f"{mse:.6f}\n\u2212{pct:.0f}%"
            color = ACCENT
            weight = "bold"
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight=weight,
            fontfamily="monospace",
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(short_labels, fontsize=11, fontfamily="sans-serif")
    ax_left.set_ylabel("Output MSE", fontsize=13, fontfamily="sans-serif")
    ax_left.set_ylim(0, max(out_mses) * 1.4)
    for spine in ax_left.spines.values():
        spine.set_color(GREY)
    ax_left.set_title(
        "GQA Ratio = 4 (Llama-3-like synthetic)",
        fontsize=13,
        color=TEXT_COLOR,
        pad=8,
    )

    # ── Attn Score MSE annotation ──
    baseline_attn = attn_mses[0]
    parts = []
    for lbl, amse in zip(["3-bit K/V", "4K/3V", "4-bit K/V"], attn_mses, strict=True):
        if amse == baseline_attn:
            parts.append(f"{lbl}: {amse:.2e}")
        else:
            pct = (1 - amse / baseline_attn) * 100
            parts.append(f"{lbl}: {amse:.2e} (\u2212{pct:.0f}%)")
    ax_left.text(
        0.5,
        -0.17,
        "Attn Score MSE   " + "   |   ".join(parts),
        fontsize=8,
        color=BLUE,
        ha="center",
        transform=ax_left.transAxes,
        fontfamily="monospace",
    )

    # ── Right: Output MSE across GQA ratios ──
    bl_mses = [r["output_mse"] for r in baseline_configs]
    ga_mses = [r["output_mse"] for r in gqa_aware_configs]

    x2 = np.arange(len(baseline_configs))
    ax_right.bar(
        x2 - 0.2,
        bl_mses,
        0.35,
        color=GREY,
        label="3-bit K/V",
        edgecolor="#1a1a22",
        linewidth=1,
    )
    ax_right.bar(
        x2 + 0.2,
        ga_mses,
        0.35,
        color=ACCENT,
        label="4-bit K / 3-bit V",
        edgecolor="#1a1a22",
        linewidth=1,
    )

    ratio_labels = [r["gqa_ratio_label"] for r in baseline_configs]
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(ratio_labels, fontsize=10, fontfamily="monospace")
    ax_right.set_xlabel("GQA Ratio", fontsize=13, fontfamily="sans-serif")
    ax_right.set_ylabel("Output MSE", fontsize=13, fontfamily="sans-serif")
    ax_right.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.3,
        facecolor=BG_COLOR,
        edgecolor=GREY,
        labelcolor=TEXT_COLOR,
    )
    for spine in ax_right.spines.values():
        spine.set_color(GREY)
    ax_right.set_title("Across GQA Ratios", fontsize=13, color=TEXT_COLOR, pad=8)

    # ── Key insight ──
    pct_reduction = (1 - out_mses[1] / out_mses[0]) * 100 if out_mses[0] > 0 else 0
    fig.text(
        0.5,
        0.04,
        f"Bumping keys to 4-bit reduces GQA attention error by {pct_reduction:.0f}%"
        f" (only keys, values stay 3-bit)",
        fontsize=14,
        color=ACCENT_DIM,
        ha="center",
        fontfamily="sans-serif",
        style="italic",
    )

    fig.text(
        0.5,
        -0.02,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11,
        color=GREY_LIGHT,
        ha="center",
        fontfamily="monospace",
    )

    fig.subplots_adjust(bottom=0.20, top=0.82)
    fig.savefig(out_path, dpi=100, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    data = load_results()
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    generate_card_d(data, assets_dir / "card-d-benchmark.png")
    generate_card_e(data, assets_dir / "card-e-scaling.png")
    generate_card_f(assets_dir / "card-f-sliding-window.png")
    generate_card_g(assets_dir / "card-g-gqa.png")


if __name__ == "__main__":
    main()
