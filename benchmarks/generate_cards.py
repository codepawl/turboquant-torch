"""Generate visual benchmark cards (PNG) for X posts.

Card D: Real model benchmark table
Card E: Scaling chart (memory at different context lengths)

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
CARD_W, CARD_H = 12, 6.75  # 1200x675 at 100 dpi

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "monospace",
    "font.size": 13,
})


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
        5, 6.3, "TurboQuant on Real LLM KV Cache",
        fontsize=24, fontweight="bold", color=ACCENT, ha="center",
        fontfamily="sans-serif",
    )
    ax.text(
        5, 5.8,
        f"{model_info['name']}  |  {model_info['layers']} layers  |  "
        f"{model_info['kv_heads']} KV heads  |  head_dim={model_info['head_dim']}",
        fontsize=13, color=GREY_LIGHT, ha="center", fontfamily="monospace",
    )

    # Table header
    cols = ["Bit-width", "Key MSE", "Value MSE", "Attn MSE", "Ratio"]
    col_x = [1.0, 2.8, 4.6, 6.4, 8.5]
    header_y = 5.1

    for x, label in zip(col_x, cols, strict=True):
        ax.text(x, header_y, label, fontsize=12, fontweight="bold",
                color=TEXT_COLOR, ha="center", fontfamily="sans-serif")

    # Divider line
    ax.plot([0.3, 9.7], [header_y - 0.25, header_y - 0.25],
            color=GREY, linewidth=1)

    # Table rows
    row_y = header_y - 0.7
    for r in real_results:
        is_3bit = r["bit_width"] == 3

        # Highlight 3-bit row
        if is_3bit:
            rect = mpatches.FancyBboxPatch(
                (0.3, row_y - 0.25), 9.4, 0.55,
                boxstyle="round,pad=0.05", facecolor=HIGHLIGHT_BG,
                edgecolor=ACCENT, linewidth=1.5,
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
            ax.text(x, row_y, val, fontsize=14, color=row_color,
                    ha="center", fontweight=weight, fontfamily="monospace")

        row_y -= 0.7

    # Bottom note
    ax.text(
        5, 1.2,
        "3-bit achieves near-zero attention distortion with ~10x memory savings",
        fontsize=14, color=ACCENT_DIM, ha="center", fontfamily="sans-serif",
        style="italic",
    )

    ax.text(
        5, 0.6,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11, color=GREY_LIGHT, ha="center", fontfamily="monospace",
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
        0.5, 0.93, "KV Cache Memory at Scale",
        fontsize=24, fontweight="bold", color=ACCENT, ha="center",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5, 0.88, "TurboQuant 3-bit vs fp32  |  batch=1",
        fontsize=13, color=GREY_LIGHT, ha="center", fontfamily="monospace",
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
        y_pos + bar_height / 2, orig_vals, bar_height,
        color=GREY, label="fp32 (original)", edgecolor=GREY_LIGHT, linewidth=0.5,
    )

    # Compressed (orange)
    bars_comp = ax.barh(
        y_pos - bar_height / 2, comp_vals, bar_height,
        color=ACCENT, label="3-bit TurboQuant", edgecolor=ACCENT_DIM, linewidth=0.5,
    )

    # Value labels
    max_val = max(orig_vals)
    for bar, val in zip(bars_orig, orig_vals, strict=True):
        label_x = val + max_val * 0.01
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} MB", fontsize=11, color=GREY_LIGHT,
                va="center", fontfamily="monospace")

    for i, (bar, val) in enumerate(zip(bars_comp, comp_vals, strict=True)):
        ratio = configs_3bit[scaling_configs[i][0]]["ratio"]
        label_x = val + max_val * 0.01
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} MB  ({ratio:.1f}x)", fontsize=11, color=ACCENT,
                va="center", fontweight="bold", fontfamily="monospace")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13, fontfamily="sans-serif")
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_xlim(0, max_val * 1.25)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=GREY, labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f} GB" if x >= 1000 else f"{x:.0f} MB"))

    # Legend
    ax.legend(
        loc="lower right", fontsize=12, framealpha=0.3,
        facecolor=BG_COLOR, edgecolor=GREY, labelcolor=TEXT_COLOR,
    )

    # Footer
    fig.text(
        0.5, 0.03,
        "turboquant-torch  |  github.com/codepawl/turboquant-torch",
        fontsize=11, color=GREY_LIGHT, ha="center", fontfamily="monospace",
    )

    fig.savefig(out_path, dpi=100, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    data = load_results()
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    generate_card_d(data, assets_dir / "card-d-benchmark.png")
    generate_card_e(data, assets_dir / "card-e-scaling.png")


if __name__ == "__main__":
    main()
