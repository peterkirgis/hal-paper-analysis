import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path

# Add the src directory to Python path
script_dir = Path(__file__).parent
quantitative_dir = script_dir.parent
src_path = quantitative_dir / 'src'
sys.path.insert(0, str(src_path))

# Change to quantitative directory for proper file paths
os.chdir(quantitative_dir)

# Now import and use the dataloader
from dataloader import load_paper_df
from plotting_utils import setup_colors


def extract_base_and_level(model_name):
    """Extract base model family and reasoning level from model name."""
    m = str(model_name).lower()
    base = "Other"
    if "o4" in m:
        base = "o4-mini"
    elif "sonnet" in m:
        if "3.7" in m:
            base = "Sonnet-3.7"
        elif "4" in m:
            base = "Sonnet-4"
        else:
            base = "Sonnet"
    elif "opus" in m:
        base = "Opus"

    if   "high" in m:     lvl = "High"
    elif "medium" in m:   lvl = "Medium"
    elif "low" in m:      lvl = "Low"
    elif "standard" in m: lvl = "None"
    else:                 lvl = "None"
    return base, lvl


def strip_level_from_model(model_name: str) -> str:
    """Remove reasoning level tokens from the model name string."""
    remove_tokens = [" High", " Low", " Medium", " None", " Standard", "Claude ", "Claude-"]
    cleaned = str(model_name)
    for tok in remove_tokens:
        cleaned = cleaned.replace(tok, "")
    return cleaned.strip()

def create_accuracy_dumbbell_plot_per_run_vertical_labels(df, save_plots=False):
    """
    Dumbbell plot per individual run, flipped horizontally:
      - Baseline always circle at 0.
      - High always diamond.
      - Sonnet-3.7 and Sonnet-4 treated separately.
      - Labels are vertical, above for negative Δ and below for positive Δ.
    """
    d = df.dropna(subset=["bench_label","model","accuracy","agent_scaffold"]).copy()
    d["base_model"], d["level"] = zip(*d["model"].map(extract_base_and_level))

    # Keep only relevant families
    d = d[d["base_model"].isin(["o4-mini","Opus","Sonnet-3.7","Sonnet-4"])].copy()

    # Normalize 'Standard' → 'None'
    mask_std = (d["base_model"].isin(["Sonnet-3.7","Sonnet-4","Opus"])) & (d["level"].str.lower()=="standard")
    d.loc[mask_std, "level"] = "None"

    pairs = []
    for (benchmark, base_model, scaffold), g in d.groupby(["bench_label","base_model","agent_scaffold"]):
        levels = set(g["level"])

        if base_model == "o4-mini":
            needed = {"Low","High"}
            if not needed.issubset(levels): continue
            baselines = g[g["level"]=="Low"]
            highs     = g[g["level"]=="High"]
        elif base_model in ("Sonnet-3.7","Sonnet-4","Opus"):
            needed = {"None","High"}
            if not needed.issubset(levels): continue
            baselines = g[g["level"]=="None"]
            highs     = g[g["level"]=="High"]
        else:
            continue

        for _, baseline_row in baselines.iterrows():
            for _, high_row in highs.iterrows():
                base_acc = float(baseline_row["accuracy"])
                high_acc = float(high_row["accuracy"])
                delta = high_acc - base_acc

                pairs.append({
                    "label": f"{benchmark} • {strip_level_from_model(high_row['model'])} • {scaffold.title()}",
                    "model": high_row['model'],
                    "benchmark": benchmark,
                    "base_model": base_model,
                    "scaffold": scaffold,
                    "difference": delta,
                    "baseline_acc": base_acc,
                    "high_acc": high_acc
                })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        print("No pairs found.")
        return

    pairs_df = pairs_df.sort_values("difference", ascending=True).reset_index(drop=True)

    # --- Plot (flipped) ---
    plt.figure(figsize=(max(8, 0.4*len(pairs_df)), 13))

    # use actual models from your data
    all_models = df["model"].unique().tolist()
    shade_of = setup_colors(all_models)
    ax = plt.gca()  # get current axis

    for i, row in pairs_df.iterrows():
        model_name = row.get("model", None) or row["base_model"]
        color = shade_of.get(model_name, "#777777")
        dy = row["difference"]

        # line from baseline (circle) at 0 to Δ (diamond)
        ax.vlines(i, 0.0, dy, colors=color, alpha=0.7, linewidth=2)

        # baseline always circle  (BIGGER)
        ax.scatter(i, 0.0, color=color, s=240, alpha=0.95,
                   marker="o", edgecolors="black", linewidth=1.5, zorder=3)

        # high always diamond    (BIGGER)
        ax.scatter(i, dy, color=color, s=240, alpha=0.95,
                   marker="D", edgecolors="black", linewidth=1.5, zorder=3)

        # label anchored to circle (baseline at 0.0)  (BIGGER FONTS)
        va  = "bottom" if dy <= 0 else "top"
        off = 20 if dy <= 0 else -20
        if dy <= 0:
            ax.annotate(
                row["label"],
                xy=(i, 0.0),
                xytext=(0, off),
                textcoords="offset points",
                rotation=90,
                ha="center", va=va,
                fontsize=20
            )
        else:
            ax.annotate(
                row["label"],
                xy=(i, 0.0),
                xytext=(0, off),
                textcoords="offset points",
                rotation=90,
                ha="center", va=va,
                fontsize=20
            )

    plt.axhline(0, color="black", linewidth=1.0, alpha=0.6)

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    mn, mx = pairs_df["difference"].min(), pairs_df["difference"].max()
    pad = max(0.03, 0.05*(mx - mn if mx != mn else 0.2))
    plt.ylim(min(0, -mx) - pad - 0.02, max(0, mx) + pad + 0.02)
    plt.yticks(fontsize=18)  # BIGGER

    ax.set_ylabel("Change in Accuracy From Higher Reasoning Level", fontsize=26, fontweight="bold")  # BIGGER
    ax.set_xlim(-0.8, len(pairs_df) - 0.2)
    ax.set_xticks([])  # hide x-axis ticks since labels are annotations
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_plots:
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        fname = plots_dir / "reasoning_accuracy_delta_vertical_labels.pdf"
        plt.savefig(fname, bbox_inches="tight", transparent=True)
        print(f"Saved plot: {fname}")

    return pairs_df


if __name__ == "__main__":
    # Load the most recent data
    model_df, agent_df, benchmark_df = load_paper_df()

    # make shortened benchmark labels from benchmark_name
    benchmark_mapping = {
        'assistantbench': 'AB',
        'corebench_hard': 'CORE',
        'gaia': 'GAIA',
        'online_mind2web': 'Mind2Web',
        'scicode': 'SciCode',
        'scienceagentbench': 'SAB',
        'swebench_verified_mini': 'SWE',
        'taubench_airline': 'TAU',
        'usaco': 'USACO'
    }
    benchmark_df['bench_label'] = benchmark_df['benchmark_name'].map(benchmark_mapping)

    scaffold_mapping = {
        'Browser-Use': 'Browser-Use',
        'CORE-Agent': 'CORE-Agent',
        'HF Open Deep Research': 'HF ODR',
        'SeeAct': 'SeeAct',
        'Scicode Tool Calling Agent': 'Tool Calling',
        'SAB Self-Debug': 'Self-Debug',
        'SWE-Agent': 'SWE-Agent',
        'USACO Episodic + Semantic': 'USACO E+S',
        'HAL Generalist Agent': 'Generalist'
    }

    benchmark_df['agent_scaffold'] = benchmark_df['agent_scaffold'].map(scaffold_mapping)

    dumbbell_df = create_accuracy_dumbbell_plot_per_run_vertical_labels(benchmark_df, save_plots=True)

    # Print count of positive vs negative deltas
    if not dumbbell_df.empty:
        pos_count = (dumbbell_df['difference'] > 0).sum()
        neg_count = (dumbbell_df['difference'] < 0).sum()
        zero_count = (dumbbell_df['difference'] == 0).sum()
        print(f"\nReasoning Level Change Results:")
        print(f"  Positive Deltas: {pos_count}")
        print(f"  Negative Deltas: {neg_count}")
        print(f"  No Change: {zero_count}")