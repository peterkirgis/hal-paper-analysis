import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import seaborn as sns
import os
import sys
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle, Patch
import matplotlib.transforms as mtransforms

# -----------------------------
# Paths and imports
# -----------------------------
script_dir = Path(__file__).parent
quantitative_dir = script_dir.parent
src_path = quantitative_dir / 'src'
sys.path.insert(0, str(src_path))

# Change CWD so relative asset paths work
os.chdir(quantitative_dir)

# Path to your variable font
font_path = Path(__file__).parent.parent / "assets" / "InterVariable.ttf"
# Register the font with matplotlib
fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = 'Inter'

from dataloader import load_paper_df
from plotting_utils import COMP_OF
from pareto_utils import _pareto_indices

# -----------------------------
# Assets
# -----------------------------
LOGO_PATHS = {
    "OpenAI":    str(quantitative_dir / "assets/openai_grey.png"),
    "Anthropic": str(quantitative_dir / "assets/anthropic_grey.png"),
    "DeepSeek":  str(quantitative_dir / "assets/deepseek_grey.png"),
    "Google":    str(quantitative_dir / "assets/gemini.png"),
}

# -----------------------------
# Matplotlib style
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter"],
    "font.weight": 400,
})


def create_accuracy_heatmap(df, save_plots=True):
    """
    Create a heatmap showing model accuracy across benchmarks,
    with model logos directly on the top edge (above heat cells),
    and x tick labels placed above the logos.
    """

    # -----------------------------
    # Pivot and ordering
    # -----------------------------
    pivot_df = df.pivot_table(
        index='bench_label',
        columns='model',
        values='accuracy',
        aggfunc='mean'
    )

    # Order columns (models) by overall mean accuracy DESC
    col_means = pivot_df.mean(axis=0)
    model_order = col_means.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df[model_order]

    # Order rows (benchmarks) by mean accuracy ASC
    row_means = pivot_df.mean(axis=1)
    benchmark_order = row_means.sort_values(ascending=True).index.tolist()
    pivot_df = pivot_df.loc[benchmark_order]

    # Recompute means after reordering
    row_means = pivot_df.mean(axis=1)
    col_means = pivot_df.mean(axis=0)

    # -----------------------------
    # Pareto frontier flags (using origin-based calculation)
    # -----------------------------
    pareto_models = set()
    for benchmark in pivot_df.index:
        bench_data = df[df['bench_label'] == benchmark].copy()
        sub = bench_data.dropna(subset=['total_cost', 'accuracy'])
        if len(sub) == 0:
            continue

        costs = sub['total_cost'].to_numpy()
        accs = sub['accuracy'].to_numpy()

        # Add origin (0,0) to the data for Pareto calculation
        costs_with_origin = np.concatenate([[0], costs])
        accs_with_origin = np.concatenate([[0], accs])

        # Get Pareto indices including origin
        p_idx_with_origin = _pareto_indices(costs_with_origin, accs_with_origin)

        # Filter out origin (index 0) and adjust indices back to original data
        p_idx = [idx - 1 for idx in p_idx_with_origin if idx > 0]

        pareto_runs = sub.iloc[p_idx].copy()

        # For ties in accuracy, only keep the one with lowest cost
        # Group by accuracy and keep only the min cost entry for each accuracy
        pareto_runs = pareto_runs.loc[pareto_runs.groupby('accuracy')['total_cost'].idxmin()]

        for _, row in pareto_runs.iterrows():
            pareto_models.add((benchmark, row['model']))

    # -----------------------------
    # Figure and axes
    # -----------------------------
    fig = plt.figure(figsize=(20, 14))
    fig.text(0.05, 0.98, "Holistic Agent Leaderboard (HAL) – Model Accuracy Across Benchmarks",
             ha='left', va='top', fontsize=20, fontweight='bold')
    fig.text(0.05, 0.96, "Comprehensive evaluation of frontier models across diverse agent benchmarks. ",
             ha='left', va='top', fontsize=16, color='.35')

    # Main axis (leave some headroom; we’ll use bbox_inches='tight' to include out-of-axes artists)
    ax = fig.add_axes([0.06, 0.0, 0.86, 0.80])

    # -----------------------------
    # Heatmap values & color scaling
    # -----------------------------
    data_range = pivot_df.max().max() - pivot_df.min().min()
    data_mid = pivot_df.mean().mean()
    color_margin = data_range * 0.4  # narrower scale for contrast
    vmin = max(0, (data_mid - color_margin) * 100)
    vmax = min(100, (data_mid + color_margin) * 100)

    annot_data = pivot_df * 100
    annot_labels = annot_data.map(lambda x: f'{x:.1f}%' if not np.isnan(x) else 'X')

    sns.heatmap(
        pivot_df * 100,
        annot=annot_labels,
        fmt='',
        cmap='RdBu',  # Blue=high, Red=low
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 18, 'weight': 'bold'},
        ax=ax,
        mask=pivot_df.isna()
    )

    # Draw X on NaNs (visual reinforcement over the 'X' label)
    for i, benchmark in enumerate(pivot_df.index):
        for j, model in enumerate(pivot_df.columns):
            if pd.isna(pivot_df.loc[benchmark, model]):
                ax.plot([j, j+1], [i, i+1], 'k-', linewidth=1.5, alpha=0.3)
                ax.plot([j, j+1], [i+1, i], 'k-', linewidth=1.5, alpha=0.3)

    # Pareto highlight
    for i, benchmark in enumerate(pivot_df.index):
        for j, model in enumerate(pivot_df.columns):
            if (benchmark, model) in pareto_models:
                rect = Rectangle((j, i), 1, 1, fill=False,
                                 edgecolor='gold', linewidth=3, zorder=10)
                ax.add_patch(rect)

    # -----------------------------
    # Ticks at top; labels above logos
    # -----------------------------
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=16)

    # Add padding so tick labels sit ABOVE logos
    ax.tick_params(axis='x', pad=26)  # adjust if logos are taller
    
    # Remove tick mark
    ax.tick_params(axis='x', which='both', length=0)

    # -----------------------------
    # Logos: directly on the top edge of the heatmap
    # -----------------------------
    # blended transform: x in data, y in axes coords
    trans_top = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    logo_y_axes = 1.005      # top edge of axes (heatmap border)
    logo_height_px = 20     # target logo height in pixels

    for i, model in enumerate(pivot_df.columns):  # displayed column order
        comp = COMP_OF.get(model, "Other")
        logo_path = LOGO_PATHS.get(comp)
        if logo_path and os.path.exists(logo_path):
            img = mpimg.imread(logo_path)
            zoom = logo_height_px / img.shape[0]
            imagebox = OffsetImage(img, zoom=zoom, resample=True)
            ab = AnnotationBbox(
                imagebox, (i + 0.5, logo_y_axes),
                xycoords=trans_top,
                box_alignment=(0.5, 0.0),  # bottom of logo sits on top edge
                frameon=False,
                pad=0,
                clip_on=False,
                zorder=5
            )
            ax.add_artist(ab)

    # -----------------------------
    # Row & column averages outside the heatmap
    # -----------------------------
    # Right side (row means): x in axes coords (>1), y in data coords
    trans_right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_right = 1.01  # just outside the right edge
    for i, val in enumerate(row_means.values):
        ax.text(x_right, i + 0.5, f"{val*100:.1f}%",
                transform=trans_right,
                ha='left', va='center', fontsize=16, fontweight='bold',
                clip_on=False, zorder=6)

    # Bottom side (column means): x in data coords, y in axes coords (<0)
    trans_bottom = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_bottom = -0.04  # just below the bottom edge
    for i, val in enumerate(col_means.values):
        ax.text(i + 0.5, y_bottom, f"{val*100:.1f}%",
                transform=trans_bottom,
                ha='center', va='top', fontsize=16, fontweight='bold',
                clip_on=False, zorder=6)

    # Labels for the averages
    ax.text(x_right, 1.0, "Avg Benchmark\nAccuracy",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=16, fontweight='bold', clip_on=False, zorder=7)

    ax.text(-0.01, y_bottom, "Avg Model\nAccuracy",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=16, fontweight='bold', clip_on=False, zorder=7)

    # IMPORTANT: Do NOT extend limits (no negative y hacks). Let bbox_inches="tight" capture out-of-axes artists.

    # -----------------------------
    # Legend for Pareto frontier
    # -----------------------------
    # Create legend at the bottom of the figure
    pareto_patch = Patch(facecolor='white', edgecolor='gold', linewidth=3,
                         label='Pareto frontier (cost-efficient models)')
    legend = fig.legend(handles=[pareto_patch],
                       loc='lower center',
                       bbox_to_anchor=(0.5, -0.10),
                       ncol=1,
                       fontsize=18,
                       frameon=True,
                       fancybox=True,
                       framealpha=0.95)

    # -----------------------------
    # Save / show
    # -----------------------------
    if save_plots:
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        fname = plots_dir / "accuracy_heatmap.pdf"
        plt.savefig(fname, bbox_inches="tight", transparent=True, dpi=600)
        print(f"Saved plot: {fname}")

    return pivot_df


if __name__ == "__main__":
    model_df, agent_df, benchmark_df = load_paper_df()

    # Filter out Sonnet 4 models
    benchmark_df = benchmark_df[~benchmark_df['model'].str.contains('Sonnet 4', case=False, na=False)]

    # Filter out TAU-bench Airline
    benchmark_df = benchmark_df[benchmark_df['bench_label'] != 'TAU-bench Airline']

    # Select highest-accuracy row for each (bench_label, model)
    model_df = benchmark_df.loc[benchmark_df.groupby(['bench_label', 'model'])['accuracy'].idxmax()]
    model_df = model_df.reset_index(drop=True)

    heatmap_df = create_accuracy_heatmap(model_df, save_plots=True)

    print("\nAccuracy Heatmap Summary:")
    print(f"Benchmarks: {len(heatmap_df)}")
    print(f"Models: {len(heatmap_df.columns)}")
    print(f"\nOverall average accuracy: {heatmap_df.values.mean():.2%}")