import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_inline.backend_inline import set_matplotlib_formats
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import patheffects as pe

import sys
from pathlib import Path
import os

# Add the src directory to Python path
script_dir = Path(__file__).parent
quantitative_dir = script_dir.parent
src_path = quantitative_dir / 'src'
sys.path.insert(0, str(src_path))

# Change to quantitative directory for proper file paths
os.chdir(quantitative_dir)

# Now import and use the dataloader
from dataloader import load_paper_df
from dataloader import load_most_recent_df
from pareto_utils import compute_pareto_frontier_with_origin, _pareto_indices, compute_pareto_statistics, compute_model_pareto_rankings
from plotting_utils import setup_colors, get_scaffold_hatch, setup_plot_style

def plot_accuracy_vs_cost_by_benchmark_individual(
    df, save_plots=False, annotate='pareto', col_wrap=3
):
    """Facet accuracy vs cost by benchmark with consistent styling.
       Colors come from your existing `setup_colors` mapping.
       Pareto frontier now always starts from (0,0).
    """
    plt.close('all')

    # ---------- Global style (aligned with bar plots) ----------
    setup_plot_style()

    # ---------- Prep data & colors ----------
    plot_df = df.dropna(subset=["bench_label","model","agent_scaffold","total_cost","accuracy"]).copy()
    if plot_df.empty:
        print("No data to plot.")
        return

    models   = sorted(plot_df['model'].unique().tolist())
    shade_of = setup_colors(models)  # <- your function (keeps colors)

    benches  = sorted(plot_df['bench_label'].unique())
    n        = len(benches)

    # Default grid
    ncols = min(col_wrap, max(1, n))
    nrows = int(np.ceil(n / ncols))
    # standard left-to-right, top-to-bottom
    fill_order = [divmod(i, ncols) for i in range(n)]

    # Square-ish small multiples
    fig_w = 4 * ncols
    fig_h = 3.2 * nrows + 1.6  # extra for legends
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Hide all to start; we'll turn on used cells
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis('off')

    # ---------- Plot each facet ----------
    for bench, (r, c) in zip(benches, fill_order):
        ax = axes[r, c]
        ax.axis('on')

        sub   = plot_df[plot_df['bench_label'] == bench].copy()
        costs = sub['total_cost'].to_numpy()
        accs  = sub['accuracy'].to_numpy()
        
        # Get Pareto frontier points starting from (0,0)
        pareto_points = compute_pareto_frontier_with_origin(costs, accs)
        
        # Get individual Pareto point indices - but adjusted for origin-based frontier
        # Add origin to the data temporarily for Pareto calculation
        costs_with_origin = np.concatenate([[0], costs])
        accs_with_origin = np.concatenate([[0], accs])
        p_idx_with_origin = _pareto_indices(costs_with_origin, accs_with_origin)
        
        # Filter out origin (index 0) and adjust indices back to original data
        p_idx = [idx - 1 for idx in p_idx_with_origin if idx > 0]
        pareto_runs = sub.iloc[p_idx].sort_values('total_cost')

        # Pareto line starting from origin (0,0)
        if len(pareto_points) >= 2:
            ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                    linestyle='--', linewidth=1.6, color='red', alpha=0.7, zorder=2)

        # Points (color by model; optional hatching by scaffold)
        for _, row in sub.iterrows():
            is_pareto = row.name in pareto_runs.index
            size      = 90 if is_pareto else 70
            lw        = 2.0 if is_pareto else 1.0
            hatch     = get_scaffold_hatch(row['agent_scaffold'])  # your mapping

            ax.scatter(row['total_cost'], row['accuracy'],
                       s=size,
                       facecolor=shade_of.get(row['model'], "#777777"),
                       edgecolor='black',
                       linewidth=lw,
                       alpha=0.95 if is_pareto else 0.78,
                       zorder=4 if is_pareto else 3,
                       rasterized=True,
                       hatch=hatch)

        # Annotate Pareto points (kept style)
        if annotate == 'pareto' and len(pareto_runs) > 0:
            x_right = ax.get_xlim()[1]
            for _, row in pareto_runs.iterrows():
                x_val, y_val = row['total_cost'], row['accuracy']
                dx, dy, ha = 8, 0, 'left'
                if x_val + dx > x_right - 12:
                    dx, ha = -8, 'right'
                txt = ax.annotate(
                    row['model'],
                    xy=(x_val, y_val),
                    xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va='center',
                    fontsize=9, weight='bold', zorder=8
                )
                txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

        # Axes cosmetics (match bar-plot look)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.5)
            ax.spines[spine].set_color('black')
        ax.tick_params(axis='both', which='major', color='black', width=1.2, length=4)
        ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.6, zorder=1)

        # Padded limits - ensure we can see the origin
        y_min, y_max = sub['accuracy'].min(), sub['accuracy'].max()
        y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 0.02
        ax.set_ylim(max(-0.02, min(0, y_min)), min(1.0, y_max + y_pad))

        x_min, x_max = sub['total_cost'].min(), sub['total_cost'].max()
        x_pad = (x_max - x_min) * 0.06 if x_max > x_min else 1.0
        # Ensure we include origin in x-axis
        ax.set_xlim(min(0, x_min - x_pad), x_max + x_pad)

        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(bench, pad=8)

        # Per-facet scaffold legend (hatching only; colors are reserved for models)
        bench_scaffolds = sorted(sub['agent_scaffold'].dropna().unique())
        if bench_scaffolds:
            scaffold_patches = [
                Patch(facecolor='white', edgecolor='black',
                      hatch=(get_scaffold_hatch(sc) or ''),
                      label=sc, linewidth=0.4)
                for sc in bench_scaffolds
            ]
            ax.legend(handles=scaffold_patches, title="Scaffolds",
                      loc='lower right', fontsize=8, title_fontsize=8,
                      frameon=True, fancybox=True, framealpha=0.9)

    # Turn off any remaining unused cells
    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    # Global y-axis label (left side)
    fig.supylabel('Accuracy', fontsize=18, fontweight='bold', x=0.005)

    # Global model legend (colors)
    model_patches = [Patch(facecolor=shade_of.get(m, "#777777"),
                           edgecolor='black', linewidth=0.8, label=m)
                     for m in models]
    if model_patches:
        leg1 = fig.legend(handles=model_patches, title="Models",
                          loc='lower center', bbox_to_anchor=(0.5, 0.055),
                          ncol=min(7, len(models)), fontsize=10, title_fontsize=11,
                          frameon=True, fancybox=True)
        leg1.get_frame().set_alpha(0.95)

    # Cost label positioned above the legend
    fig.text(0.5, 0.15, 'Cost (USD)', fontsize=18, fontweight='bold', ha='center')

    # Layout: extra room at bottom for legend and cost label
    fig.tight_layout(rect=[0, 0.17, 1, 0.98])

    if save_plots:
        # Save to plots folder at quantitative level
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        base = plots_dir / "accuracy_vs_cost_by_benchmark"
        fig.savefig(str(base) + ".pdf", bbox_inches='tight')          # vector
        print(f"Saved plot: {base}.pdf")



if __name__ == "__main__":
    # Load the most recent data
    model_df, agent_df, benchmark_df = load_paper_df()

    plot_accuracy_vs_cost_by_benchmark_individual(benchmark_df, save_plots=True, annotate='pareto', col_wrap=3)

    pareto_stats_df = compute_pareto_statistics(benchmark_df)
    print("Pareto Frontier Statistics by Benchmark (with origin baseline):")
    print("=" * 60)
    print(f"{'Benchmark':<25} {'Total':<7} {'Pareto':<8} {'Fraction':<10}")
    print("-" * 60)

    for _, row in pareto_stats_df.iterrows():
        print(f"{row['benchmark_name']:<25} {row['total_models']:<7} {row['pareto_models']:<8} {row['pareto_fraction']:<10.1%}")
    
    print("\nSummary:")
    print(f"Average Pareto fraction across benchmarks: {pareto_stats_df['pareto_fraction'].mean():.1%}")
    print(f"Range: {pareto_stats_df['pareto_fraction'].min():.1%} - {pareto_stats_df['pareto_fraction'].max():.1%}")

    print("\nComputing model rankings by Pareto frontier appearances...")

    model_rankings = compute_model_pareto_rankings(benchmark_df)

    print("Model Rankings by Pareto Frontier Frequency (with origin baseline):")
    print("=" * 75)
    print(f"{'Rank':<4} {'Model':<25} {'Benchmarks':<10} {'Pareto':<8} {'Rate':<8}")
    print("-" * 75)

    for _, row in model_rankings.iterrows():
        print(f"{row['rank']:<4} {row['model']:<25} {row['benchmarks_tested']:<10} {row['pareto_appearances']:<8} {row['pareto_rate']:<8.1%}")

    print("\nTop 5 Most Pareto-Efficient Models (relative to origin):")
    top_5 = model_rankings.head(5)
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['model']}: {row['pareto_rate']:.1%} ({row['pareto_appearances']}/{row['benchmarks_tested']} benchmarks)")

    print(f"\nModels that are ALWAYS on Pareto frontier: {len(model_rankings[model_rankings['pareto_rate'] == 1.0])}")
    print(f"Models that are NEVER on Pareto frontier: {len(model_rankings[model_rankings['pareto_rate'] == 0.0])}")
