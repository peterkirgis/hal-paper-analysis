import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib import patheffects as pe
from scipy.stats import pearsonr, spearmanr, kendalltau

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
from pareto_utils import compute_pareto_frontier_with_origin, _pareto_indices, compute_pareto_statistics, compute_model_pareto_rankings
from plotting_utils import setup_colors, get_scaffold_hatch, setup_plot_style


def plot_accuracy_vs_tokens_by_benchmark_individual(
    df, save_plots=False, annotate='pareto', col_wrap=3
):
    """Facet accuracy vs tokens by benchmark with consistent styling.
       Colors come from your existing `setup_colors` mapping.
       Pareto frontier now always starts from (0,0).
    """
    plt.close('all')

    # ---------- Global style (aligned with bar plots) ----------
    setup_plot_style()

    # ---------- Prep data & colors ----------
    plot_df = df.dropna(subset=["bench_label","model","agent_scaffold","total_tokens","accuracy"]).copy()
    if plot_df.empty:
        print("No data to plot.")
        return

    models   = sorted(plot_df['model'].unique().tolist())
    shade_of = setup_colors(models)

    benches  = sorted(plot_df['bench_label'].unique())
    n        = len(benches)

    # Default grid
    ncols = min(col_wrap, max(1, n))
    nrows = int(np.ceil(n / ncols))
    # standard left-to-right, top-to-bottom
    fill_order = [divmod(i, ncols) for i in range(n)]

    fig_w = 4 * ncols
    fig_h = 3.2 * nrows + 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis('off')

    # ---------- Plot each facet ----------
    for bench, (r, c) in zip(benches, fill_order):
        ax = axes[r, c]
        ax.axis('on')

        sub   = plot_df[plot_df['bench_label'] == bench].copy()
        tokens = sub['total_tokens'].to_numpy()
        accs   = sub['accuracy'].to_numpy()
        
        # Pareto frontier
        pareto_points = compute_pareto_frontier_with_origin(tokens, accs)
        tokens_with_origin = np.concatenate([[0], tokens])
        accs_with_origin   = np.concatenate([[0], accs])
        p_idx_with_origin  = _pareto_indices(tokens_with_origin, accs_with_origin)
        p_idx = [idx - 1 for idx in p_idx_with_origin if idx > 0]
        pareto_runs = sub.iloc[p_idx].sort_values('total_tokens')

        if len(pareto_points) >= 2:
            ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                    linestyle='--', linewidth=1.6, color='red', alpha=0.7, zorder=2)

        # Scatter points
        for _, row in sub.iterrows():
            is_pareto = row.name in pareto_runs.index
            size = 90 if is_pareto else 70
            lw   = 2.0 if is_pareto else 1.0
            hatch = get_scaffold_hatch(row['agent_scaffold'])

            ax.scatter(row['total_tokens'], row['accuracy'],
                       s=size,
                       facecolor=shade_of.get(row['model'], "#777777"),
                       edgecolor='black',
                       linewidth=lw,
                       alpha=0.95 if is_pareto else 0.78,
                       zorder=4 if is_pareto else 3,
                       rasterized=True,
                       hatch=hatch)

        # Annotate Pareto
        if annotate == 'pareto' and len(pareto_runs) > 0:
            x_right = ax.get_xlim()[1]
            for _, row in pareto_runs.iterrows():
                x_val, y_val = row['total_tokens'], row['accuracy']
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

        # Axes cosmetics
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.5)
            ax.spines[spine].set_color('black')
        ax.tick_params(axis='both', which='major', color='black', width=1.2, length=4)
        ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.6, zorder=1)


        # Limits
        y_min, y_max = sub['accuracy'].min(), sub['accuracy'].max()
        y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 0.02
        ax.set_ylim(max(-0.02, min(0, y_min)), min(1.0, y_max + y_pad))

        x_min, x_max = sub['total_tokens'].min(), sub['total_tokens'].max()
        x_pad = (x_max - x_min) * 0.06 if x_max > x_min else 1.0
        ax.set_xlim(min(0, x_min - x_pad), x_max + x_pad)

        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(bench, pad=8)

        # Scaffold legend
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

    # Global labels
    fig.supylabel('Accuracy', fontsize=18, fontweight='bold', x=0.005)

    model_patches = [Patch(facecolor=shade_of.get(m, "#777777"),
                           edgecolor='black', linewidth=0.8, label=m)
                     for m in models]
    if model_patches:
        leg1 = fig.legend(handles=model_patches, title="Models",
                          loc='lower center', bbox_to_anchor=(0.5, 0.055),
                          ncol=min(7, len(models)), fontsize=10, title_fontsize=11,
                          frameon=True, fancybox=True)
        leg1.get_frame().set_alpha(0.95)

    # X-axis label
    fig.text(0.5, 0.15, 'Token Usage (Prompt + Completion)', fontsize=18, fontweight='bold', ha='center')

    fig.tight_layout(rect=[0, 0.17, 1, 0.98])

    if save_plots:
        # Save to plots folder at quantitative level
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        base = plots_dir / "accuracy_vs_tokens_by_benchmark"
        fig.savefig(str(base) + ".pdf", bbox_inches='tight')          # vector
        print(f"Saved plot: {base}.pdf")



if __name__ == "__main__":
    # Load the most recent data
    model_df, agent_df, benchmark_df = load_paper_df()

    plot_accuracy_vs_tokens_by_benchmark_individual(benchmark_df, save_plots=True, annotate='pareto', col_wrap=3)

    model_rankings = compute_model_pareto_rankings(benchmark_df, resource_col='total_tokens')

    print("Model Rankings by Pareto Frontier Frequency (Tokens vs Accuracy):")
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

    # ═══════════════════════════════════════════════════════════════════
    # CORRELATION ANALYSIS: TOKEN USAGE VS ACCURACY BY BENCHMARK
    # ═══════════════════════════════════════════════════════════════════

    # Calculate correlations for each benchmark
    correlation_results = []

    benchmarks = sorted(benchmark_df['bench_label'].unique())

    for benchmark in benchmarks:
        # Filter data for this benchmark
        bench_data = benchmark_df[
            (benchmark_df['bench_label'] == benchmark) &
            (benchmark_df['accuracy'].notna()) &
            (benchmark_df['total_tokens'].notna())
        ].copy()

        if len(bench_data) < 3:  # Need at least 3 points for meaningful correlation
            continue

        # Extract arrays
        tokens = bench_data['total_tokens'].values
        accuracy = bench_data['accuracy'].values

        # Calculate correlations
        try:
            pearson_r, pearson_p = pearsonr(tokens, accuracy)
            spearman_r, spearman_p = spearmanr(tokens, accuracy)
            kendall_r, kendall_p = kendalltau(tokens, accuracy)

            correlation_results.append({
                'benchmark': benchmark,
                'n_models': len(bench_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_r': kendall_r,
                'kendall_p': kendall_p
            })

        except Exception as e:
            print(f"Warning: Could not calculate correlations for {benchmark}: {e}")

    # Convert to DataFrame for easier handling
    corr_df = pd.DataFrame(correlation_results)

    # Display results
    print("\nToken Usage vs Accuracy Correlations by Benchmark")
    print("=" * 70)
    print("\nNote: Positive correlation means higher token usage → higher accuracy")
    print("      Negative correlation means higher token usage → lower accuracy")
    print("      Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")

    for _, row in corr_df.iterrows():
        def significance_stars(p_value):
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            else:
                return ""

        print(f"{row['benchmark']:<25} (n={row['n_models']:2d})")
        print(f"  Pearson:  r={row['pearson_r']:6.3f}{significance_stars(row['pearson_p']):3s} (p={row['pearson_p']:.3f})")
        print(f"  Spearman: r={row['spearman_r']:6.3f}{significance_stars(row['spearman_p']):3s} (p={row['spearman_p']:.3f})")
        print(f"  Kendall:  r={row['kendall_r']:6.3f}{significance_stars(row['kendall_p']):3s} (p={row['kendall_p']:.3f})")
        print()

    # Summary statistics
    print("\nSummary Statistics Across All Benchmarks:")
    print("=" * 50)
    print(f"Benchmarks analyzed: {len(corr_df)}")
    print(f"Average Pearson correlation: {corr_df['pearson_r'].mean():.3f}")
    print(f"Average Spearman correlation: {corr_df['spearman_r'].mean():.3f}")
    print(f"Average Kendall correlation: {corr_df['kendall_r'].mean():.3f}")

    # Count significant correlations
    sig_pearson = (corr_df['pearson_p'] < 0.05).sum()
    sig_spearman = (corr_df['spearman_p'] < 0.05).sum()
    sig_kendall = (corr_df['kendall_p'] < 0.05).sum()

    print(f"\nSignificant correlations (p < 0.05):")
    print(f"  Pearson: {sig_pearson}/{len(corr_df)} benchmarks ({sig_pearson/len(corr_df)*100:.1f}%)")
    print(f"  Spearman: {sig_spearman}/{len(corr_df)} benchmarks ({sig_spearman/len(corr_df)*100:.1f}%)")
    print(f"  Kendall: {sig_kendall}/{len(corr_df)} benchmarks ({sig_kendall/len(corr_df)*100:.1f}%)")

    # Identify strongest positive and negative correlations
    strongest_positive = corr_df.loc[corr_df['pearson_r'].idxmax()]
    strongest_negative = corr_df.loc[corr_df['pearson_r'].idxmin()]

    print(f"\nStrongest positive correlation: {strongest_positive['benchmark']} (r={strongest_positive['pearson_r']:.3f})")
    print(f"Strongest negative correlation: {strongest_negative['benchmark']} (r={strongest_negative['pearson_r']:.3f})")

