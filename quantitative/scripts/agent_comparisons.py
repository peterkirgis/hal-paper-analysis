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


def setup_agent_comparison_style():
    """Set up matplotlib style for agent comparison plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "hatch.linewidth": 1.0
    })


def create_generalist_vs_specialist_comparison(df, benchmark_name, save_plots=False):
    """
    Create grouped bar chart comparing HAL Generalist Agent with task-specific agents.
    """
    # Filter to benchmark and identify agent types
    bench_data = df[df['benchmark_name'] == benchmark_name].copy()
    bench_data['agent_type'] = bench_data['agent_scaffold'].apply(
        lambda x: 'HAL Generalist' if 'Generalist' in str(x) else 'Task-Specific'
    )

    # Build comparison data
    comparison_data = []
    for model in bench_data['model'].unique():
        if pd.isna(model):
            continue

        model_data = bench_data[bench_data['model'] == model]
        gen_data = model_data[model_data['agent_type'] == 'HAL Generalist']
        spec_data = model_data[model_data['agent_type'] == 'Task-Specific']

        if len(gen_data) > 0 and len(spec_data) > 0:
            comparison_data.append({
                'model': model,
                'generalist_accuracy': gen_data['accuracy'].iloc[0],
                'specialist_accuracy': spec_data['accuracy'].iloc[0],
                'generalist_cost': gen_data['total_cost'].iloc[0] if 'total_cost' in gen_data.columns else None,
                'specialist_cost': spec_data['total_cost'].iloc[0] if 'total_cost' in spec_data.columns else None,
            })

    comparison_df = pd.DataFrame(comparison_data)

    if len(comparison_df) == 0:
        print(f"No comparison data found for {benchmark_name}")
        return comparison_df

    # Remove rows with missing cost data
    comparison_df = comparison_df.dropna(subset=['generalist_cost', 'specialist_cost'])

    if len(comparison_df) == 0:
        print(f"No cost data available for {benchmark_name}")
        return comparison_df

    # Create plot
    setup_agent_comparison_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort models by specialist accuracy
    comparison_df = comparison_df.sort_values('specialist_accuracy', ascending=False)
    models = comparison_df['model'].values
    n_models = len(models)

    # Setup positions with proper grouping
    x = np.arange(n_models)
    width = 0.35
    spacing = 0.05

    # Colors
    COLOR_SCHEME = {
        "generalist": "#5DADE2",
        "specialist": "#2E4053",
    }

    # Get values
    gen_values = comparison_df['generalist_accuracy'].values
    spec_values = comparison_df['specialist_accuracy'].values
    gen_costs = comparison_df['generalist_cost'].values
    spec_costs = comparison_df['specialist_cost'].values

    # Create bars
    gen_bars = ax.bar(x - width/2 - spacing/2, gen_values, width,
                      color=COLOR_SCHEME['generalist'],
                      edgecolor='#2C3E50', linewidth=0.8,
                      label='HAL Generalist Agent')

    spec_bars = ax.bar(x + width/2 + spacing/2, spec_values, width,
                       color=COLOR_SCHEME['specialist'],
                       edgecolor='#2C3E50', linewidth=0.8,
                       label='Task-Specific Agent')

    # Add accuracy labels above bars
    accuracy_label_offset = 0.015
    for i in range(n_models):
        # Generalist accuracy labels
        gen_label = f'{gen_values[i]:.0%}'
        ax.text(x[i] - width/2 - spacing/2, gen_values[i] + accuracy_label_offset, gen_label,
               ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Task-specific accuracy labels
        spec_label = f'{spec_values[i]:.0%}'
        ax.text(x[i] + width/2 + spacing/2, spec_values[i] + accuracy_label_offset, spec_label,
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add cost labels
    cost_label_offset = 0.005
    for i in range(n_models):
        # Generalist cost labels
        gen_cost_label = f'(${gen_costs[i]:.0f})'
        if gen_values[i] - cost_label_offset - 0.05 <= 0:
            # Too low → place cost above bar in parentheses
            ax.text(x[i] - width/2 - spacing/2,
                    gen_values[i] + accuracy_label_offset * 2,
                    gen_cost_label,
                    ha='center', va='bottom', fontsize=11,
                    fontstyle='italic', color='black')
        else:
            # Inside the bar
            ax.text(x[i] - width/2 - spacing/2,
                    gen_values[i] - cost_label_offset,
                    f'${gen_costs[i]:.0f}',
                    ha='center', va='top', fontsize=11,
                    color='white', fontweight='bold', rotation=90)

        # Specialist cost labels
        spec_cost_label = f'(${spec_costs[i]:.0f})'
        if spec_values[i] - cost_label_offset - 0.05 <= 0:
            ax.text(x[i] + width/2 + spacing/2,
                    spec_values[i] + accuracy_label_offset * 2,
                    spec_cost_label,
                    ha='center', va='bottom', fontsize=11,
                    fontstyle='italic', color='black')
        else:
            ax.text(x[i] + width/2 + spacing/2,
                    spec_values[i] - cost_label_offset,
                    f'${spec_costs[i]:.0f}',
                    ha='center', va='top', fontsize=11,
                    color='white', fontweight='bold', rotation=90)

    # Styling
    ax.set_xlabel('Model', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=16)

    # Set x-axis with proper positioning and margins
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=15)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0))

    # Set axis limits with proper margins
    ax.set_xlim(-0.7, n_models - 0.3)
    max_val = max(max(gen_values), max(spec_values))
    ax.set_ylim(0, max_val * 1.3)

    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                      shadow=False, framealpha=0.95, edgecolor='#2C3E50')
    legend.get_frame().set_linewidth(0.8)

    # Clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('#2C3E50')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.08, right=0.95)

    if save_plots:
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        bench_safe = benchmark_name.replace('_', '-')
        fname = plots_dir / f'generalist_vs_specialist_accuracy_cost_{bench_safe}.pdf'
        plt.savefig(fname, bbox_inches="tight", transparent=True)
        print(f"Saved plot: {fname}")

    return comparison_df


def create_seeact_vs_browseruse_comparison(df, save_plots=False):
    """
    Create comparison between SeeAct and Browser-Use on Online Mind2Web.
    """
    # Filter to Online Mind2Web and the two scaffolds
    bench_data = df[df['bench_label'] == 'Online Mind2Web'].copy()
    bench_data = bench_data[bench_data['agent_scaffold'].isin(['SeeAct', 'Browser-Use'])]

    if bench_data.empty:
        print("No data found for Online Mind2Web with SeeAct/Browser-Use")
        return pd.DataFrame()

    # Build comparison data
    comparison_data = []
    for model in bench_data['model'].unique():
        model_data = bench_data[bench_data['model'] == model]
        seeact_data = model_data[model_data['agent_scaffold'] == 'SeeAct']
        browser_data = model_data[model_data['agent_scaffold'] == 'Browser-Use']

        if len(seeact_data) > 0 and len(browser_data) > 0:
            comparison_data.append({
                'model': model,
                'seeact_accuracy': seeact_data['accuracy'].iloc[0],
                'browser_accuracy': browser_data['accuracy'].iloc[0],
                'seeact_cost': seeact_data['total_cost'].iloc[0] if 'total_cost' in seeact_data.columns else None,
                'browser_cost': browser_data['total_cost'].iloc[0] if 'total_cost' in browser_data.columns else None,
            })

    comparison_df = pd.DataFrame(comparison_data)

    if comparison_df.empty:
        print("No matching pairs of SeeAct and Browser-Use found")
        return comparison_df

    # Remove rows with missing cost data
    comparison_df = comparison_df.dropna(subset=['seeact_cost', 'browser_cost'])

    if len(comparison_df) == 0:
        print("No cost data available for SeeAct vs Browser-Use comparison")
        return comparison_df

    # Create plot
    setup_agent_comparison_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by Browser-Use accuracy
    comparison_df = comparison_df.sort_values('browser_accuracy', ascending=False)
    models = comparison_df['model'].values
    n_models = len(models)

    x = np.arange(n_models)
    width = 0.35
    spacing = 0.05

    # Values
    seeact_acc = comparison_df['seeact_accuracy'].values
    browser_acc = comparison_df['browser_accuracy'].values
    seeact_cost = comparison_df['seeact_cost'].values
    browser_cost = comparison_df['browser_cost'].values

    # Bars
    seeact_bars = ax.bar(x - width/2 - spacing/2, seeact_acc, width,
                         color="#E67E22", edgecolor="#2C3E50", linewidth=0.8,
                         label="SeeAct")

    browser_bars = ax.bar(x + width/2 + spacing/2, browser_acc, width,
                          color="#27AE60", edgecolor="#2C3E50", linewidth=0.8,
                          label="Browser-Use")

    # Accuracy labels
    acc_offset = 0.015
    for i in range(n_models):
        ax.text(x[i] - width/2 - spacing/2, seeact_acc[i] + acc_offset,
                f"{seeact_acc[i]:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(x[i] + width/2 + spacing/2, browser_acc[i] + acc_offset,
                f"{browser_acc[i]:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Cost labels
    cost_offset = 0.005
    for i in range(n_models):
        # SeeAct
        if seeact_acc[i] - cost_offset - 0.05 <= 0:
            ax.text(x[i] - width/2 - spacing/2, seeact_acc[i] + acc_offset*2,
                    f"(${seeact_cost[i]:.0f})", ha="center", va="bottom",
                    fontsize=11, fontstyle="italic", color="black")
        else:
            ax.text(x[i] - width/2 - spacing/2, seeact_acc[i] - cost_offset,
                    f"${seeact_cost[i]:.0f}", ha="center", va="top",
                    fontsize=11, color="white", fontweight="bold", rotation=90)

        # Browser-Use
        if browser_acc[i] - cost_offset - 0.05 <= 0:
            ax.text(x[i] + width/2 + spacing/2, browser_acc[i] + acc_offset*2,
                    f"(${browser_cost[i]:.0f})", ha="center", va="bottom",
                    fontsize=11, fontstyle="italic", color="black")
        else:
            ax.text(x[i] + width/2 + spacing/2, browser_acc[i] - cost_offset,
                    f"${browser_cost[i]:.0f}", ha="center", va="top",
                    fontsize=11, color="white", fontweight="bold", rotation=90)

    # Styling
    ax.set_xlabel("Model", fontweight="bold", fontsize=16)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=15)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.set_ylim(0, max(max(seeact_acc), max(browser_acc)) * 1.3)

    legend = ax.legend(loc="upper right", frameon=True, fancybox=True,
                       shadow=False, framealpha=0.95, edgecolor="#2C3E50")
    legend.get_frame().set_linewidth(0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_plots:
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        fname = plots_dir / "seeact_vs_browseruse_accuracy_cost_online-mind2web.pdf"
        plt.savefig(fname, bbox_inches="tight", transparent=True)
        print(f"Saved plot: {fname}")

    return comparison_df


if __name__ == "__main__":
    # Load the most recent data
    model_df, agent_df, benchmark_df = load_paper_df()

    # Create generalist vs specialist comparisons for key benchmarks
    benchmarks_to_compare = ['corebench_hard', 'swebench_verified_mini']
    benchmark_results = {}

    for benchmark in benchmarks_to_compare:
        print(f"\nAnalyzing {benchmark}...")
        results = create_generalist_vs_specialist_comparison(
            benchmark_df, benchmark, save_plots=True
        )
        benchmark_results[benchmark] = results

        if not results.empty:
            # Print summary statistics
            gen_wins = (results['generalist_accuracy'] > results['specialist_accuracy']).sum()
            spec_wins = (results['specialist_accuracy'] > results['generalist_accuracy']).sum()
            ties = (results['generalist_accuracy'] == results['specialist_accuracy']).sum()

            avg_gen_acc = results['generalist_accuracy'].mean()
            avg_spec_acc = results['specialist_accuracy'].mean()
            avg_gen_cost = results['generalist_cost'].mean()
            avg_spec_cost = results['specialist_cost'].mean()

            print(f"  Models compared: {len(results)}")
            print(f"  Generalist wins: {gen_wins}")
            print(f"  Specialist wins: {spec_wins}")
            print(f"  Ties: {ties}")
            print(f"  Average generalist accuracy: {avg_gen_acc:.1%}")
            print(f"  Average specialist accuracy: {avg_spec_acc:.1%}")
            print(f"  Average generalist cost: ${avg_gen_cost:.0f}")
            print(f"  Average specialist cost: ${avg_spec_cost:.0f}")

    # Create SeeAct vs Browser-Use comparison
    print(f"\nAnalyzing SeeAct vs Browser-Use on Mind2Web...")
    seeact_results = create_seeact_vs_browseruse_comparison(
        benchmark_df, save_plots=True
    )

    if not seeact_results.empty:
        seeact_wins = (seeact_results['seeact_accuracy'] > seeact_results['browser_accuracy']).sum()
        browser_wins = (seeact_results['browser_accuracy'] > seeact_results['seeact_accuracy']).sum()
        ties = (seeact_results['seeact_accuracy'] == seeact_results['browser_accuracy']).sum()

        avg_seeact_acc = seeact_results['seeact_accuracy'].mean()
        avg_browser_acc = seeact_results['browser_accuracy'].mean()
        avg_seeact_cost = seeact_results['seeact_cost'].mean()
        avg_browser_cost = seeact_results['browser_cost'].mean()

        print(f"  Models compared: {len(seeact_results)}")
        print(f"  SeeAct wins: {seeact_wins}")
        print(f"  Browser-Use wins: {browser_wins}")
        print(f"  Ties: {ties}")
        print(f"  Average SeeAct accuracy: {avg_seeact_acc:.1%}")
        print(f"  Average Browser-Use accuracy: {avg_browser_acc:.1%}")
        print(f"  Average SeeAct cost: ${avg_seeact_cost:.0f}")
        print(f"  Average Browser-Use cost: ${avg_browser_cost:.0f}")

    print("\nAgent comparison analysis complete!")