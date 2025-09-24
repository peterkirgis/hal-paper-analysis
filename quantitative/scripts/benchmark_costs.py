import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def benchmark_costs(df, save_plots=True):

    total_costs = df.groupby('bench_label')['total_cost'].mean().reset_index()

    # reorder by total cost descending
    total_costs = total_costs.sort_values(by='total_cost', ascending=True)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=total_costs, x='bench_label', y='total_cost')
    ax.set_xlabel('Benchmark', fontsize=15, fontweight='bold')
    ax.set_ylabel('Average Run Cost', fontsize=15, fontweight='bold')
    # add label on each bar with the total cost formatted as currency
    for i, row in enumerate(total_costs.itertuples()):
        ax.text(i, row.total_cost + 5, f'${row.total_cost:,.0f}', ha='center', va='bottom', fontsize=11)
    # use log scale for y axis
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=20, fontsize=12)

    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()

    if save_plots:
        plots_dir = quantitative_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        fname = plots_dir / "benchmark_costs.pdf"
        plt.savefig(fname, bbox_inches="tight", transparent=True)
        print(f"Saved plot: {fname}")

if __name__ == "__main__":
    model_df, agent_df, benchmark_df = load_paper_df()

    benchmark_costs(benchmark_df)
