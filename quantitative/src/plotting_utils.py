"""
Plotting utilities for benchmark analysis and visualization.

This module provides color schemes, styling functions, and plotting helpers
for consistent visualization across different analysis notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
import colorsys
import os
from typing import Dict, List, Optional
from matplotlib.patches import Patch
from matplotlib import patheffects as pe
from matplotlib.ticker import PercentFormatter

from pareto_utils import compute_pareto_frontier_with_origin, _pareto_indices


# ═══════════════════════════════════════════════════════════════════
# COLOR SCHEME AND STYLING
# ═══════════════════════════════════════════════════════════════════

# Brand colors and model mappings
BRAND_BASE = {
    "OpenAI": "#10A37F",
    "Anthropic": "#D4A27F",
    "DeepSeek": "#4D6BFE",
    "Google": "#F8BBD0"
}

COMP_OF = {
    "o3 Medium": "OpenAI", "o4-mini High": "OpenAI", "o4-mini Low": "OpenAI",
    "GPT-4.1": "OpenAI", "GPT-5 Medium": "OpenAI",
    "o3-mini High": "OpenAI", "o3-mini Low": "OpenAI",
    "GPT-OSS-120B High": "OpenAI", "GPT-OSS-120B": "OpenAI",
    "Claude-3.7 Sonnet": "Anthropic", "Claude-3.7 Sonnet High": "Anthropic",
    "Claude-4 Sonnet": "Anthropic", "Claude-4 Sonnet High": "Anthropic",
    "Claude-4.1 Opus": "Anthropic", "Claude-4.1 Opus High": "Anthropic",
    "Claude-4 Opus": "Anthropic", "Claude-4 Opus High": "Anthropic",
    "Claude Opus 4.1": "Anthropic", "Claude Opus 4.1 High": "Anthropic",
    "Claude Sonnet 4": "Anthropic", "Claude Sonnet 4 High": "Anthropic",
    "DeepSeek R1": "DeepSeek", "DeepSeek V3": "DeepSeek",
    "Gemini 2.0 Flash": "Google",
}


def make_shades(base_hex: str, k: int, span: float = 0.20) -> List[str]:
    """
    Create k shades of a base color.

    Args:
        base_hex: Base color in hex format
        k: Number of shades to create
        span: Color variation span

    Returns:
        List of hex color codes
    """
    r, g, b = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hi = min(1, l + span)
    return [mcolors.to_hex(colorsys.hls_to_rgb(h, L, s))
            for L in np.linspace(l, hi, k)]


def setup_colors(models: List[str]) -> Dict[str, str]:
    """
    Set up color scheme for models by company.

    Args:
        models: List of model names

    Returns:
        Dictionary mapping model names to hex color codes
    """
    company_models, shade_of = {}, {}
    for m in models:
        company_models.setdefault(COMP_OF.get(m, "Other"), []).append(m)
    for comp, mlist in company_models.items():
        base = BRAND_BASE.get(comp, "#777777")
        for m, c in zip(sorted(mlist), make_shades(base, len(mlist))):
            shade_of[m] = c
    return shade_of


def get_scaffold_hatch(scaffold: str) -> Optional[str]:
    """
    Get hatch pattern for agent scaffold.
    Only HAL Generalist Agent and SeeAct get hatching with denser patterns.

    Args:
        scaffold: Name of the agent scaffold

    Returns:
        Hatch pattern string or None
    """
    if scaffold == "HAL Generalist Agent":
        return "//////"  # Denser diagonal lines
    elif scaffold == "SeeAct":
        return "\\\\\\\\\\\\"  # Denser backslash lines
    return None


def get_agent_scaffold_hatch(scaffold: str) -> Optional[str]:
    """
    Return hatch pattern for agent scaffold (alternative patterns).

    Args:
        scaffold: Name of the agent scaffold

    Returns:
        Hatch pattern string or None
    """
    hatch_patterns = {
        'browser_agent': '///',      # diagonal lines
        'tool_calling_agent': '...',  # dots
        'coreagent': 'xxx',          # crosses
        'fewshot': '|||',            # vertical lines
        'default': None              # no hatch
    }
    return hatch_patterns.get(scaffold, None)


# ═══════════════════════════════════════════════════════════════════
# STYLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Set up global matplotlib style for consistent plotting."""
    sns.set_theme(style="white", rc={"axes.grid": False})
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TTF (no Type 3)
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "hatch.linewidth": 1.2,
    })


def style_axis(ax: plt.Axes):
    """
    Apply consistent styling to an axis.

    Args:
        ax: Matplotlib axis to style
    """
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')
    ax.tick_params(axis='both', which='major', color='black', width=1.2, length=4)
    ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.6, zorder=1)


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR AGENT COMPARISONS
# ═══════════════════════════════════════════════════════════════════

# Clean, professional color scheme with good contrast
AGENT_COLOR_SCHEME = {
    "generalist": "#5DADE2",  # Light blue/teal
    "specialist": "#2E4053",  # Dark blue/navy
    "generalist_accent": "#AED6F1",  # Even lighter blue for accents
    "specialist_accent": "#1B2838",  # Even darker for accents
}


def setup_agent_comparison_style():
    """Set up matplotlib style for agent comparison plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",  # Very light gray background
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E0E0E0",  # Light gray gridlines
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "hatch.linewidth": 0.8
    })


def style_agent_comparison_axis(ax: plt.Axes):
    """
    Apply styling to agent comparison plot axes.

    Args:
        ax: Matplotlib axis to style
    """
    # Clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('#2C3E50')

    # Refined grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind bars