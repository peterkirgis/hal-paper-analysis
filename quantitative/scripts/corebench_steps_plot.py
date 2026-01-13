"""
CORE-Bench Average Steps Bar Plot
Plots the average number of steps taken by each model on CORE-Bench with confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys
import matplotlib.image as mpimg
import os
from pathlib import Path

from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ═══════════════════════════════════════════════════════════════════
# SETUP AND DATA
# ═══════════════════════════════════════════════════════════════════

# Setup paths
script_dir = Path(__file__).parent
quantitative_dir = script_dir.parent

# Configure matplotlib for better quality plots with Inter font
import matplotlib.font_manager as fm
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Inter",
    "font.weight": 400,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 10,
})

# Path to your variable font
font_path = quantitative_dir / "assets" / "InterVariable.ttf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Inter'

# Data
data = {
    'model': [
        'anthropic/claude-sonnet-4-5',
        'anthropic/claude-sonnet-4-5_high',
        'claude-3-7-sonnet-20250219',
        'claude-3-7-sonnet-20250219_high',
        'claude-haiku-4-5',
        'gpt-4.1-2025-04-14',
        'o3-2025-04-16',
        'o4-mini-2025-04-16_high',
        'o4-mini-2025-04-16_low',
        'openai/deepseek-ai/DeepSeek-R1',
        'openai/deepseek-ai/DeepSeek-V3',
        'openai/gemini-2.0-flash',
        'openai/gpt-5-2025-08-07',
        'openrouter/anthropic/claude-opus-4.1',
        'openrouter/anthropic/claude-opus-4.1_high',
        'claude-opus-4-5-20251101',
        'claude-opus-4-5-20251101_high',
        'openai/gemini-3-pro-preview'
    ],
    'steps': [
        31.0833333, 30.1444444, 23.9277778, 26.1555556, 40.4222222,
        28.55, 28.3611111, 18.2222222, 17.8944444, 16.2215909,
        17.6055556, 44, 10.7272727, 27.8277778, 32.2777778,
        33.1611111, 31.1833333, 16.9444444
    ],
    'steps_ci': [
        4.39887711, 3.71816965, 3.56526884, 3.78696096, 2.881207,
        2.94933149, 2.73360207, 2.9358404, 2.26720065, 3.81241136,
        1.92236205, 2.34442953, 1.1759847, 3.24677679, 3.46402595,
        3.44696858, 3.38567727, 4.71228025
    ]
}

df = pd.DataFrame(data)

# Clean up model names
model_name_mapping = {
    'anthropic/claude-sonnet-4-5': 'Claude Sonnet 4.5',
    'anthropic/claude-sonnet-4-5_high': 'Claude Sonnet 4.5 High',
    'claude-3-7-sonnet-20250219': 'Claude-3.7 Sonnet',
    'claude-3-7-sonnet-20250219_high': 'Claude-3.7 Sonnet High',
    'claude-haiku-4-5': 'Claude Haiku 4.5',
    'gpt-4.1-2025-04-14': 'GPT-4.1',
    'o3-2025-04-16': 'o3 Medium',
    'o4-mini-2025-04-16_high': 'o4-mini High',
    'o4-mini-2025-04-16_low': 'o4-mini Low',
    'openai/deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
    'openai/deepseek-ai/DeepSeek-V3': 'DeepSeek V3',
    'openai/gemini-2.0-flash': 'Gemini 2.0 Flash',
    'openai/gpt-5-2025-08-07': 'GPT-5 Medium',
    'openrouter/anthropic/claude-opus-4.1': 'Claude Opus 4.1',
    'openrouter/anthropic/claude-opus-4.1_high': 'Claude Opus 4.1 High',
    'claude-opus-4-5-20251101': 'Claude Opus 4.5',
    'claude-opus-4-5-20251101_high': 'Claude Opus 4.5 High',
    'openai/gemini-3-pro-preview': 'Gemini 3 Pro Preview'
}

df['model'] = df['model'].map(model_name_mapping)

# Sort by steps (ascending - fewer steps is better)
df = df.sort_values('steps', ascending=True)

# ═══════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════

BRAND_BASE = {
    "OpenAI": "#10A37F",
    "Anthropic": "#D4A27F",
    "DeepSeek": "#4D6BFE",
    "Google": "#DB4437"
}

COMP_OF = {
    "o3 Medium":"OpenAI", "o4-mini High":"OpenAI", "o4-mini Low":"OpenAI",
    "GPT-4.1":"OpenAI", "GPT-5 Medium":"OpenAI",
    "Claude-3.7 Sonnet":"Anthropic", "Claude-3.7 Sonnet High":"Anthropic",
    "Claude Sonnet 4.5": "Anthropic", "Claude Sonnet 4.5 High": "Anthropic",
    "Claude Opus 4.1": "Anthropic", "Claude Opus 4.1 High": "Anthropic",
    "Claude Opus 4.5": "Anthropic", "Claude Opus 4.5 High": "Anthropic",
    "Claude Haiku 4.5":"Anthropic",
    "DeepSeek R1":"DeepSeek", "DeepSeek V3":"DeepSeek",
    "Gemini 2.0 Flash": "Google", "Gemini 3 Pro Preview": "Google"
}

LOGO_PATHS = {
    "OpenAI":    str(quantitative_dir / "assets/openai_grey.png"),
    "Anthropic": str(quantitative_dir / "assets/anthropic_grey.png"),
    "DeepSeek":  str(quantitative_dir / "assets/deepseek_grey.png"),
    "Google":   str(quantitative_dir / "assets/gemini.png"),
}

def make_shades(base_hex, k, span=.20):
    """Create k shades of a base color."""
    r,g,b = mcolors.to_rgb(base_hex)
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    hi    = min(1, l+span)
    return [mcolors.to_hex(colorsys.hls_to_rgb(h,L,s))
            for L in np.linspace(l,hi,k)]

def setup_colors(models):
    """Set up color scheme for models by company."""
    company_models, shade_of = {}, {}
    for m in models:
        company_models.setdefault(COMP_OF.get(m,"Other"), []).append(m)
    for comp, mlist in company_models.items():
        base = BRAND_BASE.get(comp, "#777777")
        for m,c in zip(sorted(mlist), make_shades(base, len(mlist))):
            shade_of[m] = c
    return shade_of

# ═══════════════════════════════════════════════════════════════════
# CREATE PLOT
# ═══════════════════════════════════════════════════════════════════

models = df['model'].tolist()
shade_of = setup_colors(models)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Bar positioning
inner_gap = 0.01
group_w = 0.90
bar_w = (group_w - (len(models)-1)*inner_gap) / len(models)
offsets = (np.arange(len(models)) - (len(models)-1)/2) * (bar_w + inner_gap)

# Plot bars with confidence intervals
for off, (idx, row) in zip(offsets, df.iterrows()):
    model = row['model']
    steps = row['steps']
    ci = row['steps_ci']

    # Draw bar
    ax.bar(off, steps, width=bar_w,
           color=shade_of[model], edgecolor='black', linewidth=0.8,
           yerr=ci, capsize=4,
           error_kw={'linewidth': 1.5, 'ecolor': 'black'})

    # Add step count label on top of bar
    ax.text(off, steps + ci + 0.8, f"{steps:.1f}",
            ha='center', va='bottom', fontsize=9, weight='bold')

    # Add company logo
    comp = COMP_OF.get(model, "Other")
    logo_path = LOGO_PATHS.get(comp)
    if logo_path and os.path.exists(logo_path):
        img = mpimg.imread(logo_path)
        zoom = 18 / img.shape[0]  # 18px height
        ax.add_artist(AnnotationBbox(
            OffsetImage(img, zoom=zoom, resample=True),
            (off, steps + ci + 2.5), frameon=False, box_alignment=(0.5, 0)
        ))

# Style axes
ax.set_xticks(offsets)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, max(df['steps'] + df['steps_ci']) * 1.25)
ax.set_xlim(offsets.min() - bar_w/2 - 0.05, offsets.max() + bar_w/2 + 0.05)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
for spine in ('left', 'bottom'):
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('black')

ax.tick_params(axis='y', which='major', color='black', width=1, length=4)
ax.tick_params(axis='x', which='major', length=4, width=1, color='black',
               direction='out', bottom=True)
ax.set_ylabel("Average Steps per Task", fontsize=13, fontweight='bold')

# Add title and subtitle
fig.text(0.05, 0.96, "Holistic Agent Leaderboard (HAL) – CORE-Bench Hard",
         ha='left', va='top', fontsize=15, fontweight='bold')
fig.text(0.05, 0.92, "Average number of agentic steps used across all tasks. Agents may use additional steps either for productive \nproblem solving or due to repeated tool calling failures. Fewer steps generally indicate more efficient problem solving.",
         ha='left', va='top', fontsize=12, color='.35')

# Add footer
fig.subplots_adjust(bottom=0.30)
footer_note = (
    "All models are evaluated using the CORE-Agent scaffold. Error bars represent 95% confidence intervals.\n"
    "See evaluations of 197 agents across 9 benchmarks at hal.cs.princeton.edu."
)

ax_note = fig.add_axes([0.05, 0.03, 0.70, 0.10])
ax_note.axis('off')
ax_note.text(0, 0, footer_note, ha='left', va='bottom',
             fontsize=10, color='.35', wrap=True, linespacing=1.25)

# Add footer logos
from matplotlib.offsetbox import HPacker, AnchoredOffsetbox

footer_logo_paths = [
    str(quantitative_dir / "assets/HAL.png"),
    str(quantitative_dir / "hal-frontend/static/pli.png"),
    str(quantitative_dir / "hal-frontend/static/princeton.png"),
]

icons = []
for p in footer_logo_paths:
    if os.path.exists(p):
        img = mpimg.imread(p)
        zoom = 28 / img.shape[0]
        icons.append(OffsetImage(img, zoom=zoom, resample=True))

if icons:
    fig.add_artist(AnchoredOffsetbox(
        loc='lower right',
        child=HPacker(children=icons, align='center', pad=0, sep=5),
        frameon=False, pad=0,
        bbox_to_anchor=(0.9, 0.03),
        bbox_transform=fig.transFigure, borderpad=0
    ))

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.30)

# Save plot
output_path = quantitative_dir / "plots"
output_path.mkdir(exist_ok=True)
output_filename = output_path / "hal_corebench_hard_steps.png"
fig.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"Saved plot: {output_filename}")

# Close the figure to free memory
plt.close(fig)

# Uncomment the line below if you want to display the plot on screen
# plt.show()