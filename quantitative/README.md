# Quantitative Analysis

Statistical analysis and visualization tools for HAL benchmark data.

## Setup

1. **Add hal-frontend as submodule** (from quantitative folder):
   ```bash
   cd quantitative
   git submodule add https://github.com/fsndzomga/hal-frontend.git hal-frontend
   git submodule update --init --recursive
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

## Structure

```
quantitative/
├── src/                      # Utility modules
│   ├── dataloader.py        # Load data from hal-frontend databases
│   ├── pareto_utils.py      # Pareto frontier computation
│   └── plotting_utils.py    # Color schemes and styling
├── scripts/                 # Standalone analysis scripts
│   ├── pareto_accuracy_cost.py      # Cost vs accuracy Pareto analysis
│   ├── pareto_accuracy_tokens.py    # Token vs accuracy analysis
│   ├── reasoning_comparisons.py     # Reasoning level impact analysis
│   ├── agent_comparisons.py         # Agent scaffold comparisons
│   └── benchmark_costs.py           # Cost analysis across benchmarks
├── plots/                   # Generated visualizations (PDF/PNG)
└── hal-frontend/            # Required submodule with SQLite databases
```

## Usage

**Run analysis scripts:**
```bash
python scripts/pareto_accuracy_cost.py      # Generates cost vs accuracy plots + statistics
python scripts/reasoning_comparisons.py     # Analyzes reasoning level impact
python scripts/agent_comparisons.py         # Compares agent scaffolds
```

**Load data directly:**
```python
from src.dataloader import load_paper_df
model_df, agent_df, benchmark_df = load_paper_df()  # Gets processed datasets
```

**Use utilities:**
```python
from src.pareto_utils import compute_pareto_frontier_with_origin
from src.plotting_utils import setup_colors
```

All scripts save plots to the `plots/` directory and print summary statistics to the terminal.