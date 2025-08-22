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
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

## Structure

```
quantitative/
├── src/
│   └── dataloader.py        # Loads data from hal-frontend databases
├── notebooks/
│   └── benchmark_bar_plots.ipynb  # Creates benchmark plots
└── plots/                 # Generated plots

hal-frontend/                # Required submodule
└── preprocessed_traces/     # SQLite database
```

## Usage

**Load data:**
```python
from src.dataloader import load_most_recent_df
df = load_most_recent_df()  # Gets latest results from all benchmarks
```

**Create plots:**
Run `notebooks/benchmark_bar_plots.ipynb` - it handles paths automatically and outputs to `hal-frontend/analysis/plots/`.