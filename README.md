# HAL Paper Analysis

Analysis tools for the Holistic Agent Leaderboard (HAL) dataset, split into qualitative and quantitative analysis sections.

## Repository Structure

```
hal-paper-analysis/
├── qualitative/          # Data extraction and conversation analysis
│   ├── full_pipeline.py # Complete extraction pipeline
│   └── results/         # Analysis notebooks and outputs
└── quantitative/        # Statistical analysis and visualization
    ├── src/             # Utility modules (dataloader, pareto, plotting)
    ├── scripts/         # Standalone analysis scripts
    ├── plots/           # Generated visualizations
    └── hal-frontend/    # HAL data submodule
```

## Getting Started

### Qualitative Analysis
Extracts agent conversations from HAL datasets and uploads to Docent for analysis.

**Setup:**
```bash
cd qualitative
pip install python-dotenv docent huggingface_hub cryptography
```

**Usage:**
```bash
python full_pipeline.py  # Processes benchmark data and uploads to Docent
```

See `qualitative/README.md` for detailed configuration options.

### Quantitative Analysis
Creates statistical analysis and visualizations from HAL benchmark results.

**Setup:**
```bash
cd quantitative
git submodule add https://github.com/fsndzomga/hal-frontend.git hal-frontend
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Usage:**
```bash
python scripts/pareto_accuracy_cost.py      # Cost vs accuracy analysis
python scripts/reasoning_comparisons.py     # Reasoning level comparisons
python scripts/agent_comparisons.py         # Agent scaffold comparisons
```

Or import utilities:
```python
from src.dataloader import load_paper_df
model_df, agent_df, benchmark_df = load_paper_df()
```

See `quantitative/README.md` for detailed script descriptions.

## Requirements

- Python 3.8+
- HuggingFace Token (for qualitative analysis)
- Docent API Key (for qualitative analysis)
