# HAL Paper Analysis

Analysis tools for the Holistic Agent Leaderboard (HAL) dataset, split into qualitative and quantitative analysis sections.

## Repository Structure

```
hal-paper-analysis/
├── qualitative/          # Data extraction and conversation analysis
│   ├── src/             # Pipeline for processing HAL data
│   ├── main.py          # Main extraction pipeline
│   ├── notebooks.py     # Testing notebooks
│   └── outputs/         # Generated JSON files
└── quantitative/        # Statistical analysis and visualization
    ├── src/             # Data loading and analysis tools
    ├── notebooks/       # Analysis notebooks
    └── plots/           # Generated visualizations

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
python main.py  # Processes benchmark data and uploads to Docent
```

See `qualitative/README.md` for detailed configuration options.

### Quantitative Analysis  
Creates statistical analysis and visualizations from HAL benchmark results.

**Setup:**
```bash
cd quantitative
git submodule add https://github.com/fsndzomga/hal-frontend.git hal-frontend
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Usage:**
```python
from src.dataloader import load_most_recent_df
df = load_most_recent_df()  # Load benchmark results
```

See `quantitative/README.md` for visualization and analysis examples.

## Requirements

- Python 3.8+
- HuggingFace Token (for qualitative analysis)
- Docent API Key (for qualitative analysis)
