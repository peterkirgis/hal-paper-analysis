# Qualitative Analysis

This section contains the data extraction and processing pipeline for qualitative analysis of HAL paper benchmarks.

## Structure

```
qualitative/
├── full_pipeline.py         # Complete extraction and analysis pipeline
├── main.py                  # Main pipeline orchestrator
├── hal_pipeline_results.json # Extracted conversation data
└── results/                 # Analysis notebooks and outputs
    ├── docent_rubric_analysis.ipynb  # Rubric-based analysis
    └── docent_figures.ipynb         # Visualization notebooks
```

## Usage

**Run extraction pipeline:**
```bash
cd qualitative
python main.py              # Basic pipeline
python full_pipeline.py     # Complete extraction with all features
```

**Analyze results:**
Open notebooks in `results/` directory to analyze extracted conversations and generate figures.

## Features

- Extracts agent conversations from encrypted benchmark data
- Handles multiple model types and benchmarks (AssistantBench, TauBench, SciCode, etc.)
- Deduplicates messages and handles role assignment for multi-model scenarios
- Extracts metadata including reasoning effort parameters
- Generates structured JSON output for further analysis
- Provides analysis notebooks for rubric-based evaluation and visualization