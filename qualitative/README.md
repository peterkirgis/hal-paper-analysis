# Qualitative Analysis

This section contains the data extraction and processing pipeline for qualitative analysis of HAL paper benchmarks.

## Pipeline Components

- **src/docent_pipeline/**: Data extraction and upload pipeline
  - `download_utils.py`: Download and parse benchmark data from HuggingFace
  - `processing_utils.py`: Convert data to docent format
  - `upload_utils.py`: Upload processed data to docent
  - `config.py`: Configuration settings

- **main.py**: Main pipeline orchestrator
- **notebooks/**: Jupyter notebooks for analysis
- **testing/**: Test scripts and utilities
- **outputs/**: Generated JSON files and results

## Usage

From the qualitative directory:

```bash
cd qualitative
python main.py
```

## Features

- Extracts agent conversations from encrypted benchmark data
- Handles multiple model types and benchmarks (AssistantBench, TauBench, SciCode, etc.)
- Deduplicates messages and handles role assignment for multi-model scenarios
- Extracts metadata including reasoning effort parameters
- Uploads to docent for analysis