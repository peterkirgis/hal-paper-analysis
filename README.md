# HAL Pipeline for Docent

Pipeline to extract, process, and upload agent conversation data from the Holistic Agent Leaderboard (HAL) dataset to Docent for qualitative analysis.

## Requirements

### API Keys
- **HuggingFace Token**: Required to access the HAL dataset
- **Docent API Key**: Required to upload data to Docent collections

### Environment Setup
Create a `.env` file in the root directory:
```bash
HF_TOKEN=your_huggingface_token_here
DOCENT_API_KEY=your_docent_api_key_here
```

### Dependencies
```bash
pip install python-dotenv docent huggingface_hub cryptography
```

## Quick Start

### 1. Configure the Pipeline
Edit `src/docent_pipeline/config.py`:

```python
# Change benchmark/agent type to process
BENCHMARK_AGENT_PREFIX = "assistantbench_assistantbench_browser_agent"

# Set processing limits (None = process all)
DEFAULT_TASK_LIMIT = 10  # Start with 10 tasks for testing

# Set collection name
COLLECTION_NAME = "your_collection_name"
```

### 2. Run the Pipeline
```bash
python main.py
```

The pipeline will:
1. Download and decrypt HAL dataset files
2. Extract agent conversations and evaluation results
3. Convert to Docent format
4. Upload to a new Docent collection

### 3. Test from Intermediate Results
If you have existing intermediate JSON results:
```bash
python test_from_intermediate.py --limit-tasks 5 --collection-name "test_run"
```

## Configuration Options

### Key Settings in `config.py`:
- `BENCHMARK_AGENT_PREFIX`: Filter files by name prefix
- `EXCLUDED_FILES`: Skip specific problematic files  
- `DEFAULT_TASK_LIMIT`: Limit tasks processed (useful for testing)
- `COLLECTION_NAME`: Default collection name for uploads

### Command Line Options:
```bash
python main.py --help
```

## Output

The pipeline produces:
- **Intermediate JSON**: Raw extracted data (`assistantbench_browser_agent_runs.json`)
- **Docent Collection**: Uploaded conversations with metadata including scores and evaluation results
- **Upload Statistics**: Success/failure counts and error details
