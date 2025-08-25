import os

# File filtering patterns - CHANGE THIS LINE TO SWITCH BENCHMARKS
BENCHMARK_AGENT_PREFIX = "scicode_scicode_tool_calling"
# BENCHMARK_AGENT_PREFIX = "assistantbench_assistantbench_browser_agent"
# BENCHMARK_AGENT_PREFIX = "taubench_airline_taubench_fewshot"  
# BENCHMARK_AGENT_PREFIX = "corebench_hard_coreagent"

# Processing limits
DEFAULT_TASK_LIMIT = 10  # None means process all tasks

# Excluded files (files to skip during processing)
EXCLUDED_FILES = {
    'assistantbench_assistantbench_browser_agent_claude37sonnet20250219_low_1748711087_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_deepseekr1_1755121049_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_gemini20flash_1746393958_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_gpt5_1754598271_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_o320250416_1746376643_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_o4mini20250416_1746227177_UPLOAD.zip',
    "taubench_airline_taubench_fewshot_o3mini20250131_high_1744743428_UPLOAD.zip",
    "taubench_airline_taubench_fewshot_o320250403_1744728447_UPLOAD.zip",
    "scicode_scicode_tool_calling_agent_o4mini20250416_1745267192_UPLOAD.zip",
    "corebench_hard_coreagent_1744839552_UPLOAD.zip"
}

# Auto-generated settings based on benchmark prefix
def _get_benchmark_name():
    """Extract clean benchmark name from prefix."""
    # Mapping for complex prefixes to clean names
    prefix_mapping = {
        "assistantbench_assistantbench_browser_agent": "assistantbench_browser_agent",
        "taubench_airline_taubench_fewshot": "taubench_fewshot",
        "scicode_scicode_tool_calling": "scicode_tool_calling",
        "corebench_hard_coreagent": "corebench_coreagent"
    }
    return prefix_mapping.get(BENCHMARK_AGENT_PREFIX, BENCHMARK_AGENT_PREFIX)

def _get_collection_base_name():
    """Extract base name for collection."""
    return BENCHMARK_AGENT_PREFIX.split('_')[0]

# Derived settings - automatically generated
DEFAULT_OUTPUT_FILE = f"outputs/{_get_benchmark_name()}_runs_test.json"
COLLECTION_NAME = f"{_get_collection_base_name()}_{DEFAULT_TASK_LIMIT}_tasks"