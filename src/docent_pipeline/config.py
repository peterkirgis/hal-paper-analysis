import os

# File filtering patterns
# BENCHMARK_AGENT_PREFIX = "assistantbench_assistantbench_browser_agent"
# BENCHMARK_AGENT_PREFIX = "taubench_airline_taubench_fewshot"
BENCHMARK_AGENT_PREFIX = "scicode_scicode_tool_calling"

# Excluded files (files to skip during processing)
# EXCLUDED_FILES = {
#     'assistantbench_assistantbench_browser_agent_claude37sonnet20250219_low_1748711087_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_deepseekr1_1755121049_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_gemini20flash_1746393958_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_gpt5_1754598271_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_o320250416_1746376643_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_o4mini20250416_1746227177_UPLOAD.zip'
# }
EXCLUDED_FILES = {}

# Processing limits
DEFAULT_TASK_LIMIT = 1  # None means process all tasks

# Output settings
DEFAULT_OUTPUT_FILE = "scicode_tool_calling_runs_test.json"

# Collection name
COLLECTION_NAME = "scicode_tool_calling_test_1"