import os

# File filtering patterns
# BENCHMARK_AGENT_PREFIX = "assistantbench_assistantbench_browser_agent"
BENCHMARK_AGENT_PREFIX = "taubench_airline_taubench_fewshot_high_reasoning_claudeopus41_1754432663_UPLOAD.zip"
# BENCHMARK_AGENT_PREFIX = "scicode_scicode_tool_calling"
# BENCHMARK_AGENT_PREFIX = "corebench_hard_coreagent"

# Excluded files (files to skip during processing)
# EXCLUDED_FILES = {
#     'assistantbench_assistantbench_browser_agent_claude37sonnet20250219_low_1748711087_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_deepseekr1_1755121049_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_gemini20flash_1746393958_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_gpt5_1754598271_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_o320250416_1746376643_UPLOAD.zip',
#     'assistantbench_assistantbench_browser_agent_o4mini20250416_1746227177_UPLOAD.zip'
# }

EXCLUDED_FILES = {
    "taubench_airline_taubench_fewshot_o3mini20250131_high_1744743428_UPLOAD.zip",
    "taubench_airline_taubench_fewshot_o320250403_1744728447_UPLOAD.zip",
    "scicode_scicode_tool_calling_agent_o4mini20250416_1745267192_UPLOAD.zip",
    "corebench_hard_coreagent_1744839552_UPLOAD.zip"
}

# Processing limits
DEFAULT_TASK_LIMIT = 1 # None means process all tasks

# Output settings
DEFAULT_OUTPUT_FILE = "taubench_fewshot_runs_test.json"
# DEFAULT_OUTPUT_FILE = "scicode_tool_calling_runs_test.json"
# DEFAULT_OUTPUT_FILE = "corebench_coreagent_runs_test.json"

# Collection name
# COLLECTION_NAME = "corebench_1_task"
COLLECTION_NAME = "taubench_fewshot_1_task"