"""
Common utilities for integrating HAL benchmark data with Docent.

This module contains shared functions and classes for processing various benchmark
data formats (TauBench, CoreBench, etc.) and converting them to Docent format.
"""

import ast
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import Field

from docent.data_models import BaseAgentRunMetadata
from docent.data_models.chat import ChatMessage, ToolCall, parse_chat_message


def extract_metadata_from_config(
    config_data: Dict[str, Any] | None, model_name_from_json: str, benchmark_name: str
) -> Dict[str, Any]:
    """
    Extract direct metadata fields from agent configuration.

    Args:
        config_data: Configuration data containing agent_args and other config info
        model_name_from_json: Model name extracted from JSON logging data
        benchmark_name: Name of the benchmark being processed

    Returns:
        Dict containing extracted metadata fields
    """
    metadata = {
        "model_name": model_name_from_json,  # Use the model from JSON as primary source
        "agent_name": None,
        "reasoning_effort": None,
        "budget": None,
        "date": None,
        "run_id": None,
        "benchmark_name": benchmark_name,
    }

    if config_data:
        # Extract from agent_args if available
        agent_args = config_data.get("agent_args", {})
        if agent_args:
            metadata["reasoning_effort"] = agent_args.get("reasoning_effort")
            metadata["budget"] = agent_args.get("budget")

        # Extract date, run_id, and agent_name from top level config
        metadata["date"] = config_data.get("date")
        metadata["run_id"] = config_data.get("run_id")
        metadata["agent_name"] = config_data.get("agent_name")

        # If model_name is not in agent_args, try to get it from there as fallback
        if agent_args.get("model_name") and not metadata["model_name"]:
            metadata["model_name"] = agent_args.get("model_name")

    return metadata


def save_failed_sanity_check_logs(
    task_logs: list, task_id: str, model_name: str, benchmark: str = "unknown"
):
    """
    Save failed sanity check logs to failed_sanity_checks directory.

    Args:
        task_logs (list): The list of logs that failed sanity check
        task_id (str): The task ID that failed
        model_name (str): The model name
        benchmark (str): The benchmark name (default: "unknown")
    """
    # Create directory structure
    failed_dir = os.path.join("failed_sanity_checks", benchmark)
    os.makedirs(failed_dir, exist_ok=True)

    # Sanitize model name for file path (replace problematic characters)
    safe_model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")

    # Create filename
    filename = f"{safe_model_name}_{task_id}.json"
    filepath = os.path.join(failed_dir, filename)

    # Save the logs
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(task_logs, f, indent=2, ensure_ascii=False)
        print(f"   💾 Saved failed logs to: {filepath}")
    except Exception as e:
        print(f"   ⚠️ Failed to save logs to {filepath}: {e}")


def extract_task_ids(filtered_logs: list, task_id_key: str = "weave_task_id") -> list:
    """
    Extract sorted unique task IDs from filtered logs.

    Args:
        filtered_logs (list): List of filtered log entries.
        task_id_key (str, optional): The key used to extract task IDs. Defaults to "weave_task_id".

    Returns:
        list: A sorted list of unique task IDs.
    """
    return sorted(
        set(entry.get(task_id_key) for entry in filtered_logs if entry.get(task_id_key))
    )


def normalize_message_for_comparison(message: dict) -> dict:
    """
    Normalize a message dict by removing None values and standardizing keys.
    This ensures consistent comparison between messages that may have None values
    vs missing keys for fields like tool_calls, function_call, etc.

    Args:
        message (dict): The message to normalize

    Returns:
        dict: Normalized message with None values removed
    """
    normalized = {}
    for key, value in message.items():
        if value is not None:
            normalized[key] = value
    return normalized


def sanity_check(task_logs: list) -> bool:
    """
    Check that the 0th (largest) log contains all smaller logs as ordered subsets.

    Args:
        task_logs (list): List of logs for a given task_id.

    Returns:
        bool: True if the largest log captures all others, False otherwise.
    """
    # Sort logs by number of messages (largest first)
    task_logs_sorted = sorted(
        task_logs, key=lambda x: len(x["inputs"]["messages"]), reverse=True
    )
    largest_log = task_logs_sorted[0]
    largest_size = len(largest_log["inputs"]["messages"])

    for log in task_logs_sorted[1:]:
        msgs_small = log["inputs"]["messages"]
        msgs_large = largest_log["inputs"]["messages"]

        # If logs have the same number of messages, check if they're identical
        if len(msgs_small) == largest_size:
            # Check if they're actually identical (same content)
            for i, m in enumerate(msgs_small):
                normalized_small = normalize_message_for_comparison(m)
                normalized_large = normalize_message_for_comparison(msgs_large[i])
                if normalized_small != normalized_large:
                    return False  # Same size but different content - that's a violation
            continue  # They're truly identical, skip to next

        # If small log is somehow larger than largest, that's a violation
        if len(msgs_small) > largest_size:
            return False  # sanity violation

        # Check if smaller log is a prefix of larger log
        for i, m in enumerate(msgs_small):
            normalized_small = normalize_message_for_comparison(m)
            normalized_large = normalize_message_for_comparison(msgs_large[i])
            if normalized_small != normalized_large:
                return False  # mismatch
    return True


def filter_logs_by_model(
    model_name: str,
    data: dict,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
) -> list:
    """
    Filter raw logging results for a given model name and system prompt prefix.

    Args:
        model_name (str): The model name to filter on.
        data (dict): The dictionary containing 'raw_logging_results'.
        system_prompt_prefix (str, optional): The prefix that the system prompt text must start with.
            Defaults to "You are an expert assistant who can solve any task using code blobs".

    Returns:
        list: A list of filtered log entries.
    """
    filtered_entries = []
    for entry in data.get("raw_logging_results", []):
        if (
            entry["inputs"].get("model") == model_name
            and entry["inputs"].get("messages")
            and entry["inputs"]["messages"][0].get("role") == "system"
        ):
            first_message = entry["inputs"]["messages"][0]
            content = first_message.get("content")

            content_matches = False
            if (
                isinstance(content, list)
                and len(content) > 0
                and isinstance(content[0], dict)
            ):
                content_text = content[0].get("text", "")
                content_matches = content_text.startswith(system_prompt_prefix)
            elif isinstance(content, str):
                content_matches = content.startswith(system_prompt_prefix)

            if content_matches:
                filtered_entries.append(entry)

    return filtered_entries


def task_id_to_transcript(
    data: dict,
    model_name: str,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
    task_id_key: str = "weave_task_id",
    benchmark: str = "unknown",
) -> Tuple[dict, int]:
    """
    Build a dictionary mapping each task_id to its full transcript (messages)
    from the largest log, after sanity check.

    Args:
        data (dict): Raw logging data.
        model_name (str): Model name to filter logs on.
        system_prompt_prefix (str, optional): Prefix that system prompt must start with.
        task_id_key (str, optional): Key for task IDs in logs. Defaults to "weave_task_id".
        benchmark (str, optional): Benchmark name for organizing failed logs. Defaults to "unknown".

    Returns:
        tuple: (transcripts_dict, failed_sanity_checks_count)
    """
    # Step 1: filter logs
    filtered_logs = filter_logs_by_model(model_name, data, system_prompt_prefix)

    # Step 2: extract task_ids
    task_ids = extract_task_ids(filtered_logs, task_id_key=task_id_key)

    transcripts = {}
    failed_sanity_checks = 0

    # Step 3: process each task_id
    for task_id in task_ids:
        task_logs = [
            entry for entry in filtered_logs if entry.get(task_id_key) == task_id
        ]

        # sanity check
        if not sanity_check(task_logs):
            # Save failed logs to file
            save_failed_sanity_check_logs(task_logs, task_id, model_name, benchmark)
            failed_sanity_checks += 1
            print(
                f"   ⚠️ Sanity check failed for task_id={task_id}, saved logs and continuing..."
            )
            continue

        # largest log is transcript
        task_logs_sorted = sorted(
            task_logs, key=lambda x: len(x["inputs"]["messages"]), reverse=True
        )
        transcripts[task_id] = task_logs_sorted[0]

    return transcripts, failed_sanity_checks


def extract_tool_calls(input_str):
    """
    Extract tool calls from assistant message content.

    Args:
        input_str (str): The message content to search for tool calls.

    Returns:
        list or None: List of parsed tool calls, or None if none found.
    """
    match = re.search(r"Calling tools:\s*(\[.*\])", input_str, re.DOTALL | re.MULTILINE)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None
    return None


def parse_messages_to_chat_messages(
    trajectories: List[Dict[str, Any]],
) -> List[ChatMessage]:
    """
    Convert raw message trajectories to ChatMessage objects.

    Args:
        trajectories (List[Dict[str, Any]]): List of raw message dictionaries.

    Returns:
        List[ChatMessage]: List of parsed ChatMessage objects.
    """
    messages: List[ChatMessage] = []

    for msg in trajectories:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, list):
            content = content[0].get("text", "")
        elif isinstance(content, str):
            pass
        else:
            content = ""

        message_data = {
            "role": role,
            "content": content,
        }
        raw_tool_calls = extract_tool_calls(content)

        if role == "assistant" and raw_tool_calls:
            parsed_tool_calls: List[ToolCall] = []
            for tc in raw_tool_calls:
                tool_call = ToolCall(
                    id=tc.get("id"),
                    function=tc.get("function", {}).get("name"),
                    arguments={
                        "code_to_run": raw_tool_calls[0]
                        .get("function", {})
                        .get("arguments", "")
                    },
                    type="function",
                    parse_error=None,
                )
                parsed_tool_calls.append(tool_call)
            message_data["tool_calls"] = parsed_tool_calls

        chat_message = parse_chat_message(message_data)
        messages.append(chat_message)

    return messages


class BaseBenchmarkMetadata(BaseAgentRunMetadata):
    """Base metadata class for benchmark agent runs."""

    benchmark_id: str = Field(description="The benchmark name")
    task_id: str = Field(
        description="The task within the benchmark that the agent is solving"
    )
    model: str = Field(description="The LLM used by the agent")
    run_id: str | None = Field(description="The run ID from config", default=None)

    # Direct metadata fields
    model_name: str = Field(description="Model name extracted from config or data")
    agent_name: str | None = Field(description="Agent name from config", default=None)
    reasoning_effort: str | None = Field(
        description="Reasoning effort level (high, medium, low, minimal, none)",
        default=None,
    )
    budget: int | float | None = Field(
        description="Budget limit for the agent run", default=None
    )
    date: str | None = Field(description="Date of the agent run", default=None)
    benchmark_name: str = Field(description="Name of the benchmark")
    accuracy: float | None = Field(
        description="Accuracy/performance score for the benchmark", default=None
    )

    agent_config: Dict[str, Any] | None = Field(
        description="Agent configuration including run parameters", default=None
    )
    additional_metadata: Dict[str, Any] = Field(
        description="Additional metadata about the task"
    )
    scoring_metadata: Dict[str, Any] | None = Field(
        description="Additional metadata about the scoring process"
    )


def download_missing_files(
    directory: str,
    file_pattern: str,
    collection_name_prefix: str,
    force_download: bool = False,
) -> List[str]:
    """
    Download benchmark files from Hugging Face repository.
    
    Args:
        directory (str): Local directory where files should be stored
        file_pattern (str): Regex pattern to match files
        collection_name_prefix (str): Prefix for determining the benchmark type
        force_download (bool): If True, download files even if they already exist locally
        
    Returns:
        List[str]: List of file paths that were downloaded (for cleanup)
    """
    try:
        from download import hf_download_decrypt_to_tempfile
    except ImportError:
        print("❌ Cannot import download module. Make sure download.py is available.")
        return []
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Map collection prefixes to HF repo paths and patterns
    # Files are directly in the root directory, not in subdirectories
    benchmark_mapping = {
        "CoreBench-Generalist": ("", r"corebench_hard_hal_generalist_agent(.+?)_\d+_UPLOAD\.zip$"),
        "CoreBench-Specialist": ("", r"corebench_hard_coreagent_\d+_UPLOAD\.zip$"),
        "SciCode-Generalist": ("", r"scicode_hal_generalist_agent(.+?)_\d+_UPLOAD\.zip$"),
        "SciCode-Specialist": ("", r"scicode_scicode_(.+?)_agent(.+?)_\d+_UPLOAD\.zip$"),
        "TauBench-Generalist": ("", r"taubench_airline_hal_generalist_(.+?)_\d+_UPLOAD\.zip$"),
        "TauBench-Specialist": ("", r"taubench_airline_taubench_fewshot_(.+?)_\d+_UPLOAD\.zip$"),
        "AssistantBench-Generalist": ("", r"assistantbench_hal_generalist_agent(.+?)_\d+_UPLOAD\.zip$"),
        "AssistantBench-Specialist": ("", r"assistantbench_assistantbench_browser_agent(.+?)_\d+_UPLOAD\.zip$"),
    }
    
    if collection_name_prefix not in benchmark_mapping:
        print(f"❌ Unknown benchmark type: {collection_name_prefix}")
        return []
    
    hf_subdir, zip_pattern = benchmark_mapping[collection_name_prefix]
    
    if force_download:
        print(f"📥 Force downloading files for {collection_name_prefix}...")
    else:
        print(f"📥 Downloading missing files for {collection_name_prefix}...")
    
    try:
        from huggingface_hub import HfFileSystem
        
        fs = HfFileSystem()
        repo_path = "datasets/agent-evals/hal_traces@main"
        hf_full_path = f"{repo_path}/{hf_subdir}" if hf_subdir else repo_path
        
        # List all files in the HF directory
        try:
            hf_files = fs.ls(hf_full_path, detail=False)
            zip_files = [f for f in hf_files if re.search(zip_pattern, os.path.basename(f))]
            
            if not zip_files:
                print(f"⚠️ No matching files found in HF repository for pattern: {zip_pattern}")
                return []
            
            downloaded_files = []
            downloaded_count = 0
            for zip_file_path in zip_files:
                zip_filename = os.path.basename(zip_file_path)
                # Convert .zip to .json for local storage
                json_filename = zip_filename.replace("_UPLOAD.zip", "_UPLOAD.json")
                local_json_path = os.path.join(directory, json_filename)
                
                # Check if file already exists locally (skip only if not force downloading)
                if os.path.exists(local_json_path) and not force_download:
                    continue
                
                print(f"   📥 Downloading: {zip_filename}")
                try:
                    # Download and decrypt the file
                    temp_json_path = hf_download_decrypt_to_tempfile(zip_filename, repo_id="agent-evals/hal_traces")
                    
                    # Read the decrypted JSON and reformat with proper indentation
                    with open(temp_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Write with 4-space indentation
                    with open(local_json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    # Clean up the temporary file
                    os.remove(temp_json_path)
                    
                    downloaded_files.append(local_json_path)
                    downloaded_count += 1
                    print(f"   ✅ Saved: {json_filename}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to download {zip_filename}: {e}")
                    continue
            
            if downloaded_count > 0:
                print(f"✅ Downloaded {downloaded_count} files to {directory}")
            else:
                print(f"ℹ️ All files already exist locally in {directory}")
            
            return downloaded_files
                
        except Exception as e:
            print(f"❌ Error listing HF repository contents: {e}")
            return []
            
    except ImportError:
        print("❌ huggingface_hub not available. Cannot download files.")
        return []
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return []


def cleanup_downloaded_files(downloaded_files: List[str]) -> None:
    """
    Clean up downloaded files to save disk space.
    
    Args:
        downloaded_files (List[str]): List of file paths to delete
    """
    if not downloaded_files:
        return
        
    print(f"🧹 Cleaning up {len(downloaded_files)} downloaded files...")
    cleaned_count = 0
    
    for file_path in downloaded_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_count += 1
                print(f"   🗑️  Deleted: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   ⚠️ Failed to delete {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} files")


def cleanup_temp_directory(temp_directory: str) -> None:
    """
    Clean up entire temporary directory to save disk space.
    
    Args:
        temp_directory (str): Path to temporary directory to delete
    """
    if not temp_directory or not os.path.exists(temp_directory):
        return
        
    print(f"🧹 Cleaning up temporary directory: {temp_directory}")
    
    try:
        shutil.rmtree(temp_directory)
        print(f"✅ Successfully removed temporary directory: {temp_directory}")
    except Exception as e:
        print(f"⚠️ Failed to remove temporary directory {temp_directory}: {e}")
        # Fallback: try to remove individual files
        try:
            for root, dirs, files in os.walk(temp_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            os.rmdir(temp_directory)
            print(f"✅ Cleaned up temporary directory using fallback method")
        except Exception as fallback_error:
            print(f"❌ Failed to clean up temporary directory: {fallback_error}")


def analyze_benchmark_files(
    directory: str,
    file_pattern: str,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
    download_if_missing: bool = False,
    collection_name_prefix: str = None,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Generic function to analyze benchmark files and extract model names from both file path and JSON content.

    Only processes entries where:
    - The entry has the expected structure with inputs/messages
    - "self" is NOT present in the inputs (filters out certain entry types)
    - The first message is a system message with the expected prompt prefix

    Args:
        directory (str): Path to the directory containing benchmark JSON files
        file_pattern (str): Regex pattern to match files and extract model names
        system_prompt_prefix (str): The prefix that the system prompt text must start with
        download_if_missing (bool): Whether to download files if directory is empty or missing
        collection_name_prefix (str): Prefix for determining benchmark type for downloads

    Returns:
        Tuple[Dict[str, Dict[str, str]], List[str]]: 
            - Dictionary mapping file paths to model name information:
              {
                  'file_path': {
                      'model_name_from_file_path': str,
                      'model_name_from_json_content': str
                  }
              }
            - List of downloaded file paths (for cleanup)

    Raises:
        ValueError: If model names are inconsistent within a file or file pattern doesn't match
        OSError: If the directory doesn't exist or can't be accessed
        json.JSONDecodeError: If a file contains invalid JSON
    """
    if not os.path.exists(directory):
        if download_if_missing and collection_name_prefix:
            print(f"📁 Directory {directory} does not exist, creating it...")
            os.makedirs(directory, exist_ok=True)
        else:
            raise OSError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise OSError(f"Path is not a directory: {directory}")

    # Download files if requested
    downloaded_files = []
    if download_if_missing and collection_name_prefix:
        # Always download when download_if_missing is True (since we're using temp directory)
        print(f"📁 Download requested, downloading files for {collection_name_prefix}...")
        downloaded_files = download_missing_files(directory, file_pattern, collection_name_prefix, force_download=True)

    results = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Skip if not a JSON file matching our pattern
        match = re.search(file_pattern, filename)
        if not match:
            continue

        # Extract model name from file path
        try:
            model_name_from_path = (
                match.group(1) if match.groups() and match.group(1) else "default"
            )
        except IndexError:
            model_name_from_path = "default"

        # Build full file path
        file_path = os.path.join(directory, filename)

        # Load JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")

        # Extract model names from JSON content
        models_from_json = set()

        for entry in data.get("raw_logging_results", []):
            # Check if entry has the expected structure and "self" is not present
            if (
                entry.get("inputs")
                and "self" not in entry["inputs"]
                and entry["inputs"].get("messages")
                and isinstance(entry["inputs"]["messages"], list)
                and len(entry["inputs"]["messages"]) > 0
            ):
                # Check system prompt

                first_message = entry["inputs"]["messages"][0]
                content = first_message.get("content")

                content_matches = False
                if (
                    isinstance(content, list)
                    and len(content) > 0
                    and isinstance(content[0], dict)
                ):
                    content_text = content[0].get("text", "")
                    content_matches = content_text.startswith(system_prompt_prefix)
                elif isinstance(content, str):
                    content_matches = content.startswith(system_prompt_prefix)

                if first_message.get("role") == "system" and content_matches:
                    # Extract model name
                    model = entry["inputs"].get("model")
                    if model:
                        models_from_json.add(model)

        # Check that there's only one unique model in the JSON
        if len(models_from_json) == 0:
            raise ValueError(f"No model found in JSON content for file: {file_path}")
        elif len(models_from_json) > 1:
            raise ValueError(
                f"Multiple models found in JSON content for file {file_path}: {models_from_json}"
            )

        model_name_from_json = list(models_from_json)[0]

        # Store results
        results[file_path] = {
            "model_name_from_file_path": model_name_from_path,
            "model_name_from_json_content": model_name_from_json,
        }

    return results, downloaded_files


def default_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs
):
    """
    Default task processor for dictionary-based eval results.

    Args:
        task_logs_dict: Dictionary mapping task_id to log entries
        data: Raw data containing eval results
        model_name: Model name from file path (may be "default")
        conversion_function: Function to convert log entry to AgentRun
        max_runs: Maximum number of runs to process

    Returns:
        List[AgentRun]: List of converted agent runs
    """
    from docent.data_models import AgentRun

    agent_runs = []
    processed = 0

    # Extract config for all tasks in this file
    config_data = data.get("config", {})

    # Use the model name from JSON extraction

    for task_id, log_entry in task_logs_dict.items():
        if processed >= max_runs:
            break

        eval_results_data = data.get("raw_eval_results", {}).get(task_id)

        if not eval_results_data:
            print(f"   ⚠️ No raw_eval_results found for task_id={task_id}, skipping.")
            continue

        try:
            agent_run = conversion_function(
                log_entry, model_name, eval_results_data, config_data
            )
            agent_runs.append(agent_run)
            processed += 1
        except Exception as e:
            print(f"   ❌ Error processing task_id={task_id}: {e}")
            continue

    return agent_runs


def process_benchmark_files(
    directory: str,
    file_pattern: str,
    conversion_function,
    collection_name_prefix: str,
    system_prompt_prefix: str,
    dry_run: bool = False,
    max_files: int = 5,
    max_runs_per_model: int = 5,
    task_processor=None,
    generate_report: bool = True,
    download_if_missing: bool = False,
):
    """
    Generic function to process benchmark files and return agent runs.

    Args:
        directory (str): Directory containing benchmark files
        file_pattern (str): Regex pattern to match files
        conversion_function: Function to convert log entry to AgentRun
        collection_name_prefix (str): Prefix for collection name
        dry_run (bool): Whether to run in dry mode
        max_files (int): Maximum files to process in dry run
        max_runs_per_model (int): Maximum runs per model in dry run
        system_prompt_prefix (str): System prompt prefix to filter on
        task_processor: Optional function to process tasks and return agent runs
        generate_report (bool): Whether to generate a PDF report of the results
        download_if_missing (bool): Whether to download files from HF if not found locally

    Returns:
        Tuple[List[AgentRun], str, str]: List of agent runs, collection name, and report path (if generated)
    """
    from docent.data_models import AgentRun

    # Create timestamped temporary directory if downloading
    temp_directory = None
    original_directory = directory
    
    if download_if_missing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_directory = f"hal_traces_{timestamp}"
        print(f"📁 Creating temporary directory: {temp_directory}")
        os.makedirs(temp_directory, exist_ok=True)
        directory = temp_directory

    # Step 1: Analyze all files to get model mappings
    print(f"🔍 Analyzing {collection_name_prefix} files...")
    file_model_mappings, downloaded_files = analyze_benchmark_files(
        directory, file_pattern, system_prompt_prefix, download_if_missing, collection_name_prefix
    )

    print(f"📁 Found {len(file_model_mappings)} {collection_name_prefix} files")

    agent_runs: List[AgentRun] = []
    failed_sanity_checks_by_model = {}
    eval_failures_by_model = {}
    model_success_stats = {}  # Track success stats by model

    # Step 2: Process files based on dry-run flag
    files_to_process = list(file_model_mappings.items())
    if dry_run:
        files_to_process = files_to_process[
            :max_files
        ]  # Only first N files for dry run
        print(f"🧪 Dry run mode: Processing {len(files_to_process)} files")
    else:
        print(f"🚀 Full mode: Processing all {len(files_to_process)} files")

    for file_path, model_info in files_to_process:
        print(f"\n📄 Processing: {os.path.basename(file_path)}")
        print(f"   Model (file): {model_info['model_name_from_file_path']}")
        print(f"   Model (JSON): {model_info['model_name_from_json_content']}")

        # Load the JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = model_info["model_name_from_json_content"]

        # Step 3: get task_id -> largest log (transcript)
        benchmark_name = collection_name_prefix.lower().replace(" ", "_")
        task_logs_dict, failed_count = task_id_to_transcript(
            data, model_name, system_prompt_prefix, benchmark=benchmark_name
        )

        # Initialize model stats if not exists
        if model_name not in model_success_stats:
            # In dry run mode, we limit the runs, so total_tasks_attempted should reflect actual processing
            total_tasks_available = len(task_logs_dict) + failed_count
            max_runs_to_process = (
                max_runs_per_model if dry_run else total_tasks_available
            )

            model_success_stats[model_name] = {
                "total_tasks_attempted": total_tasks_available,
                "total_tasks_processed": min(
                    max_runs_to_process, len(task_logs_dict)
                ),  # Actual tasks we'll process
                "successful_runs": 0,
                "sanity_check_failures": 0,
                "eval_failures": 0,
                "success_rate": 0.0,
            }

        # Track failed sanity checks by model
        if failed_count > 0:
            if model_name not in failed_sanity_checks_by_model:
                failed_sanity_checks_by_model[model_name] = 0
            failed_sanity_checks_by_model[model_name] += failed_count
            model_success_stats[model_name]["sanity_check_failures"] = failed_count

        # Step 4: process tasks using the appropriate processor
        max_runs = max_runs_per_model if dry_run else len(task_logs_dict)
        processor = task_processor or default_task_processor

        file_agent_runs = processor(
            task_logs_dict, data, model_name, conversion_function, max_runs
        )
        agent_runs.extend(file_agent_runs)

        # Update success statistics
        model_success_stats[model_name]["successful_runs"] = len(file_agent_runs)
        # Use total_tasks_processed for dry run mode to get accurate success rate
        total_processed = model_success_stats[model_name]["total_tasks_processed"]
        if total_processed > 0:
            model_success_stats[model_name]["success_rate"] = (
                len(file_agent_runs) / total_processed
            ) * 100

        print(f"   ✅ Processed {len(file_agent_runs)} agent runs from this file")

    print(f"\n🎯 Total processed agent runs: {len(agent_runs)}")

    # Print failed sanity checks summary
    if failed_sanity_checks_by_model:
        print(f"\n⚠️ Sanity Check Failures Summary:")
        for model, count in failed_sanity_checks_by_model.items():
            print(f"   {model}: {count} failed sanity checks")
        total_sanity_failures = sum(failed_sanity_checks_by_model.values())
        print(f"   Total failures: {total_sanity_failures}")
        print(
            f"   💾 Failed logs saved to: failed_sanity_checks/{collection_name_prefix.lower().replace(' ', '_')}/"
        )
    else:
        print(f"\n✅ No sanity check failures!")

    # Print model success statistics
    if model_success_stats:
        print(f"\n📊 Model Success Statistics:")
        # Sort by success rate (highest first)
        sorted_models = sorted(
            model_success_stats.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True,
        )

        for model, stats in sorted_models:
            success_rate = stats.get("success_rate", 0)
            successful_runs = stats.get("successful_runs", 0)
            total_processed = stats.get("total_tasks_processed", 0)
            total_attempted = stats.get("total_tasks_attempted", 0)
            sanity_failures = stats.get("sanity_check_failures", 0)
            eval_failures = stats.get("eval_failures", 0)

            status_emoji = (
                "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
            )

            # Show different format for dry run vs full run
            if dry_run:
                print(
                    f"   {status_emoji} {model}: {success_rate:.1f}% success ({successful_runs}/{total_processed} runs processed, {total_attempted} available)"
                )
            else:
                print(
                    f"   {status_emoji} {model}: {success_rate:.1f}% success ({successful_runs}/{total_processed} runs)"
                )

            if sanity_failures > 0 or eval_failures > 0:
                print(
                    f"      └─ Failures: {sanity_failures} sanity, {eval_failures} eval"
                )

    collection_name = (
        f"{collection_name_prefix} Collection ({'Dry Run' if dry_run else 'Full Run'})"
    )

    # Generate PDF report if requested
    report_path = None
    if generate_report:
        try:
            from report_generator import generate_benchmark_report

            processing_summary = {
                "total_files_processed": len(files_to_process),
                "total_runs_processed": len(agent_runs),
                "dry_run_mode": dry_run,
                "max_files_limit": max_files if dry_run else "No limit",
                "max_runs_per_model_limit": max_runs_per_model
                if dry_run
                else "No limit",
                "model_success_stats": model_success_stats,
            }

            report_path = generate_benchmark_report(
                benchmark_name=collection_name_prefix,
                sanity_check_failures=failed_sanity_checks_by_model,
                eval_failures=eval_failures_by_model,
                total_files_processed=len(files_to_process),
                total_runs_processed=len(agent_runs),
                processing_summary=processing_summary,
            )

            print(f"\n📄 PDF Report generated: {report_path}")

        except ImportError:
            print(f"\n⚠️ Could not generate PDF report - reportlab not available")
        except Exception as e:
            print(f"\n⚠️ Error generating PDF report: {e}")

    # Clean up temporary directory if it was created
    if download_if_missing and temp_directory:
        cleanup_temp_directory(temp_directory)

    return agent_runs, collection_name, report_path
