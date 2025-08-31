"""
Common utilities for integrating HAL benchmark data with Docent.

This module contains shared functions and classes for processing various benchmark
data formats (TauBench, CoreBench, etc.) and converting them to Docent format.
"""

import ast
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import Field

from docent.data_models import BaseAgentRunMetadata
from docent.data_models.chat import ChatMessage, ToolCall, parse_chat_message


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

    for log in task_logs_sorted[1:]:
        msgs_small = log["inputs"]["messages"]
        msgs_large = largest_log["inputs"]["messages"]

        if len(msgs_small) >= len(msgs_large):
            return False  # sanity violation

        for i, m in enumerate(msgs_small):
            if m != msgs_large[i]:
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
    return [
        entry
        for entry in data.get("raw_logging_results", [])
        if (
            entry["inputs"].get("model") == model_name
            and entry["inputs"].get("messages")
            and isinstance(entry["inputs"]["messages"][0].get("content"), list)
            and entry["inputs"]["messages"][0].get("role") == "system"
            and entry["inputs"]["messages"][0]["content"]
            and isinstance(entry["inputs"]["messages"][0]["content"][0], dict)
            and entry["inputs"]["messages"][0]["content"][0]
            .get("text", "")
            .startswith(system_prompt_prefix)
        )
    ]


def task_id_to_transcript(
    data: dict,
    model_name: str,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
    task_id_key: str = "weave_task_id",
) -> dict:
    """
    Build a dictionary mapping each task_id to its full transcript (messages)
    from the largest log, after sanity check.

    Args:
        data (dict): Raw logging data.
        model_name (str): Model name to filter logs on.
        system_prompt_prefix (str, optional): Prefix that system prompt must start with.
        task_id_key (str, optional): Key for task IDs in logs. Defaults to "weave_task_id".

    Returns:
        dict: {task_id: transcript_messages_list}
    """
    # Step 1: filter logs
    filtered_logs = filter_logs_by_model(model_name, data, system_prompt_prefix)

    # Step 2: extract task_ids
    task_ids = extract_task_ids(filtered_logs, task_id_key=task_id_key)

    transcripts = {}

    # Step 3: process each task_id
    for task_id in task_ids:
        task_logs = [
            entry for entry in filtered_logs if entry.get(task_id_key) == task_id
        ]

        # sanity check
        if not sanity_check(task_logs):
            raise ValueError(f"Sanity check failed for task_id={task_id}")

        # largest log is transcript
        task_logs_sorted = sorted(
            task_logs, key=lambda x: len(x["inputs"]["messages"]), reverse=True
        )
        transcripts[task_id] = task_logs_sorted[0]

    return transcripts


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

    benchmark_id: str = Field(description="The benchmark that the task belongs to")
    task_id: str = Field(
        description="The task within the benchmark that the agent is solving"
    )
    model: str = Field(description="The LLM used by the agent")
    additional_metadata: Dict[str, Any] = Field(
        description="Additional metadata about the task"
    )
    scoring_metadata: Dict[str, Any] | None = Field(
        description="Additional metadata about the scoring process"
    )


def analyze_benchmark_files(
    directory: str,
    file_pattern: str,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
) -> Dict[str, Dict[str, str]]:
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

    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping file paths to model name information:
        {
            'file_path': {
                'model_name_from_file_path': str,
                'model_name_from_json_content': str
            }
        }

    Raises:
        ValueError: If model names are inconsistent within a file or file pattern doesn't match
        OSError: If the directory doesn't exist or can't be accessed
        json.JSONDecodeError: If a file contains invalid JSON
    """
    if not os.path.exists(directory):
        raise OSError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise OSError(f"Path is not a directory: {directory}")

    results = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Skip if not a JSON file matching our pattern
        match = re.search(file_pattern, filename)
        if not match:
            continue

        # Extract model name from file path
        model_name_from_path = match.group(1) if match.group(1) else "default"

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
                if (
                    first_message.get("role") == "system"
                    and isinstance(first_message.get("content"), list)
                    and len(first_message["content"]) > 0
                    and isinstance(first_message["content"][0], dict)
                    and first_message["content"][0]
                    .get("text", "")
                    .startswith(system_prompt_prefix)
                ):
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

    return results


def process_benchmark_files(
    directory: str,
    file_pattern: str,
    conversion_function,
    collection_name_prefix: str,
    dry_run: bool = False,
    max_files: int = 5,
    max_runs_per_model: int = 5,
    system_prompt_prefix: str = "You are an expert assistant who can solve any task using code blobs",
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

    Returns:
        Tuple[List[AgentRun], str]: List of agent runs and collection name
    """
    from docent.data_models import AgentRun

    # Step 1: Analyze all files to get model mappings
    print(f"🔍 Analyzing {collection_name_prefix} files...")
    file_model_mappings = analyze_benchmark_files(
        directory, file_pattern, system_prompt_prefix
    )

    print(f"📁 Found {len(file_model_mappings)} {collection_name_prefix} files")

    agent_runs: List[AgentRun] = []

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
        try:
            task_logs_dict = task_id_to_transcript(
                data, model_name, system_prompt_prefix
            )
        except ValueError as e:
            print(f"   ❌ Error processing file: {e}")
            print(f"   ⏭️  Skipping this file and continuing...")
            continue

        # Step 4: iterate all task_ids for this model
        max_runs = max_runs_per_model if dry_run else len(task_logs_dict)
        processed_for_this_model = 0

        for task_id, log_entry in task_logs_dict.items():
            if processed_for_this_model >= max_runs:
                break

            eval_results_data = data.get("raw_eval_results", {}).get(task_id)

            if not eval_results_data:
                print(
                    f"   ⚠️ No raw_eval_results found for task_id={task_id}, skipping."
                )
                continue

            # Step 5: convert HAL benchmark -> Docent benchmark
            try:
                agent_run = conversion_function(
                    log_entry, model_name, eval_results_data
                )
                agent_runs.append(agent_run)
                processed_for_this_model += 1
            except Exception as e:
                print(f"   ❌ Error processing task_id={task_id}: {e}")
                continue

        print(f"   ✅ Processed {processed_for_this_model} agent runs from this file")

    print(f"\n🎯 Total processed agent runs: {len(agent_runs)}")

    collection_name = (
        f"{collection_name_prefix} Collection ({'Dry Run' if dry_run else 'Full Run'})"
    )

    return agent_runs, collection_name
