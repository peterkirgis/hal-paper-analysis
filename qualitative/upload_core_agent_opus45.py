"""
Upload CORE-Agent (smolagents) Opus 4.5 logs to Docent.

This script reads the local HAL results JSON file and uploads the agent runs
to a new Docent collection.

Usage:
    python upload_core_agent_opus45.py [--dry-run] [--collection-name "Custom Name"]
"""

import ast
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from requests.exceptions import ConnectionError, HTTPError, Timeout

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message, ToolCall

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Path to the JSON file
JSON_FILE_PATH = "/Users/peterkirgis/Documents/hal-paper-analysis/qualitative/corebench_hard_core_agent_opus_45_1764027531_UPLOAD.json"

# Default collection name
DEFAULT_COLLECTION_NAME = "CoreBench-CORE-Agent-Opus45"

# System prompt prefix used by CORE-Agent (smolagents)
SYSTEM_PROMPT_PREFIX = "You are an expert assistant who can solve any task using code blobs"


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load the JSON file."""
    print(f"📁 Loading file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"   Top-level keys: {list(data.keys())}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Message Normalization for Comparison (Deduplication)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_message_for_comparison(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a message for comparison purposes.
    Strips out fields that might vary between log entries but don't affect content.
    """
    normalized = {}

    # Include role
    if 'role' in message:
        normalized['role'] = message['role']

    # Normalize content
    content = message.get('content')
    if content is not None:
        if isinstance(content, list):
            # For list content, normalize each item
            normalized_content = []
            for item in content:
                if isinstance(item, dict):
                    normalized_item = {}
                    if 'type' in item:
                        normalized_item['type'] = item['type']
                    if 'text' in item:
                        normalized_item['text'] = item['text']
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)
            normalized['content'] = normalized_content
        else:
            normalized['content'] = content

    # Include tool_call_id only if present
    if 'tool_call_id' in message and message['tool_call_id'] is not None:
        normalized['tool_call_id'] = message['tool_call_id']

    return normalized


def sanity_check(task_logs: List[Dict[str, Any]]) -> bool:
    """
    Check that the largest log contains all smaller logs as ordered subsets.

    Args:
        task_logs: List of logs for a given task_id.

    Returns:
        True if the largest log captures all others, False otherwise.
    """
    if not task_logs:
        return False

    # Sort logs by number of messages (largest first)
    task_logs_sorted = sorted(
        task_logs,
        key=lambda x: len(x.get("inputs", {}).get("messages", [])),
        reverse=True
    )
    largest_log = task_logs_sorted[0]
    largest_messages = largest_log.get("inputs", {}).get("messages", [])
    largest_size = len(largest_messages)

    if largest_size == 0:
        return False

    # Group logs by size
    logs_by_size = defaultdict(list)
    for log in task_logs_sorted[1:]:
        size = len(log.get("inputs", {}).get("messages", []))
        logs_by_size[size].append(log)

    # Check each size group
    for size, logs_of_size in logs_by_size.items():
        if size == largest_size:
            continue
        if size > largest_size:
            return False

        # Check if at least one is a valid prefix
        has_valid_prefix = False
        for log in logs_of_size:
            msgs_small = log.get("inputs", {}).get("messages", [])
            is_valid_prefix = True

            for i, m in enumerate(msgs_small):
                if i >= largest_size:
                    is_valid_prefix = False
                    break
                normalized_small = normalize_message_for_comparison(m)
                normalized_large = normalize_message_for_comparison(largest_messages[i])
                if normalized_small != normalized_large:
                    is_valid_prefix = False
                    break

            if is_valid_prefix:
                has_valid_prefix = True
                break

        if not has_valid_prefix:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Log Filtering and Grouping
# ──────────────────────────────────────────────────────────────────────────────


def filter_and_group_logs(
    raw_logging_results: List[Dict[str, Any]],
    model_name: str,
    system_prompt_prefix: str = SYSTEM_PROMPT_PREFIX,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter logs by model and system prompt, then group by task_id.

    Args:
        raw_logging_results: List of raw log entries.
        model_name: Model name to filter on.
        system_prompt_prefix: Prefix that system prompt must start with.

    Returns:
        Dict mapping task_id to list of log entries.
    """
    logs_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in raw_logging_results:
        # Check model matches
        entry_model = entry.get("inputs", {}).get("model")
        if entry_model != model_name:
            continue

        # Check system prompt matches
        messages = entry.get("inputs", {}).get("messages", [])
        if not messages:
            continue

        first_message = messages[0]
        content = first_message.get("content")

        content_matches = False
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            content_text = content[0].get("text", "")
            content_matches = content_text.startswith(system_prompt_prefix)
        elif isinstance(content, str):
            content_matches = content.startswith(system_prompt_prefix)

        if not content_matches:
            continue

        # Get task_id
        task_id = entry.get("weave_task_id")
        if not task_id:
            task_id = entry.get("attributes", {}).get("weave_task_id")

        if task_id:
            logs_by_task[task_id].append(entry)

    return logs_by_task


def get_largest_transcript_per_task(
    logs_by_task: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    For each task, run sanity check and return the largest valid transcript.

    Args:
        logs_by_task: Dict mapping task_id to list of log entries.

    Returns:
        Dict mapping task_id to the largest log entry.
    """
    transcripts = {}
    failed_sanity_checks = 0

    for task_id, task_logs in logs_by_task.items():
        if not sanity_check(task_logs):
            print(f"   ⚠️ Sanity check failed for {task_id}, using largest anyway")
            failed_sanity_checks += 1

        # Get largest log
        task_logs_sorted = sorted(
            task_logs,
            key=lambda x: len(x.get("inputs", {}).get("messages", [])),
            reverse=True
        )
        transcripts[task_id] = task_logs_sorted[0]

    print(f"   📊 Sanity check failures: {failed_sanity_checks}/{len(logs_by_task)}")
    return transcripts


# ──────────────────────────────────────────────────────────────────────────────
# Tool Call Extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tool calls from assistant message content.

    The CORE-Agent format includes tool calls as:
    Calling tools: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    """
    if not isinstance(content, str):
        return None

    match = re.search(r"Calling tools:\s*(\[.*\])", content, re.DOTALL | re.MULTILINE)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Message Parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """
    Parse raw messages into Docent ChatMessage objects.

    Args:
        messages: List of raw message dicts.

    Returns:
        List of ChatMessage objects.
    """
    docent_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # Normalize content to string
        if isinstance(content, list):
            # Extract text from list format
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content_str = "\n".join(text_parts)
        elif isinstance(content, str):
            content_str = content
        else:
            content_str = ""

        message_data = {
            "role": role,
            "content": content_str,
        }

        # Extract tool calls from assistant messages
        if role == "assistant":
            raw_tool_calls = extract_tool_calls(content_str)
            if raw_tool_calls:
                parsed_tool_calls = []
                for tc in raw_tool_calls:
                    func_info = tc.get("function", {})
                    if isinstance(func_info, dict):
                        func_name = func_info.get("name", "unknown")
                        func_args = func_info.get("arguments", "")
                    else:
                        func_name = str(func_info)
                        func_args = ""

                    # Parse arguments if it's a string
                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            func_args = {"raw": func_args}

                    tool_call = ToolCall(
                        id=tc.get("id", f"call_{len(parsed_tool_calls)}"),
                        function=func_name,
                        arguments=func_args if isinstance(func_args, dict) else {"raw": str(func_args)},
                        type="function",
                        parse_error=None,
                    )
                    parsed_tool_calls.append(tool_call)

                message_data["tool_calls"] = parsed_tool_calls

        try:
            chat_message = parse_chat_message(message_data)
            docent_messages.append(chat_message)
        except Exception as e:
            print(f"      ⚠️ Failed to parse message: {e}")
            continue

    return docent_messages


# ──────────────────────────────────────────────────────────────────────────────
# AgentRun Creation
# ──────────────────────────────────────────────────────────────────────────────


def create_agent_run(
    task_id: str,
    model_name: str,
    run_id: str,
    docent_messages: List[Any],
    eval_data: Dict[str, Any],
    is_successful: bool,
    config_metadata: Dict[str, Any],
) -> AgentRun:
    """
    Create a Docent AgentRun from messages.
    """
    # Compute accuracy from eval data
    correct_written = eval_data.get("correct_written_answers", 0)
    correct_vision = eval_data.get("correct_vision_answers", 0)
    total_written = eval_data.get("total_written_questions", 0)
    total_vision = eval_data.get("total_vision_questions", 0)
    total_correct = correct_written + correct_vision
    total_questions = total_written + total_vision
    accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    # Determine model string (include reasoning effort if present)
    reasoning_effort = config_metadata.get("reasoning_effort")
    model_str = f"{model_name}_{reasoning_effort}" if reasoning_effort else model_name

    metadata = {
        "benchmark_id": "corebench_hard",
        "task_id": task_id,
        "model": model_str,
        "run_id": run_id,
        "agent_name": config_metadata.get("agent_name", "CORE-Agent"),
        "source": "core_agent_smolagents",
        "message_count": len(docent_messages),
        # Eval metadata
        "eval_is_successful": is_successful,
        "eval_accuracy": accuracy,
        "eval_correct_written": correct_written,
        "eval_correct_vision": correct_vision,
        "eval_total_written": total_written,
        "eval_total_vision": total_vision,
    }

    if eval_data.get("error"):
        metadata["eval_error"] = eval_data["error"]

    if reasoning_effort:
        metadata["reasoning_effort"] = reasoning_effort

    transcript = Transcript(messages=docent_messages, metadata=metadata)
    return AgentRun(transcripts={"default": transcript}, metadata=metadata)


# ──────────────────────────────────────────────────────────────────────────────
# Upload Logic
# ──────────────────────────────────────────────────────────────────────────────


def retry_with_backoff(func, max_retries=3, base_delay=2, *args, **kwargs):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout, HTTPError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"   ⚠️ Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"   Retrying in {delay}s...")
            time.sleep(delay)


def upload_agent_runs(
    client: Docent,
    collection_id: str,
    agent_runs: List[AgentRun],
) -> Dict[str, Any]:
    """Upload agent runs to Docent collection."""
    stats = {
        "total": len(agent_runs),
        "uploaded": 0,
        "failed": 0,
    }

    batch_size = 50
    for i in range(0, len(agent_runs), batch_size):
        batch = agent_runs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(agent_runs) + batch_size - 1) // batch_size

        print(f"   📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} runs)...")

        try:
            retry_with_backoff(client.add_agent_runs, 3, 2, collection_id, batch)
            stats["uploaded"] += len(batch)
        except Exception as e:
            print(f"   ❌ Failed to upload batch: {e}")
            stats["failed"] += len(batch)

        if i + batch_size < len(agent_runs):
            time.sleep(1)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(dry_run: bool = False, collection_name: Optional[str] = None):
    """
    Main function to process and upload CORE-Agent logs.

    Args:
        dry_run: If True, only process logs without uploading.
        collection_name: Custom collection name (optional).
    """
    load_dotenv()

    print("🚀 Starting CORE-Agent Opus 4.5 log upload to Docent")
    print(f"   Dry run: {dry_run}")
    print()

    # Load JSON file
    data = load_json_file(JSON_FILE_PATH)

    # Extract components
    config = data.get("config", {})
    results = data.get("results", {})
    raw_eval_results = data.get("raw_eval_results", {})
    raw_logging_results = data.get("raw_logging_results", [])

    # Get config metadata
    run_id = config.get("run_id", "unknown")
    model_name = config.get("agent_args", {}).get("model_name", "claude-opus-4-5-20251101")
    agent_name = config.get("agent_name", "CORE-Agent")

    print(f"📋 Config:")
    print(f"   Run ID: {run_id}")
    print(f"   Model: {model_name}")
    print(f"   Agent: {agent_name}")
    print(f"   Raw logging entries: {len(raw_logging_results)}")
    print()

    # Build success lookup
    successful_tasks = set(results.get("successful_tasks", []))
    failed_tasks = set(results.get("failed_tasks", []))
    print(f"📊 Results:")
    print(f"   Successful tasks: {len(successful_tasks)}")
    print(f"   Failed tasks: {len(failed_tasks)}")
    print()

    # Filter and group logs
    print("🔍 Filtering and grouping logs...")
    logs_by_task = filter_and_group_logs(raw_logging_results, model_name)
    print(f"   Found {len(logs_by_task)} unique tasks")

    # Get largest transcript per task
    print("📝 Extracting transcripts (with sanity check)...")
    transcripts = get_largest_transcript_per_task(logs_by_task)
    print(f"   Extracted {len(transcripts)} transcripts")
    print()

    # Create AgentRuns
    print("🔄 Creating AgentRuns...")
    agent_runs = []
    success_count = 0
    failure_count = 0

    for task_id, log_entry in transcripts.items():
        messages = log_entry.get("inputs", {}).get("messages", [])
        docent_messages = parse_messages(messages)

        if not docent_messages:
            print(f"   ⚠️ No valid messages for {task_id}, skipping")
            continue

        # Get eval data
        eval_data = raw_eval_results.get(task_id, {})
        is_successful = task_id in successful_tasks

        agent_run = create_agent_run(
            task_id=task_id,
            model_name=model_name,
            run_id=run_id,
            docent_messages=docent_messages,
            eval_data=eval_data,
            is_successful=is_successful,
            config_metadata={"agent_name": agent_name},
        )
        agent_runs.append(agent_run)

        if is_successful:
            success_count += 1
            status = "✅"
        else:
            failure_count += 1
            status = "❌"

        print(f"   {status} {task_id}: {len(docent_messages)} messages")

    print()
    print(f"📊 Processing Summary:")
    print(f"   Agent runs created: {len(agent_runs)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {failure_count}")
    print()

    if dry_run:
        print("🔍 Dry run complete - no uploads performed")
        return

    if not agent_runs:
        print("❌ No agent runs to upload")
        return

    # Create collection and upload
    final_collection_name = collection_name or DEFAULT_COLLECTION_NAME

    print(f"📤 Creating collection: {final_collection_name}")
    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = retry_with_backoff(
        client.create_collection,
        3, 2,
        name=final_collection_name,
        description=f"CORE-Agent (smolagents) runs on CoreBench Hard\nModel: {model_name}\nReasoning effort: high\nTotal runs: {len(agent_runs)}",
    )
    print(f"   Collection ID: {collection_id}")

    # Upload
    stats = upload_agent_runs(client, collection_id, agent_runs)

    print()
    print(f"✅ Upload Complete!")
    print(f"   Collection: {final_collection_name}")
    print(f"   Collection ID: {collection_id}")
    print(f"   Total: {stats['total']}")
    print(f"   Uploaded: {stats['uploaded']}")
    print(f"   Failed: {stats['failed']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload CORE-Agent Opus 4.5 logs to Docent collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process logs without uploading to Docent",
    )
    parser.add_argument(
        "--collection-name",
        help="Custom collection name",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, collection_name=args.collection_name)
