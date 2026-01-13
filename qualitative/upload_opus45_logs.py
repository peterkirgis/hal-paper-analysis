"""
Upload Claude Code logs from CoreBench Opus 4.5 runs to Docent.

This script extracts ALL Claude Code conversation logs from the Opus 4.5 folder
and uploads them to a Docent collection, including task success/failure metadata
from the corebench_results_updated.csv file.
"""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from requests.exceptions import ConnectionError, HTTPError, Timeout

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = "/Users/peterkirgis/Downloads"
COLLECTION_ID = "e994a85e-b637-4dcc-9061-946641df5fed"

# Opus 4.5 folder
OPUS45_FOLDER = "corebench_hard_claude_code_claudeopus45_1764537671"
MODEL_NAME = "claude-opus-4-5-20251101"

# Results CSV path
RESULTS_CSV = "/Users/peterkirgis/Documents/hal-paper-analysis/scratch/corebench_results_updated.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Load Results CSV
# ──────────────────────────────────────────────────────────────────────────────


def load_results_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the corebench results CSV and return a dict mapping capsule_id to results.

    Args:
        csv_path: Path to the corebench_results_updated.csv file

    Returns:
        Dict mapping capsule ID (e.g., "capsule-1394704") to result data
    """
    results = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                capsule_id = row.get("capsule", "")
                if capsule_id:
                    # Parse the opus4.5_success column
                    opus45_success = row.get("opus4.5_success", "").upper() == "TRUE"

                    results[capsule_id] = {
                        "opus45_success": opus45_success,
                        "total_success": int(row.get("total_success", 0)),
                        "agreement_summary": row.get("agreement_summary", ""),
                        "disagreement": row.get("disagreement", "").upper() == "TRUE",
                        "all_agree_success": row.get("all_agree_success", "").upper() == "TRUE",
                        "all_agree_failure": row.get("all_agree_failure", "").upper() == "TRUE",
                    }
    except Exception as e:
        print(f"❌ Error loading results CSV: {e}")
        return {}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Input JSON Loader
# ──────────────────────────────────────────────────────────────────────────────


def load_input_json(capsule_path: str, capsule_id: str) -> Optional[str]:
    """
    Load the input.json file from a capsule folder and extract the prompt.

    Args:
        capsule_path: Path to the capsule folder
        capsule_id: The capsule identifier (e.g., "capsule-1394704")

    Returns:
        The prompt string from input.json, or None if not found
    """
    input_path = os.path.join(capsule_path, "input.json")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The structure is {capsule_id: {prompt: "...", files: {...}, gpu: bool}}
        capsule_data = data.get(capsule_id, {})
        prompt = capsule_data.get("prompt", "")

        if prompt:
            return prompt

        # Fallback: try to get from first key if capsule_id doesn't match
        if data:
            first_key = list(data.keys())[0]
            return data[first_key].get("prompt", "")

    except Exception as e:
        print(f"   ⚠️ Could not load input.json: {e}")

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Claude Code Log Parser
# ──────────────────────────────────────────────────────────────────────────────


def parse_claude_code_log(log_path: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Parse a Claude Code JSONL log file.

    Args:
        log_path: Path to the claude_code.log.0 file

    Returns:
        Tuple of (model_name, messages) where messages is a list of normalized
        message dicts with role, content, and optionally tool_calls
    """
    model_name = None
    messages = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")

                # Extract model name from system init
                if entry_type == "system" and entry.get("subtype") == "init":
                    model_name = entry.get("model")
                    continue

                # Process assistant messages
                if entry_type == "assistant":
                    msg = entry.get("message", {})
                    role = msg.get("role", "assistant")
                    content = msg.get("content", [])

                    # Extract text content and tool calls
                    text_parts = []
                    tool_calls = []

                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif item.get("type") == "tool_use":
                                    tool_calls.append({
                                        "id": item.get("id", ""),
                                        "type": "function",
                                        "function": item.get("name", ""),
                                        "arguments": item.get("input", {}),
                                    })
                    elif isinstance(content, str):
                        text_parts.append(content)

                    normalized = {
                        "role": role,
                        "content": "\n".join(text_parts),
                    }
                    if tool_calls:
                        normalized["tool_calls"] = tool_calls

                    messages.append(normalized)

                # Process user messages (tool results)
                elif entry_type == "user":
                    msg = entry.get("message", {})
                    role = msg.get("role", "user")
                    content = msg.get("content", [])

                    # Extract tool results or text content
                    text_parts = []

                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "tool_result":
                                    # Include tool result as content
                                    result_content = item.get("content", "")
                                    tool_use_id = item.get("tool_use_id", "")
                                    text_parts.append(f"[Tool Result for {tool_use_id}]\n{result_content}")
                                elif item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                    elif isinstance(content, str):
                        text_parts.append(content)

                    normalized = {
                        "role": role,
                        "content": "\n".join(text_parts),
                    }
                    messages.append(normalized)

    except Exception as e:
        print(f"   ❌ Error parsing log file {log_path}: {e}")
        return None, []

    return model_name, messages


# ──────────────────────────────────────────────────────────────────────────────
# Docent Conversion
# ──────────────────────────────────────────────────────────────────────────────


def convert_to_docent_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert normalized messages to Docent ChatMessage objects.

    Args:
        messages: List of normalized message dicts

    Returns:
        List of ChatMessage objects
    """
    docent_messages = []

    for msg in messages:
        try:
            # Normalize tool_calls format for Docent
            normalized = msg.copy()
            if "tool_calls" in normalized and normalized["tool_calls"]:
                fixed_tool_calls = []
                for tc in normalized["tool_calls"]:
                    fixed_tc = {
                        "id": tc.get("id", f"tool_{len(fixed_tool_calls)}"),
                        "type": "function",
                        "function": tc.get("function", "unknown"),
                        "arguments": tc.get("arguments", {}),
                    }
                    # Ensure arguments is a dict
                    if isinstance(fixed_tc["arguments"], str):
                        try:
                            fixed_tc["arguments"] = json.loads(fixed_tc["arguments"])
                        except json.JSONDecodeError:
                            fixed_tc["arguments"] = {"raw_value": fixed_tc["arguments"]}
                    fixed_tool_calls.append(fixed_tc)
                normalized["tool_calls"] = fixed_tool_calls

            chat_msg = parse_chat_message(normalized)
            docent_messages.append(chat_msg)
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to convert message: {e}")
            continue

    return docent_messages


# ──────────────────────────────────────────────────────────────────────────────
# AgentRun Creation
# ──────────────────────────────────────────────────────────────────────────────


def create_agent_run(
    model_name: str,
    capsule_id: str,
    folder_name: str,
    docent_messages: List[Any],
    result_data: Dict[str, Any],
) -> AgentRun:
    """
    Create a Docent AgentRun from messages with success metadata.

    Args:
        model_name: The model name (e.g., "claude-opus-4-5-20251101")
        capsule_id: The capsule identifier (e.g., "capsule-1394704")
        folder_name: The source folder name
        docent_messages: List of Docent ChatMessage objects
        result_data: Dict containing success/failure metadata from CSV

    Returns:
        AgentRun object ready for upload
    """
    metadata = {
        "benchmark_id": "corebench_hard",
        "task_id": capsule_id,
        "model": model_name,
        "run_id": folder_name,
        "capsule_id": capsule_id,
        "source": "claude_code",
        "message_count": len(docent_messages),
        # Success metadata from CSV
        "task_success": result_data.get("opus45_success", False),
        "total_success_count": result_data.get("total_success", 0),
        "agreement_summary": result_data.get("agreement_summary", ""),
        "has_disagreement": result_data.get("disagreement", False),
        "all_models_succeeded": result_data.get("all_agree_success", False),
        "all_models_failed": result_data.get("all_agree_failure", False),
    }

    transcript = Transcript(messages=docent_messages, metadata=metadata)
    transcripts = {"default": transcript}

    return AgentRun(transcripts=transcripts, metadata=metadata)


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


def upload_agent_runs(client: Docent, collection_id: str, agent_runs: List[AgentRun]) -> Dict[str, Any]:
    """
    Upload agent runs to Docent collection.

    Args:
        client: Docent client
        collection_id: Target collection ID
        agent_runs: List of AgentRun objects to upload

    Returns:
        Upload statistics
    """
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

        # Small delay between batches
        if i + batch_size < len(agent_runs):
            time.sleep(1)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(dry_run: bool = False, capsules: Optional[List[str]] = None):
    """
    Main function to process and upload Opus 4.5 Claude Code logs.

    Args:
        dry_run: If True, only process logs without uploading
        capsules: Optional list of specific capsule IDs to upload (e.g., ["capsule-1234567"])
                  If None, all capsules are processed
    """
    load_dotenv()

    print("🚀 Starting Opus 4.5 Claude Code log upload to Docent")
    print(f"   Collection ID: {COLLECTION_ID}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Dry run: {dry_run}")
    if capsules:
        print(f"   Filtering to {len(capsules)} specific capsules")
    print()

    # Load results CSV
    print("📊 Loading results CSV...")
    results_data = load_results_csv(RESULTS_CSV)
    print(f"   Loaded {len(results_data)} capsule results")
    print()

    folder_path = os.path.join(DOWNLOADS_DIR, OPUS45_FOLDER)

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    # Find capsule folders (filtered if specific capsules requested)
    if capsules:
        # Use only the specified capsules
        capsule_folders = [c for c in capsules if os.path.exists(os.path.join(folder_path, c))]
        missing = set(capsules) - set(capsule_folders)
        if missing:
            print(f"⚠️ Warning: {len(missing)} requested capsules not found: {missing}")
    else:
        # Find all capsule folders
        capsule_folders = []
        for item in os.listdir(folder_path):
            if item.startswith("capsule-"):
                capsule_folders.append(item)

    capsule_folders.sort()
    print(f"📁 Processing {len(capsule_folders)} capsule folders in {OPUS45_FOLDER}")
    print()

    agent_runs = []
    processed_count = 0
    skipped_count = 0
    success_count = 0
    failure_count = 0

    # Process each capsule
    for capsule_id in capsule_folders:
        capsule_path = os.path.join(folder_path, capsule_id)
        log_path = os.path.join(capsule_path, "claude_code.log.0")

        if not os.path.exists(log_path):
            print(f"   ⚠️ Log not found: {capsule_id}")
            skipped_count += 1
            continue

        # Load the input.json to get the task prompt
        task_prompt = load_input_json(capsule_path, capsule_id)

        # Parse the log
        model_name, messages = parse_claude_code_log(log_path)

        if not messages:
            print(f"   ⚠️ No messages found in {capsule_id}")
            skipped_count += 1
            continue

        # Prepend the task prompt as a system message if available
        if task_prompt:
            system_message = {
                "role": "system",
                "content": task_prompt,
            }
            messages.insert(0, system_message)

        # Use model from log if available, otherwise use default
        final_model = model_name or MODEL_NAME

        # Convert to Docent format
        docent_messages = convert_to_docent_messages(messages)

        if not docent_messages:
            print(f"   ⚠️ No valid messages after conversion for {capsule_id}")
            skipped_count += 1
            continue

        # Get result data for this capsule
        result_data = results_data.get(capsule_id, {
            "opus45_success": False,
            "total_success": 0,
            "agreement_summary": "Unknown",
            "disagreement": False,
            "all_agree_success": False,
            "all_agree_failure": False,
        })

        # Create AgentRun with success metadata
        agent_run = create_agent_run(
            model_name=final_model,
            capsule_id=capsule_id,
            folder_name=OPUS45_FOLDER,
            docent_messages=docent_messages,
            result_data=result_data,
        )
        agent_runs.append(agent_run)
        processed_count += 1

        # Track success/failure counts
        if result_data.get("opus45_success", False):
            success_count += 1
            status = "✅"
        else:
            failure_count += 1
            status = "❌"

        print(f"   {status} {capsule_id}: {len(docent_messages)} messages (success={result_data.get('opus45_success', False)})")

    print()
    print(f"📊 Processing Summary:")
    print(f"   Total processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Successful tasks: {success_count}")
    print(f"   Failed tasks: {failure_count}")
    print(f"   Agent runs created: {len(agent_runs)}")
    print()

    if dry_run:
        print("🔍 Dry run complete - no uploads performed")
        return

    if not agent_runs:
        print("❌ No agent runs to upload")
        return

    # Upload to Docent
    print("📤 Uploading to Docent...")
    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    stats = upload_agent_runs(client, COLLECTION_ID, agent_runs)

    print()
    print(f"✅ Upload Complete!")
    print(f"   Total: {stats['total']}")
    print(f"   Uploaded: {stats['uploaded']}")
    print(f"   Failed: {stats['failed']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload Opus 4.5 Claude Code logs to Docent collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process logs without uploading to Docent",
    )
    parser.add_argument(
        "--capsules",
        nargs="+",
        help="Specific capsule IDs to upload (e.g., --capsules capsule-1234567 capsule-2345678)",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, capsules=args.capsules)
