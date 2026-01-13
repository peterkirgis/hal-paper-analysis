"""
Upload Claude Code logs from CoreBench runs to Docent.

This script extracts Claude Code conversation logs from local folders and uploads
them to an existing Docent collection.
"""

import glob
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
COLLECTION_ID = "ac047e81-c9a2-44a1-911d-cc0054aae70c"

# Model folders to process
MODEL_FOLDERS = [
    "corebench_hard_claude_code_claudeopus41_1764560915",
    "corebench_hard_claude_code_claudeopus45_1764537671",
    "corebench_hard_claude_code_claudesonnet420250514_1764558813",
]

# Target capsules to extract
TARGET_CAPSULES = [
    "capsule-1394704",
    "capsule-2345790",
    "capsule-3262218",
    "capsule-3418007",
    "capsule-3449234",
    "capsule-3639589",
    "capsule-4252248",
    "capsule-5136217",
    "capsule-7716865",
    "capsule-8234136",
    "capsule-8807709",
    "capsule-9054015",
]

# Model name mapping from folder names
MODEL_NAME_MAP = {
    "claudeopus41": "claude-opus-4-20250514",
    "claudeopus45": "claude-opus-4-5-20251101",
    "claudesonnet420250514": "claude-sonnet-4-20250514",
}


# ──────────────────────────────────────────────────────────────────────────────
# Claude Code Log Parser
# ──────────────────────────────────────────────────────────────────────────────


def extract_model_name_from_folder(folder_name: str) -> str:
    """Extract model name from folder name."""
    # Pattern: corebench_hard_claude_code_{model}_{timestamp}
    parts = folder_name.split("_")
    # Find the model part after "code"
    for i, part in enumerate(parts):
        if part == "code" and i + 1 < len(parts):
            model_key = parts[i + 1]
            return MODEL_NAME_MAP.get(model_key, model_key)
    return "unknown"


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
) -> AgentRun:
    """
    Create a Docent AgentRun from messages.

    Args:
        model_name: The model name (e.g., "claude-opus-4-5-20251101")
        capsule_id: The capsule identifier (e.g., "capsule-1394704")
        folder_name: The source folder name
        docent_messages: List of Docent ChatMessage objects

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


def main(dry_run: bool = False):
    """
    Main function to process and upload Claude Code logs.

    Args:
        dry_run: If True, only process logs without uploading
    """
    load_dotenv()

    print("🚀 Starting Claude Code log upload to Docent")
    print(f"   Collection ID: {COLLECTION_ID}")
    print(f"   Dry run: {dry_run}")
    print()

    agent_runs = []
    processed_count = 0
    skipped_count = 0

    # Process each model folder
    for folder_name in MODEL_FOLDERS:
        folder_path = os.path.join(DOWNLOADS_DIR, folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        model_from_folder = extract_model_name_from_folder(folder_name)
        print(f"📁 Processing folder: {folder_name}")
        print(f"   Model: {model_from_folder}")

        # Process each target capsule
        for capsule_id in TARGET_CAPSULES:
            capsule_path = os.path.join(folder_path, capsule_id)
            log_path = os.path.join(capsule_path, "claude_code.log.0")

            if not os.path.exists(log_path):
                print(f"   ⚠️ Log not found: {capsule_id}")
                skipped_count += 1
                continue

            # Parse the log
            model_name, messages = parse_claude_code_log(log_path)

            if not messages:
                print(f"   ⚠️ No messages found in {capsule_id}")
                skipped_count += 1
                continue

            # Use model from log if available, otherwise from folder
            final_model = model_name or model_from_folder

            # Convert to Docent format
            docent_messages = convert_to_docent_messages(messages)

            if not docent_messages:
                print(f"   ⚠️ No valid messages after conversion for {capsule_id}")
                skipped_count += 1
                continue

            # Create AgentRun
            agent_run = create_agent_run(
                model_name=final_model,
                capsule_id=capsule_id,
                folder_name=folder_name,
                docent_messages=docent_messages,
            )
            agent_runs.append(agent_run)
            processed_count += 1

            print(f"   ✅ {capsule_id}: {len(docent_messages)} messages (model: {final_model})")

        print()

    print(f"📊 Processing Summary:")
    print(f"   Total processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
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
        description="Upload Claude Code logs to Docent collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process logs without uploading to Docent",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run)
