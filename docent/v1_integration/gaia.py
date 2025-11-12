r"""
GAIA Benchmark Integration with Docent - V1 (Zero Abstractions)

This script processes GAIA benchmark data and uploads it to Docent.

Supports both generalist and specialist agent patterns:
- Generalist: gaia_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$
- Specialist: gaia_hf_open_deep_research(.+?)_\d+_UPLOAD\.json$
"""

import argparse
import ast
import json
import logging
import os
import re
from datetime import datetime
from dateutil import parser as dateutil_parser
from typing import Any, Dict, List

from dotenv import load_dotenv
from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ContentReasoning,
    ContentText,
    SystemMessage,
    ToolCall,
    ToolCallContent,
    ToolMessage,
    UserMessage,
)
from pydantic import BaseModel, Field

from common_utils import (
    parse_message_dict_to_chat_message,
    deduplicate_log_entries,
    check_transcript_contains_largest_entry,
    load_and_organize_benchmark_file,
    extract_tool_calls,
    validate_agent_run,
    get_entry_with_most_messages,
)


# ============================================================================
# PATTERNS
# ============================================================================

GAIA_GENERALIST_PATTERN = r"gaia_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
GAIA_SPECIALIST_PATTERN = r"gaia_hf_open_deep_research(.+?)_\d+_UPLOAD\.json$"




# ============================================================================
# DATA MODELS
# ============================================================================


class GAIAMetadata(BaseModel):
    """Metadata for a GAIA task run."""

    benchmark_id: str = Field(..., description="Benchmark identifier")
    task_id: str = Field(..., description="Task identifier")
    model: str = Field(..., description="Model identifier")
    run_id: str = Field(..., description="Run identifier")
    model_name: str = Field(..., description="Model name")
    agent_name: str = Field(..., description="Agent name")
    reasoning_effort: str | None = Field(None, description="Reasoning effort level")
    budget: float | None = Field(None, description="Budget for the run")
    date: str = Field(..., description="Date of the run")
    benchmark_name: str = Field(..., description="Name of the benchmark")
    accuracy: float = Field(..., description="Accuracy score")
    agent_config: Dict[str, Any] | None = Field(None, description="Agent configuration")
    scores: Dict[str, Any] | None = Field(None, description="Scores from evaluation")
    additional_metadata: Dict[str, Any] | None = Field(
        None, description="Additional metadata"
    )
    scoring_metadata: Any = Field(None, description="Scoring metadata")
    question: str | None = Field(None, description="GAIA question/task")
    level: int | None = Field(None, description="GAIA difficulty level (1, 2, or 3)")


# ============================================================================
# CORE CONVERSION LOGIC
# ============================================================================



def reconstruct_conversation_from_log_entries_specialist(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[SystemMessage | UserMessage | AssistantMessage | ToolMessage]:
    """
    Reconstruct conversation for SPECIALIST agents (single message increments).

    For specialist agents:
    - First entry contains the full context (system + all initial messages)
    - Subsequent entries add one new user message + assistant response

    Strategy:
    - Use ALL messages from the first entry
    - For subsequent entries, add only the last message from inputs (new user message) + output

    Args:
        log_entries: List of log entry dictionaries sorted by created_timestamp
        verbose: Enable verbose logging

    Returns:
        List of ChatMessage objects representing the full conversation
    """
    if not log_entries:
        return []

    conversation = []

    # Process first entry: add ALL input messages
    first_entry = log_entries[0]
    first_entry_messages = first_entry.get("inputs", {}).get("messages", [])

    if verbose:
        print(
            f"      Entry 1 (first): {len(first_entry_messages)} messages in inputs - adding ALL"
        )

    for msg in first_entry_messages:
        conversation.append(parse_message_dict_to_chat_message(msg, verbose=verbose))

    # Add first entry's output
    output = first_entry.get("output")
    if output is not None:
        choices = output.get("choices", [])
        if choices and len(choices) > 0:
            assistant_message = choices[0].get("message", {})
            assistant_content = assistant_message.get("content", "")
            thinking_blocks = assistant_message.get("thinking_blocks", [])
            tool_calls = assistant_message.get("tool_calls", [])

            if verbose:
                content_preview = str(assistant_content)[:100]
                print(f"      ✅ Adding assistant output from entry 1")
                print(f"      Assistant content preview: {content_preview}")
                print(f"      Thinking blocks: {len(thinking_blocks)}")
                print(f"      Tool calls: {len(tool_calls) if tool_calls else 0}")

            output_msg_dict = {
                "role": "assistant",
                "content": assistant_content,
                "thinking_blocks": thinking_blocks,
                "tool_calls": tool_calls,
            }
            conversation.append(parse_message_dict_to_chat_message(output_msg_dict, verbose=verbose))

    # Process subsequent entries: add only the LAST message from inputs + output
    for idx, entry in enumerate(log_entries[1:], start=2):
        entry_input_messages = entry.get("inputs", {}).get("messages", [])

        if verbose:
            print(f"      Entry {idx}: {len(entry_input_messages)} messages in inputs")
            print(
                f"      Current conversation length before: {len(conversation)} messages"
            )

        # Add the last message from entry inputs (the NEW user message for this turn)
        if entry_input_messages:
            last_input_msg = entry_input_messages[-1]
            if verbose:
                print(f"      ✅ Adding last input message (new user message)")
                content_raw = last_input_msg.get("content", "")
                if isinstance(content_raw, list) and len(content_raw) > 0:
                    first_item = content_raw[0]
                    if isinstance(first_item, dict):
                        content_preview = str(first_item.get("text", ""))[:100]
                    else:
                        content_preview = str(first_item)[:100]
                else:
                    content_preview = str(content_raw)[:100]
                print(f"      Content preview: {content_preview}")
            conversation.append(parse_message_dict_to_chat_message(last_input_msg, verbose=verbose))

        # Get the output from this entry
        output = entry.get("output")

        if verbose:
            print(f"      Output type={type(output)}, has_output={output is not None}")

        # Skip entries without output
        if output is None:
            if verbose:
                print(f"      ⚠️ Skipping - no output")
            continue

        choices = output.get("choices", [])

        if choices and len(choices) > 0:
            assistant_message = choices[0].get("message", {})
            assistant_content = assistant_message.get("content", "")
            thinking_blocks = assistant_message.get("thinking_blocks", [])
            tool_calls = assistant_message.get("tool_calls", [])

            if verbose:
                content_preview = str(assistant_content)[:100]
                print(f"      ✅ Adding assistant output")
                print(f"      Assistant content preview: {content_preview}")
                print(f"      Thinking blocks: {len(thinking_blocks)}")
                print(f"      Tool calls: {len(tool_calls) if tool_calls else 0}")

            # Add the assistant's output message
            output_msg_dict = {
                "role": "assistant",
                "content": assistant_content,
                "thinking_blocks": thinking_blocks,
                "tool_calls": tool_calls,
            }
            conversation.append(parse_message_dict_to_chat_message(output_msg_dict, verbose=verbose))

    return conversation


def reconstruct_conversation_from_log_entries_generalist(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[SystemMessage | UserMessage | AssistantMessage | ToolMessage]:
    """
    Reconstruct conversation for GENERALIST agents (can have multi-message gaps).

    For generalist agents, there can be sudden jumps of 4+ messages at once (e.g., tool calls).
    We add all messages that are not already in the conversation, then add the output.

    Args:
        log_entries: List of log entry dictionaries sorted by created_timestamp
        verbose: Enable verbose logging

    Returns:
        List of ChatMessage objects representing the full conversation
    """
    if not log_entries:
        return []

    conversation = []

    # For each entry, add all new messages not in conversation, then add output
    for idx, entry in enumerate(log_entries, start=1):
        entry_input_messages = entry.get("inputs", {}).get("messages", [])

        if verbose:
            print(f"      Entry {idx}: {len(entry_input_messages)} messages in inputs")
            print(
                f"      Current conversation length before: {len(conversation)} messages"
            )

        # Calculate how many new messages to add
        num_existing = len(conversation)
        num_in_entry = len(entry_input_messages)
        num_new = num_in_entry - num_existing

        if verbose:
            print(f"      New messages to add: {num_new}")

        # Add all new messages that aren't in conversation yet
        if num_new > 0:
            new_messages = entry_input_messages[num_existing:]
            if verbose:
                print(f"      ✅ Adding {len(new_messages)} new input messages")
            for i, msg_dict in enumerate(new_messages):
                if verbose:
                    content_raw = msg_dict.get("content", "")
                    if isinstance(content_raw, list) and len(content_raw) > 0:
                        first_item = content_raw[0]
                        if isinstance(first_item, dict):
                            content_preview = str(first_item.get("text", ""))[:100]
                        else:
                            content_preview = str(first_item)[:100]
                    else:
                        content_preview = str(content_raw)[:100]
                    print(f"         Message {i + 1} preview: {content_preview}")
                conversation.append(parse_message_dict_to_chat_message(msg_dict, verbose=verbose))

        # Get the output from this entry
        output = entry.get("output")

        if verbose:
            print(f"      Output type={type(output)}, has_output={output is not None}")

        # Skip entries without output
        if output is None:
            if verbose:
                print(f"      ⚠️ Skipping - no output")
            continue

        choices = output.get("choices", [])

        if choices and len(choices) > 0:
            assistant_message = choices[0].get("message", {})
            assistant_content = assistant_message.get("content", "")
            thinking_blocks = assistant_message.get("thinking_blocks", [])
            tool_calls = assistant_message.get("tool_calls", [])

            if verbose:
                content_preview = str(assistant_content)[:100]
                print(f"      ✅ Adding assistant output")
                print(f"      Assistant content preview: {content_preview}")
                print(f"      Thinking blocks: {len(thinking_blocks)}")
                print(f"      Tool calls: {len(tool_calls) if tool_calls else 0}")

            # Add the assistant's output message
            output_msg_dict = {
                "role": "assistant",
                "content": assistant_content,
                "thinking_blocks": thinking_blocks,
                "tool_calls": tool_calls,
            }
            conversation.append(parse_message_dict_to_chat_message(output_msg_dict, verbose=verbose))

    return conversation


def hal_gaia_to_docent_gaia(
    log_entries: List[Dict[str, Any]],
    model_name: str,
    eval_results_data: Dict[str, Any],
    config_data: Dict[str, Any],
    is_generalist: bool = False,
    verbose: bool = False,
) -> AgentRun:
    """
    Convert HAL GAIA log entries for a task into a Docent AgentRun.

    Args:
        log_entries: List of log entry dictionaries for the same task_id (sorted by timestamp)
        model_name: The model name to assert against the log entries
        eval_results_data: Evaluation results containing task results and metadata
        config_data: Configuration data for the run
        is_generalist: Whether this is a generalist agent (affects conversation reconstruction)
        verbose: Enable verbose logging

    Returns:
        AgentRun object containing parsed transcript messages, metadata, and evaluation results
    """
    assert len(log_entries) > 0
    first_entry = log_entries[0]
    entry_model_full = first_entry["inputs"]["model"]
    
    # Split by '/' and take the last part for comparison
    entry_model = entry_model_full.split('/')[-1] if '/' in entry_model_full else entry_model_full
    
    if entry_model != model_name:
        if verbose:
            print(
                f"   ⚠️  Model mismatch: expected '{model_name}', got '{entry_model}' (from '{entry_model_full}')"
            )
        assert entry_model == model_name, f"Model mismatch: expected '{model_name}', got '{entry_model}' (from '{entry_model_full}')"
    task_id = first_entry["weave_task_id"]

    # Pick only the entry with the most messages (handles agent restarts)
    if verbose:
        print(f"   📏 Selecting entry with most messages for task {task_id}")
    log_entries = get_entry_with_most_messages(log_entries, verbose=verbose)
    
    if verbose:
        print(f"   📊 After selection: {len(log_entries)} entry(ies)")

    # Reconstruct the full conversation from all log entries
    if verbose:
        agent_type = "generalist" if is_generalist else "specialist"
        print(
            f"   🔄 Reconstructing conversation for task {task_id} ({agent_type}) from {len(log_entries)} log entries"
        )

    if is_generalist:
        messages = reconstruct_conversation_from_log_entries_generalist(
            log_entries, verbose=verbose
        )
    else:
        messages = reconstruct_conversation_from_log_entries_specialist(
            log_entries, verbose=verbose
        )

    # Extract metadata from config
    run_id = config_data.get("run_id", "unknown")
    agent_name = config_data.get("agent_name", "unknown")
    date_str = config_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    benchmark_name = config_data.get("benchmark_name", "gaia")

    # Extract agent args
    agent_args = config_data.get("agent_args", {})
    reasoning_effort = agent_args.get("reasoning_effort")
    budget = agent_args.get("budget")

    # Determine task success from successful_tasks and failed_tasks lists
    results = eval_results_data
    successful_tasks = results.get("successful_tasks", [])
    failed_tasks = results.get("failed_tasks", [])

    task_success = 1 if task_id in successful_tasks else 0
    accuracy = float(task_success)

    # Extract question and level from the first user message if available
    question = None
    level = None

    # Try to get question from first user message in log entries
    if log_entries and len(log_entries) > 0:
        first_entry_messages = log_entries[0].get("inputs", {}).get("messages", [])
        for msg in first_entry_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from list format
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    question = " ".join(text_parts)[:500]  # Limit to 500 chars
                else:
                    question = str(content)[:500]  # Limit to 500 chars
                break

    metadata = GAIAMetadata(
        benchmark_id="gaia",
        task_id=task_id,
        model=model_name,
        run_id=run_id,
        model_name=model_name,
        agent_name=agent_name,
        reasoning_effort=reasoning_effort,
        budget=budget,
        date=date_str,
        benchmark_name=benchmark_name,
        accuracy=accuracy,
        agent_config=config_data,
        scores={
            "task_success": task_success,
        },
        additional_metadata={
            "task_success": task_success,
        },
        scoring_metadata=None,
        question=question,
        level=level,
    )

    # Convert metadata to dict
    metadata_dict = metadata.model_dump()

    transcript = Transcript(
        messages=messages,
        metadata=metadata_dict,
    )

    agent_run = AgentRun(
        transcripts=[transcript],
        metadata=metadata_dict,
    )

    return agent_run


# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_gaia_file(
    file_path: str,
    max_tasks: int | None = None,
    target_first_message_prefix: str = "",
    is_generalist: bool = False,
    verbose: bool = False,
) -> List[AgentRun]:
    """
    Process a single GAIA JSON file and extract agent runs.

    Args:
        file_path: Path to the JSON file
        max_tasks: Maximum number of tasks to process (for dry runs)
        target_first_message_prefix: The expected first message prefix for filtering (optional)
        is_generalist: Whether this is a generalist agent (affects conversation reconstruction)
        verbose: Enable verbose logging

    Returns:
        List of AgentRun objects
    """
    result = load_and_organize_benchmark_file(
        file_path=file_path,
        target_first_message_prefix=target_first_message_prefix,
        is_generalist=is_generalist,
        verbose=verbose,
        timestamp_based_resolving=False,
    )

    if result is None:
        return []

    # Extract results from the common loader
    file_name = result["file_name"]
    config_data = result["config_data"]
    eval_results_data_wrapper = result["eval_results_data"]
    deduped_task_logs = result["deduped_task_logs"]
    model_name = result["model_name"]

    # Extract eval results from wrapper
    raw_eval_results = eval_results_data_wrapper.get("raw_eval_results", {})
    results_data = eval_results_data_wrapper.get("results", {})

    # Process tasks
    agent_runs = []
    transcript_check_results = []
    processed = 0

    for task_id, log_entries in deduped_task_logs.items():
        if max_tasks and processed >= max_tasks:
            break

        print(f"\n   {'-' * 70}")
        print(f"   🔧 Processing task_id: {task_id}")

        # Print log entries info
        message_counts = [
            len(entry.get("inputs", {}).get("messages", [])) for entry in log_entries
        ]
        print(f"   📊 Log entries: {len(log_entries)}")
        print(f"   📊 Message counts per entry: {message_counts}")

        # Process without try-catch - let errors stop the pipeline
        if verbose:
            print(f"   🔧 Task has {len(log_entries)} log entries")
        agent_run = hal_gaia_to_docent_gaia(
            log_entries,
            model_name,
            results_data,
            config_data,
            is_generalist=is_generalist,
            verbose=verbose,
        )
        agent_runs.append(agent_run)
        
        # Check if largest entry is subset of transcript
        transcript_messages = agent_run.transcripts[0].messages
        check_result = check_transcript_contains_largest_entry(
            log_entries,
            transcript_messages,
            task_id=task_id,
            file_name=file_name,
            is_generalist=is_generalist,
            verbose=verbose,
        )
        transcript_check_results.append(check_result)
        
        processed += 1
        print(f"   {'-' * 70}")

    # Print transcript check summary
    if transcript_check_results:
        tasks_passed = sum(1 for r in transcript_check_results if r["passed"])
        tasks_failed = len(transcript_check_results) - tasks_passed
        total_tasks = len(transcript_check_results)
        pass_percentage = (tasks_passed / total_tasks) * 100 if total_tasks > 0 else 0
        
        print(f"\n   📊 Transcript Check Results:")
        print(f"      ✅ Tasks passed (largest entry ⊆ transcript): {tasks_passed}/{total_tasks} ({pass_percentage:.1f}%)")
        print(f"      ❌ Tasks failed: {tasks_failed}/{total_tasks} ({100-pass_percentage:.1f}%)")

    print(f"   ✅ Successfully processed {len(agent_runs)} agent runs")
    return agent_runs


def process_all_gaia_files(
    directory: str,
    pattern: str,
    log_dir: str,
    max_files: int | None = None,
    max_tasks_per_file: int | None = None,
    target_first_message_prefix: str = "",
    is_generalist: bool = False,
    verbose: bool = False,
) -> List[AgentRun]:
    """
    Process all GAIA JSON files in a directory matching the pattern.

    Args:
        directory: Directory containing the JSON files
        pattern: Regex pattern to match files (generalist or specialist)
        log_dir: Directory to write individual log files
        max_files: Maximum number of files to process
        max_tasks_per_file: Maximum number of tasks to process per file
        target_first_message_prefix: The expected first message prefix for filtering (optional)

    Returns:
        List of all AgentRun objects
    """
    import sys
    from contextlib import redirect_stdout
    
    all_agent_runs = []

    # Find all JSON files in the directory matching the pattern
    json_files = []
    pattern_regex = re.compile(pattern)

    for f in os.listdir(directory):
        if f.endswith("_UPLOAD.json") and pattern_regex.search(f):
            json_files.append(f)

    json_files.sort()

    print(f"\n🔍 Found {len(json_files)} JSON files matching pattern in {directory}")
    print(f"   Pattern: {pattern}")
    print(f"   Log directory: {log_dir}")

    if max_files:
        json_files = json_files[:max_files]
        print(f"   Limiting to first {max_files} files")

    for json_file in json_files:
        log_filename = json_file.replace("_UPLOAD.json", ".log")
        log_path = os.path.join(log_dir, log_filename)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\n📄 Processing file: {json_file} -> {log_path}")
        
        with open(log_path, 'w') as log_file:
            with redirect_stdout(log_file):
                print(f"{'=' * 80}")
                print(f"Processing file: {json_file}")
                print(f"{'=' * 80}")
                
                file_path = os.path.join(directory, json_file)
                agent_runs = process_gaia_file(
                    file_path,
                    max_tasks=max_tasks_per_file,
                    target_first_message_prefix=target_first_message_prefix,
                    is_generalist=is_generalist,
                    verbose=verbose,
                )
                all_agent_runs.extend(agent_runs)
                
                print(f"\n✅ Processed {len(agent_runs)} agent runs from this file")

    print(f"\n✅ Total agent runs collected: {len(all_agent_runs)}")
    return all_agent_runs


# ============================================================================
# MAIN
# ============================================================================


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload GAIA agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 3 tasks from a single model (default: process all)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["generalist", "specialist"],
        default="generalist",
        help="Type of agent data to process (generalist or specialist)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Name of the Docent collection (auto-generated if not provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write log files (one per input file)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level
    directory = os.path.join(project_root, "hal_traces", "gaia_data")
    
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(script_dir, "logs", "gaia", args.agent_type)

    if args.agent_type == "generalist":
        pattern = GAIA_GENERALIST_PATTERN
        collection_prefix = "GAIA-Generalist"
        target_first_message_prefix = (
            "You are an expert assistant who can solve any task using code blobs."
        )
        is_generalist = True
    else:
        pattern = GAIA_SPECIALIST_PATTERN
        collection_prefix = "GAIA-Specialist"
        target_first_message_prefix = (
            "You are an expert assistant who can solve any task using tool calls."
        )
        is_generalist = False

    # Generate collection name
    if args.collection_name:
        collection_name = args.collection_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"{collection_prefix}-V1-{timestamp}"

    # Dry run processes just 3 tasks from one model file
    max_files = 1 if args.dry_run else None
    max_tasks_per_file = 3 if args.dry_run else None

    print(f"\n{'=' * 60}")
    print(f"GAIA Benchmark Integration - V1")
    print(f"{'=' * 60}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Pattern: {pattern}")
    print(f"Directory: {directory}")
    if args.dry_run:
        print(
            f"Dry Run Mode: Processing {max_tasks_per_file} tasks from {max_files} model file"
        )
    print(f"{'=' * 60}\n")

    # Process files
    agent_runs = process_all_gaia_files(
        directory=directory,
        pattern=pattern,
        log_dir=log_dir,
        max_files=max_files,
        max_tasks_per_file=max_tasks_per_file,
        target_first_message_prefix=target_first_message_prefix,
        is_generalist=is_generalist,
        verbose=args.verbose,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        return

    print(f"\n📤 Uploading {len(agent_runs)} agent runs to Docent...")

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"GAIA {args.agent_type} agent runs - {len(agent_runs)} runs processed",
    )

    chunk_size = 100
    total_runs = len(agent_runs)

    for i in range(0, total_runs, chunk_size):
        chunk = agent_runs[i : i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        print(f"📤 Uploading chunk {chunk_num}/{total_chunks} ({len(chunk)} runs)...")
        client.add_agent_runs(collection_id, chunk)

    print(f"\n{'=' * 60}")
    print(f"✅ Upload Complete!")
    print(f"{'=' * 60}")
    print(f"Total Runs: {len(agent_runs)}")
    print(f"Collection ID: {collection_id}")
    print(f"Collection Name: {collection_name}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
