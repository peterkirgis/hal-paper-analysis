r"""
TAU-bench Benchmark Integration with Docent - V1 (Zero Abstractions)

This script processes TAU-bench benchmark data and uploads it to Docent.

Supports both generalist and specialist agent patterns for airline domain:
- Generalist: taubench_airline_hal_generalist(.+?)_\d+_UPLOAD\.json$
- Specialist: taubench_airline_taubench_few_?shot(.+?)_\d+_UPLOAD\.json$
"""

import argparse
import ast
import json
import os
import re
from datetime import datetime
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


# ============================================================================
# PATTERNS
# ============================================================================

TAUBENCH_GENERALIST_PATTERN = (
    r"taubench_airline_hal_generalist(.+?)_\d+_UPLOAD\.json$"
)
TAUBENCH_SPECIALIST_PATTERN = (
    r"taubench_airline_taubench_few_?shot(.+?)_\d+_UPLOAD\.json$"
)


# ============================================================================
# DATA MODELS
# ============================================================================


class TAUBenchMetadata(BaseModel):
    """Metadata for a TAU-bench task run."""

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
    domain: str = Field(..., description="TAU-bench domain (e.g., airline)")
    task_number: str | None = Field(None, description="TAU-bench task number")


# ============================================================================
# CORE CONVERSION LOGIC
# ============================================================================


def extract_tool_calls(input_str: str) -> List[Dict[str, Any]] | None:
    """
    Extract tool calls from assistant message content.

    Args:
        input_str: The message content to search for tool calls.

    Returns:
        List of parsed tool calls, or None if none found.
    """
    match = re.search(r"Calling tools:\s*(\[.*\])", input_str, re.DOTALL | re.MULTILINE)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None
    return None


def parse_message_dict_to_chat_message(
    msg: Dict[str, Any],
) -> SystemMessage | UserMessage | AssistantMessage | ToolMessage:
    """
    Convert a single message dictionary to a Docent ChatMessage object.

    Args:
        msg: Message dictionary with 'role' and 'content'

    Returns:
        ChatMessage object (SystemMessage, UserMessage, AssistantMessage, or ToolMessage)
    """
    role = msg.get("role", "user")
    content_raw = msg.get("content", "")
    tool_calls = msg.get("tool_calls")
    tool_call_id = msg.get("tool_call_id")
    function_name = msg.get("name")  # For tool messages
    thinking_blocks = msg.get("thinking_blocks", [])

    # Handle content that might be a list of dicts with 'type' and 'text'
    if isinstance(content_raw, list):
        # Extract text from list of content objects
        content_parts = []
        for item in content_raw:
            if isinstance(item, dict):
                content_parts.append(item.get("text", ""))
            else:
                content_parts.append(str(item))
        content = "\n".join(content_parts)
    else:
        content = str(content_raw) if content_raw else ""

    # Map role to Docent's proper message types
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return UserMessage(content=[ContentText(text=content)])
    elif role == "assistant":
        # Build content list with reasoning and text
        content_list = []
        
        # Add thinking blocks as ContentReasoning
        if thinking_blocks:
            for block in thinking_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    if thinking_text:
                        content_list.append(ContentReasoning(reasoning=thinking_text))
        
        # Add main content as ContentText
        if content:
            content_list.append(ContentText(text=content))
        
        # Check if there are tool calls in the content
        extracted_tool_calls = extract_tool_calls(content)

        if extracted_tool_calls:
            # Parse tool calls into ToolCall objects
            tool_call_objects = []
            for tc in extracted_tool_calls:
                tool_call_id = tc.get("id", "")
                tool_type = tc.get("type", "function")
                func_data = tc.get("function", {})
                func_name = func_data.get("name", "")
                func_args = func_data.get("arguments", "")
                
                # Convert arguments to dict if it's a string
                if isinstance(func_args, str):
                    # Wrap the string in a dict with "code" key for code-based tools
                    args_dict = {"code": func_args}
                    args_str = func_args
                else:
                    args_dict = func_args if isinstance(func_args, dict) else {}
                    # Convert dict to string for ToolCallContent
                    args_str = str(func_args) if func_args else ""

                # Create ToolCall with view (content must be string)
                tool_call_obj = ToolCall(
                    id=tool_call_id,
                    function=func_name,
                    arguments=args_dict,
                    view=ToolCallContent(format="markdown", content=args_str),
                )
                tool_call_objects.append(tool_call_obj)

            # Use content_list if we have reasoning, otherwise use string content
            if content_list and any(isinstance(c, ContentReasoning) for c in content_list):
                return AssistantMessage(content=content_list, tool_calls=tool_call_objects)
            else:
                return AssistantMessage(content=content, tool_calls=tool_call_objects)
        else:
            # Use content_list if we have reasoning, otherwise use string content
            if content_list and any(isinstance(c, ContentReasoning) for c in content_list):
                return AssistantMessage(content=content_list)
            else:
                return AssistantMessage(content=content)
    elif role == "tool":
        if tool_call_id and function_name:
            return ToolMessage(
                content=content, tool_call_id=tool_call_id, function=function_name
            )
        else:
            # Fallback to assistant message if tool info is missing
            return AssistantMessage(content=content)
    else:
        # Default to user message for unknown roles
        return UserMessage(content=[ContentText(text=content)])


def reconstruct_conversation_from_log_entries_specialist(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[SystemMessage | UserMessage | AssistantMessage | ToolMessage]:
    """
    Reconstruct conversation for SPECIALIST agents (single message increments).
    
    For specialist agents, each turn adds exactly one user message and one assistant response.
    We simply take the last message from inputs and add the assistant's output.

    Args:
        log_entries: List of log entry dictionaries sorted by created_timestamp
        verbose: Enable verbose logging

    Returns:
        List of ChatMessage objects representing the full conversation
    """
    if not log_entries:
        return []

    conversation = []

    # For each entry, add the last input message and then the output
    for idx, entry in enumerate(log_entries, start=1):
        entry_input_messages = entry.get("inputs", {}).get("messages", [])
        
        if verbose:
            print(f"      Entry {idx}: {len(entry_input_messages)} messages in inputs")
            print(f"      Current conversation length before: {len(conversation)} messages")
        
        # Add the last message from entry inputs (the user message for this turn)
        if entry_input_messages:
            last_input_msg = entry_input_messages[-1]
            if verbose:
                print(f"      ✅ Adding last input message")
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
            conversation.append(parse_message_dict_to_chat_message(last_input_msg))
        
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
            
            if verbose:
                content_preview = str(assistant_content)[:100]
                print(f"      ✅ Adding assistant output")
                print(f"      Assistant content preview: {content_preview}")
                print(f"      Thinking blocks: {len(thinking_blocks)}")

            # Add the assistant's output message
            output_msg_dict = {
                "role": "assistant",
                "content": assistant_content,
                "thinking_blocks": thinking_blocks
            }
            conversation.append(parse_message_dict_to_chat_message(output_msg_dict))

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
            print(f"      Current conversation length before: {len(conversation)} messages")
        
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
                conversation.append(parse_message_dict_to_chat_message(msg_dict))
        
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
            
            if verbose:
                content_preview = str(assistant_content)[:100]
                print(f"      ✅ Adding assistant output")
                print(f"      Assistant content preview: {content_preview}")
                print(f"      Thinking blocks: {len(thinking_blocks)}")

            # Add the assistant's output message
            output_msg_dict = {
                "role": "assistant",
                "content": assistant_content,
                "thinking_blocks": thinking_blocks
            }
            conversation.append(parse_message_dict_to_chat_message(output_msg_dict))

    return conversation


def hal_taubench_to_docent_taubench(
    log_entries: List[Dict[str, Any]],
    model_name: str,
    eval_results_data: Dict[str, Any],
    config_data: Dict[str, Any],
    is_generalist: bool = False,
    verbose: bool = False,
) -> AgentRun:
    """
    Convert HAL TAU-bench log entries for a task into a Docent AgentRun.

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
    entry_model = first_entry["inputs"]["model"]
    if entry_model != model_name:
        if verbose:
            print(f"   ⚠️  Model mismatch: expected '{model_name}', got '{entry_model}'")
        assert entry_model == model_name
    task_id = first_entry["weave_task_id"]

    # Reconstruct the full conversation from all log entries
    if verbose:
        agent_type = "generalist" if is_generalist else "specialist"
        print(f"   🔄 Reconstructing conversation for task {task_id} ({agent_type}) from {len(log_entries)} log entries")
    
    if is_generalist:
        messages = reconstruct_conversation_from_log_entries_generalist(log_entries, verbose=verbose)
    else:
        messages = reconstruct_conversation_from_log_entries_specialist(log_entries, verbose=verbose)

    # Extract metadata from config
    run_id = config_data.get("run_id", "unknown")
    agent_name = config_data.get("agent_name", "unknown")
    date_str = config_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    benchmark_name = config_data.get("benchmark_name", "taubench")

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

    # TAU-bench task IDs are typically numeric (e.g., "46", "73")
    task_number = str(task_id)
    domain = "airline"  # TAU-bench airline domain

    metadata = TAUBenchMetadata(
        benchmark_id="taubench",
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
            "domain": domain,
            "task_number": task_number,
        },
        scoring_metadata=None,
        domain=domain,
        task_number=task_number,
    )

    # Convert metadata to dict
    metadata_dict = metadata.model_dump()
    
    transcript = Transcript(
        messages=messages,
        metadata=metadata_dict,
    )

    agent_run = AgentRun(
        transcripts={"default": transcript},
        metadata=metadata_dict,
    )

    return agent_run


# ============================================================================
# FILE PROCESSING
# ============================================================================


def deduplicate_log_entries(
    log_entries: List[Dict[str, Any]],
    model_name: str,
    target_first_message_prefix: str,
    task_id: str = "unknown",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Deduplicate log entries through three filtering stages:
    1. Filter by first message prefix
    2. Filter by model name
    3. Remove duplicate entries based on message content and role
    
    Args:
        log_entries: List of log entries for a single task
        model_name: The model name to filter by
        target_first_message_prefix: The expected first message prefix
        task_id: Task ID for logging purposes
        verbose: Whether to print detailed logging
        
    Returns:
        Deduplicated list of log entries
    """
    if not log_entries:
        return []
    
    if verbose:
        print(f"\n   🔍 Processing {task_id}:")
        print(f"      Initial: {len(log_entries)} log entries")
        # Show message counts with entry numbers
        for i, entry in enumerate(log_entries, start=1):
            messages = entry.get("inputs", {}).get("messages", [])
            entry_model = entry.get("inputs", {}).get("model", "unknown")
            print(f"         - Log {i}: {len(messages)} messages in inputs, model={entry_model}")
    
    # Stage 1: Filter by first message prefix
    first_stage_filtered = []
    for log_entry in log_entries:
        messages = log_entry.get("inputs", {}).get("messages", [])
        if messages and len(messages) > 0:
            first_message_content = messages[0].get("content", "")

            # Handle content that might be a list of dicts with 'type' and 'text'
            first_message_text = ""
            if isinstance(first_message_content, list):
                # Filter dicts with type="text" and extract the "text" field
                text_dicts = [
                    item
                    for item in first_message_content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_dicts:
                    # Usually just one dict, take the first one
                    first_message_text = text_dicts[0].get("text", "")
            else:
                first_message_text = str(first_message_content)

            if first_message_text.startswith(target_first_message_prefix):
                first_stage_filtered.append(log_entry)
    
    if verbose:
        removed_entries = [entry for entry in log_entries if entry not in first_stage_filtered]
        removed_msg_counts = [len(entry.get("inputs", {}).get("messages", [])) for entry in removed_entries]
        print(f"      Stage 1 (prefix filter): {len(first_stage_filtered)} entries (removed {len(log_entries) - len(first_stage_filtered)})")
        if removed_msg_counts:
            print(f"         Removed entries had message counts: {removed_msg_counts}")
    
    if not first_stage_filtered:
        return []
    
    # Stage 2: Filter by model name
    second_stage_filtered = []
    for log_entry in first_stage_filtered:
        entry_model = log_entry.get("inputs", {}).get("model", "unknown")
        if entry_model == model_name:
            second_stage_filtered.append(log_entry)
    
    if verbose:
        removed_entries = [entry for entry in first_stage_filtered if entry not in second_stage_filtered]
        removed_msg_counts = [len(entry.get("inputs", {}).get("messages", [])) for entry in removed_entries]
        print(f"      Stage 2 (model filter): {len(second_stage_filtered)} entries (removed {len(first_stage_filtered) - len(second_stage_filtered)})")
        if removed_msg_counts:
            print(f"         Removed entries had message counts: {removed_msg_counts}")
    
    if not second_stage_filtered:
        return []
    
    # Stage 3: Remove duplicates based on content
    # Group by message count, then deduplicate within each group
    by_length = {}
    for entry in second_stage_filtered:
        messages = entry.get("inputs", {}).get("messages", [])
        msg_count = len(messages)
        if msg_count not in by_length:
            by_length[msg_count] = []
        by_length[msg_count].append(entry)
    
    # For each message count group, deduplicate by content
    unique_entries = []
    for msg_count, entries in sorted(by_length.items()):
        seen_signatures = set()
        for entry in entries:
            messages = entry.get("inputs", {}).get("messages", [])
            
            # Create a signature from the messages (role + full content)
            signature_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Handle content that might be a list
                if isinstance(content, list):
                    content_str = str([
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    ])
                else:
                    content_str = str(content)
                
                signature_parts.append(f"{role}:{content_str}")
            
            signature = "||".join(signature_parts)
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_entries.append(entry)
    
    if verbose:
        removed_entries = [entry for entry in second_stage_filtered if entry not in unique_entries]
        removed_msg_counts = [len(entry.get("inputs", {}).get("messages", [])) for entry in removed_entries]
        print(f"      Stage 3 (deduplication): {len(unique_entries)} entries (removed {len(second_stage_filtered) - len(unique_entries)})")
        if removed_msg_counts:
            print(f"         Removed entries had message counts: {removed_msg_counts}")
        # Show final message counts with entry numbers
        print(f"      Final deduplicated entries:")
        for i, entry in enumerate(unique_entries, start=1):
            messages = entry.get("inputs", {}).get("messages", [])
            entry_model = entry.get("inputs", {}).get("model", "unknown")
            print(f"         - Log {i}: {len(messages)} messages in inputs, model={entry_model}")
        print(f"      ✅ Total removed: {len(log_entries) - len(unique_entries)} duplicate/filtered entries")
    
    return unique_entries


def process_taubench_file(
    file_path: str,
    max_tasks: int | None = None,
    target_first_message_prefix: str = "",
    is_generalist: bool = False,
    verbose: bool = False,
) -> List[AgentRun]:
    """
    Process a single TAU-bench JSON file and extract agent runs.

    Args:
        file_path: Path to the JSON file
        max_tasks: Maximum number of tasks to process (for dry runs)
        target_first_message_prefix: The expected first message prefix for filtering (optional)
        is_generalist: Whether this is a generalist agent (affects conversation reconstruction)
        verbose: Enable verbose logging

    Returns:
        List of AgentRun objects
    """
    print(f"\n📂 Processing file: {os.path.basename(file_path)}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract config and eval results
    config_data = data.get("config", {})
    results_data = data.get("results", {})

    if not results_data:
        print("   ❌ No results found, skipping file")
        return []

    # Get unique task IDs from results.latencies
    results = data.get("results", {})
    latencies = results.get("latencies", {})
    unique_task_ids = set(latencies.keys())

    print(f"   📊 Found {len(unique_task_ids)} unique task IDs in results.latencies")

    # Organize logs by task_id - each task may have multiple log entries
    logs = data.get("raw_logging_results", [])
    task_logs_dict = {}  # task_id -> list of log entries

    for log_entry in logs:
        task_id = log_entry.get("weave_task_id")
        if task_id and task_id in unique_task_ids:  # Only include if in latencies
            if task_id not in task_logs_dict:
                task_logs_dict[task_id] = []
            task_logs_dict[task_id].append(log_entry)

    # Sort log entries by timestamp for each task
    for task_id in task_logs_dict:
        task_logs_dict[task_id].sort(
            key=lambda x: x.get("created_timestamp", ""), reverse=False
        )

    print(f"   📊 Found {len(task_logs_dict)} tasks with log entries")

    # Debug: Print info for each task
    for task_id, log_entries in sorted(task_logs_dict.items())[:5]:  # Show first 5
        print(f"   🔍 Task {task_id}: {len(log_entries)} log entries")
        for i, entry in enumerate(log_entries):
            messages = entry.get("inputs", {}).get("messages", [])
            print(f"      - Log {i + 1}: {len(messages)} messages in inputs")
    if len(task_logs_dict) > 5:
        print(f"   ... and {len(task_logs_dict) - 5} more tasks")

    # Get the model name from config
    agent_args = config_data.get("agent_args", {})
    model_name = agent_args.get("model_name", "unknown")
    
    # Normalize model name: remove provider prefixes
    original_model_name = model_name
    if model_name.startswith("gemini/"):
        model_name = model_name.replace("gemini/", "")
        print(f"   🔧 Normalized model name from '{original_model_name}' to: '{model_name}'")
    elif model_name.startswith("together_ai/"):
        model_name = model_name.replace("together_ai/", "")
        print(f"   🔧 Normalized model name from '{original_model_name}' to: '{model_name}'")

    # Deduplicate log entries for each task through all three filtering stages
    deduped_task_logs = {}
    print("\n   🔄 Deduplicating log entries for each task...")
    
    for task_id, log_entries in task_logs_dict.items():
        deduped_entries = deduplicate_log_entries(
            log_entries, 
            model_name, 
            target_first_message_prefix,
            task_id=task_id,
            verbose=True
        )
        if deduped_entries:
            deduped_task_logs[task_id] = deduped_entries
    
    print(f"\n   📊 Final result: {len(deduped_task_logs)} tasks with deduplicated log entries")

    # Process tasks
    agent_runs = []
    processed = 0

    for task_id, log_entries in deduped_task_logs.items():
        if max_tasks and processed >= max_tasks:
            break

        print(f"\n   {'-' * 70}")
        print(f"   🔧 Processing task_id: {task_id}")
        
        # Print log entries info
        message_counts = [len(entry.get("inputs", {}).get("messages", [])) for entry in log_entries]
        print(f"   📊 Log entries: {len(log_entries)}")
        print(f"   📊 Message counts per entry: {message_counts}")

        # Process without try-catch - let errors stop the pipeline
        if verbose:
            print(f"   🔧 Task has {len(log_entries)} log entries")
        agent_run = hal_taubench_to_docent_taubench(
            log_entries, model_name, results_data, config_data, is_generalist=is_generalist, verbose=verbose
        )
        agent_runs.append(agent_run)
        processed += 1
        print(f"   {'-' * 70}")

    print(f"   ✅ Successfully processed {len(agent_runs)} agent runs")
    return agent_runs


def process_all_taubench_files(
    directory: str,
    pattern: str,
    max_files: int | None = None,
    max_tasks_per_file: int | None = None,
    target_first_message_prefix: str = "",
    is_generalist: bool = False,
    verbose: bool = False,
) -> List[AgentRun]:
    """
    Process all TAU-bench JSON files in a directory matching the pattern.

    Args:
        directory: Directory containing the JSON files
        pattern: Regex pattern to match files
        max_files: Maximum number of files to process
        max_tasks_per_file: Maximum number of tasks to process per file
        target_first_message_prefix: The expected first message prefix for filtering (optional)
        is_generalist: Whether this is a generalist agent (affects conversation reconstruction)
        verbose: Enable verbose logging

    Returns:
        List of all AgentRun objects
    """
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

    if max_files:
        json_files = json_files[:max_files]
        print(f"   Limiting to first {max_files} files")

    for json_file in json_files:
        print(f"\n{'=' * 80}")
        print(f"📄 Processing file: {json_file}")
        print(f"{'=' * 80}")
        
        file_path = os.path.join(directory, json_file)
        agent_runs = process_taubench_file(
            file_path,
            max_tasks=max_tasks_per_file,
            target_first_message_prefix=target_first_message_prefix,
            is_generalist=is_generalist,
            verbose=verbose,
        )
        all_agent_runs.extend(agent_runs)

    print(f"\n✅ Total agent runs collected: {len(all_agent_runs)}")
    return all_agent_runs


# ============================================================================
# MAIN
# ============================================================================


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload TAU-bench agent runs to Docent collection."
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
    args = parser.parse_args()

    # Configuration
    # Get the directory relative to the script location (one level up from v1_integration)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level
    directory = os.path.join(project_root, "hal_traces", "tau_bench_data")

    if args.agent_type == "specialist":
        pattern = TAUBENCH_SPECIALIST_PATTERN
        collection_prefix = "TAUBench-Specialist"
        target_first_message_prefix = "# Airline Agent Policy\n\nThe current time"
        is_generalist = False
    else:
        pattern = TAUBENCH_GENERALIST_PATTERN
        collection_prefix = "TAUBench-Generalist"
        target_first_message_prefix = "You are an expert assistant who can solve any task using code blobs."
        is_generalist = True

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
    print(f"TAU-bench Benchmark Integration - V1")
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
    agent_runs = process_all_taubench_files(
        directory=directory,
        pattern=pattern,
        max_files=max_files,
        max_tasks_per_file=max_tasks_per_file,
        target_first_message_prefix=target_first_message_prefix,
        is_generalist=is_generalist,
        verbose=args.verbose,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        return

    # Upload to Docent
    print(f"\n📤 Uploading {len(agent_runs)} agent runs to Docent...")

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"TAU-bench {args.agent_type} agent runs - {len(agent_runs)} runs processed",
    )

    # Upload agent runs in chunks
    chunk_size = 5
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
