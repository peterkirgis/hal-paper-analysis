"""
Common utility functions for benchmark processing.

This module contains shared functions used across multiple benchmark scripts:
- normalize_generalist_content: Clean assistant message content
- extract_tool_calls: Extract tool calls from content strings
- parse_message_dict_to_chat_message: Convert message dicts to Docent ChatMessage objects
- deduplicate_log_entries: Deduplicate log entries with multi-stage filtering
- check_transcript_contains_largest_entry: Validate transcript against raw logs
- load_and_organize_benchmark_file: Load, filter, and organize log entries from benchmark files
- validate_agent_run: Validate AgentRun objects to catch serialization issues early
- timestamp_based_ordering: Order log entries by timestamp and cut off when message counts decrease
- get_entry_with_most_messages: Return the single log entry with the largest number of messages
- resolve_duplicates_by_timestamp: Resolve duplicate message counts by keeping the latest timestamp
"""

import ast
import json
import os
import re
from typing import Any, Dict, List

from dateutil import parser as dateutil_parser
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


def validate_agent_run(agent_run, task_id: str = "unknown", verbose: bool = False):
    """
    Validate an AgentRun object to catch serialization issues early.

    This validates that the AgentRun can be properly serialized and deserialized,
    which helps catch issues before uploading to Docent.

    Args:
        agent_run: The AgentRun object to validate
        task_id: Task ID for error reporting
        verbose: Enable verbose logging

    Raises:
        Exception: If validation fails
    """
    try:
        validated_dict = agent_run.model_dump()
        from docent.data_models import AgentRun

        AgentRun.model_validate(validated_dict)
        if verbose:
            print(f"   ✅ AgentRun validation passed for task {task_id}")
    except Exception as e:
        print(f"\n❌ AgentRun validation failed for task {task_id}:")
        print(f"   Error: {e}")
        if hasattr(agent_run, "transcripts") and agent_run.transcripts:
            messages = agent_run.transcripts[0].messages
            print(f"   Transcript has {len(messages)} messages:")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                print(f"      Message {i}: {msg_type}")
        raise


def normalize_generalist_content(content: str) -> str:
    """
    Normalize generalist agent content by removing <end_code> and tool call metadata.

    For generalist agents:
    - Entry content: may or may not have <end_code>
    - Transcript content: may have <end_code> at the end

    We need to normalize both by removing <end_code> markers and anything after them.

    Args:
        content: Raw content string from log entry or transcript

    Returns:
        Normalized content string
    """
    # Remove <end_code> and everything after it (including "Calling tools:" section)
    if "<end_code>" in content:
        content = content.split("<end_code>")[0]

    # Also handle the "Calling tools:" pattern even if <end_code> is missing
    if "Calling tools:" in content:
        content = content.split("Calling tools:")[0]

    return content.strip()


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
    verbose: bool = False,
) -> SystemMessage | UserMessage | AssistantMessage | ToolMessage:
    """
    Convert a single message dictionary to a Docent ChatMessage object.

    Args:
        msg: Message dictionary with 'role' and 'content'
        verbose: Enable verbose logging

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
        if verbose:
            print(f"         → Returning SystemMessage")
        return SystemMessage(content=content)
    elif role == "user":
        if verbose:
            print(f"         → Returning UserMessage")
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

        # Check for DeepSeek <think> blocks in content and extract them as reasoning
        think_content = ""
        main_content = content
        if content and "<think>" in content and "</think>" in content:
            # Extract the thinking block
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                # Remove the <think> block from main content
                main_content = re.sub(
                    r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
                ).strip()
                # Add thinking as ContentReasoning
                if think_content:
                    content_list.append(ContentReasoning(reasoning=think_content))

        # Add main content as ContentText (only if not empty and not None)
        if main_content:
            content_list.append(ContentText(text=main_content))

        # Collect tool calls from both sources and unify them:
        # 1. Structured tool_calls field from message dict
        # 2. Extracted from content string using extract_tool_calls
        all_tool_calls = []

        # Add structured tool_calls if present
        if tool_calls:
            all_tool_calls.extend(tool_calls)

        # Add extracted tool calls from content if present
        extracted_tool_calls = extract_tool_calls(content)
        if extracted_tool_calls:
            all_tool_calls.extend(extracted_tool_calls)

        # Process all collected tool calls
        tool_call_objects = []
        if all_tool_calls:
            if verbose:
                print(f"         Processing {len(all_tool_calls)} tool calls:")
            for idx, tc in enumerate(all_tool_calls):
                if verbose:
                    print(f"         Tool call {idx}: {tc}")

                tool_call_id = tc.get("id", "")
                tool_type = tc.get("type", "function")
                func_data = tc.get("function", {})
                
                # Handle None or missing function data
                if func_data is None:
                    func_data = {}
                
                func_name = func_data.get("name") or ""  # Ensure not None
                func_args = func_data.get("arguments") or ""  # Ensure not None

                if verbose:
                    print(f"           - id: {tool_call_id}")
                    print(f"           - function name: {func_name}")
                    print(f"           - arguments type: {type(func_args)}")
                    print(f"           - arguments value: {func_args}")

                # Convert arguments to dict if it's a string
                if isinstance(func_args, str):
                    # Try to parse JSON arguments, otherwise wrap in dict
                    try:
                        args_dict = json.loads(func_args)
                        if not isinstance(args_dict, dict):
                            # If parsed result is not a dict, wrap it
                            args_dict = {"code": func_args}
                    except:
                        # If JSON parsing fails, wrap the string in a dict
                        args_dict = {"code": func_args}
                elif isinstance(func_args, dict):
                    # Already a dict, use as-is
                    args_dict = func_args
                else:
                    # Neither string nor dict (e.g., None, int, etc.), create empty dict
                    args_dict = {}

                # Create ToolCall with view (content must be string)
                tool_call_obj = ToolCall(
                    id=tool_call_id,
                    function=func_name,
                    arguments=args_dict,
                    type="function",
                )
                tool_call_objects.append(tool_call_obj)

        # Return AssistantMessage with tool calls if we have any
        if tool_call_objects:
            # Use content_list if we have reasoning or if content is empty
            # When content is null/empty but we have tool calls, we need at least an empty list
            if verbose:
                print(
                    f"         → Returning AssistantMessage with {len(tool_call_objects)} tool call(s)"
                )
            if content_list:
                return AssistantMessage(
                    content=content_list, tool_calls=tool_call_objects
                )
            elif content:
                return AssistantMessage(content=content, tool_calls=tool_call_objects)
            else:
                # Content is null/empty, but we have tool calls - use empty content list
                return AssistantMessage(content=[], tool_calls=tool_call_objects)
        else:
            # Use content_list if we have reasoning, otherwise use string content
            if verbose:
                has_reasoning = (
                    any(isinstance(c, ContentReasoning) for c in content_list)
                    if content_list
                    else False
                )
                if has_reasoning:
                    print(
                        f"         → Returning AssistantMessage with reasoning blocks"
                    )
                else:
                    print(f"         → Returning AssistantMessage")
            if content_list and any(
                isinstance(c, ContentReasoning) for c in content_list
            ):
                return AssistantMessage(content=content_list)
            else:
                return AssistantMessage(content=content)
    elif role == "tool":
        if verbose:
            print(f"         → Returning ToolMessage (function: {function_name})")
        return ToolMessage(
            content=content, tool_call_id=tool_call_id, function=function_name
        )
    else:
        # Default to user message for unknown roles
        if verbose:
            print(f"         → Returning UserMessage (unknown role: {role})")
        return UserMessage(content=[ContentText(text=content)])


def deduplicate_log_entries(
    log_entries: List[Dict[str, Any]],
    model_name: str,
    target_first_message_prefix: str,
    task_id: str = "unknown",
    file_name: str = "unknown",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Deduplicate log entries through three filtering stages:
    1. Filter by first message prefix
    2. Filter by model name
    3. Remove duplicate entries based on message content and role

    Args:
        log_entries: List of log entries for a single task
        model_name: The model name to filter by (last part after splitting by '/')
        target_first_message_prefix: The expected first message prefix
        task_id: Task ID for logging purposes
        file_name: File name for debug output
        verbose: Whether to print detailed logging

    Returns:
        List of deduplicated log entries
    """
    if verbose:
        print(f"   🔄 Deduplicating task {task_id}:")
        print(f"      Input: {len(log_entries)} log entries")

    # Stage 1: Filter by first message prefix (if specified)
    if target_first_message_prefix:
        first_stage_filtered = []
        for log_entry in log_entries:
            messages = log_entry.get("inputs", {}).get("messages", [])
            if not messages:
                continue

            first_message = messages[0]
            first_message_content = first_message.get("content", "")

            # Handle list-type content
            if isinstance(first_message_content, list):
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
            removed_entries = [
                entry for entry in log_entries if entry not in first_stage_filtered
            ]
            removed_msg_counts = [
                len(entry.get("inputs", {}).get("messages", []))
                for entry in removed_entries
            ]
            print(
                f"      Stage 1 (prefix filter): {len(first_stage_filtered)} entries (removed {len(log_entries) - len(first_stage_filtered)})"
            )
            if removed_msg_counts:
                print(
                    f"         Removed entries had message counts: {removed_msg_counts}"
                )

        if not first_stage_filtered:
            return []
    else:
        first_stage_filtered = log_entries

    # Stage 2: Filter by model name (compare last part after splitting by '/')
    second_stage_filtered = []
    for log_entry in first_stage_filtered:
        entry_model_full = log_entry.get("inputs", {}).get("model", "unknown")
        # Extract the last part after splitting by '/'
        entry_model = entry_model_full.split("/")[-1]
        if entry_model == model_name:
            second_stage_filtered.append(log_entry)

    if verbose:
        removed_entries = [
            entry
            for entry in first_stage_filtered
            if entry not in second_stage_filtered
        ]
        removed_msg_counts = [
            len(entry.get("inputs", {}).get("messages", []))
            for entry in removed_entries
        ]
        removed_models = [
            entry.get("inputs", {}).get("model", "unknown").split("/")[-1]
            for entry in removed_entries
        ]
        print(
            f"      Stage 2 (model filter): {len(second_stage_filtered)} entries (removed {len(first_stage_filtered) - len(second_stage_filtered)})"
        )
        if removed_msg_counts:
            print(f"         Removed entries had message counts: {removed_msg_counts}")
            print(f"         Removed models: {removed_models}")

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
        # First, create a content-only signature (without role) to detect entries with same content but different roles
        content_signatures = {}  # content_signature -> list of entries
        for entry in entries:
            messages = entry.get("inputs", {}).get("messages", [])

            # Create content-only signature
            content_parts = []
            for msg in messages:
                content = msg.get("content", "")

                # Handle content that might be a list
                if isinstance(content, list):
                    content_str = "\n".join(
                        [
                            item.get("text", "")
                            if isinstance(item, dict)
                            else str(item)
                            for item in content
                        ]
                    )
                else:
                    # Normalize None to empty string
                    content_str = "" if content is None else str(content)

                content_parts.append(content_str)

            content_sig = "||".join(content_parts).strip()

            if content_sig not in content_signatures:
                content_signatures[content_sig] = []
            content_signatures[content_sig].append(entry)

        # For entries with same content sequences, prefer the one with first role="system" over first role="user"
        filtered_entries = []
        for content_sig, same_content_entries in content_signatures.items():
            if len(same_content_entries) > 1:
                # Multiple entries with same content sequences - check if they differ only in first role
                # Group by first role
                by_first_role = {}
                for entry in same_content_entries:
                    msgs = entry.get("inputs", {}).get("messages", [])
                    if msgs:
                        first_role = msgs[0].get("role", "")
                        if first_role not in by_first_role:
                            by_first_role[first_role] = []
                        by_first_role[first_role].append(entry)

                # If we have both "system" and "user" as first roles, prefer "system"
                if "system" in by_first_role and "user" in by_first_role:
                    # Keep only entries with first role="system"
                    filtered_entries.extend(by_first_role["system"])
                    if verbose:
                        removed_count = len(by_first_role["user"])
                        print(
                            f"         Filtered {removed_count} entries with first role='user' (keeping entries with first role='system')"
                        )
                else:
                    # No role conflict, keep all
                    filtered_entries.extend(same_content_entries)
            else:
                # Only one entry with this content
                filtered_entries.extend(same_content_entries)

        # Now deduplicate by full signature (role + content)
        seen_signatures = {}  # signature -> entry
        for entry in filtered_entries:
            messages = entry.get("inputs", {}).get("messages", [])

            # Create a signature from the messages (role + full content)
            signature_parts = []
            for msg_idx, msg in enumerate(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Handle content that might be a list
                if isinstance(content, list):
                    # Join text content with newlines (same as .text property)
                    content_str = "\n".join(
                        [
                            item.get("text", "")
                            if isinstance(item, dict)
                            else str(item)
                            for item in content
                        ]
                    )
                else:
                    # Normalize None to empty string for comparison
                    if content is None:
                        content_str = ""

                        # Log the entire message JSON to file
                        # Use agent_type string for directory structure
                        agent_type_str = (
                            "generalist"
                            if "hal_generalist_agent" in file_name
                            else "specialist"
                        )
                        none_content_dir = os.path.join(
                            "none_content_entries", agent_type_str, file_name, task_id
                        )
                        os.makedirs(none_content_dir, exist_ok=True)

                        # Create a unique filename based on message index
                        none_content_file = os.path.join(
                            none_content_dir, f"message_{msg_idx}_none_content.json"
                        )

                        with open(none_content_file, "w") as f:
                            json.dump(msg, f, indent=4)
                    else:
                        content_str = str(content)

                signature_parts.append(f"{role}:{content_str}")

            signature = "||".join(signature_parts).strip()

            # If we haven't seen this signature, add it
            if signature not in seen_signatures:
                seen_signatures[signature] = entry
            else:
                # We have a duplicate - decide which one to keep
                existing_entry = seen_signatures[signature]

                # Get output status
                existing_has_output = existing_entry.get("output") is not None
                current_has_output = entry.get("output") is not None

                # Priority 1: Prefer entries with output over those without
                # Priority 2: Among entries with same output status, prefer latest timestamp
                should_replace = False

                if current_has_output and not existing_has_output:
                    # Current has output, existing doesn't - always replace
                    should_replace = True
                elif current_has_output == existing_has_output:
                    # Both have same output status - compare timestamps
                    existing_timestamp_str = existing_entry.get("created_timestamp", "")
                    current_timestamp_str = entry.get("created_timestamp", "")

                    # Parse timestamps for proper comparison
                    try:
                        existing_ts = (
                            dateutil_parser.parse(existing_timestamp_str)
                            if existing_timestamp_str
                            else None
                        )
                        current_ts = (
                            dateutil_parser.parse(current_timestamp_str)
                            if current_timestamp_str
                            else None
                        )

                        if current_ts and existing_ts and current_ts > existing_ts:
                            should_replace = True
                        elif current_ts and not existing_ts:
                            # Current has valid timestamp, existing doesn't
                            should_replace = True
                    except (ValueError, TypeError):
                        # If parsing fails, fall back to string comparison
                        if current_timestamp_str > existing_timestamp_str:
                            should_replace = True
                # If existing has output and current doesn't, keep existing (don't replace)

                if should_replace:
                    seen_signatures[signature] = entry

        # Add all unique entries (after deduplication) to the result
        unique_entries.extend(seen_signatures.values())

    if verbose:
        removed_entries = [
            entry for entry in second_stage_filtered if entry not in unique_entries
        ]
        removed_msg_counts = [
            len(entry.get("inputs", {}).get("messages", []))
            for entry in removed_entries
        ]
        print(
            f"      Stage 3 (deduplication): {len(unique_entries)} entries (removed {len(second_stage_filtered) - len(unique_entries)})"
        )
        if removed_msg_counts:
            print(f"         Removed entries had message counts: {removed_msg_counts}")
        # Show final message counts with entry numbers
        print(f"      Final deduplicated entries:")
        for i, entry in enumerate(unique_entries, start=1):
            messages = entry.get("inputs", {}).get("messages", [])
            entry_model = entry.get("inputs", {}).get("model", "unknown").split("/")[-1]
            print(
                f"         - Log {i}: {len(messages)} messages in inputs, model={entry_model}"
            )
        print(
            f"      ✅ Total removed: {len(log_entries) - len(unique_entries)} duplicate/filtered entries"
        )

    return unique_entries


def check_transcript_contains_largest_entry(
    log_entries: List[Dict[str, Any]],
    transcript_messages: List[Any],
    task_id: str = "unknown",
    file_name: str = "unknown",
    is_generalist: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check if the largest log entry is a subset of the built transcript.

    Args:
        log_entries: List of log entries for a single task (after deduplication)
        transcript_messages: Built transcript messages (Docent format)
        task_id: Task identifier for logging
        file_name: File name for debug output
        is_generalist: Whether this is a generalist agent (affects content normalization)
        verbose: Whether to print detailed information

    Returns:
        Dictionary with check results
    """
    if not log_entries:
        return {"is_subset": True, "reason": "No entries to check", "passed": True}

    # Find entry with largest message count
    max_entry = max(
        log_entries, key=lambda e: len(e.get("inputs", {}).get("messages", []))
    )
    max_messages = max_entry.get("inputs", {}).get("messages", [])
    max_count = len(max_messages)

    if verbose:
        print(
            f"      Checking if largest entry ({max_count} messages) is subset of transcript ({len(transcript_messages)} messages)"
        )

    # Check if max_messages is a prefix subset of transcript_messages
    is_subset = True
    details = []

    if max_count > len(transcript_messages):
        is_subset = False
        details.append(
            f"Largest entry has {max_count} messages > transcript has {len(transcript_messages)}"
        )
    else:
        # Check if first max_count messages match
        for i in range(max_count):
            entry_msg = max_messages[i]

            # Get the transcript message (convert from Docent format)
            transcript_msg = transcript_messages[i]

            # Extract role from both
            entry_role = entry_msg.get("role", "")

            # Get role from Docent message object
            if hasattr(transcript_msg, "role"):
                transcript_role = transcript_msg.role
            else:
                # Fallback: infer from type
                transcript_role = (
                    type(transcript_msg).__name__.replace("Message", "").lower()
                )

            if entry_role != transcript_role:
                is_subset = False
                details.append(
                    f"Role mismatch at position {i}: entry='{entry_role}', transcript='{transcript_role}'"
                )
                if not verbose:
                    break

            # Extract content from entry message
            entry_content = entry_msg.get("content", "")
            if isinstance(entry_content, list):
                # Join text content with newlines (same as .text property)
                entry_content_str = "\n".join(
                    [
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in entry_content
                    ]
                )
            else:
                # Normalize None to empty string for comparison
                entry_content_str = "" if entry_content is None else str(entry_content)

            # Extract content from transcript message using .text property
            transcript_content_str = (
                transcript_msg.text
                if hasattr(transcript_msg, "text")
                else str(
                    transcript_msg.content if hasattr(transcript_msg, "content") else ""
                )
            )

            # Normalize content for generalist agents (remove <end_code> and tool call metadata from BOTH sides)
            if is_generalist and entry_role == "assistant":
                entry_content_str = normalize_generalist_content(entry_content_str)
                transcript_content_str = normalize_generalist_content(
                    transcript_content_str
                )

            if entry_content_str.strip() != transcript_content_str.strip():
                is_subset = False
                entry_preview = (
                    entry_content_str[:30]
                    if len(entry_content_str) > 30
                    else entry_content_str
                )
                transcript_preview = (
                    transcript_content_str[:30]
                    if len(transcript_content_str) > 30
                    else transcript_content_str
                )
                details.append(
                    f"Content mismatch at position {i}: entry='{entry_preview}...' vs transcript='{transcript_preview}...'"
                )

                # Write full content to debug files
                debug_dir = os.path.join("debug_mismatches", file_name, task_id)
                os.makedirs(debug_dir, exist_ok=True)

                entry_file = os.path.join(debug_dir, f"entry_pos_{i}.txt")
                transcript_file = os.path.join(debug_dir, f"transcript_pos_{i}.txt")

                with open(entry_file, "w") as f:
                    f.write(f"Position: {i}\n")
                    f.write(f"Role: {entry_role}\n")
                    f.write(f"Length: {len(entry_content_str)} characters\n")
                    f.write(f"{'=' * 80}\n")
                    f.write(entry_content_str)

                with open(transcript_file, "w") as f:
                    f.write(f"Position: {i}\n")
                    f.write(f"Role: {transcript_role}\n")
                    f.write(f"Length: {len(transcript_content_str)} characters\n")
                    f.write(f"{'=' * 80}\n")
                    f.write(transcript_content_str)

                if not verbose:
                    break

    result = {
        "is_subset": is_subset,
        "passed": is_subset,
        "max_entry_count": max_count,
        "transcript_count": len(transcript_messages),
        "details": details if not is_subset else [],
    }

    if verbose and not is_subset:
        print(f"      ❌ Check failed: {'; '.join(details)}")
    elif verbose:
        print(f"      ✅ Check passed")

    return result


def load_and_organize_benchmark_file(
    file_path: str,
    target_first_message_prefix: str = "",
    is_generalist: bool = False,
    verbose: bool = False,
    timestamp_based_resolving: bool = False,
):
    """
    Load a benchmark JSON file, organize logs by task, and deduplicate entries.

    This function handles the common file processing logic that is shared across
    all benchmark scripts:
    1. Load JSON file
    2. Extract config, results, and raw logs
    3. Organize logs by task_id
    4. Extract and normalize model name
    5. Deduplicate log entries
    6. Check for duplicate message counts

    Args:
        file_path: Path to the benchmark JSON file
        target_first_message_prefix: Expected first message prefix for filtering
        is_generalist: Whether this is a generalist agent
        verbose: Enable verbose logging
        timestamp_based_resolving: If True, prefer latest timestamp in case of duplicate message counts

    Returns:
        Dictionary containing:
        - file_name: Base filename without _UPLOAD.json
        - config_data: Configuration dictionary
        - eval_results_data: Dictionary with 'results' and 'raw_eval_results' keys
        - deduped_task_logs: Dict of task_id -> list of deduplicated log entries
        - model_name: Normalized model name (last part after /)
        - model_name_full: Original full model name
        - duplicate_stats: Stats about duplicate message counts
    """
    import json
    import os

    file_name = os.path.basename(file_path).replace("_UPLOAD.json", "")
    print(f"\n📂 Processing file: {file_name}")

    with open(file_path, "r") as f:
        data = json.load(f)

    config_data = data.get("config", {})
    eval_results_data = {
        "results": data.get("results", {}),
        "raw_eval_results": data.get("raw_eval_results", {}),
    }

    if not eval_results_data["results"]:
        print("   ❌ No results found, skipping file")
        return None

    # Get unique task IDs from results.latencies
    latencies = eval_results_data["results"].get("latencies", {})
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
    model_name_full = agent_args.get("model_name")

    # Fallback to agent.model.name if model_name is not found
    if not model_name_full:
        model_name_full = agent_args.get("agent.model.name", "unknown")
        if model_name_full != "unknown":
            print(f"   ℹ️  Using agent.model.name: '{model_name_full}'")

    # Extract the last part after splitting by '/' for comparison
    # e.g., "openrouter/anthropic/claude-opus-4.1" -> "claude-opus-4.1"
    model_name = model_name_full.split("/")[-1]

    if model_name != model_name_full:
        print(
            f"   🔧 Using model identifier: '{model_name}' (from '{model_name_full}')"
        )

    # Deduplicate log entries for each task through all three filtering stages
    deduped_task_logs = {}
    print("\n   🔄 Deduplicating log entries for each task...")

    for task_id, log_entries in task_logs_dict.items():
        deduped_entries = deduplicate_log_entries(
            log_entries,
            model_name,
            target_first_message_prefix,
            task_id=task_id,
            file_name=file_name,
            verbose=True,
        )
        if deduped_entries:
            deduped_task_logs[task_id] = deduped_entries

    print(
        f"\n   📊 Final result: {len(deduped_task_logs)} tasks with deduplicated log entries"
    )

    # Check for duplicate message counts in deduplicated entries
    print("\n   🔍 Checking for duplicate message counts...")
    tasks_with_duplicates = 0
    tasks_without_duplicates = 0

    agent_type_str = "generalist" if is_generalist else "specialist"

    for task_id, log_entries in deduped_task_logs.items():
        message_counts = [
            len(entry.get("inputs", {}).get("messages", [])) for entry in log_entries
        ]
        # Check if there are duplicate counts
        if len(message_counts) != len(set(message_counts)):
            tasks_with_duplicates += 1
            if verbose:
                print(
                    f"      ⚠️  Task {task_id} has duplicate message counts: {message_counts}"
                )

            # Write duplicate entries to files
            duplicate_dir = os.path.join(
                "duplicates", agent_type_str, file_name, task_id
            )
            os.makedirs(duplicate_dir, exist_ok=True)

            # Group entries by message count to identify duplicates
            count_to_entries = {}
            for entry in log_entries:
                msg_count = len(entry.get("inputs", {}).get("messages", []))
                if msg_count not in count_to_entries:
                    count_to_entries[msg_count] = []
                count_to_entries[msg_count].append(entry)

            # Print timestamps for duplicate message counts
            print(f"      Timestamps for duplicate message counts:")
            for msg_count, entries in sorted(count_to_entries.items()):
                if (
                    len(entries) > 1
                ):  # Only print if there are duplicates for this count
                    print(f"         Messages count {msg_count}:")
                    for idx, entry in enumerate(entries):
                        timestamp = entry.get("created_timestamp", "N/A")
                        print(f"            {msg_count}_{idx}: {timestamp}")

            # Resolve duplicates based on timestamp if flag is set
            if timestamp_based_resolving:
                deduped_task_logs[task_id] = resolve_duplicates_by_timestamp(
                    log_entries, verbose=verbose
                )

            # Write files for duplicate message counts
            for msg_count, entries in count_to_entries.items():
                if (
                    len(entries) > 1
                ):  # Only write if there are duplicates for this count
                    for idx, entry in enumerate(entries):
                        output_file = os.path.join(
                            duplicate_dir, f"messages_len_{msg_count}_idx_{idx}.log"
                        )
                        messages = entry.get("inputs", {}).get("messages", [])

                        with open(output_file, "w") as f:
                            for i, msg in enumerate(messages):
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")

                                # Extract text from content
                                if isinstance(content, list):
                                    content_str = "\n".join(
                                        [
                                            item.get("text", "")
                                            if isinstance(item, dict)
                                            else str(item)
                                            for item in content
                                        ]
                                    )
                                else:
                                    content_str = str(content)

                                f.write(f"# Role: {role}\n\n")
                                f.write(f"# Message {i + 1}:\n\n")
                                f.write(f"{content_str}\n\n")
                                f.write("=" * 80 + "\n\n")
        else:
            tasks_without_duplicates += 1

    total_tasks_checked = tasks_with_duplicates + tasks_without_duplicates
    duplicate_stats = {
        "tasks_with_duplicates": tasks_with_duplicates,
        "tasks_without_duplicates": tasks_without_duplicates,
        "total_tasks": total_tasks_checked,
    }

    if total_tasks_checked > 0:
        duplicate_percentage = (tasks_with_duplicates / total_tasks_checked) * 100
        print(f"\n   📊 Duplicate Message Count Check:")
        print(
            f"      ✅ Tasks without duplicates: {tasks_without_duplicates}/{total_tasks_checked} ({100 - duplicate_percentage:.1f}%)"
        )
        print(
            f"      ⚠️  Tasks with duplicates: {tasks_with_duplicates}/{total_tasks_checked} ({duplicate_percentage:.1f}%)"
        )
        if tasks_with_duplicates > 0:
            print(
                f"      📁 Duplicate entries written to: duplicates/{agent_type_str}/{file_name}/"
            )
    else:
        print(f"\n   📊 No tasks to check for duplicates")

    return {
        "file_name": file_name,
        "config_data": config_data,
        "eval_results_data": eval_results_data,  # Contains both 'results' and 'raw_eval_results' from trace
        "deduped_task_logs": deduped_task_logs,
        "model_name": model_name,
        "model_name_full": model_name_full,
        "duplicate_stats": duplicate_stats,
    }


def timestamp_based_ordering(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Order log entries by timestamp and cut off when message counts start decreasing.

    This function:
    1. Sorts all log entries by timestamp (ascending)
    2. Tracks message counts: [2, 4, 6, 8, 10, 2, 4, ...]
    3. Cuts off at the first decrease (returns [2, 4, 6, 8, 10])

    This handles cases where an agent was restarted or had multiple runs,
    keeping only the longest continuous increasing sequence.

    Args:
        log_entries: List of log entry dictionaries
        verbose: Enable verbose logging

    Returns:
        Filtered list of log entries up to the first message count decrease
    """
    if not log_entries:
        return []

    if len(log_entries) <= 1:
        return log_entries

    from dateutil import parser as date_parser

    # Sort entries by timestamp (ascending - oldest first)
    sorted_entries = sorted(
        log_entries,
        key=lambda e: date_parser.parse(
            e.get("created_timestamp", "1970-01-01T00:00:00")
        ),
    )

    # Get message counts for each entry
    message_counts = [
        len(entry.get("inputs", {}).get("messages", [])) for entry in sorted_entries
    ]

    if verbose:
        print(f"      📊 Message counts (timestamp-sorted): {message_counts}")

    # Find the cutoff point - first time message count decreases
    cutoff_index = len(sorted_entries)  # Default: keep all entries

    for i in range(1, len(message_counts)):
        if message_counts[i] <= message_counts[i - 1]:
            # Message count decreased or stayed same - cut off here
            cutoff_index = i
            if verbose:
                print(
                    f"      ✂️  Cutoff at index {i}: count went from {message_counts[i - 1]} to {message_counts[i]}"
                )
            break

    # Return entries up to (but not including) the cutoff point
    filtered_entries = sorted_entries[:cutoff_index]

    if verbose:
        print(f"      ✅ Kept {len(filtered_entries)}/{len(sorted_entries)} entries")
        if len(filtered_entries) < len(sorted_entries):
            kept_counts = message_counts[:cutoff_index]
            dropped_counts = message_counts[cutoff_index:]
            print(f"         Kept message counts: {kept_counts}")
            print(f"         Dropped message counts: {dropped_counts}")

    return filtered_entries


def get_entry_with_most_messages(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return the single log entry with the largest number of messages.

    This function is useful when you want to keep only the most complete entry,
    for example when dealing with agent restarts where you want the longest run.

    Args:
        log_entries: List of log entry dictionaries
        verbose: Enable verbose logging

    Returns:
        List containing only the entry with the most messages (single-element list)
        Returns empty list if input is empty
    """
    if not log_entries:
        return []

    if len(log_entries) == 1:
        return log_entries

    # Find entry with maximum number of messages
    max_entry = None
    max_message_count = -1

    for entry in log_entries:
        message_count = len(entry.get("inputs", {}).get("messages", []))
        if message_count > max_message_count:
            max_message_count = message_count
            max_entry = entry

    if verbose:
        all_counts = [len(e.get("inputs", {}).get("messages", [])) for e in log_entries]
        print(f"      📊 Message counts in all entries: {all_counts}")
        print(f"      ✅ Selected entry with {max_message_count} messages")
        if len(log_entries) > 1:
            print(f"      🗑️  Dropped {len(log_entries) - 1} entries")

    return [max_entry] if max_entry else []


def resolve_duplicates_by_timestamp(
    log_entries: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Resolve duplicate message counts by keeping the entry with the latest timestamp.

    When multiple log entries have the same message count (e.g., two entries with 4 messages),
    this function keeps only the entry with the latest timestamp for each duplicate count.

    This is useful for handling agent restarts or retries where the same message count
    appears multiple times.

    Args:
        log_entries: List of log entry dictionaries
        verbose: Enable verbose logging

    Returns:
        List of log entries with duplicates resolved (sorted by timestamp)
    """
    if not log_entries:
        return []

    if len(log_entries) <= 1:
        return log_entries

    from dateutil import parser as date_parser

    # Group entries by message count
    count_to_entries = {}
    for entry in log_entries:
        msg_count = len(entry.get("inputs", {}).get("messages", []))
        if msg_count not in count_to_entries:
            count_to_entries[msg_count] = []
        count_to_entries[msg_count].append(entry)

    if verbose:
        duplicate_counts = [
            count for count, entries in count_to_entries.items() if len(entries) > 1
        ]
        if duplicate_counts:
            print(f"      🔍 Found duplicate message counts: {duplicate_counts}")
        else:
            print(f"      ✅ No duplicate message counts found")

    # Resolve duplicates by keeping the latest timestamp
    resolved_entries = []
    for msg_count, entries in count_to_entries.items():
        if len(entries) > 1:
            # Sort by timestamp (latest first) and pick the latest
            sorted_entries = sorted(
                entries,
                key=lambda e: date_parser.parse(
                    e.get("created_timestamp", "1970-01-01T00:00:00")
                ),
                reverse=True,
            )
            latest_entry = sorted_entries[0]
            resolved_entries.append(latest_entry)
            if verbose:
                timestamps = [e.get("created_timestamp") for e in sorted_entries]
                print(
                    f"         ⏰ Resolved duplicate for message count {msg_count}: keeping entry with timestamp {latest_entry.get('created_timestamp')}"
                )
                print(f"            All timestamps: {timestamps}")
                print(f"            Dropped {len(entries) - 1} duplicate(s)")
        else:
            resolved_entries.append(entries[0])

    # Sort resolved entries by timestamp (oldest first)
    resolved_entries = sorted(
        resolved_entries,
        key=lambda e: date_parser.parse(
            e.get("created_timestamp", "1970-01-01T00:00:00")
        ),
    )

    if verbose:
        print(
            f"      ✅ Resolved from {len(log_entries)} to {len(resolved_entries)} entries"
        )
        original_counts = [
            len(e.get("inputs", {}).get("messages", [])) for e in log_entries
        ]
        resolved_counts = [
            len(e.get("inputs", {}).get("messages", [])) for e in resolved_entries
        ]
        print(f"         Original message counts: {original_counts}")
        print(f"         Resolved message counts: {resolved_counts}")

    return resolved_entries
