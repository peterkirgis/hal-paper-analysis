#!/usr/bin/env python3
"""
Script to read a HAL trace file and print all unique weave_task_ids
with their corresponding system messages.

Usage:
    python print_task_system_messages.py <trace_file_path>
    
Example:
    python print_task_system_messages.py hal_traces/swe_bench_mini_data/swebench_verified_mini_sweagentgpt520250807_1754592641_UPLOAD.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set


def extract_system_message(messages):
    """
    Extract the system message from a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        str: The system message content, or None if not found
    """
    if not messages:
        return None
    
    for msg in messages:
        if isinstance(msg, dict):
            # Check if this is a system message
            if msg.get("role") == "system":
                content = msg.get("content", "")
                
                # Handle case where content is a list (always size 1) with objects containing type and text
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        return first_item["text"]
                    elif isinstance(first_item, str):
                        return first_item
                
                # Handle case where content is already a string
                return content if content else None
    
    return None


def process_trace_file(file_path: str, no_truncate: bool = False, output_file: str = None):
    """
    Process a trace file and print unique weave_task_ids with their system messages.
    
    Args:
        file_path: Path to the trace JSON file
        no_truncate: If True, don't truncate long system messages
        output_file: Optional file path to save output
    """
    print(f"📂 Reading trace file: {file_path}\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    raw_logging = data.get("raw_logging_results")
    
    if not raw_logging:
        print("❌ Error: No 'raw_logging_results' section found in the trace file")
        sys.exit(1)
    
    # Dictionary to store unique task_id -> list of system messages
    task_system_messages: Dict[str, Dict[str, int]] = {}
    # Dictionary to count occurrences of each task_id
    task_id_counts: Dict[str, int] = {}
    
    # Process each entry in raw_logging_results (can be dict or list)
    if isinstance(raw_logging, dict):
        entries = raw_logging.values()
        print(f"📋 Processing raw_logging_results as dict with {len(raw_logging)} entries\n")
    elif isinstance(raw_logging, list):
        entries = raw_logging
        print(f"📋 Processing raw_logging_results as list with {len(raw_logging)} entries\n")
    else:
        print(f"❌ Error: Unexpected raw_logging_results format: {type(raw_logging)}")
        sys.exit(1)
    
    for entry_data in entries:
        if isinstance(entry_data, dict):
            # Extract weave_task_id
            weave_task_id = entry_data.get("weave_task_id")
            
            if weave_task_id:
                # Count this occurrence
                task_id_counts[weave_task_id] = task_id_counts.get(weave_task_id, 0) + 1
                
                # Extract messages from inputs
                inputs = entry_data.get("inputs", {})
                messages = inputs.get("messages", [])
                
                # Extract system message
                system_message = extract_system_message(messages)
                
                # Initialize the task's system message counter if not exists
                if weave_task_id not in task_system_messages:
                    task_system_messages[weave_task_id] = {}
                
                # Count this system message for this task
                if system_message is not None:
                    if system_message not in task_system_messages[weave_task_id]:
                        task_system_messages[weave_task_id][system_message] = 0
                    task_system_messages[weave_task_id][system_message] += 1
                else:
                    # Track None as a special key
                    if None not in task_system_messages[weave_task_id]:
                        task_system_messages[weave_task_id][None] = 0
                    task_system_messages[weave_task_id][None] += 1
    
    # Prepare output
    output_lines = []
    
    output_lines.append(f"📊 Found {len(task_system_messages)} unique weave_task_id(s)\n")
    output_lines.append("=" * 80)
    
    for idx, (task_id, system_msg_counts) in enumerate(sorted(task_system_messages.items()), 1):
        count = task_id_counts.get(task_id, 0)
        duplicate_info = f" (appears {count} times)" if count > 1 else f" (appears {count} time)"
        
        output_lines.append(f"\n🔹 Task #{idx}: {task_id}{duplicate_info}")
        output_lines.append("-" * 80)
        
        # Count unique system messages for this task
        num_unique_messages = len(system_msg_counts)
        
        if num_unique_messages == 0:
            output_lines.append("⚠️  No system messages found for this task")
        elif num_unique_messages == 1 and None in system_msg_counts:
            output_lines.append("⚠️  No system messages found for this task (all entries have None)")
        else:
            output_lines.append(f"📝 Found {num_unique_messages} unique system message(s):")
            output_lines.append("")
            
            # Sort by count (most common first)
            sorted_messages = sorted(system_msg_counts.items(), key=lambda x: x[1], reverse=True)
            
            for msg_idx, (system_message, msg_count) in enumerate(sorted_messages, 1):
                if system_message is None:
                    output_lines.append(f"   Message #{msg_idx}: [No system message] (appears {msg_count} times)")
                else:
                    # Truncate very long system messages for readability (unless no_truncate is set)
                    if not no_truncate and len(system_message) > 500:
                        output_lines.append(f"   Message #{msg_idx} (appears {msg_count} times, {len(system_message)} chars, truncated):")
                        output_lines.append(f"   {system_message[:500]}")
                        output_lines.append(f"   ... [truncated {len(system_message) - 500} more characters] ...")
                    else:
                        output_lines.append(f"   Message #{msg_idx} (appears {msg_count} times, {len(system_message)} chars):")
                        # Indent each line of the message
                        for line in system_message.split('\n'):
                            output_lines.append(f"   {line}")
                
                if msg_idx < len(sorted_messages):
                    output_lines.append("")  # Blank line between messages
        
        output_lines.append("-" * 80)
    
    output_lines.append(f"\n✅ Processed {len(task_system_messages)} unique task(s)")
    
    # Print summary statistics
    tasks_with_messages = sum(1 for msg_counts in task_system_messages.values() 
                              if msg_counts and any(msg is not None for msg in msg_counts.keys()))
    tasks_without_messages = len(task_system_messages) - tasks_with_messages
    
    # Count tasks with multiple different system messages
    tasks_with_multiple_messages = sum(1 for msg_counts in task_system_messages.values() 
                                       if len([m for m in msg_counts.keys() if m is not None]) > 1)
    
    # Calculate duplicate statistics
    total_entries = sum(task_id_counts.values())
    tasks_with_duplicates = sum(1 for count in task_id_counts.values() if count > 1)
    total_duplicate_entries = sum(count - 1 for count in task_id_counts.values() if count > 1)
    
    output_lines.append(f"\n📈 Summary:")
    output_lines.append(f"   Total entries in raw_logging_results: {total_entries}")
    output_lines.append(f"   Unique weave_task_ids: {len(task_system_messages)}")
    output_lines.append(f"   Tasks with duplicates: {tasks_with_duplicates}")
    output_lines.append(f"   Total duplicate occurrences: {total_duplicate_entries}")
    output_lines.append(f"   Tasks with system messages: {tasks_with_messages}")
    output_lines.append(f"   Tasks without system messages: {tasks_without_messages}")
    output_lines.append(f"   Tasks with multiple different system messages: {tasks_with_multiple_messages}")
    
    # Show tasks with most duplicates
    if tasks_with_duplicates > 0:
        output_lines.append(f"\n🔄 Tasks with duplicates (sorted by count):")
        sorted_by_count = sorted(task_id_counts.items(), key=lambda x: x[1], reverse=True)
        for task_id, count in sorted_by_count:
            if count > 1:
                output_lines.append(f"   {task_id}: {count} occurrences")
    
    # Show tasks with multiple different system messages
    if tasks_with_multiple_messages > 0:
        output_lines.append(f"\n⚠️  Tasks with multiple different system messages:")
        for task_id, msg_counts in sorted(task_system_messages.items()):
            non_none_messages = {msg: count for msg, count in msg_counts.items() if msg is not None}
            if len(non_none_messages) > 1:
                output_lines.append(f"   {task_id}: {len(non_none_messages)} different system messages")
                for msg, count in sorted(non_none_messages.items(), key=lambda x: x[1], reverse=True):
                    # Show first 100 chars of each message
                    msg_preview = msg[:100] + "..." if len(msg) > 100 else msg
                    msg_preview = msg_preview.replace('\n', ' ')  # Remove newlines for preview
                    output_lines.append(f"      - ({count}x) {msg_preview}")
    
    # Join all output lines
    full_output = "\n".join(output_lines)
    
    # Print to console
    print(full_output)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_output)
            print(f"\n💾 Output saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving to file: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract and print unique weave_task_ids with their system messages from a HAL trace file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific trace file
  python print_task_system_messages.py hal_traces/swe_bench_mini_data/swebench_verified_mini_sweagentgpt520250807_1754592641_UPLOAD.json
  
  # Process with full output (no truncation)
  python print_task_system_messages.py --no-truncate trace_file.json
        """
    )
    
    parser.add_argument(
        "trace_file",
        type=str,
        help="Path to the HAL trace JSON file"
    )
    
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Don't truncate long system messages (show full content)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Save output to a file"
    )
    
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.trace_file).exists():
        print(f"❌ Error: File does not exist: {args.trace_file}")
        sys.exit(1)
    
    process_trace_file(args.trace_file, no_truncate=args.no_truncate, output_file=args.output)


if __name__ == "__main__":
    main()

