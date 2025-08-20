
from typing import Dict, Any, List, Tuple

from docent.data_models.chat import parse_chat_message


def normalize_message_for_docent(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize AssistantBench message format to docent-compatible format"""
    normalized = msg.copy()
    
    # Fix tool_calls format if present
    if 'tool_calls' in normalized and normalized['tool_calls']:
        fixed_tool_calls = []
        for tool_call in normalized['tool_calls']:
            fixed_tc = {}
            
            # Required fields for docent ToolCall
            fixed_tc['id'] = tool_call.get('id', f"tool_{len(fixed_tool_calls)}")  # Ensure ID exists
            fixed_tc['type'] = 'function'  # docent expects 'function'
            
            # Function name - should be a string, not a dict
            if isinstance(tool_call.get('function'), str):
                fixed_tc['function'] = tool_call['function']
            elif isinstance(tool_call.get('function'), dict):
                # If it's a dict with 'name', use that
                fixed_tc['function'] = tool_call['function'].get('name', 'unknown_function')
            else:
                fixed_tc['function'] = 'unknown_function'
            
            # Arguments - should be a dict at the top level
            if 'arguments' in tool_call:
                fixed_tc['arguments'] = tool_call['arguments']
            elif isinstance(tool_call.get('function'), dict) and 'arguments' in tool_call['function']:
                fixed_tc['arguments'] = tool_call['function']['arguments']
            else:
                fixed_tc['arguments'] = {}
            
            fixed_tool_calls.append(fixed_tc)
        
        normalized['tool_calls'] = fixed_tool_calls
    
    return normalized


def convert_to_docent_messages(loaded_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert AssistantBench results to docent ChatMessage format.
    
    Args:
        loaded_results: Dictionary mapping zip_name -> tasks -> agent_runs
        
    Returns:
        Tuple of (converted_results, conversion_stats)
    """
    docent_results = {}
    
    print("Converting to docent ChatMessage format...")
    conversion_stats = {
        'total_messages': 0,
        'successful': 0,
        'failed': 0,
        'failed_tasks': [],
        'error_types': {}
    }
    
    for zip_name, tasks in loaded_results.items():
        if "error" in tasks:
            docent_results[zip_name] = tasks  # Keep error info
            continue
            
        docent_results[zip_name] = {}
        
        for task_id, agent_run in tasks.items():
            messages = agent_run.get('messages', [])
            conversion_stats['total_messages'] += len(messages)
            
            # Convert each message to docent ChatMessage
            docent_messages = []
            task_failed_count = 0
            
            for i, msg in enumerate(messages):
                try:
                    # Normalize the message format first
                    normalized_msg = normalize_message_for_docent(msg)
                    
                    # Parse using docent's parse_chat_message function
                    chat_msg = parse_chat_message(normalized_msg)
                    docent_messages.append(chat_msg)
                    conversion_stats['successful'] += 1
                    
                except Exception as e:
                    error_type = type(e).__name__
                    conversion_stats['error_types'][error_type] = conversion_stats['error_types'].get(error_type, 0) + 1
                    
                    # Only print first few errors to avoid spam
                    if conversion_stats['failed'] < 5:
                        print(f"Warning: Failed to parse message {i} in task {task_id[:12]}...: {e}")
                    
                    conversion_stats['failed'] += 1
                    task_failed_count += 1
                    continue
            
            if task_failed_count > 0:
                conversion_stats['failed_tasks'].append((task_id, task_failed_count))
            
            # Store the converted data
            docent_results[zip_name][task_id] = {
                'weave_task_id': agent_run.get('weave_task_id'),
                'model': agent_run.get('model'),
                'eval': agent_run.get('eval'),
                'original_message_count': len(messages),
                'docent_message_count': len(docent_messages),
                'failed_message_count': task_failed_count,
                'docent_messages': docent_messages,  # These are now ChatMessage objects
                'original_messages': messages  # Keep original for reference
            }
    
    return docent_results, conversion_stats