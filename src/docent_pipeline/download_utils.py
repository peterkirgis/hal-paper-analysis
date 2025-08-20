import base64
import json
import os
import tempfile
import zipfile
from typing import List, Dict, Any, Optional, Set
import ijson
import re
from typing import Iterator

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from huggingface_hub import HfFileSystem

DEFAULT_PASSWORD = "hal1234"

# Regex for detecting code fences
_CODE_FENCE_RE = re.compile(r"^\s*```[\w+-]*\n(.*?)\n```$", re.DOTALL)

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password and salt."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def decrypt_token_bytes(encrypted_data_b64: str, salt_b64: str, password: str = DEFAULT_PASSWORD) -> bytes:
    """Decrypt base64-encoded encrypted data using password and salt."""
    ct = base64.b64decode(encrypted_data_b64)
    salt = base64.b64decode(salt_b64)
    f = Fernet(_derive_key(password, salt))
    return f.decrypt(ct)


def decrypt_container_to_tempfile(container: Dict[str, Any], password: str = DEFAULT_PASSWORD) -> str:
    """
    Decrypt an encrypted container and write to a temporary file.
    
    Args:
        container: Dictionary containing 'encrypted_data' and 'salt' keys
        password: Password for decryption
        
    Returns:
        Path to temporary file containing decrypted data
    """
    plaintext = decrypt_token_bytes(container["encrypted_data"], container["salt"], password)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tf.write(plaintext)
        tf.flush()
        return tf.name
    finally:
        tf.close()


# HuggingFace client functions
def create_hf_client(repo_id: str = "agent-evals/hal_traces", revision: str = "main") -> Dict[str, Any]:
    """Create HuggingFace client configuration."""
    fs = HfFileSystem()
    repo_path = f"datasets/{repo_id}@{revision}"
    return {
        "repo_id": repo_id,
        "revision": revision,
        "fs": fs,
        "repo_path": repo_path
    }

def list_hf_files(client_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List all files in the repository."""
    return client_config["fs"].ls(client_config["repo_path"], detail=True)

def find_files_by_prefix(client_config: Dict[str, Any], prefix: str, excluded_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """
    Find ZIP files matching a prefix, optionally excluding specific filenames.
    
    Args:
        client_config: HuggingFace client configuration
        prefix: File name prefix to match (e.g., "assistantbench_assistantbench_browser_agent")
        excluded_files: Set of specific filenames to exclude
        
    Returns:
        List of file info dictionaries with 'name', 'size', and 'path' keys
    """
    files = list_hf_files(client_config)
    matching_files = []
    excluded_files = excluded_files or set()
    
    for file_info in files:
        if file_info['name'].endswith('.zip'):
            file_path = file_info['name']
            file_name = file_path.split('/')[-1]
            
            # Check if file matches prefix
            if file_name.lower().startswith(prefix.lower()):
                # Skip if filename is excluded
                if file_name in excluded_files:
                    continue
                
                file_size = file_info.get('size', 0)
                matching_files.append({
                    'name': file_name,
                    'size': file_size,
                    'path': file_path
                })
    
    # Sort by name for consistent processing
    matching_files.sort(key=lambda x: x['name'])
    return matching_files

def open_hf_file(client_config: Dict[str, Any], file_path: str, mode: str = "rb"):
    """Open a file from the repository."""
    full_path = f"datasets/{client_config['repo_id']}@{client_config['revision']}/{file_path}"
    return client_config["fs"].open(full_path, mode)


def extract_contextual_messages_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract user, system, and assistant messages that are contextual to this specific log item"""
    
    def _collapse_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for seg in content:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    parts.append(seg["text"])
            return "\n".join(p for p in parts if p)
        return str(content)
    
    messages = []
    ts = item.get("started_at") or item.get("created_timestamp")
    
    # Check inputs.messages for contextual user/system messages
    input_messages = item.get("inputs", {}).get("messages", [])
    if isinstance(input_messages, list):
        for msg in input_messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                if role in ["user", "system"]:
                    content = _collapse_content_to_text(msg.get("content"))
                    # Include the message if it has content OR if it's a system message
                    if content or role == "system":
                        messages.append({
                            "role": role,
                            "content": content,
                            "tool_calls": [],
                            "ts": ts
                        })
    
    # Also check for assistant messages in inputs.messages (some might be there)
    for msg in input_messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = _collapse_content_to_text(msg.get("content"))
            if content:  # Only include assistant messages with content
                messages.append({
                    "role": "assistant", 
                    "content": content,
                    "tool_calls": msg.get("tool_calls", []),
                    "ts": ts
                })
    
    return messages


def _canon_text(s: str) -> str:
    """Strip code fences, normalize whitespace so identical answers hash the same."""
    if not s:
        return ""
    s = s.strip()
    m = _CODE_FENCE_RE.match(s)
    if m:
        s = m.group(1)
    lines = [ln.rstrip() for ln in s.splitlines()]
    out, prev_blank = [], False
    for ln in lines:
        blank = (ln == "")
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return "\n".join(out).strip()


def _tc_canon(tc):
    """Canonicalize a tool call for deduplication."""
    typ = (tc.get("type") or "function")
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = fn.get("name")
    if fn is None:
        fn = ""
    args = tc.get("arguments", {})
    if isinstance(args, (dict, list)):
        args_canon = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    else:
        args_canon = str(args)
    return (typ, str(fn), args_canon)


def _msg_fingerprint(m: Dict[str, Any]) -> tuple:
    """Fingerprint of a normalized chat message."""
    role = m.get("role")
    content = _canon_text(m.get("content") or "")
    tool_calls = tuple(sorted(_tc_canon(tc) for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)))
    return (role, content, tool_calls)


def dedupe_messages(messages, mode: str = "consecutive"):
    """Drop duplicate messages."""
    out = []
    last_fp = None
    seen = set()
    for m in messages:
        fp = _msg_fingerprint(m)
        if mode == "consecutive":
            if fp == last_fp:
                continue
            last_fp = fp
        else:  # global
            if fp in seen:
                continue
            seen.add(fp)
        out.append(m)
    return out


def _collapse_content_to_text(content) -> str:
    """Collapse various content formats to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                parts.append(seg["text"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _map_role(raw_type, raw__type):
    """Map various role formats to standard roles."""
    t = (raw_type or raw__type or "").lower()
    if t in ("ai", "assistant"): 
        return "assistant"
    if t in ("human", "user"):   
        return "user"
    if t == "system":            
        return "system"
    return None


def normalize_weave_log_item(item: Dict[str, Any]):
    """Prefer the 'inputs.raw' single-message rows."""
    raw = item.get("inputs", {}).get("raw")
    if not isinstance(raw, dict):
        return None
    role = _map_role(raw.get("type"), raw.get("_type"))
    if role is None:
        return None
    content_text = _collapse_content_to_text(raw.get("content"))
    tool_calls = raw.get("tool_calls") or []
    ts = item.get("started_at") or item.get("created_timestamp")
    return {
        "role": role,
        "content": content_text,
        "tool_calls": [
            {
                "id": tc.get("id"),
                "function": (tc.get("name") or (tc.get("function") or {}).get("name")),
                "arguments": (tc.get("args")  or (tc.get("function") or {}).get("arguments", {})),
                "type": tc.get("type") or "function",
            }
            for tc in tool_calls if isinstance(tc, dict)
        ],
        "ts": ts,
    }


def normalize_assistant_output(item: Dict[str, Any]):
    """Pick up OpenAI-style assistant messages from the 'output' side."""
    out = item.get("output") or {}
    choices = out.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not content:
        return None
    return {
        "role": "assistant",
        "content": content if isinstance(content, str) else _collapse_content_to_text(content),
        "tool_calls": [],
        "ts": item.get("ended_at") or item.get("created_timestamp"),
    }


def extract_eval_results_for_task(tid: str, task_id_list: List[str], raw_eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract evaluation results for a specific task based on its index in the task list.
    
    Args:
        tid: The task ID to find eval results for
        task_id_list: Ordered list of task IDs as they appear in the eval results
        raw_eval_results: Dictionary containing raw evaluation results with list values
        
    Returns:
        Dictionary with eval results for this task, or empty dict if not found
    """
    if not task_id_list or not raw_eval_results:
        return {}
        
    try:
        task_index = task_id_list.index(tid)
    except ValueError:
        # Task ID not found in list
        return {}
    
    task_eval_data = {}
    
    # Extract data from any list-type fields in raw_eval_results
    for key, value in raw_eval_results.items():
        if isinstance(value, list) and task_index < len(value):
            task_eval_data[key] = value[task_index]
        elif not isinstance(value, list):
            # Non-list data is global/metadata, include it
            task_eval_data[f"global_{key}"] = value
    
    return task_eval_data

def build_agent_run_from_bucket(tid: str, bucket, model, eval_blob=None, task_eval_results=None):
    """Build an agent run from a bucket of messages."""
    # Include messages with empty content (important for system/user messages)
    msgs_sorted = sorted([m for m in bucket if m.get("role")], 
                         key=lambda m: m.get("ts") or "")
    
    # Strip ts from final output
    final_messages = []
    for m in msgs_sorted:
        mm = {"role": m["role"], "content": m.get("content") or ""}  # Default to empty string
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        final_messages.append(mm)

    agent_run = {
        "weave_task_id": tid,
        "model": model,
        "messages": final_messages,
    }
    if eval_blob:
        agent_run["eval"] = {
            "reward": eval_blob.get("reward"),
            "task": eval_blob.get("task", eval_blob.get("info", {})) or {},
        }
    
    # Add task-specific eval results if available
    if task_eval_results:
        if "eval" not in agent_run:
            agent_run["eval"] = {}
        agent_run["eval"]["raw_results"] = task_eval_results
    
    agent_run["messages"] = dedupe_messages(agent_run["messages"])
    return agent_run

def stream_agent_runs_by_task(
    client_config: Dict[str, Any],
    zip_name: str,
    member_name: Optional[str] = None,
    require_model: Optional[str] = None,
    include_eval: bool = False,
    include_eval_results: bool = True,
    limit: Optional[int] = None,
    aggregate_all: bool = True,
    password: str = "hal1234"
) -> Iterator[tuple[str, Dict[str, Any]]]:
    """
    Stream agent runs from a ZIP file in the Hugging Face repository.
    
    Args:
        client_config: HuggingFace client configuration
        zip_name: Name of the ZIP file to process
        member_name: Specific member to extract (if None, uses first non-directory)
        require_model: Only process items with this model
        include_eval: Include evaluation data in results
        include_eval_results: Include raw evaluation results (scores, answers) mapped by task index
        limit: Maximum number of tasks to process
        aggregate_all: Whether to aggregate all messages per task
        password: Decryption password
        
    Yields:
        Tuples of (task_id, agent_run_dict)
    """

    # Open ZIP from HF
    hf_file = open_hf_file(client_config, zip_name, "rb")
    zf = zipfile.ZipFile(hf_file)
    
    if member_name:
        info = zf.getinfo(member_name)
    else:
        info = next(i for i in zf.infolist() if not i.filename.endswith("/"))
    
    try:
        with zf.open(info, "r") as member:
            container = json.load(member)
    finally:
        try:
            zf.close()
        except:
            pass
        try:
            hf_file.close()
        except:
            pass

    plaintext_path = decrypt_container_to_tempfile(container, password)

    tasks_bucket = {}
    model_by_tid = {}
    eval_by_tid = {}
    raw_eval_results = {}
    task_id_list = []  # Keep track of task order for index mapping
    produced = 0
    
    # First pass: Extract raw_eval_results if requested
    if include_eval_results:
        try:
            with open(plaintext_path, "r") as f:
                data = json.load(f)
                raw_eval_results = data.get('raw_eval_results', {})
                # Get ordered list of task IDs from results section
                results = data.get('results', {})
                if 'successful_tasks' in results and 'failed_tasks' in results:
                    # Combine successful and failed tasks in the order they appear
                    task_id_list = results['successful_tasks'] + results['failed_tasks']
        except Exception as e:
            print(f"Warning: Could not extract raw_eval_results: {e}")
            raw_eval_results = {}
    
    try:
        with open(plaintext_path, "rb") as f:
            for item in ijson.items(f, "raw_logging_results.item"):
                tid = item.get("weave_task_id")
                if not tid:
                    continue

                # Model discovery
                mdl = (item.get("inputs", {}) or {}).get("model") or (item.get("output", {}) or {}).get("model")
                if mdl:
                    model_by_tid[tid] = mdl

                # Eval blob
                if ("reward" in item) or ("task" in item) or ("info" in item):
                    eval_by_tid[tid] = item
                    continue

                # Filter by model if requested
                if require_model and ((item.get("inputs", {}) or {}).get("model") != require_model):
                    continue

                # Start bucket
                if tid not in tasks_bucket:
                    tasks_bucket[tid] = []

                # Extract ALL contextual messages from this item
                contextual_messages = extract_contextual_messages_from_item(item)
                tasks_bucket[tid].extend(contextual_messages)

                # Normalize individual messages from inputs.raw
                nm = normalize_weave_log_item(item)
                if nm:
                    tasks_bucket[tid].append(nm)

                # Assistant messages from outputs
                ao = normalize_assistant_output(item)
                if ao:
                    tasks_bucket[tid].append(ao)

        # Emit once per task
        for tid, bucket in tasks_bucket.items():
            # Extract task-specific eval results if available
            task_eval_results = None
            if include_eval_results and raw_eval_results:
                task_eval_results = extract_eval_results_for_task(tid, task_id_list, raw_eval_results)
            
            run = build_agent_run_from_bucket(
                tid=tid,
                bucket=bucket,
                model=model_by_tid.get(tid),
                eval_blob=eval_by_tid.get(tid) if include_eval else None,
                task_eval_results=task_eval_results,
            )
            yield tid, run
            produced += 1
            if limit and produced >= limit:
                break
    finally:
        try:
            os.remove(plaintext_path)
        except OSError:
            pass