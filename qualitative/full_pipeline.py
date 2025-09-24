from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, List, Optional, Iterator, Set, Tuple

import ijson
from tqdm import tqdm
from dotenv import load_dotenv

# Crypto imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# HF FS
from huggingface_hub import HfFileSystem

# Docent SDK
from requests.exceptions import ConnectionError, Timeout, HTTPError
from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message

# ──────────────────────────────────────────────────────────────────────────────
# Defaults / Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PASSWORD = "hal1234"
DEFAULT_REPO_ID = "agent-evals/hal_traces"
DEFAULT_REVISION = "main"


BENCHMARK_AGENT_PREFIX = "scicode_scicode_tool_calling_agent"
# BENCHMARK_AGENT_PREFIX = "assistantbench_assistantbench_browser_agent"
# BENCHMARK_AGENT_PREFIX = "taubench_airline_taubench_fewshot"  
# BENCHMARK_AGENT_PREFIX = "corebench_hard_coreagent"

DEFAULT_TASK_LIMIT: Optional[int] = None  # Keep all tasks by default
if DEFAULT_TASK_LIMIT is None:
    DEFAULT_COLLECTION_NAME = f"{BENCHMARK_AGENT_PREFIX}_all_tasks"
else:
    DEFAULT_COLLECTION_NAME = f"{BENCHMARK_AGENT_PREFIX}_{DEFAULT_TASK_LIMIT}_tasks"
DEFAULT_OUTPUT_FILE = f"{BENCHMARK_AGENT_PREFIX}_results.json"

EXCLUDED_FILES = [
    'assistantbench_assistantbench_browser_agent_claude37sonnet20250219_low_1748711087_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_deepseekr1_1755121049_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_gemini20flash_1746393958_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_gpt5_1754598271_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_o320250416_1746376643_UPLOAD.zip',
    'assistantbench_assistantbench_browser_agent_o4mini20250416_1746227177_UPLOAD.zip',
    "taubench_airline_taubench_fewshot_o3mini20250131_high_1744743428_UPLOAD.zip",
    "taubench_airline_taubench_fewshot_o320250403_1744728447_UPLOAD.zip",
    "scicode_scicode_tool_calling_agent_o4mini20250416_1745267192_UPLOAD.zip",
    "corebench_hard_coreagent_1744839552_UPLOAD.zip",
    "corebench_hard_coreagent_1745118908_UPLOAD.zip",
    "corebench_hard_coreagent_1744922343_UPLOAD.zip",
    "corebench_hard_coreagent_1744922265_UPLOAD.zip",
    "corebench_hard_coreagentdeepseekv31_1755793007_UPLOAD.zip",
    "corebench_hard_coreagent_1754539776_UPLOAD.zip",
    "corebench_hard_coreagent_1754492673_UPLOAD.zip",
    "corebench_hard_coreagentdeepseekr1_1757604883_UPLOAD.zip",
    "corebench_hard_coreagentclaudesonnet4_1755796611_UPLOAD.zip",
    "corebench_hard_coreagentclaudesonnet4high_1755814601_UPLOAD.zip"
    
]

_CODE_FENCE_RE = re.compile(r"^\s*```[\w+-]*\n(.*?)\n```$", re.DOTALL)

# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def decrypt_token_bytes(encrypted_data_b64: str, salt_b64: str, password: str = DEFAULT_PASSWORD) -> bytes:
    ct = base64.b64decode(encrypted_data_b64)
    salt = base64.b64decode(salt_b64)
    f = Fernet(_derive_key(password, salt))
    return f.decrypt(ct)


def decrypt_container_to_tempfile(container: Dict[str, Any], password: str = DEFAULT_PASSWORD) -> str:
    plaintext = decrypt_token_bytes(container["encrypted_data"], container["salt"], password)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tf.write(plaintext)
        tf.flush()
        return tf.name
    finally:
        tf.close()


def _canon_text(s: str) -> str:
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


def _tc_canon(tc: Dict[str, Any]) -> Tuple[str, str, str]:
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


def _msg_fingerprint(m: Dict[str, Any]) -> Tuple[Any, ...]:
    role = m.get("role")
    content = _canon_text(m.get("content") or "")
    tool_calls = tuple(sorted(_tc_canon(tc) for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)))
    return (role, content, tool_calls)


def _msg_content_fingerprint(m: Dict[str, Any]) -> Tuple[Any, ...]:
    content = _canon_text(m.get("content") or "")
    tool_calls = tuple(sorted(_tc_canon(tc) for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)))
    return (content, tool_calls)


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


def _map_role(raw_type, raw__type):
    t = (raw_type or raw__type or "").lower()
    if t in ("ai", "assistant"):
        return "assistant"
    if t in ("human", "user"):
        return "user"
    if t == "system":
        return "system"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# HF client
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HFClient:
    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    fs: HfFileSystem = field(default_factory=HfFileSystem)

    @property
    def repo_path(self) -> str:
        return f"datasets/{self.repo_id}@{self.revision}"

    def list_files(self) -> List[Dict[str, Any]]:
        return self.fs.ls(self.repo_path, detail=True)

    def open(self, file_path: str, mode: str = "rb"):
        full_path = f"{self.repo_path}/{file_path}"
        return self.fs.open(full_path, mode)

    def find_files_by_prefix(self, prefix: str, excluded_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        files = self.list_files()
        matching: List[Dict[str, Any]] = []
        excluded_files = excluded_files or set()
        for file_info in files:
            if file_info['name'].endswith('.zip'):
                file_path = file_info['name']
                file_name = file_path.split('/')[-1]
                if file_name.lower().startswith(prefix.lower()) and file_name not in excluded_files:
                    matching.append({'name': file_name, 'size': file_info.get('size', 0), 'path': file_path})
        matching.sort(key=lambda x: x['name'])
        return matching

# ──────────────────────────────────────────────────────────────────────────────
# Weave → AgentRun extraction
# ──────────────────────────────────────────────────────────────────────────────

class WeaveExtractor:
    @staticmethod
    def normalize_weave_log_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                    "arguments": (tc.get("args") or (tc.get("function") or {}).get("arguments", {})),
                    "type": tc.get("type") or "function",
                }
                for tc in tool_calls if isinstance(tc, dict)
            ],
            "ts": ts,
        }

    @staticmethod
    def normalize_assistant_output(item: Dict[str, Any], primary_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        out = item.get("output") or {}
        choices = out.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            return None
        current_model = (item.get("inputs", {}) or {}).get("model") or out.get("model")
        role = "assistant"
        if (primary_model and current_model and current_model.lower() != primary_model.lower() and current_model.lower().startswith('gpt-4o')):
            role = "user"
        return {
            "role": role,
            "content": content if isinstance(content, str) else _collapse_content_to_text(content),
            "tool_calls": [],
            "ts": item.get("ended_at") or item.get("created_timestamp"),
        }

    @staticmethod
    def extract_contextual_messages_from_item(item: Dict[str, Any], primary_model: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        ts = item.get("started_at") or item.get("created_timestamp")
        current_model = (item.get("inputs", {}) or {}).get("model")
        input_messages = item.get("inputs", {}).get("messages", [])
        if isinstance(input_messages, list):
            for msg in input_messages:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    if role in ["user", "system"]:
                        content = _collapse_content_to_text(msg.get("content"))
                        if content or role == "system":
                            messages.append({"role": role, "content": content, "tool_calls": [], "ts": ts})
        for msg in input_messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = _collapse_content_to_text(msg.get("content"))
                if content:
                    role = "assistant"
                    if (primary_model and current_model and current_model.lower() != primary_model.lower() and current_model.lower().startswith('gpt-4o')):
                        role = "user"
                    messages.append({"role": role, "content": content, "tool_calls": msg.get("tool_calls", []), "ts": ts})
        return messages

    @staticmethod
    def dedupe_messages(messages: List[Dict[str, Any]], mode: str = "cross_role") -> List[Dict[str, Any]]:
        if mode == "consecutive":
            out: List[Dict[str, Any]] = []
            last_fp = None
            for m in messages:
                fp = _msg_fingerprint(m)
                if fp == last_fp:
                    continue
                last_fp = fp
                out.append(m)
            return out
        elif mode == "global":
            out = []
            seen = set()
            for m in messages:
                fp = _msg_fingerprint(m)
                if fp in seen:
                    continue
                seen.add(fp)
                out.append(m)
            return out
        elif mode == "cross_role":
            out = []
            seen_content = set()
            for m in messages:
                content_fp = _msg_content_fingerprint(m)
                if content_fp in seen_content:
                    continue
                seen_content.add(content_fp)
                out.append(m)
            return out
        else:
            raise ValueError(f"Unknown deduplication mode: {mode}")

    @staticmethod
    def truncate_before_evaluator_message(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Truncate conversation to the last message before a system message containing
        the evaluator prompt about giving up on tasks.
        """
        evaluator_text = "You are an evaluator determining if an agent's answer indicates giving up on a task."
        
        for i, message in enumerate(messages):
            if (message.get("role") == "system" and 
                isinstance(message.get("content"), str) and 
                evaluator_text in message["content"]):
                # Found the evaluator message, truncate here
                print(f"   🔄 Truncating conversation at message {i} (evaluator system message found)")
                return messages[:i]
        
        # No evaluator message found, return all messages
        return messages
    
    @staticmethod
    def _normalize_msg_for_cmp(msg: dict) -> dict:
        """Drop None values so dict equality doesn't break on None-vs-missing."""
        if not isinstance(msg, dict):
            return msg
        return {k: v for k, v in msg.items() if v is not None}

    @staticmethod
    def _messages_are_prefix(smaller: list, larger: list) -> bool:
        """Return True iff `smaller` is an ordered prefix of `larger` under normalized comparison."""
        if len(smaller) > len(larger):
            return False
        for i, m in enumerate(smaller):
            if WeaveExtractor._normalize_msg_for_cmp(m) != WeaveExtractor._normalize_msg_for_cmp(larger[i]):
                return False
        return True

    @staticmethod
    def _sanity_check_task_logs(task_entries: List[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Ensure the largest inputs.messages contains all other logs as ordered subsets/prefixes.
        Returns (ok, largest_entry).
        """
        # Keep only entries that actually have message lists
        def _get_msgs(entry):
            return ((entry.get("inputs") or {}).get("messages") or [])
        entries = [e for e in task_entries if isinstance(_get_msgs(e), list)]
        if not entries:
            return True, None

        entries_sorted = sorted(entries, key=lambda e: len(_get_msgs(e)), reverse=True)
        largest = entries_sorted[0]
        largest_msgs = _get_msgs(largest)

        for e in entries_sorted[1:]:
            msgs_small = _get_msgs(e)
            if len(msgs_small) == len(largest_msgs):
                # Same length ⇒ must be element-wise equal
                for i, m in enumerate(msgs_small):
                    if WeaveExtractor._normalize_msg_for_cmp(m) != WeaveExtractor._normalize_msg_for_cmp(largest_msgs[i]):
                        return False, largest
            else:
                if not WeaveExtractor._messages_are_prefix(msgs_small, largest_msgs):
                    return False, largest
        return True, largest


    @staticmethod
    def _normalize_msg_for_cmp(msg: dict) -> dict:
        """Drop None values so dict equality doesn't break on None-vs-missing."""
        if not isinstance(msg, dict):
            return msg
        return {k: v for k, v in msg.items() if v is not None}

    @staticmethod
    def _messages_are_prefix(smaller: list, larger: list) -> bool:
        """Return True iff `smaller` is an ordered prefix of `larger` under normalized comparison."""
        if len(smaller) > len(larger):
            return False
        for i, m in enumerate(smaller):
            if WeaveExtractor._normalize_msg_for_cmp(m) != WeaveExtractor._normalize_msg_for_cmp(larger[i]):
                return False
        return True

    @staticmethod
    def _sanity_check_task_logs(task_entries: List[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Ensure the largest inputs.messages contains all other logs as ordered subsets/prefixes.
        Returns (ok, largest_entry).
        """
        # Keep only entries that actually have message lists
        def _get_msgs(entry):
            return ((entry.get("inputs") or {}).get("messages") or [])
        entries = [e for e in task_entries if isinstance(_get_msgs(e), list)]
        if not entries:
            return True, None

        entries_sorted = sorted(entries, key=lambda e: len(_get_msgs(e)), reverse=True)
        largest = entries_sorted[0]
        largest_msgs = _get_msgs(largest)

        for e in entries_sorted[1:]:
            msgs_small = _get_msgs(e)
            if len(msgs_small) == len(largest_msgs):
                # Same length ⇒ must be element-wise equal
                for i, m in enumerate(msgs_small):
                    if WeaveExtractor._normalize_msg_for_cmp(m) != WeaveExtractor._normalize_msg_for_cmp(largest_msgs[i]):
                        return False, largest
            else:
                if not WeaveExtractor._messages_are_prefix(msgs_small, largest_msgs):
                    return False, largest
        return True, largest


    @staticmethod
    def fix_benchmark_specific_issues(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        fixed: List[Dict[str, Any]] = []
        for i, message in enumerate(messages):
            content = message.get("content", "")
            if not isinstance(content, str):
                fixed.append(message)
                continue

            # Parse "Calling tools: [...]" JSON into tool_calls
            if "Calling tools:" in content and message.get("role") == "assistant":
                calling_start = content.find("Calling tools:")
                if calling_start != -1:
                    bracket_start = content.find("[", calling_start)
                    if bracket_start != -1:
                        bracket_count = 0
                        bracket_end = -1
                        for j in range(bracket_start, len(content)):
                            if content[j] == '[':
                                bracket_count += 1
                            elif content[j] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    bracket_end = j + 1
                                    break
                        if bracket_end != -1:
                            tools_json = content[bracket_start:bracket_end]
                            parsed_tools = None
                            # First try strict JSON
                            try:
                                parsed_tools = json.loads(tools_json)
                            except json.JSONDecodeError:
                                # Many logs serialize this array using Python repr (single quotes).
                                # Fall back to a safe Python-literal parser.
                                try:
                                    import ast
                                    parsed_tools = ast.literal_eval(tools_json)
                                except Exception:
                                    parsed_tools = None
                            if parsed_tools:
                                fm = message.copy()
                                existing = fm.get("tool_calls", []) or []
                                for tool in parsed_tools:
                                    if isinstance(tool, dict):
                                        func = tool.get("function", {})
                                        existing.append({
                                            "id": tool.get("id", f"call_{len(existing)}"),
                                            "type": tool.get("type", "function"),
                                            # keep dict if present; Docent normalization handles string/dict
                                            "function": func if func else tool.get("function", {}),
                                        })
                                # Remove the entire "Calling tools: [...]" block from content
                                cleaned = (content[:calling_start] + content[bracket_end:]).strip()
                                fm["content"], fm["tool_calls"] = cleaned, existing
                                fixed.append(fm)
                                continue

            # TauBench embedded "assistant: None\ntool: {...}" → tool_calls
            if (message.get("role") == "assistant" and "\nassistant:" in content and "\ntool:" in content):
                tool_pattern = r'\ntool:\s*(\{.*?\})\s*(?=\n|$)'
                tool_matches = re.findall(tool_pattern, content, re.DOTALL)
                if tool_matches:
                    fm = message.copy()
                    tool_calls = fm.get("tool_calls", []) or []
                    for tjson in tool_matches:
                        try:
                            tool_data = json.loads(tjson.strip())
                            tool_calls.append({
                                "id": f"tool_{len(tool_calls)}",
                                "type": "function",
                                "function": "tool_use",
                                "arguments": tool_data,
                            })
                        except json.JSONDecodeError:
                            pass
                    cleaned = re.sub(r'\nassistant:\s*None\s*', '', content)
                    cleaned = re.sub(r'\ntool:\s*\{.*?\}\s*', '', cleaned, flags=re.DOTALL)
                    cleaned = re.sub(r'^\nassistant:\s*', '', cleaned.strip())
                    fm["content"], fm["tool_calls"] = cleaned.strip(), tool_calls
                    fixed.append(fm)
                    continue

            # TauBench: assistant content "None" but has tool_calls → keep calls, drop content
            if (message.get("role") == "assistant" and content.strip().lower() == "none" and message.get("tool_calls")):
                fm = message.copy()
                fm["content"] = ""
                fixed.append(fm)
                continue

            # Role fix for specific greeting
            if content.strip() == "Hi! How can I help you today?" and message.get("role") == "user":
                fm = message.copy(); fm["role"] = "assistant"
                fixed.append(fm); continue

            # AssistantBench spurious items
            if content.strip() == "What is the capital of France? Respond with a single word." and message.get("role") == "user":
                continue
            if content.strip() == "Paris" and message.get("role") == "assistant":
                continue
            if content.strip() == "None" and message.get("role") == "assistant":
                continue

            # Remove spurious Apple/iPhone tool calls
            tcs = message.get("tool_calls", [])
            if tcs and isinstance(tcs, list):
                cleaned_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        if isinstance(func, dict):
                            args = func.get("arguments", "")
                            name = func.get("name", "")
                            if (name == "AgentOutput" and isinstance(args, str) and "Apple" in args and "iPhone" in args and "click_element" in args and "Best Buy" in args):
                                continue
                    cleaned_tcs.append(tc)
                if len(cleaned_tcs) != len(tcs):
                    fm = message.copy(); fm["tool_calls"] = cleaned_tcs
                    fixed.append(fm); continue

            # ###STOP### handling
            if "###STOP###" in content:
                cleaned_content = content.replace("###STOP###", "").strip()
                is_last = (i == len(messages) - 1)
                if is_last and message.get("role") == "assistant":
                    if not cleaned_content:
                        continue
                    fm = message.copy(); fm["role"] = "user"; fm["content"] = cleaned_content
                    fixed.append(fm)
                else:
                    if cleaned_content:
                        fm = message.copy(); fm["content"] = cleaned_content
                        fixed.append(fm)
            else:
                fixed.append(message)
        return fixed

    @staticmethod
    def build_agent_run_from_bucket(tid: str, bucket: List[Dict[str, Any]], model: Optional[str], eval_blob: Optional[Dict[str, Any]] = None, task_eval_results: Optional[Dict[str, Any]] = None, config_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        msgs_sorted = sorted([m for m in bucket if m.get("role")], key=lambda m: m.get("ts") or "")
        final_messages: List[Dict[str, Any]] = []
        for m in msgs_sorted:
            mm = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = m["tool_calls"]
            final_messages.append(mm)
        agent_run: Dict[str, Any] = {"weave_task_id": tid, "model": model, "messages": final_messages}
        if eval_blob:
            agent_run["eval"] = {"reward": eval_blob.get("reward"), "task": eval_blob.get("task", eval_blob.get("info", {})) or {}}
        if task_eval_results:
            if "eval" not in agent_run:
                agent_run["eval"] = {}
            agent_run["eval"]["raw_results"] = task_eval_results
        if config_metadata:
            agent_run["config_metadata"] = config_metadata
        agent_run["messages"] = WeaveExtractor.dedupe_messages(agent_run["messages"], mode="cross_role")
        agent_run["messages"] = WeaveExtractor.fix_benchmark_specific_issues(agent_run["messages"]) 
        agent_run["messages"] = WeaveExtractor.truncate_before_evaluator_message(agent_run["messages"])
        return agent_run

    @staticmethod
    def _extract_model_from_filename(zip_name: str) -> Optional[str]:
        base_name = zip_name.split('/')[-1].replace('.zip', '').replace('_UPLOAD', '')
        parts = base_name.split('_')
        for part in parts:
            pl = part.lower()
            if 'gpt-4o' in pl or 'gpt4o' in pl:
                return 'gpt-4o'
            elif 'gpt4' in pl:
                return 'gpt-4'
            elif 'deepseekr1' in pl or ('deepseekai' in pl and 'deepseekr1' in pl):
                return 'deepseek-r1'
            elif 'deepseekv3' in pl or ('deepseekai' in pl and 'deepseekv3' in pl):
                return 'deepseek-v3'
            elif 'claude' in pl:
                return 'claude-3.5-sonnet'
            elif 'gemini' in pl:
                return 'gemini-2.0-flash'
            elif 'o1' in pl:
                return 'o1'
            elif 'o3' in pl:
                return 'o3'
        return None

    @staticmethod
    def extract_task_order_from_raw_logging(plaintext_path: str) -> List[str]:
        task_order: List[str] = []
        seen: Set[str] = set()
        try:
            with open(plaintext_path, "rb") as f:
                for item in ijson.items(f, "raw_logging_results.item"):
                    tid = item.get("weave_task_id") or (item.get("attributes") or {}).get("weave_task_id")
                    if tid and tid not in seen:
                        task_order.append(tid)
                        seen.add(tid)
        except Exception as e:
            print(f"Warning: Could not extract task order from raw logging: {e}")
            return []
        return task_order

    @staticmethod
    def extract_eval_results_for_task(task_id: str, task_id_list: List[str], raw_eval_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result: Dict[str, Any] = {}
        if 'results' in raw_eval_results:
            results = raw_eval_results['results']
            successful_tasks = results.get('successful_tasks', [])
            task_id_str = str(task_id)
            result['is_successful'] = task_id_str in [str(t) for t in successful_tasks]
            result['successful_tasks'] = successful_tasks
            result['failed_tasks'] = results.get('failed_tasks', [])
        if 'details' in raw_eval_results:
            details = raw_eval_results['details']
            task_id_str = str(task_id)
            if task_id_str in details:
                successful_subtasks = details[task_id_str]
                if isinstance(successful_subtasks, list):
                    if len(successful_subtasks) > 0:
                        result['has_successful_subtasks'] = True
                    else:
                        result['has_successful_subtasks'] = False
                    result['successful_subtasks'] = successful_subtasks
        if 'scores' in raw_eval_results:
            try:
                search_list = (raw_eval_results.get('actual_task_order') or raw_eval_results.get('original_task_list') or task_id_list)
                if search_list:
                    task_id_str = str(task_id)
                    task_index = None
                    for i, tid in enumerate(search_list):
                        if str(tid) == task_id_str:
                            task_index = i
                            break
                    if task_index is not None:
                        scores = raw_eval_results['scores']
                        if isinstance(scores, list) and task_index < len(scores):
                            reverse_index = len(scores) - 1 - task_index
                            result['score'] = scores[reverse_index]
                            result['task_index_used'] = task_index
                            result['reverse_index_used'] = reverse_index
                            result['task_order_source'] = 'actual_task_order' if 'actual_task_order' in raw_eval_results else 'original_task_list'
                        answers = raw_eval_results.get('answers', [])
                        if isinstance(answers, list) and task_index < len(answers):
                            reverse_index = len(answers) - 1 - task_index
                            result['answer'] = answers[reverse_index]
                        has_answer = raw_eval_results.get('has_answer', [])
                        if isinstance(has_answer, list) and task_index < len(has_answer):
                            reverse_index = len(has_answer) - 1 - task_index
                            result['has_answer'] = has_answer[reverse_index]
                    else:
                        print(f"Warning: Task {task_id} not found in task order list")
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not extract score for task {task_id}: {e}")
        try:
            if isinstance(raw_eval_results, list):
                task_index = task_id_list.index(task_id) if task_id_list and task_id in task_id_list else None
                if task_index is not None and task_index < len(raw_eval_results):
                    indexed_result = raw_eval_results[task_index]
                    if indexed_result:
                        result.update(indexed_result)
        except (ValueError, IndexError, TypeError):
            pass
        return result if result else None
    
    @staticmethod
    def stream_agent_runs_by_task(
        client: HFClient,
        zip_name: str,
        member_name: Optional[str] = None,
        require_model: Optional[str] = None,
        include_eval: bool = False,
        include_eval_results: bool = True,
        limit: Optional[int] = None,
        aggregate_all: bool = True,
        password: str = DEFAULT_PASSWORD
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        # Open and decrypt the container (unchanged)
        hf_file = client.open(zip_name, "rb")
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
            except Exception:
                pass
            try:
                hf_file.close()
            except Exception:
                pass
        plaintext_path = decrypt_container_to_tempfile(container, password)

        # State we keep (mostly unchanged)
        model_by_tid: Dict[str, str] = {}
        eval_by_tid: Dict[str, Any] = {}
        raw_eval_results: Dict[str, Any] = {}
        task_id_list: List[str] = []
        config_metadata: Dict[str, Any] = {}
        produced = 0

        # For the new strategy we collect ALL entries per task_id first
        # but keep memory modest: we keep full items for each task_id, which is
        # usually manageable; if not, swap to storing only inputs/messages for non-largest.
        entries_by_tid: Dict[str, List[Dict[str, Any]]] = {}

        # Optionally extract eval + config metadata up front (unchanged)
        if include_eval_results:
            try:
                with open(plaintext_path, "r") as f:
                    data = json.load(f)
                    raw_eval_results = data.get('raw_eval_results', {})
                    results = data.get('results', {})
                    if 'successful_tasks' in results and 'failed_tasks' in results:
                        original_task_list = results['successful_tasks'] + results['failed_tasks']
                        raw_eval_results['original_task_list'] = original_task_list
                        raw_eval_results['results'] = results
                    actual_task_order = WeaveExtractor.extract_task_order_from_raw_logging(plaintext_path)
                    if actual_task_order:
                        raw_eval_results['actual_task_order'] = actual_task_order
                        print(f"✅ Extracted actual task order: {len(actual_task_order)} tasks")
                    else:
                        print("⚠️ Could not extract actual task order from raw logging")
                    config_data = data.get('config', {})
                    agent_args = (config_data or {}).get('agent_args', {})
                    if 'reasoning_effort' in agent_args:
                        config_metadata['reasoning_effort'] = agent_args['reasoning_effort']
            except Exception as e:
                print(f"Warning: Could not extract raw_eval_results and config metadata: {e}")
                raw_eval_results = {}

        # ---- First pass: stream and bucket raw entries per task_id ----
        try:
            with open(plaintext_path, "rb") as f:
                for item in ijson.items(f, "raw_logging_results.item"):
                    tid = item.get("weave_task_id") or (item.get("attributes") or {}).get("weave_task_id")
                    if not tid:
                        continue

                    # Record model preference heuristics (unchanged)
                    mdl = (item.get("inputs", {}) or {}).get("model") or (item.get("output", {}) or {}).get("model")
                    if mdl:
                        existing = model_by_tid.get(tid)
                        if existing:
                            if mdl.lower().startswith('gpt-4o') and not existing.lower().startswith('gpt-4o'):
                                pass
                            elif not mdl.lower().startswith('gpt-4o') and existing.lower().startswith('gpt-4o'):
                                model_by_tid[tid] = mdl
                            elif not mdl.lower().startswith('gpt-4o') and not existing.lower().startswith('gpt-4o'):
                                if len(mdl) > len(existing):
                                    model_by_tid[tid] = mdl
                        else:
                            model_by_tid[tid] = mdl

                    # Stash eval items (unchanged)
                    if ("reward" in item) or ("task" in item) or ("info" in item):
                        eval_by_tid[tid] = item
                        continue

                    # Optional filter by required model (unchanged)
                    if require_model and ((item.get("inputs", {}) or {}).get("model") != require_model):
                        continue

                    # Bucket full entry for sanity subset logic
                    entries_by_tid.setdefault(tid, []).append(item)
        finally:
            try:
                os.remove(plaintext_path)
            except OSError:
                pass

        # ---- Second pass: per task decide largest-entry transcript or fallback ----
        filename_model = WeaveExtractor._extract_model_from_filename(zip_name)
        # For error-file labeling
        benchmark_label = zip_name.split('_')[0] if '_' in zip_name else "unknown"

        for tid in sorted(entries_by_tid.keys(), key=lambda x: str(x)):
            task_entries = entries_by_tid[tid]

            # Run sanity check and choose largest entry
            ok, largest_entry = WeaveExtractor._sanity_check_task_logs(task_entries)

            # Prepare a bucket of normalized messages using the chosen strategy
            bucket: List[Dict[str, Any]] = []
            primary_model = model_by_tid.get(tid)

            if ok and largest_entry is not None:
                # ✅ Preferred path: parse ONLY the largest entry for this task_id
                ctx_msgs = WeaveExtractor.extract_contextual_messages_from_item(largest_entry, primary_model)
                bucket.extend(ctx_msgs)
                nm = WeaveExtractor.normalize_weave_log_item(largest_entry)
                if nm:
                    bucket.append(nm)
                ao = WeaveExtractor.normalize_assistant_output(largest_entry, primary_model)
                if ao:
                    bucket.append(ao)
            else:
                # ⚠️ Fallback to the original aggregation + dedupe path
                # WeaveExtractor._save_failed_sanity(task_entries, str(tid), model_by_tid.get(tid) or (filename_model or "unknown"), benchmark=benchmark_label)
                for item in task_entries:
                    ctx_msgs = WeaveExtractor.extract_contextual_messages_from_item(item, primary_model)
                    bucket.extend(ctx_msgs)
                    nm = WeaveExtractor.normalize_weave_log_item(item)
                    if nm:
                        bucket.append(nm)
                    ao = WeaveExtractor.normalize_assistant_output(item, primary_model)
                    if ao:
                        bucket.append(ao)

            # Build agent_run (unchanged downstream cleaning steps stay in build_agent_run_from_bucket)
            task_eval_results = None
            if include_eval_results and raw_eval_results:
                task_eval_results = WeaveExtractor.extract_eval_results_for_task(tid, task_id_list, raw_eval_results)
            task_model = model_by_tid.get(tid) or filename_model
            run = WeaveExtractor.build_agent_run_from_bucket(
                tid=tid,
                bucket=bucket,
                model=task_model,
                eval_blob=eval_by_tid.get(tid) if include_eval else None,
                task_eval_results=task_eval_results,
                config_metadata=config_metadata,
            )
            yield tid, run
            produced += 1
            if limit and produced >= limit:
                break

# ──────────────────────────────────────────────────────────────────────────────
# Docent conversion
# ──────────────────────────────────────────────────────────────────────────────

class DocentConverter:
    @staticmethod
    def normalize_message_for_docent(msg: Dict[str, Any]) -> Dict[str, Any]:
        normalized = msg.copy()
        if 'tool_calls' in normalized and normalized['tool_calls']:
            fixed_tool_calls = []
            for tool_call in normalized['tool_calls']:
                fixed_tc: Dict[str, Any] = {}
                fixed_tc['id'] = tool_call.get('id', f"tool_{len(fixed_tool_calls)}")
                fixed_tc['type'] = 'function'
                if isinstance(tool_call.get('function'), str):
                    fixed_tc['function'] = tool_call['function']
                elif isinstance(tool_call.get('function'), dict):
                    fixed_tc['function'] = tool_call['function'].get('name', 'unknown_function')
                else:
                    fixed_tc['function'] = 'unknown_function'
                arguments = None
                if 'arguments' in tool_call:
                    arguments = tool_call['arguments']
                elif isinstance(tool_call.get('function'), dict) and 'arguments' in tool_call['function']:
                    arguments = tool_call['function']['arguments']
                if isinstance(arguments, str):
                    try:
                        fixed_tc['arguments'] = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        fixed_tc['arguments'] = {'raw_value': arguments} if arguments else {}
                elif isinstance(arguments, dict):
                    fixed_tc['arguments'] = arguments
                else:
                    fixed_tc['arguments'] = {}
                fixed_tool_calls.append(fixed_tc)
            normalized['tool_calls'] = fixed_tool_calls
        return normalized

    @staticmethod
    def convert_to_docent_messages(loaded_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        docent_results: Dict[str, Any] = {}
        print("Converting to docent ChatMessage format...")
        stats = {'total_messages': 0, 'successful': 0, 'failed': 0, 'failed_tasks': [], 'error_types': {}}
        for zip_name, tasks in loaded_results.items():
            if "error" in tasks:
                docent_results[zip_name] = tasks
                continue
            docent_results[zip_name] = {}
            for task_id, agent_run in tasks.items():
                try:
                    messages = agent_run.get('messages', [])
                    stats['total_messages'] += len(messages)
                    docent_messages: List[Any] = []
                    task_failed_count = 0
                    for i, msg in enumerate(messages):
                        try:
                            normalized_msg = DocentConverter.normalize_message_for_docent(msg)
                            chat_msg = parse_chat_message(normalized_msg)
                            docent_messages.append(chat_msg)
                            stats['successful'] += 1
                        except Exception as e:
                            et = type(e).__name__
                            stats['error_types'][et] = stats['error_types'].get(et, 0) + 1
                            if stats['failed'] < 5:
                                print(f"Warning: Failed to parse message {i} in task {str(task_id)[:12]}...: {e}")
                            stats['failed'] += 1
                            task_failed_count += 1
                            continue
                    if task_failed_count > 0:
                        stats['failed_tasks'].append((task_id, task_failed_count))
                    docent_results[zip_name][task_id] = {
                        'weave_task_id': agent_run.get('weave_task_id'),
                        'model': agent_run.get('model'),
                        'eval': agent_run.get('eval'),
                        'config_metadata': agent_run.get('config_metadata'),
                        'original_message_count': len(messages),
                        'docent_message_count': len(docent_messages),
                        'failed_message_count': task_failed_count,
                        'docent_messages': docent_messages,
                        'original_messages': messages,
                    }
                except Exception as e:
                    et = type(e).__name__
                    stats['error_types'][et] = stats['error_types'].get(et, 0) + 1
                    print(f"❌ Failed to process task {str(task_id)[:12]}...: {e}")
                    stats['failed'] += 1
                    docent_results[zip_name][task_id] = None
        return docent_results, stats

# ──────────────────────────────────────────────────────────────────────────────
# Docent upload
# ──────────────────────────────────────────────────────────────────────────────

class DocentUploader:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DOCENT_API_KEY")
        self.client = Docent(api_key=self.api_key)

    @staticmethod
    def _retry_with_backoff(func, max_retries=3, base_delay=2, *args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, Timeout, HTTPError) as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)

    def create_collection(self, name: str, description: str) -> str:
        print(f"🔧 Creating collection: {name}")
        return self._retry_with_backoff(self.client.create_collection, 3, 2, name=name, description=description)

    @staticmethod
    def _create_agent_run(zip_name: str, task_id: str, agent_run_data: Dict[str, Any]) -> AgentRun:
        benchmark_label = zip_name.split('_')[0]
        base_model = agent_run_data.get('model', 'unknown')
        config_metadata = agent_run_data.get('config_metadata', {}) or {}
        reasoning_effort = config_metadata.get('reasoning_effort') if config_metadata else None
        model_name = f"{base_model}_{reasoning_effort}" if reasoning_effort else base_model
        metadata: Dict[str, Any] = {
            "benchmark_id": benchmark_label,
            "task_id": task_id,
            "model": model_name,
            "run_id": zip_name,
            "weave_task_id": agent_run_data.get('weave_task_id'),
            "original_message_count": agent_run_data['original_message_count'],
            "docent_message_count": agent_run_data['docent_message_count'],
            "failed_message_count": agent_run_data['failed_message_count'],
        }
        if config_metadata:
            metadata.update(config_metadata)
        eval_data = agent_run_data.get('eval', {})
        if eval_data and 'raw_results' in eval_data:
            raw_results = eval_data['raw_results']
            for key, value in raw_results.items():
                if not str(key).startswith('global_'):
                    metadata[f"eval_{key}"] = value
                else:
                    metadata[key] = value
        transcript = Transcript(messages=agent_run_data['docent_messages'], metadata=metadata)
        transcripts = {"default": transcript}
        return AgentRun(transcripts=transcripts, metadata=metadata)

    def upload_transcripts(self, docent_results: Dict[str, Any], collection_id: str, batch_by_model: bool = True) -> Dict[str, Any]:
        upload_stats = {'total_runs': 0, 'successful_uploads': 0, 'failed_uploads': 0, 'skipped_runs': 0, 'failed_runs': []}
        print("🚀 Processing docent_results for upload...")
        if batch_by_model:
            # Group by model
            model_groups: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
            for zip_name, tasks in docent_results.items():
                if "error" in tasks:
                    print(f"⚠️  Skipping {zip_name}: contains error"); upload_stats['skipped_runs'] += 1; continue
                for task_id, agent_run_data in tasks.items():
                    if agent_run_data is None:
                        print(f"⚠️  Skipping task {str(task_id)[:12]}...: agent_run_data is None (conversion failed)")
                        upload_stats['skipped_runs'] += 1; continue
                    model = agent_run_data.get('model', 'unknown')
                    model_groups.setdefault(model, []).append((zip_name, task_id, agent_run_data))
            print(f"📊 Found {len(model_groups)} models:")
            for model, runs in model_groups.items():
                print(f"   {model}: {len(runs)} runs")
            for model, runs in model_groups.items():
                print(f"\n🔄 Processing model: {model} ({len(runs)} runs)")
                agent_runs: List[AgentRun] = []
                for zip_name, task_id, agent_run_data in runs:
                    upload_stats['total_runs'] += 1
                    try:
                        agent_runs.append(self._create_agent_run(zip_name, task_id, agent_run_data))
                    except Exception as e:
                        print(f"❌ Failed to create AgentRun for task {str(task_id)[:12]}...: {e}")
                        upload_stats['failed_uploads'] += 1
                        upload_stats['failed_runs'].append({'task_id': task_id, 'zip_name': zip_name, 'error': str(e)})
                if agent_runs:
                    try:
                        batch_size = min(50, len(agent_runs))
                        total_uploaded = 0
                        for i in range(0, len(agent_runs), batch_size):
                            batch = agent_runs[i:i + batch_size]
                            print(f"   🔄 Uploading batch {i//batch_size + 1} ({len(batch)} runs) for {model}...")
                            self._retry_with_backoff(self.client.add_agent_runs, 3, 2, collection_id, batch)
                            total_uploaded += len(batch)
                            if i + batch_size < len(agent_runs):
                                time.sleep(1)
                        upload_stats['successful_uploads'] += total_uploaded
                        print(f"   ✅ Successfully uploaded {total_uploaded} runs for {model}!")
                    except Exception as e:
                        print(f"   ❌ Failed to upload runs for {model}: {e}")
                        upload_stats['failed_uploads'] += len(agent_runs)
                        for ar in agent_runs:
                            upload_stats['failed_runs'].append({'agent_run_id': getattr(ar, 'id', None), 'model': model, 'error': str(e)})
        else:
            # Upload all at once (still chunked)
            agent_runs: List[AgentRun] = []
            for zip_name, tasks in docent_results.items():
                if "error" in tasks:
                    print(f"⚠️  Skipping {zip_name}: contains error"); upload_stats['skipped_runs'] += 1; continue
                print(f"📁 Processing {zip_name}...")
                for task_id, agent_run_data in tasks.items():
                    upload_stats['total_runs'] += 1
                    try:
                        ar = self._create_agent_run(zip_name, task_id, agent_run_data)
                        agent_runs.append(ar)
                        if len(agent_runs) % 10 == 0:
                            print(f"   ✅ Prepared {len(agent_runs)} agent runs...")
                    except Exception as e:
                        print(f"❌ Failed to create AgentRun for task {str(task_id)[:12]}...: {e}")
                        upload_stats['failed_uploads'] += 1
                        upload_stats['failed_runs'].append({'task_id': task_id, 'zip_name': zip_name, 'error': str(e)})
            print(f"📊 Prepared {len(agent_runs)} agent runs for upload")
            if agent_runs:
                try:
                    batch_size = min(50, len(agent_runs))
                    total_uploaded = 0
                    for i in range(0, len(agent_runs), batch_size):
                        batch = agent_runs[i:i + batch_size]
                        print(f"🔄 Uploading batch {i//batch_size + 1} ({len(batch)} runs)...")
                        self._retry_with_backoff(self.client.add_agent_runs, 3, 2, collection_id, batch)
                        total_uploaded += len(batch)
                        if i + batch_size < len(agent_runs):
                            time.sleep(1)
                    upload_stats['successful_uploads'] = total_uploaded
                    print(f"✅ Successfully uploaded {total_uploaded} agent runs!")
                except Exception as e:
                    print(f"❌ Failed to upload agent runs: {e}")
                    upload_stats['failed_uploads'] += len(agent_runs)
                    for ar in agent_runs:
                        upload_stats['failed_runs'].append({'agent_run_id': getattr(ar, 'id', None), 'error': str(e)})
        DocentUploader._print_upload_stats(upload_stats)
        return upload_stats

    @staticmethod
    def _print_upload_stats(upload_stats: Dict[str, Any]):
        print(f"\n📈 Upload Statistics:")
        print(f"   Total runs processed: {upload_stats['total_runs']}")
        print(f"   Successfully uploaded: {upload_stats['successful_uploads']}")
        print(f"   Failed uploads: {upload_stats['failed_uploads']}")
        print(f"   Skipped runs: {upload_stats['skipped_runs']}")
        if upload_stats['failed_runs']:
            print(f"   Failed runs: {len(upload_stats['failed_runs'])}")

# ──────────────────────────────────────────────────────────────────────────────
# High-level orchestration (formerly main.py)
# ──────────────────────────────────────────────────────────────────────────────

def find_browser_agent_files(client: HFClient, prefix: str = BENCHMARK_AGENT_PREFIX, excluded_files: Optional[Set[str]] = EXCLUDED_FILES) -> List[Dict[str, Any]]:
    files = client.find_files_by_prefix(prefix, excluded_files)
    print(f"Found {len(files)} AssistantBench browser agent ZIP files:")
    for i, file_info in enumerate(files):
        size_mb = file_info['size'] / (1024 * 1024)
        print(f"  {i+1:2d}. {file_info['name']:<80} ({size_mb:>8.1f} MB)")
    return files


def process_all_files(client: HFClient, files: Optional[List[Dict[str, Any]]] = None, task_limit: Optional[int] = DEFAULT_TASK_LIMIT, require_model: Optional[str] = None, include_eval: bool = True, include_eval_results: bool = True) -> Dict[str, Any]:
    if files is None:
        files = find_browser_agent_files(client)
    all_results: Dict[str, Any] = {}
    overall_progress = tqdm(files, desc="Processing ZIP files")
    for file_info in overall_progress:
        zip_name = file_info['name']
        overall_progress.set_description(f"Processing {zip_name[:50]}...")
        try:
            zip_results: Dict[str, Any] = {}
            for tid, agent_run in WeaveExtractor.stream_agent_runs_by_task(client=client, zip_name=zip_name, member_name=None, require_model=require_model, include_eval=include_eval, include_eval_results=include_eval_results, limit=task_limit, aggregate_all=True):
                zip_results[tid] = agent_run
            all_results[zip_name] = zip_results
            print(f"\n✅ {zip_name}: {len(zip_results)} tasks processed")
            if zip_results:
                sample_task_id = next(iter(zip_results))
                sample_run = zip_results[sample_task_id]
                print(f"   Sample: Task {str(sample_task_id)[:12]}... has {len(sample_run.get('messages', []))} messages")
                if sample_run.get('model'):
                    print(f"   Model: {sample_run['model']}")
        except Exception as e:
            print(f"\n❌ Error processing {zip_name}: {e}")
            all_results[zip_name] = {"error": str(e)}
            continue
    overall_progress.close()
    print(f"\n🎉 Processing complete!")
    print(f"Total files processed: {len(all_results)}")
    successful_files = sum(1 for v in all_results.values() if "error" not in v)
    print(f"Successful files: {successful_files}")
    total_tasks = sum(len(v) for v in all_results.values() if "error" not in v)
    print(f"Total tasks extracted: {total_tasks:,}")
    return all_results


def save_results(results: Dict[str, Any], output_file: str = DEFAULT_OUTPUT_FILE) -> None:
    def _json_default(o):
        from datetime import date, datetime
        if isinstance(o, Decimal):
            return format(o, "f")
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("ascii")
        try:
            import numpy as np
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return str(o)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=_json_default, sort_keys=True)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved complete results to: {output_file}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    total_files = len(results)
    total_tasks = sum(len(v) for v in results.values() if isinstance(v, dict) and "error" not in v)
    print(f"📋 Contains {total_files} ZIP files with {total_tasks} total tasks")


def upload_to_docent(docent_client: DocentUploader, docent_results: Dict[str, Any], collection_name: str = DEFAULT_COLLECTION_NAME, collection_description: str = "", batch_by_model: bool = True) -> Tuple[str, Dict[str, Any]]:
    collection_id = docent_client.create_collection(name=collection_name, description=collection_description or f"HAL paper analysis: {collection_name}")
    upload_stats = docent_client.upload_transcripts(docent_results, collection_id, batch_by_model=batch_by_model)
    return collection_id, upload_stats


def run_full_pipeline(collection_name: str = DEFAULT_COLLECTION_NAME, hf_token: Optional[str] = None, docent_api_key: Optional[str] = None, output_file: Optional[str] = None, task_limit: Optional[int] = DEFAULT_TASK_LIMIT, require_model: Optional[str] = None, include_eval: bool = True, batch_by_model: bool = True, save_intermediate: bool = True) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run the complete pipeline from data extraction to Docent upload."""
    load_dotenv()
    print("🚀 Starting full HAL pipeline...")
    client = HFClient(repo_id=os.getenv("HF_REPO_ID", DEFAULT_REPO_ID), revision=os.getenv("HF_REVISION", DEFAULT_REVISION))
    docent_uploader = DocentUploader(api_key=docent_api_key or os.getenv("DOCENT_API_KEY"))

    print("\n📁 Step 1: Processing ZIP files...")
    all_results = process_all_files(client, task_limit=task_limit, require_model=require_model, include_eval=include_eval, include_eval_results=True)

    if save_intermediate:
        intermediate_file = output_file or DEFAULT_OUTPUT_FILE
        print(f"\n💾 Step 2: Saving intermediate results to {intermediate_file}...")
        save_results(all_results, intermediate_file)

    print("\n🔄 Step 3: Converting to docent format...")
    docent_results, conversion_stats = DocentConverter.convert_to_docent_messages(all_results)
    print(f"✅ Conversion complete!")
    print("📊 Conversion stats:")
    print(f"   Total messages: {conversion_stats['total_messages']}")
    print(f"   Successfully converted: {conversion_stats['successful']}")
    print(f"   Failed to convert: {conversion_stats['failed']}")
    if conversion_stats['total_messages'] > 0:
        success_rate = conversion_stats['successful']/max(1, conversion_stats['total_messages'])*100
        print(f"   Success rate: {success_rate:.1f}%")

    print(f"\n📤 Step 4: Uploading to docent collection '{collection_name}'...")
    collection_id, upload_stats = upload_to_docent(docent_uploader, docent_results, collection_name, batch_by_model=batch_by_model)

    print(f"\n🎉 Pipeline complete!")
    print(f"📋 Collection ID: {collection_id}")
    return collection_id, upload_stats, docent_results


if __name__ == "__main__":
    # Minimal CLI execution
    run_full_pipeline()
