from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


ModelUsage = Dict[
    str, int
]  


@dataclass
class Config:
    agent_name: str
    benchmark_name: str
    date: str
    run_id: str
    agent_args: Any
    run_command: str


@dataclass
class Results:
    accuracy: float
    successful_tasks: List[str]
    failed_tasks: List[str]
    total_cost: float
    latencies: List[float]


@dataclass
class Task:
    description: str
    requirements: List[str]
    expected_output: Any


@dataclass
class RawEvalResult:
    reward: float
    taken_actions: List[Any] 
    task: Task


@dataclass
class LoggingInputs:
    self: Any
    messages: List[Dict[str, Any]]
    model: str
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None


@dataclass
class RawLoggingResult:
    id: str
    project_id: str
    op_name: str
    display_name: str
    trace_id: str
    parent_id: Optional[str]
    started_at: datetime
    attributes: Dict[str, Any]
    inputs: LoggingInputs
    ended_at: Optional[datetime]
    exception: Optional[str]
    output: Any
    summary: Optional[Dict[str, Any]]
    wb_user_id: str
    wb_run_id: str
    deleted_at: Optional[datetime]
    storage_size_bytes: int
    total_storage_size_bytes: int
    weave_task_id: str
    created_timestamp: datetime


@dataclass
class GitInfo:
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    repo_url: Optional[str] = None
    dirty: Optional[bool] = None


@dataclass
class EvaluationData:
    config: Config
    results: Results
    raw_eval_results: Dict[str, RawEvalResult]  
    raw_logging_results: List[RawLoggingResult]
    total_usage: ModelUsage
    total_cost: float
    git_info: GitInfo
