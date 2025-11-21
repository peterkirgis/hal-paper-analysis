import argparse
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import Field

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent_integration import (
    BaseBenchmarkMetadata,
    extract_metadata_from_config,
    parse_messages_to_chat_messages,
    process_benchmark_files,
)


SCIENCEAGENT_GENERALIST_PATTERN = r"scienceagentbench_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
SCIENCEAGENT_SPECIALIST_PATTERN = r"scienceagentbench_sab_selfdebug(.+?)_\d+_UPLOAD\.json$"


class ScienceAgentMetadata(BaseBenchmarkMetadata):
    instance_id: str = Field(..., description="ScienceAgent instance ID")
    task_type: str = Field(..., description="Type of ScienceAgent task")


def hal_scienceagent_to_docent_scienceagent(
    logging_data: dict[str, Any],
    model_name: str,
    eval_results_data: dict[str, Any],
    config_data: dict[str, Any] = None,
) -> AgentRun:
    """
    Convert a HAL ScienceAgent log entry into a Docent ScienceAgent AgentRun.

    Args:
        logging_data (dict[str, Any]): Raw logging data containing model inputs and messages.
        model_name (str): The model name to assert against the log entry.
        eval_results_data (dict[str, Any]): Evaluation results containing task results and metadata.
        config_data (dict[str, Any]): Configuration data for the run.

    Returns:
        AgentRun: An AgentRun object containing parsed transcript messages, metadata,
                  and evaluation results.
    """
    assert logging_data["inputs"]["model"] == model_name
    task_id = logging_data["weave_task_id"]

    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)

    # Extract individual task evaluation result from eval_result section
    eval_result_data = eval_results_data.get("eval_result", {})
    task_eval_result = eval_result_data.get(task_id, {})
    
    # Extract task-specific metrics
    valid_program = task_eval_result.get("valid_program", 0)
    codebert_score = task_eval_result.get("codebert_score", 0.0)
    success_rate = task_eval_result.get("success_rate", 0.0)
    log_info = task_eval_result.get("log_info", "")
    task_cost = task_eval_result.get("cost", 0.0)
    
    # Extract overall metrics from results section
    overall_success_rate = eval_results_data.get("success_rate", 0.0)
    overall_codebert_score = eval_results_data.get("codebert_score", 0.0)
    overall_valid_program_rate = eval_results_data.get("valid_program_rate", 0.0)
    total_cost = eval_results_data.get("total_cost", 0.0)
    
    # Calculate derived metrics
    task_success = 1 if success_rate > 0 else 0
    program_validity = 1 if valid_program > 0 else 0

    scores = {
        "valid_program": int(valid_program),
        "codebert_score": float(codebert_score),
        "success_rate": float(success_rate),
        "task_success": task_success,
        "program_validity": program_validity,
        "task_cost": float(task_cost),
        "overall_success_rate": float(overall_success_rate),
        "overall_codebert_score": float(overall_codebert_score),
        "overall_valid_program_rate": float(overall_valid_program_rate),
        "total_cost": float(total_cost),
    }

    additional_metadata = {
        "task_id": task_id,
        "instance_id": task_id,  # In ScienceAgent, task_id is the instance_id
        "task_success": task_success,
        "program_validity": program_validity,
        "log_info": log_info,
        "task_cost": float(task_cost),
        "evaluation_metrics": {
            "valid_program": int(valid_program),
            "codebert_score": float(codebert_score),
            "success_rate": float(success_rate),
        },
        "overall_metrics": {
            "success_rate": float(overall_success_rate),
            "codebert_score": float(overall_codebert_score),
            "valid_program_rate": float(overall_valid_program_rate),
            "total_cost": float(total_cost),
        },
    }

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(
        config_data, model_name, "scienceagentbench"
    )

    # Determine task type from instance_id or other metadata
    task_type = "data_driven_discovery"  # Default type for ScienceAgent
    
    # Try to extract more specific task type from agent_output if available
    agent_output = eval_results_data.get("agent_output", {})
    if task_id in agent_output:
        task_history = agent_output[task_id].get("history", [])
        if task_history:
            # Could analyze the task content to determine specific type
            # For now, keep it generic
            pass

    metadata = ScienceAgentMetadata(
        benchmark_id="scienceagentbench",
        task_id=task_id,
        model=model_name,
        run_id=config_metadata["run_id"],
        model_name=config_metadata["model_name"],
        agent_name=config_metadata["agent_name"],
        reasoning_effort=config_metadata["reasoning_effort"],
        budget=config_metadata["budget"],
        date=config_metadata["date"],
        benchmark_name=config_metadata["benchmark_name"],
        accuracy=float(task_success),  # Use task success as accuracy
        agent_config=config_data,
        scores=scores,
        additional_metadata=additional_metadata,
        scoring_metadata=None,
        instance_id=task_id,
        task_type=task_type,
    )

    transcript = Transcript(
        messages=messages,
        metadata=metadata,
    )

    agent_run = AgentRun(
        transcripts={"default": transcript},
        metadata=metadata,
    )

    return agent_run


def scienceagent_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs
):
    """
    Custom task processor for ScienceAgent that handles its unique evaluation structure.
    
    ScienceAgent stores evaluation results in a nested structure:
    - raw_eval_results.eval_result[task_id] contains per-task metrics
    - Overall metrics are at the top level of raw_eval_results
    
    Args:
        task_logs_dict: Dictionary mapping task_id to log entries
        data: Raw data containing eval results
        model_name: Model name from file path
        conversion_function: Function to convert log entry to AgentRun
        max_runs: Maximum number of runs to process

    Returns:
        List[AgentRun]: List of converted agent runs
    """
    from docent.data_models import AgentRun

    agent_runs = []
    processed = 0

    # Extract config for all tasks in this file
    config_data = data.get("config", {})
    
    # Extract evaluation results
    raw_eval_results = data.get("raw_eval_results", {})
    
    if not raw_eval_results:
        print(f"   ❌ No raw_eval_results found, skipping file")
        return agent_runs

    # Check if we have eval_result data
    eval_result_data = raw_eval_results.get("eval_result", {})
    if not eval_result_data:
        print(f"   ❌ No eval_result data found in raw_eval_results, skipping file")
        return agent_runs

    for task_id, log_entry in task_logs_dict.items():
        if processed >= max_runs:
            break

        # Check if we have evaluation results for this task
        if task_id not in eval_result_data:
            print(f"   ⚠️ No evaluation results found for task_id={task_id}, skipping.")
            continue

        try:
            # Pass the entire raw_eval_results which contains both per-task and overall metrics
            agent_run = conversion_function(
                log_entry, model_name, raw_eval_results, config_data
            )
            agent_runs.append(agent_run)
            processed += 1
        except Exception as e:
            print(f"   ❌ Error processing task_id={task_id}: {e}")
            continue

    return agent_runs


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload ScienceAgent agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 2 logs from 3 models (default: process all logs from all models)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["generalist", "specialist"],
        default="generalist",
        help="Type of agent data to process (generalist or specialist)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files from Hugging Face (will overwrite existing files)",
    )
    args = parser.parse_args()

    if args.agent_type == "specialist":
        directory = os.path.join(os.getcwd(), "hal_traces", "scienceagent_data")
        file_pattern = SCIENCEAGENT_SPECIALIST_PATTERN
        collection_prefix = "ScienceAgent-Specialist"
        system_prompt_prefix = (
            "You are a specialist ScienceAgent"
        )
    else:
        directory = os.path.join(os.getcwd(), "hal_traces", "scienceagent_data")
        file_pattern = SCIENCEAGENT_GENERALIST_PATTERN
        collection_prefix = "ScienceAgent-Generalist"
        system_prompt_prefix = (
            "You are an expert assistant who can solve any task using code blobs"
        )
    
    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_scienceagent_to_docent_scienceagent,
        collection_name_prefix=collection_prefix,
        system_prompt_prefix=system_prompt_prefix,
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
        task_processor=scienceagent_task_processor,
        download_if_missing=args.download,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"ScienceAgent agent runs - {len(agent_runs)} runs processed",
    )

    # Upload agent runs in chunks to avoid payload size limits
    chunk_size = 300
    total_runs = len(agent_runs)

    for i in range(0, total_runs, chunk_size):
        chunk = agent_runs[i : i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        print(f"📤 Uploading chunk {chunk_num}/{total_chunks} ({len(chunk)} runs)...")
        client.add_agent_runs(collection_id, chunk)

    print(f"✅ Uploaded {len(agent_runs)} agent runs to collection {collection_id}")
    print(f"📊 Collection: {collection_name}")

    if report_path:
        print(f"📄 Analysis report: {report_path}")
