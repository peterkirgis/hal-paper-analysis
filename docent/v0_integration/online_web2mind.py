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


ONLINE_WEB2MIND_BROWSER_USE_PATTERN = r"browser-use_(.+?)_UPLOAD\.json$"
ONLINE_WEB2MIND_SEEACT_PATTERN = r"seeact_(.+?)_UPLOAD\.json$"


class OnlineWeb2MindMetadata(BaseBenchmarkMetadata):
    task_level: str = Field(..., description="Task difficulty level (easy, medium, hard)")
    website: str = Field(..., description="Target website for the task")
    reference_length: int = Field(..., description="Reference solution length")


def hal_online_web2mind_to_docent_online_web2mind(
    logging_data: dict[str, Any],
    model_name: str,
    eval_results_data: dict[str, Any],
    config_data: dict[str, Any] = None,
) -> AgentRun:
    """
    Convert a HAL Online Web2Mind log entry into a Docent Online Web2Mind AgentRun.

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

    # Extract individual task data from raw_eval_results
    task_data = eval_results_data.get(task_id, {})
    
    # Extract task-specific information
    confirmed_task = task_data.get("confirmed_task", "")
    website = task_data.get("website", "")
    level = task_data.get("level", "medium")
    reference_length = task_data.get("reference_length", 0)
    urls = task_data.get("urls", [])
    
    # Extract overall metrics from eval_results_data
    overall_accuracy = eval_results_data.get("accuracy", 0.0)
    successful_tasks = eval_results_data.get("successful_tasks", [])
    failed_tasks = eval_results_data.get("failed_tasks", [])
    total_tasks = len(successful_tasks) + len(failed_tasks)
    
    # Determine task success
    task_success = 1 if task_id in successful_tasks else 0
    
    # Calculate level-specific metrics
    level_counts = {"easy": 0, "medium": 0, "hard": 0}
    level_success_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    # Count tasks by level from all task data
    for tid, tdata in eval_results_data.items():
        if isinstance(tdata, dict) and "level" in tdata:
            task_level = tdata["level"]
            if task_level in level_counts:
                level_counts[task_level] += 1
                if tid in successful_tasks:
                    level_success_counts[task_level] += 1

    # Calculate level accuracies
    level_accuracies = {}
    for level_name in ["easy", "medium", "hard"]:
        if level_counts[level_name] > 0:
            level_accuracies[f"{level_name}_accuracy"] = level_success_counts[level_name] / level_counts[level_name]
        else:
            level_accuracies[f"{level_name}_accuracy"] = 0.0

    scores = {
        "task_success": task_success,
        "accuracy": float(overall_accuracy),
        "reference_length": reference_length,
        "total_tasks": total_tasks,
        "successful_count": len(successful_tasks),
        "failed_count": len(failed_tasks),
        **level_accuracies,
    }

    additional_metadata = {
        "task_id": task_id,
        "confirmed_task": confirmed_task,
        "website": website,
        "level": level,
        "reference_length": reference_length,
        "urls": urls,
        "task_success": task_success,
        "overall_accuracy": float(overall_accuracy),
        "level_statistics": {
            "counts": level_counts,
            "success_counts": level_success_counts,
            "accuracies": level_accuracies,
        },
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
    }

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(
        config_data, model_name, "online_mind2web"
    )

    metadata = OnlineWeb2MindMetadata(
        benchmark_id="online_mind2web",
        task_id=task_id,
        model=model_name,
        run_id=config_metadata["run_id"],
        model_name=config_metadata["model_name"],
        agent_name=config_metadata["agent_name"],
        reasoning_effort=config_metadata["reasoning_effort"],
        budget=config_metadata["budget"],
        date=config_metadata["date"],
        benchmark_name=config_metadata["benchmark_name"],
        accuracy=float(task_success),  # Individual task accuracy
        agent_config=config_data,
        scores=scores,
        additional_metadata=additional_metadata,
        scoring_metadata=None,
        task_level=level,
        website=website,
        reference_length=reference_length,
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


def online_web2mind_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs
):
    """
    Custom task processor for Online Web2Mind that handles its evaluation structure.
    
    Online Web2Mind stores task-specific data in raw_eval_results[task_id] and
    overall metrics (accuracy, successful_tasks, failed_tasks) at the top level.
    
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
    
    # Extract evaluation results - for online_web2mind, we need both the results
    # section and the raw_eval_results section
    results_data = data.get("results", {})
    raw_eval_results = data.get("raw_eval_results", {})
    
    if not raw_eval_results:
        print(f"   ❌ No raw_eval_results found, skipping file")
        return agent_runs

    # Combine results and raw_eval_results for the conversion function
    combined_eval_data = {**raw_eval_results, **results_data}

    for task_id, log_entry in task_logs_dict.items():
        if processed >= max_runs:
            break

        # Check if we have task data for this task_id
        if task_id not in raw_eval_results:
            print(f"   ⚠️ No task data found for task_id={task_id}, skipping.")
            continue

        try:
            # Pass the combined evaluation data which contains both per-task and overall metrics
            agent_run = conversion_function(
                log_entry, model_name, combined_eval_data, config_data
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
        description="Upload Online Web2Mind agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 2 logs from 3 models (default: process all logs from all models)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["browser-use", "seeact"],
        default="browser-use",
        help="Type of agent data to process (browser-use or seeact)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files from Hugging Face (will overwrite existing files)",
    )
    args = parser.parse_args()

    if args.agent_type == "seeact":
        directory = os.path.join(os.getcwd(), "hal_traces", "online_web2mind_data")
        file_pattern = ONLINE_WEB2MIND_SEEACT_PATTERN
        collection_prefix = "OnlineWeb2Mind-SeeAct"
        system_prompt_prefix = (
            "You are a SeeAct web navigation agent"
        )
    else:
        directory = os.path.join(os.getcwd(), "hal_traces", "online_web2mind_data")
        file_pattern = ONLINE_WEB2MIND_BROWSER_USE_PATTERN
        collection_prefix = "OnlineWeb2Mind-BrowserUse"
        system_prompt_prefix = (
            "You are a Browser-Use web navigation agent"
        )
    
    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_online_web2mind_to_docent_online_web2mind,
        collection_name_prefix=collection_prefix,
        system_prompt_prefix=system_prompt_prefix,
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
        task_processor=online_web2mind_task_processor,
        download_if_missing=args.download,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"Online Web2Mind agent runs - {len(agent_runs)} runs processed",
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
