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


GAIA_GENERALIST_PATTERN = r"gaia_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
GAIA_SPECIALIST_PATTERN = r"gaia_hf_open_deep_research(.+?)_\d+_UPLOAD\.json$"


class GAIAMetadata(BaseBenchmarkMetadata):
    level: int = Field(..., description="GAIA difficulty level (1, 2, or 3)")
    task_type: str = Field(..., description="Type of GAIA task")


def hal_gaia_to_docent_gaia(
    logging_data: dict[str, Any],
    model_name: str,
    eval_results_data: dict[str, Any],
    config_data: dict[str, Any] = None,
) -> AgentRun:
    """
    Convert a HAL GAIA log entry into a Docent GAIA AgentRun.

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

    # Extract individual task evaluation result
    task_eval_result = eval_results_data.get(task_id, {})
    task_score = task_eval_result.get("score", False)
    explanation = task_eval_result.get("explanation", "")
    
    # Convert boolean score to float for consistency
    score_value = 1.0 if task_score else 0.0
    
    # Extract overall metrics from eval_results_data
    overall_accuracy = eval_results_data.get("accuracy", 0.0)
    successful_tasks = eval_results_data.get("successful_tasks", [])
    failed_tasks = eval_results_data.get("failed_tasks", [])
    total_tasks = len(successful_tasks) + len(failed_tasks)
    
    # Extract level-specific accuracies
    level_1_accuracy = eval_results_data.get("level_1_accuracy", 0.0)
    level_2_accuracy = eval_results_data.get("level_2_accuracy", 0.0)
    level_3_accuracy = eval_results_data.get("level_3_accuracy", 0.0)

    scores = {
        "score": score_value,
        "task_success": int(task_score),
        "accuracy": float(overall_accuracy),
        "level_1_accuracy": float(level_1_accuracy),
        "level_2_accuracy": float(level_2_accuracy),
        "level_3_accuracy": float(level_3_accuracy),
        "total_tasks": total_tasks,
        "successful_count": len(successful_tasks),
        "failed_count": len(failed_tasks),
    }

    additional_metadata = {
        "task_id": task_id,
        "task_success": int(task_score),
        "explanation": explanation,
        "overall_accuracy": float(overall_accuracy),
        "level_accuracies": {
            "level_1": float(level_1_accuracy),
            "level_2": float(level_2_accuracy),
            "level_3": float(level_3_accuracy),
        },
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
    }

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(
        config_data, model_name, "gaia"
    )

    # Determine task level and type from task_id or other metadata
    # GAIA task_ids typically contain level information
    level = 1  # Default level
    task_type = "general"  # Default type
    
    # Try to extract level from various sources
    if "level" in task_eval_result:
        level = int(task_eval_result["level"])
    elif any(f"level_{i}" in str(task_id).lower() for i in [1, 2, 3]):
        for i in [1, 2, 3]:
            if f"level_{i}" in str(task_id).lower():
                level = i
                break

    metadata = GAIAMetadata(
        benchmark_id="gaia",
        task_id=task_id,
        model=model_name,
        run_id=config_metadata["run_id"],
        model_name=config_metadata["model_name"],
        agent_name=config_metadata["agent_name"],
        reasoning_effort=config_metadata["reasoning_effort"],
        budget=config_metadata["budget"],
        date=config_metadata["date"],
        benchmark_name=config_metadata["benchmark_name"],
        accuracy=score_value,  # Individual task accuracy
        agent_config=config_data,
        scores=scores,
        additional_metadata=additional_metadata,
        scoring_metadata=None,
        level=level,
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


def gaia_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs
):
    """
    Custom task processor for GAIA that handles per-task evaluation results.
    
    GAIA stores evaluation results per task_id with score and explanation.
    
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

    for task_id, log_entry in task_logs_dict.items():
        if processed >= max_runs:
            break

        try:
            # For GAIA, we pass the entire raw_eval_results as eval_results_data
            # The conversion function will extract the specific task data
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
        description="Upload GAIA agent runs to Docent collection."
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
        directory = os.path.join(os.getcwd(), "hal_traces", "gaia_data")
        file_pattern = GAIA_SPECIALIST_PATTERN
        collection_prefix = "GAIA-Specialist"
        system_prompt_prefix = (
            "You are a specialist GAIA agent"
        )
    else:
        directory = os.path.join(os.getcwd(), "hal_traces", "gaia_data")
        file_pattern = GAIA_GENERALIST_PATTERN
        collection_prefix = "GAIA-Generalist"
        system_prompt_prefix = (
            "You are an expert assistant who can solve any task using code blobs"
        )
    
    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_gaia_to_docent_gaia,
        collection_name_prefix=collection_prefix,
        system_prompt_prefix=system_prompt_prefix,
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
        task_processor=gaia_task_processor,
        download_if_missing=args.download,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"GAIA agent runs - {len(agent_runs)} runs processed",
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
