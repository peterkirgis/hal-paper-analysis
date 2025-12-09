import argparse
import os
from typing import Any, Tuple

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


TAUBENCH_GENERALIST_PATTERN = r"taubench_airline_hal_generalist_(.+?)_\d+_UPLOAD\.json$"
TAUBENCH_SPECIALIST_PATTERN = (
    r"taubench_airline_taubench_fewshot_(.+?)_\d+_UPLOAD\.json$"
)

# Global variable to track failed tasks by model
failed_tasks_by_model = {}


class TauBenchMetadata(BaseBenchmarkMetadata):
    pass


def hal_tau_bench_to_docent_tau_bench(
    logging_data: dict[str, Any],
    model_name: str,
    eval_results_data: Any,
    config_data: dict[str, Any] = None,
) -> AgentRun:
    """
    Convert a HAL TauBench log entry into a Docent TauBench AgentRun.

    Args:
        logging_data (dict[str, Any]): Raw logging data containing model inputs and messages.
        model_name (str): The model name to assert against the log entry.
        eval_results_data (Any): Evaluation results containing reward and task info, or error string.

    Returns:
        AgentRun: An AgentRun object containing parsed transcript messages, metadata,
                  and evaluation results.
    """
    assert logging_data["inputs"]["model"] == model_name
    task_id = logging_data["weave_task_id"]
    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)

    # Handle case where eval_results_data is an error string instead of dict
    if isinstance(eval_results_data, str):
        # Task failed, create default values
        print(f"   ⚠️  Task failed - model: {model_name}, task_id: {task_id}")
        # Track failed tasks by model
        if model_name not in failed_tasks_by_model:
            failed_tasks_by_model[model_name] = 0
        failed_tasks_by_model[model_name] += 1
        scores = {"reward": 0.0, "error": True}
        info = {"error": eval_results_data}
        accuracy = 0.0
    else:
        # Normal case with dict
        reward = round(eval_results_data["reward"], 3)
        scores = {"reward": reward}
        info = eval_results_data["task"]
        accuracy = reward  # For TauBench, reward is the accuracy metric

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(
        config_data, model_name, "taubench_airline"
    )

    # Prepare metadata
    additional_metadata = info.copy() if isinstance(info, dict) else {"task_info": info}

    metadata = TauBenchMetadata(
        benchmark_id="taubench_airline",
        task_id=task_id,
        model=model_name,
        run_id=config_metadata["run_id"],
        model_name=config_metadata["model_name"],
        agent_name=config_metadata["agent_name"],
        reasoning_effort=config_metadata["reasoning_effort"],
        budget=config_metadata["budget"],
        date=config_metadata["date"],
        benchmark_name=config_metadata["benchmark_name"],
        accuracy=accuracy,
        agent_config=config_data,
        scores=scores,
        additional_metadata=additional_metadata,
        scoring_metadata=None,
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


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Upload TauBench agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 5 logs from 5 models (default: process all logs from all models)",
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
        directory = os.path.join(os.getcwd(), "hal_traces", "tau_bench_data")
        file_pattern = TAUBENCH_SPECIALIST_PATTERN
        collection_prefix = "TauBench-Specialist"
        system_prompt_prefix = "# Airline Agent Policy\n\nThe current time"
    else:
        directory = os.path.join(os.getcwd(), "hal_traces", "tau_bench_data")
        file_pattern = TAUBENCH_GENERALIST_PATTERN
        collection_prefix = "TauBench-Generalist"
        system_prompt_prefix = (
            "You are an expert assistant who can solve any task using code blobs"
        )

    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_tau_bench_to_docent_tau_bench,
        collection_name_prefix=collection_prefix,
        system_prompt_prefix=system_prompt_prefix,
        dry_run=args.dry_run,
        max_files=5,
        max_runs_per_model=5,
        download_if_missing=args.download,
    )

    # Print failed tasks summary
    if failed_tasks_by_model:
        print(f"\n📊 Failed Tasks Summary (Tracebacks):")
        total_failed = 0
        for model, count in failed_tasks_by_model.items():
            print(f"   {model}: {count} failed tasks")
            total_failed += count
        print(f"   📈 Total failed tasks: {total_failed}")
    else:
        print(f"\n✅ No failed tasks!")

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))
    collection_id = client.create_collection(
        name=collection_name,
        description=f"TauBench agent runs - {len(agent_runs)} runs processed",
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
