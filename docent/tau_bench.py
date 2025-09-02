import argparse
import os
from typing import Any, Tuple

from dotenv import load_dotenv
from pydantic import Field

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent_integration import (
    BaseBenchmarkMetadata,
    parse_messages_to_chat_messages,
    process_benchmark_files,
)


TAUBENCH_FILE_PATTERN = r"taubench_airline_hal_generalist_(.+?)_\d+_UPLOAD\.json$"

# Global variable to track failed tasks by model
failed_tasks_by_model = {}


class TauBenchMetadata(BaseBenchmarkMetadata):
    pass


def hal_tau_bench_to_docent_tau_bench(
    logging_data: dict[str, Any], model_name: str, eval_results_data: Any, config_data: dict[str, Any] = None
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
    else:
        # Normal case with dict
        scores = {"reward": round(eval_results_data["reward"], 3)}
        info = eval_results_data["task"]

    # Prepare metadata
    additional_metadata = info.copy() if isinstance(info, dict) else {"task_info": info}

    metadata = TauBenchMetadata(
        benchmark_id="taubench_airline",
        task_id=task_id,
        model=model_name,
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
    args = parser.parse_args()
    directory = "/Users/saitejautpala/work/hal_explore/tau_bench_data"
    agent_runs, collection_name = process_benchmark_files(
        directory=directory,
        file_pattern=TAUBENCH_FILE_PATTERN,
        conversion_function=hal_tau_bench_to_docent_tau_bench,
        collection_name_prefix="TauBench",
        dry_run=args.dry_run,
        max_files=5,
        max_runs_per_model=5,
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
    client.add_agent_runs(collection_id, agent_runs)
    print(f"✅ Uploaded {len(agent_runs)} agent runs to collection {collection_id}")
    print(f"📊 Collection: {collection_name}")
