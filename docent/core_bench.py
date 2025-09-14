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


COREBENCH_GENERALIST_PATTERN = r"corebench_hard_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
COREBENCH_SPECIALIST_PATTERN = r"corebench_hard_coreagent_\d+_UPLOAD\.json$"


class CoreBenchMetadata(BaseBenchmarkMetadata):
    pass


def hal_core_bench_to_docent_core_bench(
    logging_data: dict[str, Any], model_name: str, eval_results_data: dict[str, Any], config_data: dict[str, Any] = None
) -> AgentRun:
    """
    Convert a HAL CoreBench log entry into a Docent CoreBench AgentRun.

    Args:
        logging_data (dict[str, Any]): Raw logging data containing model inputs and messages.
        model_name (str): The model name to assert against the log entry.
        eval_results_data (dict[str, Any]): Evaluation results containing task results and metadata.

    Returns:
        AgentRun: An AgentRun object containing parsed transcript messages, metadata,
                  and evaluation results.
    """
    assert logging_data["inputs"]["model"] == model_name
    task_id = logging_data["weave_task_id"]

    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)

    correct_written = eval_results_data.get("correct_written_answers", 0)
    total_written = eval_results_data.get("total_written_questions", 0)
    correct_vision = eval_results_data.get("correct_vision_answers", 0)
    total_vision = eval_results_data.get("total_vision_questions", 0)

    total_correct = correct_written + correct_vision
    total_questions = total_written + total_vision

    accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    scores = {
        "accuracy": round(accuracy, 3),
        "correct_written_answers": correct_written,
        "total_written_questions": total_written,
        "correct_vision_answers": correct_vision,
        "total_vision_questions": total_vision,
    }

    additional_metadata = dict(eval_results_data)

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(config_data, model_name, "corebench_hard")

    metadata = CoreBenchMetadata(
        benchmark_id="corebench_hard",
        task_id=task_id,
        model=model_name,
        model_name=config_metadata["model_name"],
        agent_name=config_metadata["agent_name"],
        reasoning_effort=config_metadata["reasoning_effort"],
        budget=config_metadata["budget"],
        date=config_metadata["date"],
        benchmark_name=config_metadata["benchmark_name"],
        accuracy=accuracy,  # For CoreBench, use the calculated accuracy
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
        description="Upload CoreBench agent runs to Docent collection."
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
    args = parser.parse_args()

    if args.agent_type == "specialist":
        directory = "/Users/saitejautpala/work/hal_explore/hal_traces/core_bench_data"
        file_pattern = COREBENCH_SPECIALIST_PATTERN
        collection_prefix = "CoreBench-Specialist"
    else:
        directory = "/Users/saitejautpala/work/hal_explore/hal_traces/core_bench_data"
        file_pattern = COREBENCH_GENERALIST_PATTERN
        collection_prefix = "CoreBench-Generalist"

    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_core_bench_to_docent_core_bench,
        collection_name_prefix=collection_prefix,
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"CoreBench agent runs - {len(agent_runs)} runs processed",
    )

    # Upload agent runs in chunks to avoid payload size limits
    chunk_size = 300
    total_runs = len(agent_runs)
    
    for i in range(0, total_runs, chunk_size):
        chunk = agent_runs[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (total_runs + chunk_size - 1) // chunk_size
        
        print(f"📤 Uploading chunk {chunk_num}/{total_chunks} ({len(chunk)} runs)...")
        client.add_agent_runs(collection_id, chunk)

    print(f"✅ Uploaded {len(agent_runs)} agent runs to collection {collection_id}")
    print(f"📊 Collection: {collection_name}")
    
    if report_path:
        print(f"📄 Analysis report: {report_path}")
