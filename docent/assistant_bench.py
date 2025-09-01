import argparse
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import Field

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent_integration import (
    BaseBenchmarkMetadata,
    parse_messages_to_chat_messages,
    process_benchmark_files,
)


ASSISTANTBENCH_FILE_PATTERN = (
    r"assistantbench_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
)


class AssistantBenchMetadata(BaseBenchmarkMetadata):
    benchmark_id: str = Field(
        description="The benchmark that the task belongs to", default="assistant_bench"
    )


def hal_assistant_bench_to_docent_assistant_bench(
    logging_data: dict[str, Any],
    model_name: str,
    score: float,
    has_answer: float,
    exact_match: int,
    task_index: int,
) -> AgentRun:
    """
    Convert a HAL AssistantBench log entry into a Docent AssistantBench AgentRun.

    Args:
        logging_data (dict[str, Any]): Raw logging data containing model inputs and messages.
        model_name (str): The model name to assert against the log entry.
        score (float): Task score from evaluation.
        has_answer (float): Whether the task has an answer.
        exact_match (int): Whether the answer is an exact match.
        task_index (int): Index of this task in the evaluation arrays.

    Returns:
        AgentRun: An AgentRun object containing parsed transcript messages, metadata,
                  and evaluation results.
    """
    assert logging_data["inputs"]["model"] == model_name
    task_id = logging_data["weave_task_id"]

    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)

    scores = {
        "score": float(score),
        "has_answer": float(has_answer),
        "exact_match": int(exact_match),
        "task_index": task_index,
    }

    additional_metadata = {
        "task_index": task_index,
        "score": score,
        "has_answer": has_answer,
        "exact_match": exact_match,
    }

    metadata = AssistantBenchMetadata(
        benchmark_id=task_id,
        task_id=task_id,
        model=model_name,
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


def assistant_bench_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs
):
    """
    Custom task processor for AssistantBench with array-based evaluation results.

    Args:
        task_logs_dict: Dictionary mapping task_id to log entries
        data: Raw data containing eval results
        model_name: Model name for validation
        conversion_function: Function to convert log entry to AgentRun
        max_runs: Maximum number of runs to process

    Returns:
        List[AgentRun]: List of converted agent runs
    """
    from docent.data_models import AgentRun

    agent_runs = []

    raw_eval_results = data.get("raw_eval_results", {})
    scores = raw_eval_results.get("scores", [])
    answers = raw_eval_results.get("answers", [])
    exact_matches = raw_eval_results.get("exact_matches", [])

    if not scores or not answers:
        print(f"   ❌ No evaluation results found, skipping file")
        return agent_runs

    task_ids = sorted(task_logs_dict.keys())

    for i, task_id in enumerate(task_ids):
        if i >= max_runs or i >= len(scores):
            break

        log_entry = task_logs_dict[task_id]
        score = scores[i] if i < len(scores) else 0.0
        has_answer = answers[i] if i < len(answers) else 0.0
        exact_match = exact_matches[i] if i < len(exact_matches) else 0

        try:
            agent_run = conversion_function(
                log_entry, model_name, score, has_answer, exact_match, i
            )
            agent_runs.append(agent_run)
        except Exception as e:
            print(f"   ❌ Error processing task_id={task_id}: {e}")
            continue

    return agent_runs


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload AssistantBench agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 3 logs from 3 models (default: process all logs from all models)",
    )
    args = parser.parse_args()

    directory = "/Users/saitejautpala/work/hal_explore/assistant_bench_data"

    agent_runs, collection_name = process_benchmark_files(
        directory=directory,
        file_pattern=ASSISTANTBENCH_FILE_PATTERN,
        conversion_function=hal_assistant_bench_to_docent_assistant_bench,
        collection_name_prefix="AssistantBench",
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
        task_processor=assistant_bench_task_processor,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"AssistantBench agent runs - {len(agent_runs)} runs processed",
    )

    client.add_agent_runs(collection_id, agent_runs)

    print(f"✅ Uploaded {len(agent_runs)} agent runs to collection {collection_id}")
    print(f"📊 Collection: {collection_name}")
