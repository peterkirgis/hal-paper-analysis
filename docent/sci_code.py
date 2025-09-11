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


SCICODE_GENERALIST_PATTERN = r"scicode_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
SCICODE_SPECIALIST_PATTERN = r"scicode_scicode_(.+?)_agent(.+?)_\d+_UPLOAD\.json$"


class SciCodeMetadata(BaseBenchmarkMetadata):
    pass


def hal_sci_code_to_docent_sci_code(
    logging_data: dict[str, Any], model_name: str, eval_results_data: dict[str, Any], config_data: dict[str, Any] = None
) -> AgentRun:
    assert logging_data["inputs"]["model"] == model_name
    task_id = logging_data["weave_task_id"]

    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)

    score = eval_results_data.get("score", 0.0)
    accuracy = float(score)

    scores = {
        "score": float(score),
    }

    additional_metadata = dict(eval_results_data)

    config_metadata = extract_metadata_from_config(config_data, model_name, "scicode")

    metadata = SciCodeMetadata(
        benchmark_id="scicode",
        task_id=task_id,
        model=model_name,
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
        description="Upload SciCode agent runs to Docent collection."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, process only 3 logs from 3 models (default: process all logs from all models)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["generalist", "specialist"],
        default="generalist",
        help="Type of agent data to process (generalist or specialist)",
    )
    args = parser.parse_args()

    if args.agent_type == "specialist":
        directory = "/Users/saitejautpala/work/hal_explore/hal_traces/sci_code_data"
        file_pattern = SCICODE_SPECIALIST_PATTERN
        collection_prefix = "SciCode-Specialist"
    else:
        directory = "/Users/saitejautpala/work/hal_explore/sci_code_data"
        file_pattern = SCICODE_GENERALIST_PATTERN
        collection_prefix = "SciCode-Generalist"

    agent_runs, collection_name = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_sci_code_to_docent_sci_code,
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
        description=f"SciCode agent runs - {len(agent_runs)} runs processed",
    )

    client.add_agent_runs(collection_id, agent_runs)

    print(f"✅ Uploaded {len(agent_runs)} agent runs to collection {collection_id}")
    print(f"📊 Collection: {collection_name}")
