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


USACO_GENERALIST_PATTERN = r"usaco_hal_generalist_agent(.+?)_\d+_UPLOAD\.json$"
USACO_SPECIALIST_PATTERN = r"usaco_usaco_episodic__semantic(.+?)_\d+_UPLOAD\.json$"


class USACOMetadata(BaseBenchmarkMetadata):
    problem_id: str = Field(..., description="USACO problem ID")
    difficulty: str = Field(..., description="USACO difficulty level (bronze, silver, gold, platinum)")
    problem_name: str = Field(..., description="Problem name")


def hal_usaco_to_docent_usaco(
    logging_data: dict[str, Any],
    model_name: str,
    eval_results_data: dict[str, Any],
    config_data: dict[str, Any] = None,
    verbose: bool = False,
) -> AgentRun:
    """
    Convert a HAL USACO log entry into a Docent USACO AgentRun.

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
    
    if verbose:
        print(f"      🔍 [VERBOSE] Converting task_id: {task_id}")
        print(f"      🔍 [VERBOSE] Model: {model_name}")

    trajectories = logging_data["inputs"]["messages"]
    messages = parse_messages_to_chat_messages(trajectories)
    
    if verbose:
        print(f"      🔍 [VERBOSE] Parsed {len(messages)} messages from trajectories")

    # Extract individual task evaluation results from rdict
    rdict = eval_results_data.get("rdict", {})
    task_results = rdict.get(task_id, [])
    
    if verbose:
        print(f"      🔍 [VERBOSE] Found {len(task_results)} result attempts for this task")
    
    # Calculate task success based on test results
    # Each problem has multiple test cases
    task_success = 0
    total_tests = 0
    passed_tests = 0
    test_details = []
    
    if task_results and len(task_results) > 0:
        # Get the first attempt (could have multiple attempts)
        attempt = task_results[0]
        result_list = attempt.get("result_list", [])
        
        total_tests = len(result_list)
        for test_result in result_list:
            result_type = test_result.get("result_type", 0)
            status = test_result.get("status", "unknown")
            test_details.append({
                "result_type": result_type,
                "status": status
            })
            # result_type 1 means passed
            if result_type == 1:
                passed_tests += 1
        
        # Task is successful if all tests pass
        if total_tests > 0 and passed_tests == total_tests:
            task_success = 1
    
    # Calculate test pass rate for this task
    test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    if verbose:
        print(f"      🔍 [VERBOSE] Test results: {passed_tests}/{total_tests} passed (success={task_success})")
    
    # Extract overall metrics
    overall_accuracy = eval_results_data.get("accuracy", 0.0)
    successful_tasks = eval_results_data.get("successful_tasks", [])
    failed_tasks = eval_results_data.get("failed_tasks", [])
    total_tasks = len(successful_tasks) + len(failed_tasks)
    
    # Parse problem ID to extract difficulty and name
    # Format: "1234_bronze_problem_name" or "1234_silver_problem_name"
    parts = task_id.split("_", 2)
    problem_number = parts[0] if len(parts) > 0 else ""
    difficulty = parts[1] if len(parts) > 1 else "unknown"
    problem_name = parts[2] if len(parts) > 2 else task_id
    
    if verbose:
        print(f"      🔍 [VERBOSE] Problem: {problem_number}, Difficulty: {difficulty}, Name: {problem_name}")
    
    # Calculate difficulty-specific metrics
    difficulty_counts = {"bronze": 0, "silver": 0, "gold": 0, "platinum": 0}
    difficulty_success_counts = {"bronze": 0, "silver": 0, "gold": 0, "platinum": 0}
    
    # Count tasks by difficulty from all tasks
    for tid in successful_tasks + failed_tasks:
        task_parts = tid.split("_", 2)
        if len(task_parts) > 1:
            task_diff = task_parts[1]
            if task_diff in difficulty_counts:
                difficulty_counts[task_diff] += 1
                if tid in successful_tasks:
                    difficulty_success_counts[task_diff] += 1
    
    # Calculate difficulty accuracies
    difficulty_accuracies = {}
    for diff in ["bronze", "silver", "gold", "platinum"]:
        if difficulty_counts[diff] > 0:
            difficulty_accuracies[f"{diff}_accuracy"] = difficulty_success_counts[diff] / difficulty_counts[diff]
        else:
            difficulty_accuracies[f"{diff}_accuracy"] = 0.0

    scores = {
        "task_success": task_success,
        "test_pass_rate": test_pass_rate,
        "tests_passed": passed_tests,
        "tests_total": total_tests,
        "accuracy": float(overall_accuracy),
        "total_tasks": total_tasks,
        "successful_count": len(successful_tasks),
        "failed_count": len(failed_tasks),
        **difficulty_accuracies,
    }

    additional_metadata = {
        "task_id": task_id,
        "problem_id": task_id,
        "problem_number": problem_number,
        "difficulty": difficulty,
        "problem_name": problem_name,
        "task_success": task_success,
        "test_details": test_details,
        "tests_passed": passed_tests,
        "tests_total": total_tests,
        "test_pass_rate": test_pass_rate,
        "overall_accuracy": float(overall_accuracy),
        "difficulty_statistics": {
            "counts": difficulty_counts,
            "success_counts": difficulty_success_counts,
            "accuracies": difficulty_accuracies,
        },
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
    }

    # Extract metadata from config
    config_metadata = extract_metadata_from_config(
        config_data, model_name, "usaco"
    )

    metadata = USACOMetadata(
        benchmark_id="usaco",
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
        problem_id=task_id,
        difficulty=difficulty,
        problem_name=problem_name,
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


def usaco_task_processor(
    task_logs_dict, data, model_name, conversion_function, max_runs, verbose=False
):
    """
    Custom task processor for USACO that handles its evaluation structure.
    
    USACO stores evaluation results in raw_eval_results.rdict[task_id] with
    test-by-test results for each problem. Overall metrics are at the top level.
    
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
    
    if verbose:
        print(f"   🔍 [VERBOSE] Processing {len(task_logs_dict)} task logs")
        print(f"   🔍 [VERBOSE] Max runs to process: {max_runs}")
    
    # Extract evaluation results
    results_data = data.get("results", {})
    raw_eval_results = data.get("raw_eval_results", {})
    
    if not raw_eval_results:
        print(f"   ❌ No raw_eval_results found, skipping file")
        return agent_runs
    
    # Check if we have rdict data
    rdict = raw_eval_results.get("rdict", {})
    if not rdict:
        print(f"   ❌ No rdict data found in raw_eval_results, skipping file")
        return agent_runs
    
    if verbose:
        print(f"   🔍 [VERBOSE] Found rdict with {len(rdict)} task results")

    # Combine results and raw_eval_results for the conversion function
    combined_eval_data = {**raw_eval_results, **results_data}

    for task_id, log_entry in task_logs_dict.items():
        if processed >= max_runs:
            if verbose:
                print(f"   🔍 [VERBOSE] Reached max runs limit ({max_runs}), stopping")
            break

        # Check if we have test results for this task
        if task_id not in rdict:
            print(f"   ⚠️ No test results found for task_id={task_id}, skipping.")
            continue

        try:
            if verbose:
                print(f"   🔍 [VERBOSE] Processing task {processed+1}/{max_runs}: {task_id}")
            
            # Pass the combined evaluation data
            agent_run = conversion_function(
                log_entry, model_name, combined_eval_data, config_data, verbose
            )
            agent_runs.append(agent_run)
            processed += 1
            
            if verbose:
                print(f"   🔍 [VERBOSE] ✓ Successfully processed task {task_id}")
        except Exception as e:
            print(f"   ❌ Error processing task_id={task_id}: {e}")
            if verbose:
                import traceback
                print(f"   🔍 [VERBOSE] Traceback: {traceback.format_exc()}")
            continue

    if verbose:
        print(f"   🔍 [VERBOSE] Completed processing: {len(agent_runs)} agent runs created")

    return agent_runs


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload USACO agent runs to Docent collection."
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    args = parser.parse_args()
    
    if args.verbose:
        print("🔍 [VERBOSE] Verbose logging enabled")
        print(f"🔍 [VERBOSE] Arguments: dry_run={args.dry_run}, agent_type={args.agent_type}, download={args.download}")

    if args.agent_type == "specialist":
        directory = os.path.join(os.getcwd(), "hal_traces", "uscao_data")
        file_pattern = USACO_SPECIALIST_PATTERN
        collection_prefix = "USACO-Specialist"
        system_prompt_prefix = (
            "You are a USACO competitive programming specialist"
        )
    else:
        directory = os.path.join(os.getcwd(), "hal_traces", "uscao_data")
        file_pattern = USACO_GENERALIST_PATTERN
        collection_prefix = "USACO-Generalist"
        system_prompt_prefix = (
            "You are an expert assistant who can solve any task using code blobs"
        )
    
    if args.verbose:
        print(f"🔍 [VERBOSE] Configuration:")
        print(f"🔍 [VERBOSE]   Directory: {directory}")
        print(f"🔍 [VERBOSE]   Pattern: {file_pattern}")
        print(f"🔍 [VERBOSE]   Collection: {collection_prefix}")
    
    agent_runs, collection_name, report_path = process_benchmark_files(
        directory=directory,
        file_pattern=file_pattern,
        conversion_function=hal_usaco_to_docent_usaco,
        collection_name_prefix=collection_prefix,
        system_prompt_prefix=system_prompt_prefix,
        dry_run=args.dry_run,
        max_files=3,
        max_runs_per_model=3,
        task_processor=usaco_task_processor,
        download_if_missing=args.download,
        verbose=args.verbose,
    )

    if len(agent_runs) == 0:
        print("❌ No agent runs to upload. Exiting.")
        exit(1)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name=collection_name,
        description=f"USACO agent runs - {len(agent_runs)} runs processed",
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
