"""
Transcript Data Processor

Downloads transcript messages and metadata from Docent agent runs into a unified dataframe.
"""

import os
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from docent import Docent


# Configuration: maps collection_id -> rubric_id (for getting run lists)
# All rubrics in a collection share the same runs, so we only need one rubric ID per collection
COLLECTION_RUBRICS = {
    "b64b3811-7210-46f1-84fb-c97693ac4f56": "83762c94-012f-4648-bec5-5a6f24582f03",  # Instruction Following
    "02879d7f-5e50-4bca-8dd7-33a39da4762d": "ef6f2d13-0a1f-4931-a157-03ae68f03064",  # Instruction Following
    "e80ca13a-028c-4cc8-9ea5-1977aa04297d": "538d2104-8eed-45a7-8caf-36928912763e",  # Verification
    "db38b00b-3097-4e63-abc6-b36a7c58a7e8": "fa1b91a7-c777-4d4b-a593-22bb9c3d2703",  # Environmental Barrier
}


def extract_messages_from_run(run_info):
    """Extract messages from an AgentRun object."""
    messages = []

    # Iterate through transcripts
    for transcript in run_info.transcripts:
        if hasattr(transcript, 'messages') and transcript.messages:
            # Convert messages to serializable format
            for msg in transcript.messages:
                message_dict = {
                    'role': msg.role,
                    'content': msg.content if hasattr(msg, 'content') else None,
                }

                # Add optional fields if they exist
                if hasattr(msg, 'id') and msg.id:
                    message_dict['id'] = msg.id
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_dict['tool_calls'] = [
                        {
                            'id': tc.id if hasattr(tc, 'id') else None,
                            'function': tc.function if hasattr(tc, 'function') else None,
                            'arguments': tc.arguments if hasattr(tc, 'arguments') else None,
                            'type': tc.type if hasattr(tc, 'type') else None,
                        }
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    message_dict['tool_call_id'] = msg.tool_call_id

                messages.append(message_dict)

    return messages


def fetch_collection_data(client, collection_id, rubric_id, collection_name, test_mode=False):
    """Fetch transcript data for all runs in a collection.

    Uses the rubric run state to get the list of agent_run_ids.
    Since all rubrics share the same runs, we only need one rubric ID per collection.
    """
    print(f"\nProcessing: {collection_name}")

    # Get run IDs from the rubric
    try:
        run_state = client.get_rubric_run_state(collection_id, rubric_id)
        run_state_df = pd.DataFrame(run_state['results'])

        if len(run_state_df) == 0:
            print(f"  No runs found in collection")
            return pd.DataFrame()

        # Limit runs in test mode
        if test_mode:
            run_state_df = run_state_df.head(5)
            print(f"  [TEST MODE] Processing {len(run_state_df)} runs (out of {len(run_state['results'])} total)...")
        else:
            print(f"  Processing {len(run_state_df)} runs...")

    except Exception as e:
        print(f"  Error fetching run list: {e}")
        return pd.DataFrame()

    # Fetch data for each run
    results = []
    for _, run in tqdm(run_state_df.iterrows(), total=len(run_state_df), desc="  Fetching"):
        run_id = run['agent_run_id']
        try:
            run_info = client.get_agent_run(collection_id, run_id)

            # Extract metadata
            metadata = run_info.metadata if hasattr(run_info, 'metadata') else {}

            # Extract messages
            messages = extract_messages_from_run(run_info)

            # Create row
            row = {
                'collection_id': collection_id,
                'collection_name': collection_name,
                'agent_run_id': run_id,
                'benchmark_id': metadata.get('benchmark_id'),
                'task_id': metadata.get('task_id'),
                'model': metadata.get('model'),
                'run_id': metadata.get('run_id'),
                'weave_task_id': metadata.get('weave_task_id'),
                'reasoning_effort': metadata.get('reasoning_effort'),
                'docent_message_count': metadata.get('docent_message_count'),
                'failed_message_count': metadata.get('failed_message_count'),
                'eval_is_successful': metadata.get('eval_is_successful'),
                'eval_successful_tasks': metadata.get('eval_successful_tasks'),
                'eval_failed_tasks': metadata.get('eval_failed_tasks'),
                'eval_has_successful_subtasks': metadata.get('eval_has_successful_subtasks'),
                'eval_answer': metadata.get('eval_answer'),
                'eval_score': metadata.get('eval_score'),
                'messages': json.dumps(messages),  # Store as JSON string
                'message_count': len(messages),
                'metadata': json.dumps(metadata)  # Store full metadata as JSON
            }

            results.append(row)

        except Exception as e:
            print(f"    Error fetching run {run_id}: {e}")
            results.append({
                'collection_id': collection_id,
                'collection_name': collection_name,
                'agent_run_id': run_id,
                'error': str(e)
            })

    df = pd.DataFrame(results)
    print(f"  ✓ {len(df)} rows")
    return df


def process_all_collections(api_key=None, output_file=None, test_mode=False):
    """Process all collections and return combined DataFrame."""
    client = Docent(api_key=api_key or os.getenv("DOCENT_API_KEY"))

    # Get collection names
    collections = {c['id']: c['name'] for c in client.list_collections()}

    if test_mode:
        print("\n🧪 TEST MODE: Processing only 5 runs per collection")
        print("=" * 60)

    all_data = []

    # Process each collection
    for collection_id, rubric_id in COLLECTION_RUBRICS.items():
        collection_name = collections.get(collection_id, f"Unknown ({collection_id[:8]}...)")

        try:
            df = fetch_collection_data(client, collection_id, rubric_id, collection_name, test_mode)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"  ✗ Error processing collection: {e}")

    # Combine results
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"✓ Processed {len(all_data)} collections")
        print(f"✓ Total rows: {len(combined_df)}")
        print(f"✓ Total messages: {combined_df['message_count'].sum()}")

        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"✓ Saved to {output_file}")

        return combined_df

    return pd.DataFrame()


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract transcript data from Docent collections')
    parser.add_argument('--api-key', help='Docent API key (or use DOCENT_API_KEY env var)')
    parser.add_argument('--output-file', default='transcript_data.csv',
                       help='Output CSV file path (default: transcript_data.csv)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only 5 runs per collection')
    args = parser.parse_args()

    df = process_all_collections(
        api_key=args.api_key,
        output_file=args.output_file,
        test_mode=args.test
    )

    if not df.empty:
        print(f"\n✅ Complete! {len(df)} runs processed")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head(2))
    else:
        print("\n❌ No data retrieved")

    return df


if __name__ == "__main__":
    main()