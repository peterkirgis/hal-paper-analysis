"""
Rubric Data Processor

Downloads and processes rubric evaluation data from Docent into a unified dataframe.
"""

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from docent import Docent


# Configuration: maps collection_id -> list of (rubric_id, rubric_label) tuples
COLLECTION_RUBRICS = {
    "b64b3811-7210-46f1-84fb-c97693ac4f56": [
        ("83762c94-012f-4648-bec5-5a6f24582f03", "Instruction Following"),
        ("e9d7dd28-2454-41d5-b493-7c8e9ed431c0", "Tool Use"),
        ("82b30d80-cab4-4ee3-a9bc-24793745228e", "Verification"),
        ("c27fac3a-01e1-4241-9348-8fa197592d3b", "Self Correction"),
        ("31002fcf-8750-4615-9cdd-28206aad0636", "Environmental Barrier")
    ],
    "02879d7f-5e50-4bca-8dd7-33a39da4762d": [
        ("ef6f2d13-0a1f-4931-a157-03ae68f03064", "Instruction Following"),
        ("c89c0b32-8bfb-481b-a51a-62b719c25790", "Verification"),
        ("c64bc823-445c-4ddd-b709-391537d8d5c3", "Tool Use"),
        ("43a0958b-32e5-49b3-823e-020c5e39ea9c", "Environmental Barrier"),
        ("5e0d744b-7305-44a0-baed-319db7da7bc0", "Self Correction")
    ],
    "e80ca13a-028c-4cc8-9ea5-1977aa04297d": [
        ("538d2104-8eed-45a7-8caf-36928912763e", "Verification"),
        ("448c003d-b1d0-4e0b-9ea4-56c7f38bf95e", "Tool Use"),
        ("e73e972f-bdae-4451-acdb-cd7fe4f0152d", "Instruction Following"),
        ("e18f048c-5bfd-4c10-b71d-cb9dd7ea4960", "Environmental Barrier"),
        ("899f56a8-45b0-4acf-a690-7fc03929cd01", "Self Correction")
    ],
    "db38b00b-3097-4e63-abc6-b36a7c58a7e8": [
        ("fa1b91a7-c777-4d4b-a593-22bb9c3d2703", "Environmental Barrier"),
        ("e62515fe-f31f-4f0c-8b5c-1a88e98be26a", "Verification"),
        ("8390c2fb-cfe4-49a1-857b-c534137cda00", "Self Correction"),
        ("492d0340-c7d2-45c7-bcd9-55bae7de4c9d", "Tool Use"),
        ("c968596a-1512-4f45-98ca-10283da3b034", "Instruction Following")
    ]
}

# Columns to extract in final output
FINAL_COLUMNS = [
    'benchmark_id', 'rubric', 'model', 'task_id', 'agent_run_id', 'eval_is_successful', 'label',
    'output_explanation.text', 'docent_message_count', 'eval_has_successful_subtasks', 'eval_answer', 'eval_score'
]


def fetch_rubric_data(client, collection_id, rubric_id, rubric_label, test_mode=False):
    """Fetch and process data for a single rubric."""
    # Get rubric run state and run details
    run_state = client.get_rubric_run_state(collection_id, rubric_id)
    run_state_df = pd.DataFrame(run_state['results'])

    # Limit to 1 run in test mode
    if test_mode and len(run_state_df) > 0:
        run_state_df = run_state_df.head(1)
        print(f"  [TEST MODE] Processing 1 run (out of {len(run_state['results'])} total)...")
    else:
        print(f"  Processing {len(run_state_df)} runs...")

    # Fetch metadata for each run
    run_data = []
    for _, run in tqdm(run_state_df.iterrows(), total=len(run_state_df), desc="  Fetching"):
        try:
            run_info = client.get_agent_run(collection_id, run['agent_run_id'])
            run_data.append({'agent_run_id': run['agent_run_id'], **run_info.metadata})
        except Exception as e:
            print(f"    Error: {e}")
            run_data.append({'agent_run_id': run['agent_run_id'], 'error': str(e)})

    # Merge run_state with metadata
    runs_df = pd.DataFrame(run_data)
    merged_df = pd.merge(run_state_df, runs_df, on='agent_run_id', how='left')

    # Extract 'output' from the nested 'results' column
    # Each cell in 'results' is a list containing one dict with 'output' field
    # Example: [{'id': '...', 'output': {'label': '...', 'explanation': {...}}, ...}]
    merged_df['output'] = merged_df['results'].apply(
        lambda x: x[0]['output'] if isinstance(x, list) and len(x) > 0 else None
    )

    # Expand output column (contains label and explanation)
    output_expanded = pd.json_normalize(merged_df['output']).add_prefix('output_')
    result_df = pd.concat([merged_df.drop(columns=['output', 'results']), output_expanded], axis=1)

    # Add rubric label and extract label from output
    result_df['rubric'] = rubric_label
    result_df['label'] = result_df.get('output_label', None)

    # Select only needed columns
    available_cols = [col for col in FINAL_COLUMNS if col in result_df.columns]
    return result_df[available_cols]


def process_all_rubrics(api_key=None, save_individual=False, output_dir=None, test_mode=False):
    """Process all rubrics and return combined DataFrame."""
    client = Docent(api_key=api_key or os.getenv("DOCENT_API_KEY"))

    # Setup output directory
    if save_individual:
        output_dir = Path(output_dir or Path(__file__).parent / "rubrics")
        output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Get collection names once
    collections = {c['id']: c['name'] for c in client.list_collections()}

    if test_mode:
        print("\n🧪 TEST MODE: Processing only 1 run per rubric")
        print("=" * 60)

    # Process each collection and rubric
    for collection_id, rubrics in COLLECTION_RUBRICS.items():
        print(f"\nProcessing: {collections.get(collection_id, collection_id)}")

        for rubric_id, rubric_label in rubrics:
            print(f"\n  Rubric: {rubric_label}")
            try:
                df = fetch_rubric_data(client, collection_id, rubric_id, rubric_label, test_mode)
                print(f"  ✓ {len(df)} rows")

                # Save individual file
                if save_individual and len(df) > 0:
                    filename = f"{df['benchmark_id'].iloc[0]}_{rubric_label.lower().replace(' ', '_')}.csv"
                    df.to_csv(output_dir / filename, index=False)
                    print(f"  ✓ Saved {filename}")

                all_data.append(df)
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Combine results
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"✓ {len(all_data)} rubrics, {combined_df.shape} total")
        return combined_df
    return pd.DataFrame()


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Process Docent rubric data')
    parser.add_argument('--api-key', help='Docent API key')
    parser.add_argument('--save-individual', action='store_true', help='Save per-rubric CSVs')
    parser.add_argument('--output-dir', default='rubrics', help='Output directory')
    parser.add_argument('--output-file', help='Combined CSV output path')
    parser.add_argument('--test', action='store_true', help='Test mode: process only 1 run per rubric')
    args = parser.parse_args()

    df = process_all_rubrics(args.api_key, args.save_individual, args.output_dir, args.test)

    if args.output_file and not df.empty:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_file, index=False)
        print(f"\n✓ Saved to {args.output_file}")

    return df


if __name__ == "__main__":
    main()
