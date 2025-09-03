"""
Database loader for HAL benchmark results.

This module loads and processes data from all SQLite database files in the 
preprocessed_traces/ directory, combining them into a unified DataFrame with
the most recent results for each benchmark-agent combination.
"""

import pandas as pd
import sqlite3
import re
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_database_tables(db_path):
    """Load all tables from a SQLite database into a dictionary of DataFrames."""
    tables = {}
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get list of all tables in the database
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            print(f"\nTables in {db_path.name}: {table_names}")
            
            # Load each table into a DataFrame
            for table_name in table_names:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                tables[table_name] = df
                print(f"  - {table_name}: {len(df)} rows, {len(df.columns)} columns")
                
    except Exception as e:
        print(f"Error loading {db_path.name}: {e}")
        
    return tables


def pretty_benchmark(name: str) -> str:
    """Convert benchmark names to prettier labels."""
    pretty_map = {
        'assistantbench'        : 'AssistantBench',
        'gaia'                  : 'GAIA',
        'online_mind2web'       : 'Online Mind2Web',
        'taubench_airline'      : 'TauBench – Airline',
        'corebench_hard'        : 'CORE-Bench (Hard)',
        'scicode'               : 'SciCode',
        'scienceagentbench'     : 'ScienceAgentBench',
        'swebench_verified_mini': 'SWE-bench Verified (Mini)',
    }
    
    if pd.isna(name):
        return name
    name = str(name)
    if name in pretty_map:
        return pretty_map[name]
    # generic fallback: nice-ish formatting
    s = name.replace('_', ' ')
    # special casing some common tokens
    s = re.sub(r'\bswebench\b', 'SWE-bench', s, flags=re.IGNORECASE)
    s = re.sub(r'\bmind2web\b', 'Mind2Web', s, flags=re.IGNORECASE)
    # Title-case remaining words (keeps Mind2Web and SWE-bench as set above)
    return ' '.join(w if re.search(r'[A-Z0-9]', w) else w.capitalize() for w in s.split())


def map_task_type(benchmark_name):
    """Map benchmark names to task types."""
    task_mapping = {
        'Web Assistance' : ['assistantbench', 'gaia', 'online_mind2web'],
        'Scientific Programming' : ['corebench_hard', 'scicode', 'scienceagentbench'],
        'Software Engineering' : ['swebench_verified_mini'],
        'Customer Service' : ['taubench_airline']
    }
    
    for task, benchmarks in task_mapping.items():
        if benchmark_name in benchmarks:
            return task
    return 'Other'


def load_most_recent_df():
    """
    Load all database files and return the most recent results DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with the most recent results for each
                     benchmark-agent combination, including derived columns for
                     task types, agent scaffolds, models, and scaled metrics.
    """
    # Define the path to the preprocessed_traces directory
    db_dir = Path('hal-frontend/preprocessed_traces')
    print(f"Looking for DB files in: {db_dir.absolute()}")
    
    # Get all .db files in the directory
    db_files = list(db_dir.glob('*.db'))
    print(f"Found {len(db_files)} database files:")
    for db_file in db_files:
        print(f"  - {db_file.name}")
    
    # Load all databases
    all_databases = {}
    for db_file in db_files:
        db_name = db_file.stem  # filename without extension
        print(f"\n{'='*50}")
        print(f"Loading database: {db_name}")
        print(f"{'='*50}")
        
        tables = load_database_tables(db_file)
        all_databases[db_name] = tables
    
    # Define the columns we want to keep
    desired_columns = [
        'benchmark_name', 'agent_name', 'date', 'run_id', 
        'successful_tasks', 'failed_tasks', 'total_cost', 'accuracy'
    ]
    
    # Collect all parsed_results DataFrames with token usage data
    parsed_results_dfs = []
    
    for db_name, tables in all_databases.items():
        if 'parsed_results' in tables:
            df = tables['parsed_results'].copy()
            
            # Check which desired columns exist in this DataFrame
            available_columns = [col for col in desired_columns if col in df.columns]
            missing_columns = [col for col in desired_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: {db_name} missing columns: {missing_columns}")
            
            # Select only the available desired columns
            df_filtered = df[available_columns]
            
            # Add any missing columns with None values
            for col in missing_columns:
                df_filtered[col] = None
                
            # Reorder columns to match desired order
            df_filtered = df_filtered[desired_columns]
            
            # Add token usage data if available
            if 'token_usage' in tables:
                token_df = tables['token_usage'].copy()
            
                # Select token columns
                token_columns = ['run_id', 'benchmark_name', 'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens']
                
                # Filter token_df to only these columns if they exist
                filtered_token_df = token_df[[col for col in token_columns if col in token_df.columns]]

                # Remove if benchmark_name is taubench_airline and model_name is "GPT-4o (August 2024)" from filtered_token_df
                filtered_token_df = filtered_token_df[~((filtered_token_df['benchmark_name'] == 'taubench_airline') & (filtered_token_df['model_name'] == 'GPT-4o (August 2024)'))]

                # Drop benchmark_name and model_name
                filtered_token_df = filtered_token_df.drop(columns=['benchmark_name', 'model_name'])

                # Merge with parsed results
                df_filtered = df_filtered.merge(
                    filtered_token_df, 
                    on='run_id', 
                    how='left')
               
            else:
                print(f"No token_usage table found in {db_name}")
                # Add empty token columns
                for col in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    if col not in df_filtered.columns:
                        df_filtered[col] = None
            
            parsed_results_dfs.append(df_filtered)
            print(f"Added {len(df_filtered)} rows from {db_name}")
    
    # Union all DataFrames
    if not parsed_results_dfs:
        print("No parsed_results tables found!")
        return pd.DataFrame()
    
    union_df = pd.concat(parsed_results_dfs, ignore_index=True)
    print(f"\nTotal rows after union: {len(union_df)}")
    
    # Keep only the most recent date for each benchmark_name + agent_name combination
    print("\nFiltering to most recent date for each benchmark_name + agent_name combination...")
    
    # Convert date column to datetime if it's not already
    union_df['date'] = pd.to_datetime(union_df['date'])
    
    # Sort by date (most recent first) and keep the first occurrence of each combination
    most_recent_df = (union_df
                     .sort_values(['benchmark_name', 'agent_name', 'date'], ascending=[True, True, False])
                     .groupby(['benchmark_name', 'agent_name'])
                     .first()
                     .reset_index())

    print(f"Rows before filtering: {len(union_df)}")
    print(f"Rows after filtering to most recent dates: {len(most_recent_df)}")
    
    # Split agent_name into Agent Scaffold and Model columns
    print("Splitting agent_name into Agent Scaffold and Model...")
    
    # Split agent_name by the first '(' character
    agent_split = most_recent_df['agent_name'].str.split('(', n=1, expand=True)
    
    # Create new columns
    most_recent_df['agent_scaffold'] = agent_split[0].str.strip()  # Remove trailing spaces
    most_recent_df['model'] = agent_split[1].apply(lambda x: '(' + x if pd.notna(x) else None)  # Add back the '('
    
    # Apply min-max scaling to accuracy and cost within each benchmark + agent scaffold combination
    print("Applying min-max scaling to accuracy and cost by benchmark + agent scaffold...")
    
    # Initialize new columns for scaled values
    most_recent_df['accuracy_scaled'] = np.nan
    most_recent_df['cost_scaled'] = np.nan
    
    # Group by benchmark_name and agent_scaffold
    grouped = most_recent_df.groupby(['benchmark_name', 'agent_scaffold'])
    
    for (benchmark, scaffold), group in grouped:
        if len(group) > 1:  # Only scale if there are multiple models to compare
            # Filter out null values for scaling
            accuracy_mask = group['accuracy'].notna()
            cost_mask = group['total_cost'].notna()
            
            if accuracy_mask.sum() > 1:  # Need at least 2 non-null values to scale
                scaler = MinMaxScaler()
                accuracy_values = group.loc[accuracy_mask, 'accuracy'].values.reshape(-1, 1)
                scaled_accuracy = scaler.fit_transform(accuracy_values).flatten()
                most_recent_df.loc[group.index[accuracy_mask], 'accuracy_scaled'] = scaled_accuracy
                
            if cost_mask.sum() > 1:  # Need at least 2 non-null values to scale
                scaler = MinMaxScaler()
                cost_values = group.loc[cost_mask, 'total_cost'].values.reshape(-1, 1)
                scaled_cost = scaler.fit_transform(cost_values).flatten()
                most_recent_df.loc[group.index[cost_mask], 'cost_scaled'] = scaled_cost
        else:
            # Single model - set scaled values to 0.5 (middle of 0-1 range)
            most_recent_df.loc[group.index, 'accuracy_scaled'] = 0.5
            most_recent_df.loc[group.index, 'cost_scaled'] = 0.5
    
    # Add task type mapping
    most_recent_df['task_type'] = most_recent_df['benchmark_name'].apply(map_task_type)
    
    # Add pretty benchmark labels
    most_recent_df['bench_label'] = most_recent_df['benchmark_name'].map(pretty_benchmark)

    # Total tokens
    most_recent_df['total_tokens'] = most_recent_df['prompt_tokens'] + most_recent_df['completion_tokens']

    # Create model release date by filtering after first ( and before second )
    most_recent_df['model_release_date'] = most_recent_df['model'].str.extract(r'\(([^)]+)\)')[0]

    # Clean up model names - remove " (Month Year)" suffix and outer parentheses
    most_recent_df['model'] = most_recent_df['model'].str.replace(
        r'\s*\([A-Za-z]+\s+\d{4}\)',  # a space + "(Month Year)"
        '',
        regex=True
    )
    most_recent_df['model'] = most_recent_df['model'].str.strip('()')
    
    print(f"\nFinal dataset shape: {most_recent_df.shape}")
    print(f"Date range: {most_recent_df['date'].min()} to {most_recent_df['date'].max()}")
    
    return most_recent_df

def load_paper_df():
    df = load_most_recent_df()

    # Remove benchmarks that contain 'colbench'
    df = df[~df['benchmark_name'].str.contains('colbench', case=False)]

    # Filter to a subset of models by name
    model_subset = [
        'Claude Opus 4.1', 'Claude Opus 4.1 High', 'Claude-3.7 Sonnet',
       'Claude-3.7 Sonnet High', 'DeepSeek R1', 'DeepSeek V3', 'GPT-4.1',
       'GPT-5 Medium', 'Gemini 2.0 Flash', 'o3 Medium', 'o4-mini High',
       'o4-mini Low'
    ]

    model_subset_df = df[df['model'].isin(model_subset)]

    rate_in  = 2 / 1_000_000   # $2 per million input tokens
    rate_out = 8 / 1_000_000   # $8 per million output tokens

    mask = model_subset_df["model"].eq("o3 Medium")

    model_subset_df.loc[mask, "total_cost"] = (
        rate_in  * model_subset_df.loc[mask, "prompt_tokens"]
    + rate_out * model_subset_df.loc[mask, "completion_tokens"]
    )

    # Remove generalist agent and secondary agent scaffolds
    model_df = model_subset_df[~model_subset_df['agent_scaffold'].str.contains('Generalist|Zero|SeeAct')]

    # filter to only corebench, taubench, and swebench for generalist comparison
    task_df_compare = model_df[model_df['benchmark_name'].str.contains('core|tau|swe')]

    # Only include generalist agent scaffold for corebench, taubench, and swebench
    generalist_df = model_subset_df[model_subset_df['agent_scaffold'].str.contains('Generalist') & model_subset_df['benchmark_name'].str.contains('core|tau|swe')]

    # Union task_df_compare and generalist_df
    agent_df = pd.concat([task_df_compare, generalist_df])

    # Create master dataframe including generalist agent and seeact
    all_task_df = df[~df['agent_scaffold'].str.contains('Generalist|Zero')]

    benchmark_df = pd.concat([all_task_df, generalist_df])

    return model_df, agent_df, benchmark_df


if __name__ == "__main__":
    # Example usage
    df = load_most_recent_df()
    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample of the results
    print(f"\nSample of the data:")
    display_cols = ['benchmark_name', 'agent_scaffold', 'model', 'accuracy', 'total_cost', 'prompt_tokens', 'completion_tokens', 'task_type']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head().to_string(index=False))