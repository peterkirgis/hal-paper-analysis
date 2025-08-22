import json
import os
from typing import Dict, Any, Optional, Set
from tqdm import tqdm
from dotenv import load_dotenv

from src.docent_pipeline.download_utils import (
    create_hf_client, find_files_by_prefix, stream_agent_runs_by_task
)
from src.docent_pipeline.processing_utils import convert_to_docent_messages
from src.docent_pipeline.upload_utils import (
    create_docent_client, create_collection, upload_transcripts
)
from src.docent_pipeline.config import (
    BENCHMARK_AGENT_PREFIX,
    EXCLUDED_FILES,
    DEFAULT_TASK_LIMIT,
    DEFAULT_OUTPUT_FILE,
    COLLECTION_NAME
)

load_dotenv()

def find_browser_agent_files(
    client_config: Dict[str, Any],
    prefix: str = BENCHMARK_AGENT_PREFIX,
    excluded_files: Optional[Set[str]] = None
) -> list:
    """Find all browser agent ZIP files to process."""
    if excluded_files is None:
        excluded_files = EXCLUDED_FILES
    
    files = find_files_by_prefix(client_config, prefix, excluded_files)
    
    print(f"Found {len(files)} AssistantBench browser agent ZIP files:")
    for i, file_info in enumerate(files):
        size_mb = file_info['size'] / (1024 * 1024)
        print(f"  {i+1:2d}. {file_info['name']:<80} ({size_mb:>8.1f} MB)")
    
    return files


def process_all_files(
    client_config: Dict[str, Any],
    files: Optional[list] = None,
    task_limit: Optional[int] = DEFAULT_TASK_LIMIT,
    require_model: Optional[str] = None,
    include_eval: bool = True,
    include_eval_results: bool = True
) -> Dict[str, Any]:
    """
    Process all ZIP files and extract agent runs.
    
    Args:
        client_config: HuggingFace client configuration
        files: List of file info dicts (if None, will find files automatically)
        task_limit: Maximum tasks per file to process (None for all)
        require_model: Only process runs with this model
        include_eval: Include evaluation data in results
        include_eval_results: Include raw evaluation results (scores, answers) mapped by task index
        
    Returns:
        Dictionary mapping zip_name -> {task_id -> agent_run}
    """
    if files is None:
        files = find_browser_agent_files(client_config)
    
    all_results = {}
    overall_progress = tqdm(files, desc="Processing ZIP files")

    for file_info in overall_progress:
        zip_name = file_info['name']
        overall_progress.set_description(f"Processing {zip_name[:50]}...")
        
        try:
            zip_results = {}
            
            for tid, agent_run in stream_agent_runs_by_task(
                client_config=client_config,
                zip_name=zip_name,
                member_name=None,
                require_model=require_model,
                include_eval=include_eval,
                include_eval_results=include_eval_results,
                limit=task_limit,
                aggregate_all=True,
            ):
                zip_results[tid] = agent_run
            
            all_results[zip_name] = zip_results
            
            print(f"\n✅ {zip_name}: {len(zip_results)} tasks processed")
            
            # Show sample info
            if zip_results:
                sample_task_id = next(iter(zip_results))
                sample_run = zip_results[sample_task_id]
                print(f"   Sample: Task {sample_task_id[:12]}... has {len(sample_run.get('messages', []))} messages")
                if sample_run.get('model'):
                    print(f"   Model: {sample_run['model']}")
            
        except Exception as e:
            print(f"\n❌ Error processing {zip_name}: {e}")
            all_results[zip_name] = {"error": str(e)}
            continue

    overall_progress.close()

    print(f"\n🎉 Processing complete!")
    print(f"Total files processed: {len(all_results)}")
    successful_files = sum(1 for v in all_results.values() if "error" not in v)
    print(f"Successful files: {successful_files}")
    total_tasks = sum(len(v) for v in all_results.values() if "error" not in v)
    print(f"Total tasks extracted: {total_tasks:,}")
    
    return all_results


def save_results(
    results: Dict[str, Any], 
    output_file: str = DEFAULT_OUTPUT_FILE
) -> None:
    """Save results to JSON file."""
    def _json_default(o):
        """JSON serializer for special types."""
        import base64
        from datetime import date, datetime
        from decimal import Decimal
        
        if isinstance(o, Decimal):
            return format(o, "f")
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("ascii")
        
        try:
            import numpy as np
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        
        return str(o)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=_json_default, sort_keys=True)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved complete results to: {output_file}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    total_files = len(results)
    total_tasks = sum(len(v) for v in results.values() if isinstance(v, dict) and "error" not in v)
    print(f"📋 Contains {total_files} ZIP files with {total_tasks} total tasks")


def upload_to_docent(
    docent_client,
    docent_results: Dict[str, Any], 
    collection_name: str = COLLECTION_NAME,
    collection_description: str = "",
    batch_by_model: bool = True
) -> tuple:
    """
    Upload results to a docent collection.
    
    Args:
        docent_client: Docent client instance
        docent_results: Converted docent format results
        collection_name: Name for the new collection
        collection_description: Description for the collection
        batch_by_model: Whether to batch uploads by model
        
    Returns:
        Tuple of (collection_id, upload_stats)
    """
    # Create collection
    collection_id = create_collection(
        docent_client,
        name=collection_name,
        description=collection_description or f"HAL paper analysis: {collection_name}"
    )
    
    # Upload transcripts
    upload_stats = upload_transcripts(
        docent_client, 
        docent_results, 
        collection_id, 
        batch_by_model=batch_by_model
    )
    
    return collection_id, upload_stats


def run_full_pipeline(
    collection_name: str = COLLECTION_NAME,
    hf_token: Optional[str] = None,
    docent_api_key: Optional[str] = None,
    output_file: Optional[str] = None,
    task_limit: Optional[int] = DEFAULT_TASK_LIMIT,
    require_model: Optional[str] = None,
    include_eval: bool = True,
    batch_by_model: bool = True,
    save_intermediate: bool = True
) -> tuple:
    """
    Run the complete pipeline from data extraction to docent upload.
    
    Args:
        collection_name: Name for the docent collection
        hf_token: HuggingFace token (optional, will use env var if not provided)
        docent_api_key: Docent API key (optional, will use env var if not provided) 
        output_file: Optional file to save raw results
        task_limit: Limit on tasks per file
        require_model: Only process specific model
        include_eval: Include evaluation data
        batch_by_model: Batch uploads by model
        save_intermediate: Save intermediate results to file
        
    Returns:
        Tuple of (collection_id, upload_stats, docent_results)
    """
    print("🚀 Starting full HAL pipeline...")
    
    # Set up environment
    hf_token = hf_token or os.getenv("HF_TOKEN")
        
    # Create clients
    hf_client_config = create_hf_client()
    docent_client = create_docent_client(docent_api_key or os.getenv("DOCENT_API_KEY"))
    
    # Step 1: Process all files
    print("\n📁 Step 1: Processing ZIP files...")
    all_results = process_all_files(
        hf_client_config,
        task_limit=task_limit,
        require_model=require_model,
        include_eval=include_eval,
        include_eval_results=True  # Enable raw eval results extraction
    )
    
    # Step 2: Save intermediate results
    if save_intermediate:
        intermediate_file = output_file or DEFAULT_OUTPUT_FILE
        print(f"\n💾 Step 2: Saving intermediate results to {intermediate_file}...")
        save_results(all_results, intermediate_file)
    
    # Step 3: Convert to docent format
    print("\n🔄 Step 3: Converting to docent format...")
    docent_results, conversion_stats = convert_to_docent_messages(all_results)
    
    print(f"✅ Conversion complete!")
    print(f"📊 Conversion stats:")
    print(f"   Total messages: {conversion_stats['total_messages']}")
    print(f"   Successfully converted: {conversion_stats['successful']}")
    print(f"   Failed to convert: {conversion_stats['failed']}")
    if conversion_stats['total_messages'] > 0:
        success_rate = conversion_stats['successful']/conversion_stats['total_messages']*100
        print(f"   Success rate: {success_rate:.1f}%")
    
    # Step 4: Upload to docent
    print(f"\n📤 Step 4: Uploading to docent collection '{collection_name}'...")
    collection_id, upload_stats = upload_to_docent(
        docent_client,
        docent_results,
        collection_name,
        batch_by_model=batch_by_model
    )
    
    print(f"\n🎉 Pipeline complete!")
    print(f"📋 Collection ID: {collection_id}")
    
    return collection_id, upload_stats, docent_results


if __name__ == "__main__":
    
    # Run the full pipeline
    collection_id, upload_stats, docent_results = run_full_pipeline()
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"Collection ID: {collection_id}")
    print(f"Upload stats: {upload_stats}")