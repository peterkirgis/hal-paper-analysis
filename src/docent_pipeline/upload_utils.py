
from typing import Dict, Any, Optional, Set, List
import os
import time
from requests.exceptions import ConnectionError, Timeout, HTTPError

from docent import Docent
from docent.data_models import AgentRun, Transcript

DOCENT_API_KEY = os.getenv("DOCENT_API_KEY")

def _retry_with_backoff(func, max_retries=3, base_delay=2, *args, **kwargs):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout, HTTPError) as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"⚠️  Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"   Retrying in {delay}s...")
            time.sleep(delay)

def create_docent_client(api_key: str = None) -> Docent:
    """Create a Docent client."""
    if api_key is None:
        api_key = DOCENT_API_KEY
    return Docent(api_key=api_key)

def create_collection(client: Docent, name: str, description: str) -> str:
    """Create a new collection and return its ID with retry logic."""
    print(f"🔧 Creating collection: {name}")
    return _retry_with_backoff(client.create_collection, 3, 2, name=name, description=description)
    
def upload_transcripts(
    client: Docent,
    docent_results: Dict[str, Any], 
    collection_id: str, 
    batch_by_model: bool = True
) -> Dict[str, Any]:
    """
    Upload all transcripts from docent_results to a collection.
    
    Args:
        client: Docent client instance
        docent_results: Dictionary containing converted transcript data
        collection_id: The collection ID to upload to
        batch_by_model: If True, upload runs one model at a time
    
    Returns:
        Dictionary with upload statistics
    """
    upload_stats = {
        'total_runs': 0,
        'successful_uploads': 0,
        'failed_uploads': 0,
        'skipped_runs': 0,
        'failed_runs': []
    }
    
    print("🚀 Processing docent_results for upload...")
    
    if batch_by_model:
        return _upload_batched_by_model(client, docent_results, collection_id, upload_stats)
    else:
        return _upload_all_at_once(client, docent_results, collection_id, upload_stats)
    
def _upload_batched_by_model(
    client: Docent,
    docent_results: Dict[str, Any], 
    collection_id: str, 
    upload_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload runs batched by model."""
    # Group agent runs by model
    model_groups = {}
    
    for zip_name, tasks in docent_results.items():
        # Skip entries with errors
        if "error" in tasks:
            print(f"⚠️  Skipping {zip_name}: contains error")
            upload_stats['skipped_runs'] += 1
            continue
            
        for task_id, agent_run_data in tasks.items():
            model = agent_run_data.get('model', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append((zip_name, task_id, agent_run_data))
    
    print(f"📊 Found {len(model_groups)} models:")
    for model, runs in model_groups.items():
        print(f"   {model}: {len(runs)} runs")
    
    # Process each model group separately
    for model, runs in model_groups.items():
        print(f"\n🔄 Processing model: {model} ({len(runs)} runs)")
        agent_runs = []
        
        for zip_name, task_id, agent_run_data in runs:
            upload_stats['total_runs'] += 1
            
            try:
                agent_run = _create_agent_run(zip_name, task_id, agent_run_data)
                agent_runs.append(agent_run)
                    
            except Exception as e:
                print(f"❌ Failed to create AgentRun for task {task_id[:12]}...: {e}")
                upload_stats['failed_uploads'] += 1
                upload_stats['failed_runs'].append({
                    'task_id': task_id,
                    'zip_name': zip_name,
                    'error': str(e)
                })
                continue
        
        # Upload this model's runs
        if agent_runs:
            try:
                # Upload in smaller batches to avoid connection issues
                batch_size = min(50, len(agent_runs))  # Limit batch size
                total_uploaded = 0
                
                for i in range(0, len(agent_runs), batch_size):
                    batch = agent_runs[i:i + batch_size]
                    print(f"   🔄 Uploading batch {i//batch_size + 1} ({len(batch)} runs) for {model}...")
                    
                    _retry_with_backoff(client.add_agent_runs, 3, 2, collection_id, batch)
                    total_uploaded += len(batch)
                    
                    # Small delay between batches to avoid overwhelming the server
                    if i + batch_size < len(agent_runs):
                        time.sleep(1)
                
                upload_stats['successful_uploads'] += total_uploaded
                print(f"   ✅ Successfully uploaded {total_uploaded} runs for {model}!")
                
            except Exception as e:
                print(f"   ❌ Failed to upload runs for {model}: {e}")
                upload_stats['failed_uploads'] += len(agent_runs)
                for agent_run in agent_runs:
                    upload_stats['failed_runs'].append({
                        'agent_run_id': agent_run.id,
                        'model': model,
                        'error': str(e)
                    })
    
    _print_upload_stats(upload_stats)
    return upload_stats
    
def _upload_all_at_once(
    client: Docent,
    docent_results: Dict[str, Any], 
    collection_id: str, 
    upload_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload all runs at once."""
    agent_runs = []
    
    for zip_name, tasks in docent_results.items():
        # Skip entries with errors
        if "error" in tasks:
            print(f"⚠️  Skipping {zip_name}: contains error")
            upload_stats['skipped_runs'] += 1
            continue
            
        print(f"📁 Processing {zip_name}...")
        
        for task_id, agent_run_data in tasks.items():
            upload_stats['total_runs'] += 1
            
            try:
                agent_run = _create_agent_run(zip_name, task_id, agent_run_data)
                agent_runs.append(agent_run)
                
                if len(agent_runs) % 10 == 0:
                    print(f"   ✅ Prepared {len(agent_runs)} agent runs...")
                    
            except Exception as e:
                print(f"❌ Failed to create AgentRun for task {task_id[:12]}...: {e}")
                upload_stats['failed_uploads'] += 1
                upload_stats['failed_runs'].append({
                    'task_id': task_id,
                    'zip_name': zip_name,
                    'error': str(e)
                })
                continue
    
    print(f"📊 Prepared {len(agent_runs)} agent runs for upload")
    
    # Upload all agent runs to the collection
    if agent_runs:
        try:
            # Upload in smaller batches to avoid connection issues
            batch_size = min(50, len(agent_runs))  # Limit batch size
            total_uploaded = 0
            
            for i in range(0, len(agent_runs), batch_size):
                batch = agent_runs[i:i + batch_size]
                print(f"🔄 Uploading batch {i//batch_size + 1} ({len(batch)} runs)...")
                
                _retry_with_backoff(client.add_agent_runs, 3, 2, collection_id, batch)
                total_uploaded += len(batch)
                
                # Small delay between batches to avoid overwhelming the server
                if i + batch_size < len(agent_runs):
                    time.sleep(1)
            
            upload_stats['successful_uploads'] = total_uploaded
            print(f"✅ Successfully uploaded {total_uploaded} agent runs!")
            
        except Exception as e:
            print(f"❌ Failed to upload agent runs: {e}")
            upload_stats['failed_uploads'] += len(agent_runs)
            for agent_run in agent_runs:
                upload_stats['failed_runs'].append({
                    'agent_run_id': agent_run.id,
                    'error': str(e)
                })
    
    _print_upload_stats(upload_stats)
    return upload_stats
    
def _create_agent_run(zip_name: str, task_id: str, agent_run_data: Dict[str, Any]) -> AgentRun:
    """Create an AgentRun from agent run data."""

    # Get benchmark label by splitting at the first underscore
    benchmark_label = zip_name.split('_')[0]
    # Extract metadata
    metadata = {
        "benchmark_id": benchmark_label,
        "task_id": task_id,
        "model": agent_run_data.get('model', 'unknown'),
        "run_id": zip_name,
        "weave_task_id": agent_run_data.get('weave_task_id'),
        "original_message_count": agent_run_data['original_message_count'],
        "docent_message_count": agent_run_data['docent_message_count'],
        "failed_message_count": agent_run_data['failed_message_count']
    }
    
    # Add raw eval results to metadata if available
    eval_data = agent_run_data.get('eval', {})
    if eval_data and 'raw_results' in eval_data:
        raw_results = eval_data['raw_results']
        # Flatten raw results into metadata with clear prefixes
        for key, value in raw_results.items():
            if not key.startswith('global_'):
                # These are task-specific results like scores, answers
                metadata[f"eval_{key}"] = value
            else:
                # These are global/benchmark metadata 
                metadata[key] = value
    
    # Create transcript from docent messages
    transcript = Transcript(
        messages=agent_run_data['docent_messages'],
        metadata=metadata
    )
    
    # Create transcripts dict (AgentRun expects plural)
    transcripts = {
        "default": transcript
    }
    
    # Create AgentRun
    return AgentRun(
        transcripts=transcripts,
        metadata=metadata
    )

def _print_upload_stats(upload_stats: Dict[str, Any]):
    """Print upload statistics."""
    print(f"\n📈 Upload Statistics:")
    print(f"   Total runs processed: {upload_stats['total_runs']}")
    print(f"   Successfully uploaded: {upload_stats['successful_uploads']}")
    print(f"   Failed uploads: {upload_stats['failed_uploads']}")
    print(f"   Skipped runs: {upload_stats['skipped_runs']}")
    
    if upload_stats['failed_runs']:
        print(f"   Failed runs: {len(upload_stats['failed_runs'])}")