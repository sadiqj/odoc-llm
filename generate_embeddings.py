#!/usr/bin/env python3
"""
Generate embeddings for OCaml module descriptions using OpenAI-compatible API.

This script processes module descriptions from the module-descriptions/ directory
and generates embeddings using a local vLLM server. Embeddings are stored as
numpy arrays with JSON metadata in the package_embeddings/ directory.
"""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

try:
    import requests
except ImportError:
    print("Error: requests package not found. Install with: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm package not found. Install with: pip install tqdm")
    sys.exit(1)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logging.info(f"Received signal {signum}, initiating graceful shutdown...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    llm_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    workers: int = 12
    batch_size: int = 32
    rate_limit: float = 10.0  # requests per second
    output_dir: str = "package_embeddings"
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    resume: bool = False


class RateLimiter:
    """Thread-safe rate limiter for API requests."""
    
    def __init__(self, max_requests_per_second: float = 10.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_times = {}
        self.lock = threading.Lock()
    
    def acquire(self, worker_id: str):
        """Acquire permission to make a request, blocking if necessary."""
        with self.lock:
            now = time.time()
            if worker_id in self.last_request_times:
                elapsed = now - self.last_request_times[worker_id]
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    time.sleep(sleep_time)
            self.last_request_times[worker_id] = time.time()


class EmbeddingClient:
    """Client for the custom embedding endpoint."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.endpoint_url = f"{config.llm_url.rstrip('/')}/embedding"
        self.stats = {
            'requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_texts': 0,
            'total_tokens': 0
        }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts with retry logic."""
        if not texts:
            return []
        
        # For the /embedding endpoint, we need to process texts one by one
        all_embeddings = []
        
        for text in texts:
            for attempt in range(self.config.max_retries):
                try:
                    response = requests.post(
                        self.endpoint_url,
                        json={"content": text},
                        headers={"Content-Type": "application/json"},
                        timeout=self.config.timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Extract embedding from response format: [{"index": 0, "embedding": [[...]]}]
                        if data and len(data) > 0 and "embedding" in data[0]:
                            embedding = data[0]["embedding"][0]  # Unwrap nested array
                            all_embeddings.append(embedding)
                            break
                        else:
                            raise ValueError(f"Unexpected response format: {data}")
                    else:
                        raise ValueError(f"HTTP {response.status_code}: {response.text}")
                        
                except Exception as e:
                    self.stats['requests'] += 1
                    self.stats['failed_requests'] += 1
                    
                    if attempt == self.config.max_retries - 1:
                        logging.error(f"Failed to get embedding for text after {self.config.max_retries} attempts: {e}")
                        raise
                    else:
                        wait_time = (2 ** attempt) * 1.0  # exponential backoff
                        logging.warning(f"Embedding request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
        
        # Update stats
        self.stats['requests'] += len(texts)
        self.stats['successful_requests'] += len(all_embeddings)
        self.stats['total_texts'] += len(all_embeddings)
        
        return all_embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return self.stats.copy()


class ProgressCheckpoint:
    """Manages progress checkpointing for resume capability."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.completed_packages = set()
        self.failed_packages = set()
        self.lock = threading.Lock()
    
    def load_checkpoint(self):
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                    self.completed_packages = set(data.get("completed", []))
                    self.failed_packages = set(data.get("failed", []))
                    logging.info(f"Loaded checkpoint: {len(self.completed_packages)} completed, {len(self.failed_packages)} failed")
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        with self.lock:
            try:
                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint_file, 'w') as f:
                    json.dump({
                        "completed": list(self.completed_packages),
                        "failed": list(self.failed_packages),
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
    
    def mark_completed(self, package_name: str):
        """Mark a package as completed."""
        with self.lock:
            self.completed_packages.add(package_name)
            self.failed_packages.discard(package_name)
    
    def mark_failed(self, package_name: str):
        """Mark a package as failed."""
        with self.lock:
            self.failed_packages.add(package_name)
    
    def is_completed(self, package_name: str) -> bool:
        """Check if a package is already completed."""
        with self.lock:
            return package_name in self.completed_packages
    
    def is_failed(self, package_name: str) -> bool:
        """Check if a package previously failed."""
        with self.lock:
            return package_name in self.failed_packages


class ProgressTracker:
    """Tracks and displays progress with ETA calculation."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total_tasks, desc="Processing packages")
        self.stats = defaultdict(int)
    
    def update_progress(self, status: str, package_name: str = None, stats: Dict = None):
        """Update progress based on task status."""
        with self.lock:
            if status == 'completed':
                self.completed += 1
                self.in_progress -= 1
                self.pbar.update(1)
            elif status == 'failed':
                self.failed += 1
                self.in_progress -= 1
                self.pbar.update(1)
            elif status == 'started':
                self.in_progress += 1
            
            # Update statistics
            if stats:
                for key, value in stats.items():
                    self.stats[key] += value
            
            # Update progress bar description
            self.pbar.set_description(
                f"Processing packages (✓{self.completed} ✗{self.failed} ⧗{self.in_progress})"
            )
    
    def get_eta(self) -> str:
        """Calculate estimated time of arrival."""
        with self.lock:
            if self.completed == 0:
                return "Unknown"
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed
            remaining = (self.total_tasks - self.completed - self.failed) / rate
            return str(timedelta(seconds=int(remaining)))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'total_tasks': self.total_tasks,
                'completed': self.completed,
                'failed': self.failed,
                'in_progress': self.in_progress,
                'elapsed_time': elapsed,
                'eta': self.get_eta(),
                'completion_rate': self.completed / elapsed if elapsed > 0 else 0,
                'stats': dict(self.stats)
            }
    
    def close(self):
        """Close the progress bar."""
        self.pbar.close()


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.completed = set()
        self.failed = set()
        self.in_progress = set()
        self.lock = threading.Lock()
    
    def add_task(self, package_path: Path, priority: int = 0):
        """Add a task to the queue with priority."""
        # Priority based on file size (smaller files first for quick wins)
        try:
            file_size = package_path.stat().st_size
            priority_score = priority + (file_size // 1024)  # Size in KB
            self.queue.put((priority_score, str(package_path)))
        except OSError:
            # Handle case where file doesn't exist
            self.queue.put((priority, str(package_path)))
    
    def get_task(self, timeout: float = 5.0) -> Optional[Path]:
        """Get the next task from the queue."""
        try:
            priority, task_path = self.queue.get(timeout=timeout)
            task = Path(task_path)
            with self.lock:
                self.in_progress.add(task.stem)
            return task
        except queue.Empty:
            return None
    
    def mark_done(self, task_path: Path, success: bool = True):
        """Mark a task as completed."""
        with self.lock:
            package_name = task_path.stem
            self.in_progress.discard(package_name)
            if success:
                self.completed.add(package_name)
            else:
                self.failed.add(package_name)


def is_meaningful_description(description: str) -> bool:
    """Check if a module description contains meaningful content."""
    if not description or len(description.strip()) < 50:
        return False
    
    desc_lower = description.lower().strip()
    
    # Exact empty pattern (most common)
    if "empty module with no functions, types, or documentation" in description:
        return False
    
    # Common empty patterns
    empty_patterns = [
        "contains no",
        "provides no", 
        "serves as a placeholder",
        "no examples applicable",
        "no functional capabilities",
        "no computational role"
    ]
    
    if any(pattern in desc_lower for pattern in empty_patterns):
        return False
    
    # Too short to be meaningful (less than 100 chars)
    if len(description) < 100:
        return False
        
    return True


def clean_description(text: str) -> str:
    """Clean and normalize description text."""
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    return text


def load_package_data(package_path: Path) -> Dict[str, Any]:
    """Load package data from JSON file."""
    try:
        with open(package_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load package data from {package_path}: {e}")
        return {}


def process_single_package(
    package_path: Path,
    client: EmbeddingClient,
    config: EmbeddingConfig,
    rate_limiter: RateLimiter,
    worker_id: str
) -> Dict[str, Any]:
    """Process a single package to generate embeddings."""
    package_name = package_path.stem
    
    try:
        # Load package data
        package_data = load_package_data(package_path)
        if not package_data:
            return {"success": False, "error": "Failed to load package data"}
        
        descriptions = package_data.get("descriptions", {})
        if not descriptions:
            logging.warning(f"No descriptions found in {package_name}")
            return {"success": False, "error": "No descriptions found"}
        
        # Extract and filter meaningful module descriptions
        modules = []
        texts = []
        filtered_count = 0
        
        for module_path, description in descriptions.items():
            # Check if description is meaningful before processing
            if not is_meaningful_description(description):
                filtered_count += 1
                continue
                
            cleaned_text = clean_description(description)
            if cleaned_text:
                modules.append({
                    "module_path": module_path,
                    "description": cleaned_text,
                    "description_length": len(cleaned_text)
                })
                texts.append(cleaned_text)
        
        if not texts:
            total_modules = len(descriptions)
            return {"success": False, "error": f"No meaningful descriptions found ({filtered_count}/{total_modules} filtered as empty)"}
        
        # Log filtering statistics
        total_modules = len(descriptions)
        meaningful_modules = len(texts)
        if filtered_count > 0:
            logging.info(f"Package {package_name}: {meaningful_modules}/{total_modules} modules meaningful, filtered {filtered_count} empty modules")
        
        # Generate embeddings in batches
        all_embeddings = []
        batch_size = config.batch_size
        
        for i in range(0, len(texts), batch_size):
            if shutdown_requested:
                return {"success": False, "error": "Shutdown requested"}
            
            batch_texts = texts[i:i + batch_size]
            
            # Apply rate limiting
            rate_limiter.acquire(worker_id)
            
            # Get embeddings for batch
            try:
                batch_embeddings = client.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Failed to get embeddings for {package_name} batch {i//batch_size}: {e}")
                return {"success": False, "error": f"Embedding generation failed: {e}"}
        
        if len(all_embeddings) != len(modules):
            return {"success": False, "error": f"Embedding count mismatch: {len(all_embeddings)} vs {len(modules)}"}
        
        # Create output directory
        output_dir = Path(config.output_dir) / "packages" / package_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as NPZ
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        embeddings_file = output_dir / "embeddings.npz"
        np.savez_compressed(
            embeddings_file,
            embeddings=embeddings_array,
            module_indices=np.arange(len(modules), dtype=np.int32)
        )
        
        # Add embedding norms to module metadata
        for i, module in enumerate(modules):
            module["index"] = i
            module["embedding_norm"] = float(np.linalg.norm(embeddings_array[i]))
        
        # Create metadata
        metadata = {
            "package": package_name,
            "embedding_model": config.model,
            "embedding_dimension": len(all_embeddings[0]),
            "total_modules": len(modules),
            "creation_timestamp": datetime.now().isoformat(),
            "modules": modules,
            "filtering": {
                "total_modules_in_package": total_modules,
                "meaningful_modules": meaningful_modules,
                "filtered_empty_modules": filtered_count,
                "retention_rate": meaningful_modules / total_modules if total_modules > 0 else 0
            },
            "statistics": {
                "max_description_length": max(m["description_length"] for m in modules),
                "min_description_length": min(m["description_length"] for m in modules),
                "avg_description_length": sum(m["description_length"] for m in modules) / len(modules),
                "embedding_file_size_mb": embeddings_file.stat().st_size / 1024 / 1024
            }
        }
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "modules_count": len(modules),
            "embedding_dimension": len(all_embeddings[0]),
            "file_size_mb": embeddings_file.stat().st_size / 1024 / 1024
        }
        
    except Exception as e:
        logging.error(f"Error processing package {package_name}: {e}")
        return {"success": False, "error": str(e)}


def package_worker(
    task_queue: TaskQueue,
    progress_tracker: ProgressTracker,
    checkpoint: ProgressCheckpoint,
    config: EmbeddingConfig,
    worker_id: int
) -> None:
    """Worker function for processing packages."""
    worker_name = f"worker-{worker_id}"
    client = EmbeddingClient(config)
    rate_limiter = RateLimiter(config.rate_limit / config.workers)  # Distribute rate limit
    
    logging.info(f"Worker {worker_name} starting")
    
    while not shutdown_requested:
        # Get next task
        task = task_queue.get_task()
        if task is None:
            logging.info(f"Worker {worker_name} finished - no more tasks")
            break
        
        package_name = task.stem
        
        try:
            # Check if already completed
            if checkpoint.is_completed(package_name):
                logging.info(f"Skipping {package_name} - already completed")
                task_queue.mark_done(task, success=True)
                progress_tracker.update_progress('completed', package_name)
                continue
            
            # Update progress
            progress_tracker.update_progress('started', package_name)
            
            # Process package
            result = process_single_package(task, client, config, rate_limiter, worker_name)
            
            if result["success"]:
                checkpoint.mark_completed(package_name)
                task_queue.mark_done(task, success=True)
                progress_tracker.update_progress('completed', package_name, {
                    'modules': result.get('modules_count', 0),
                    'embedding_mb': result.get('file_size_mb', 0)
                })
                logging.info(f"Completed {package_name}: {result.get('modules_count', 0)} modules")
            else:
                checkpoint.mark_failed(package_name)
                task_queue.mark_done(task, success=False)
                progress_tracker.update_progress('failed', package_name)
                logging.error(f"Failed {package_name}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logging.error(f"Worker {worker_name} error processing {package_name}: {e}")
            checkpoint.mark_failed(package_name)
            task_queue.mark_done(task, success=False)
            progress_tracker.update_progress('failed', package_name)
    
    # Log worker stats
    stats = client.get_stats()
    logging.info(f"Worker {worker_name} completed. Stats: {stats}")


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('embedding_generation.log')
        ]
    )


def create_global_metadata(config: EmbeddingConfig, packages: List[Path], results: Dict[str, Any]):
    """Create global metadata file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count successful packages
    successful_packages = {}
    total_modules = 0
    total_size_mb = 0
    
    for package_path in packages:
        package_name = package_path.stem
        package_dir = output_dir / "packages" / package_name
        metadata_file = package_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    successful_packages[package_name] = {
                        "modules_count": metadata["total_modules"],
                        "file_size_mb": metadata["statistics"]["embedding_file_size_mb"],
                        "last_updated": metadata["creation_timestamp"]
                    }
                    total_modules += metadata["total_modules"]
                    total_size_mb += metadata["statistics"]["embedding_file_size_mb"]
            except Exception as e:
                logging.error(f"Failed to read metadata for {package_name}: {e}")
    
    # Create global metadata
    global_metadata = {
        "format_version": "1.0",
        "embedding_model": config.model,
        "embedding_dimension": results.get('embedding_dimension', 0),
        "total_packages": len(successful_packages),
        "total_modules": total_modules,
        "total_size_mb": total_size_mb,
        "creation_timestamp": datetime.now().isoformat(),
        "generation_config": {
            "workers": config.workers,
            "batch_size": config.batch_size,
            "rate_limit": config.rate_limit
        },
        "packages": successful_packages
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    logging.info(f"Created global metadata for {len(successful_packages)} packages")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate embeddings for OCaml module descriptions")
    parser.add_argument("--llm-url", default="http://localhost:8000", help="LLM API endpoint URL")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for API requests")
    parser.add_argument("--rate-limit", type=float, default=10.0, help="API requests per second")
    parser.add_argument("--output-dir", default="package_embeddings", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--packages", help="Comma-separated list of specific packages to process")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EmbeddingConfig(
        llm_url=args.llm_url,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        output_dir=args.output_dir,
        resume=args.resume,
        log_level=args.log_level
    )
    
    # Setup logging
    setup_logging(config.log_level)
    logging.info(f"Starting embedding generation with config: {config}")
    
    # Find package files
    module_descriptions_dir = Path("module-descriptions")
    if not module_descriptions_dir.exists():
        logging.error(f"Module descriptions directory not found: {module_descriptions_dir}")
        return 1
    
    # Get all package files
    all_packages = list(module_descriptions_dir.glob("*.json"))
    
    # Filter packages if specified
    if args.packages:
        specified_packages = set(args.packages.split(","))
        all_packages = [p for p in all_packages if p.stem in specified_packages]
        logging.info(f"Processing {len(all_packages)} specified packages")
    
    if not all_packages:
        logging.error("No packages found to process")
        return 1
    
    logging.info(f"Found {len(all_packages)} packages to process")
    
    # Initialize components
    checkpoint_file = Path(config.output_dir) / "checkpoint.json"
    checkpoint = ProgressCheckpoint(checkpoint_file)
    
    if config.resume:
        checkpoint.load_checkpoint()
    
    # Filter out completed packages
    packages_to_process = [p for p in all_packages if not checkpoint.is_completed(p.stem)]
    logging.info(f"Processing {len(packages_to_process)} packages (excluding completed)")
    
    if not packages_to_process:
        logging.info("All packages already completed")
        return 0
    
    # Create task queue and progress tracker
    task_queue = TaskQueue()
    progress_tracker = ProgressTracker(len(packages_to_process))
    
    # Add tasks to queue
    for package_path in packages_to_process:
        task_queue.add_task(package_path)
    
    # Start workers
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            # Submit worker tasks
            futures = []
            for i in range(config.workers):
                future = executor.submit(
                    package_worker,
                    task_queue,
                    progress_tracker,
                    checkpoint,
                    config,
                    i
                )
                futures.append(future)
            
            # Monitor and save checkpoints periodically
            checkpoint_counter = 0
            while not all(future.done() for future in futures):
                time.sleep(5)  # Check every 5 seconds
                
                checkpoint_counter += 1
                if checkpoint_counter >= 12:  # Save every minute (5s * 12 = 60s)
                    checkpoint.save_checkpoint()
                    checkpoint_counter = 0
                
                if shutdown_requested:
                    logging.info("Shutdown requested, waiting for workers to finish...")
                    break
            
            # Wait for all workers to complete
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logging.error(f"Worker error: {e}")
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Final checkpoint save
        checkpoint.save_checkpoint()
        
        # Get final results
        summary = progress_tracker.get_summary()
        progress_tracker.close()
        
        logging.info(f"Processing completed. Summary: {summary}")
        
        # Create global metadata
        create_global_metadata(config, all_packages, summary)
        
        # Print final statistics
        print(f"\n=== Final Results ===")
        print(f"Total packages: {len(all_packages)}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['completed']/(summary['completed']+summary['failed'])*100:.1f}%")
        print(f"Total modules: {summary['stats'].get('modules', 0)}")
        print(f"Total size: {summary['stats'].get('embedding_mb', 0):.1f} MB")
        print(f"Processing time: {timedelta(seconds=int(summary['elapsed_time']))}")
        
        return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())