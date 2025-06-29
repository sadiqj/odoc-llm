#!/usr/bin/env python3
"""
Generate semantic descriptions for OCaml modules using LLM.
Hierarchically merges module descriptions into package summaries.
"""

import json
import os
import logging
import re
import queue
import threading
import psutil
import gc
import signal
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence HTTP request logging from OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global exception and signal handlers
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))
    logger.error(f"Exception type: {exc_type}")
    logger.error(f"Exception value: {exc_value}")
    logger.error(f"Exception traceback: {''.join(traceback.format_tb(exc_traceback))}")

def handle_signal(signum, frame):
    logger.error(f"RECEIVED SIGNAL {signum} at frame {frame}")
    logger.error(f"Signal name: {signal.Signals(signum).name}")
    logger.error(f"Current thread: {threading.current_thread()}")
    logger.error(f"Active threads: {threading.active_count()}")
    for thread in threading.enumerate():
        logger.error(f"Thread: {thread.name}, alive: {thread.is_alive()}")
    
    # Print stack traces for all threads
    for thread_id, frame in sys._current_frames().items():
        logger.error(f"Stack trace for thread {thread_id}:")
        logger.error(''.join(traceback.format_stack(frame)))
    
    raise KeyboardInterrupt(f"Signal {signum}")

# Install global handlers
sys.excepthook = handle_exception
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)
signal.signal(signal.SIGQUIT, handle_signal)

@dataclass
class ModuleContent:
    """Extracted content from a module."""
    name: str
    path: str
    functions: List[Dict[str, Any]]
    types: List[Dict[str, Any]]
    modules: List[Dict[str, Any]]
    documentation: str
    parent: Optional[str] = None
    children: List[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

class LLMClient:
    """OpenAI-compatible client for generating descriptions."""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen/Qwen3-30B-A3B-FP8"):
        try:
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="dummy_key",  # Local endpoint doesn't need real key
                timeout=120.0,  # 2 minute timeout for requests
                max_retries=1  # Only retry once on failure
            )
            self.model = model
            logger.info(f"LLMClient initialized with base_url={base_url}, model={model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_module_description(self, module: ModuleContent, log_prompts: bool = False) -> str:
        """Generate a concise description for a single module using chunking strategy."""
        
        # If module has many functions, use chunking strategy
        all_items = module.functions + module.types
        
        if len(all_items) <= 20:
            return self._generate_simple_description(module, log_prompts)
        else:
            return self._generate_chunked_description(module, log_prompts)
    
    def _generate_simple_description(self, module: ModuleContent, log_prompts: bool = False) -> str:
        """Generate description for modules with ≤20 functions/types."""
        
        context_parts = [f"Module: {module.name}"]
        
        if module.documentation:
            context_parts.append(f"Module Documentation: {module.documentation}")
        
        # Include all functions with their documentation
        if module.functions:
            context_parts.append("Functions:")
            for func in module.functions:
                func_line = f"- {func.get('signature', func.get('name', 'unnamed'))}"
                if func.get('documentation'):
                    func_line += f" // {func['documentation']}"
                context_parts.append(func_line)
        
        # Include all types with their documentation  
        if module.types:
            context_parts.append("Types:")
            for typ in module.types:
                type_line = f"- {typ.get('signature', typ.get('name', 'unnamed'))}"
                if typ.get('documentation'):
                    type_line += f" // {typ['documentation']}"
                context_parts.append(type_line)
        
        if module.modules:
            submodule_names = [m.get("name", "unnamed") for m in module.modules[:8]]
            context_parts.append(f"Submodules: {', '.join(submodule_names)}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Write a 2-3 sentence description focusing on:
- The specific operations and functions this module provides
- What data types or structures it works with
- Concrete use cases (avoid generic terms like "utility functions" or "common operations")

Do NOT:
- Repeat the module name in the description
- Mention "functional programming patterns" or "code clarity"
- Use filler phrases like "provides functionality for" or "collection of functions"
- Describe how it works with other modules

{context}

Description:"""

        if log_prompts:
            logger.info(f"=== PROMPT for {module.name} ===")
            logger.info(prompt)
            logger.info("=== END PROMPT ===")
        
        try:
            logger.info(f"Sending LLM request for module {module.name} (length: {len(prompt)} chars)")
            import time
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert OCaml developer. Provide concise, direct answers without thinking aloud or explanation. /no_think"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            elapsed = time.time() - start_time
            logger.info(f"LLM request completed for module {module.name} in {elapsed:.1f}s")
            
            if not response.choices:
                logger.error(f"LLM returned no choices for module {module.name}")
                return f"OCaml module {module.name} - no response from LLM"
            
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== RESPONSE for {module.name} ===")
                logger.info(result)
                logger.info("=== END RESPONSE ===")
            return result
        except Exception as e:
            import time
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"LLM error for module {module.name} after {elapsed:.1f}s: {type(e).__name__}: {e}")
            
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error(f"LLM timeout for module {module.name}: {e}")
                return f"OCaml module {module.name} - description generation timed out after {elapsed:.1f}s."
            else:
                logger.error(f"LLM error type: {type(e)}, message: {e}")
                import traceback
                logger.error(f"LLM error traceback: {traceback.format_exc()}")
                return f"OCaml module {module.name} - description generation failed: {type(e).__name__}"
    
    def _generate_chunked_description(self, module: ModuleContent, log_prompts: bool = False) -> str:
        """Generate description for large modules using chunking strategy."""
        
        all_items = module.functions + module.types
        chunk_size = 20
        chunk_summaries = []
        
        # Process each chunk of 20 functions/types
        for i in range(0, len(all_items), chunk_size):
            chunk = all_items[i:i + chunk_size]
            chunk_summary = self._summarize_chunk(module, chunk, i // chunk_size + 1, log_prompts)
            chunk_summaries.append(chunk_summary)
        
        # Now combine all chunk summaries with module info
        return self._combine_chunk_summaries(module, chunk_summaries, log_prompts)
    
    def _summarize_chunk(self, module: ModuleContent, chunk: List[Dict], chunk_num: int, log_prompts: bool = False) -> str:
        """Summarize a chunk of functions/types."""
        
        context_parts = [
            f"Module: {module.name} (Chunk {chunk_num})",
        ]
        
        if module.documentation:
            context_parts.append(f"Module Documentation: {module.documentation}")
        
        context_parts.append(f"Functions/Types in this chunk:")
        for item in chunk:
            item_line = f"- {item.get('signature', item.get('name', 'unnamed'))}"
            if item.get('documentation'):
                item_line += f" // {item['documentation']}"
            context_parts.append(item_line)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Summarize this chunk in 1-2 sentences by identifying:
- The specific operations these functions provide
- What data they operate on
- Any patterns in their functionality (e.g., all string manipulation, all file operations, etc.)

Avoid generic terms. Be specific about what these functions actually do.

{context}

Chunk Summary:"""

        if log_prompts:
            logger.info(f"=== CHUNK PROMPT for {module.name} chunk {chunk_num} ===")
            logger.info(prompt)
            logger.info("=== END CHUNK PROMPT ===")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert OCaml developer. Provide concise, direct answers without thinking aloud or explanation."},
                    {"role": "user", "content": prompt + "\n\n/no_think"}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            import re
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== CHUNK RESPONSE for {module.name} chunk {chunk_num} ===")
                logger.info(result)
                logger.info("=== END CHUNK RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout for chunk {chunk_num} of module {module.name}: {e}")
                return f"Chunk {chunk_num}: {len(chunk)} functions/types (timed out)"
            else:
                logger.error(f"LLM error for chunk {chunk_num} of module {module.name}: {e}")
                return f"Chunk {chunk_num}: {len(chunk)} functions/types"
    
    def _combine_chunk_summaries(self, module: ModuleContent, chunk_summaries: List[str], log_prompts: bool = False) -> str:
        """Combine chunk summaries into final module description."""
        
        context_parts = [f"Module: {module.name}"]
        
        if module.documentation:
            context_parts.append(f"Module Documentation: {module.documentation}")
        
        if module.modules:
            submodule_names = [m.get("name", "unnamed") for m in module.modules[:8]]
            context_parts.append(f"Submodules: {', '.join(submodule_names)}")
        
        context_parts.append("Function/Type Summaries:")
        for i, summary in enumerate(chunk_summaries, 1):
            context_parts.append(f"{i}. {summary}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Based on the chunk summaries below, write a 2-3 sentence description that:
- Identifies the main types of operations this module provides
- Specifies what data structures or types it works with
- Mentions specific use cases where applicable

Do NOT use generic phrases or repeat the module name.

{context}

Module Description:"""

        if log_prompts:
            logger.info(f"=== FINAL PROMPT for {module.name} ===")
            logger.info(prompt)
            logger.info("=== END FINAL PROMPT ===")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            import re
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== FINAL RESPONSE for {module.name} ===")
                logger.info(result)
                logger.info("=== END FINAL RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout combining chunks for module {module.name}: {e}")
                return f"OCaml module {module.name} with {len(chunk_summaries)} functional areas (timed out)"
            else:
                logger.error(f"LLM error combining chunks for module {module.name}: {e}")
                return f"OCaml module {module.name} with {len(chunk_summaries)} functional areas"
    
    def merge_descriptions(self, module_name: str, child_descriptions: List[str], module_content: Optional[ModuleContent] = None, log_prompts: bool = False) -> str:
        """Merge child module descriptions into a parent description."""
        
        context_parts = [f"Parent Module: {module_name}"]
        
        if module_content and module_content.documentation:
            context_parts.append(f"Documentation: {module_content.documentation}")
        
        if child_descriptions:
            context_parts.append("Child module descriptions:")
            for i, desc in enumerate(child_descriptions, 1):
                context_parts.append(f"{i}. {desc}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Write a 3-4 sentence description that:
- Synthesizes the functionality of the child modules into a coherent overview
- Identifies the main data types and operations available
- Provides specific examples of what can be done with this module

Do NOT:
- Use generic phrases like "provides functionality" or "collection of modules"
- Repeat the module name
- Use filler words about code quality or programming patterns

{context}

Merged description:"""

        if log_prompts:
            logger.info(f"=== MERGE PROMPT for {module_name} ===")
            logger.info(prompt)
            logger.info("=== END MERGE PROMPT ===")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert OCaml developer. Provide concise, direct answers without thinking aloud or explanation."},
                    {"role": "user", "content": prompt + "\n\n/no_think"}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            import re
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== MERGE RESPONSE for {module_name} ===")
                logger.info(result)
                logger.info("=== END MERGE RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout merging descriptions for {module_name}: {e}")
                return f"OCaml module {module_name} containing: {'; '.join(child_descriptions[:3])} (timed out)"
            else:
                logger.error(f"LLM error merging descriptions for {module_name}: {e}")
                return f"OCaml module {module_name} containing: {'; '.join(child_descriptions[:3])}"


class ModuleExtractor:
    """Extract modules from parsed JSON data."""
    
    def extract_from_parsed_json(self, json_file: Path) -> List[ModuleContent]:
        """Extract module content from parsed JSON file."""
        with open(json_file) as f:
            data = json.load(f)
        
        modules = []
        package_name = data.get("package", "unknown")
        
        def extract_module_recursive(module_data: Dict, parent_path: str = "") -> ModuleContent:
            # Extract name from module_path if available, otherwise use name field
            module_path = module_data.get("module_path", "")
            if module_path:
                # Extract the last component of the module path as the name
                name = module_path.split(".")[-1] if module_path else "unnamed"
                path = module_path
            else:
                name = module_data.get("name", "unnamed")
                path = f"{parent_path}.{name}" if parent_path else name
            
            # Extract content
            functions = module_data.get("values", [])
            types = module_data.get("types", [])
            submodules = module_data.get("modules", [])
            
            # Get documentation from preamble or documentation_sections
            documentation = module_data.get("documentation", "")
            if not documentation:
                # Try preamble
                preamble = module_data.get("preamble", "")
                if preamble:
                    # Simple HTML stripping
                    import re
                    documentation = re.sub('<[^<]+?>', '', preamble).strip()
                
                # Also check documentation_sections
                doc_sections = module_data.get("documentation_sections", [])
                if doc_sections and not documentation:
                    documentation = " ".join(doc_sections)
            
            module = ModuleContent(
                name=name,
                path=path,
                functions=functions,
                types=types,
                modules=submodules,
                documentation=documentation,
                parent=parent_path if parent_path else None
            )
            
            # Process submodules
            for submodule_data in submodules:
                child_module = extract_module_recursive(submodule_data, path)
                modules.append(child_module)
                module.children.append(child_module.path)
            
            return module
        
        # Process modules - handle both hierarchical and flat structures
        if "modules" in data:
            # Check if this is a flat list of parsed modules
            if data["modules"] and "module_path" in data["modules"][0]:
                # Flat structure from parsed documentation
                for module_data in data["modules"]:
                    # Don't process recursively for flat structure
                    module = extract_module_recursive(module_data)
                    
                    # For empty main modules, try to get documentation from README
                    if (not module.functions and not module.types and 
                        module_data.get("module_path", "") == ""):
                        # Check if documentation is just a version string (like "package 1.2.3")
                        doc_words = module.documentation.strip().split()
                        is_just_version = (len(doc_words) <= 3 and 
                                         any(c.isdigit() or c == '.' for word in doc_words for c in word))
                        
                        if not module.documentation.strip() or is_just_version:
                            # This is likely the main package module with no meaningful API docs
                            readme_doc = ""
                            if "documentation" in data and "README" in data["documentation"]:
                                readme = data["documentation"]["README"]
                                readme_sections = readme.get("documentation_sections", [])
                                if readme_sections:
                                    # Use first few sections of README for context
                                    readme_doc = " ".join(readme_sections[:3])
                            
                            if readme_doc:
                                module.documentation = readme_doc
                                logger.info(f"Enhanced empty module {module.path} with README documentation ({len(readme_doc)} chars)")
                    
                    modules.append(module)
            else:
                # Hierarchical structure
                for module_data in data["modules"]:
                    main_module = extract_module_recursive(module_data)
                    modules.append(main_module)
        
        return modules

def process_single_package(json_file: Path, output_dir: Path, llm_client: LLMClient, extractor: ModuleExtractor, log_prompts: bool) -> bool:
    """Process a single package and return success status."""
    package_name = json_file.stem
    
    # Memory monitoring
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    file_size_mb = json_file.stat().st_size / 1024 / 1024
    
    logger.info(f"Starting processing of package: {package_name} (file size: {file_size_mb:.1f}MB, memory: {initial_memory:.1f}MB)")
    
    try:
        # Extract modules
        modules = extractor.extract_from_parsed_json(json_file)
        if not modules:
            logger.warning(f"No modules found in {package_name}")
            return False
        
        # Memory check after loading
        after_load_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Package {package_name}: loaded {len(modules)} modules, memory: {after_load_memory:.1f}MB (+{after_load_memory-initial_memory:.1f}MB)")
        
        # Generate descriptions for leaf modules first
        descriptions = {}
        
        # Sort modules by depth (leaf modules first)
        modules_by_depth = sorted(modules, key=lambda m: m.path.count('.'), reverse=True)
        
        # Process modules
        for module in modules_by_depth:
            # Skip empty modules (no content to describe)
            if not module.children and not module.functions and not module.types and not module.documentation.strip():
                logger.info(f"Skipping empty module: {module.path}")
                descriptions[module.path] = f"Empty module with no functions, types, or documentation."
                continue
            
            # Generate new description
            if not module.children:  # Leaf module
                description = llm_client.generate_module_description(module, log_prompts)
            else:  # Parent module - merge child descriptions
                child_descriptions = [descriptions.get(child, "") for child in module.children if child in descriptions]
                description = llm_client.merge_descriptions(module.name, child_descriptions, module, log_prompts)
            
            descriptions[module.path] = description
        
        # Clean up module paths - replace "unnamed" with package name
        cleaned_descriptions = {}
        for module_path, description in descriptions.items():
            # Replace "unnamed" with package name in the path
            if module_path.startswith("unnamed."):
                clean_path = module_path.replace("unnamed.", f"{package_name}.", 1)
            elif module_path == "unnamed":
                clean_path = package_name
            else:
                clean_path = module_path
            cleaned_descriptions[clean_path] = description
        
        # Save package descriptions
        output_file = output_dir / f"{package_name}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "package": package_name,
                "descriptions": cleaned_descriptions
            }, f, indent=2)
        
        # Memory check at completion
        final_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Successfully completed package: {package_name} ({len(modules)} modules, {len(descriptions)} descriptions)")
        logger.info(f"Package {package_name}: final memory: {final_memory:.1f}MB (+{final_memory-initial_memory:.1f}MB total)")
        
        # Force garbage collection for large packages
        if file_size_mb > 5:
            logger.info(f"Large package {package_name}: forcing garbage collection")
            gc.collect()
            after_gc_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Package {package_name}: after GC memory: {after_gc_memory:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED processing package {package_name}: {e}")
        import traceback
        logger.error(f"Full traceback for {package_name}: {traceback.format_exc()}")
        return False

def package_worker(package_queue: queue.Queue, output_dir: Path, llm_url: str, model: str, log_prompts: bool, 
                  progress_lock: threading.Lock, completed_count: list, failed_count: list):
    """Worker function that processes packages from the queue."""
    import threading
    import time
    import traceback
    worker_id = threading.current_thread().name
    logger.info(f"Worker {worker_id} started successfully")
    
    # Install signal handlers for this worker thread
    def worker_signal_handler(signum, frame):
        logger.error(f"Worker {worker_id} RECEIVED SIGNAL {signum}")
        logger.error(f"Worker {worker_id} signal traceback: {''.join(traceback.format_stack(frame))}")
        raise KeyboardInterrupt(f"Worker {worker_id} signal {signum}")
    
    try:
        # Each worker gets its own LLM client and extractor
        logger.info(f"Worker {worker_id} initializing LLM client...")
        llm_client = LLMClient(llm_url, model)
        logger.info(f"Worker {worker_id} initializing extractor...")
        extractor = ModuleExtractor()
        logger.info(f"Worker {worker_id} initialization complete")
        
        package_count = 0
        last_heartbeat = time.time()
        
        while True:
            json_file = None
            try:
                # Heartbeat every 30 seconds
                if time.time() - last_heartbeat > 30:
                    logger.info(f"Worker {worker_id} HEARTBEAT - alive and processing (packages done: {package_count})")
                    last_heartbeat = time.time()
                
                logger.info(f"Worker {worker_id} waiting for package from queue...")
                json_file = package_queue.get(timeout=5)  # Increased timeout
                package_count += 1
                package_name = json_file.stem
                
                logger.info(f"Worker {worker_id} picked up package #{package_count}: {package_name}")
                
                # Add heartbeat logging
                start_time = time.time()
                logger.info(f"Worker {worker_id} starting processing of {package_name} at {time.strftime('%H:%M:%S')}")
                
                # Wrap in try-catch for individual package processing
                try:
                    success = process_single_package(json_file, output_dir, llm_client, extractor, log_prompts)
                    logger.info(f"Worker {worker_id} process_single_package returned: {success}")
                except Exception as package_error:
                    logger.error(f"Worker {worker_id} EXCEPTION in process_single_package for {package_name}: {package_error}")
                    logger.error(f"Worker {worker_id} package traceback: {traceback.format_exc()}")
                    success = False
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Worker {worker_id} finished processing {package_name} in {duration:.1f}s")
                
                # Update progress counters thread-safely
                try:
                    with progress_lock:
                        if success:
                            completed_count[0] += 1
                            logger.info(f"Worker {worker_id} completed package: {package_name} (SUCCESS) - total completed: {completed_count[0]}")
                        else:
                            failed_count[0] += 1
                            logger.error(f"Worker {worker_id} failed package: {package_name} (FAILED) - total failed: {failed_count[0]}")
                except Exception as count_error:
                    logger.error(f"Worker {worker_id} EXCEPTION updating progress counts: {count_error}")
                    logger.error(f"Worker {worker_id} count traceback: {traceback.format_exc()}")
                
                try:
                    logger.info(f"Worker {worker_id} marking task done for {package_name}")
                    package_queue.task_done()
                    logger.info(f"Worker {worker_id} successfully completed task for {package_name}")
                except Exception as task_done_error:
                    logger.error(f"Worker {worker_id} EXCEPTION in task_done: {task_done_error}")
                    logger.error(f"Worker {worker_id} task_done traceback: {traceback.format_exc()}")
                
            except queue.Empty:
                # No more packages in queue
                logger.info(f"Worker {worker_id} finished - no more packages in queue after processing {package_count} packages")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered CRITICAL error processing {json_file.stem if json_file else 'unknown'}: {e}")
                logger.error(f"Worker {worker_id} full traceback: {traceback.format_exc()}")
                
                # Still mark task as done to prevent hanging
                if json_file:
                    try:
                        package_queue.task_done()
                        logger.info(f"Worker {worker_id} marked failed task as done")
                    except Exception as task_done_error:
                        logger.error(f"Worker {worker_id} could not mark task as done: {task_done_error}")
                
                # Update failed count
                try:
                    with progress_lock:
                        failed_count[0] += 1
                        logger.error(f"Worker {worker_id} incremented failed count to {failed_count[0]}")
                except Exception as count_error:
                    logger.error(f"Worker {worker_id} could not update failed count: {count_error}")
    
    except Exception as fatal_error:
        logger.error(f"Worker {worker_id} FATAL ERROR in main loop: {fatal_error}")
        logger.error(f"Worker {worker_id} FATAL traceback: {traceback.format_exc()}")
        logger.error(f"Worker {worker_id} FATAL error type: {type(fatal_error)}")
    
    logger.info(f"Worker {worker_id} exiting after processing {package_count} packages")

def monitor_thread(workers, stop_event):
    """Background thread to monitor worker health"""
    import time
    while not stop_event.is_set():
        try:
            process = psutil.Process()
            active_threads = threading.active_count()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            logger.info(f"[MONITOR] Active threads: {active_threads}, Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
            
            for i, worker in enumerate(workers):
                if worker.done():
                    try:
                        worker.exception()
                    except Exception as e:
                        logger.error(f"[MONITOR] Worker {i} died with exception: {e}")
            
            # Check for memory issues
            if memory_mb > 8000:  # 8GB
                logger.warning(f"[MONITOR] High memory usage: {memory_mb:.1f}MB")
            
            time.sleep(10)
        except Exception as e:
            logger.error(f"[MONITOR] Monitor thread error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate semantic descriptions for OCaml modules")
    parser.add_argument("--input-dir", default="parsed-docs", help="Directory with parsed JSON files")
    parser.add_argument("--output-dir", default="module-descriptions", help="Output directory for descriptions")
    parser.add_argument("--limit", type=int, help="Limit number of packages to process")
    parser.add_argument("--package", help="Process specific package only")
    parser.add_argument("--llm-url", default="http://localhost:8000", help="LLM endpoint URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8", help="LLM model name")
    parser.add_argument("--log-prompts", action="store_true", help="Log prompts and responses sent to LLM")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get input files
    input_dir = Path(args.input_dir)
    json_files = list(input_dir.glob("*.json"))
    
    if args.package:
        json_files = [f for f in json_files if f.stem == args.package]
    
    # Exclude tezos and octez packages
    json_files = [f for f in json_files if "tezos" not in f.stem.lower() and "octez" not in f.stem.lower()]
    
    # Skip packages that already have generated descriptions
    json_files = [f for f in json_files if not (output_dir / f"{f.stem}.json").exists()]
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    logger.info(f"Processing {len(json_files)} packages with {args.workers} workers")
    
    # Estimate time based on ~30 seconds per package
    estimated_hours = (len(json_files) * 30) / 3600
    logger.info(f"Estimated time: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    
    # Create package queue with filtered files
    package_queue = queue.Queue()
    for json_file in json_files:
        package_queue.put(json_file)
    
    # Shared progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference
    failed_count = [0]
    
    # Main thread signal handlers
    def main_signal_handler(signum, frame):
        logger.error(f"MAIN thread received signal {signum} - process being killed!")
        logger.error(f"MAIN signal traceback: {''.join(traceback.format_stack(frame))}")
        raise KeyboardInterrupt(f"Main signal {signum}")
    
    signal.signal(signal.SIGTERM, main_signal_handler)
    signal.signal(signal.SIGINT, main_signal_handler)
    
    logger.info(f"Starting ThreadPoolExecutor with {args.workers} workers...")
    
    # Shuffle the queue to distribute load better
    import random
    random.shuffle(json_files)
    logger.info(f"Shuffled package order to distribute load")
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            logger.info("ThreadPoolExecutor created successfully")
            
            # Submit worker tasks
            workers = []
            for i in range(args.workers):
                logger.info(f"Submitting worker {i}...")
                future = executor.submit(
                    package_worker, 
                    package_queue, 
                    output_dir, 
                    args.llm_url, 
                    args.model, 
                    args.log_prompts,
                    progress_lock,
                    completed_count,
                    failed_count
                )
                workers.append(future)
                logger.info(f"Worker {i} submitted successfully")
            
            logger.info(f"All {len(workers)} workers submitted")
            
            # Start background monitor thread
            stop_monitor = threading.Event()
            monitor = threading.Thread(target=monitor_thread, args=(workers, stop_monitor), daemon=True)
            monitor.start()
            logger.info("Background monitor thread started")
            
            # Monitor progress
            total_packages = len(json_files)
            logger.info(f"Starting progress monitoring for {total_packages} packages...")
            
            try:
                with tqdm(total=total_packages, desc="Processing packages", unit="package") as pbar:
                    last_completed = 0
                    monitoring_cycles = 0
                    
                    while True:
                        monitoring_cycles += 1
                        
                        # Check worker status periodically
                        if monitoring_cycles % 20 == 0:  # Every 10 seconds
                            logger.info(f"Progress monitor cycle #{monitoring_cycles}")
                            alive_workers = sum(1 for w in workers if not w.done())
                            logger.info(f"Alive workers: {alive_workers}/{len(workers)}")
                            
                            # Check for worker exceptions
                            for i, worker in enumerate(workers):
                                if worker.done():
                                    try:
                                        worker.result(timeout=1)  # 1 second timeout to avoid hanging
                                    except Exception as worker_error:
                                        logger.error(f"Worker {i} failed with exception: {worker_error}")
                                        import traceback
                                        logger.error(f"Worker {i} traceback: {traceback.format_exc()}")
                        
                        with progress_lock:
                            current_completed = completed_count[0] + failed_count[0]
                            logger.debug(f"Progress check: {current_completed}/{total_packages} completed")
                        
                        # Update progress bar
                        if current_completed > last_completed:
                            delta = current_completed - last_completed
                            pbar.update(delta)
                            last_completed = current_completed
                            logger.info(f"Progress updated: {current_completed}/{total_packages} packages processed")
                        
                        # Check if all packages are done
                        if current_completed >= total_packages:
                            logger.info("All packages completed!")
                            break
                        
                        # Small delay to avoid busy waiting
                        threading.Event().wait(0.5)
                
            except Exception as monitor_error:
                logger.error(f"Progress monitoring failed: {monitor_error}")
                import traceback
                logger.error(f"Monitor traceback: {traceback.format_exc()}")
                raise
            
            logger.info("Progress monitoring complete, waiting for queue to finish...")
            
            # Wait for all workers to complete
            try:
                package_queue.join()
                logger.info("Queue join completed successfully")
            except Exception as join_error:
                logger.error(f"Queue join failed: {join_error}")
                raise
            
            logger.info("Waiting for worker futures to complete...")
            for i, worker in enumerate(workers):
                try:
                    worker.result(timeout=30)  # 30 second timeout per worker
                    logger.info(f"Worker {i} completed successfully")
                except Exception as worker_error:
                    logger.error(f"Worker {i} failed: {worker_error}")
            
            # Stop monitor thread
            stop_monitor.set()
            logger.info("Monitor thread stopped")
    
    except Exception as executor_error:
        logger.error(f"ThreadPoolExecutor failed: {executor_error}")
        import traceback
        logger.error(f"Executor traceback: {traceback.format_exc()}")
        raise
    
    logger.info(f"Processing complete! Successful: {completed_count[0]}, Failed: {failed_count[0]}")

if __name__ == "__main__":
    try:
        logger.info("=== STARTING DESCRIPTION GENERATION ===")
        logger.info(f"=== PYTHON VERSION: {sys.version} ===")
        logger.info(f"=== CURRENT PID: {os.getpid()} ===")
        logger.info(f"=== CURRENT WORKING DIR: {os.getcwd()} ===")
        
        # Install global exception handlers before starting
        logger.info("=== INSTALLING GLOBAL EXCEPTION HANDLERS ===")
        
        main()
        logger.info("=== DESCRIPTION GENERATION COMPLETED SUCCESSFULLY ===")
    except KeyboardInterrupt as kb_interrupt:
        logger.error(f"=== PROCESS INTERRUPTED BY USER OR SIGNAL: {kb_interrupt} ===")
        logger.error(f"=== KEYBOARD INTERRUPT TRACEBACK: {traceback.format_exc()} ===")
        sys.exit(1)
    except SystemExit as sys_exit:
        logger.error(f"=== SYSTEM EXIT: {sys_exit} ===") 
        logger.error(f"=== SYSTEM EXIT TRACEBACK: {traceback.format_exc()} ===")
        raise
    except Exception as main_error:
        logger.error(f"=== FATAL ERROR IN MAIN: {main_error} ===")
        logger.error(f"=== FATAL ERROR TYPE: {type(main_error)} ===")
        logger.error(f"=== MAIN TRACEBACK: {traceback.format_exc()} ===")
        
        # Try to get additional debugging info
        try:
            import psutil
            process = psutil.Process()
            logger.error(f"=== PROCESS STATUS: {process.status()} ===")
            logger.error(f"=== MEMORY INFO: {process.memory_info()} ===")
            logger.error(f"=== OPEN FILES: {len(process.open_files())} ===")
            logger.error(f"=== THREADS: {process.num_threads()} ===")
        except Exception as debug_error:
            logger.error(f"=== COULD NOT GET DEBUG INFO: {debug_error} ===")
        
        sys.exit(1)