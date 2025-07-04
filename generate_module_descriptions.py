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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

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
    elements: List[Dict[str, Any]]  # Ordered list of all elements
    modules: List[Dict[str, Any]]   # Submodules
    documentation: str
    library: Optional[str] = None
    parent: Optional[str] = None
    children: List[str] = None
    is_module_type: bool = False
    preamble: str = ""

    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class LibraryWorkItem:
    """Work item for processing a single library."""
    package_file: Path
    package_name: str
    package_version: str
    library_name: Optional[str]  # None for modules without library
    modules: List[ModuleContent]
    extractor: 'ModuleExtractor'  # Shared extractor instance

class LLMClient:
    """OpenAI-compatible client for generating descriptions."""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen/Qwen3-30B-A3B-FP8", api_key: str = "dummy_key"):
        try:
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key=api_key,
                timeout=120.0,  # 2 minute timeout for requests
                max_retries=1  # Only retry once on failure
            )
            self.model = model
            logger.info(f"LLMClient initialized with base_url={base_url}, model={model}, api_key={'***' if api_key != 'dummy_key' else 'dummy_key'}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _is_module_type(self, module: ModuleContent) -> bool:
        """Check if a module is actually a module type."""
        return module.is_module_type
    
    def _get_ancestor_preambles(self, module: ModuleContent, all_modules: Dict[str, ModuleContent]) -> List[Tuple[str, str]]:
        """Get preambles from ancestor modules.
        Returns list of (module_path, preamble) tuples."""
        ancestor_preambles = []
        
        # Split the module path to find ancestors
        path_parts = module.path.split('.')
        
        # Build ancestor paths (e.g., for "Eio.Private.Cells" -> ["Eio", "Eio.Private"])
        for i in range(1, len(path_parts)):
            ancestor_path = '.'.join(path_parts[:i])
            if ancestor_path in all_modules:
                ancestor = all_modules[ancestor_path]
                if ancestor.preamble:
                    ancestor_preambles.append((ancestor_path, ancestor.preamble))
        
        return ancestor_preambles
    
    def generate_module_description(self, module: ModuleContent, all_modules: Dict[str, ModuleContent], log_prompts: bool = False) -> str:
        """Generate a concise description for a single module using chunking strategy."""
        
        # Count code elements (functions, types, modules) from ordered elements
        code_elements = [elem for elem in module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
        total_items = len(code_elements)
        
        if total_items <= 20:
            return self._generate_simple_description(module, all_modules, log_prompts)
        else:
            return self._generate_chunked_description(module, all_modules, log_prompts)
    
    def _build_simple_description_prompt(self, module: ModuleContent, all_modules: Dict[str, ModuleContent]) -> str:
        """Build prompt for modules with ≤20 functions/types."""
        
        # Check if this is a module type
        is_module_type = self._is_module_type(module)
        module_label = "Module Type" if is_module_type else "Module"
        
        context_parts = [f"{module_label}: {module.path}"]
        
        # Add ancestor preambles for context
        ancestor_preambles = self._get_ancestor_preambles(module, all_modules)
        if ancestor_preambles:
            context_parts.append("\nAncestor Module Context:")
            for ancestor_path, preamble in ancestor_preambles:
                context_parts.append(f"- {ancestor_path}: {preamble}")
        
        if module.documentation:
            context_parts.append(f"\nModule Documentation: {module.documentation}")
        
        # Process elements in order they appear in documentation
        current_section = None
        for elem in module.elements:
            if elem.get('kind') == 'section':
                current_section = elem.get('title', '')
                if current_section:
                    context_parts.append(f"\n## {current_section}")
                    if elem.get('content'):
                        context_parts.append(elem['content'])
            elif elem.get('kind') in ['value', 'type', 'module', 'module-type']:
                # Format based on kind
                if elem.get('kind') == 'value':
                    # Use signature if available, otherwise format as val name
                    if elem.get('signature'):
                        elem_line = f"- {elem['signature']}"
                    else:
                        elem_line = f"- val {elem.get('name', 'unnamed')}"
                elif elem.get('kind') == 'type':
                    # Use signature if available, otherwise format as type name
                    if elem.get('signature'):
                        elem_line = f"- {elem['signature']}"
                    else:
                        elem_line = f"- type {elem.get('name', 'unnamed')}"
                elif elem.get('kind') == 'module':
                    # Format as module name
                    elem_line = f"- module {elem.get('name', 'unnamed')}"
                elif elem.get('kind') == 'module-type':
                    # Format as module type name
                    elem_line = f"- module type {elem.get('name', 'unnamed')}"
                else:
                    # Fallback
                    elem_line = f"- {elem.get('signature', elem.get('name', 'unnamed'))}"
                
                if elem.get('documentation'):
                    elem_line += f" (* {elem['documentation']} *)"
                context_parts.append(elem_line)
        
        
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
        
        return prompt
    
    def _generate_simple_description(self, module: ModuleContent, all_modules: Dict[str, ModuleContent], log_prompts: bool = False) -> str:
        """Generate description for modules with ≤20 functions/types."""
        
        prompt = self._build_simple_description_prompt(module, all_modules)

        if log_prompts:
            logger.info(f"=== PROMPT for {module.path} ===")
            logger.info(prompt)
            logger.info("=== END PROMPT ===")
        
        try:
            logger.info(f"Sending LLM request for module {module.path} (length: {len(prompt)} chars)")
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
            logger.info(f"LLM request completed for module {module.path} in {elapsed:.1f}s")
            
            if not response.choices:
                logger.error(f"LLM returned no choices for module {module.path}")
                return f"OCaml module {module.path} - no response from LLM"
            
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== RESPONSE for {module.path} ===")
                logger.info(result)
                logger.info("=== END RESPONSE ===")
            return result
        except Exception as e:
            import time
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"LLM error for module {module.path} after {elapsed:.1f}s: {type(e).__name__}: {e}")
            
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error(f"LLM timeout for module {module.path}: {e}")
                return f"OCaml module {module.path} - description generation timed out after {elapsed:.1f}s."
            else:
                logger.error(f"LLM error type: {type(e)}, message: {e}")
                import traceback
                logger.error(f"LLM error traceback: {traceback.format_exc()}")
                return f"OCaml module {module.path} - description generation failed: {type(e).__name__}"
    
    def _generate_chunked_description(self, module: ModuleContent, all_modules: Dict[str, ModuleContent], log_prompts: bool = False) -> str:
        """Generate description for large modules using chunking strategy."""
        
        # Extract code elements (functions, types, modules) from ordered elements
        code_elements = [elem for elem in module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
        
        chunk_size = 20
        chunk_summaries = []
        
        # Process each chunk of 20 functions/types
        for i in range(0, len(code_elements), chunk_size):
            chunk = code_elements[i:i + chunk_size]
            chunk_summary = self._summarize_chunk(module, chunk, i // chunk_size + 1, all_modules, log_prompts)
            chunk_summaries.append(chunk_summary)
        
        # Now combine all chunk summaries with module info
        return self._combine_chunk_summaries(module, chunk_summaries, all_modules, log_prompts)
    
    def _build_chunk_prompt(self, module: ModuleContent, chunk: List[Dict], chunk_num: int, all_modules: Dict[str, ModuleContent]) -> str:
        """Build prompt for summarizing a chunk of functions/types/modules."""
        # Check if this is a module type
        is_module_type = self._is_module_type(module)
        module_label = "Module Type" if is_module_type else "Module"
        
        context_parts = [
            f"{module_label}: {module.path} (Chunk {chunk_num})",
        ]
        
        # Add ancestor preambles for context (only in first chunk)
        if chunk_num == 1:
            ancestor_preambles = self._get_ancestor_preambles(module, all_modules)
            if ancestor_preambles:
                context_parts.append("\nAncestor Module Context:")
                for ancestor_path, preamble in ancestor_preambles:
                    context_parts.append(f"- {ancestor_path}: {preamble}")
        
        if module.documentation:
            context_parts.append(f"\nModule Documentation: {module.documentation}")
        
        context_parts.append(f"Functions/Types/Modules in this chunk:")
        for item in chunk:
            # Format based on kind
            if item.get('kind') == 'value':
                # Use signature if available, otherwise format as val name
                if item.get('signature'):
                    item_line = f"- {item['signature']}"
                else:
                    item_line = f"- val {item.get('name', 'unnamed')}"
            elif item.get('kind') == 'type':
                # Use signature if available, otherwise format as type name
                if item.get('signature'):
                    item_line = f"- {item['signature']}"
                else:
                    item_line = f"- type {item.get('name', 'unnamed')}"
            elif item.get('kind') == 'module':
                # Format as module name
                item_line = f"- module {item.get('name', 'unnamed')}"
            elif item.get('kind') == 'module-type':
                # Format as module type name
                item_line = f"- module type {item.get('name', 'unnamed')}"
            else:
                # Fallback for old format or other kinds
                item_line = f"- {item.get('signature', item.get('name', 'unnamed'))}"
            
            if item.get('documentation'):
                item_line += f" (* {item['documentation']} *)"
            context_parts.append(item_line)
        
        context = "\n".join(context_parts)
        
        return f"""You are an expert OCaml developer. Summarize this chunk in 1-2 sentences by identifying:
- The specific operations these functions provide
- What data they operate on
- Any patterns in their functionality (e.g., all string manipulation, all file operations, etc.)

Avoid generic terms. Be specific about what these functions actually do.

{context}

Chunk Summary:"""

    def _summarize_chunk(self, module: ModuleContent, chunk: List[Dict], chunk_num: int, all_modules: Dict[str, ModuleContent], log_prompts: bool = False) -> str:
        """Summarize a chunk of functions/types."""
        
        prompt = self._build_chunk_prompt(module, chunk, chunk_num, all_modules)

        if log_prompts:
            logger.info(f"=== CHUNK PROMPT for {module.path} chunk {chunk_num} ===")
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
                logger.info(f"=== CHUNK RESPONSE for {module.path} chunk {chunk_num} ===")
                logger.info(result)
                logger.info("=== END CHUNK RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout for chunk {chunk_num} of module {module.path}: {e}")
                return f"Chunk {chunk_num}: {len(chunk)} functions/types (timed out)"
            else:
                logger.error(f"LLM error for chunk {chunk_num} of module {module.path}: {e}")
                return f"Chunk {chunk_num}: {len(chunk)} functions/types"
    
    def _build_chunk_combine_prompt(self, module: ModuleContent, chunk_summaries: List[str], all_modules: Dict[str, ModuleContent]) -> str:
        """Build prompt for combining chunk summaries into final module description."""
        # Check if this is a module type
        is_module_type = self._is_module_type(module)
        module_label = "Module Type" if is_module_type else "Module"
        
        context_parts = [f"{module_label}: {module.path}"]
        
        # Add ancestor preambles for context
        ancestor_preambles = self._get_ancestor_preambles(module, all_modules)
        if ancestor_preambles:
            context_parts.append("\nAncestor Module Context:")
            for ancestor_path, preamble in ancestor_preambles:
                context_parts.append(f"- {ancestor_path}: {preamble}")
        
        if module.documentation:
            context_parts.append(f"\nModule Documentation: {module.documentation}")
        
        if module.modules:
            submodule_names = [m.get("name", "unnamed") for m in module.modules[:8]]
            context_parts.append(f"Submodules: {', '.join(submodule_names)}")
        
        context_parts.append("Function/Type Summaries:")
        for i, summary in enumerate(chunk_summaries, 1):
            context_parts.append(f"{i}. {summary}")
        
        context = "\n".join(context_parts)
        
        return f"""You are an expert OCaml developer. Based on the chunk summaries below, write a 2-3 sentence description that:
- Identifies the main types of operations this module provides
- Specifies what data structures or types it works with
- Mentions specific use cases where applicable

Do NOT use generic phrases or repeat the module name.

{context}

Module Description:"""

    def _combine_chunk_summaries(self, module: ModuleContent, chunk_summaries: List[str], all_modules: Dict[str, ModuleContent], log_prompts: bool = False) -> str:
        """Combine chunk summaries into final module description."""
        
        prompt = self._build_chunk_combine_prompt(module, chunk_summaries, all_modules)

        if log_prompts:
            logger.info(f"=== FINAL PROMPT for {module.path} ===")
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
                logger.info(f"=== FINAL RESPONSE for {module.path} ===")
                logger.info(result)
                logger.info("=== END FINAL RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout combining chunks for module {module.path}: {e}")
                return f"OCaml module {module.path} with {len(chunk_summaries)} functional areas (timed out)"
            else:
                logger.error(f"LLM error combining chunks for module {module.path}: {e}")
                return f"OCaml module {module.path} with {len(chunk_summaries)} functional areas"
    
    def generate_library_summary(self, library_name: str, module_descriptions: List[str], log_prompts: bool = False) -> str:
        """Generate a summary for a library based on its module descriptions."""
        
        context_parts = [f"Library: {library_name}"]
        
        context_parts.append("Module descriptions:")
        for i, desc in enumerate(module_descriptions, 1):
            context_parts.append(f"{i}. {desc}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Write a 3-4 sentence summary of this library that:
- Identifies the main purpose and functionality of the library
- Describes the types of operations and data structures it provides
- Explains what developers would use this library for
- Mentions specific capabilities where relevant

Do NOT:
- Use generic phrases like "provides functionality" or "collection of modules"
- Repeat the library name
- Use filler words about code quality or programming patterns

{context}

Library Summary:"""

        if log_prompts:
            logger.info(f"=== LIBRARY SUMMARY PROMPT for {library_name} ===")
            logger.info(prompt)
            logger.info("=== END LIBRARY SUMMARY PROMPT ===")
        
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
                logger.info(f"=== LIBRARY SUMMARY RESPONSE for {library_name} ===")
                logger.info(result)
                logger.info("=== END LIBRARY SUMMARY RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout generating library summary for {library_name}: {e}")
                return f"Library containing {len(module_descriptions)} modules (timed out)"
            else:
                logger.error(f"LLM error generating library summary for {library_name}: {e}")
                return f"Library containing {len(module_descriptions)} modules"

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

    def merge_descriptions_with_own_content(self, module_name: str, own_description: str, child_descriptions: List[str], module_content: Optional[ModuleContent] = None, log_prompts: bool = False) -> str:
        """Merge a module's own description with its child module descriptions."""
        
        context_parts = [f"Parent Module: {module_name}"]
        
        if module_content and module_content.documentation:
            context_parts.append(f"Documentation: {module_content.documentation}")
        
        context_parts.append(f"Own functionality: {own_description}")
        
        if child_descriptions:
            context_parts.append("Child module descriptions:")
            for i, desc in enumerate(child_descriptions, 1):
                context_parts.append(f"{i}. {desc}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Write a 3-4 sentence description that:
- Combines the module's own functionality with its child modules into a coherent overview
- Identifies the main data types and operations available
- Provides specific examples of what can be done with this module
- Balances coverage of both the module's direct API and its submodules

Do NOT:
- Use generic phrases like "provides functionality" or "collection of modules"
- Repeat the module name
- Use filler words about code quality or programming patterns

{context}

Merged description:"""

        if log_prompts:
            logger.info(f"=== HYBRID MERGE PROMPT for {module_name} ===")
            logger.info(prompt)
            logger.info("=== END HYBRID MERGE PROMPT ===")
        
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
                logger.info(f"=== HYBRID MERGE RESPONSE for {module_name} ===")
                logger.info(result)
                logger.info("=== END HYBRID MERGE RESPONSE ===")
            return result
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"LLM timeout merging hybrid descriptions for {module_name}: {e}")
                return f"OCaml module {module_name}: {own_description} Also contains: {'; '.join(child_descriptions[:2])} (timed out)"
            else:
                logger.error(f"LLM error merging hybrid descriptions for {module_name}: {e}")
                return f"OCaml module {module_name}: {own_description} Also contains: {'; '.join(child_descriptions[:2])}"


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
                # Handle both string and list formats for module_path
                if isinstance(module_path, list):
                    # Join list elements with dots
                    path = ".".join(module_path)
                    name = module_path[-1] if module_path else "unnamed"
                else:
                    # Extract the last component of the module path as the name
                    name = module_path.split(".")[-1] if module_path else "unnamed"
                    path = module_path
            else:
                name = module_data.get("name", "unnamed")
                path = f"{parent_path}.{name}" if parent_path else name
            
            # Extract content
            elements = module_data.get("elements", [])  # Ordered elements
            submodules = module_data.get("modules", [])
            
            # Get documentation from preamble or documentation_sections
            documentation = module_data.get("documentation", "")
            if not documentation:
                # Try module_documentation
                documentation = module_data.get("module_documentation", "")
                
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
            
            # Extract library, is_module_type, and preamble from the module data
            library = module_data.get("library")
            is_module_type = module_data.get("is_module_type", False)
            preamble = module_data.get("preamble", "")
            
            module = ModuleContent(
                name=name,
                path=path,
                elements=elements,
                modules=submodules,
                documentation=documentation,
                library=library,
                parent=parent_path if parent_path else None,
                is_module_type=is_module_type,
                preamble=preamble
            )
            
            # For flat JSON structures, don't process submodules since they exist as separate entries
            # Use the children field that was calculated by build_module_hierarchy in extract_docs.py
            # This includes both submodules and sub-module-types
            children_from_json = module_data.get("children", [])
            module.children.extend(children_from_json)
            
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
                    code_elements = [elem for elem in module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
                    if (not code_elements and module_data.get("module_path", "") == ""):
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

def process_library(work_item: LibraryWorkItem, llm_client: LLMClient, log_prompts: bool) -> Dict[str, Any]:
    """Process a single library and return its descriptions."""
    library_name = work_item.library_name
    modules = work_item.modules
    package_name = work_item.package_name
    
    if library_name:
        logger.info(f"Processing library {library_name} from package {package_name} with {len(modules)} modules")
    else:
        logger.info(f"Processing {len(modules)} modules without library from package {package_name}")
    
    # Create a dictionary of all modules for ancestor lookup
    all_modules = {module.path: module for module in modules}
    
    # Sort modules by depth (leaf modules first)
    modules_by_depth = sorted(modules, key=lambda m: m.path.count('.'), reverse=True)
    
    # Generate descriptions for modules
    module_descriptions = {}
    
    for module in modules_by_depth:
        # Skip empty modules (no content to describe)
        code_elements = [elem for elem in module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
        if not module.children and not code_elements and not module.documentation.strip():
            logger.info(f"Skipping empty module: {module.path}")
            module_descriptions[module.path] = f"Empty module with no functions, types, or documentation."
            continue
        
        # Generate new description
        # Check if module has its own content (functions, types, elements)
        has_own_content = any(elem.get('kind') in ['value', 'type', 'module-type'] for elem in module.elements)
        
        if not module.children:  # Leaf module
            description = llm_client.generate_module_description(module, all_modules, log_prompts)
        elif not has_own_content:  # Pure parent module - merge child descriptions only
            child_descriptions = [module_descriptions.get(child, "") for child in module.children if child in module_descriptions]
            description = llm_client.merge_descriptions(module.name, child_descriptions, module, log_prompts)
        else:  # Hybrid module - has both own content and children
            # First generate description for own content
            own_description = llm_client.generate_module_description(module, all_modules, log_prompts)
            
            # Then get child descriptions
            child_descriptions = [module_descriptions.get(child, "") for child in module.children if child in module_descriptions]
            
            # Combine both own content and children
            if child_descriptions:
                description = llm_client.merge_descriptions_with_own_content(
                    module.name, own_description, child_descriptions, module, log_prompts)
            else:
                description = own_description
        
        module_descriptions[module.path] = description
    
    # Generate library summary if this is a library
    library_summary = None
    if library_name:
        valid_descriptions = [desc for desc in module_descriptions.values() if desc and not desc.startswith("Empty module")]
        if valid_descriptions:
            library_summary = llm_client.generate_library_summary(library_name, valid_descriptions, log_prompts)
        else:
            library_summary = f"Library with {len(module_descriptions)} modules (no valid descriptions)"
    
    # Clean up module paths
    cleaned_modules = {}
    for module_path, description in module_descriptions.items():
        # Replace "unnamed" with package name in the path
        if module_path.startswith("unnamed."):
            clean_path = module_path.replace("unnamed.", f"{package_name}.", 1)
        elif module_path == "unnamed":
            clean_path = package_name
        else:
            clean_path = module_path
        cleaned_modules[clean_path] = description
    
    result = {
        "library_name": library_name,
        "modules": cleaned_modules
    }
    
    if library_summary:
        result["summary"] = library_summary
    
    return result

def load_package_and_create_work_items(json_file: Path, extractor: ModuleExtractor) -> List[LibraryWorkItem]:
    """Load a package and create work items for each library."""
    package_name = json_file.stem
    work_items = []
    
    try:
        # Load package data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        package_version = data.get('version', 'unknown')
        
        # Extract modules
        modules = extractor.extract_from_parsed_json(json_file)
        if not modules:
            logger.warning(f"No modules found in {package_name}")
            return work_items
        
        # Group modules by library
        libraries = {}
        no_library_modules = []
        
        for module in modules:
            if module.library:
                if module.library not in libraries:
                    libraries[module.library] = []
                libraries[module.library].append(module)
            else:
                no_library_modules.append(module)
        
        logger.info(f"Package {package_name}: {len(libraries)} libraries, {len(no_library_modules)} modules without library")
        
        # Track expected libraries for this package
        with results_lock:
            package_expected_libraries[package_name].clear()
            for library_name in libraries:
                package_expected_libraries[package_name].add(library_name)
            if no_library_modules:
                package_expected_libraries[package_name].add(None)  # None represents modules without library
        
        # Create work items for each library
        for library_name, library_modules in libraries.items():
            work_items.append(LibraryWorkItem(
                package_file=json_file,
                package_name=package_name,
                package_version=package_version,
                library_name=library_name,
                modules=library_modules,
                extractor=extractor
            ))
        
        # Create work item for modules without library
        if no_library_modules:
            work_items.append(LibraryWorkItem(
                package_file=json_file,
                package_name=package_name,
                package_version=package_version,
                library_name=None,
                modules=no_library_modules,
                extractor=extractor
            ))
        
        return work_items
        
    except Exception as e:
        logger.error(f"Failed to load package {package_name}: {e}")
        return work_items

# Global storage for package results and tracking
package_results = defaultdict(lambda: {'libraries': {}, 'modules_without_library': {}})
package_expected_libraries = defaultdict(set)  # Track expected libraries per package
package_completed_libraries = defaultdict(set)  # Track completed libraries per package
results_lock = threading.Lock()

def save_package_results(output_dir: Path, package_name: str, completed_packages: set) -> bool:
    """Save results for a package when all its libraries are processed."""
    try:
        with results_lock:
            if package_name not in package_results:
                return False
            
            result_data = package_results[package_name]
            
            # Build final result structure
            result = {
                "package": package_name,
                "libraries": result_data['libraries']
            }
            
            if result_data['modules_without_library']:
                result["modules_without_library"] = result_data['modules_without_library']
            
            # Save to file
            output_file = output_dir / f"{package_name}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Clean up memory
            del package_results[package_name]
            
            # Mark package as completed
            completed_packages.add(package_name)
            
            return True
    except Exception as e:
        logger.error(f"Failed to save results for package {package_name}: {e}")
        return False

def library_worker(work_queue: queue.Queue, output_dir: Path, llm_url: str, model: str, api_key: str, log_prompts: bool, 
                   progress_lock: threading.Lock, completed_count: list, failed_count: list, completed_packages: set):
    """Worker function that processes library work items from the queue."""
    import threading
    import time
    import traceback
    worker_id = threading.current_thread().name
    logger.info(f"Worker {worker_id} started successfully")
    
    try:
        # Each worker gets its own LLM client
        logger.info(f"Worker {worker_id} initializing LLM client...")
        llm_client = LLMClient(llm_url, model, api_key)
        logger.info(f"Worker {worker_id} initialization complete")
        
        work_count = 0
        last_heartbeat = time.time()
        
        while True:
            work_item = None
            try:
                # Heartbeat every 30 seconds
                if time.time() - last_heartbeat > 30:
                    logger.info(f"Worker {worker_id} HEARTBEAT - alive and processing (work items done: {work_count})")
                    last_heartbeat = time.time()
                
                logger.info(f"Worker {worker_id} waiting for work item from queue...")
                work_item = work_queue.get(timeout=5)
                work_count += 1
                
                package_name = work_item.package_name
                library_name = work_item.library_name
                work_description = f"{package_name}::{library_name if library_name else 'no-library'}"
                
                logger.info(f"Worker {worker_id} picked up work item #{work_count}: {work_description}")
                
                start_time = time.time()
                
                # Process the library
                try:
                    result = process_library(work_item, llm_client, log_prompts)
                    
                    # Store result in global package results
                    with results_lock:
                        if library_name:
                            package_results[package_name]['libraries'][library_name] = {
                                "summary": result["summary"],
                                "modules": result["modules"]
                            }
                        else:
                            package_results[package_name]['modules_without_library'] = result["modules"]
                    
                    success = True
                    
                except Exception as work_error:
                    logger.error(f"Worker {worker_id} EXCEPTION processing {work_description}: {work_error}")
                    logger.error(f"Worker {worker_id} work traceback: {traceback.format_exc()}")
                    success = False
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Worker {worker_id} finished processing {work_description} in {duration:.1f}s")
                
                # Update progress counters
                try:
                    with progress_lock:
                        if success:
                            completed_count[0] += 1
                            logger.info(f"Worker {worker_id} completed work item: {work_description} (SUCCESS) - total completed: {completed_count[0]}")
                        else:
                            failed_count[0] += 1
                            logger.error(f"Worker {worker_id} failed work item: {work_description} (FAILED) - total failed: {failed_count[0]}")
                
                        # Track completed libraries and check if package is complete
                        if success:
                            # Mark this library as completed
                            library_completed = library_name if library_name else None
                            package_completed_libraries[package_name].add(library_completed)
                            
                            # Check if all expected libraries for this package are completed
                            expected_libs = package_expected_libraries[package_name]
                            completed_libs = package_completed_libraries[package_name]
                            
                            if expected_libs <= completed_libs:  # All libraries completed
                                if save_package_results(output_dir, package_name, completed_packages):
                                    logger.info(f"Package {package_name} completed and saved (libraries: {expected_libs})")
                                    # Clean up tracking
                                    del package_expected_libraries[package_name]
                                    del package_completed_libraries[package_name]
                            
                except Exception as count_error:
                    logger.error(f"Worker {worker_id} EXCEPTION updating progress counts: {count_error}")
                
                try:
                    work_queue.task_done()
                except Exception as task_done_error:
                    logger.error(f"Worker {worker_id} EXCEPTION in task_done: {task_done_error}")
                
            except queue.Empty:
                logger.info(f"Worker {worker_id} finished - no more work items in queue after processing {work_count} items")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered CRITICAL error processing {work_item.package_name if work_item else 'unknown'}: {e}")
                logger.error(f"Worker {worker_id} full traceback: {traceback.format_exc()}")
                
                if work_item:
                    try:
                        work_queue.task_done()
                    except Exception:
                        pass
                    
                    try:
                        with progress_lock:
                            failed_count[0] += 1
                    except Exception:
                        pass
    
    except Exception as fatal_error:
        logger.error(f"Worker {worker_id} FATAL ERROR in main loop: {fatal_error}")
        logger.error(f"Worker {worker_id} FATAL traceback: {traceback.format_exc()}")
    
    logger.info(f"Worker {worker_id} exiting after processing {work_count} work items")

def create_work_items(json_files: List[Path]) -> List[LibraryWorkItem]:
    """Create all work items from a list of package files."""
    all_work_items = []
    extractor = ModuleExtractor()
    
    for json_file in json_files:
        work_items = load_package_and_create_work_items(json_file, extractor)
        all_work_items.extend(work_items)
    
    return all_work_items

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

def debug_module_prompts(args):
    """Debug mode: show prompts for a specific module without making API calls."""
    
    module_path = args.debug
    input_dir = Path(args.input_dir)
    
    print(f"=== DEBUG MODE: {module_path} ===")
    print(f"Input directory: {input_dir}")
    print()
    
    # Find the package file
    if args.package:
        json_file = input_dir / f"{args.package}.json"
        if not json_file.exists():
            print(f"Error: Package file {json_file} does not exist")
            return
    else:
        # Try to infer package from module path
        package_name = module_path.split('.')[0].lower()
        json_file = input_dir / f"{package_name}.json"
        if not json_file.exists():
            print(f"Error: Could not find package file for '{module_path}'. Try using --package flag.")
            print(f"Looked for: {json_file}")
            return
    
    print(f"Package file: {json_file}")
    
    # Extract module using the same logic as the main script
    extractor = ModuleExtractor()
    try:
        modules = extractor.extract_from_parsed_json(json_file)
    except Exception as e:
        print(f"Error extracting modules: {e}")
        return
    
    # Find our target module
    target_module = None
    for module in modules:
        if module.path == module_path:
            target_module = module
            break
    
    if not target_module:
        print(f"Module '{module_path}' not found")
        print("\nAvailable modules:")
        for module in modules:
            print(f"  - {module.path}")
        return
    
    print(f"Module name: {target_module.name}")
    print(f"Module path: {target_module.path}")
    print(f"Elements: {len(target_module.elements)}")
    code_elements = [elem for elem in target_module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
    print(f"Code elements: {len(code_elements)}")
    print(f"Submodules: {len(target_module.modules)}")
    print(f"Children: {target_module.children}")
    print()
    
    # Check processing strategy
    has_own_content = any(elem.get('kind') in ['value', 'type', 'module-type'] for elem in target_module.elements)
    
    print(f"Has own content: {has_own_content}")
    print(f"Has children: {bool(target_module.children)}")
    
    if not target_module.children:
        strategy = "Leaf module"
    elif not has_own_content:
        strategy = "Pure parent module"
    else:
        strategy = "Hybrid module"
    
    print(f"Processing strategy: {strategy}")
    print()
    
    # Create a mock LLM client to generate prompts without API calls
    class DebugLLMClient(LLMClient):
        def __init__(self):
            # Skip the parent __init__ to avoid API setup
            pass
            
        def _show_prompt(self, title, prompt):
            print("="*80)
            print(title)
            print("="*80)
            print(prompt)
            print("="*80)
            print()
            
        def _show_chunked_prompts(self, module: ModuleContent, all_modules: Dict[str, ModuleContent]):
            """Show all prompts that would be generated for chunked description."""
            # Extract code elements (functions, types, modules) from ordered elements
            code_elements = [elem for elem in module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
            
            chunk_size = 20
            
            # Show chunk prompts using the actual prompt building logic
            for i in range(0, len(code_elements), chunk_size):
                chunk = code_elements[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                
                chunk_prompt = self._build_chunk_prompt(module, chunk, chunk_num, all_modules)
                self._show_prompt(f"CHUNK {chunk_num} PROMPT", chunk_prompt)
            
            # Show final combining prompt using the actual prompt building logic
            mock_summaries = [f"Chunk {i+1}: [would contain summary of chunk {i+1}]" 
                            for i in range(len(range(0, len(code_elements), chunk_size)))]
            
            final_prompt = self._build_chunk_combine_prompt(module, mock_summaries, all_modules)
            self._show_prompt("FINAL COMBINING PROMPT", final_prompt)
    
    debug_client = DebugLLMClient()
    
    # Create all_modules dictionary for ancestor lookup
    all_modules = {module.path: module for module in modules}
    
    if strategy == "Leaf module":
        print("PROCESSING AS LEAF MODULE")
        print()
        
        # Check if simple or chunked
        code_elements = [elem for elem in target_module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
        total_items = len(code_elements)
        
        if total_items <= 20:
            prompt = debug_client._build_simple_description_prompt(target_module, all_modules)
            debug_client._show_prompt("SIMPLE DESCRIPTION PROMPT", prompt)
        else:
            print(f"This module has {total_items} items - using CHUNKED DESCRIPTION")
            print()
            debug_client._show_chunked_prompts(target_module, all_modules)
            
    elif strategy == "Pure parent module":
        print("PROCESSING AS PURE PARENT MODULE")
        print("This would use MERGE DESCRIPTIONS with child modules")
        print("Child descriptions would be generated first, then merged")
        print()
        print("Example merge prompt:")
        
        mock_child_descs = [f"Child {child}: [would contain description of {child}]" 
                           for child in target_module.children[:3]]  # Show first 3
        if len(target_module.children) > 3:
            mock_child_descs.append(f"... and {len(target_module.children) - 3} more children")
        
        merge_prompt = f"""You are an expert OCaml developer. Write a 2-3 sentence description for the {target_module.name} module based on its submodules:

Module: {target_module.name}
Documentation: {target_module.documentation or "No documentation"}

Submodule descriptions:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(mock_child_descs))}

Description:"""
        
        debug_client._show_prompt("MERGE DESCRIPTIONS PROMPT", merge_prompt)
        
    else:  # Hybrid module
        print("PROCESSING AS HYBRID MODULE")
        print()
        print("1. First generating own description:")
        
        # Check if simple or chunked
        code_elements = [elem for elem in target_module.elements if elem.get('kind') in ['value', 'type', 'module', 'module-type']]
        total_items = len(code_elements)
        
        if total_items <= 20:
            prompt = debug_client._build_simple_description_prompt(target_module, all_modules)
            debug_client._show_prompt("OWN FUNCTIONALITY PROMPT", prompt)
        else:
            print(f"Own functionality has {total_items} items - using CHUNKED DESCRIPTION")
            print()
            debug_client._show_chunked_prompts(target_module, all_modules)
        
        if target_module.children:
            print(f"2. Then merging with {len(target_module.children)} children using HYBRID MERGE")
            print()
            
            mock_child_descs = [f"Child {child}: [would contain description of {child}]" 
                               for child in target_module.children[:3]]  # Show first 3
            if len(target_module.children) > 3:
                mock_child_descs.append(f"... and {len(target_module.children) - 3} more children")
            
            hybrid_merge_prompt = f"""You are an expert OCaml developer. Write a comprehensive 3-4 sentence description combining this module's own functionality with its submodules:

Module: {target_module.name}
Own functionality: [would contain description generated from step 1]

Submodule descriptions:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(mock_child_descs))}

Combined Description:"""
            
            debug_client._show_prompt("HYBRID MERGE PROMPT", hybrid_merge_prompt)
        else:
            print("2. No children to merge with")

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
    parser.add_argument("--api-key-file", help="File containing API key (if needed for remote endpoint)")
    parser.add_argument("--debug", help="Debug mode: show prompts for specific module path (e.g., 'Cmdliner.Term') without making API calls")
    
    args = parser.parse_args()
    
    # Handle debug mode
    if args.debug:
        debug_module_prompts(args)
        return
    
    # Read API key from file if provided
    api_key = "dummy_key"
    if args.api_key_file:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key = f.read().strip()
            logger.info(f"API key loaded from {args.api_key_file}")
        except Exception as e:
            logger.error(f"Failed to read API key from {args.api_key_file}: {e}")
            return
    
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
    
    # Create work items for all libraries
    logger.info("Creating work items for all libraries...")
    all_work_items = create_work_items(json_files)
    total_work_items = len(all_work_items)
    
    logger.info(f"Created {total_work_items} library work items from {len(json_files)} packages")
    
    # Estimate time based on ~15 seconds per library (improved parallelism)
    estimated_hours = (total_work_items * 15) / 3600
    logger.info(f"Estimated time: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    
    # Shuffle the work items to distribute load better
    import random
    random.shuffle(all_work_items)
    logger.info(f"Shuffled work items to distribute load")
    
    # Create work queue
    work_queue = queue.Queue()
    for work_item in all_work_items:
        work_queue.put(work_item)
    
    # Shared progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference
    failed_count = [0]
    completed_packages = set()  # Track completed packages
    
    # Main thread signal handlers
    def main_signal_handler(signum, frame):
        logger.error(f"MAIN thread received signal {signum} - process being killed!")
        logger.error(f"MAIN signal traceback: {''.join(traceback.format_stack(frame))}")
        raise KeyboardInterrupt(f"Main signal {signum}")
    
    signal.signal(signal.SIGTERM, main_signal_handler)
    signal.signal(signal.SIGINT, main_signal_handler)
    
    logger.info(f"Starting ThreadPoolExecutor with {args.workers} workers...")
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            logger.info("ThreadPoolExecutor created successfully")
            
            # Submit worker tasks
            workers = []
            for i in range(args.workers):
                logger.info(f"Submitting worker {i}...")
                future = executor.submit(
                    library_worker, 
                    work_queue, 
                    output_dir, 
                    args.llm_url, 
                    args.model, 
                    api_key,
                    args.log_prompts,
                    progress_lock,
                    completed_count,
                    failed_count,
                    completed_packages
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
            logger.info(f"Starting progress monitoring for {total_work_items} work items...")
            
            try:
                with tqdm(total=total_work_items, desc="Processing libraries", unit="library") as pbar:
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
                            logger.debug(f"Progress check: {current_completed}/{total_work_items} completed")
                        
                        # Update progress bar
                        if current_completed > last_completed:
                            delta = current_completed - last_completed
                            pbar.update(delta)
                            last_completed = current_completed
                            logger.info(f"Progress updated: {current_completed}/{total_work_items} work items processed")
                        
                        # Check if all work items are done
                        if current_completed >= total_work_items:
                            logger.info("All work items completed!")
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
                work_queue.join()
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