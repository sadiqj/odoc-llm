#!/usr/bin/env python3
"""
Generate concise package descriptions for OCaml packages using LLM.
Extracts README content and generates 3-4 sentence summaries.
"""

import json
import os
import logging
import re
import signal
import sys
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence HTTP request logging from OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global shutdown flag
shutdown_requested = False

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@dataclass
class PackageInfo:
    """Information extracted from a package."""
    name: str
    version: str
    readme_content: str
    changes_content: str = ""
    license_content: str = ""
    statistics: Optional[Dict[str, Any]] = None

class LLMClient:
    """OpenAI-compatible client for generating package descriptions."""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen/Qwen3-30B-A3B-FP8"):
        try:
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="dummy_key",  # Local endpoint doesn't need real key
                timeout=60.0,  # 1 minute timeout for requests
                max_retries=1
            )
            self.model = model
            logger.info(f"LLMClient initialized with base_url={base_url}, model={model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_package_description(self, package: PackageInfo, log_prompts: bool = False) -> str:
        """Generate a concise 3-4 sentence description for a package."""
        
        # Build context from available documentation
        context_parts = [f"Package: {package.name}"]
        
        if package.readme_content:
            # Clean up README content - remove HTML tags and excessive whitespace
            clean_readme = re.sub(r'<[^>]+>', '', package.readme_content)
            clean_readme = re.sub(r'\s+', ' ', clean_readme).strip()
            # Limit README content to avoid overly long prompts
            if len(clean_readme) > 2000:
                clean_readme = clean_readme[:2000] + "..."
            context_parts.append(f"README content: {clean_readme}")
        
        if package.statistics:
            stats = package.statistics
            context_parts.append(f"Package contains {stats.get('total_modules', 0)} modules, {stats.get('total_values', 0)} functions, {stats.get('total_types', 0)} types")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert OCaml developer. Based on the package information below, write a concise 3-4 sentence description of what this OCaml package does.

Focus on:
- The main purpose and functionality of the package
- What problems it solves or what domain it addresses
- Key capabilities or features it provides
- Practical use cases

Write in a clear, informative style. Do not:
- Repeat the package name unnecessarily
- Use generic phrases like "provides functionality for"
- Mention implementation details or internal architecture
- Include installation or usage instructions

{context}

Package description:"""

        if log_prompts:
            logger.info(f"=== PROMPT for {package.name} ===")
            logger.info(prompt)
            logger.info("=== END PROMPT ===")
        
        try:
            logger.info(f"Generating description for package {package.name}")
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert OCaml developer. Provide concise, direct answers without thinking aloud or explanation. /no_think"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1
            )
            elapsed = time.time() - start_time
            logger.info(f"Generated description for {package.name} in {elapsed:.1f}s")
            
            if not response.choices:
                logger.error(f"LLM returned no choices for package {package.name}")
                return f"OCaml package for {package.name.replace('-', ' ')}"
            
            result = response.choices[0].message.content.strip()
            
            # Filter out think tags and their content
            result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL).strip()
            
            if log_prompts:
                logger.info(f"=== RESPONSE for {package.name} ===")
                logger.info(result)
                logger.info("=== END RESPONSE ===")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"LLM error for package {package.name} after {elapsed:.1f}s: {type(e).__name__}: {e}")
            return f"OCaml package for {package.name.replace('-', ' ')}"

def extract_package_info(json_file: Path) -> Optional[PackageInfo]:
    """Extract package information from parsed JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        package_name = data.get('package', json_file.stem)
        version = data.get('version', 'unknown')
        statistics = data.get('statistics', {})
        
        # Extract README content - handle both old and new format
        readme_content = ""
        # Try new format first (package_documentation)
        package_documentation = data.get('package_documentation', {})
        # Fall back to old format (documentation)
        if not package_documentation:
            package_documentation = data.get('documentation', {})
            
        if 'README' in package_documentation and isinstance(package_documentation['README'], dict):
            readme_doc = package_documentation['README']
            # Try different fields that might contain the actual content
            readme_content = (
                readme_doc.get('preamble', '') +
                ' '.join(readme_doc.get('documentation_sections', []))
            ).strip()
        
        # Extract other documentation if available
        changes_content = ""
        if 'CHANGES' in package_documentation and isinstance(package_documentation['CHANGES'], dict):
            changes_doc = package_documentation['CHANGES']
            changes_content = (
                changes_doc.get('preamble', '') +
                ' '.join(changes_doc.get('documentation_sections', []))
            ).strip()
        
        # If no README content, try to extract from main module documentation
        if not readme_content and 'modules' in data and data['modules']:
            # Look for package-level module with documentation
            for module in data['modules']:
                if module.get('module_path') == package_name or not module.get('module_path'):
                    module_docs = module.get('documentation_sections', [])
                    if module_docs:
                        readme_content = ' '.join(module_docs[:2])  # Take first 2 sections
                        break
        
        if not readme_content and not changes_content:
            logger.warning(f"No documentation content found for package {package_name}")
            return None
        
        return PackageInfo(
            name=package_name,
            version=version,
            readme_content=readme_content,
            changes_content=changes_content,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Error extracting package info from {json_file}: {e}")
        return None

def process_single_package(json_file: Path, output_dir: Path, llm_client: LLMClient, log_prompts: bool = False) -> bool:
    """Process a single package and generate its description."""
    global shutdown_requested
    
    if shutdown_requested:
        return False
    
    try:
        package_info = extract_package_info(json_file)
        if not package_info:
            logger.warning(f"Skipping {json_file.stem} - no extractable content")
            return False
        
        # Generate description
        description = llm_client.generate_package_description(package_info, log_prompts)
        
        # Save to output file
        output_file = output_dir / f"{package_info.name}.json"
        output_data = {
            "package": package_info.name,
            "version": package_info.version,
            "description": description
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated description for {package_info.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing package {json_file.stem}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate concise descriptions for OCaml packages")
    parser.add_argument("--input-dir", default="parsed-docs", help="Directory with parsed JSON files")
    parser.add_argument("--output-dir", default="package-descriptions", help="Output directory for package descriptions")
    parser.add_argument("--limit", type=int, help="Limit number of packages to process")
    parser.add_argument("--package", help="Process specific package only")
    parser.add_argument("--llm-url", default="http://localhost:8000", help="LLM endpoint URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8", help="LLM model name")
    parser.add_argument("--log-prompts", action="store_true", help="Log prompts and responses sent to LLM")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return 1
    
    output_dir.mkdir(exist_ok=True)
    
    # Collect input files
    json_files = list(input_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Filter by specific package if specified
    if args.package:
        json_files = [f for f in json_files if f.stem == args.package]
        logger.info(f"Filtered to {len(json_files)} files for package: {args.package}")
    
    # Skip packages that already have descriptions
    json_files = [f for f in json_files if not (output_dir / f"{f.stem}.json").exists()]
    logger.info(f"Processing {len(json_files)} packages (skipping existing descriptions)")
    
    # Apply limit if specified
    if args.limit:
        json_files = json_files[:args.limit]
        logger.info(f"Limited to {len(json_files)} packages")
    
    if not json_files:
        logger.info("No packages to process")
        return 0
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(args.llm_url, args.model)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return 1
    
    # Process packages
    logger.info(f"Starting to process {len(json_files)} packages with {args.workers} workers")
    
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_package, json_file, output_dir, llm_client, args.log_prompts): json_file
            for json_file in json_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(json_files), desc="Processing packages") as pbar:
            for future in as_completed(future_to_file):
                if shutdown_requested:
                    logger.info("Shutdown requested, cancelling remaining tasks")
                    break
                
                json_file = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Task failed for {json_file.stem}: {e}")
                    failed += 1
                
                pbar.update(1)
                pbar.set_postfix(completed=completed, failed=failed)
    
    logger.info(f"Processing complete. Completed: {completed}, Failed: {failed}")
    return 0 if not shutdown_requested else 1

if __name__ == "__main__":
    sys.exit(main())