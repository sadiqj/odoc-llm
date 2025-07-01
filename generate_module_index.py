#!/usr/bin/env python3
"""
Generate BM25 indexes for OCaml module documentation.

This script creates per-package BM25 indexes from parsed documentation,
enabling fast full-text search across module documentation.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import bm25s
from tqdm import tqdm
import signal
import sys
import numpy as np
import re


def signal_handler(signum, frame):
    """Handle interruption signals gracefully."""
    print("\nInterrupted. Exiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def extract_module_documentation(module: Dict) -> Tuple[str, str]:
    """Extract all documentation text from a module.
    
    Returns:
        Tuple of (module_path, combined_documentation_text)
    """
    texts = []
    
    # Add module documentation
    if "documentation" in module and module["documentation"]:
        texts.append(module["documentation"])
    
    # Add elements documentation
    if "elements" in module and module["elements"]:
        for element in module["elements"]:
            if element.get("kind") in ["value", "type", "module", "module-type"]:
                if "documentation" in element and element["documentation"]:
                    texts.append(element["documentation"])
    
    # Combine all documentation with space separation
    combined_text = " ".join(texts).strip()
    
    return (module.get("module_path", ""), combined_text)


def build_package_index(package_file: Path, output_dir: Path) -> None:
    """Build BM25 index for a single package."""
    print(f"Processing {package_file.stem}...")
    
    # Load package data
    with open(package_file, 'r') as f:
        data = json.load(f)
    
    # Extract documentation from all modules
    module_docs = []
    module_paths = []
    
    for module in data.get("modules", []):
        module_path, doc_text = extract_module_documentation(module)
        if doc_text:  # Only include modules with documentation
            module_paths.append(module_path)
            module_docs.append(doc_text)
    
    if not module_docs:
        print(f"  No documentation found in {package_file.stem}, skipping...")
        return
    
    # Create BM25 index
    corpus_tokens = bm25s.tokenize(module_docs, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Save index and metadata
    package_output_dir = output_dir / package_file.stem
    package_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the BM25 index
    retriever.save(str(package_output_dir / "index"))
    
    # Save module paths for result mapping
    with open(package_output_dir / "module_paths.json", 'w') as f:
        json.dump(module_paths, f, indent=2)
    
    # Save package metadata
    metadata = {
        "package": data.get("package", package_file.stem),
        "version": data.get("version", "unknown"),
        "module_count": len(module_paths),
        "total_documents": len(module_docs)
    }
    with open(package_output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Indexed {len(module_docs)} modules with documentation")


def main():
    parser = argparse.ArgumentParser(description="Generate BM25 indexes for OCaml documentation")
    parser.add_argument("--input-dir", default="parsed-docs", 
                        help="Directory containing parsed documentation JSON files")
    parser.add_argument("--output-dir", default="module-indexes",
                        help="Directory to save BM25 indexes")
    parser.add_argument("--packages", nargs="+",
                        help="Specific packages to index (default: all)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of packages to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Get list of packages to process
    if args.packages:
        package_files = [input_dir / f"{pkg}.json" for pkg in args.packages]
        package_files = [f for f in package_files if f.exists()]
    else:
        package_files = sorted(input_dir.glob("*.json"))
    
    if args.limit:
        package_files = package_files[:args.limit]
    
    print(f"Found {len(package_files)} packages to index")
    
    # Process each package
    for package_file in tqdm(package_files, desc="Building indexes"):
        try:
            build_package_index(package_file, output_dir)
        except Exception as e:
            print(f"Error processing {package_file.stem}: {e}")
            continue
    
    print(f"\nIndexing complete. Indexes saved to {output_dir}/")


if __name__ == "__main__":
    main()