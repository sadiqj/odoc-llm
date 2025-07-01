#!/usr/bin/env python3
"""
Extract and parse OCaml documentation from odoc-generated markdown files.
For each package, finds the latest version and extracts all module documentation
into a structured JSON format.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from version_utils import find_latest_version
from parse_markdown import parse_markdown_documentation, extract_module_path

def find_all_packages(docs_dir: Path) -> List[str]:
    """Find all package directories in the docs folder."""
    packages = []
    for item in docs_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            packages.append(item.name)
    return sorted(packages)

def find_package_versions(package_dir: Path) -> List[str]:
    """Find all version directories for a package."""
    versions = []
    for item in package_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            versions.append(item.name)
    return versions

def check_build_status(version_dir: Path) -> Dict[str, Any]:
    """Check the build status of a package version."""
    status_file = version_dir / 'status.json'
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    
    # For markdown docs, if there's a doc directory, assume the build succeeded
    doc_dir = version_dir / 'doc'
    if doc_dir.exists() and doc_dir.is_dir():
        return {'failed': False}
    
    return {'failed': True, 'error': 'No documentation found'}

def is_valid_ocaml_module_name(name: str) -> bool:
    """Check if a string is a valid OCaml module name."""
    if not name:
        return False
    # Module names must start with a capital letter and contain only letters, numbers, underscores, and apostrophes
    if not name[0].isupper():
        return False
    return all(c.isalnum() or c in ('_', "'") for c in name)

def is_library_directory(dir_path: Path) -> bool:
    """Check if a directory under doc/ is a library directory."""
    # Library directories in markdown format have:
    # 1. An index.md file in them  
    # 2. One or more .md files with module names (e.g., Module.md or Module-Submodule.md)
    
    index_file = dir_path / 'index.md'
    if not index_file.exists():
        return False
    
    # Check for module markdown files
    has_module_files = False
    for item in dir_path.iterdir():
        if item.is_file() and item.suffix == '.md' and item.name != 'index.md':
            # Check if the filename looks like a module (starts with capital letter)
            base_name = item.stem
            if base_name and base_name[0].isupper():
                has_module_files = True
                break
    
    return has_module_files

def find_documentation_files(version_dir: Path) -> List[Tuple[Path, Optional[str]]]:
    """Recursively find all .md files in the documentation.
    Returns list of (file_path, library_name) tuples where library_name is None for non-library files."""
    doc_files = []
    
    # Main documentation directory
    doc_dir = version_dir / 'doc'
    if doc_dir.exists():
        # First, process markdown files directly in doc/
        for md_file in doc_dir.glob('*.md'):
            doc_files.append((md_file, None))
        
        # Then process library directories
        for item in doc_dir.iterdir():
            if item.is_dir():
                if is_library_directory(item):
                    # This is a library directory
                    library_name = item.name
                    library_index = item / 'index.md'
                    
                    # Process all markdown files in the library
                    for md_file in item.rglob('*.md'):
                        if md_file != library_index:  # Skip the library's main index - it's just a landing page
                            doc_files.append((md_file, library_name))
                else:
                    # This is a non-library directory, get all markdown files
                    for md_file in item.rglob('*.md'):
                        doc_files.append((md_file, None))
    
    return doc_files

def load_package_metadata(version_dir: Path) -> Optional[Dict[str, Any]]:
    """Load package.json metadata if available."""
    package_file = version_dir / 'package.json'
    if package_file.exists():
        with open(package_file, 'r') as f:
            return json.load(f)
    return None

def build_module_hierarchy(modules: List[Dict[str, Any]]) -> None:
    """Build parent-child relationships between modules based on their paths."""
    # Convert string paths to list format for consistent handling
    for module in modules:
        if "module_path" in module:
            if isinstance(module["module_path"], str):
                module["module_path"] = module["module_path"].split(".")
    
    # Build parent-child relationships
    for module in modules:
        module_path = module.get("module_path", [])
        if not module_path:
            continue
            
        # Initialize children list if not present
        if "children" not in module:
            module["children"] = []
        
        # Look for potential children
        for other_module in modules:
            other_path = other_module.get("module_path", [])
            if not other_path or len(other_path) <= len(module_path):
                continue
                
            # Check if other_module is a direct child of this module
            if (len(other_path) == len(module_path) + 1 and 
                other_path[:len(module_path)] == module_path):
                
                child_path_str = ".".join(other_path)
                if child_path_str not in module["children"]:
                    module["children"].append(child_path_str)

def process_documentation_file(doc_file: Path, package_name: str, version: str, version_dir: Path, library_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Process a single documentation file."""
    try:
        # Parse the documentation
        parsed = parse_markdown_documentation(str(doc_file))
        
        # Add metadata
        parsed['package'] = package_name
        parsed['version'] = version
        parsed['file_path'] = str(doc_file.relative_to(version_dir))  # Relative to version dir
        parsed['library'] = library_name  # Will be None for non-library files
        
        # Extract module path from file path
        parsed['module_path'] = extract_module_path(str(doc_file))
        
        # Determine if this is a module type based on file path
        filename = doc_file.name
        if filename.endswith('.md'):
            filename = filename[:-3]
        parsed['is_module_type'] = '-module-type-' in filename and filename.split('-module-type-')[-1] != ''
        
        return parsed
    except Exception as e:
        print(f"Error processing {doc_file}: {e}")
        return None

def process_package_version(package_name: str, version: str, docs_dir: Path) -> Optional[Dict[str, Any]]:
    """Process all documentation for a specific package version."""
    version_dir = docs_dir / package_name / version
    
    # Check build status
    status = check_build_status(version_dir)
    if status.get('failed', False):
        print(f"Skipping {package_name} {version}: build failed")
        return None
    
    # Find all documentation files
    doc_files = find_documentation_files(version_dir)
    if not doc_files:
        print(f"No documentation files found for {package_name} {version}")
        return None
    
    # Load package metadata
    metadata = load_package_metadata(version_dir)
    
    # Process all documentation files
    modules = []
    package_docs = {}
    other_docs = {}
    
    for doc_file, library_name in doc_files:
        result = process_documentation_file(doc_file, package_name, version, version_dir, library_name)
        if result:
            # Categorize the documentation
            rel_path = str(doc_file.relative_to(version_dir))
            if rel_path.startswith('doc/'):
                if library_name:
                    # This is a library module
                    modules.append(result)
                else:
                    # This is package-level documentation
                    doc_type = doc_file.stem.split('.')[0].upper()
                    package_docs[doc_type] = result
            else:
                # It's a README, LICENSE, etc. at the version root
                doc_type = doc_file.stem.split('.')[0].upper()
                other_docs[doc_type] = result
    
    # Build parent-child relationships between modules
    build_module_hierarchy(modules)
    
    # Build the package documentation structure
    package_doc = {
        'package': package_name,
        'version': version,
        'metadata': metadata,
        'modules': modules,
        'package_documentation': package_docs,
        'documentation': other_docs,
        'statistics': {
            'total_modules': len(modules),
            'total_types': sum(len(m.get('types', [])) for m in modules),
            'total_values': sum(len(m.get('values', [])) for m in modules),
            'total_submodules': sum(len(m.get('modules', [])) for m in modules),
            'total_elements': sum(len(m.get('elements', [])) for m in modules)
        }
    }
    
    return package_doc

def process_package(package_name: str, docs_dir: Path) -> Optional[Dict[str, Any]]:
    """Process a single package - find latest version and extract documentation."""
    package_dir = docs_dir / package_name
    
    # Find all versions
    versions = find_package_versions(package_dir)
    if not versions:
        print(f"No versions found for {package_name}")
        return None
    
    # Find the latest version
    latest_version, _ = find_latest_version(versions)
    if not latest_version:
        print(f"Could not determine latest version for {package_name}")
        return None
    
    print(f"Processing {package_name} {latest_version}")
    
    # Process the latest version
    return process_package_version(package_name, latest_version, docs_dir)

def save_package_documentation(package_doc: Dict[str, Any], output_dir: Path):
    """Save package documentation to a JSON file."""
    if not package_doc:
        return
    
    package_name = package_doc['package']
    output_file = output_dir / f"{package_name}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(package_doc, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Extract OCaml documentation to structured JSON')
    parser.add_argument('--docs-dir', type=Path, default=Path('docs-md'),
                        help='Directory containing OCaml documentation')
    parser.add_argument('--output-dir', type=Path, default=Path('parsed-docs'),
                        help='Output directory for parsed documentation')
    parser.add_argument('--package', type=str, help='Process only a specific package')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, help='Limit number of packages to process')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Find packages to process
    if args.package:
        packages = [args.package]
    else:
        packages = find_all_packages(args.docs_dir)
        if args.limit:
            packages = packages[:args.limit]
    
    print(f"Found {len(packages)} packages to process")
    
    # Process packages
    successful = 0
    failed = 0
    
    # Create index for all packages
    index = {
        'total_packages': len(packages),
        'packages': []
    }
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        future_to_package = {
            executor.submit(process_package, pkg, args.docs_dir): pkg 
            for pkg in packages
        }
        
        # Process results with progress bar
        with tqdm(total=len(packages), desc="Processing packages") as pbar:
            for future in as_completed(future_to_package):
                package_name = future_to_package[future]
                
                try:
                    result = future.result()
                    if result:
                        save_package_documentation(result, args.output_dir)
                        successful += 1
                        
                        # Add to index
                        index['packages'].append({
                            'name': package_name,
                            'version': result['version'],
                            'modules': result['statistics']['total_modules'],
                            'types': result['statistics']['total_types'],
                            'values': result['statistics']['total_values']
                        })
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error processing {package_name}: {e}")
                    failed += 1
                
                pbar.update(1)
    
    # Save index
    index_file = args.output_dir / 'index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()