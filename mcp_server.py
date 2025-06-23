#!/usr/bin/env python3
"""
FastMCP Server for OCaml Documentation Search

This server exposes tools for searching and interacting with the OCaml
documentation dataset through the Model Context Protocol (MCP) using HTTP SSE.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import shlex

from mcp.server.fastmcp import FastMCP

# Import semantic search components
from semantic_search import SemanticSearch
from unified_search import UnifiedSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ocaml-search")

# Global search engine instances (lazy loaded)
search_engine: Optional[SemanticSearch] = None
unified_engine: Optional[UnifiedSearchEngine] = None
embeddings_dir = Path("package-embeddings")
package_descriptions_dir = Path("package-descriptions")
indexes_dir = Path("module-indexes")

@mcp.tool()
async def find_ocaml_packages(functionality: str) -> Dict[str, Any]:
    """
    Find OCaml packages that provide specific functionality.
    
    Takes a natural language description of the desired functionality and 
    returns the top 5 most relevant packages with their modules.
    
    Args:
        functionality: Natural language description of the desired functionality 
                      (e.g., 'HTTP server', 'JSON parsing', 'cryptographic hashing')
    
    Returns:
        Dictionary containing query and list of matching packages with modules
    """
    global search_engine
    
    # Initialize search engine on first use (lazy loading)
    if search_engine is None:
        logger.info("Initializing semantic search engine...")
        try:
            search_engine = SemanticSearch(embeddings_dir)
        except Exception as e:
            error_msg = f"Failed to initialize search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform semantic search
        results = search_engine.search(functionality, top_k=5)
        
        # Format results for MCP response
        return {
            "query": functionality,
            "packages": [
                {
                    "package": result["package"],
                    "module": result["module_path"],
                    "description": result["description"],
                    "similarity": round(result["similarity_score"], 4)
                }
                for result in results
            ]
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def get_package_summary(package_name: str) -> Dict[str, Any]:
    """
    Get a concise summary of an OCaml package.
    
    Returns a 3-4 sentence description of what the package does, its main purpose,
    key capabilities, and practical use cases.
    
    Args:
        package_name: Name of the OCaml package (e.g., 'lwt', 'base', 'cohttp')
    
    Returns:
        Dictionary containing package name, version, and description
    """
    try:
        description_file = package_descriptions_dir / f"{package_name}.json"
        
        if not description_file.exists():
            return {
                "error": f"No description found for package '{package_name}'. Package may not exist or description not yet generated."
            }
        
        with open(description_file, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        return {
            "package": package_data["package"],
            "version": package_data["version"],
            "description": package_data["description"]
        }
        
    except Exception as e:
        error_msg = f"Failed to get package summary: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def test_opam_compatibility(packages: List[str]) -> Dict[str, Any]:
    """
    Test opam package compatibility for a list of packages.
    
    Runs 'opam list --installable --all-versions -s' to find concrete package 
    versions that are compatible with the current switch.

    Requires the current opam switch for the project to be the active one.
    
    Args:
        packages: List of opam package names to test for compatibility
    
    Returns:
        Dictionary containing the list of compatible package versions and any errors
    """
    if not packages:
        return {"error": "No packages provided"}
    
    # Join packages with spaces and ensure proper escaping
    package_list = " ".join(shlex.quote(pkg) for pkg in packages)
    cmd = f"opam list --installable --all-versions -s {package_list}"
    
    logger.info(f"Running opam compatibility test: {cmd}")
    
    try:
        # Run the opam command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode != 0:
            return {
                "error": f"opam command failed with exit code {result.returncode}",
                "stderr": result.stderr.strip()
            }
        
        # Parse the output - opam list returns one package per line
        output_lines = result.stdout.strip().split('\n')
        compatible_versions = []
        
        for line in output_lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments and empty lines
                compatible_versions.append(line)
        
        return {
            "packages_tested": packages,
            "compatible_versions": compatible_versions,
            "count": len(compatible_versions)
        }
        
    except subprocess.TimeoutExpired:
        return {
            "error": "opam command timed out after 30 seconds",
            "packages_tested": packages
        }
    except Exception as e:
        error_msg = f"Failed to run opam command: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def search_ocaml_modules(query: str, packages: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    Search OCaml modules using both semantic similarity and keyword matching.
    
    Combines two search methods for comprehensive results:
    1. Semantic search using embeddings to find conceptually related modules
    2. Full-text search using BM25 to find modules containing specific keywords
    
    Results are deduplicated so each module appears only once across both methods.
    
    Args:
        query: Natural language query describing the desired functionality
               (e.g., 'HTTP server', 'JSON parsing', 'list operations')
        packages: List of package names to search within
                 (e.g., ['base', 'lwt', 'cohttp', 'yojson'])
        top_k: Maximum number of results to return from each search method (default: 5)
    
    Returns:
        Dictionary containing:
        - query: The search query
        - packages_searched: List of packages that were searched
        - semantic_results: Results from embedding-based semantic search
        - keyword_results: Results from BM25 full-text search
    """
    global unified_engine
    
    if not packages:
        return {"error": "No packages specified. Please provide a list of package names to search."}
    
    # Initialize unified search engine on first use (lazy loading)
    if unified_engine is None:
        logger.info("Initializing unified search engine...")
        try:
            unified_engine = UnifiedSearchEngine(
                embedding_dir=str(embeddings_dir),
                index_dir=str(indexes_dir)
            )
        except Exception as e:
            error_msg = f"Failed to initialize unified search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform unified search
        results = unified_engine.unified_search(query, packages, top_k)
        
        # Format results for MCP response
        return {
            "query": results["query"],
            "packages_searched": results["packages_searched"],
            "semantic_results": [
                {
                    "package": r["package"],
                    "module": r["module_name"],
                    "module_path": r["module_path"],
                    "similarity_score": round(r["score"], 4),
                    "description": r["description"]
                }
                for r in results["embedding_results"]
            ],
            "keyword_results": [
                {
                    "package": r["package"],
                    "module": r["module_name"],
                    "module_path": r["module_path"],
                    "relevance_score": round(r["score"], 4)
                }
                for r in results["bm25_results"]
            ],
            "total_results": len(results["embedding_results"]) + len(results["bm25_results"])
        }
        
    except Exception as e:
        error_msg = f"Unified search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


# Additional tool functions can be added here using the @mcp.tool() decorator
#     # Implementation here
#     pass


def main():
    """Main entry point for the FastMCP server."""
    import sys
    import asyncio
    
    # Check for --test flag for direct testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run a simple test
        async def test():
            if len(sys.argv) > 2 and sys.argv[2] == "--summary":
                # Test package summary functionality
                package = sys.argv[3] if len(sys.argv) > 3 else "lwt"
                print(f"Testing package summary for: {package}\n")
                result = await get_package_summary(package)
                print(json.dumps(result, indent=2))
            elif len(sys.argv) > 2 and sys.argv[2] == "opam":
                # Test opam compatibility
                packages = sys.argv[3:] if len(sys.argv) > 3 else ["base", "core"]
                print(f"Testing opam compatibility for: {packages}\n")
                result = await test_opam_compatibility(packages)
                print(json.dumps(result, indent=2))
            elif "--packages" in sys.argv:
                # Test unified search functionality
                query_idx = sys.argv.index("--test") + 1
                packages_idx = sys.argv.index("--packages") + 1
                query = sys.argv[query_idx] if query_idx < len(sys.argv) else "HTTP server"
                packages = sys.argv[packages_idx:] if packages_idx < len(sys.argv) else ["base", "lwt", "cohttp"]
                print(f"Testing unified search query: {query}")
                print(f"Packages: {packages}\n")
                result = await search_ocaml_modules(query, packages)
                print(json.dumps(result, indent=2))
            else:
                # Test search functionality
                query = sys.argv[2] if len(sys.argv) > 2 else "HTTP server"
                print(f"Testing search query: {query}\n")
                result = await find_ocaml_packages(query)
                print(json.dumps(result, indent=2))
        
        asyncio.run(test())
    else:
        # Run FastMCP server with HTTP SSE transport
        # Default SSE endpoint will be available at /sse
        # Using default port (probably 8000), may conflict with embedding server
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()