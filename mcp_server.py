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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ocaml-search")

# Global search engine instance (lazy loaded)
search_engine: Optional[SemanticSearch] = None
embeddings_dir = Path("package_embeddings")


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


# Additional tool functions can be added here using the @mcp.tool() decorator
# Example structure for future tools:
#
# @mcp.tool()
# async def analyze_package_dependencies(package_name: str) -> Dict[str, Any]:
#     """Analyze dependencies of a given OCaml package."""
#     # Implementation here
#     pass
#
# @mcp.tool()
# async def get_module_documentation(package_name: str, module_path: str) -> Dict[str, Any]:
#     """Get detailed documentation for a specific module."""
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
            if len(sys.argv) > 2 and sys.argv[2] == "opam":
                # Test opam compatibility
                packages = sys.argv[3:] if len(sys.argv) > 3 else ["base", "core"]
                print(f"Testing opam compatibility for: {packages}\n")
                result = await test_opam_compatibility(packages)
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
