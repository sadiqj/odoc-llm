#!/usr/bin/env python3
"""
FastMCP Server for OCaml Documentation Search

This server exposes tools for searching and interacting with the OCaml
documentation dataset through the Model Context Protocol (MCP) using HTTP SSE.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
embeddings_dir = Path("package-embeddings")
package_descriptions_dir = Path("package-descriptions")

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
            if len(sys.argv) > 2:
                if sys.argv[2].startswith("--summary="):
                    # Test package summary
                    package_name = sys.argv[2].split("=", 1)[1]
                    print(f"Testing package summary: {package_name}\n")
                    result = await get_package_summary(package_name)
                    print(json.dumps(result, indent=2))
                else:
                    # Test semantic search
                    query = sys.argv[2]
                    print(f"Testing semantic search: {query}\n")
                    result = await find_ocaml_packages(query)
                    print(json.dumps(result, indent=2))
            else:
                # Default test
                print("Testing semantic search: HTTP server\n")
                result = await find_ocaml_packages("HTTP server")
                print(json.dumps(result, indent=2))
        
        asyncio.run(test())
    else:
        # Run FastMCP server with HTTP SSE transport
        # Default SSE endpoint will be available at /sse
        # Using default port (probably 8000), may conflict with embedding server
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()