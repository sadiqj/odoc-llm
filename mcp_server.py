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
    Discover OCaml packages across the entire ecosystem for specific functionality.
    
    Use this when you don't know which packages might contain what you need.
    This searches across all available packages to find the most relevant ones.
    
    Args:
        functionality: Describe what you're looking for. Be specific:
                      - "WebSocket client implementation"
                      - "Machine learning matrix operations" 
                      - "CSV file parsing and writing"
                      - "OAuth2 authentication flow"
                      - "Image processing and filtering"
    
    Returns:
        List of packages ranked by relevance, each with:
        - package: Package name
        - module: Primary module providing the functionality  
        - description: What the module does
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
                    "description": result["description"]
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
    Get a concise overview of what an OCaml package does.
    
    Provides a 3-4 sentence summary explaining the package's purpose, main features,
    and typical use cases. Helpful for understanding a package before using it.
    
    Args:
        package_name: The OCaml package name you want to learn about:
                     - 'lwt' (asynchronous programming)
                     - 'base' (alternative standard library)  
                     - 'cohttp' (HTTP client/server)
                     - 'cmdliner' (command-line interfaces)
                     - 'yojson' (JSON processing)
    
    Returns:
        Package name, version, and description explaining what it does
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
async def search_ocaml_modules(query: str, packages: List[str], top_k: int = 8) -> Dict[str, Any]:
    """
    Find OCaml modules that provide specific functionality within chosen packages.
    
    Provide a clear description of the functionality you need and specify which 
    packages to search. The tool will find relevant modules using both conceptual 
    understanding and exact keyword matching.
    
    Args:
        query: Specific functionality you're looking for. Be precise about what you need:
               - "MD5 hash function" 
               - "HTTP client for making requests"
               - "JSON parsing and serialization"
               - "list sorting operations"
               - "TCP socket server"
        packages: List of OCaml package names to search within. You must specify
                 which packages to search - common ones include:
                 ['base', 'core', 'lwt', 'async', 'cohttp', 'yojson', 'cmdliner']
        top_k: Maximum number of results to return (default: 8)
    
    Returns:
        Two lists of matching modules:
        - semantic_results: Modules with conceptually similar functionality (includes descriptions)
        - keyword_results: Modules containing your exact keywords in their documentation
        
        Each result includes the package name, module name, and module path.
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
        # Perform unified search - split top_k between the two methods
        results_per_method = top_k // 2
        results = unified_engine.unified_search(query, packages, results_per_method)
        
        # Format results for MCP response
        return {
            "query": results["query"],
            "packages_searched": results["packages_searched"],
            "semantic_results": [
                {
                    "package": r["package"],
                    "module": r["module_name"],
                    "module_path": r["module_path"],
                    "description": r["description"]
                }
                for r in results["embedding_results"]
            ],
            "keyword_results": [
                {
                    "package": r["package"],
                    "module": r["module_name"],
                    "module_path": r["module_path"]
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