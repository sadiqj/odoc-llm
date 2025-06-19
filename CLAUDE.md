# OCaml Documentation Dataset Analysis

 When responding keep the language accessible. It is offensive to use too complicated of language because it isn't inclusive of less educated individuals. Favour the simplest solution you can. All code in this project is AI-generated so feel free to be critical. You should avoid being overly agreeable and flattering in your responses.

## Overview

This repository contains tools for extracting and analyzing OCaml package documentation from odoc-generated HTML/JSON files. The project provides a complete pipeline for processing OCaml documentation:

1. **Documentation Extraction**: Parses HTML/JSON documentation into structured, machine-readable format
2. **Module Description Generation**: Uses LLMs to generate natural language descriptions of individual modules
3. **Package Description Generation**: Uses LLMs to generate concise summaries of entire packages
4. **Embedding Generation**: Creates high-dimensional vector embeddings from module descriptions
5. **Semantic Search**: Enables natural language queries to find relevant OCaml modules
6. **MCP Server**: Provides access to search and package information through Model Context Protocol

## Development Environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. To set up the development environment:

```bash
# Install dependencies
uv sync

# Run extraction tools
uv run python extract_docs.py --help
uv run python generate_module_descriptions.py --help
uv run python generate_package_descriptions.py --help
uv run python generate_embeddings.py --help
uv run python semantic_search.py --help

# Run MCP server
uv run python mcp_server.py
```

## Tool 1: Documentation Extraction (`extract_docs.py`)

Extracts OCaml documentation from odoc-generated files into structured JSON format.

### How it Works

1. **Package Discovery**: Scans the `docs/` directory for package folders
2. **Version Selection**: Uses semantic versioning to find the latest version of each package
3. **Build Status Check**: Skips packages that failed to build (checks `status.json`)
4. **File Collection**: Recursively finds all `index.html.json` documentation files
5. **Content Parsing**: Uses BeautifulSoup to extract:
   - Function signatures and documentation
   - Type definitions
   - Module hierarchies
   - Code examples
   - Cross-references
6. **Output Generation**: Creates one JSON file per package with all extracted data

### Quick Start

```bash
# Extract a specific package
uv run python extract_docs.py --package base --output-dir parsed-docs

# Extract first 10 packages for testing
uv run python extract_docs.py --limit 10

# Extract all packages with 8 parallel workers
uv run python extract_docs.py --parallel 8
```

### Output Structure

Each package generates a JSON file in `parsed-docs/` with:
- Package metadata (name, version, dependencies)
- All modules with their types, functions, and documentation
- Statistics (total modules, types, values)
- Documentation sections and code examples

## Tool 2: Module Description Generator (`generate_module_descriptions.py`)

Generates semantic descriptions for OCaml modules using an LLM.

### How it Works

1. **Module Extraction**: Reads parsed JSON files from `extract_docs.py` output
2. **Hierarchical Processing**: Processes leaf modules first, then merges up
3. **Smart Chunking**: For large modules (>20 functions), uses chunking strategy:
   - Summarizes each chunk of 20 functions/types
   - Combines chunk summaries into final description
4. **Description Merging**: Parent modules get descriptions synthesized from children
5. **Output**: Creates JSON files with module paths mapped to descriptions

### Quick Start

```bash
# Generate descriptions for specific package
uv run python generate_module_descriptions.py --package base

# Generate for first 10 packages
uv run python generate_module_descriptions.py --limit 10

# Use custom LLM endpoint and parallel workers
uv run python generate_module_descriptions.py --llm-url http://localhost:8000 --model Qwen/Qwen3-30B-A3B-FP8 --workers 12

# Debug prompts/responses
uv run python generate_module_descriptions.py --package base --log-prompts

# Test syntax
uv run python -m py_compile generate_module_descriptions.py
```

### Configuration

- **LLM Endpoint**: Default expects local OpenAI-compatible server at `http://localhost:8000`
- **Model**: Default uses `Qwen/Qwen3-30B-A3B-FP8`
- **Processing**: Excludes Tezos packages and skips already-processed files
- **Parallel Processing**: Supports up to 12 workers for faster processing
- **Signal Handling**: Graceful shutdown support for robust operation

## Tool 3: Package Description Generator (`generate_package_descriptions.py`)

Generates concise package descriptions using README content and LLM.

### How it Works

1. **Package Information Extraction**: Reads parsed JSON files and extracts README content
2. **Content Cleaning**: Removes HTML tags and normalizes whitespace from documentation
3. **LLM Description Generation**: Uses README content to generate 3-4 sentence package summaries
4. **Simple Output**: Creates JSON files with package name, version, and description
5. **Parallel Processing**: Supports multiple workers for efficient batch processing

### Quick Start

```bash
# Generate description for specific package
uv run python generate_package_descriptions.py --package lwt

# Generate for first 10 packages
uv run python generate_package_descriptions.py --limit 10

# Use custom LLM endpoint and parallel workers
uv run python generate_package_descriptions.py --llm-url http://localhost:8000 --model Qwen/Qwen3-30B-A3B-FP8 --workers 8

# Debug prompts/responses
uv run python generate_package_descriptions.py --package base --log-prompts
```

### Configuration

- **LLM Endpoint**: Default expects local OpenAI-compatible server at `http://localhost:8000`
- **Model**: Default uses `Qwen/Qwen3-30B-A3B-FP8`
- **Processing**: Skips packages that already have descriptions
- **Parallel Processing**: Supports up to 8 workers for faster processing
- **Signal Handling**: Graceful shutdown support for robust operation

### Output Structure

Creates `package-descriptions/` directory with:
- `{package_name}.json`: Package name, version, and 3-4 sentence description

Example output:
```json
{
  "package": "lwt",
  "version": "5.9.1", 
  "description": "Lwt is a concurrent programming library for OCaml that simplifies asynchronous and parallel execution through a promise-based model. It enables efficient handling of I/O operations and background tasks without requiring manual thread management or synchronization. Key features include non-blocking I/O, cooperative multitasking, and support for reactive programming, making it suitable for building scalable networked applications and event-driven systems."
}
```

## Supporting Modules

### HTML Parser (`parse_html.py`)

Utility module for parsing OCaml documentation HTML.

**Key Functions:**
- `parse_json_documentation()`: Main entry point for parsing JSON files
- `parse_module_content()`: Extracts structured data from HTML content
- `parse_type_definition()`: Parses type specifications
- `parse_value_definition()`: Parses function signatures and docs
- `extract_code_examples()`: Finds code snippets in documentation

### Version Utils (`version_utils.py`)

Handles OCaml-specific version string normalization.

**Features:**
- Converts OCaml versions to semantic versioning format
- Handles special cases like `v0.17.1`, `4.2.1-1`
- Finds latest version from a list of version strings

## Tool 4: Embedding Generator (`generate_embeddings.py`)

Generates high-dimensional embeddings for OCaml module descriptions using a local embedding model server.

### How it Works

1. **Module Loading**: Reads module descriptions from `module-descriptions/` JSON files
2. **Intelligent Filtering**: Automatically filters out empty/placeholder modules using pattern detection:
   - Removes modules with "Empty module with no functions, types, or documentation"
   - Filters descriptions containing "contains no", "provides no", "serves as a placeholder"
   - Excludes descriptions shorter than 100 characters
3. **Embedding Generation**: Sends filtered descriptions to local embedding server (http://localhost:8000/embedding) using Qwen/Qwen3-Embedding-0.6B model
4. **Parallel Processing**: Uses configurable worker threads for concurrent package processing
5. **Progress Tracking**: Maintains checkpoint file for resumable processing
6. **Output Generation**: Creates compressed NPZ arrays and detailed metadata for each package

### Quick Start

```bash
# Generate embeddings for all packages
uv run python generate_embeddings.py

# Process specific packages
uv run python generate_embeddings.py --packages base,core,lwt

# Resume interrupted processing
uv run python generate_embeddings.py --resume

# Custom configuration
uv run python generate_embeddings.py --workers 8 --batch-size 32 --rate-limit 15.0
```

### Output Structure

Creates `package-embeddings/` directory with:
- `packages/{package_name}/embeddings.npz`: Compressed numpy array of embeddings (float32)
- `packages/{package_name}/metadata.json`: Module information with filtering statistics
- `checkpoint.json`: Progress tracking for resume capability
- `metadata.json`: Global statistics across all packages

### Key Features

- **Intelligent Filtering**: ~87% retention rate, focusing on meaningful content
- **High Performance**: Processes ~3,900 packages in ~2 hours with 12 workers
- **Fault Tolerant**: Automatic retry, progress checkpointing, graceful error handling
- **Storage Efficient**: ~600MB for 137,000+ embeddings using NPZ compression
- **Quality Assurance**: Validates embeddings, checks for NaN/inf values, ensures normalization

## Tool 5: Semantic Search (`semantic_search.py`)

Enables natural language search for OCaml modules using query embeddings and cosine similarity.

### How it Works

1. **Query Embedding**: Uses Qwen3-Embedding-0.6B model to embed user queries
2. **Embedding Loading**: Loads all module embeddings from `package-embeddings/` into memory
3. **Similarity Calculation**: Computes cosine similarity between query and all module embeddings
4. **Result Ranking**: Returns top-K most similar modules with descriptions
5. **Output Formatting**: Supports both text and JSON output formats

### Quick Start

```bash
# Search for HTTP server modules
uv run python semantic_search.py "http server"

# Find JSON parsing modules (top 10 results)
uv run python semantic_search.py "JSON parser" --top-k 10

# Search with JSON output format
uv run python semantic_search.py "cryptographic hash" --format json

# Verbose mode for debugging
uv run python semantic_search.py "async IO" --verbose
```

### Key Features

- **Fast Search**: Sub-second query response time (~200ms)
- **Natural Language**: Understands semantic meaning, not just keywords
- **GPU Support**: Automatically uses CUDA if available
- **Comprehensive Results**: Returns package name, module path, and description
- **Scalable**: Efficiently searches across 137,000+ module embeddings

## Tool 6: MCP Server (`mcp_server.py`)

Exposes OCaml module search functionality through the Model Context Protocol (MCP) using FastMCP with HTTP SSE (Server-Sent Events) transport, allowing integration with Claude Desktop and other MCP-compatible clients.

### How it Works

1. **FastMCP Framework**: Uses FastMCP for simplified server implementation with automatic schema generation
2. **HTTP SSE Transport**: Provides HTTP-based Server-Sent Events endpoint instead of stdio
3. **Tool Exposure**: Provides tools for semantic search, package summaries, and opam compatibility testing using decorators
4. **Lazy Loading**: Initializes search engine only when first query is made
5. **Type-Safe API**: Uses Python type hints and docstrings for automatic tool schema generation
6. **Extensible Design**: Easy to add new tools using `@mcp.tool()` decorator

### Quick Start

```bash
# Run the FastMCP server with HTTP SSE transport
uv run python mcp_server.py

# Test the functionality directly (without MCP protocol)
uv run python mcp_server.py --test "HTTP server"
uv run python mcp_server.py --test --summary=lwt
uv run python mcp_server.py --test opam base core
```

### HTTP SSE Endpoint

The server runs on HTTP with Server-Sent Events transport:
- **Default URL**: `http://localhost:8000/sse` (SSE endpoint)
- **Protocol**: MCP over HTTP SSE
- **Content-Type**: `text/event-stream`

**Note**: The MCP server uses the same port (8000) as the LLM/embedding server. If you're running the LLM server for descriptions/embeddings, you'll need to stop it before starting the MCP server, or configure one to use a different port.

### Available Tools

#### find_ocaml_packages
- **Description**: Find OCaml packages providing specific functionality
- **Input**: `functionality` (string) - Natural language description
- **Output**: Top 5 matching packages with modules and similarity scores

#### get_package_summary
- **Description**: Get a concise summary of an OCaml package
- **Input**: `package_name` (string) - Name of the OCaml package
- **Output**: Package name, version, and 3-4 sentence description

#### test_opam_compatibility
- **Description**: Test opam package compatibility for a list of packages
- **Input**: `packages` (list of strings) - List of opam package names to test
- **Output**: Compatible package versions for the current opam switch
- **Note**: Requires the current opam switch for the project to be active

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "ocaml-search": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/odoc-llm"
    }
  }
}
```

### Key Features

- **HTTP SSE Transport**: Server-Sent Events over HTTP instead of stdio
- **Triple Capabilities**: Semantic search, package summaries, and opam compatibility testing
- **Package Summaries**: Instant access to concise 3-4 sentence package descriptions
- **Opam Integration**: Test package compatibility with current opam switch
- **FastMCP Framework**: Simplified development with automatic schema generation
- **Type Safety**: Automatic validation using Python type hints
- **Low Latency**: Reuses existing search infrastructure
- **Error Handling**: Graceful degradation with informative error messages
- **Extensible**: Easy to add new tools using decorators
- **Standard Protocol**: Compatible with all MCP clients

## Repository Updates and Progress

### Recent Project Developments
- **Package Description Generation**: Added tool for generating concise package summaries from README content
- **MCP Server Integration**: Implemented FastMCP server with semantic search, package summary, and opam compatibility tools
- **Opam Compatibility Testing**: Added tool to test package compatibility with current opam switch
- **Directory Structure Standardization**: Unified naming conventions using hyphens throughout
- **Code Cleanup**: Removed over-engineered infrastructure (progress_tracker.py, error_handling.py)
- **Tool Separation**: Split module and package description generation into focused, specialized tools

### Ongoing Research and Improvements
- Continuous refinement of LLM-based description generation
- Exploring advanced embedding techniques
- Expanding package coverage and documentation quality assessment
- Developing more sophisticated semantic search algorithms

### Challenges and Future Work
- Handling documentation variations across different packages
- Improving description generation for complex or minimal documentation
- Scaling the infrastructure for even larger package ecosystems
- Developing more advanced natural language understanding for module search

## Repository Statistics

### Documentation Extraction
- **Total packages**: 4,695 OCaml packages (original dataset)
- **Extracted packages**: 4,164 packages (89% completion)
- **Total documentation files**: 2,199,957 HTML JSON files
- **Extracted structured data**: 1.3GB of parsed JSON
- **Functions extracted**: 2,693,805 functions with signatures
- **Types extracted**: 353,539 type definitions
- **Modules extracted**: 229,730 modules
- **Coverage**: Spans multiple versions per package, showing API evolution over time

### Module Descriptions
- **Packages with descriptions**: 3,938 packages
- **Total modules described**: 158,229 modules
- **Description generation**: Using LLM with hierarchical summarization

### Package Descriptions
- **Packages with descriptions**: Growing dataset of concise package summaries
- **Description generation**: Using README content and LLM processing
- **Output format**: 3-4 sentence summaries focusing on purpose and key capabilities

### Embeddings Dataset
- **Packages embedded**: 3,934 packages (99.9% success rate)
- **Meaningful modules**: 137,782 modules (87.1% retention after filtering)
- **Empty modules filtered**: 20,447 modules (12.9%)
- **Embedding dimension**: 1,024D vectors
- **Total storage**: 600MB compressed (NPZ format)
- **Average modules per package**: 35.0

## Project Structure

### Core Files

```
├── extract_docs.py               # Documentation extraction from HTML/JSON
├── generate_module_descriptions.py  # LLM-based module description generation
├── generate_package_descriptions.py # LLM-based package description generation
├── generate_embeddings.py        # Vector embedding generation
├── semantic_search.py            # Natural language module search
├── mcp_server.py                  # MCP server for tool access
├── parse_html.py                  # HTML parsing utilities
├── version_utils.py               # Version handling utilities
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Dependency lock file
└── CLAUDE.md               # This documentation
```

## Conclusion

This OCaml documentation dataset and toolset provides:

- **Comprehensive Coverage**: 4,695+ OCaml packages with detailed extraction and analysis
- **Multi-Level Descriptions**: Both detailed module descriptions and concise package summaries
- **Semantic Search**: Natural language queries across 137,000+ module embeddings
- **Modern Integration**: MCP server for seamless Claude Desktop integration
- **Clean Architecture**: Focused, single-purpose tools with consistent naming and structure

The project demonstrates effective use of LLMs for documentation analysis while maintaining practical, efficient tooling that scales to large codebases.
