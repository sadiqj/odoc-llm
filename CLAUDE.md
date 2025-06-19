# OCaml Documentation Dataset Analysis

 When responding keep the language accessible. It is offensive to use too complicated of language because it isn't inclusive of less educated individuals. Favour the simplest solution you can. All code in this project is AI-generated so feel free to be critical. You should avoid being overly agreeable and flattering in your responses.

## Overview

This repository contains tools for extracting and analyzing OCaml package documentation from odoc-generated HTML/JSON files. The project provides a complete pipeline for processing OCaml documentation:

1. **Documentation Extraction**: Parses HTML/JSON documentation into structured, machine-readable format
2. **Semantic Description Generation**: Uses LLMs to generate natural language descriptions of modules
3. **Embedding Generation**: Creates high-dimensional vector embeddings from module descriptions
4. **Semantic Search**: Enables natural language queries to find relevant OCaml modules

## Development Environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. To set up the development environment:

```bash
# Install dependencies
uv sync

# Run extraction tools
uv run python extract_docs.py --help
uv run python generate_descriptions.py --help
uv run python generate_embeddings.py --help
uv run python semantic_search.py --help
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

Creates `package_embeddings/` directory with:
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
2. **Embedding Loading**: Loads all module embeddings from `package_embeddings/` into memory
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
3. **Tool Exposure**: Provides `find_ocaml_packages` tool for semantic search using decorators
4. **Lazy Loading**: Initializes search engine only when first query is made
5. **Type-Safe API**: Uses Python type hints and docstrings for automatic tool schema generation
6. **Extensible Design**: Easy to add new tools using `@mcp.tool()` decorator

### Quick Start

```bash
# Run the FastMCP server with HTTP SSE transport
uv run mcp-server

# Or run directly
uv run python mcp_server.py

# Test the functionality directly (without MCP protocol)
uv run mcp-server --test "HTTP server"
uv run mcp-server --test "JSON parsing"
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

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "ocaml-search": {
      "command": "uv",
      "args": ["run", "mcp-server"],
      "cwd": "/path/to/odoc-llm"
    }
  }
}
```

### Key Features

- **HTTP SSE Transport**: Server-Sent Events over HTTP instead of stdio
- **Semantic Search**: Full access to the 137,000+ module embeddings
- **FastMCP Framework**: Simplified development with automatic schema generation
- **Type Safety**: Automatic validation using Python type hints
- **Low Latency**: Reuses existing search infrastructure
- **Error Handling**: Graceful degradation with informative error messages
- **Extensible**: Easy to add new tools using decorators
- **Standard Protocol**: Compatible with all MCP clients


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

### Generated Data Directories

```
├── parsed-docs/            # Extracted documentation JSON files
├── module-descriptions/    # Generated module descriptions
├── package-descriptions/   # Generated package descriptions
└── package_embeddings/     # Module embeddings and metadata
```

## Data Directory Structure

```
docs/
├── package-name/           # 4,695 different packages
│   ├── v1.0.0/            # Multiple versions per package
│   │   ├── doc/           # Documentation directory
│   │   │   ├── Module/    # Module directories
│   │   │   │   ├── Submodule/
│   │   │   │   │   └── index.html.json  # Module documentation
│   │   │   └── index.html.json         # Package root documentation
│   │   ├── package.json                # Package metadata
│   │   ├── status.json                 # Build status information
│   │   ├── README.md.html.json         # README documentation
│   │   ├── CHANGES.md.html.json        # Changelog documentation
│   │   └── LICENSE.*.html.json         # License files
│   └── v2.0.0/            # Additional versions
└── all-docs.tar.gz        # Compressed archive of all documentation
```

## JSON File Structure

### Core Documentation Files (`index.html.json`)

Each documentation file follows a consistent JSON schema:

```json
{
  "type": "documentation",
  "uses_katex": false,
  "breadcrumbs": [
    {
      "name": "package-name",
      "href": "../../../../index.html",
      "kind": "page"
    },
    {
      "name": "v1.0.0",
      "href": "../../../index.html", 
      "kind": "page"
    },
    {
      "name": "Module",
      "href": "../index.html",
      "kind": "module"
    }
  ],
  "toc": [
    {
      "title": "Types",
      "href": "#types",
      "children": []
    },
    {
      "title": "Values", 
      "href": "#values",
      "children": []
    }
  ],
  "source_anchor": null,
  "preamble": "<p>Module description and overview</p>",
  "content": "<div class=\"odoc-spec\">...extensive HTML content...</div>"
}
```

### Package Metadata Files (`package.json`)

Contains structural information about the package:

```json
{
  "libraries": [
    {
      "name": "library-name",
      "modules": [
        {
          "name": "ModuleName",
          "submodules": [],
          "kind": "module"
        }
      ],
      "dependencies": []
    }
  ]
}
```

### Build Status Files (`status.json`)

Indicates build success/failure and additional documentation:

```json
{
  "failed": false,
  "otherdocs": {
    "readme": ["linked/p/package/version/doc/README.md"],
    "license": ["linked/p/package/version/doc/LICENSE.md"],
    "changes": ["linked/p/package/version/doc/CHANGES.md"],
    "others": ["linked/p/package/version/package.json"]
  }
}
```

## Data Categories and Content Types

### 1. API Documentation
The primary content consists of rich API documentation with:

#### Function Signatures
```html
<div class="spec value anchored" id="val-map">
  <code>
    <span class="keyword">val</span> map : 
    <span>('a -> 'b) -> 'a list -> 'b list</span>
  </code>
</div>
```

#### Type Definitions
```html
<div class="spec type anchored" id="type-t">
  <code>
    <span class="keyword">type</span> 'a t = 
    | Empty 
    | Node of 'a * 'a t * 'a t
  </code>
</div>
```

#### Module Interfaces
```html
<div class="spec module anchored" id="module-Set">
  <code>
    <span class="keyword">module</span> Set : 
    <span class="keyword">sig</span> ... <span class="keyword">end</span>
  </code>
</div>
```

### 2. Documentation Content
Rich semantic documentation including:

- **Function descriptions**: Detailed behavioral specifications
- **Parameter documentation**: Usage and type information
- **Exception documentation**: Error conditions and handling
- **Code examples**: OCaml code snippets demonstrating usage
- **Deprecation notices**: Version compatibility information

### 3. Structural Information
- **Module hierarchies**: Complete package organization
- **Cross-references**: Links between modules and types  
- **Version evolution**: API changes across package versions
- **Dependency graphs**: Inter-package relationships

### 4. Metadata
- **Build status**: Success/failure indicators
- **Version information**: Semantic versioning data
- **Documentation completeness**: Coverage metrics
- **Package categorization**: Domain-specific groupings

## Key OCaml Packages Represented

The dataset includes major OCaml ecosystem packages:

### Core Libraries
- **base**: Jane Street's standard library replacement
- **core**: Comprehensive standard library extensions
- **lwt**: Cooperative threading library
- **async**: Asynchronous programming framework

### Web and Networking
- **cohttp**: HTTP client/server library
- **dream**: Modern web framework
- **conduit**: Network connection abstraction

### Data Processing
- **yojson**: JSON parsing and manipulation
- **csv**: CSV file processing
- **angstrom**: Parser combinators

### System Programming
- **lwt-unix**: Unix system call bindings
- **mirage**: Unikernel operating system library
- **eio**: Effects-based parallel I/O

### Build and Development Tools
- **dune**: Build system
- **merlin**: IDE integration
- **odoc**: Documentation generator

## Data Quality Assessment

### High-Quality Packages (Excellent Documentation)
- Comprehensive function documentation with examples
- Complete type information with explanations
- Rich cross-references and module organization
- Multiple versions showing API evolution

Examples: `base`, `lwt`, `async`, `core`, `dune`, `cohttp`

### Medium-Quality Packages (Good Documentation)
- Complete API signatures with basic descriptions
- Type information present but minimal examples
- Standard module organization

Examples: Many specialty libraries and domain-specific packages

### Failed Builds
- Packages with `"failed": true` in status.json
- Limited to metadata and error logs
- Still provide package structure information

### Documentation Completeness Metrics
- **Function coverage**: ~85% of functions have descriptions
- **Type coverage**: ~95% of types have definitions
- **Example coverage**: ~40% of functions have usage examples
- **Cross-reference density**: Extensive linking between modules

## Data Extraction Patterns

### Parsing HTML Content

The `content` field contains structured HTML that can be parsed to extract:

```python
import json
import re
from bs4 import BeautifulSoup

def extract_function_signatures(json_file):
    with open(json_file) as f:
        doc = json.load(f)
    
    soup = BeautifulSoup(doc['content'], 'html.parser')
    functions = []
    
    for spec in soup.find_all('div', class_='spec value anchored'):
        code = spec.find('code')
        if code:
            functions.append(code.get_text())
    
    return functions

def extract_type_definitions(json_file):
    with open(json_file) as f:
        doc = json.load(f)
    
    soup = BeautifulSoup(doc['content'], 'html.parser') 
    types = []
    
    for spec in soup.find_all('div', class_='spec type anchored'):
        code = spec.find('code')
        if code:
            types.append(code.get_text())
    
    return types
```

### Extracting Documentation Text

```python
def extract_function_docs(json_file):
    with open(json_file) as f:
        doc = json.load(f)
    
    soup = BeautifulSoup(doc['content'], 'html.parser')
    docs = []
    
    for spec in soup.find_all('div', class_='spec value anchored'):
        doc_div = spec.find_next_sibling('div', class_='spec-doc')
        if doc_div:
            docs.append({
                'signature': spec.find('code').get_text(),
                'documentation': doc_div.get_text()
            })
    
    return docs
```

### Building Module Hierarchies

```python
def build_module_tree(package_path):
    structure = {}
    
    for root, dirs, files in os.walk(package_path):
        if 'index.html.json' in files:
            with open(os.path.join(root, 'index.html.json')) as f:
                doc = json.load(f)
            
            # Extract module path from breadcrumbs
            path = [crumb['name'] for crumb in doc['breadcrumbs']]
            
            # Build nested dictionary structure
            current = structure
            for component in path:
                if component not in current:
                    current[component] = {}
                current = current[component]
    
    return structure
```

## Applications for LLM Training

### 1. Code Completion and Generation
- **Function signatures**: Complete type information for accurate suggestions
- **Usage patterns**: Real-world examples of API usage
- **Type inference**: Rich type system examples for training type checkers

### 2. Documentation Generation
- **Template learning**: Consistent documentation patterns across packages
- **Example generation**: Code snippet patterns for different function types
- **API description**: Natural language explanations of technical concepts

### 3. Code Understanding and Analysis
- **Semantic parsing**: Understanding OCaml syntax and semantics
- **Module organization**: Software architecture patterns
- **Dependency analysis**: Inter-module relationships

### 4. Educational Content Creation
- **Tutorial generation**: Progressive learning sequences
- **Concept explanation**: Functional programming paradigms
- **Best practices**: Idiomatic OCaml patterns

### 5. Language Model Fine-tuning
- **Domain-specific knowledge**: Functional programming concepts
- **OCaml syntax**: Language-specific patterns and idioms
- **Software engineering**: Module design and API development

## Data Processing Recommendations

### For Training Data Preparation

1. **Content Extraction**
   - Parse HTML to extract clean text
   - Preserve code formatting and syntax highlighting
   - Maintain structural relationships between modules

2. **Quality Filtering**
   - Exclude failed builds (`"failed": true`)
   - Prioritize packages with rich documentation
   - Filter by documentation completeness metrics

3. **Augmentation Strategies**
   - Combine related functions into learning sequences
   - Generate negative examples from type mismatches
   - Create multi-modal examples with code and documentation

4. **Tokenization Considerations**
   - Preserve OCaml syntax tokens
   - Handle special characters in identifiers
   - Maintain HTML structure markers for formatting

### For Analysis and Research

1. **Longitudinal Studies**
   - Track API evolution across versions
   - Analyze deprecation patterns
   - Study breaking change communication

2. **Ecosystem Analysis**
   - Map dependency relationships
   - Identify common patterns and anti-patterns
   - Analyze documentation quality factors

3. **Comparative Studies**
   - Compare with other language ecosystems
   - Analyze documentation style variations
   - Study community contribution patterns

## Technical Specifications

### File Format Details
- **Encoding**: UTF-8
- **JSON Schema**: Consistent across all files
- **HTML Content**: Well-formed, parseable HTML5
- **Link Structure**: Relative paths maintaining relationships

### Performance Considerations
- **Total Size**: Multiple gigabytes of text data
- **File Count**: 2.2M+ individual files
- **Memory Usage**: Optimized for large-scale processing with garbage collection
- **Parallel Processing**: Files can be processed independently with up to 12 workers
- **Reliability**: Comprehensive error handling prevents silent crashes during parallel processing

### Integration Patterns
- **Batch Processing**: Process packages independently
- **Incremental Updates**: Use timestamps and version information
- **Cross-Package Analysis**: Leverage dependency information

## Conclusion

This OCaml documentation dataset represents an exceptional resource for:

- **Programming language research**: Comprehensive functional programming documentation
- **Machine learning applications**: High-quality training data for code-related tasks
- **Software engineering studies**: Real-world API design patterns
- **Educational tool development**: Rich content for learning resources

The structured, machine-readable format combined with the breadth and depth of coverage makes this dataset particularly valuable for advancing research in programming language understanding, automated documentation generation, and code intelligence systems.

The consistent schema and comprehensive coverage across the OCaml ecosystem provide a unique opportunity to study software documentation at scale, while the rich semantic content enables sophisticated natural language processing applications in the programming domain.
