"""Markdown parsing utilities for OCaml documentation using standard markdown library."""
import re
import markdown
from markdown.extensions import codehilite
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    return text.strip()

def parse_type_definition(name: str, signature: str) -> Dict[str, Any]:
    """Parse a type definition."""
    return {
        'kind': 'type',
        'name': name,
        'signature': signature,
        'anchor': f'type-{name}'
    }

def parse_value_definition(name: str, signature: str, doc: str = "") -> Dict[str, Any]:
    """Parse a value (function) definition."""
    return {
        'kind': 'value',
        'name': name,
        'signature': signature,
        'documentation': doc,
        'anchor': f'val-{name}'
    }

def extract_code_blocks_from_html(html_content: str) -> List[Dict[str, str]]:
    """Extract code blocks from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    code_blocks = []
    
    # Find all code blocks (both inline and pre-formatted)
    for pre_elem in soup.find_all('pre'):
        code_elem = pre_elem.find('code')
        if code_elem:
            code_text = code_elem.get_text().strip()
            if code_text and (code_text.startswith('val ') or code_text.startswith('type ') or code_text.startswith('module ')):
                # Get the text immediately following the code block for documentation
                next_text = ""
                next_elem = pre_elem.find_next_sibling()
                while next_elem and next_elem.name in ['p', 'div']:
                    text = next_elem.get_text().strip()
                    if text:
                        next_text = text
                        break
                    next_elem = next_elem.find_next_sibling()
                
                code_blocks.append({
                    'code': code_text,
                    'following_text': next_text
                })
    
    return code_blocks

def parse_ocaml_code_block(code_text: str, following_text: str = "") -> Optional[Dict[str, Any]]:
    """Parse a single OCaml code block."""
    code_text = code_text.strip()
    
    # Parse value definitions
    val_match = re.match(r'val\s+(\w+)\s*:\s*(.+)', code_text, re.DOTALL)
    if val_match:
        name = val_match.group(1)
        sig = val_match.group(2).strip()
        return parse_value_definition(name, f"val {name} : {sig}", following_text)
    
    # Parse type definitions
    type_match = re.match(r'type\s+(\w+)', code_text)
    if type_match:
        name = type_match.group(1)
        return parse_type_definition(name, code_text)
    
    # Parse module type definitions
    module_type_match = re.match(r'module\s+type\s+(\w+)\s*=\s*sig\s+\.\.\.\s+end', code_text)
    if module_type_match:
        name = module_type_match.group(1)
        return {
            'name': name,
            'kind': 'module-type'
        }
    
    # Parse module definitions - handle various patterns
    # Pattern 1: module Name : sig ... end
    module_match1 = re.match(r'module\s+(\w+)\s*:\s*sig\s+\.\.\.\s+end', code_text)
    if module_match1:
        name = module_match1.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    # Pattern 2: module Name (Arg : Type) : ReturnType with constraints
    module_match2 = re.match(r'module\s+(\w+)\s*\([^)]+\)\s*:\s*\w+', code_text)
    if module_match2:
        name = module_match2.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    # Pattern 3: module Name : Type  (simple module signature)
    module_match3 = re.match(r'module\s+(\w+)\s*:\s*\w+', code_text)
    if module_match3:
        name = module_match3.group(1)
        return {
            'name': name,
            'kind': 'module'
        }
    
    return None

def parse_module_markdown(content: str) -> Dict[str, Any]:
    """Parse a markdown file containing OCaml module documentation."""
    result = {
        'elements': [],  # New ordered list of all elements
        'types': [],     # Keep for backward compatibility 
        'values': [],    # Keep for backward compatibility
        'modules': [],   # Keep for backward compatibility
        'module_documentation': "",
        'sections': []   # Keep for backward compatibility
    }
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['codehilite', 'fenced_code'])
    html_content = md.convert(content)
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract module name from the first h1
    h1 = soup.find('h1')
    module_name = None
    if h1:
        h1_text = h1.get_text()
        match = re.search(r'Module `([^`]+)`', h1_text)
        if match:
            module_name = match.group(1)
    
    # Extract main module documentation (text before the first h2)
    main_doc_parts = []
    for elem in soup.find_all(['p', 'h2']):
        if elem.name == 'h2':
            break
        if elem.name == 'p':
            main_doc_parts.append(elem.get_text().strip())
    
    if main_doc_parts:
        result['module_documentation'] = '\n'.join(main_doc_parts)
    
    # Process all elements in order
    all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre'])
    
    for elem in all_elements:
        if elem.name.startswith('h'):
            # This is a header
            level = int(elem.name[1])  # h1 -> 1, h2 -> 2, etc.
            title = elem.get_text().strip()
            
            # Skip the main module title
            if level == 1 and 'Module' in title:
                continue
                
            section = {
                'kind': 'section',
                'title': title,
                'level': level,
                'content': ""
            }
            
            # Collect content until the next header or code block
            content_parts = []
            next_elem = elem.find_next_sibling()
            while next_elem and not (next_elem.name and (next_elem.name.startswith('h') or next_elem.name == 'pre')):
                if next_elem.name == 'p':
                    content_parts.append(next_elem.get_text().strip())
                next_elem = next_elem.find_next_sibling()
            
            section['content'] = '\n'.join(content_parts)
            result['elements'].append(section)
            result['sections'].append(section)  # Keep for backward compatibility
            
        elif elem.name == 'pre':
            # This is a code block
            code_elem = elem.find('code')
            if code_elem:
                code_text = code_elem.get_text().strip()
                if code_text and (code_text.startswith('val ') or code_text.startswith('type ') or code_text.startswith('module ')):
                    # Get the text immediately following the code block for documentation
                    next_text = ""
                    next_elem = elem.find_next_sibling()
                    while next_elem and next_elem.name in ['p', 'div']:
                        text = next_elem.get_text().strip()
                        if text:
                            next_text = text
                            break
                        next_elem = next_elem.find_next_sibling()
                    
                    parsed = parse_ocaml_code_block(code_text, next_text)
                    if parsed:
                        result['elements'].append(parsed)
                        
                        # Keep for backward compatibility
                        if parsed.get('kind') == 'value':
                            result['values'].append(parsed)
                        elif parsed.get('kind') == 'type':
                            result['types'].append(parsed)
                        elif parsed.get('kind') == 'module':
                            result['modules'].append(parsed)
    
    return result

def parse_markdown_documentation(file_path: str) -> Dict[str, Any]:
    """Parse a markdown documentation file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if this is a module documentation file
    if 'Module `' in content or '# Module' in content:
        return parse_module_markdown(content)
    
    # Otherwise, it's a general documentation file (README, CHANGES, etc.)
    return {
        'content': content,
        'type': 'documentation'
    }

def extract_module_path(file_path: str) -> List[str]:
    """Extract module path from file path."""
    # Example: docs-md/package/version/doc/module/Module-Submodule.md
    # Should return ['Module', 'Submodule']
    # Special case: Module-Submodule-module-type-TypeName.md
    # Should return ['Module', 'Submodule', 'TypeName']
    
    parts = file_path.split('/')
    
    # Find the 'doc' directory
    try:
        doc_index = parts.index('doc')
    except ValueError:
        return []
    
    # Get the file name without extension
    if parts[-1].endswith('.md'):
        filename = parts[-1][:-3]  # Remove .md
    else:
        return []
    
    # If it's index.md, use the parent directory name
    if filename == 'index':
        if len(parts) > doc_index + 1:
            return [parts[-2]]
        return []
    
    # Check for module-type pattern and handle specially
    if '-module-type-' in filename:
        # Split on '-module-type-' to separate the module path from the type name
        module_part, type_name = filename.split('-module-type-', 1)
        # Split the module part on '-' for nested modules
        module_parts = module_part.split('-')
        # Add the type name at the end
        module_parts.append(type_name)
        return module_parts
    
    # Split on '-' for nested modules
    module_parts = filename.split('-')
    
    return module_parts