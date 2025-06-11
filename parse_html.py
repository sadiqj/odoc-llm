"""HTML parsing utilities for OCaml documentation."""
import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Decode HTML entities
    text = text.replace('&#45;', '-').replace('&gt;', '>').replace('&lt;', '<')
    return text

def extract_signature(code_elem) -> str:
    """Extract clean signature from code element."""
    if not code_elem:
        return ""
    # Get text and preserve some formatting
    text = code_elem.get_text(separator=' ')
    return clean_text(text)

def parse_type_definition(spec_div) -> Optional[Dict[str, Any]]:
    """Parse a type definition from a spec div."""
    anchor = spec_div.get('id', '')
    if not anchor.startswith('type-'):
        return None
    
    type_name = anchor[5:]  # Remove 'type-' prefix
    code = spec_div.find('code')
    signature = extract_signature(code) if code else ""
    
    return {
        'kind': 'type',
        'name': type_name,
        'signature': signature,
        'anchor': anchor
    }

def parse_value_definition(spec_div, doc_div=None) -> Optional[Dict[str, Any]]:
    """Parse a value (function) definition from a spec div."""
    anchor = spec_div.get('id', '')
    if not anchor.startswith('val-'):
        return None
    
    val_name = anchor[4:]  # Remove 'val-' prefix
    code = spec_div.find('code')
    signature = extract_signature(code) if code else ""
    
    # Extract documentation if available
    documentation = ""
    if doc_div:
        documentation = clean_text(doc_div.get_text())
    
    return {
        'kind': 'value',
        'name': val_name,
        'signature': signature,
        'documentation': documentation,
        'anchor': anchor
    }

def parse_module_definition(spec_div) -> Optional[Dict[str, Any]]:
    """Parse a module definition from a spec div."""
    anchor = spec_div.get('id', '')
    if not anchor.startswith('module-'):
        return None
    
    module_name = anchor[7:]  # Remove 'module-' prefix
    code = spec_div.find('code')
    signature = extract_signature(code) if code else ""
    
    return {
        'kind': 'module',
        'name': module_name,
        'signature': signature,
        'anchor': anchor
    }

def parse_exception_definition(spec_div) -> Optional[Dict[str, Any]]:
    """Parse an exception definition from a spec div."""
    anchor = spec_div.get('id', '')
    if not anchor.startswith('exception-'):
        return None
    
    exception_name = anchor[10:]  # Remove 'exception-' prefix
    code = spec_div.find('code')
    signature = extract_signature(code) if code else ""
    
    return {
        'kind': 'exception',
        'name': exception_name,
        'signature': signature,
        'anchor': anchor
    }

def parse_class_definition(spec_div) -> Optional[Dict[str, Any]]:
    """Parse a class definition from a spec div."""
    anchor = spec_div.get('id', '')
    if not anchor.startswith('class-'):
        return None
    
    class_name = anchor[6:]  # Remove 'class-' prefix
    code = spec_div.find('code')
    signature = extract_signature(code) if code else ""
    
    return {
        'kind': 'class',
        'name': class_name,
        'signature': signature,
        'anchor': anchor
    }

def extract_code_examples(content: str) -> List[str]:
    """Extract code examples from documentation."""
    soup = BeautifulSoup(content, 'lxml')
    examples = []
    
    # Find code blocks in <pre> tags
    for pre in soup.find_all('pre', class_='language-ocaml'):
        code = pre.find('code')
        if code:
            examples.append(clean_text(code.get_text()))
    
    return examples

def parse_module_content(html_content: str) -> Dict[str, Any]:
    """
    Parse the HTML content of a module documentation page.
    Returns structured data with types, values, modules, etc.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    result = {
        'types': [],
        'values': [],
        'modules': [],
        'exceptions': [],
        'classes': [],
        'includes': [],
        'documentation_sections': []
    }
    
    # Process all spec divs
    for spec_div in soup.find_all('div', class_='spec'):
        # Check if it's followed by documentation
        next_sibling = spec_div.find_next_sibling('div', class_='spec-doc')
        
        # Determine type and parse accordingly
        if 'type' in spec_div.get('class', []):
            parsed = parse_type_definition(spec_div)
            if parsed:
                result['types'].append(parsed)
        elif 'value' in spec_div.get('class', []):
            parsed = parse_value_definition(spec_div, next_sibling)
            if parsed:
                result['values'].append(parsed)
        elif 'module' in spec_div.get('class', []):
            parsed = parse_module_definition(spec_div)
            if parsed:
                result['modules'].append(parsed)
        elif 'exception' in spec_div.get('class', []):
            parsed = parse_exception_definition(spec_div)
            if parsed:
                result['exceptions'].append(parsed)
        elif 'class' in spec_div.get('class', []):
            parsed = parse_class_definition(spec_div)
            if parsed:
                result['classes'].append(parsed)
    
    # Extract include statements
    for include_elem in soup.find_all('summary', class_='spec include'):
        code = include_elem.find('code')
        if code:
            include_text = extract_signature(code)
            result['includes'].append(include_text)
    
    # Extract general documentation paragraphs
    for p in soup.find_all('p'):
        # Skip if it's inside a spec-doc (already captured)
        if not p.find_parent('div', class_='spec-doc'):
            text = clean_text(p.get_text())
            if text and len(text) > 10:  # Skip very short texts
                result['documentation_sections'].append(text)
    
    # Extract code examples
    result['code_examples'] = extract_code_examples(html_content)
    
    return result

def parse_json_documentation(json_path: str) -> Dict[str, Any]:
    """
    Parse a JSON documentation file and extract structured content.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {
        'type': data.get('type', 'documentation'),
        'breadcrumbs': data.get('breadcrumbs', []),
        'preamble': clean_text(data.get('preamble', '')),
        'uses_katex': data.get('uses_katex', False),
        'toc': data.get('toc', [])
    }
    
    # Parse the HTML content
    if 'content' in data:
        parsed_content = parse_module_content(data['content'])
        result.update(parsed_content)
    
    return result

def extract_module_path(breadcrumbs: List[Dict[str, str]]) -> str:
    """Extract the full module path from breadcrumbs."""
    # Skip non-module breadcrumbs (pages, versions, etc.)
    module_parts = []
    for crumb in breadcrumbs:
        if crumb.get('kind') == 'module':
            module_parts.append(crumb.get('name', ''))
    
    return '.'.join(module_parts) if module_parts else ""