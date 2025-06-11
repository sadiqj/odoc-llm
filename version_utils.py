"""Utilities for handling semantic versioning of OCaml packages."""
import re
import semantic_version

def normalize_version(version_str):
    """
    Normalize version strings to be compatible with semantic versioning.
    Handles OCaml-specific version formats like:
    - v0.17.1 -> 0.17.1
    - 4.2.1-1 -> 4.2.1+1
    - 2.5.2 -> 2.5.2
    """
    # Remove 'v' prefix if present
    version = version_str.lstrip('v')
    
    # Handle OCaml-style suffixes (e.g., 4.2.1-1)
    # Convert hyphen to plus for prerelease versions
    if '-' in version and not version.startswith('-'):
        parts = version.split('-', 1)
        if parts[1].isdigit():
            # It's a build number, use + notation
            version = f"{parts[0]}+{parts[1]}"
        else:
            # It's a prerelease identifier
            version = f"{parts[0]}-{parts[1]}"
    
    # Ensure version has at least major.minor.patch
    parts = version.split('.', 2)
    if len(parts) == 1:
        version = f"{parts[0]}.0.0"
    elif len(parts) == 2:
        # Check if there's a suffix
        if '+' in parts[1] or '-' in parts[1]:
            base = parts[1].split('+')[0].split('-')[0]
            suffix = parts[1][len(base):]
            version = f"{parts[0]}.{base}.0{suffix}"
        else:
            version = f"{parts[0]}.{parts[1]}.0"
    
    return version


def parse_version(version_str):
    """Parse a version string into a semantic_version.Version object."""
    normalized = normalize_version(version_str)
    try:
        return semantic_version.Version(normalized)
    except ValueError:
        # If parsing fails, create a simple version
        # Extract just numbers and create a basic version
        numbers = re.findall(r'\d+', version_str)
        if numbers:
            major = numbers[0] if len(numbers) > 0 else '0'
            minor = numbers[1] if len(numbers) > 1 else '0'
            patch = numbers[2] if len(numbers) > 2 else '0'
            return semantic_version.Version(f"{major}.{minor}.{patch}")
        else:
            # Last resort - treat as 0.0.0
            return semantic_version.Version("0.0.0")


def find_latest_version(versions):
    """
    Find the latest version from a list of version strings.
    Returns tuple of (latest_version_string, parsed_version).
    """
    if not versions:
        return None, None
    
    version_map = {}
    for v in versions:
        try:
            parsed = parse_version(v)
            version_map[v] = parsed
        except Exception as e:
            print(f"Warning: Could not parse version {v}: {e}")
            # Use a very low version for unparseable versions
            version_map[v] = semantic_version.Version("0.0.0")
    
    # Sort by parsed version
    sorted_versions = sorted(version_map.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_versions:
        return sorted_versions[0][0], sorted_versions[0][1]
    
    return None, None


def compare_versions(v1, v2):
    """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
    parsed_v1 = parse_version(v1)
    parsed_v2 = parse_version(v2)
    
    if parsed_v1 < parsed_v2:
        return -1
    elif parsed_v1 > parsed_v2:
        return 1
    else:
        return 0