"""Utilities for handling opam-style version ordering."""
import re
from typing import List, Tuple, Union, Optional


def split_version(version: str) -> List[Union[str, int]]:
    """
    Split a version string into alternating non-digit and digit sequences.
    Always starts with a non-digit sequence (empty string if version starts with digit).
    
    Examples:
    - "1.0~beta2" -> ["", 1, ".", 0, "~beta", 2]
    - "3.0.0" -> ["", 3, ".", 0, ".", 0]
    - "dev" -> ["dev"]
    """
    parts = []
    current = ""
    in_digit = False
    
    # Always start with a non-digit sequence
    if version and version[0].isdigit():
        parts.append("")
    
    for char in version:
        if char.isdigit():
            if not in_digit:
                # Switching from non-digit to digit
                if current:
                    parts.append(current)
                current = char
                in_digit = True
            else:
                current += char
        else:
            if in_digit:
                # Switching from digit to non-digit
                parts.append(int(current))
                current = char
                in_digit = False
            else:
                current += char
    
    # Add the last part
    if current:
        if in_digit:
            parts.append(int(current))
        else:
            parts.append(current)
    
    return parts


def compare_version_parts(part1: Union[str, int], part2: Union[str, int]) -> int:
    """
    Compare two version parts according to opam rules.
    Returns -1 if part1 < part2, 0 if equal, 1 if part1 > part2.
    
    Rules:
    - Integers are compared numerically
    - For strings:
      - '~' sorts before everything, even empty string
      - Letters sort before non-letters
      - Non-letters sorted by ASCII order
    """
    # Both integers
    if isinstance(part1, int) and isinstance(part2, int):
        if part1 < part2:
            return -1
        elif part1 > part2:
            return 1
        else:
            return 0
    
    # Both strings
    if isinstance(part1, str) and isinstance(part2, str):
        # Special handling for ~
        if part1.startswith('~') and not part2.startswith('~'):
            return -1
        if part2.startswith('~') and not part1.startswith('~'):
            return 1
        
        # Empty string handling
        if not part1 and part2:
            return -1
        if part1 and not part2:
            return 1
        
        # Character-by-character comparison with special rules
        for i in range(min(len(part1), len(part2))):
            c1, c2 = part1[i], part2[i]
            
            # Special case for ~
            if c1 == '~' and c2 != '~':
                return -1
            if c2 == '~' and c1 != '~':
                return 1
            
            # Letters before non-letters
            if c1.isalpha() and not c2.isalpha():
                return -1
            if c2.isalpha() and not c1.isalpha():
                return 1
            
            # ASCII comparison
            if c1 < c2:
                return -1
            if c1 > c2:
                return 1
        
        # If all characters match, shorter string comes first
        if len(part1) < len(part2):
            return -1
        elif len(part1) > len(part2):
            return 1
        else:
            return 0
    
    # Type mismatch should not happen in well-formed versions
    # But if it does, treat numbers as less than strings
    if isinstance(part1, int):
        return -1
    else:
        return 1


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings according to opam ordering.
    Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
    """
    parts1 = split_version(v1)
    parts2 = split_version(v2)
    
    # Compare part by part
    for i in range(min(len(parts1), len(parts2))):
        result = compare_version_parts(parts1[i], parts2[i])
        if result != 0:
            return result
    
    # If all compared parts are equal, we need to check what's left
    # A version without additional parts is greater than one with ~ parts
    if len(parts1) < len(parts2):
        # Check if the remaining parts in parts2 start with ~
        for i in range(len(parts1), len(parts2)):
            if isinstance(parts2[i], str) and parts2[i].startswith('~'):
                return 1  # v1 is greater because v2 has ~ suffix
        return -1
    elif len(parts1) > len(parts2):
        # Check if the remaining parts in parts1 start with ~
        for i in range(len(parts2), len(parts1)):
            if isinstance(parts1[i], str) and parts1[i].startswith('~'):
                return -1  # v1 is less because it has ~ suffix
        return 1
    else:
        return 0


def find_latest_version(versions: List[str]) -> Tuple[Optional[str], None]:
    """
    Find the latest version from a list of version strings using opam ordering.
    Returns tuple of (latest_version_string, None) for compatibility.
    """
    if not versions:
        return None, None
    
    # Create a custom class for comparison
    class OpamVersion:
        def __init__(self, version):
            self.version = version
        
        def __lt__(self, other):
            return compare_versions(self.version, other.version) < 0
        
        def __le__(self, other):
            return compare_versions(self.version, other.version) <= 0
        
        def __gt__(self, other):
            return compare_versions(self.version, other.version) > 0
        
        def __ge__(self, other):
            return compare_versions(self.version, other.version) >= 0
        
        def __eq__(self, other):
            return compare_versions(self.version, other.version) == 0
    
    # Sort using the custom comparison class
    opam_versions = [OpamVersion(v) for v in versions]
    opam_versions.sort(reverse=True)
    
    return opam_versions[0].version, None


# Keep old functions for compatibility but mark as deprecated
def normalize_version(version_str):
    """DEPRECATED: Use opam version ordering instead."""
    return version_str


def parse_version(version_str):
    """DEPRECATED: Use opam version ordering instead."""
    return None