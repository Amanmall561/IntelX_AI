"""
JSON file parser module.
Extracts data from JSON files and converts to text format for LLM processing.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import JSONParserError, FileHandlerError


def parse_json_file(json_path: str) -> str:
    """
    Parse JSON file and convert to text format for LLM extraction.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        String representation of JSON data
    """
    if not json_path:
        raise JSONParserError("JSON file path is required", json_path)
    
    path = Path(json_path)
    
    if not path.exists():
        raise FileHandlerError(f"JSON file not found: {json_path}", json_path)
    
    if not path.is_file():
        raise FileHandlerError(f"Path is not a file: {json_path}", json_path)
    
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        
        # Convert JSON to readable text format
        text_parts = []
        
        if isinstance(data, dict):
            text_parts.append("JSON Document Structure:")
            text_parts.append(_dict_to_text(data, indent=0))
        elif isinstance(data, list):
            text_parts.append(f"JSON Array with {len(data)} items:")
            for idx, item in enumerate(data, 1):
                text_parts.append(f"\nItem {idx}:")
                if isinstance(item, dict):
                    text_parts.append(_dict_to_text(item, indent=2))
                else:
                    text_parts.append(f"  {str(item)}")
        else:
            text_parts.append(f"JSON Value: {str(data)}")
        
        return "\n".join(text_parts)
    
    except FileNotFoundError as e:
        raise FileHandlerError(f"JSON file not found: {json_path}", json_path) from e
    except PermissionError as e:
        raise FileHandlerError(f"Permission denied reading JSON file: {json_path}", json_path) from e
    except json.JSONDecodeError as e:
        raise JSONParserError(f"Invalid JSON format: {str(e)}", json_path) from e
    except UnicodeDecodeError as e:
        raise JSONParserError(f"Encoding error reading JSON file: {json_path}", json_path) from e
    except Exception as e:
        raise JSONParserError(f"Unexpected error parsing JSON file: {str(e)}", json_path) from e


def _dict_to_text(data: Dict[str, Any], indent: int = 0) -> str:
    """Recursively convert dictionary to readable text format."""
    lines = []
    prefix = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dict_to_text(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}: [List with {len(value)} items]")
            for idx, item in enumerate(value[:5], 1):  # Show first 5 items
                if isinstance(item, dict):
                    lines.append(f"{prefix}  Item {idx}:")
                    lines.append(_dict_to_text(item, indent + 2))
                else:
                    lines.append(f"{prefix}  Item {idx}: {str(item)}")
            if len(value) > 5:
                lines.append(f"{prefix}  ... and {len(value) - 5} more items")
        else:
            lines.append(f"{prefix}{key}: {str(value)}")
    
    return "\n".join(lines)


def extract_json_data(json_path: str) -> Dict[str, Any]:
    """
    Extract structured data from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary with JSON structure and data
        
    Raises:
        JSONParserError: If parsing fails
        FileHandlerError: If file operations fail
    """
    if not json_path:
        raise JSONParserError("JSON file path is required", json_path)
    
    path = Path(json_path)
    
    if not path.exists():
        raise FileHandlerError(f"JSON file not found: {json_path}", json_path)
    
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        
        return {
            "file_path": str(path),
            "data_type": type(data).__name__,
            "data": data
        }
    
    except FileNotFoundError as e:
        raise FileHandlerError(f"JSON file not found: {json_path}", json_path) from e
    except PermissionError as e:
        raise FileHandlerError(f"Permission denied reading JSON file: {json_path}", json_path) from e
    except json.JSONDecodeError as e:
        raise JSONParserError(f"Invalid JSON format: {str(e)}", json_path) from e
    except UnicodeDecodeError as e:
        raise JSONParserError(f"Encoding error reading JSON file: {json_path}", json_path) from e
    except Exception as e:
        raise JSONParserError(f"Unexpected error extracting JSON data: {str(e)}", json_path) from e

