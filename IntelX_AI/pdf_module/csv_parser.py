"""
CSV file parser module.
Extracts data from CSV files and converts to text format for LLM processing.
"""
import csv
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import CSVParserError, FileHandlerError


def parse_csv_file(csv_path: str) -> str:
    """
    Parse CSV file and convert to text format for LLM extraction.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        String representation of CSV data
    """
    if not csv_path:
        raise CSVParserError("CSV file path is required", csv_path)
    
    path = Path(csv_path)
    
    if not path.exists():
        raise FileHandlerError(f"CSV file not found: {csv_path}", csv_path)
    
    if not path.is_file():
        raise FileHandlerError(f"Path is not a file: {csv_path}", csv_path)
    
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            # Read sample to detect dialect
            sample = file_obj.read(1024)
            file_obj.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample)
            except (csv.Error, UnicodeDecodeError):
                dialect = csv.excel
            
            reader = csv.DictReader(file_obj, dialect=dialect)
            rows = list(reader)
        
        # Convert CSV data to readable text format
        text_parts = []
        
        if rows:
            # Add headers
            headers = list(rows[0].keys())
            text_parts.append("CSV Headers: " + ", ".join(headers))
            text_parts.append("\n")
            
            # Add rows
            for idx, row in enumerate(rows, 1):
                text_parts.append(f"Row {idx}:")
                for key, value in row.items():
                    if value:  # Only include non-empty values
                        text_parts.append(f"  {key}: {value}")
                text_parts.append("")
        
        return "\n".join(text_parts)
    
    except FileNotFoundError as e:
        raise FileHandlerError(f"CSV file not found: {csv_path}", csv_path) from e
    except PermissionError as e:
        raise FileHandlerError(f"Permission denied reading CSV file: {csv_path}", csv_path) from e
    except UnicodeDecodeError as e:
        raise CSVParserError(f"Encoding error reading CSV file: {csv_path}", csv_path) from e
    except csv.Error as e:
        raise CSVParserError(f"CSV parsing error: {str(e)}", csv_path) from e
    except Exception as e:
        raise CSVParserError(f"Unexpected error parsing CSV file: {str(e)}", csv_path) from e


def extract_csv_data(csv_path: str) -> Dict[str, Any]:
    """
    Extract structured data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with CSV structure and data
        
    Raises:
        CSVParserError: If parsing fails
        FileHandlerError: If file operations fail
    """
    if not csv_path:
        raise CSVParserError("CSV file path is required", csv_path)
    
    path = Path(csv_path)
    
    if not path.exists():
        raise FileHandlerError(f"CSV file not found: {csv_path}", csv_path)
    
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            sample = file_obj.read(1024)
            file_obj.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample)
            except (csv.Error, UnicodeDecodeError):
                dialect = csv.excel
            
            reader = csv.DictReader(file_obj, dialect=dialect)
            rows = list(reader)
        
        return {
            "file_path": str(path),
            "total_rows": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "data": rows
        }
    
    except FileNotFoundError as e:
        raise FileHandlerError(f"CSV file not found: {csv_path}", csv_path) from e
    except PermissionError as e:
        raise FileHandlerError(f"Permission denied reading CSV file: {csv_path}", csv_path) from e
    except UnicodeDecodeError as e:
        raise CSVParserError(f"Encoding error reading CSV file: {csv_path}", csv_path) from e
    except csv.Error as e:
        raise CSVParserError(f"CSV parsing error: {str(e)}", csv_path) from e
    except Exception as e:
        raise CSVParserError(f"Unexpected error extracting CSV data: {str(e)}", csv_path) from e

