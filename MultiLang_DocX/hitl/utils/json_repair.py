"""
hitl.utils.json_repair — Robust JSON/Dictionary Repair (Phase 5.3 Supplement)
===========================================================================
Handles malformed LLM outputs that occasionally:
  1. Return Python dictionary strings (single quotes) instead of JSON.
  2. Are truncated or missing required delimiters (e.g. trailing keys).
  3. Include markdown artifacts (e.g. ```json blocks).
"""
import json
import logging
import ast
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("hitl.utils.json_repair")

def repair_and_parse(raw_str: str) -> Dict[str, Any]:
    """
    Attempts to parse a string as JSON or a Python dictionary, repairing common
    formatting issues if needed.
    """
    if not raw_str or not isinstance(raw_str, str):
        return {}

    # Cleanup: Remove markdown code blocks if present
    cleaned = raw_str.strip()
    if cleaned.startswith("```"):
        # Remove starting ```json or ``` and ending ```
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # Step 1: Standard json.loads
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 2: ast.literal_eval for Python-style dict strings (single quotes)
    try:
        # literal_eval is safer than eval()
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Step 3: Repair common truncation errors
    repaired = cleaned
    
    # Case A: Missing closing brace
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
        logger.warning("Repaired truncated JSON/Dict: Added missing closing brace.")

    # Case B: Ends with a key but no colon or value (e.g. ... 'key'} )
    # This regex looks for a trailing single or double quoted key just before the closing brace
    trailing_key_match = re.search(r"(['\"][\w\s\-\.\/]+['\"])\s*\}\s*$", repaired)
    if trailing_key_match:
        key = trailing_key_match.group(1)
        # Patch it by adding a colon and empty value
        repaired = repaired.replace(key + "}", f"{key}: \"\"}}")
        logger.warning("Repaired truncated JSON/Dict: Added empty value for trailing key.")

    # Case C: JSON but single quotes
    if "'" in repaired and '"' not in repaired:
        # Very risky to just replace ' with ", but often works for simple LLM outputs
        # Only do this as a last resort if literal_eval failed (which it will if it's tried already)
        pass

    # Step 4: Final attempt after repair
    try:
        # Try both again with the repaired string
        try:
            return json.loads(repaired)
        except:
            result = ast.literal_eval(repaired)
            if isinstance(result, dict):
                return result
    except Exception as e:
        logger.error("Failed to repair and parse string: %s. Raw was: %s...", e, cleaned[:100])
        return {}

    return {}
