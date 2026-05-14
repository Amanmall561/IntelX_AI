"""
tests/test_json_repair.py — Unit test for robust repair logic
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hitl.utils.json_repair import repair_and_parse

def test_truncation_fix():
    # Example provided by user (truncated key)
    failing_str = "{'warranty': '1 year', 'delay': 'Penalty Rs. 500', 'Procuring entity'}"
    
    result = repair_and_parse(failing_str)
    print(f"Result: {result}")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "Procuring entity" in result, "Repaired dict should contain 'Procuring entity'"
    assert result["Procuring entity"] == "", f"Value should be empty string, got: {result['Procuring entity']}"
    print("Test passed!")

if __name__ == "__main__":
    test_truncation_fix()
