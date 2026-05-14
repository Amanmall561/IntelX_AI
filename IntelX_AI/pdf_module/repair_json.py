import json
import re


def fix_broken_json(text):
    """
    Repair common broken JSON issues:
    - Missing closing braces/brackets based on last unmatched opening
    - Trailing commas
    - Unclosed strings
    - Preserves correct order of closing symbols
    """

    text = text.strip()
    bad_json_string=text

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix unclosed quotes (odd number of ")
    if text.count('"') % 2 != 0:
        text += '"'

    # ------------------------------
    # Detect unmatched opening symbols in correct order
    # ------------------------------
    
    stack = []  # to track opening symbols in order

    for ch in text:
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                # Match if correct closing is found
                if (ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "["):
                    stack.pop()

    # Now stack contains UNMATCHED opening brackets in order
    # We need to close them in reverse order
    closing_map = {"{": "}", "[": "]"}

    while stack:
        last_open = stack.pop()  # close last unmatched
        text += closing_map[last_open]

    # Try parsing now
    try:
        return json.loads(text)
    except:
        pass

    # Last attempt: If looks like a dict but missing outer braces
    if ":" in text:
        if not text.startswith("{"):
            text = "{" + text
        if not text.endswith("}"):
            text = text + "}"

    try:
        return json.loads(text)
    except:
        # pip install json-repair
        from json_repair import repair_json
        good_json_string = repair_json(bad_json_string)
        return good_json_string
