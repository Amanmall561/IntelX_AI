"""
Centralised configuration loader for IntelX_AI.

Values are sourced from environment variables. To configure locally,
copy `env.template` to `.env` (in the same directory) and adjust as needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Attempt to load `.env` placed next to this file.
_env_file = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=False)
if _env_file:
    load_dotenv(_env_file)
else:
    # Fallback: load from env.template if user renamed differently (best-effort).
    template_path = BASE_DIR / "env.template"
    if template_path.exists():
        load_dotenv(template_path)


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and (value is None or value == ""):
        raise EnvironmentError(f"Missing required environment variable `{key}`.")
    return value


DEFAULT_INPUT_FILE = _get_env("DEFAULT_INPUT_FILE")
YOLO_MODEL_PATH = _get_env("YOLO_MODEL_PATH", required=True)
LLM_MODEL_NAME = _get_env("LLM_MODEL_NAME", "mosaicml/mpt-7b-instruct")
TICKET_MODEL_ID = _get_env("TICKET_MODEL_ID", "allenai/olmOCR-7B-0725")
TEMP_IMAGE_DIR = _get_env("TEMP_IMAGE_DIR", "temp_img")

