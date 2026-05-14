"""
HITL Module Configuration
==========================
All tunable parameters for the Human-in-the-Loop validation pipeline.
Override values via environment variables or a local `.env` file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "hitl" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Confidence gate ───────────────────────────────────────────────────────────
# Documents with all-field mean confidence >= this threshold are auto-approved.
CONFIDENCE_THRESHOLD: float = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.55"))

# Fields below this individual score are flagged as low-confidence (amber).
LOW_CONFIDENCE_FIELD_THRESHOLD: float = float(
    os.getenv("HITL_LOW_CONF_FIELD_THRESHOLD", "0.55")
)

# ── Storage ───────────────────────────────────────────────────────────────────
SQLITE_DB_PATH: str = os.getenv(
    "HITL_SQLITE_DB_PATH", str(DATA_DIR / "review_queue.db")
)
ACTIVE_LEARNING_LOG: str = os.getenv(
    "HITL_AL_LOG_PATH", str(DATA_DIR / "active_learning.jsonl")
)

# ── Server ports ──────────────────────────────────────────────────────────────
REVIEWER_API_HOST: str = os.getenv("HITL_API_HOST", "0.0.0.0")
REVIEWER_API_PORT: int = int(os.getenv("HITL_API_PORT", "7860"))
HITL_UI_PORT: int = int(os.getenv("HITL_UI_PORT", "8501"))

# ── Deduplication ─────────────────────────────────────────────────────────────
# Minimum Levenshtein similarity ratio (0–100) to treat two entity strings as
# the same value during deduplication.
DEDUP_SIMILARITY_THRESHOLD: int = int(
    os.getenv("HITL_DEDUP_THRESHOLD", "90")
)

# ── Required fields per document type ─────────────────────────────────────────
# Only these fields are checked during the confidence gate evaluation.
REQUIRED_FIELDS_BY_DOC_TYPE: dict[str, list[str]] = {
    "AADHAR": ["people", "identifiers", "dates"],
    "PAN": ["people", "identifiers"],
    "Passport": ["people", "identifiers", "dates", "locations"],
    "DL": ["people", "identifiers", "dates", "vehicle_details"],
    "AIRTICKET": ["people", "identifiers", "dates", "locations", "organization_details"],
    "Bank Statement": ["Account_holder", "Statement"],
    "FIR": ["people", "identifiers", "dates", "locations", "organization_details"],
    "Other": ["people", "identifiers"],
    "Unknown": [],
}

# ── Active learning ───────────────────────────────────────────────────────────
# Minimum number of corrections before a pattern report is generated.
AL_MIN_CORRECTIONS_FOR_REPORT: int = int(
    os.getenv("HITL_AL_MIN_CORRECTIONS", "5")
)

# ── Multilingual ──────────────────────────────────────────────────────────────
# Target language for transliteration output.
TRANSLITERATION_TARGET_LANG: str = os.getenv("HITL_TRANSLITERATION_LANG", "en")

# ── Supported document types (for classification fallback) ────────────────────
KNOWN_DOC_TYPES: list[str] = [
    "AADHAR", "PAN", "Passport", "DL", "AIRTICKET",
    "Bank Statement", "FIR", "Other", "Unknown",
]
