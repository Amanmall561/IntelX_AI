"""
Active Learning Feedback Logger & Pattern Advisor (Phase 5.3)
==============================================================
Every correction made by a human reviewer is:
  1. Written to a JSONL training log (one JSON line per correction).
  2. Stored in the ReviewQueueStore SQLite DB (via queue_store.save_al_entry).
  3. Analysed by ActiveLearningAnalyser to detect systematic error patterns.
  4. Used by PromptTweakAdvisor to suggest targeted prompt improvements.

The JSONL file is the primary artifact for fine-tuning downstream:
  - YOLO bounding-box corrections  → YOLO fine-tune dataset
  - Field corrections               → LLM instruction fine-tune dataset
"""
from __future__ import annotations

import json
import logging
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hitl_config import ACTIVE_LEARNING_LOG, AL_MIN_CORRECTIONS_FOR_REPORT
from hitl.models import ActiveLearningEntry, Correction, FieldCorrection

logger = logging.getLogger(__name__)


# ── JSONL writer ──────────────────────────────────────────────────────────────

class ActiveLearningLogger:
    """
    Phase 5.3: Persists every human correction as an active-learning record.
    Thread-safe via a lock on the JSONL file.
    """

    def __init__(
        self,
        log_path: str = ACTIVE_LEARNING_LOG,
        queue_store=None,          # Optional ReviewQueueStore for DB-side logging
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue_store = queue_store

    def log_correction(
        self,
        correction: Correction,
        original_ai_output: Dict[str, Any],
        original_confidence_scores: Optional[Dict[str, float]] = None,
    ) -> ActiveLearningEntry:
        """
        Build an ActiveLearningEntry from a reviewer Correction and append
        it to the JSONL log.

        Args:
            correction: The Correction object submitted by the reviewer.
            original_ai_output: The AI-extracted JSON before corrections.
            original_confidence_scores: Per-field confidence at extraction time.

        Returns:
            The created ActiveLearningEntry.
        """
        error_fields = [fc.field_path for fc in correction.field_corrections]

        entry = ActiveLearningEntry(
            document_id=correction.document_id,
            document_type=correction.document_type,
            file_path="",   # enriched by caller if available
            ai_output=original_ai_output,
            ground_truth=correction.corrected_json,
            field_diffs=correction.field_corrections,
            error_fields=error_fields,
            original_confidence_scores=original_confidence_scores or {},
            bbox_corrections=correction.bbox_corrections,
            reviewer_id=correction.reviewer_id,
        )

        self._write_jsonl(entry)

        # Also persist to DB if store is wired in
        if self._queue_store is not None:
            try:
                self._queue_store.save_al_entry(entry.model_dump(mode="json"))
            except Exception as e:
                logger.warning("Failed to save AL entry to DB: %s", e)

        logger.info(
            "Active learning entry %s logged (%d field diffs, doc_type=%s)",
            entry.entry_id,
            len(error_fields),
            correction.document_type,
        )
        return entry

    def _write_jsonl(self, entry: ActiveLearningEntry) -> None:
        """Append one JSON line to the active learning log (thread-safe)."""
        line = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def export_training_data(
        self,
        output_path: str,
        doc_type: Optional[str] = None,
        format: str = "jsonl",   # "jsonl" | "json"
    ) -> int:
        """
        Export corrections as a fine-tuning dataset.

        Args:
            output_path: Where to write the export.
            doc_type:    If set, export only records for that document type.
            format:      "jsonl" (one record per line) or "json" (list).

        Returns:
            Number of records exported.
        """
        records = self._load_all(doc_type)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            if format == "json":
                json.dump(records, f, ensure_ascii=False, indent=2)
            else:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Exported %d AL records to %s", len(records), output_path)
        return len(records)

    def _load_all(
        self, doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load and optionally filter all JSONL records."""
        records: List[Dict[str, Any]] = []
        if not self._log_path.exists():
            return records
        with self._log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if doc_type is None or rec.get("document_type") == doc_type:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        return records


# ── Pattern analyser ──────────────────────────────────────────────────────────

class ActiveLearningAnalyser:
    """
    Phase 5.3: Analyses accumulated corrections to surface systematic errors.
    """

    def __init__(self, logger_instance: ActiveLearningLogger) -> None:
        self._al_logger = logger_instance

    def analyse_patterns(
        self, doc_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns a report dict identifying:
          - Most frequently corrected fields
          - Confidence level at time of error
          - Field coverage gaps (missing fields)
          - Total corrections analysed
        """
        records = self._al_logger._load_all(doc_type)
        if len(records) < AL_MIN_CORRECTIONS_FOR_REPORT:
            return {
                "status": "insufficient_data",
                "records_found": len(records),
                "min_required": AL_MIN_CORRECTIONS_FOR_REPORT,
                "message": f"Need at least {AL_MIN_CORRECTIONS_FOR_REPORT} corrections to generate a report.",
            }

        field_error_count: Counter = Counter()
        field_confidence_sum: Dict[str, float] = defaultdict(float)
        doc_type_counts: Counter = Counter()
        missing_field_count: Counter = Counter()
        total_corrections = 0

        for rec in records:
            dt = rec.get("document_type", "Unknown")
            doc_type_counts[dt] += 1

            field_diffs = rec.get("field_diffs", [])
            original_scores = rec.get("original_confidence_scores", {})

            for diff in field_diffs:
                field_path = diff.get("field_path", "unknown")
                field_error_count[field_path] += 1
                total_corrections += 1
                # Track what the confidence was when the error occurred
                score = original_scores.get(field_path, 0.0)
                field_confidence_sum[field_path] += score

        # Compute average confidence-at-error for each field
        field_avg_confidence = {
            field: round(field_confidence_sum[field] / count, 4)
            for field, count in field_error_count.items()
        }

        top_errors = field_error_count.most_common(15)

        return {
            "status": "ok",
            "records_analysed": len(records),
            "total_field_corrections": total_corrections,
            "doc_type_filter": doc_type,
            "doc_type_breakdown": dict(doc_type_counts),
            "top_error_fields": [
                {
                    "field": field,
                    "error_count": count,
                    "avg_confidence_at_error": field_avg_confidence.get(field, 0.0),
                }
                for field, count in top_errors
            ],
            "most_problematic_field": top_errors[0][0] if top_errors else None,
        }


# ── Prompt tweak advisor ──────────────────────────────────────────────────────

class PromptTweakAdvisor:
    """
    Phase 5.3: Generates targeted prompt improvement suggestions based on
    the pattern analysis report.
    """

    # Template suggestions per common error pattern
    _SUGGESTIONS: Dict[str, str] = {
        "name": (
            "The 'name' field is frequently corrected. "
            "Add this rule to the prompt: "
            "'Extract the FULL legal name as printed. Do NOT split into first/last.'"
        ),
        "date": (
            "Date fields are frequently wrong. "
            "Add: 'Return all dates in ISO format YYYY-MM-DD. "
            "If only partial date is visible, fill missing parts with XX (e.g. 2024-XX-15).'"
        ),
        "identifier": (
            "Identifier fields are being corrected often. "
            "Add: 'Extract identifier numbers exactly as printed, including spaces. "
            "Validate Aadhaar as 12 digits, PAN as 10 chars AAAAA9999A.'"
        ),
        "address": (
            "Address fields have high error rates. "
            "Add: 'Extract the complete address including house number, street, "
            "city, state, and PIN code as separate sub-fields.'"
        ),
        "account": (
            "Account/IFSC fields need improvement. "
            "Add: 'Account numbers are purely numeric 9-18 digits. "
            "IFSC codes match pattern: 4 alpha + 0 + 6 alphanumeric.'"
        ),
        "transaction": (
            "Transaction extraction needs work. "
            "Add: 'For each row in the transaction table, extract date, withdrawal, "
            "deposit, and closing balance as separate numeric fields (no commas, "
            "no currency symbols).'"
        ),
    }

    def suggest_prompt_tweak(
        self, pattern_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Given a pattern_report from ActiveLearningAnalyser, return actionable
        prompt improvement suggestions.
        """
        if pattern_report.get("status") != "ok":
            return {
                "status": "no_suggestion",
                "reason": pattern_report.get("message", "Insufficient data."),
            }

        top_errors = pattern_report.get("top_error_fields", [])
        suggestions: List[Dict[str, Any]] = []

        for error in top_errors[:5]:
            field = error["field"].lower()
            suggestion_text = None

            for keyword, text in self._SUGGESTIONS.items():
                if keyword in field:
                    suggestion_text = text
                    break

            if suggestion_text is None:
                suggestion_text = (
                    f"Field '{error['field']}' has {error['error_count']} corrections. "
                    f"Consider adding explicit extraction rules for this field in the prompt."
                )

            suggestions.append({
                "field": error["field"],
                "error_count": error["error_count"],
                "avg_confidence_at_error": error["avg_confidence_at_error"],
                "suggestion": suggestion_text,
            })

        return {
            "status": "ok",
            "doc_type": pattern_report.get("doc_type_filter", "all"),
            "records_analysed": pattern_report.get("records_analysed", 0),
            "suggestions": suggestions,
            "note": (
                "These suggestions target the highest-frequency error fields. "
                "Apply them by editing the relevant extraction prompt template."
            ),
        }
