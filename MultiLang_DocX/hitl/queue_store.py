"""
SQLite-backed Review Queue Store (Phase 5.2)
============================================
Persistent storage for PENDING_REVIEW documents and all downstream
review actions:  claim → approve | correct | reject.

Tables:
  review_queue       — one row per ReviewItem
  corrections        — one row per Correction (field-level diffs)
  active_learning_log— JSONL-like rows for fine-tuning (Phase 5.3 fallback)

All DB operations are synchronous (using sqlite3) and protected by a
threading.Lock so the store is safe to use from multiple asyncio.to_thread
calls simultaneously.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hitl_config import SQLITE_DB_PATH
from hitl.models import (
    Correction,
    DocumentState,
    FieldCorrection,
    ReviewItem,
)

logger = logging.getLogger(__name__)


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
-- Phase 5.2: Review queue
CREATE TABLE IF NOT EXISTS review_queue (
    item_id           TEXT PRIMARY KEY,
    document_id       TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    document_type     TEXT NOT NULL DEFAULT 'Unknown',
    aggregated_json   TEXT NOT NULL DEFAULT '{}',
    confidence_scores TEXT NOT NULL DEFAULT '{}',
    overall_confidence REAL NOT NULL DEFAULT 0.0,
    flagged_fields    TEXT NOT NULL DEFAULT '[]',
    missing_fields    TEXT NOT NULL DEFAULT '[]',
    state             TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    reviewer_id       TEXT,
    claimed_at        TEXT,
    resolved_at       TEXT,
    reviewer_notes    TEXT NOT NULL DEFAULT '',
    priority          INTEGER NOT NULL DEFAULT 5,
    page_thumbnails   TEXT NOT NULL DEFAULT '[]',
    multilingual_mappings TEXT NOT NULL DEFAULT '[]',
    queued_at         TEXT NOT NULL
);

-- Phase 5.3: Corrections
CREATE TABLE IF NOT EXISTS corrections (
    correction_id   TEXT PRIMARY KEY,
    item_id         TEXT NOT NULL REFERENCES review_queue(item_id),
    document_id     TEXT NOT NULL,
    document_type   TEXT NOT NULL,
    reviewer_id     TEXT NOT NULL,
    field_corrections TEXT NOT NULL DEFAULT '[]',
    corrected_json  TEXT NOT NULL DEFAULT '{}',
    bbox_corrections TEXT NOT NULL DEFAULT '[]',
    reviewer_notes  TEXT NOT NULL DEFAULT '',
    corrected_at    TEXT NOT NULL
);

-- Phase 5.3: Active learning log (supplemental to JSONL file)
CREATE TABLE IF NOT EXISTS active_learning_log (
    entry_id          TEXT PRIMARY KEY,
    document_id       TEXT NOT NULL,
    document_type     TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    ai_output         TEXT NOT NULL DEFAULT '{}',
    ground_truth      TEXT NOT NULL DEFAULT '{}',
    field_diffs       TEXT NOT NULL DEFAULT '[]',
    error_fields      TEXT NOT NULL DEFAULT '[]',
    original_confidence_scores TEXT NOT NULL DEFAULT '{}',
    bbox_corrections  TEXT NOT NULL DEFAULT '[]',
    reviewer_id       TEXT NOT NULL DEFAULT '',
    logged_at         TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_rq_state      ON review_queue(state);
CREATE INDEX IF NOT EXISTS idx_rq_doc_id     ON review_queue(document_id);
CREATE INDEX IF NOT EXISTS idx_rq_queued_at  ON review_queue(queued_at);
CREATE INDEX IF NOT EXISTS idx_corr_item_id  ON corrections(item_id);
CREATE INDEX IF NOT EXISTS idx_al_doc_type   ON active_learning_log(document_type);
"""


class QueueStats:
    """Snapshot of review queue statistics."""
    def __init__(
        self,
        total: int,
        pending: int,
        approved: int,
        corrected: int,
        rejected: int,
        avg_confidence: float,
    ) -> None:
        self.total = total
        self.pending = pending
        self.approved = approved
        self.corrected = corrected
        self.rejected = rejected
        self.avg_confidence = avg_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "pending_review": self.pending,
            "approved": self.approved,
            "corrected": self.corrected,
            "rejected": self.rejected,
            "avg_confidence": round(self.avg_confidence, 4),
        }


class ReviewQueueStore:
    """
    Thread-safe SQLite-backed store for review items and corrections.
    Instantiate once and share across all threads/coroutines.
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(_DDL)
        logger.info("Review queue DB initialized at %s", self._db_path)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _j(obj: Any) -> str:
        return json.dumps(obj, default=str)

    @staticmethod
    def _pj(text: str) -> Any:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    def _row_to_review_item(self, row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            item_id=row["item_id"],
            document_id=row["document_id"],
            file_path=row["file_path"],
            document_type=row["document_type"],
            aggregated_json=self._pj(row["aggregated_json"]),
            confidence_scores=self._pj(row["confidence_scores"]),
            overall_confidence=row["overall_confidence"],
            flagged_fields=self._pj(row["flagged_fields"]),
            missing_required_fields=self._pj(row["missing_fields"]),
            state=DocumentState(row["state"]),
            reviewer_id=row["reviewer_id"],
            claimed_at=datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None,
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
            reviewer_notes=row["reviewer_notes"] or "",
            priority=row["priority"],
            page_thumbnails=self._pj(row["page_thumbnails"]),
            multilingual_mappings=self._pj(row["multilingual_mappings"]),
            queued_at=datetime.fromisoformat(row["queued_at"]),
        )

    # ── Phase 5.2 — Queue Operations ─────────────────────────────────────────

    def push(self, item: ReviewItem) -> None:
        """Insert a new ReviewItem into the review queue."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO review_queue
                    (item_id, document_id, file_path, document_type,
                     aggregated_json, confidence_scores, overall_confidence,
                     flagged_fields, missing_fields, state, reviewer_id,
                     claimed_at, resolved_at, reviewer_notes, priority,
                     page_thumbnails, multilingual_mappings, queued_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.item_id,
                        item.document_id,
                        item.file_path,
                        item.document_type,
                        self._j(item.aggregated_json),
                        self._j(item.confidence_scores),
                        item.overall_confidence,
                        self._j(item.flagged_fields),
                        self._j(item.missing_required_fields),
                        item.state.value,
                        item.reviewer_id,
                        item.claimed_at.isoformat() if item.claimed_at else None,
                        item.resolved_at.isoformat() if item.resolved_at else None,
                        item.reviewer_notes,
                        item.priority,
                        self._j(item.page_thumbnails),
                        self._j(item.multilingual_mappings),
                        item.queued_at.isoformat(),
                    ),
                )
        logger.info("Queued ReviewItem %s for document %s", item.item_id, item.document_id)

    def peek(
        self,
        limit: int = 20,
        offset: int = 0,
        state_filter: Optional[str] = "PENDING_REVIEW",
        reviewer_id: Optional[str] = None,
        unclaimed_only: bool = False,
    ) -> List[ReviewItem]:
        """
        List ReviewItems with dynamic filtering.
        - state_filter: Filter by document state (e.g. 'PENDING_REVIEW')
        - reviewer_id:  Filter to items claimed by a specific reviewer
        - unclaimed_only: Filter to items where reviewer_id is NULL
        """
        query = "SELECT * FROM review_queue WHERE 1=1"
        params = []

        if state_filter:
            query += " AND state = ?"
            params.append(state_filter)
        
        if reviewer_id:
            query += " AND reviewer_id = ?"
            params.append(reviewer_id)
        elif unclaimed_only:
            query += " AND reviewer_id IS NULL"

        query += " ORDER BY priority ASC, queued_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_review_item(r) for r in rows]

    def get(self, item_id: str) -> Optional[ReviewItem]:
        """Retrieve a single ReviewItem by ID."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM review_queue WHERE item_id = ?", (item_id,)
                ).fetchone()
        return self._row_to_review_item(row) if row else None

    def claim(self, item_id: str, reviewer_id: str, force: bool = False) -> Optional[ReviewItem]:
        """
        Atomically claim or re-assign a review item.
        - If force=True, overwrites any current reviewer (reassignment).
        - Returns None if the item doesn't exist or is already claimed by someone else (if force is False).
        """
        now = self._now()
        with self._lock:
            with self._connect() as conn:
                if force:
                    # Forced claim: overwrite regardless of current reviewer
                    updated = conn.execute(
                        """
                        UPDATE review_queue
                        SET reviewer_id = ?, claimed_at = ?, state = 'PENDING_REVIEW'
                        WHERE item_id = ?
                        """,
                        (reviewer_id, now, item_id),
                    ).rowcount
                else:
                    # standard claim: only if unclaimed or already yours
                    updated = conn.execute(
                        """
                        UPDATE review_queue
                        SET reviewer_id = ?, claimed_at = ?, state = 'PENDING_REVIEW'
                        WHERE item_id = ?
                          AND (reviewer_id IS NULL OR reviewer_id = ?)
                        """,
                        (reviewer_id, now, item_id, reviewer_id),
                    ).rowcount
                
                if updated == 0:
                    return None
                
                row = conn.execute(
                    "SELECT * FROM review_queue WHERE item_id = ?", (item_id,)
                ).fetchone()
        
        logger.info(
            "ReviewItem %s %s by reviewer %s",
            item_id,
            "FORCE-CLAIMED" if force else "claimed",
            reviewer_id,
        )
        return self._row_to_review_item(row) if row else None

    def approve(self, item_id: str, reviewer_id: str, notes: str = "") -> bool:
        """Mark a review item as APPROVED (no corrections needed)."""
        now = self._now()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    UPDATE review_queue
                    SET state = 'APPROVED', resolved_at = ?, reviewer_notes = ?
                    WHERE item_id = ? AND reviewer_id = ?
                    """,
                    (now, notes, item_id, reviewer_id),
                ).rowcount
        logger.info("ReviewItem %s APPROVED by %s", item_id, reviewer_id)
        return rows > 0

    def reject(self, item_id: str, reviewer_id: str, notes: str = "") -> bool:
        """Mark a review item as REJECTED."""
        now = self._now()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    UPDATE review_queue
                    SET state = 'REJECTED', resolved_at = ?, reviewer_notes = ?
                    WHERE item_id = ? AND reviewer_id = ?
                    """,
                    (now, notes, item_id, reviewer_id),
                ).rowcount
        logger.info("ReviewItem %s REJECTED by %s", item_id, reviewer_id)
        return rows > 0

    def delete(self, item_id: str) -> bool:
        """Permanently delete a review item and any associated corrections."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM corrections WHERE item_id = ?", (item_id,))
                rows = conn.execute("DELETE FROM review_queue WHERE item_id = ?", (item_id,)).rowcount
        logger.info("Deleted ReviewItem %s", item_id)
        return rows > 0

    def save_correction(self, correction: Correction) -> bool:
        """
        Phase 5.3: Save a field-level correction and mark the queue item
        as CORRECTED.
        """
        now = self._now()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO corrections
                    (correction_id, item_id, document_id, document_type,
                     reviewer_id, field_corrections, corrected_json,
                     bbox_corrections, reviewer_notes, corrected_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        correction.correction_id,
                        correction.item_id,
                        correction.document_id,
                        correction.document_type,
                        correction.reviewer_id,
                        self._j([fc.model_dump() for fc in correction.field_corrections]),
                        self._j(correction.corrected_json),
                        self._j(correction.bbox_corrections),
                        correction.reviewer_notes,
                        correction.corrected_at.isoformat(),
                    ),
                )
                conn.execute(
                    """
                    UPDATE review_queue
                    SET state = 'CORRECTED', resolved_at = ?, document_type = ?
                    WHERE item_id = ? AND reviewer_id = ?
                    """,
                    (now, correction.document_type, correction.item_id, correction.reviewer_id),
                )
        logger.info(
            "Correction %s saved for ReviewItem %s (%d field diffs)",
            correction.correction_id,
            correction.item_id,
            len(correction.field_corrections),
        )
        return True

    def get_corrections(self, item_id: str) -> List[Dict[str, Any]]:
        """Retrieve all corrections for a given review item."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE item_id = ?", (item_id,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Queue Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> QueueStats:
        """Return aggregate queue statistics."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*)                                          AS total,
                        SUM(CASE WHEN state='PENDING_REVIEW' THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN state='APPROVED'       THEN 1 ELSE 0 END) AS approved,
                        SUM(CASE WHEN state='CORRECTED'      THEN 1 ELSE 0 END) AS corrected,
                        SUM(CASE WHEN state='REJECTED'       THEN 1 ELSE 0 END) AS rejected,
                        AVG(overall_confidence)                           AS avg_conf
                    FROM review_queue
                    """
                ).fetchone()

        return QueueStats(
            total=row["total"] or 0,
            pending=row["pending"] or 0,
            approved=row["approved"] or 0,
            corrected=row["corrected"] or 0,
            rejected=row["rejected"] or 0,
            avg_confidence=row["avg_conf"] or 0.0,
        )

    # ── Active learning log (DB side) ─────────────────────────────────────────

    def save_al_entry(self, entry: Dict[str, Any]) -> None:
        """Persist an active learning entry to the DB (in addition to JSONL)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO active_learning_log
                    (entry_id, document_id, document_type, file_path,
                     ai_output, ground_truth, field_diffs, error_fields,
                     original_confidence_scores, bbox_corrections,
                     reviewer_id, logged_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        entry.get("entry_id", ""),
                        entry.get("document_id", ""),
                        entry.get("document_type", ""),
                        entry.get("file_path", ""),
                        self._j(entry.get("ai_output", {})),
                        self._j(entry.get("ground_truth", {})),
                        self._j(entry.get("field_diffs", [])),
                        self._j(entry.get("error_fields", [])),
                        self._j(entry.get("original_confidence_scores", {})),
                        self._j(entry.get("bbox_corrections", [])),
                        entry.get("reviewer_id", ""),
                        entry.get("logged_at", self._now()),
                    ),
                )

    def get_al_entries(
        self, doc_type: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Retrieve active learning entries, optionally filtered by doc type."""
        with self._lock:
            with self._connect() as conn:
                if doc_type:
                    rows = conn.execute(
                        "SELECT * FROM active_learning_log WHERE document_type=? LIMIT ?",
                        (doc_type, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM active_learning_log LIMIT ?", (limit,)
                    ).fetchall()
        return [dict(r) for r in rows]
