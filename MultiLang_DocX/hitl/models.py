"""
HITL Pydantic Data Models
==========================
Central type definitions shared across every HITL component.

Major/minor plan mapping:
  - DocumentState  → Phase 5.1 state machine (COMPLETED / PENDING_REVIEW)
  - BatchChunk     → Phase 1.2 chunk identity (Batch_ID, Document_ID)
  - ReviewItem     → Phase 5.2 review queue entry
  - Correction     → Phase 5.3 active learning diff
  - ActiveLearningEntry → Phase 5.3 training log record
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Phase 5.1 — Document State Machine ───────────────────────────────────────

class DocumentState(str, Enum):
    """All possible lifecycle states of a document in the HITL pipeline."""
    PENDING     = "PENDING"          # File received, not yet processed
    PROCESSING  = "PROCESSING"       # AI pipeline running
    COMPLETED   = "COMPLETED"        # Confidence ≥ threshold; auto-approved
    PENDING_REVIEW = "PENDING_REVIEW"# Confidence < threshold; in review queue
    APPROVED    = "APPROVED"         # Human reviewer approved AI output
    CORRECTED   = "CORRECTED"        # Human reviewer submitted corrections
    REJECTED    = "REJECTED"         # Document rejected; needs re-processing


# ── Phase 1.2 — Chunk Identity ────────────────────────────────────────────────

class BatchChunk(BaseModel):
    """Represents a single page-batch produced by the Map Splitter (Phase 1.2)."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_range: List[int]                      # e.g. [1, 5]
    classification: Optional[str] = None      # e.g. "Bank Statement" (Phase 3.1)
    extracted_json: Dict[str, Any] = Field(default_factory=dict)

    # Phase 3.3 — per-field confidence scores keyed by field path
    # e.g. {"people[0].name": 0.92, "identifiers[0]": 0.65}
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Script profiling result (Phase 1.3)
    detected_scripts: List[str] = Field(default_factory=list)   # e.g. ["latin", "devanagari"]
    has_handwritten: bool = False

    # Aggregated mean confidence for this chunk
    chunk_confidence: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Phase 4.2 / 4.3 — Aggregated Document Profile ────────────────────────────

class AggregatedDocument(BaseModel):
    """Merged, de-duplicated document profile produced by the Reduce stage."""
    document_id: str
    file_path: str
    document_type: str = "Unknown"

    # De-duplicated, merged JSON payload (Phase 4.1 + 4.2)
    aggregated_json: Dict[str, Any] = Field(default_factory=dict)

    # Per-field confidence scores after merging (Phase 3.3)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Overall confidence (0.0 – 1.0)
    overall_confidence: float = 0.0

    # Multilingual mappings for native-script entities (Phase 4.3)
    multilingual_mappings: List[Dict[str, str]] = Field(default_factory=list)

    # Source chunks that contributed to this document
    source_chunk_ids: List[str] = Field(default_factory=list)

    # Relationships resolved across chunks (Phase 4.2)
    cross_chunk_relationships: List[Dict[str, Any]] = Field(default_factory=list)

    state: DocumentState = DocumentState.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


# ── Phase 5.2 — Review Queue Entry ───────────────────────────────────────────

class ReviewItem(BaseModel):
    """A document queued for human review."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    file_path: str
    document_type: str = "Unknown"

    # The aggregated JSON that needs review
    aggregated_json: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = 0.0

    # Fields that failed the confidence gate
    flagged_fields: List[str] = Field(default_factory=list)

    # Which required fields are missing
    missing_required_fields: List[str] = Field(default_factory=list)

    state: DocumentState = DocumentState.PENDING_REVIEW

    # Reviewer assignment
    reviewer_id: Optional[str] = None
    claimed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Reviewer notes
    reviewer_notes: str = ""

    # Queue metadata
    queued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 5   # 1 = highest, 10 = lowest

    # Page thumbnail paths for the reviewer UI
    page_thumbnails: List[str] = Field(default_factory=list)

    # Multilingual mapping results (Phase 4.3)
    multilingual_mappings: List[Dict[str, str]] = Field(default_factory=list)


# ── Phase 5.3 — Field-Level Correction ───────────────────────────────────────

class FieldCorrection(BaseModel):
    """A single field-level correction made by a human reviewer."""
    field_path: str         # e.g. "people[0].name"
    original_value: Any     # AI-extracted value
    corrected_value: Any    # Human-corrected value
    confidence_was: float = 0.0
    correction_note: str = ""


class Correction(BaseModel):
    """Full correction record for a review item."""
    correction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_id: str
    document_id: str
    document_type: str
    reviewer_id: str

    # Individual field corrections
    field_corrections: List[FieldCorrection] = Field(default_factory=list)

    # Final corrected JSON (full document)
    corrected_json: Dict[str, Any] = Field(default_factory=dict)

    # Bounding-box corrections for future YOLO fine-tuning (Phase 5.3)
    bbox_corrections: List[Dict[str, Any]] = Field(default_factory=list)

    reviewer_notes: str = ""
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Phase 5.3 — Active Learning Training Entry ───────────────────────────────

class ActiveLearningEntry(BaseModel):
    """Serialized correction saved to the JSONL training log."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    document_type: str
    file_path: str

    # The original AI-extracted JSON
    ai_output: Dict[str, Any] = Field(default_factory=dict)

    # The human-corrected JSON
    ground_truth: Dict[str, Any] = Field(default_factory=dict)

    # Individual field diffs for targeted fine-tuning
    field_diffs: List[FieldCorrection] = Field(default_factory=list)

    # Which fields were consistently wrong (pattern tracking)
    error_fields: List[str] = Field(default_factory=list)

    # Confidence scores at time of extraction
    original_confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Optional bounding box corrections for YOLO fine-tuning
    bbox_corrections: List[Dict[str, Any]] = Field(default_factory=list)

    reviewer_id: str = ""
    logged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Gate result ───────────────────────────────────────────────────────────────

class GateOutcome(str, Enum):
    PASS    = "PASS"     # All required fields present AND confidence >= threshold
    PARTIAL = "PARTIAL"  # Some fields present but low confidence
    FAIL    = "FAIL"     # Missing required fields OR classification failure


class GateResult(BaseModel):
    """Confidence gate evaluation result (Phase 5.1)."""
    outcome: GateOutcome
    overall_confidence: float
    threshold_used: float
    flagged_fields: List[str] = Field(default_factory=list)
    missing_required_fields: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ── HITL Orchestrator final result ────────────────────────────────────────────

class HITLResult(BaseModel):
    """Return type of HITLOrchestrator.process()."""
    document_id: str
    file_path: str
    state: DocumentState
    gate_result: GateResult
    aggregated_json: Dict[str, Any] = Field(default_factory=dict)
    multilingual_mappings: List[Dict[str, str]] = Field(default_factory=list)

    # Set when routed to review queue
    review_item_id: Optional[str] = None

    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
