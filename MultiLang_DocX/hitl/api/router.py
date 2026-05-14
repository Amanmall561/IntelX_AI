"""
HITL REST API Router (Phase 5.2)
=================================
FastAPI router exposing all review-queue operations to the Reviewer UI
and any external integrations.

Endpoints:
  GET  /hitl/queue              — list PENDING_REVIEW items
  GET  /hitl/queue/{item_id}    — get single item
  POST /hitl/queue/{item_id}/claim    — claim for review
  POST /hitl/queue/{item_id}/approve  — approve (no corrections)
  POST /hitl/queue/{item_id}/correct  — submit field corrections
  POST /hitl/queue/{item_id}/reject   — reject document
  GET  /hitl/stats              — queue statistics
  GET  /hitl/active-learning/report   — pattern analysis report
  POST /hitl/active-learning/export   — export training dataset
  GET  /hitl/active-learning/suggest  — prompt tweak suggestions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from hitl_config import ACTIVE_LEARNING_LOG, AL_MIN_CORRECTIONS_FOR_REPORT
from hitl.active_learning import (
    ActiveLearningAnalyser,
    ActiveLearningLogger,
    PromptTweakAdvisor,
)
from hitl.models import (
    Correction,
    DocumentState,
    FieldCorrection,
    ReviewItem,
)
from hitl.queue_store import ReviewQueueStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hitl", tags=["hitl"])

# ── Singleton store (shared with orchestrator if wired) ───────────────────────

_store: Optional[ReviewQueueStore] = None
_al_logger: Optional[ActiveLearningLogger] = None
_analyser: Optional[ActiveLearningAnalyser] = None
_advisor: Optional[PromptTweakAdvisor] = None


def get_store() -> ReviewQueueStore:
    global _store
    if _store is None:
        _store = ReviewQueueStore()
    return _store


def get_al_logger() -> ActiveLearningLogger:
    global _al_logger, _store
    if _al_logger is None:
        _al_logger = ActiveLearningLogger(queue_store=get_store())
    return _al_logger


def get_analyser() -> ActiveLearningAnalyser:
    global _analyser
    if _analyser is None:
        _analyser = ActiveLearningAnalyser(get_al_logger())
    return _analyser


def get_advisor() -> PromptTweakAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = PromptTweakAdvisor()
    return _advisor


# ── Request / Response schemas ────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    reviewer_id: str = Field(..., description="Unique reviewer identifier")
    force: bool = Field(default=False, description="Whether to take over an existing claim")


class ApproveRequest(BaseModel):
    reviewer_id: str
    notes: str = ""


class RejectRequest(BaseModel):
    reviewer_id: str
    notes: str = ""


class FieldCorrectionSchema(BaseModel):
    field_path: str
    original_value: Any
    corrected_value: Any
    confidence_was: float = 0.0
    correction_note: str = ""


class CorrectRequest(BaseModel):
    reviewer_id: str
    document_type: Optional[str] = None
    field_corrections: List[FieldCorrectionSchema] = Field(default_factory=list)
    corrected_json: Dict[str, Any] = Field(default_factory=dict)
    bbox_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    reviewer_notes: str = ""


class ExportRequest(BaseModel):
    output_path: str = Field(default="/tmp/hitl_training_export.jsonl")
    doc_type: Optional[str] = None
    format: str = "jsonl"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/queue", summary="List pending review items")
async def list_queue(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    state: str = Query(default="PENDING_REVIEW"),
    reviewer_id: Optional[str] = Query(default=None),
    unclaimed: bool = Query(default=False),
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Phase 5.2: List queued review items, ordered by priority then age.
    """
    actual_state = None if state == "ALL" else state
    items = store.peek(
        limit=limit,
        offset=offset,
        state_filter=actual_state,
        reviewer_id=reviewer_id,
        unclaimed_only=unclaimed,
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "count": len(items),
        "offset": offset,
        "limit": limit,
        "state_filter": state,
        "reviewer_filter": reviewer_id,
        "unclaimed_filter": unclaimed,
    }


@router.get("/queue/{item_id}", summary="Get a single review item")
async def get_item(
    item_id: str,
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """Retrieve a specific ReviewItem and its corrections."""
    item = store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"ReviewItem {item_id} not found")
    corrections = store.get_corrections(item_id)
    return {
        "item": item.model_dump(mode="json"),
        "corrections": corrections,
    }


@router.post("/queue/{item_id}/claim", summary="Claim an item for review")
async def claim_item(
    item_id: str,
    body: ClaimRequest,
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Phase 5.2: Atomically assign this review item to a reviewer.
    Returns 409 if the item is already claimed by someone else.
    """
    item = store.claim(item_id, body.reviewer_id, force=body.force)
    if item is None:
        raise HTTPException(
            status_code=409,
            detail=f"Item {item_id} could not be claimed (already claimed or not found).",
        )
    status = "reassigned" if body.force else "claimed"
    return {"status": status, "item": item.model_dump(mode="json")}


@router.post("/queue/{item_id}/approve", summary="Approve AI extraction")
async def approve_item(
    item_id: str,
    body: ApproveRequest,
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Phase 5.1: Human reviewer confirms AI output is correct.
    Sets document state to APPROVED.
    """
    ok = store.approve(item_id, body.reviewer_id, body.notes)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found or not yours")
    logger.info("Item %s approved by %s", item_id, body.reviewer_id)
    return {"status": "approved", "item_id": item_id}


@router.post("/queue/{item_id}/correct", summary="Submit field corrections")
async def correct_item(
    item_id: str,
    body: CorrectRequest,
    store: ReviewQueueStore = Depends(get_store),
    al_logger: ActiveLearningLogger = Depends(get_al_logger),
) -> Dict[str, Any]:
    """
    Phase 5.3: Submit field-level corrections.
    Saves the correction, updates the item state to CORRECTED,
    and logs to the active learning JSONL file.
    """
    existing = store.get(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    # Build Correction model
    field_corrections = [
        FieldCorrection(
            field_path=fc.field_path,
            original_value=fc.original_value,
            corrected_value=fc.corrected_value,
            confidence_was=fc.confidence_was,
            correction_note=fc.correction_note,
        )
        for fc in body.field_corrections
    ]

    correction = Correction(
        item_id=item_id,
        document_id=existing.document_id,
        document_type=body.document_type or existing.document_type,
        reviewer_id=body.reviewer_id,
        field_corrections=field_corrections,
        corrected_json=body.corrected_json or existing.aggregated_json,
        bbox_corrections=body.bbox_corrections,
        reviewer_notes=body.reviewer_notes,
    )

    store.save_correction(correction)

    # Phase 5.3: Log to active learning
    try:
        al_logger.log_correction(
            correction=correction,
            original_ai_output=existing.aggregated_json,
            original_confidence_scores=existing.confidence_scores,
        )
    except Exception as e:
        logger.warning("AL logging failed for item %s: %s", item_id, e)

    logger.info(
        "Correction saved for item %s (%d field diffs) by %s",
        item_id,
        len(field_corrections),
        body.reviewer_id,
    )
    return {
        "status": "corrected",
        "item_id": item_id,
        "correction_id": correction.correction_id,
        "fields_corrected": len(field_corrections),
    }


@router.post("/queue/{item_id}/reject", summary="Reject document")
async def reject_item(
    item_id: str,
    body: RejectRequest,
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Phase 5.2: Reject a document (needs re-processing or re-scan).
    """
    ok = store.reject(item_id, body.reviewer_id, body.notes)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found or not yours")
    return {"status": "rejected", "item_id": item_id}


@router.delete("/queue/{item_id}", summary="Delete document permanently")
async def delete_item(
    item_id: str,
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Phase 5.2: Delete a document permanently from the review queue and corrections.
    """
    ok = store.delete(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return {"status": "deleted", "item_id": item_id}


@router.get("/stats", summary="Review queue statistics")
async def queue_stats(
    store: ReviewQueueStore = Depends(get_store),
) -> Dict[str, Any]:
    """Overall review queue health metrics."""
    stats = store.get_stats()
    return stats.to_dict()


@router.get("/active-learning/report", summary="Pattern analysis report")
async def al_report(
    doc_type: Optional[str] = Query(default=None),
    analyser: ActiveLearningAnalyser = Depends(get_analyser),
) -> Dict[str, Any]:
    """
    Phase 5.3: Analyse accumulated corrections to identify systematic errors.
    """
    return analyser.analyse_patterns(doc_type=doc_type)


@router.get("/active-learning/suggest", summary="Prompt tweak suggestions")
async def al_suggest(
    doc_type: Optional[str] = Query(default=None),
    analyser: ActiveLearningAnalyser = Depends(get_analyser),
    advisor: PromptTweakAdvisor = Depends(get_advisor),
) -> Dict[str, Any]:
    """
    Phase 5.3: Generate actionable prompt improvement suggestions based on
    the most frequently corrected fields.
    """
    report = analyser.analyse_patterns(doc_type=doc_type)
    return advisor.suggest_prompt_tweak(report)


@router.post("/active-learning/export", summary="Export training dataset")
async def al_export(
    body: ExportRequest,
    al_logger: ActiveLearningLogger = Depends(get_al_logger),
) -> Dict[str, Any]:
    """
    Phase 5.3: Export the full correction log as a fine-tuning dataset.
    """
    count = al_logger.export_training_data(
        output_path=body.output_path,
        doc_type=body.doc_type,
        format=body.format,
    )
    return {
        "status": "exported",
        "records_exported": count,
        "output_path": body.output_path,
        "format": body.format,
    }
