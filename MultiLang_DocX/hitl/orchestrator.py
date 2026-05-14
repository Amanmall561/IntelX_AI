"""
HITL Orchestrator — Master Entry Point (Phase 4.1 → 5.2)
=========================================================
HITLOrchestrator.process() is the single function you call after the
IntelX_AI pipeline returns its extracted chunks.  It implements the
full Reduce + Validation pipeline:

  Phase 4.1  JSON Compilation         (collect chunks)
  Phase 4.2  Entity Deduplication     (EntityDeduplicator.merge_chunks)
  Phase 4.3  Multilingual Key Mapping (MultilingualMapper.process_document)
  Phase 5.1  Confidence Gate          (confidence_gate)
  Phase 5.2  Review Queue Routing     (ReviewQueueStore.push if FAIL/PARTIAL)

Usage:
    from hitl.orchestrator import HITLOrchestrator

    orchestrator = HITLOrchestrator()
    result = orchestrator.process(
        pipeline_output=pipeline_main(file_path),   # list[dict] from moduler_call.main()
        document_id="doc-uuid",
        file_path="/path/to/original/file.pdf",
    )
    print(result.state)     # DocumentState.COMPLETED or PENDING_REVIEW
    print(result.review_item_id)  # set if routed to review queue
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hitl_config import CONFIDENCE_THRESHOLD, KNOWN_DOC_TYPES
from hitl.confidence import (
    aggregate_confidence,
    compute_field_confidence_scores,
    confidence_gate,
)
from hitl.deduplication import EntityDeduplicator
from hitl.models import (
    AggregatedDocument,
    BatchChunk,
    DocumentState,
    GateOutcome,
    GateResult,
    HITLResult,
    ReviewItem,
)
from hitl.multilingual_mapper import MultilingualMapper
from hitl.queue_store import ReviewQueueStore
from hitl.utils.json_repair import repair_and_parse

logger = logging.getLogger("hitl.orchestrator")


class HITLOrchestrator:
    """
    Central coordinator for the HITL validation loop.

    Args:
        queue_store:  A ReviewQueueStore instance (auto-created if None).
        threshold:    Confidence threshold (default from config.py).
    """

    def __init__(
        self,
        queue_store: Optional[ReviewQueueStore] = None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._queue_store = queue_store or ReviewQueueStore()
        self._dedup = EntityDeduplicator()
        self._mapper = MultilingualMapper()
        self._threshold = threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process(
        self,
        pipeline_output: Any,
        document_id: Optional[str] = None,
        file_path: str = "",
        page_thumbnails: Optional[List[str]] = None,
    ) -> HITLResult:
        """
        Main entry point.  Accepts the raw output from moduler_call.main()
        (which can be a list[dict], a single dict, or None/error).

        Returns a HITLResult with the document's final state.
        """
        doc_id = document_id or str(uuid.uuid4())

        # ── Step 1: Normalise pipeline output ────────────────────────────────
        chunks = self._normalise_pipeline_output(pipeline_output)
        if not chunks:
            logger.warning("Empty pipeline output for document %s", doc_id)
            # Return a FAIL gate result immediately (uses module-level GateResult/GateOutcome)
            gate = GateResult(
                outcome=GateOutcome.FAIL,
                overall_confidence=0.0,
                threshold_used=self._threshold,
                reasoning="Pipeline returned no extractable data.",
            )
            return HITLResult(
                document_id=doc_id,
                file_path=file_path,
                state=DocumentState.PENDING_REVIEW,
                gate_result=gate,
            )

        # ── Step 2 (Phase 4.2): Deduplicate & merge chunks (Reduce stage) ───
        raw_json_chunks = [c.get("extracted_data", c) for c in chunks]
        merged_json = self._dedup.merge_chunks(raw_json_chunks)

        # Merge per-chunk confidence scores
        chunk_score_lists = [
            c.get("confidence_scores", {}) for c in chunks
            if isinstance(c.get("confidence_scores"), dict)
        ]
        merged_scores = self._dedup.merge_confidence_scores(chunk_score_lists)

        # ── Step 3 (Phase 4.3): Multilingual mapping ─────────────────────────
        _, multilingual_mappings = self._mapper.enrich_document(merged_json)

        # ── Step 4 (Phase 5.1): Detect document type ─────────────────────────
        doc_type = self._detect_doc_type(chunks, merged_json)
        merged_json["document_type"] = merged_json.get("document_type") or doc_type

        # ── Step 5 (Phase 5.1): Run confidence gate ───────────────────────────
        gate_result = confidence_gate(
            extracted_json=merged_json,
            doc_type=doc_type,
            existing_scores=merged_scores if merged_scores else None,
            threshold=self._threshold,
        )

        logger.info(
            "Document %s | type=%s | gate=%s | confidence=%.2f%%",
            doc_id,
            doc_type,
            gate_result.outcome.value,
            gate_result.overall_confidence * 100,
        )

        # ── Step 6 (Phase 5.1 / 5.2): Route based on gate outcome ─────────────
        if gate_result.outcome == GateOutcome.PASS:
            # Auto-approved — mark COMPLETED and save to db so it shows in Queue Dashboard!
            review_item = ReviewItem(
                document_id=doc_id,
                file_path=file_path,
                document_type=doc_type,
                aggregated_json=merged_json,
                confidence_scores=merged_scores if merged_scores else {},
                overall_confidence=gate_result.overall_confidence,
                flagged_fields=[],
                missing_required_fields=[],
                state=DocumentState.COMPLETED,
                page_thumbnails=page_thumbnails or [],
                multilingual_mappings=multilingual_mappings,
                priority=10,
            )
            logger.info("Auto-approved document %s (PASS) - Persisting to review_queue", doc_id)
            self._queue_store.push(review_item)

            return HITLResult(
                document_id=doc_id,
                file_path=file_path,
                state=DocumentState.COMPLETED,
                gate_result=gate_result,
                aggregated_json=merged_json,
                multilingual_mappings=multilingual_mappings,
                review_item_id=review_item.item_id,
            )
        else:
            # FAIL or PARTIAL → route to review queue
            review_item = ReviewItem(
                document_id=doc_id,
                file_path=file_path,
                document_type=doc_type,
                aggregated_json=merged_json,
                confidence_scores=gate_result.flagged_fields and merged_scores or {},
                overall_confidence=gate_result.overall_confidence,
                flagged_fields=gate_result.flagged_fields,
                missing_required_fields=gate_result.missing_required_fields,
                state=DocumentState.PENDING_REVIEW,
                page_thumbnails=page_thumbnails or [],
                multilingual_mappings=multilingual_mappings,
                # Higher priority for classification failures (FAIL)
                priority=2 if gate_result.outcome == GateOutcome.FAIL else 5,
            )
            self._queue_store.push(review_item)

            return HITLResult(
                document_id=doc_id,
                file_path=file_path,
                state=DocumentState.PENDING_REVIEW,
                gate_result=gate_result,
                aggregated_json=merged_json,
                multilingual_mappings=multilingual_mappings,
                review_item_id=review_item.item_id,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_pipeline_output(
        self, pipeline_output: Any
    ) -> List[Dict[str, Any]]:
        """
        The existing pipeline can return:
          - list[dict]      (most common: one dict per chunk/page)
          - dict            (single-page or single-document result)
          - None / error    (extraction failure)
        
        Normalise to list[dict] where each item is a chunk.
        """
        if pipeline_output is None:
            return []
        
        # Handle case where output is a single string (not wrapped in dict or list)
        if isinstance(pipeline_output, str):
            repaired = repair_and_parse(pipeline_output)
            return [repaired] if repaired else []

        if isinstance(pipeline_output, dict):
            if "error" in pipeline_output:
                logger.warning("Pipeline returned error: %s", pipeline_output.get("error"))
                return []
            return [pipeline_output]

        if isinstance(pipeline_output, list):
            # Filter out error entries and repair string entries
            valid = []
            for item in pipeline_output:
                if isinstance(item, dict):
                    if "error" not in item:
                        valid.append(item)
                    else:
                        logger.warning("Skipping error chunk: %s", item.get("error"))
                elif isinstance(item, str):
                    repaired = repair_and_parse(item)
                    if repaired:
                        valid.append(repaired)
            return valid
        return []

    def _detect_doc_type(
        self,
        chunks: List[Dict[str, Any]],
        merged_json: Dict[str, Any],
    ) -> str:
        """
        Phase 3.1 fallback: detect document type from the merged JSON or chunks.
        Priority order:
          1. Explicit 'document_type' already set in merged_json (that isn't generic)
          2. 'document_type' from chunk metadata
          3. Specific high-value keys from extracted data (e.g. 'form_number', 'plan_name')
          4. Fallback to generic "Other" or "Unknown"
        """
        # 1. Check merged output
        dt = merged_json.get("document_type", "")
        if dt and dt not in ("Unknown", "Other"):
            return str(dt)

        # 2. Check individual chunks
        best_dt = ""
        for chunk in chunks:
            extracted = chunk.get("extracted_data", {})
            if isinstance(extracted, dict):
                cdt = extracted.get("document_type") or extracted.get("Document Type", "")
                if cdt and str(cdt) not in ("Unknown", "Other"):
                    best_dt = str(cdt)
                    break
            cdt = chunk.get("document_type", "") or chunk.get("doc_type", "")
            if cdt and str(cdt) not in ("Unknown", "Other"):
                best_dt = str(cdt)
                break

        if best_dt:
            return best_dt

        # 3. If it's still 'Other' or 'Unknown', try to extract the literal form/document name
        for key in ["form", "form_number", "form_name", "plan_name", "certificate_name", "document_name", "title"]:
            val = merged_json.get(key)
            if val and isinstance(val, str) and len(val.strip()) >= 2:
                # E.g., if the user extracts {"form": "5500-EZ"}, use "5500-EZ" as the doc type
                return val.strip()

        return dt if dt else "Unknown"

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: process a single already-merged dict
    # ──────────────────────────────────────────────────────────────────────────

    def process_merged(
        self,
        merged_json: Dict[str, Any],
        document_id: Optional[str] = None,
        file_path: str = "",
        doc_type: str = "Unknown",
        existing_scores: Optional[Dict[str, float]] = None,
        page_thumbnails: Optional[List[str]] = None,
    ) -> HITLResult:
        """
        Alternative entry point when the caller has already merged chunks.
        Useful for testing individual components in isolation.
        """
        doc_id = document_id or str(uuid.uuid4())

        _, multilingual_mappings = self._mapper.enrich_document(merged_json)

        if not doc_type or doc_type == "Unknown":
            doc_type = merged_json.get("document_type", "Unknown")

        gate_result = confidence_gate(
            extracted_json=merged_json,
            doc_type=doc_type,
            existing_scores=existing_scores,
            threshold=self._threshold,
        )

        if gate_result.outcome == GateOutcome.PASS:
            # Sync: Also push COMPLETED results to the DB in the merged path
            review_item = ReviewItem(
                document_id=doc_id,
                file_path=file_path,
                document_type=doc_type,
                aggregated_json=merged_json,
                confidence_scores=existing_scores or {},
                overall_confidence=gate_result.overall_confidence,
                flagged_fields=[],
                missing_required_fields=[],
                state=DocumentState.COMPLETED,
                page_thumbnails=page_thumbnails or [],
                multilingual_mappings=multilingual_mappings,
                priority=10,
            )
            logger.info("Auto-approved document %s (PASS/merged) - Persisting to review_queue", doc_id)
            self._queue_store.push(review_item)

            return HITLResult(
                document_id=doc_id,
                file_path=file_path,
                state=DocumentState.COMPLETED,
                gate_result=gate_result,
                aggregated_json=merged_json,
                multilingual_mappings=multilingual_mappings,
                review_item_id=review_item.item_id,
            )

        review_item = ReviewItem(
            document_id=doc_id,
            file_path=file_path,
            document_type=doc_type,
            aggregated_json=merged_json,
            confidence_scores=existing_scores or {},
            overall_confidence=gate_result.overall_confidence,
            flagged_fields=gate_result.flagged_fields,
            missing_required_fields=gate_result.missing_required_fields,
            state=DocumentState.PENDING_REVIEW,
            page_thumbnails=page_thumbnails or [],
            multilingual_mappings=multilingual_mappings,
            priority=2 if gate_result.outcome == GateOutcome.FAIL else 5,
        )
        self._queue_store.push(review_item)

        return HITLResult(
            document_id=doc_id,
            file_path=file_path,
            state=DocumentState.PENDING_REVIEW,
            gate_result=gate_result,
            aggregated_json=merged_json,
            multilingual_mappings=multilingual_mappings,
            review_item_id=review_item.item_id,
        )
