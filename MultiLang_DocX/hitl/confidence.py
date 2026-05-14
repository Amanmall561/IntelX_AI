"""
Confidence Gate & Score Aggregation (Phase 5.1 + Phase 3.3)
=============================================================
Evaluates aggregated pipeline output and decides whether a document
should be auto-approved (COMPLETED) or routed to human review (PENDING_REVIEW).

Confidence scoring strategy (no changes to existing extraction models):
  - If the extraction model returned per-field confidence_scores, use them directly.
  - Otherwise, fall back to heuristic scoring:
      * Field presence score   (is the field non-empty?)            → 0.70 base
      * Field type score       (correct type, e.g. list vs string?) → +0.10 bonus
      * Identifier format score (regex check for known patterns)    → +0.15 bonus
      * Text plausibility      (no gibberish / numeric noise)        → +0.05 bonus
  Maximum heuristic score: 1.0
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from hitl_config import (
    CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_FIELD_THRESHOLD,
    REQUIRED_FIELDS_BY_DOC_TYPE,
)
from hitl.models import GateOutcome, GateResult


# ── Known identifier format patterns ─────────────────────────────────────────

_IDENTIFIER_PATTERNS: dict[str, re.Pattern] = {
    "aadhaar":   re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$"),
    "pan":        re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
    "passport":   re.compile(r"^[A-Z]\d{7}$"),
    "dl":         re.compile(r"^[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}$"),
    "voter_id":   re.compile(r"^[A-Z]{3}\d{7}$"),
    "account_no": re.compile(r"^\d{9,18}$"),
    "ifsc":       re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$"),
    "pincode":    re.compile(r"^\d{6}$"),
}


# ── Heuristic per-field confidence scoring ────────────────────────────────────

def _score_value(field_name: str, value: Any) -> float:
    """
    Return a heuristic confidence score (0.0–1.0) for a single extracted value.
    This replaces LLM-level confidence when the model doesn't output scores.
    """
    if value is None:
        return 0.0
    if isinstance(value, list):
        if len(value) == 0:
            return 0.0
        # Score each item and average
        return round(sum(_score_value(field_name, v) for v in value) / len(value), 4)
    if isinstance(value, dict):
        if not value:
            return 0.0
        scores = [_score_value(k, v) for k, v in value.items()]
        return round(sum(scores) / len(scores), 4)

    text = str(value).strip()
    if not text:
        return 0.0

    score = 0.70  # Base: field is present

    # Bonus: no suspicious garbage characters
    gibberish_ratio = sum(1 for c in text if unicodedata.category(c) in ("Cs", "Cn")) / max(len(text), 1)
    if gibberish_ratio < 0.05:
        score += 0.05

    # Bonus: identifier format check
    fkey = field_name.lower().replace(" ", "_").replace("-", "_")
    for id_type, pattern in _IDENTIFIER_PATTERNS.items():
        if id_type in fkey and pattern.match(text.upper().replace(" ", "")):
            score += 0.15
            break

    # Bonus: reasonable text length (not a single stray character)
    if len(text) > 2:
        score += 0.10

    return min(round(score, 4), 1.0)


def compute_field_confidence_scores(
    extracted_json: Dict[str, Any],
    existing_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Return per-field confidence scores for every leaf value in extracted_json.
    If the model already provided scores in existing_scores, those take priority.
    Otherwise falls back to heuristic _score_value().

    Returns a flat dict keyed by dot-notation field paths:
        e.g. {"people[0].name": 0.92, "identifiers[0]": 0.83}
    """
    existing = existing_scores or {}
    result: Dict[str, float] = {}

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                _walk(v, path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                path = f"{prefix}[{i}]"
                _walk(item, path)
        else:
            # Leaf node
            if prefix in existing:
                result[prefix] = float(existing[prefix])
            else:
                # Derive field name from last segment of path
                field_name = re.split(r"[.\[\]]+", prefix)[-1] or prefix
                result[prefix] = _score_value(field_name, obj)

    _walk(extracted_json, "")
    return result


def aggregate_confidence(
    field_scores: Dict[str, float],
    doc_type: str = "Unknown",
) -> float:
    """
    Compute overall document confidence as the weighted mean of field scores,
    giving higher weight to required-field paths.

    Phase 3.3 + 5.1.
    """
    if not field_scores:
        return 0.0

    required = set(REQUIRED_FIELDS_BY_DOC_TYPE.get(doc_type, []))
    weighted_scores: List[Tuple[float, float]] = []  # (score, weight)

    for path, score in field_scores.items():
        top_level = path.split(".")[0].rstrip("]").rstrip("[0123456789")
        top_level = re.split(r"[\[\].]", path)[0]
        weight = 2.0 if top_level in required else 1.0
        weighted_scores.append((score, weight))

    total_weight = sum(w for _, w in weighted_scores)
    if total_weight == 0:
        return 0.0

    return round(
        sum(s * w for s, w in weighted_scores) / total_weight, 4
    )


# ── Field coverage check ──────────────────────────────────────────────────────

def evaluate_field_coverage(
    extracted_json: Dict[str, Any],
    doc_type: str,
) -> Tuple[List[str], List[str]]:
    """
    Returns (present_required, missing_required) lists.
    Checks only the top-level required fields defined in config.
    """
    required = REQUIRED_FIELDS_BY_DOC_TYPE.get(doc_type, [])
    missing: List[str] = []
    present: List[str] = []

    for field in required:
        val = extracted_json.get(field)
        if val is None or val == "" or val == [] or val == {}:
            missing.append(field)
        else:
            present.append(field)

    return present, missing


# ── Low-confidence field detection ────────────────────────────────────────────

def detect_flagged_fields(
    field_scores: Dict[str, float],
    threshold: float = LOW_CONFIDENCE_FIELD_THRESHOLD,
) -> List[str]:
    """Return list of field paths whose confidence is below the amber threshold."""
    return [path for path, score in field_scores.items() if score < threshold]


# ── The Confidence Gate (Phase 5.1) ──────────────────────────────────────────

def confidence_gate(
    extracted_json: Dict[str, Any],
    doc_type: str = "Unknown",
    existing_scores: Optional[Dict[str, float]] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> GateResult:
    """
    Core confidence gate evaluation.

    Returns GateResult with:
      - PASS    → all required fields present AND overall_confidence >= threshold
      - PARTIAL → required fields present but confidence < threshold
      - FAIL    → required fields missing OR json is empty / None
    """
    if not extracted_json:
        return GateResult(
            outcome=GateOutcome.FAIL,
            overall_confidence=0.0,
            threshold_used=threshold,
            missing_required_fields=REQUIRED_FIELDS_BY_DOC_TYPE.get(doc_type, []),
            reasoning="Extracted JSON is empty or None.",
        )

    # 1. Per-field scores
    field_scores = compute_field_confidence_scores(extracted_json, existing_scores)

    # 2. Overall confidence
    overall = aggregate_confidence(field_scores, doc_type)

    # 3. Required field coverage
    _, missing = evaluate_field_coverage(extracted_json, doc_type)

    # 4. Flagged fields (below amber threshold)
    flagged = detect_flagged_fields(field_scores)

    # 5. Decide gate outcome
    if missing:
        outcome = GateOutcome.FAIL
        reasoning = (
            f"Required fields missing: {missing}. "
            f"Overall confidence: {overall:.2%}."
        )
    elif overall >= threshold:
        outcome = GateOutcome.PASS
        reasoning = (
            f"All required fields present. "
            f"Overall confidence {overall:.2%} ≥ threshold {threshold:.2%}."
        )
    else:
        outcome = GateOutcome.PARTIAL
        reasoning = (
            f"All required fields present but confidence {overall:.2%} "
            f"< threshold {threshold:.2%}. "
            f"Low-confidence fields: {flagged[:5]}{'...' if len(flagged) > 5 else ''}."
        )

    return GateResult(
        outcome=outcome,
        overall_confidence=overall,
        threshold_used=threshold,
        flagged_fields=flagged,
        missing_required_fields=missing,
        reasoning=reasoning,
    )
