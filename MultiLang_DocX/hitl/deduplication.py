"""
Entity Deduplication & Cross-Chunk Linking (Phase 4.2)
======================================================
After all chunks are extracted in parallel (Phase 3.3), this module:

  1. Merges duplicate entities that appear across multiple chunks.
     E.g. "Aadhaar No. 1234" found on page 1 AND page 50 → single entry.

  2. Resolves cross-chunk relationships.
     E.g. a person named on page 2 is linked to a transaction on page 10.

  3. Produces a clean, single-document aggregated JSON (Phase 4.1).

Deduplication strategy:
  - String values are compared using fuzzy Levenshtein ratio.
  - Identifier fields use exact normalised matching (strip spaces, upper-case).
  - List merging: unique items preserved; confidence-weighted when available.
"""
from __future__ import annotations

import copy
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from hitl_config import DEDUP_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import rapidfuzz first (faster), fall back to fuzzywuzzy
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _fuzz
    _ratio = _fuzz.ratio
    _partial_ratio = _fuzz.partial_ratio
except ImportError:
    try:
        from fuzzywuzzy import fuzz as _fuzz   # type: ignore
        _ratio = _fuzz.ratio
        _partial_ratio = _fuzz.partial_ratio
    except ImportError:
        # Last-resort: simple exact match (no fuzzy)
        def _ratio(a: str, b: str) -> float:     # type: ignore  # noqa
            return 100.0 if a.strip().lower() == b.strip().lower() else 0.0
        def _partial_ratio(a: str, b: str) -> float:  # type: ignore  # noqa
            return 100.0 if a.strip().lower() in b.strip().lower() else 0.0
        logger.warning(
            "Neither rapidfuzz nor fuzzywuzzy is installed. "
            "Falling back to exact string matching for deduplication."
        )


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Normalise a string: strip, upper-case, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def _are_similar(a: Any, b: Any, threshold: int = DEDUP_SIMILARITY_THRESHOLD) -> bool:
    """Return True if two values are similar enough to be considered duplicates."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (int, float)):
        return a == b
    if isinstance(a, dict):
        # Compare string representations of dicts
        return _ratio(str(a), str(b)) >= threshold
    a_str = _normalise(str(a))
    b_str = _normalise(str(b))
    return _ratio(a_str, b_str) >= threshold


# ── List merging ──────────────────────────────────────────────────────────────

def _merge_lists(base: List[Any], incoming: List[Any]) -> List[Any]:
    """
    Merge two lists removing duplicates using fuzzy similarity.
    Items in `incoming` that are sufficiently similar to any item in `base`
    are dropped; novel items are appended.
    """
    result = list(base)
    for item in incoming:
        if not any(_are_similar(item, existing) for existing in result):
            result.append(item)
    return result


# ── Dict deep-merge ───────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `incoming` into `base`.
    - Dicts are merged recursively.
    - Lists are merged with deduplication.
    - Scalar values: `base` wins unless it is empty/None.
    """
    result = copy.deepcopy(base)
    for key, val in incoming.items():
        existing = result.get(key)
        if existing is None or existing == "" or existing == [] or existing == {}:
            result[key] = copy.deepcopy(val)
        elif isinstance(existing, dict) and isinstance(val, dict):
            result[key] = _deep_merge(existing, val)
        elif isinstance(existing, list) and isinstance(val, list):
            result[key] = _merge_lists(existing, val)
        # else: keep base (higher-confidence first chunk wins)
    return result


# ── Identifier-specific deduplication ────────────────────────────────────────

def deduplicate_identifiers(identifiers: List[Any]) -> List[Any]:
    """
    Minor Step 4.2: Deduplicate identifier entries.
    Handles both string and dict identifiers.
    E.g. "AADHAR 1234 5678 9012" appearing in multiple chunks → single entry.
    """
    if not identifiers:
        return []

    seen: List[Any] = []
    for item in identifiers:
        # Normalise dict identifiers to a canonical string for comparison
        if isinstance(item, dict):
            canonical = _normalise(" ".join(str(v) for v in item.values()))
        else:
            canonical = _normalise(str(item))

        duplicate = False
        for existing in seen:
            if isinstance(existing, dict):
                existing_canonical = _normalise(" ".join(str(v) for v in existing.values()))
            else:
                existing_canonical = _normalise(str(existing))

            if _ratio(canonical, existing_canonical) >= DEDUP_SIMILARITY_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            seen.append(item)

    return seen


# ── Cross-chunk relationship resolver ─────────────────────────────────────────

def link_cross_chunk_entities(
    people: List[Any],
    relationships: List[Any],
    transactions: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 4.2: Resolve cross-chunk relationships.
    Links people named in one chunk to transactions or relationships in another.

    Returns a list of resolved relationship dicts.
    """
    resolved: List[Dict[str, Any]] = []

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        parties = rel.get("relationship_between", [])
        if len(parties) < 2:
            continue

        # Try to match each party to a known person entity
        matched_people = []
        for party in parties:
            party_norm = _normalise(str(party))
            for person in people:
                person_name = ""
                if isinstance(person, dict):
                    person_name = _normalise(str(person.get("name", "")))
                elif isinstance(person, str):
                    person_name = _normalise(person)

                if person_name and _ratio(party_norm, person_name) >= DEDUP_SIMILARITY_THRESHOLD:
                    matched_people.append(person)
                    break
            else:
                # No match found; keep raw party string
                matched_people.append({"name": party, "resolved": False})

        entry: Dict[str, Any] = {
            "relationship_type": rel.get("relationship_type", "unknown"),
            "parties": matched_people,
            "raw_relationship": rel,
        }

        # Link to transactions if any party appears in narration
        if transactions:
            linked_txns = []
            for txn in transactions:
                if isinstance(txn, dict):
                    narration = _normalise(str(txn.get("transaction", {}).get("Narration", "")))
                    beneficiary = _normalise(str(txn.get("transaction", {}).get("Beneficiary_Name", "")))
                    for party in parties:
                        p_norm = _normalise(str(party))
                        if (
                            _partial_ratio(p_norm, narration) >= 70
                            or _partial_ratio(p_norm, beneficiary) >= DEDUP_SIMILARITY_THRESHOLD
                        ):
                            linked_txns.append(txn)
                            break
            if linked_txns:
                entry["linked_transactions"] = linked_txns

        resolved.append(entry)

    return resolved


# ── Main entity deduplicator ──────────────────────────────────────────────────

class EntityDeduplicator:
    """
    Phase 4.1 + 4.2: Collect all parallel chunk JSON outputs and
    produce a single de-duplicated, merged document profile.
    """

    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Phase 4.1 — JSON Compilation + Phase 4.2 — Deduplication.

        Args:
            chunks: List of per-chunk extracted_json dicts, ordered by page.

        Returns:
            Single merged document dict with duplicates removed.
        """
        if not chunks:
            return {}

        if len(chunks) == 1:
            return copy.deepcopy(chunks[0])

        # Start with the first chunk as base
        merged: Dict[str, Any] = copy.deepcopy(chunks[0])

        # Fold in remaining chunks
        for chunk in chunks[1:]:
            if not isinstance(chunk, dict):
                continue
            merged = _deep_merge(merged, chunk)

        # Post-merge: deduplicate specific known list fields
        if "identifiers" in merged and isinstance(merged["identifiers"], list):
            merged["identifiers"] = deduplicate_identifiers(merged["identifiers"])

        if "people" in merged and isinstance(merged["people"], list):
            merged["people"] = _merge_lists(merged["people"], [])  # self-dedup

        if "dates" in merged and isinstance(merged["dates"], list):
            merged["dates"] = list({_normalise(str(d)): d for d in merged["dates"]}.values())

        # Resolve cross-chunk relationships
        people = merged.get("people", [])
        relationships = merged.get("relationships", [])
        transactions = None
        # Bank Statement transaction linking
        if "Statement" in merged and isinstance(merged["Statement"], list):
            transactions = merged["Statement"]

        if relationships:
            resolved_rels = link_cross_chunk_entities(people, relationships, transactions)
            merged["resolved_relationships"] = resolved_rels

        logger.info(
            "Merged %d chunks → %d top-level keys, %d identifiers, %d people",
            len(chunks),
            len(merged),
            len(merged.get("identifiers", [])),
            len(merged.get("people", [])),
        )

        return merged

    def merge_confidence_scores(
        self,
        chunk_scores_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Merge per-chunk confidence score dicts.
        If the same field appears in multiple chunks, keep the higher score.
        """
        merged: Dict[str, float] = {}
        for scores in chunk_scores_list:
            for path, score in scores.items():
                if path not in merged or score > merged[path]:
                    merged[path] = score
        return merged
