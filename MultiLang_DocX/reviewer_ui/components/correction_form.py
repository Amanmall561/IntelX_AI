"""
Correction Form Component (Phase 5.3)
======================================
Renders an interactive form for human reviewers to:
  1. Edit individual field values in the extracted JSON
  2. Add per-field correction notes and see confidence scores
  3. Submit corrections via the HITL REST API

Diff viewer: shows AI value vs corrected value side-by-side.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

from hitl_config import REVIEWER_API_PORT


API_BASE = f"http://localhost:{REVIEWER_API_PORT}"


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict to dot-notation keys for the correction form."""
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(_flatten(v, path))
            else:
                items[path] = v
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                items.update(_flatten(item, path))
            else:
                items[path] = item
    return items


def _diff_color(original: Any, corrected: Any) -> str:
    """Return a CSS colour string based on whether the value changed."""
    return "#f85149" if str(original) != str(corrected) else "#3fb950"


def render_correction_form(
    item: Dict[str, Any],
    reviewer_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Render the full correction form for a ReviewItem.
    Returns the submitted correction payload or None if not submitted yet.

    Phase 5.3: The form generates field_corrections (list of diffs) and
    corrected_json (the full corrected document).
    """
    aggregated_json: Dict[str, Any] = item.get("aggregated_json", {})
    confidence_scores: Dict[str, float] = item.get("confidence_scores", {})
    flagged_fields: List[str] = item.get("flagged_fields", [])

    st.markdown(
        "<p class='section-header'>Field-Level Corrections (Phase 5.3)</p>",
        unsafe_allow_html=True,
    )

    # Flatten the JSON for editing
    flat = _flatten(aggregated_json)

    if not flat:
        st.info("No fields to correct (empty extracted data).")
        return None

    corrections: List[Dict[str, Any]] = []
    corrected_flat: Dict[str, Any] = {}

    # ── Show flagged fields first ─────────────────────────────────────────────
    flagged_set = set(flagged_fields)

    if flagged_set:
        st.markdown(
            """
            <div style="font-size:0.72rem;color:#d29922;font-weight:600;
                        margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">
              ⚠ Low-confidence fields appear first
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Custom Fields Management (Phase 5.3 Supplement) ──────────────────────
    if "custom_fields" not in st.session_state:
        st.session_state["custom_fields"] = {}
    
    # Simple UI to add a new key-value pair (outside the form so buttons work)
    st.markdown("---")
    st.markdown("<p style='font-size:0.9rem;font-weight:600;color:#e6edf3;'>➕ Add Custom Field</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        new_key = st.text_input("Key Name", placeholder="e.g. document_date", key="new_field_key")
    with c2:
        new_val = st.text_input("Value", placeholder="e.g. 2023-10-27", key="new_field_val")
    with c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Add", use_container_width=True):
            if new_key and new_key not in flat and new_key not in st.session_state["custom_fields"]:
                st.session_state["custom_fields"][new_key] = new_val
                st.rerun()
            elif new_key in flat:
                st.error("Key already exists in extracted data.")
    
    # Sort AI fields: flagged first, then alphabetically
    sorted_fields = sorted(flat.keys(), key=lambda k: (k not in flagged_set, k))
    
    # Merge custom fields into the fields to be rendered
    all_fields = list(sorted_fields)
    custom_keys = list(st.session_state["custom_fields"].keys())
    all_fields.extend(custom_keys)

    form_key = f"correction_form_{item.get('item_id', 'unknown')}"
    with st.form(key=form_key):
        document_type_input = st.text_input(
            "Document Type",
            value=item.get("document_type", "Unknown"),
        )
        reviewer_notes = st.text_area(
            "Reviewer Notes (overall)",
            placeholder="Add any overall notes about this document...",
            height=80,
        )

        st.divider()

        st.divider()

        for path in all_fields:
            if path in flat:
                original_val = flat[path]
                is_flagged = path in flagged_set
                score = confidence_scores.get(path, None)
            else:
                # This is a custom field
                original_val = None
                is_flagged = False
                score = None

            # Field header with confidence badge
            conf_html = ""
            if score is not None:
                pct = int(score * 100)
                colour = "#f85149" if score < 0.65 else ("#d29922" if score < 0.85 else "#3fb950")
                conf_html = (
                    f"<span style='font-size:0.7rem;font-weight:600;color:{colour};"
                    f"background:rgba(0,0,0,0.2);border-radius:4px;padding:1px 6px;'>"
                    f"conf: {pct}%</span>"
                )

            flag_html = (
                "<span class='flag-chip'>⚠ LOW CONF</span>"
                if is_flagged else ""
            )

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;margin-top:12px;">
                  <code style="font-size:0.78rem;color:#79c0ff;">{path}</code>
                  {flag_html}
                  {conf_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(
                    f"<p style='font-size:0.72rem;color:#8b949e;margin:0;'>AI Extracted</p>",
                    unsafe_allow_html=True,
                )
                st.code(
                    str(original_val) if original_val is not None else "(empty)",
                    language=None,
                )

            with col2:
                st.markdown(
                    f"<p style='font-size:0.72rem;color:#8b949e;margin:0;'>Corrected Value</p>",
                    unsafe_allow_html=True,
                )
                corrected_val = st.text_input(
                    label=path,
                    value=str(original_val) if original_val is not None else "",
                    key=f"field_{path}_{item.get('item_id', '')}",
                    label_visibility="collapsed",
                )
            
            # Special handling for custom (added) fields
            if path in custom_keys:
                with col1:
                    if st.markdown("<span style='color:#f85149;font-size:0.7rem;cursor:pointer;'>[Remove]</span>", unsafe_allow_html=True):
                        # Note: we can't easily trigger a rerun from inside the form loop
                        # but we can check it on next submission or use a separate button
                        pass

            corrected_flat[path] = corrected_val

            # Track actual changes only
            if str(corrected_val) != str(original_val):
                correction_note = st.text_input(
                    f"Note for `{path}`",
                    placeholder="Why was this corrected?",
                    key=f"note_{path}_{item.get('item_id', '')}",
                    label_visibility="visible",
                )
                corrections.append({
                    "field_path": path,
                    "original_value": original_val,
                    "corrected_value": corrected_val,
                    "confidence_was": score or 0.0,
                    "correction_note": correction_note,
                })

            if is_flagged:
                st.divider()

        # Bounding box corrections (optional, for YOLO fine-tune — Phase 5.3)
        with st.expander("📐 Bounding Box Corrections (optional — for YOLO fine-tuning)"):
            bbox_raw = st.text_area(
                "Paste bounding box corrections as JSON array",
                placeholder='e.g. [{"field": "name", "bbox": [x1,y1,x2,y2], "page": 1}]',
                height=100,
                key=f"bbox_{item.get('item_id', '')}",
            )

        submitted = st.form_submit_button(
            "💾 Submit Corrections",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    # Parse bbox corrections
    bbox_corrections = []
    if bbox_raw and bbox_raw.strip():
        try:
            bbox_corrections = json.loads(bbox_raw)
        except json.JSONDecodeError:
            st.warning("⚠ Invalid JSON in bounding box corrections. Ignoring.")

    # Rebuild corrected_json from flat corrected values
    # (simple flat dict is sufficient for API; orchestrator can re-nest if needed)
    corrected_json = dict(aggregated_json)  # Start from original, shallow copy
    for path, val in corrected_flat.items():
        # Simple top-level override; deep path injection is handled server-side
        top = path.split(".")[0].split("[")[0]
        if top in corrected_json and path == top:
            corrected_json[top] = val

    return {
        "reviewer_id": reviewer_id,
        "document_type": document_type_input,
        "field_corrections": corrections,
        "corrected_json": corrected_flat,   # flat representation for audit trail
        "bbox_corrections": bbox_corrections,
        "reviewer_notes": reviewer_notes,
    }


def render_diff_summary(corrections: List[Dict[str, Any]]) -> None:
    """
    Render a clean diff summary of changes made (shown after submission).
    """
    if not corrections:
        st.success("✓ No field changes — document approved as-is.")
        return

    st.markdown(
        f"<p class='section-header'>Correction Diff ({len(corrections)} changes)</p>",
        unsafe_allow_html=True,
    )
    for c in corrections:
        original = c.get("original_value", "")
        corrected = c.get("corrected_value", "")
        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid rgba(240,246,252,0.08);
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;font-family:monospace;">
              <code style="color:#79c0ff;font-size:0.75rem;">{c['field_path']}</code>
              <div style="display:flex;gap:24px;margin-top:6px;">
                <div>
                  <span style="font-size:0.68rem;color:#8b949e;">BEFORE</span><br/>
                  <span style="color:#f85149;font-size:0.82rem;">{original}</span>
                </div>
                <div style="color:#484f58;align-self:center;">→</div>
                <div>
                  <span style="font-size:0.68rem;color:#8b949e;">AFTER</span><br/>
                  <span style="color:#3fb950;font-size:0.82rem;">{corrected}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def submit_correction_to_api(
    item_id: str,
    payload: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    POST the correction payload to the HITL REST API.
    Returns (success: bool, message: str).
    """
    try:
        url = f"{API_BASE}/hitl/queue/{item_id}/correct"
        response = httpx.post(url, json=payload, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            return True, (
                f"Correction saved. ID: {data.get('correction_id', '?')} | "
                f"{data.get('fields_corrected', 0)} fields corrected."
            )
        else:
            return False, f"API error {response.status_code}: {response.text}"
    except httpx.ConnectError:
        return False, (
            "Cannot connect to HITL API. "
            f"Make sure it is running on port {REVIEWER_API_PORT}."
        )
    except Exception as e:
        return False, f"Unexpected error: {e}"


def submit_approve_to_api(
    item_id: str,
    reviewer_id: str,
    notes: str = "",
) -> Tuple[bool, str]:
    """POST an approval to the HITL REST API."""
    try:
        url = f"{API_BASE}/hitl/queue/{item_id}/approve"
        response = httpx.post(url, json={"reviewer_id": reviewer_id, "notes": notes}, timeout=10.0)
        if response.status_code == 200:
            return True, "Document approved successfully."
        return False, f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"API error: {e}"


def submit_reject_to_api(
    item_id: str,
    reviewer_id: str,
    notes: str = "",
) -> Tuple[bool, str]:
    """POST a rejection to the HITL REST API."""
    try:
        url = f"{API_BASE}/hitl/queue/{item_id}/reject"
        response = httpx.post(url, json={"reviewer_id": reviewer_id, "notes": notes}, timeout=10.0)
        if response.status_code == 200:
            return True, "Document rejected."
        return False, f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"API error: {e}"
