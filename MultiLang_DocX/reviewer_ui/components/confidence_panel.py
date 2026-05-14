"""
Confidence Panel Component (Phase 5.1 + 5.2 visual)
=====================================================
Renders a colour-coded confidence breakdown for a single review item.

Colour strategy:
  green  (≥ 0.85)  → PASS
  amber  (0.65–0.84) → REVIEW
  red    (< 0.65)  → FLAG

Used in the Review View of the Streamlit app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _colour_for_score(score: float) -> str:
    if score >= 0.85:
        return "#3fb950"     # green
    elif score >= 0.65:
        return "#d29922"     # amber
    else:
        return "#f85149"     # red


def _bar_class(score: float) -> str:
    if score >= 0.85:
        return "conf-high"
    elif score >= 0.65:
        return "conf-medium"
    return "conf-low"


def render_overall_gauge(overall: float, threshold: float = 0.85) -> None:
    """Render the overall document confidence as a large progress bar."""
    pct = int(overall * 100)
    colour = _colour_for_score(overall)
    label = "AUTO-APPROVED ✓" if overall >= threshold else (
        "PARTIAL — NEEDS REVIEW" if overall >= 0.65 else "LOW CONFIDENCE — FAILED"
    )
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:0.75rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;">
              Overall Confidence
            </span>
            <span style="font-size:1.5rem;font-weight:700;color:{colour};">{pct}%</span>
          </div>
          <div class="conf-bar-container">
            <div class="conf-bar-fill {_bar_class(overall)}" style="width:{pct}%;"></div>
          </div>
          <div style="margin-top:6px;font-size:0.75rem;font-weight:600;color:{colour};">
            {label}
          </div>
          <div style="font-size:0.7rem;color:#484f58;margin-top:2px;">
            Threshold: {int(threshold*100)}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_field_confidence_table(
    confidence_scores: Dict[str, float],
    flagged_fields: Optional[List[str]] = None,
    max_rows: int = 30,
) -> None:
    """
    Render per-field confidence scores as a colour-coded table.
    Shows only the bottom-scoring fields (most likely to be wrong).
    """
    if not confidence_scores:
        st.info("No per-field confidence data available.")
        return

    flagged = set(flagged_fields or [])

    # Sort by score ascending (worst first)
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1])[:max_rows]

    st.markdown(
        "<p class='section-header'>Per-Field Confidence Scores (sorted worst → best)</p>",
        unsafe_allow_html=True,
    )

    for path, score in sorted_scores:
        pct = int(score * 100)
        colour = _colour_for_score(score)
        is_flagged = path in flagged
        flag_html = " <span class='flag-chip'>⚠ LOW</span>" if is_flagged else ""

        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
                <code style="font-size:0.75rem;color:#79c0ff;">{path}</code>
                {flag_html}
                <span style="font-size:0.78rem;font-weight:600;color:{colour};">{pct}%</span>
              </div>
              <div class="conf-bar-container" style="height:5px;">
                <div class="conf-bar-fill {_bar_class(score)}" style="width:{pct}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_missing_fields_alert(missing_fields: List[str]) -> None:
    """Render a warning block listing missing required fields."""
    if not missing_fields:
        st.success("✓ All required fields are present.", icon="✅")
        return

    chips = "".join(
        f"<span class='missing-chip'>{f}</span>" for f in missing_fields
    )
    st.markdown(
        f"""
        <div style="background:rgba(210,153,34,0.08);border:1px solid rgba(210,153,34,0.3);
                    border-radius:8px;padding:12px 16px;margin-bottom:12px;">
          <span style="font-size:0.75rem;font-weight:700;color:#d29922;
                       text-transform:uppercase;letter-spacing:0.06em;">
            ⚠ Missing Required Fields
          </span>
          <div style="margin-top:8px;">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_panel(
    item: Dict[str, Any],
    threshold: float = 0.85,
) -> None:
    """
    Full confidence panel: overall gauge + missing fields alert + field table.
    `item` is a ReviewItem dict from the API.
    """
    overall = float(item.get("overall_confidence", 0.0))
    field_scores = item.get("confidence_scores", {})
    flagged = item.get("flagged_fields", [])
    missing = item.get("missing_required_fields", [])

    render_overall_gauge(overall, threshold)
    st.divider()
    render_missing_fields_alert(missing)
    render_field_confidence_table(field_scores, flagged)
