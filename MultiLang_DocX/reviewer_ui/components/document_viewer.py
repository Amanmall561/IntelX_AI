"""
Document Viewer Component
=========================
Side-by-side document page thumbnail viewer alongside the
extracted JSON for the Streamlit Reviewer UI.

Features:
  - Renders page images (PNG/JPG) by thumbnail path stored in ReviewItem
  - Displays extracted JSON with syntax highlighting
  - Highlights (via border/overlay) low-confidence page regions
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


def render_document_pages(
    page_thumbnails: List[str],
    selected_page: int = 0,
    file_path: Optional[str] = None,
) -> int:
    """
    Render native PDF or image if file_path is provided.
    Falls back to page thumbnail navigation if thumbnail paths exist.
    """
    if file_path:
        p = Path(file_path)
        if p.exists():
            if p.suffix.lower() == ".pdf":
                try:
                    with open(p, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                    pdf_display = (
                        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                        f'width="100%" height="800px" style="border:none;border-radius:8px;"></iframe>'
                    )
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    return 0
                except Exception as e:
                    st.error(f"Could not load PDF: {e}")
            elif p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                st.image(str(p), use_container_width=True, caption=p.name)
                return 0
    if not page_thumbnails:
        st.markdown(
            """
            <div style="background:#161b22;border:1px dashed rgba(240,246,252,0.1);
                        border-radius:12px;padding:48px;text-align:center;color:#484f58;">
              <div style="font-size:3rem;margin-bottom:12px;">📄</div>
              <div style="font-size:0.85rem;">No page thumbnails available.</div>
              <div style="font-size:0.72rem;margin-top:4px;">
                Re-process the document with image export enabled.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return 0

    # Page selector tabs
    page_labels = [f"Page {i+1}" for i in range(len(page_thumbnails))]
    selected = st.selectbox(
        "Select page",
        options=range(len(page_thumbnails)),
        format_func=lambda i: page_labels[i],
        index=selected_page,
        key="doc_page_selector",
        label_visibility="collapsed",
    )

    thumb_path = page_thumbnails[selected]
    p = Path(thumb_path)

    if p.exists():
        st.image(
            str(p),
            use_container_width=True,
            caption=f"Page {selected + 1} of {len(page_thumbnails)}",
        )
    else:
        st.warning(f"Thumbnail not found: `{thumb_path}`")
        st.markdown(
            f"""
            <div style="background:#21262d;border:1px solid rgba(240,246,252,0.1);
                        border-radius:8px;padding:32px;text-align:center;color:#8b949e;">
              <div style="font-size:2rem;margin-bottom:8px;">🖼️</div>
              <div>Page {selected+1}</div>
              <code style="font-size:0.7rem;color:#484f58;">{thumb_path}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected


def _render_dict_as_html_table(data: Dict[str, Any], flagged_fields: Optional[List[str]] = None) -> str:
    """Recursively render a JSON dictionary as a styled HTML hierarchy table, highlighting flagged paths."""
    flagged = set(flagged_fields or [])

    def _to_html(obj: Any, path: str = "", depth: int = 0) -> str:
        if isinstance(obj, dict):
            if not obj:
                return "<span style='color:#8b949e;'>{}</span>"
            html = "<table style='width:100%; border-collapse:collapse; margin:2px 0;'>"
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else str(k)
                is_flagged = curr_path in flagged
                bg = "rgba(240,246,252,0.02)" if depth % 2 == 0 else "transparent"
                if is_flagged:
                    bg = "rgba(248,81,73,0.1)"
                key_style = "color:#f85149; font-weight:700;" if is_flagged else "color:#c9d1d9;"
                warn_icon = "⚠ " if is_flagged else ""
                
                html += f"<tr style='background:{bg}; border-bottom:1px solid rgba(240,246,252,0.1);'>"
                html += f"<td style='padding:6px 10px; width:35%; min-width:120px; vertical-align:top; font-weight:600; {key_style} font-size:0.8rem; border-right:1px solid rgba(240,246,252,0.05);'>{warn_icon}{k}</td>"
                html += f"<td style='padding:6px 10px; vertical-align:top; font-size:0.8rem;'>{_to_html(v, curr_path, depth+1)}</td>"
                html += "</tr>"
            html += "</table>"
            return html
        elif isinstance(obj, list):
            if not obj:
                return "<span style='color:#8b949e;'>[]</span>"
            html = "<div style='display:flex; flex-direction:column; gap:6px;'>"
            for i, item in enumerate(obj):
                curr_path = f"{path}[{i}]"
                is_flagged = curr_path in flagged
                border_col = "#f85149" if is_flagged else "#30363d"
                html += f"<div style='border-left:2px solid {border_col}; padding-left:10px;'>{_to_html(item, curr_path, depth+1)}</div>"
            html += "</div>"
            return html
        else:
            if obj is None:
                return "<span style='color:#8b949e; font-style:italic;'>null</span>"
            is_flagged = path in flagged
            col = "#f85149" if is_flagged else "#a5d6ff"
            return f"<span style='color:{col}; font-family:monospace;'>{str(obj)}</span>"
    
    return f"<div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; overflow:hidden;'>{_to_html(data)}</div>"


def render_extracted_json(
    extracted_json: Dict[str, Any],
    flagged_fields: Optional[List[str]] = None,
    title: str = "Extracted Data",
) -> None:
    """
    Display extracted JSON natively as a custom-styled hierarchical HTML table, 
    so Streamlit `st.json` string parsing errors are avoided.
    """
    flagged = flagged_fields or []

    st.markdown(
        f"<p class='section-header'>{title}</p>",
        unsafe_allow_html=True,
    )

    if flagged:
        chips = "".join(
            f"<span class='flag-chip'>⚠ {f}</span>" for f in flagged[:10]
        )
        if len(flagged) > 10:
            chips += f"<span class='flag-chip'>+{len(flagged)-10} more</span>"
        st.markdown(
            f"""
            <div style="background:rgba(248,81,73,0.07);border:1px solid rgba(248,81,73,0.25);
                        border-radius:8px;padding:10px 14px;margin-bottom:10px;">
              <span style="font-size:0.72rem;font-weight:700;color:#f85149;
                           text-transform:uppercase;letter-spacing:0.06em;">
                Low-confidence fields
              </span>
              <div style="margin-top:6px;">{chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        _render_dict_as_html_table(extracted_json, flagged_fields=flagged),
        unsafe_allow_html=True,
    )


def render_multilingual_mappings(
    mappings: List[Dict[str, str]],
) -> None:
    """
    Phase 4.3: Display multilingual entity mappings in a readable table.
    """
    if not mappings:
        return

    st.markdown(
        "<p class='section-header'>Multilingual Entity Mappings (Phase 4.3)</p>",
        unsafe_allow_html=True,
    )

    rows = []
    for m in mappings:
        rows.append({
            "Field Path": m.get("field_path", ""),
            "Original Script": m.get("original_script", ""),
            "Script": m.get("script_detected", ""),
            "Transliterated": m.get("transliterated", ""),
            "English": m.get("english", ""),
        })

    st.dataframe(
        rows,
        use_container_width=True,
        column_config={
            "Field Path":     st.column_config.TextColumn(width="medium"),
            "Original Script":st.column_config.TextColumn(width="small"),
            "Script":         st.column_config.TextColumn(width="small"),
            "Transliterated": st.column_config.TextColumn(width="medium"),
            "English":        st.column_config.TextColumn(width="medium"),
        },
        hide_index=True,
    )
