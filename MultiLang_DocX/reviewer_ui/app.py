"""
IntelX_AI HITL Reviewer Dashboard
====================================
Streamlit application for the Human-in-the-Loop review pipeline.

Three views accessible from the sidebar:
  1. 📋 Queue Dashboard  — overview table of all PENDING_REVIEW items
  2. 🔍 Review Document  — side-by-side document + correction form
  3. 📊 Analytics        — active learning stats and pattern analysis

Run:
    streamlit run reviewer_ui/app.py --server.port 8501
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import streamlit as st

from hitl_config import REVIEWER_API_PORT, CONFIDENCE_THRESHOLD

API_BASE = f"http://localhost:{REVIEWER_API_PORT}"

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IntelX_AI · HITL Reviewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "IntelX_AI Human-in-the-Loop Document Review Dashboard",
    },
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
_css_path = Path(__file__).parent / "assets" / "style.css"
if _css_path.exists():
    with _css_path.open() as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_get(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _api_post(path: str, payload: Dict) -> Optional[Dict]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _check_api() -> bool:
    result = _api_get("/health")
    return result is not None and result.get("status") == "ok"


# ── State helpers ─────────────────────────────────────────────────────────────

def _api_delete(endpoint: str) -> Optional[Dict[str, Any]]:
    try:
        response = httpx.delete(f"{API_BASE}{endpoint}", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def _init_state() -> None:
    defaults = {
        "reviewer_id": "reviewer_001",
        "selected_item_id": None,
        "view": "queue",
        "last_refresh": 0.0,
        "submitted_correction": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:20px 0 8px;">
              <div style="font-size:2rem;">🔍</div>
              <div style="font-size:1.1rem;font-weight:700;color:#e6edf3;">IntelX_AI</div>
              <div style="font-size:0.72rem;color:#8b949e;letter-spacing:0.1em;
                           text-transform:uppercase;">HITL Reviewer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Reviewer identity
        st.markdown("<p style='font-size:0.72rem;color:#8b949e;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;'>Reviewer ID</p>",
                    unsafe_allow_html=True)
        st.session_state["reviewer_id"] = st.text_input(
            "reviewer_id",
            value=st.session_state["reviewer_id"],
            label_visibility="collapsed",
        )

        st.divider()

        def sync_view():
            st.session_state["view"] = st.session_state["sidebar_nav"]

        options = ["queue", "review", "upload", "analytics"]
        idx = options.index(st.session_state.get("view", "queue")) if st.session_state.get("view", "queue") in options else 0

        st.radio(
            "Navigation",
            options=options,
            index=idx,
            format_func=lambda v: {
                "queue":     "📋  Queue Dashboard",
                "review":    "🔍  Review Document",
                "upload":    "🚀  Process Document",
                "analytics": "📊  Analytics",
            }[v],
            label_visibility="collapsed",
            key="sidebar_nav",
            on_change=sync_view,
        )

        st.divider()

        # API status indicator
        api_ok = _check_api()
        status_colour = "#3fb950" if api_ok else "#f85149"
        status_label = "API Connected" if api_ok else "API Offline"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;font-size:0.75rem;color:#8b949e;">
              <div style="width:8px;height:8px;border-radius:50%;
                           background:{status_colour};flex-shrink:0;
                           box-shadow:0 0 6px {status_colour};"></div>
              {status_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<p style='font-size:0.68rem;color:#484f58;margin-top:4px;"
                    f"'>API: {API_BASE}</p>", unsafe_allow_html=True)




# ── View 1: Queue Dashboard ───────────────────────────────────────────────────

def render_queue_dashboard() -> None:
    """Phase 5.2: Display the review queue with key metrics and item table."""

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown("## 📋 Review Queue")
        st.markdown(
            "<p style='color:#8b949e;margin-top:-8px;'>Documents requiring human validation.</p>"
            "<p style='font-size:0.75rem;color:#484f58;border-left:3px solid #1f6feb;padding-left:8px;'>"
            "Note: <b>COMPLETED</b> documents bypass this queue safely.</p>",
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Process New Document", type="primary", use_container_width=True):
            st.session_state["view"] = "upload"
            st.rerun()

    # Stats row
    stats_data = _api_get("/hitl/stats")
    if stats_data:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Documents", stats_data.get("total", 0))
        c2.metric("⏳ Pending Review", stats_data.get("pending_review", 0))
        c3.metric("✅ Approved", stats_data.get("approved", 0))
        c4.metric("✏️ Corrected", stats_data.get("corrected", 0))
        c5.metric("❌ Rejected", stats_data.get("rejected", 0))

        avg_conf = stats_data.get("avg_confidence", 0.0)
        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid rgba(240,246,252,0.1);
                        border-radius:8px;padding:10px 16px;margin:12px 0;
                        display:flex;align-items:center;gap:12px;">
              <span style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
                           letter-spacing:0.06em;font-weight:600;">
                Queue Avg Confidence
              </span>
              <span style="font-size:1.3rem;font-weight:700;
                           color:{'#3fb950' if avg_conf >= CONFIDENCE_THRESHOLD else '#d29922'};">
                {int(avg_conf*100)}%
              </span>
              <span style="font-size:0.72rem;color:#484f58;">
                Threshold: {int(CONFIDENCE_THRESHOLD*100)}%
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Cannot connect to HITL API. Start it with: `python -m hitl.api.main`")
        return

    st.divider()

    # Tabs for organization
    tab_all , tab_mine, tab_unclaimed = st.tabs([
        "📚 All Documents",
        f"👤 My Tasks ({st.session_state.get('reviewer_id', 'Guest')})",
        "📥 Unclaimed"
    ])

    with tab_unclaimed:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        q_unclaimed = _api_get("/hitl/queue", params={"state": "PENDING_REVIEW", "unclaimed": True, "limit": 20})
        if q_unclaimed and q_unclaimed.get("items"):
            for item in q_unclaimed["items"]:
                _render_queue_card(item, key_prefix="unclaimed")
        else:
            st.success("✓ No unclaimed documents! Everything is being handled.")

    with tab_mine:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        r_id = st.session_state.get("reviewer_id")
        q_mine = _api_get("/hitl/queue", params={"state": "PENDING_REVIEW", "reviewer_id": r_id, "limit": 20})
        if q_mine and q_mine.get("items"):
            for item in q_mine["items"]:
                _render_queue_card(item, key_prefix="mine")
        else:
            st.info("You haven't claimed any documents yet. Check the 'Unclaimed' tab!")

    with tab_all:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        # Filters for the "All" view
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            state_filter = st.selectbox(
                "Filter by state",
                options=["ALL", "PENDING_REVIEW", "APPROVED", "CORRECTED", "REJECTED", "COMPLETED"],
                index=0,
                key="state_filter_all"
            )
        with fc2:
            page_size = st.selectbox("Items per page", [10, 20, 50], index=1, key="page_size_all")
        with fc3:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_all"):
                st.rerun()

        params = {"limit": page_size, "offset": 0, "state": state_filter}
        queue_data = _api_get("/hitl/queue", params=params)
        if queue_data and queue_data.get("items"):
            st.markdown(f"<p style='font-size:0.75rem;color:#8b949e;'>Showing {len(queue_data['items'])} items</p>", unsafe_allow_html=True)
            for item in queue_data["items"]:
                _render_queue_card(item, key_prefix="all")
        else:
            st.info("No documents found matching this filter.")


def _render_queue_card(item: Dict[str, Any], key_prefix: str = "") -> None:
    """Render a single review item as a styled card."""
    state = item.get("state", "PENDING_REVIEW")
    reviewer_id = item.get("reviewer_id")
    current_reviewer = st.session_state.get("reviewer_id")
    
    state_colours = {
        "PENDING_REVIEW": ("#d29922", "⏳"),
        "APPROVED":       ("#3fb950", "✅"),
        "CORRECTED":      ("#79c0ff", "✏️"),
        "REJECTED":       ("#f85149", "❌"),
        "COMPLETED":      ("#388bfd", "🚀"),
    }
    colour, icon = state_colours.get(state, ("#8b949e", "?"))
    
    # Assignment tag
    assign_html = ""
    if state == "PENDING_REVIEW":
        if not reviewer_id:
            assign_html = "<span style='font-size:0.65rem;background:#238636;color:white;padding:2px 6px;border-radius:4px;margin-left:8px;'>🆕 UNCLAIMED</span>"
        elif reviewer_id == current_reviewer:
            assign_html = "<span style='font-size:0.65rem;background:#1f6feb;color:white;padding:2px 6px;border-radius:4px;margin-left:8px;'>👤 ASSIGNED TO YOU</span>"
        else:
            assign_html = f"<span style='font-size:0.65rem;background:#30363d;color:#8b949e;padding:2px 6px;border-radius:4px;margin-left:8px;'>👤 {reviewer_id}</span>"

    overall = float(item.get("overall_confidence", 0))
    pct = int(overall * 100)
    flagged_count = len(item.get("flagged_fields", []))
    missing_count = len(item.get("missing_required_fields", []))

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([5, 2, 2, 2])
        with c1:
            html_c1 = f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
  <span style="font-size:1.2rem;">{icon}</span>
  <span style="font-weight:600;font-size:1.1rem;color:#e6edf3;">
    {item.get('document_type', 'Unknown')}
  </span>
  <span style="font-size:0.7rem;color:{colour};background:rgba(0,0,0,0.3);border-radius:4px;padding:2px 8px;font-weight:600;">
    {state}
  </span>
  {assign_html}
</div>
<div style="font-size:0.8rem;color:#8b949e;margin-top:4px;margin-bottom:6px;">
  📄 {Path(item.get('file_path','')).name} &nbsp;|&nbsp; 🕒 {item.get('queued_at','')[:16].replace('T', ' ')}
</div>
"""
            st.markdown(html_c1, unsafe_allow_html=True)
            st.code(item.get('item_id', 'unknown_id'), language=None)
            
        with c2:
            pct_col = '#f85149' if pct < 55 else ('#d29922' if pct < 75 else '#3fb950')
            html_c2 = f"""
<div style="text-align:right;">
  <span style="font-size:1.4rem;font-weight:700;color:{pct_col};">{pct}%</span><br/>
  <span style="font-size:0.7rem;color:#79c0ff;text-transform:uppercase;">confidence</span>
"""
            if flagged_count:
                html_c2 += f'\n  <div style="font-size:0.75rem;color:#f85149;margin-top:4px;font-weight:600;">⚠ {flagged_count} flagged</div>'
            if missing_count:
                html_c2 += f'\n  <div style="font-size:0.75rem;color:#d29922;font-weight:600;">⚠ {missing_count} missing</div>'
            html_c2 += "\n</div>"
            
            st.markdown(html_c2, unsafe_allow_html=True)
            
        with c3:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if st.button("Review →", key=f"{key_prefix}_btn_{item['item_id']}", type="primary", use_container_width=True):
                st.session_state["selected_item_id"] = item["item_id"]
                st.session_state["view"] = "review"
                st.rerun()
                
        with c4:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"{key_prefix}_del_{item['item_id']}", use_container_width=True):
                if _api_delete(f"/hitl/queue/{item['item_id']}"):
                    st.rerun()

    st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)


def _render_mini_conf_bar(overall: float, pct: int) -> str:
    bar_class = "conf-high" if overall >= 0.75 else ("conf-medium" if overall >= 0.55 else "conf-low")
    return f"""
    <div style="margin-top:8px;">
      <div class="conf-bar-container" style="height:4px;">
        <div class="conf-bar-fill {bar_class}" style="width:{pct}%;"></div>
      </div>
    </div>
    """


# ── View 2: Review Document ───────────────────────────────────────────────────

def render_review_view() -> None:
    """Full document review: side-by-side document + confidence panel + correction form."""
    from reviewer_ui.components.confidence_panel import render_confidence_panel
    from reviewer_ui.components.document_viewer import (
        render_document_pages,
        render_extracted_json,
        render_multilingual_mappings,
    )
    from reviewer_ui.components.correction_form import (
        render_correction_form,
        render_diff_summary,
        submit_correction_to_api,
        submit_approve_to_api,
        submit_reject_to_api,
    )

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown("## 🔍 Review Document")
    with col_btn:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🗑️ Delete"):
                item_to_del = st.session_state.get("selected_item_id")
                if item_to_del and _api_delete(f"/hitl/queue/{item_to_del}"):
                    st.session_state["view"] = "queue"
                    st.rerun()
        with b2:
            if st.button("← Back", use_container_width=True):
                st.session_state["view"] = "queue"
                st.rerun()

    # Item selector
    item_id = st.session_state.get("selected_item_id")

    col_id, col_btn = st.columns([4, 1])
    with col_id:
        item_id_input = st.text_input(
            "Item ID",
            value=item_id or "",
            placeholder="Paste review item ID or select from Queue Dashboard",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Load Item →", use_container_width=True) and item_id_input:
            st.session_state["selected_item_id"] = item_id_input.strip()
            item_id = item_id_input.strip()

    if not item_id:
        st.info("Select an item from the Queue Dashboard or enter an item ID above.")
        return

    # Fetch item
    item_data = _api_get(f"/hitl/queue/{item_id}")
    if not item_data:
        st.error(f"Item `{item_id}` not found.")
        return

    item: Dict[str, Any] = item_data.get("item", {})
    corrections_history: List[Dict[str, Any]] = item_data.get("corrections", [])
    reviewer_id: str = st.session_state.get("reviewer_id", "reviewer_001")

    # Edit mode state management
    if "edit_active" not in st.session_state:
        st.session_state["edit_active"] = False
    
    # If the user switched items, reset edit mode
    current_edit_item = st.session_state.get("current_edit_item")
    if current_edit_item != item_id:
        st.session_state["edit_active"] = False
        st.session_state["current_edit_item"] = item_id

    # Item header
    state = item.get("state", "PENDING_REVIEW")
    doc_type = item.get("document_type", "Unknown")
    overall = float(item.get("overall_confidence", 0))

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#161b22,#21262d);
                    border:1px solid rgba(240,246,252,0.1);border-radius:12px;
                    padding:16px 20px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-size:1.2rem;font-weight:700;color:#e6edf3;">{doc_type}</span>
            <span style="font-size:0.72rem;color:#8b949e;background:#0d1117;
                         padding:2px 10px;border-radius:99px;">{state}</span>
            <span style="font-size:0.72rem;color:#8b949e;">
              Confidence: <b style="color:{'#3fb950' if overall>=0.75 else '#d29922'};">
              {int(overall*100)}%</b>
            </span>
          </div>
          <div style="font-size:0.72rem;color:#484f58;margin-top:6px;">
            Item: {item_id}  &nbsp;|&nbsp;  
            File: {Path(item.get('file_path','')).name}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    import json
    export_json = json.dumps(item_data, indent=2)
    st.download_button(
        label="📥 Export to JSON",
        data=export_json,
        file_name=f"document_{item_id}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Claim item ────────────────────────────────────────────────────────────
    current_assignee = item.get("reviewer_id")

    if not current_assignee:
        # Case 1: Unclaimed
        if st.button(f"🆕 Claim for Review ({reviewer_id})", type="primary", use_container_width=True):
            result = _api_post(f"/hitl/queue/{item_id}/claim", {"reviewer_id": reviewer_id})
            if result:
                st.success("Item claimed!")
                time.sleep(0.5)
                st.rerun()
    elif current_assignee == reviewer_id:
        # Case 2: Assigned to current user
        st.info(f"👤 You are the assigned reviewer for this document.")
    else:
        # Case 3: Assigned to someone else
        st.warning(f"👤 Assigned to **{current_assignee}**")
        if st.button(f"⚠️ Take Over Claim as {reviewer_id}", type="secondary", use_container_width=True):
            result = _api_post(f"/hitl/queue/{item_id}/claim", {"reviewer_id": reviewer_id, "force": True})
            if result:
                st.success(f"Claim taken over by {reviewer_id}!")
                time.sleep(0.5)
                st.rerun()

    # ── Three-column layout ───────────────────────────────────────────────────
    doc_col, conf_col = st.columns([3, 2])

    with doc_col:
        st.markdown("<p class='section-header'>Document Pages</p>", unsafe_allow_html=True)
        render_document_pages(
            item.get("page_thumbnails", []),
            file_path=item.get("file_path"),
        )

        st.divider()
        
        display_json = item.get("aggregated_json", {})
        flagged = item.get("flagged_fields", [])
        title = "AI Extracted Data"

        if corrections_history:
            import json
            try:
                latest = corrections_history[-1]
                c_json = latest.get("corrected_json", {})
                if isinstance(c_json, str):
                    c_json = json.loads(c_json)
                display_json = c_json
                flagged = []
                title = "✅ Human Corrected Data"
            except Exception:
                pass

        render_extracted_json(
            display_json,
            flagged_fields=flagged,
            title=title,
        )

        if corrections_history:
            with st.expander("Original AI Extracted Data", expanded=True):
                render_extracted_json(
                    item.get("aggregated_json", {}),
                    flagged_fields=item.get("flagged_fields", []),
                    title="Original AI Extracted",
                )

        # Multilingual mappings (Phase 4.3)
        ml_mappings = item.get("multilingual_mappings", [])
        if ml_mappings:
            st.divider()
            render_multilingual_mappings(ml_mappings)

    with conf_col:
        # Confidence panel
        st.markdown("<p class='section-header'>Confidence Analysis</p>", unsafe_allow_html=True)
        render_confidence_panel(item, threshold=CONFIDENCE_THRESHOLD)

        st.divider()

        # can_edit is true if the current user owns the claim AND (state is pending OR edit mode is active)
        is_owner = (current_assignee == reviewer_id)
        is_resolved = (state in ["APPROVED", "CORRECTED", "COMPLETED"])
        edit_active = st.session_state.get("edit_active", False)

        # Allow editing if pending OR manually unlocked
        can_edit = is_owner and (not is_resolved or edit_active)
        
        if is_owner and is_resolved and not edit_active:
            # Case 1: Resolved and locked — show Edit button
            st.markdown(f"<p style='font-size:0.8rem;color:#8b949e;margin-bottom:8px;'>Current State: <b>{state}</b></p>", unsafe_allow_html=True)
            if st.button("📝 Edit Data", use_container_width=True, type="secondary"):
                st.session_state["edit_requested"] = True
            
            if st.session_state.get("edit_requested"):
                st.warning("⚠️ Enter Edit Mode? Any previous resolutions will be updated upon submission.")
                if st.button("✅ Confirm & Edit", use_container_width=True, type="primary"):
                    st.session_state["edit_active"] = True
                    st.session_state["edit_requested"] = False
                    st.rerun()
                if st.button("Cancel", use_container_width=True):
                    st.session_state["edit_requested"] = False
                    st.rerun()

        elif can_edit:
            # Quick action buttons (approve / reject)
            st.markdown(f"<p style='font-size:0.8rem;color:#8b949e;margin-bottom:8px;'>Current State: <b>{state}</b></p>", unsafe_allow_html=True)
            a_col, r_col = st.columns(2)
            with a_col:
                if st.button("✅ Approve", type="primary", use_container_width=True,
                             key="approve_btn"):
                    notes = st.session_state.get("reviewer_notes_approve", "")
                    ok, msg = submit_approve_to_api(item_id, reviewer_id, notes)
                    if ok:
                        st.success(msg)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)
            with r_col:
                if st.button("❌ Reject", use_container_width=True, key="reject_btn"):
                    notes = st.session_state.get("reviewer_notes_reject", "")
                    ok, msg = submit_reject_to_api(item_id, reviewer_id, notes)
                    if ok:
                        st.warning(msg)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)

            reject_notes = st.text_input(
                "Rejection reason",
                placeholder="Optional reason for rejection...",
                key="reviewer_notes_reject",
            )
        elif not current_assignee:
             st.info("🔓 Claim this document above to enable editing and resolution.")
        else:
            st.info(f"🔒 This document is locked to **{current_assignee}**. Click 'Take Over Claim' above to make changes.")

    # ── Full-width correction form ────────────────────────────────────────────
    if can_edit:
        st.divider()
        st.markdown("### ✏️ Field Corrections")

        correction_payload = render_correction_form(item, reviewer_id)

        if correction_payload is not None:
            ok, msg = submit_correction_to_api(item_id, correction_payload)
            if ok:
                st.session_state["submitted_correction"] = correction_payload
                st.session_state["edit_active"] = False
                st.session_state["custom_fields"] = {}
                st.success(f"✅ {msg}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"❌ {msg}")
                
    if not can_edit and corrections_history:
        try:
            import json
            latest = corrections_history[-1]
            diffs_str = latest.get("field_corrections", "[]")
            diffs = json.loads(diffs_str) if isinstance(diffs_str, str) else diffs_str
            if diffs:
                st.divider()
                render_diff_summary(diffs)
        except Exception:
            pass


# ── View 3: Analytics ─────────────────────────────────────────────────────────

def render_analytics() -> None:
    """Phase 5.3: Active learning analytics — error patterns and prompt suggestions."""
    st.markdown("## 📊 Active Learning Analytics")
    st.markdown(
        "<p style='color:#8b949e;margin-top:-8px;'>Pattern analysis from human corrections</p>",
        unsafe_allow_html=True,
    )

    # Stats summary
    stats = _api_get("/hitl/stats")
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Items Processed", stats.get("total", 0))
        c2.metric("Total Corrected", stats.get("corrected", 0))
        c3.metric("Approval Rate", 
                  f"{round(stats.get('approved',0) / max(stats.get('total',1), 1) * 100)}%")

    st.divider()

    # Doc type filter
    doc_type_options = ["All"] + [
        "AADHAR", "PAN", "Passport", "DL", "AIRTICKET",
        "Bank Statement", "FIR", "Other"
    ]
    selected_type = st.selectbox(
        "Filter by document type", doc_type_options, index=0
    )
    doc_type_param = None if selected_type == "All" else selected_type

    tab1, tab2, tab3 = st.tabs(["🔍 Error Patterns", "💡 Prompt Suggestions", "📤 Export"])

    with tab1:
        report = _api_get("/hitl/active-learning/report",
                          params={"doc_type": doc_type_param} if doc_type_param else {})
        if not report:
            st.warning("No report available.")
        elif report.get("status") == "insufficient_data":
            st.info(report.get("message", "Insufficient data."))
            st.metric("Corrections found", report.get("records_found", 0))
            st.metric("Minimum required", report.get("min_required", 5))
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records Analysed", report.get("records_analysed", 0))
                st.metric("Total Field Corrections", report.get("total_field_corrections", 0))
            with col2:
                most_prob = report.get("most_problematic_field", "N/A")
                st.metric("Most Problematic Field", most_prob or "N/A")

            st.subheader("Top Error Fields")
            errors = report.get("top_error_fields", [])
            if errors:
                import pandas as pd
                df = pd.DataFrame(errors)
                df.columns = ["Field Path", "Error Count", "Avg Confidence at Error"]
                df["Avg Confidence at Error"] = (df["Avg Confidence at Error"] * 100).round(1).astype(str) + "%"
                st.dataframe(df, use_container_width=True, hide_index=True)

            if report.get("doc_type_breakdown"):
                st.subheader("Corrections by Document Type")
                breakdown = report["doc_type_breakdown"]
                import pandas as pd
                df2 = pd.DataFrame(
                    list(breakdown.items()), columns=["Document Type", "Count"]
                ).sort_values("Count", ascending=False)
                st.bar_chart(df2.set_index("Document Type"))

    with tab2:
        suggest = _api_get("/hitl/active-learning/suggest",
                           params={"doc_type": doc_type_param} if doc_type_param else {})
        if not suggest or suggest.get("status") != "ok":
            st.info(suggest.get("reason", "No suggestions available yet.") if suggest else "API unavailable.")
        else:
            st.markdown(f"**{suggest.get('records_analysed', 0)} corrections analysed**")
            st.caption(suggest.get("note", ""))
            for s in suggest.get("suggestions", []):
                with st.expander(
                    f"🔧 `{s['field']}` — {s['error_count']} errors "
                    f"(avg conf: {int(s['avg_confidence_at_error']*100)}%)"
                ):
                    st.info(s["suggestion"])

    with tab3:
        st.subheader("Export Training Dataset")
        st.markdown(
            "Export all human corrections as a JSONL fine-tuning dataset "
            "for LLM or YOLO model improvement."
        )
        export_path = st.text_input(
            "Output path",
            value="/tmp/hitl_training_export.jsonl",
        )
        export_type = st.selectbox(
            "Filter by doc type for export",
            ["All"] + doc_type_options[1:],
        )
        fmt = st.radio("Format", ["jsonl", "json"], horizontal=True)

        if st.button("📤 Export Now", type="primary"):
            payload = {
                "output_path": export_path,
                "doc_type": None if export_type == "All" else export_type,
                "format": fmt,
            }
            result = _api_post("/hitl/active-learning/export", payload)
            if result:
                st.success(
                    f"✅ Exported {result.get('records_exported', 0)} records → `{result.get('output_path')}`"
                )
            else:
                st.error("Export failed. Check API connection.")


# ── View 4: Upload & Process ──────────────────────────────────────────────────

def render_upload_view() -> None:
    """End-to-end processing of a new document right from the UI."""
    st.markdown("## 🚀 Process New Document")
    st.markdown(
        "<p style='color:#8b949e;margin-top:-8px;'>Upload a file to run IntelX_AI extraction natively.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose a document...", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Clear stale results if they just uploaded a new file
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state["upload_result"] = None
            st.session_state["last_uploaded"] = uploaded_file.name

        if st.button("▶ Run AI Pipeline", type="primary", use_container_width=True):
            import tempfile
            import uuid
            
            with st.spinner("⏳ Extracting document... (Models may take 20-60s to run)"):
                tmp_dir = Path(tempfile.gettempdir()) / "hitl_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(uploaded_file.name).suffix
                out_path = tmp_dir / f"{uuid.uuid4()}{ext}"
                
                with open(out_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Dynamically import run_hitl to execute the pipeline locally
                    sys.path.insert(0, str(ROOT))
                    from run_hitl import run
                    result = run(str(out_path), threshold=CONFIDENCE_THRESHOLD)
                    st.session_state["upload_result"] = result
                except Exception as e:
                    st.error(f"Pipeline crashed: {e}")
                    return

    # Display results
    result = st.session_state.get("upload_result")
    if result:
        st.divider()
        state = result.get("state")
        
        if state == "COMPLETED":
            st.success("✅ **Extraction Successful!** Confidence was high enough that this skipped the human review queue.")
            if st.button("📊 Go to Queue Dashboard →", type="primary"):
                st.session_state["view"] = "queue"
                st.session_state["upload_result"] = None
                st.rerun()
            st.json(result.get("aggregated_json", {}))
        elif state == "PENDING_REVIEW":
            st.warning("⚠️ **Low Confidence / Missing Fields**")
            item_id = result.get("review_item_id")
            st.info(f"Routed to human review. The document is safely queued (ID: `{item_id}`).")
            
            if st.button("🔍 Open this document in Review Dashboard →", type="primary"):
                st.session_state["selected_item_id"] = item_id
                st.session_state["view"] = "review"
                st.session_state["upload_result"] = None
                st.rerun()
        else:
            st.error("Document Rejected or completely failed pipeline extraction.")
            st.json(result)


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    render_sidebar()
    view = st.session_state.get("view", "queue")

    # Auto-switch to review if item selected from queue
    if view == "queue":
        render_queue_dashboard()
    elif view == "review":
        render_review_view()
    elif view == "upload":
        render_upload_view()
    elif view == "analytics":
        render_analytics()


if __name__ == "__main__":
    main()
