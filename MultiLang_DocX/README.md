# IntelX_AI HITL Module

> **Full Human-in-the-Loop validation pipeline** for the IntelX_AI document extraction system.  
> Covers all 5 phases from the master plan: Ingestion → Routing → Extraction → Aggregation → HITL Validation.

---

## Architecture

```
moduler_call.main(file_path)              ← existing pipeline (unchanged)
         │
         ▼
HITLOrchestrator.process()
  ├── Phase 4.1  JSON Compilation       → collect parallel chunk outputs
  ├── Phase 4.2  EntityDeduplicator     → merge duplicates, link cross-chunk entities
  ├── Phase 4.3  MultilingualMapper     → script detection + transliteration
  ├── Phase 5.1  confidence_gate()      → PASS / PARTIAL / FAIL
  │       │
  │   PASS └──→ DocumentState.COMPLETED  (auto-saved, done)
  │       │
  │   FAIL/PARTIAL
  │       └──→ ReviewQueueStore.push()  (DocumentState.PENDING_REVIEW)
  │                    │
  │             Reviewer Dashboard (Streamlit)
  │             REST API (FastAPI)
  │                    │
  │             Human reviewer:  Approve / Correct / Reject
  │                    │
  └── Phase 5.3  ActiveLearningLogger  → JSONL training log + DB
                 PromptTweakAdvisor    → prompt improvement suggestions
```

---

## Directory Structure

```
MultiLang_DocX/
├── config.py                       ← All thresholds, paths, required-field map
├── run_hitl.py                     ← Drop-in wrapper (no changes to DocX_AI)
├── requirements.txt
├── hitl/
│   ├── models.py                   ← Pydantic models (all phases)
│   ├── confidence.py               ← Phase 5.1 — gate + per-field scoring
│   ├── deduplication.py            ← Phase 4.2 — entity dedup + cross-chunk linking
│   ├── multilingual_mapper.py      ← Phase 4.3 — script detection + transliteration
│   ├── queue_store.py              ← Phase 5.2 — SQLite review queue
│   ├── orchestrator.py             ← Master coordinator (4.1 → 5.2)
│   ├── active_learning.py          ← Phase 5.3 — correction logger + advisor
│   ├── data/                       ← Auto-created: review_queue.db, active_learning.jsonl
│   └── api/
│       ├── main.py                 ← FastAPI server entrypoint
│       ├── router.py               ← 10 REST endpoints
│       └── websocket.py            ← Real-time queue updates (WebSocket)
└── reviewer_ui/
    ├── app.py                      ← Streamlit 3-view dashboard
    ├── components/
    │   ├── confidence_panel.py     ← Colour-coded confidence gauge + per-field bars
    │   ├── document_viewer.py      ← Page thumbnails + JSON display
    │   └── correction_form.py      ← Field-level edit form + diff viewer + API submit
    └── assets/
        └── style.css               ← Dark-mode premium UI stylesheet
```

---

## Quick Start

### 1. Install dependencies
```bash
cd /home/ubuntu/MultiLang_DocX
pip install -r requirements.txt
```

### 2. Process a document through the pipeline + HITL
```bash
python run_hitl.py /path/to/document.pdf --pretty
# Exit code 0 = COMPLETED (auto-approved)
# Exit code 2 = PENDING_REVIEW (routed to human queue)
```

### 3. Start the HITL REST API
```bash
python -m hitl.api.main
# Running at http://localhost:7860
# Swagger docs at http://localhost:7860/docs
```

### 4. Start the Reviewer Dashboard
```bash
streamlit run reviewer_ui/app.py --server.port 8501
# Open http://localhost:8501
```

---

## REST API Endpoints

| Method | Path | Phase | Description |
|--------|------|-------|-------------|
| `GET`  | `/hitl/queue` | 5.2 | List pending review items |
| `GET`  | `/hitl/queue/{item_id}` | 5.2 | Get single item + corrections |
| `POST` | `/hitl/queue/{item_id}/claim` | 5.2 | Claim item for review |
| `POST` | `/hitl/queue/{item_id}/approve` | 5.1 | Approve AI output |
| `POST` | `/hitl/queue/{item_id}/correct` | 5.3 | Submit field corrections |
| `POST` | `/hitl/queue/{item_id}/reject` | 5.2 | Reject document |
| `GET`  | `/hitl/stats` | 5.2 | Queue statistics |
| `GET`  | `/hitl/active-learning/report` | 5.3 | Error pattern analysis |
| `GET`  | `/hitl/active-learning/suggest` | 5.3 | Prompt tweak suggestions |
| `POST` | `/hitl/active-learning/export` | 5.3 | Export training dataset |
| `WS`   | `/hitl/ws` | 5.2 | Real-time queue events |

---

## Configuration (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.75` | Auto-approval threshold (Phase 5.1) |
| `LOW_CONFIDENCE_FIELD_THRESHOLD` | `0.55` | Per-field amber flag threshold |
| `DEDUP_SIMILARITY_THRESHOLD` | `90` | Levenshtein ratio for entity dedup (Phase 4.2) |
| `SQLITE_DB_PATH` | `hitl/data/review_queue.db` | Review queue database |
| `ACTIVE_LEARNING_LOG` | `hitl/data/active_learning.jsonl` | Training correction log |
| `REVIEWER_API_PORT` | `7860` | FastAPI server port |
| `HITL_UI_PORT` | `8501` | Streamlit UI port |
| `REQUIRED_FIELDS_BY_DOC_TYPE` | `{AADHAR:[...], ...}` | Per-doc-type required fields |

---

## Integration (no changes to existing code)

```python
# In your existing code, replace:
result = moduler_call.main(file_path)

# With:
from hitl.orchestrator import HITLOrchestrator
result_raw = moduler_call.main(file_path)
hitl = HITLOrchestrator()
hitl_result = hitl.process(result_raw, document_id="...", file_path=file_path)

if hitl_result.state.value == "COMPLETED":
    # Save to database — AI was confident enough
    save_to_db(hitl_result.aggregated_json)
else:
    # Document is in the review queue
    print(f"Review queued: {hitl_result.review_item_id}")
```

---

## Active Learning Loop (Phase 5.3)

Every human correction is automatically:
1. Saved to `hitl/data/active_learning.jsonl` (JSONL fine-tuning dataset)
2. Persisted to SQLite for querying
3. Analysed by `ActiveLearningAnalyser` to surface error patterns
4. Processed by `PromptTweakAdvisor` to suggest LLM prompt improvements

**Export training data:**
```bash
curl -X POST http://localhost:7860/hitl/active-learning/export \
  -H "Content-Type: application/json" \
  -d '{"output_path": "/tmp/training_data.jsonl", "doc_type": "AADHAR"}'
```

---

## Multilingual Support (Phase 4.3)

Supported scripts with transliteration:
- **Indic**: Devanagari (Hindi/Marathi), Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Gurmukhi, Odia
- **Semitic**: Arabic, Hebrew
- **Other**: Cyrillic, Greek, Armenian, Thai, CJK (detection only)

Every non-Latin entity is annotated with:
```json
{
  "field_path": "people[0].name",
  "original_script": "குமார்",
  "script_detected": "tamil",
  "transliterated": "Kumār",
  "english": "Kumār"
}
```
