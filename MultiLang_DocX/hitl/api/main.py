"""
Standalone FastAPI server for the HITL REST API + WebSocket.
Run with: uvicorn hitl.api.main:app --host 0.0.0.0 --port 7860 --reload
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow imports from MultiLang_DocX root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hitl.api.router import router
from hitl.api.websocket import ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(str(ROOT / "hitl.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="IntelX_AI HITL Review API",
    description=(
        "Human-in-the-Loop validation API for the IntelX_AI document extraction pipeline. "
        "Provides queue management, reviewer actions, and active learning endpoints."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the Streamlit UI and any front-end to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(router)
app.include_router(ws_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "service": "hitl-api"}


if __name__ == "__main__":
    from hitl_config import REVIEWER_API_HOST, REVIEWER_API_PORT
    logger.info("Starting HITL API on %s:%d", REVIEWER_API_HOST, REVIEWER_API_PORT)
    uvicorn.run(
        "hitl.api.main:app",
        host=REVIEWER_API_HOST,
        port=REVIEWER_API_PORT,
        reload=False,
        log_level="info",
    )
