"""
WebSocket – Real-time Queue Update Broadcaster (Phase 5.2)
===========================================================
Provides a WebSocket endpoint at /hitl/ws that pushes JSON events
to all connected reviewer dashboards whenever queue state changes.

Event types:
  item_queued      — a new document entered PENDING_REVIEW
  item_claimed     — a reviewer claimed an item
  item_approved    — an item was approved
  item_corrected   — corrections were submitted
  item_rejected    — an item was rejected
  stats_update     — periodic queue statistics broadcast

Integration:
  The router.py endpoints call `broadcast_event()` after each state change.
  The Streamlit UI connects to this endpoint for live updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["hitl-websocket"])

# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages all active WebSocket connections."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._active.add(websocket)
        logger.info(
            "WebSocket client connected. Total connections: %d", len(self._active)
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._active.discard(websocket)
        logger.info(
            "WebSocket client disconnected. Remaining: %d", len(self._active)
        )

    async def broadcast(self, event: Dict[str, Any]) -> None:
        """Send a JSON event to all connected clients."""
        if not self._active:
            return

        message = json.dumps(event, default=str)
        dead: List[WebSocket] = []

        async with self._lock:
            targets = list(self._active)

        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._active.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)


# Module-level singleton
manager = ConnectionManager()


async def broadcast_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Utility called by the REST router after any state-changing action.
    Thread-safe: can be called from asyncio tasks.
    """
    event = {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    await manager.broadcast(event)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@ws_router.websocket("/hitl/ws")
async def hitl_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time reviewer dashboard updates.

    Connection lifecycle:
      1. Client connects → receives a 'connected' welcome event.
      2. Client receives broadcast events when any queue item changes state.
      3. Client can send a 'ping' message to keep the connection alive.
      4. Connection closed on disconnect or error.
    """
    await manager.connect(websocket)
    try:
        # Send welcome event
        await websocket.send_text(
            json.dumps({
                "type": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "message": "Connected to HITL review queue stream.",
                    "active_connections": manager.connection_count,
                },
            })
        )

        # Keep alive loop — wait for client messages (ping/pong)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data.strip().lower() == "ping":
                    await websocket.send_text(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    )
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_text(
                    json.dumps({
                        "type": "keepalive",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        await manager.disconnect(websocket)
