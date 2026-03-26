"""
serve.py — QUARM HQ WebSocket Server
========================================
FastAPI server with two jobs:
  1. Serve static files (quarm_hq.html, any assets)
  2. Accept POST /update from the orchestrator and instantly broadcast
     the payload to all connected WebSocket clients

Architecture:
  orchestrator.py
      └── status_bridge.py
              └── POST http://localhost:8000/update  (fire-and-forget)
                        │
                   FastAPI (serve.py)
                        ├── GET  /             → quarm_hq.html
                        ├── POST /update       → broadcast to all WS clients
                        └── WS   /ws           → browser connects here

Requirements:
  pip install fastapi uvicorn python-multipart

Run:
  python serve.py
  # Or with auto-reload during dev:
  # uvicorn serve:app --reload --port 8000
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quarm")

# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._last_status: dict | None = None   # replay to new connections

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        log.info(f"WS connected  ({len(self.active)} total)")
        # Immediately send last known state so a page refresh isn't blank
        if self._last_status:
            try:
                await ws.send_json(self._last_status)
            except Exception:
                pass

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active = [c for c in self.active if c is not ws]
        log.info(f"WS disconnected  ({len(self.active)} total)")

    async def broadcast(self, payload: dict):
        self._last_status = payload
        dead = []
        async with self._lock:
            clients = list(self.active)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("QUARM HQ server starting")
    log.info(f"  Open: http://localhost:{PORT}/")
    log.info(f"  WS  : ws://localhost:{PORT}/ws")
    log.info(f"  POST: http://localhost:{PORT}/update")
    yield
    log.info("Server shutting down")

app = FastAPI(title="QUARM HQ", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent
PORT = int(os.environ.get("QUARM_PORT", 8000))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "quarm_hq.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="quarm_hq.html not found in this directory")
    return HTMLResponse(html_path.read_text())


@app.get("/{filename}")
async def static_file(filename: str):
    path = STATIC_DIR / filename
    if path.exists() and path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"{filename} not found")


@app.post("/update")
async def receive_update(request: Request):
    """
    Called by status_bridge.py on every orchestrator state change.
    Immediately broadcasts the JSON payload to all WebSocket clients.
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    await manager.broadcast(payload)
    return {"ok": True, "clients": len(manager.active)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; we only push, never pull
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",   # suppress uvicorn noise; our log handler covers it
        access_log=False,
    )
