"""
serve.py — QUARM HQ WebSocket Server + Plan Queue Manager
============================================================
Single entry point for the entire QUARM system. Run:
  python serve.py
  # Open http://localhost:8000/

Features:
  - Serve the dashboard UI (quarm_hq.html)
  - Accept POST /update from the orchestrator status bridge
  - Plan generation, queue management, and orchestrator execution via API
  - Real-time WebSocket broadcasts for all state changes

API:
  POST /api/generate        — Generate a plan from a description
  GET  /api/plans           — List all plans in queue order
  GET  /api/plans/{id}      — Get a single plan's content + metadata
  POST /api/plans/reorder   — Reorder the queue
  POST /api/plans/{id}/run  — Start orchestrator for a plan
  DELETE /api/plans/{id}    — Remove a plan

Requirements:
  pip install fastapi uvicorn python-multipart
"""

import os
import re
import json
import asyncio
import logging
import uuid
import time
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quarm")

STATIC_DIR = Path(__file__).parent
PORT = int(os.environ.get("QUARM_PORT", 8000))
PLANS_DIR = STATIC_DIR / "plans"
QUEUE_FILE = PLANS_DIR / "queue.json"


# ── Plan storage helpers ─────────────────────────────────────────────────────

def _ensure_plans_dir():
    PLANS_DIR.mkdir(exist_ok=True)
    if not QUEUE_FILE.exists():
        QUEUE_FILE.write_text("[]")


def _load_queue() -> list[dict]:
    _ensure_plans_dir()
    try:
        return json.loads(QUEUE_FILE.read_text())
    except Exception:
        return []


def _save_queue(queue: list[dict]):
    _ensure_plans_dir()
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))


def _extract_title(plan_text: str) -> str:
    m = re.search(r"^# PROJECT PLAN:\s*(.+)", plan_text, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled Plan"


def _add_plan(plan_id: str, title: str, description: str, status: str = "queued"):
    queue = _load_queue()
    queue.append({
        "id": plan_id,
        "title": title,
        "description": description,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_queue(queue)


def _update_plan_status(plan_id: str, status: str, title: str | None = None):
    queue = _load_queue()
    for entry in queue:
        if entry["id"] == plan_id:
            entry["status"] = status
            if title:
                entry["title"] = title
            break
    _save_queue(queue)


def _remove_plan(plan_id: str):
    queue = _load_queue()
    queue = [e for e in queue if e["id"] != plan_id]
    _save_queue(queue)
    plan_file = PLANS_DIR / f"{plan_id}.md"
    if plan_file.exists():
        plan_file.unlink()


# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._last_status: dict | None = None
        self._last_queue: list[dict] | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        log.info(f"WS connected  ({len(self.active)} total)")
        # Send queue state (from memory or disk)
        queue_payload = None
        if self._last_queue:
            queue_payload = {"type": "queue", "plans": self._last_queue}
        else:
            disk_queue = _load_queue()
            if disk_queue:
                queue_payload = {"type": "queue", "plans": disk_queue}
        if queue_payload:
            try:
                await ws.send_json(queue_payload)
            except Exception:
                pass
        # Send last orchestrator state
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
        if payload.get("type") == "queue":
            self._last_queue = payload.get("plans")
        else:
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

# Reference to the running event loop (set during lifespan)
_loop: asyncio.AbstractEventLoop | None = None

# Track running orchestrator so we don't double-start
_running_plan_id: str | None = None
_running_lock = threading.Lock()


def _broadcast_queue():
    """Broadcast current queue state to all WS clients (callable from any thread)."""
    queue = _load_queue()
    payload = {"type": "queue", "plans": queue}
    if _loop:
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _loop)


def _broadcast_plan_event(plan_id: str, event: str, **extra):
    """Broadcast a plan-level event (generating, ready, error, etc.)."""
    payload = {"type": "plan_event", "plan_id": plan_id, "event": event, **extra}
    if _loop:
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _loop)


# ── Background workers ────────────────────────────────────────────────────────

def _generate_plan_worker(plan_id: str, description: str):
    """Run plan generation in a background thread, streaming updates."""
    from generate_plan import generate_plan_streaming
    try:
        plan_file = str(PLANS_DIR / f"{plan_id}.md")
        chunk_buffer = ""
        last_flush = time.time()

        for event in generate_plan_streaming(description, plan_file):
            evt_type = event.get("event")

            if evt_type == "model":
                _broadcast_plan_event(plan_id, "generating_model",
                    model=event["model"],
                    context_window=event["context_window"],
                    estimated_input_tokens=event["estimated_input_tokens"])

            elif evt_type == "chunk":
                chunk_buffer += event["text"]
                now = time.time()
                if now - last_flush > 0.05 or len(chunk_buffer) > 20:
                    _broadcast_plan_event(plan_id, "generating_chunk", text=chunk_buffer)
                    chunk_buffer = ""
                    last_flush = now

            elif evt_type == "done":
                if chunk_buffer:
                    _broadcast_plan_event(plan_id, "generating_chunk", text=chunk_buffer)
                    chunk_buffer = ""
                title = _extract_title(event["plan_text"])
                _update_plan_status(plan_id, "queued", title=title)
                _broadcast_queue()
                _broadcast_plan_event(plan_id, "ready", title=title, usage=event.get("usage"))
                log.info(f"Plan generated: {plan_id} — {title}")

            elif evt_type == "error":
                raise RuntimeError(event["message"])

    except Exception as e:
        _update_plan_status(plan_id, "failed")
        _broadcast_queue()
        _broadcast_plan_event(plan_id, "error", message=str(e))
        log.error(f"Plan generation failed: {plan_id} — {e}")


def _run_orchestrator_worker(plan_id: str):
    """Run the orchestrator in a background thread."""
    global _running_plan_id
    from orchestrator import run as orchestrator_run
    try:
        plan_file = str(PLANS_DIR / f"{plan_id}.md")
        if not Path(plan_file).exists():
            raise FileNotFoundError(f"Plan file not found: {plan_file}")

        _update_plan_status(plan_id, "running")
        _broadcast_queue()
        _broadcast_plan_event(plan_id, "started")
        log.info(f"Orchestrator started: {plan_id}")

        orchestrator_run(plan_file)

        _update_plan_status(plan_id, "done")
        _broadcast_queue()
        _broadcast_plan_event(plan_id, "done")
        log.info(f"Orchestrator finished: {plan_id}")
    except Exception as e:
        _update_plan_status(plan_id, "failed")
        _broadcast_queue()
        _broadcast_plan_event(plan_id, "error", message=str(e))
        log.error(f"Orchestrator failed: {plan_id} — {e}")
    finally:
        with _running_lock:
            _running_plan_id = None
        # Auto-advance: start next queued plan if any
        _auto_advance()


def _auto_advance():
    """Start the next queued plan if nothing is running."""
    global _running_plan_id
    with _running_lock:
        if _running_plan_id:
            return
        queue = _load_queue()
        for entry in queue:
            if entry["status"] == "queued":
                _running_plan_id = entry["id"]
                break
        else:
            return
    t = threading.Thread(
        target=_run_orchestrator_worker,
        args=(_running_plan_id,),
        daemon=True,
    )
    t.start()


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    _ensure_plans_dir()
    log.info("QUARM HQ server starting")
    log.info(f"  Dashboard : http://localhost:{PORT}/")
    log.info(f"  API       : http://localhost:{PORT}/api/plans")
    log.info(f"  WebSocket : ws://localhost:{PORT}/ws")
    yield
    _loop = None
    log.info("Server shutting down")

app = FastAPI(title="QUARM HQ", lifespan=lifespan)


# ── Page routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "quarm_hq.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="quarm_hq.html not found")
    return HTMLResponse(html_path.read_text())


# ── Orchestrator bridge route (existing) ─────────────────────────────────────

@app.post("/update")
async def receive_update(request: Request):
    """Called by status_bridge.py on every orchestrator state change."""
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    # Tag as orchestrator status so UI can distinguish from queue updates
    payload["type"] = "orchestrator"
    await manager.broadcast(payload)
    return {"ok": True, "clients": len(manager.active)}


# ── Plan API routes ──────────────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(request: Request):
    """Generate a plan from a text description."""
    body = await request.json()
    description = body.get("description", "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="description is required")

    plan_id = uuid.uuid4().hex[:12]
    _add_plan(plan_id, "Generating...", description, status="generating")
    _broadcast_queue()

    t = threading.Thread(
        target=_generate_plan_worker,
        args=(plan_id, description),
        daemon=True,
    )
    t.start()

    return {"ok": True, "id": plan_id}


@app.get("/api/plans")
async def api_list_plans():
    """Return the ordered plan queue."""
    return _load_queue()


@app.get("/api/plans/{plan_id}")
async def api_get_plan(plan_id: str):
    """Return a single plan's metadata and content."""
    queue = _load_queue()
    entry = next((e for e in queue if e["id"] == plan_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Plan not found")
    plan_file = PLANS_DIR / f"{plan_id}.md"
    content = plan_file.read_text() if plan_file.exists() else ""
    return {**entry, "content": content}


@app.post("/api/plans/reorder")
async def api_reorder(request: Request):
    """Reorder the plan queue. Expects {"order": ["id1", "id2", ...]}."""
    body = await request.json()
    new_order = body.get("order", [])
    if not new_order:
        raise HTTPException(status_code=400, detail="order is required")

    queue = _load_queue()
    by_id = {e["id"]: e for e in queue}
    reordered = [by_id[pid] for pid in new_order if pid in by_id]
    # Append any plans not in the order list (shouldn't happen, but safe)
    seen = set(new_order)
    for e in queue:
        if e["id"] not in seen:
            reordered.append(e)
    _save_queue(reordered)
    _broadcast_queue()
    return {"ok": True}


@app.post("/api/plans/{plan_id}/run")
async def api_run_plan(plan_id: str):
    """Start the orchestrator for a specific plan."""
    global _running_plan_id
    queue = _load_queue()
    entry = next((e for e in queue if e["id"] == plan_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Plan not found")
    if entry["status"] not in ("queued", "failed", "done"):
        raise HTTPException(status_code=409, detail=f"Plan is {entry['status']}, cannot run")

    with _running_lock:
        if _running_plan_id:
            raise HTTPException(status_code=409, detail="Another plan is already running")
        _running_plan_id = plan_id

    t = threading.Thread(
        target=_run_orchestrator_worker,
        args=(plan_id,),
        daemon=True,
    )
    t.start()
    return {"ok": True, "id": plan_id}


@app.delete("/api/plans/{plan_id}")
async def api_delete_plan(plan_id: str):
    """Remove a plan from the queue."""
    with _running_lock:
        if _running_plan_id == plan_id:
            raise HTTPException(status_code=409, detail="Cannot delete a running plan")
    _remove_plan(plan_id)
    _broadcast_queue()
    return {"ok": True}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


# ── Static file fallback (must be last) ──────────────────────────────────────

@app.get("/{filename}")
async def static_file(filename: str):
    path = STATIC_DIR / filename
    if path.exists() and path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"{filename} not found")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        access_log=False,
    )
