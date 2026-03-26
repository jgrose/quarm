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
from dotenv import load_dotenv
from checkpoint import has_checkpoint, clear_checkpoint

load_dotenv()
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


CONFIG_FILE = STATIC_DIR / "config.json"

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_config(cfg: dict):
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

_TIER_KEYWORDS_SERVE = {
    "high": ["opus", "gpt-4o", "nova-premier"],
    "mid":  ["sonnet", "gpt-4o-mini", "nova-pro", "llama-4-maverick"],
    "low":  ["haiku", "nova-lite", "llama3.2-3b", "llama3.2-1b"],
}

def _tier_for_model(model_id: str) -> str:
    low = model_id.lower()
    for tier, keywords in _TIER_KEYWORDS_SERVE.items():
        if any(kw in low for kw in keywords):
            return tier
    return "mid"


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

        orchestrator_run(plan_file, plan_id=plan_id)

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


def _resume_interrupted_runs():
    """On startup, resume plans that were running when server died."""
    global _running_plan_id
    queue = _load_queue()
    for entry in queue:
        if entry["status"] == "running":
            plan_id = entry["id"]
            if has_checkpoint(plan_id):
                log.info(f"Resuming interrupted run: {plan_id} ({entry.get('title', '')})")
                with _running_lock:
                    if _running_plan_id:
                        log.warning(f"Cannot resume {plan_id} — another plan already running")
                        continue
                    _running_plan_id = plan_id
                t = threading.Thread(
                    target=_run_orchestrator_worker,
                    args=(plan_id,),
                    daemon=True,
                )
                t.start()
                break  # One at a time; _auto_advance handles the rest
            else:
                log.warning(f"No checkpoint for interrupted run {plan_id} — marking failed")
                _update_plan_status(plan_id, "failed")
                _broadcast_queue()


def _cleanup_running_plans():
    """Mark any in-flight plan as failed on shutdown (graceful Ctrl+C)."""
    with _running_lock:
        plan_id = _running_plan_id
    if plan_id:
        log.info(f"Shutdown: marking running plan {plan_id} as failed")
        _update_plan_status(plan_id, "failed")


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
    _resume_interrupted_runs()
    yield
    _cleanup_running_plans()
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
    global _last_update_time
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    _last_update_time = time.time()
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

    # Clear any stale checkpoint from a previous run
    clear_checkpoint(plan_id)

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


# ── Model config API routes ──────────────────────────────────────────────────

@app.get("/api/models")
async def api_list_models():
    """Return all available models with their enabled/disabled status."""
    from openai import OpenAI
    try:
        client = OpenAI()
        all_models = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach model API: {e}")

    config = _load_config()
    allowed = config.get("allowed_models")

    result = []
    for model_id in all_models:
        result.append({
            "id": model_id,
            "enabled": allowed is None or model_id in allowed,
            "tier": _tier_for_model(model_id),
        })
    return {"models": result, "all_allowed": allowed is None}


@app.post("/api/models")
async def api_save_models(request: Request):
    """Save the list of enabled model IDs."""
    body = await request.json()
    allowed = body.get("allowed_models")

    if allowed is not None and not isinstance(allowed, list):
        raise HTTPException(status_code=400, detail="allowed_models must be a list or null")

    config = _load_config()
    config["allowed_models"] = allowed
    _save_config(config)
    return {"ok": True, "allowed_count": len(allowed) if allowed else "all"}


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


# ── Health endpoint ──────────────────────────────────────────────────────────

_server_start = time.time()
_last_update_time = time.time()


@app.get("/api/health")
async def health():
    stuck_threshold = 1800  # 30 minutes
    is_running = _running_plan_id is not None
    since_update = time.time() - _last_update_time
    status = "idle"
    if is_running:
        status = "stuck" if since_update > stuck_threshold else "running"
    return {
        "status": status,
        "running_plan": _running_plan_id,
        "uptime_seconds": int(time.time() - _server_start),
        "seconds_since_update": int(since_update),
    }


# ── Analytics endpoints ─────────────────────────────────────────────────────

@app.get("/api/analytics/costs")
async def analytics_costs():
    from tracking import get_cost_analytics
    return get_cost_analytics()


@app.get("/api/analytics/scores")
async def analytics_scores():
    from tracking import get_score_analytics
    return get_score_analytics()


# ── Webhook config endpoint ─────────────────────────────────────────────────

CONFIG_FILE = STATIC_DIR / "config.json"

def _load_config():
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}

def _save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

@app.get("/api/config")
async def get_config():
    return _load_config()

@app.post("/api/config")
async def save_config(request: Request):
    data = await request.json()
    cfg = _load_config()
    cfg.update(data)
    _save_config(cfg)
    return {"ok": True}

@app.post("/api/webhook/test")
async def test_webhook(request: Request):
    """Send a test payload to the configured webhook URL."""
    cfg = _load_config()
    url = cfg.get("webhook_url", "")
    if not url:
        raise HTTPException(status_code=400, detail="No webhook URL configured")
    import urllib.request
    payload = json.dumps({
        "project": "QUARM Test",
        "tasks_completed": 0,
        "total_revisions": 0,
        "tokens_used": 0,
        "elapsed_seconds": 0,
        "summary": "This is a test webhook from QUARM HQ.",
    }).encode()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Plan directory watcher ──────────────────────────────────────────────────

INCOMING_DIR = PLANS_DIR / "incoming"

def _watch_incoming():
    """Poll plans/incoming/ for new .md files and auto-queue them."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            for f in sorted(INCOMING_DIR.glob("*.md")):
                plan_id = uuid.uuid4().hex[:12]
                dest = PLANS_DIR / f"{plan_id}.md"
                content = f.read_text()
                dest.write_text(content)
                # Extract title from first heading
                title_match = re.search(r"^#\s+(?:PROJECT PLAN:\s*)?(.+)", content, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else f.stem
                # Add to queue
                plans = json.loads(QUEUE_FILE.read_text()) if QUEUE_FILE.exists() else []
                plans.append({
                    "id": plan_id,
                    "title": title,
                    "status": "queued",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                QUEUE_FILE.write_text(json.dumps(plans, indent=2))
                f.unlink()  # remove from incoming
                log.info(f"Auto-queued: {title} ({plan_id})")
        except Exception as e:
            log.debug(f"Watcher error: {e}")
        time.sleep(5)

_watcher_thread = threading.Thread(target=_watch_incoming, daemon=True)
_watcher_thread.start()


# ── RAG endpoints ────────────────────────────────────────────────────────────

@app.get("/api/rag/stats")
async def rag_stats():
    from rag import get_stats
    return get_stats()

@app.get("/api/rag/search")
async def rag_search_endpoint(q: str = ""):
    if not q:
        return {"results": []}
    from rag import search
    return {"results": search(q, top_k=10)}


# ── Tool approval endpoints ─────────────────────────────────────────────────

@app.get("/api/approvals")
async def get_approvals():
    from tools import get_pending_approvals
    return {"pending": get_pending_approvals()}

@app.post("/api/approvals/{tool_call_id}")
async def resolve_approval_endpoint(tool_call_id: str, request: Request):
    data = await request.json()
    from tools import resolve_approval
    resolve_approval(tool_call_id, data.get("approved", False))
    return {"ok": True}


# ── Artifacts endpoint ───────────────────────────────────────────────────────

ARTIFACTS_DIR = STATIC_DIR / "artifacts"

@app.get("/api/artifacts/{plan_id}")
async def list_artifacts(plan_id: str):
    plan_dir = ARTIFACTS_DIR / plan_id
    if not plan_dir.exists():
        return {"files": []}
    files = []
    for f in sorted(plan_dir.rglob("*")):
        if f.is_file():
            files.append({
                "path": str(f.relative_to(ARTIFACTS_DIR)),
                "size": f.stat().st_size,
                "name": f.name,
            })
    return {"files": files}


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
