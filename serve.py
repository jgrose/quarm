"""
serve.py — NORT HQ WebSocket Server + Plan Queue Manager
============================================================
Single entry point for the entire NORT system. Run:
  python serve.py
  # Open http://localhost:8000/

Features:
  - Serve the dashboard UI (nort_hq.html)
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
import zipfile
import io
from dotenv import load_dotenv
from checkpoint import has_checkpoint, clear_checkpoint

load_dotenv()
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nort")

STATIC_DIR = Path(__file__).parent
TEMPLATES_DIR = STATIC_DIR / "templates"
PORT = int(os.environ.get("NORT_PORT", os.environ.get("QUARM_PORT", 8000)))
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
PLANS_DIR = STATIC_DIR / "plans"
QUEUE_FILE = PLANS_DIR / "queue.json"
_queue_lock = threading.Lock()


# ── Plan storage helpers ─────────────────────────────────────────────────────

def _ensure_plans_dir():
    PLANS_DIR.mkdir(exist_ok=True)
    if not QUEUE_FILE.exists():
        QUEUE_FILE.write_text("[]")


def _load_queue() -> list[dict]:
    """Read queue from disk. Caller MUST hold _queue_lock for read-modify-write."""
    _ensure_plans_dir()
    try:
        return json.loads(QUEUE_FILE.read_text())
    except Exception as e:
        log.warning(f"Failed to load queue file: {e}")
        return []


def _save_queue(queue: list[dict]):
    """Write queue to disk. Caller MUST hold _queue_lock for read-modify-write."""
    _ensure_plans_dir()
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))


CONFIG_FILE = STATIC_DIR / "config.json"

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            log.warning(f"Failed to load config file: {e}")
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
    with _queue_lock:
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
    with _queue_lock:
        queue = _load_queue()
        for entry in queue:
            if entry["id"] == plan_id:
                entry["status"] = status
                if title:
                    entry["title"] = title
                break
        _save_queue(queue)


def _remove_plan(plan_id: str):
    with _queue_lock:
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
        self._sessions: dict[str, dict] = {}  # session_id → last status payload
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
            with _queue_lock:
                disk_queue = _load_queue()
            if disk_queue:
                queue_payload = {"type": "queue", "plans": disk_queue}
        if queue_payload:
            try:
                await ws.send_json(queue_payload)
            except Exception as e:
                log.debug(f"Failed to send queue state to new WS client: {e}")
        # Send all active session states
        for session_id, status in self._sessions.items():
            try:
                await ws.send_json(status)
            except Exception as e:
                log.debug(f"Failed to send session {session_id} state to new WS client: {e}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active = [c for c in self.active if c is not ws]
        log.info(f"WS disconnected  ({len(self.active)} total)")

    def cleanup_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    async def broadcast(self, payload: dict):
        if payload.get("type") == "queue":
            self._last_queue = payload.get("plans")
        else:
            session_id = payload.get("session_id") or "default"
            self._sessions[session_id] = payload
        dead = []
        async with self._lock:
            clients = list(self.active)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception as e:
                log.debug(f"WS send failed (marking dead): {e}")
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()

# Reference to the running event loop (set during lifespan)
_loop: asyncio.AbstractEventLoop | None = None

# Track running orchestrators — multiple plans can run simultaneously
_running_plan_ids: set[str] = set()
_running_lock = threading.Lock()
_stop_flags: set[str] = set()  # plan IDs that should be stopped


def _broadcast_queue():
    """Broadcast current queue state to all WS clients (callable from any thread)."""
    with _queue_lock:
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
            _running_plan_ids.discard(plan_id)
        _auto_advance()


def _auto_advance():
    """Start any queued plans that aren't already running."""
    with _running_lock:
        with _queue_lock:
            queue = _load_queue()
        to_start = []
        for entry in queue:
            if entry["status"] == "queued" and entry["id"] not in _running_plan_ids:
                to_start.append(entry["id"])
                _running_plan_ids.add(entry["id"])
    for plan_id in to_start:
        t = threading.Thread(
            target=_run_orchestrator_worker,
            args=(plan_id,),
            daemon=True,
        )
        t.start()


def _resume_interrupted_runs():
    """On startup, resume plans that were running when server died."""
    with _queue_lock:
        queue = _load_queue()
    for entry in queue:
        if entry["status"] == "running":
            plan_id = entry["id"]
            if has_checkpoint(plan_id):
                log.info(f"Resuming interrupted run: {plan_id} ({entry.get('title', '')})")
                with _running_lock:
                    _running_plan_ids.add(plan_id)
                t = threading.Thread(
                    target=_run_orchestrator_worker,
                    args=(plan_id,),
                    daemon=True,
                )
                t.start()
            else:
                log.warning(f"No checkpoint for interrupted run {plan_id} — marking failed")
                _update_plan_status(plan_id, "failed")
                _broadcast_queue()


def _cleanup_running_plans():
    """On graceful shutdown, leave running plans resumable if checkpointed."""
    with _running_lock:
        plan_ids = list(_running_plan_ids)
    for plan_id in plan_ids:
        if has_checkpoint(plan_id):
            log.info(f"Shutdown: plan {plan_id} has checkpoint — will resume on restart")
        else:
            log.info(f"Shutdown: marking plan {plan_id} as failed (no checkpoint)")
            _update_plan_status(plan_id, "failed")


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    _ensure_plans_dir()
    log.info("NORT HQ server starting")
    log.info(f"  Dashboard : http://localhost:{PORT}/")
    log.info(f"  API       : http://localhost:{PORT}/api/plans")
    log.info(f"  WebSocket : ws://localhost:{PORT}/ws")
    _resume_interrupted_runs()
    yield
    _cleanup_running_plans()
    _loop = None
    log.info("Server shutting down")

app = FastAPI(title="NORT HQ", lifespan=lifespan)


# ── Page routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "base.html", {"port": PORT})

@app.get("/flow", response_class=HTMLResponse)
async def flow_view(request: Request):
    return templates.TemplateResponse(request, "flow.html", {"port": PORT})


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
    # Tag as orchestrator status if no type set (preserves approval_request etc.)
    if "type" not in payload:
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
    with _queue_lock:
        return _load_queue()


@app.get("/api/plans/summary")
async def api_plans_summary():
    """Return every plan with cheap disk-state flags so the output browser's
    plan picker can populate in one round-trip. has_artifacts/has_output tell the
    UI whether there's anything worth showing for each plan."""
    with _queue_lock:
        queue = _load_queue()
    summary = []
    for entry in queue:
        pid = entry["id"]
        art_dir = ARTIFACTS_DIR / pid
        task_count = 0
        if art_dir.is_dir():
            task_count = sum(1 for d in art_dir.iterdir() if d.is_dir() and d.name.startswith("TASK-"))
        summary.append({
            "id": pid,
            "title": entry.get("title", ""),
            "status": entry.get("status", ""),
            "created_at": entry.get("created_at", ""),
            "has_artifacts": art_dir.is_dir() and task_count > 0,
            "has_output": _find_output_dir(pid) is not None,
            "task_count": task_count,
        })
    return {"plans": summary}


@app.get("/api/plans/{plan_id}")
async def api_get_plan(plan_id: str):
    """Return a single plan's metadata and content."""
    with _queue_lock:
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

    with _queue_lock:
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
async def api_run_plan(plan_id: str, request: Request):
    """Start the orchestrator for a specific plan."""
    # Accept optional human-input policy on the run payload.
    try:
        data = await request.json()
    except Exception:
        data = {}
    if isinstance(data, dict) and "human_input_policy" in data:
        from tools import set_plan_policy
        try:
            set_plan_policy(
                plan_id,
                policy=data.get("human_input_policy", "block"),
                timeout_s=int(data.get("human_input_timeout_s", 900)),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    with _queue_lock:
        queue = _load_queue()
    entry = next((e for e in queue if e["id"] == plan_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Plan not found")
    if entry["status"] not in ("queued", "failed", "done"):
        raise HTTPException(status_code=409, detail=f"Plan is {entry['status']}, cannot run")

    # Clear any stale checkpoint from a previous run
    clear_checkpoint(plan_id)

    with _running_lock:
        if plan_id in _running_plan_ids:
            raise HTTPException(status_code=409, detail="Plan is already running")
        _running_plan_ids.add(plan_id)

    t = threading.Thread(
        target=_run_orchestrator_worker,
        args=(plan_id,),
        daemon=True,
    )
    t.start()
    return {"ok": True, "id": plan_id}


@app.post("/api/plans/{plan_id}/stop")
async def api_stop_plan(plan_id: str):
    """Stop a running plan."""
    with _running_lock:
        if plan_id not in _running_plan_ids:
            raise HTTPException(status_code=409, detail="Plan is not running")
    # Set a stop flag the orchestrator thread can check
    _stop_flags.add(plan_id)
    _update_plan_status(plan_id, "failed")
    _broadcast_queue()
    log.info(f"Stop requested: {plan_id}")
    return {"ok": True}


@app.delete("/api/plans/{plan_id}")
async def api_delete_plan(plan_id: str):
    """Remove a plan from the queue."""
    with _running_lock:
        if plan_id in _running_plan_ids:
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


# ── Session API routes ────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """Return a list of active sessions with summary info."""
    sessions = []
    for sid, status in manager._sessions.items():
        sessions.append({
            "id": sid,
            "project": status.get("project", "NORT"),
            "phase": status.get("phase", "idle"),
            "tasks_done": status.get("results_count", 0),
            "tasks_total": status.get("total_tasks", 0),
            "tokens": status.get("tokens_used", 0),
            "updated_at": status.get("updated_at", ""),
        })
    return {"sessions": sessions}


@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """Return the transcript for a session."""
    status = manager._sessions.get(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"transcript": status.get("transcript", [])}


@app.get("/api/files/{session_id}")
async def get_files_touched(session_id: str):
    """Return file attention data for a session."""
    status = manager._sessions.get(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"files": status.get("files_touched", [])}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            # Handle heartbeat ping from client
            try:
                msg = json.loads(raw)
                if isinstance(msg, dict) and msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except (json.JSONDecodeError, TypeError):
                pass
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        log.warning(f"WebSocket error (disconnecting): {e}")
        await manager.disconnect(websocket)


# ── Health endpoint ──────────────────────────────────────────────────────────

_server_start = time.time()
_last_update_time = time.time()


@app.get("/api/health")
async def health():
    stuck_threshold = 1800  # 30 minutes
    with _running_lock:
        running = list(_running_plan_ids)
    is_running = len(running) > 0
    since_update = time.time() - _last_update_time
    status = "idle"
    if is_running:
        status = "stuck" if since_update > stuck_threshold else "running"
    return {
        "status": status,
        "running_plans": running,
        "running_count": len(running),
        "uptime_seconds": int(time.time() - _server_start),
        "seconds_since_update": int(since_update),
    }


# ── Analytics endpoints ─────────────────────────────────────────────────────

@app.get("/api/analytics/costs/{run_id}")
async def analytics_costs_by_run(run_id: str):
    from tracking import get_run_cost_breakdown
    return get_run_cost_breakdown(run_id)


@app.get("/api/analytics/costs")
async def analytics_costs():
    from tracking import get_cost_analytics
    return get_cost_analytics()


@app.get("/api/analytics/scores")
async def analytics_scores():
    from tracking import get_score_analytics
    return get_score_analytics()


@app.get("/api/review-stats")
async def review_stats():
    from tracking import get_review_stats
    return get_review_stats()


# ── Specialization endpoint ──────────────────────────────────────────────────

@app.get("/api/specializations")
async def api_specializations():
    """Return the current agent specialization matrix."""
    from specialization import get_specialization_matrix
    return get_specialization_matrix()


# ── Webhook config endpoint ─────────────────────────────────────────────────

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

@app.get("/api/tolerance")
async def get_tolerance():
    cfg = _load_config()
    overrides = cfg.get("tolerance_overrides", {})
    default_tol = cfg.get("default_tolerance", 6)

    builtin = [
        {"name": "security_engineer", "title": "Senior Security Engineer"},
        {"name": "ux_designer", "title": "Senior UX/UI Designer"},
        {"name": "user_tester", "title": "End-User Representative"},
        {"name": "creative_director", "title": "Creative Director"},
        {"name": "devils_advocate", "title": "Devil's Advocate"},
        {"name": "performance_engineer", "title": "Performance Engineer"},
    ]

    # Also pull any managers/custom reviewers from active sessions
    seen = {r["name"] for r in builtin}
    for name, tol in overrides.items():
        if name not in seen:
            builtin.append({"name": name, "title": name.replace("_", " ").title()})
            seen.add(name)

    agents = []
    for r in builtin:
        agent_tol = overrides.get(r["name"])
        agents.append({
            "name": r["name"],
            "title": r["title"],
            "tolerance": agent_tol,
            "effective": agent_tol if agent_tol is not None else default_tol,
        })

    return {
        "default_tolerance": default_tol,
        "agents": agents,
        "overrides": overrides,
        "active_preset": cfg.get("active_preset"),
    }

@app.post("/api/tolerance")
async def save_tolerance(request: Request):
    body = await request.json()
    cfg = _load_config()

    # Handle preset selection
    if "preset" in body:
        preset_name = body["preset"]
        preset_values = {"prototype": 8, "production": 5, "audit": 3}
        if preset_name not in preset_values:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_name}")
        cfg["default_tolerance"] = preset_values[preset_name]
        cfg["tolerance_overrides"] = {}
        cfg["active_preset"] = preset_name
        _save_config(cfg)
        return {"ok": True, "preset": preset_name}

    # Clear active preset on manual changes
    cfg.pop("active_preset", None)

    if "default_tolerance" in body:
        val = body["default_tolerance"]
        if val is not None and (not isinstance(val, int) or val < 1 or val > 10):
            raise HTTPException(status_code=400, detail="default_tolerance must be 1-10 or null")
        cfg["default_tolerance"] = val

    if "tolerance_overrides" in body:
        overrides = body["tolerance_overrides"]
        if not isinstance(overrides, dict):
            raise HTTPException(status_code=400, detail="tolerance_overrides must be an object")
        for name, val in overrides.items():
            if val is not None and (not isinstance(val, int) or val < 1 or val > 10):
                raise HTTPException(status_code=400, detail=f"Tolerance for '{name}' must be 1-10 or null")
        cfg["tolerance_overrides"] = overrides

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
        "project": "NORT Test",
        "tasks_completed": 0,
        "total_revisions": 0,
        "tokens_used": 0,
        "elapsed_seconds": 0,
        "summary": "This is a test webhook from NORT HQ.",
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
                # Add to queue (use helper for thread safety)
                _add_plan(plan_id, title, description="")
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


# ── Ask-human question endpoints ────────────────────────────────────────────

@app.get("/api/questions")
async def get_questions():
    from tools import get_pending_questions
    return {"pending": get_pending_questions()}


@app.post("/api/questions/{tool_call_id}")
async def resolve_question_endpoint(tool_call_id: str, request: Request):
    data = await request.json()
    from tools import resolve_question
    resolve_question(tool_call_id, str(data.get("answer", "")))
    return {"ok": True}


# ── Operator guidance injection ─────────────────────────────────────────────

def _validate_task_id(task_id: str):
    if not re.match(r'^[a-zA-Z0-9_.\-]+$', task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")


@app.post("/api/tasks/{plan_id}/{task_id}/guidance")
async def post_task_guidance(plan_id: str, task_id: str, request: Request):
    """Queue an operator nudge that the agent will see on its next LLM turn."""
    _validate_plan_id(plan_id)
    _validate_task_id(task_id)
    data = await request.json()
    message = str(data.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    from tools import queue_guidance
    pending = queue_guidance(plan_id, task_id, message)
    return {"ok": True, "pending": pending}


@app.get("/api/tasks/{plan_id}/{task_id}/guidance")
async def get_task_guidance(plan_id: str, task_id: str):
    """Return pending (not-yet-consumed) guidance for a task, for the UI badge."""
    _validate_plan_id(plan_id)
    _validate_task_id(task_id)
    from tools import peek_guidance
    msgs = peek_guidance(plan_id, task_id)
    return {"plan_id": plan_id, "task_id": task_id, "pending": len(msgs), "messages": msgs}


@app.post("/api/plans/{plan_id}/human_policy")
async def set_plan_human_policy(plan_id: str, request: Request):
    _validate_plan_id(plan_id)
    data = await request.json()
    policy = data.get("policy", "block")
    timeout_s = int(data.get("timeout_s", 900))
    from tools import set_plan_policy
    try:
        set_plan_policy(plan_id, policy=policy, timeout_s=timeout_s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


# ── Artifacts endpoint ───────────────────────────────────────────────────────

ARTIFACTS_DIR = STATIC_DIR / "artifacts"

def _validate_plan_id(plan_id: str):
    if not re.match(r'^[a-zA-Z0-9_.\-]+$', plan_id):
        raise HTTPException(status_code=400, detail="Invalid plan_id")

@app.get("/api/artifacts/{plan_id}")
async def list_artifacts(plan_id: str):
    _validate_plan_id(plan_id)
    plan_dir = ARTIFACTS_DIR / plan_id
    if not plan_dir.exists():
        return {"files": [], "tree": {}}
    files = []
    tree = {}
    for f in sorted(plan_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(plan_dir)
            parts = rel.parts
            task_id = parts[0] if len(parts) > 1 else ""
            st_size = f.stat().st_size
            artifact_path = str(f.relative_to(ARTIFACTS_DIR))
            ext = f.suffix.lstrip(".")
            files.append({
                "path": artifact_path,
                "rel_path": str(rel),
                "size": st_size,
                "name": f.name,
                "ext": ext,
                "task_id": task_id,
            })
            # Build nested tree
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[f.name] = {"file": True, "path": artifact_path, "size": st_size, "ext": ext}
    return {"files": files, "tree": tree}


@app.get("/api/artifacts/{plan_id}/file")
async def get_artifact_file(plan_id: str, path: str = ""):
    """Return the content of a specific artifact file."""
    _validate_plan_id(plan_id)
    if not path:
        raise HTTPException(status_code=400, detail="path parameter required")
    # Security: prevent path traversal
    plan_dir = (ARTIFACTS_DIR / plan_id).resolve()
    target = (plan_dir / path).resolve()
    try:
        target.relative_to(plan_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Binary files: return metadata only
    binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".zip", ".tar", ".gz"}
    if target.suffix.lower() in binary_exts:
        return {"path": path, "binary": True, "size": target.stat().st_size}
    try:
        content = target.read_text(errors="replace")
        return {"path": path, "binary": False, "content": content, "size": target.stat().st_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/artifacts/{plan_id}/download")
async def download_artifacts(plan_id: str):
    """Download all artifacts for a plan as a zip file."""
    _validate_plan_id(plan_id)
    plan_dir = ARTIFACTS_DIR / plan_id
    if not plan_dir.exists():
        raise HTTPException(status_code=404, detail="No artifacts found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(plan_dir.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(plan_dir)))
    buf.seek(0)
    zip_size = buf.getbuffer().nbytes
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{re.sub(r"[^a-zA-Z0-9_-]", "", plan_id)}_artifacts.zip"',
            "Content-Length": str(zip_size),
        },
    )


@app.get("/api/artifacts/{plan_id}/revisions/{task_id}")
async def list_revisions(plan_id: str, task_id: str):
    """List revision snapshots for a task."""
    _validate_plan_id(plan_id)
    if not re.match(r'^[a-zA-Z0-9_.\-]+$', task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id")
    rev_dir = ARTIFACTS_DIR / plan_id / task_id / "revisions"
    if not rev_dir.exists():
        return {"revisions": []}
    revisions = []
    for d in sorted(rev_dir.iterdir()):
        if d.is_dir():
            files = [str(f.relative_to(d)) for f in d.rglob("*") if f.is_file()]
            revisions.append({
                "revision": d.name,
                "files": files,
                "file_count": len(files),
            })
    return {"revisions": revisions}


# ── Output Directory API ───────────────────────────────────────────────────

OUTPUT_DIR = STATIC_DIR / "output"


def _find_output_dir(plan_id: str) -> Path | None:
    """Locate the output folder for a plan_id (named {plan_id}_{slug})."""
    if not OUTPUT_DIR.is_dir():
        return None
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(plan_id):
            return d
    return None


@app.get("/api/output/{plan_id}/files")
async def list_output_files(plan_id: str):
    """Return a JSON tree of files in the assembled output directory."""
    _validate_plan_id(plan_id)
    out_dir = _find_output_dir(plan_id)
    if out_dir is None:
        return {"files": [], "tree": {}}
    files = []
    tree = {}
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            parts = rel.parts
            st_size = f.stat().st_size
            ext = f.suffix.lstrip(".")
            files.append({"path": str(rel), "size": st_size, "name": f.name, "ext": ext})
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[f.name] = {"file": True, "path": str(rel), "size": st_size, "ext": ext}
    return {"files": files, "tree": tree}


@app.get("/api/output/{plan_id}/file")
async def get_output_file(plan_id: str, path: str = ""):
    """Return the content of a specific file from the output directory."""
    _validate_plan_id(plan_id)
    if not path:
        raise HTTPException(status_code=400, detail="path parameter required")
    out_dir = _find_output_dir(plan_id)
    if out_dir is None:
        raise HTTPException(status_code=404, detail="Output directory not found")
    resolved_dir = out_dir.resolve()
    target = (resolved_dir / path).resolve()
    try:
        target.relative_to(resolved_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".zip", ".tar", ".gz"}
    if target.suffix.lower() in binary_exts:
        return {"path": path, "binary": True, "size": target.stat().st_size}
    try:
        content = target.read_text(errors="replace")
        return {"path": path, "binary": False, "content": content, "size": target.stat().st_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/output/{plan_id}/download")
async def download_output(plan_id: str):
    """Download the assembled output folder for a plan as a ZIP."""
    _validate_plan_id(plan_id)
    out_dir = _find_output_dir(plan_id)
    if out_dir is None:
        raise HTTPException(status_code=404, detail="No output found for this plan")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(out_dir)))
    buf.seek(0)
    zip_size = buf.getbuffer().nbytes
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", out_dir.name)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.zip"',
            "Content-Length": str(zip_size),
        },
    )


# ── Results API ────────────────────────────────────────────────────────────


@app.get("/api/results/{plan_id}")
async def get_plan_results(plan_id: str):
    """Return the task results for a plan (text output from agents)."""
    _validate_plan_id(plan_id)
    results_file = PLANS_DIR / f"{plan_id}_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="No results found")
    try:
        data = json.loads(results_file.read_text())
        return {
            "task_results": data.get("task_results", {}),
            "summary": data.get("summary", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent Registry API ──────────────────────────────────────────────────────

# ── Agent Import/Export ────────────────────────────────────────────────────
# NOTE: These must be registered BEFORE parameterized /api/agents/{agent_type}
# routes, otherwise FastAPI matches "export"/"import" as agent_type.

@app.get("/api/agents/export")
async def api_export_agents():
    from agent_registry import export_agents
    return export_agents()

@app.post("/api/agents/import")
async def api_import_agents(request: Request):
    from agent_registry import import_agents
    body = await request.json()
    data = body.get("data", body)
    overwrite = body.get("overwrite", False)
    try:
        summary = import_agents(data, overwrite=overwrite)
        return {"ok": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/agents/import-single")
async def api_import_single_agent(request: Request):
    """Import a single agent from an exported JSON payload."""
    body = await request.json()
    if not body.get("nort_agent_export"):
        raise HTTPException(status_code=400, detail="Missing nort_agent_export marker")
    agent_type = body.get("agent_type")
    agent_data = body.get("agent")
    overwrite = body.get("overwrite", False)
    if not agent_type or not agent_data:
        raise HTTPException(status_code=400, detail="agent_type and agent required")
    from agent_registry import get_agent, create_agent, merge_agent_from_plan
    name = agent_data.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="agent.name is required")
    existing = get_agent(agent_type, name)
    if existing and not overwrite:
        raise HTTPException(status_code=409, detail=f"Agent {name} already exists. Set overwrite=true to replace.")
    try:
        if existing:
            merge_agent_from_plan(agent_type, agent_data)
        else:
            create_agent(agent_type, agent_data)
        return {"ok": True, "created": not existing, "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agents")
async def api_list_agents(type: str = None):
    """List all agents, optionally filtered by type (sub_agents, managers, reviewers)."""
    try:
        from agent_registry import list_agents
        agents = list_agents(type)
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_type}/{name}")
async def api_get_agent(agent_type: str, name: str):
    from agent_registry import get_agent
    agent = get_agent(agent_type, name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/api/agents/{agent_type}")
async def api_create_agent(agent_type: str, request: Request):
    from agent_registry import create_agent
    spec = await request.json()
    try:
        agent = create_agent(agent_type, spec)
        return {"ok": True, "agent": agent}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/agents/{agent_type}/{name}")
async def api_update_agent(agent_type: str, name: str, request: Request):
    from agent_registry import update_agent
    updates = await request.json()
    agent = update_agent(agent_type, name, updates)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"ok": True, "agent": agent}

@app.delete("/api/agents/{agent_type}/{name}")
async def api_delete_agent(agent_type: str, name: str):
    from agent_registry import delete_agent
    try:
        ok = delete_agent(agent_type, name)
        if not ok:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


# ── Agent Versioning ───────────────────────────────────────────────────────

@app.get("/api/agents/{agent_type}/{name}/versions")
async def api_get_agent_versions(agent_type: str, name: str):
    from agent_registry import get_agent_versions
    versions = get_agent_versions(agent_type, name)
    return {"versions": versions}

@app.post("/api/agents/{agent_type}/{name}/rollback")
async def api_rollback_agent(agent_type: str, name: str, request: Request):
    from agent_registry import rollback_agent
    body = await request.json()
    version = body.get("version")
    if version is None:
        raise HTTPException(status_code=400, detail="version is required")
    agent = rollback_agent(agent_type, name, int(version))
    if not agent:
        raise HTTPException(status_code=404, detail="Agent or version not found")
    return {"ok": True, "agent": agent}


# ── Agent Cloning ──────────────────────────────────────────────────────────

@app.post("/api/agents/{agent_type}/{name}/clone")
async def api_clone_agent(agent_type: str, name: str, request: Request):
    from agent_registry import clone_agent
    body = await request.json() if await request.body() else {}
    new_name = body.get("new_name")
    try:
        agent = clone_agent(agent_type, name, new_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"ok": True, "agent": agent}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── Agent Retirement ───────────────────────────────────────────────────────

@app.post("/api/agents/{agent_type}/{name}/retire")
async def api_retire_agent(agent_type: str, name: str, request: Request):
    from agent_registry import retire_agent
    body = await request.json() if await request.body() else {}
    retired = body.get("retired", True)
    agent = retire_agent(agent_type, name, retired)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"ok": True, "agent": agent}


# ── Single Agent Export/Import ────────────────────────────────────────────

@app.get("/api/agents/{agent_type}/{name}/export")
async def api_export_single_agent(agent_type: str, name: str, format: str = "json"):
    """Export a single agent as JSON or claude-code markdown."""
    if format == "claude":
        from agent_registry import export_agent_as_claude_code
        md = export_agent_as_claude_code(agent_type, name)
        if md is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        from fastapi.responses import Response
        return Response(
            content=md,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={name}.md"},
        )
    else:
        from agent_registry import export_single_agent
        data = export_single_agent(agent_type, name)
        if data is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return data


# ── Team Presets ───────────────────────────────────────────────────────────

@app.get("/api/teams")
async def api_list_teams():
    from agent_registry import get_teams
    return {"teams": get_teams()}

@app.post("/api/teams")
async def api_create_team(request: Request):
    from agent_registry import create_team
    spec = await request.json()
    try:
        team = create_team(spec)
        return {"ok": True, "team": team}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# NOTE: Preset routes must be registered BEFORE parameterized /api/teams/{name}
@app.get("/api/teams/presets")
async def api_list_presets():
    from agent_registry import list_presets
    return {"presets": list_presets()}

@app.post("/api/teams/presets/{preset_name}/apply")
async def api_apply_preset(preset_name: str, request: Request):
    from agent_registry import apply_preset
    body = await request.json()
    team_name = body.get("team_name", "").strip()
    if not team_name:
        raise HTTPException(status_code=400, detail="team_name is required")
    try:
        team = apply_preset(preset_name, team_name)
        return {"ok": True, "team": team}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/teams/{name}")
async def api_delete_team(name: str):
    from agent_registry import delete_team
    ok = delete_team(name)
    if not ok:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"ok": True}


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
