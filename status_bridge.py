"""
status_bridge.py
================
Pushes orchestrator state to serve.py via a fire-and-forget background POST.
Now includes full agent/manager/reviewer rosters so the UI can build all
room mappings dynamically — zero manual config required.

Environment:
  NORT_SERVER    (default: http://localhost:8000)  [QUARM_SERVER also accepted]
  NORT_SECRET    optional shared secret header     [QUARM_SECRET also accepted]
"""

import json
import threading
import os
import logging
from collections import deque
from datetime import datetime, timezone

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request as _urllib
    _HAS_REQUESTS = False

SERVER_URL = os.environ.get("NORT_SERVER", os.environ.get("QUARM_SERVER", "http://localhost:8000")).rstrip("/")
UPDATE_URL = f"{SERVER_URL}/update"
SECRET     = os.environ.get("NORT_SECRET", os.environ.get("QUARM_SECRET", ""))
MAX_LOG    = 80

log = logging.getLogger("nort.bridge")

# ── Internal state ────────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_tls = threading.local()  # thread-local session_id for concurrent plan runs

# Per-session state — keyed by session_id
_session_logs:        dict[str, deque] = {}   # session_id → deque[str]
_session_transcripts: dict[str, deque] = {}   # session_id → deque[dict]
_session_files:       dict[str, dict]  = {}   # session_id → {path: {reads, writes, agents}}
_session_projects:    dict[str, str]   = {}   # session_id → project name
_session_reviewers:   dict[str, str | None] = {}  # session_id → active reviewer
_session_rosters:     dict[str, dict]  = {}   # session_id → {sub_agents, managers, reviewers}

# Fallback globals for backward compat (single-session usage)
_project:         str             = "NORT"
_active_reviewer: str | None      = None


def _get_sid() -> str:
    return getattr(_tls, 'session_id', '')


# ── Registration (called once from orchestrator at startup) ───────────────────

def set_project(name: str):
    global _project
    _project = name
    sid = _get_sid()
    if sid:
        with _state_lock:
            _session_projects[sid] = name


def set_active_reviewer(name: str | None):
    global _active_reviewer
    _active_reviewer = name
    sid = _get_sid()
    if sid:
        with _state_lock:
            _session_reviewers[sid] = name


def set_session_id(sid: str):
    _tls.session_id = sid
    with _state_lock:
        if sid not in _session_logs:
            _session_logs[sid] = deque(maxlen=80)
        if sid not in _session_transcripts:
            _session_transcripts[sid] = deque(maxlen=200)
        if sid not in _session_files:
            _session_files[sid] = {}
        if sid not in _session_rosters:
            _session_rosters[sid] = {"sub_agents": [], "managers": [], "reviewers": []}


def record_file_touch(path: str, operation: str, agent: str):
    """Record a file read/write for the file attention heatmap."""
    sid = _get_sid()
    with _state_lock:
        ft = _session_files.get(sid, {}) if sid else {}
        if path not in ft:
            ft[path] = {"reads": 0, "writes": 0, "agents": set()}
        if operation == "read":
            ft[path]["reads"] += 1
        elif operation == "write":
            ft[path]["writes"] += 1
        ft[path]["agents"].add(agent)
        if sid:
            _session_files[sid] = ft


def add_transcript_entry(role: str, content: str, agent: str = "", task_id: str = ""):
    """Add a message to the transcript for replay."""
    sid = _get_sid()
    with _state_lock:
        entry = {
            "role": role,
            "content": content[:2000],
            "agent": agent,
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if sid and sid in _session_transcripts:
            _session_transcripts[sid].append(entry)


def register_rosters(
    sub_agents:  list[dict],
    managers:    list[dict],
    reviewers:   list[dict],
):
    """
    Call this once after parse_plan() so the bridge knows the full cast.
    sub_agents / managers / reviewers are the raw __dict__ lists from the
    parsed dataclass objects — we only need name + title/description for the UI.
    """
    sa = [{"name": a["name"], "title": _title_from(a)} for a in sub_agents]
    mg = [{"name": m["name"], "title": m.get("title", m["name"])} for m in managers]
    rv = [{"name": r["name"], "title": r.get("title", r["name"])} for r in reviewers]

    sid = _get_sid()
    with _state_lock:
        if sid:
            _session_rosters[sid] = {"sub_agents": sa, "managers": mg, "reviewers": rv}


def _title_from(agent: dict) -> str:
    """Derive a display title from an agent dict."""
    # Use 'title' if present, else prettify the name
    if "title" in agent and agent["title"]:
        return agent["title"]
    return agent["name"].replace("_", " ").title()


def log_event(msg: str):
    sid = _get_sid()
    with _state_lock:
        if sid and sid in _session_logs:
            _session_logs[sid].append(msg)


# ── Push ──────────────────────────────────────────────────────────────────────

def _post(payload: dict):
    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["X-Gauntlet-Secret"] = SECRET
    body = json.dumps(payload).encode()
    try:
        if _HAS_REQUESTS:
            _req.post(UPDATE_URL, data=body, headers=headers, timeout=2)
        else:
            req = _urllib.Request(UPDATE_URL, data=body, headers=headers, method="POST")
            _urllib.urlopen(req, timeout=2)
    except Exception as e:
        log.debug(f"POST failed (server running?): {e}")


def write_status(state: dict):
    """Serialise relevant state and fire a background POST to the WS server."""
    sid = _get_sid()
    tasks = state.get("tasks", [])

    with _state_lock:
        rosters = _session_rosters.get(sid, {"sub_agents": [], "managers": [], "reviewers": []})
        log_lines = list(_session_logs.get(sid, []))
        transcript = list(_session_transcripts.get(sid, deque()))[-50:]
        files = _session_files.get(sid, {})
        files_list = [
            {"path": p, "reads": d["reads"], "writes": d["writes"], "agents": list(d["agents"])}
            for p, d in files.items()
        ]
        project = _session_projects.get(sid, _project)
        reviewer = _session_reviewers.get(sid, _active_reviewer)

    payload = {
        # ── Identity ──────────────────────────────────────────────────
        "project":         project,
        "session_id":      sid,
        # ── Rosters (dynamic — drives UI room mapping + labels) ───────
        "sub_agents":      rosters["sub_agents"],
        "managers":        rosters["managers"],
        "reviewers":       rosters["reviewers"],
        # ── Live state ────────────────────────────────────────────────
        "phase":           state.get("phase", "dispatch"),
        "active_task_id":  state.get("active_task_id"),
        "active_reviewer": reviewer,
        "tasks": [
            {
                "id":             t["id"],
                "title":          t["title"],
                "agent":          t["agent"],
                "status":         t["status"],
                "revision_count": t.get("revision_count", 0),
                "manager_notes":  t.get("manager_notes",  ""),
                "reviewer_notes": t.get("reviewer_notes", ""),
                "last_score":     t.get("last_score", 0),
                "current_model":  t.get("current_model", ""),
                "task_tokens":    t.get("task_tokens", 0),
                "depends_on":     t.get("depends_on", []),
                "result_preview": (t.get("result", "") or "")[:500],
                "tool_calls":     t.get("tool_calls", []),
                "spawned_at":     t.get("spawned_at", ""),
                "completed_at":   t.get("completed_at", ""),
            }
            for t in tasks
        ],
        "results_count":    len(state.get("results", {})),
        "total_tasks":      len(tasks),
        "tokens_used":      state.get("tokens_used", 0),
        "last_verdict":     state.get("last_verdict"),
        "synthesis_report": state.get("synthesis_report", ""),
        "validation":       state.get("validation", {}),
        "coherence_report": state.get("coherence_report", {}),
        "log":              log_lines,
        "transcript":       transcript,
        "files_touched":    files_list,
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }
    t = threading.Thread(target=_post, args=(payload,), daemon=True)
    t.start()
