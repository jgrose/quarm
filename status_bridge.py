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
_log_lines:       deque[str]       = deque(maxlen=80)
_project:         str             = "NORT"
_active_reviewer: str | None      = None
_session_id:      str             = ""

# Rosters — set once at plan-parse time, sent in every payload
_sub_agents:  list[dict] = []   # [{"name": "backend_engineer", "title": "Backend Engineer"}, ...]
_managers:    list[dict] = []   # [{"name": "eng_director",     "title": "Engineering Director"}, ...]
_reviewers:   list[dict] = []   # [{"name": "security_engineer","title": "Senior Security Engineer"}, ...]

# File attention heatmap — tracks read/write operations across agents
_files_touched: dict[str, dict] = {}  # path → {reads: int, writes: int, agents: set}

# Transcript — ordered log of agent communications for replay
_transcript: deque[dict] = deque(maxlen=200)


# ── Registration (called once from orchestrator at startup) ───────────────────

def set_project(name: str):
    global _project
    _project = name


def set_active_reviewer(name: str | None):
    global _active_reviewer
    _active_reviewer = name


def set_session_id(sid: str):
    global _session_id
    _session_id = sid


def record_file_touch(path: str, operation: str, agent: str):
    """Record a file read/write for the file attention heatmap."""
    with _state_lock:
        if path not in _files_touched:
            _files_touched[path] = {"reads": 0, "writes": 0, "agents": set()}
        if operation == "read":
            _files_touched[path]["reads"] += 1
        elif operation == "write":
            _files_touched[path]["writes"] += 1
        _files_touched[path]["agents"].add(agent)


def add_transcript_entry(role: str, content: str, agent: str = "", task_id: str = ""):
    """Add a message to the transcript for replay."""
    with _state_lock:
        _transcript.append({
            "role": role,
            "content": content[:2000],
            "agent": agent,
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


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
    global _sub_agents, _managers, _reviewers

    _sub_agents = [
        {"name": a["name"], "title": _title_from(a)}
        for a in sub_agents
    ]
    _managers = [
        {"name": m["name"], "title": m.get("title", m["name"])}
        for m in managers
    ]
    _reviewers = [
        {"name": r["name"], "title": r.get("title", r["name"])}
        for r in reviewers
    ]


def _title_from(agent: dict) -> str:
    """Derive a display title from an agent dict."""
    # Use 'title' if present, else prettify the name
    if "title" in agent and agent["title"]:
        return agent["title"]
    return agent["name"].replace("_", " ").title()


def log_event(msg: str):
    _log_lines.append(msg)


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
    tasks = state.get("tasks", [])
    payload = {
        # ── Identity ──────────────────────────────────────────────────
        "project":         _project,
        "session_id":      _session_id,
        # ── Rosters (dynamic — drives UI room mapping + labels) ───────
        "sub_agents":      _sub_agents,
        "managers":        _managers,
        "reviewers":       _reviewers,
        # ── Live state ────────────────────────────────────────────────
        "phase":           state.get("phase", "dispatch"),
        "active_task_id":  state.get("active_task_id"),
        "active_reviewer": _active_reviewer,
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
        "log":              list(_log_lines),
        "transcript":       list(_transcript)[-50:],  # last 50 entries
        "files_touched": [
            {"path": p, "reads": d["reads"], "writes": d["writes"], "agents": list(d["agents"])}
            for p, d in _files_touched.items()
        ],
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }
    t = threading.Thread(target=_post, args=(payload,), daemon=True)
    t.start()
