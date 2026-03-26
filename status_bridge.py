"""
status_bridge.py
================
Pushes orchestrator state to serve.py via a fire-and-forget background POST.
Now includes full agent/manager/reviewer rosters so the UI can build all
room mappings dynamically — zero manual config required.

Environment:
  QUARM_SERVER   (default: http://localhost:8000)
  QUARM_SECRET   optional shared secret header
"""

import json
import threading
import os
import logging
from datetime import datetime, timezone

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request as _urllib
    _HAS_REQUESTS = False

SERVER_URL = os.environ.get("QUARM_SERVER", "http://localhost:8000").rstrip("/")
UPDATE_URL = f"{SERVER_URL}/update"
SECRET     = os.environ.get("QUARM_SECRET", "")
MAX_LOG    = 80

log = logging.getLogger("quarm.bridge")

# ── Internal state ────────────────────────────────────────────────────────────

_log_lines:       list[str]       = []
_project:         str             = "QUARM"
_active_reviewer: str | None      = None

# Rosters — set once at plan-parse time, sent in every payload
_sub_agents:  list[dict] = []   # [{"name": "backend_engineer", "title": "Backend Engineer"}, ...]
_managers:    list[dict] = []   # [{"name": "eng_director",     "title": "Engineering Director"}, ...]
_reviewers:   list[dict] = []   # [{"name": "security_engineer","title": "Senior Security Engineer"}, ...]


# ── Registration (called once from orchestrator at startup) ───────────────────

def set_project(name: str):
    global _project
    _project = name


def set_active_reviewer(name: str | None):
    global _active_reviewer
    _active_reviewer = name


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
    if len(_log_lines) > MAX_LOG:
        _log_lines.pop(0)


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
            }
            for t in tasks
        ],
        "results_count":    len(state.get("results", {})),
        "total_tasks":      len(tasks),
        "tokens_used":      state.get("tokens_used", 0),
        "last_verdict":     state.get("last_verdict"),
        "synthesis_report": state.get("synthesis_report", ""),
        "log":              list(_log_lines),
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }
    t = threading.Thread(target=_post, args=(payload,), daemon=True)
    t.start()
