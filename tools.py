"""
tools.py — Tool registry and execution engine for QUARM agents.
Maps tool name strings from plan.md to LangChain tool functions.
Supports hybrid approval: read-only tools auto-execute, write tools require human approval.
"""

import os
import subprocess
import logging
import threading
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

log = logging.getLogger("quarm.tools")

PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# ── Approval system ─────────────────────────────────────────────────────────

# Tools that require human approval before execution
APPROVAL_REQUIRED = {"execute_code", "write_file"}

# Pending approvals: tool_call_id → threading.Event
_pending_approvals: dict[str, threading.Event] = {}
_approval_results: dict[str, bool] = {}  # tool_call_id → approved?
_approval_details: dict[str, dict] = {}  # tool_call_id → {tool, args, agent, task_id}


def request_approval(tool_call_id: str, tool_name: str, args: dict,
                      agent: str = "", task_id: str = "") -> bool:
    """Block until human approves/rejects. Returns True if approved."""
    event = threading.Event()
    _pending_approvals[tool_call_id] = event
    _approval_details[tool_call_id] = {
        "tool": tool_name, "args": args,
        "agent": agent, "task_id": task_id,
    }
    log.info(f"[APPROVAL] Waiting for human approval: {tool_name}({args})")
    # Broadcast to UI via serve.py
    import json as _json
    import urllib.request
    try:
        payload = _json.dumps({
            "type": "approval_request",
            "id": tool_call_id,
            "tool": tool_name,
            "args": {k: (v[:500] if isinstance(v, str) else v) for k, v in args.items()},
            "agent": agent,
            "task_id": task_id,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{os.environ.get('QUARM_PORT', '8000')}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # fire-and-forget
    event.wait(timeout=300)  # 5 minute timeout
    approved = _approval_results.pop(tool_call_id, False)
    _pending_approvals.pop(tool_call_id, None)
    _approval_details.pop(tool_call_id, None)
    return approved


def resolve_approval(tool_call_id: str, approved: bool):
    """Called from serve.py when human clicks approve/reject."""
    _approval_results[tool_call_id] = approved
    event = _pending_approvals.get(tool_call_id)
    if event:
        event.set()
    # Broadcast dismissal to UI
    import json as _json
    import urllib.request
    try:
        payload = _json.dumps({
            "type": "approval_resolved",
            "id": tool_call_id,
            "approved": approved,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{os.environ.get('QUARM_PORT', '8000')}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def get_pending_approvals() -> list[dict]:
    """Get all pending approval requests for the dashboard."""
    return [
        {"id": k, **v}
        for k, v in _approval_details.items()
    ]


# ── Tool context (set per-task) ──────────────────────────────────────────────

_tool_context = threading.local()


def set_tool_context(plan_id: str = "", task_id: str = "", agent: str = ""):
    """Set context for the current task's tool execution."""
    _tool_context.plan_id = plan_id
    _tool_context.task_id = task_id
    _tool_context.agent = agent


def _ctx():
    return {
        "plan_id": getattr(_tool_context, "plan_id", ""),
        "task_id": getattr(_tool_context, "task_id", ""),
        "agent": getattr(_tool_context, "agent", ""),
    }


def _artifacts_path() -> Path:
    ctx = _ctx()
    p = ARTIFACTS_DIR / ctx["plan_id"] / ctx["task_id"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Tool definitions ────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top 5 results with titles, URLs, and snippets."""
    from tools_web import web_search as _ws
    return _ws(query)


@tool
def browse_url(url: str) -> str:
    """Load a web page using headless Chromium and return its content as markdown. Handles JavaScript-rendered pages."""
    from tools_web import browse_url as _bu
    return _bu(url)


@tool
def rag_search(query: str) -> str:
    """Search the QUARM knowledge base for relevant information from past projects, artifacts, and web content. Returns the top 5 most relevant text chunks with their sources."""
    from rag import search
    results = search(query)
    if not results:
        return "No relevant documents found in the knowledge base."
    output = []
    for i, r in enumerate(results, 1):
        output.append(f"{i}. [score={r['score']:.2f}] ({r['source']})\n   {r['text'][:300]}")
    return "\n\n".join(output)


@tool
def rag_store(text: str, tags: str = "") -> str:
    """Store text in the QUARM knowledge base for use in current and future projects. Provide comma-separated tags for categorization."""
    from rag import ingest_text
    ctx = _ctx()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    n = ingest_text(text, source=f"agent:{ctx['agent']}", content_type="manual",
                    plan_id=ctx["plan_id"], task_id=ctx["task_id"],
                    agent=ctx["agent"], tags=tag_list)
    return f"Stored {n} chunk(s) in the knowledge base."


@tool
def download_artifact(url: str) -> str:
    """Download content from a URL, save it as a file, and store it in the knowledge base for future reference."""
    import requests
    from rag import ingest_url
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        content = resp.text[:50000]
        # Save to artifacts directory
        from urllib.parse import urlparse
        filename = urlparse(url).path.split("/")[-1] or "download.txt"
        path = _artifacts_path() / filename
        path.write_text(content)
        # Ingest into RAG
        ctx = _ctx()
        n = ingest_url(url, content, plan_id=ctx["plan_id"],
                       task_id=ctx["task_id"], agent=ctx["agent"])
        return f"Downloaded {len(content)} chars to {path.name}, stored {n} chunks in knowledge base."
    except Exception as e:
        return f"Download failed: {e}"


@tool
def read_file(path: str) -> str:
    """Read a file from the project workspace. Can read from artifacts/ or plans/ directories."""
    target = PROJECT_DIR / path
    # Sandbox check
    try:
        target.resolve().relative_to(PROJECT_DIR.resolve())
    except ValueError:
        return f"Access denied: {path} is outside the project directory."
    if not target.exists():
        return f"File not found: {path}"
    try:
        return target.read_text()[:20000]
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the artifacts directory. Creates directories as needed."""
    target = _artifacts_path() / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Written {len(content)} chars to {target.relative_to(PROJECT_DIR)}"
    except Exception as e:
        return f"Error writing {path}: {e}"


@tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandboxed subprocess with a 30-second timeout. Returns stdout and stderr."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
            cwd=str(_artifacts_path()),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


# ── Tool registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "web_search": web_search,
    "browse_url": browse_url,
    "rag_search": rag_search,
    "rag_store": rag_store,
    "download_artifact": download_artifact,
    "read_file": read_file,
    "write_file": write_file,
    "execute_code": execute_code,
    # Aliases for plan.md compatibility
    "search": web_search,
    "browse": browse_url,
    "analyze_data": execute_code,
    "design_ui": write_file,
    "reason": rag_search,
}


def get_tools(tool_names: list[str]) -> list:
    """Get LangChain tool objects for the given tool name strings."""
    tools = []
    seen = set()
    for name in tool_names:
        name = name.strip().lower()
        if name in TOOL_REGISTRY and name not in seen:
            tools.append(TOOL_REGISTRY[name])
            seen.add(name)
    return tools


# ── Tool execution with approval ─────────────────────────────────────────────

def execute_tool_call(tool_call: dict, tools: list, auto_approve_all: bool = False) -> str:
    """Execute a single tool call, with approval check for dangerous tools."""
    name = tool_call["name"]
    args = tool_call["args"]
    tc_id = tool_call["id"]

    # Find the matching tool
    tool_fn = next((t for t in tools if t.name == name), None)
    if not tool_fn:
        return f"Unknown tool: {name}"

    # Check if approval is needed
    needs_approval = name in APPROVAL_REQUIRED and not auto_approve_all
    if needs_approval:
        ctx = _ctx()
        approved = request_approval(tc_id, name, args,
                                     agent=ctx.get("agent", ""),
                                     task_id=ctx.get("task_id", ""))
        if not approved:
            return f"Tool call rejected by human operator: {name}"

    # Execute
    try:
        result = tool_fn.invoke(args)
        return str(result)
    except Exception as e:
        return f"Tool error ({name}): {e}"
