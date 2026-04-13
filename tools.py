"""
tools.py — Tool registry and execution engine for NORT agents.
Maps tool name strings from plan.md to LangChain tool functions.
Supports hybrid approval: read-only tools auto-execute, write tools require human approval.
"""

import os
import json
import time
import subprocess
import logging
import threading
import tempfile
import shlex
import resource
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from status_bridge import record_file_touch, log_event

log = logging.getLogger("nort.tools")

PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
QUESTIONS_LOG_DIR = PROJECT_DIR / "plans"

SANDBOX_MODE = os.environ.get("NORT_SANDBOX_MODE", "subprocess").lower()
_SENSITIVE_ENV_PATTERNS = {"API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "OPENAI", "ANTHROPIC", "AWS_"}

# ── Approval system ─────────────────────────────────────────────────────────

# Tools that require human approval before execution
APPROVAL_REQUIRED = {"execute_code"}

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
            f"http://localhost:{os.environ.get('NORT_PORT', os.environ.get('QUARM_PORT', '8000'))}/update",
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
            f"http://localhost:{os.environ.get('NORT_PORT', os.environ.get('QUARM_PORT', '8000'))}/update",
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


# ── Operator guidance (push nudges into a running task) ─────────────────────
#
# Inverse of the ask_human pattern: the operator pushes a message that the
# running agent will see on its next LLM turn. Keyed by (plan_id, task_id) so
# guidance doesn't leak across plans.

_guidance_lock = threading.Lock()
_guidance_queue: dict[str, list[str]] = {}  # "{plan_id}:{task_id}" → [messages]


def _guidance_key(plan_id: str, task_id: str) -> str:
    return f"{plan_id or ''}:{task_id or ''}"


def _broadcast_guidance(payload: dict) -> None:
    import json as _json
    import urllib.request
    try:
        req = urllib.request.Request(
            f"http://localhost:{os.environ.get('NORT_PORT', os.environ.get('QUARM_PORT', '8000'))}/update",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def queue_guidance(plan_id: str, task_id: str, message: str) -> int:
    """Append operator guidance for the agent working on this task.
    Returns the new pending count. Safe to call even if the task is no longer running —
    the message simply stays queued until drained (or forever)."""
    message = (message or "").strip()
    if not message:
        return 0
    key = _guidance_key(plan_id, task_id)
    with _guidance_lock:
        _guidance_queue.setdefault(key, []).append(message)
        pending = len(_guidance_queue[key])
    log.info(f"[GUIDANCE] queued for {key}: {message[:100]}")
    _broadcast_guidance({
        "type": "guidance_queued",
        "plan_id": plan_id,
        "task_id": task_id,
        "pending": pending,
        "message_preview": message[:200],
    })
    return pending


def drain_guidance(plan_id: str, task_id: str) -> list[str]:
    """Pop all pending guidance messages for a task. Orchestrator calls this
    between LLM turns to fold operator nudges into the next agent invocation."""
    key = _guidance_key(plan_id, task_id)
    with _guidance_lock:
        msgs = _guidance_queue.pop(key, [])
    if msgs:
        _broadcast_guidance({
            "type": "guidance_consumed",
            "plan_id": plan_id,
            "task_id": task_id,
            "count": len(msgs),
        })
    return msgs


def peek_guidance(plan_id: str, task_id: str) -> list[str]:
    """Non-destructive read of pending guidance — for the UI's 'N nudges pending' badge."""
    key = _guidance_key(plan_id, task_id)
    with _guidance_lock:
        return list(_guidance_queue.get(key, []))


# ── Human-input (ask_human) system ──────────────────────────────────────────
#
# Parallel to the approval system: an agent calls `ask_human(question)`, which
# blocks the agent's thread until a human submits an answer through the dashboard
# (or a timeout elapses if the plan's policy is "timeout").

QUESTION_TIMEOUT_SENTINEL = "[NO HUMAN RESPONSE — PROCEED WITH BEST JUDGMENT]"

_pending_questions: dict[str, threading.Event] = {}
_question_answers: dict[str, str] = {}
_question_details: dict[str, dict] = {}
_questions_lock = threading.Lock()

# plan_id → {"policy": "block"|"timeout", "timeout_s": int}
_plan_policies: dict[str, dict] = {}


def _append_question_log(plan_id: str, entry: dict) -> None:
    """Atomic-append one JSON line to plans/{plan_id}_questions.jsonl."""
    if not plan_id:
        return
    try:
        QUESTIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = QUESTIONS_LOG_DIR / f"{plan_id}_questions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.debug(f"[ask_human] JSONL log write failed: {exc}")


def _broadcast_questions_snapshot() -> None:
    """POST a questions_snapshot event to serve.py for WS fan-out."""
    try:
        import urllib.request
        with _questions_lock:
            pending = [
                {
                    "id": k,
                    "plan_id": v.get("plan_id", ""),
                    "agent": v.get("agent", ""),
                    "task_id": v.get("task_id", ""),
                    "question": (v.get("question") or "")[:2000],
                    "context": (v.get("context") or "")[:2000],
                    "received_at": v.get("received_at", 0),
                }
                for k, v in _question_details.items()
            ]
        payload = json.dumps({"type": "questions_snapshot", "pending": pending}).encode()
        port = os.environ.get("NORT_PORT", os.environ.get("QUARM_PORT", "8000"))
        req = urllib.request.Request(
            f"http://localhost:{port}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # fire-and-forget


def set_plan_policy(plan_id: str, policy: str = "block", timeout_s: int = 900) -> None:
    """Set the human-input policy for a plan. Called at plan submission and on runtime edits."""
    if policy not in ("block", "timeout"):
        raise ValueError(f"Unknown policy {policy!r}; expected 'block' or 'timeout'")
    _plan_policies[plan_id] = {"policy": policy, "timeout_s": int(timeout_s)}


def _get_plan_policy(plan_id: str) -> dict:
    """Return the stored policy for a plan, defaulting to 'block' if none set."""
    return _plan_policies.get(plan_id, {"policy": "block", "timeout_s": 900})


def request_question(tc_id: str, question: str, context: str = "",
                      agent: str = "", task_id: str = "", plan_id: str = "") -> str:
    """Block until a human answers. Returns the answer or the timeout sentinel."""
    event = threading.Event()
    with _questions_lock:
        _pending_questions[tc_id] = event
        _question_details[tc_id] = {
            "question": question, "context": context,
            "agent": agent, "task_id": task_id, "plan_id": plan_id,
            "received_at": int(time.time()),
        }
    _append_question_log(plan_id, {
        "ts": int(time.time()),
        "type": "request",
        "id": tc_id,
        "agent": agent,
        "task_id": task_id,
        "question": question,
    })
    log.info(f"[ASK_HUMAN] Waiting for human answer: {question[:120]}")
    # Broadcast to UI via serve.py
    import json as _json
    import urllib.request
    try:
        payload = _json.dumps({
            "type": "question_request",
            "id": tc_id,
            "question": question[:2000],
            "context": (context or "")[:2000],
            "agent": agent,
            "task_id": task_id,
            "plan_id": plan_id,
            "received_at": _question_details[tc_id].get("received_at", int(time.time())),
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{os.environ.get('NORT_PORT', os.environ.get('QUARM_PORT', '8000'))}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # fire-and-forget
    _broadcast_questions_snapshot()

    policy = _get_plan_policy(plan_id)
    if policy["policy"] == "timeout":
        got = event.wait(timeout=policy["timeout_s"])
    else:
        event.wait()
        got = True

    with _questions_lock:
        answer = _question_answers.pop(tc_id, None)
        _pending_questions.pop(tc_id, None)
        _question_details.pop(tc_id, None)
    if not got or answer is None:
        _append_question_log(plan_id, {
            "ts": int(time.time()),
            "type": "timeout",
            "id": tc_id,
        })
        _broadcast_questions_snapshot()
        return QUESTION_TIMEOUT_SENTINEL
    _broadcast_questions_snapshot()
    return answer


def resolve_question(tc_id: str, answer: str) -> None:
    """Called from serve.py when the human submits an answer."""
    with _questions_lock:
        _question_answers[tc_id] = answer
        details = dict(_question_details.get(tc_id, {}))
        event = _pending_questions.get(tc_id)
    _append_question_log(details.get("plan_id", ""), {
        "ts": int(time.time()),
        "type": "resolve",
        "id": tc_id,
        "answer": answer,
    })
    if event:
        event.set()
    import json as _json
    import urllib.request
    try:
        payload = _json.dumps({
            "type": "question_resolved",
            "id": tc_id,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{os.environ.get('NORT_PORT', os.environ.get('QUARM_PORT', '8000'))}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass
    _broadcast_questions_snapshot()


def get_pending_questions() -> list[dict]:
    """Get all pending question requests for the dashboard."""
    with _questions_lock:
        return [
            {"id": k, **v}
            for k, v in _question_details.items()
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


def _path_is_within(target: Path, allowed_base: Path) -> bool:
    """Check that target path resolves within allowed_base (prevents path traversal)."""
    try:
        target.resolve().relative_to(allowed_base.resolve())
        return True
    except ValueError:
        return False


# ── Tool definitions ────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top 5 results with titles, URLs, and snippets."""
    from tools_web import web_search as _ws
    return _ws(query)


@tool
def browse_url(url: str) -> str:
    """Load a web page using headless Chromium and return its content as markdown. Handles JavaScript-rendered pages."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return "Blocked: file:// URLs are not allowed."
    _blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]", "169.254.169.254"}
    hostname = (parsed.hostname or "").lower()
    if hostname in _blocked_hosts or hostname == "::1":
        return f"Blocked: requests to {hostname} are not allowed."
    from tools_web import browse_url as _bu
    return _bu(url)


@tool
def rag_search(query: str) -> str:
    """Search the NORT knowledge base for relevant information from past projects, artifacts, and web content. Returns the top 5 most relevant text chunks with their sources."""
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
    """Store text in the NORT knowledge base for use in current and future projects. Provide comma-separated tags for categorization."""
    from rag import ingest_text
    truncated = False
    if len(text) > 100_000:
        text = text[:100_000]
        truncated = True
    ctx = _ctx()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    n = ingest_text(text, source=f"agent:{ctx['agent']}", content_type="manual",
                    plan_id=ctx["plan_id"], task_id=ctx["task_id"],
                    agent=ctx["agent"], tags=tag_list)
    msg = f"Stored {n} chunk(s) in the knowledge base."
    if truncated:
        msg += " Note: input was truncated to 100,000 characters."
    return msg


@tool
def download_artifact(url: str) -> str:
    """Download content from a URL, save it as a file, and store it in the knowledge base for future reference."""
    import requests
    from rag import ingest_url
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        content = resp.text[:50000]
        from urllib.parse import urlparse
        filename = Path(urlparse(url).path.split("/")[-1]).name
        if not filename:
            filename = "download.txt"
        artifacts = _artifacts_path()
        path = artifacts / filename
        if not _path_is_within(path, artifacts):
            return "Access denied: filename would escape artifacts directory."
        path.write_text(content)
        ctx = _ctx()
        n = ingest_url(url, content, plan_id=ctx["plan_id"],
                       task_id=ctx["task_id"], agent=ctx["agent"])
        return f"Downloaded {len(content)} chars to {path.name}, stored {n} chunks in knowledge base."
    except Exception as e:
        return f"Download failed: {e}"


@tool
def read_file(path: str) -> str:
    """Read a file by path relative to the project root. If a directory path is given, lists its contents instead."""
    target = PROJECT_DIR / path
    if not _path_is_within(target, PROJECT_DIR):
        return f"Access denied: {path} is outside the project directory."
    if not target.exists():
        return f"File not found: {path}"
    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"{path} is a directory. Contents:\n" + "\n".join(entries)
    try:
        content = target.read_text()[:20000]
        record_file_touch(path, "read", _ctx().get("agent", ""))
        return content
    except Exception as e:
        log.warning(f"read_file error for {target}: {e}")
        return f"Error reading {path}: operation failed."


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the artifacts directory. Creates directories as needed."""
    artifacts = _artifacts_path()
    target = artifacts / path
    if not _path_is_within(target, artifacts):
        return f"Access denied: {path} would escape the artifacts directory."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        record_file_touch(str(target.relative_to(PROJECT_DIR)), "write", _ctx().get("agent", ""))
        return f"Written {len(content)} chars to {target.relative_to(PROJECT_DIR)}"
    except Exception as e:
        log.warning(f"write_file error for {target}: {e}")
        return f"Error writing {path}: operation failed."


# ── Sandbox helpers for execute_code ────────────────────────────────────────

def _format_exec_output(stdout: str, stderr: str) -> str:
    """Format subprocess stdout/stderr into a single result string."""
    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += f"\nSTDERR:\n{stderr}"
    return output.strip() or "(no output)"


_cached_sanitized_env: dict[str, str] | None = None


def _sanitized_env() -> dict[str, str]:
    """Return a copy of os.environ with sensitive variables stripped. Cached after first call."""
    global _cached_sanitized_env
    if _cached_sanitized_env is not None:
        return dict(_cached_sanitized_env)
    safe_keys = {"PYTHONPATH", "PATH", "HOME", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE"}
    env = {}
    for k, v in os.environ.items():
        if k in safe_keys:
            env[k] = v
        elif not any(pat in k.upper() for pat in _SENSITIVE_ENV_PATTERNS):
            env[k] = v
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    _cached_sanitized_env = dict(env)
    return env


def _preexec_limits():
    """Set resource limits for sandboxed subprocess (512 MB memory, 30s CPU)."""
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))


def _execute_none(code: str, artifacts_path: Path) -> str:
    """No sandbox -- bare subprocess, full env. For trusted environments."""
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=30,
        cwd=str(artifacts_path),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return _format_exec_output(result.stdout, result.stderr)


def _execute_subprocess(code: str, artifacts_path: Path) -> str:
    """Sandboxed subprocess with temp dir, stripped env, resource limits, and optional network isolation."""
    tmpdir = tempfile.mkdtemp(prefix="nort_sandbox_")
    try:
        env = _sanitized_env()
        try:
            proc = subprocess.Popen(
                ["unshare", "--net", "--map-root-user", "python3", "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=tmpdir, env=env,
                preexec_fn=_preexec_limits,
            )
            stdout, stderr = proc.communicate(timeout=30)
        except (FileNotFoundError, PermissionError, OSError):
            proc = subprocess.Popen(
                ["python3", "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=tmpdir, env=env,
                preexec_fn=_preexec_limits,
            )
            stdout, stderr = proc.communicate(timeout=30)
        return _format_exec_output(stdout, stderr)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _execute_docker(code: str, artifacts_path: Path) -> str:
    """Docker-based sandbox with no network, memory/cpu/pid limits."""
    cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "--memory=256m",
        "--cpus=0.5",
        "--pids-limit=50",
        "-v", f"{artifacts_path}:/work:rw",
        "-w", "/work",
        "python:3.12-slim",
        "python3", "-c", code,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        return _format_exec_output(result.stdout, result.stderr)
    except FileNotFoundError:
        return "Error: docker is not installed or not in PATH. Set NORT_SANDBOX_MODE=subprocess to use process-based sandboxing."
    except subprocess.CalledProcessError as e:
        return f"Error: docker execution failed ({e}). Set NORT_SANDBOX_MODE=subprocess to use process-based sandboxing."


@tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandboxed subprocess with a 30-second timeout. Returns stdout and stderr."""
    artifacts_path = _artifacts_path()
    try:
        if SANDBOX_MODE == "none":
            return _execute_none(code, artifacts_path)
        elif SANDBOX_MODE == "docker":
            return _execute_docker(code, artifacts_path)
        else:
            return _execute_subprocess(code, artifacts_path)
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


# ── ask_human tool ──────────────────────────────────────────────────────────

@tool
def ask_human(question: str, context: str = "") -> str:
    """Ask the human operator a clarifying question and wait for their answer.

    Use this only when ambiguity or missing information would materially change
    your deliverable. Do not use it for minor choices — make those yourself and
    flag the assumption in your output.

    Args:
        question: The specific question for the human to answer.
        context: Optional extra context explaining why the answer matters.

    Returns:
        The human's typed answer, or a sentinel string if the plan's policy
        timed out without a response.
    """
    ctx = _ctx()
    import uuid
    tc_id = f"ask-{uuid.uuid4().hex[:12]}"
    return request_question(
        tc_id=tc_id,
        question=question,
        context=context,
        agent=ctx.get("agent", ""),
        task_id=ctx.get("task_id", ""),
        plan_id=ctx.get("plan_id", ""),
    )


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
    "ask_human": ask_human,
    # Aliases for plan.md compatibility
    "search": web_search,
    "browse": browse_url,
    "analyze_data": execute_code,
    "design_ui": write_file,
    "reason": rag_search,
}


def load_default_tools() -> list[str]:
    """Load default_tools from config.json. Returns empty list if not configured."""
    try:
        import json
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            return json.load(f).get("default_tools", [])
    except Exception:
        return []


def init_mcp_tools():
    """Discover MCP tools and register them in TOOL_REGISTRY. Called once at startup."""
    try:
        from mcp_tool_wrapper import register_mcp_tools_in_registry
        count = register_mcp_tools_in_registry(TOOL_REGISTRY)
        if count:
            log.info(f"Registered {count} MCP tools into TOOL_REGISTRY")
    except ImportError:
        pass  # mcp package not installed
    except Exception as e:
        log.warning(f"MCP tool init failed: {e}")


def get_tools(tool_names: list[str], allowed_tools: list[str] = None,
              include_defaults: bool = True) -> list:
    """Get LangChain tool objects for the given tool name strings.
    Merges with default_tools from config.json unless disabled.
    Pass tool_names=["none"] to disable all tools including defaults.
    If allowed_tools is non-empty, only include tools in that list."""
    # "none" sentinel -- agent opts out of all tools
    if tool_names == ["none"]:
        return []

    # Merge with defaults
    if include_defaults:
        defaults = load_default_tools()
        merged = list(dict.fromkeys(defaults + tool_names))
    else:
        merged = tool_names

    tools = []
    seen = set()
    allowed_set = set(t.strip().lower() for t in allowed_tools) if allowed_tools else None
    for name in merged:
        name = name.strip().lower()
        if name in TOOL_REGISTRY and name not in seen:
            if allowed_set and name not in allowed_set:
                log_event(f"  [SECURITY] Tool '{name}' not in allowed_tools — skipped")
                continue
            tools.append(TOOL_REGISTRY[name])
            seen.add(name)
    return tools


# ── Tool execution with approval ─────────────────────────────────────────────

def execute_tool_call(tool_call: dict, tools: list, auto_approve_all: bool = False,
                      allowed_tools: list[str] = None) -> str:
    """Execute a single tool call, with approval check for dangerous tools."""
    name = tool_call["name"]
    args = tool_call["args"]
    tc_id = tool_call["id"]

    # Find the matching tool
    tool_fn = next((t for t in tools if t.name == name), None)
    if not tool_fn:
        return f"Unknown tool: {name}"

    # Allowlist enforcement (defense-in-depth — tools are also filtered in get_tools)
    if allowed_tools:
        allowed_set = set(t.strip().lower() for t in allowed_tools)
        if name.lower() not in allowed_set:
            log_event(f"  [SECURITY] Agent attempted disallowed tool '{name}' — blocked")
            return f"Tool '{name}' is not in this agent's allowed tools list."

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
