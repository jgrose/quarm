"""
4-Layer Multi-Agent Orchestrator — Zero-Config Live Edition
============================================================
Registers sub-agent / manager / reviewer rosters with status_bridge
immediately after plan parsing, so the UI dynamically labels every
room without any manual AGENT_TO_ROOM mapping.

Run:
  # Terminal 1
  python serve.py

  # Terminal 2
  python orchestrator.py plan.md

  # Browser
  http://localhost:8000/
"""

import re, json, os, shutil
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Sequence, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from status_bridge import (
    write_status, log_event, set_project,
    set_active_reviewer, register_rosters,
)
from model_config import load_allowed_models
from tracking import track_run_start, track_score, track_run_end
from tools import get_tools, execute_tool_call, set_tool_context
from checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

load_dotenv()
MAX_REVISIONS = 3
DEFAULT_TOLERANCE = 6
_run_id = ""  # set at run() start
_plan_id = ""  # set at run() start, used for checkpointing


def _infer_tags(description: str, extras: list) -> list:
    """Extract relevant tags from a description string."""
    keywords = {"python", "javascript", "react", "node", "api", "database", "sql",
                "frontend", "backend", "fullstack", "devops", "docker", "kubernetes",
                "security", "testing", "documentation", "design", "ui", "ux",
                "html", "css", "infrastructure", "cloud", "aws", "data", "ml",
                "mobile", "ios", "android", "performance", "architecture"}
    desc_words = set(description.lower().split())
    found = list(desc_words & keywords)
    # Add any extras that look like tags
    for e in extras:
        if isinstance(e, str) and len(e) < 30:
            found.append(e.lower().replace(" ", "_"))
    return list(set(found))[:10]  # cap at 10 tags

# ── Model discovery & auto-selection ─────────────────────────────────────────

AVAILABLE_MODELS: list[str] = []
DEFAULT_MODEL = "bedrock-claude-opus-4-6"

# Tier keywords — first match wins, ordered most-capable to least
_TIER_KEYWORDS = {
    "high":   ["opus", "gpt-4o", "nova-premier"],
    "mid":    ["sonnet", "gpt-4o-mini", "nova-pro", "llama-4-maverick"],
    "low":    ["haiku", "nova-lite", "llama3.2-3b", "llama3.2-1b"],
}


def fetch_available_models() -> list[str]:
    """Query the /models endpoint, filter by config, and cache results."""
    global AVAILABLE_MODELS
    try:
        client = OpenAI()
        all_models = sorted(m.id for m in client.models.list().data)
        allowed = load_allowed_models()
        if allowed is not None:
            AVAILABLE_MODELS = [m for m in all_models if m in allowed]
            if not AVAILABLE_MODELS:
                print(f"[WARN] No allowed models found, using all {len(all_models)}")
                AVAILABLE_MODELS = all_models
            else:
                print(f"Allowed models ({len(AVAILABLE_MODELS)}/{len(all_models)}): {AVAILABLE_MODELS}")
        else:
            AVAILABLE_MODELS = all_models
            print(f"Available models ({len(AVAILABLE_MODELS)}): {AVAILABLE_MODELS}")
    except Exception as e:
        print(f"[WARN] Could not fetch models: {e} — using default: {DEFAULT_MODEL}")
        AVAILABLE_MODELS = [DEFAULT_MODEL]
    return AVAILABLE_MODELS


def _tier_for(model_id: str) -> str:
    low = model_id.lower()
    for tier, keywords in _TIER_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return tier
    return "mid"


def _pick_from_tier(target_tier: str) -> str:
    """Pick the first available model matching target tier, else fall back."""
    for m in AVAILABLE_MODELS:
        if _tier_for(m) == target_tier:
            return m
    return AVAILABLE_MODELS[0] if AVAILABLE_MODELS else DEFAULT_MODEL


def auto_select_model(role: str) -> str:
    """Pick a model based on the role of the LLM call."""
    if role == "execute":
        return _pick_from_tier("high")
    elif role in ("review", "synthesis"):
        return _pick_from_tier("mid")
    return _pick_from_tier("mid")


def resolve_model(*preferred: str, role: str = "execute") -> str:
    """Walk a priority list of preferred model names, fall back to auto-select."""
    for p in preferred:
        if p and p in AVAILABLE_MODELS:
            return p
    selected = auto_select_model(role)
    return selected


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SubAgentSpec:
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    model: str = ""

@dataclass
class ManagerSpec:
    name: str
    title: str
    description: str
    expertise_blend: list[str]
    oversees: list[str]
    model: str = ""
    tolerance: int = 0

@dataclass
class ReviewerSpec:
    name: str
    title: str
    description: str
    focus_areas: list[str]
    applies_to: list[str]
    model: str = ""
    tolerance: int = 0

@dataclass
class TaskSpec:
    id: str
    title: str
    agent: str
    description: str
    task_type:  list[str] = field(default_factory=list)
    reviewers:  list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    model: str = ""
    status: str = "pending"
    result: str = ""
    manager_notes:  str = ""
    reviewer_notes: str = ""
    revision_count: int = 0


# ── Built-in specialist reviewers ─────────────────────────────────────────────

BUILTIN_REVIEWERS = [
    ReviewerSpec("security_engineer", "Senior Security Engineer",
        "Review for OWASP Top 10, broken auth, secrets exposure, input validation, least privilege. Think like an attacker.",
        ["OWASP Top 10","auth & secrets","input validation","least privilege","dependency risk"],
        ["code","api","auth","data","config","infrastructure","backend","security"]),
    ReviewerSpec("ux_designer", "Senior UX/UI Designer",
        "Review for WCAG 2.1 AA, visual hierarchy, information architecture, cognitive load, interaction quality.",
        ["WCAG accessibility","visual hierarchy","info architecture","interaction patterns","cognitive load"],
        ["ui","frontend","ux","design","report","dashboard","form","user_flow"]),
    ReviewerSpec("user_tester", "End-User Representative",
        "Review as a non-technical first-time user: clarity, plain language, workflow intuitiveness, value delivered.",
        ["first-use clarity","plain language","workflow intuitiveness","value delivered"],
        ["ui","report","documentation","user_flow","dashboard","api","frontend"]),
    ReviewerSpec("creative_director", "Creative Director",
        "Challenge conventional thinking. Is this the obvious boring solution or something genuinely clever? "
        "Push for innovation, elegance, and delight. Ask: what would make someone say 'that's brilliant'? "
        "FLAG safe, generic, or copy-paste solutions that lack originality or miss creative opportunities.",
        ["innovation","elegance","originality","lateral thinking","user delight","bold alternatives"],
        ["code","api","ui","frontend","ux","report","dashboard","user_flow","backend","documentation"]),
    ReviewerSpec("devils_advocate", "Devil's Advocate",
        "Assume everything is wrong. Find the hidden assumptions, logical flaws, unstated dependencies, "
        "and failure modes nobody mentioned. Ask: what happens when this breaks at 3am? What did they forget? "
        "What looks right but is subtly wrong? Be ruthless but specific — vague skepticism is useless.",
        ["hidden assumptions","logical flaws","edge cases","failure modes","unstated dependencies","silent failures"],
        ["code","api","auth","data","config","infrastructure","backend","security","ui","frontend","user_flow"]),
    ReviewerSpec("performance_engineer", "Performance Engineer",
        "Review for scalability, efficiency, and production readiness. Find N+1 queries, unbounded loops, "
        "memory leaks, missing indexes, chatty APIs, blocking calls in async paths, missing caching, "
        "and anything that will fall over at 10x traffic. Think in terms of p99 latency and cost per request.",
        ["scalability","N+1 queries","memory management","caching","concurrency","latency","cost efficiency"],
        ["code","api","data","backend","infrastructure","config"]),
]


# ── Plan parser ───────────────────────────────────────────────────────────────

def parse_plan(path: str):
    text = open(path).read()

    obj_m = re.search(r"## Objective\s+(.*?)(?=\n##|\Z)", text, re.DOTALL)
    objective = obj_m.group(1).strip() if obj_m else "No objective."

    proj_m = re.search(r"^# PROJECT PLAN: (.+)", text, re.MULTILINE)
    set_project(proj_m.group(1).strip() if proj_m else "QUARM")

    def rx(pat, raw, default=""):
        m = re.search(pat, raw)
        return m.group(1).strip() if m else default

    def rxl(pat, raw):
        m = re.search(pat, raw)
        return [x.strip() for x in m.group(1).split(",")] if m and m.group(1).strip() else []

    sub_agents = [
        SubAgentSpec(
            name=b.group(1).lower(),
            description=rx(r"- description:\s*(.+)", b.group(2)),
            tools=rxl(r"- tools:\s*(.+)", b.group(2)),
            model=rx(r"- model:\s*(.+)", b.group(2)),
        )
        for b in re.finditer(r"### AGENT: (\S+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL)
    ]

    managers = [
        ManagerSpec(
            name=b.group(1).lower(),
            title=rx(r"- title:\s*(.+)", b.group(2), b.group(1)),
            description=rx(r"- description:\s*(.+)", b.group(2)),
            expertise_blend=rxl(r"- expertise_blend:\s*\[(.+?)\]", b.group(2)),
            oversees=rxl(r"- oversees:\s*\[(.+?)\]", b.group(2)),
            model=rx(r"- model:\s*(.+)", b.group(2)),
            tolerance=int(rx(r"- tolerance:\s*(\d+)", b.group(2), "0")),
        )
        for b in re.finditer(r"### MANAGER: (\S+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL)
    ]

    custom = [
        ReviewerSpec(
            name=b.group(1).lower(),
            title=rx(r"- title:\s*(.+)", b.group(2), b.group(1)),
            description=rx(r"- description:\s*(.+)", b.group(2)),
            focus_areas=rxl(r"- focus_areas:\s*\[(.+?)\]", b.group(2)),
            applies_to=rxl(r"- applies_to:\s*\[(.+?)\]", b.group(2)),
            model=rx(r"- model:\s*(.+)", b.group(2)),
            tolerance=int(rx(r"- tolerance:\s*(\d+)", b.group(2), "0")),
        )
        for b in re.finditer(r"### REVIEWER: (\S+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL)
    ]

    all_reviewers = list(
        {**{r.name: r for r in BUILTIN_REVIEWERS}, **{r.name: r for r in custom}}.values()
    )

    tasks = [
        TaskSpec(
            id=b.group(1),
            title=rx(r"- title:\s*(.+)", b.group(2), b.group(1)),
            agent=rx(r"- agent:\s*(.+)", b.group(2), "unknown").lower(),
            description=rx(r"- description:\s*(.+)", b.group(2)),
            task_type=rxl(r"- task_type:\s*\[(.+?)\]", b.group(2)),
            reviewers=rxl(r"- reviewers:\s*\[(.+?)\]", b.group(2)),
            depends_on=rxl(r"- depends_on:\s*\[(.+?)\]", b.group(2)),
            model=rx(r"- model:\s*(.+)", b.group(2)),
        )
        for b in re.finditer(r"### (TASK-\w+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL)
    ]

    # ── Register rosters with bridge so UI gets dynamic room mapping ──────────
    register_rosters(
        sub_agents  = [a.__dict__ for a in sub_agents],
        managers    = [m.__dict__ for m in managers],
        reviewers   = [r.__dict__ for r in all_reviewers],
    )

    # ── Auto-register new agents into the persistent registry ──────────────
    try:
        from agent_registry import get_agent, create_agent
        for agent in sub_agents:
            d = agent.__dict__
            if not get_agent("sub_agents", d["name"]):
                tags = _infer_tags(d.get("description", ""), d.get("tools", []))
                create_agent("sub_agents", {**d, "tags": tags})
        for mgr in managers:
            d = mgr.__dict__
            if not get_agent("managers", d["name"]):
                tags = _infer_tags(d.get("description", ""), d.get("expertise_blend", []))
                create_agent("managers", {**d, "tags": tags})
        for rev in custom:  # custom reviewers only, not builtins
            d = rev.__dict__
            if not get_agent("reviewers", d["name"]):
                tags = d.get("applies_to", []) + d.get("focus_areas", [])[:3]
                create_agent("reviewers", {**d, "tags": tags})
    except Exception as e:
        print(f"[DEBUG] Registry auto-save: {e}")

    return objective, managers, sub_agents, tasks, all_reviewers


# ── LangGraph state ───────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    messages:       Annotated[Sequence[BaseMessage], add_messages]
    objective:      str
    managers:       list[dict]
    sub_agents:     list[dict]
    reviewers:      list[dict]
    tasks:          list[dict]
    active_task_id:  Optional[str]
    active_task_ids: list[str]           # batch of tasks for parallel execution
    results:         dict[str, str]
    finished:        bool
    phase:           str
    tokens_used:     int
    last_verdict:    Optional[dict]
    synthesis_report: str
    coherence_report: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def llm(model: str = ""):
    m = model or DEFAULT_MODEL
    return ChatOpenAI(model=m, temperature=0.2)

def extract_tokens(resp) -> int:
    """Extract total token count from a LangChain response."""
    try:
        meta = resp.response_metadata or {}
        usage = meta.get("usage", meta.get("token_usage", {}))
        return usage.get("total_tokens", 0)
    except Exception:
        return 0

def get_task(tid, tasks):
    return next((t for t in tasks if t["id"] == tid), None)

def upd(tasks, tid, **kw):
    return [{**t, **kw} if t["id"] == tid else t for t in tasks]

def find_mgr(agent, mgrs):
    return next((m for m in mgrs if agent in m.get("oversees", [])), None)

def _load_orchestrator_config() -> dict:
    """Load config.json with safe fallback."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
            return json.load(f)
    except Exception:
        return {}

def _resolve_tolerance(agent_name: str, agent_dict: dict, task_tolerance: int = 0) -> int:
    """Resolve effective tolerance for a reviewer/manager.

    Precedence: config per-agent > task-level > plan per-agent > config global > DEFAULT_TOLERANCE
    Then: +1 earned bonus if agent has avg_score > 8 over 5+ runs.
    """
    cfg = _load_orchestrator_config()

    per_agent = cfg.get("tolerance_overrides", {}).get(agent_name)
    if per_agent is not None and 1 <= per_agent <= 10:
        base = per_agent
    elif task_tolerance and 1 <= task_tolerance <= 10:
        base = task_tolerance
    else:
        plan_tol = agent_dict.get("tolerance", 0)
        if plan_tol and 1 <= plan_tol <= 10:
            base = plan_tol
        else:
            global_tol = cfg.get("default_tolerance")
            if global_tol is not None and 1 <= global_tol <= 10:
                base = global_tol
            else:
                base = DEFAULT_TOLERANCE

    # Earned tolerance bonus: high-performing agents get +1
    try:
        from agent_registry import check_earned_tolerance
        for atype in ("sub_agents", "managers", "reviewers"):
            if check_earned_tolerance(atype, agent_name):
                bonus = min(base + 1, 10)
                if bonus > base:
                    log_event(f"[TOLERANCE] {agent_name} earned +1 bonus (score>8, 5+ runs): {base} → {bonus}")
                return bonus
    except Exception:
        pass

    return base

def _auto_ingest(task, results):
    """Auto-ingest completed task output into RAG for cross-run knowledge."""
    try:
        from rag import ingest_text
        result_text = results.get(task["id"], task.get("result", ""))
        if result_text and len(result_text) > 50:
            n = ingest_text(
                result_text, source=f"task:{task['id']}",
                content_type="output", plan_id=_plan_id,
                task_id=task["id"], agent=task.get("agent", ""),
                tags=task.get("task_type", []),
            )
            log_event(f"  [RAG] Auto-ingested {task['id']} → {n} chunks")
    except Exception as e:
        log_event(f"  [RAG] Ingest failed: {e}")


def snapshot_artifacts(plan_id: str, task_id: str, revision_num: int):
    """Snapshot current task artifacts before a revision overwrites them."""
    if not plan_id:
        return
    artifacts_root = Path(__file__).parent / "artifacts" / plan_id / task_id
    if not artifacts_root.is_dir():
        return
    rev_dir = artifacts_root / "revisions" / f"rev_{revision_num}"
    rev_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in artifacts_root.rglob("*"):
        if src.is_file() and "revisions" not in src.parts:
            rel = src.relative_to(artifacts_root)
            dest = rev_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
    log_event(f"  [SNAPSHOT] {task_id} rev_{revision_num}: {copied} files")


def applicable_reviewers(task, reviewers):
    named  = set(task.get("reviewers", []))
    ttypes = set(task.get("task_type",  []))
    return [r for r in reviewers
            if r["name"] in named
            or (not named and ttypes & set(r.get("applies_to", [])))]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def master_node(state):
    set_active_reviewer(None)
    tasks, results = state["tasks"], state["results"]

    # Save checkpoint at each dispatch boundary (all completed tasks are persisted)
    if _plan_id:
        save_checkpoint(_plan_id, state)

    # ── Priority 1: route tasks waiting for review ──
    for task in tasks:
        if task["status"] == "in_manager_review":
            msg = f"[MASTER] Routing {task['id']} → manager review"
            print(f"\n{msg}"); log_event(msg)
            s = {**state, "tasks": tasks, "active_task_id": task["id"],
                 "active_task_ids": [], "phase": "manager_review"}
            write_status(s); return s
        if task["status"] == "in_specialist_review":
            msg = f"[MASTER] Routing {task['id']} → specialist review"
            print(f"\n{msg}"); log_event(msg)
            s = {**state, "tasks": tasks, "active_task_id": task["id"],
                 "active_task_ids": [], "phase": "specialist_review"}
            write_status(s); return s

    # ── Priority 2: find ALL ready tasks and dispatch ──
    ready = [t for t in tasks
             if t["status"] == "pending"
             and all(results.get(d) for d in t["depends_on"])]

    if ready:
        ready_ids = []
        for t in ready:
            msg = f"[MASTER] Dispatching → {t['id']}: {t['title']}"
            print(f"\n{msg}"); log_event(msg)
            tasks = upd(tasks, t["id"], status="in_progress")
            ready_ids.append(t["id"])

        if len(ready_ids) > 1:
            log_event(f"[MASTER] ⚡ Parallel batch: {len(ready_ids)} tasks")

        s = {**state, "tasks": tasks,
             "active_task_id": ready_ids[0],
             "active_task_ids": ready_ids,
             "phase": "execute"}
        write_status(s); return s

    # ── All done or blocked ──
    if all(t["status"] in ("done", "failed") for t in tasks):
        msg = "[MASTER] All tasks complete — synthesising"
        print(f"\n{msg}"); log_event(msg)
        s = {**state, "active_task_id": None, "active_task_ids": [], "phase": "done"}
        write_status(s); return s

    msg = "[MASTER] Remaining tasks blocked — forcing done"
    print(f"\n{msg}"); log_event(msg)
    s = {**state, "active_task_id": None, "active_task_ids": [], "phase": "done"}
    write_status(s); return s


def _execute_single_task(tid, tasks, results, sub_agents_list):
    """Run a single sub-agent task. Returns (tid, draft, toks, tool_calls_log, model).
    Thread-safe — called from ThreadPoolExecutor for parallel execution."""
    task   = get_task(tid, tasks)
    agents = {a["name"]: a for a in sub_agents_list}
    agent  = agents.get(task["agent"], {})
    rev    = task.get("revision_count", 0)

    system = (
        f"You are the '{task['agent']}' specialist. {agent.get('description','')}\n"
        "Produce thorough, production-quality work. "
        "Address all reviewer feedback explicitly and specifically."
    )

    ctx = [f"Output from {d}:\n{results[d]}"
           for d in task.get("depends_on", []) if d in results]

    if rev > 0:
        parts = []
        if task.get("manager_notes"):
            parts.append(f"MANAGER FEEDBACK:\n{task['manager_notes']}")
        if task.get("reviewer_notes"):
            parts.append(f"SPECIALIST PANEL FEEDBACK:\n{task['reviewer_notes']}")
        if parts:
            ctx.append(
                f"\n--- REVISION {rev} FEEDBACK ---\n"
                + "\n\n".join(parts)
                + "\n\nAddress every point explicitly in your revised output."
            )
        msg = f"[{task['agent'].upper()}] Revising {tid} (attempt {rev+1})"
    else:
        msg = f"[{task['agent'].upper()}] Executing {tid}: {task['title']}"

    print(f"  {msg}"); log_event(msg)

    model = resolve_model(task.get("model", ""), agent.get("model", ""), role="execute")
    log_event(f"  [MODEL] {model}")

    # ── Smart context injection: auto-search RAG for relevant past knowledge ──
    try:
        from rag import search as rag_search_fn
        rag_hits = rag_search_fn(f"query: {task['title']} {task['description'][:200]}", top_k=3)
        if rag_hits:
            rag_context = "\n\n".join(
                f"[Knowledge Base — {h['source']}]\n{h['text'][:400]}"
                for h in rag_hits if h['score'] > 0.5
            )
            if rag_context:
                ctx.append(f"\n--- RELEVANT KNOWLEDGE ---\n{rag_context}")
    except Exception:
        pass

    set_tool_context(plan_id=_plan_id, task_id=tid, agent=task["agent"])
    agent_tools = get_tools(agent.get("tools", []))
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Task: {task['title']}\nDescription: {task['description']}"
                              + ("\n\n" + "\n\n".join(ctx) if ctx else "")),
    ]
    total_toks = 0
    tool_calls_log = []

    if agent_tools:
        llm_with_tools = llm(model).bind_tools(agent_tools)
        max_iterations = 5
        for _iter in range(max_iterations):
            resp = llm_with_tools.invoke(messages)
            total_toks += extract_tokens(resp)
            if not resp.tool_calls:
                break
            messages.append(resp)
            for tc in resp.tool_calls:
                tc_name = tc["name"]
                tc_args = tc["args"]
                log_event(f"  [TOOL] {tc_name}({str(tc_args)[:80]})")
                result = execute_tool_call(tc, agent_tools)
                result_str = str(result)[:4000]
                messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
                tool_calls_log.append({"tool": tc_name, "args_preview": str(tc_args)[:100],
                                       "result_preview": result_str[:200]})
        draft = resp.content or ""
        if not draft.strip():
            log_event(f"  [TOOL] Loop exhausted — requesting final summary")
            messages.append(HumanMessage(
                content="You've completed your tool calls. Now produce your final written output for this task. "
                        "Synthesize everything you learned from the tools into your deliverable."
            ))
            resp = llm(model).invoke(messages)
            total_toks += extract_tokens(resp)
            draft = resp.content or "(No output generated)"
    else:
        resp = llm(model).invoke(messages)
        total_toks = extract_tokens(resp)
        draft = resp.content

    tools_used = len(tool_calls_log)
    done_msg = f"[{task['agent'].upper()}] Draft done ({len(draft)} chars, {total_toks} tokens, {tools_used} tool calls) → manager review"
    print(f"  {done_msg}"); log_event(done_msg)

    return (tid, draft, total_toks, tool_calls_log, model)


def sub_agent_node(state):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    active_ids = state.get("active_task_ids", [])
    tasks      = state["tasks"]
    results    = state["results"]
    sub_agents = state["sub_agents"]
    set_active_reviewer(None)

    # Fall back to single task if active_task_ids not populated
    if not active_ids:
        tid = state.get("active_task_id")
        if tid:
            active_ids = [tid]
        else:
            s = {**state, "phase": "dispatch", "active_task_id": None, "active_task_ids": []}
            write_status(s); return s

    if len(active_ids) == 1:
        # ── Single task — run directly ──
        tid = active_ids[0]
        tid, draft, toks, tool_log, model = _execute_single_task(tid, tasks, results, sub_agents)
        task = get_task(tid, tasks)
        if not task:
            s = {**state, "phase": "dispatch", "active_task_id": None, "active_task_ids": []}
            write_status(s); return s
        total_tokens = state.get("tokens_used", 0) + toks
        tasks = upd(tasks, tid, status="in_manager_review", result=draft,
                    manager_notes="", reviewer_notes="",
                    current_model=model, task_tokens=task.get("task_tokens", 0) + toks,
                    tool_calls=tool_log)
        s = {**state, "tasks": tasks, "phase": "manager_review",
             "tokens_used": total_tokens, "active_task_ids": []}
        write_status(s); return s
    else:
        # ── Parallel execution — multiple agents working simultaneously ──
        log_event(f"[PARALLEL] ⚡ Running {len(active_ids)} tasks concurrently")
        total_new_toks = 0

        with ThreadPoolExecutor(max_workers=len(active_ids)) as pool:
            futures = {
                pool.submit(_execute_single_task, tid, tasks, results, sub_agents): tid
                for tid in active_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    tid, draft, toks, tool_log, model = future.result()
                    task = get_task(tid, tasks)
                    tasks = upd(tasks, tid, status="in_manager_review", result=draft,
                                manager_notes="", reviewer_notes="",
                                current_model=model,
                                task_tokens=task.get("task_tokens", 0) + toks,
                                tool_calls=tool_log)
                    total_new_toks += toks
                except Exception as e:
                    log_event(f"[PARALLEL] {tid} failed: {e}")
                    tasks = upd(tasks, tid, status="failed")

        total_tokens = state.get("tokens_used", 0) + total_new_toks
        log_event(f"[PARALLEL] ⚡ Batch complete — routing to reviews")
        # Route back to master to dispatch reviews one at a time
        s = {**state, "tasks": tasks, "phase": "dispatch",
             "tokens_used": total_tokens, "active_task_id": None,
             "active_task_ids": []}
        write_status(s); return s


def manager_review_node(state):
    tid     = state["active_task_id"]
    tasks   = state["tasks"]
    results = state["results"]

    # After parallel batch, active_task_id may be None — find first task needing review
    if not tid:
        for t in tasks:
            if t["status"] == "in_manager_review":
                tid = t["id"]
                break
    if not tid:
        s = {**state, "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    task    = get_task(tid, tasks)
    if not task:
        s = {**state, "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    rev     = task.get("revision_count", 0)
    set_active_reviewer(None)
    manager = find_mgr(task["agent"], state["managers"])

    if not manager:
        log_event(f"[NO MANAGER] Auto-approving {tid} → panel")
        tasks = upd(tasks, tid, status="in_specialist_review")
        s = {**state, "tasks": tasks, "phase": "specialist_review"}
        write_status(s); return s

    if rev >= MAX_REVISIONS:
        log_event(f"[{manager['name'].upper()}] Max revisions — force-approving {tid}")
        results = {**results, tid: task["result"]}
        tasks   = upd(tasks, tid, status="done")
        _auto_ingest(task, results)
        s = {**state, "tasks": tasks, "results": results,
             "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    task_tolerance = task.get("tolerance", 0)
    tolerance = _resolve_tolerance(manager["name"], manager, task_tolerance)
    if tolerance >= 8:
        strictness = " Only FAIL for critical blocking issues that would cause real harm."
    elif tolerance >= 5:
        strictness = " FAIL only for real, substantive problems — not minor preferences."
    else:
        strictness = " Apply thorough scrutiny. FAIL any real problems you find."
    system = (
        f"You are the {manager['title']}. {manager['description']}\n"
        f"Expertise: {', '.join(manager.get('expertise_blend', []))}.\n"
        f"{strictness}\n"
        'Return ONLY JSON: {"verdict":"PASS"|"FAIL","score":1-10,'
        '"issues":["..."],"feedback":"..."}'
    )
    model = resolve_model(task.get("model", ""), manager.get("model", ""), role="review")
    log_event(f"  [MODEL] {model}")

    prior = "".join(
        f"\n{d}:\n{results[d][:400]}...\n"
        for d in task.get("depends_on", []) if d in results
    )
    resp = llm(model).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Task:{task['title']}\nReqs:{task['description']}"
                              + (f"\nContext:{prior}" if prior else "")
                              + f"\n\n---\n{task['result']}"),
    ])
    toks = extract_tokens(resp)
    total_tokens = state.get("tokens_used", 0) + toks
    try:
        v = json.loads(resp.content.strip().replace("```json","").replace("```",""))
    except Exception:
        v = {"verdict":"PASS","score":7,"issues":[],"feedback":""}

    verdict = v.get("verdict","PASS")
    score   = v.get("score",7)

    if verdict == "FAIL" and score >= tolerance:
        log_event(f"[{manager['name'].upper()}] Score {score} >= tolerance {tolerance} — overriding FAIL → PASS")
        verdict = "PASS"

    msg     = f"[{manager['name'].upper()}] {tid}: {verdict} ({score}/10)"
    print(f"  {msg}"); log_event(msg)
    for iss in v.get("issues",[]): log_event(f"    ↳ {iss}")

    last_v = {"reviewer": manager["name"], "verdict": verdict, "score": score, "task_id": tid}
    if _run_id:
        track_score(_run_id, tid, task["agent"], score, verdict, manager["name"], model, toks)

    try:
        from agent_registry import record_agent_performance
        record_agent_performance("sub_agents", task["agent"], score, task.get("revision_count", 0))
    except Exception:
        pass

    if verdict == "PASS":
        # Check for conditional specialist review skipping
        skip_specialist = False
        if score >= 9:
            skip_specialist = _load_orchestrator_config().get("skip_specialist_on_high_score", False)

        if skip_specialist:
            skip_msg = f"[{manager['name'].upper()}] Score {score} >= 9 with skip_specialist enabled — skipping specialist review"
            print(f"  {skip_msg}"); log_event(skip_msg)
            results = {**results, tid: task["result"]}
            tasks = upd(tasks, tid, status="done", manager_notes="", last_score=score)
            _auto_ingest(task, results)
            s = {**state, "tasks": tasks, "results": results,
                 "phase": "dispatch", "active_task_id": None,
                 "tokens_used": total_tokens, "last_verdict": last_v}
            write_status(s); return s

        log_event(f"[{manager['name'].upper()}] Approved → panel")
        tasks = upd(tasks, tid, status="in_specialist_review", manager_notes="",
                    last_score=score)
        s = {**state, "tasks": tasks, "phase": "specialist_review",
             "tokens_used": total_tokens, "last_verdict": last_v}
        write_status(s); return s
    else:
        log_event(f"[{manager['name'].upper()}] Returning {tid} for revision")
        snapshot_artifacts(_plan_id, tid, rev + 1)
        tasks = upd(tasks, tid, status="revision",
                    manager_notes=v.get("feedback",""), revision_count=rev+1,
                    last_score=score)
        s = {**state, "tasks": tasks, "phase": "execute",
             "tokens_used": total_tokens, "last_verdict": last_v}
        write_status(s); return s


def _build_reviewer_prompt(reviewer: dict, tolerance: int) -> str:
    if tolerance >= 8:
        strictness = (
            "You have a HIGH tolerance threshold. Only FLAG critical, blocking issues "
            "that would cause real harm in production. Minor style preferences, theoretical "
            "concerns, and nice-to-haves should receive PASS with a note. "
            "FLAG only for real, specific, severe problems."
        )
    elif tolerance >= 5:
        strictness = (
            "FLAG only for real, specific problems that meaningfully impact quality. "
            "Do not flag minor style issues, theoretical edge cases with low probability, "
            "or subjective preferences."
        )
    else:
        strictness = (
            "Apply thorough scrutiny. FLAG any real problems you find, including "
            "moderate issues that could affect quality. Be specific and actionable."
        )
    return (
        f"You are the {reviewer['title']}. {reviewer['description']}\n"
        f"Focus areas: {', '.join(reviewer.get('focus_areas', []))}\n"
        "Review from YOUR domain perspective only.\n"
        f'{{"reviewer":"{reviewer["name"]}","verdict":"PASS"|"FLAG",'
        '"score":1-10,"issues":["..."],"feedback":"Precise revision instructions"}}\n'
        f"Return ONLY JSON in the format above.\n{strictness}"
    )


def specialist_review_node(state):
    tid      = state["active_task_id"]
    tasks    = state["tasks"]
    results  = state["results"]

    # After routing, active_task_id may be None — find first task needing specialist review
    if not tid:
        for t in tasks:
            if t["status"] == "in_specialist_review":
                tid = t["id"]
                break
    if not tid:
        s = {**state, "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    task     = get_task(tid, tasks)
    if not task:
        s = {**state, "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    rev      = task.get("revision_count", 0)
    rev_list = applicable_reviewers(task, state["reviewers"])

    if not rev_list:
        log_event(f"[PANEL] No reviewers for {tid} — accepted")
        set_active_reviewer(None)
        results = {**results, tid: task["result"]}
        tasks   = upd(tasks, tid, status="done")
        _auto_ingest(task, results)
        s = {**state, "tasks": tasks, "results": results,
             "phase": "dispatch", "active_task_id": None}
        write_status(s); return s

    flags = []; verdicts = []

    for reviewer in rev_list:
        if rev >= MAX_REVISIONS:
            break

        # Push active reviewer — instant room highlight in the UI
        set_active_reviewer(reviewer["name"])
        write_status({**state, "tasks": tasks})

        model = resolve_model(task.get("model", ""), reviewer.get("model", ""), role="review")
        log_event(f"  [MODEL] {reviewer['name']} → {model}")

        tolerance = _resolve_tolerance(reviewer["name"], reviewer)
        system = _build_reviewer_prompt(reviewer, tolerance)
        resp = llm(model).invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Task:{task['title']}\nReqs:{task['description']}"
                                  f"\n\n---\n{task['result']}"),
        ])
        toks = extract_tokens(resp)
        total_tokens = state.get("tokens_used", 0) + toks
        try:
            v = json.loads(resp.content.strip().replace("```json","").replace("```",""))
        except Exception:
            v = {"reviewer": reviewer["name"], "verdict":"PASS","score":8,
                 "issues":[],"feedback":""}

        verdict = v.get("verdict","PASS")
        score   = v.get("score",8)

        if verdict == "FLAG" and score >= tolerance:
            log_event(f"[{reviewer['name'].upper()}] Score {score} >= tolerance {tolerance} — overriding FLAG → PASS")
            verdict = "PASS"

        msg     = f"[{reviewer['name'].upper()}] {tid}: {verdict} ({score}/10)"
        print(f"  {msg}"); log_event(msg)
        for iss in v.get("issues",[]): log_event(f"    ↳ {iss}")
        verdicts.append(verdict)
        last_v = {"reviewer": reviewer["name"], "verdict": verdict, "score": score, "task_id": tid}
        if _run_id:
            track_score(_run_id, tid, task["agent"], score, verdict, reviewer["name"], model, toks)
        try:
            from agent_registry import record_agent_performance
            record_agent_performance("reviewers", reviewer["name"], score, 0)
        except Exception:
            pass
        if verdict == "FLAG" and v.get("feedback"):
            flags.append(f"[{reviewer['title']}]\n{v['feedback']}")
        # Update score to latest reviewer score
        tasks = upd(tasks, tid, last_score=score)
        state = {**state, "tokens_used": total_tokens, "last_verdict": last_v}

    set_active_reviewer(None)
    any_flags = any(vd == "FLAG" for vd in verdicts)

    if any_flags and rev < MAX_REVISIONS:
        msg = f"[PANEL] {len(flags)} reviewer(s) flagged {tid} — revising"
        print(f"  {msg}"); log_event(msg)
        snapshot_artifacts(_plan_id, tid, rev + 1)
        tasks = upd(tasks, tid, status="revision",
                    reviewer_notes="\n\n".join(flags), revision_count=rev+1)
        s = {**state, "tasks": tasks, "phase": "execute"}
        write_status(s); return s

    msg = (f"[PANEL] Force-accepting {tid}" if any_flags
           else f"[PANEL] All reviewers passed {tid} ✓")
    print(f"  {msg}"); log_event(msg)
    results = {**results, tid: task["result"]}
    tasks   = upd(tasks, tid, status="done", reviewer_notes="")
    _auto_ingest(task, results)
    s = {**state, "tasks": tasks, "results": results,
         "phase": "dispatch", "active_task_id": None}
    write_status(s); return s


def synthesis_node(state):
    log_event("[MASTER] Writing final report...")
    model = resolve_model(role="synthesis")
    log_event(f"  [MODEL] {model}")
    summaries = "\n\n".join(
        f"[{tid}]\n{res[:800]}" for tid, res in state["results"].items()
    )
    resp = llm(model).invoke([HumanMessage(content=(
        f"Project: {state['objective']}\n\nOutputs:\n{summaries}\n\n"
        "Write a concise final executive report: accomplishments, key outputs, "
        "quality signals (revision counts), risks, and next steps."
    ))])
    toks = extract_tokens(resp)
    total_tokens = state.get("tokens_used", 0) + toks
    log_event("[MASTER] Done.")
    s = {**state, "messages": [AIMessage(content=resp.content)],
         "finished": True, "phase": "done",
         "tokens_used": total_tokens,
         "synthesis_report": resp.content}
    write_status(s); return s


def composition_node(state):
    """Check cross-file coherence of assembled output."""
    plan_id = _plan_id
    if not plan_id:
        return state

    artifacts_root = Path(__file__).parent / "artifacts" / plan_id
    if not artifacts_root.is_dir():
        log_event("[COMPOSITION] No artifacts to check")
        return state

    # Gather file listing and content samples (cap at 20 files)
    all_files = []
    file_contents = {}
    for f in sorted(artifacts_root.rglob("*")):
        if f.is_file() and "revisions" not in f.parts:
            rel = str(f.relative_to(artifacts_root))
            all_files.append(rel)
            if len(file_contents) < 20:
                try:
                    content = f.read_text(errors="replace")
                    if len(content) < 10000:
                        file_contents[rel] = content
                except Exception:
                    pass

    if not all_files:
        log_event("[COMPOSITION] No files to check")
        return state

    log_event("[COMPOSITION] Checking cross-file coherence...")
    model = resolve_model(role="review")
    log_event(f"  [MODEL] {model}")

    file_listing = "\n".join(all_files)
    content_samples = "\n\n".join(
        f"=== {path} ===\n{content[:2000]}"
        for path, content in list(file_contents.items())[:20]
    )

    prompt = (
        f"You are a composition reviewer checking cross-file coherence for a project.\n\n"
        f"## All Files\n{file_listing}\n\n"
        f"## File Contents (samples)\n{content_samples}\n\n"
        "Check for these issues:\n"
        "1. Import references: Do any files import/reference other files that don't exist in the listing?\n"
        "2. Config coherence: Do config values (paths, URLs, names) match actual file paths?\n"
        "3. README accuracy: If there's a README, do its instructions match the actual project structure?\n"
        "4. Missing files: Are there obvious missing files (e.g., referenced but not present)?\n\n"
        "Return JSON:\n"
        '{"coherent": true/false, "issues": [{"file": "path", "type": "import|config|readme|missing", '
        '"description": "..."}], "summary": "1-2 sentence overall assessment"}'
    )

    try:
        resp = llm(model).invoke([HumanMessage(content=prompt)])
        toks = extract_tokens(resp)
        total_tokens = state.get("tokens_used", 0) + toks

        text = resp.content
        json_match = re.search(r'\{[\s\S]*?\}(?=\s*$)', text)
        if json_match:
            report = json.loads(json_match.group())
        else:
            report = {"coherent": True, "issues": [], "summary": text[:500]}

        log_event(f"[COMPOSITION] {'Coherent' if report.get('coherent') else 'Issues found'}: {report.get('summary', '')[:100]}")

        return {**state, "coherence_report": report, "tokens_used": total_tokens}
    except Exception as e:
        log_event(f"[COMPOSITION] Error: {e}")
        return {**state, "coherence_report": {"coherent": True, "issues": [], "summary": f"Check failed: {e}"}}


# ── Graph ─────────────────────────────────────────────────────────────────────

def route_master(s):
    p = s.get("phase", "dispatch")
    return "synthesis" if p == "done" else "sub_agent" if p == "execute" else "master"

def route_manager(s):
    p = s.get("phase")
    return ("sub_agent"         if p == "execute"           else
            "specialist_review" if p == "specialist_review" else "master")

def route_specialist(s):
    return "sub_agent" if s.get("phase") == "execute" else "master"


def build_graph():
    g = StateGraph(OrchestratorState)
    for name, fn in [
        ("master",            master_node),
        ("sub_agent",         sub_agent_node),
        ("manager_review",    manager_review_node),
        ("specialist_review", specialist_review_node),
        ("synthesis",         synthesis_node),
        ("composition",       composition_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("master")
    g.add_conditional_edges("master", route_master,
        {"sub_agent":"sub_agent","synthesis":"synthesis","master":"master"})
    g.add_edge("sub_agent", "manager_review")
    g.add_conditional_edges("manager_review", route_manager,
        {"sub_agent":"sub_agent","specialist_review":"specialist_review","master":"master"})
    g.add_conditional_edges("specialist_review", route_specialist,
        {"sub_agent":"sub_agent","master":"master"})
    g.add_edge("synthesis", "composition")
    g.add_edge("composition", END)
    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def _send_webhook(data: dict):
    """Fire-and-forget webhook notification."""
    import threading
    url = os.environ.get("QUARM_WEBHOOK_URL", "")
    if not url:
        # Try config.json
        try:
            with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
                url = json.load(f).get("webhook_url", "")
        except Exception:
            pass
    if not url:
        return
    def _post():
        try:
            import urllib.request
            req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[WEBHOOK] Failed: {e}")
    threading.Thread(target=_post, daemon=True).start()


def _slugify(text: str, max_len: int = 60) -> str:
    """Turn an objective string into a filesystem-safe folder name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def assemble_output(plan_id: str, objective: str, tasks: list,
                    plan_path: str = "", results_path: str = "") -> tuple[str, dict]:
    """Merge per-task artifacts into a single deliverable output folder.

    Layout:  output/<slug>/
               ├── <merged project files>
               ├── MANIFEST.md
               ├── plan.md          (copy of source plan)
               └── results.json     (copy of results)
    Returns (output_dir_path, artifacts_by_task) or ("", {}) if no artifacts.
    """
    artifacts_root = Path(__file__).parent / "artifacts" / plan_id
    if not artifacts_root.is_dir():
        return "", {}

    slug = _slugify(objective) or plan_id
    output_dir = Path(__file__).parent / "output" / f"{plan_id}_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [f"# {objective}\n", f"**Plan ID:** `{plan_id}`\n", "## Deliverables\n"]
    files_copied = 0
    artifacts_by_task = {}  # {TASK-001: ["src/index.html", ...]}

    # Process tasks in order so later tasks overwrite earlier ones on conflict
    task_ids = sorted(
        [d.name for d in artifacts_root.iterdir() if d.is_dir()],
        key=lambda t: int(re.search(r"\d+", t).group()) if re.search(r"\d+", t) else 0,
    )

    for tid in task_ids:
        task_dir = artifacts_root / tid
        task_label = next((t["id"] for t in tasks if t["id"] == tid), tid)
        task_desc = next((t.get("description", "") for t in tasks if t["id"] == tid), "")
        task_files = []

        for src in sorted(task_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(task_dir)
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            task_files.append(str(rel))
            files_copied += 1

        if task_files:
            artifacts_by_task[tid] = task_files
            manifest_lines.append(f"### {task_label}")
            if task_desc:
                manifest_lines.append(f"{task_desc}\n")
            for f in task_files:
                manifest_lines.append(f"- `{f}`")
            manifest_lines.append("")

    if files_copied == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        return "", {}

    # Bundle source plan and results into the output package
    if plan_path and os.path.isfile(plan_path):
        shutil.copy2(plan_path, output_dir / "plan.md")
    if results_path and os.path.isfile(results_path):
        shutil.copy2(results_path, output_dir / "results.json")

    manifest_lines.append(f"\n---\n*{files_copied} file(s) assembled from {len(task_ids)} task(s).*\n")
    (output_dir / "MANIFEST.md").write_text("\n".join(manifest_lines))

    return str(output_dir), artifacts_by_task


def validate_outputs(output_dir: str) -> dict:
    """Run basic validation on assembled output files.

    Returns dict: {"passed": [...], "failed": [...], "summary": str}
    """
    if not output_dir or not Path(output_dir).is_dir():
        return {"passed": [], "failed": [], "summary": "No output directory"}

    import subprocess
    passed = []
    failed = []

    for f in sorted(Path(output_dir).rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(output_dir))

        if f.suffix == ".py":
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(f)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    passed.append({"file": rel, "check": "py_compile", "status": "pass"})
                else:
                    failed.append({"file": rel, "check": "py_compile", "status": "fail",
                                   "error": result.stderr.strip()[:500]})
            except FileNotFoundError:
                passed.append({"file": rel, "check": "py_compile", "status": "skip",
                               "error": "python3 not available"})
            except Exception as e:
                failed.append({"file": rel, "check": "py_compile", "status": "fail",
                               "error": str(e)[:200]})

        elif f.suffix == ".js":
            try:
                result = subprocess.run(
                    ["node", "--check", str(f)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    passed.append({"file": rel, "check": "node_check", "status": "pass"})
                else:
                    failed.append({"file": rel, "check": "node_check", "status": "fail",
                                   "error": result.stderr.strip()[:500]})
            except FileNotFoundError:
                passed.append({"file": rel, "check": "node_check", "status": "skip",
                               "error": "node not available"})
            except Exception as e:
                failed.append({"file": rel, "check": "node_check", "status": "fail",
                               "error": str(e)[:200]})

        elif f.suffix == ".json":
            try:
                json.loads(f.read_text())
                passed.append({"file": rel, "check": "json_parse", "status": "pass"})
            except json.JSONDecodeError as e:
                failed.append({"file": rel, "check": "json_parse", "status": "fail",
                               "error": str(e)[:200]})

    total = len(passed) + len(failed)
    summary = f"{len(passed)}/{total} files passed validation"
    if failed:
        summary += f" ({len(failed)} failed)"

    return {"passed": passed, "failed": failed, "summary": summary}


def run(plan_path="plan.md", plan_id: str = ""):
    global _run_id, _plan_id
    _plan_id = plan_id
    import time as _time
    _start_time = _time.time()
    print(f"\nLoading: {plan_path}\n{'='*60}")
    fetch_available_models()

    # ── Check for existing checkpoint ──
    checkpoint = load_checkpoint(plan_id) if plan_id else None

    if checkpoint:
        # ── RESUME MODE ──
        print(f"  ** Resuming from checkpoint (saved {checkpoint['saved_at']})")
        _run_id = checkpoint.get("run_id", "") or track_run_start(os.path.basename(plan_path))

        # Re-parse to re-register rosters with status bridge (in-memory, lost on restart)
        parse_plan(plan_path)

        # Reset any in-flight tasks back to pending
        for task in checkpoint["tasks"]:
            if task["status"] in ("in_progress", "in_manager_review",
                                   "in_specialist_review", "revision"):
                print(f"  ** Resetting interrupted task {task['id']} → pending")
                task["status"] = "pending"
                task["result"] = ""
                task["manager_notes"] = ""
                task["reviewer_notes"] = ""

        completed = [t["id"] for t in checkpoint["tasks"] if t["status"] == "done"]
        pending = [t["id"] for t in checkpoint["tasks"] if t["status"] == "pending"]
        print(f"  Completed: {completed}")
        print(f"  Pending:   {pending}")
        print("=" * 60)

        initial_state = {
            "messages":          [],
            "objective":         checkpoint["objective"],
            "managers":          checkpoint["managers"],
            "sub_agents":        checkpoint["sub_agents"],
            "reviewers":         checkpoint["reviewers"],
            "tasks":             checkpoint["tasks"],
            "active_task_id":    None,
            "active_task_ids":   [],
            "results":           checkpoint["results"],
            "finished":          checkpoint.get("finished", False),
            "phase":             "dispatch",
            "tokens_used":       checkpoint.get("tokens_used", 0),
            "last_verdict":      None,
            "synthesis_report":  checkpoint.get("synthesis_report", ""),
        }
    else:
        # ── FRESH START ──
        _run_id = track_run_start(os.path.basename(plan_path))
        objective, managers, sub_agents, tasks, reviewers = parse_plan(plan_path)
        print(f"Managers  : {[m.name for m in managers]}")
        print(f"Sub-agents: {[a.name for a in sub_agents]}")
        print(f"Tasks     : {[t.id for t in tasks]}")
        print("=" * 60)

        initial_state = {
            "messages":          [],
            "objective":         objective,
            "managers":          [m.__dict__ for m in managers],
            "sub_agents":        [a.__dict__ for a in sub_agents],
            "reviewers":         [r.__dict__ for r in reviewers],
            "tasks":             [t.__dict__ for t in tasks],
            "active_task_id":    None,
            "active_task_ids":   [],
            "results":           {},
            "finished":          False,
            "phase":             "dispatch",
            "tokens_used":       0,
            "last_verdict":      None,
            "synthesis_report":  "",
        }

    objective = initial_state["objective"]
    final = build_graph().invoke(initial_state)

    print("\n" + "="*60 + "\nFINAL REPORT\n" + "="*60)
    for msg in final["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)

    base, _ = os.path.splitext(plan_path)
    results_path = f"{base}_results.json"

    # ── Assemble output package ──
    output_path = ""
    artifacts_by_task = {}
    if plan_id:
        output_path, artifacts_by_task = assemble_output(
            plan_id, objective, final["tasks"],
            plan_path=plan_path, results_path="",  # results not written yet
        )

    # ── Validate output files ──
    validation = {}
    if output_path:
        log_event("[VALIDATE] Running output validation...")
        validation = validate_outputs(output_path)
        log_event(f"[VALIDATE] {validation.get('summary', '')}")

    with open(results_path, "w") as f:
        results_data = {
            "task_results": final["results"],
            "quality_log": [
                {"id": t["id"], "status": t["status"],
                 "revision_count": t.get("revision_count", 0)}
                for t in final["tasks"]
            ],
            "summary": next(
                (m.content for m in final["messages"] if isinstance(m, AIMessage)), ""
            ),
        }
        if artifacts_by_task:
            results_data["artifacts"] = artifacts_by_task
        if output_path:
            results_data["output_dir"] = output_path
        if validation:
            results_data["validation"] = validation
        if final.get("coherence_report"):
            results_data["coherence_report"] = final["coherence_report"]
        json.dump(results_data, f, indent=2)
    print(f"\nSaved → {results_path}")

    # Copy results.json into the output package now that it's written
    if output_path:
        shutil.copy2(results_path, os.path.join(output_path, "results.json"))
        print(f"Output → {output_path}/")

    # ── Tracking & webhook ──
    total_tokens = final.get("tokens_used", 0)
    total_revisions = sum(t.get("revision_count", 0) for t in final["tasks"])
    track_run_end(_run_id, total_tokens, total_revisions, len(final["tasks"]))
    elapsed = int(_time.time() - _start_time)
    summary_preview = next(
        (m.content[:300] for m in final["messages"] if isinstance(m, AIMessage)), ""
    )
    _send_webhook({
        "project": objective[:100],
        "plan": os.path.basename(plan_path),
        "tasks_completed": len(final["results"]),
        "total_revisions": total_revisions,
        "tokens_used": total_tokens,
        "elapsed_seconds": elapsed,
        "summary": summary_preview,
    })

    # Clean up checkpoint on successful completion
    if plan_id:
        clear_checkpoint(plan_id)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "plan.md")
