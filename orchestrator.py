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

import re, json, os
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Sequence, Optional
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
_run_id = ""  # set at run() start
_plan_id = ""  # set at run() start, used for checkpointing

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

@dataclass
class ReviewerSpec:
    name: str
    title: str
    description: str
    focus_areas: list[str]
    applies_to: list[str]
    model: str = ""

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

    return objective, managers, sub_agents, tasks, all_reviewers


# ── LangGraph state ───────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    messages:       Annotated[Sequence[BaseMessage], add_messages]
    objective:      str
    managers:       list[dict]
    sub_agents:     list[dict]
    reviewers:      list[dict]
    tasks:          list[dict]
    active_task_id: Optional[str]
    results:        dict[str, str]
    finished:       bool
    phase:          str
    tokens_used:    int
    last_verdict:   Optional[dict]
    synthesis_report: str


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

    for task in tasks:
        if task["status"] != "pending":
            continue
        if all(results.get(d) for d in task["depends_on"]):
            msg = f"[MASTER] Dispatching → {task['id']}: {task['title']}"
            print(f"\n{msg}"); log_event(msg)
            tasks = upd(tasks, task["id"], status="in_progress")
            s = {**state, "tasks": tasks, "active_task_id": task["id"], "phase": "execute"}
            write_status(s); return s

    if all(t["status"] in ("done", "failed") for t in tasks):
        msg = "[MASTER] All tasks complete — synthesising"
        print(f"\n{msg}"); log_event(msg)
        s = {**state, "active_task_id": None, "phase": "done"}
        write_status(s); return s

    msg = "[MASTER] Remaining tasks blocked — forcing done"
    print(f"\n{msg}"); log_event(msg)
    s = {**state, "active_task_id": None, "phase": "done"}
    write_status(s); return s


def sub_agent_node(state):
    tid    = state["active_task_id"]
    tasks  = state["tasks"]
    results = state["results"]
    agents = {a["name"]: a for a in state["sub_agents"]}
    task   = get_task(tid, tasks)
    agent  = agents.get(task["agent"], {})
    rev    = task.get("revision_count", 0)
    set_active_reviewer(None)

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
        pass  # RAG unavailable — continue without

    # ── Set tool context for this task ──
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
        # ── Agentic tool loop ──
        llm_with_tools = llm(model).bind_tools(agent_tools)
        max_iterations = 10
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
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                tool_calls_log.append({"tool": tc_name, "args_preview": str(tc_args)[:100],
                                       "result_preview": str(result)[:200]})
        draft = resp.content
    else:
        # ── Plain text generation (no tools) ──
        resp = llm(model).invoke(messages)
        total_toks = extract_tokens(resp)
        draft = resp.content

    total_tokens = state.get("tokens_used", 0) + total_toks
    tools_used = len(tool_calls_log)
    done = f"[{task['agent'].upper()}] Draft done ({len(draft)} chars, {total_toks} tokens, {tools_used} tool calls) → manager review"
    print(f"  {done}"); log_event(done)

    tasks = upd(tasks, tid, status="in_manager_review", result=draft,
                manager_notes="", reviewer_notes="",
                current_model=model, task_tokens=task.get("task_tokens", 0) + total_toks,
                tool_calls=tool_calls_log)
    s = {**state, "tasks": tasks, "phase": "manager_review", "tokens_used": total_tokens}
    write_status(s); return s


def manager_review_node(state):
    tid     = state["active_task_id"]
    tasks   = state["tasks"]
    results = state["results"]
    task    = get_task(tid, tasks)
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

    system = (
        f"You are the {manager['title']}. {manager['description']}\n"
        f"Expertise: {', '.join(manager.get('expertise_blend', []))}.\n"
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
    msg     = f"[{manager['name'].upper()}] {tid}: {verdict} ({score}/10)"
    print(f"  {msg}"); log_event(msg)
    for iss in v.get("issues",[]): log_event(f"    ↳ {iss}")

    last_v = {"reviewer": manager["name"], "verdict": verdict, "score": score, "task_id": tid}
    if _run_id:
        track_score(_run_id, tid, task["agent"], score, verdict, manager["name"], model, toks)

    if verdict == "PASS":
        log_event(f"[{manager['name'].upper()}] Approved → panel")
        tasks = upd(tasks, tid, status="in_specialist_review", manager_notes="",
                    last_score=score)
        s = {**state, "tasks": tasks, "phase": "specialist_review",
             "tokens_used": total_tokens, "last_verdict": last_v}
        write_status(s); return s
    else:
        log_event(f"[{manager['name'].upper()}] Returning {tid} for revision")
        tasks = upd(tasks, tid, status="revision",
                    manager_notes=v.get("feedback",""), revision_count=rev+1,
                    last_score=score)
        s = {**state, "tasks": tasks, "phase": "execute",
             "tokens_used": total_tokens, "last_verdict": last_v}
        write_status(s); return s


REVIEWER_PROMPT = (
    "You are the {title}. {description}\n"
    "Focus areas: {focus_areas}\n"
    "Review from YOUR domain perspective only.\n"
    'Return ONLY JSON: {{"reviewer":"{name}","verdict":"PASS"|"FLAG",'
    '"score":1-10,"issues":["..."],"feedback":"Precise revision instructions"}}\n'
    "FLAG only for real, specific problems."
)


def specialist_review_node(state):
    tid      = state["active_task_id"]
    tasks    = state["tasks"]
    results  = state["results"]
    task     = get_task(tid, tasks)
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

        system = REVIEWER_PROMPT.format(
            title=reviewer["title"],
            description=reviewer["description"],
            focus_areas=", ".join(reviewer.get("focus_areas", [])),
            name=reviewer["name"],
        )
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
        msg     = f"[{reviewer['name'].upper()}] {tid}: {verdict} ({score}/10)"
        print(f"  {msg}"); log_event(msg)
        for iss in v.get("issues",[]): log_event(f"    ↳ {iss}")
        verdicts.append(verdict)
        last_v = {"reviewer": reviewer["name"], "verdict": verdict, "score": score, "task_id": tid}
        if _run_id:
            track_score(_run_id, tid, task["agent"], score, verdict, reviewer["name"], model, toks)
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
    g.add_edge("synthesis", END)
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
    with open(results_path, "w") as f:
        json.dump({
            "task_results": final["results"],
            "quality_log": [
                {"id": t["id"], "status": t["status"],
                 "revision_count": t.get("revision_count", 0)}
                for t in final["tasks"]
            ],
            "summary": next(
                (m.content for m in final["messages"] if isinstance(m, AIMessage)), ""
            ),
        }, f, indent=2)
    print(f"\nSaved → {results_path}")

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
