"""
Plan Generator — 4-Layer Hierarchy
Generates plan.md with Master → Managers → Sub-Agents + Specialist Reviewer assignments.

Usage:
    python generate_plan.py "Build a web dashboard for AWS cost monitoring"
    python generate_plan.py   # interactive
"""

import sys
import os
import threading
import time
import json
import openai
from dotenv import load_dotenv
from model_config import load_allowed_models

load_dotenv()

DEFAULT_MODEL = "bedrock-claude-opus-4-6"


def _pick_best_opus() -> str:
    """Query available models and return the best opus-tier model."""
    try:
        client = openai.OpenAI()
        available = sorted(m.id for m in client.models.list().data)
        allowed = load_allowed_models()
        if allowed is not None:
            available = [m for m in available if m in allowed]
            if not available:
                print("[WARN] No allowed models, falling back to all available")
                available = sorted(m.id for m in client.models.list().data)
        opus_models = [m for m in available if "opus" in m.lower()]
        if opus_models:
            return sorted(opus_models, reverse=True)[0]
        print(f"[WARN] No opus model found, falling back to: {available[0]}")
        return available[0]
    except Exception as e:
        print(f"[WARN] Could not fetch models: {e} — using default: {DEFAULT_MODEL}")
        return DEFAULT_MODEL


CONTEXT_WINDOWS = {
    "opus": 200000,
    "sonnet": 200000,
    "haiku": 200000,
    "gpt-4o": 128000,
    "gpt-4": 128000,
}


SYSTEM_PROMPT = """You are a technical project planner designing a 4-layer AI agent system.

Given a project description, produce a structured plan.md.
Return ONLY the raw markdown — no code fences, no preamble, no commentary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA (follow exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# PROJECT PLAN: [Short project name]

## Objective
[1-3 sentence description of what will be built and why]

## Sub-Agents
### AGENT: [agent_name]
- description: [Specialist role and exact output format this agent produces]
- tools: [choose from: web_search, browse_url, rag_search, rag_store, download_artifact, read_file, write_file, execute_code]
- model: [optional — specific model ID to use for this agent, omit to auto-select]

[Define 2-5 sub-agents. Each should be a distinct specialist.]

## Managers
### MANAGER: [manager_name]
- title: [e.g. "Security Architecture Director", "Frontend Delivery Lead"]
- description: [What domain this manager owns and how they judge quality]
- expertise_blend: [comma-separated domain terms, e.g. threat_modeling, secure_coding, OWASP, API_design]
- oversees: [comma-separated agent names]
- model: [optional — specific model ID for this manager's reviews, omit to auto-select]

[1-3 managers. Each oversees 1-3 related sub-agents.
Manager expertise_blend MUST span ALL domains of sub-agents they oversee.]

## Tasks
### TASK-001
- title: [Short action title]
- agent: [agent_name — must match a sub-agent above]
- description: [Specific, detailed instructions. What exactly should the agent produce?]
- task_type: [comma-separated tags from: code, api, auth, data, config, ui, frontend, ux, report, documentation, backend, infrastructure, security, user_flow, dashboard, form]
- reviewers: [comma-separated from: security_engineer, ux_designer, user_tester, creative_director, devils_advocate, performance_engineer — or leave empty as []]
- depends_on: []
- model: [optional — override model for this specific task, omit to auto-select]

[4-8 tasks. Only assign to sub-agents. Managers review automatically.
Use depends_on to model real sequencing.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REVIEWER ASSIGNMENT GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
security_engineer → Any task involving: code, APIs, auth flows, data handling,
  config files, infrastructure, backend logic, secrets, access control.

ux_designer → Any task involving: user interfaces, frontend components, dashboards,
  forms, reports displayed to users, user flows, visual design.

user_tester → Any task involving: interfaces the end-user touches, reports
  delivered to non-technical users, documentation, workflows a user must navigate.

creative_director → Any task where the solution could be generic or conventional.
  Challenges safe thinking, pushes for innovation and elegance. Good for: architecture
  decisions, API design, user flows, any task where there might be a bolder approach.

devils_advocate → Any task with complexity, assumptions, or risk. Finds hidden
  flaws, unstated dependencies, edge cases, and failure modes. Good for: code,
  auth, data pipelines, infrastructure, anything that could break silently.

performance_engineer → Any task involving: code that runs at scale, APIs,
  data pipelines, database queries, caching, infrastructure. Finds N+1 queries,
  memory leaks, blocking calls, missing indexes, scalability bottlenecks.

Multiple reviewers can be assigned to one task. Be generous with reviewers.
A backend API might get [security_engineer, devils_advocate, performance_engineer].
A user dashboard might get [security_engineer, ux_designer, user_tester, creative_director].
A core architecture task might get [devils_advocate, creative_director, performance_engineer].
A pure-analysis or internal task might get [devils_advocate].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- All names: lowercase_with_underscores
- TASK IDs: TASK-001, TASK-002, etc.
- depends_on and reviewers: use [] for none, [ITEM1, ITEM2] for multiple
- Do NOT create tasks for managers or reviewers — they run automatically
- The master agent is implicit — do not include it
- Be specific in task descriptions — vague tasks produce poor results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARALLELISM (critical for performance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tasks with no dependencies (depends_on: []) run IN PARALLEL automatically.
Tasks that depend on other tasks wait for those to finish first.

MAXIMIZE parallelism by minimizing unnecessary depends_on:
- Only add a dependency when a task truly NEEDS the output of another task.
- Do NOT chain tasks sequentially just because they are numbered in order.
- If two tasks can be done independently, they MUST have depends_on: [].

Good example (3 tasks run in parallel, 1 waits):
  TASK-001: Build API endpoints          depends_on: []
  TASK-002: Build frontend components    depends_on: []
  TASK-003: Write security policies      depends_on: []
  TASK-004: Integration testing          depends_on: [TASK-001, TASK-002]

Bad example (all sequential, wastes time):
  TASK-001: Build API endpoints          depends_on: []
  TASK-002: Build frontend components    depends_on: [TASK-001]  ← unnecessary!
  TASK-003: Write security policies      depends_on: [TASK-002]  ← unnecessary!

Aim for a WIDE dependency graph (many parallel tasks) not a TALL one (long chain).
"""


def generate_plan_streaming(description: str, output_path: str = "plan.md"):
    """Generator that yields streaming events as dicts for the web UI."""
    client = openai.OpenAI()
    model = _pick_best_opus()

    ctx_window = 200000
    for key, size in CONTEXT_WINDOWS.items():
        if key in model.lower():
            ctx_window = size
            break

    est_input = (len(SYSTEM_PROMPT) + len(description)) // 4
    yield {"event": "model", "model": model, "context_window": ctx_window,
           "estimated_input_tokens": est_input}

    try:
        stream = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
        )
    except Exception as e:
        yield {"event": "error", "message": str(e)}
        return

    chunks = []
    usage_info = None
    for chunk in stream:
        if hasattr(chunk, 'usage') and chunk.usage:
            usage_info = {
                "input_tokens": getattr(chunk.usage, 'prompt_tokens', 0),
                "output_tokens": getattr(chunk.usage, 'completion_tokens', 0),
            }
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            chunks.append(text)
            yield {"event": "chunk", "text": text}

    plan = "".join(chunks).strip()
    if plan.startswith("```"):
        plan = "\n".join(plan.split("\n")[1:])
    if plan.endswith("```"):
        plan = "\n".join(plan.split("\n")[:-1])

    with open(output_path, "w") as f:
        f.write(plan)

    yield {"event": "done", "plan_text": plan,
           "usage": usage_info or {"input_tokens": est_input, "output_tokens": len(plan) // 4}}


def generate_plan(description: str, output_path: str = "plan.md") -> str:
    client = openai.OpenAI()
    model = _pick_best_opus()
    print(f"Generating 4-layer plan for:\n  {description}")
    print(f"Using model: {model}\n")

    # Spinner to show progress during generation
    stop_spinner = threading.Event()
    def _spinner():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        start = time.time()
        i = 0
        while not stop_spinner.is_set():
            elapsed = int(time.time() - start)
            print(f"\r  {frames[i % len(frames)]} Generating plan... ({elapsed}s)", end="", flush=True)
            i += 1
            stop_spinner.wait(0.1)
        elapsed = int(time.time() - start)
        print(f"\r  ✓ Plan generated in {elapsed}s          ")

    spinner = threading.Thread(target=_spinner, daemon=True)
    spinner.start()

    try:
        msg = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
        )
    finally:
        stop_spinner.set()
        spinner.join()

    plan = msg.choices[0].message.content.strip()
    if plan.startswith("```"):
        plan = "\n".join(plan.split("\n")[1:])
    if plan.endswith("```"):
        plan = "\n".join(plan.split("\n")[:-1])

    with open(output_path, "w") as f:
        f.write(plan)

    print(f"Plan written → {output_path}\n{'='*60}")
    print(plan)
    print("="*60)
    return plan


if __name__ == "__main__":
    if len(sys.argv) > 1:
        desc = " ".join(sys.argv[1:])
    else:
        desc = input("Describe your project: ").strip()
        if not desc:
            sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "plan.md"
    generate_plan(desc, out)
    print(f"\nNext: python orchestrator.py {out}")
