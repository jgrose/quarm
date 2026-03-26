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
import openai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "bedrock-claude-opus-4-6"


def _pick_best_opus() -> str:
    """Query available models and return the best opus-tier model."""
    try:
        client = openai.OpenAI()
        available = sorted(m.id for m in client.models.list().data)
        opus_models = [m for m in available if "opus" in m.lower()]
        if opus_models:
            # Prefer the highest-versioned opus model (sorted descending)
            return sorted(opus_models, reverse=True)[0]
        # No opus found — fall back to first available
        print(f"[WARN] No opus model found, falling back to: {available[0]}")
        return available[0]
    except Exception as e:
        print(f"[WARN] Could not fetch models: {e} — using default: {DEFAULT_MODEL}")
        return DEFAULT_MODEL


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
- tools: [web_search, write_file, execute_code, read_file, analyze_data, design_ui, etc.]
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
- reviewers: [comma-separated from: security_engineer, ux_designer, user_tester — or leave empty as []]
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

Multiple reviewers can be assigned to one task.
A backend data pipeline might get [security_engineer] only.
A user dashboard might get [security_engineer, ux_designer, user_tester].
A pure-analysis or internal task might get [].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- All names: lowercase_with_underscores
- TASK IDs: TASK-001, TASK-002, etc.
- depends_on and reviewers: use [] for none, [ITEM1, ITEM2] for multiple
- Do NOT create tasks for managers or reviewers — they run automatically
- The master agent is implicit — do not include it
- Be specific in task descriptions — vague tasks produce poor results
"""


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
