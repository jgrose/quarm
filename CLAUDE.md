# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never include Co-Authored-By, "Claude", or any AI attribution in commit messages or anywhere in the codebase.**

## What This Is

NORT is a 4-layer multi-agent orchestrator built on LangGraph. It takes a structured plan (markdown), dispatches tasks to specialist sub-agents, and runs each result through two quality gates: a domain manager review and a specialist reviewer panel (security engineer, UX designer, user tester). Results and a final executive report are written to `results.json`.

## Commands

```bash
# Install dependencies
pip install langgraph langchain langchain-openai python-dotenv fastapi uvicorn python-multipart jinja2

# Generate a plan from a project description
python generate_plan.py "Build a web dashboard for AWS cost monitoring"

# Run the orchestrator against a plan
python orchestrator.py plan.md

# Start the live dashboard server (run before orchestrator for real-time UI)
python serve.py
# Then open http://localhost:8000/
```

The live dashboard requires running `serve.py` in one terminal and `orchestrator.py` in another. The orchestrator pushes state to the server via HTTP POST; the server broadcasts to browser clients over WebSocket.

## Architecture

```
generate_plan.py  →  plan.md  →  orchestrator.py  →  results.json
                                      │
                                 status_bridge.py  ──POST──→  serve.py  ──WS──→  browser
                                                                │
                                                     Jinja2 templates/
```

**orchestrator.py** — Core LangGraph state machine. Defines the graph nodes (`master_node`, `sub_agent_node`, `manager_review_node`, `specialist_review_node`, `synthesis_node`) and conditional routing between them. Parses `plan.md` into dataclasses (`SubAgentSpec`, `ManagerSpec`, `ReviewerSpec`, `TaskSpec`), then invokes a compiled `StateGraph`. All LLM calls go through `langchain-openai` (`ChatOpenAI`). State is `OrchestratorState` (TypedDict). Dynamic model selection queries `/models` at startup and auto-selects by role (opus-tier for execution, sonnet-tier for reviews).

**generate_plan.py** — Uses the `openai` SDK to generate a structured `plan.md` from a natural-language project description. The system prompt enforces the plan schema.

**status_bridge.py** — Fire-and-forget bridge between orchestrator and dashboard. Maintains an in-memory event log, roster registry, transcript log, and file attention tracker. Pushes serialized state to `serve.py` via background `threading.Thread` POSTs.

**serve.py** — FastAPI WebSocket server with Jinja2 template rendering. Receives POST `/update` from the bridge and broadcasts to all connected WebSocket clients. Supports multi-session tracking keyed by session_id. Replays all active session states to new connections.

## Dashboard (Jinja2 Templates)

The dashboard is composed from `templates/` via Jinja2 `{% include %}` directives, served as a single HTML response:

```
templates/
├── base.html                     # Composition shell
├── styles/base.css               # Glass morphism + holographic palette
├── components/                   # HTML partials (top bar, control bar, panels)
└── scripts/                      # 16 JS modules
    ├── colors.js, constants.js   # Palette + animation config
    ├── nodes.js, force.js        # Node model + force simulation
    ├── draw_*.js (5 files)       # Canvas rendering (hex nodes, edges, particles, effects, tools)
    ├── bloom.js, camera.js       # Post-processing + pan/zoom
    ├── render.js                 # 60fps render loop
    ├── websocket.js              # WS + applyStatus()
    ├── api.js, panels.js         # API calls + UI panel logic
    └── init.js                   # Boot + keyboard shortcuts
```

Node hierarchy: NEXUS (master, 36px) → SENTINEL (manager, 28px) → DRONE (worker, 22px) → PROBE (reviewer, 18px) → SHARD (parallel sub-agent, 16px).

## Plan Schema

Plans are markdown files with sections: `## Objective`, `## Sub-Agents` (### AGENT:), `## Managers` (### MANAGER:), `## Tasks` (### TASK-NNN:), and optionally `## Custom Reviewers` (### REVIEWER:). See `plan.example.md` for a complete reference. The parser in `orchestrator.py` uses regex to extract fields like `- description:`, `- tools:`, `- reviewers: [...]`, `- depends_on: [...]`.

## Key Configuration

- `MAX_REVISIONS = 3` in `orchestrator.py` — max revision cycles per task across both review gates
- `NORT_PORT` env var — dashboard server port (default 8000). `QUARM_PORT` also accepted.
- `NORT_SERVER` env var — bridge target URL (default `http://localhost:8000`). `QUARM_SERVER` also accepted.
- `NORT_SECRET` env var — optional shared secret for bridge-to-server auth. `QUARM_SECRET` also accepted.
- `.env` file — must contain `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- LLM model is auto-selected at runtime; optional `- model:` field in plan.md overrides per agent/task

## Review Flow

Tasks follow: `pending → in_progress → in_manager_review → in_specialist_review → done`. On FAIL/FLAG from either gate, the task loops back to `revision` status and the sub-agent re-executes with consolidated feedback. After `MAX_REVISIONS`, the result is force-accepted. Custom reviewers defined in `plan.md` override builtins with the same name.
