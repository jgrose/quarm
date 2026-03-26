# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never include Co-Authored-By, "Claude", or any AI attribution in commit messages or anywhere in the codebase.**

## What This Is

QUARM is a 4-layer multi-agent orchestrator built on LangGraph. It takes a structured plan (markdown), dispatches tasks to specialist sub-agents, and runs each result through two quality gates: a domain manager review and a specialist reviewer panel (security engineer, UX designer, user tester). Results and a final executive report are written to `results.json`.

## Commands

```bash
# Install dependencies
pip install langgraph langchain langchain-anthropic python-dotenv anthropic fastapi uvicorn python-multipart

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
                                 status_bridge.py  ──POST──→  serve.py  ──WS──→  quarm_hq.html
```

**orchestrator.py** — Core LangGraph state machine. Defines the graph nodes (`master_node`, `sub_agent_node`, `manager_review_node`, `specialist_review_node`, `synthesis_node`) and conditional routing between them. Parses `plan.md` into dataclasses (`SubAgentSpec`, `ManagerSpec`, `ReviewerSpec`, `TaskSpec`), then invokes a compiled `StateGraph`. All LLM calls go through `langchain-openai` (`ChatOpenAI`). State is `OrchestratorState` (TypedDict). Dynamic model selection queries `/models` at startup and auto-selects by role (opus-tier for execution, sonnet-tier for reviews).

**generate_plan.py** — Uses the `openai` SDK to generate a structured `plan.md` from a natural-language project description. The system prompt enforces the plan schema.

**status_bridge.py** — Fire-and-forget bridge between orchestrator and dashboard. Maintains an in-memory event log and roster registry. Pushes serialized state to `serve.py` via background `threading.Thread` POSTs. Falls back to `urllib` if `requests` is not installed.

**serve.py** — FastAPI WebSocket server. Receives POST `/update` from the bridge and broadcasts to all connected WebSocket clients. Replays last known state to new connections.

**quarm_hq.html** — Single-file dashboard UI (HTML/CSS/JS). Connects to `ws://localhost:8000/ws` and renders agent rooms, task progress, and event logs in real time.

## Plan Schema

Plans are markdown files with sections: `## Objective`, `## Sub-Agents` (### AGENT:), `## Managers` (### MANAGER:), `## Tasks` (### TASK-NNN:), and optionally `## Custom Reviewers` (### REVIEWER:). See `plan.example.md` for a complete reference. The parser in `orchestrator.py` uses regex to extract fields like `- description:`, `- tools:`, `- reviewers: [...]`, `- depends_on: [...]`.

## Key Configuration

- `MAX_REVISIONS = 3` in `orchestrator.py` — max revision cycles per task across both review gates
- `QUARM_PORT` env var — dashboard server port (default 8000)
- `QUARM_SERVER` env var — bridge target URL (default `http://localhost:8000`)
- `QUARM_SECRET` env var — optional shared secret for bridge-to-server auth
- `.env` file — must contain `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- LLM model is auto-selected at runtime; optional `- model:` field in plan.md overrides per agent/task

## Review Flow

Tasks follow: `pending → in_progress → in_manager_review → in_specialist_review → done`. On FAIL/FLAG from either gate, the task loops back to `revision` status and the sub-agent re-executes with consolidated feedback. After `MAX_REVISIONS`, the result is force-accepted. Custom reviewers defined in `plan.md` override builtins with the same name.
