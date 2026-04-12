# NORT — Technical Rebuild Brief

## What It Is

A **multi-agent orchestrator** that takes a markdown plan, dispatches tasks to LLM-powered specialist agents, runs each result through two independent quality gates, and visualizes the entire process in real-time as an animated isometric pixel-art city.

## Core Loop (4 layers)

```
Plan (markdown) → Master dispatches tasks by dependency order
  → Sub-Agent executes (with tools: read/write files, execute code, browse web)
    → Manager Review (domain expert, scores 1-10, PASS/FAIL)
      → Specialist Panel (security, UX, user testing — scores 1-10)
        → FAIL? Loop back with feedback (max 3 revisions)
        → PASS? Store result, pick next task
→ Synthesis report + assembled output folder
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Python, LangGraph (StateGraph with conditional routing) |
| LLM | OpenAI-compatible API (any provider via OPENAI_BASE_URL) |
| Server | FastAPI + Uvicorn, Jinja2 templates, WebSocket |
| Database | SQLite (analytics), JSON files (registry, queue, config) |
| Dashboard | Vanilla JS, HTML5 Canvas 2D (no framework, no build step) |
| Tests | pytest (78+ tests), Playwright (smoke tests) |

## Three Subsystems to Build

### 1. The Orchestrator (~4 Python files, ~100K lines total)

A LangGraph state machine with 6 nodes: `master_node` (dispatches), `sub_agent_node` (executes), `manager_review_node` (gate 1), `specialist_review_node` (gate 2), `synthesis_node` (final report), `composition_node` (cross-file coherence check). Tasks have dependencies (`depends_on`), a tolerance system (configurable score threshold that can override FAIL verdicts), and checkpoint/resume for crash recovery. Output is assembled into a folder with MANIFEST.md + per-task artifacts with revision snapshots.

### 2. The Server (~1 Python file, ~50K lines)

FastAPI serving two HTML pages (`/` city view, `/flow` node graph view) and 57 REST API endpoints covering: plan CRUD + queue + generation, agent registry with versioning/clone/retire/import/export, team presets, tolerance config with presets (Prototype/Production/Audit), cost + score analytics (SQLite), artifact browsing + ZIP download, model management, RAG knowledge base, tool approval workflow, and health checks. A WebSocket at `/ws` broadcasts live state from the orchestrator via a bridge module (`status_bridge.py`) that POSTs state updates from the orchestrator process to the server.

### 3. The Dashboard (~34 JS modules, ~8000 lines, no build step)

Two views sharing the same WebSocket data:

**City View** — An isometric (Sim City SNES-style) pixel-art city with:
- 12 Tron-themed buildings mapped to task states (Code Forge = in_progress, Tribunal = in_manager_review, Data Vault = done, etc.)
- 6 ambient pixel-art programs (agents) that walk between buildings, enter bunkers, carry task sprites, and level up with XP
- 17 canvas draw systems composited at 60fps: background grid, buildings, programs, hex agent nodes, bezier edges with particle trails, tool call cards, message bubbles, token context bars, cost pills, dependency lines, discovery cards, thought bubbles, bloom post-processing, day/night cycle (120s), weather (data rain + lightning), minimap
- 20 overlay panels (queue, config, costs, agents, chat, roster, dependencies, performance profiler, output browser, review analytics, DAG, timeline, transcript, etc.)
- Keyboard shortcuts for every panel (Q, R, M, A, C, L, P, D, $, ?)

**Flow View** — A holographic node graph with void background (#050510), depth particles, pulsing hex grid, Claude spark logo on the master node, free-form force-directed layout (no zone banding). Same data, focused on agent status clarity.

## Key Design Decisions

1. **No frontend framework** — all 34 JS modules are vanilla JS inlined via Jinja2 `{% include %}`. Zero build step. This keeps iteration instant but means no TypeScript, no component model, no hot reload.

2. **Canvas 2D, not WebGL** — all rendering is 2D canvas with manual compositing. Bloom is multi-pass box blur on an offscreen canvas. This keeps it portable but limits to ~100 nodes before performance degrades.

3. **Bridge architecture** — the orchestrator runs as a separate process from the server. `status_bridge.py` POSTs state snapshots to the server via HTTP, which broadcasts to browsers via WebSocket. This decouples execution from visualization.

4. **Tolerance over strict pass/fail** — review verdicts are scores (1-10), not binary. A configurable tolerance threshold can auto-override FAIL if the score is "close enough." Precedence chain: per-agent config > per-task field > per-plan agent > global config > default (6). Agents that consistently score > 8 earn a +1 bonus automatically.

5. **Everything is a panel** — the UI has 20 toggleable panels rather than fixed layouts. Each panel is a self-contained HTML + CSS + JS file. New features are added by creating a new panel file and including it.

## Data Shapes

**Plan** (markdown): `## Objective`, `## Sub-Agents` (name, description, tools), `## Managers` (expertise_blend, oversees), `## Tasks` (agent, depends_on, reviewers, tolerance)

**WebSocket payload** (JSON): `{ session_id, phase, tasks: [{id, title, agent, status, task_tokens, depends_on, tool_calls, last_score}], tokens_used, log[], transcript[], files_touched[] }`

**Output**: `results.json` (task_results, quality_log, artifacts, validation, coherence_report) + `output/{plan_id}/` folder with assembled files + MANIFEST.md

## What Makes It Non-Trivial

- The canvas rendering pipeline (17 composited systems with bloom, particles, viewport culling, idle detection, settled-skip for force simulation)
- The tolerance system (5-level precedence chain with earned bonuses and conditional specialist skip)
- The agent registry (CRUD + versioning + rollback + clone + retire + import/export + team presets + performance tracking)
- The checkpoint/resume system (atomic writes, structural validation, corrupted file recovery)
- The multi-session architecture (concurrent plans, per-session state isolation, session switching without layout jitter via localStorage persistence)
