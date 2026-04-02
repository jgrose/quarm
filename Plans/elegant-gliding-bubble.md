# README.md Rewrite Plan

## Context

The current README.md is outdated — it only describes the orchestrator's 4-layer architecture, built-in reviewers, and basic CLI usage. It doesn't mention the live dashboard, WebSocket integration, 54 API endpoints, 20 dashboard panels, sessions, agent registry, tolerance system, cost tracking, or any of the 60+ features shipped. The install instructions reference `langchain-anthropic` when the codebase uses `langchain-openai`. The "Extending" section lists features that are all now implemented.

**Goal**: Rewrite README.md to comprehensively document the current state of NORT — scannable in 30 seconds, actionable quickstart in < 5 minutes, with full feature coverage.

## Approach

Single agent rewrites `README.md` in place. No other files modified. The agent reads current source files for accuracy and uses the exploration data below.

## File to Modify

- `/home/localuser/projects/quarm/README.md` — full rewrite

## Source Files to Read for Accuracy

- `CLAUDE.md` — architecture diagram, node hierarchy, config
- `roadmap.md` — completed features list (authoritative)
- `plan.example.md` — canonical plan schema example
- `serve.py` lines 1-50 — PORT config, imports
- `orchestrator.py` lines 1-50 — docstring, constants (MAX_REVISIONS, DEFAULT_TOLERANCE)

## README Structure (18 sections)

### 1. Hero Block
- Existing logo image (`assets/images/quarm_logo.png`)
- One-line tagline: "4-layer multi-agent orchestrator with a Tron-themed isometric city dashboard"
- Stat badges: `54 API endpoints` | `20 panels` | `78 tests` | `17 canvas systems`

### 2. Table of Contents
- Linked anchors for all major sections (10-12 entries)

### 3. What is NORT?
- 3-4 sentence description covering: markdown plan → specialist agents → two quality gates → isometric city dashboard
- 6 bullet differentiators: LangGraph state machine, two quality gates, dynamic model selection, tolerance system, pixel-art dashboard, checkpoint/resume
- Two dashboard views: City (`/`) and Flow (`/flow`)

### 4. Quickstart
- **Prerequisites**: Python 3.11+, OpenAI-compatible LLM endpoint
- **Install**: `pip install langgraph langchain langchain-openai python-dotenv fastapi uvicorn python-multipart jinja2`
- **.env**: `OPENAI_API_KEY` + `OPENAI_BASE_URL`
- **Generate a plan**: `python generate_plan.py "description"`
- **Run with dashboard**: Terminal 1: `python serve.py` / Terminal 2: `python orchestrator.py plan.md` / Browser: `http://localhost:8000/`
- **Run headless**: `python orchestrator.py plan.md` → `results.json`

### 5. Architecture
- ASCII data flow diagram (from CLAUDE.md): `generate_plan.py → plan.md → orchestrator.py → results.json` with `status_bridge.py → serve.py → browser` branch
- 4-layer pipeline diagram (from current README, cleaned up): MASTER → SUB-AGENT → MANAGER REVIEW → SPECIALIST REVIEW → DONE
- File map table (13 Python files with one-line purpose each)

### 6. Plan Format
- Four sections: `## Objective`, `## Sub-Agents`, `## Managers`, `## Tasks`, optional `## Custom Reviewers`
- Condensed inline example (1 agent, 1 manager, 2 tasks with dependency)
- Field reference tables for tasks, agents, managers
- Mention `python validate_plan.py plan.md` for validation

### 7. Quality Gates
- Task lifecycle: `pending → in_progress → in_manager_review → in_specialist_review → done`
- Revision loop: FAIL/FLAG → consolidated feedback → sub-agent revises (MAX_REVISIONS=3)
- Built-in reviewers table (security_engineer, ux_designer, user_tester) — keep from current README
- Custom reviewers — keep from current README
- **Tolerance system**: precedence chain, earned bonus (+1 for avg > 8 over 5+ runs), specialist skip (score >= 9), presets (Prototype/Production/Audit)

### 8. Dashboard — City View (`/`)
- One-paragraph description of the isometric Sim City SNES-style pixel art city
- Key visual features as bullets:
  - 11 Tron-themed buildings (table: 6 idle + 5 work with task-state mapping)
  - 5 agent tiers (NEXUS 36px → SHARD 16px)
  - 6 ambient programs with walking, bunker entry, light cycle trails
  - Day/night cycle (120s), weather (data rain + lightning storms)
  - Building upgrades at task thresholds (3/7/15)
  - Bloom post-processing, depth particles

### 9. Dashboard — Flow View (`/flow`)
- Holographic node graph with void background (#050510)
- Claude spark logo on NEXUS, free-form force layout
- Same WebSocket data, focused on agent status
- "CITY VIEW" button to switch back

### 10. Dashboard Panels & Shortcuts
- Table of all 20 panels with keyboard shortcut and one-line purpose
- Full keyboard shortcut reference: Q, R, M, A, C, L, P, D, $, ?, ESC

### 11. Canvas Systems
- Table of 17 draw modules with one-line descriptions
- Note: 60fps render loop, bloom post-processing, render caching, viewport culling, idle pause

### 12. API Reference
- Grouped endpoint tables (HTTP method | path | description):
  - Plan Management (7 endpoints)
  - Sessions (3)
  - Agents (12)
  - Teams (5)
  - Analytics (5)
  - Artifacts & Output (7)
  - Configuration (6)
  - RAG (2)
  - Approvals (2)
  - Models (2)
  - System (2: health, WebSocket)

### 13. Agent Tools
- Table of available tools: read_file, write_file, execute_code, browse_url, web_search, etc.
- Tool approval system (execute_code requires human approval)
- Content scanning for secrets/injection patterns

### 14. Configuration
- Environment variables table (NORT_PORT, NORT_SERVER, NORT_SECRET, OPENAI_API_KEY, OPENAI_BASE_URL)
- config.json overview (tolerance, models, webhook)
- Agent registry (`agents/registry.json`)

### 15. Testing
- `pytest tests/` — 78 tests in 0.10s
- Test file list with categories
- Note: conftest.py stubs all external dependencies for isolation

### 16. Project Structure
- Directory tree showing: Python files, templates/, tests/, agents/, plans/, artifacts/, output/

### 17. results.json
- Output format example (keep from current README, add artifacts and validation fields)
- Revision count as quality signal

### 18. Security
- Content scanner, path traversal prevention, tool approval, sandbox mode

## What Gets Removed from Current README

- **"Extending" section** — all 5 items are now implemented (parallel reviewers, real tools, checkpointing, human-in-loop, quality metrics)
- **Incorrect install** — `langchain-anthropic` → `langchain-openai`, `ANTHROPIC_API_KEY` → `OPENAI_API_KEY`

## What Gets Kept from Current README

- Logo image reference
- 4-layer architecture ASCII diagram (cleaned up)
- Built-in reviewers table
- Reviewer assignment guide
- Custom reviewers section
- results.json example (expanded)

## Execution

**Single agent** — one Engineer agent rewrites README.md. No parallelization needed since it's a single file. The agent should:

1. Read: `README.md`, `CLAUDE.md`, `roadmap.md` (completed section), `plan.example.md`, `serve.py` (first 50 lines + grep for all route decorators), `orchestrator.py` (first 50 lines)
2. Write the full README.md following the 18-section structure above
3. Verify: no broken markdown, all sections present, accurate install instructions

## Verification

1. Read the output README.md and verify all 18 sections are present
2. Confirm install command matches actual dependencies (`langchain-openai`, not `langchain-anthropic`)
3. Confirm API endpoint count matches (grep `@app.` in serve.py)
4. Confirm keyboard shortcuts match init.js
5. Confirm building count matches draw_locations.js LOCATION_DEFS
