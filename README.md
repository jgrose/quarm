<p align="center">
  <img src="assets/images/quarm_logo.png" alt="NORT" width="500">
</p>

# NORT -- Multi-Agent Orchestrator

4-layer multi-agent orchestrator built on LangGraph with a Tron-themed isometric city dashboard.

`57 API endpoints` | `20 dashboard panels` | `84 tests` | `17 canvas systems`

---

## Table of Contents

- [What is NORT?](#what-is-nort)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Plan Format](#plan-format)
- [Quality Gates](#quality-gates)
- [Dashboard -- City View](#dashboard----city-view)
- [Dashboard -- Flow View](#dashboard----flow-view)
- [Dashboard Panels and Shortcuts](#dashboard-panels-and-shortcuts)
- [Canvas Systems](#canvas-systems)
- [API Reference](#api-reference)
- [Agent Tools](#agent-tools)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [results.json](#resultsjson)
- [Security](#security)

---

## What is NORT?

NORT takes a markdown plan, dispatches tasks to specialist sub-agents, runs each result through two quality gates (domain manager review + specialist reviewer panel), and visualizes the entire process as an animated isometric city.

- **Two quality gates per task** -- domain manager review followed by a specialist reviewer panel (security engineer, UX designer, user tester)
- **Tron-themed city dashboard** -- isometric Sim City SNES-style pixel art with 12 buildings, walking programs, day/night cycle, and weather
- **Flow dashboard** -- holographic node graph with free-form force layout at `/flow`
- **Dynamic model selection** -- queries your LLM provider's `/models` endpoint at startup, auto-selects opus-tier for execution and sonnet-tier for reviews
- **Real agent tools** -- web search, code execution, file I/O, RAG knowledge base, and URL browsing with human-in-the-loop approval for dangerous operations
- **Output assembly** -- merges per-task artifacts into a single deliverable folder with MANIFEST.md, post-assembly validation, artifact versioning across revisions, and downloadable ZIP

---

## Quickstart

### Prerequisites

- Python 3.11+
- An OpenAI-compatible LLM provider (OpenAI, Anthropic via proxy, local models, etc.)

### Install

```bash
pip install langgraph langchain langchain-openai python-dotenv fastapi uvicorn python-multipart jinja2
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Generate a plan

```bash
python generate_plan.py "Build a web dashboard for AWS cost monitoring"
# Writes plan.md with agents, managers, tasks, and reviewer assignments
```

### Run with the dashboard

```bash
# Terminal 1 -- start the dashboard server
python serve.py

# Terminal 2 -- run the orchestrator
python orchestrator.py plan.md

# Browser -- open the dashboard
http://localhost:8000/
```

### Run headless

```bash
python orchestrator.py plan.md
# Streams review decisions to stdout, writes results.json on completion
```

---

## Architecture

### Data Flow

```
generate_plan.py  ->  plan.md  ->  orchestrator.py  ->  results.json
                                        |
                                   status_bridge.py  --POST-->  serve.py  --WS-->  browser
                                                                  |
                                                       Jinja2 templates/
```

### 4-Layer Pipeline

```
MASTER  (Program Manager -- dispatches tasks, synthesises final report)
  |
  v
SUB-AGENT  (executes task -- specialist with a defined role and toolset)
  |
  v
MANAGER REVIEW  -- Quality Gate 1 -----------------------------------------------
  |  Blended domain expertise. Reviews for correctness, completeness,
  |  and adherence to requirements.
  |  FAIL -> feedback -> sub-agent revises
  |  PASS v
  v
SPECIALIST REVIEW PANEL  -- Quality Gate 2 --------------------------------------
  |-- Security Engineer   Checks: OWASP, auth, secrets, input validation,
  |                       access control, dependency risk, data handling
  |
  |-- UX/UI Designer      Checks: visual hierarchy, WCAG accessibility,
  |                       information architecture, interaction patterns,
  |                       typography, cognitive load
  |
  +-- User Tester         Checks: clarity of purpose, ease of first use,
                          plain language, workflow intuitiveness,
                          actual value delivered to a real user
  |
  |  Any reviewer FLAGs -> consolidated feedback -> sub-agent revises
  |  All PASS v
  v
DONE  (result stored, Master picks next task)
```

### File Map

| File | Purpose |
|---|---|
| `orchestrator.py` | Core LangGraph state machine -- graph nodes, conditional routing, plan parser, model selection |
| `serve.py` | FastAPI server -- dashboard UI, WebSocket broadcasts, plan queue, 57 API endpoints |
| `generate_plan.py` | Plan generator -- natural-language description to structured plan.md |
| `status_bridge.py` | Fire-and-forget bridge -- event log, roster registry, transcript log, file attention tracker |
| `tools.py` | Tool registry -- 8 core tools with approval system and content scanning |
| `tools_web.py` | Web tool implementations -- browser and search backends |
| `validate_plan.py` | Plan schema validator -- CLI and importable validation |
| `model_config.py` | Model selection -- allowed model list, role-based auto-selection |
| `agent_registry.py` | Persistent agent definitions -- performance tracking, versioning, import/export |
| `tracking.py` | Run analytics -- per-agent cost tracking, score history |
| `content_scanner.py` | Security scanner -- detects secrets, credentials, malicious patterns in artifacts |
| `checkpoint.py` | Run checkpointing -- save/resume orchestrator state mid-run |
| `rag.py` | RAG knowledge base -- vector store for cross-project context |
| `specialization.py` | Agent learning -- auto-enhance agent descriptions based on task performance |

---

## Plan Format

Plans are markdown files with five sections: `Objective`, `Sub-Agents`, `Managers`, `Tasks`, and optionally `Custom Reviewers`.

### Example

```markdown
## Objective
Build a web dashboard for AWS cost monitoring.

## Sub-Agents
### AGENT: backend_engineer
- description: Python/FastAPI backend engineer. Builds secure REST APIs.
- tools: execute_code, write_file, read_file

### AGENT: frontend_engineer
- description: React/TypeScript frontend engineer. Builds accessible UI.
- tools: write_file, design_ui

## Managers
### MANAGER: engineering_director
- title: Engineering Architecture Director
- description: Reviews backend code and security architecture.
- expertise_blend: [API_design, Python_architecture, AWS_cloud]
- oversees: [backend_engineer]

## Tasks
### TASK-001
- title: Build cost data API
- agent: backend_engineer
- description: Build a FastAPI service with GET /costs endpoint.
- task_type: [code, api, backend]
- reviewers: [security_engineer]
- depends_on: []

### TASK-002
- title: Build dashboard UI
- agent: frontend_engineer
- description: React dashboard with cost charts.
- task_type: [code, ui, frontend]
- reviewers: [security_engineer, ux_designer, user_tester]
- depends_on: [TASK-001]
```

### Agent Fields

| Field | Required | Description |
|---|---|---|
| `description` | Yes | Role, expertise, and expected output format |
| `tools` | Yes | Comma-separated tool names from the tool registry |
| `model` | No | Override the auto-selected LLM model for this agent |

### Manager Fields

| Field | Required | Description |
|---|---|---|
| `title` | Yes | Display title for the manager |
| `description` | Yes | Domain expertise and review focus |
| `expertise_blend` | Yes | List of domain tags for blended review |
| `oversees` | Yes | List of agent names this manager reviews |

### Task Fields

| Field | Required | Description |
|---|---|---|
| `title` | Yes | Short task name |
| `agent` | Yes | Agent name to execute this task |
| `description` | Yes | Detailed instructions for the agent |
| `task_type` | Yes | Category tags for reviewer matching |
| `reviewers` | Yes | List of specialist reviewers (can be empty `[]`) |
| `depends_on` | Yes | List of task IDs that must complete first |
| `tolerance` | No | Per-task review score threshold (1-10) |
| `model` | No | Override the auto-selected LLM model for this task |

### Validation

```bash
python validate_plan.py plan.md
```

See `plan.example.md` for a complete reference plan.

---

## Quality Gates

### Task Lifecycle

```
pending -> in_progress -> in_manager_review -> in_specialist_review -> done
                ^                                        |
                |              FAIL / FLAG               |
                +----------------------------------------+
                            (revision)
```

On FAIL or FLAG from either gate, the task loops back to `revision` status and the sub-agent re-executes with consolidated feedback. After `MAX_REVISIONS` (default 3), the result is force-accepted.

### Built-in Reviewers

| Reviewer | Domain | When to assign |
|---|---|---|
| `security_engineer` | OWASP, auth, secrets, least-privilege | Any task with code, APIs, auth, config, infrastructure |
| `ux_designer` | WCAG, visual hierarchy, interaction patterns | Any user-facing UI, dashboard, form, report |
| `user_tester` | First-use clarity, plain language, workflow | Any output a non-technical user will touch |

### Reviewer Assignment Guide

```
Backend API only:          reviewers: [security_engineer]
Frontend component:        reviewers: [security_engineer, ux_designer, user_tester]
Internal data pipeline:    reviewers: []
User-facing documentation: reviewers: [ux_designer, user_tester]
Auth architecture:         reviewers: [security_engineer]
```

### Custom Reviewers

Define project-specific reviewers in plan.md alongside the builtins:

```markdown
## Custom Reviewers
### REVIEWER: compliance_officer
- title: Regulatory Compliance Officer
- description: Reviews outputs for HIPAA/GDPR compliance...
- focus_areas: [data_minimization, consent_flows, audit_logging, PII_handling]
- applies_to: [data, api, auth, report]
```

Custom reviewers with the same name as a builtin will override the builtin.

### Tolerance System

Review tolerance controls the minimum score (1-10) a task must achieve to pass a quality gate.

**Precedence chain** (highest priority first):
1. Config per-agent override
2. Task-level `- tolerance:` field
3. Plan per-agent override
4. Config global setting
5. `DEFAULT_TOLERANCE` (6)

**Earned tolerance bonus**: Agents with `avg_score > 8` over 5+ completed tasks automatically earn a `+1` tolerance bonus.

**Conditional review skipping**: Tasks scoring 9+ at manager review can skip the specialist panel entirely (configurable via `skip_specialist_on_high_score`).

**Presets**: One-click profiles in the tolerance config panel:

| Preset | Tolerance | Use case |
|---|---|---|
| Prototype | 8 | Rapid iteration, accept most output |
| Production | 5 | Standard quality bar |
| Audit | 3 | Maximum scrutiny, nearly everything gets reviewed |

---

## Dashboard -- City View

The city view (`/`) renders an isometric Sim City SNES-style pixel art world. Agent programs walk between buildings, enter through animated doors, and carry out tasks in real time as the orchestrator runs.

### Buildings

| Building | Category | Task State | Description |
|---|---|---|---|
| END OF LINE | Idle | -- | Lounge for idle programs |
| CYCLE ARENA | Idle | -- | Arena for idle programs |
| I/O TOWER | Idle | -- | Communications tower |
| DISC RING | Idle | -- | Training ring |
| RECOGNIZER PAD | Idle | -- | Landing pad |
| PORTAL | Idle | -- | Entry/exit portal |
| CODE FORGE | Work | `in_progress` | Where agents execute tasks |
| TRIBUNAL | Work | `in_manager_review` | Manager review chamber |
| ANALYSIS BAY | Work | `in_specialist_review` | Specialist review lab |
| RECOMPILE | Work | `revision` | Revision workshop |
| DATA VAULT | Work | `done` | Completed task storage |
| DEREZZED | Work | `failed` | Failed task graveyard |

### Agent Tiers

| Tier | Role | Sprite Size |
|---|---|---|
| NEXUS | Master (program manager) | 36px |
| SENTINEL | Manager (domain reviewer) | 28px |
| DRONE | Worker (sub-agent) | 22px |
| PROBE | Reviewer (specialist) | 18px |
| SHARD | Parallel sub-agent | 16px |

### Ambient Features

- **Day/night cycle** -- 120-second rotation through dawn, day, dusk, and night phases
- **Weather** -- data rain particles and lightning strikes between buildings during storms
- **Cyber roads** -- glowing circuit-line roads connecting buildings
- **Cyber trees** -- round canopy trees, data bushes, and light poles along roads
- **Light cycle trails** -- thick bright trails for long-distance agent travel
- **Sound design** -- ambient hum, footsteps, door whoosh, level-up chimes, thunder (toggle with sound config)

### Building Upgrades

Buildings visually evolve as tasks complete through them:

| Threshold | Level | Effect |
|---|---|---|
| 3 completions | Level 1 | First visual upgrade |
| 7 completions | Level 2 | Second visual upgrade |
| 15 completions | Level 3 | Final visual upgrade |

---

## Dashboard -- Flow View

The flow view (`/flow`) provides a holographic node graph visualization. Agents and tasks appear as connected nodes in a free-form force-directed layout against a void background with depth particles and a pulsing hex grid.

- Route: `http://localhost:8000/flow`
- Same WebSocket data as the city view
- Claude spark logo watermark
- Nodes colored by tier and status

---

## Dashboard Panels and Shortcuts

### Panels

20 panel overlays in `templates/components/panels/`, plus the agent chat and agent list side drawers.

| Panel | Shortcut | Purpose |
|---|---|---|
| Agent Chat | `C` | Group-chat-style event log with per-agent avatars, tier icons, colored bubbles, verdict badges |
| Agent List | `L` | Left-side drawer showing all active agents grouped by session with live status |
| Plans List | `P` | Plan browser with session switching (sub-panel of Agent List) |
| Queue | `Q` | Plan queue with drag-to-reorder and run controls |
| Roster | `R` | Agent roster with Tron names, XP bars, leveling |
| Agents | `A` | Agent registry -- create, edit, version, clone, retire, import/export |
| Help | `?` | Keyboard shortcut reference |
| Config | -- | Settings toggles for all visual systems |
| Model Config | -- | LLM model selection and role assignment |
| Tolerance Config | -- | Per-agent/global tolerance sliders with presets |
| Cost Panel | -- | Per-agent cost tracking with LIVE and HISTORY tabs |
| Review Analytics | -- | Reviewer pass/fail rates, score distributions |
| DAG Panel | -- | Task dependency graph visualization |
| Output Browser | -- | File tree with syntax-highlighted preview |
| Performance | -- | Per-system timing with EMA smoothing |
| Ledger | -- | Run history and score ledger |
| Completion | -- | Final report overlay on run completion |
| Plan Viewer | -- | Full plan markdown viewer overlay |
| Thinking | -- | Agent reasoning trace display |
| Transcript | -- | Full session transcript log |
| Timeline | -- | Task execution timeline visualization |
| File Attention | -- | Files touched by agents during the run |

### Keyboard Reference

| Key | Action |
|---|---|
| `C` | Toggle agent chat panel |
| `L` | Toggle agent list panel |
| `P` | Toggle plans list |
| `Q` | Toggle queue panel |
| `R` | Toggle roster panel |
| `A` | Toggle agents panel |
| `M` | Toggle minimap |
| `D` | Toggle dependency edges |
| `?` | Toggle help overlay |
| `ESC` | Close topmost panel |

---

## Canvas Systems

| Module | Description |
|---|---|
| `draw_locations.js` | Isometric building sprites, placement, door animations, upgrade visuals |
| `draw_programs.js` | Agent program sprites (4-directional walk animation), tier-based sizing |
| `draw_agents.js` | Agent labels, status indicators, task badges above programs |
| `draw_edges.js` | Connection lines between buildings with animated particles |
| `draw_effects.js` | Completion effects, level-up flashes, status transition particles |
| `draw_background.js` | Hex grid background with pulsing glow, isometric ground plane |
| `draw_weather.js` | Data rain particles, lightning strikes between buildings |
| `draw_minimap.js` | Corner overview map with viewport rectangle and click-to-pan |
| `draw_tools.js` | Floating tool cards showing active tool calls per agent |
| `draw_dependencies.js` | Animated dashed bezier lines between dependent tasks, BLOCKED badges |
| `draw_roster.js` | Roster panel rendering with XP bars and tier icons |
| `draw_cost.js` | Cost panel bar charts and history drill-down |
| `draw_atmosphere.js` | Day/night cycle lighting, ambient glow, sky color transitions |
| `draw_bubbles.js` | Speech/thought bubbles above agents during task execution |
| `draw_thought_bubbles.js` | Extended thought visualization for agent reasoning |
| `draw_discoveries.js` | Discovery notification animations for new findings |
| `draw_context.js` | Contextual information overlays on hover/select |

**Performance**: 60fps render loop with bloom post-processing, offscreen grid canvas, glow cache (500 cap), viewport culling, force simulation idle-skip, and conditional bloom bypass.

---

## API Reference

57 endpoints served by `serve.py`. Grouped by category:

### Pages (2)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | City dashboard (HTML) |
| GET | `/flow` | Flow dashboard (HTML) |

### WebSocket (1)

| Method | Endpoint | Description |
|---|---|---|
| WS | `/ws` | Real-time state broadcasts |

### Bridge (1)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/update` | Receive state from orchestrator bridge |

### Plan Management (6)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/generate` | Generate a plan from a description |
| GET | `/api/plans` | List all plans in queue order |
| GET | `/api/plans/{id}` | Get a single plan's content and metadata |
| POST | `/api/plans/reorder` | Reorder the plan queue |
| POST | `/api/plans/{id}/run` | Start orchestrator for a plan |
| DELETE | `/api/plans/{id}` | Remove a plan |

### Orchestrator Control (1)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/plans/{id}/stop` | Stop a running orchestrator |

### Models (2)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/models` | List available LLM models |
| POST | `/api/models` | Update model configuration |

### Sessions and History (3)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/sessions` | List active sessions |
| GET | `/api/transcript/{session_id}` | Get session transcript |
| GET | `/api/files/{session_id}` | Get files touched in session |

### Health (1)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health check |

### Analytics (4)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/analytics/costs/{run_id}` | Cost breakdown for a run |
| GET | `/api/analytics/costs` | Aggregate cost analytics |
| GET | `/api/analytics/scores` | Score analytics across runs |
| GET | `/api/review-stats` | Reviewer pass/fail statistics |

### Configuration (4)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/config` | Get current configuration |
| POST | `/api/config` | Update configuration |
| GET | `/api/tolerance` | Get tolerance settings |
| POST | `/api/tolerance` | Update tolerance settings |

### Specializations (1)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/specializations` | Get agent specialization data |

### Webhooks (1)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/webhook/test` | Test webhook delivery |

### RAG Knowledge Base (2)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/rag/stats` | Knowledge base statistics |
| GET | `/api/rag/search` | Search the knowledge base |

### Approvals (2)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/approvals` | List pending tool approvals |
| POST | `/api/approvals/{id}` | Approve or reject a tool call |

### Artifacts and Output (7)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/artifacts/{plan_id}` | List artifacts for a plan |
| GET | `/api/artifacts/{plan_id}/file` | Get a single artifact file |
| GET | `/api/artifacts/{plan_id}/download` | Download all artifacts as ZIP |
| GET | `/api/artifacts/{plan_id}/revisions/{task_id}` | List revision snapshots |
| GET | `/api/output/{plan_id}/files` | List output files |
| GET | `/api/output/{plan_id}/file` | Get a single output file |
| GET | `/output/{plan_id}/download` | Download assembled output |

### Agent Registry (13)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/agents/export` | Export all agents as JSON |
| POST | `/api/agents/import` | Import agents from JSON |
| POST | `/api/agents/import-single` | Import a single agent |
| GET | `/api/agents` | List all registered agents |
| GET | `/api/agents/{type}/{name}` | Get a specific agent |
| POST | `/api/agents/{type}` | Create a new agent |
| PUT | `/api/agents/{type}/{name}` | Update an agent |
| DELETE | `/api/agents/{type}/{name}` | Delete an agent |
| GET | `/api/agents/{type}/{name}/versions` | List agent version history |
| POST | `/api/agents/{type}/{name}/rollback` | Rollback to a previous version |
| POST | `/api/agents/{type}/{name}/clone` | Clone an agent |
| POST | `/api/agents/{type}/{name}/retire` | Soft-delete an underperforming agent |
| GET | `/api/agents/{type}/{name}/export` | Export a single agent |

### Teams (5)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/teams` | List all teams |
| POST | `/api/teams` | Create a team |
| GET | `/api/teams/presets` | List team presets |
| POST | `/api/teams/presets/{name}/apply` | Apply a team preset |
| DELETE | `/api/teams/{name}` | Delete a team |

### Static (1)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/{filename}` | Serve static files |

---

## Agent Tools

8 core tools available to sub-agents, mapped from `tools:` fields in plan.md.

| Tool | Description | Approval |
|---|---|---|
| `web_search` | Search the web via DuckDuckGo, returns top 5 results | Auto |
| `browse_url` | Load a web page via headless Chromium, returns content as markdown | Auto |
| `rag_search` | Search the NORT knowledge base for relevant past context | Auto |
| `rag_store` | Store text in the knowledge base with tags for future retrieval | Auto |
| `download_artifact` | Download content from a URL and store in knowledge base | Auto |
| `read_file` | Read a file relative to project root (or list directory contents) | Auto |
| `write_file` | Write content to a file in the artifacts directory | Auto |
| `execute_code` | Execute Python code in a sandboxed subprocess (30s timeout) | **Human approval required** |

**Aliases**: `search` -> `web_search`, `browse` -> `browse_url`, `analyze_data` -> `execute_code`, `design_ui` -> `write_file`, `reason` -> `rag_search`

**Approval system**: Tools in `APPROVAL_REQUIRED` (currently `execute_code`) block until a human clicks approve/reject in the dashboard. An approval banner appears in the UI with the tool name, arguments, agent, and task context.

**Content scanning**: Agent-written artifacts are scanned for secrets, credentials, and known malicious patterns before inclusion in output.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | -- | API key for your LLM provider (required) |
| `OPENAI_BASE_URL` | -- | Base URL for your LLM provider (required) |
| `NORT_PORT` | `8000` | Dashboard server port |
| `NORT_SERVER` | `http://localhost:8000` | Bridge target URL |
| `NORT_SECRET` | -- | Shared secret for bridge-to-server auth |
| `NORT_SANDBOX_MODE` | `subprocess` | Code execution sandbox mode |

`QUARM_PORT`, `QUARM_SERVER`, and `QUARM_SECRET` are accepted as aliases.

### config.json

Server-side configuration stored in `config.json` alongside `serve.py`. Manages:

- Global tolerance threshold
- Per-agent tolerance overrides
- `skip_specialist_on_high_score` toggle
- Review analytics preferences

### Agent Registry

Persistent agent definitions stored in `agents/registry.json`. Tracks:

- Agent descriptions and tool allowlists
- Performance history (scores, run counts)
- Version history with rollback support
- Earned tolerance bonuses

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

84 tests across 8 test files:

| File | Tests | Coverage |
|---|---|---|
| `test_plan_parser.py` | 9 | Plan parsing, field extraction, edge cases |
| `test_routing.py` | 10 | Graph routing, conditional edges, state transitions |
| `test_status_bridge.py` | 22 | Event log, roster registry, session isolation |
| `test_tolerance.py` | 5 | Tolerance precedence, earned bonus, presets |
| `test_error_recovery.py` | 21 | Checkpoint save/resume, failure handling |
| `test_validation_wiring.py` | 10 | Plan validator integration, schema enforcement |
| `test_review_integration.py` | 1 | End-to-end review pipeline |
| `test_smoke.py` | 6 | Playwright headless browser tests (page load, canvas, FPS, WebSocket, shortcuts, health) |

Additional test file: `test_specialization.py` (17 tests for agent specialization learning).

---

## Project Structure

```
.
|-- orchestrator.py            # Core LangGraph state machine
|-- serve.py                   # FastAPI server (57 endpoints)
|-- generate_plan.py           # Plan generator
|-- status_bridge.py           # Orchestrator-to-dashboard bridge
|-- tools.py                   # Tool registry (8 tools + aliases)
|-- tools_web.py               # Web tool backends
|-- validate_plan.py           # Plan schema validator
|-- model_config.py            # LLM model selection
|-- agent_registry.py          # Persistent agent definitions
|-- tracking.py                # Cost and score analytics
|-- content_scanner.py         # Artifact security scanner
|-- checkpoint.py              # Run state checkpointing
|-- rag.py                     # RAG knowledge base
|-- specialization.py          # Agent learning system
|-- plan.example.md            # Reference plan
|-- .env                       # API keys (not committed)
|-- config.json                # Server configuration
|-- agents/
|   +-- registry.json          # Agent definitions + history
|-- artifacts/                 # Per-plan task artifacts
|-- templates/
|   |-- base.html              # City view shell
|   |-- flow.html              # Flow view shell
|   |-- styles/
|   |   +-- base.css           # Glass morphism + holographic palette
|   |-- components/
|   |   |-- top_bar.html       # Header bar
|   |   |-- control_bar.html   # Bottom control bar
|   |   |-- event_log.html     # Chat panel
|   |   |-- approval_banner.html
|   |   +-- panels/            # 20 panel HTML partials
|   +-- scripts/
|       |-- colors.js          # Palette definitions
|       |-- constants.js       # Animation config
|       |-- nodes.js           # Node model + config object
|       |-- force.js           # Force simulation
|       |-- bloom.js           # Post-processing bloom
|       |-- camera.js          # Pan/zoom controls
|       |-- render.js          # 60fps render loop
|       |-- render_cache.js    # Offscreen canvas caching
|       |-- websocket.js       # WebSocket + applyStatus()
|       |-- api.js             # API client functions
|       |-- panels.js          # Panel toggle logic
|       |-- audio.js           # Sound design system
|       |-- dag.js             # DAG panel rendering
|       |-- init.js            # Boot + keyboard shortcuts
|       |-- flow_background.js # Flow view background
|       |-- flow_init.js       # Flow view boot
|       |-- flow_render.js     # Flow view render loop
|       +-- draw_*.js          # 17 canvas rendering modules
+-- tests/
    |-- conftest.py
    +-- test_*.py              # 8 test files (84 tests)
```

---

## results.json

```json
{
  "objective": "Build a web dashboard for AWS cost monitoring",
  "quality_log": [
    {
      "id": "TASK-001",
      "title": "Design auth architecture",
      "agent": "security_architect",
      "status": "done",
      "revision_count": 1,
      "scores": {
        "manager": 8,
        "security_engineer": 7
      }
    }
  ],
  "task_results": {
    "TASK-001": "... full task output ...",
    "TASK-002": "..."
  },
  "artifacts": {
    "TASK-001": {
      "files": ["auth_architecture.md", "iam_policy.json"],
      "output_dir": "artifacts/plan_abc123/TASK-001"
    }
  },
  "validation": {
    "TASK-002": {
      "checked": 3,
      "passed": 3,
      "errors": []
    }
  },
  "summary": "Final master report..."
}
```

`revision_count` tells you which tasks needed rework and how many cycles -- useful for evaluating sub-agent quality or prompt quality over time. `artifacts` maps each task to its output files and directory. `validation` records post-assembly linter/syntax check results.

---

## Security

### Content Scanner

Agent-written artifacts are scanned by `content_scanner.py` for:
- API keys and secrets (pattern matching against known formats)
- Credentials and passwords
- Known malicious code patterns
- Environment variable leaks

### Path Traversal Prevention

Both `read_file` and `write_file` tools enforce path boundaries. File operations are restricted to the project root and artifacts directory. Directory traversal attempts (`../`) are blocked.

### Tool Approval

The `execute_code` tool requires explicit human approval before execution. The approval request appears as a banner in the dashboard UI showing the code to be executed, the requesting agent, and the task context. Approvals time out after 5 minutes.

### Environment Isolation

Sensitive environment variables matching patterns like `API_KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL` are scrubbed from the subprocess environment when executing agent code.
