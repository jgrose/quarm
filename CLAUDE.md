# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never include Co-Authored-By, "Claude", or any AI attribution in commit messages or anywhere in the codebase.**

## Tools & MCP Servers

Agents have access to tools during task execution. Tools are configured in `config.json`.

### Default Tools

All agents automatically receive default tools unless they opt out. Configure in `config.json`:

```json
"default_tools": ["web_search", "browse_url", "read_file"]
```

- Agents with no `- tools:` line in the plan get the defaults
- Agents with explicit `- tools: [execute_code]` get defaults PLUS their listed tools
- Use `- tools: none` in a plan to disable all tools (including defaults) for that agent

### Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search, returns top 5 results |
| `browse_url` | Headless Chromium (Playwright), returns page as markdown |
| `rag_search` | Search NORT knowledge base from past projects |
| `rag_store` | Ingest text into knowledge base |
| `download_artifact` | Download URL content, save to artifacts |
| `read_file` | Read project files |
| `write_file` | Write to artifacts directory |
| `execute_code` | Run Python in sandboxed subprocess (requires approval) |

### MCP Server Integration

Agents can use tools from external MCP (Model Context Protocol) servers. Add servers to `config.json`:

```json
"mcp_servers": {
  "brave_search": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@anthropic/brave-search-mcp"],
    "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
  },
  "filesystem": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@anthropic/filesystem-mcp", "/tmp/sandbox"]
  },
  "remote_api": {
    "type": "sse",
    "url": "http://localhost:3001/sse"
  }
}
```

**Server types:**

- `stdio` — Launches a local subprocess. Set `command`, `args`, and optionally `env`. Environment variables use `${VAR}` syntax to pull from your shell environment.
- `sse` — Connects to a remote MCP server over HTTP/SSE. Set `url`.

**How it works:**

1. On startup, the orchestrator connects to each configured MCP server
2. Tools are auto-discovered via `list_tools()` and registered as `server_name.tool_name`
3. Agents can reference them in plans: `- tools: [brave_search.brave_web_search]`
4. MCP tools can also be added to `default_tools` for automatic availability
5. Connections are lazy (established on first use) and reconnect on failure

**Setup steps:**

```bash
# 1. Install the mcp package (already in .venv)
pip install mcp

# 2. Install any MCP servers you want (example: Brave Search)
npm install -g @anthropic/brave-search-mcp

# 3. Set required API keys in your environment or .env
export BRAVE_API_KEY=your-key-here

# 4. Add the server to config.json under "mcp_servers" (see examples above)

# 5. Run the orchestrator — MCP tools are discovered automatically
python orchestrator.py plan.md
```

**Using MCP tools in a plan:**

```markdown
### AGENT: researcher
- description: Web researcher who finds and analyzes sources
- tools: [brave_search.brave_web_search, browse_url]
```

Or add to defaults so all agents get them:

```json
"default_tools": ["web_search", "browse_url", "read_file", "brave_search.brave_web_search"]
```

## What This Is

NORT is a 4-layer multi-agent orchestrator built on LangGraph. It takes a structured plan (markdown), dispatches tasks to specialist sub-agents, and runs each result through two quality gates: a domain manager review and a specialist reviewer panel (security engineer, UX designer, user tester). Results and a final executive report are written to `results.json`.

## Commands

```bash
# Install dependencies
pip install langgraph langchain langchain-openai python-dotenv fastapi uvicorn python-multipart jinja2

# Generate a plan from a project description
python generate_plan.py "Build a web dashboard for AWS cost monitoring"

# Validate a plan before running
python validate_plan.py plan.md

# Run the orchestrator against a plan
python orchestrator.py plan.md

# Start the live dashboard server (run before orchestrator for real-time UI)
python serve.py
# Then open http://localhost:8000/ (city view) or http://localhost:8000/flow (flow view)

# Run all tests (no LLM or network required — conftest.py stubs all external deps)
pytest tests/ -v

# Run a single test file
pytest tests/test_plan_parser.py -v

# Run a single test by name
pytest tests/test_routing.py -v -k "test_manager_review_pass"

# Smoke tests (require serve.py running + playwright installed)
pip install playwright pytest-playwright && python -m playwright install chromium
pytest tests/test_smoke.py -v
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

**status_bridge.py** — Fire-and-forget bridge between orchestrator and dashboard. Maintains per-session in-memory event logs, roster registry, transcript log, and file attention tracker. Pushes serialized state to `serve.py` via background `threading.Thread` POSTs. Uses `requests` if available, falls back to `urllib`. Thread-local storage for session IDs enables concurrent plan runs.

**serve.py** — FastAPI WebSocket server (57 API endpoints) with Jinja2 template rendering. Receives POST `/update` from the bridge and broadcasts to all connected WebSocket clients. Supports multi-session tracking keyed by session_id. Replays all active session states to new connections.

**tools.py** — Tool registry mapping plan.md tool names to LangChain tool functions. 8 core tools with aliases (e.g., `search` → `web_search`, `browse` → `browse_url`). Hybrid approval system: most tools auto-execute, `execute_code` blocks until human approval via dashboard.

**agent_registry.py** — Persistent agent definitions in `agents/registry.json`. Uses atomic writes (tmp file + fsync + `os.replace`) with `.json.bak` backup. Tracks performance history, version history with rollback, and earned tolerance bonuses.

**checkpoint.py** — Task-level state persistence for crash recovery. Saves `OrchestratorState` to `plans/{plan_id}_checkpoint.json` at each dispatch boundary. Uses atomic write for crash safety. On resume, in-flight tasks reset to pending.

**content_scanner.py** — Scans agent-written artifacts for secrets, credentials, and malicious patterns before inclusion in output.

## Dashboard

Two views share the same WebSocket data:

- **City view** (`/`) — Isometric Sim City SNES-style pixel art. Agent programs walk between 12 buildings, enter through animated doors. Day/night cycle, weather, building upgrades.
- **Flow view** (`/flow`) — Holographic node graph with force-directed layout against a void background with depth particles and pulsing hex grid.

The dashboard is composed from `templates/` via Jinja2 `{% include %}` directives, served as a single HTML response per view. 20 panel overlays in `templates/components/panels/`. Canvas rendering split across 17 `draw_*.js` modules.

Node hierarchy: NEXUS (master, 36px) → SENTINEL (manager, 28px) → DRONE (worker, 22px) → PROBE (reviewer, 18px) → SHARD (parallel sub-agent, 16px).

## Testing Patterns

**Critical: module stubbing in conftest.py.** The orchestrator imports `status_bridge`, `model_config`, `tracking`, `tools`, `checkpoint`, `agent_registry`, and `rag` at module level — all of which have import-time side effects (DB init, network calls). `tests/conftest.py` replaces these with `ModuleType` stubs via `sys.modules.setdefault()` BEFORE `import orchestrator`. When adding new test files that import `orchestrator`, they must go through `conftest.py` (pytest auto-loads it) or replicate the stubbing.

Key test fixtures: `parsed_simple_plan`, `parsed_complex_plan` (parse fixture plans), `mock_llm_pass`/`mock_llm_fail` (patch `orchestrator.llm`), `make_base_state()` (builds minimal `OrchestratorState` dict). Test fixtures live in `tests/fixtures/`.

Smoke tests (`test_smoke.py`) use Playwright and require `serve.py` to be running. They auto-start the server if not reachable. Marked with `@pytest.mark.smoke`.

## Plan Schema

Plans are markdown files with sections: `## Objective`, `## Sub-Agents` (### AGENT:), `## Managers` (### MANAGER:), `## Tasks` (### TASK-NNN:), and optionally `## Custom Reviewers` (### REVIEWER:). See `plan.example.md` for a complete reference. The parser in `orchestrator.py` uses regex to extract fields like `- description:`, `- tools:`, `- reviewers: [...]`, `- depends_on: [...]`.

## Key Configuration

- `MAX_REVISIONS = 3` in `orchestrator.py` — max revision cycles per task across both review gates
- `NORT_PORT` env var — dashboard server port (default 8000). `QUARM_PORT` also accepted.
- `NORT_SERVER` env var — bridge target URL (default `http://localhost:8000`). `QUARM_SERVER` also accepted.
- `NORT_SECRET` env var — optional shared secret for bridge-to-server auth. `QUARM_SECRET` also accepted.
- `.env` file — must contain `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- `config.json` — server-side config: allowed models, tolerance settings, webhook URL, active preset
- `agents/registry.json` — persistent agent definitions, performance history, version history
- LLM model is auto-selected at runtime; optional `- model:` field in plan.md overrides per agent/task

## Review Flow

Tasks follow: `pending → in_progress → in_manager_review → in_specialist_review → done`. On FAIL/FLAG from either gate, the task loops back to `revision` status and the sub-agent re-executes with consolidated feedback. After `MAX_REVISIONS`, the result is force-accepted. Custom reviewers defined in `plan.md` override builtins with the same name.

### Tolerance Precedence (highest wins)

1. `config.json` per-agent override
2. Task-level `- tolerance:` field in plan.md
3. Plan per-agent override
4. `config.json` global setting (`default_tolerance`)
5. `DEFAULT_TOLERANCE` constant (6)

Agents with `avg_score > 8` over 5+ tasks earn a `+1` tolerance bonus. Tasks scoring 9+ at manager review can skip the specialist panel (`skip_specialist_on_high_score` config toggle).
