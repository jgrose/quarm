# Plan: Dynamic Agent Registry & Adaptive Role Assignment

## Context

Currently, agents (sub-agents, managers, reviewers) are defined only inside plan markdown files — they're ephemeral, recreated from scratch each time. There's no way to reuse a well-tuned agent definition across plans, and the plan generator doesn't know about agents that have worked well before.

We need:
1. **Persistent agent registry** — store agent definitions as JSON, reusable across plans
2. **Dynamic agent creation** — plan generator selects/creates agents based on what the project needs
3. **Adaptive enhancement** — agent definitions improve over time based on performance data
4. **CRUD API** — create, read, update, delete agents via REST + UI

---

## Agent Registry Design

### Storage: `agents/registry.json`

```json
{
  "sub_agents": {
    "python_backend_dev": {
      "name": "python_backend_dev",
      "title": "Senior Python Backend Developer",
      "description": "Expert in Python, FastAPI, SQLAlchemy, async patterns. Produces production-quality code with tests.",
      "tools": ["write_file", "execute_code", "read_file", "web_search"],
      "model": "",
      "tags": ["python", "backend", "api", "database"],
      "created_at": "2026-04-01T...",
      "updated_at": "2026-04-01T...",
      "runs": 0,
      "avg_score": 0,
      "total_revisions": 0,
      "builtin": false
    }
  },
  "managers": {
    "tech_lead": {
      "name": "tech_lead",
      "title": "Technical Lead",
      "description": "Reviews code quality, architecture decisions, and production readiness.",
      "expertise_blend": ["architecture", "code_quality", "testing", "performance"],
      "oversees": [],
      "model": "",
      "tags": ["technical", "code", "architecture"],
      "created_at": "...",
      "runs": 0,
      "avg_score": 0,
      "builtin": false
    }
  },
  "reviewers": {
    "security_engineer": {
      "name": "security_engineer",
      "title": "Senior Security Engineer",
      "description": "...",
      "focus_areas": ["OWASP Top 10", ...],
      "applies_to": ["code", "api", "auth", ...],
      "model": "",
      "tags": ["security"],
      "runs": 0,
      "avg_score": 0,
      "builtin": true
    }
  }
}
```

Key additions over current dataclasses:
- **`tags`** — searchable keywords for plan generator to find relevant agents
- **`runs`** — how many times this agent has been used
- **`avg_score`** — average review score (tracks quality over time)
- **`total_revisions`** — how many revisions triggered (lower = better)
- **`builtin`** — whether it's a system-provided agent (can't delete, can customize)

### Seed Data

On first run, seed the registry with:
- The 6 builtin reviewers (security_engineer, ux_designer, user_tester, creative_director, devils_advocate, performance_engineer) marked `builtin: true`
- A few common sub-agents: `general_developer`, `frontend_developer`, `backend_developer`, `technical_writer`
- A few common managers: `tech_lead`, `project_manager`

---

## Changes by File

### 1. NEW: `agents/registry.json` — persistent agent storage

JSON file with the structure above. Created on first access if missing (seeded with builtins).

### 2. NEW: `agent_registry.py` — registry CRUD module

```python
REGISTRY_FILE = Path("agents/registry.json")

def load_registry() -> dict
def save_registry(data: dict)
def seed_registry()  # creates initial builtins + common agents

# CRUD
def list_agents(agent_type=None) -> list[dict]
def get_agent(agent_type, name) -> dict | None
def create_agent(agent_type, spec: dict) -> dict
def update_agent(agent_type, name, updates: dict) -> dict
def delete_agent(agent_type, name) -> bool  # blocks builtin deletion

# Performance tracking
def record_agent_performance(agent_type, name, score, revisions)
def get_top_agents(agent_type, tags=None, limit=5) -> list[dict]

# Plan generation helper
def suggest_agents_for_description(description: str) -> dict
  # Returns recommended sub_agents, managers, reviewers based on tags matching
```

### 3. MODIFY: `generate_plan.py` — use registry for agent selection

Update the system prompt to include available agents from the registry:

```python
from agent_registry import list_agents, get_top_agents

# Build available agent catalog for the LLM
sub_agents = list_agents("sub_agents")
managers = list_agents("managers")
reviewers = list_agents("reviewers")

# Add to system prompt:
AGENT_CATALOG = f"""
## Available Agents (prefer reusing these over creating new ones)

### Sub-Agents:
{format_agent_list(sub_agents)}

### Managers:
{format_agent_list(managers)}

### Reviewers:
{format_agent_list(reviewers)}

You may create new agent definitions if none of the above fit.
When creating new agents, use descriptive names and comprehensive descriptions.
"""
```

The LLM sees the catalog and picks from existing agents or creates new ones. This is appended to the existing SYSTEM_PROMPT.

### 4. MODIFY: `orchestrator.py` — save new agents + track performance

**After `parse_plan()`:** Any agent defined in the plan markdown that isn't in the registry gets auto-saved:

```python
from agent_registry import get_agent, create_agent, record_agent_performance

# In parse_plan(), after parsing:
for agent in sub_agents:
    if not get_agent("sub_agents", agent.name):
        create_agent("sub_agents", agent.__dict__)

for mgr in managers:
    if not get_agent("managers", mgr.name):
        create_agent("managers", mgr.__dict__)
```

**After task completion:** Record performance metrics:

```python
# In manager_review_node, after verdict:
record_agent_performance("sub_agents", task["agent"], score, task["revision_count"])

# In specialist_review_node, after each reviewer verdict:
record_agent_performance("reviewers", reviewer["name"], score, 0)
```

### 5. MODIFY: `serve.py` — REST API for agent CRUD

New endpoints:

```
GET    /api/agents                    — list all agents (optional ?type=sub_agents)
GET    /api/agents/{type}/{name}      — get single agent
POST   /api/agents/{type}             — create new agent
PUT    /api/agents/{type}/{name}      — update agent
DELETE /api/agents/{type}/{name}      — delete agent (blocks builtins)
GET    /api/agents/suggest?desc=...   — suggest agents for a project description
```

### 6. MODIFY: `templates/` — Agent management UI panel

New panel accessible from the top bar (new "AGENTS" button):

- List all agents grouped by type (sub-agents, managers, reviewers)
- Each shows: name, title, tags, runs count, avg score
- Click to edit (name, description, tools, tags, etc.)
- "Create Agent" button with form
- Performance sparkline or badge (avg score, run count)
- Can't delete builtins but can customize their description

---

## Execution Plan (3 parallel subagents)

| Agent | Files | Scope |
|-------|-------|-------|
| **Agent 1** | NEW `agent_registry.py`, NEW `agents/registry.json` | Registry module + seed data |
| **Agent 2** | `generate_plan.py`, `orchestrator.py` | Plan gen integration + performance tracking |
| **Agent 3** | `serve.py`, NEW `templates/components/panels/agents.html`, `templates/scripts/panels.js` | REST API + UI panel |

After agents complete, I wire the UI button into `base.html` and add keyboard shortcut "A" in `init.js`.

---

## Performance Enhancement Loop

Over time, the system self-improves:

1. **Plan generates** → LLM sees agent catalog with scores → picks best agents
2. **Tasks run** → scores and revision counts recorded
3. **Registry updates** → avg_score and runs increment
4. **Next plan** → LLM sees updated scores → prefers higher-scoring agents
5. **Low-scoring agents** → LLM avoids them or the user can edit/delete them

Agents with `runs > 5` and `avg_score > 7` naturally become the preferred picks because the catalog shows their track record.

---

## Verification

1. **Fresh start**: `agents/registry.json` auto-created with 6 builtin reviewers + common agents
2. **Generate a plan**: system prompt shows agent catalog, LLM picks from existing agents
3. **Run plan**: new agents auto-saved to registry, scores tracked
4. **API**: `GET /api/agents` returns full catalog, CRUD works
5. **UI**: Press "A" or click AGENTS button, see all agents with stats, edit one
6. **Second plan**: LLM sees previously created agents with scores, reuses good ones
7. **Delete**: can delete custom agents, blocked for builtins
