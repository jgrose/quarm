# NORT Tool Orchestration System — Deep Dive Report

## Executive Summary

NORT implements a sophisticated multi-layer tool binding and execution system for LangGraph sub-agents:
- **Tools defined**: 8 primary tools via LangChain `@tool` decorators in `tools.py`
- **Registry pattern**: `TOOL_REGISTRY` dict maps string names to callable tool objects
- **Binding mechanism**: `ChatOpenAI.bind_tools(agent_tools)` at execution time
- **Approval system**: Hybrid approval (auto-execute read-only, human-gate dangerous operations)
- **Plan-driven configuration**: Tools specified per-agent in plan.md schema

---

## 1. TOOLS.PY — TOOL REGISTRY & EXECUTION ENGINE

### 1.1 Tool Definitions (Lines 150-401)

Eight tools decorated with `@tool` (LangChain decorator):

| Line | Tool Name | Signature | Type | Description |
|------|-----------|-----------|------|-------------|
| 150-154 | `web_search` | `(query: str) -> str` | Read-only | DuckDuckGo search (delegates to `tools_web.web_search`) |
| 157-169 | `browse_url` | `(url: str) -> str` | Read-only | Headless Chromium page load + markdown conversion; blocks file://, localhost, 169.254.x.x |
| 172-182 | `rag_search` | `(query: str) -> str` | Read-only | Knowledge base search; returns top-5 chunks with scores and sources |
| 185-201 | `rag_store` | `(text: str, tags: str = "") -> str` | Write | Ingest text into knowledge base; truncates to 100k chars; supports tagging |
| 204-227 | `download_artifact` | `(url: str) -> str` | Write | Download URL content, save to artifacts dir, auto-ingest to RAG |
| 230-247 | `read_file` | `(path: str) -> str` | Read-only | Read file from project root (20k char limit) or list directory; path-traversal protected |
| 250-264 | `write_file` | `(path: str, content: str) -> str` | Write | Write to artifacts directory only; prevents escape via path-traversal validation |
| 367-381 | `execute_code` | `(code: str) -> str` | Write (dangerous) | Python code execution with 30s timeout; **requires human approval** |

All tools call `set_tool_context()` to populate thread-local metadata (plan_id, task_id, agent).

### 1.2 Tool Registry (Lines 386-401)

```python
TOOL_REGISTRY = {
    "web_search": web_search,
    "browse_url": browse_url,
    "rag_search": rag_search,
    "rag_store": rag_store,
    "download_artifact": download_artifact,
    "read_file": read_file,
    "write_file": write_file,
    "execute_code": execute_code,
    # Aliases for plan.md compatibility
    "search": web_search,
    "browse": browse_url,
    "analyze_data": execute_code,
    "design_ui": write_file,
    "reason": rag_search,
}
```

**Key**: String-to-callable mapping. Supports 8 primary + 5 aliases = 13 total resolvable names.

### 1.3 Approval System (Lines 29-110)

**APPROVAL_REQUIRED set** (line 32):
```python
APPROVAL_REQUIRED = {"execute_code"}
```

Only `execute_code` requires human approval. All other tools auto-execute.

**Core Functions**:

| Line | Function | Signature | Purpose |
|------|----------|-----------|---------|
| 40-75 | `request_approval()` | `(tool_call_id, tool_name, args, agent, task_id) -> bool` | Block until human approves/rejects via 5-min timeout; broadcasts to UI via HTTP POST to `/update` |
| 78-101 | `resolve_approval()` | `(tool_call_id, approved) -> bool` | Called from serve.py when human clicks approve/reject; signals pending event |
| 104-109 | `get_pending_approvals()` | `() -> list[dict]` | Returns all pending approval requests for dashboard |

**Threading**: Uses `threading.Event()` and thread-safe dicts `_pending_approvals`, `_approval_results`, `_approval_details`.

### 1.4 Tool Context (Lines 112-136)

Thread-local context set before task execution:

```python
_tool_context = threading.local()

def set_tool_context(plan_id: str = "", task_id: str = "", agent: str = ""):
    """Set context for the current task's tool execution."""
    _tool_context.plan_id = plan_id
    _tool_context.task_id = task_id
    _tool_context.agent = agent

def _ctx():
    return {
        "plan_id": getattr(_tool_context, "plan_id", ""),
        "task_id": getattr(_tool_context, "task_id", ""),
        "agent": getattr(_tool_context, "agent", ""),
    }
```

Used by tools to:
- Build artifact paths: `artifacts/[plan_id]/[task_id]/`
- Log file access: `record_file_touch(path, operation, agent)`
- Tag RAG ingestion with origin metadata

### 1.5 get_tools() Function (Lines 404-418)

**Signature**:
```python
def get_tools(tool_names: list[str], allowed_tools: list[str] = None) -> list:
    """Get LangChain tool objects for the given tool name strings.
    If allowed_tools is non-empty, only include tools in that list."""
```

**Logic**:
1. Accepts list of tool name strings
2. Optional allowlist enforcement (security)
3. Deduplicates via `seen` set
4. Looks up each name in `TOOL_REGISTRY`
5. Returns list of actual LangChain tool objects

**Allowlist check** (lines 409-417):
```python
allowed_set = set(t.strip().lower() for t in allowed_tools) if allowed_tools else None
for name in tool_names:
    name = name.strip().lower()
    if name in TOOL_REGISTRY and name not in seen:
        if allowed_set and name not in allowed_set:
            log_event(f"  [SECURITY] Tool '{name}' not in allowed_tools — skipped")
            continue
        tools.append(TOOL_REGISTRY[name])
        seen.add(name)
```

**Returns**: List of callable LangChain tool objects ready for `bind_tools()`.

### 1.6 execute_tool_call() Function (Lines 423-457)

**Signature**:
```python
def execute_tool_call(tool_call: dict, tools: list, auto_approve_all: bool = False,
                      allowed_tools: list[str] = None) -> str:
    """Execute a single tool call, with approval check for dangerous tools."""
```

**Parameters**:
- `tool_call`: Dict with `{"name", "args", "id"}` from LLM tool_calls
- `tools`: List of LangChain tool objects (from `get_tools()`)
- `auto_approve_all`: If True, skip approval even for `execute_code`
- `allowed_tools`: Optional allowlist for defense-in-depth

**Execution flow** (lines 426-457):
```
1. Extract name, args, tc_id from tool_call dict
2. Find matching tool object in tools list by name
3. Defense-in-depth: check allowlist again
4. If name in APPROVAL_REQUIRED and not auto_approve_all:
   - Call request_approval(tc_id, name, args, agent, task_id)
   - Block until human responds (5 min timeout)
   - Return error message if rejected
5. Call tool_fn.invoke(args)
6. Return str(result) or error message
```

**Critical detail** (line 661 in orchestrator.py):
```python
result = execute_tool_call(tc, agent_tools, auto_approve_all=True)
```
During sub-agent task execution, **all tools auto-approve** (even `execute_code`). Approval system is currently disabled for agents.

---

## 2. ORCHESTRATOR.PY — TOOL BINDING & EXECUTION

### 2.1 Data Class: SubAgentSpec (Lines 167-172)

```python
@dataclass
class SubAgentSpec:
    name: str
    description: str
    tools: list[str] = field(default_factory=list)  # Tool names for this agent
    model: str = ""
```

**Key field**: `tools: list[str]` stores comma-separated tool names parsed from plan.md.

### 2.2 Plan Parser: Tool Extraction (Lines 250-356)

**Location**: `parse_plan(path: str)` function

**Tool parsing** (line 271):
```python
tools=rxl(r"- tools:\s*(.+)", b.group(2)),
```

**Helper function** (lines 263-265):
```python
def rxl(pat, raw):
    m = re.search(pat, raw)
    return [x.strip() for x in m.group(1).split(",")] if m and m.group(1).strip() else []
```

**Regex pattern**:
- Matches: `- tools: web_search, read_file, execute_code`
- Splits on comma, strips whitespace
- Returns: `["web_search", "read_file", "execute_code"]`

**Example agent from plan.md**:
```markdown
### AGENT: backend_engineer
- description: Build RESTful APIs with FastAPI and MongoDB
- tools: web_search, read_file, write_file, execute_code
- model: bedrock-claude-opus-4-6
```

### 2.3 Tool Binding in Execution Loop (Lines 586-687)

**Function**: `_execute_single_task(tid, tasks, results, sub_agents_list)`

**Tool setup** (lines 638-649):

```python
# Line 638: Set thread-local context for tool execution
set_tool_context(plan_id=_plan_id, task_id=tid, agent=task["agent"])

# Line 639: Get tool objects from string names in agent spec
agent_tools = get_tools(agent.get("tools", []))

# Line 649: Bind tools to LLM instance
if agent_tools:
    llm_with_tools = llm(model).bind_tools(agent_tools)
```

**Key points**:
1. `agent.get("tools", [])` extracts list of tool name strings from SubAgentSpec
2. `get_tools()` resolves names to LangChain tool objects
3. `ChatOpenAI.bind_tools()` (from langchain_openai) configures LLM for tool use

**Tool execution loop** (lines 651-667):

```python
max_iterations = 5
for _iter in range(max_iterations):
    resp = _invoke_with_retry(llm_with_tools, messages)
    total_toks += extract_tokens(resp)
    if not resp.tool_calls:
        break
    messages.append(resp)
    for tc in resp.tool_calls:
        tc_name = tc["name"]
        tc_args = tc["args"]
        log_event(f"  [TOOL] {tc_name}({str(tc_args)[:80]})")
        
        # Execute tool call
        result = execute_tool_call(tc, agent_tools, auto_approve_all=True)
        result_str = str(result)[:4000]
        
        # Append ToolMessage to conversation for next LLM call
        messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
        
        # Log for audit trail
        tool_calls_log.append({"name": tc_name, "state": "complete",
                               "args_preview": str(tc_args)[:100],
                               "result_preview": result_str[:200]})
```

**Flow**:
1. Call LLM with `bind_tools()` applied
2. LLM generates `resp.tool_calls` (list of tool invocations)
3. For each tool call:
   - Execute via `execute_tool_call()`
   - Wrap result in `ToolMessage`
   - Append to messages for next LLM iteration
4. Max 5 iterations to prevent infinite loops
5. If LLM stops calling tools, break

### 2.4 LLM Helper Function (Lines 381-383)

```python
def llm(model: str = ""):
    m = model or DEFAULT_MODEL
    return ChatOpenAI(model=m, temperature=0.2)
```

Returns a `ChatOpenAI` instance ready for `.bind_tools()`.

### 2.5 Tool Call Logging (Lines 658-666)

Tool calls tracked in three ways:

1. **Event log** (line 660):
   ```python
   log_event(f"  [TOOL] {tc_name}({str(tc_args)[:80]})")
   ```

2. **Tool calls log** (lines 664-666):
   ```python
   tool_calls_log.append({"name": tc_name, "state": "complete",
                          "args_preview": str(tc_args)[:100],
                          "result_preview": result_str[:200]})
   ```
   Stored in task record: `tool_calls` field

3. **Transcript** (via status_bridge):
   - Tools auto-logged via `execute_tool_call()` → `log_event()`
   - Events pushed to UI via `write_status()`

---

## 3. GENERATE_PLAN.PY — TOOL SPECIFICATION SCHEMA

### 3.1 System Prompt: Tool Documentation (Lines 62-189)

**Location**: `SYSTEM_PROMPT` used in `generate_plan_streaming()` and `generate_plan()`

**Tool list in schema** (lines 79):
```markdown
- tools: [choose from: web_search, browse_url, rag_search, rag_store, download_artifact, read_file, write_file, execute_code]
```

**Agent definition block** (lines 77-81):
```markdown
### AGENT: [agent_name]
- description: [Specialist role and exact output format this agent produces]
- tools: [choose from: web_search, browse_url, rag_search, rag_store, download_artifact, read_file, write_file, execute_code]
- model: [optional — specific model ID to use for this agent, omit to auto-select]
```

**Validation rules** (lines 141-147):
- All names: lowercase_with_underscores
- Tool list must be CSV of valid tool names
- No custom tools allowed (must come from predefined set)

**Prompt guidance**: The system prompt documents each tool's purpose and hints at appropriate use cases. LLM is instructed to choose tools matching agent's role.

---

## 4. CONFIG.JSON — TOOL & APPROVAL CONFIGURATION

**Location**: `/home/localuser/projects/quarm/config.json`

```json
{
  "allowed_models": [...],
  "default_tolerance": 8,
  "tolerance_overrides": {},
  "webhook_url": "",
  "active_preset": "prototype"
}
```

**Current state**: No tool-level configuration in config.json.

**Potential for expansion**:
- `"allowed_tools": ["web_search", "read_file", "write_file"]` (per-agent allowlist)
- `"auto_approve_tools": ["web_search", "read_file"]` (override APPROVAL_REQUIRED)
- `"tool_timeout": 30` (per-tool execution timeout)
- `"tool_resource_limits": {"execute_code": {"memory_mb": 512, "cpu_seconds": 30}}`

---

## 5. STATUS_BRIDGE.PY — TOOL EXECUTION TRACKING

### 5.1 Tool Context Recording (Lines 121-134)

```python
def record_file_touch(path: str, operation: str, agent: str):
    """Record a file read/write for the file attention heatmap."""
    sid = _get_sid()
    with _state_lock:
        ft = _session_files.get(sid, {}) if sid else {}
        if path not in ft:
            ft[path] = {"reads": 0, "writes": 0, "agents": set()}
        if operation == "read":
            ft[path]["reads"] += 1
        elif operation == "write":
            ft[path]["writes"] += 1
        ft[path]["agents"].add(agent)
```

Called from:
- `read_file()` tool (line 243): `record_file_touch(path, "read", agent)`
- `write_file()` tool (line 260): `record_file_touch(path, "write", agent)`

**Purpose**: Build file access heatmap for UI (which agents touched which files).

### 5.2 Event Logging (Lines 180-187)

```python
def log_event(msg: str):
    sid = _get_sid()
    with _state_lock:
        if sid and sid in _session_logs:
            dq = _session_logs[sid]
            if len(dq) == dq.maxlen:
                _session_dropped_events[sid] = _session_dropped_events.get(sid, 0) + 1
            dq.append(msg)
```

All tool calls logged via `log_event()`, persisted in per-session deque (max 80 events).

---

## 6. LANGGRAPH INTEGRATION POINTS

### 6.1 OrchestratorState TypedDict (Lines 361-376)

```python
class OrchestratorState(TypedDict):
    messages:        Annotated[Sequence[BaseMessage], add_messages]
    objective:       str
    managers:        list[dict]
    sub_agents:      list[dict]      # Includes 'tools' field per agent
    reviewers:       list[dict]
    tasks:           list[dict]
    active_task_id:  Optional[str]
    active_task_ids: list[str]
    results:         dict[str, str]
    finished:        bool
    phase:           str
    tokens_used:     int
    last_verdict:    Optional[dict]
    synthesis_report: str
    coherence_report: dict
```

**Tool-relevant fields**:
- `sub_agents`: List of dicts with `tools` field
- `messages`: LangChain message list (accumulates tool calls and results)

### 6.2 Graph Nodes (Lines 527-1093)

Tool execution happens in: **`sub_agent_node()` → `_execute_single_task()`**

Other nodes (manager_review, specialist_review, synthesis) do NOT use tools — they only review/critique outputs.

---

## 7. DATA FLOW: PLAN → BINDING → EXECUTION

```
plan.md
  └─ Parse with rxl()
     └─ SubAgentSpec.tools = ["web_search", "read_file", ...]
        └─ register in orchestrator state
           └─ _execute_single_task() called
              └─ get_tools(agent["tools"]) → [LangChain tool objects]
                 └─ llm().bind_tools([...])
                    └─ LLM invoked, generates tool_calls
                       └─ for each tool_call:
                          └─ execute_tool_call(tc, agent_tools, auto_approve_all=True)
                             └─ tool_fn.invoke(args)
                                └─ ToolMessage(result) appended
                                   └─ next LLM iteration
```

---

## 8. SANDBOX EXECUTION (execute_code tool)

**Function**: `execute_code()` (lines 367-381)

**Three sandbox modes** (configurable via `NORT_SANDBOX_MODE` env var):

| Mode | Implementation | Isolation | Security |
|------|---|---|---|
| `none` | `_execute_none()` (lines 305-313) | None — bare subprocess | Trusted environment only |
| `subprocess` | `_execute_subprocess()` (lines 316-340) | Temp dir + env stripping + resource limits + optional network isolation via `unshare` | Good — memory/CPU limits, no network by default |
| `docker` | `_execute_docker()` (lines 343-364) | Docker container: no network, 256MB memory, 0.5 CPU, 50 PID limit | Best — full isolation |

**Resource limits** (lines 299-302):
```python
def _preexec_limits():
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
```

512 MB memory, 30 second CPU timeout.

**Environment sanitization** (lines 282-296):
```python
def _sanitized_env() -> dict[str, str]:
    safe_keys = {"PYTHONPATH", "PATH", "HOME", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE"}
    env = {}
    for k, v in os.environ.items():
        if k in safe_keys:
            env[k] = v
        elif not any(pat in k.upper() for pat in _SENSITIVE_ENV_PATTERNS):
            env[k] = v
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    _cached_sanitized_env = dict(env)
    return env
```

Strips: API keys, secrets, tokens, passwords, credentials, AWS, OpenAI, Anthropic vars.

---

## 9. KEY FUNCTION SIGNATURES

| Location | Function | Signature | Returns |
|----------|----------|-----------|---------|
| tools.py:404-418 | `get_tools()` | `(tool_names: list[str], allowed_tools: list[str] = None) -> list` | List of LangChain tool objects |
| tools.py:423-457 | `execute_tool_call()` | `(tool_call: dict, tools: list, auto_approve_all: bool = False, allowed_tools: list[str] = None) -> str` | String result of tool execution |
| tools.py:117-121 | `set_tool_context()` | `(plan_id: str = "", task_id: str = "", agent: str = "")` | None (sets thread-local) |
| orchestrator.py:586-687 | `_execute_single_task()` | `(tid, tasks, results, sub_agents_list)` | `(tid, draft, toks, tool_log, model)` |
| orchestrator.py:250-356 | `parse_plan()` | `(path: str)` | `(objective, managers, sub_agents, tasks, reviewers)` |

---

## 10. TOOL CALL LIFECYCLE

### 10.1 Full Sequence

```
1. Task Dispatch
   │
   ├─ load SubAgentSpec.tools = ["web_search", "read_file", ...]
   │
   ├─ set_tool_context(plan_id, task_id, agent_name)
   │
   ├─ agent_tools = get_tools(["web_search", "read_file", ...])
   │   └─ resolves to [web_search_fn, read_file_fn, ...]
   │
   ├─ llm_with_tools = llm(model).bind_tools(agent_tools)
   │
   ├─ LLM invoked with tool-aware system prompt
   │
   └─ Loop (max 5 iterations):
       │
       ├─ resp = llm_with_tools.invoke(messages)
       │
       ├─ IF resp.tool_calls:
       │   │
       │   └─ FOR each tool_call in resp.tool_calls:
       │       │
       │       ├─ extract: name, args, id
       │       │
       │       ├─ execute_tool_call(tool_call, agent_tools, auto_approve_all=True)
       │       │   │
       │       │   ├─ find tool by name in agent_tools
       │       │   ├─ check allowlist (if provided)
       │       │   ├─ check approval (APPROVAL_REQUIRED)
       │       │   ├─ call tool_fn.invoke(args)
       │       │   ├─ catch exceptions → error string
       │       │   └─ return result string
       │       │
       │       ├─ log to event bridge: log_event(f"[TOOL] {name}(...)")
       │       │
       │       ├─ wrap result in ToolMessage(content, tool_call_id)
       │       │
       │       ├─ append ToolMessage to messages list
       │       │
       │       └─ record in tool_calls_log for audit trail
       │
       ├─ ELSE (no tool_calls):
       │   └─ break (LLM done with tools)
       │
       └─ draft = resp.content (final text output)

2. Result Persistence
   ├─ tasks = upd(tasks, tid, status="in_manager_review", result=draft, tool_calls=tool_log)
   └─ state pushed to bridge → UI updates
```

### 10.2 Message Flow

```
SystemMessage("You are X agent...")
 + HumanMessage("Task: Y")
 + [optional: context from depends_on]
 + [optional: RAG knowledge]

↓ (invoke with bind_tools)

AIMessage (with tool_calls=[{name, args, id}, ...])

↓ (loop iteration 1)

ToolMessage(content="tool result", tool_call_id="...")

↓ (append, next iteration)

AIMessage (with tool_calls or final content)

↓ (if no tool_calls, break)

[final AIMessage.content becomes task result]
```

---

## 11. SECURITY ARCHITECTURE

### 11.1 Multi-Layer Defense

1. **Tool registry allowlist** (tools.py:404-418)
   - Only tools in `TOOL_REGISTRY` can be invoked
   - Agent specs must list valid tool names (enforced by plan validation)

2. **Per-agent allowlist** (optional, via `allowed_tools` param)
   - `get_tools()` can filter further
   - `execute_tool_call()` checks allowlist again (defense-in-depth)
   - Currently unused in orchestrator (all agents get all requested tools)

3. **Path traversal prevention** (tools.py:139-145)
   ```python
   def _path_is_within(target: Path, allowed_base: Path) -> bool:
       try:
           target.resolve().relative_to(allowed_base.resolve())
           return True
       except ValueError:
           return False
   ```
   - Used by `read_file()`, `write_file()`, `download_artifact()`
   - Ensures paths stay within project/artifacts directories

4. **Network blocking** (tools.py:162-167)
   - `browse_url()` blocks: file://, localhost, 127.0.0.1, 0.0.0.0, [::1], 169.254.x.x
   - Prevents SSRF and information disclosure

5. **Environment sanitization** (tools.py:282-296)
   - `execute_code()` strips sensitive env vars before subprocess exec
   - Prevents accidental credential leakage to untrusted code

6. **Approval gate** (tools.py:40-75)
   - `execute_code` requires human approval
   - 5-minute timeout, broadcast to UI
   - Can be auto-approved with `auto_approve_all=True` (currently always enabled for agents)

7. **Resource limits** (tools.py:299-302)
   - `execute_code`: 512 MB memory, 30 sec CPU timeout
   - Prevents DoS via infinite loops or memory bombs

8. **Sandbox isolation** (tools.py:305-364)
   - Three modes: none, subprocess (preferred), docker (best)
   - Network isolation via `unshare --net` (if available)
   - Docker: complete containerization with 256MB memory cap

### 11.2 Approval Status

**Current**: `auto_approve_all=True` in orchestrator.py:661
- Approval system is **implemented but disabled** for agent execution
- Can be re-enabled by changing one parameter

---

## 12. TOOL ADDITION ROADMAP FOR MCP INTEGRATION

### 12.1 Current Architecture

```
LangChain @tool decorator
  └─ TOOL_REGISTRY dict
     └─ get_tools(tool_names) → [LangChain tool objects]
        └─ llm.bind_tools([...])
```

### 12.2 MCP Integration Points

1. **Tool discovery**: Instead of hardcoded `TOOL_REGISTRY`, query MCP server for available tools
2. **Tool binding**: Create adapter layer:
   ```python
   class MCPToolAdapter:
       def __init__(self, mcp_tool_def):
           self.name = mcp_tool_def["name"]
           self.schema = mcp_tool_def["schema"]
       
       def invoke(self, args):
           # Call MCP server via stdio/SSE
           return mcp_client.call_tool(self.name, args)
   ```

3. **Registry expansion**:
   ```python
   TOOL_REGISTRY = {
       # Builtin tools
       "web_search": web_search,
       ...
       # MCP tools (loaded at runtime)
       "mcp_server_name/tool_name": MCPToolAdapter(...),
   }
   ```

4. **Plan schema update**: Allow `- tools: mcp_server/tool_name` syntax

5. **Config default tools**:
   ```json
   {
       "default_tools": ["web_search", "read_file", "mcp_server/advanced_search"],
       "mcp_servers": [
           {"name": "mcp_server", "command": "python -m mcp_server", "auto_start": true}
       ]
   }
   ```

---

## 13. CODE LOCATIONS SUMMARY

| Aspect | File | Lines |
|--------|------|-------|
| Tool definitions | tools.py | 150-401 |
| Tool registry | tools.py | 386-401 |
| Tool retrieval | tools.py | 404-418 |
| Tool execution | tools.py | 423-457 |
| Approval system | tools.py | 29-110 |
| Sandbox execution | tools.py | 305-364 |
| Resource limits | tools.py | 299-302 |
| Tool context | tools.py | 112-136 |
| Binding in agent | orchestrator.py | 639, 649 |
| Tool loop | orchestrator.py | 651-667 |
| Plan parsing | orchestrator.py | 250-356 |
| Tool name extraction | orchestrator.py | 271 (rxl regex) |
| SubAgentSpec | orchestrator.py | 167-172 |
| Tool logging | orchestrator.py | 658-666 |
| File touch tracking | status_bridge.py | 121-134 |
| Event logging | status_bridge.py | 180-187 |
| Plan schema | generate_plan.py | 62-189 |
| Tool list prompt | generate_plan.py | 79 |
| Config schema | config.json | all |

---

## 14. CALL SIGNATURES FOR IMPLEMENTATION

**To add a tool to NORT:**

```python
from langchain_core.tools import tool

@tool
def my_tool(required_arg: str, optional_arg: str = "default") -> str:
    """Clear docstring describing what the tool does and what it returns."""
    # Implementation
    return "result string"

# Add to TOOL_REGISTRY
TOOL_REGISTRY["my_tool"] = my_tool

# Add to generate_plan.py system prompt (line 79)
# "- tools: [choose from: ..., my_tool, ...]"

# Plan schema:
# ### AGENT: some_agent
# - tools: my_tool, web_search
```

**To bind tools for an agent:**

```python
agent_tools = get_tools(agent["tools"])
llm_with_tools = llm(model).bind_tools(agent_tools)
resp = llm_with_tools.invoke(messages)  # Auto-calls tools
```

**To execute a tool call:**

```python
result = execute_tool_call(
    {"name": "my_tool", "args": {"required_arg": "value"}, "id": "tc_123"},
    agent_tools,
    auto_approve_all=True
)
```

