# NORT Tool Orchestration — Quick Reference

## Core Data Flow

```
SubAgentSpec.tools: ["web_search", "read_file"]
  ↓
get_tools(["web_search", "read_file"])
  ↓ (lookup in TOOL_REGISTRY)
[web_search_fn, read_file_fn]  ← LangChain tool objects
  ↓
llm().bind_tools([...])  ← Configure LLM for tool use
  ↓
LLM generates: resp.tool_calls = [{name, args, id}, ...]
  ↓
FOR each tool_call:
  execute_tool_call(tool_call, agent_tools)
    ↓
    find matching tool by name
    ↓
    call tool_fn.invoke(args)
    ↓
    return result string
  ↓
  ToolMessage(content=result, tool_call_id=...)
  ↓ (append to messages, continue)
```

---

## Key Files & Locations

| File | Purpose | Key Functions | Lines |
|------|---------|---|---|
| **tools.py** | Tool definitions & registry | `get_tools()`, `execute_tool_call()`, `set_tool_context()` | 1-457 |
| **orchestrator.py** | Tool binding & execution | `_execute_single_task()`, tool loop | 639, 649, 651-667 |
| **generate_plan.py** | Plan schema | Tool documentation in SYSTEM_PROMPT | 79 |
| **config.json** | Configuration | (none currently for tools) | — |
| **status_bridge.py** | Logging | `log_event()`, `record_file_touch()` | 121-187 |

---

## Function Signatures (Copy-Paste Ready)

### get_tools()
```python
def get_tools(tool_names: list[str], allowed_tools: list[str] = None) -> list:
    """Get LangChain tool objects for the given tool name strings.
    If allowed_tools is non-empty, only include tools in that list."""
    tools = []
    seen = set()
    allowed_set = set(t.strip().lower() for t in allowed_tools) if allowed_tools else None
    for name in tool_names:
        name = name.strip().lower()
        if name in TOOL_REGISTRY and name not in seen:
            if allowed_set and name not in allowed_set:
                log_event(f"  [SECURITY] Tool '{name}' not in allowed_tools — skipped")
                continue
            tools.append(TOOL_REGISTRY[name])
            seen.add(name)
    return tools
```

### execute_tool_call()
```python
def execute_tool_call(tool_call: dict, tools: list, auto_approve_all: bool = False,
                      allowed_tools: list[str] = None) -> str:
    """Execute a single tool call, with approval check for dangerous tools."""
    name = tool_call["name"]
    args = tool_call["args"]
    tc_id = tool_call["id"]
    
    tool_fn = next((t for t in tools if t.name == name), None)
    if not tool_fn:
        return f"Unknown tool: {name}"
    
    if allowed_tools:
        allowed_set = set(t.strip().lower() for t in allowed_tools)
        if name.lower() not in allowed_set:
            log_event(f"  [SECURITY] Agent attempted disallowed tool '{name}' — blocked")
            return f"Tool '{name}' is not in this agent's allowed tools list."
    
    needs_approval = name in APPROVAL_REQUIRED and not auto_approve_all
    if needs_approval:
        ctx = _ctx()
        approved = request_approval(tc_id, name, args,
                                     agent=ctx.get("agent", ""),
                                     task_id=ctx.get("task_id", ""))
        if not approved:
            return f"Tool call rejected by human operator: {name}"
    
    try:
        result = tool_fn.invoke(args)
        return str(result)
    except Exception as e:
        return f"Tool error ({name}): {e}"
```

### set_tool_context()
```python
def set_tool_context(plan_id: str = "", task_id: str = "", agent: str = ""):
    """Set context for the current task's tool execution."""
    _tool_context.plan_id = plan_id
    _tool_context.task_id = task_id
    _tool_context.agent = agent
```

---

## TOOL_REGISTRY Contents

```python
TOOL_REGISTRY = {
    # Primary tools (8)
    "web_search": web_search,          # DuckDuckGo search
    "browse_url": browse_url,          # Chromium headless + markdown
    "rag_search": rag_search,          # Knowledge base search
    "rag_store": rag_store,            # Knowledge base ingest
    "download_artifact": download_artifact,  # Download + store
    "read_file": read_file,            # Read from project
    "write_file": write_file,          # Write to artifacts
    "execute_code": execute_code,      # Python sandbox (30s timeout, approval required)
    
    # Aliases (5)
    "search": web_search,
    "browse": browse_url,
    "analyze_data": execute_code,
    "design_ui": write_file,
    "reason": rag_search,
}
```

---

## Tool Call Loop in Orchestrator

```python
# From orchestrator.py lines 638-667

set_tool_context(plan_id=_plan_id, task_id=tid, agent=task["agent"])
agent_tools = get_tools(agent.get("tools", []))

if agent_tools:
    llm_with_tools = llm(model).bind_tools(agent_tools)
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
            result = execute_tool_call(tc, agent_tools, auto_approve_all=True)
            result_str = str(result)[:4000]
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
            tool_calls_log.append({"name": tc_name, "state": "complete",
                                   "args_preview": str(tc_args)[:100],
                                   "result_preview": result_str[:200]})
else:
    resp = _invoke_with_retry(llm(model), messages)
    total_toks = extract_tokens(resp)
    draft = resp.content
```

---

## Plan Schema: Tool Specification

```markdown
### AGENT: agent_name
- description: What this agent does and produces
- tools: web_search, read_file, write_file, execute_code
- model: bedrock-claude-opus-4-6
```

**Parsing** (orchestrator.py line 271):
```python
tools=rxl(r"- tools:\s*(.+)", b.group(2))

# rxl() helper (lines 263-265):
def rxl(pat, raw):
    m = re.search(pat, raw)
    return [x.strip() for x in m.group(1).split(",")] if m and m.group(1).strip() else []
```

---

## Tool Context & Metadata

Thread-local context stored in `_tool_context`:

```python
def _ctx():
    return {
        "plan_id": getattr(_tool_context, "plan_id", ""),
        "task_id": getattr(_tool_context, "task_id", ""),
        "agent": getattr(_tool_context, "agent", ""),
    }
```

Used by tools for:
- Artifact paths: `artifacts/{plan_id}/{task_id}/`
- File tracking: `record_file_touch(path, "read"|"write", agent)`
- RAG tagging: `ingest_text(text, source=f"agent:{agent}", ...)`

---

## Security Layers

1. **Registry allowlist**: Only tools in TOOL_REGISTRY can be invoked
2. **Allowlist enforcement** (optional): per-agent tool filtering
3. **Path traversal protection**: `_path_is_within()` check in read/write tools
4. **Network blocking**: `browse_url()` blocks localhost, file://, 169.254.x.x
5. **Environment sanitization**: `execute_code()` strips API keys before exec
6. **Approval gate**: `execute_code` requires human approval (currently disabled)
7. **Resource limits**: 512 MB memory, 30s CPU timeout
8. **Sandbox modes**: none, subprocess (preferred), docker (best)

---

## Approval System (Currently Disabled)

```python
APPROVAL_REQUIRED = {"execute_code"}

def request_approval(tool_call_id: str, tool_name: str, args: dict, ...) -> bool:
    """Block until human approves/rejects (5 min timeout)"""
    
def resolve_approval(tool_call_id: str, approved: bool):
    """Called from serve.py when human clicks approve/reject"""
    
def get_pending_approvals() -> list[dict]:
    """Dashboard query for pending approvals"""
```

**Current state**: `auto_approve_all=True` in orchestrator.py:661 (line 661)
- Approval system implemented but disabled for agents
- Can re-enable by changing one parameter

---

## Error Handling in Tool Execution

All exceptions caught and returned as strings:

```python
try:
    result = tool_fn.invoke(args)
    return str(result)
except Exception as e:
    return f"Tool error ({name}): {e}"
```

Tool results truncated to 4000 chars before ToolMessage (orchestrator.py:662).

---

## MCP Integration — Minimal Required Changes

To add MCP support, need:

1. `MCPToolAdapter` class in tools.py to wrap MCP tools
2. `mcp_manager.py` to handle server lifecycle
3. Update `TOOL_REGISTRY` to include MCP tools dynamically
4. Config schema for MCP servers in config.json
5. Update generate_plan.py SYSTEM_PROMPT to reference MCP tools

**No changes needed** to:
- Orchestrator.py (get_tools() and execute_tool_call() already generic)
- Tool binding (bind_tools() works with any callable)
- Tool execution loop (same code works for MCP)

---

## Configuration Schema (Potential Expansion)

```json
{
  "default_tolerance": 8,
  "webhook_url": "",
  
  "default_agent_tools": ["web_search", "read_file", "write_file"],
  
  "mcp_servers": [
    {"name": "filesystem", "command": "python -m mcp.servers.filesystem", "auto_start": true}
  ],
  
  "tool_timeout_seconds": 30,
  
  "sandbox_mode": "subprocess"
}
```

---

## Logging & Monitoring

All tool calls logged via:

```python
log_event(f"  [TOOL] {tc_name}({str(tc_args)[:80]})")
```

Persisted in per-session event deque (max 80 events).

File access tracked:

```python
record_file_touch(path, "read"|"write", agent_name)
```

Generates heatmap of which agents touched which files.

---

## Checklist: Adding a New Tool

1. Define with `@tool` decorator in tools.py
2. Add to `TOOL_REGISTRY` dict
3. Update SYSTEM_PROMPT in generate_plan.py line 79
4. No changes to orchestrator.py needed (generic execution)
5. Update validate_plan.py to accept tool name
6. Document in README

Example:

```python
@tool
def my_search(query: str) -> str:
    """Search for information about a topic."""
    return f"Results for {query}"

TOOL_REGISTRY["my_search"] = my_search
```

Then in plan.md:
```markdown
### AGENT: researcher
- tools: web_search, my_search, read_file
```

---

## Next Steps for MCP

1. Create `mcp_manager.py` with MCPManager class (~150 lines)
2. Add MCPToolAdapter to tools.py (~50 lines)
3. Create `mcp_client.py` for JSON-RPC (~100 lines)
4. Update config.json with mcp_servers schema (~15 lines)
5. Update generate_plan.py to hint at MCP tools (~20 lines)
6. Test with example MCP server

See `MCP_Integration_Roadmap.md` for detailed implementation guide.

