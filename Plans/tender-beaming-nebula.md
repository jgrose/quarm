# Plan: MCP Server Support + Default Tools

## Context

NORT agents have a working tool system (8 builtin tools via LangChain `@tool` in [tools.py](tools.py)), but two capabilities are missing:

1. **No MCP support** -- Agents can only use hardcoded Python tools. No way to connect to external MCP servers (Brave Search, filesystem, databases, etc.)
2. **No default tools** -- Every plan must explicitly specify `- tools: [...]` per agent. Common tools like `web_search` and `browse_url` should be available automatically.

This plan adds MCP server integration and configurable default tools while maintaining 100% backward compatibility.

---

## Feature 1: Default Tools

### Approach

Add a `default_tools` list to `config.json`. Modify `get_tools()` in `tools.py` to merge defaults with agent-specified tools. Support `- tools: none` in plans to opt out.

### Files to Modify

| File | Change |
|------|--------|
| [config.json](config.json) | Add `"default_tools": ["web_search", "browse_url", "read_file"]` |
| [tools.py:404](tools.py#L404) | Add `load_default_tools()` function; modify `get_tools()` to merge defaults; handle `["none"]` sentinel |
| [generate_plan.py:79](generate_plan.py#L79) | Update SYSTEM_PROMPT to document that tools line is optional (defaults apply) and `none` disables all |

### Logic

```python
def get_tools(tool_names, allowed_tools=None, include_defaults=True):
    if tool_names == ["none"]:
        return []
    if include_defaults:
        defaults = load_default_tools()  # from config.json
        merged = list(dict.fromkeys(defaults + tool_names))  # union, defaults first
    else:
        merged = tool_names
    # ... existing lookup logic unchanged ...
```

- No `- tools:` line in plan -> `agent.get("tools", [])` returns `[]` -> defaults apply
- `- tools: [execute_code]` -> merged with defaults -> `[web_search, browse_url, read_file, execute_code]`
- `- tools: none` -> returns `[]` -> no tools at all

---

## Feature 2: MCP Server Support

### Approach

Create an MCP client manager that lazily connects to configured MCP servers and wraps their tools as LangChain `BaseTool` objects, which get registered into the existing `TOOL_REGISTRY`. MCP tools are then usable identically to builtins.

### Configuration

Add `mcp_servers` to [config.json](config.json):

```json
{
  "default_tools": ["web_search", "browse_url", "read_file"],
  "mcp_servers": {
    "brave_search": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@anthropic/brave-search-mcp"],
      "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
    },
    "filesystem": {
      "type": "sse",
      "url": "http://localhost:3001/sse"
    }
  }
}
```

### New Files

#### [mcp_client.py](mcp_client.py) (~200 lines)

MCP client lifecycle manager. Handles lazy connection, tool discovery, and the async-to-sync bridge.

Key design: Each MCP server gets a dedicated background daemon thread running an `asyncio` event loop. Sync callers use `asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)` to bridge. This is thread-safe for the `ThreadPoolExecutor` parallel task execution.

```python
class MCPClientManager:
    def load_config(self, mcp_servers: dict) -> None
    def discover_tools(self, server_name: str) -> list[dict]
    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str
    def discover_all_tools(self) -> dict[str, list[dict]]
    def shutdown(self) -> None

def init_mcp_from_config() -> MCPClientManager  # loads config.json, creates singleton
def get_mcp_manager() -> MCPClientManager        # returns singleton
def shutdown_mcp() -> None                        # cleanup
```

#### [mcp_tool_wrapper.py](mcp_tool_wrapper.py) (~80 lines)

LangChain `BaseTool` subclass wrapping a single MCP tool.

```python
class MCPTool(BaseTool):
    name: str           # "brave_search.web_search"
    description: str    # from MCP schema
    server_name: str
    mcp_tool_name: str

    def _run(self, **kwargs) -> str:
        return get_mcp_manager().call_tool(self.server_name, self.mcp_tool_name, kwargs)

def create_mcp_tools(server_name, tool_schemas) -> list[MCPTool]
def register_mcp_tools_in_registry(registry: dict) -> int
```

### Files to Modify

| File | Change |
|------|--------|
| [tools.py:401](tools.py#L401) | Add `init_mcp_tools()` that calls `register_mcp_tools_in_registry(TOOL_REGISTRY)` |
| [orchestrator.py:36](orchestrator.py#L36) | Import `init_mcp_from_config`, `shutdown_mcp` (with ImportError fallback) |
| [orchestrator.py:1414](orchestrator.py#L1414) | After `fetch_available_models()`, call `init_mcp_from_config()` + `init_mcp_tools()` |
| [orchestrator.py](orchestrator.py) | At end of `run()`, call `shutdown_mcp()` |
| [tests/conftest.py](tests/conftest.py) | Add `mcp_client` and `mcp_tool_wrapper` module stubs |

### Data Flow

```
config.json {mcp_servers, default_tools}
  |
  v (startup)
MCPClientManager.load_config()         -- parses server defs, NO connections yet
  |
  v
init_mcp_tools()                       -- discovers tools from each server (lazy-connects)
  |                                       wraps as MCPTool, adds to TOOL_REGISTRY
  v
TOOL_REGISTRY = {
  "web_search": <builtin>,             -- existing
  "brave_search.web_search": <MCPTool> -- new, from MCP
  ...
}
  |
  v (per-task)
get_tools(agent.tools)                 -- merges defaults, looks up TOOL_REGISTRY
  |
  v
llm.bind_tools(agent_tools)            -- MCPTool.args_schema works with bind_tools
  |
  v
execute_tool_call(tc, tools)           -- MCPTool._run() -> MCPClientManager.call_tool()
                                          -> async bridge -> MCP server -> result string
```

### Async/Sync Bridge

```
Worker Thread (ThreadPoolExecutor)     MCP Loop Thread (per-server, daemon)
  |                                      |
  | MCPTool._run(**kwargs)               |
  |   -> manager.call_tool(srv, tool, args)
  |   -> asyncio.run_coroutine_threadsafe(
  |        session.call_tool(name, args),
  |        loop
  |      ).result(timeout=30) ---------->|
  |                                      |-- await session.call_tool(...)
  |                                      |<- CallToolResult
  |<---- result string -----------------|
```

### Error Handling

- **`mcp` package not installed**: `ImportError` caught at import time, all MCP features silently disabled
- **Server connection fails**: Warning logged, that server's tools not registered, others unaffected
- **Server crashes mid-run**: `call_tool()` catches errors, attempts one reconnect, returns error string to agent on failure
- **Unknown tool name in config**: `get_tools()` already silently skips unknown names (existing behavior)
- **No `mcp_servers` in config**: No MCP behavior at all, fully backward compatible

---

## Implementation Order

1. **Install dependency**: `pip install mcp` into `.venv`
2. **Default tools** (no new dependencies): `config.json` + `tools.py` + `generate_plan.py`
3. **MCP client manager**: Create `mcp_client.py`
4. **MCP tool wrapper**: Create `mcp_tool_wrapper.py`
5. **Orchestrator integration**: Hook init/shutdown into `orchestrator.py`
6. **Test stubs**: Update `tests/conftest.py`

## Verification

1. **Default tools**: Run `python orchestrator.py plan.example.md` -- agents without explicit `- tools:` should get defaults; agents with `- tools: none` should get no tools
2. **MCP integration**: Configure a test MCP server in `config.json`, run a plan that references its tools, verify tool calls flow through and results appear in agent output
3. **Backward compat**: Run existing plans unchanged -- behavior should be identical (plus defaults)
4. **Tests**: `python -m pytest tests/ -v` -- all existing tests should pass
