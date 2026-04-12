# MCP Integration Roadmap for NORT

Based on deep analysis of NORT's tool orchestration system, this document outlines the minimal changes needed to add MCP server support.

---

## 1. Architecture Overview

Current flow:
```
plan.md → parse_plan() → SubAgentSpec.tools=[strings]
         → _execute_single_task()
            → get_tools(agent["tools"]) → [LangChain tool objects]
               → llm().bind_tools([...])
                  → LLM invokes tools
                     → execute_tool_call()
```

MCP integration points:
1. **Discovery**: Load MCP servers at startup, populate TOOL_REGISTRY dynamically
2. **Binding**: Wrap MCP tools as LangChain-compatible callables
3. **Execution**: Route MCP tool calls to appropriate MCP server
4. **Configuration**: Define MCP servers and default tools in config.json

---

## 2. Implementation Strategy

### Phase 1: MCP Tool Wrapper (tools.py)

Create adapter to make MCP tools callable like LangChain tools:

```python
class MCPToolAdapter:
    """Wraps an MCP tool definition to work with LangChain bind_tools()"""
    
    def __init__(self, mcp_client, tool_def: dict):
        self.mcp_client = mcp_client
        self.name = tool_def["name"]
        self.description = tool_def.get("description", "")
        self.schema = tool_def.get("inputSchema", {})
        
        # Convert MCP schema to LangChain-compatible format
        self.input_schema = self._convert_schema(self.schema)
    
    def invoke(self, input: dict) -> str:
        """Call the MCP server tool"""
        try:
            result = self.mcp_client.call_tool(self.name, input)
            return str(result)
        except Exception as e:
            return f"MCP error ({self.name}): {e}"
    
    def _convert_schema(self, mcp_schema: dict):
        """Convert MCP JSONSchema to Python type hints"""
        # Extract input parameters from MCP schema
        properties = mcp_schema.get("properties", {})
        required = mcp_schema.get("required", [])
        
        # Build Pydantic model dynamically for LangChain compatibility
        # (LangChain expects tool inputs to have schema via Pydantic)
        pass
```

**Location**: `tools.py` lines 112-140 (after tool context, before tool definitions)

---

### Phase 2: MCP Client Initialization (new file: mcp_manager.py)

```python
# mcp_manager.py
"""Manages MCP server connections and tool discovery."""

import json
import subprocess
from typing import Optional
from pathlib import Path

class MCPManager:
    def __init__(self, config_path: str):
        self.servers = {}
        self.tools = {}
        self.clients = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load MCP server definitions from config.json"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            mcp_servers = config.get("mcp_servers", [])
            for server_def in mcp_servers:
                self._register_server(server_def)
        except Exception as e:
            log_event(f"[MCP] Failed to load config: {e}")
    
    def _register_server(self, server_def: dict):
        """Start MCP server and discover its tools"""
        name = server_def["name"]
        command = server_def.get("command", "")
        
        if not command:
            log_event(f"[MCP] Skipping {name} — no command specified")
            return
        
        try:
            # Start server process
            proc = subprocess.Popen(
                command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )
            # Initialize JSON-RPC client
            client = MCPClient(proc)
            
            # Discover tools
            tools = client.list_tools()
            for tool in tools:
                tool_id = f"{name}/{tool['name']}"
                self.tools[tool_id] = tool
                tool["_server"] = name
            
            self.servers[name] = server_def
            self.clients[name] = client
            log_event(f"[MCP] Loaded {name}: {len(tools)} tools")
        except Exception as e:
            log_event(f"[MCP] Failed to register {name}: {e}")
    
    def get_tool_callable(self, tool_id: str):
        """Get a callable wrapper for an MCP tool"""
        if tool_id not in self.tools:
            return None
        
        tool_def = self.tools[tool_id]
        server_name = tool_def.get("_server")
        client = self.clients.get(server_name)
        
        return MCPToolAdapter(client, tool_def)
    
    def populate_registry(self, registry: dict):
        """Add all MCP tools to TOOL_REGISTRY"""
        for tool_id, tool_def in self.tools.items():
            registry[tool_id] = self.get_tool_callable(tool_id)

_mcp_manager: Optional[MCPManager] = None

def init_mcp(config_path: str = "config.json"):
    global _mcp_manager
    _mcp_manager = MCPManager(config_path)

def get_mcp_tools() -> dict:
    """Return all MCP tools for registry population"""
    if not _mcp_manager:
        return {}
    return {tid: _mcp_manager.get_tool_callable(tid) 
            for tid in _mcp_manager.tools.keys()}
```

---

### Phase 3: Registry Population (tools.py)

Modify `get_tools()` to include MCP tools:

```python
# At module initialization (after TOOL_REGISTRY definition, line 401)

def _populate_mcp_tools():
    """Load MCP tools into registry at startup"""
    try:
        from mcp_manager import get_mcp_tools
        mcp_tools = get_mcp_tools()
        TOOL_REGISTRY.update(mcp_tools)
        log_event(f"[MCP] Added {len(mcp_tools)} MCP tools to registry")
    except Exception as e:
        log_event(f"[MCP] Failed to load MCP tools: {e}")

# Call at module load time (if config has mcp_servers defined)
_populate_mcp_tools()
```

**Alternative**: Call `_populate_mcp_tools()` from orchestrator.py:run() before execute phase.

---

### Phase 4: Configuration Schema (config.json)

Add MCP server definitions:

```json
{
  "allowed_models": [...],
  "default_tolerance": 8,
  "webhook_url": "",
  
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": "python -m mcp.servers.filesystem",
      "auto_start": true,
      "env": {}
    },
    {
      "name": "github",
      "command": "python -m mcp.servers.github",
      "auto_start": false,
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  ],
  
  "default_tools": [
    "web_search",
    "read_file",
    "write_file",
    "execute_code",
    "filesystem/list_directory",
    "filesystem/read_file",
    "github/search_repositories"
  ]
}
```

---

### Phase 5: Plan Schema Update (generate_plan.py)

Modify SYSTEM_PROMPT to include MCP tools in the available tool list:

**Location**: `generate_plan.py` line 79

Current:
```markdown
- tools: [choose from: web_search, browse_url, rag_search, rag_store, download_artifact, read_file, write_file, execute_code]
```

Updated (dynamic):
```markdown
- tools: [choose from builtin tools or MCP tools — see catalog below]
```

Add MCP tools to agent catalog hint (similar to existing agent_registry logic):

```python
# In generate_plan_streaming(), after format_agent_catalog() (line 204)

def format_mcp_tool_catalog():
    """Build hint for available MCP tools"""
    try:
        from mcp_manager import _mcp_manager
        if not _mcp_manager or not _mcp_manager.tools:
            return ""
        
        lines = ["## Available MCP Tools\n"]
        by_server = {}
        for tool_id, tool_def in _mcp_manager.tools.items():
            server = tool_def.get("_server", "unknown")
            if server not in by_server:
                by_server[server] = []
            by_server[server].append(tool_id)
        
        for server, tools in sorted(by_server.items()):
            lines.append(f"### {server}")
            for tool in tools:
                lines.append(f"- {tool}")
            lines.append("")
        
        return "\n".join(lines)
    except Exception:
        return ""

mcp_hint = format_mcp_tool_catalog()
if mcp_hint:
    agent_hint += "\n" + mcp_hint
```

---

### Phase 6: Tool Name Validation (validate_plan.py)

Update plan validation to allow `server/tool_name` syntax:

```python
# In validate_plan.py, wherever tool names are validated

def _get_valid_tool_names() -> set[str]:
    """Get all valid tool names: builtin + MCP"""
    from tools import TOOL_REGISTRY
    return set(TOOL_REGISTRY.keys())

# When validating agent tools:
valid_tools = _get_valid_tool_names()
for tool_name in agent["tools"]:
    if tool_name not in valid_tools:
        errors.append(f"Unknown tool '{tool_name}'")
```

---

### Phase 7: Execution Path (orchestrator.py)

No changes needed — `get_tools()` and `execute_tool_call()` work with any callable.

The existing code at lines 639, 649, 661 automatically works with MCP tools:

```python
agent_tools = get_tools(agent.get("tools", []))
                ↑ resolves both builtin and MCP tools
if agent_tools:
    llm_with_tools = llm(model).bind_tools(agent_tools)
    ↑ LLM binds both types the same way

result = execute_tool_call(tc, agent_tools, auto_approve_all=True)
    ↑ calls MCPToolAdapter.invoke() same as builtin tools
```

---

## 3. Integration Checklist

- [ ] Create `mcp_manager.py` with MCPManager class
- [ ] Add MCPToolAdapter to `tools.py`
- [ ] Create `mcp_client.py` with JSON-RPC transport
- [ ] Update `TOOL_REGISTRY` population to include MCP tools
- [ ] Add config schema for `mcp_servers` to `config.json`
- [ ] Update SYSTEM_PROMPT in `generate_plan.py` to reference MCP tools
- [ ] Add MCP tool validation to `validate_plan.py`
- [ ] Test with example MCP server (e.g., GitHub API)
- [ ] Document MCP server setup in README
- [ ] Add error handling for failed MCP connections

---

## 4. Default Tools Configuration

Add to config.json to give agents a default set when they specify no tools:

```json
{
  "default_agent_tools": ["web_search", "read_file", "write_file", "rag_search"]
}
```

Modify `get_tools()` to use defaults:

```python
def get_tools(tool_names: list[str], allowed_tools: list[str] = None) -> list:
    if not tool_names:
        # Fall back to defaults from config
        config = _load_orchestrator_config()
        tool_names = config.get("default_agent_tools", [])
    
    # ... rest of function
```

---

## 5. Security Considerations for MCP

1. **Input validation**: MCPToolAdapter must validate inputs against tool schema
2. **Timeout handling**: Each MCP call should have timeout (same as execute_code: 30s)
3. **Environment variables**: Support env var substitution in MCP server command (e.g., `${GITHUB_TOKEN}`)
4. **Approval gate**: Consider requiring approval for MCP tools, similar to execute_code
5. **Rate limiting**: Track MCP tool calls to prevent abuse
6. **Error handling**: MCP server crashes should not crash orchestrator

---

## 6. Testing Strategy

```python
# test_mcp_integration.py

def test_mcp_tool_discovery():
    """Test that MCP server tools are discovered"""
    manager = MCPManager("config.json")
    assert len(manager.tools) > 0
    
def test_mcp_tool_invocation():
    """Test that MCP tools can be invoked"""
    adapter = manager.get_tool_callable("filesystem/list_directory")
    result = adapter.invoke({"path": "."})
    assert isinstance(result, str)

def test_mcp_in_registry():
    """Test that MCP tools appear in TOOL_REGISTRY"""
    tools = get_tools(["filesystem/list_directory"])
    assert len(tools) == 1
    assert tools[0].name == "filesystem/list_directory"

def test_mcp_in_plan():
    """Test that agents can specify MCP tools in plan.md"""
    plan = """
    ### AGENT: search_engine
    - tools: mcp_server/search, web_search
    """
    obj, mgrs, agents, tasks, revs = parse_plan("test_plan.md")
    assert "mcp_server/search" in agents[0]["tools"]
```

---

## 7. Backward Compatibility

- Existing plans without MCP tools continue to work
- TOOL_REGISTRY expansion doesn't break existing lookups
- Builtin tools always available (no breaking changes)
- MCP servers optional (graceful degradation if unavailable)

---

## 8. Future Enhancements

1. **Tool composition**: Combine multiple MCP tools into workflows
2. **Caching**: Cache MCP tool results per session
3. **Tool metrics**: Track which MCP tools are used most frequently
4. **Dynamic registration**: Auto-discover MCP servers from environment
5. **Nested tools**: Allow MCP tools to call other MCP tools
6. **Tool versioning**: Support multiple versions of same MCP tool
7. **Fallback chains**: Try alternative tools if primary fails

---

## 9. File Modification Summary

| File | Changes | Scope |
|------|---------|-------|
| `tools.py` | Add MCPToolAdapter class | ~50 lines |
| `mcp_manager.py` | NEW: MCP server lifecycle | ~150 lines |
| `mcp_client.py` | NEW: JSON-RPC transport | ~100 lines |
| `orchestrator.py` | Call `init_mcp()` in run() | ~5 lines |
| `generate_plan.py` | Add MCP tool hint to prompt | ~20 lines |
| `validate_plan.py` | Validate MCP tool names | ~10 lines |
| `config.json` | Add mcp_servers schema | ~15 lines |
| `status_bridge.py` | Log MCP operations | ~5 lines (optional) |

**Total new lines**: ~350 lines (all non-breaking)

