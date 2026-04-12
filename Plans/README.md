# NORT Tool Orchestration Documentation

## Overview

This folder contains comprehensive analysis of NORT's tool orchestration system for planning MCP (Model Context Protocol) server integration.

## Documents

### 1. **NORT_Tool_Orchestration_Deep_Dive.md** (762 lines, 26 KB)

**Complete reference guide** covering every aspect of tool handling:

- **Section 1**: tools.py — Tool definitions, registry, approval system, context management
- **Section 2**: orchestrator.py — Tool binding, plan parsing, execution loop, logging
- **Section 3**: generate_plan.py — Tool specification schema, system prompt
- **Section 4**: config.json — Configuration possibilities for tool management
- **Section 5**: status_bridge.py — Tool tracking and logging infrastructure
- **Section 6**: LangGraph integration points
- **Section 7**: Full data flow from plan to execution
- **Section 8**: Sandbox execution modes (none, subprocess, docker)
- **Section 9**: All key function signatures with complete code
- **Section 10**: Tool call lifecycle with message flow
- **Section 11**: Multi-layer security architecture
- **Section 12**: MCP integration roadmap (high-level overview)
- **Section 13**: Code location index
- **Section 14**: Implementation examples

**Use this for**: Understanding every detail of current implementation, making architectural decisions, security review.

---

### 2. **MCP_Integration_Roadmap.md** (441 lines, 13 KB)

**Detailed implementation plan** for adding MCP server support:

- **Phase 1**: MCPToolAdapter class (wraps MCP tools as LangChain tools)
- **Phase 2**: MCPManager class (server lifecycle management)
- **Phase 3**: Registry population (dynamic tool loading)
- **Phase 4**: config.json schema (MCP server definitions)
- **Phase 5**: Plan schema updates (SYSTEM_PROMPT modifications)
- **Phase 6**: Validation updates (tool name checking)
- **Phase 7**: No changes needed to execution (backward compatible)
- **Integration checklist**: 10 items for complete implementation
- **Default tools configuration**: Per-agent tool defaults
- **Security considerations**: Input validation, timeouts, env vars, approval gates
- **Testing strategy**: Unit test examples
- **Backward compatibility**: How existing systems continue to work
- **Future enhancements**: Caching, composition, metrics, versioning

**Use this for**: Planning implementation sprint, estimating effort, understanding integration points.

---

### 3. **QUICK_REFERENCE.md** (357 lines, 11 KB)

**Developer quick-start guide** with copy-paste ready code:

- Core data flow diagram
- Key files and their purposes
- Complete function signatures for: `get_tools()`, `execute_tool_call()`, `set_tool_context()`
- TOOL_REGISTRY contents with descriptions
- Tool call loop from orchestrator.py
- Plan schema specification and parsing
- Tool context and metadata usage
- Security layers (8 different protections)
- Approval system details
- Error handling patterns
- MCP integration summary (minimal changes needed)
- Configuration schema examples
- Logging and monitoring patterns
- Checklist for adding new tools
- Next steps for MCP implementation

**Use this for**: Quick lookup during development, copy-paste code patterns, understanding security model.

---

## Key Findings

### Current Tool Architecture

- **8 builtin tools** via LangChain `@tool` decorators
- **5 aliases** for backward compatibility
- **TOOL_REGISTRY** dict for name-to-callable mapping
- **get_tools()** resolves tool names to LangChain objects
- **execute_tool_call()** invokes tools with error handling
- **Thread-local context** for artifact paths and metadata

### Execution Flow

```
plan.md → SubAgentSpec.tools (strings)
  ↓
get_tools() → LangChain tool objects
  ↓
llm().bind_tools([...]) → Configure LLM
  ↓
LLM invokes → resp.tool_calls
  ↓
execute_tool_call() → tool_fn.invoke(args)
  ↓
ToolMessage(result) → next iteration
```

### MCP Integration Requirements

1. **MCPToolAdapter** class to wrap MCP tools (50 lines)
2. **MCPManager** for server lifecycle (150 lines)
3. **Registry expansion** to include MCP tools dynamically (5 lines)
4. **Config schema** for MCP server definitions (15 lines)
5. **Plan schema** updates to reference MCP tools (20 lines)

**Total**: ~240 lines of new code, fully backward compatible.

### Security Architecture

- Registry allowlist (only known tools)
- Per-agent allowlist (optional filtering)
- Path traversal prevention
- Network blocking (SSRF prevention)
- Environment sanitization
- Approval gate (currently disabled)
- Resource limits (512 MB, 30s timeout)
- Sandbox isolation (3 modes)

### Critical Implementation Details

- **Tools are fully generic**: MCP tools need only implement `invoke(args) -> str`
- **No orchestrator changes needed**: Existing binding code works with any callable
- **Config-driven**: MCP servers defined in config.json, not hardcoded
- **Thread-safe**: Tool context via threading.local()
- **Fully logged**: All tool calls tracked for audit trail
- **Error-resilient**: All exceptions caught and returned as strings

---

## For MCP Planning

### Minimal Changes Required

| Component | Change | Effort |
|-----------|--------|--------|
| tools.py | Add MCPToolAdapter | Low |
| NEW mcp_manager.py | Server lifecycle | Medium |
| NEW mcp_client.py | JSON-RPC transport | Medium |
| orchestrator.py | Call init_mcp() | Trivial |
| generate_plan.py | Update SYSTEM_PROMPT | Low |
| validate_plan.py | Allow server/tool syntax | Low |
| config.json | Add mcp_servers schema | Trivial |

### Backward Compatibility

- Existing plans work unchanged
- Builtin tools always available
- MCP servers optional (graceful degradation)
- No breaking changes to APIs

### Testing Strategy

- Unit tests for MCPManager
- Integration tests with example MCP server
- Plan parsing with MCP tool names
- Registry population tests
- Execution flow tests

---

## Code Locations Reference

| Aspect | File | Lines |
|--------|------|-------|
| Tool definitions | tools.py | 150-401 |
| Tool registry | tools.py | 386-401 |
| get_tools() | tools.py | 404-418 |
| execute_tool_call() | tools.py | 423-457 |
| Approval system | tools.py | 29-110 |
| Tool context | tools.py | 112-136 |
| Plan parsing | orchestrator.py | 250-356 |
| Tool binding | orchestrator.py | 639, 649 |
| Tool loop | orchestrator.py | 651-667 |
| SubAgentSpec | orchestrator.py | 167-172 |
| Plan schema | generate_plan.py | 62-189 |
| Event logging | status_bridge.py | 180-187 |
| File tracking | status_bridge.py | 121-134 |

---

## How to Use These Documents

1. **Start here**: Read QUICK_REFERENCE.md for overview (5 min)
2. **Deep dive**: Read NORT_Tool_Orchestration_Deep_Dive.md sections 1-3 for details (15 min)
3. **Implementation**: Read MCP_Integration_Roadmap.md for phased approach (10 min)
4. **Code references**: Use Deep Dive sections 9-14 for exact line numbers and patterns (as needed)

---

## Questions Answered

### "How are tools bound to the LLM?"
See Deep Dive Section 2.3, Quick Reference: Tool Call Loop

### "What's the full tool execution path?"
See Deep Dive Section 7 (Data Flow diagram), Quick Reference: Core Data Flow

### "How do I add a new tool?"
See Quick Reference: Checklist section

### "What security protections exist?"
See Deep Dive Section 11, Quick Reference: Security Layers

### "How can I integrate MCP?"
See MCP_Integration_Roadmap.md, all 7 phases

### "What config options are available?"
See Deep Dive Section 4, Quick Reference: Configuration Schema

### "How are tools logged and monitored?"
See Deep Dive Section 5, Quick Reference: Logging & Monitoring

### "What happens if a tool fails?"
See Deep Dive Section 10.1, Quick Reference: Error Handling

---

## Key Insights for Implementation

1. **Tools are just callables**: Any object with `invoke(args) -> str` method works
2. **Binding is generic**: `llm().bind_tools([...])` doesn't care about tool source
3. **Registry is extensible**: Just add new entries at runtime
4. **Execution is async-friendly**: Tool loop can be parallelized per message
5. **Context is thread-local**: Safe for concurrent agent execution
6. **Errors are strings**: All failures return human-readable error messages
7. **Config-driven**: MCP servers should be defined in config.json, not code

---

## Next Steps

1. Review Quick Reference (5 min) — understand current model
2. Read Deep Dive Section 2-3 (10 min) — see binding and execution
3. Study MCP_Integration_Roadmap Phase 1-3 (15 min) — design adapter
4. Create MCPToolAdapter class — extend tools.py
5. Create MCPManager class — new mcp_manager.py
6. Add test cases — verify tool discovery and invocation
7. Update config schema — add mcp_servers definition
8. Test with example MCP server — GitHub API, filesystem, etc.

---

Created: 2026-04-02
Last Updated: 2026-04-02
Analysis Depth: Complete (all functions, data flows, security layers)
