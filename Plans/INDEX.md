# NORT Tool Orchestration Analysis — Complete Index

**Created**: 2026-04-02  
**Status**: Complete deep-dive analysis  
**Scope**: All tool binding, execution, and MCP integration planning  

---

## Document Overview

| Document | Purpose | Size | Read Time |
|----------|---------|------|-----------|
| **README.md** | Overview & navigation guide | 5 KB | 5 min |
| **INDEX.md** | This file — complete index | 3 KB | 3 min |
| **QUICK_REFERENCE.md** | Copy-paste code patterns & quick lookup | 11 KB | 10 min |
| **NORT_Tool_Orchestration_Deep_Dive.md** | Complete technical reference | 26 KB | 30 min |
| **MCP_Integration_Roadmap.md** | Implementation plan with 7 phases | 13 KB | 20 min |

**Total**: 58 KB across 5 comprehensive documents

---

## Quick Navigation

### By Use Case

**"I'm new to NORT tools"**
1. Read README.md (overview)
2. Read QUICK_REFERENCE.md (core data flow + patterns)
3. Skim Deep Dive Section 1 (tool definitions)

**"I need to add a new builtin tool"**
1. See QUICK_REFERENCE.md: Checklist section
2. See Deep Dive Section 1.1 (tool definitions)
3. See Deep Dive Section 1.2 (TOOL_REGISTRY)

**"I'm planning MCP integration"**
1. Read MCP_Integration_Roadmap.md (all sections)
2. See QUICK_REFERENCE.md: MCP Integration section
3. See Deep Dive Section 12 (high-level overview)

**"I'm implementing MCP support"**
1. Read MCP_Integration_Roadmap.md Phase 1-3 (adapter + manager)
2. See QUICK_REFERENCE.md: Function signatures
3. See Deep Dive Section 2.3 (tool binding internals)

**"I need to understand security"**
1. See QUICK_REFERENCE.md: Security Layers (8 protections)
2. See Deep Dive Section 11 (complete security architecture)
3. See Deep Dive Section 8 (sandbox modes)

**"I'm debugging a tool execution failure"**
1. See Deep Dive Section 10 (tool call lifecycle)
2. See QUICK_REFERENCE.md: Error Handling
3. See Deep Dive Section 1.6 (execute_tool_call internals)

---

## Content By Topic

### Tool Architecture

| Topic | Quick Ref | Deep Dive | MCP Roadmap |
|-------|-----------|-----------|------------|
| Tool definitions | TOOL_REGISTRY Contents | Section 1.1 | — |
| Tool registry | TOOL_REGISTRY Contents | Section 1.2 | Phase 3 |
| get_tools() | Function Signatures | Section 1.5 | — |
| execute_tool_call() | Function Signatures | Section 1.6 | — |
| set_tool_context() | Function Signatures | Section 1.4 | — |
| Tool approval | Approval System | Section 1.3 | Phase 5+ |

### Tool Execution

| Topic | Quick Ref | Deep Dive |
|-------|-----------|-----------|
| Core data flow | Core Data Flow | Section 7 |
| Tool binding | Tool Call Loop | Section 2.3 |
| LLM integration | Tool Call Loop | Section 2.4 |
| Tool invocation | Tool Call Loop | Section 2.3 |
| Message handling | Tool Call Loop | Section 10.2 |
| Error handling | Error Handling | Section 10.1 |
| Logging | Logging & Monitoring | Section 5.2 |

### Planning & Implementation

| Topic | Quick Ref | MCP Roadmap |
|-------|-----------|-------------|
| Architecture overview | MCP Integration | Section 1 |
| Phase 1: Adapter | Next Steps | Phase 1 |
| Phase 2: Manager | Next Steps | Phase 2 |
| Phase 3: Registry | Next Steps | Phase 3 |
| Phase 4: Config | Configuration Schema | Phase 4 |
| Phase 5: Plan schema | Next Steps | Phase 5 |
| Testing | Next Steps | Section 6 |
| Backward compat | — | Section 7 |
| Security | — | Section 5 |

### Configuration

| Topic | Quick Ref | Deep Dive | MCP Roadmap |
|-------|-----------|-----------|-------------|
| Tool timeout | Config Schema | Section 4 | — |
| Sandbox mode | Config Schema | Section 8 | — |
| Default tools | Config Schema | — | Section 4 |
| MCP servers | Config Schema | — | Section 4 |
| Tool allowlist | — | Section 1.5 | — |

---

## Code Locations

### Primary Files

| File | Purpose | Key Content |
|------|---------|---|
| **tools.py** | Tool registry & execution | Lines 1-457 |
| **orchestrator.py** | Tool binding & planning | Lines 1-1585 |
| **generate_plan.py** | Plan schema | Lines 1-378 |
| **status_bridge.py** | Logging infrastructure | Lines 1-380 |
| **config.json** | Configuration | All |

### By Line Number (tools.py)

- 29-32: APPROVAL_REQUIRED set
- 40-75: request_approval() function
- 78-101: resolve_approval() function
- 104-109: get_pending_approvals() function
- 112-136: Tool context (threading.local)
- 117-121: set_tool_context()
- 124-129: _ctx()
- 139-145: _path_is_within()
- 150-401: Tool definitions (@tool decorators)
  - 150-154: web_search
  - 157-169: browse_url
  - 172-182: rag_search
  - 185-201: rag_store
  - 204-227: download_artifact
  - 230-247: read_file
  - 250-264: write_file
  - 367-381: execute_code
- 282-296: _sanitized_env()
- 299-302: _preexec_limits()
- 305-313: _execute_none()
- 316-340: _execute_subprocess()
- 343-364: _execute_docker()
- 386-401: TOOL_REGISTRY dict
- 404-418: get_tools()
- 423-457: execute_tool_call()

### By Line Number (orchestrator.py)

- 167-172: SubAgentSpec dataclass
- 250-356: parse_plan() function
- 263-265: rxl() helper (regex list parsing)
- 271: Tool name extraction (regex pattern)
- 381-383: llm() helper
- 586-687: _execute_single_task()
- 638: set_tool_context() call
- 639: get_tools() call
- 649: llm().bind_tools() call
- 651-667: Tool invocation loop
- 661: execute_tool_call() call

### By Line Number (generate_plan.py)

- 62-189: SYSTEM_PROMPT (complete schema)
- 79: Tool list in schema
- 192-281: generate_plan_streaming()
- 204: Agent catalog formatting
- 284-366: generate_plan()

### By Line Number (status_bridge.py)

- 121-134: record_file_touch()
- 180-187: log_event()
- 152-169: register_rosters()

---

## Key Concepts

### Tool Name Resolution

```
Agent spec: "tools: web_search, read_file"
          ↓
TOOL_REGISTRY lookup: {"web_search": web_search_fn, "read_file": read_file_fn, ...}
          ↓
get_tools(): [web_search_fn, read_file_fn]
          ↓
llm().bind_tools([...])
```

### Tool Execution Cycle

```
LLM → resp.tool_calls=[{name, args, id}]
  → execute_tool_call()
    → find tool by name
    → check approval
    → call tool_fn.invoke(args)
    → return result string
  → ToolMessage(result, id)
  → messages.append()
  → next iteration
```

### MCP Tool Integration

```
config.json: mcp_servers=[{name, command, ...}]
          ↓
MCPManager: starts server, discovers tools
          ↓
MCPToolAdapter: wraps MCP tool as callable
          ↓
TOOL_REGISTRY: {"mcp_server/tool_name": adapter}
          ↓
Rest is identical to builtin tools
```

---

## Approval System Status

**Current**: `auto_approve_all=True` (line 661 of orchestrator.py)
- Approval system fully implemented
- Approval system currently disabled for agents
- Can be re-enabled by changing one parameter
- Approval gate only on `execute_code` tool
- 5-minute timeout for human response
- Broadcast to UI via HTTP POST

See: Deep Dive Section 1.3, QUICK_REFERENCE: Approval System

---

## Security Architecture (8 Layers)

1. **Registry allowlist**: Only known tools accessible
2. **Per-agent allowlist**: Optional per-agent filtering
3. **Path traversal**: `_path_is_within()` prevents escape
4. **Network blocking**: `browse_url()` blocks localhost/private IPs
5. **Env sanitization**: `_sanitized_env()` strips credentials
6. **Approval gate**: `execute_code` requires human approval
7. **Resource limits**: 512 MB memory, 30s CPU timeout
8. **Sandbox isolation**: 3 modes (none, subprocess, docker)

See: QUICK_REFERENCE: Security Layers, Deep Dive Section 11

---

## MCP Integration Phases

| Phase | Component | Lines | Effort | Status |
|-------|-----------|-------|--------|--------|
| 1 | MCPToolAdapter | ~50 | Low | Planned |
| 2 | MCPManager | ~150 | Medium | Planned |
| 3 | Registry population | ~5 | Trivial | Planned |
| 4 | Config schema | ~15 | Trivial | Planned |
| 5 | Plan schema | ~20 | Low | Planned |
| 6 | Validation | ~10 | Low | Planned |
| 7 | Orchestrator | ~5 | Trivial | Planned |

**Total**: ~255 lines of new code, 100% backward compatible

See: MCP_Integration_Roadmap.md all sections

---

## Testing Checklist

- [ ] MCPManager discovers tools from config
- [ ] MCPToolAdapter invokes tools correctly
- [ ] TOOL_REGISTRY includes MCP tools
- [ ] get_tools() resolves MCP tool names
- [ ] execute_tool_call() works with MCP tools
- [ ] Plan parsing accepts server/tool_name syntax
- [ ] Validation rejects invalid tool names
- [ ] Existing plans work unchanged
- [ ] LLM binding works with MCP tools
- [ ] Tool results appear in ToolMessage

See: MCP_Integration_Roadmap.md Section 6

---

## Related Files (Read-Only)

Key files for reference (not in this analysis):

- `/home/localuser/projects/quarm/tools.py` (457 lines)
- `/home/localuser/projects/quarm/orchestrator.py` (1585 lines)
- `/home/localuser/projects/quarm/generate_plan.py` (378 lines)
- `/home/localuser/projects/quarm/config.json` (34 lines)
- `/home/localuser/projects/quarm/status_bridge.py` (380 lines)
- `/home/localuser/projects/quarm/validate_plan.py` (142 lines)

---

## Implementation Roadmap

### Week 1: Planning & Setup
- [ ] Review all documents (2 hours)
- [ ] Design MCPToolAdapter interface (1 hour)
- [ ] Design MCPManager API (1 hour)
- [ ] Review config schema (30 min)

### Week 2: Core Implementation
- [ ] Implement MCPToolAdapter (2 hours)
- [ ] Implement MCPManager (4 hours)
- [ ] Implement JSON-RPC client (3 hours)
- [ ] Unit tests (2 hours)

### Week 3: Integration
- [ ] Update config schema (1 hour)
- [ ] Update generate_plan.py (2 hours)
- [ ] Update validate_plan.py (1 hour)
- [ ] Integration tests (3 hours)

### Week 4: Testing & Docs
- [ ] Test with GitHub API server (2 hours)
- [ ] Test with filesystem server (2 hours)
- [ ] Update README (1 hour)
- [ ] Test backward compatibility (2 hours)

**Total Effort**: ~30 hours, can be parallelized

---

## Success Criteria

- [ ] Builtin tools work unchanged
- [ ] MCP servers can be configured in config.json
- [ ] Agents can specify MCP tools in plan.md
- [ ] Tool execution works with MCP tools
- [ ] All tool results logged correctly
- [ ] Error handling works for MCP failures
- [ ] Backward compatible (no breaking changes)
- [ ] Security maintained (no new vulnerabilities)

---

## Future Enhancements

After basic MCP integration:

1. **Tool composition**: Chain multiple tools together
2. **Caching**: Cache MCP tool results per session
3. **Metrics**: Track most-used tools
4. **Dynamic discovery**: Auto-find MCP servers
5. **Nested tools**: Tools calling other tools
6. **Versioning**: Multiple tool versions
7. **Fallback chains**: Try alternatives on failure

See: MCP_Integration_Roadmap.md Section 8

---

## Document Maintenance

| Document | Last Updated | Review Frequency |
|----------|--------------|------------------|
| README.md | 2026-04-02 | Quarterly |
| INDEX.md | 2026-04-02 | Quarterly |
| QUICK_REFERENCE.md | 2026-04-02 | When tools.py changes |
| Deep Dive | 2026-04-02 | When orchestrator.py changes |
| MCP Roadmap | 2026-04-02 | During implementation |

---

## Contact & Questions

All line numbers and code references are accurate as of 2026-04-02.
For questions about specific sections, refer to the document table of contents.

