# Fix "unknown" tool call cards on page refresh

## Context

When the page is refreshed, rectangular tool call cards appear near agent nodes showing "unknown" instead of the actual tool name. This is caused by a **field name mismatch** between the Python orchestrator and the JavaScript UI.

## Root Cause

**The orchestrator stores tool calls with the key `"tool"`, but the UI reads `tool.name`.**

In [orchestrator.py:606](orchestrator.py#L606):
```python
tool_calls_log.append({"tool": tc_name, "args_preview": ..., "result_preview": ...})
```

In [draw_tools.js:88](templates/scripts/draw_tools.js#L88):
```javascript
var toolName = truncateText(tool.name || 'unknown', 18);
```

Since `tool.name` is always `undefined`, every card falls back to `'unknown'`. The same mismatch exists in [draw_cost.js:63](templates/scripts/draw_cost.js#L63) (`tc.name || 'unknown'`).

Additionally, the data has no `state` or `tokens` properties, so tool cards show no state indicators (running spinner, complete checkmark) and no per-tool token counts.

## Secondary Issue

`drawAllToolCards` renders cards for **all nodes with toolCalls**, including completed tasks. Once a task finishes, its `tool_calls` persist in the cached session state, so on refresh every completed task's cards reappear — stacking up the "unknown" boxes.

## Fix

### 1. Fix field name in orchestrator.py (~line 606)

Change `"tool"` key to `"name"` and add a `"state"` field:

```python
tool_calls_log.append({
    "name": tc_name,
    "state": "complete",
    "args_preview": str(tc_args)[:100],
    "result_preview": result_str[:200],
})
```

### 2. Only render tool cards for active tasks (draw_tools.js)

In `drawAllToolCards` (~line 111), skip nodes whose state is `done`, `failed`, or `pending`:

```javascript
function drawAllToolCards(ctx, time) {
  for (var entry of nodes) {
    var node = entry[1];
    if (!node.toolCalls || node.toolCalls.length === 0) continue;
    if (node.state === 'done' || node.state === 'failed' || node.state === 'pending') continue;
    // ... rest unchanged
  }
}
```

### 3. Clear tool calls on task completion (websocket.js)

In `applyStatus`, when a task transitions to `done` or `failed`, clear its tool calls so they don't persist visually:

In the task loop (~line 101), after setting `node.toolCalls`:
```javascript
if (task.status === 'done' || task.status === 'failed') {
    node.toolCalls = [];
}
```

## Files to Modify

| File | Change |
|------|--------|
| [orchestrator.py:606](orchestrator.py#L606) | Rename `"tool"` → `"name"`, add `"state": "complete"` |
| [templates/scripts/draw_tools.js:111-118](templates/scripts/draw_tools.js#L111-L118) | Skip rendering for done/failed/pending nodes |
| [templates/scripts/websocket.js:101](templates/scripts/websocket.js#L101) | Clear toolCalls for completed tasks |

## Verification

1. Run `python serve.py` and `python orchestrator.py plan.example.md`
2. During execution, verify tool cards show real tool names (e.g., "read_file", "bash") near active agent nodes
3. Verify cards disappear when tasks complete
4. Refresh the page — no "unknown" boxes should appear
5. Check the cost breakdown bar in cost pills still works (uses `tc.name`)
