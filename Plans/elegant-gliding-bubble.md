# Tier 2 Features: Cost Tracking + Dependency Visualization

## Context

After completing Tier 1 stability work (30 tests, FPS profiling, WebSocket hardening, render optimizations), the user wants to tackle Tier 2 high-priority roadmap items. Three were selected:

1. **Cost tracking per agent** — dashboard panel showing overall plan token/cost AND per-agent breakdown
2. **Task dependency visualization** — visual connections between dependent tasks, blocked indicators
3. **Output browser + zip download** — **already implemented** (discovered during exploration: `output_browser.html`, `/api/artifacts/{plan_id}`, `/api/artifacts/{plan_id}/download` all exist)

This plan covers Features 1 and 2 only. They can be built in parallel by two agents with minimal file conflicts.

---

## Feature 1: Cost Tracking Dashboard Panel

**Agent**: Cost Agent  
**Goal**: Add a "COST TRACKER" panel with live per-plan and per-agent token/cost breakdown

### What Already Exists
- `/api/analytics/costs` endpoint (`serve.py:660`) returns `{ total_tokens, recent_runs, by_agent, by_model }`
- `tracking.py:110-130` — `get_cost_analytics()` with SQLite aggregation by agent
- `task_scores` table has `run_id, task_id, agent, tokens, model` columns
- Status bridge transmits `task_tokens` per task + `tokens_used` run total via WebSocket
- `draw_cost.js` draws canvas-based cost pills (existing)
- `_sessions[sid].data` in frontend has all live cost data

### Backend Changes

**`tracking.py`** — add `get_run_cost_breakdown(run_id)` function (after line 131):
- Query `task_scores` grouped by agent and model for a specific run
- Return `{ run: {...}, by_agent: [...], by_model: [...], by_task: [...] }`

**`serve.py`** — add endpoint (after line 663):
- `GET /api/analytics/costs/{run_id}` → calls `get_run_cost_breakdown()`

### Frontend Changes

**`templates/components/panels/cost_panel.html`** — NEW file (self-contained HTML + CSS + JS):
- Glass-card panel, fixed position top-right, 380px wide
- Two tabs: **LIVE** (from WebSocket `_sessions` data) and **HISTORY** (from API)
- **LIVE tab**: Summary stats (total tokens, est. cost, task count) + per-agent bar chart aggregated from `data.tasks[].task_tokens` grouped by `data.tasks[].agent`
- **HISTORY tab**: Recent runs list (clickable), drill into per-run agent/model breakdown via `/api/analytics/costs/{run_id}`
- Auto-refreshes every 2s while visible
- `_estimateCost(tokens)` helper using configurable rate per million tokens

**`templates/base.html`** — add `{% include "components/panels/cost_panel.html" %}` (after line 38, after perf_panel)

**`templates/components/top_bar.html`** — add COSTS button (after LEDGER at line 11):
```html
<button onclick="toggleCostPanel()">COSTS</button>
```

**`templates/scripts/init.js`** — add ESC dismiss for `costPanelOverlay` (in ESC block, line 87) + keyboard shortcut `$` (after line 106):
```javascript
if (e.key === '$' && ...) toggleCostPanel();
```

### Files Touched
| File | Action |
|------|--------|
| `tracking.py` | Add 1 function (~25 lines) |
| `serve.py` | Add 1 endpoint (~4 lines) |
| `templates/components/panels/cost_panel.html` | CREATE (~200 lines) |
| `templates/base.html` | Add 1 include line |
| `templates/components/top_bar.html` | Add 1 button |
| `templates/scripts/init.js` | Add ESC handler + shortcut (~5 lines) |

---

## Feature 2: Task Dependency Visualization

**Agent**: Deps Agent  
**Goal**: Draw dependency lines between programs and show blocked task indicators

### What Already Exists
- `node.dependsOn` already captured from WebSocket data (`websocket.js:209,368`)
- Task flow arrows system in `draw_locations.js:1220-1281` (animated dashed lines between buildings)
- Bezier edge drawing in `draw_edges.js:25-87` (tapered curves with control points)
- Active building indicators in `draw_locations.js:1132-1159` (pulsing ring + badge)
- `ambientPrograms[i].assignedTask = { id, status, title }` links programs to tasks
- **No backend changes needed** — all data already transmitted

### Design Decisions
- **Lines connect programs to programs** (not buildings), since dependencies are task-to-task and each task maps to an agent/program
- **Lines shown** when a dependency exists AND is unresolved (prerequisite not `done`)
- **Lines fade out** when prerequisite completes
- **Blocked indicator**: lock badge + "BLOCKED" text above programs assigned to tasks with unmet deps
- **New file** `draw_dependencies.js` (not extending draw_locations.js which is 1300+ lines)

### Frontend Changes

**`templates/scripts/draw_dependencies.js`** — NEW file (~180 lines):
- `rebuildDependencyState(tasks)` — called from websocket.js on status updates. Scans tasks for unmet `depends_on`, builds `_depLines[]` and `_blockedTasks{}` maps
- `drawDependencyLines(ctx, time)` — draws animated dashed bezier curves between programs. Amber for active deps, green fade-out for resolved. Uses `_computeCP` pattern from draw_edges.js
- `drawBlockedIndicators(ctx, time)` — draws lock badge + "BLOCKED" text above blocked programs with bob animation
- `_findProgramForTask(taskId)` — scans `ambientPrograms` for matching `assignedTask.id`

**`templates/scripts/websocket.js`** — add `rebuildDependencyState(tasks)` call:
- After `routeProgramsToTasks(tasks)` at line 273
- After session switch task processing at line 368

**`templates/scripts/render.js`** — add draw calls in world-space section (after programs, ~line 170):
```javascript
if (config.dependencies && typeof drawDependencyLines === 'function') {
  drawDependencyLines(ctx, currentTime);
}
if (config.dependencies && typeof drawBlockedIndicators === 'function') {
  drawBlockedIndicators(ctx, currentTime);
}
```

**`templates/scripts/nodes.js`** — add `dependencies: true` to config (after `activeIndicators` at line 49)

**`templates/base.html`** — add `{% include "scripts/draw_dependencies.js" %}` (after draw_cost.js at line 54)

**`templates/scripts/init.js`** — add keyboard shortcut `d` (after line 106):
```javascript
if (e.key === 'd' && ...) config.dependencies = !config.dependencies;
```

**`templates/components/panels/config.html`** — add DEPENDENCIES toggle row in DISPLAY tab

### Files Touched
| File | Action |
|------|--------|
| `templates/scripts/draw_dependencies.js` | CREATE (~180 lines) |
| `templates/scripts/websocket.js` | Add 2 function calls (~4 lines) |
| `templates/scripts/render.js` | Add draw calls (~6 lines) |
| `templates/scripts/nodes.js` | Add 1 config flag |
| `templates/base.html` | Add 1 include line |
| `templates/scripts/init.js` | Add 1 shortcut (~3 lines) |
| `templates/components/panels/config.html` | Add 1 toggle row (~4 lines) |

---

## File Conflict Analysis

| File | Cost Agent | Deps Agent | Risk |
|------|-----------|-----------|------|
| `templates/base.html` | Add panel include (HTML section) | Add script include (JS section) | **NONE** — different sections |
| `templates/scripts/init.js` | Add ESC + `$` shortcut | Add `d` shortcut | **LOW** — adjacent lines |
| `templates/scripts/nodes.js` | — | Add config flag | **NONE** |
| `templates/scripts/render.js` | — | Add draw calls | **NONE** |
| `templates/scripts/websocket.js` | — | Add rebuildDependencyState | **NONE** |
| `tracking.py` | Add 1 function | — | **NONE** |
| `serve.py` | Add 1 endpoint | — | **NONE** |

**Verdict**: Safe to parallelize in worktrees. The only shared file is `init.js` with adjacent line additions — trivial merge.

---

## Execution Plan

**Single wave, 2 agents in parallel** (worktree isolation):

| Agent | Type | Files | Isolation |
|-------|------|-------|-----------|
| Cost Agent | Engineer | `tracking.py`, `serve.py`, `cost_panel.html` (new), `base.html`, `top_bar.html`, `init.js` | worktree |
| Deps Agent | Engineer | `draw_dependencies.js` (new), `websocket.js`, `render.js`, `nodes.js`, `base.html`, `init.js`, `config.html` | worktree |

### After merge
1. Run `python3 -m pytest tests/ -v --ignore=tests/test_smoke.py` — verify no regressions
2. Start `serve.py`, open dashboard
3. Verify cost panel: press `$`, see LIVE and HISTORY tabs, check per-agent bars
4. Verify dependency viz: press `d`, run a plan with dependencies, see lines between programs + blocked indicators
5. Update `roadmap.md` — check off cost tracking, dependency visualization, output browser, and zip download

---

## Roadmap Items to Mark Done After This

- [x] Cost tracking per agent
- [x] Task dependency visualization
- [x] Output browser in dashboard (already implemented)
- [x] Download output as zip (already implemented)
