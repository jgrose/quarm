# Plan: Add Agent Performance Stats to NORT Dashboard

## Objective

Enhance the existing agents panel in the NORT dashboard to show per-agent performance statistics (score bars, rejection rate, task breakdown, last-used timestamps, sort controls) using data already available from `/api/agents`.

## Analysis

### Current State
- **agents.html** (`templates/components/panels/agents.html`) -- 38 lines. Glass-card overlay with tabs (SUB-AGENTS, MANAGERS, REVIEWERS, TEAMS), plus export/import/create buttons and a `#agentsList` container.
- **panels.js** -- `renderAgentsList()` at line 1456 already renders each agent as a compact row with: name, badges (BUILTIN, RETIRED, LOW PERF, TRUSTED), avg_score + runs count, description, tags.
- **agent_registry.py** -- `list_agents()` returns all fields including: `runs`, `avg_score`, `total_revisions`, `tasks_passed`, `tasks_failed`, `tasks_force_accepted`, `rejection_rate`, `last_task_at`.
- **colors.js** -- Palette: `#66ffaa` (green/done), `#ffbb44` (yellow/review), `#ff5566` (red/failed), `#66ccff` (blue/holo), `#cc88ff` (purple/specialist).
- **base.css** -- Agent panel styles at section 23 (line 2578+). Uses `.agent-item`, `.agent-item-header`, `.agent-item-name`, `.agent-item-score`, `.agent-item-desc`, `.agent-item-tags`, `.agent-tag`.

### Data Available from API
Each agent object from `/api/agents` includes:
- `runs` (int) -- number of task executions
- `avg_score` (float) -- average quality score
- `tasks_passed` (int) -- tasks that passed review
- `tasks_failed` (int) -- tasks that failed review
- `tasks_force_accepted` (int) -- tasks force-accepted after MAX_REVISIONS
- `rejection_rate` (float, 0-1) -- (failed + forced) / total
- `last_task_at` (ISO timestamp or null) -- when agent last ran

All performance fields already exist -- no new API endpoints needed.

### Key Design Constraint
Enhance the existing `renderAgentsList()` function. Do NOT create a new panel. Keep the compact, holographic aesthetic. The agent list already shows basic score info -- we're extending each row with a small performance stats section.

## Changes

### 1. panels.js -- Enhance `renderAgentsList()` (primary change)

**Add sort state variable** near line 1410:
```javascript
var _agentsSortBy = 'score'; // 'score', 'runs', 'rejection'
```

**Add sort toggle rendering** in `renderAgentsList()` -- insert sort control bar above the agent list items, after filtering.

**Add sort logic** -- sort the `filtered` array based on `_agentsSortBy`:
- `'score'` -- sort descending by `avg_score`
- `'runs'` -- sort descending by `runs`
- `'rejection'` -- sort ascending by `rejection_rate` (best first)

**Enhance each agent row** -- after the existing description + tags, add a compact performance stats section:

For each agent with `runs > 0`:
1. **Score bar** -- a thin horizontal bar (width proportional to avg_score/10), colored green (>7), yellow (5-7), red (<5), with glow effect
2. **Rejection rate** -- small text with percentage, colored green (<20%), yellow (20-40%), red (>40%)
3. **Task breakdown mini-bar** -- a small stacked horizontal bar showing passed (green), failed (red), force-accepted (orange) proportionally
4. **Last used** -- relative timestamp ("2h ago", "3d ago") using `last_task_at`

For agents with `runs === 0`: show "No runs yet" in muted text.

**Add helper function** `_relativeTime(isoString)` -- converts ISO timestamp to relative human-readable string.

**Add helper function** `_renderPerfStats(agent)` -- returns HTML string for the performance stats section.

**Add function** `setAgentsSort(sortBy)` -- updates `_agentsSortBy` and re-renders.

### 2. agents.html -- Add sort controls to header

Add a small sort control bar inside the `agents-header-actions` or as a sub-bar below the tabs. Three small buttons: SCORE, RUNS, REJECT -- clicking each calls `setAgentsSort()`.

### 3. base.css -- Add performance stats styles

New CSS classes (compact, fits existing aesthetic):
- `.agent-perf` -- container for the stats row (flexbox, small padding, subtle top border)
- `.agent-perf-bar` -- thin bar container (4px height, dark background, border-radius)
- `.agent-perf-bar-fill` -- the colored fill inside the bar (with glow box-shadow)
- `.agent-perf-breakdown` -- the stacked task bar container
- `.agent-perf-label` -- tiny label text (7px monospace, muted)
- `.agent-perf-value` -- the value text (8px monospace)
- `.agents-sort-bar` -- the sort toggle button bar
- `.agents-sort-btn` -- individual sort button (styled like `.agents-tab` but smaller)
- `.agents-sort-btn.active` -- highlighted state

## Implementation Order

1. Add CSS classes to `base.css` (styles ready before JS renders)
2. Add sort controls to `agents.html` (below tabs)
3. Enhance `panels.js`:
   a. Add `_agentsSortBy` state variable
   b. Add `_relativeTime()` helper
   c. Add `_renderPerfStats()` helper
   d. Add `setAgentsSort()` function
   e. Modify `renderAgentsList()` to sort and render enhanced rows

## Verification

- The existing agent list appearance is preserved (name, badges, score, description, tags)
- Performance stats appear below tags for agents with runs > 0
- Sort buttons work and re-sort the list
- Colors match the holographic palette
- Panel doesn't become too tall or busy -- stats are compact (one additional row per agent)
