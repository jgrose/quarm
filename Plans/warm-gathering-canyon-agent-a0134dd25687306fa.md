# Plan: Add Agent Performance Stats to Agents Panel

## Summary
Enhance the NORT dashboard agents panel with inline performance statistics (score bar, rejection rate, task outcomes, last active time) and a sort control bar to sort agents by score, runs, or rejection rate.

## Analysis

### Current State
- `agents.html` has a tabs bar (SUB-AGENTS, MANAGERS, REVIEWERS, TEAMS) followed by the agents list div
- `renderAgentsList()` in `panels.js` (line 1447) filters agents by tab and renders each with name, badges, score summary, description, and tags
- Agent data objects contain: `runs`, `avg_score`, `rejection_rate`, `tasks_passed`, `tasks_failed`, `tasks_force_accepted`, `last_task_at`
- CSS styles for agent items exist at lines 2619-2671 in `base.css`

### Available Performance Fields (from `agent_registry.py`)
- `runs` (int) - total task runs
- `avg_score` (float) - running average score 0-10
- `rejection_rate` (float, 0-1) - proportion of failed + force-accepted tasks
- `tasks_passed` (int) - count of PASS verdicts
- `tasks_failed` (int) - count of FAIL verdicts  
- `tasks_force_accepted` (int) - count of force-accepted tasks
- `last_task_at` (ISO string or null) - when agent last ran

## Changes

### 1. `templates/components/panels/agents.html` (1 insertion)

**After** the `.agents-tabs` div (line 19), **before** the `#agentsList` div (line 21), insert the sort control bar:

```html
<div class="agent-sort-bar">
  <span class="agent-sort-label">SORT:</span>
  <button class="agent-sort-btn active" onclick="setAgentsSort('score')">SCORE</button>
  <button class="agent-sort-btn" onclick="setAgentsSort('runs')">RUNS</button>
  <button class="agent-sort-btn" onclick="setAgentsSort('rejection')">REJECT</button>
</div>
```

### 2. `templates/scripts/panels.js` (3 additions)

**A) Module-level variable** - Add after `var _showRetired = false;` (line 1404):
```js
var _agentsSortBy = 'score';
```

**B) New functions** - Add after `toggleShowRetired()` (after line 1445), before `renderAgentsList()`:

1. `setAgentsSort(key)` - updates `_agentsSortBy`, toggles `.active` on sort buttons, calls `renderAgentsList()`
2. `_relativeTime(isoString)` - returns human-readable relative time ("2h ago", "3d ago", "never")

**C) Inside `renderAgentsList()`** - Two modifications:

1. **After filtering** (after line 1464, the retired filter), add sort logic:
   - Sort `filtered` array based on `_agentsSortBy`:
     - `'score'` -> sort by `avg_score` descending
     - `'runs'` -> sort by `runs` descending
     - `'rejection'` -> sort by `rejection_rate` descending

2. **After the tags section** (after line 1504, before the closing `</div>` of agent-item), append the perf stats HTML:
   - If `a.runs > 0`: render `.agent-perf` div with:
     - SCORE row: label + progress bar (width = avg_score/10*100%) + color-coded value
     - REJECT row: label + color-coded percentage
     - TASKS row: label + checkmark/pass + x/fail + lightning/force-accepted counts
     - LAST row: label + relative time from `last_task_at`
   - If `a.runs == 0`: render `.agent-perf-empty` with "No runs yet"

**Color thresholds:**
- Score: green (#66ffaa) >= 7, yellow (#ffbb44) 5-7, red (#ff5566) < 5
- Rejection: green (#66ffaa) < 20%, yellow (#ffbb44) 20-40%, red (#ff5566) > 40%
- Score bar width: `(avg_score / 10) * 100`%

### 3. `templates/styles/base.css` (1 insertion)

**After** `.agent-tag` block (after line 2671), insert all new CSS classes:

```css
.agent-sort-bar { display: flex; align-items: center; gap: 6px; padding: 4px 12px; border-bottom: 1px solid rgba(100,200,255,0.08); }
.agent-sort-label { font-size: 8px; color: var(--text-muted); letter-spacing: 1px; }
.agent-sort-btn { background: none; border: 1px solid rgba(100,200,255,0.15); color: var(--text-dim); font: 9px monospace; padding: 2px 8px; cursor: pointer; border-radius: 2px; }
.agent-sort-btn.active { background: rgba(100,200,255,0.12); color: var(--text-primary); border-color: rgba(100,200,255,0.3); }
.agent-perf { padding: 4px 0 0; margin-top: 4px; border-top: 1px solid rgba(100,200,255,0.06); }
.agent-perf-row { display: flex; align-items: center; gap: 6px; font-size: 9px; padding: 1px 0; }
.agent-perf-label { width: 42px; color: var(--text-muted); font-size: 8px; letter-spacing: 0.5px; }
.agent-perf-bar { flex: 1; height: 4px; background: rgba(100,200,255,0.08); border-radius: 2px; overflow: hidden; }
.agent-perf-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
.agent-perf-val { font-size: 9px; min-width: 32px; text-align: right; }
.agent-perf-empty { font-size: 8px; color: var(--text-muted); padding: 4px 0; font-style: italic; }
```

## File Inventory
| File | Action |
|------|--------|
| `templates/components/panels/agents.html` | Insert sort bar after tabs |
| `templates/scripts/panels.js` | Add variable, 2 functions, modify renderAgentsList() |
| `templates/styles/base.css` | Add 11 new CSS rules |

## Commit
Single commit: "Add agent performance stats and sort controls to agents panel"
