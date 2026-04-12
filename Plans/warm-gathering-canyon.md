# Plan: Agent Persistence, Performance Grading & Portable Export

## Context

NORT agents (sub-agents, managers, reviewers) are defined in plan files and auto-registered to `agents/registry.json` on first use. However, the system has several gaps:

1. **Skip-if-exists** — when a plan redefines an existing agent with a better description, the improvement is discarded (orchestrator.py:334)
2. **Coarse performance tracking** — only `runs`, `avg_score`, `total_revisions`. No pass/fail breakdown, no rejection rate
3. **Managers never scored** — `record_agent_performance()` is never called for managers
4. **Reviewer revisions hardcoded to 0** — line 1000: `record_agent_performance("reviewers", ..., 0)`
5. **No portable export** — can't export a single agent as a standalone file usable in another NORT instance or as a Claude Code agent (`.claude/agents/*.md`)

Goal: agents should persist, improve over time, be graded on their actual performance, and be exportable in standard formats.

## Implementation

### Change 1: Enhanced Performance Fields — `agent_registry.py`

Add five new fields to `_base()` helper (line ~49):
- `tasks_passed: 0`
- `tasks_failed: 0`
- `tasks_force_accepted: 0`
- `rejection_rate: 0.0`
- `last_task_at: None`

Expand `record_agent_performance()` (lines 443-457) signature:
```python
def record_agent_performance(agent_type, name, score, revisions=0, verdict="PASS", force_accepted=False):
```
Body updates outcome counters and computes `rejection_rate = (failed + force_accepted) / total_outcomes`.

Update `format_agent_catalog()` (line 507) to include rejection rate in score_info.

Update `check_earned_tolerance()` (line 712) to gate on `rejection_rate < 0.3`.

### Change 2: Merge-on-Reuse — `agent_registry.py` + `orchestrator.py`

New function `merge_agent_from_plan(agent_type, spec)` in `agent_registry.py`:
- If incoming description is longer than existing → update description, title, tools
- Merge tags (union, not replace)
- Never overwrite performance data or version history
- Uses existing `update_agent()` which auto-snapshots versions

Replace skip-if-exists block in `orchestrator.py` (lines 329-348):
```python
if not get_agent("sub_agents", d["name"]):
    create_agent(...)
else:
    merge_agent_from_plan(...)
```
Same pattern for managers and custom reviewers.

### Change 3: Manager Scoring — `orchestrator.py`

After the sub-agent scoring at line 848, add:
```python
record_agent_performance("managers", manager["name"], score, 0, verdict)
```
This records each manager review as a "run" with their verdict. Gives us usage counts + score distributions for managers.

### Change 4: Fix Reviewer & Sub-Agent Scoring — `orchestrator.py`

**Line 1000** — pass task's `revision_count` and verdict instead of hardcoded 0:
```python
record_agent_performance("reviewers", reviewer["name"], score,
    revisions=task.get("revision_count", 0),
    verdict="PASS" if verdict == "PASS" else "FAIL")
```

**Line 848** — pass verdict to sub-agent scoring:
```python
record_agent_performance("sub_agents", task["agent"], score,
    revisions=task.get("revision_count", 0), verdict=verdict)
```

**Force-accept paths** (lines 782-784 manager max-revisions, lines 1021-1025 panel) — record with `force_accepted=True`:
```python
record_agent_performance("sub_agents", task["agent"],
    score=task.get("last_score", 5), revisions=rev,
    verdict="FAIL", force_accepted=True)
```

### Change 5: Single-Agent Export — `agent_registry.py` + `serve.py`

**Two new functions in `agent_registry.py`:**

`export_single_agent(agent_type, name)` → self-contained JSON dict:
```json
{
  "nort_agent_export": true,
  "version": 1,
  "agent_type": "sub_agents",
  "agent": { ...all fields except versions... }
}
```

`export_agent_as_claude_code(agent_type, name)` → markdown file compatible with `.claude/agents/`:
```markdown
---
name: GeneralDeveloper
description: Versatile full-stack developer...
model: sonnet
permissions:
  allow:
    - "Bash(*)"
    - "Edit(*)"
    - "Read(*)"
---
# General Developer
Versatile full-stack developer. Handles any coding task.
## Performance History (from NORT registry)
- Runs: 45, Score: 7.8/10, Rejection rate: 12%
```

**Two new API endpoints in `serve.py`:**
- `GET /api/agents/{type}/{name}/export?format=json|claude` — download single agent
- `POST /api/agents/import-single` — import a single exported agent

### Change 6: Specialization Data in Catalog — `agent_registry.py`

Enhance `format_agent_catalog()` to pull top specialization tags from `specialization.py`:
```
- **ui_artist**: UI Artist -- Creates beautiful interfaces [score: 8.2, runs: 12, rej: 8%]
  tags: frontend, ui, design
  strengths: frontend(8.5), css(8.1), responsive(7.9)
```

This enriches plan generation with per-domain competency signals.

## Files Modified

| File | Changes |
|------|---------|
| `agent_registry.py` | New performance fields, expanded `record_agent_performance()`, `merge_agent_from_plan()`, `export_single_agent()`, `export_agent_as_claude_code()`, enhanced catalog |
| `orchestrator.py` | Merge-on-reuse (replace skip-if-exists), manager scoring, fix reviewer/sub-agent scoring calls, force-accept tracking |
| `serve.py` | Two new export/import endpoints |
| `tests/conftest.py` | Add stubs for new functions |

## Verification

1. Run existing tests — 78 should still pass (backward-compatible signature change)
2. Run a plan with an agent that already exists in registry with a shorter description → verify description is updated
3. After a plan completes, check `agents/registry.json` — verify managers have `runs > 0`
4. After a task gets force-accepted, verify `tasks_force_accepted` increments
5. `curl localhost:8000/api/agents/sub_agents/general_developer/export` → verify JSON export
6. `curl localhost:8000/api/agents/sub_agents/general_developer/export?format=claude` → verify markdown output is valid Claude Code agent format
7. Check `format_agent_catalog()` output includes rejection rates and specialization strengths
