# Plan: Replace Raw Manager Scoring with Specialist-Alignment Scoring

## Problem

In `manager_review_node`, `record_agent_performance("managers", manager["name"], score, 0, verdict)` records the score the manager **gave** to the sub-agent's work -- not a measure of the manager's own quality. This is misleading. A manager who gives a generous 9/10 to bad work would appear to be a "high performing manager" when they're actually doing a poor job.

The fix: score managers based on whether specialists **agree** with the manager's verdict. If the manager passes work and specialists also pass it, the manager made a good call. If the manager passes work and specialists flag it, the manager missed issues.

## Code Locations (orchestrator.py)

| Location | Lines | What's There Now |
|---|---|---|
| `manager_review_node` - misleading scoring | 868-872 | `record_agent_performance("managers", manager["name"], score, 0, verdict)` -- records raw manager score |
| `manager_review_node` - force-accept path | 793-801 | `record_agent_performance("managers", ...)` with force_accepted=True -- KEEP this one |
| `specialist_review_node` - after `any_flags` | 1041 | No manager scoring currently exists here |
| `specialist_review_node` - no-reviewers early return | 961-977 | No manager scoring currently exists here |
| `find_mgr` helper | 416 | `find_mgr(agent, mgrs)` -- takes agent name, returns manager dict |

## Changes

### 1. REMOVE: Misleading manager scoring in `manager_review_node` (lines 868-872)

Remove the entire try/except block:
```python
    try:
        from agent_registry import record_agent_performance
        record_agent_performance("managers", manager["name"], score, 0, verdict)
    except Exception as e:
        log_event(f"[ERROR] Failed to record manager performance for {manager['name']}: {e}")
```

Keep the sub-agent scoring block above it (lines 861-866).

### 2. KEEP: Force-accept path in `manager_review_node` (lines 793-801)

The force-accept path (when `rev >= MAX_REVISIONS`) already records both sub-agent and manager with `force_accepted=True`. This stays as-is.

### 3. ADD: Alignment-based scoring in `specialist_review_node` main completion path (after line 1058)

After `_auto_ingest(task, results)` and before the state return, add manager alignment scoring:

```python
    mgr = find_mgr(task["agent"], state.get("managers", []))
    if mgr:
        try:
            mgr_score = 7 if task.get("revision_count", 0) > 0 else (4 if any_flags else 9)
            mgr_verdict = "FAIL" if any_flags and task.get("revision_count", 0) == 0 else "PASS"
            record_agent_performance("managers", mgr["name"], mgr_score, 0, mgr_verdict)
        except Exception:
            pass
```

Scoring logic:
- `any_flags=False`, `revision_count=0`: Manager passed it, specialists agreed -> score=9, PASS (good judgment)
- `any_flags=True`, `revision_count=0`: Manager passed it, specialists flagged issues -> score=4, FAIL (missed problems)
- `revision_count > 0`: Manager previously failed it (caused revision), showing appropriate skepticism -> score=7, PASS

Note: `record_agent_performance` is already imported in this function's scope at line 1028. But since it's inside a try/except block there, we need our own import or we can rely on the existing one being available after the loop. To be safe, add the import inside our new try block.

### 4. ADD: Manager scoring in no-reviewers early return path (after line 974)

When no reviewers are assigned, the manager's PASS is the final word. Score the manager positively:

```python
    mgr = find_mgr(task["agent"], state.get("managers", []))
    if mgr:
        try:
            from agent_registry import record_agent_performance
            record_agent_performance("managers", mgr["name"], 9, 0, "PASS")
        except Exception:
            pass
```

## Test Plan

The existing test infrastructure in `conftest.py` already stubs `record_agent_performance` as a MagicMock. Tests to verify:

1. **manager_review_node no longer records manager performance** -- call manager_review_node with PASS verdict, assert `record_agent_performance` was NOT called with `"managers"` as first arg (only with `"sub_agents"`)
2. **specialist_review_node records alignment-based manager score on clean pass** -- no flags, revision_count=0, verify score=9, verdict="PASS"
3. **specialist_review_node records alignment-based manager score on flagged** -- flags present, revision_count=0, verify score=4, verdict="FAIL"
4. **specialist_review_node records alignment-based manager score after revision** -- revision_count > 0, verify score=7, verdict="PASS"
5. **specialist_review_node no-reviewers path records manager score=9** -- empty rev_list, verify score=9, verdict="PASS"

## Execution Order

1. Write tests first (TDD -- red phase)
2. Run tests to confirm they fail
3. Make code changes in orchestrator.py
4. Run tests to confirm they pass (green phase)
5. Run full test suite: `python3 -m pytest tests/ -x -v --ignore=tests/test_smoke.py`
6. Commit (no AI attribution)
