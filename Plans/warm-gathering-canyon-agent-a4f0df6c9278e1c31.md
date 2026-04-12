# Plan: Fix Manager Scoring to Track Specialist-Alignment

## Problem

In `orchestrator.py` line 868-872, the manager's performance is recorded in `manager_review_node` using the score the manager assigned to the sub-agent's work. This is the wrong signal -- it conflates the manager's judgment quality with the agent's output quality. A manager who gives everything a 5 isn't necessarily performing poorly.

The correct signal: **did the specialist review panel agree with the manager's verdict?**

## Current Code (What Exists)

### manager_review_node (lines 868-872)
```python
try:
    from agent_registry import record_agent_performance
    record_agent_performance("managers", manager["name"], score, 0, verdict)
except Exception as e:
    log_event(f"[ERROR] Failed to record manager performance for {manager['name']}: {e}")
```
This records the raw score the manager gave -- misleading.

### specialist_review_node (lines 938-1061)
- Runs each reviewer, collects verdicts
- `any_flags = any(vd == "FLAG" for vd in verdicts)` (line 1041)
- If any_flags and rev < MAX_REVISIONS: sends back for revision (line 1043)
- Otherwise: task is done (line 1052-1061) -- but no manager alignment scoring here

### find_mgr (line 416-417)
```python
def find_mgr(agent, mgrs):
    return next((m for m in mgrs if agent in m.get("oversees", [])), None)
```
This finds the manager for a given agent by checking the `oversees` list. Available in `specialist_review_node` via `state["managers"]` and `task["agent"]`.

### record_agent_performance (agent_registry.py lines 493-526)
Takes: `agent_type, name, score, revisions=0, verdict="PASS", force_accepted=False`
- verdict="PASS" increments `tasks_passed`
- verdict != "PASS" increments `tasks_failed`
- Already computes `rejection_rate` from passed/failed/forced counters

## Changes

### Step 1: Remove raw manager scoring from manager_review_node

**File:** `orchestrator.py`, lines 868-872

Remove the entire try/except block that calls `record_agent_performance("managers", ...)`. The sub-agent recording (lines 861-866) stays.

### Step 2: Add alignment-based manager scoring at end of specialist_review_node

At the END of `specialist_review_node`, after the final verdict is known (after line 1041 `any_flags` is computed), and BEFORE the return statements:

1. Look up the manager: `manager = find_mgr(task["agent"], state["managers"])`
2. Score based on alignment:
   - **Manager passed + specialists all passed** (not any_flags): `score=9, verdict="PASS"` (good alignment)
   - **Manager passed + specialists flagged** (any_flags): `score=4, verdict="FAIL"` (false positive, missed issues)  
   - **Task is in revision** (rev > 0, meaning manager previously failed it and it came back): `score=7, verdict="PASS"` (appropriate skepticism, caught real issues)
3. Record: `record_agent_performance("managers", manager["name"], alignment_score, 0, alignment_verdict)`

The `tasks_passed` vs `tasks_failed` counters in the existing `record_agent_performance` function already naturally track alignment rate since we pass verdict="PASS" when the manager was right and verdict="FAIL" when wrong. No new `alignment_rate` field is needed -- the existing `rejection_rate` field effectively becomes the misalignment rate for managers.

### Step 3: Keep force-accept path (lines 798-799)

The existing force-accept recording in `manager_review_node` (lines 794-801) stays as-is. When a manager repeatedly fails a task that gets force-accepted, that's valid performance data (`force_accepted=True`).

### Step 4: Handle edge cases in specialist_review_node

- **No manager found** (`find_mgr` returns None): Skip manager alignment recording (just like manager_review_node does at line 781-785)
- **No reviewers** (early return at line 961-977): Also record manager alignment as score=9/PASS since the work was accepted without issues
- **Force-accept after max revisions** (any_flags is True but rev >= MAX_REVISIONS at line 1052): Record score=4/FAIL since specialists disagreed

### Step 5: Handle skip_specialist path in manager_review_node

There's a path where `skip_specialist` is True (lines 876-890) and the task goes directly to done without specialist review. In this case, no specialist alignment is possible, so we should NOT record manager performance here -- or we record a neutral score=7 since we can't verify alignment. Actually, skipping means the manager gave score >= 9 -- this is a high-confidence pass. Best to leave this path without manager recording since there's no specialist signal to align against.

## Detailed Edit Locations

1. **DELETE lines 868-872** in orchestrator.py (the raw manager `record_agent_performance` call in manager_review_node)

2. **ADD after line 1040** (after `any_flags` computation, before the if/else branches) in specialist_review_node:
```python
    # Record manager alignment with specialist panel verdict
    manager = find_mgr(task["agent"], state["managers"])
    if manager:
        rev = task.get("revision_count", 0)
        if any_flags:
            # Manager passed but specialists flagged issues -- poor judgment
            alignment_score, alignment_verdict = 4, "FAIL"
        elif rev > 0:
            # Task was revised (manager previously failed it) -- appropriate skepticism
            alignment_score, alignment_verdict = 7, "PASS"
        else:
            # Manager passed and specialists agreed -- good alignment
            alignment_score, alignment_verdict = 9, "PASS"
        try:
            from agent_registry import record_agent_performance
            record_agent_performance("managers", manager["name"],
                alignment_score, 0, alignment_verdict)
        except Exception:
            pass
```

3. **ADD in the no-reviewers early return** (after line 966, before the sub_agent recording): Similar alignment recording with score=9/PASS since task was accepted.

## Tests

The existing tests mock `record_agent_performance` via conftest.py stub. The changes shouldn't break existing tests since we're just changing where/when the manager recording happens. Verify with `pytest`.

## Risk Assessment

- **Low risk**: The change is purely about when and with what values `record_agent_performance` is called for managers
- **No schema changes**: Using existing function signature and existing counters
- **Backwards compatible**: Registry data format unchanged
- **Existing force-accept path preserved**: No behavioral change there
