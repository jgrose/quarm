# Plan: Comprehensive Tests for agent_registry.py

## Summary

Create `tests/test_agent_registry.py` with 23 test cases across 4 test classes covering `merge_agent_from_plan`, `export_single_agent`, `export_agent_as_claude_code`, and `record_agent_performance`.

## Analysis

### Module Under Test
`agent_registry.py` is a persistent agent registry with JSON file storage. Key functions to test:

1. **`merge_agent_from_plan(agent_type, spec)`** - Merges plan agent definitions into existing registry entries. Only merges if incoming description is longer. Merges tags as union. Updates title/tools if different. Returns updated agent or None.

2. **`export_single_agent(agent_type, name)`** - Returns a self-contained export dict with `nort_agent_export: True`, `version: 1`, `agent_type`, and the agent data (minus `versions` key).

3. **`export_agent_as_claude_code(agent_type, name)`** - Produces a markdown string with YAML frontmatter, permissions mapping, performance history, tags, and origin note.

4. **`record_agent_performance(agent_type, name, score, revisions, verdict, force_accepted)`** - Updates running average score, revision count, outcome counters (passed/failed/force_accepted), rejection_rate, and last_task_at.

### Test Infrastructure Pattern
Following the established pattern from `test_status_bridge.py` and `test_tolerance.py`:
- Force-reload the real module at top of file (bypassing conftest stub)
- Use `unittest.TestCase` (consistent with existing tests)
- Use `tempfile.mkdtemp()` for isolated registry file
- In `setUp`: redirect `REGISTRY_FILE` to temp dir, call `seed_registry()`
- In `tearDown`: restore original `REGISTRY_FILE`, remove temp dir

### Test Cases (23 total)

#### TestMergeAgentFromPlan (7 tests)
1. **Shorter description returns None** - Spec with shorter desc than existing -> None, desc unchanged
2. **Longer description updates** - Spec with longer desc -> returns updated agent with new desc
3. **Tags merged as union** - Existing tags preserved, new tags added
4. **Tools updated when different** - New tools list replaces old
5. **Title updated when different** - New title replaces old
6. **Performance fields not overwritten** - runs, avg_score stay intact after merge
7. **Non-existent agent returns None** - Agent not in registry -> None

#### TestExportSingleAgent (4 tests)
1. **Valid agent returns dict with markers** - Contains `nort_agent_export: True` and `version: 1`
2. **Correct agent_type field** - Export dict has correct `agent_type` value
3. **Versions key excluded** - The `versions` list is stripped from exported agent
4. **Non-existent agent returns None** - Missing agent -> None

#### TestExportAgentAsClaudeCode (6 tests)
1. **Starts with YAML frontmatter** - Output string starts with `---`
2. **Contains permissions section** - String includes `permissions:`
3. **write_file maps to Edit and Write** - Both "Edit" and "Write" appear in output
4. **read_file maps to Read and Grep** - Both "Read" and "Grep" appear in output
5. **Agent with runs>0 has Performance History** - Section present when runs > 0
6. **Non-existent agent returns None** - Missing agent -> None

#### TestRecordAgentPerformanceExpanded (6 tests)
1. **PASS increments tasks_passed** - verdict="PASS" -> tasks_passed goes from 0 to 1
2. **FAIL increments tasks_failed** - verdict="FAIL" -> tasks_failed goes from 0 to 1
3. **force_accepted increments tasks_force_accepted** - force_accepted=True -> counter increments
4. **Rejection rate correct after mixed outcomes** - 2 pass, 1 fail, 1 force -> rejection_rate = 0.5
5. **last_task_at is set** - After recording, last_task_at is not None
6. **Backward compatible** - Calling with just (type, name, score) works without error

## Implementation Steps

1. Create `/home/localuser/projects/quarm/tests/test_agent_registry.py`
2. At top of file: force-remove the conftest stub from `sys.modules`, import and reload real `agent_registry`
3. Create base `_RegistryTestCase` or just repeat setUp/tearDown in each class (following existing pattern of per-class setUp)
4. Implement all 23 test methods
5. Run `python3 -m pytest tests/ -x -v --ignore=tests/test_smoke.py` from project root
6. Fix any failures
7. Commit with no AI attribution

## File to Create

- `/home/localuser/projects/quarm/tests/test_agent_registry.py`

## Verification

```bash
cd /home/localuser/projects/quarm
python3 -m pytest tests/test_agent_registry.py -x -v
```

All 23 tests should pass.
