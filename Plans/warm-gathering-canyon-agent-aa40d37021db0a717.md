# Plan: Add Comprehensive Tests for agent_registry.py Functions

## Summary

Create `tests/test_agent_registry.py` with comprehensive test coverage for four areas:
1. `merge_agent_from_plan()` - 7 test cases
2. `export_single_agent()` - 4 test cases
3. `export_agent_as_claude_code()` - 6 test cases
4. `record_agent_performance()` expanded kwargs - 6 test cases

Total: 23 test cases.

## Analysis

### Functions Under Test

**`merge_agent_from_plan(agent_type, spec)`** (lines 327-369):
- Returns None if `spec` has no name or agent doesn't exist in registry
- Compares incoming description length against existing - shorter or equal = no merge (returns None)
- When incoming description is longer: updates description, optionally title, tools, and merges tags as sorted union
- Calls `update_agent()` which auto-snapshots versions
- Performance fields (runs, avg_score, etc.) are NOT in the updates dict, so they survive untouched

**`export_single_agent(agent_type, name)`** (lines 802-813):
- Returns None if agent doesn't exist
- Returns dict with `nort_agent_export: True`, `version: 1`, `agent_type`, and `agent` (cleaned of versions)

**`export_agent_as_claude_code(agent_type, name)`** (lines 816-913):
- Returns None if agent doesn't exist
- Builds markdown with YAML frontmatter (---), name (PascalCase), description, model, permissions
- Maps NORT tools to Claude Code permissions via `tool_map`
- Includes performance history section only when runs > 0
- Includes focus_areas and expertise_blend sections when present
- Includes tags section when present

**`record_agent_performance(agent_type, name, score, revisions=0, verdict="PASS", force_accepted=False)`** (lines 493-526):
- Increments runs, computes running avg_score
- When force_accepted=True: increments tasks_force_accepted (regardless of verdict)
- When verdict="PASS" and not force_accepted: increments tasks_passed
- When verdict != "PASS" and not force_accepted: increments tasks_failed
- Computes rejection_rate = (failed + forced) / (passed + failed + forced)
- Sets last_task_at and updated_at
- Backward compatible: calling with just positional (type, name, score) uses defaults (verdict="PASS", force_accepted=False)

### Test Infrastructure Pattern

From existing tests (test_error_recovery.py, test_validation_wiring.py):
- Tests use `unittest.TestCase` classes grouped by feature
- `PROJECT_ROOT` computed from `Path(__file__).resolve().parent.parent`
- Use `tmp_path` (pytest) or `tempfile` for isolation
- Use `setUp/tearDown` for per-test state management

For agent_registry tests:
- Must avoid touching the real `agents/registry.json`
- Strategy: patch `agent_registry.REGISTRY_FILE` to point to a temp dir, call `seed_registry()` to initialize, test, cleanup
- Import the real `agent_registry` module directly (bypassing conftest stub) using `importlib` force-load pattern (same as `_load_real_checkpoint()` and `_load_real_bridge()` in existing tests)

## Implementation Plan

### Step 1: Create `tests/test_agent_registry.py`

Structure:
```
def _load_real_registry():
    """Force-load the real agent_registry module."""

class TestMergeAgentFromPlan(unittest.TestCase):
    def setUp(self): ...  # tmp dir, patch REGISTRY_FILE, seed
    def tearDown(self): ... # cleanup tmp dir
    
    test_shorter_description_no_merge
    test_longer_description_updates
    test_new_tags_merged_as_union
    test_tools_updated_when_different
    test_title_updates_when_different
    test_performance_fields_not_overwritten
    test_nonexistent_agent_returns_none

class TestExportSingleAgent(unittest.TestCase):
    def setUp(self): ...
    def tearDown(self): ...
    
    test_valid_agent_returns_export_dict
    test_agent_has_correct_type_field
    test_versions_excluded_from_export
    test_nonexistent_agent_returns_none

class TestExportAgentAsClaudeCode(unittest.TestCase):
    def setUp(self): ...
    def tearDown(self): ...
    
    test_valid_sub_agent_starts_with_frontmatter
    test_contains_yaml_frontmatter_fields
    test_tools_mapped_correctly
    test_performance_section_when_runs_gt_0
    test_no_performance_section_when_runs_eq_0
    test_nonexistent_agent_returns_none

class TestRecordAgentPerformanceExpanded(unittest.TestCase):
    def setUp(self): ...
    def tearDown(self): ...
    
    test_verdict_pass_increments_tasks_passed
    test_verdict_fail_increments_tasks_failed
    test_force_accepted_increments_counter
    test_rejection_rate_computed_correctly
    test_last_task_at_is_set
    test_backward_compatible_positional_only
```

### Step 2: Run tests
```bash
python3 -m pytest tests/test_agent_registry.py -x -v
```

### Step 3: Run full test suite
```bash
python3 -m pytest tests/ -x -v --ignore=tests/test_smoke.py
```

### Step 4: Commit when all pass

## Key Design Decisions

1. **Force-load real module**: Use `importlib.util.spec_from_file_location` to bypass conftest's stub, same pattern as `_load_real_checkpoint()` and `_load_real_bridge()`.

2. **Temp directory isolation**: Each test class creates a temp dir, patches `REGISTRY_FILE` to `tmpdir/registry.json`, seeds the registry, tests, then removes.

3. **setUp/tearDown**: Each test class manages its own temp state. No shared state between test methods.

4. **No mocking of agent_registry internals**: Test the real functions end-to-end against a temp registry file. This follows the existing pattern of integration-style testing.

5. **Assertions**: Direct value assertions on returned dicts/strings. No broad "is truthy" checks - test specific field values.
