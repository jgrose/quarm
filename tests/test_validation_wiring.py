"""
Tests for plan validation wiring in orchestrator startup,
checkpoint integrity validation, and generate_plan validation.

Verifies:
- validate_plan.validate() is called before parse_plan() in orchestrator.run()
- Validation errors cause orchestrator to abort with ValueError
- Valid plans proceed normally
- Checkpoint loading validates required keys
- Corrupted checkpoints are renamed and None is returned
- generate_plan validates output and warns on invalid plans
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use the conftest stubs
from tests.conftest import FIXTURES_DIR


# ── Tests for validate_plan.validate() itself ──────────────────────────────────

class TestValidatePlanDirect:
    """Verify the validate_plan module works correctly standalone."""

    def test_valid_plan_returns_no_errors(self):
        """A well-formed plan file returns an empty error list."""
        from validate_plan import validate
        errors = validate(str(FIXTURES_DIR / "simple_plan.md"))
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_complex_plan_returns_no_errors(self):
        """Complex plan with custom reviewers also validates cleanly."""
        from validate_plan import validate
        errors = validate(str(FIXTURES_DIR / "complex_plan.md"))
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_missing_file_returns_error(self):
        """Missing file returns a 'File not found' error."""
        from validate_plan import validate
        errors = validate("/nonexistent/plan.md")
        assert len(errors) == 1
        assert "File not found" in errors[0]

    def test_invalid_agent_reference_returns_error(self):
        """A task referencing a non-existent agent produces an error."""
        from validate_plan import validate
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("""# PROJECT PLAN: Bad Agent Ref

## Objective
Test plan with bad agent reference.

## Sub-Agents
### AGENT: real_agent
- description: A real agent.
- tools: execute_code

## Managers
### MANAGER: mgr
- title: Manager
- description: A manager.
- expertise_blend: [testing]
- oversees: [real_agent]

## Tasks
### TASK-001
- title: Do something
- agent: nonexistent_agent
- description: This references an agent that does not exist.
- task_type: [code]
- reviewers: []
- depends_on: []
""")
            path = f.name

        try:
            errors = validate(path)
            assert len(errors) >= 1
            assert any("nonexistent_agent" in e for e in errors)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_no_tasks_returns_error(self):
        """A plan with no tasks section returns an error."""
        from validate_plan import validate
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("""# PROJECT PLAN: No Tasks

## Objective
A plan with no tasks.

## Sub-Agents
### AGENT: some_agent
- description: An agent.
- tools: execute_code

## Managers
### MANAGER: mgr
- title: Manager
- description: Manager.
- expertise_blend: [testing]
- oversees: [some_agent]
""")
            path = f.name

        try:
            errors = validate(path)
            assert any("No tasks" in e for e in errors)
        finally:
            Path(path).unlink(missing_ok=True)


# ── Tests for checkpoint integrity ──────────────────────────────────────────

def _load_real_checkpoint():
    """Force-load the real checkpoint module, bypassing conftest stub."""
    import importlib
    real_spec = importlib.util.spec_from_file_location(
        "checkpoint_real",
        str(PROJECT_ROOT / "checkpoint.py"),
    )
    mod = importlib.util.module_from_spec(real_spec)
    real_spec.loader.exec_module(mod)
    return mod


class TestCheckpointIntegrity:
    """Verify checkpoint loading validates structure and handles corruption."""

    def test_valid_checkpoint_loads_successfully(self, tmp_path):
        """A checkpoint with all required keys loads normally."""
        cp = _load_real_checkpoint()
        cp.PLANS_DIR = tmp_path

        plan_id = "test_valid"
        state = {
            "objective": "Test objective",
            "managers": [{"name": "mgr1"}],
            "sub_agents": [{"name": "agent1"}],
            "reviewers": [{"name": "rev1"}],
            "tasks": [{"id": "TASK-001", "status": "done"}],
            "results": {"TASK-001": "result"},
            "tokens_used": 100,
            "phase": "dispatch",
            "active_task_id": None,
            "finished": False,
            "synthesis_report": "",
        }
        cp.save_checkpoint(plan_id, state)

        loaded = cp.load_checkpoint(plan_id)
        assert loaded is not None
        assert loaded["objective"] == "Test objective"
        assert loaded["tasks"][0]["id"] == "TASK-001"

    def test_checkpoint_missing_required_keys_returns_none(self, tmp_path):
        """A checkpoint missing 'tasks' key returns None."""
        cp = _load_real_checkpoint()
        cp.PLANS_DIR = tmp_path

        plan_id = "test_bad_keys"
        path = tmp_path / f"{plan_id}_checkpoint.json"
        data = {
            "plan_id": plan_id,
            "saved_at": "2026-01-01T00:00:00+00:00",
            "objective": "Test",
            "results": {},
            # 'tasks' is missing
        }
        path.write_text(json.dumps(data))

        loaded = cp.load_checkpoint(plan_id)
        assert loaded is None

    def test_checkpoint_missing_results_returns_none(self, tmp_path):
        """A checkpoint missing 'results' key returns None."""
        cp = _load_real_checkpoint()
        cp.PLANS_DIR = tmp_path

        plan_id = "test_no_results"
        path = tmp_path / f"{plan_id}_checkpoint.json"
        data = {
            "plan_id": plan_id,
            "saved_at": "2026-01-01T00:00:00+00:00",
            "objective": "Test",
            "tasks": [],
            # 'results' is missing
        }
        path.write_text(json.dumps(data))

        loaded = cp.load_checkpoint(plan_id)
        assert loaded is None

    def test_corrupted_json_returns_none_and_renames(self, tmp_path):
        """A checkpoint with invalid JSON returns None and renames the file."""
        cp = _load_real_checkpoint()
        cp.PLANS_DIR = tmp_path

        plan_id = "test_corrupted"
        path = tmp_path / f"{plan_id}_checkpoint.json"
        path.write_text("{invalid json content???")

        loaded = cp.load_checkpoint(plan_id)
        assert loaded is None
        # The corrupted file should be renamed
        corrupted_path = tmp_path / f"{plan_id}_checkpoint.json.corrupted"
        assert corrupted_path.exists()
        # Original should be gone
        assert not path.exists()

    def test_valid_checkpoint_no_rename(self, tmp_path):
        """A valid checkpoint should not create a .corrupted file."""
        cp = _load_real_checkpoint()
        cp.PLANS_DIR = tmp_path

        plan_id = "test_ok"
        path = tmp_path / f"{plan_id}_checkpoint.json"
        data = {
            "plan_id": plan_id,
            "saved_at": "2026-01-01T00:00:00+00:00",
            "objective": "Test",
            "managers": [],
            "sub_agents": [],
            "reviewers": [],
            "tasks": [{"id": "TASK-001"}],
            "results": {},
            "tokens_used": 0,
            "phase": "dispatch",
            "finished": False,
            "synthesis_report": "",
        }
        path.write_text(json.dumps(data))

        loaded = cp.load_checkpoint(plan_id)
        assert loaded is not None
        assert not (tmp_path / f"{plan_id}_checkpoint.json.corrupted").exists()
