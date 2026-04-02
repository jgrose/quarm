"""
Tests for the NORT state machine routing logic.

Validates master_node dispatch, review routing, revision cycles,
force-acceptance at MAX_REVISIONS, and tolerance-based overrides.
"""

import json
from unittest.mock import MagicMock, patch

from tests.conftest import make_base_state

import orchestrator


class TestMasterNodeDispatch:
    """Verify master_node routes tasks based on status and dependencies."""

    def test_route_master_dispatches_ready_tasks(self):
        """Pending task with all deps met is dispatched (phase='execute')."""
        tasks = [
            {"id": "TASK-001", "title": "First", "agent": "backend_engineer",
             "description": "Do something", "task_type": ["code"], "reviewers": [],
             "depends_on": [], "model": "", "tolerance": 0,
             "status": "pending", "result": "", "manager_notes": "",
             "reviewer_notes": "", "revision_count": 0},
        ]
        state = make_base_state(tasks)

        result = orchestrator.master_node(state)

        assert result["phase"] == "execute"
        assert result["active_task_id"] == "TASK-001"
        # The task should be marked in_progress
        dispatched = next(t for t in result["tasks"] if t["id"] == "TASK-001")
        assert dispatched["status"] == "in_progress"

    def test_route_master_prioritizes_reviews(self):
        """Task in_manager_review is routed before pending tasks."""
        tasks = [
            {"id": "TASK-001", "title": "Reviewed", "agent": "backend_engineer",
             "description": "Already done", "task_type": ["code"], "reviewers": [],
             "depends_on": [], "model": "", "tolerance": 0,
             "status": "in_manager_review", "result": "some output",
             "manager_notes": "", "reviewer_notes": "", "revision_count": 0},
            {"id": "TASK-002", "title": "Waiting", "agent": "backend_engineer",
             "description": "Not started", "task_type": ["code"], "reviewers": [],
             "depends_on": [], "model": "", "tolerance": 0,
             "status": "pending", "result": "", "manager_notes": "",
             "reviewer_notes": "", "revision_count": 0},
        ]
        state = make_base_state(tasks)

        result = orchestrator.master_node(state)

        # Reviews take priority over pending dispatch
        assert result["phase"] == "manager_review"
        assert result["active_task_id"] == "TASK-001"

    def test_route_master_triggers_synthesis(self):
        """All tasks done triggers phase='done' for synthesis."""
        tasks = [
            {"id": "TASK-001", "title": "Done1", "agent": "backend_engineer",
             "description": "Finished", "task_type": ["code"], "reviewers": [],
             "depends_on": [], "model": "", "tolerance": 0,
             "status": "done", "result": "output1", "manager_notes": "",
             "reviewer_notes": "", "revision_count": 0},
            {"id": "TASK-002", "title": "Done2", "agent": "backend_engineer",
             "description": "Also finished", "task_type": ["code"], "reviewers": [],
             "depends_on": ["TASK-001"], "model": "", "tolerance": 0,
             "status": "done", "result": "output2", "manager_notes": "",
             "reviewer_notes": "", "revision_count": 0},
        ]
        state = make_base_state(tasks)
        state["results"] = {"TASK-001": "output1", "TASK-002": "output2"}

        result = orchestrator.master_node(state)

        assert result["phase"] == "done"

    def test_route_master_router_function(self):
        """The route_master function maps phase to correct next node."""
        assert orchestrator.route_master({"phase": "done"}) == "synthesis"
        assert orchestrator.route_master({"phase": "execute"}) == "sub_agent"
        assert orchestrator.route_master({"phase": "dispatch"}) == "master"
        assert orchestrator.route_master({"phase": "manager_review"}) == "manager_review"


class TestManagerReviewRouting:
    """Verify manager_review_node routes based on verdict."""

    def _make_review_state(self, revision_count=0, task_tolerance=0):
        """Build a state with a task in manager review."""
        tasks = [
            {"id": "TASK-001", "title": "Under Review", "agent": "backend_engineer",
             "description": "Needs review", "task_type": ["code"], "reviewers": [],
             "depends_on": [], "model": "", "tolerance": task_tolerance,
             "status": "in_manager_review", "result": "task output here",
             "manager_notes": "", "reviewer_notes": "",
             "revision_count": revision_count},
        ]
        managers = [
            {"name": "engineering_director", "title": "Engineering Director",
             "description": "Reviews code", "expertise_blend": ["API_design"],
             "oversees": ["backend_engineer"], "model": "", "tolerance": 0},
        ]
        state = make_base_state(tasks, managers=managers)
        state["active_task_id"] = "TASK-001"
        return state

    def test_route_manager_pass_to_specialist(self, mock_llm_pass):
        """PASS verdict routes task to specialist_review."""
        state = self._make_review_state()

        with patch.object(orchestrator, "resolve_model", return_value="test-model"):
            result = orchestrator.manager_review_node(state)

        task = next(t for t in result["tasks"] if t["id"] == "TASK-001")
        assert task["status"] == "in_specialist_review"
        assert result["phase"] == "specialist_review"

    def test_route_manager_fail_to_revision(self, mock_llm_fail):
        """FAIL verdict routes task back to execute with revision_count++."""
        state = self._make_review_state(revision_count=0)

        with patch.object(orchestrator, "resolve_model", return_value="test-model"), \
             patch.object(orchestrator, "snapshot_artifacts"):
            result = orchestrator.manager_review_node(state)

        task = next(t for t in result["tasks"] if t["id"] == "TASK-001")
        assert task["status"] == "revision"
        assert task["revision_count"] == 1
        assert result["phase"] == "execute"

    def test_route_manager_force_accept_at_max_revisions(self, mock_llm_fail):
        """revision_count >= MAX_REVISIONS forces approval regardless of verdict."""
        state = self._make_review_state(revision_count=3)

        result = orchestrator.manager_review_node(state)

        task = next(t for t in result["tasks"] if t["id"] == "TASK-001")
        assert task["status"] == "done"
        assert result["phase"] == "dispatch"
        # Result should be captured
        assert "TASK-001" in result["results"]

    def test_tolerance_override_converts_fail_to_pass(self, mock_llm_fail_high_score):
        """Score >= tolerance overrides FAIL verdict to PASS."""
        # Task has tolerance=7, mock returns FAIL with score=7
        state = self._make_review_state(task_tolerance=7)

        with patch.object(orchestrator, "resolve_model", return_value="test-model"), \
             patch.object(orchestrator, "_load_orchestrator_config", return_value={}):
            result = orchestrator.manager_review_node(state)

        task = next(t for t in result["tasks"] if t["id"] == "TASK-001")
        # Score 7 >= tolerance 7 means FAIL is overridden to PASS
        assert task["status"] == "in_specialist_review"
        assert result["phase"] == "specialist_review"

    def test_route_manager_router_function(self):
        """The route_manager function maps phase to correct next node."""
        assert orchestrator.route_manager({"phase": "execute"}) == "sub_agent"
        assert orchestrator.route_manager({"phase": "specialist_review"}) == "specialist_review"
        assert orchestrator.route_manager({"phase": "dispatch"}) == "master"


class TestRouteSpecialist:
    """Verify the route_specialist router function."""

    def test_route_specialist_router_function(self):
        """route_specialist routes to sub_agent on execute, master otherwise."""
        assert orchestrator.route_specialist({"phase": "execute"}) == "sub_agent"
        assert orchestrator.route_specialist({"phase": "dispatch"}) == "master"
        assert orchestrator.route_specialist({"phase": "done"}) == "master"
