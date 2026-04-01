"""
Integration tests for the NORT orchestrator review cycle.

Covers:
  6. After a FAIL verdict from manager_review_node, the task enters revision
     status with manager_notes populated, and on re-execution the sub-agent
     receives that feedback in context.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_task(tid="TASK-001", agent="backend_dev", status="in_manager_review",
               result="initial draft output", revision_count=0, **overrides):
    """Build a minimal task dict matching orchestrator conventions."""
    base = {
        "id": tid,
        "title": "Build auth module",
        "description": "Implement JWT-based authentication",
        "agent": agent,
        "status": status,
        "result": result,
        "revision_count": revision_count,
        "manager_notes": "",
        "reviewer_notes": "",
        "last_score": 0,
        "current_model": "",
        "task_tokens": 0,
        "depends_on": [],
        "tool_calls": [],
        "spawned_at": "",
        "completed_at": "",
        "tolerance": 0,
    }
    base.update(overrides)
    return base


def _make_state(tasks, **overrides):
    """Build a minimal OrchestratorState dict."""
    base = {
        "tasks": tasks,
        "results": {},
        "sub_agents": [
            {"name": "backend_dev", "description": "Backend specialist", "tools": []},
        ],
        "managers": [
            {
                "name": "tech_lead",
                "title": "Technical Lead",
                "description": "Reviews code quality and architecture",
                "expertise_blend": ["architecture", "code_quality"],
                "oversees": ["backend_dev"],
            },
        ],
        "reviewers": [],
        "phase": "manager_review",
        "active_task_id": "TASK-001",
        "active_task_ids": [],
        "tokens_used": 0,
        "last_verdict": None,
        "synthesis_report": "",
        "validation": {},
        "coherence_report": {},
    }
    base.update(overrides)
    return base


class TestRevisionCycleAttachesFeedback(unittest.TestCase):
    """After a FAIL from manager_review_node, the task should have manager_notes
    populated, and the sub_agent should receive that feedback on re-execution."""

    @patch("orchestrator.snapshot_artifacts")
    @patch("orchestrator._auto_ingest")
    @patch("orchestrator.write_status")
    @patch("orchestrator.set_active_reviewer")
    @patch("orchestrator.log_event")
    @patch("orchestrator.track_score")
    @patch("orchestrator._load_orchestrator_config", return_value={})
    @patch("orchestrator.resolve_model", return_value="test-model")
    @patch("orchestrator.llm")
    def test_revision_cycle_attaches_feedback(
        self, mock_llm, mock_resolve_model, mock_config,
        mock_track, mock_log, mock_set_reviewer,
        mock_write_status, mock_auto_ingest, mock_snapshot,
    ):
        from orchestrator import manager_review_node, _execute_single_task

        # Configure LLM to return a FAIL verdict with specific feedback
        fail_response = MagicMock()
        fail_response.content = json.dumps({
            "verdict": "FAIL",
            "score": 4,
            "issues": ["Missing input validation", "No rate limiting"],
            "feedback": "Add input validation for all endpoints and implement rate limiting",
        })
        fail_response.response_metadata = {"token_usage": {"total_tokens": 100}}
        mock_llm.return_value.invoke.return_value = fail_response

        task = _make_task()
        state = _make_state([task])

        # Patch _run_id to empty string to skip tracking imports
        with patch("orchestrator._run_id", ""):
            result_state = manager_review_node(state)

        # Verify task enters revision status with feedback attached
        result_task = result_state["tasks"][0]
        self.assertEqual(result_task["status"], "revision")
        self.assertEqual(
            result_task["manager_notes"],
            "Add input validation for all endpoints and implement rate limiting",
        )
        self.assertEqual(result_task["revision_count"], 1)
        self.assertEqual(result_state["phase"], "execute")

        # Now verify the sub-agent would receive that feedback during re-execution
        # by calling _execute_single_task with the revised task
        revised_response = MagicMock()
        revised_response.content = "Revised output with validation and rate limiting"
        revised_response.response_metadata = {"token_usage": {"total_tokens": 200}}
        revised_response.tool_calls = []
        mock_llm.return_value.invoke.return_value = revised_response

        tid, draft, toks, tool_log, model = _execute_single_task(
            "TASK-001",
            result_state["tasks"],
            result_state["results"],
            state["sub_agents"],
        )

        # Verify the LLM was invoked with feedback in context
        call_args = mock_llm.return_value.invoke.call_args
        messages = call_args[0][0]
        # The HumanMessage should contain the manager feedback
        human_msg = messages[1].content
        self.assertIn("MANAGER FEEDBACK", human_msg)
        self.assertIn("Add input validation for all endpoints", human_msg)


if __name__ == "__main__":
    unittest.main()
