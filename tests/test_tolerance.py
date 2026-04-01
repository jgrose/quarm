"""
Integration tests for the NORT tolerance resolution system.

Covers:
  7.  _resolve_tolerance precedence chain:
      config per-agent > task-level > plan per-agent > config global > DEFAULT_TOLERANCE
  8.  Earned tolerance bonus: agent with avg_score > 8 over 5+ runs gets +1
  9.  track_tolerance_override() inserts to SQLite correctly
  10. Specialist skip on high score: score >= 9 with config toggle skips panel
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestResolveTolerancePrecedence(unittest.TestCase):
    """Config per-agent > task-level > plan per-agent > config global > DEFAULT_TOLERANCE."""

    def _resolve(self, config, agent_dict, task_tolerance=0):
        with patch("orchestrator._load_orchestrator_config", return_value=config):
            # Patch check_earned_tolerance to return False so bonus doesn't interfere
            with patch("agent_registry.check_earned_tolerance", return_value=False):
                from orchestrator import _resolve_tolerance
                return _resolve_tolerance("test_agent", agent_dict, task_tolerance)

    def test_resolve_tolerance_precedence(self):
        """Walk the entire precedence chain from highest to lowest priority."""
        # 1. Config per-agent override wins over everything
        result = self._resolve(
            config={"tolerance_overrides": {"test_agent": 9}, "default_tolerance": 3},
            agent_dict={"tolerance": 5},
            task_tolerance=7,
        )
        self.assertEqual(result, 9, "Config per-agent override should win")

        # 2. Task-level tolerance is next when no config per-agent
        result = self._resolve(
            config={"tolerance_overrides": {}, "default_tolerance": 3},
            agent_dict={"tolerance": 5},
            task_tolerance=7,
        )
        self.assertEqual(result, 7, "Task-level tolerance should be second priority")

        # 3. Plan per-agent tolerance is next when no task-level
        result = self._resolve(
            config={"tolerance_overrides": {}, "default_tolerance": 3},
            agent_dict={"tolerance": 5},
            task_tolerance=0,
        )
        self.assertEqual(result, 5, "Plan per-agent tolerance should be third priority")

        # 4. Config global default is next when no plan per-agent
        result = self._resolve(
            config={"tolerance_overrides": {}, "default_tolerance": 3},
            agent_dict={"tolerance": 0},
            task_tolerance=0,
        )
        self.assertEqual(result, 3, "Config global default should be fourth priority")

        # 5. DEFAULT_TOLERANCE (6) is the final fallback
        result = self._resolve(
            config={},
            agent_dict={},
            task_tolerance=0,
        )
        self.assertEqual(result, 6, "DEFAULT_TOLERANCE (6) should be the final fallback")


class TestEarnedToleranceBonus(unittest.TestCase):
    """Agent with avg_score > 8 over 5+ runs gets +1 earned bonus."""

    def test_earned_tolerance_bonus(self):
        from orchestrator import _resolve_tolerance

        # Agent qualifies for bonus: avg_score > 8 and 5+ runs
        with patch("orchestrator._load_orchestrator_config", return_value={}):
            with patch("agent_registry.check_earned_tolerance", return_value=True):
                result = _resolve_tolerance("high_performer", {}, 0)

        # DEFAULT_TOLERANCE is 6, bonus gives +1 = 7
        self.assertEqual(result, 7, "Earned bonus should add +1 to base tolerance")

    def test_no_bonus_when_ineligible(self):
        from orchestrator import _resolve_tolerance

        with patch("orchestrator._load_orchestrator_config", return_value={}):
            with patch("agent_registry.check_earned_tolerance", return_value=False):
                result = _resolve_tolerance("low_performer", {}, 0)

        # DEFAULT_TOLERANCE is 6, no bonus
        self.assertEqual(result, 6, "No bonus when agent is ineligible")


class TestToleranceOverrideTracking(unittest.TestCase):
    """track_tolerance_override() inserts to SQLite correctly."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._orig_db_path = None

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_tolerance_override_tracking(self):
        import tracking

        # Redirect DB_PATH to a temp file
        orig_path = tracking.DB_PATH
        tracking.DB_PATH = self._tmp.name

        try:
            # Initialize schema in temp DB
            conn = sqlite3.connect(self._tmp.name)
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    plan_name TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    total_revisions INTEGER DEFAULT 0,
                    task_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );
                CREATE TABLE IF NOT EXISTS tolerance_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    task_id TEXT,
                    reviewer TEXT,
                    original_verdict TEXT,
                    score INTEGER,
                    tolerance INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
            """)
            conn.execute(
                "INSERT INTO runs (id, plan_name, started_at, status) VALUES (?, ?, ?, ?)",
                ("run-123", "test-plan", "2026-01-01T00:00:00Z", "running"),
            )
            conn.commit()
            conn.close()

            # Call track_tolerance_override
            tracking.track_tolerance_override(
                run_id="run-123",
                task_id="TASK-005",
                reviewer="tech_lead",
                original_verdict="FAIL",
                score=7,
                tolerance=6,
            )

            # Verify the row was inserted
            conn = sqlite3.connect(self._tmp.name)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tolerance_overrides WHERE run_id = ?", ("run-123",)
            ).fetchall()
            conn.close()

            self.assertEqual(len(rows), 1)
            row = dict(rows[0])
            self.assertEqual(row["run_id"], "run-123")
            self.assertEqual(row["task_id"], "TASK-005")
            self.assertEqual(row["reviewer"], "tech_lead")
            self.assertEqual(row["original_verdict"], "FAIL")
            self.assertEqual(row["score"], 7)
            self.assertEqual(row["tolerance"], 6)
            self.assertIsNotNone(row["created_at"])
        finally:
            tracking.DB_PATH = orig_path


class TestSpecialistSkipOnHighScore(unittest.TestCase):
    """Score >= 9 with skip_specialist_on_high_score config skips specialist panel."""

    @patch("orchestrator.snapshot_artifacts")
    @patch("orchestrator._auto_ingest")
    @patch("orchestrator.write_status")
    @patch("orchestrator.set_active_reviewer")
    @patch("orchestrator.log_event")
    @patch("orchestrator.track_score")
    @patch("orchestrator.resolve_model", return_value="test-model")
    @patch("orchestrator.llm")
    def test_specialist_skip_on_high_score(
        self, mock_llm, mock_resolve_model, mock_track,
        mock_log, mock_set_reviewer, mock_write_status,
        mock_auto_ingest, mock_snapshot,
    ):
        from orchestrator import manager_review_node

        # LLM returns PASS with score 9
        pass_response = MagicMock()
        pass_response.content = json.dumps({
            "verdict": "PASS",
            "score": 9,
            "issues": [],
            "feedback": "",
        })
        pass_response.response_metadata = {"token_usage": {"total_tokens": 80}}
        mock_llm.return_value.invoke.return_value = pass_response

        task = {
            "id": "TASK-010",
            "title": "Build dashboard",
            "description": "Create monitoring dashboard",
            "agent": "frontend_dev",
            "status": "in_manager_review",
            "result": "Dashboard implementation code",
            "revision_count": 0,
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

        state = {
            "tasks": [task],
            "results": {},
            "sub_agents": [
                {"name": "frontend_dev", "description": "Frontend specialist", "tools": []},
            ],
            "managers": [
                {
                    "name": "tech_lead",
                    "title": "Technical Lead",
                    "description": "Reviews code quality",
                    "expertise_blend": ["architecture"],
                    "oversees": ["frontend_dev"],
                },
            ],
            "reviewers": [
                {
                    "name": "ux_designer",
                    "title": "UX Designer",
                    "description": "UX review",
                    "focus_areas": ["accessibility"],
                },
            ],
            "phase": "manager_review",
            "active_task_id": "TASK-010",
            "active_task_ids": [],
            "tokens_used": 0,
            "last_verdict": None,
            "synthesis_report": "",
            "validation": {},
            "coherence_report": {},
        }

        # Config has skip_specialist_on_high_score enabled
        config_with_skip = {"skip_specialist_on_high_score": True}

        with patch("orchestrator._run_id", ""):
            with patch("orchestrator._load_orchestrator_config", return_value=config_with_skip):
                with patch("agent_registry.check_earned_tolerance", return_value=False):
                    result_state = manager_review_node(state)

        # Task should go directly to done, skipping specialist review
        result_task = result_state["tasks"][0]
        self.assertEqual(result_task["status"], "done",
                         "Task should be 'done' (specialist review skipped)")
        # Phase should be dispatch, not specialist_review
        self.assertEqual(result_state["phase"], "dispatch")
        # Result should be captured
        self.assertIn("TASK-010", result_state["results"])


if __name__ == "__main__":
    unittest.main()
