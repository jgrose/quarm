"""
Integration tests for the NORT status bridge event pipeline.

Covers:
  1. write_status payload structure (required keys present)
  2. Session isolation (separate sessions don't leak state)
  3. register_rosters populates _session_rosters correctly
  4. log_event appends to the correct session's deque
  5. add_transcript_entry includes role, content, agent, timestamp
"""

import threading
import unittest
from collections import deque
from unittest.mock import patch, MagicMock

import sys, os
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force-load the real status_bridge (conftest.py may have stubbed it)
if "status_bridge" in sys.modules:
    del sys.modules["status_bridge"]
import status_bridge
importlib.reload(status_bridge)


def _reset_bridge():
    """Reset all bridge internal state between tests."""
    with status_bridge._state_lock:
        status_bridge._session_logs.clear()
        status_bridge._session_transcripts.clear()
        status_bridge._session_files.clear()
        status_bridge._session_projects.clear()
        status_bridge._session_reviewers.clear()
        status_bridge._session_rosters.clear()
    status_bridge._project = "NORT"
    status_bridge._active_reviewer = None
    # Clear thread-local session_id
    status_bridge._tls.session_id = ""


class TestWriteStatusPayloadStructure(unittest.TestCase):
    """Verify write_status produces a payload with every required key."""

    def setUp(self):
        _reset_bridge()
        status_bridge.set_session_id("test-payload")

    def tearDown(self):
        _reset_bridge()

    @patch.object(status_bridge, "_post")
    def test_write_status_payload_structure(self, mock_post):
        """Payload must contain project, session_id, tasks, phase, rosters, and other live state keys."""
        state = {
            "phase": "dispatch",
            "active_task_id": "TASK-001",
            "tasks": [
                {
                    "id": "TASK-001",
                    "title": "Build auth module",
                    "agent": "backend_dev",
                    "status": "in_progress",
                    "revision_count": 0,
                    "manager_notes": "",
                    "reviewer_notes": "",
                    "last_score": 0,
                    "current_model": "gpt-4",
                    "task_tokens": 120,
                    "depends_on": [],
                    "result": "some output",
                    "tool_calls": [],
                    "spawned_at": "2026-01-01T00:00:00Z",
                    "completed_at": "",
                }
            ],
            "results": {},
            "tokens_used": 500,
            "last_verdict": None,
            "synthesis_report": "",
            "validation": {},
            "coherence_report": {},
        }

        status_bridge.write_status(state)

        # _post is called in a daemon thread; wait briefly for the call
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][0]

        required_keys = {
            "project", "session_id",
            "sub_agents", "managers", "reviewers",
            "phase", "active_task_id", "active_reviewer",
            "tasks", "results_count", "total_tasks",
            "tokens_used", "last_verdict",
            "synthesis_report", "validation", "coherence_report",
            "log", "transcript", "files_touched", "updated_at",
        }
        self.assertTrue(
            required_keys.issubset(payload.keys()),
            f"Missing keys: {required_keys - payload.keys()}"
        )
        self.assertEqual(payload["session_id"], "test-payload")
        self.assertEqual(payload["phase"], "dispatch")
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["id"], "TASK-001")


class TestSessionIsolation(unittest.TestCase):
    """Two sessions must not leak state (separate log deques, rosters)."""

    def setUp(self):
        _reset_bridge()

    def tearDown(self):
        _reset_bridge()

    def test_session_isolation(self):
        """Events logged in session A must not appear in session B."""
        # Set up session A
        status_bridge.set_session_id("session-A")
        status_bridge.log_event("event-A-1")
        status_bridge.log_event("event-A-2")
        status_bridge.register_rosters(
            sub_agents=[{"name": "agent_a", "title": "Agent A"}],
            managers=[],
            reviewers=[],
        )

        # Set up session B
        status_bridge.set_session_id("session-B")
        status_bridge.log_event("event-B-1")
        status_bridge.register_rosters(
            sub_agents=[],
            managers=[{"name": "mgr_b", "title": "Manager B"}],
            reviewers=[],
        )

        # Verify isolation
        with status_bridge._state_lock:
            logs_a = list(status_bridge._session_logs["session-A"])
            logs_b = list(status_bridge._session_logs["session-B"])
            rosters_a = status_bridge._session_rosters["session-A"]
            rosters_b = status_bridge._session_rosters["session-B"]

        self.assertEqual(logs_a, ["event-A-1", "event-A-2"])
        self.assertEqual(logs_b, ["event-B-1"])
        self.assertNotIn("event-A-1", logs_b)
        self.assertNotIn("event-B-1", logs_a)

        self.assertEqual(len(rosters_a["sub_agents"]), 1)
        self.assertEqual(rosters_a["sub_agents"][0]["name"], "agent_a")
        self.assertEqual(len(rosters_a["managers"]), 0)

        self.assertEqual(len(rosters_b["managers"]), 1)
        self.assertEqual(rosters_b["managers"][0]["name"], "mgr_b")
        self.assertEqual(len(rosters_b["sub_agents"]), 0)


class TestRegisterRosters(unittest.TestCase):
    """register_rosters() must populate _session_rosters correctly."""

    def setUp(self):
        _reset_bridge()

    def tearDown(self):
        _reset_bridge()

    def test_register_rosters_stores_agents(self):
        status_bridge.set_session_id("roster-test")
        status_bridge.register_rosters(
            sub_agents=[
                {"name": "dev_1", "title": "Developer 1"},
                {"name": "dev_2", "description": "Backend dev"},
            ],
            managers=[{"name": "tech_lead", "title": "Tech Lead"}],
            reviewers=[
                {"name": "security_eng", "title": "Security Engineer"},
                {"name": "ux_reviewer", "title": "UX Reviewer"},
            ],
        )

        with status_bridge._state_lock:
            rosters = status_bridge._session_rosters["roster-test"]

        self.assertEqual(len(rosters["sub_agents"]), 2)
        self.assertEqual(rosters["sub_agents"][0]["name"], "dev_1")
        self.assertEqual(rosters["sub_agents"][0]["title"], "Developer 1")
        # Agent without title gets name-derived title via _title_from
        self.assertEqual(rosters["sub_agents"][1]["name"], "dev_2")

        self.assertEqual(len(rosters["managers"]), 1)
        self.assertEqual(rosters["managers"][0]["name"], "tech_lead")

        self.assertEqual(len(rosters["reviewers"]), 2)
        self.assertEqual(rosters["reviewers"][0]["name"], "security_eng")
        self.assertEqual(rosters["reviewers"][1]["name"], "ux_reviewer")


class TestLogEvent(unittest.TestCase):
    """log_event() must append to the correct session's deque."""

    def setUp(self):
        _reset_bridge()

    def tearDown(self):
        _reset_bridge()

    def test_log_event_appends_to_session(self):
        status_bridge.set_session_id("log-test")

        status_bridge.log_event("first event")
        status_bridge.log_event("second event")
        status_bridge.log_event("third event")

        with status_bridge._state_lock:
            log_entries = list(status_bridge._session_logs["log-test"])

        self.assertEqual(len(log_entries), 3)
        self.assertEqual(log_entries[0], "first event")
        self.assertEqual(log_entries[1], "second event")
        self.assertEqual(log_entries[2], "third event")


class TestTranscriptEntryFormat(unittest.TestCase):
    """add_transcript_entry() must include role, content, agent, and timestamp."""

    def setUp(self):
        _reset_bridge()

    def tearDown(self):
        _reset_bridge()

    def test_transcript_entry_format(self):
        status_bridge.set_session_id("transcript-test")

        status_bridge.add_transcript_entry(
            role="assistant",
            content="Task completed successfully",
            agent="backend_dev",
            task_id="TASK-042",
        )

        with status_bridge._state_lock:
            entries = list(status_bridge._session_transcripts["transcript-test"])

        self.assertEqual(len(entries), 1)
        entry = entries[0]

        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(entry["content"], "Task completed successfully")
        self.assertEqual(entry["agent"], "backend_dev")
        self.assertEqual(entry["task_id"], "TASK-042")
        self.assertIn("timestamp", entry)
        # Timestamp should be ISO format with timezone
        self.assertIn("T", entry["timestamp"])


if __name__ == "__main__":
    unittest.main()
