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


class TestRetryQueueOnPostFailure(unittest.TestCase):
    """When _post fails, the payload should be queued for retry."""

    def setUp(self):
        _reset_bridge()
        status_bridge.set_session_id("retry-test")
        # Reset retry-related state
        status_bridge._retry_queue.clear()
        status_bridge._posts_sent = 0
        status_bridge._posts_failed = 0
        status_bridge._posts_retried = 0

    def tearDown(self):
        _reset_bridge()
        status_bridge._retry_queue.clear()
        status_bridge._posts_sent = 0
        status_bridge._posts_failed = 0
        status_bridge._posts_retried = 0

    def test_failed_post_queued_for_retry(self):
        """A failed _post call should add the payload to _retry_queue."""
        payload = {"test": "data", "session_id": "retry-test"}
        # Make the actual HTTP call fail
        mock_req = MagicMock()
        mock_req.post.side_effect = Exception("Connection refused")
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._post(payload)

        self.assertEqual(len(status_bridge._retry_queue), 1)
        queued_item = status_bridge._retry_queue[0]
        self.assertEqual(queued_item["payload"], payload)
        self.assertEqual(queued_item["attempts"], 1)

    def test_successful_post_not_queued(self):
        """A successful _post call should NOT add anything to _retry_queue."""
        payload = {"test": "data"}
        mock_req = MagicMock()
        mock_req.post.return_value = MagicMock(status_code=200)
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._post(payload)

        self.assertEqual(len(status_bridge._retry_queue), 0)

    def test_posts_sent_counter_incremented_on_success(self):
        """_posts_sent counter should increment on successful POST."""
        payload = {"test": "data"}
        mock_req = MagicMock()
        mock_req.post.return_value = MagicMock(status_code=200)
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._post(payload)

        self.assertEqual(status_bridge._posts_sent, 1)

    def test_posts_failed_counter_incremented_on_failure(self):
        """_posts_failed counter should increment on failed POST."""
        payload = {"test": "data"}
        mock_req = MagicMock()
        mock_req.post.side_effect = Exception("Connection refused")
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._post(payload)

        self.assertEqual(status_bridge._posts_failed, 1)

    def test_retry_queue_maxlen(self):
        """_retry_queue should not exceed its maxlen (100)."""
        self.assertEqual(status_bridge._retry_queue.maxlen, 100)


class TestRetryThreadDrains(unittest.TestCase):
    """The retry thread should drain the queue and re-POST failed payloads."""

    def setUp(self):
        _reset_bridge()
        status_bridge._retry_queue.clear()
        status_bridge._posts_retried = 0

    def tearDown(self):
        _reset_bridge()
        status_bridge._retry_queue.clear()
        status_bridge._posts_retried = 0

    def test_retry_drain_succeeds(self):
        """_drain_retry_queue should successfully retry and remove items."""
        status_bridge._retry_queue.append({
            "payload": {"test": "retry-data"},
            "attempts": 1,
        })
        mock_req = MagicMock()
        mock_req.post.return_value = MagicMock(status_code=200)
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._drain_retry_queue()

        self.assertEqual(len(status_bridge._retry_queue), 0)
        self.assertEqual(status_bridge._posts_retried, 1)

    def test_retry_drain_requeues_on_failure(self):
        """Failed retry should put the item back with incremented attempts."""
        status_bridge._retry_queue.append({
            "payload": {"test": "retry-data"},
            "attempts": 1,
        })
        mock_req = MagicMock()
        mock_req.post.side_effect = Exception("Still down")
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._drain_retry_queue()

        self.assertEqual(len(status_bridge._retry_queue), 1)
        self.assertEqual(status_bridge._retry_queue[0]["attempts"], 2)

    def test_retry_drain_discards_after_max_attempts(self):
        """After 3 total attempts, the item should be discarded with a warning."""
        status_bridge._retry_queue.append({
            "payload": {"test": "retry-data"},
            "attempts": 3,
        })
        mock_req = MagicMock()
        mock_req.post.side_effect = Exception("Still down")
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._drain_retry_queue()

        # Should be discarded, not requeued
        self.assertEqual(len(status_bridge._retry_queue), 0)

    def test_retry_drain_handles_multiple_items(self):
        """Multiple queued items should all be processed in one drain cycle."""
        for i in range(3):
            status_bridge._retry_queue.append({
                "payload": {"index": i},
                "attempts": 1,
            })
        mock_req = MagicMock()
        mock_req.post.return_value = MagicMock(status_code=200)
        with patch.object(status_bridge, "_HAS_REQUESTS", True), \
             patch.object(status_bridge, "_req", mock_req):
            status_bridge._drain_retry_queue()

        self.assertEqual(len(status_bridge._retry_queue), 0)
        self.assertEqual(status_bridge._posts_retried, 3)


class TestSessionCleanup(unittest.TestCase):
    """cleanup_session() must remove all per-session state."""

    def setUp(self):
        _reset_bridge()

    def tearDown(self):
        _reset_bridge()

    def test_cleanup_removes_all_session_state(self):
        """cleanup_session should remove logs, transcripts, files, rosters, etc."""
        status_bridge.set_session_id("cleanup-test")
        status_bridge.log_event("some event")
        status_bridge.add_transcript_entry("user", "hello", "agent1")
        status_bridge.register_rosters(
            sub_agents=[{"name": "a", "title": "A"}],
            managers=[], reviewers=[],
        )
        status_bridge.set_project("TestProject")
        status_bridge.set_active_reviewer("reviewer1")

        # Verify state exists before cleanup
        with status_bridge._state_lock:
            self.assertIn("cleanup-test", status_bridge._session_logs)
            self.assertIn("cleanup-test", status_bridge._session_transcripts)
            self.assertIn("cleanup-test", status_bridge._session_rosters)

        # Clean up
        status_bridge.cleanup_session("cleanup-test")

        # Verify all session state is removed
        with status_bridge._state_lock:
            self.assertNotIn("cleanup-test", status_bridge._session_logs)
            self.assertNotIn("cleanup-test", status_bridge._session_transcripts)
            self.assertNotIn("cleanup-test", status_bridge._session_files)
            self.assertNotIn("cleanup-test", status_bridge._session_projects)
            self.assertNotIn("cleanup-test", status_bridge._session_reviewers)
            self.assertNotIn("cleanup-test", status_bridge._session_rosters)

    def test_cleanup_nonexistent_session_is_safe(self):
        """cleanup_session on a session that doesn't exist should not raise."""
        # Should not raise
        status_bridge.cleanup_session("does-not-exist")

    def test_cleanup_removes_dropped_events_counter(self):
        """cleanup_session should also remove the dropped events counter."""
        status_bridge.set_session_id("drop-cleanup")
        status_bridge._session_dropped_events["drop-cleanup"] = 5

        status_bridge.cleanup_session("drop-cleanup")

        self.assertNotIn("drop-cleanup", status_bridge._session_dropped_events)


class TestDroppedEventTracking(unittest.TestCase):
    """Track events lost when a deque is full."""

    def setUp(self):
        _reset_bridge()
        status_bridge._session_dropped_events.clear()

    def tearDown(self):
        _reset_bridge()
        status_bridge._session_dropped_events.clear()

    def test_dropped_events_counted_when_log_deque_full(self):
        """When the log deque is full and a new event is added, dropped count increments."""
        status_bridge.set_session_id("drop-test")

        # Fill the log deque to capacity (maxlen=80)
        for i in range(80):
            status_bridge.log_event(f"event-{i}")

        with status_bridge._state_lock:
            self.assertEqual(len(status_bridge._session_logs["drop-test"]), 80)

        # Adding one more should cause the oldest to be evicted
        status_bridge.log_event("overflow-event")

        self.assertEqual(
            status_bridge._session_dropped_events.get("drop-test", 0), 1
        )

    def test_dropped_events_included_in_write_status_payload(self):
        """write_status payload should include dropped_events count."""
        status_bridge.set_session_id("drop-payload")
        status_bridge._session_dropped_events["drop-payload"] = 7

        state = {
            "phase": "dispatch",
            "tasks": [],
            "results": {},
            "tokens_used": 0,
        }

        with patch.object(status_bridge, "_post") as mock_post:
            status_bridge.write_status(state)
            mock_post.assert_called_once()
            payload = mock_post.call_args[0][0]
            self.assertEqual(payload["dropped_events"], 7)

    def test_dropped_events_zero_by_default(self):
        """write_status payload should include dropped_events=0 for new sessions."""
        status_bridge.set_session_id("drop-zero")

        state = {
            "phase": "dispatch",
            "tasks": [],
            "results": {},
            "tokens_used": 0,
        }

        with patch.object(status_bridge, "_post") as mock_post:
            status_bridge.write_status(state)
            mock_post.assert_called_once()
            payload = mock_post.call_args[0][0]
            self.assertEqual(payload["dropped_events"], 0)


class TestRetryThreadIsDaemon(unittest.TestCase):
    """The retry background thread must be a daemon thread."""

    def test_retry_thread_is_daemon(self):
        """_retry_thread should be a daemon thread."""
        self.assertTrue(status_bridge._retry_thread.daemon)

    def test_retry_thread_is_alive(self):
        """_retry_thread should be running."""
        self.assertTrue(status_bridge._retry_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
