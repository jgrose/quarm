"""
Tests for error recovery, retry logic, and edge cases in the NORT orchestrator.

Covers:
  1. _invoke_with_retry — success, transient retry, exhaustion, logging
  2. Status bridge _post failure handling and session deque limits
  3. Edge cases — empty plans, bad agent refs, no pending tasks, malformed tokens
"""

import sys
import os
import time
import json
import logging
import importlib
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use conftest stubs so orchestrator can import cleanly
from tests.conftest import make_base_state, FIXTURES_DIR

import orchestrator


# ============================================================================
# 1. LLM Retry Logic (_invoke_with_retry)
# ============================================================================


class TestInvokeWithRetrySuccess(unittest.TestCase):
    """_invoke_with_retry succeeds on first call without any retry."""

    def test_first_attempt_succeeds(self):
        """When the LLM responds on the first try, the result is returned immediately."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Hello world"
        mock_llm.invoke.return_value = mock_resp

        result = orchestrator._invoke_with_retry(mock_llm, ["test message"])

        assert result is mock_resp
        mock_llm.invoke.assert_called_once_with(["test message"])


class TestInvokeWithRetryTransient(unittest.TestCase):
    """_invoke_with_retry retries on transient failure and succeeds on 2nd attempt."""

    @patch("time.sleep")  # Don't actually sleep in tests
    def test_retry_on_transient_failure(self, mock_sleep):
        """First call raises, second call succeeds => result returned, sleep called once."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Success on retry"

        mock_llm.invoke.side_effect = [
            ConnectionError("server unavailable"),
            mock_resp,
        ]

        result = orchestrator._invoke_with_retry(
            mock_llm, ["msg"], max_retries=3, base_delay=1.0
        )

        assert result is mock_resp
        assert mock_llm.invoke.call_count == 2
        # Should have slept once (base_delay * 2^0 = 1.0s)
        mock_sleep.assert_called_once_with(1.0)


class TestInvokeWithRetryExhausted(unittest.TestCase):
    """_invoke_with_retry raises after max_retries exhausted."""

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        """All attempts fail => original exception is re-raised."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("persistent failure")

        with pytest.raises(RuntimeError, match="persistent failure"):
            orchestrator._invoke_with_retry(
                mock_llm, ["msg"], max_retries=3, base_delay=1.0
            )

        assert mock_llm.invoke.call_count == 3
        # Should have slept twice (before retries 2 and 3, not after final failure)
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_raises_after_single_retry_allowed(self, mock_sleep):
        """With max_retries=1, only one attempt is made, then exception raised."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            orchestrator._invoke_with_retry(
                mock_llm, ["msg"], max_retries=1, base_delay=1.0
            )

        assert mock_llm.invoke.call_count == 1
        mock_sleep.assert_not_called()


class TestInvokeWithRetryLogging(unittest.TestCase):
    """_invoke_with_retry logs retry attempts via log_event."""

    @patch("time.sleep")
    def test_logs_retry_attempts(self, mock_sleep):
        """Each retry logs a message containing the attempt number."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "ok"

        # Fail twice, succeed on third attempt
        mock_llm.invoke.side_effect = [
            ConnectionError("fail1"),
            TimeoutError("fail2"),
            mock_resp,
        ]

        with patch.object(orchestrator, "log_event") as mock_log:
            result = orchestrator._invoke_with_retry(
                mock_llm, ["msg"], max_retries=3, base_delay=2.0
            )

        assert result is mock_resp
        assert mock_log.call_count == 2

        # First retry log
        first_call_msg = mock_log.call_args_list[0][0][0]
        assert "RETRY" in first_call_msg
        assert "1/3" in first_call_msg
        assert "ConnectionError" in first_call_msg

        # Second retry log
        second_call_msg = mock_log.call_args_list[1][0][0]
        assert "RETRY" in second_call_msg
        assert "2/3" in second_call_msg
        assert "TimeoutError" in second_call_msg

    @patch("time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Delay doubles on each retry: base_delay * 2^attempt."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "ok"

        mock_llm.invoke.side_effect = [
            ConnectionError("fail1"),
            ConnectionError("fail2"),
            mock_resp,
        ]

        with patch.object(orchestrator, "log_event"):
            orchestrator._invoke_with_retry(
                mock_llm, ["msg"], max_retries=3, base_delay=2.0
            )

        # Delays: 2.0 * 2^0 = 2.0, 2.0 * 2^1 = 4.0
        assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


# ============================================================================
# 2. Status Bridge Failure Handling
# ============================================================================


def _load_real_bridge():
    """Force-load the real status_bridge module, bypassing conftest stub."""
    if "status_bridge" in sys.modules:
        del sys.modules["status_bridge"]
    import status_bridge
    importlib.reload(status_bridge)
    return status_bridge


def _reset_bridge(bridge):
    """Reset all bridge internal state between tests."""
    with bridge._state_lock:
        bridge._session_logs.clear()
        bridge._session_transcripts.clear()
        bridge._session_files.clear()
        bridge._session_projects.clear()
        bridge._session_reviewers.clear()
        bridge._session_rosters.clear()
    bridge._project = "NORT"
    bridge._active_reviewer = None
    bridge._tls.session_id = ""


class TestPostFailureLogging(unittest.TestCase):
    """_post() failure is logged, not silently swallowed."""

    def setUp(self):
        self.bridge = _load_real_bridge()
        _reset_bridge(self.bridge)

    def tearDown(self):
        _reset_bridge(self.bridge)

    def test_post_failure_is_logged(self):
        """When the POST request fails, it is logged via the logger."""
        with patch.object(self.bridge.log, "debug") as mock_debug:
            # Force the POST to fail by patching the HTTP layer
            if self.bridge._HAS_REQUESTS:
                with patch.object(self.bridge._req, "post", side_effect=ConnectionError("refused")):
                    self.bridge._post({"test": "data"})
            else:
                with patch.object(self.bridge._urllib, "urlopen", side_effect=ConnectionError("refused")):
                    self.bridge._post({"test": "data"})

            mock_debug.assert_called_once()
            log_msg = mock_debug.call_args[0][0]
            assert "POST failed" in log_msg


class TestWriteStatusServerUnreachable(unittest.TestCase):
    """write_status() must not crash when the server is unreachable."""

    def setUp(self):
        self.bridge = _load_real_bridge()
        _reset_bridge(self.bridge)

    def tearDown(self):
        _reset_bridge(self.bridge)

    def test_write_status_no_crash_on_unreachable_server(self):
        """write_status completes without exception even if _post fails."""
        self.bridge.set_session_id("unreachable-test")

        state = {
            "phase": "dispatch",
            "active_task_id": None,
            "tasks": [],
            "results": {},
            "tokens_used": 0,
            "last_verdict": None,
            "synthesis_report": "",
            "validation": {},
            "coherence_report": {},
        }

        # Patch _post to raise an exception
        with patch.object(self.bridge, "_post", side_effect=ConnectionError("server down")):
            # write_status fires _post in a background thread, so we patch
            # threading.Thread to run synchronously and verify no crash
            original_thread = self.bridge.threading.Thread

            def sync_thread(*args, **kwargs):
                """Run the thread target synchronously for test determinism."""
                t = original_thread(*args, **kwargs)
                t.daemon = True
                # Just run the target directly
                target = kwargs.get("target") or (args[0] if args else None)
                targs = kwargs.get("args", ())
                if target:
                    try:
                        target(*targs)
                    except Exception:
                        pass  # In production, background threads swallow exceptions
                return MagicMock()  # Return a mock thread (start() is a no-op)

            with patch.object(self.bridge.threading, "Thread", side_effect=sync_thread):
                # This should not raise
                self.bridge.write_status(state)


class TestSessionDequeMaxlen(unittest.TestCase):
    """Session log deque respects maxlen limit."""

    def setUp(self):
        self.bridge = _load_real_bridge()
        _reset_bridge(self.bridge)

    def tearDown(self):
        _reset_bridge(self.bridge)

    def test_log_deque_respects_maxlen(self):
        """Adding more than maxlen events drops oldest entries."""
        self.bridge.set_session_id("maxlen-test")

        # Deque maxlen is 80 (MAX_LOG)
        for i in range(100):
            self.bridge.log_event(f"event-{i}")

        with self.bridge._state_lock:
            logs = self.bridge._session_logs["maxlen-test"]

        assert len(logs) == 80
        # Oldest entries (0-19) should have been dropped
        log_list = list(logs)
        assert log_list[0] == "event-20"
        assert log_list[-1] == "event-99"

    def test_transcript_deque_respects_maxlen(self):
        """Adding more than 200 transcript entries drops oldest."""
        self.bridge.set_session_id("transcript-maxlen")

        for i in range(220):
            self.bridge.add_transcript_entry(
                role="assistant",
                content=f"msg-{i}",
                agent="test_agent",
                task_id="TASK-001",
            )

        with self.bridge._state_lock:
            transcripts = self.bridge._session_transcripts["transcript-maxlen"]

        assert len(transcripts) == 200
        assert transcripts[0]["content"] == "msg-20"
        assert transcripts[-1]["content"] == "msg-219"


# ============================================================================
# 3. Edge Cases
# ============================================================================


class TestParsePlanNoTasks(unittest.TestCase):
    """parse_plan with a plan that has no tasks section."""

    def test_no_tasks_section_returns_empty_tasks(self):
        """A plan with agents/managers but no ### TASK- entries returns empty task list."""
        import tempfile

        plan_text = """# PROJECT PLAN: Empty Tasks Test

## Objective
A plan with no tasks at all.

## Sub-Agents
### AGENT: backend_engineer
- description: Builds backend services with FastAPI.
- tools: execute_code, write_file

## Managers
### MANAGER: engineering_director
- title: Engineering Director
- description: Senior engineering leader.
- expertise_blend: [API_design]
- oversees: [backend_engineer]
"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False
        ) as f:
            f.write(plan_text)
            path = f.name

        try:
            objective, managers, agents, tasks, reviewers = orchestrator.parse_plan(path)
            assert tasks == []
            assert len(agents) == 1
            assert len(managers) == 1
        finally:
            os.unlink(path)


class TestParsePlanBadAgentReference(unittest.TestCase):
    """parse_plan with a task referencing a non-existent agent."""

    def test_task_with_nonexistent_agent_still_parses(self):
        """Parser does not validate agent references - task gets the raw agent name."""
        import tempfile

        plan_text = """# PROJECT PLAN: Bad Agent Ref

## Objective
Test plan with an agent that does not exist.

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
- agent: ghost_agent
- description: This references a non-existent agent.
- task_type: [code]
- reviewers: []
- depends_on: []
"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False
        ) as f:
            f.write(plan_text)
            path = f.name

        try:
            objective, managers, agents, tasks, reviewers = orchestrator.parse_plan(path)
            assert len(tasks) == 1
            # The parser stores whatever agent name is in the plan
            assert tasks[0].agent == "ghost_agent"
            # Only the real agent is in the agents list
            agent_names = [a.name for a in agents]
            assert "ghost_agent" not in agent_names
            assert "real_agent" in agent_names
        finally:
            os.unlink(path)


class TestRoutingNoPendingTasks(unittest.TestCase):
    """When no tasks are pending (all done), master_node routes to synthesis."""

    def test_all_done_routes_to_synthesis(self):
        """All tasks with status='done' triggers phase='done' for synthesis."""
        tasks = [
            {
                "id": "TASK-001", "title": "Finished",
                "agent": "backend_engineer",
                "description": "Done task", "task_type": ["code"],
                "reviewers": [], "depends_on": [], "model": "",
                "tolerance": 0, "status": "done", "result": "output",
                "manager_notes": "", "reviewer_notes": "",
                "revision_count": 0,
            },
        ]
        state = make_base_state(tasks)
        state["results"] = {"TASK-001": "output"}

        result = orchestrator.master_node(state)

        assert result["phase"] == "done"

    def test_blocked_tasks_force_done(self):
        """Tasks blocked on unresolvable deps forces phase='done'."""
        tasks = [
            {
                "id": "TASK-001", "title": "Done",
                "agent": "backend_engineer",
                "description": "Completed", "task_type": ["code"],
                "reviewers": [], "depends_on": [], "model": "",
                "tolerance": 0, "status": "done", "result": "output",
                "manager_notes": "", "reviewer_notes": "",
                "revision_count": 0,
            },
            {
                "id": "TASK-002", "title": "Blocked",
                "agent": "backend_engineer",
                "description": "Needs TASK-999 which doesn't exist",
                "task_type": ["code"], "reviewers": [],
                "depends_on": ["TASK-999"], "model": "",
                "tolerance": 0, "status": "pending", "result": "",
                "manager_notes": "", "reviewer_notes": "",
                "revision_count": 0,
            },
        ]
        state = make_base_state(tasks)
        state["results"] = {"TASK-001": "output"}

        result = orchestrator.master_node(state)

        # Not all tasks are done/failed, but no task can proceed => forced done
        assert result["phase"] == "done"

    def test_empty_task_list_routes_to_done(self):
        """An empty task list should route to phase='done'."""
        state = make_base_state([])

        result = orchestrator.master_node(state)

        assert result["phase"] == "done"


class TestExtractTokensMalformed(unittest.TestCase):
    """extract_tokens returns 0 on malformed or missing response metadata."""

    def test_returns_zero_on_none_metadata(self):
        """response_metadata is None => returns 0."""
        mock_resp = MagicMock()
        mock_resp.response_metadata = None

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 0

    def test_returns_zero_on_missing_usage(self):
        """response_metadata has no 'usage' key => returns 0."""
        mock_resp = MagicMock()
        mock_resp.response_metadata = {"model": "gpt-4"}

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 0

    def test_returns_zero_on_empty_usage(self):
        """usage dict is empty => returns 0."""
        mock_resp = MagicMock()
        mock_resp.response_metadata = {"usage": {}}

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 0

    def test_returns_zero_on_exception(self):
        """If response_metadata is a non-dict type that breaks .get(), returns 0."""
        mock_resp = MagicMock()
        # Set response_metadata to a string, which has no .get() method
        # The try/except in extract_tokens should catch the AttributeError
        mock_resp.response_metadata = "not a dict"

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 0

    def test_returns_correct_tokens_on_valid_response(self):
        """Sanity check: valid metadata returns correct total_tokens."""
        mock_resp = MagicMock()
        mock_resp.response_metadata = {"usage": {"total_tokens": 1234}}

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 1234

    def test_returns_tokens_from_token_usage_key(self):
        """Some providers use 'token_usage' instead of 'usage'."""
        mock_resp = MagicMock()
        mock_resp.response_metadata = {"token_usage": {"total_tokens": 567}}

        result = orchestrator.extract_tokens(mock_resp)
        assert result == 567


if __name__ == "__main__":
    unittest.main()
