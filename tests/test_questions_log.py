"""Append-only JSONL log for ask_human requests/resolutions."""
import json
import sys
import threading
from pathlib import Path

import pytest

# conftest.py stubs `tools` in sys.modules; pop it so we can import the real module.
sys.modules.pop("tools", None)


@pytest.fixture
def tmp_plans_dir(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    import tools
    monkeypatch.setattr(tools, "QUESTIONS_LOG_DIR", plans_dir, raising=False)
    monkeypatch.setattr(tools, "_plan_policies", {})
    monkeypatch.setattr(tools, "_pending_questions", {})
    monkeypatch.setattr(tools, "_question_answers", {})
    monkeypatch.setattr(tools, "_question_details", {})
    return plans_dir


def test_request_writes_jsonl_entry(tmp_plans_dir):
    import tools

    def caller():
        tools.request_question(
            "tc-log-1",
            "Should we proceed?",
            context="context text",
            agent="worker-a",
            task_id="TASK-001",
            plan_id="plan-xyz",
        )

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    # Give the thread a moment to write before we resolve it.
    import time; time.sleep(0.05)

    log_path = tmp_plans_dir / "plan-xyz_questions.jsonl"
    assert log_path.exists(), "JSONL log file should be created on first request"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"] == "request"
    assert entry["id"] == "tc-log-1"
    assert entry["agent"] == "worker-a"
    assert entry["task_id"] == "TASK-001"
    assert entry["question"] == "Should we proceed?"
    assert isinstance(entry["ts"], int)

    tools.resolve_question("tc-log-1", "yes")
    t.join(timeout=1.0)


def test_resolve_writes_jsonl_entry(tmp_plans_dir):
    import tools

    def caller():
        tools.request_question(
            "tc-log-2", "Do the thing?", plan_id="plan-xyz",
            agent="worker-b", task_id="TASK-002",
        )

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    import time; time.sleep(0.05)
    tools.resolve_question("tc-log-2", "affirmative")
    t.join(timeout=1.0)

    log_path = tmp_plans_dir / "plan-xyz_questions.jsonl"
    entries = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    types = [e["type"] for e in entries]
    assert types == ["request", "resolve"]
    resolve = entries[1]
    assert resolve["id"] == "tc-log-2"
    assert resolve["answer"] == "affirmative"


def test_timeout_writes_jsonl_entry(tmp_plans_dir):
    import tools
    tools.set_plan_policy("plan-xyz", "timeout", timeout_s=0)

    def caller():
        tools.request_question(
            "tc-log-3", "Timeout me", plan_id="plan-xyz",
            agent="worker-c", task_id="TASK-003",
        )

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    t.join(timeout=2.0)

    log_path = tmp_plans_dir / "plan-xyz_questions.jsonl"
    entries = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    types = [e["type"] for e in entries]
    assert types == ["request", "timeout"]
    assert entries[1]["id"] == "tc-log-3"
