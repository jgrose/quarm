"""Tests for the ask_human tool and its request/resolve primitives in tools.py."""

import sys
import threading
import time

# conftest.py stubs `tools` in sys.modules; pop it so we can import the real module.
sys.modules.pop("tools", None)

import tools  # noqa: E402


def test_resolve_question_unblocks_request_question():
    """resolve_question should unblock a waiting request_question and return the answer."""
    tc_id = "test-resolve-unblock"
    answers = []

    def caller():
        ans = tools.request_question(
            tc_id=tc_id,
            question="What should I do?",
            context="",
            agent="tester",
            task_id="T1",
            plan_id="PLAN_BLOCK",
        )
        answers.append(ans)

    tools.set_plan_policy("PLAN_BLOCK", policy="block")
    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.1)
    tools.resolve_question(tc_id, "do the thing")
    t.join(timeout=2)
    assert not t.is_alive(), "Thread should have unblocked after resolve_question"
    assert answers == ["do the thing"]


def test_request_question_returns_sentinel_on_timeout():
    """policy='timeout' returns the timeout sentinel when no resolution arrives."""
    tools.set_plan_policy("PLAN_TIMEOUT", policy="timeout", timeout_s=1)
    ans = tools.request_question(
        tc_id="timeout-test",
        question="any",
        context="",
        agent="tester",
        task_id="T1",
        plan_id="PLAN_TIMEOUT",
    )
    assert "NO HUMAN RESPONSE" in ans


def test_pending_questions_listed_while_blocked():
    """get_pending_questions exposes the active request while a caller is waiting."""
    tc_id = "pending-test"
    tools.set_plan_policy("PLAN_PENDING", policy="block")

    def caller():
        tools.request_question(
            tc_id=tc_id,
            question="Q?",
            context="ctx",
            agent="a",
            task_id="T",
            plan_id="PLAN_PENDING",
        )

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.1)
    try:
        pending = tools.get_pending_questions()
        assert any(p["id"] == tc_id for p in pending), f"Expected {tc_id} in pending, got {pending}"
    finally:
        tools.resolve_question(tc_id, "")
        t.join(timeout=2)


def test_unknown_plan_id_defaults_to_block():
    """If no policy was set for the plan, request_question should default to blocking."""
    tc_id = "unknown-plan-test"
    answers = []

    def caller():
        ans = tools.request_question(
            tc_id=tc_id,
            question="Q?",
            context="",
            agent="a",
            task_id="T",
            plan_id="NEVER_REGISTERED",
        )
        answers.append(ans)

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive(), "Should block indefinitely with unknown plan_id"
    tools.resolve_question(tc_id, "ok")
    t.join(timeout=2)
    assert answers == ["ok"]


def test_ask_human_tool_invokes_request_question(monkeypatch):
    """The @tool-decorated ask_human reads _tool_context and delegates to request_question."""
    captured = {}

    def fake_request_question(tc_id, question, context="", agent="", task_id="", plan_id=""):
        captured.update({"question": question, "context": context, "agent": agent,
                         "task_id": task_id, "plan_id": plan_id})
        return "stub-answer"

    monkeypatch.setattr(tools, "request_question", fake_request_question)
    tools.set_tool_context(plan_id="P1", task_id="T1", agent="alice")
    result = tools.ask_human.invoke({"question": "help?", "context": "details"})
    assert result == "stub-answer"
    assert captured["question"] == "help?"
    assert captured["context"] == "details"
    assert captured["plan_id"] == "P1"
    assert captured["task_id"] == "T1"
    assert captured["agent"] == "alice"


def test_broadcast_called_on_request_and_resolve(monkeypatch):
    """request_question and resolve_question should broadcast questions_snapshot."""
    import tools
    calls = []

    def fake_post(url, data=None, timeout=None, headers=None):
        import json as _json
        # production passes a Request object as the first positional arg
        if data is None and hasattr(url, "data"):
            data = url.data
        try:
            payload = _json.loads(data.decode() if hasattr(data, "decode") else data)
        except Exception:
            payload = {}
        calls.append(payload)

        class _Stub:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b""
        return _Stub()

    monkeypatch.setattr("urllib.request.urlopen", fake_post)

    def caller():
        tools.request_question("tc-bc-1", "Q?", plan_id="plan-bc",
                               agent="a", task_id="T")

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    import time; time.sleep(0.05)
    tools.resolve_question("tc-bc-1", "A")
    t.join(timeout=1.0)

    types = [c.get("type") for c in calls]
    # Exact count per-call isn't the contract; what matters is at least one snapshot
    # on request and one on resolve alongside the existing request/resolved events.
    assert "questions_snapshot" in types
    snapshots = [c for c in calls if c.get("type") == "questions_snapshot"]
    assert any("pending" in s for s in snapshots)
    # The resolve must produce a snapshot whose pending list no longer contains tc-bc-1.
    final = snapshots[-1]
    ids = [p.get("id") for p in final["pending"]]
    assert "tc-bc-1" not in ids


def test_multi_question_preserves_request_order():
    """get_pending_questions returns entries in insertion order."""
    import tools

    threads = []
    for i in range(3):
        tc = f"tc-order-{i}"
        def caller(tc=tc, i=i):
            tools.request_question(tc, f"Q{i}?", plan_id="plan-order",
                                   agent=f"a{i}", task_id=f"T{i}")
        t = threading.Thread(target=caller, daemon=True)
        t.start()
        threads.append((tc, t))

    import time; time.sleep(0.1)
    try:
        pending = tools.get_pending_questions()
        ids = [p["id"] for p in pending if p["id"].startswith("tc-order-")]
        assert ids == ["tc-order-0", "tc-order-1", "tc-order-2"]
    finally:
        for tc, t in threads:
            tools.resolve_question(tc, "")
            t.join(timeout=1.0)


def test_resolve_nonactive_question_leaves_others_pending():
    """Resolving one of many pending questions must not disturb the rest."""
    import tools

    for i in range(2):
        tc = f"tc-leave-{i}"
        def caller(tc=tc, i=i):
            tools.request_question(tc, f"Q{i}?", plan_id="plan-leave",
                                   agent=f"a{i}", task_id=f"T{i}")
        threading.Thread(target=caller, daemon=True).start()

    import time; time.sleep(0.05)
    tools.resolve_question("tc-leave-0", "ok")
    time.sleep(0.05)
    remaining = {p["id"] for p in tools.get_pending_questions()}
    assert "tc-leave-1" in remaining
    assert "tc-leave-0" not in remaining

    tools.resolve_question("tc-leave-1", "ok")
