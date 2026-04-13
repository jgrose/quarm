"""Tests for the /api/questions and /api/plans/{id}/human_policy endpoints."""

import sys
import threading
import time

# Ensure we use the real tools module (conftest.py stubs it for other tests).
sys.modules.pop("tools", None)

from fastapi.testclient import TestClient  # noqa: E402


def _client():
    # Import here so the tools stub is gone before serve imports tools.
    import serve  # noqa: E402
    return TestClient(serve.app)


def test_post_question_resolves_pending_request():
    """POST /api/questions/{id} with {answer} should unblock a waiting request_question."""
    import tools
    tools.set_plan_policy("TEST_PLAN_A", policy="block")
    answers = []

    def caller():
        ans = tools.request_question(
            tc_id="endpoint-resolve",
            question="Q?",
            plan_id="TEST_PLAN_A",
        )
        answers.append(ans)

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.1)

    client = _client()
    r = client.post("/api/questions/endpoint-resolve", json={"answer": "the answer"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    t.join(timeout=2)
    assert answers == ["the answer"]


def test_get_questions_lists_pending():
    """GET /api/questions should return the list of pending questions while one is active."""
    import tools
    tools.set_plan_policy("TEST_PLAN_B", policy="block")

    def caller():
        tools.request_question(tc_id="endpoint-list", question="Waiting?", plan_id="TEST_PLAN_B")

    t = threading.Thread(target=caller, daemon=True)
    t.start()
    time.sleep(0.1)

    try:
        client = _client()
        r = client.get("/api/questions")
        assert r.status_code == 200
        pending = r.json()["pending"]
        assert any(p["id"] == "endpoint-list" for p in pending)
    finally:
        tools.resolve_question("endpoint-list", "")
        t.join(timeout=2)


def test_set_human_policy_endpoint():
    """POST /api/plans/{id}/human_policy should update the stored policy for the plan."""
    client = _client()
    r = client.post(
        "/api/plans/policy-test-plan/human_policy",
        json={"policy": "timeout", "timeout_s": 42},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    import tools
    stored = tools._get_plan_policy("policy-test-plan")
    assert stored == {"policy": "timeout", "timeout_s": 42}


def test_set_human_policy_rejects_unknown_policy():
    """Unknown policy values should return a 400."""
    client = _client()
    r = client.post(
        "/api/plans/bad-policy-plan/human_policy",
        json={"policy": "nonsense"},
    )
    assert r.status_code == 400


def test_questions_snapshot_endpoint_returns_pending():
    """GET /api/questions returns the current pending set (used for HTTP fallback)."""
    from fastapi.testclient import TestClient
    import serve
    import tools

    # Seed a pending question.
    tools._question_details["tc-snap-1"] = {
        "question": "Live?", "context": "", "agent": "a",
        "task_id": "T", "plan_id": "P", "received_at": 1,
    }
    try:
        client = TestClient(serve.app)
        resp = client.get("/api/questions")
        assert resp.status_code == 200
        data = resp.json()
        assert any(p["id"] == "tc-snap-1" for p in data["pending"])
    finally:
        tools._question_details.pop("tc-snap-1", None)
