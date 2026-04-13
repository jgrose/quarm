# Ask-Human Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-popup `ask_human` behavior with a queue that shows a persistent ASKS badge, glowing in-view agent indicators, a browsable queue panel, localStorage-backed draft answers, and an append-only JSONL audit log.

**Architecture:** Server-side changes are additive — a new JSONL log plus a new `questions_snapshot` WebSocket message broadcast whenever the pending set changes. Client keeps a `_pendingQuestions` Map as the single source of truth for banner, badge, queue panel, and agent glow. The orchestrator already runs as a background thread inside `serve.py`, so `tools._pending_questions` is already shared between the agent thread and HTTP handlers.

**Tech Stack:** Python (FastAPI, threading), vanilla JS (canvas + DOM), CSS animations, pytest.

**Related spec:** [docs/superpowers/specs/2026-04-13-ask-human-queue-design.md](../specs/2026-04-13-ask-human-queue-design.md)

---

## Task 1: JSONL question log writer in tools.py

Append one JSON line per `ask_human` request / resolve / timeout to `plans/{plan_id}_questions.jsonl`. This is the audit trail the queue panel will later use for a "recent" view and that debugging will rely on.

**Files:**
- Modify: `tools.py` (around `request_question`, `resolve_question`, timeout branch)
- Test: `tests/test_questions_log.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_questions_log.py`:

```python
"""Append-only JSONL log for ask_human requests/resolutions."""
import json
import threading
from pathlib import Path

import pytest


@pytest.fixture
def tmp_plans_dir(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    import tools
    monkeypatch.setattr(tools, "QUESTIONS_LOG_DIR", plans_dir, raising=False)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_questions_log.py -v`
Expected: All three FAIL (JSONL file missing / `QUESTIONS_LOG_DIR` attribute missing).

- [ ] **Step 3: Add the JSONL log writer to `tools.py`**

Near the top of `tools.py`, with the other module-level constants, add:

```python
QUESTIONS_LOG_DIR = Path("plans")
```

Below `QUESTION_TIMEOUT_SENTINEL` and the other `_question_*` state, add the helper:

```python
def _append_question_log(plan_id: str, entry: dict) -> None:
    """Atomic-append one JSON line to plans/{plan_id}_questions.jsonl."""
    if not plan_id:
        return
    try:
        QUESTIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = QUESTIONS_LOG_DIR / f"{plan_id}_questions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.debug(f"[ask_human] JSONL log write failed: {exc}")
```

Then inside `request_question`, right after `_question_details[tc_id] = {...}`, add:

```python
_append_question_log(plan_id, {
    "ts": int(time.time()),
    "type": "request",
    "id": tc_id,
    "agent": agent,
    "task_id": task_id,
    "question": question,
})
```

Inside `request_question`, at the end where the timeout sentinel is returned (the `if not got or answer is None:` branch), replace the `return QUESTION_TIMEOUT_SENTINEL` line so it logs first:

```python
if not got or answer is None:
    _append_question_log(plan_id, {
        "ts": int(time.time()),
        "type": "timeout",
        "id": tc_id,
    })
    return QUESTION_TIMEOUT_SENTINEL
```

Inside `resolve_question`, right after `_question_answers[tc_id] = answer`, add:

```python
details = _question_details.get(tc_id, {})
_append_question_log(details.get("plan_id", ""), {
    "ts": int(time.time()),
    "type": "resolve",
    "id": tc_id,
    "answer": answer,
})
```

Also add these two imports at the top of `tools.py` if not already present: `import json`, `import time` (check first — don't duplicate).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_questions_log.py -v`
Expected: 3 passed.

Run: `pytest tests/test_ask_human.py tests/test_question_endpoints.py -v`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_questions_log.py
git commit -m "Append ask_human lifecycle events to plans/{plan_id}_questions.jsonl"
```

---

## Task 2: Broadcast `questions_snapshot` on every change

Every time the pending question set changes (new request, resolve, timeout), emit a full-list `questions_snapshot` WebSocket message so clients can re-render badge and queue without round-tripping the REST endpoint.

**Files:**
- Modify: `tools.py` (add broadcast helper; call it from request/resolve/timeout)
- Test: extend `tests/test_ask_human.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ask_human.py`:

```python
def test_broadcast_called_on_request_and_resolve(monkeypatch):
    """request_question and resolve_question should broadcast questions_snapshot."""
    import tools
    calls = []

    def fake_post(url, data=None, timeout=None, headers=None):
        import json as _json
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ask_human.py::test_broadcast_called_on_request_and_resolve -v`
Expected: FAIL — `questions_snapshot` is never in the list of broadcast types.

- [ ] **Step 3: Implement the broadcaster**

In `tools.py`, below `_append_question_log`, add:

```python
def _broadcast_questions_snapshot() -> None:
    """POST a questions_snapshot event to serve.py for WS fan-out."""
    try:
        pending = [
            {
                "id": k,
                "plan_id": v.get("plan_id", ""),
                "agent": v.get("agent", ""),
                "task_id": v.get("task_id", ""),
                "question": (v.get("question") or "")[:2000],
                "context": (v.get("context") or "")[:2000],
                "received_at": v.get("received_at", 0),
            }
            for k, v in _question_details.items()
        ]
        payload = json.dumps({"type": "questions_snapshot", "pending": pending}).encode()
        port = os.environ.get("NORT_PORT", os.environ.get("QUARM_PORT", "8000"))
        req = urllib.request.Request(
            f"http://localhost:{port}/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # fire-and-forget
```

Add `import urllib.request` at the top of `tools.py` if not already imported at module level (the existing code imports it inline; move to module scope or keep inline in the new function for parity — keep inline to minimize diff: put `import urllib.request` as the first line inside `_broadcast_questions_snapshot`).

In `request_question`, after `_question_details[tc_id] = {...}`, add a `received_at` field:

```python
_question_details[tc_id] = {
    "question": question, "context": context,
    "agent": agent, "task_id": task_id, "plan_id": plan_id,
    "received_at": int(time.time()),
}
```

Call `_broadcast_questions_snapshot()` in three places:

1. In `request_question`, right after the existing `question_request` POST block.
2. In `resolve_question`, right after the existing `question_resolved` POST block.
3. In `request_question`, right before `return QUESTION_TIMEOUT_SENTINEL` (so the timeout path also broadcasts).

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ask_human.py -v`
Expected: all pass including the new `test_broadcast_called_on_request_and_resolve`.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_ask_human.py
git commit -m "Broadcast questions_snapshot on every ask_human state change"
```

---

## Task 3: Replay snapshot on WebSocket connect

A client that connects mid-run must see the current pending set immediately, without having to wait for the next state change.

**Files:**
- Modify: `serve.py` (WebSocket connect handler)
- Test: extend `tests/test_question_endpoints.py`

- [ ] **Step 1: Locate the WebSocket connect path**

Run: `grep -n 'websocket' serve.py | head -20`

Find the `async def websocket_endpoint(...)` (or similarly named) handler. Identify the line that currently sends initial state to a new client.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_question_endpoints.py`:

```python
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
```

- [ ] **Step 3: Run test to verify current behavior**

Run: `pytest tests/test_question_endpoints.py::test_questions_snapshot_endpoint_returns_pending -v`

This should already pass (the endpoint exists). If it fails, fix the import path / endpoint URL to match current code.

- [ ] **Step 4: Add snapshot send on WS connect**

In `serve.py`, in the WebSocket connect handler (look for `await websocket.accept()` — there's only one), right after the existing code that replays session states to the new connection, add:

```python
try:
    from tools import get_pending_questions
    await websocket.send_text(json.dumps({
        "type": "questions_snapshot",
        "pending": get_pending_questions(),
    }))
except Exception:
    pass
```

Make sure `json` is imported at the top of `serve.py` (it already is).

- [ ] **Step 5: Run all server-side tests**

Run: `pytest tests/test_ask_human.py tests/test_question_endpoints.py tests/test_questions_log.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add serve.py tests/test_question_endpoints.py
git commit -m "Replay pending questions snapshot on WebSocket connect"
```

---

## Task 4: Client-side question state & snapshot handler

The client keeps a `_pendingQuestions` Map as the single source of truth. `_activeQuestionId` tracks which question is currently shown in the banner.

**Files:**
- Modify: `templates/scripts/panels.js` (add module-level state, render trigger)
- Modify: `templates/scripts/websocket.js` (handle `questions_snapshot`)

- [ ] **Step 1: Add state + render trigger in `panels.js`**

Near the top of `templates/scripts/panels.js`, below the `_planPolicyCache` var, add:

```javascript
// ── Ask-human queue state ────────────────────────────────────────────────────
// Single source of truth for banner, badge, queue panel, and agent glow.
var _pendingQuestions = new Map();         // tool_call_id → question record
var _activeQuestionId = null;              // id currently shown in the banner

function _rerenderQuestionUI() {
  if (typeof renderAsksBadge === 'function') renderAsksBadge();
  if (typeof renderAsks === 'function') renderAsks();
  if (typeof refreshActiveBanner === 'function') refreshActiveBanner();
}

function applyQuestionsSnapshot(pending) {
  var next = new Map();
  for (var i = 0; i < pending.length; i++) {
    var q = pending[i];
    if (q && q.id) next.set(q.id, q);
  }
  _pendingQuestions = next;
  // If the active id disappeared, clear it.
  if (_activeQuestionId && !_pendingQuestions.has(_activeQuestionId)) {
    _activeQuestionId = null;
  }
  _rerenderQuestionUI();
}
```

(The `renderAsksBadge`, `renderAsks`, and `refreshActiveBanner` functions don't exist yet — they're added in later tasks. The `typeof ... === 'function'` guard makes the handler safe to call before those land.)

- [ ] **Step 2: Wire the WebSocket handler**

In `templates/scripts/websocket.js`, just before the `if (data.type === 'question_request')` block (around line 142), add:

```javascript
if (data.type === 'questions_snapshot') {
  if (typeof applyQuestionsSnapshot === 'function') {
    applyQuestionsSnapshot(data.pending || []);
  }
  return;
}
```

Also modify the existing `question_request` handler so it *also* upserts into the Map (in case snapshots lag):

```javascript
if (data.type === 'question_request') {
  if (typeof _pendingQuestions !== 'undefined' && data.id) {
    _pendingQuestions.set(data.id, {
      id: data.id,
      plan_id: data.plan_id || '',
      agent: data.agent || '',
      task_id: data.task_id || '',
      question: data.question || '',
      context: data.context || '',
      received_at: Math.floor(Date.now() / 1000),
    });
    if (typeof _rerenderQuestionUI === 'function') _rerenderQuestionUI();
  }
  showQuestion(data);
  return;
}
```

And the `question_resolved` handler must remove from the Map:

```javascript
if (data.type === 'question_resolved') {
  if (typeof _pendingQuestions !== 'undefined' && data.id) {
    _pendingQuestions.delete(data.id);
    if (_activeQuestionId === data.id) _activeQuestionId = null;
    if (typeof _rerenderQuestionUI === 'function') _rerenderQuestionUI();
  }
  hideQuestion();
  return;
}
```

- [ ] **Step 3: Smoke-test in the browser**

Run the server: `python serve.py`
Open `http://localhost:8000/` in a browser. Open the devtools console and check there are no JS errors on page load. In the console, run:

```javascript
applyQuestionsSnapshot([{id:'tc1', agent:'a', plan_id:'p', question:'Q?', received_at:1}]);
console.log(_pendingQuestions.size);  // expect 1
applyQuestionsSnapshot([]);
console.log(_pendingQuestions.size);  // expect 0
```

Expected: no errors, sizes match.

- [ ] **Step 4: Commit**

```bash
git add templates/scripts/panels.js templates/scripts/websocket.js
git commit -m "Add client-side pending-question Map and snapshot handler"
```

---

## Task 5: Draft-answer localStorage helper

Save the banner / queue textarea contents as the user types so drafts survive refresh and server restart. The key hashes the question text so the exact same question matches across reruns.

**Files:**
- Modify: `templates/scripts/panels.js`

- [ ] **Step 1: Add the helper**

Below the `applyQuestionsSnapshot` function added in Task 4, add:

```javascript
// ── Draft answers ───────────────────────────────────────────────────────────
// Keyed by plan+question-hash+agent so the same question across runs matches.

function _hashStr(s) {
  // Tiny non-crypto hash; collisions are fine for draft matching.
  var h = 5381;
  for (var i = 0; i < s.length; i++) {
    h = ((h << 5) + h) + s.charCodeAt(i);
    h |= 0;
  }
  return (h >>> 0).toString(16).slice(0, 12);
}

function _draftKey(q) {
  if (!q) return null;
  return 'nort_ask_draft::' + (q.plan_id || '') + '::' +
         _hashStr(q.question || '') + '::' + (q.agent || '');
}

function loadDraft(q) {
  try {
    var k = _draftKey(q);
    return k ? (localStorage.getItem(k) || '') : '';
  } catch (e) { return ''; }
}

function saveDraft(q, value) {
  try {
    var k = _draftKey(q);
    if (!k) return;
    if (value) localStorage.setItem(k, value);
    else localStorage.removeItem(k);
  } catch (e) { /* quota or disabled — silently ignore */ }
}

function clearDraft(q) { saveDraft(q, ''); }

// Debounced wrapper for typing-driven saves.
var _draftSaveTimer = null;
function scheduleSaveDraft(q, value) {
  if (_draftSaveTimer) clearTimeout(_draftSaveTimer);
  _draftSaveTimer = setTimeout(function () { saveDraft(q, value); }, 400);
}
```

- [ ] **Step 2: Sanity-check in the console**

Reload the page, then in devtools:

```javascript
var q = {plan_id:'p', agent:'a', question:'Test?'};
saveDraft(q, 'hello');
loadDraft(q);   // 'hello'
clearDraft(q);
loadDraft(q);   // ''
```

Expected outputs as noted.

- [ ] **Step 3: Commit**

```bash
git add templates/scripts/panels.js
git commit -m "Add localStorage draft helpers for ask_human answers"
```

---

## Task 6: Banner dismiss (X button + Escape key)

Let the user close the popup without answering. The question stays in the pending Map and remains reachable via the badge / queue.

**Files:**
- Modify: `templates/components/question_banner.html`
- Modify: `templates/scripts/panels.js`
- Modify: `templates/styles/base.css`

- [ ] **Step 1: Add the X button and carousel scaffolding to the banner markup**

Replace the contents of `templates/components/question_banner.html` with:

```html
<div id="questionBanner" class="hidden">
  <button class="question-close" title="Close (Esc)" onclick="dismissQuestion()">&times;</button>
  <div class="question-body">
    <div class="question-header">
      <span class="question-label">AGENT NEEDS INPUT</span>
      <span id="questionCarousel" class="question-carousel hidden">
        <button class="carousel-btn" onclick="carouselPrev()" title="Previous">&lsaquo;</button>
        <span id="questionCarouselText">1 of 1</span>
        <button class="carousel-btn" onclick="carouselNext()" title="Next">&rsaquo;</button>
      </span>
    </div>
    <div id="questionAgent" class="question-agent"></div>
    <div id="questionText" class="question-text"></div>
    <div id="questionContext" class="question-context"></div>
    <div class="question-input-row">
      <textarea id="questionAnswerInput"
                placeholder="Type your answer and press Ctrl+Enter to submit..."
                rows="2"></textarea>
      <button class="btn-submit-answer"
              onclick="submitQuestionAnswer(document.getElementById('questionBanner').dataset.toolCallId, document.getElementById('questionAnswerInput').value)">
        SUBMIT
      </button>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Add `dismissQuestion` + Escape key handler in `panels.js`**

In `templates/scripts/panels.js`, find the existing `hideQuestion()` function. Below it, add:

```javascript
function dismissQuestion() {
  // User closed the banner without answering — question stays in the queue.
  var banner = document.getElementById('questionBanner');
  if (banner) banner.classList.add('hidden');
  _activeQuestionId = null;
  _rerenderQuestionUI();
}

// Escape dismisses the banner if it's visible.
document.addEventListener('keydown', function (e) {
  if (e.key !== 'Escape') return;
  var banner = document.getElementById('questionBanner');
  if (banner && !banner.classList.contains('hidden')) {
    e.preventDefault();
    dismissQuestion();
  }
});
```

Also, inside the existing `showQuestion(data)` function, at the very top, set the active id:

```javascript
function showQuestion(data) {
  if (data && data.id) _activeQuestionId = data.id;
  // ... rest unchanged
```

- [ ] **Step 3: Add `refreshActiveBanner` helper**

Below `dismissQuestion`, add:

```javascript
function refreshActiveBanner() {
  var banner = document.getElementById('questionBanner');
  if (!banner) return;
  if (!_activeQuestionId || !_pendingQuestions.has(_activeQuestionId)) {
    // Nothing to show; hide silently (not a dismiss).
    banner.classList.add('hidden');
    return;
  }
  var q = _pendingQuestions.get(_activeQuestionId);
  banner.classList.remove('hidden');
  banner.dataset.toolCallId = _activeQuestionId;
  var agentEl = document.getElementById('questionAgent');
  if (agentEl) {
    var task = q.task_id ? ' · ' + q.task_id : '';
    agentEl.textContent = (q.agent || 'agent') + task;
  }
  var textEl = document.getElementById('questionText');
  if (textEl) textEl.textContent = q.question || '';
  var ctxEl = document.getElementById('questionContext');
  if (ctxEl) ctxEl.textContent = q.context || '';
  var input = document.getElementById('questionAnswerInput');
  if (input && document.activeElement !== input) {
    input.value = loadDraft(q);
    input.oninput = function () { scheduleSaveDraft(q, input.value); };
  }
}
```

- [ ] **Step 4: Add CSS for the X button**

In `templates/styles/base.css`, find the existing question-banner styles (search for `#questionBanner` or `.question-body`) and add these rules after the existing banner block:

```css
#questionBanner .question-close {
  position: absolute;
  top: 6px;
  right: 8px;
  background: transparent;
  border: none;
  color: rgba(255,255,255,0.6);
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
  padding: 2px 8px;
  line-height: 1;
}
#questionBanner .question-close:hover {
  color: #ff5f8c;
}
#questionBanner .question-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
#questionBanner .question-carousel {
  font-size: 11px;
  color: rgba(255,255,255,0.7);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
#questionBanner .question-carousel.hidden { display: none; }
#questionBanner .carousel-btn {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.3);
  color: #fff;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  padding: 0 6px;
  border-radius: 3px;
}
#questionBanner .carousel-btn:hover {
  background: rgba(255,255,255,0.1);
}
```

- [ ] **Step 5: Manual smoke**

Start `serve.py` and run a plan that calls `ask_human` (or in devtools: `applyQuestionsSnapshot([{id:'t1',plan_id:'p',agent:'a',question:'Test?',received_at:1}]); _activeQuestionId='t1'; refreshActiveBanner();`). Confirm:
- Banner shows with X button top-right.
- Clicking X hides the banner.
- Pressing Escape with the banner visible hides it.
- The `_pendingQuestions` Map still has the question after dismiss (check `_pendingQuestions.size` in console).

- [ ] **Step 6: Commit**

```bash
git add templates/components/question_banner.html templates/scripts/panels.js templates/styles/base.css
git commit -m "Let users dismiss ask_human banner with X or Escape without answering"
```

---

## Task 7: Carousel navigation between pending questions

When there are multiple pending questions, the banner shows `‹ N of M ›` controls to step through them in request order without dismissing.

**Files:**
- Modify: `templates/scripts/panels.js`

- [ ] **Step 1: Add carousel navigation functions**

Below `refreshActiveBanner`, add:

```javascript
function _orderedPendingIds() {
  var arr = [];
  _pendingQuestions.forEach(function (v, k) { arr.push({id: k, at: v.received_at || 0}); });
  arr.sort(function (a, b) { return a.at - b.at; });
  return arr.map(function (x) { return x.id; });
}

function carouselPrev() { _carouselStep(-1); }
function carouselNext() { _carouselStep(+1); }

function _carouselStep(delta) {
  var ids = _orderedPendingIds();
  if (ids.length <= 1) return;
  var idx = ids.indexOf(_activeQuestionId);
  if (idx < 0) idx = 0;
  var next = (idx + delta + ids.length) % ids.length;
  _activeQuestionId = ids[next];
  refreshActiveBanner();
  _updateCarouselUI();
}

function _updateCarouselUI() {
  var el = document.getElementById('questionCarousel');
  var txt = document.getElementById('questionCarouselText');
  if (!el || !txt) return;
  var ids = _orderedPendingIds();
  if (ids.length <= 1) {
    el.classList.add('hidden');
    return;
  }
  var idx = ids.indexOf(_activeQuestionId);
  if (idx < 0) idx = 0;
  el.classList.remove('hidden');
  txt.textContent = (idx + 1) + ' of ' + ids.length;
}
```

- [ ] **Step 2: Wire carousel refresh into `refreshActiveBanner`**

Modify `refreshActiveBanner`: at the end of the function (after the input.oninput assignment), add:

```javascript
_updateCarouselUI();
```

- [ ] **Step 3: Manual smoke**

In devtools on the running dashboard:

```javascript
applyQuestionsSnapshot([
  {id:'a', plan_id:'p1', agent:'A', question:'First?', received_at:1},
  {id:'b', plan_id:'p1', agent:'B', question:'Second?', received_at:2},
  {id:'c', plan_id:'p2', agent:'C', question:'Third?', received_at:3},
]);
_activeQuestionId = 'a';
refreshActiveBanner();
```

- Banner shows "1 of 3" with both arrows visible.
- Click right arrow → banner shows "Second?" / "2 of 3".
- Click right arrow again → "Third?" / "3 of 3".
- Click right again → wraps to "First?" / "1 of 3".

- [ ] **Step 4: Commit**

```bash
git add templates/scripts/panels.js
git commit -m "Carousel nav between pending ask_human questions in banner"
```

---

## Task 8: Toast announcement for arrivals while banner is open

A new `question_request` arriving while the banner is already showing a different question fires a non-blocking toast instead of stealing focus. Sound and OS notification are suppressed in this case.

**Files:**
- Modify: `templates/components/question_banner.html` (add toast container)
- Modify: `templates/scripts/panels.js` (toast logic; update notification gating)
- Modify: `templates/styles/base.css` (toast styling)

- [ ] **Step 1: Add toast container markup**

In `templates/components/question_banner.html`, add this div *below* the closing `</div>` of `#questionBanner`:

```html
<div id="questionToastStack"></div>
```

- [ ] **Step 2: Add toast CSS to `base.css`**

After the carousel styles added in Task 6:

```css
#questionToastStack {
  position: fixed;
  right: 16px;
  bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 9000;
  pointer-events: none;
}
.ask-toast {
  pointer-events: auto;
  background: rgba(20, 20, 30, 0.92);
  border-left: 3px solid #ff5f8c;
  color: #fff;
  padding: 10px 14px;
  font-size: 12px;
  max-width: 320px;
  box-shadow: 0 4px 18px rgba(0,0,0,0.5);
  animation: askToastIn 200ms ease-out, askToastOut 300ms ease-in 3.7s both;
  cursor: pointer;
}
.ask-toast:hover { border-left-color: #ffc83c; }
@keyframes askToastIn {
  from { transform: translateX(20px); opacity: 0; }
  to   { transform: translateX(0); opacity: 1; }
}
@keyframes askToastOut {
  to { transform: translateX(20px); opacity: 0; }
}
```

- [ ] **Step 3: Add toast function in `panels.js`**

Below `_updateCarouselUI`:

```javascript
function showAskToast(q) {
  var stack = document.getElementById('questionToastStack');
  if (!stack) return;
  var el = document.createElement('div');
  el.className = 'ask-toast';
  var total = _pendingQuestions.size;
  el.innerHTML = '<strong>' + escapeHtml(q.agent || 'agent') +
                 '</strong> also needs help — ' + total + ' in queue';
  el.onclick = function () {
    _activeQuestionId = q.id;
    refreshActiveBanner();
    el.remove();
  };
  stack.appendChild(el);
  setTimeout(function () { if (el.parentNode) el.remove(); }, 4000);
}
```

- [ ] **Step 4: Update notification gating in `showQuestion`**

Modify the existing `showQuestion(data)` in `panels.js`. The current code plays audio / fires a browser Notification every time. Replace the body so those only fire when the banner was previously hidden:

At the top of `showQuestion(data)`, replace the existing function body with:

```javascript
function showQuestion(data) {
  if (!data || !data.id) return;
  var banner = document.getElementById('questionBanner');
  var wasHidden = !banner || banner.classList.contains('hidden');

  // Always mirror into the Map — receiver may predate a snapshot.
  if (typeof _pendingQuestions !== 'undefined') {
    _pendingQuestions.set(data.id, {
      id: data.id,
      plan_id: data.plan_id || '',
      agent: data.agent || '',
      task_id: data.task_id || '',
      question: data.question || '',
      context: data.context || '',
      received_at: Math.floor(Date.now() / 1000),
    });
  }

  if (wasHidden) {
    // Banner was idle — raise it for this question with full fanfare.
    _activeQuestionId = data.id;
    refreshActiveBanner();
    try {
      if (typeof config !== 'undefined' && config.browserAlerts &&
          typeof Notification !== 'undefined' &&
          Notification.permission === 'granted') {
        new Notification('Agent needs input', {
          body: (data.question || '').slice(0, 200),
        });
      }
    } catch (e) { /* ignore */ }
    try {
      if (typeof config !== 'undefined' && config.sound && config.questionSound &&
          typeof playQuestion === 'function') {
        playQuestion();
      }
    } catch (e) { /* ignore */ }
  } else if (data.id !== _activeQuestionId) {
    // Banner already busy with a different question — toast only.
    showAskToast(_pendingQuestions.get(data.id));
  }

  _rerenderQuestionUI();
}
```

- [ ] **Step 5: Manual smoke**

In devtools with the page open:

```javascript
// 1. New arrival with banner hidden — should fire the banner (and sound if enabled).
showQuestion({id:'t1', plan_id:'p', agent:'Aleph', question:'Hidden?'});

// 2. New arrival with banner showing a different question — should toast, not steal focus.
showQuestion({id:'t2', plan_id:'p', agent:'Bet', question:'Second?'});
```

Expected: After step 1 the banner shows "Hidden?". After step 2 the banner still shows "Hidden?" but a toast slides in bottom-right reading "**Bet** also needs help — 2 in queue" and auto-dismisses after ~4 seconds. Clicking the toast switches the banner to the second question.

- [ ] **Step 6: Commit**

```bash
git add templates/components/question_banner.html templates/scripts/panels.js templates/styles/base.css
git commit -m "Toast for concurrent ask_human arrivals; suppress audio while banner open"
```

---

## Task 9: ASKS top-bar button + pulsing badge

A new always-visible button in the top bar shows the pending count and pulses when > 0. Clicking toggles the ASKS panel (implemented in Task 10). Keyboard shortcut `K` toggles it too.

**Files:**
- Modify: `templates/components/top_bar.html`
- Modify: `templates/scripts/panels.js`
- Modify: `templates/styles/base.css`

- [ ] **Step 1: Add the button**

In `templates/components/top_bar.html`, insert this line *between* the `QUEUE` and `LEDGER` buttons (after the `QUEUE` button):

```html
<button id="asksBtn" onclick="toggleAsksPanel()" title="Pending agent questions (K)">
  ASKS (<span id="asksCount">0</span>)
</button>
```

- [ ] **Step 2: Add CSS**

In `templates/styles/base.css`, below the toast styles from Task 8:

```css
#asksBtn { position: relative; }
#asksBtn.has-pending #asksCount {
  color: #ff5f8c;
  font-weight: 900;
}
#asksBtn.has-pending {
  animation: asksPulse 1.8s ease-in-out infinite;
}
@keyframes asksPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(255,95,140,0.0); }
  50%     { box-shadow: 0 0 12px 2px rgba(255,95,140,0.6); }
}
```

- [ ] **Step 3: Add `renderAsksBadge` + `toggleAsksPanel` + keyboard shortcut**

In `templates/scripts/panels.js`, below the toast function:

```javascript
function renderAsksBadge() {
  var btn = document.getElementById('asksBtn');
  var count = document.getElementById('asksCount');
  if (!btn || !count) return;
  var n = _pendingQuestions.size;
  count.textContent = n;
  if (n > 0) btn.classList.add('has-pending');
  else btn.classList.remove('has-pending');
}

function toggleAsksPanel() {
  var p = document.getElementById('asksPanel');
  if (!p) return;
  p.classList.toggle('hidden');
  if (!p.classList.contains('hidden') && typeof renderAsks === 'function') {
    renderAsks();
  }
}

// Keyboard shortcut: K toggles ASKS (unless user is typing).
document.addEventListener('keydown', function (e) {
  if (e.key !== 'k' && e.key !== 'K') return;
  if (_isTyping()) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  e.preventDefault();
  toggleAsksPanel();
});
```

- [ ] **Step 4: Manual smoke**

Reload the dashboard. Badge should show `ASKS (0)`. In devtools:

```javascript
applyQuestionsSnapshot([{id:'t1', plan_id:'p', agent:'A', question:'?', received_at:1}]);
```

Badge should read `ASKS (1)`, the count turns hot pink, and the button pulses. Press `K` — nothing visible yet (Task 10 adds the panel), but no error.

- [ ] **Step 5: Commit**

```bash
git add templates/components/top_bar.html templates/scripts/panels.js templates/styles/base.css
git commit -m "Add pulsing ASKS badge to top bar for pending ask_human questions"
```

---

## Task 10: ASKS panel template + inclusion

Add the empty panel so `toggleAsksPanel` has a target. Rendering logic lands in Task 11.

**Files:**
- Create: `templates/components/panels/asks.html`
- Modify: `templates/base.html`
- Modify: `templates/flow.html`
- Modify: `templates/styles/base.css`

- [ ] **Step 1: Create the panel template**

Create `templates/components/panels/asks.html`:

```html
<div id="asksPanel" class="glass-card hidden">
  <div class="panel-header">
    <span>ASKS &mdash; <span id="asksPanelCount">0</span> WAITING</span>
    <button class="close-btn" onclick="toggleAsksPanel()">&times;</button>
  </div>
  <div id="asksBody"></div>
</div>
```

- [ ] **Step 2: Include the panel in both views**

In `templates/base.html`, find `{% include "components/panels/queue.html" %}` and add the line below it:

```html
{% include "components/panels/asks.html" %}
```

Do the same in `templates/flow.html` — add `{% include "components/panels/asks.html" %}` next to the existing panel includes.

- [ ] **Step 3: Add layout CSS**

In `templates/styles/base.css`, find the existing `#queuePanel` styles (search for `#queuePanel`). Add a sibling rule right below:

```css
#asksPanel {
  position: fixed;
  top: 56px;
  right: 16px;
  width: 360px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  z-index: 800;
  padding: 12px;
}
#asksPanel.hidden { display: none; }
#asksBody { font-size: 12px; }
.asks-group {
  border-top: 1px solid rgba(255,255,255,0.08);
  padding-top: 8px;
  margin-top: 8px;
}
.asks-group:first-child { border-top: none; margin-top: 0; padding-top: 0; }
.asks-group-header {
  font-size: 10px;
  letter-spacing: 1px;
  color: rgba(255,255,255,0.55);
  cursor: pointer;
  user-select: none;
  margin-bottom: 6px;
}
.asks-group-header .caret { display: inline-block; width: 10px; }
.asks-item {
  padding: 8px;
  margin-bottom: 4px;
  background: rgba(255,95,140,0.06);
  border-left: 2px solid #ff5f8c;
  cursor: pointer;
  transition: background 120ms;
}
.asks-item:hover { background: rgba(255,95,140,0.14); }
.asks-item.active { background: rgba(255,200,60,0.15); border-left-color: #ffc83c; }
.asks-item-head {
  display: flex;
  justify-content: space-between;
  font-weight: 700;
  color: #fff;
  margin-bottom: 2px;
}
.asks-item-time { font-size: 10px; color: rgba(255,255,255,0.5); }
.asks-item-preview {
  color: rgba(255,255,255,0.8);
  font-size: 11px;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.asks-empty {
  color: rgba(255,255,255,0.5);
  font-style: italic;
  text-align: center;
  padding: 24px 0;
}
```

- [ ] **Step 4: Manual smoke**

Reload. Press `K`. The empty `#asksPanel` should appear top-right. Press `K` again or click the × to close.

- [ ] **Step 5: Commit**

```bash
git add templates/components/panels/asks.html templates/base.html templates/flow.html templates/styles/base.css
git commit -m "Add ASKS panel container (empty) to both dashboard views"
```

---

## Task 11: `renderAsks` — grouped queue rendering + click-to-select

Populate the ASKS panel body. Questions group by `plan_id`; each group is collapsible. Clicking an item makes it the banner's active question without forcing an answer.

**Files:**
- Modify: `templates/scripts/panels.js`

- [ ] **Step 1: Module-level state for collapsed groups**

Below `toggleAsksPanel`, add:

```javascript
var _asksCollapsedPlans = new Set();   // plan_ids the user has collapsed

function toggleAsksPlanGroup(planId) {
  if (_asksCollapsedPlans.has(planId)) _asksCollapsedPlans.delete(planId);
  else _asksCollapsedPlans.add(planId);
  renderAsks();
}
```

- [ ] **Step 2: Implement `renderAsks`**

Below that:

```javascript
function renderAsks() {
  var body = document.getElementById('asksBody');
  var countEl = document.getElementById('asksPanelCount');
  if (!body) return;
  if (countEl) countEl.textContent = _pendingQuestions.size;

  if (_pendingQuestions.size === 0) {
    body.innerHTML = '<div class="asks-empty">No pending questions.</div>';
    return;
  }

  // Group by plan_id. Keep a stable order: plans sorted by oldest question first.
  var groups = {};
  _pendingQuestions.forEach(function (q) {
    var pid = q.plan_id || '(no plan)';
    if (!groups[pid]) groups[pid] = [];
    groups[pid].push(q);
  });
  Object.keys(groups).forEach(function (pid) {
    groups[pid].sort(function (a, b) {
      return (a.received_at || 0) - (b.received_at || 0);
    });
  });
  var planIds = Object.keys(groups).sort(function (a, b) {
    return (groups[a][0].received_at || 0) - (groups[b][0].received_at || 0);
  });

  var nowSec = Math.floor(Date.now() / 1000);
  var html = '';
  for (var i = 0; i < planIds.length; i++) {
    var pid = planIds[i];
    var items = groups[pid];
    var collapsed = _asksCollapsedPlans.has(pid);
    var caret = collapsed ? '&#9656;' : '&#9662;';
    html += '<div class="asks-group">';
    html += '<div class="asks-group-header" onclick="toggleAsksPlanGroup(' +
            JSON.stringify(pid) + ')">' +
            '<span class="caret">' + caret + '</span> PLAN ' +
            escapeHtml(pid) + ' (' + items.length + ')</div>';
    if (!collapsed) {
      for (var j = 0; j < items.length; j++) {
        var q = items[j];
        var ago = _formatAgo(nowSec - (q.received_at || nowSec));
        var activeCls = (q.id === _activeQuestionId) ? ' active' : '';
        html += '<div class="asks-item' + activeCls + '" onclick="selectAsk(' +
                JSON.stringify(q.id) + ')">';
        html += '<div class="asks-item-head">';
        html += '<span>' + escapeHtml(q.agent || 'agent') +
                (q.task_id ? ' &middot; ' + escapeHtml(q.task_id) : '') + '</span>';
        html += '<span class="asks-item-time">' + ago + '</span>';
        html += '</div>';
        html += '<div class="asks-item-preview">' +
                escapeHtml((q.question || '').slice(0, 160)) + '</div>';
        html += '</div>';
      }
    }
    html += '</div>';
  }
  body.innerHTML = html;
}

function _formatAgo(deltaSec) {
  if (deltaSec < 60) return deltaSec + 's ago';
  if (deltaSec < 3600) return Math.floor(deltaSec / 60) + 'm ago';
  return Math.floor(deltaSec / 3600) + 'h ago';
}

function selectAsk(id) {
  if (!_pendingQuestions.has(id)) return;
  _activeQuestionId = id;
  refreshActiveBanner();
  renderAsks();  // refresh the .active highlight
}
```

- [ ] **Step 3: Manual smoke**

In devtools:

```javascript
applyQuestionsSnapshot([
  {id:'a', plan_id:'web-dashboard', agent:'Researcher', task_id:'T-1', question:'Which API?', received_at: Math.floor(Date.now()/1000) - 120},
  {id:'b', plan_id:'web-dashboard', agent:'Designer', task_id:'T-2', question:'Palette?', received_at: Math.floor(Date.now()/1000) - 30},
  {id:'c', plan_id:'etl-pipeline', agent:'Extractor', task_id:'T-1', question:'Source path?', received_at: Math.floor(Date.now()/1000) - 10},
]);
toggleAsksPanel();
```

Expected:
- Panel shows two plan groups with the oldest-question plan first (`web-dashboard` before `etl-pipeline`).
- Each item shows agent + task, time ago, and a two-line preview.
- Clicking an item highlights it and pops the banner for that question.
- Clicking the group header collapses/expands.
- No forced answer — the banner's X button still works.

- [ ] **Step 4: Commit**

```bash
git add templates/scripts/panels.js
git commit -m "Render ASKS panel with plan-grouped items and click-to-select"
```

---

## Task 12: Live clock tick for "time ago" labels

"4m ago" labels must update without a WS event. A 30-second re-render keeps them current without being wasteful.

**Files:**
- Modify: `templates/scripts/panels.js`

- [ ] **Step 1: Add a periodic tick**

Below `_formatAgo`:

```javascript
setInterval(function () {
  var panel = document.getElementById('asksPanel');
  if (panel && !panel.classList.contains('hidden') && _pendingQuestions.size > 0) {
    renderAsks();
  }
}, 30000);
```

- [ ] **Step 2: Verify by inspection**

Open the ASKS panel with pending items. Wait 30 seconds; the "Xs ago" labels should increase.

- [ ] **Step 3: Commit**

```bash
git add templates/scripts/panels.js
git commit -m "Tick ASKS time-ago labels every 30s while panel is open"
```

---

## Task 13: Agent "needs help" glow in city view

City-view sprite sprites get the amber pulse + `?!` chip + beacon beam when their tool_call_id is in `_pendingQuestions`.

**Files:**
- Modify: `templates/scripts/draw_programs.js` (city view)

- [ ] **Step 1: Look up the existing program/agent drawing**

Run: `grep -n 'drawAmbientPrograms\|drawProgram' templates/scripts/draw_programs.js | head`

Identify where a single program is drawn (look inside `drawAmbientPrograms(ctx, time)` for the per-program render loop).

- [ ] **Step 2: Add the "needs help" renderer**

Above `drawAmbientPrograms` (near the other private helpers `_drawPixelTrail` etc.), add:

```javascript
function _programHasPendingQuestion(p) {
  if (typeof _pendingQuestions === 'undefined' || !_pendingQuestions.size) return false;
  var name = (p && p.agentName) ? p.agentName : '';
  if (!name) return false;
  var hit = false;
  _pendingQuestions.forEach(function (q) {
    if (!hit && q.agent && q.agent.toLowerCase() === name.toLowerCase()) hit = true;
  });
  return hit;
}

function _drawHelpIndicator(ctx, x, y, time) {
  // Amber pulse ring around the sprite.
  var pulse = 0.5 + 0.5 * Math.sin(time / 0.9);
  ctx.save();
  ctx.globalAlpha = 0.35 + 0.45 * pulse;
  ctx.beginPath();
  ctx.arc(x, y, 18 + 4 * pulse, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,200,60,' + (0.6 + 0.3 * pulse) + ')';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  // Beacon beam.
  ctx.save();
  var grad = ctx.createLinearGradient(x, y - 12, x, y - 72);
  grad.addColorStop(0, 'rgba(95,220,255,0.55)');
  grad.addColorStop(1, 'rgba(95,220,255,0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(x - 5, y - 72, 10, 60);
  ctx.restore();

  // "?!" chip.
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#ff5f8c';
  ctx.fillRect(x - 10, y - 34, 20, 14);
  ctx.fillStyle = '#fff';
  ctx.font = '900 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('?!', x, y - 27);
  ctx.restore();
}
```

- [ ] **Step 3: Call `_drawHelpIndicator` from the program draw loop**

In `drawAmbientPrograms(ctx, time)`, find the per-program draw loop. After the sprite is drawn for program `p`, add:

```javascript
if (_programHasPendingQuestion(p)) {
  _drawHelpIndicator(ctx, p.x, p.y, time);
}
```

(Use whatever `p.x` / `p.y` property names the existing code uses — inspect the surrounding draw code to match.)

- [ ] **Step 4: Manual smoke**

Start `serve.py`, open `/`, and in devtools trigger the snapshot while a city program exists:

```javascript
// Pick a real agent name from the city — check ambientPrograms in devtools
var name = ambientPrograms[0].agentName;
applyQuestionsSnapshot([{id:'g1', plan_id:'p', agent:name, question:'Help?', received_at:1}]);
```

Expected: that program starts showing the amber ring, pink `?!` chip, and cyan beam. Clearing with `applyQuestionsSnapshot([])` removes the indicator within one frame.

- [ ] **Step 5: Commit**

```bash
git add templates/scripts/draw_programs.js
git commit -m "Show amber pulse + ?! chip + beacon above agents needing human input (city)"
```

---

## Task 14: Agent glow in flow view

Same indicator, but for the flow-view node renderer in `draw_agents.js`.

**Files:**
- Modify: `templates/scripts/draw_agents.js`

- [ ] **Step 1: Locate the per-node render**

Run: `grep -n 'drawAllAgents\|function draw' templates/scripts/draw_agents.js | head`

Inside `drawAllAgents(ctx, time, W, H)` find where each node is drawn. Match the loop variable name (likely `n` or `node`).

- [ ] **Step 2: Reuse the indicator helper**

The helper `_drawHelpIndicator` is defined in `draw_programs.js` (Task 13) and is available globally because all scripts share one window scope. Add to `draw_agents.js` above `drawAllAgents`:

```javascript
function _agentHasPendingQuestion(node) {
  if (typeof _pendingQuestions === 'undefined' || !_pendingQuestions.size) return false;
  var agent = node && (node.agent || node.label || '');
  if (!agent) return false;
  var hit = false;
  _pendingQuestions.forEach(function (q) {
    if (!hit && q.agent && q.agent.toLowerCase() === String(agent).toLowerCase()) hit = true;
  });
  return hit;
}
```

In `drawAllAgents`, inside the per-node draw loop, after the node's main render, add:

```javascript
if (_agentHasPendingQuestion(node) && typeof _drawHelpIndicator === 'function') {
  _drawHelpIndicator(ctx, node.x, node.y, time);
}
```

- [ ] **Step 3: Manual smoke**

Navigate to `/flow` while a plan runs, or simulate:

```javascript
var node = nodes[0];
applyQuestionsSnapshot([{id:'f1', plan_id:'p', agent:node.agent || node.label, question:'Help?', received_at:1}]);
```

Expected: the flow node shows the indicator.

- [ ] **Step 4: Commit**

```bash
git add templates/scripts/draw_agents.js
git commit -m "Show needs-help indicator on flow-view nodes"
```

---

## Task 15: Click-glowing-agent opens its question

The existing click-to-locate handler (`1a6815e`) opens the agent detail card. Extend the click path so that if the clicked agent has a pending question, the banner opens for that question too.

**Files:**
- Modify: `templates/scripts/panels.js`

- [ ] **Step 1: Find the existing click-to-locate handler**

Run: `grep -n 'showAgentDetail\|click.*agent\|onAgentClick' templates/scripts/panels.js | head`

Identify the function that handles the "clicked agent" action (likely `showAgentDetail(node)`).

- [ ] **Step 2: Extend it**

Inside `showAgentDetail(node)`, right after `selectedNode = node;`, add:

```javascript
// If this agent has a pending ask_human question, also open its banner entry.
if (typeof _pendingQuestions !== 'undefined' && _pendingQuestions.size > 0) {
  var target = null;
  var name = (node.agent || node.label || '').toLowerCase();
  _pendingQuestions.forEach(function (q) {
    if (!target && q.agent && q.agent.toLowerCase() === name) target = q.id;
  });
  if (target) {
    _activeQuestionId = target;
    refreshActiveBanner();
  }
}
```

- [ ] **Step 3: Manual smoke**

Run a plan that calls `ask_human`. When the agent glows in the city view, click it. Expected: the detail card opens and the banner populates with that agent's question.

- [ ] **Step 4: Commit**

```bash
git add templates/scripts/panels.js
git commit -m "Clicking a glowing agent opens its pending question in the banner"
```

---

## Task 16: Extend ask_human unit tests

Harden the Python-side contract for ordering, resolve-doesn't-disturb, and the JSONL path.

**Files:**
- Modify: `tests/test_ask_human.py`

- [ ] **Step 1: Add the tests**

Append to `tests/test_ask_human.py`:

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_ask_human.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ask_human.py
git commit -m "Add multi-question ordering and selective-resolve tests"
```

---

## Task 17: Browser smoke test (optional — behind smoke marker)

Extend the Playwright smoke test so CI (or a human running `pytest -m smoke`) can catch regressions in the queue UI.

**Files:**
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Add a smoke test**

Append to `tests/test_smoke.py` (guard with the existing `@pytest.mark.smoke` marker and `skip_if_server_down` pattern — match whatever the file already uses):

```python
@pytest.mark.smoke
def test_asks_badge_updates_with_snapshot(page, server_url):
    """Pending-question snapshot should bump the ASKS badge count."""
    page.goto(server_url + "/")
    page.wait_for_selector("#asksBtn")

    page.evaluate("""
        applyQuestionsSnapshot([
          {id:'s1', plan_id:'p', agent:'a1', question:'one?', received_at:1},
          {id:'s2', plan_id:'p', agent:'a2', question:'two?', received_at:2},
        ]);
    """)

    count = page.text_content("#asksCount")
    assert count.strip() == "2"

    btn = page.locator("#asksBtn")
    assert "has-pending" in (btn.get_attribute("class") or "")


@pytest.mark.smoke
def test_asks_panel_open_and_click_selects(page, server_url):
    """Opening the panel and clicking a row should set the banner's active id."""
    page.goto(server_url + "/")
    page.wait_for_selector("#asksBtn")

    page.evaluate("""
        applyQuestionsSnapshot([
          {id:'qa', plan_id:'p1', agent:'R', question:'First?', received_at:1},
          {id:'qb', plan_id:'p1', agent:'D', question:'Second?', received_at:2},
        ]);
    """)
    page.click("#asksBtn")
    page.wait_for_selector("#asksPanel:not(.hidden)")
    page.click(".asks-item:nth-child(1)")
    active = page.evaluate("_activeQuestionId")
    assert active in ("qa", "qb")
```

- [ ] **Step 2: Run smoke tests**

Run: `pytest tests/test_smoke.py -v -m smoke`
Expected: new tests pass if Playwright + server are available. Skipped otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "Smoke test ASKS badge + panel item click"
```

---

## Task 18: Final integration pass

End-to-end verification with the whole thing wired up.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass. If anything fails, fix it before continuing.

- [ ] **Step 2: Run two concurrent plans with ask_human**

Start `python serve.py`. In another terminal, submit two plans back-to-back that each call `ask_human` (either via the dashboard Generate button with a prompt like "Build X but ask me to confirm the approach" or by crafting a fixture plan). Confirm:

- Badge reads `ASKS (2)`.
- Banner shows question #1 with `1 of 2` carousel.
- Both agents glow (amber ring + `?!` chip + beam) in city and flow views.
- Press Escape — banner closes, badge stays lit.
- Press K — ASKS panel opens with both questions grouped by plan.
- Click the second item — banner opens to question #2.
- Type a draft answer, refresh the page — draft is restored on the matching question.
- Submit answer — both the banner and queue entry disappear, badge updates.

- [ ] **Step 3: Verify the JSONL log**

Check `plans/{plan_id}_questions.jsonl` for both plans. Each should contain matching `request` and `resolve` (or `timeout`) entries with timestamps.

- [ ] **Step 4: Commit any follow-up fixes**

If anything needed fixing during the integration pass, commit with a descriptive message. Otherwise, no commit needed for this task.

---

## Notes for the executor

- The spec (`docs/superpowers/specs/2026-04-13-ask-human-queue-design.md`) is the contract. If anything in this plan disagrees with the spec, follow the spec and flag the conflict.
- The orchestrator and `serve.py` share a process when plans are launched via `/api/plans/{id}/run`. Do not assume IPC between them — `tools._question_details` is the shared source of truth.
- Fire-and-forget POSTs (`_broadcast_questions_snapshot`, `request_question`, `resolve_question`) use short timeouts and swallow errors by design. Do not tighten them.
- Visual verification steps in UI tasks use devtools; they're not a substitute for the smoke test, but are faster for iteration.
