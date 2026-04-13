# Ask-Human Queue Design

**Date:** 2026-04-13
**Status:** Approved for implementation
**Related:** commit `4b2f9b1` (initial `ask_human` tool)

## Problem

When multiple plans run concurrently and agents call `ask_human`, the single popup banner (`#questionBanner`) is overwritten by each new question. A user cannot respond to an earlier question because the later question has replaced its UI. There is also no way to browse which agents are waiting for human input once a popup has been dismissed or missed — the state is effectively invisible outside of the moment it pops up.

## Goals

- Stop losing questions. Every pending `ask_human` request must remain reachable until it is answered, dismissed, or resolved by timeout.
- Let the user browse pending questions at will. No forced response when opening the queue or a queue item.
- Make it obvious in the city/flow view which agents are blocked on human input.
- Preserve drafted answers across page refresh and server restart.

## Non-goals

- Changing the server-side `ask_human` blocking semantics (`threading.Event` + policy).
- Cross-session / cross-user queue sharing. The queue is local to the dashboard instance.
- Answer routing between multiple human operators.

## Architecture

The orchestrator runs as a background thread inside `serve.py` (`_run_orchestrator_worker`), so `tools._pending_questions` / `tools._question_answers` are already shared between the agent thread and the HTTP endpoints. No IPC changes are required.

```
agent thread  ──ask_human()──>  tools.request_question()
                                      │
                                      ├── stores in _pending_questions / _question_details
                                      ├── appends to plans/{plan_id}_questions.jsonl
                                      ├── POSTs {type:"question_request"} to serve.py /update
                                      └── blocks on threading.Event

serve.py      ──broadcasts──>   WebSocket clients
                                      │
                                      ├── type:"question_request"   (single event, triggers toast/banner)
                                      └── type:"questions_snapshot" (full list, drives badge + queue panel)

user clicks submit ──POST /api/questions/{id}──> tools.resolve_question()
                                                      │
                                                      ├── sets event, releases agent thread
                                                      ├── appends resolve entry to jsonl
                                                      └── broadcasts questions_snapshot
```

### New WebSocket message: `questions_snapshot`

Emitted whenever the pending set changes. Payload:

```json
{
  "type": "questions_snapshot",
  "pending": [
    { "id": "...", "plan_id": "...", "agent": "...", "task_id": "...",
      "question": "...", "context": "...", "received_at": 1776057927 }
  ]
}
```

Sent on: new request, resolution, timeout, and once on WebSocket connect so a fresh client gets the full state.

## Client state (`templates/scripts/panels.js`)

Single source of truth for both badge, banner, and queue panel:

- `_pendingQuestions: Map<tool_call_id, QuestionRecord>` — rebuilt on each `questions_snapshot`.
- `_activeQuestionId: string | null` — the id currently displayed in the banner.
- All rendering (badge count, banner fields, queue list, agent glow check) reads from these two values. Submit / dismiss / carousel actions mutate them and trigger a re-render.

## Banner behavior (`#questionBanner`)

Existing banner is extended, not replaced:

- **Dismiss without answering:** new `×` button and `Escape` key clear the banner. The question stays in `_pendingQuestions`. Badge remains lit.
- **Carousel:** when `_pendingQuestions.size > 1`, render `‹ N of M ›` controls. Arrows update `_activeQuestionId` in order of `received_at`.
- **Arrival while banner open:** a new `question_request` for a different id does *not* replace the active one. Instead, a small non-blocking toast slides in (`"SPECIALIST-42 also needs help — N in queue"`) and auto-dismisses after ~4 s. Badge updates.
- **Notification gating:**
  - `browserAlerts` / `questionSound` fire only when the banner is hidden and this is the first pending question (unchanged behavior for single-question cases).
  - With the banner already visible, subsequent arrivals use toast only (no audio, no OS notification).
  - When the banner is dismissed with questions still pending, the badge keeps pulsing but no re-chime fires.
- **Coexists with queue panel:** opening `#asksPanel` does not close the banner. Clicking a queue item sets `_activeQuestionId` to that item and ensures the banner is visible.

## ASKS badge + queue panel

### Top-bar button

Added to `templates/components/top_bar.html` between `QUEUE` and `LEDGER`:

```html
<button id="asksBtn" onclick="toggleAsksPanel()" title="Pending questions (ASK)">
  ASKS (<span id="asksCount">0</span>)
</button>
```

- Count element hot-pink and pulsing when > 0. Styling mirrors the existing `.panel-header` pink accent; the pulse uses the same cadence as the agent-body glow so they feel like one system.
- Keyboard shortcut `K` (ASK) opens/closes the panel. `_isTyping()` guard already exists in `panels.js`.

### Panel structure

New file `templates/components/panels/asks.html` modelled on `queue.html`:

```html
<div id="asksPanel" class="glass-card hidden">
  <div class="panel-header">
    <span>ASKS — <span id="asksPanelCount">0</span> WAITING</span>
    <button class="close-btn" onclick="toggleAsksPanel()">&times;</button>
  </div>
  <div id="asksBody"></div>
</div>
```

Body rendering (`renderAsks()` in `panels.js`):

- Items grouped by `plan_id`. Each group header is collapsible: `▸ PLAN <plan-id> (N)`.
- Group ordering: plans with the oldest pending question first.
- Item row:
  - Agent name + task id (e.g. `SPECIALIST-42 · TASK-007`).
  - First ~120 chars of the question text.
  - Relative time (`4m ago`) computed from `received_at`.
  - Inline `Answer` chip (optional quick path — opens inline textarea + submit button inside the row).
- Clicking any row (outside the inline Answer chip) sets `_activeQuestionId` to that question and ensures the banner is shown. The panel stays open; the user may dismiss or switch items freely without answering.

## Agent "needs help" indicator

Triggered in both city view (`draw_programs.js`) and flow view (whatever renders flow nodes) whenever the agent's tool_call_id is in `_pendingQuestions`. Cleared when it leaves the set.

Visual composition (all three layered together):

- **Soft amber pulse glow** on the agent body. 1.8 s ease-in-out; peak `box-shadow: 0 0 24px 6px rgba(255,200,60,0.6)`.
- **Hot-pink `?!` chip** floating ~30px above the agent.
- **Vertical beacon beam** (~80 px) rising above the chip, gradient from transparent at the top to `rgba(95,220,255,0.55)` at the base of the chip. Visible at distance even on crowded views.

Interaction:

- Click-to-locate already exists (`1a6815e`). Extend the handler so that clicking a glowing agent also does `_activeQuestionId = agent.pending_question_id` and shows the banner.

## Persistence

### Draft answers (client)

LocalStorage key format:

```
nort_ask_draft::<plan_id>::<sha1(question_text)[:12]>::<agent>
```

- On any input in the banner textarea or the queue's inline textarea, debounce 400 ms and write the current value.
- On incoming `question_request`, compute the key and pre-fill the textarea if a value exists.
- On successful submit, delete the key. Dismissal does not clear drafts.

### Question log (server)

Per-plan append-only JSONL at `plans/{plan_id}_questions.jsonl`:

```jsonl
{"ts":1776057927,"type":"request","id":"tc_abc","agent":"specialist-42","task_id":"TASK-007","question":"..."}
{"ts":1776058001,"type":"resolve","id":"tc_abc","answer":"..."}
{"ts":1776058200,"type":"timeout","id":"tc_def"}
```

- Written atomically via append on every `request_question`, `resolve_question`, and timeout code path.
- Not used for state restoration on restart. Present for audit, debugging, and (future) a "recently answered" view in the queue.

### Server restart behaviour

- `_pending_questions` starts empty.
- The plan run resumes from the last checkpoint (`plans/{plan_id}_checkpoint.json`). If an agent was blocked on `ask_human`, the LLM re-calls it on replay and a new `tool_call_id` is issued for the same question text.
- The matching localStorage draft (keyed by question text hash) pre-fills the new banner so the user does not retype.

## Testing

Extend `tests/test_ask_human.py`:

- Multi-question ordering: request three, assert `get_pending_questions` returns them in request order.
- `resolve_question` on a non-active question leaves the active one untouched.
- Carousel navigation correctness via a light shim around `_activeQuestionId`.
- Draft key is deterministic across identical question text / plan / agent.

New `tests/test_questions_log.py`:

- Request → resolve writes two entries to the correct JSONL with expected fields.
- Timeout path writes a `timeout` entry.
- Invalid answers are not written.

Smoke (`tests/test_smoke.py`, behind `@pytest.mark.smoke`):

- Submit two plans each containing an `ask_human` call.
- Assert `#asksBtn` count reaches `2`.
- Assert carousel arrows work and answering one leaves count `1`.
- Assert dismissing the banner keeps the badge lit.

## Out of scope

- Multi-operator queue assignment.
- Persistent resume of in-flight questions without replay (would require serialising agent thread state, not worth it given checkpoint replay works).
- Mobile layout — the dashboard is desktop-only (confirmed `bed38a2`).
