# Plan: Per-Plan Chat Filtering Tabs

## Context

When multiple plans run concurrently, the Agent Chat panel only shows messages from the currently active canvas session. There's no way to view one plan's chat without switching the entire canvas. The user wants to filter agent chat by plan independently of the canvas view.

The agent-flow project (`agent-flow/web/components/agent-visualizer/message-feed-panel.tsx`) provides a proven reference: horizontal tab strip with "All" + per-entity tabs, unread dot indicators, and filter state decoupled from canvas state.

## Approach: Horizontal Tab Strip

Add a compact tab strip between the "AGENT CHAT" header and the progress bar. Tabs: **ALL** (default, combined view) + one tab per running plan. Clicking a tab filters chat without changing the canvas session. Unread dots appear when filtered-out plans get new messages.

**Why tabs over dropdown**: Instant visual feedback on active plans + status, one-click switching, unread indicators visible at a glance — same pattern proven in agent-flow.

## Implementation Steps

### 1. Add state variables — [websocket.js:11](templates/scripts/websocket.js#L11)

Add after existing `_activeSessionId`:
- `_chatFilterId = 'all'` — which plan's chat to show (`'all'` or a `session_id`)
- `_chatUnread = {}` — `session_id -> true` when non-viewed plan gets new messages
- `_chatLastSeenCount = {}` — `session_id -> number` to detect new log lines

### 2. Add tab container — [event_log.html](templates/components/event_log.html)

Insert `<div id="chatTabStrip" class="chat-tab-strip"></div>` between `.panel-header` and `#eventLogProgress`.

### 3. Add tab strip CSS — [base.css](templates/styles/base.css) (after line ~510)

- `.chat-tab-strip` — flex row, `overflow-x: auto`, hidden scrollbar, border-bottom separator
- `.chat-tab` — compact buttons (8px font, monospace), transparent default, flex-shrink:0
- `.chat-tab.active` — tinted background using `--tab-color` CSS variable (set per-tab via inline style to match plan phase color)
- `.chat-tab.unread::after` — 5px red dot pseudo-element (matches agent-flow's indicator pattern)
- `.chat-tab-phase` — 7px sub-label showing plan phase (EXEC, DONE, etc.)

### 4. Add rendering functions — [panels.js](templates/scripts/panels.js) (after line 189)

**`renderChatTabs()`** — Builds tab strip HTML from `_sessions`. Hides strip when <= 1 session. Each tab shows truncated plan name + phase badge. Falls back to `_chatFilterId = 'all'` if filtered session no longer exists.

**`setChatFilter(filterId)`** — Sets `_chatFilterId`, clears unread for that tab, re-renders tabs + chat body. Does NOT call `switchSession()`.

**`_maybeRenderChat()`** — Dispatcher: if filter is `'all'`, calls `_renderAllChat()`; otherwise calls existing `renderEventLog()` with the filtered session's data.

**`_renderAllChat()`** — Collects last 40 lines from each session, concatenates them (per-session order, since lines lack global timestamps), runs through existing `parseLogLine` / `groupChatMessages` / `buildChatHTML` pipeline. Shows aggregate progress bar.

### 5. Wire up call sites — [websocket.js](templates/scripts/websocket.js)

**In `applyStatus()` (~line 61)**:
- After storing session data, track unread: if incoming `data.log` has grown and `_chatFilterId` doesn't match this session, set `_chatUnread[sid] = true`
- Replace line 194 (`renderEventLog(...)`) with `_maybeRenderChat()`
- Add `renderChatTabs()` call (always, for all sessions — to show phase changes)

**In `switchSession()` (line 271)**:
- Replace `renderEventLog(...)` call with `_maybeRenderChat()` — canvas switches but chat filter stays independent

### 6. Edge cases

- **Single session**: Tab strip hidden, behavior identical to today
- **Session removed**: `renderChatTabs()` resets `_chatFilterId` to `'all'` if filtered session disappears
- **"ALL" unread**: Selecting "ALL" clears all unread indicators

## Files to Modify

| File | Change |
|------|--------|
| [templates/scripts/websocket.js](templates/scripts/websocket.js) | State vars, unread tracking, replace renderEventLog calls |
| [templates/scripts/panels.js](templates/scripts/panels.js) | `renderChatTabs`, `setChatFilter`, `_maybeRenderChat`, `_renderAllChat` |
| [templates/components/event_log.html](templates/components/event_log.html) | Add `#chatTabStrip` div |
| [templates/styles/base.css](templates/styles/base.css) | Tab strip + tab + unread CSS |

## Verification

1. Start `serve.py`, open dashboard
2. Run two plans concurrently (queue two plans via UI or CLI)
3. Verify tab strip appears when second session arrives
4. Click a plan tab — chat filters to that plan only, canvas stays unchanged
5. Switch to other tab — verify unread dot appeared on first plan's tab while viewing other
6. Click "ALL" — verify combined view shows both plans' messages
7. Verify single-plan runs still work (tab strip hidden)
8. Verify `toggleChat()` (C key) still opens/closes the panel with tabs present
