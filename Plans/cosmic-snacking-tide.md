# Redesign Agent Chat as iMessage-Style Conversation Cards

## Context

The agent chat currently uses small bubbles with left-border accents and avatar circles. The user wants a cleaner iMessage-like layout where each message is a full-width rounded card with the agent name as a header, accent-colored background tint, and matching text color — like the screenshot showing "Edit", "CLAUDE", "Grep" as card titles.

## Approach

Restyle the chat bubbles and modify `buildChatHTML()` to produce card-style markup. Remove avatar circles. Each message becomes a standalone rounded card with agent-colored background and name header.

## Files to Modify

### 1. `templates/scripts/panels.js` — `buildChatHTML()` (~line 669)

**Current structure per agent message group:**
```html
<div class="chat-group">
  <div class="chat-avatar" style="border-color:COLOR;background:rgba;color:COLOR">●</div>
  <div class="chat-body">
    <div class="chat-name" style="color:COLOR">NAME</div>
    <div class="chat-bubble" style="border-left-color:COLOR;background:rgba(COLOR,0.06)">text</div>
    <div class="chat-meta">[TOOL] ...</div>
  </div>
</div>
```

**New structure — card-style:**
```html
<div class="chat-card" style="background:rgba(COLOR,0.08);border-color:rgba(COLOR,0.2)">
  <div class="chat-card-header" style="color:COLOR">
    <span class="chat-card-icon">●</span> NAME
  </div>
  <div class="chat-card-body" style="color:COLOR_LIGHTER">text</div>
  <div class="chat-card-meta">[TOOL] ...</div>
</div>
```

Changes to `buildChatHTML()`:
- Remove avatar circle div
- Replace `.chat-group` + `.chat-body` with single `.chat-card` wrapper
- Agent name becomes `.chat-card-header` inside the card
- Message text becomes `.chat-card-body`
- Meta/issues go below body inside the same card
- Continuation messages from same agent merge into one card (multiple `.chat-card-body` divs)
- System/master messages become minimal centered cards with subtle styling

### 2. `templates/styles/base.css` — Chat styles (~line 972)

**Remove:** `.chat-group`, `.chat-group.continuation`, `.chat-avatar`, `.chat-body`, `.chat-name`, `.chat-bubble`, `.chat-group.continuation .chat-bubble`

**Add:**
```css
.chat-card {
  border-radius: 8px;
  border: 1px solid;
  padding: 10px 12px;
  margin-bottom: 6px;
  transition: opacity 0.2s;
}

.chat-card-header {
  font-size: 10px;
  font-family: 'Courier New', monospace;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.chat-card-icon {
  font-size: 8px;
}

.chat-card-body {
  font-size: 11px;
  font-family: 'Courier New', monospace;
  line-height: 1.5;
  color: var(--text-primary);
  word-wrap: break-word;
}

.chat-card-body + .chat-card-body {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(100, 200, 255, 0.06);
}

.chat-card-meta {
  font-size: 9px;
  font-family: 'Courier New', monospace;
  color: var(--text-muted);
  margin-top: 6px;
  padding-top: 4px;
  border-top: 1px solid rgba(100, 200, 255, 0.06);
}

.chat-card-issue {
  font-size: 9px;
  font-family: 'Courier New', monospace;
  color: var(--text-dim);
  padding: 1px 0 1px 8px;
  line-height: 1.5;
}

.chat-card-issue::before {
  content: '\21B3 ';
  color: var(--text-muted);
}

/* System messages — minimal centered card */
.chat-card.system {
  background: rgba(100, 200, 255, 0.03);
  border-color: rgba(100, 200, 255, 0.08);
  text-align: center;
  padding: 6px 10px;
}

.chat-card.system .chat-card-body {
  font-size: 10px;
  color: var(--text-dim);
}

.chat-card.master {
  background: rgba(255, 215, 0, 0.06);
  border-color: rgba(255, 215, 0, 0.15);
}

.chat-card.master .chat-card-body {
  color: #ffd700;
}
```

**Update responsive breakpoints** (~line 3437): change the responsive chat class names to match new card classes.

## Verification

1. Start `serve.py`, open dashboard
2. Run a plan with multiple agents
3. Verify each message appears as a rounded card with agent name header
4. Verify accent colors match per agent (same palette as before)
5. Verify system/master messages are visually distinct
6. Verify continuation messages from same agent merge into one card
7. Verify meta (TOOL, MODEL) and issues render inside cards
8. Verify verdict badges still render correctly
9. Check responsive layout at narrow widths
