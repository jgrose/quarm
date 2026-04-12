# Plan: Remove Dead Responsive CSS from base.css

## Summary
Remove all 4 responsive `@media (max-width: ...)` blocks from `templates/styles/base.css`. The NORT dashboard is desktop-only (canvas mouse events, keyboard shortcuts, WebSocket monitoring, no touch/PWA) so these ~407 lines of responsive CSS are dead code.

## Blocks to Remove

### Block 1: Lines 930-951
- Comment: `/* -- Responsive: Plans list */`
- Rule: `@media (max-width: 900px) { ... }` (plans-list-panel responsive override)
- 22 lines

### Block 2: Lines 3066-3110
- Comment block: `/* RESPONSIVE BREAKPOINTS ... */`
- Tier comment: `/* -- Tier 1: Compact Desktop (max-width: 1200px) */`
- Rule: `@media (max-width: 1200px) { ... }`
- 45 lines (including comment header)

### Block 3: Lines 3112-3297
- Tier comment: `/* -- Tier 2: Tablet / Narrow (max-width: 900px) */`
- Rule: `@media (max-width: 900px) { ... }`
- 186 lines

### Block 4: Lines 3299-3471
- Tier comment: `/* -- Tier 3: Mobile / Small Screen (max-width: 600px) */`
- Rule: `@media (max-width: 600px) { ... }`
- 173 lines

**Total removal: ~407 lines**

## Non-responsive @media to Keep
- None found. All 4 `@media` rules are responsive breakpoints. No print, prefers-color-scheme, or other non-breakpoint media queries exist.

## Steps

1. Remove Block 1 (lines 930-951) -- the inline responsive plans-list override
2. Remove Blocks 2-4 (lines 3066-3471) -- the large responsive breakpoints section
3. Verify no remaining `@media` rules via grep
4. Verify balanced braces with a quick brace-count check
5. Commit with no AI attribution

## Risk
Zero -- these styles only activate at viewport widths the desktop-only dashboard never encounters. Removal is purely dead code cleanup.
