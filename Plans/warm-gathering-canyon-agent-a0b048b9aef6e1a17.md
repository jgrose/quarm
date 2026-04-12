# Plan: Remove All Responsive Breakpoint @media Blocks from base.css

## Task
Remove all `@media (max-width: ...)` blocks from `templates/styles/base.css`. There are exactly 4 blocks, totaling ~420 lines.

## Blocks to Remove

### Block 1: Lines 930-951 (22 lines)
- Comment: `/* -- Responsive: Plans list */`
- Rule: `@media (max-width: 900px)`
- Content: `.plans-list-panel` responsive overrides

### Block 2: Lines 3072-3110 (39 lines)
- Comment: `/* -- Tier 1: Compact Desktop (max-width: 1200px) */`
- Rule: `@media (max-width: 1200px)`
- Content: Width narrowing for eventLog, thinkingPanel, agentListPanel, transcriptPanel, fileAttention, roster-panel, queuePanel

### Block 3: Lines 3112-3297 (186 lines)
- Comment: `/* -- Tier 2: Tablet / Narrow (max-width: 900px) */`
- Rule: `@media (max-width: 900px)`
- Content: Side panel collapse/overlay, full-width panels, modal fluid widths, stats grid columns

### Block 4: Lines 3299-3471 (173 lines)
- Comment: `/* -- Tier 3: Mobile / Small Screen (max-width: 600px) */`
- Rule: `@media (max-width: 600px)`
- Content: Top bar wrapping, control bar stacking, full-width overlays, typography reduction, modal near-fullscreen

## Verification
- Grep for `@media` after removal -- expect 0 results
- Commit with no AI attribution

## Status: READY TO EXECUTE
Awaiting plan mode exit to implement.
