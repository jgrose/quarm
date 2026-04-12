# Commit: Dashboard UI improvements

## Context
Committing completed work from this session: model config fixes, filter input, model presets, settings close button, and ESC hints across all overlay panels.

## Files to stage
- `templates/components/panels/config.html` — added close button + ESC hint
- `templates/components/panels/model_config.html` — ESC hint, filter input, preset row
- `templates/components/panels/tolerance_config.html` — ESC hint
- `templates/components/panels/review_analytics.html` — ESC hint
- `templates/components/panels/ledger.html` — ESC hint
- `templates/components/panels/agents.html` — ESC hint
- `templates/components/panels/output_browser.html` — ESC hint
- `templates/scripts/panels.js` — model presets, filter, renderModelConfig refactor
- `templates/scripts/init.js` — model overlay .visible class fix
- `templates/styles/base.css` — model card max-height, filter input, preset button styles

## Commit message
Fix model config panel, add filter/presets, add close button and ESC hints to all overlays
