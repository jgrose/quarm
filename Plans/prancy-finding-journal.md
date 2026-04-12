# Fix Artifact/Output Browsing and Downloading in Dashboard

## Context

Completed jobs in the queue show action buttons (download ⬇, browse 📂) but clicking them either shows nothing or displays an empty overlay. The **backend APIs are fully functional** — the bugs are entirely in the frontend JavaScript (`panels.js`). There are two categories of issues:

1. **Source-unaware functions**: `previewArtifact()` and `downloadArtifacts()` are hardcoded to the artifacts API, ignoring the active tab (ARTIFACTS vs OUTPUT)
2. **No fallback for text-only plans**: Plans whose agents produce only LLM text responses (no `write_file()` calls) have zero files on disk, so both tabs show empty — but the actual results exist in `{plan_id}_results.json`

Verified with curl: `GET /api/artifacts/34da0a472a13` returns files correctly (HTTP 200). Plan `66dcec9a4c40` (sparky_short_story) has `artifacts: {}` in its results.json — agents never called `write_file()`.

---

## Bug Fixes (4 bugs, 2 files)

### Bug 1: `previewArtifact()` always uses artifacts API
**File:** `templates/scripts/panels.js:2186-2221`

**Problem:** When the OUTPUT tab is active, clicking a file calls `previewArtifact(filePath)` which:
- Parses `filePath.split('/')[0]` as planId — but output paths have no plan_id prefix (e.g., `"index.html"` not `"34da0a472a13/TASK-001/index.html"`)
- Always fetches from `/api/artifacts/` — wrong endpoint for output files

**Fix:** Make the function source-aware:
- When `_outputSource === 'output'`: use `_outputPlanId`, treat entire path as relPath, fetch from `/api/output/{planId}/file`
- When `_outputSource === 'artifacts'`: keep existing behavior (parse planId from path, fetch from `/api/artifacts/`)

### Bug 2: `showOutputBrowserForPlan()` defaults to wrong tab
**File:** `templates/scripts/panels.js:2067-2075`

**Problem:** Line 2069 hardcodes `_outputSource = 'output'`. Plans without assembled output (or text-only plans) show empty on this tab, even though artifacts may exist.

**Fix:** Probe `/api/output/{planId}/files` first. If it returns files, default to OUTPUT tab. Otherwise default to ARTIFACTS tab. Reset `_cachedResults = null` on plan switch.

### Bug 3: Text-only plans have no viewable/downloadable content
**Files:** `templates/scripts/panels.js` (new functions) + `serve.py` (new endpoint)

**Problem:** Plans like `sparky_short_story` where agents produce only LLM text (no file artifacts) show "No artifacts" in the browser and 404 on download. The actual deliverable text is in `plans/{plan_id}_results.json` under `task_results` but the dashboard can't access it.

**Fix — serve.py:** Add `GET /api/results/{plan_id}` endpoint (after line ~1074) that returns `task_results` and `summary` from the results JSON file.

**Fix — panels.js:** 
- In `loadOutputTree()`: when both tabs return zero files, call new `_loadResultsAsPreview(planId)` 
- New `_loadResultsAsPreview()`: fetches `/api/results/{planId}`, renders task IDs in the tree panel, auto-shows first result text
- New `_showResultText(taskId)`: displays task result text in the preview pane with caching
- In `downloadOutputZip()`: when both output and artifact downloads fail (404), fall back to downloading results.json as a blob

### Bug 4: `downloadArtifacts()` in overlay always uses artifacts API
**File:** `templates/scripts/panels.js:2294-2301`

**Problem:** The DOWNLOAD ZIP button in the output browser overlay always calls `/api/artifacts/{planId}/download`, even when the OUTPUT tab is active.

**Fix:** Check `_outputSource` and use `/output/{planId}/download` when viewing output. Fall back to artifacts endpoint if output download fails.

---

## Implementation Sequence

| Step | What | File | Lines |
|------|------|------|-------|
| 1 | Add `/api/results/{plan_id}` endpoint | `serve.py` | after ~1074 |
| 2 | Fix `previewArtifact()` source-awareness | `panels.js` | 2186-2221 |
| 3 | Fix `downloadArtifacts()` source-awareness | `panels.js` | 2294-2301 |
| 4 | Fix `showOutputBrowserForPlan()` default tab | `panels.js` | 2067-2075 |
| 5 | Add results fallback (`_loadResultsAsPreview`, `_showResultText`, `downloadOutputZip` fallback) | `panels.js` | 2116-2131, 2095-2100, new functions |

---

## Edge Cases to Handle
- Clear `_cachedResults = null` when switching plans
- Guard against race conditions in async source probe (check `_outputPlanId === planId` in callback)
- Large result text: `<pre>` with overflow scrolling handles this already

## Verification
1. **Plan with artifacts** (`34da0a472a13` hello_world): Browse defaults to tab with content, preview files from both tabs, download ZIP works from both tabs
2. **Text-only plan** (`66dcec9a4c40` sparky_short_story): Browse shows task result text in tree + preview, download offers results.json
3. **Tab switching**: Preview and download use correct API endpoint per active tab
4. **curl tests**: `curl localhost:8000/api/results/66dcec9a4c40` returns task_results
