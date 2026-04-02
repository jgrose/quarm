# Agent Status Dashboard — Agent-Flow Visual Style

## Context

The user needs a way to see agent status clearly. The current NORT dashboard is an isometric city with buildings, walking programs, weather, and day/night — visually rich but agent status is buried across multiple panels. The `agent-flow/` directory contains a separate React/Canvas2D visualization with a clean holographic node-graph style focused on agent execution flow.

**Goal**: Create a new `/flow` route that renders an agent-flow-style dashboard using the same WebSocket data, reusing existing draw modules where possible. No isometric city — pure node graph focused on agent status.

**Architecture decision**: New Jinja2 page with vanilla JS (not React). The NORT dashboard already ported ~80% of agent-flow's canvas rendering to vanilla JS (bloom, hex nodes, edges, particles, effects, tool cards, bubbles). The delta is small. Adding React/Vite would introduce a build toolchain the project doesn't use.

---

## What Gets Reused (no modification needed)

These existing NORT scripts already contain agent-flow's visual effects:

| Module | Agent-Flow Equivalent | Notes |
|--------|----------------------|-------|
| `bloom.js` | `bloom-renderer.ts` | Same multi-pass blur, additive blend |
| `draw_agents.js` | `draw-agents.ts` | Hex nodes, glows, state colors, badges |
| `draw_edges.js` | `draw-edges.ts` | Tapered bezier curves, particle trails |
| `draw_effects.js` | `draw-effects.ts` | Spawn/complete/error VFX |
| `draw_tools.js` | `draw-tool-calls.ts` | Tool call cards |
| `draw_bubbles.js` | `draw-bubbles.ts` | Message overlays |
| `draw_context.js` | Context ring/bar | Token usage visualization |
| `draw_cost.js` | `draw-cost.ts` | Cost pills per agent |
| `draw_discoveries.js` | `draw-discoveries.ts` | Discovery cards |
| `draw_dependencies.js` | N/A (NORT-only) | Task dependency lines |
| `camera.js` | `use-canvas-camera.ts` | Pan/zoom with inertia |
| `render_cache.js` | `render-cache.ts` | Glow sprite caching |
| `colors.js` | `colors.ts` | Holographic palette |
| `constants.js` | `canvas-constants.ts` | Animation config |
| `nodes.js` | `agent-types.ts` | Node model + config |
| `websocket.js` | N/A | WebSocket + applyStatus() |
| `api.js` | N/A | API fetch helpers |
| `panels.js` | N/A | Panel UI logic |

**What gets dropped** (city-only, not included in flow.html):
`draw_programs.js`, `draw_locations.js`, `draw_atmosphere.js`, `draw_weather.js`, `draw_minimap.js`, `draw_roster.js`, `draw_background.js` (isometric grid portion), `audio.js`

---

## New Files (4)

### 1. `templates/flow.html` (~80 lines)

Page shell modeled after `base.html` but focused on agent status:

**Includes (HTML panels):**
- `components/top_bar.html` — connection status, session tabs, stats
- Agent detail card (inline glass-card)
- `components/event_log.html` — agent chat
- `components/panels/agent_list.html` — agent status sidebar
- `components/panels/cost_panel.html` — token breakdown
- `components/panels/config.html` — settings (display toggles)
- `components/panels/perf_panel.html` — FPS monitoring

**Does NOT include**: queue, thinking, completion, ledger, plan_viewer, model_config, tolerance_config, transcript, timeline, file_attention, roster, agents registry, output_browser, review_analytics, DAG panel

**Includes (JS scripts — ~20 of the 29):**
- `colors.js`, `constants.js`, `render_cache.js`, `nodes.js`, `force.js`
- `draw_agents.js`, `draw_edges.js`, `draw_effects.js`, `draw_tools.js`, `draw_bubbles.js`, `draw_context.js`, `draw_cost.js`, `draw_dependencies.js`, `draw_discoveries.js`
- `bloom.js`, `camera.js`
- `websocket.js`, `api.js`, `panels.js`
- **NEW**: `flow_background.js`, `flow_render.js`, `flow_init.js`

### 2. `templates/scripts/flow_background.js` (~60 lines)

Stripped-down background with agent-flow's void aesthetic:
- Background color: `#050510` (deep void)
- Depth particles: reuse the `initDepthParticles()` / `drawDepthParticles()` functions from `draw_background.js` (copy the ~40 lines of depth particle code)
- Hex grid: pulsing hexagonal grid from agent-flow (distance-based alpha with `Math.sin(time * 0.5 + dist * 0.005)`), NOT the isometric diamond grid
- No isometric tiles, no offscreen canvas caching (those are city-only)

### 3. `templates/scripts/flow_render.js` (~100 lines)

Simplified render loop (vs the 250-line `render.js`):

```
function flowRender(timestamp) {
  // dt, currentTime calculation
  // FPS sampling (reuse _fps, _frameCount pattern)
  // Clear to #050510
  // Draw flow background (depth particles + hex grid)
  // Apply camera transform
  // Draw edges (drawAllEdges)
  // Draw particles (drawAllParticles)
  // Draw agents (drawAllAgents) 
  // Draw context bars (drawAllContextBars)
  // Draw tool cards (drawAllToolCards)
  // Draw bubbles (drawAllBubbles)
  // Draw discoveries (drawAllDiscoveries)
  // Draw dependencies (drawDependencyLines, drawBlockedIndicators)
  // Draw cost pills (drawAllCostPills)
  // Draw effects (drawEffects)
  // Tick force simulation (free mode)
  // Apply bloom
  // FPS badge (if perfOverlay)
  // requestAnimationFrame(flowRender)
}
```

Key difference from `render.js`: no buildings, programs, weather, atmosphere, minimap, roster badges. Just the agent node graph.

### 4. `templates/scripts/flow_init.js` (~80 lines)

Boot script for the flow page:
- Initialize canvas + camera
- Set `config.flowMode = true` (used by force.js to skip zone attraction)
- Set background to void color
- Connect WebSocket (reuses `connectWS()` from websocket.js)
- Register keyboard shortcuts (subset: Q, C, L, A, P, $, ESC)
- Add "CITY VIEW" button in top bar linking back to `/`
- Start `flowRender()` loop

---

## Modified Files (4)

### 1. `serve.py` — Add `/flow` route (3 lines)

After the existing root route (around line 394):
```python
@app.get("/flow", response_class=HTMLResponse)
async def flow_view(request: Request):
    return templates.TemplateResponse(request, "flow.html", {"port": PORT})
```

### 2. `templates/scripts/force.js` — Add free layout mode (~6 lines)

At lines 41-48 (zone attraction block), wrap in a config check:
```javascript
if (!config.flowMode) {
  // Zone attraction — pull toward zone's Y band
  var targetY = zoneH * (n.zone + 1);
  ...
}
```
When `config.flowMode` is true, nodes use pure charge/spring layout without zone banding — same physics as agent-flow's D3-force.

### 3. `templates/scripts/draw_agents.js` — Add Claude spark logo (~25 lines)

Port the `drawClaudeSpark()` function from `agent-flow/web/components/agent-visualizer/canvas/draw-agents.ts:16-27`:
- Add `CLAUDE_SPARK_D` SVG path constant (from `draw-misc.ts:68`)
- Add `_claudeSparkPath = null` lazy Path2D cache
- Add `drawClaudeSpark(ctx, cx, cy, r, color)` function
- In the main agent draw function, when `config.flowMode && node.tier === 'nexus'`, draw spark instead of current icon

### 4. `templates/styles/base.css` — Add `.flow-page` styles (~30 lines)

```css
.flow-page { background: #050510; }
.flow-page #canvasWrap { background: transparent; }
.flow-page .top-bar-actions button { /* agent-flow glass style */ }
```

---

## Parallelization

**Two agents, single wave:**

| Agent | Files | Scope |
|-------|-------|-------|
| **Flow-Shell** | `serve.py`, `flow.html`, `flow_init.js`, `flow_background.js` | Route, page shell, boot, background |
| **Flow-Render** | `flow_render.js`, `force.js`, `draw_agents.js`, `base.css` | Render loop, force mode, spark logo, styles |

**File conflict**: None. Each agent touches completely separate files.

**Dependency**: Flow-Render needs to know that `flow_background.js` exports `drawFlowBackground(ctx, W, H, time)` and `initFlowParticles(W, H)` — just the function signatures.

---

## Verification

1. Start `serve.py`, navigate to `http://localhost:8000/flow`
2. See void background with depth particles and pulsing hex grid
3. Run an orchestrator plan — agents appear as hexagonal nodes with state colors
4. NEXUS node shows Claude spark logo
5. Edges connect managers to agents with particle trails
6. Tool cards appear near active agents
7. Nodes use free-form layout (no zone banding)
8. Press `C` for chat, `L` for agent list, `$` for costs — panels work
9. Press `P` — FPS badge appears
10. Click "CITY VIEW" to return to main dashboard at `/`
11. Run `python3 -m pytest tests/ -v --ignore=tests/test_smoke.py` — no regressions
