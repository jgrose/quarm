# Plan: Isometric (Sim City SNES) View Conversion

## Context

The NORT dashboard currently uses a top-down hex grid with front-facing pixel art. The user wants a Sim City SNES-style isometric view — diamond tile grid, 3/4 perspective buildings, directional character sprites. This is primarily a visual/rendering overhaul; the camera, force simulation, node graph, and WebSocket systems are coordinate-agnostic and need no changes.

---

## Isometric Coordinate System

**2:1 isometric projection** (same as Sim City SNES):
```
TILE_W = 64   (diamond width in pixels)
TILE_H = 32   (diamond height — half of width)

Grid (col, row) → Screen (x, y):
  x = (col - row) * TILE_W / 2
  y = (col + row) * TILE_H / 2

Screen (x, y) → Grid (col, row):
  col = (x / (TILE_W/2) + y / (TILE_H/2)) / 2
  row = (y / (TILE_H/2) - x / (TILE_W/2)) / 2
```

Origin (0,0) is at top-center of the diamond grid. The grid extends down-left and down-right.

---

## What Changes

### 1. Grid Renderer — `draw_background.js`

Replace hex grid with isometric diamond tile grid:
- Replace `HEX_GRID_SIZE`, `HEX_OFFSETS`, `drawHexPath`, `drawHexGrid`
- New: `TILE_W = 64`, `TILE_H = 32`, `drawIsoDiamond(ctx, cx, cy)`, `drawIsoGrid(ctx, W, H, time)`
- Diamond tiles drawn as 4-point polygons (top, right, bottom, left vertices)
- Pulsing glow effect preserved (distance-based alpha modulation)
- Viewport-aware: uses `getVisibleRect()` to only draw visible tiles

### 2. Coordinate Helpers — `draw_background.js`

New public functions replacing hex math:
- `isoToScreen(col, row)` — returns `{x, y}` in world space
- `screenToIso(x, y)` — returns `{col, row}` (for click detection)
- `TILE_W` and `TILE_H` exported as globals (replace `HEX_GRID_SIZE`)

### 3. Location Sprites — `draw_locations.js` (all 12 redrawn)

Isometric 3/4 perspective buildings. Each building shows:
- Diamond-shaped footprint (matching tile grid)
- Front face (south-facing wall)
- Right side face (east-facing wall, darker shading)
- Roof/top surface
- Circuit glow lines on walls and edges

**Sprite dimensions** (at LOC_PX=6, roughly 2-3 tile footprint):
- Small buildings: 32w × 32h pixels → 192×192 screen px
- Large buildings: 40w × 40h pixels → 240×240 screen px
- Tall buildings (I/O Tower): 24w × 48h pixels → 144×288 screen px

**Same palette system** (indices 0-9) — no change to `_renderLocationSprite` color derivation. Only the sprite arrays change.

**Locations to redraw:**

| Location | Isometric Visual |
|----------|-----------------|
| END OF LINE | Bar with angled roof, neon sign on front face, arched door |
| CYCLE ARENA | Flat diamond arena floor with raised walls, track lines |
| I/O TOWER | Tall spire rising from diamond base, light beam at top |
| DISC RING | Circular ring sitting on diamond platform |
| RECOGNIZER PAD | Flat landing pad with T-shape recognizer on top |
| PORTAL | Two angled pillars with arch between, energy in gap |
| CODE FORGE | Terminal desk viewed from angle, floating code above |
| TRIBUNAL | Stepped pyramid platform with bench on top |
| ANALYSIS BAY | Scanner arch viewed from side angle |
| RECOMPILE | Open bay with crane arm, viewed from angle |
| DATA VAULT | Cubic vault with lock on front face |
| DEREZZED | Cracked diamond floor tiles, debris floating up |

### 4. Character Sprites — `draw_programs.js` (4 directions × 4 frames)

Isometric characters need **4 walking directions** (since the grid has 4 primary axes):
- **SE** (down-right) — primary facing direction
- **SW** (down-left) — mirror of SE
- **NE** (up-right) — back view, mirror of NW
- **NW** (up-left) — back view

Each direction has 4 walk frames (same cycle: mid → stride-A → mid → stride-B).

**Sprite dimensions**: 16w × 28h pixels at PX=3 (48×84 screen px) — slightly shorter than current 16×32 to fit isometric proportions.

**Direction selection**: Based on movement vector angle:
```
angle = atan2(dy, dx)
SE: -45° to 45°  (moving right-ish)
SW: 45° to 135°  (moving down-ish)
NW: 135° to 225° (moving left-ish)
NE: 225° to 315° (moving up-ish)
```

**Optimization**: Only draw SE and NE sprites. SW = horizontal flip of SE. NW = horizontal flip of NE. So we only need **2 directions × 4 frames = 8 sprite arrays** plus flipping.

### 5. Movement System — `draw_programs.js`

Replace axis-aligned movement with **isometric grid-aligned** movement:
- Programs move along iso grid axes (NE/SW and NW/SE diagonals in screen space)
- `_pickRandomHex` → `_pickRandomTile` — uses `isoToScreen(col, row)`
- `_pickTarget` — same logic, just uses iso coordinates via location positions
- Movement still interpolates linearly between world positions (no change to the lerp)

### 6. Agent Node Rendering — `draw_agents.js`

Agent nodes (NEXUS, SENTINEL, DRONE, PROBE) currently render as hexagons. Two options:
- **Keep hexagons** — they're floating indicators, not ground objects. Hexagons still look good.
- **Change to diamonds** — match the iso grid aesthetic.

**Recommendation**: Keep hexagons. They visually distinguish agent nodes from the grid.

### 7. Depth Sorting

Isometric view requires back-to-front rendering for proper occlusion:
- Objects with smaller `(col + row)` values (further from camera) draw first
- Locations and programs need to be sorted by their iso row before drawing
- Add a sort step in `drawAllLocations` and `drawAmbientPrograms`

---

## What Does NOT Change

| Component | Why |
|-----------|-----|
| Camera system (`camera.js`) | Coordinate-agnostic pan/zoom |
| Force simulation (`force.js`) | Uses zones and Cartesian x/y, not grid |
| Node graph (`nodes.js`) | Cartesian positions |
| Edge/particle rendering (`draw_edges.js`) | Bezier curves in Cartesian |
| WebSocket system (`websocket.js`) | Data only, no rendering |
| Palette/color system (`colors.js`) | Unchanged |
| Bloom/effects (`bloom.js`, `draw_effects.js`) | Coordinate-agnostic |
| All panel UI (`panels.js`, templates) | DOM-based, no canvas coords |
| Sprite cache system | Same approach, just new sprite data |

---

## Files to Modify

| File | Scope |
|------|-------|
| `templates/scripts/draw_background.js` | **REWRITE** — iso grid + coordinate helpers |
| `templates/scripts/draw_locations.js` | **REWRITE sprites** — all 12 iso building sprites + update `_locGridToScreen` → `isoToScreen`, add depth sort |
| `templates/scripts/draw_programs.js` | **REWRITE sprites** — 8 directional frames + direction selection + update `_pickRandomHex` → iso |
| `templates/scripts/draw_agents.js` | **MINOR** — optionally replace `drawHexPath` calls with diamond path |
| `templates/scripts/constants.js` | **MINOR** — add `TILE_W`, `TILE_H` |
| `templates/scripts/render.js` | **MINOR** — depth sort call in render loop |

---

## Implementation Order

### Phase 1: Grid Foundation
1. Add `TILE_W`, `TILE_H` to constants.js
2. Rewrite `draw_background.js` — iso diamond grid with `isoToScreen()`, `screenToIso()`, `drawIsoGrid()`
3. Update `getVisibleRect` usage for iso viewport coverage

### Phase 2: Isometric Building Sprites (12 buildings)
4. Redesign all 12 location sprites as isometric 3/4 view pixel art
5. Update `_locGridToScreen` → use `isoToScreen`
6. Update `initLocations` grid math for iso tile layout
7. Add depth sorting to `drawAllLocations`

### Phase 3: Isometric Character Sprites (4 dir × 4 frames)
8. Create SE and NE directional sprite sets (4 frames each)
9. Add direction selection logic based on movement vector
10. Update `_pickRandomHex` → `_pickRandomTile` with iso coords
11. Add depth sorting to `drawAmbientPrograms`

### Phase 4: Polish
12. Update agent node shapes if desired (hex → diamond)
13. Tune depth sort to handle programs near/inside buildings
14. Adjust camera default zoom for iso perspective feel

---

## Verification

1. **Refresh with no plan** — see isometric diamond grid, 12 iso buildings at correct positions, programs walking between them in 4 directions
2. **Zoom in** — buildings and characters stay crisp (pixel art at PX=3 / LOC_PX=6)
3. **Zoom out** — grid tiles infinitely based on viewport
4. **Pan** — world scrolls correctly in iso space
5. **Depth sorting** — programs behind buildings are occluded by buildings in front
6. **Run a plan** — agent nodes still render (hexagons in world space), locations dim to 40%
