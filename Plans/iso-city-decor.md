# Plan: Space Out Buildings, Add Cyber Roads & Trees

## Context

Buildings are too close together on the isometric grid. Need to spread them out and add visual connective tissue — cyber roads between buildings and decorative cyber trees — to make it feel like a Tron city.

## Changes

### 1. Space Out Buildings — `draw_locations.js`

Update the `zoneMap` in `initLocations()` to spread buildings across a larger grid area. Current positions cluster in a ~20×20 grid. Spread to ~40×40 with more separation:

```
Current: buildings at cols 1-19, rows 0-19 (cramped)
New: buildings at cols 2-38, rows 2-38 (generous spacing, ~6-8 tiles between buildings)
```

Work locations stay in the center cluster but with more breathing room. Idle locations spread to the periphery.

### 2. Add Cyber Roads — `draw_locations.js`

Define road segments as pairs of iso grid coordinates `(fromCol, fromRow, toCol, toRow)`. Draw them as lit-up tile strips connecting buildings.

Each road tile is a diamond (same as grid) but filled with a dark surface + circuit line down the center. Drawn BEFORE buildings in the render order.

Road data:
```javascript
var ROAD_SEGMENTS = [
  // Connect work cluster
  { from: 'code_forge', to: 'tribunal' },
  { from: 'code_forge', to: 'analysis_bay' },
  { from: 'code_forge', to: 'recompile' },
  // Connect idle locations to work district
  { from: 'eol_club', to: 'tribunal' },
  { from: 'arena', to: 'analysis_bay' },
  { from: 'portal', to: 'code_forge' },
  // etc.
];
```

Roads are drawn by iterating iso tiles between two endpoints (Bresenham-style on the iso grid) and drawing a filled diamond tile with circuit glow.

### 3. Add Cyber Trees — `draw_locations.js`

Small isometric tree sprites (8w × 12h pixel art at LOC_PX scale) scattered between buildings. Tron-style: geometric/crystalline shapes with circuit glow, not organic.

Place ~15-20 trees at fixed grid positions, avoiding road tiles and building footprints. Trees are purely decorative (no occupancy, no interaction).

Tree varieties:
- **Tall crystal** — narrow vertical spike with glow tip
- **Data bush** — low dome of pixels with circuit traces
- **Light pole** — thin post with bright orb at top

### 4. Render Order

In `drawAllLocations`:
1. Draw roads (behind everything)
2. Draw trees (behind buildings, depth-sorted)
3. Draw buildings (depth-sorted)

## Files Modified

| File | Changes |
|------|---------|
| `templates/scripts/draw_locations.js` | Spread zone map, add road drawing, add tree sprites + placement, update render order |

No other files need changes.

## Verification

1. Refresh — buildings visibly spread out with clear spacing
2. Glowing cyber roads connect buildings across the grid
3. Decorative trees scattered between buildings
4. Programs still walk between locations correctly (pathfinding uses location coordinates, unaffected by spacing)
5. Zoom out to see full city layout
