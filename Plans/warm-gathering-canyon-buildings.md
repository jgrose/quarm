# Plan: Building Perspective Alignment & Larger Labels

## Context

Buildings are pixel-art sprites drawn flat via `drawImage` at isometric coordinates. The grid is 2:1 isometric (TILE_W=64, TILE_H=32), and the sprites already have isometric perspective baked into their pixel art (diamond roofs, angled walls). However two issues:

1. **Ground anchoring is wrong** — the ground light pool beneath buildings is a flat rectangle (`fillRect` at [draw_locations.js:1106-1108](templates/scripts/draw_locations.js#L1106-L1108)), not an isometric diamond. This makes buildings look like they're floating rather than sitting on the grid.

2. **Labels are too small** — building names are `10px monospace` at 65% opacity ([draw_locations.js:1113](templates/scripts/draw_locations.js#L1113)) with no background. Nearly invisible at normal zoom. "X ACTIVE" badge is `bold 8px monospace` — even worse.

## Implementation

All changes in [draw_locations.js](templates/scripts/draw_locations.js).

### Change 1: Replace rectangular ground pool with isometric diamond shadow (lines 1105-1108)

Replace:
```js
ctx.fillStyle = hexToRgba(loc.glowColor, 0.06);
var gw = Math.ceil(sprite.width * 0.7);
ctx.fillRect(Math.floor(loc.x - gw / 2), Math.floor(loc.y + 6), gw, LOC_PX * 2);
```

With an isometric diamond that matches the grid perspective:
```js
var groundW = sprite.width * 0.75;
var groundH = groundW * 0.5;
ctx.fillStyle = hexToRgba(loc.glowColor, 0.06);
drawIsoDiamond(ctx, loc.x, loc.y + LOC_PX, groundW, groundH);
ctx.fill();
```

### Change 2: Larger labels with dark background pill (lines 1110-1117)

Replace:
```js
ctx.save();
ctx.fillStyle = hexToRgba(loc.glowColor, 0.65);
ctx.font = '10px monospace';
ctx.textAlign = 'center';
ctx.textBaseline = 'top';
ctx.fillText(loc.name, loc.x, loc.y + 14);
ctx.restore();
```

With larger text, zoom-aware scaling, and a dark pill background for contrast:
```js
ctx.save();
var labelSize = 13;
// Scale up labels when zoomed out so they stay readable
if (config.lodEnabled && camera.zoom < 0.8) {
  labelSize = Math.round(13 / Math.max(camera.zoom, 0.25));
}
ctx.font = 'bold ' + labelSize + 'px monospace';
ctx.textAlign = 'center';
ctx.textBaseline = 'top';
var labelText = loc.name;
var labelY = loc.y + LOC_PX * 3;
var tw = ctx.measureText(labelText).width;
// Dark background pill
ctx.fillStyle = 'rgba(5, 5, 16, 0.55)';
ctx.beginPath();
ctx.roundRect(loc.x - tw/2 - 6, labelY - 2, tw + 12, labelSize + 6, 3);
ctx.fill();
// Glow-colored text
ctx.fillStyle = hexToRgba(loc.glowColor, 0.85);
ctx.fillText(labelText, loc.x, labelY);
ctx.restore();
```

### Change 3: Larger active badge (lines 1140-1141)

Change from:
```js
ctx.font = 'bold 8px monospace';
```
To:
```js
ctx.font = 'bold 11px monospace';
```

## Files Modified

| File | Changes |
|------|---------|
| `templates/scripts/draw_locations.js` | Iso diamond ground shadow, larger labels with pill bg, larger active badge |

## Verification

1. Zoom to 1x — building labels should be clearly readable with dark pill behind text
2. Zoom out to 0.3x — labels should scale up to remain legible
3. Diamond shadow beneath buildings should match the isometric grid angle
4. "X ACTIVE" badge should be readable at normal zoom
5. Ground shadow should be a diamond, not a rectangle
