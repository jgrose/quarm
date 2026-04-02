// ═══ NORT BACKGROUND LAYER — ISOMETRIC ═══
// Sim City SNES-style 2:1 isometric diamond tile grid

var depthParticles = null;

// ─── Isometric tile constants ───
var TILE_W = 64;           // diamond width
var TILE_H = 32;           // diamond height (2:1 ratio)
var ISO_GRID_COLS = 40;    // default grid size (expands as needed)
var ISO_GRID_ROWS = 40;

// Keep HEX_GRID_SIZE as alias for any legacy references
var HEX_GRID_SIZE = TILE_W;

// ─── Isometric coordinate conversion ───

function isoToScreen(col, row) {
  return {
    x: (col - row) * (TILE_W / 2),
    y: (col + row) * (TILE_H / 2),
  };
}

function screenToIso(sx, sy) {
  return {
    col: (sx / (TILE_W / 2) + sy / (TILE_H / 2)) / 2,
    row: (sy / (TILE_H / 2) - sx / (TILE_W / 2)) / 2,
  };
}

// ─── Depth particles (screen-space, unchanged) ───

function initDepthParticles(W, H) {
  depthParticles = [];
  for (var i = 0; i < 80; i++) {
    depthParticles.push({
      x: Math.random() * W * 2 - W * 0.5,
      y: Math.random() * H * 2 - H * 0.5,
      size: Math.random() * 1.5 + 0.5,
      brightness: Math.random() * 0.3 + 0.05,
      speed: Math.random() * 0.15 + 0.05,
      depth: Math.random(),
    });
  }
}

function drawDepthParticles(ctx, W, H, time) {
  if (!depthParticles) initDepthParticles(W, H);
  for (var i = 0; i < depthParticles.length; i++) {
    var p = depthParticles[i];
    p.x += p.speed * 0.16 * 10 * (1 - p.depth * 0.5);
    p.y -= p.speed * 0.16 * 5 * (1 - p.depth * 0.3);
    if (p.x > W * 1.5) p.x = -W * 0.5;
    if (p.y < -H * 0.5) p.y = H * 1.5;

    var px = p.x + camera.x * (0.3 + p.depth * 0.7) * 0.1;
    var py = p.y + camera.y * (0.3 + p.depth * 0.7) * 0.1;
    var sz = p.size * (0.5 + p.depth * 0.5);
    var al = p.brightness * (0.5 + p.depth * 0.5);

    ctx.beginPath();
    ctx.fillStyle = C.holoBase + alphaHex(al);
    ctx.arc(px, py, sz, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ─── Draw isometric diamond tile ───

function drawIsoDiamond(ctx, cx, cy, w, h) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - h / 2);        // top
  ctx.lineTo(cx + w / 2, cy);        // right
  ctx.lineTo(cx, cy + h / 2);        // bottom
  ctx.lineTo(cx - w / 2, cy);        // left
  ctx.closePath();
}

// ─── Keep drawHexPath for agent node rendering (still hexagonal) ───

var HEX_OFFSETS = [];
(function() {
  for (var i = 0; i < 6; i++) {
    var angle = (Math.PI / 3) * i - Math.PI / 6;
    HEX_OFFSETS.push({ cos: Math.cos(angle), sin: Math.sin(angle) });
  }
})();

function drawHexPath(ctx, cx, cy, r) {
  ctx.beginPath();
  ctx.moveTo(cx + r * HEX_OFFSETS[0].cos, cy + r * HEX_OFFSETS[0].sin);
  for (var i = 1; i < 6; i++) {
    ctx.lineTo(cx + r * HEX_OFFSETS[i].cos, cy + r * HEX_OFFSETS[i].sin);
  }
  ctx.closePath();
}

// ─── Tron grid line renderer ───
// Two sets of parallel diagonal lines forming the isometric diamond pattern.
// O(cols + rows) line draws instead of O(cols × rows) diamond strokes.

function drawHexGrid(ctx, W, H, time) {
  var view = (typeof getVisibleRect === 'function')
    ? getVisibleRect(W, H)
    : { x: 0, y: 0, w: W, h: H };

  // Iso grid line directions:
  //   Col-axis lines run along (TILE_W/2, TILE_H/2) — slope = TILE_H/TILE_W
  //   Row-axis lines run along (-TILE_W/2, TILE_H/2) — slope = -TILE_H/TILE_W
  var halfW = TILE_W / 2;   // 32
  var halfH = TILE_H / 2;   // 16
  var slope = halfH / halfW; // 0.5

  // LOD: skip every other line when zoomed out
  var step = 1;
  if (config.lodEnabled && camera.zoom < 0.35) step = 3;
  else if (config.lodEnabled && camera.zoom < 0.55) step = 2;

  // How many lines needed to cover the visible viewport
  // Lines are spaced halfH apart perpendicular to their direction
  var diagExtent = Math.sqrt(view.w * view.w + view.h * view.h) * 0.6;
  var numLines = Math.ceil(diagExtent / halfH) + 4;

  // Viewport center in world coords
  var vcx = view.x + view.w / 2;
  var vcy = view.y + view.h / 2;

  // Line length: extend well past viewport edges
  var lineLen = diagExtent + 200;

  var timeFactor = time * 0.8;
  var gridAlpha = typeof atmGridAlpha !== 'undefined' ? atmGridAlpha : 0.35;

  ctx.save();
  ctx.lineWidth = 0.6;

  // ─── Set 1: Lines parallel to col-axis (row = constant) ───
  // Each line passes through world origin offset by row index.
  // Perpendicular offset from origin: row * halfH in the row-perp direction.
  // Screen: base point at (-row * halfW, row * halfH), direction (1, slope)
  for (var i = -numLines; i <= numLines; i += step) {
    var baseX = vcx + (-i * halfW);
    var baseY = vcy + (i * halfH);

    // Distance from world origin for pulse
    var dist = Math.abs(i) * halfH;
    var pulse = Math.sin(timeFactor + dist * 0.006) * 0.25 + 0.6;
    var edgeFade = 1.0 - Math.min(1, Math.abs(i) / numLines);
    var alpha = gridAlpha * pulse * edgeFade;
    if (alpha < 0.008) continue;

    ctx.strokeStyle = hexToRgba(C.hexGrid, alpha);
    ctx.beginPath();
    ctx.moveTo(baseX - lineLen, baseY - lineLen * slope);
    ctx.lineTo(baseX + lineLen, baseY + lineLen * slope);
    ctx.stroke();
  }

  // ─── Set 2: Lines parallel to row-axis (col = constant) ───
  // Direction (-1, slope) or equivalently (1, -slope) — going top-right to bottom-left
  for (var i = -numLines; i <= numLines; i += step) {
    var baseX = vcx + (i * halfW);
    var baseY = vcy + (i * halfH);

    var dist = Math.abs(i) * halfH;
    var pulse = Math.sin(timeFactor + dist * 0.006) * 0.25 + 0.6;
    var edgeFade = 1.0 - Math.min(1, Math.abs(i) / numLines);
    var alpha = gridAlpha * pulse * edgeFade;
    if (alpha < 0.008) continue;

    ctx.strokeStyle = hexToRgba(C.hexGrid, alpha);
    ctx.beginPath();
    ctx.moveTo(baseX - lineLen, baseY + lineLen * slope);
    ctx.lineTo(baseX + lineLen, baseY - lineLen * slope);
    ctx.stroke();
  }

  ctx.restore();
}
