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

// ─── Isometric grid renderer ───

function drawHexGrid(ctx, W, H, time) {
  var view = (typeof getVisibleRect === 'function')
    ? getVisibleRect(W, H)
    : { x: 0, y: 0, w: W, h: H };

  // Convert viewport corners to iso grid coords to find visible range
  var tl = screenToIso(view.x - TILE_W, view.y - TILE_H);
  var tr = screenToIso(view.x + view.w + TILE_W, view.y - TILE_H);
  var bl = screenToIso(view.x - TILE_W, view.y + view.h + TILE_H);
  var br = screenToIso(view.x + view.w + TILE_W, view.y + view.h + TILE_H);

  var minCol = Math.floor(Math.min(tl.col, bl.col)) - 1;
  var maxCol = Math.ceil(Math.max(tr.col, br.col)) + 1;
  var minRow = Math.floor(Math.min(tl.row, tr.row)) - 1;
  var maxRow = Math.ceil(Math.max(bl.row, br.row)) + 1;

  var timeSin = time * 0.5;

  ctx.save();
  ctx.strokeStyle = C.hexGrid;
  ctx.lineWidth = 0.6;

  for (var col = minCol; col <= maxCol; col++) {
    for (var row = minRow; row <= maxRow; row++) {
      var pos = isoToScreen(col, row);
      var dist = Math.sqrt(pos.x * pos.x + pos.y * pos.y);
      var pulse = Math.sin(timeSin + dist * 0.003) * 0.3 + 0.7;
      var alpha = 0.3 * pulse;

      ctx.globalAlpha = alpha;
      drawIsoDiamond(ctx, pos.x, pos.y, TILE_W, TILE_H);
      ctx.stroke();
    }
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}
