// ═══ TRON MINIMAP — ISOMETRIC CITY OVERVIEW ═══
// Fixed-position minimap showing the entire grid, buildings, programs, and viewport

var _minimapBounds = null;  // { minX, minY, maxX, maxY } world coords of iso grid
var MINIMAP_W = 220;
var MINIMAP_H = 140;
var MINIMAP_PAD = 12;       // padding from screen edge
var MINIMAP_INSET = 8;      // inner padding inside panel

// ═══════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════

function initMinimap() {
  // Compute world bounds from the iso grid extremes (0,0) to (39,39)
  var topLeft = isoToScreen(0, 0);
  var topRight = isoToScreen(ISO_GRID_COLS - 1, 0);
  var bottomLeft = isoToScreen(0, ISO_GRID_ROWS - 1);
  var bottomRight = isoToScreen(ISO_GRID_COLS - 1, ISO_GRID_ROWS - 1);

  // The iso diamond has its extremes at: top=(topLeft), right=(topRight),
  // bottom=(bottomRight), left=(bottomLeft) — but we need the bounding box
  var allX = [topLeft.x, topRight.x, bottomLeft.x, bottomRight.x];
  var allY = [topLeft.y, topRight.y, bottomLeft.y, bottomRight.y];

  _minimapBounds = {
    minX: Math.min.apply(null, allX) - TILE_W,
    maxX: Math.max.apply(null, allX) + TILE_W,
    minY: Math.min.apply(null, allY) - TILE_H,
    maxY: Math.max.apply(null, allY) + TILE_H,
  };
}

// ═══════════════════════════════════════════════════
//  COORDINATE MAPPING
// ═══════════════════════════════════════════════════

function worldToMinimap(wx, wy, mx, my, mw, mh) {
  if (!_minimapBounds) return { x: mx, y: my };
  var bounds = _minimapBounds;
  var nx = (wx - bounds.minX) / (bounds.maxX - bounds.minX);
  var ny = (wy - bounds.minY) / (bounds.maxY - bounds.minY);
  return {
    x: mx + nx * mw,
    y: my + ny * mh,
  };
}

// ═══════════════════════════════════════════════════
//  CLICK HANDLING
// ═══════════════════════════════════════════════════

function handleMinimapClick(e) {
  if (!config.minimap) return false;
  if (!_minimapBounds) return false;

  var canvas = document.getElementById('canvas');
  if (!canvas) return false;

  var rect = canvas.getBoundingClientRect();
  var sx = e.clientX - rect.left;
  var sy = e.clientY - rect.top;

  // Minimap screen position (bottom-left)
  var H = canvas.height / dpr;
  var panelX = MINIMAP_PAD;
  var panelY = H - MINIMAP_H - MINIMAP_PAD;
  var drawX = panelX + MINIMAP_INSET;
  var drawY = panelY + MINIMAP_INSET;
  var drawW = MINIMAP_W - MINIMAP_INSET * 2;
  var drawH = MINIMAP_H - MINIMAP_INSET * 2;

  // Check if click is inside the minimap
  if (sx < panelX || sx > panelX + MINIMAP_W) return false;
  if (sy < panelY || sy > panelY + MINIMAP_H) return false;

  // Convert click to world coordinates
  var bounds = _minimapBounds;
  var nx = (sx - drawX) / drawW;
  var ny = (sy - drawY) / drawH;
  nx = Math.max(0, Math.min(1, nx));
  ny = Math.max(0, Math.min(1, ny));

  var worldX = bounds.minX + nx * (bounds.maxX - bounds.minX);
  var worldY = bounds.minY + ny * (bounds.maxY - bounds.minY);

  // Center camera on this world position
  var W = canvas.width / dpr;
  camera.x = W / 2 - worldX;
  camera.y = H / 2 - worldY;

  return true;  // consumed the click
}

// ═══════════════════════════════════════════════════
//  DRAW (screen-space)
// ═══════════════════════════════════════════════════

function drawMinimap(ctx, W, H, time) {
  if (!config.minimap) return;
  if (!_minimapBounds) initMinimap();
  if (!_minimapBounds) return;  // still null = can't compute

  ctx.save();
  ctx.scale(dpr, dpr);

  // Panel position: bottom-left corner
  var panelX = MINIMAP_PAD;
  var panelY = H - MINIMAP_H - MINIMAP_PAD;

  // ─── Glass panel background ───
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = C.glassBg;
  ctx.strokeStyle = C.glassBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, MINIMAP_W, MINIMAP_H, 6);
  ctx.fill();
  ctx.stroke();

  // Drawing area inside panel (with inset)
  var drawX = panelX + MINIMAP_INSET;
  var drawY = panelY + MINIMAP_INSET;
  var drawW = MINIMAP_W - MINIMAP_INSET * 2;
  var drawH = MINIMAP_H - MINIMAP_INSET * 2;

  // Clip to minimap area
  ctx.save();
  ctx.beginPath();
  ctx.rect(drawX, drawY, drawW, drawH);
  ctx.clip();

  // ─── Road lines (faint) ───
  if (_roadTiles && _roadTiles.length > 0) {
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = C.hexGrid;
    for (var r = 0; r < _roadTiles.length; r++) {
      var rt = _roadTiles[r];
      var rp = worldToMinimap(rt.x, rt.y, drawX, drawY, drawW, drawH);
      ctx.fillRect(rp.x - 0.5, rp.y - 0.5, 1, 1);
    }
  }

  // ─── Building dots ───
  if (typeof gridLocations !== 'undefined' && gridLocations.length > 0) {
    for (var i = 0; i < gridLocations.length; i++) {
      var loc = gridLocations[i];
      var lp = worldToMinimap(loc.x, loc.y, drawX, drawY, drawW, drawH);

      ctx.globalAlpha = 0.8;
      ctx.fillStyle = loc.glowColor || C.holoBase;
      ctx.fillRect(lp.x - 2, lp.y - 2, 4, 4);
    }
  }

  // ─── Ambient program dots ───
  if (typeof ambientPrograms !== 'undefined' && ambientPrograms.length > 0) {
    for (var j = 0; j < ambientPrograms.length; j++) {
      var prog = ambientPrograms[j];
      var pp = worldToMinimap(prog.x, prog.y, drawX, drawY, drawW, drawH);

      ctx.globalAlpha = 0.6;
      ctx.fillStyle = prog.glow || C.holoBase;
      ctx.fillRect(pp.x - 1, pp.y - 1, 2, 2);
    }
  }

  // ─── Viewport rectangle ───
  var view = getVisibleRect(W, H);
  var vtl = worldToMinimap(view.x, view.y, drawX, drawY, drawW, drawH);
  var vbr = worldToMinimap(view.x + view.w, view.y + view.h, drawX, drawY, drawW, drawH);
  var vw = vbr.x - vtl.x;
  var vh = vbr.y - vtl.y;

  ctx.globalAlpha = 0.7;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.strokeRect(vtl.x, vtl.y, vw, vh);

  ctx.restore();  // end clip

  // ─── Label ───
  ctx.globalAlpha = 0.5;
  ctx.font = '7px monospace';
  ctx.fillStyle = C.textDim;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('MINIMAP', panelX + 6, panelY - 2);

  ctx.restore();  // end dpr scale
}
