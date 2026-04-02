// ═══ NORT FLOW VIEW — BACKGROUND LAYER ═══
// Deep void with depth particles and pulsing hex grid

var flowParticles = null;

var FLOW_HEX_SIZE = 60;
var FLOW_HEX_HEIGHT = FLOW_HEX_SIZE * Math.sqrt(3);
var FLOW_HEX_R = FLOW_HEX_SIZE * 0.4;

// Pre-computed hex vertex offsets (flat-top orientation)
var FLOW_HEX_OFFSETS = [];
(function() {
  for (var i = 0; i < 6; i++) {
    var angle = (Math.PI / 3) * i - Math.PI / 2;
    FLOW_HEX_OFFSETS.push({ cos: Math.cos(angle), sin: Math.sin(angle) });
  }
})();

// ─── Depth particles ───

function initFlowParticles(W, H) {
  flowParticles = [];
  for (var i = 0; i < 80; i++) {
    flowParticles.push({
      x: Math.random() * W * 2 - W * 0.5,
      y: Math.random() * H * 2 - H * 0.5,
      size: Math.random() * 1.5 + 0.5,
      brightness: Math.random() * 0.3 + 0.05,
      speed: Math.random() * 0.15 + 0.05,
      depth: Math.random() * 0.7 + 0.3,
    });
  }
}

function _drawFlowParticles(ctx, W, H, time) {
  if (!flowParticles) initFlowParticles(W, H);
  for (var i = 0; i < flowParticles.length; i++) {
    var p = flowParticles[i];
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

// ─── Pulsing hex grid ───

function _drawFlowHexGrid(ctx, W, H, time) {
  ctx.save();
  applyCamera(ctx, _canvas);

  var size = FLOW_HEX_SIZE;
  var hexH = FLOW_HEX_HEIGHT;
  var r = FLOW_HEX_R;
  var startX = Math.floor(-camera.x / (size * 1.5)) * (size * 1.5) - size * 3;
  var startY = Math.floor(-camera.y / hexH) * hexH - hexH * 2;
  var endX = startX + W / camera.zoom + size * 6;
  var endY = startY + H / camera.zoom + hexH * 4;

  var timeSin = time * 0.5;
  ctx.strokeStyle = C.hexGrid;
  ctx.lineWidth = 0.5;

  // Quantize alpha into 4 levels to batch draw calls
  var buckets = {};
  var alphaLevels = [0.03, 0.06, 0.09, 0.12];

  for (var x = startX; x < endX; x += size * 1.5) {
    for (var y = startY; y < endY; y += hexH) {
      var colIdx = Math.round((x - startX) / (size * 1.5));
      var offsetY = (colIdx % 2 === 0) ? 0 : hexH / 2;
      var cx = x;
      var cy = y + offsetY;
      var dist = Math.sqrt(cx * cx + cy * cy);
      var pulse = Math.sin(timeSin + dist * 0.005) * 0.3 + 0.7;
      var rawAlpha = 0.12 * pulse;

      // Quantize to nearest level
      var alpha = alphaLevels[0];
      for (var a = 1; a < alphaLevels.length; a++) {
        if (Math.abs(rawAlpha - alphaLevels[a]) < Math.abs(rawAlpha - alpha)) {
          alpha = alphaLevels[a];
        }
      }

      if (!buckets[alpha]) buckets[alpha] = [];
      buckets[alpha].push(cx, cy);
    }
  }

  // Draw each alpha bucket as a single batched path
  var keys = Object.keys(buckets);
  for (var k = 0; k < keys.length; k++) {
    var al = parseFloat(keys[k]);
    var coords = buckets[keys[k]];
    ctx.globalAlpha = al;
    ctx.beginPath();
    for (var j = 0; j < coords.length; j += 2) {
      var hx = coords[j];
      var hy = coords[j + 1];
      ctx.moveTo(hx + r * FLOW_HEX_OFFSETS[0].cos, hy + r * FLOW_HEX_OFFSETS[0].sin);
      for (var v = 1; v < 6; v++) {
        ctx.lineTo(hx + r * FLOW_HEX_OFFSETS[v].cos, hy + r * FLOW_HEX_OFFSETS[v].sin);
      }
      ctx.closePath();
    }
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}

// ─── Combined background draw ───

function drawFlowBackground(ctx, W, H, time) {
  _drawFlowParticles(ctx, W, H, time);
  _drawFlowHexGrid(ctx, W, H, time);
}
