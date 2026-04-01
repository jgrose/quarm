// ═══ NORT EDGE AND PARTICLE RENDERING ═══
// Tapered bezier beams with wobble, ported from agent-flow draw-edges.ts / draw-particles.ts

// ─── Bezier helpers ───

var _EDGE_SEGMENTS = 16;
var _EDGE_CURVATURE = 0.15;
var _EDGE_CP1 = 0.33;
var _EDGE_CP2 = 0.66;

// Edge width presets
var _BW_PARENT = { startW: 3, endW: 1 };
var _BW_TOOL   = { startW: 1.5, endW: 0.5 };
var _BW_GLOW   = { startW: 3, endW: 1, alpha: 0.08 };

// Edge alpha
var _EDGE_IDLE_ALPHA   = 0.08;
var _EDGE_ACTIVE_ALPHA = 0.3;

function _cubicBez(t, p0, p1, p2, p3) {
  var mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}

function _computeCP(fromX, fromY, toX, toY) {
  var dx = toX - fromX, dy = toY - fromY;
  var dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null;
  var curvature = dist * _EDGE_CURVATURE;
  var perpX = (-dy / dist) * curvature;
  var perpY = (dx / dist) * curvature;
  return {
    cp1x: fromX + dx * _EDGE_CP1 + perpX,
    cp1y: fromY + dy * _EDGE_CP1 + perpY,
    cp2x: fromX + dx * _EDGE_CP2 - perpX,
    cp2y: fromY + dy * _EDGE_CP2 - perpY,
    dist: dist, dx: dx, dy: dy,
  };
}

function _bezNormal(t, fx, fy, c1x, c1y, c2x, c2y, tx, ty, halfW) {
  var x = _cubicBez(t, fx, c1x, c2x, tx);
  var y = _cubicBez(t, fy, c1y, c2y, ty);
  var dt = 0.001;
  var t0 = Math.max(0, t - dt), t1 = Math.min(1, t + dt);
  var tdx = _cubicBez(t1, fx, c1x, c2x, tx) - _cubicBez(t0, fx, c1x, c2x, tx);
  var tdy = _cubicBez(t1, fy, c1y, c2y, ty) - _cubicBez(t0, fy, c1y, c2y, ty);
  var len = Math.sqrt(tdx * tdx + tdy * tdy) || 1;
  return { x: x, y: y, nx: (-tdy / len) * halfW, ny: (tdx / len) * halfW };
}

// ─── Public bezierPoint (curved control points) ───

function bezierPoint(fromNode, toNode, t) {
  var cp = _computeCP(fromNode.x, fromNode.y, toNode.x, toNode.y);
  if (!cp) return { x: fromNode.x, y: fromNode.y };
  return {
    x: _cubicBez(t, fromNode.x, cp.cp1x, cp.cp2x, toNode.x),
    y: _cubicBez(t, fromNode.y, cp.cp1y, cp.cp2y, toNode.y),
  };
}

// ─── Tapered bezier fill ───

function _drawTaperedBezier(ctx, fx, fy, c1x, c1y, c2x, c2y, tx, ty, startW, endW, color, alpha) {
  if (alpha < 0.005) return;
  var steps = _EDGE_SEGMENTS;
  ctx.beginPath();
  // Forward pass: left side
  for (var i = 0; i <= steps; i++) {
    var t = i / steps;
    var halfW = (startW + (endW - startW) * t) / 2;
    var p = _bezNormal(t, fx, fy, c1x, c1y, c2x, c2y, tx, ty, halfW);
    if (i === 0) ctx.moveTo(p.x + p.nx, p.y + p.ny);
    else ctx.lineTo(p.x + p.nx, p.y + p.ny);
  }
  // Reverse pass: right side
  for (var i = steps; i >= 0; i--) {
    var t = i / steps;
    var halfW = (startW + (endW - startW) * t) / 2;
    var p = _bezNormal(t, fx, fy, c1x, c1y, c2x, c2y, tx, ty, halfW);
    ctx.lineTo(p.x - p.nx, p.y - p.ny);
  }
  ctx.closePath();
  ctx.fillStyle = hexToRgba(color, alpha);
  ctx.fill();
}

// ─── Active edge detection ───

function _getActiveEdgeKeys(particleList) {
  var keys = {};
  for (var i = 0; i < particleList.length; i++) {
    keys[particleList[i].from + '|' + particleList[i].to] = true;
  }
  return keys;
}

// ─── Draw all edges ───

function drawAllEdges(ctx, time) {
  var activeKeys = _getActiveEdgeKeys(particles);

  for (var i = 0; i < edges.length; i++) {
    var e = edges[i];
    var fromNode = nodes.get(e.from);
    var toNode = nodes.get(e.to);
    if (!fromNode || !toNode) continue;
    if (fromNode.opacity < 0.05 || toNode.opacity < 0.05) continue;

    var cp = _computeCP(fromNode.x, fromNode.y, toNode.x, toNode.y);
    if (!cp) continue;

    var edgeKey = e.from + '|' + e.to;
    var hasActive = !!activeKeys[edgeKey];
    var baseAlpha = hasActive ? _EDGE_ACTIVE_ALPHA : _EDGE_IDLE_ALPHA;
    var pulsing = hasActive ? (Math.sin(time * 4) * 0.1 + 0.9) : 1;
    var alpha = baseAlpha * pulsing * Math.min(fromNode.opacity, toNode.opacity);

    var isToolReview = (e.activity > 0.85 && e.label === 'REVIEW');
    var bw = isToolReview ? _BW_TOOL : _BW_PARENT;

    ctx.save();

    // Main tapered beam
    _drawTaperedBezier(ctx,
      fromNode.x, fromNode.y, cp.cp1x, cp.cp1y, cp.cp2x, cp.cp2y, toNode.x, toNode.y,
      bw.startW, bw.endW, e.color, alpha);

    // Glow layer (wider, dimmer)
    if (hasActive || e.activity > 0.5) {
      _drawTaperedBezier(ctx,
        fromNode.x, fromNode.y, cp.cp1x, cp.cp1y, cp.cp2x, cp.cp2y, toNode.x, toNode.y,
        bw.startW + _BW_GLOW.startW, bw.endW + _BW_GLOW.endW, e.color, _BW_GLOW.alpha * pulsing);
    }

    ctx.restore();
  }
}

// ─── Particles ───

function updateParticles(dt) {
  for (var i = particles.length - 1; i >= 0; i--) {
    var p = particles[i];
    p.progress += p.speed * dt;
    if (p.progress >= 1) {
      var toNode = nodes.get(p.to);
      if (toNode) {
        spawnCompleteEffect(toNode.x, toNode.y, p.color);
      }
      particles.splice(i, 1);
    }
  }
}

function drawAllParticles(ctx, time) {
  for (var i = 0; i < particles.length; i++) {
    var p = particles[i];
    var fromNode = nodes.get(p.from);
    var toNode = nodes.get(p.to);
    if (!fromNode || !toNode) continue;

    var cp = _computeCP(fromNode.x, fromNode.y, toNode.x, toNode.y);
    if (!cp) continue;

    var t = p.progress;

    // Compute tangent direction for wobble
    var tangentX = cp.dx / cp.dist;
    var tangentY = cp.dy / cp.dist;
    var normalX = -tangentY;
    var normalY = tangentX;

    // Wobble: perpendicular displacement, amplitude 3px, freq 10 rad/s
    var wobblePhase = i * 7;
    var wobbleAmt = Math.sin(time * 10 + wobblePhase) * 3 * Math.sin(t * Math.PI);

    var baseX = _cubicBez(t, fromNode.x, cp.cp1x, cp.cp2x, toNode.x);
    var baseY = _cubicBez(t, fromNode.y, cp.cp1y, cp.cp2y, toNode.y);
    var px = baseX + normalX * wobbleAmt;
    var py = baseY + normalY * wobbleAmt;

    ctx.save();

    // Comet trail: 8 segments behind particle
    for (var s = 8; s >= 0; s--) {
      var trailOffset = (s / 8) * 0.12;
      var tt = Math.max(0, t - trailOffset);
      var wob = Math.sin(time * 10 + wobblePhase) * 3 * Math.sin(tt * Math.PI);
      var tx = _cubicBez(tt, fromNode.x, cp.cp1x, cp.cp2x, toNode.x) + normalX * wob;
      var ty = _cubicBez(tt, fromNode.y, cp.cp1y, cp.cp2y, toNode.y) + normalY * wob;
      var trailAlpha = ((8 - s) / 8) * 0.6;
      var trailSize = p.size * ((8 - s) / 8);
      if (trailSize < 0.2) continue;
      ctx.beginPath();
      ctx.fillStyle = p.color + alphaHex(trailAlpha);
      ctx.arc(tx, ty, trailSize, 0, Math.PI * 2);
      ctx.fill();
    }

    // Glow core: pre-rendered sprite
    var glowSprite = getGlowSprite(p.color, 15, 0.4, 0);
    ctx.drawImage(glowSprite, px - 15, py - 15);

    // Particle core
    ctx.beginPath();
    ctx.fillStyle = p.color;
    ctx.arc(px, py, p.size, 0, Math.PI * 2);
    ctx.fill();

    // Bright highlight at 40% size
    ctx.beginPath();
    ctx.fillStyle = C.holoHot + '80';
    ctx.arc(px, py, p.size * 0.4, 0, Math.PI * 2);
    ctx.fill();

    // Label near particle (between t=0.2 and t=0.8)
    if (p.label && t > 0.2 && t < 0.8) {
      ctx.fillStyle = p.color + 'aa';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(p.label, px, py - 12);
    }

    ctx.restore();
  }
}
