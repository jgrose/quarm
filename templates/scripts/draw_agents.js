// ═══ NORT AGENT NODE RENDERING ═══
// Enhanced hexagonal agent nodes with depth shadows, pre-rendered glows, richer state animations

// drawHexagon aliases drawHexPath from draw_background.js
var drawHexagon = drawHexPath;

function truncateText(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars - 1) + '\u2026';
}

// ─── Glow intensity by state ───

function _glowIntensity(state) {
  switch (state) {
    case 'in_progress': return 0.2;
    case 'in_manager_review':
    case 'in_specialist_review': return 0.3;
    default: return 0.1;
  }
}

// ─── Depth shadow ───

function _drawDepthShadow(ctx, node, r) {
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0)';
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.ellipse(node.x + 3, node.y + 5, r * 0.8, r * 0.4, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.shadowColor = 'rgba(0,0,0,0.3)';
  ctx.shadowBlur = 15;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;
  ctx.fill();
  ctx.restore();
}

// ─── Pre-rendered glow ───

function _drawAgentGlow(ctx, node, r, color, state, isHover) {
  var glowAlpha = _glowIntensity(state);
  if (isHover) glowAlpha = 0.35;
  var outerR = r + 20;
  var sprite = getAgentGlowSprite(color, r * 0.5, outerR, glowAlpha);
  ctx.drawImage(sprite, node.x - outerR, node.y - outerR);
}

// ─── Ambient outer ring ───

function _drawAmbientRing(ctx, node, r, color) {
  drawHexagon(ctx, node.x, node.y, r + 3);
  ctx.strokeStyle = hexToRgba(color, 0.2);
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ─── Hex body ───

function _drawHexBody(ctx, node, r, color, state, time) {
  // Fill
  drawHexagon(ctx, node.x, node.y, r);
  ctx.fillStyle = C.nodeInterior;
  ctx.fill();

  // State ring
  drawHexagon(ctx, node.x, node.y, r);
  ctx.strokeStyle = color;

  if (state === 'done') {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = hexToRgba(color, 0.6);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  } else if (state === 'in_manager_review' || state === 'in_specialist_review') {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineDashOffset = -time * 25;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  } else {
    ctx.lineWidth = (node._selected || state === 'in_progress') ? 2.5 : 2;
    ctx.stroke();
  }
}

// ─── Enhanced breathing ───

function getBreathScale(state, time) {
  switch (state) {
    case 'pending':
    case 'done':
      return 1 + Math.sin(time * 0.7) * 0.015;
    case 'in_progress':
      return 1 + Math.sin(time * 2) * 0.03;
    case 'revision':
      return 1 + Math.sin(time * 2.5) * 0.04;
    case 'in_manager_review':
    case 'in_specialist_review':
      return 1 + Math.sin(time * 1.2) * 0.08;
    default:
      return 1;
  }
}

// ─── Enhanced scanline (clipped to hex) ───

function _drawScanline(ctx, node, r, color, time, isThinking) {
  var speed = isThinking ? 40 : 15;
  var scanY = node.y - r + ((time * speed) % (r * 2));
  var halfH = 4;
  ctx.save();
  drawHexagon(ctx, node.x, node.y, r);
  ctx.clip();
  var scanGrad = ctx.createLinearGradient(node.x, scanY - halfH, node.x, scanY + halfH);
  scanGrad.addColorStop(0, hexToRgba(color, 0));
  scanGrad.addColorStop(0.5, hexToRgba(color, 0.2));
  scanGrad.addColorStop(1, hexToRgba(color, 0));
  ctx.fillStyle = scanGrad;
  ctx.fillRect(node.x - r, scanY - halfH, r * 2, halfH * 2);
  ctx.restore();
}

// ─── Orbiting particles (in_progress, 4 particles) ───

function _drawOrbitParticles(ctx, node, r, color, time) {
  for (var i = 0; i < 4; i++) {
    var angle = time * 1.5 + (i / 4) * Math.PI * 2;
    var orbR = r + 12;
    var ox = node.x + Math.cos(angle) * orbR;
    var oy = node.y + Math.sin(angle) * orbR;
    ctx.beginPath();
    ctx.fillStyle = hexToRgba(color, 0.8);
    ctx.arc(ox, oy, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ─── Waiting ripples (review states, 2 concentric hex rings) ───

function _drawWaitingRipples(ctx, node, r, color, time) {
  for (var i = 0; i < 2; i++) {
    var phase = ((time * 0.65 + i * 0.5) % 1);
    var rippleR = r + 5 + phase * 45;
    var rippleAlpha = (1 - phase) * 0.4;
    var rippleLW = 1.5 * (1 - phase);
    if (rippleLW < 0.1) continue;
    drawHexagon(ctx, node.x, node.y, rippleR);
    ctx.strokeStyle = hexToRgba(color, rippleAlpha);
    ctx.lineWidth = rippleLW;
    ctx.stroke();
  }
}

// ─── Waiting orbit particles (review states, 3 particles) ───

function _drawWaitingOrbitParticles(ctx, node, r, color, time) {
  for (var i = 0; i < 3; i++) {
    var angle = time * 0.8 + (i / 3) * Math.PI * 2;
    var orbR = r + 14;
    var ox = node.x + Math.cos(angle) * orbR;
    var oy = node.y + Math.sin(angle) * orbR;
    ctx.beginPath();
    ctx.fillStyle = hexToRgba(color, 0.8);
    ctx.arc(ox, oy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ─── Dashed ring (revision / review) ───

function _drawDashedRing(ctx, node, r, color, time) {
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineDashOffset = -time * 25;
  drawHexagon(ctx, node.x, node.y, r + 6);
  ctx.strokeStyle = hexToRgba(color, 0.5);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

// ─── Error arc ───

function _drawErrorArc(ctx, node, r, color, time) {
  var intensity = 0.3 + Math.sin(time * 4) * 0.2;
  var grad = ctx.createRadialGradient(node.x, node.y, r * 0.8, node.x, node.y, r + 15);
  grad.addColorStop(0, hexToRgba(color, intensity * 0.5));
  grad.addColorStop(1, hexToRgba(color, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(node.x, node.y, r + 15, 0, Math.PI * 2);
  ctx.fill();
}

// ─── Center icons by state ───

function _drawCenterIcon(ctx, node, r, color, time) {
  var fontSize = Math.floor(r * 0.4);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  switch (node.state) {
    case 'in_progress': {
      // Spinning gear
      ctx.save();
      ctx.translate(node.x, node.y);
      ctx.rotate(time * 2);
      ctx.fillStyle = hexToRgba(color, 0.8);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText('\u2699', 0, 0);
      ctx.restore();
      return;
    }
    case 'done': {
      // Checkmark
      ctx.fillStyle = hexToRgba(color, 0.8);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText('\u2713', node.x, node.y);
      return;
    }
    case 'in_manager_review': {
      // Diamond
      ctx.fillStyle = hexToRgba(color, 0.7);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText('\u25C7', node.x, node.y);
      return;
    }
    case 'in_specialist_review': {
      // Eye
      ctx.fillStyle = hexToRgba(color, 0.7);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText('\u25C8', node.x, node.y);
      return;
    }
    case 'failed': {
      // X mark
      ctx.fillStyle = hexToRgba(color, 0.9);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText('\u2715', node.x, node.y);
      return;
    }
    default: {
      // Idle/pending: tier icon
      var t = TIERS[node.tier] || TIERS.drone;
      ctx.fillStyle = hexToRgba(color, 0.7);
      ctx.font = fontSize + 'px monospace';
      ctx.fillText(t.icon, node.x, node.y);
      return;
    }
  }
}

// ─── Name label ───

function _drawNameLabel(ctx, node, r) {
  ctx.fillStyle = C.textPrimary;
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(truncateText(node.name, 18), node.x, node.y + r + 8);
}

// ─── Model badge (above) ───

function _drawModelBadge(ctx, node, r) {
  if (!node.model) return;
  ctx.fillStyle = C.textMuted;
  ctx.font = '7px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(node.model, node.x, node.y - r - 6);
}

// ─── Token badge ───

function _drawTokenBadge(ctx, node, r) {
  if (!node.tokens) return;
  ctx.fillStyle = C.textMuted;
  ctx.font = '7px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  var label = node.tokens >= 1000 ? (node.tokens / 1000).toFixed(1) + 'k' : String(node.tokens);
  ctx.fillText(label + ' tok', node.x, node.y + r + 20);
}

// ─── Draw single agent ───

function _drawSingleAgent(ctx, node, time) {
  var color = getStateColor(node.state);
  var breath = getBreathScale(node.state, time);
  var r = node.radius * breath * node.scale;
  if (r < 1) return;

  var isHover = !!node._hover;

  ctx.save();
  ctx.globalAlpha = node.opacity;

  // 1. Depth shadow
  _drawDepthShadow(ctx, node, r);

  // 2. Pre-rendered glow
  _drawAgentGlow(ctx, node, r, color, node.state, isHover);

  // 3. Ambient outer ring
  _drawAmbientRing(ctx, node, r, color);

  // 4. Hex body with state ring
  _drawHexBody(ctx, node, r, color, node.state, time);

  // 5. State-specific effects
  if (node.state === 'in_progress') {
    _drawScanline(ctx, node, r, color, time, true);
    _drawOrbitParticles(ctx, node, r, color, time);
  }

  if (node.state === 'in_manager_review' || node.state === 'in_specialist_review') {
    _drawWaitingRipples(ctx, node, r, color, time);
    _drawWaitingOrbitParticles(ctx, node, r, color, time);
    _drawDashedRing(ctx, node, r, color, time);
  }

  if (node.state === 'revision') {
    _drawDashedRing(ctx, node, r, color, time);
    _drawScanline(ctx, node, r, color, time, false);
  }

  if (node.state === 'failed') {
    _drawErrorArc(ctx, node, r, color, time);
  }

  // 6. Center icon
  _drawCenterIcon(ctx, node, r, color, time);

  // 7. Name label below
  _drawNameLabel(ctx, node, r);

  // 8. Optional stat badges
  if (config.nodeStats) {
    _drawModelBadge(ctx, node, r);
    _drawTokenBadge(ctx, node, r);
  }

  ctx.restore();
}

// ─── Draw all agents (public API) ───

function drawAllAgents(ctx, time) {
  for (var entry of nodes) {
    var node = entry[1];
    _drawSingleAgent(ctx, node, time);
  }
}
