// ═══ NORT AGENT NODE RENDERING ═══
// Enhanced hexagonal agent nodes with depth shadows, pre-rendered glows, richer state animations

var drawHexagon = drawHexPath;
var CLAUDE_SPARK_D = 'M142.27 316.619l73.655-41.326 1.238-3.589-1.238-1.996-3.589-.001-12.31-.759-42.084-1.138-36.498-1.516-35.361-1.896-8.897-1.895-8.34-10.995.859-5.484 7.482-5.03 10.717.935 23.683 1.617 35.537 2.452 25.782 1.517 38.193 3.968h6.064l.86-2.451-2.073-1.517-1.618-1.517-36.776-24.922-39.81-26.338-20.852-15.166-11.273-7.683-5.687-7.204-2.451-15.721 10.237-11.273 13.75.935 3.513.936 13.928 10.716 29.749 23.027 38.848 28.612 5.687 4.727 2.275-1.617.278-1.138-2.553-4.271-21.13-38.193-22.546-38.848-10.035-16.101-2.654-9.655c-.935-3.968-1.617-7.304-1.617-11.374l11.652-15.823 6.445-2.073 15.545 2.073 6.547 5.687 9.655 22.092 15.646 34.78 24.265 47.291 7.103 14.028 3.791 12.992 1.416 3.968 2.449-.001v-2.275l1.997-26.641 3.69-32.707 3.589-42.084 1.239-11.854 5.863-14.206 11.652-7.683 9.099 4.348 7.482 10.716-1.036 6.926-4.449 28.915-8.72 45.294-5.687 30.331h3.313l3.792-3.791 15.342-20.372 25.782-32.227 11.374-12.789 13.27-14.129 8.517-6.724 16.1-.001 11.854 17.617-5.307 18.199-16.581 21.029-13.75 17.819-19.716 26.54-12.309 21.231 1.138 1.694 2.932-.278 44.536-9.479 24.062-4.347 28.714-4.928 12.992 6.066 1.416 6.167-5.106 12.613-30.71 7.583-36.018 7.204-53.636 12.689-.657.48.758.935 24.164 2.275 10.337.556h25.301l47.114 3.514 12.309 8.139 7.381 9.959-1.238 7.583-18.957 9.655-25.579-6.066-59.702-14.205-20.474-5.106-2.83-.001v1.694l17.061 16.682 31.266 28.233 39.152 36.397 1.997 8.999-5.03 7.102-5.307-.758-34.401-25.883-13.27-11.651-30.053-25.302-1.996-.001v2.654l6.926 10.136 36.574 54.975 1.895 16.859-2.653 5.485-9.479 3.311-10.414-1.895-21.408-30.054-22.092-33.844-17.819-30.331-2.173 1.238-10.515 113.261-4.929 5.788-11.374 4.348-9.478-7.204-5.03-11.652 5.03-23.027 6.066-30.052 4.928-23.886 4.449-29.674 2.654-9.858-.177-.657-2.173.278-22.37 30.71-34.021 45.977-26.919 28.815-6.445 2.553-11.173-5.789 1.037-10.337 6.243-9.2 37.257-47.392 22.47-29.371 14.508-16.961-.101-2.451h-.859l-98.954 64.251-17.618 2.275-7.583-7.103.936-11.652 3.589-3.791 29.749-20.474-.101.102.024.101z';
var _claudeSparkPath = null;

function _getClaudeSparkPath() {
  if (!_claudeSparkPath) _claudeSparkPath = new Path2D(CLAUDE_SPARK_D);
  return _claudeSparkPath;
}

function drawClaudeSpark(ctx, cx, cy, r, color) {
  ctx.save();
  ctx.translate(cx, cy);
  var scale = (r * 0.45) / 256;  // sparkScale=0.45, sparkViewBox=256
  ctx.scale(scale, scale);
  ctx.translate(-256, -256 + 1);
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 6 / scale;
  ctx.fill(_getClaudeSparkPath());
  ctx.restore();
}

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
  var sq = config.shadowQuality || 'high';
  if (sq === 'off') return;
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0)';
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.ellipse(node.x + 3, node.y + 5, r * 0.8, r * 0.4, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.shadowColor = 'rgba(0,0,0,0.3)';
  ctx.shadowBlur = sq === 'low' ? 5 : 15;
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

// ─── Orbiting particles (parameterized) ───

function _drawOrbitParticles(ctx, node, r, color, time, count, speed, orbOffset, dotSize) {
  for (var i = 0; i < (count || 4); i++) {
    var angle = time * (speed || 1.5) + (i / (count || 4)) * Math.PI * 2;
    var orbR = r + (orbOffset || 12);
    ctx.beginPath();
    ctx.fillStyle = hexToRgba(color, 0.8);
    ctx.arc(node.x + Math.cos(angle) * orbR, node.y + Math.sin(angle) * orbR, dotSize || 1.5, 0, Math.PI * 2);
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
  // Flow mode: nexus node gets the spark logo instead of state icon
  if (config.flowMode && node.tier === 'nexus') {
    drawClaudeSpark(ctx, node.x, node.y, r, color);
    return;
  }

  var fontSize = Math.floor(r * 0.4);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = fontSize + 'px monospace';

  var _icons = { in_progress: ['\u2699', 0.8], done: ['\u2713', 0.8], in_manager_review: ['\u25C7', 0.7],
    in_specialist_review: ['\u25C8', 0.7], failed: ['\u2715', 0.9] };
  var iconInfo = _icons[node.state];

  if (node.state === 'in_progress') {
    ctx.save();
    ctx.translate(node.x, node.y);
    ctx.rotate(time * 2);
    ctx.fillStyle = hexToRgba(color, 0.8);
    ctx.fillText('\u2699', 0, 0);
    ctx.restore();
  } else if (iconInfo) {
    ctx.fillStyle = hexToRgba(color, iconInfo[1]);
    ctx.fillText(iconInfo[0], node.x, node.y);
  } else {
    var t = TIERS[node.tier] || TIERS.drone;
    ctx.fillStyle = hexToRgba(color, 0.7);
    ctx.fillText(t.icon, node.x, node.y);
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

// ─── LOD level based on zoom ───

function _getLodLevel() {
  if (!config.lodEnabled) return 0;
  if (camera.zoom >= 0.8) return 0;
  if (camera.zoom >= 0.4) return 1;
  return 2;
}

// ─── Draw single agent ───

function _drawSingleAgent(ctx, node, time, lod) {
  var color = getStateColor(node.state);
  var breath = getBreathScale(node.state, time);
  var r = node.radius * breath * node.scale;
  if (r < 1) return;

  var isHover = !!node._hover;

  ctx.save();
  ctx.globalAlpha = node.opacity;

  if (lod <= 1) {
    // 1. Depth shadow (LOD 0-1, simplified at LOD 1)
    _drawDepthShadow(ctx, node, r);
  }

  if (lod === 0) {
    // 2. Pre-rendered glow (LOD 0 only)
    _drawAgentGlow(ctx, node, r, color, node.state, isHover);
  }

  if (lod <= 1) {
    // 3. Ambient outer ring (LOD 0-1)
    _drawAmbientRing(ctx, node, r, color);
    // 4. Hex body with state ring
    _drawHexBody(ctx, node, r, color, node.state, time);
  } else {
    // LOD 2: simple filled circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = C.nodeInterior;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // 5. State-specific effects (LOD 0 only)
  if (lod === 0) {
    if (node.state === 'in_progress') {
      _drawScanline(ctx, node, r, color, time, true);
      _drawOrbitParticles(ctx, node, r, color, time);
    }

    if (node.state === 'in_manager_review' || node.state === 'in_specialist_review') {
      _drawWaitingRipples(ctx, node, r, color, time);
      _drawOrbitParticles(ctx, node, r, color, time, 3, 0.8, 14, 2);
      _drawDashedRing(ctx, node, r, color, time);
    }

    if (node.state === 'revision') {
      _drawDashedRing(ctx, node, r, color, time);
      _drawScanline(ctx, node, r, color, time, false);
    }
  }

  if (lod <= 1 && node.state === 'failed') {
    _drawErrorArc(ctx, node, r, color, time);
  }

  // 6. Center icon (LOD 0-1)
  if (lod <= 1) {
    _drawCenterIcon(ctx, node, r, color, time);
  }

  // 7. Name label below
  _drawNameLabel(ctx, node, r);

  // 8. Optional stat badges (LOD 0 only)
  if (lod === 0 && config.nodeStats) {
    _drawModelBadge(ctx, node, r);
    _drawTokenBadge(ctx, node, r);
  }

  ctx.restore();
}

// ─── Draw all agents (public API) ───

function drawAllAgents(ctx, time, W, H) {
  var lod = _getLodLevel();
  var view = null;
  var margin = 60;

  if (config.viewportCulling && W && H) {
    view = getVisibleRect(W, H);
  }

  for (var entry of nodes) {
    var node = entry[1];
    if (view) {
      if (node.x < view.x - margin || node.x > view.x + view.w + margin ||
          node.y < view.y - margin || node.y > view.y + view.h + margin) continue;
    }
    _drawSingleAgent(ctx, node, time, lod);
  }
}
