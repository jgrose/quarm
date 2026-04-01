// ═══ NORT CONTEXT BREAKDOWN VISUALIZATION ═══
// Token usage bars below agents and context ring around NEXUS

var DEFAULT_CONTEXT_WINDOW = 200000;

function _formatTokenCount(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return String(n);
}

function _contextSegments(node) {
  var hasTools = node.toolCalls && node.toolCalls.length > 0;
  if (hasTools) {
    return [
      { ratio: 0.10, color: C.contextSystem },
      { ratio: 0.20, color: C.contextUser },
      { ratio: 0.30, color: C.contextTool },
      { ratio: 0.40, color: C.contextReasoning },
    ];
  }
  return [
    { ratio: 0.20, color: C.contextSystem },
    { ratio: 0.30, color: C.contextUser },
    { ratio: 0.50, color: C.contextReasoning },
  ];
}

function _drawContextBar(ctx, node) {
  var tokens = node.tokens;
  if (!tokens || tokens <= 0) return;

  var ctxWindow = node.contextWindow || DEFAULT_CONTEXT_WINDOW;
  var barWidth = Math.max(60, node.radius * 2.2);
  var barHeight = 6;
  var barX = node.x - barWidth / 2;
  var barY = node.y + node.radius + 22;

  // Background
  ctx.fillStyle = 'rgba(100,200,255,0.05)';
  ctx.beginPath();
  ctx.roundRect(barX, barY, barWidth, barHeight, 3);
  ctx.fill();

  // Segments
  var segs = _contextSegments(node);
  var fillWidth = (tokens / ctxWindow) * barWidth;
  var segX = barX;
  for (var i = 0; i < segs.length; i++) {
    var segW = segs[i].ratio * fillWidth;
    if (segW < 0.5) continue;
    ctx.fillStyle = segs[i].color;
    ctx.fillRect(segX, barY, segW, barHeight);
    segX += segW;
  }

  // Border
  ctx.strokeStyle = C.glassBorder;
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.roundRect(barX, barY, barWidth, barHeight, 3);
  ctx.stroke();

  // Token count label
  ctx.fillStyle = C.textMuted;
  ctx.font = '7px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(
    _formatTokenCount(tokens) + ' / ' + _formatTokenCount(ctxWindow) + ' tokens',
    node.x,
    barY + barHeight + 3
  );
}

function _drawContextRing(ctx, node, time) {
  var tokens = node.tokens;
  if (!tokens || tokens <= 0) return;
  if (node.tier !== 'nexus') return;

  var ctxWindow = node.contextWindow || DEFAULT_CONTEXT_WINDOW;
  var usage = tokens / ctxWindow;
  var ringR = node.radius + 8;
  var ringW = 4;
  var startAngle = -Math.PI / 2;

  // Background ring (empty capacity)
  ctx.beginPath();
  ctx.arc(node.x, node.y, ringR, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(100,200,255,0.06)';
  ctx.lineWidth = ringW;
  ctx.stroke();

  // Filled segments as arcs
  var segs = _contextSegments(node);
  var currentAngle = startAngle;
  for (var i = 0; i < segs.length; i++) {
    var sweep = segs[i].ratio * usage * Math.PI * 2;
    if (sweep < 0.005) { currentAngle += sweep; continue; }
    ctx.beginPath();
    ctx.arc(node.x, node.y, ringR, currentAngle, currentAngle + sweep);
    ctx.strokeStyle = segs[i].color;
    ctx.lineWidth = ringW;
    ctx.stroke();
    currentAngle += sweep;
  }

  // Warning / critical pulsing glow
  if (usage > 0.8) {
    var isCritical = usage > 0.9;
    var warningColor = isCritical ? C.failed : C.in_manager_review;
    var intensity = isCritical
      ? 0.35 + Math.sin(time * 6) * 0.2
      : 0.15 + Math.sin(time * 3) * 0.1;

    ctx.save();
    ctx.beginPath();
    ctx.arc(node.x, node.y, ringR + 4, 0, Math.PI * 2);
    ctx.strokeStyle = warningColor;
    ctx.lineWidth = 2;
    ctx.globalAlpha = intensity;
    ctx.shadowColor = warningColor;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.restore();
  }

  // Percentage label when usage > 70%
  if (usage > 0.7) {
    var pctColor = usage > 0.9 ? C.failed : usage > 0.8 ? C.in_manager_review : C.textDim;
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = pctColor;
    ctx.fillText(Math.floor(usage * 100) + '%', node.x, node.y - node.radius - 10);
  }
}

function drawAllContextBars(ctx, time) {
  for (var entry of nodes) {
    var node = entry[1];
    if (node.opacity < 0.05) continue;
    ctx.save();
    ctx.globalAlpha = node.opacity;
    _drawContextRing(ctx, node, time);
    _drawContextBar(ctx, node);
    ctx.restore();
  }
}
