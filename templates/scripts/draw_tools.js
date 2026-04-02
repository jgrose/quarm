// ═══ NORT TOOL CALL CARD RENDERING ═══
// Small cards rendered near DRONE nodes for active tool calls

function drawToolCard(ctx, node, tool, index, time) {
  // Radial position around the parent node
  var count = node.toolCalls.length || 1;
  var angleStep = (Math.PI * 2) / Math.max(count, 3);
  var baseAngle = -Math.PI / 2;
  var angle = baseAngle + angleStep * index;
  var dist = node.radius + 60;
  var cx = node.x + Math.cos(angle) * dist;
  var cy = node.y + Math.sin(angle) * dist;

  var w = TOOL_CARD_W;
  var h = TOOL_CARD_H;
  var halfW = w / 2;
  var halfH = h / 2;
  var toolOpacity = tool.opacity !== undefined ? tool.opacity : 1;

  ctx.save();
  ctx.globalAlpha = node.opacity * toolOpacity;

  // Background
  ctx.beginPath();
  ctx.roundRect(cx - halfW, cy - halfH, w, h, 4);
  ctx.fillStyle = 'rgba(5, 5, 16, 0.8)';
  ctx.fill();

  // Border color based on tool state
  var borderColor;
  if (tool.state === 'running') {
    borderColor = C.review; // amber
  } else if (tool.state === 'complete') {
    borderColor = C.done; // green
  } else if (tool.state === 'error') {
    borderColor = C.failed; // red
  } else {
    borderColor = C.holoBase;
  }

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Spinning arc indicator for running state
  if (tool.state === 'running') {
    ctx.save();
    ctx.strokeStyle = hexToRgba(borderColor, 0.7);
    ctx.lineWidth = 2;
    var spinAngle = time * 3;
    ctx.beginPath();
    ctx.arc(cx - halfW + 12, cy, 6, spinAngle, spinAngle + Math.PI * 1.2);
    ctx.stroke();
    ctx.restore();
  }

  // Checkmark for complete state
  if (tool.state === 'complete') {
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - halfW + 8, cy);
    ctx.lineTo(cx - halfW + 11, cy + 3);
    ctx.lineTo(cx - halfW + 16, cy - 3);
    ctx.stroke();
  }

  // Error pulsing glow
  if (tool.state === 'error') {
    var errIntensity = 0.3 + Math.sin(time * 4) * 0.15;
    ctx.save();
    ctx.shadowColor = borderColor;
    ctx.shadowBlur = 8 + Math.sin(time * 4) * 4;
    ctx.beginPath();
    ctx.roundRect(cx - halfW, cy - halfH, w, h, 4);
    ctx.strokeStyle = hexToRgba(borderColor, errIntensity);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  // Tool name text
  var textX = (tool.state === 'running' || tool.state === 'complete') ? cx - halfW + 22 : cx - halfW + 8;
  ctx.fillStyle = C.textPrimary;
  ctx.font = '8px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  var toolName = truncateText(tool.name || 'unknown', 18);
  ctx.fillText(toolName, textX, cy);

  // Token cost on completed cards
  if (tool.state === 'complete' && tool.tokens) {
    var tokLabel = tool.tokens >= 1000 ? (tool.tokens / 1000).toFixed(1) + 'k' : String(tool.tokens);
    ctx.fillStyle = C.textMuted;
    ctx.font = '7px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(tokLabel + ' tok', cx + halfW - 4, cy);
  }

  // Connection line to parent node
  ctx.beginPath();
  ctx.moveTo(node.x, node.y);
  ctx.lineTo(cx, cy);
  ctx.strokeStyle = hexToRgba(borderColor, 0.15);
  ctx.lineWidth = 0.5;
  ctx.stroke();

  ctx.restore();
}

function drawAllToolCards(ctx, time) {
  for (var entry of nodes) {
    var node = entry[1];
    if (!node.toolCalls || node.toolCalls.length === 0) continue;
    if (node.state === 'done' || node.state === 'failed' || node.state === 'pending') continue;
    for (var i = 0; i < node.toolCalls.length; i++) {
      drawToolCard(ctx, node, node.toolCalls[i], i, time);
    }
  }
}
