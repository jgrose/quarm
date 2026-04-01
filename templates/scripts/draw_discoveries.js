// ═══ NORT DISCOVERY CARDS ═══
// Floating discovery cards showing file/pattern findings

var discoveries = [];

var DISCOVERY_COLORS = {
  file: '#66ccff',
  pattern: '#cc88ff',
  finding: '#66ffaa',
  code: '#ffbb44',
};

var DISCOVERY_HOLD_S = 8;
var DISCOVERY_CHAR_W = 5.5;
var DISCOVERY_LABEL_CHAR_W = 6;

function addDiscovery(agentId, type, label, contentLines) {
  var agent = nodes.get(agentId);
  var startX = agent ? agent.x + (Math.random() - 0.5) * 60 : 200;
  var startY = agent ? agent.y + 60 + Math.random() * 40 : 200;
  var targetAngle = Math.random() * Math.PI * 2;
  var targetDist = 80 + Math.random() * 60;

  discoveries.push({
    agentId: agentId,
    type: type || 'file',
    label: label || '',
    content: contentLines || [],
    x: startX,
    y: startY,
    targetX: (agent ? agent.x : startX) + Math.cos(targetAngle) * targetDist,
    targetY: (agent ? agent.y : startY) + Math.sin(targetAngle) * targetDist,
    opacity: 0,
    createdAt: currentTime,
  });
}

function _discoveryCardDims(label, lines) {
  var maxLineW = 0;
  for (var i = 0; i < lines.length; i++) {
    var w = lines[i].length * DISCOVERY_CHAR_W;
    if (w > maxLineW) maxLineW = w;
  }
  var labelW = label.length * DISCOVERY_LABEL_CHAR_W;
  if (labelW > maxLineW) maxLineW = labelW;
  var cardW = Math.min(Math.max(80, maxLineW + 16), 150);
  var cardH = 16 + lines.length * 11;
  return { w: cardW, h: cardH };
}

function drawAllDiscoveries(ctx, time, dt) {
  var alive = [];

  for (var i = 0; i < discoveries.length; i++) {
    var disc = discoveries[i];
    var age = time - disc.createdAt;

    // Position lerp toward target
    disc.x += (disc.targetX - disc.x) * 3 * dt;
    disc.y += (disc.targetY - disc.y) * 3 * dt;

    // Opacity: fade in, hold, fade out
    if (age < DISCOVERY_HOLD_S) {
      disc.opacity = Math.min(1, disc.opacity + 2 * dt);
    } else {
      disc.opacity -= 0.5 * dt;
    }

    if (disc.opacity < 0.05) continue;
    alive.push(disc);

    var typeColor = DISCOVERY_COLORS[disc.type] || DISCOVERY_COLORS.file;
    var dims = _discoveryCardDims(disc.label, disc.content);
    var cardX = disc.x - dims.w / 2;
    var cardY = disc.y - dims.h / 2;

    ctx.save();
    ctx.globalAlpha = disc.opacity;

    // Connection line to agent
    var agent = nodes.get(disc.agentId);
    if (agent) {
      ctx.strokeStyle = hexToRgba(C.holoBase, 0.3);
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(agent.x, agent.y);
      ctx.lineTo(disc.x, disc.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Card background
    ctx.fillStyle = 'rgba(10,15,30,0.6)';
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, dims.w, dims.h, 4);
    ctx.fill();

    // Left stripe
    ctx.fillStyle = hexToRgba(typeColor, 0.6);
    ctx.fillRect(cardX, cardY, 2, dims.h);

    // Border
    ctx.strokeStyle = hexToRgba(typeColor, 0.3);
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, dims.w, dims.h, 4);
    ctx.stroke();

    // Label
    ctx.fillStyle = typeColor;
    ctx.font = 'bold 8px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(disc.label, cardX + 6, cardY + 3);

    // Content lines
    ctx.fillStyle = C.textMuted;
    ctx.font = '7px monospace';
    for (var j = 0; j < disc.content.length; j++) {
      var line = disc.content[j];
      if (line.length * DISCOVERY_CHAR_W > dims.w - 10) {
        var maxChars = Math.floor((dims.w - 16) / DISCOVERY_CHAR_W);
        line = line.slice(0, maxChars - 1) + '\u2026';
      }
      ctx.fillText(line, cardX + 6, cardY + 14 + j * 11);
    }

    ctx.restore();
  }

  discoveries = alive;
}
