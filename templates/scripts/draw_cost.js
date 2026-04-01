// ═══ NORT COST VISUALIZATION ═══
// Floating cost pills above agents and summary panel top-right

var COST_RATE_PER_M = 6;
var COST_MIN_DISPLAY = 0.0001;

function _agentCost(tokens) {
  return (tokens / 1000000) * COST_RATE_PER_M;
}

function _toolTypeColor(name) {
  if (!name) return C.contextSubagent;
  var n = name.toLowerCase();
  if (n.indexOf('read') >= 0 || n.indexOf('glob') >= 0 || n.indexOf('grep') >= 0) return C.contextUser;
  if (n.indexOf('edit') >= 0 || n.indexOf('write') >= 0) return C.contextReasoning;
  if (n.indexOf('bash') >= 0) return C.contextTool;
  return C.contextSubagent;
}

function drawAllCostPills(ctx, time) {
  for (var entry of nodes) {
    var node = entry[1];
    if (node.opacity < 0.05 || !node.tokens) continue;
    var cost = _agentCost(node.tokens);
    if (cost < COST_MIN_DISPLAY) continue;

    var r = node.radius;
    var pillY = node.y - r - 22;
    var label = '$' + (cost < 0.01 ? cost.toFixed(4) : cost.toFixed(3));

    ctx.save();
    ctx.globalAlpha = node.opacity * 0.9;
    ctx.font = 'bold 8px monospace';
    var labelW = measureTextCached(ctx, label);
    var pillW = labelW + 12;
    var pillH = 16;
    var pillX = node.x - pillW / 2;

    // Pill background
    ctx.fillStyle = 'rgba(10,20,40,0.75)';
    ctx.strokeStyle = 'rgba(102,255,170,0.3)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, pillW, pillH, 8);
    ctx.fill();
    ctx.stroke();

    // Cost text
    ctx.fillStyle = '#66ffaa';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, node.x, pillY + pillH / 2);

    // Mini bar below pill showing tool type breakdown
    var tools = node.toolCalls;
    if (tools && tools.length > 0) {
      var byType = {};
      var totalToolTokens = 0;
      for (var i = 0; i < tools.length; i++) {
        var tc = tools[i];
        var tok = tc.tokens || 0;
        if (tok <= 0) continue;
        var key = tc.name || 'unknown';
        byType[key] = (byType[key] || 0) + tok;
        totalToolTokens += tok;
      }
      if (totalToolTokens > 0) {
        var barW = Math.min(pillW + 10, 80);
        var barH = 3;
        var barX = node.x - barW / 2;
        var barY = pillY + pillH + 3;

        ctx.fillStyle = 'rgba(100,200,255,0.06)';
        ctx.beginPath();
        ctx.roundRect(barX, barY, barW, barH, 1.5);
        ctx.fill();

        var segX = barX;
        for (var tName in byType) {
          var segW = (byType[tName] / totalToolTokens) * barW;
          if (segW < 1) continue;
          ctx.fillStyle = _toolTypeColor(tName);
          ctx.globalAlpha = node.opacity * 0.7;
          ctx.beginPath();
          ctx.roundRect(segX, barY, segW, barH, 1.5);
          ctx.fill();
          segX += segW;
        }
      }
    }

    ctx.restore();
  }
}

function drawCostPanel(ctx, W, H) {
  var agentList = [];
  var totalTokens = 0;
  for (var entry of nodes) {
    var node = entry[1];
    if (node.tokens > 0) {
      agentList.push(node);
      totalTokens += node.tokens;
    }
  }
  if (agentList.length === 0) return;

  var totalCost = _agentCost(totalTokens);

  // Per-agent breakdown sorted by cost desc
  var breakdown = agentList
    .map(function(a) { return { name: a.name, cost: _agentCost(a.tokens) }; })
    .sort(function(a, b) { return b.cost - a.cost; });

  var maxRows = Math.min(breakdown.length, 5);

  // Panel dimensions
  var panelW = 200;
  var panelX = W - panelW - 16;
  var panelY = 48;
  var lineH = 16;
  var headerH = 28;
  var panelH = headerH + maxRows * lineH + 12;

  ctx.save();

  // Panel background
  ctx.fillStyle = C.glassBg;
  ctx.strokeStyle = C.glassBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 8);
  ctx.fill();
  ctx.stroke();

  var y = panelY + 8;

  // Header: total cost
  ctx.font = 'bold 9px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#66ffaa';
  var totalLabel = 'TOTAL: $' + totalCost.toFixed(3);
  ctx.fillText(totalLabel, panelX + 10, y);

  // Token count dim
  ctx.font = '8px monospace';
  ctx.fillStyle = C.textMuted;
  var tokLabel = (totalTokens >= 1000 ? (totalTokens / 1000).toFixed(1) + 'K' : String(totalTokens)) + ' tokens';
  var totalLabelW = measureTextCached(ctx, totalLabel);
  ctx.fillText(tokLabel, panelX + 10 + totalLabelW + 10, y + 1);

  y += headerH;

  // Agent breakdown rows
  var barW = panelW - 20;
  for (var i = 0; i < maxRows; i++) {
    var a = breakdown[i];
    var ratio = totalCost > 0 ? a.cost / totalCost : 0;

    // Background bar
    ctx.fillStyle = 'rgba(102,204,255,0.15)';
    ctx.beginPath();
    ctx.roundRect(panelX + 10, y + 1, barW * ratio, lineH - 3, 3);
    ctx.fill();

    // Agent name (truncated)
    ctx.font = '8px monospace';
    ctx.fillStyle = C.textPrimary;
    ctx.textAlign = 'left';
    var displayName = a.name.length > 15 ? a.name.slice(0, 14) + '\u2026' : a.name;
    ctx.fillText(displayName, panelX + 14, y + 3);

    // Cost right-aligned
    ctx.textAlign = 'right';
    ctx.fillStyle = '#66ffaa';
    ctx.fillText('$' + a.cost.toFixed(3), panelX + 10 + barW - 4, y + 3);

    y += lineH;
  }

  ctx.restore();
}
