// ═══ NORT MESSAGE BUBBLES ═══
// Floating message bubbles attached to agent nodes showing output,
// review notes, and status messages.

var bubbles = [];

var BUBBLE_ROLES = {
  drone:    { bg: 'rgba(80, 160, 220, 0.12)', text: '#a0d4f0', label: 'DRONE' },
  sentinel: { bg: 'rgba(255, 187, 68, 0.12)', text: '#e0c888', label: 'SENTINEL' },
  probe:    { bg: 'rgba(140, 100, 200, 0.12)', text: '#c0a0e0', label: 'PROBE' },
  nexus:    { bg: 'rgba(255, 215, 0, 0.12)',   text: '#ffd700', label: 'NEXUS' },
};

var BUBBLE_MAX_COUNT = 20;
var BUBBLE_MAX_LINES = 8;
var BUBBLE_PADDING = 6;
var BUBBLE_HEADER_H = 12;
var BUBBLE_LINE_H = 10;
var BUBBLE_BORDER_R = 5;
var BUBBLE_DEFAULT_MAX_W = 220;
var BUBBLE_TRI_W = 5;
var BUBBLE_TRI_H = 4;
var BUBBLE_STACK_GAP = 6;
var BUBBLE_FADE_IN_DUR = 0.3;
var BUBBLE_HOLD_DUR = 10;
var BUBBLE_FADE_OUT_DUR = 1.5;
var BUBBLE_TOTAL_DUR = BUBBLE_FADE_IN_DUR + BUBBLE_HOLD_DUR + BUBBLE_FADE_OUT_DUR;

// ── addBubble ───────────────────────────────────────────────────────────────

function addBubble(agentId, role, text) {
  var bubble = {
    agentId: agentId,
    role: role || 'drone',
    text: text || '',
    lines: [],
    createdAt: currentTime,
    maxWidth: BUBBLE_DEFAULT_MAX_W,
  };
  bubbles.push(bubble);
  // Cap total count, remove oldest first
  while (bubbles.length > BUBBLE_MAX_COUNT) {
    bubbles.shift();
  }
}

// ── wrapText ────────────────────────────────────────────────────────────────

function wrapBubbleText(ctx, text, maxWidth) {
  var lines = [];
  var paragraphs = text.split('\n');

  for (var p = 0; p < paragraphs.length; p++) {
    var para = paragraphs[p];
    if (para.trim() === '') {
      lines.push('');
      if (lines.length >= BUBBLE_MAX_LINES) break;
      continue;
    }

    var words = para.split(/\s+/);
    var currentLine = '';

    for (var w = 0; w < words.length; w++) {
      var word = words[w];
      var test = currentLine ? currentLine + ' ' + word : word;
      var testW = ctx.measureText(test).width;

      if (testW > maxWidth && currentLine) {
        lines.push(currentLine);
        if (lines.length >= BUBBLE_MAX_LINES) break;
        currentLine = word;
      } else {
        currentLine = test;
      }

      // Force-break single tokens wider than maxWidth
      while (ctx.measureText(currentLine).width > maxWidth && currentLine.length > 1) {
        var breakAt = currentLine.length - 1;
        while (breakAt > 1 && ctx.measureText(currentLine.slice(0, breakAt)).width > maxWidth) {
          breakAt--;
        }
        lines.push(currentLine.slice(0, breakAt));
        if (lines.length >= BUBBLE_MAX_LINES) break;
        currentLine = currentLine.slice(breakAt);
      }
      if (lines.length >= BUBBLE_MAX_LINES) break;
    }

    if (currentLine && lines.length < BUBBLE_MAX_LINES) {
      lines.push(currentLine);
    }
    if (lines.length >= BUBBLE_MAX_LINES) break;
  }

  // Truncation indicator
  var totalParaLines = 0;
  for (var i = 0; i < paragraphs.length; i++) {
    totalParaLines += Math.max(1, Math.ceil(paragraphs[i].length / 30));
  }
  if (totalParaLines > BUBBLE_MAX_LINES && lines.length >= BUBBLE_MAX_LINES) {
    lines[BUBBLE_MAX_LINES - 1] = lines[BUBBLE_MAX_LINES - 1].slice(0, -3) + '...';
  }

  return lines.length > 0 ? lines : [''];
}

// ── bubbleAlpha ─────────────────────────────────────────────────────────────

function getBubbleAlpha(age) {
  if (age < 0) return 0;
  if (age < BUBBLE_FADE_IN_DUR) {
    return age / BUBBLE_FADE_IN_DUR;
  }
  if (age < BUBBLE_FADE_IN_DUR + BUBBLE_HOLD_DUR) {
    return 1;
  }
  var fadeAge = age - BUBBLE_FADE_IN_DUR - BUBBLE_HOLD_DUR;
  if (fadeAge < BUBBLE_FADE_OUT_DUR) {
    return 1 - fadeAge / BUBBLE_FADE_OUT_DUR;
  }
  return 0;
}

// ── drawAllBubbles ──────────────────────────────────────────────────────────

function drawAllBubbles(ctx, time) {
  // Remove expired bubbles
  var i = bubbles.length;
  while (i--) {
    if (time - bubbles[i].createdAt > BUBBLE_TOTAL_DUR) {
      bubbles.splice(i, 1);
    }
  }

  if (bubbles.length === 0) return;

  // Group bubbles by agentId for vertical stacking
  var grouped = {};
  for (var b = 0; b < bubbles.length; b++) {
    var bubble = bubbles[b];
    if (!grouped[bubble.agentId]) grouped[bubble.agentId] = [];
    grouped[bubble.agentId].push(bubble);
  }

  for (var agentId in grouped) {
    var stack = grouped[agentId];
    var node = nodes.get(agentId);
    if (!node) continue;

    // Wrap lines for each bubble (cache on first render)
    var fontSize = 7;
    ctx.font = fontSize + 'px monospace';

    for (var s = 0; s < stack.length; s++) {
      var bub = stack[s];
      if (!bub.lines || bub.lines.length === 0) {
        bub.lines = wrapBubbleText(ctx, bub.text, bub.maxWidth - BUBBLE_PADDING * 2);
      }
    }

    // Calculate total stack height (bottom-up: newest on top)
    var stackHeight = 0;
    for (var s = 0; s < stack.length; s++) {
      var bub = stack[s];
      var bubH = BUBBLE_HEADER_H + bub.lines.length * BUBBLE_LINE_H + BUBBLE_PADDING;
      stackHeight += bubH + (s > 0 ? BUBBLE_STACK_GAP : 0);
    }

    // Position: centered above the agent node
    var baseY = node.y - node.radius - 20 - BUBBLE_TRI_H;
    var cursorY = baseY - stackHeight;

    for (var s = 0; s < stack.length; s++) {
      var bub = stack[s];
      var age = time - bub.createdAt;
      var alpha = getBubbleAlpha(age);
      if (alpha < 0.01) continue;

      var roleStyle = BUBBLE_ROLES[bub.role] || BUBBLE_ROLES.drone;
      var lineCount = bub.lines.length;
      var bubH = BUBBLE_HEADER_H + lineCount * BUBBLE_LINE_H + BUBBLE_PADDING;

      // Measure actual max line width
      ctx.font = fontSize + 'px monospace';
      var maxLineW = 0;
      for (var l = 0; l < lineCount; l++) {
        var lw = ctx.measureText(bub.lines[l]).width;
        if (lw > maxLineW) maxLineW = lw;
      }
      var bubW = Math.min(bub.maxWidth, maxLineW + BUBBLE_PADDING * 2 + 4);
      var bubX = node.x - bubW / 2;

      ctx.save();
      ctx.globalAlpha = alpha;

      // ── Background rounded rect ──
      ctx.beginPath();
      ctx.roundRect(bubX, cursorY, bubW, bubH, BUBBLE_BORDER_R);
      ctx.fillStyle = roleStyle.bg;
      ctx.fill();

      // ── Border ──
      ctx.strokeStyle = hexToRgba(roleStyle.text, 0.3);
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // ── Role label ──
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillStyle = roleStyle.text;
      ctx.globalAlpha = alpha * 0.7;
      ctx.fillText(roleStyle.label, bubX + BUBBLE_PADDING, cursorY + 2);

      // ── Text body ──
      ctx.globalAlpha = alpha;
      ctx.font = fontSize + 'px monospace';
      ctx.fillStyle = roleStyle.text;
      for (var l = 0; l < lineCount; l++) {
        ctx.fillText(
          bub.lines[l],
          bubX + BUBBLE_PADDING,
          cursorY + BUBBLE_HEADER_H + l * BUBBLE_LINE_H
        );
      }

      // ── Triangle pointer (on bottom bubble only) ──
      if (s === stack.length - 1) {
        ctx.beginPath();
        ctx.moveTo(node.x - BUBBLE_TRI_W, cursorY + bubH);
        ctx.lineTo(node.x, cursorY + bubH + BUBBLE_TRI_H);
        ctx.lineTo(node.x + BUBBLE_TRI_W, cursorY + bubH);
        ctx.closePath();
        ctx.fillStyle = roleStyle.bg;
        ctx.fill();
      }

      ctx.restore();

      cursorY += bubH + BUBBLE_STACK_GAP;
    }
  }
}
