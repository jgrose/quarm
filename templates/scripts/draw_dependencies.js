// ═══ DEPENDENCY VISUALIZATION ═══
// Animated lines between dependent tasks + BLOCKED indicators

var _depLines = [];      // [{fromTaskId, toTaskId, resolved, resolvedAt}]
var _blockedTasks = {};  // {taskId: {deps: [...], metCount: N, totalCount: N, agent: string}}

// Called from websocket.js on every status update
function rebuildDependencyState(tasks) {
  _depLines = [];
  _blockedTasks = {};

  var now = typeof currentTime !== 'undefined' ? currentTime : 0;

  for (var i = 0; i < tasks.length; i++) {
    var task = tasks[i];
    var deps = task.depends_on || [];
    if (deps.length === 0) continue;

    var metCount = 0;
    for (var j = 0; j < deps.length; j++) {
      var depTask = _findTaskById(tasks, deps[j]);
      var resolved = depTask && depTask.status === 'done';
      if (resolved) metCount++;

      _depLines.push({
        fromTaskId: deps[j],      // the prerequisite
        toTaskId: task.id,         // the dependent task
        resolved: resolved,
        resolvedAt: resolved ? now : 0
      });
    }

    // Task is blocked if pending and has unmet deps
    if (task.status === 'pending' && metCount < deps.length) {
      _blockedTasks[task.id] = {
        deps: deps,
        metCount: metCount,
        totalCount: deps.length,
        agent: task.agent
      };
    }
  }
}

// Draw animated dependency lines between programs
function drawDependencyLines(ctx, time) {
  if (!config.dependencies) return;

  for (var i = 0; i < _depLines.length; i++) {
    var line = _depLines[i];

    var fromProg = _findProgramForTask(line.fromTaskId);
    var toProg = _findProgramForTask(line.toTaskId);

    // Need at least the source program to draw
    if (!fromProg) continue;

    var fromX = fromProg.x;
    var fromY = fromProg.y;
    var toX, toY;

    if (toProg) {
      toX = toProg.x;
      toY = toProg.y;
    } else {
      // Dependent task has no program yet (still pending/blocked)
      // Skip drawing line — the blocked indicator is enough
      continue;
    }

    // Resolved lines fade out
    if (line.resolved) {
      var fadeAge = time - line.resolvedAt;
      if (fadeAge > 2.0) continue; // fully faded
      var alpha = 0.3 * (1 - fadeAge / 2.0);
      _drawDepLine(ctx, fromX, fromY, toX, toY, '#66ffaa', alpha, time);
    } else {
      // Active dependency — amber, pulsing
      var alpha = 0.35 + Math.sin(time * 2) * 0.1;
      _drawDepLine(ctx, fromX, fromY, toX, toY, '#ffbb44', alpha, time);
    }
  }
}

// Draw a single dependency line (animated dashed bezier with arrowhead)
function _drawDepLine(ctx, fromX, fromY, toX, toY, color, alpha, time) {
  var dx = toX - fromX, dy = toY - fromY;
  var dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 20) return; // too close, skip

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([8, 6]);
  ctx.lineDashOffset = -time * 20;

  // Simple bezier curve (slight arc)
  var perpX = (-dy / dist) * dist * 0.15;
  var perpY = (dx / dist) * dist * 0.15;
  var cpX = (fromX + toX) / 2 + perpX;
  var cpY = (fromY + toY) / 2 + perpY;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.quadraticCurveTo(cpX, cpY, toX, toY);
  ctx.stroke();

  // Arrowhead
  var angle = Math.atan2(toY - cpY, toX - cpX);
  var headLen = 8;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - 0.4), toY - headLen * Math.sin(angle - 0.4));
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle + 0.4), toY - headLen * Math.sin(angle + 0.4));
  ctx.stroke();

  ctx.restore();
}

// Draw blocked indicators above programs with unmet dependencies
function drawBlockedIndicators(ctx, time) {
  if (!config.dependencies) return;

  var taskIds = Object.keys(_blockedTasks);
  for (var i = 0; i < taskIds.length; i++) {
    var info = _blockedTasks[taskIds[i]];
    var prog = _findProgramForTask(taskIds[i]);

    // If the task has an assigned program, draw above it
    // Otherwise try to find an idle program with this agent name
    var drawX, drawY;
    if (prog) {
      drawX = prog.x;
      drawY = prog.y;
    } else if (info.agent) {
      // Find program by agent name
      for (var j = 0; j < ambientPrograms.length; j++) {
        if (ambientPrograms[j].agentName === info.agent) {
          drawX = ambientPrograms[j].x;
          drawY = ambientPrograms[j].y;
          break;
        }
      }
      if (drawX === undefined) continue;
    } else {
      continue;
    }

    // Bob animation
    var bob = Math.sin(time * 2) * 2;
    var badgeY = drawY - 30 + bob;

    // Background pill
    var text = 'BLOCKED ' + info.metCount + '/' + info.totalCount;
    ctx.save();
    ctx.font = 'bold 7px monospace';
    var tw = ctx.measureText(text).width;
    var px = 4, py = 3;

    ctx.fillStyle = 'rgba(10, 15, 30, 0.85)';
    ctx.strokeStyle = 'rgba(255, 136, 0, 0.5)';
    ctx.lineWidth = 1;

    // roundRect with fallback for older browsers
    var rx = drawX - tw / 2 - px;
    var ry = badgeY - 7 - py;
    var rw = tw + px * 2;
    var rh = 14 + py * 2;
    var radius = 3;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(rx, ry, rw, rh, radius);
    } else {
      ctx.moveTo(rx + radius, ry);
      ctx.lineTo(rx + rw - radius, ry);
      ctx.arcTo(rx + rw, ry, rx + rw, ry + radius, radius);
      ctx.lineTo(rx + rw, ry + rh - radius);
      ctx.arcTo(rx + rw, ry + rh, rx + rw - radius, ry + rh, radius);
      ctx.lineTo(rx + radius, ry + rh);
      ctx.arcTo(rx, ry + rh, rx, ry + rh - radius, radius);
      ctx.lineTo(rx, ry + radius);
      ctx.arcTo(rx, ry, rx + radius, ry, radius);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();

    // Text
    ctx.fillStyle = '#ff8866';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, drawX, badgeY);

    ctx.restore();
  }
}

// ─── Helpers ───

function _findProgramForTask(taskId) {
  for (var i = 0; i < ambientPrograms.length; i++) {
    if (ambientPrograms[i].assignedTask && ambientPrograms[i].assignedTask.id === taskId) {
      return ambientPrograms[i];
    }
  }
  return null;
}

function _findTaskById(tasks, taskId) {
  for (var i = 0; i < tasks.length; i++) {
    if (tasks[i].id === taskId) return tasks[i];
  }
  return null;
}
