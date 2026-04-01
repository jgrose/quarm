// ═══ TRON ROSTER PANEL — Agent XP & Leveling System ═══
// DOM-based UI panel tracking ambient programs with XP/levels
// Supports both ambient Tron names and live orchestrator agent names

var rosterData = [];
var _rosterInitialized = false;
var _rosterInterval = null;

var TRON_NAMES = ['CLU', 'TRON', 'RINZLER', 'QUORRA', 'ZEN', 'YORI'];

var XP_LEVELS = [0, 50, 150, 300, 500]; // L1-L5 thresholds

var TIER_ICONS = {
  sentinel: '\u25C7',  // ◇
  drone:    '\u25B8',  // ▸
  probe:    '\u25C8',  // ◈
};

var TASK_STATUS_LABELS = {
  in_progress:           'WORKING',
  in_manager_review:     'IN REVIEW',
  in_specialist_review:  'SPECIALIST REVIEW',
  revision:              'REVISING',
  done:                  'COMPLETE',
  failed:                'FAILED',
};

// ─── Persistence ───

function _loadRosterXP() {
  try {
    var raw = localStorage.getItem('nort_roster_xp');
    if (!raw) return {};
    var parsed = JSON.parse(raw);
    // Support legacy array format: convert to keyed object
    if (Array.isArray(parsed)) {
      var obj = {};
      for (var i = 0; i < parsed.length; i++) {
        var legacyKey = TRON_NAMES[i % TRON_NAMES.length];
        obj[legacyKey] = parsed[i];
      }
      return obj;
    }
    return parsed;
  } catch (e) {
    return {};
  }
}

function _saveRosterXP() {
  try {
    var data = {};
    for (var i = 0; i < rosterData.length; i++) {
      var key = rosterData[i].agentName || rosterData[i].name;
      data[key] = { xp: rosterData[i].xp, level: rosterData[i].level };
    }
    localStorage.setItem('nort_roster_xp', JSON.stringify(data));
  } catch (e) {}
}

// ─── Level Calculation ───

function _getLevelForXP(xp) {
  var level = 1;
  for (var i = 0; i < XP_LEVELS.length; i++) {
    if (xp >= XP_LEVELS[i]) level = i + 1;
  }
  return level;
}

function _getXPProgress(xp, level) {
  var currentThreshold = XP_LEVELS[level - 1] || 0;
  var nextThreshold = XP_LEVELS[level] || XP_LEVELS[XP_LEVELS.length - 1];
  if (level >= XP_LEVELS.length) return 1; // Max level
  var progress = (xp - currentThreshold) / (nextThreshold - currentThreshold);
  return Math.max(0, Math.min(1, progress));
}

// ─── Status from program state ───

function _getProgramStatus(program) {
  if (program.assignedTask) {
    var status = program.assignedTask.status || 'working';
    if (status === 'in_progress') return 'working';
    if (status === 'in_manager_review' || status === 'in_specialist_review') return 'review';
    if (status === 'revision') return 'revising';
    if (status === 'done') return 'idle';
    return 'working';
  }
  if (!program.idle) return 'traveling';
  if (program.atLocation) return 'inside';
  return 'idle';
}

function _getProgramStatusLabel(program) {
  if (program.assignedTask) {
    var status = program.assignedTask.status || 'in_progress';
    return TASK_STATUS_LABELS[status] || 'WORKING';
  }
  if (!program.idle) return 'TRAVELING';
  if (program.atLocation) return 'INSIDE';
  return 'IDLE';
}

function _getLocationName(program) {
  if (program.assignedTask && program.assignedTask.title) {
    return program.assignedTask.title;
  }
  if (program.atLocation && program.atLocation.name) {
    return program.atLocation.name;
  }
  return '';
}

function _truncate(str, maxLen) {
  if (!str) return '';
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen - 1) + '\u2026';
}

// ─── Init ───

function initRoster() {
  if (!ambientPrograms || !ambientPrograms.length) return;

  // Check if this is an orchestrator-bound roster
  var isLive = ambientPrograms[0] && ambientPrograms[0].agentName;

  var saved = _loadRosterXP();

  rosterData = [];
  for (var i = 0; i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];
    var name = isLive ? (p.displayName || p.agentName || TRON_NAMES[i % TRON_NAMES.length])
                      : TRON_NAMES[i % TRON_NAMES.length];

    // Restore saved XP if available
    var key = p.agentName || name;
    var savedEntry = saved[key] || {};
    var xp = savedEntry.xp || 0;
    var level = _getLevelForXP(xp);

    var taskTitle = '';
    if (p.assignedTask && p.assignedTask.title) {
      taskTitle = _truncate(p.assignedTask.title, 25);
    }

    rosterData.push({
      name: name,
      agentName: p.agentName || null,
      glow: p.glow,
      tier: p.tier || 'drone',
      xp: xp,
      level: level,
      status: _getProgramStatus(p),
      statusLabel: _getProgramStatusLabel(p),
      location: _getLocationName(p),
      taskTitle: taskTitle,
      taskStatus: (p.assignedTask && p.assignedTask.status) || null,
    });
  }

  _rosterInitialized = true;

  // Only create the interval once
  if (!_rosterInterval) {
    _rosterInterval = setInterval(updateRoster, 2000);
  }

  renderRosterPanel();
}

// ─── XP System ───

function addXP(programIndex, amount) {
  if (!_rosterInitialized) return;
  if (programIndex < 0 || programIndex >= rosterData.length) return;

  var entry = rosterData[programIndex];
  var oldLevel = entry.level;
  entry.xp += amount;
  entry.level = _getLevelForXP(entry.xp);

  // Level-up effect
  if (entry.level > oldLevel && typeof effects !== 'undefined') {
    var program = ambientPrograms[programIndex];
    if (program) {
      effects.push({
        type: 'complete',
        x: program.x,
        y: program.y,
        color: program.glow,
        age: 0,
        duration: 1.0,
      });
    }
  }

  _saveRosterXP();

  // Update the panel if visible
  var panel = document.getElementById('rosterPanel');
  if (panel && !panel.classList.contains('hidden')) {
    renderRosterPanel();
  }
}

// ─── Update (sync from ambientPrograms) ───

function updateRoster() {
  if (!_rosterInitialized) return;
  if (!ambientPrograms || !ambientPrograms.length) return;

  // Detect roster size change — reinitialize if programs changed
  if (rosterData.length !== ambientPrograms.length) {
    _rosterInitialized = false;
    initRoster();
    return;
  }

  for (var i = 0; i < rosterData.length && i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];
    rosterData[i].status = _getProgramStatus(p);
    rosterData[i].statusLabel = _getProgramStatusLabel(p);
    rosterData[i].location = _getLocationName(p);
    rosterData[i].glow = p.glow;

    // Sync task info
    if (p.assignedTask) {
      rosterData[i].taskTitle = _truncate(p.assignedTask.title || '', 25);
      rosterData[i].taskStatus = p.assignedTask.status || null;
    } else {
      rosterData[i].taskTitle = '';
      rosterData[i].taskStatus = null;
    }

    // Sync agent name changes (e.g., late binding)
    if (p.agentName && !rosterData[i].agentName) {
      rosterData[i].agentName = p.agentName;
      rosterData[i].name = p.displayName || p.agentName;
    }

    // Sync tier if it changed
    if (p.tier && rosterData[i].tier !== p.tier) {
      rosterData[i].tier = p.tier;
    }
  }

  // Re-render if panel is visible
  var panel = document.getElementById('rosterPanel');
  if (panel && !panel.classList.contains('hidden')) {
    renderRosterPanel();
  }
}

// ─── Toggle Panel ───

function toggleRoster() {
  var panel = document.getElementById('rosterPanel');
  if (!panel) return;

  // Lazy-init on first open
  if (!_rosterInitialized) initRoster();

  panel.classList.toggle('hidden');

  if (!panel.classList.contains('hidden')) {
    renderRosterPanel();
  }
}

// ─── Render Panel HTML ───

function renderRosterPanel() {
  var list = document.getElementById('rosterList');
  if (!list) return;
  if (!rosterData.length) {
    list.innerHTML = '<div style="font-size:9px;color:rgba(102,204,255,0.4);text-align:center;padding:12px">No programs active</div>';
    return;
  }

  var html = '';
  for (var i = 0; i < rosterData.length; i++) {
    var r = rosterData[i];
    var progress = _getXPProgress(r.xp, r.level);

    // Tier icon
    var tierIcon = TIER_ICONS[r.tier] || TIER_ICONS.drone;

    // Build status line
    var statusLine = r.statusLabel || r.status.toUpperCase();
    if (r.taskTitle) {
      statusLine += ' \u00b7 ' + r.taskTitle;
    } else if (r.location && !r.taskTitle) {
      statusLine += ' \u00b7 ' + r.location;
    }

    // Determine status color: use task state color when actively assigned
    var statusColor = r.glow;
    if (r.taskStatus && typeof getStateColor === 'function') {
      statusColor = getStateColor(r.taskStatus);
    }

    html += '<div class="roster-item" onclick="panToProgram(' + i + ')" title="Click to track">';
    html += '<div class="roster-swatch" style="background:' + r.glow + ';box-shadow:0 0 6px ' + r.glow + '"></div>';
    html += '<div class="roster-info">';
    html += '<div class="roster-name"><span style="opacity:0.5;margin-right:3px">' + tierIcon + '</span>' + escapeHtml(r.name) + '</div>';
    html += '<div class="roster-status" style="color:' + statusColor + '">' + escapeHtml(statusLine) + '</div>';
    html += '<div class="xp-bar-outer"><div class="xp-bar-inner" style="width:' + (progress * 100).toFixed(1) + '%;background:' + r.glow + '"></div></div>';
    html += '</div>';
    html += '<div class="roster-level">L' + r.level + '</div>';
    html += '</div>';
  }

  list.innerHTML = html;
}

// ─── Pan Camera to Program ───

function panToProgram(index) {
  if (index < 0 || !ambientPrograms || index >= ambientPrograms.length) return;
  if (typeof camera === 'undefined') return;

  var p = ambientPrograms[index];
  var canvas = document.getElementById('canvas');
  if (!canvas) return;

  var W = canvas.width / (window.devicePixelRatio || 1);
  var H = canvas.height / (window.devicePixelRatio || 1);

  // Center camera on program position
  camera.x = -(p.x - W / 2);
  camera.y = -(p.y - H / 2);
}

// ─── Level Badge Drawing (called from render pipeline) ───
// Call this after drawAmbientPrograms in the render loop
// to overlay level badges above each program sprite.

function drawRosterBadges(ctx) {
  if (!_rosterInitialized) return;
  if (!ambientPrograms || !ambientPrograms.length) return;
  if (typeof nodes !== 'undefined' && nodes.size > 0) return;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';

  for (var i = 0; i < ambientPrograms.length && i < rosterData.length; i++) {
    var p = ambientPrograms[i];
    var r = rosterData[i];
    var label = 'L' + r.level;

    var badgeY = p.y - (24 * PX * p.scale) - 4;

    // Badge background
    ctx.fillStyle = 'rgba(5, 5, 16, 0.7)';
    var textW = ctx.measureText(label).width || 14;
    ctx.fillRect(p.x - textW / 2 - 3, badgeY - 10, textW + 6, 12);

    // Badge border
    ctx.strokeStyle = hexToRgba(p.glow, 0.4);
    ctx.lineWidth = 1;
    ctx.strokeRect(p.x - textW / 2 - 3, badgeY - 10, textW + 6, 12);

    // Badge text
    ctx.font = '8px "Courier New", monospace';
    ctx.fillStyle = p.glow;
    ctx.fillText(label, p.x, badgeY);
  }

  ctx.restore();
}
