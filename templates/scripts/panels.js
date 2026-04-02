// ═══ NORT PANEL LOGIC ═══
// UI panels: config, agent detail, event log, queue, thinking, overlays

// ── Config System ───────────────────────────────────────────────────────────

// config is declared in nodes.js with all default keys — apply localStorage overrides
try {
  var _saved = JSON.parse(localStorage.getItem('nort_config') || '{}');
  for (var _sk in _saved) {
    if (_saved.hasOwnProperty(_sk)) config[_sk] = _saved[_sk];
  }
} catch(_e) {}

function saveConfig() { localStorage.setItem('nort_config', JSON.stringify(config)); }

function toggleConfig(key) {
  config[key] = !config[key];
  if (key === 'sound' && typeof setAudioMuted === 'function') {
    setAudioMuted(!config.sound);
    if (config.sound && typeof resumeAudio === 'function') resumeAudio();
  }
  saveConfig();
  syncConfigUI();
}

function setConfigValue(key, value) {
  config[key] = value;
  saveConfig();
  syncConfigUI();
}

function _switchTab(tabClass, contentClass, prefix, tab, btn) {
  document.querySelectorAll('.' + tabClass).forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('.' + contentClass).forEach(function(s) { s.classList.remove('active'); });
  if (btn) btn.classList.add('active');
  var section = document.getElementById(prefix + tab);
  if (section) section.classList.add('active');
}

function switchConfigTab(tab, btn) { _switchTab('cfg-tab', 'cfg-tab-content', 'cfgTab-', tab, btn); }

var _qualityPresets = {
  low:    { shadowQuality: 'off', bloomQuality: 'off', particles: false, maxParticles: 10, maxEffects: 5, maxTrailLength: 10, edgeDetail: 4, weather: false, hexGrid: false, completionFx: false, lodEnabled: true, viewportCulling: true },
  medium: { shadowQuality: 'low', bloomQuality: 'off', particles: true, maxParticles: 25, maxEffects: 15, maxTrailLength: 20, edgeDetail: 8, weather: false, hexGrid: true, completionFx: true, lodEnabled: true, viewportCulling: true },
  high:   { shadowQuality: 'high', bloomQuality: 'low', particles: true, maxParticles: 50, maxEffects: 30, maxTrailLength: 30, edgeDetail: 16, weather: true, hexGrid: true, completionFx: true, lodEnabled: true, viewportCulling: true },
  ultra:  { shadowQuality: 'high', bloomQuality: 'high', particles: true, maxParticles: 100, maxEffects: 50, maxTrailLength: 50, edgeDetail: 16, weather: true, hexGrid: true, completionFx: true, lodEnabled: false, viewportCulling: true },
};

function applyQualityPreset(preset) {
  var p = _qualityPresets[preset];
  if (!p) return;
  config.qualityPreset = preset;
  for (var k in p) {
    if (p.hasOwnProperty(k)) config[k] = p[k];
  }
  // Sync bloom toggle with bloomQuality
  config.bloom = config.bloomQuality !== 'off';
  saveConfig();
  syncConfigUI();
}

// ── Agent Detail Card ───────────────────────────────────────────────────────

var selectedNode = null;

function showAgentDetail(node) {
  hideBuildingDetail(); // Mutual exclusion with building card
  selectedNode = node;
  var card = document.getElementById('agentDetailCard');
  card.classList.remove('hidden');
  var tierInfo = TIERS[node.tier] || TIERS.drone;
  var stateColor = getStateColor(node.state);
  var stateLabel = node.state.replace(/_/g, ' ').toUpperCase();

  var html = '<button class="close-btn" onclick="hideAgentDetail()">&times;</button>';
  html += '<div class="agent-tier">' + tierInfo.label + '</div>';
  html += '<div class="agent-name">' + escapeHtml(node.label) + '</div>';
  html += '<div class="agent-state">';
  html += '<span class="state-dot" style="background:' + stateColor + '"></span>';
  html += '<span>' + stateLabel + '</span>';
  html += '</div>';

  if (node.taskId) {
    html += '<div style="margin-top:8px;font-size:9px;color:' + C.textDim + '">' +
      escapeHtml(node.taskId) + ': ' + escapeHtml(node.taskTitle || '') + '</div>';
  }

  html += '<div class="agent-stats" style="margin-top:8px">';
  html += '<div><span class="stat-label">MODEL</span><span class="stat-value">' +
    escapeHtml(node.model || '\u2014') + '</span></div>';
  html += '<div><span class="stat-label">TOKENS</span><span class="stat-value">' +
    (node.tokens ? (node.tokens / 1000).toFixed(1) + 'K' : '\u2014') + '</span></div>';
  html += '<div><span class="stat-label">SCORE</span><span class="stat-value">' +
    (node.lastScore || '\u2014') + '</span></div>';
  html += '<div><span class="stat-label">REVISIONS</span><span class="stat-value">' +
    (node.revisionCount || 0) + '</span></div>';
  html += '</div>';

  if (node.toolCalls && node.toolCalls.length) {
    html += '<div style="margin-top:8px;font-size:8px;color:' + C.textDim + ';letter-spacing:1px">TOOL CALLS</div>';
    html += '<ul class="tool-list">';
    for (var i = 0; i < node.toolCalls.length; i++) {
      var tc = node.toolCalls[i];
      var tcName = tc.name || tc;
      var tcDone = tc.status === 'complete' || tc.state === 'complete';
      html += '<li class="tool-item">' + escapeHtml(String(tcName)) + ' ' +
        (tcDone ? '\u2713' : '\u2699') + '</li>';
    }
    html += '</ul>';
  }

  if (node.resultPreview) {
    html += '<div style="margin-top:8px;font-size:8px;color:' + C.textDim + ';letter-spacing:1px">OUTPUT</div>';
    html += '<div class="result-preview">' + escapeHtml(node.resultPreview) + '</div>';
  }

  card.innerHTML = html;
}

function hideAgentDetail() {
  selectedNode = null;
  document.getElementById('agentDetailCard').classList.add('hidden');
}

// ── Building Detail Card ───────────────────────────────────────────────────

var _selectedBuilding = null;
var _lastBuildingHash = '';

function showBuildingDetail(loc, e) {
  // Mutual exclusion with agent detail card
  hideAgentDetail();

  var card = document.getElementById('buildingDetailCard');
  if (!card) return;
  var wrap = document.getElementById('canvasWrap');
  if (!wrap) return;
  var rect = wrap.getBoundingClientRect();

  var cardX = e.clientX - rect.left + 16;
  var cardY = e.clientY - rect.top - 20;

  var cardWidth = 280;
  var cardHeight = 320;
  if (cardX + cardWidth > rect.width) cardX = e.clientX - rect.left - cardWidth - 16;
  if (cardY + cardHeight > rect.height) cardY = rect.height - cardHeight - 8;
  if (cardY < 8) cardY = 8;
  if (cardX < 8) cardX = 8;

  card.style.left = cardX + 'px';
  card.style.top = cardY + 'px';

  _selectedBuilding = loc;
  _lastBuildingHash = '';
  _renderBuildingCardContent(card, loc);
  card.classList.remove('hidden');
}

function hideBuildingDetail() {
  _selectedBuilding = null;
  _lastBuildingHash = '';
  var card = document.getElementById('buildingDetailCard');
  if (card) card.classList.add('hidden');
}

function _renderBuildingCardContent(card, loc) {
  var html = '<button class="close-btn" onclick="hideBuildingDetail()">&times;</button>';

  // Category + upgrade badge
  var catLabel = loc.category === 'work' ? 'WORK STATION' : 'IDLE ZONE';
  var upgLabel = loc.upgradeLevel > 0 ? 'LVL ' + loc.upgradeLevel : '';
  html += '<div class="bld-header">';
  html += '<span class="bld-category">' + catLabel + '</span>';
  if (upgLabel) html += '<span class="bld-upgrade" style="color:' + loc.glowColor + '">' + upgLabel + '</span>';
  html += '</div>';

  // Building name
  html += '<div class="bld-name" style="color:' + loc.glowColor + '">' + escapeHtml(loc.name) + '</div>';

  // Occupancy bar
  var occCount = loc.occupants ? loc.occupants.length : 0;
  html += '<div class="bld-occupancy">';
  html += '<span class="bld-category">OCCUPANCY</span>';
  html += '<span class="bld-occ-count">' + occCount + '/' + loc.capacity + '</span>';
  html += '</div>';
  html += '<div class="bld-occ-bar">';
  for (var i = 0; i < loc.capacity; i++) {
    var filled = i < occCount;
    var slotColor = filled ? loc.glowColor : 'rgba(100,200,255,0.1)';
    html += '<div class="bld-occ-slot" style="background:' + slotColor +
            (filled ? ';box-shadow:0 0 6px ' + loc.glowColor : '') + '"></div>';
  }
  html += '</div>';

  // Status section (work buildings only)
  if (loc.category === 'work') {
    var stateLabel = loc.taskState ? loc.taskState.replace(/_/g, ' ').toUpperCase() : 'IDLE';
    var stateColor = loc.taskState ? getStateColor(loc.taskState) : C.textDim;
    html += '<div class="bld-section-label">STATUS</div>';
    html += '<div class="bld-status">';
    html += '<span class="state-dot" style="background:' + stateColor + ';width:8px;height:8px;border-radius:50%;display:inline-block"></span>';
    html += '<span>' + stateLabel + '</span>';
    html += '</div>';
    html += '<div class="bld-stat-row"><span class="stat-label">COMPLETIONS</span><span class="stat-value">' + (loc.taskCompletions || 0) + '</span></div>';
  }

  // Occupants list
  html += '<div class="bld-section-label">PROGRAMS' + (occCount > 0 ? ' (' + occCount + ')' : '') + '</div>';
  if (occCount === 0) {
    html += '<div class="bld-empty">No programs present</div>';
  } else {
    for (var j = 0; j < loc.occupants.length; j++) {
      var p = loc.occupants[j];
      var tierInfo = TIERS[p.tier] || TIERS.drone;
      var name = p.displayName || p.agentName || ('Program ' + j);
      var taskTitle = p.assignedTask ? p.assignedTask.title : null;
      var taskStatus = p.assignedTask ? p.assignedTask.status : 'idle';
      var taskColor = getStateColor(taskStatus);

      html += '<div class="bld-occupant">';
      html += '<div class="bld-occ-header">';
      html += '<span class="bld-occ-icon" style="color:' + p.glow + '">' + tierInfo.icon + '</span>';
      html += '<span class="bld-occ-name">' + escapeHtml(name) + '</span>';
      html += '<span class="bld-occ-tier">' + (tierInfo.label || p.tier).toUpperCase() + '</span>';
      html += '</div>';
      if (taskTitle) {
        html += '<div class="bld-occ-task">';
        html += '<span style="background:' + taskColor + ';width:6px;height:6px;border-radius:50%;display:inline-block;flex-shrink:0"></span>';
        html += '<span>' + escapeHtml(taskTitle.length > 30 ? taskTitle.slice(0, 30) + '\u2026' : taskTitle) + '</span>';
        html += '</div>';
      } else {
        html += '<div class="bld-occ-task idle">IDLE</div>';
      }
      html += '</div>';
    }
  }

  card.innerHTML = html;
}

function _updateBuildingDetailIfNeeded() {
  if (!_selectedBuilding) return;
  var card = document.getElementById('buildingDetailCard');
  if (!card || card.classList.contains('hidden')) { _selectedBuilding = null; return; }

  var loc = _selectedBuilding;
  var hash = (loc.occupants ? loc.occupants.length : 0) + '|' +
    (loc.occupants || []).map(function(o) {
      return (o.agentName || '') + ':' + (o.assignedTask ? o.assignedTask.status : 'x');
    }).join(',') + '|' + (loc.upgradeLevel || 0) + '|' + (loc.taskCompletions || 0);

  if (hash !== _lastBuildingHash) {
    _lastBuildingHash = hash;
    _renderBuildingCardContent(card, loc);
  }
}

// ── Shared panel toggle helper ──────────────────────────────────────────────

function _togglePanel(panelId, btnId) {
  var panel = document.getElementById(panelId);
  if (!panel) return;
  panel.classList.toggle('collapsed');
  var btn = document.getElementById(btnId);
  if (btn) btn.innerHTML = panel.classList.contains('collapsed') ? '&#x25B6;' : '&#x25C0;';
}

// ── Agent List Panel ────────────────────────────────────────────────────────

function toggleAgentList() {
  _togglePanel('agentListPanel', 'agentListToggle');
}

function renderAgentList() {
  var body = document.getElementById('agentListBody');
  if (!body) return;
  if (typeof _sessions === 'undefined') return;

  var html = '';
  for (var sid in _sessions) {
    var sess = _sessions[sid];
    var d = sess.data;
    if (!d) continue;

    var project = d.project || sid;
    var phase = d.phase || 'idle';
    var isActive = sid === _activeSessionId;
    var done = d.results_count || 0;
    var total = d.total_tasks || 0;

    // Phase color
    var phaseColor;
    if (phase === 'done') phaseColor = C.done;
    else if (phase === 'dispatch' || phase === 'execute') phaseColor = C.in_progress;
    else phaseColor = C.textDim;

    // Session header
    html += '<div class="al-session">';
    html += '<div class="al-session-header' + (isActive ? ' active' : '') +
            '" onclick="switchSession(\'' + sid + '\')">';
    html += '<span class="al-session-name">' + escapeHtml(project) + '</span>';
    html += '<span class="al-session-badge" style="color:' + phaseColor +
            ';border-color:' + phaseColor + '">' +
            done + '/' + total + ' ' + phase.toUpperCase() + '</span>';
    html += '</div>';

    var tasks = d.tasks || [];
    var roster = [['sentinel', d.managers], ['drone', d.sub_agents], ['probe', d.reviewers]];
    for (var ri = 0; ri < roster.length; ri++) {
      var agents = roster[ri][1] || [];
      for (var ai = 0; ai < agents.length; ai++) html += _renderAgentRow(agents[ai], roster[ri][0], tasks);
    }

    html += '</div>';
  }

  if (!html) {
    html = '<div class="al-empty">No active sessions</div>';
  }

  body.innerHTML = html;
}

function _renderAgentRow(agent, tier, tasks) {
  var tierInfo = TIERS[tier] || TIERS.drone;
  var icon = tierInfo.icon;
  var task = _findTaskForAgent(agent.name, tasks);
  var status = task ? task.status : 'idle';
  var statusColor = getStateColor(status);
  var statusLabel = status.replace(/_/g, ' ').toUpperCase();
  var taskTitle = task ? (task.title || task.id || '') : '';
  var name = agent.title || prettify(agent.name);

  var html = '<div class="al-agent">';
  html += '<span class="al-agent-icon" style="color:' + statusColor + '">' + icon + '</span>';
  html += '<div class="al-agent-info">';
  html += '<div class="al-agent-name">' + escapeHtml(name) + '</div>';
  if (taskTitle) {
    html += '<div class="al-agent-task">' + escapeHtml(taskTitle) + '</div>';
  }
  html += '</div>';
  html += '<span class="al-agent-status" style="color:' + statusColor + '">' + statusLabel + '</span>';
  html += '</div>';
  return html;
}

function _findTaskForAgent(agentName, tasks) {
  for (var i = 0; i < tasks.length; i++) {
    if (tasks[i].agent === agentName) return tasks[i];
  }
  return null;
}

// ── Agent Chat (Event Log) ──────────────────────────────────────────────────

function toggleChat() {
  _togglePanel('eventLog', 'chatToggle');
  var panel = document.getElementById('eventLog');
  if (panel && panel.classList.contains('collapsed')) closePlansList();
}

// ── Chat Plan Tabs ─────────────────────────────────────────────────────────

function renderChatTabs() {
  if (typeof _sessions === 'undefined') return;

  // Reset filter if filtered session disappeared
  if (_chatFilterId !== 'all' && !_sessions[_chatFilterId]) {
    _chatFilterId = 'all';
  }

  // Update plans list if open, and always update header indicator
  if (_plansListOpen) renderPlansList();
  updateChatHeaderIndicator();
}

function setChatFilter(filterId) {
  _chatFilterId = filterId;
  if (filterId === 'all') {
    _chatUnread = {};
  } else {
    delete _chatUnread[filterId];
  }
  renderChatTabs();
  _renderFilteredChat();
}

// ── Plans List Sub-Panel ──────────────────────────────────────────────────

var _plansListOpen = false;

function togglePlansList() {
  var panel = document.getElementById('plansListPanel');
  var btn = document.getElementById('plansListBtn');
  if (!panel) return;
  _plansListOpen = !_plansListOpen;
  if (_plansListOpen) {
    panel.classList.remove('plans-list-closed');
    if (btn) btn.classList.add('open');
    renderPlansList();
  } else {
    panel.classList.add('plans-list-closed');
    if (btn) btn.classList.remove('open');
  }
}

function closePlansList() {
  if (!_plansListOpen) return;
  _plansListOpen = false;
  var panel = document.getElementById('plansListPanel');
  var btn = document.getElementById('plansListBtn');
  if (panel) panel.classList.add('plans-list-closed');
  if (btn) btn.classList.remove('open');
}

function getPhaseColor(phase) {
  var p = (phase || '').toLowerCase();
  if (p === 'done' || p === 'complete') return C.done;
  if (p === 'failed' || p === 'error') return C.failed;
  if (p === 'review' || p === 'reviewing') return C.in_specialist_review;
  if (p === 'revision') return C.revision;
  if (p === 'running' || p === 'executing' || p === 'work') return C.in_progress;
  return C.pending;
}

function renderPlansList() {
  var body = document.getElementById('plansListBody');
  var countEl = document.getElementById('plansListCount');
  if (!body) return;
  if (typeof _sessions === 'undefined') { body.innerHTML = ''; return; }

  var sessionIds = Object.keys(_sessions);
  if (countEl) countEl.textContent = sessionIds.length + ' session' + (sessionIds.length !== 1 ? 's' : '');

  var html = '';

  // "ALL PLANS" item
  var totalUnread = Object.keys(_chatUnread).length;
  html += '<div class="plan-item all-plans' + (_chatFilterId === 'all' ? ' active' : '') +
          '" onclick="selectPlanFromList(\'all\')" tabindex="0" onkeydown="if(event.key===\'Enter\')selectPlanFromList(\'all\')">';
  html += '<div class="plan-item-name">ALL PLANS</div>';
  html += '<div class="plan-item-meta">';
  html += '<span class="plan-item-progress">' + sessionIds.length + ' session' + (sessionIds.length !== 1 ? 's' : '') + '</span>';
  if (totalUnread > 0) html += '<span class="plan-item-unread"></span>';
  html += '</div></div>';

  // Per-session items
  for (var i = 0; i < sessionIds.length; i++) {
    var sid = sessionIds[i];
    var sess = _sessions[sid];
    var d = sess.data;
    if (!d) continue;
    var project = d.project || sid;
    var phase = d.phase || 'idle';
    var done = d.results_count || 0;
    var total = d.total_tasks || 0;
    var hasUnread = !!_chatUnread[sid];
    var phaseColor = getPhaseColor(phase);

    html += '<div class="plan-item' + (_chatFilterId === sid ? ' active' : '') +
            '" onclick="selectPlanFromList(\'' + sid + '\')" tabindex="0" onkeydown="if(event.key===\'Enter\')selectPlanFromList(\'' + sid + '\')">';
    html += '<div class="plan-item-name">' + escapeHtml(project) + '</div>';
    html += '<div class="plan-item-meta">';
    html += '<span class="plan-item-phase" style="background:' + hexToRgba(phaseColor, 0.15) + ';color:' + phaseColor + '">' +
            escapeHtml(phase.toUpperCase()) + '</span>';
    if (total > 0) {
      html += '<span class="plan-item-progress">' + done + '/' + total + '</span>';
    }
    if (hasUnread) html += '<span class="plan-item-unread"></span>';
    html += '</div></div>';
  }

  body.innerHTML = html;
}

function selectPlanFromList(filterId) {
  setChatFilter(filterId);
  closePlansList();
  updateChatHeaderIndicator();
}

function updateChatHeaderIndicator() {
  var el = document.getElementById('chatCurrentPlan');
  if (!el) return;
  if (typeof _sessions === 'undefined' || Object.keys(_sessions).length === 0) {
    el.textContent = 'AGENT CHAT';
    return;
  }
  if (_chatFilterId === 'all') {
    var count = Object.keys(_sessions).length;
    el.textContent = count > 1 ? 'ALL PLANS' : 'AGENT CHAT';
  } else {
    var sess = _sessions[_chatFilterId];
    if (sess && sess.data && sess.data.project) {
      el.textContent = sess.data.project;
    } else {
      el.textContent = _chatFilterId;
    }
  }
}

function _renderFilteredChat() {
  if (_chatFilterId === 'all') {
    _renderAllChat();
  } else {
    var sess = (typeof _sessions !== 'undefined') ? _sessions[_chatFilterId] : null;
    if (sess && sess.data && sess.data.log) {
      renderEventLog(sess.data.log, sess.data.project || 'NORT',
                     sess.data.results_count || 0, sess.data.total_tasks || 0);
    }
  }
}

function _renderAllChat() {
  var progressEl = document.getElementById('eventLogProgress');
  var bodyEl = document.getElementById('eventLogBody');
  if (!bodyEl) return;
  if (typeof _sessions === 'undefined') return;

  var totalDone = 0, totalAll = 0;
  var projectNames = [];
  var allLines = [];

  for (var sid in _sessions) {
    var d = _sessions[sid].data;
    if (!d) continue;
    totalDone += (d.results_count || 0);
    totalAll += (d.total_tasks || 0);
    projectNames.push(d.project || sid);

    var lines = d.log || [];
    var recent = lines.slice(-40);
    for (var i = 0; i < recent.length; i++) {
      allLines.push(recent[i]);
    }
  }

  _renderProgressBar(progressEl, projectNames.join(' + '), totalDone, totalAll);

  // Render combined lines through existing pipeline
  var combined = allLines.slice(-60);
  var parsed = combined.map(parseLogLine);
  var groups = groupChatMessages(parsed);
  var html = buildChatHTML(groups);

  var isAtBottom = bodyEl.scrollHeight - bodyEl.scrollTop - bodyEl.clientHeight < 40;
  bodyEl.innerHTML = html;
  if (isAtBottom) bodyEl.scrollTop = bodyEl.scrollHeight;
}

var _agentColorMap = {};
var _CHAT_PALETTE = [
  '#66ccff', '#cc88ff', '#ffbb44', '#66ffaa', '#ff8866',
  '#88ddff', '#dd99ff', '#ffcc66', '#88ffcc', '#ff99aa',
  '#44ddbb', '#bb88ff'
];
var _chatColorIdx = 0;
var _chatRosters = { sub_agents: [], managers: [], reviewers: [] };

function getChatColor(agentName) {
  if (!agentName) return C.textDim;
  var key = agentName.toLowerCase();
  if (key === 'master') return '#ffd700';
  if (_agentColorMap[key]) return _agentColorMap[key];
  _agentColorMap[key] = _CHAT_PALETTE[_chatColorIdx % _CHAT_PALETTE.length];
  _chatColorIdx++;
  return _agentColorMap[key];
}

function updateChatRosters(data) {
  if (data.sub_agents) _chatRosters.sub_agents = data.sub_agents;
  if (data.managers) _chatRosters.managers = data.managers;
  if (data.reviewers) _chatRosters.reviewers = data.reviewers;
}

function getAgentTier(agentKey) {
  var name = agentKey.toLowerCase();
  if (name === 'master') return 'nexus';
  var i;
  for (i = 0; i < _chatRosters.managers.length; i++) {
    if (_chatRosters.managers[i].name === name) return 'sentinel';
  }
  for (i = 0; i < _chatRosters.reviewers.length; i++) {
    if (_chatRosters.reviewers[i].name === name) return 'probe';
  }
  for (i = 0; i < _chatRosters.sub_agents.length; i++) {
    if (_chatRosters.sub_agents[i].name === name) return 'drone';
  }
  return null;
}

function getAgentTitle(agentKey) {
  var name = agentKey.toLowerCase();
  var all = _chatRosters.sub_agents.concat(_chatRosters.managers, _chatRosters.reviewers);
  for (var i = 0; i < all.length; i++) {
    if (all[i].name === name) return all[i].title || prettify(name);
  }
  if (name === 'master') return 'NEXUS';
  return agentKey.replace(/_/g, ' ');
}

var _SYSTEM_AGENTS = ['TOOL', 'MODEL', 'RAG'];
var _META_AGENTS = ['PARALLEL', 'PANEL', 'NO MANAGER'];

function parseLogLine(line) {
  var trimmed = line.replace(/^\s+/, '');

  // Issue sub-item (reviewer feedback lines starting with ↳ or indented ↳)
  if (/^\u21b3/.test(trimmed) || /^\s{2,}\u21b3/.test(line)) {
    return { agent: null, type: 'issue', content: trimmed.replace(/^\u21b3\s*/, ''), raw: line };
  }

  // Bracketed agent: [AGENT_NAME] ...
  var m = trimmed.match(/^\[([A-Z][A-Z0-9_ ]*)\]\s*(.*)/);
  if (m) {
    var agentKey = m[1];
    var content = m[2];
    if (_SYSTEM_AGENTS.indexOf(agentKey) >= 0) {
      return { agent: agentKey, type: 'meta', content: content, raw: line };
    }
    if (_META_AGENTS.indexOf(agentKey) >= 0) {
      return { agent: agentKey, type: 'system', content: content, raw: line };
    }
    if (agentKey === 'MASTER') {
      return { agent: agentKey, type: 'master', content: content, raw: line };
    }
    return { agent: agentKey, type: 'message', content: content, raw: line };
  }

  // Fallback: plain text system line
  return { agent: null, type: 'system', content: trimmed, raw: line };
}

function groupChatMessages(parsed) {
  var groups = [];
  var current = null;

  for (var i = 0; i < parsed.length; i++) {
    var p = parsed[i];

    // Meta and issue lines attach to current group
    if (p.type === 'meta' || p.type === 'issue') {
      if (current) {
        current.meta.push(p);
      }
      continue;
    }

    // System/master messages are standalone
    if (p.type === 'system' || p.type === 'master') {
      if (current) groups.push(current);
      groups.push({ agent: p.agent, type: p.type, messages: [p.content], meta: [] });
      current = null;
      continue;
    }

    // Agent message — group consecutive same-agent
    if (current && current.agent === p.agent && current.type === 'message') {
      current.messages.push(p.content);
    } else {
      if (current) groups.push(current);
      current = { agent: p.agent, type: 'message', messages: [p.content], meta: [] };
    }
  }
  if (current) groups.push(current);
  return groups;
}

function buildChatHTML(groups) {
  var html = '';

  for (var i = 0; i < groups.length; i++) {
    var g = groups[i];

    // System / master: card style
    if (g.type === 'system' || g.type === 'master') {
      var cardCls = g.type === 'master' ? 'chat-card master' : 'chat-card system';
      html += '<div class="' + cardCls + '">';
      html += '<div class="chat-card-body">' + escapeHtml(g.messages.join(' ')) + '</div>';
      html += '</div>';
      continue;
    }

    // Agent message card
    var agentKey = g.agent;
    var color = getChatColor(agentKey);
    var tier = getAgentTier(agentKey);
    var tierInfo = tier ? TIERS[tier] : null;
    var icon = tierInfo ? tierInfo.icon : '\u25CF';
    var displayName = getAgentTitle(agentKey);

    html += '<div class="chat-card" style="background:' + hexToRgba(color, 0.08) +
            ';border-color:' + hexToRgba(color, 0.2) + '">';

    // Header with icon and name
    html += '<div class="chat-card-header" style="color:' + color + '">' +
            '<span class="chat-card-icon">' + icon + '</span> ' +
            escapeHtml(displayName) + '</div>';

    // Message body/bodies
    for (var j = 0; j < g.messages.length; j++) {
      var msg = g.messages[j];
      var verdictMatch = msg.match(/^(\S+):\s*(PASS|FAIL|FLAG)\s*\((\d+)\/10\)/);

      html += '<div class="chat-card-body">';
      if (verdictMatch) {
        var vCls = verdictMatch[2].toLowerCase();
        html += '<span class="chat-verdict ' + vCls + '">' + verdictMatch[2] +
                '</span> ' + escapeHtml(verdictMatch[1]) +
                ' <span style="color:' + C.textDim + '">(' + verdictMatch[3] + '/10)</span>';
        var remainder = msg.slice(verdictMatch[0].length).trim();
        if (remainder) html += '<br>' + escapeHtml(remainder);
      } else {
        html += escapeHtml(msg);
      }
      html += '</div>';
    }

    // Attached meta (TOOL, MODEL) and issues
    for (var k = 0; k < g.meta.length; k++) {
      var meta = g.meta[k];
      if (meta.type === 'issue') {
        html += '<div class="chat-card-issue">' + escapeHtml(meta.content) + '</div>';
      } else {
        html += '<div class="chat-card-meta">' + escapeHtml((meta.agent ? '[' + meta.agent + '] ' : '') + meta.content) + '</div>';
      }
    }

    html += '</div>'; // .chat-card
  }

  return html;
}

function _renderProgressBar(el, label, done, total) {
  if (!el) return;
  var pct = total ? Math.round(done / total * 100) : 0;
  var bar = '\u2588'.repeat(Math.floor(pct / 5)) + '\u2591'.repeat(20 - Math.floor(pct / 5));
  el.innerHTML = '<div class="chat-progress">' + escapeHtml(label) + ' ' + bar + ' ' + pct + '% ' + done + '/' + total + '</div>';
}

function renderEventLog(lines, project, done, total) {
  var progressEl = document.getElementById('eventLogProgress');
  var bodyEl = document.getElementById('eventLogBody');
  if (!bodyEl) return;
  _renderProgressBar(progressEl, project, done, total);

  // Parse, group, render
  var recent = lines.slice(-40);
  var parsed = recent.map(parseLogLine);
  var groups = groupChatMessages(parsed);
  var html = buildChatHTML(groups);

  // Smart scroll: only auto-scroll if user was already at bottom
  var isAtBottom = bodyEl.scrollHeight - bodyEl.scrollTop - bodyEl.clientHeight < 40;
  bodyEl.innerHTML = html;
  if (isAtBottom) {
    bodyEl.scrollTop = bodyEl.scrollHeight;
  }
}

// ── Queue Panel ─────────────────────────────────────────────────────────────

var QUEUE_STATUS_COLORS = {
  generating: C.revision,
  queued:     C.holoBase,
  running:    C.done,
  done:       C.done,
  failed:     C.failed,
};

var _draggedPlanId = null;

function renderQueue(plans) {
  var panel = document.getElementById('queueBody');
  if (!panel) return;
  if (!plans || !plans.length) {
    panel.innerHTML = '<div style="padding:12px;font-size:9px;color:' + C.textMuted + '">No plans in queue</div>';
    return;
  }

  panel.innerHTML = '';
  plans.forEach(function(p) {
    var item = document.createElement('div');
    item.className = 'queue-item';
    item.draggable = true;
    item.dataset.id = p.id;

    // Drag handle
    var handle = document.createElement('span');
    handle.className = 'drag-handle';
    handle.textContent = '\u2261';
    item.appendChild(handle);

    // Title
    var title = document.createElement('span');
    title.className = 'queue-title';
    title.textContent = p.title || p.id || 'Untitled';
    title.style.cursor = 'pointer';
    if (p.status !== 'generating') {
      title.onclick = (function(planId) { return function(e) { e.stopPropagation(); viewPlan(planId); }; })(p.id);
    }
    item.appendChild(title);

    // Status badge
    var badge = document.createElement('span');
    badge.className = 'queue-status';
    var statusText = (p.status || 'queued').toUpperCase();
    var statusColor = QUEUE_STATUS_COLORS[p.status] || '#444';
    badge.textContent = statusText;
    badge.style.color = statusColor;
    badge.style.borderColor = statusColor + '88';
    if (p.status === 'generating' || p.status === 'running') {
      badge.classList.add('badge-pulse');
    }
    item.appendChild(badge);

    // Action buttons
    var actions = document.createElement('span');
    actions.className = 'queue-actions';

    if (p.status === 'queued' || p.status === 'failed' || p.status === 'done') {
      var runBtn = document.createElement('button');
      runBtn.textContent = 'RUN';
      runBtn.onclick = (function(planId) { return function(e) { e.stopPropagation(); runPlan(planId); }; })(p.id);
      actions.appendChild(runBtn);
    }

    if (p.status === 'done') {
      var dlBtn = document.createElement('button');
      dlBtn.textContent = '\u2B07';
      dlBtn.title = 'Download output ZIP';
      dlBtn.className = 'queue-dl-btn';
      dlBtn.onclick = (function(planId) { return function(e) { e.stopPropagation(); downloadOutputZip(planId, e.currentTarget); }; })(p.id);
      actions.appendChild(dlBtn);
      var browseBtn = document.createElement('button');
      browseBtn.textContent = '\uD83D\uDCC2';
      browseBtn.title = 'Browse output files';
      browseBtn.onclick = (function(planId) { return function(e) { e.stopPropagation(); showOutputBrowserForPlan(planId); }; })(p.id);
      actions.appendChild(browseBtn);
    }

    if (p.status === 'running' || p.status === 'generating') {
      var stopBtn = document.createElement('button');
      stopBtn.textContent = 'STOP';
      stopBtn.onclick = (function(planId) { return function(e) { e.stopPropagation(); stopPlan(planId); }; })(p.id);
      actions.appendChild(stopBtn);
    } else {
      var delBtn = document.createElement('button');
      delBtn.textContent = '\u2715';
      delBtn.onclick = (function(planId) { return function(e) { e.stopPropagation(); deletePlan(planId); }; })(p.id);
      actions.appendChild(delBtn);
    }

    item.appendChild(actions);

    // Drag events
    item.addEventListener('dragstart', onDragStart);
    item.addEventListener('dragover', onDragOver);
    item.addEventListener('dragleave', onDragLeave);
    item.addEventListener('drop', onDrop);
    item.addEventListener('dragend', onDragEnd);

    panel.appendChild(item);
  });
}

function onDragStart(e) {
  _draggedPlanId = e.currentTarget.dataset.id;
  e.currentTarget.style.opacity = '0.4';
  e.dataTransfer.effectAllowed = 'move';
}
function onDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  var item = e.currentTarget.closest('.queue-item');
  if (item && item.dataset.id !== _draggedPlanId) {
    item.classList.add('drag-over');
  }
}
function onDragLeave(e) {
  var item = e.currentTarget.closest('.queue-item');
  if (item) item.classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  var target = e.currentTarget.closest('.queue-item');
  if (!target) return;
  target.classList.remove('drag-over');
  var container = document.getElementById('queueBody');
  var items = Array.from(container.querySelectorAll('.queue-item'));
  var draggedEl = items.find(function(el) { return el.dataset.id === _draggedPlanId; });
  if (!draggedEl || draggedEl === target) return;
  var targetRect = target.getBoundingClientRect();
  var mid = targetRect.top + targetRect.height / 2;
  if (e.clientY < mid) {
    container.insertBefore(draggedEl, target);
  } else {
    container.insertBefore(draggedEl, target.nextSibling);
  }
  var newOrder = Array.from(container.querySelectorAll('.queue-item')).map(function(el) { return el.dataset.id; });
  reorderQueue(newOrder);
}
function onDragEnd(e) {
  e.currentTarget.style.opacity = '1';
  _draggedPlanId = null;
  document.querySelectorAll('.queue-item.drag-over').forEach(function(el) { el.classList.remove('drag-over'); });
}

// ── Thinking Panel (plan generation streaming) ──────────────────────────────

var _thinking = {
  planId: null,
  contextWindow: 200000,
  estimatedInput: 0,
  outputTokens: 0,
  textBuffer: '',
  displayedLen: 0,
  typewriterTimer: null,
};

function handlePlanEvent(data) {
  var evt = data.event;

  if (evt === 'generating_model') {
    showThinkingPanel(data.plan_id);
    _thinking.contextWindow = data.context_window || 200000;
    _thinking.estimatedInput = data.estimated_input_tokens || 0;
    var labelEl = document.querySelector('.thinking-label');
    if (labelEl) labelEl.textContent = 'PLAN ARCHITECT (' + data.model + ') IS THINKING...';
  }
  else if (evt === 'generating_chunk' && data.plan_id === _thinking.planId) {
    _thinking.textBuffer += data.text;
    _thinking.outputTokens = Math.round(_thinking.textBuffer.length / 4);
    var body = document.getElementById('thinkingBody');
    if (body) {
      body.textContent = _thinking.textBuffer;
      body.scrollTop = body.scrollHeight;
    }
  }
  else if (evt === 'ready') {
    if (data.plan_id === _thinking.planId) {
      _thinking.displayedLen = _thinking.textBuffer.length;
      var bodyEl = document.getElementById('thinkingBody');
      if (bodyEl) bodyEl.textContent = _thinking.textBuffer;
      var label = document.querySelector('.thinking-label');
      if (label) label.textContent = 'PLAN COMPLETE: ' + (data.title || 'UNTITLED');
      setTimeout(function() {
        if (_thinking.planId === data.plan_id) hideThinkingPanel();
      }, 8000);
    }
  }
  else if (evt === 'error' && data.plan_id === _thinking.planId) {
    var tBody = document.getElementById('thinkingBody');
    if (tBody) tBody.textContent += '\n\n[ERROR] ' + (data.message || 'Generation failed');
    var errLabel = document.querySelector('.thinking-label');
    if (errLabel) { errLabel.textContent = 'PLAN GENERATION FAILED'; errLabel.style.color = '#ff4444'; }
    setTimeout(function() { hideThinkingPanel(); }, 10000);
  }

  refreshQueue();
}

function showThinkingPanel(planId) {
  _thinking.planId = planId;
  _thinking.textBuffer = '';
  _thinking.displayedLen = 0;
  _thinking.outputTokens = 0;
  var panel = document.getElementById('thinkingPanel');
  var body = document.getElementById('thinkingBody');
  if (body) body.textContent = '';
  if (panel) panel.classList.remove('hidden');
}

function hideThinkingPanel() {
  var panel = document.getElementById('thinkingPanel');
  if (panel) panel.classList.add('hidden');
  _thinking.planId = null;
  if (_thinking.typewriterTimer) {
    clearInterval(_thinking.typewriterTimer);
    _thinking.typewriterTimer = null;
  }
}

// ── Overlay Toggles ─────────────────────────────────────────────────────────

function toggleQueue() {
  document.getElementById('queuePanel').classList.toggle('hidden');
  refreshQueue();
}

function toggleLedger() {
  document.getElementById('ledgerOverlay').classList.toggle('hidden');
  loadLedgerData();
}

function toggleHelp() {
  var overlay = document.getElementById('helpOverlay');
  if (overlay) overlay.classList.toggle('visible');
}

function toggleConfigOverlay() {
  var overlay = document.getElementById('configOverlay');
  if (overlay) overlay.classList.toggle('visible');
  syncConfigUI();
  // Load webhook URL
  fetch('/api/config').then(function(r) { return r.json(); }).then(function(c) {
    var wh = document.getElementById('webhookInput');
    if (wh) wh.value = c.webhook_url || '';
  }).catch(function() {});
}

function syncConfigUI() {
  // Boolean toggles
  document.querySelectorAll('.cfg-toggle').forEach(function(el) {
    var key = el.dataset.key;
    if (key && key in config) {
      if (config[key]) el.classList.add('on');
      else el.classList.remove('on');
    }
  });
  // Button groups
  document.querySelectorAll('.cfg-btn-group').forEach(function(group) {
    var cfgKey = group.dataset.cfg;
    if (!cfgKey || !(cfgKey in config)) return;
    var val = String(config[cfgKey]);
    group.querySelectorAll('.cfg-btn').forEach(function(btn) {
      if (String(btn.dataset.val) === val) btn.classList.add('active');
      else btn.classList.remove('active');
    });
  });
  // Range sliders
  document.querySelectorAll('.cfg-range').forEach(function(range) {
    var cfgKey = range.dataset.cfg;
    if (cfgKey && cfgKey in config) {
      range.value = config[cfgKey];
    }
  });
  // Range value displays
  document.querySelectorAll('[data-cfg-display]').forEach(function(el) {
    var cfgKey = el.dataset.cfgDisplay;
    if (cfgKey && cfgKey in config) {
      el.textContent = config[cfgKey];
    }
  });
}

// ── Completion Overlay ──────────────────────────────────────────────────────

var _completionShown = false;

function showCompletion(data) {
  if (!config.completionFx || _completionShown) return;
  _completionShown = true;

  var overlay = document.getElementById('completionOverlay');
  if (!overlay) return;

  var totalRevisions = (data.tasks || []).reduce(function(s, t) { return s + (t.revision_count || 0); }, 0);
  var elapsed = liveStartedAt ? Math.round((Date.now() - new Date(liveStartedAt).getTime()) / 1000) : 0;
  var mins = Math.floor(elapsed / 60), secs = elapsed % 60;

  var stats = document.getElementById('completionStats');
  if (stats) {
    stats.innerHTML =
      '<div class="stat-block"><div class="stat-val">' + (data.total_tasks || 0) + '</div><div class="stat-lbl">TASKS</div></div>' +
      '<div class="stat-block"><div class="stat-val">' + totalRevisions + '</div><div class="stat-lbl">REVISIONS</div></div>' +
      '<div class="stat-block"><div class="stat-val">' + (data.tokens_used || 0).toLocaleString() + '</div><div class="stat-lbl">TOKENS</div></div>' +
      '<div class="stat-block"><div class="stat-val">' + mins + 'm' + secs + 's</div><div class="stat-lbl">ELAPSED</div></div>';
  }

  var body = document.getElementById('completionBody');
  if (body) body.textContent = data.synthesis_report || '';

  // Show validation results
  var valSection = document.getElementById('completionValidation');
  var valBody = document.getElementById('completionValidationBody');
  var validation = data.validation || {};
  if (valBody && (validation.passed || validation.failed)) {
    var valHtml = '<div style="margin-bottom:6px;color:var(--text-dim)">' + (validation.summary || '') + '</div>';
    var failed = validation.failed || [];
    var passed = validation.passed || [];
    if (failed.length) {
      valHtml += '<div style="margin-bottom:4px;color:var(--state-error);font-weight:600">FAILURES</div>';
      for (var fi = 0; fi < failed.length; fi++) {
        valHtml += '<div class="validation-fail">\u2717 ' + escapeHtml(failed[fi].file) + ' — ' + escapeHtml(failed[fi].error || '') + '</div>';
      }
    }
    valHtml += '<div style="margin-top:4px;color:var(--state-done)">' + passed.length + ' file(s) passed</div>';
    valBody.innerHTML = valHtml;
    if (valSection) valSection.classList.remove('hidden');
  }

  // Show coherence report
  var coherence = data.coherence_report || {};
  if (coherence.summary && body) {
    var cohHtml = '<div style="margin-top:12px;padding-top:8px;border-top:1px solid var(--glass-border)">';
    cohHtml += '<div style="font-size:9px;letter-spacing:1px;color:var(--text-dim);margin-bottom:4px">COHERENCE CHECK</div>';
    cohHtml += '<div style="color:' + (coherence.coherent ? 'var(--state-done)' : 'var(--state-error)') + '">';
    cohHtml += (coherence.coherent ? '\u2713 ' : '\u2717 ') + escapeHtml(coherence.summary);
    cohHtml += '</div>';
    var issues = coherence.issues || [];
    if (issues.length) {
      cohHtml += '<ul style="margin-top:4px;padding-left:16px;font-size:9px">';
      for (var ci = 0; ci < issues.length; ci++) {
        cohHtml += '<li style="color:var(--state-error)">' + escapeHtml(issues[ci].file || '') + ': ' + escapeHtml(issues[ci].description || '') + '</li>';
      }
      cohHtml += '</ul>';
    }
    cohHtml += '</div>';
    body.innerHTML = body.textContent + cohHtml;
  }

  overlay.classList.remove('hidden');
}

function hideCompletion() {
  document.getElementById('completionOverlay').classList.add('hidden');
}

// ── Approval Banner ─────────────────────────────────────────────────────────

function showApproval(data) {
  var banner = document.getElementById('approvalBanner');
  if (!banner) return;
  banner.classList.remove('hidden');

  var text = document.getElementById('approvalText');
  if (text) {
    var agent = data.agent || 'unknown';
    var tool = data.tool || 'unknown';
    var argsStr = '';
    if (data.args) {
      try { argsStr = JSON.stringify(data.args).slice(0, 100); } catch(e) { argsStr = '...'; }
    }
    text.textContent = agent + ' wants to ' + tool + '(' + argsStr + ')';
  }

  banner.dataset.toolCallId = data.id || data.tool_call_id || '';
}

function hideApproval() {
  var banner = document.getElementById('approvalBanner');
  if (banner) banner.classList.add('hidden');
}

// ── Plan Viewer ─────────────────────────────────────────────────────────────

function showPlanViewer(content, title) {
  var overlay = document.getElementById('planViewer');
  if (!overlay) return;
  overlay.classList.remove('hidden');
  var contentEl = document.getElementById('planContent');
  if (contentEl) contentEl.textContent = content;
}

function hidePlanViewer() {
  var overlay = document.getElementById('planViewer');
  if (overlay) overlay.classList.add('hidden');
}

// ── Verdict Popup ───────────────────────────────────────────────────────────

function showVerdict(verdict) {
  var node = getNodeByAgent(verdict.agent);
  if (!node) return;
  var isPASS = verdict.verdict === 'PASS';
  spawnEffect(node.x, node.y, isPASS ? C.done : C.failed);
  console.log('Verdict: ' + verdict.verdict + ' score=' + verdict.score + ' for ' + verdict.task_id);
}

// ── Ledger Rendering ────────────────────────────────────────────────────────

function switchLedgerTab(tab, btn) { _switchTab('ledger-tab', 'ledger-section', 'ledger-', tab, btn); }

function renderLedger(costs, scores) {
  var body = document.getElementById('ledgerBody');
  if (!body) return;

  var html = '';

  // Costs section
  html += '<div id="ledger-costs" class="ledger-section active">';
  html += '<div class="stats-grid" style="margin-bottom:12px">';
  html += '<div class="stat-block"><div class="stat-val">' + (costs.total_tokens || 0).toLocaleString() + '</div><div class="stat-lbl">TOTAL TOKENS</div></div>';
  html += '<div class="stat-block"><div class="stat-val">' + ((costs.recent_runs || []).length) + '</div><div class="stat-lbl">RUNS</div></div>';
  html += '</div>';

  // By agent
  var agents = costs.by_agent || [];
  if (agents.length) {
    var agentMax = Math.max.apply(null, agents.map(function(a) { return a.total_tokens; })) || 1;
    html += '<div style="font-size:9px;color:' + C.textDim + ';letter-spacing:1px;margin:8px 0 4px">TOKENS BY AGENT</div>';
    html += _barChart(agents.map(function(a) { return { label: a.agent, val: a.total_tokens }; }), agentMax, '#0088cc');
  }

  // By model
  var models = costs.by_model || [];
  if (models.length) {
    var modelMax = Math.max.apply(null, models.map(function(m) { return m.total_tokens; })) || 1;
    html += '<div style="font-size:9px;color:' + C.textDim + ';letter-spacing:1px;margin:8px 0 4px">TOKENS BY MODEL</div>';
    html += _barChart(models.map(function(m) { return { label: m.model, val: m.total_tokens }; }), modelMax, '#44aaff');
  }

  // Recent runs
  var runs = costs.recent_runs || [];
  if (runs.length) {
    html += '<div style="font-size:9px;color:' + C.textDim + ';letter-spacing:1px;margin:8px 0 4px">RECENT RUNS</div>';
    html += '<table class="ledger-table"><tr><th>PLAN</th><th>TOKENS</th><th>TASKS</th><th>STATUS</th></tr>';
    runs.forEach(function(r) {
      var sColor = r.status === 'done' ? '#00ff88' : '#ff8800';
      html += '<tr><td>' + escapeHtml(r.plan_name || '?') + '</td><td>' + (r.total_tokens || 0).toLocaleString() +
        '</td><td>' + (r.task_count || 0) + '</td><td style="color:' + sColor + '">' + (r.status || '?').toUpperCase() + '</td></tr>';
    });
    html += '</table>';
  }
  html += '</div>';

  // Scores section
  html += '<div id="ledger-scores" class="ledger-section">';
  var dist = scores.distribution || { high: 0, mid: 0, low: 0 };
  html += '<div class="stats-grid" style="margin-bottom:12px">';
  html += '<div class="stat-block"><div class="stat-val" style="color:#00ff88">' + (dist.high || 0) + '</div><div class="stat-lbl">SCORE 8+</div></div>';
  html += '<div class="stat-block"><div class="stat-val" style="color:#ffaa00">' + (dist.mid || 0) + '</div><div class="stat-lbl">SCORE 5-7</div></div>';
  html += '<div class="stat-block"><div class="stat-val" style="color:#ff4444">' + (dist.low || 0) + '</div><div class="stat-lbl">SCORE &lt;5</div></div>';
  html += '</div>';

  // Avg score by agent
  var sAgents = scores.by_agent || [];
  if (sAgents.length) {
    var scoreItems = sAgents.map(function(a) {
      return { label: a.agent + ' (' + a.reviews + ' reviews)', val: Math.round((a.avg_score || 0) * 10) / 10 };
    });
    html += '<div style="font-size:9px;color:' + C.textDim + ';letter-spacing:1px;margin:8px 0 4px">AVG SCORE BY AGENT</div>';
    html += _barChart(scoreItems, 10, '#00ff88');
  }

  // Recent scores
  var recent = scores.recent_scores || [];
  if (recent.length) {
    html += '<div style="font-size:9px;color:' + C.textDim + ';letter-spacing:1px;margin:8px 0 4px">RECENT REVIEWS</div>';
    html += '<table class="ledger-table"><tr><th>TASK</th><th>AGENT</th><th>SCORE</th><th>VERDICT</th><th>REVIEWER</th></tr>';
    recent.forEach(function(s) {
      var vColor = s.verdict === 'PASS' ? '#00ff88' : '#ff4444';
      html += '<tr><td>' + escapeHtml(s.task_id) + '</td><td>' + escapeHtml(s.agent) + '</td><td>' + s.score + '/10</td>' +
        '<td style="color:' + vColor + '">' + s.verdict + '</td><td>' + escapeHtml(s.reviewer) + '</td></tr>';
    });
    html += '</table>';
  }
  html += '</div>';

  body.innerHTML = html;
}

function _barChart(items, maxVal, color) {
  if (!maxVal) maxVal = 1;
  return items.map(function(it) {
    var w = Math.max(2, (it.val / maxVal) * 200);
    return '<div class="ledger-bar-row">' +
      '<span class="ledger-bar-label">' + escapeHtml(String(it.label)) + '</span>' +
      '<div class="ledger-bar-fill" style="width:' + w + 'px;background:' + color + '"></div>' +
      '<span class="ledger-bar-val">' + it.val.toLocaleString() + '</span>' +
      '</div>';
  }).join('');
}

// ── Model Config ────────────────────────────────────────────────────────────

var _modelConfigData = [];

var MODEL_PRESETS = [
  { name: 'recommended', label: 'RECOMMENDED',
    include: ['opus', 'sonnet', 'gemini', 'nemotron', 'qwen', 'gpt-4o', 'o3-', 'o4-', 'llama-4', 'deepseek'],
    exclude: ['embed', 'vision', 'tts', 'whisper', 'dall', 'stable-diffusion', 'moderation', 'realtime', 'image'] },
  { name: 'claude', label: 'CLAUDE', include: ['claude'], exclude: [] },
  { name: 'openai', label: 'OPENAI', include: ['gpt', 'o1-', 'o3-', 'o4-'], exclude: [] },
  { name: 'aws', label: 'AWS', include: ['bedrock', 'nova'], exclude: [] },
  { name: 'google', label: 'GOOGLE', include: ['gemini', 'gemma'], exclude: [] },
  { name: 'opensource', label: 'OPEN SOURCE',
    include: ['llama', 'qwen', 'nemotron', 'mistral', 'deepseek', 'phi-'], exclude: [] },
  { name: 'all', label: 'ALL', include: [], exclude: [] },
];

function _modelMatchesPreset(modelId, preset) {
  var lower = modelId.toLowerCase();
  if (preset.exclude.length > 0) {
    for (var i = 0; i < preset.exclude.length; i++) {
      if (lower.indexOf(preset.exclude[i]) !== -1) return false;
    }
  }
  if (preset.include.length === 0) return true;
  for (var j = 0; j < preset.include.length; j++) {
    if (lower.indexOf(preset.include[j]) !== -1) return true;
  }
  return false;
}

function renderModelConfig(data) {
  var overlay = document.getElementById('modelConfigOverlay');
  if (overlay) overlay.classList.add('visible');
  _modelConfigData = data.models || [];
  var filterInput = document.getElementById('modelFilterInput');
  if (filterInput) filterInput.value = '';
  _renderModelPresets();
  _renderModelList('');
}

function _renderModelPresets() {
  var row = document.getElementById('modelPresetRow');
  if (!row) return;
  row.innerHTML = MODEL_PRESETS.map(function(p) {
    return '<button class="model-preset-btn" data-preset="' + p.name + '" onclick="applyModelPreset(\'' + p.name + '\')">' + p.label + '</button>';
  }).join('');
}

function applyModelPreset(presetName) {
  var preset = MODEL_PRESETS.filter(function(p) { return p.name === presetName; })[0];
  if (!preset) return;

  var enabledIds;
  if (presetName === 'all') {
    enabledIds = null;
    _modelConfigData.forEach(function(m) { m.enabled = true; });
  } else {
    enabledIds = [];
    _modelConfigData.forEach(function(m) {
      var match = _modelMatchesPreset(m.id, preset);
      m.enabled = match;
      if (match) enabledIds.push(m.id);
    });
  }

  saveModels(enabledIds);

  // Update toggle UI in place
  var toggles = document.querySelectorAll('#modelConfigBody .cfg-toggle');
  toggles.forEach(function(el) {
    var model = _modelConfigData.filter(function(m) { return m.id === el.dataset.model; })[0];
    if (model) {
      if (model.enabled) el.classList.add('on');
      else el.classList.remove('on');
    }
  });

  // Highlight active preset
  var btns = document.querySelectorAll('.model-preset-btn');
  btns.forEach(function(btn) {
    if (btn.dataset.preset === presetName) btn.classList.add('active');
    else btn.classList.remove('active');
  });
}

function _renderModelList(filter) {
  var body = document.getElementById('modelConfigBody');
  if (!body) return;
  var lowerFilter = filter.toLowerCase();
  var filtered = _modelConfigData.filter(function(m) {
    return !lowerFilter || m.id.toLowerCase().indexOf(lowerFilter) !== -1;
  });
  body.innerHTML = filtered.map(function(m) {
    return '<div class="config-row">' +
      '<span class="config-label">' + escapeHtml(m.id) + ' <span style="color:' + C.textMuted + '">[' + (m.tier || '?') + ']</span></span>' +
      '<div class="cfg-toggle ' + (m.enabled ? 'on' : '') + '" data-model="' + escapeHtml(m.id) + '" onclick="toggleModel(this)"></div>' +
      '</div>';
  }).join('');
}

function filterModels(value) {
  _renderModelList(value);
}

function toggleModel(el) {
  el.classList.toggle('on');
  // Update internal state
  var modelId = el.dataset.model;
  var model = _modelConfigData.filter(function(m) { return m.id === modelId; })[0];
  if (model) model.enabled = el.classList.contains('on');
  // Clear preset highlight on manual toggle
  var btns = document.querySelectorAll('.model-preset-btn');
  btns.forEach(function(btn) { btn.classList.remove('active'); });
  // Save
  var allToggles = document.querySelectorAll('#modelConfigBody .cfg-toggle');
  var enabledToggles = document.querySelectorAll('#modelConfigBody .cfg-toggle.on');
  var allEnabled = allToggles.length === enabledToggles.length;
  var enabledModels = allEnabled ? null : Array.from(enabledToggles).map(function(e) { return e.dataset.model; });
  saveModels(enabledModels);
}

// ── Utility ─────────────────────────────────────────────────────────────────

function _slugify(s) { return s.trim().toLowerCase().replace(/\s+/g, '_'); }
function _csvTags(s) { return s.split(',').map(function(t) { return t.trim(); }).filter(Boolean); }

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── Agent Registry Panel ───────────────────────────────────────────────────

var _agentsActiveTab = 'sub_agents';
var _agentsData = null;
var _teamsData = null;
var _showRetired = false;
var _agentsSortBy = 'score';

function setAgentsSort(metric, btn) {
  _agentsSortBy = metric;
  var btns = document.querySelectorAll('.agent-sort-btn');
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
  if (btn) btn.classList.add('active');
  renderAgentsList();
}

function _relativeTime(ts) {
  if (!ts) return 'never';
  var diff = Date.now() - new Date(ts).getTime();
  if (diff < 0) diff = 0;
  var sec = Math.floor(diff / 1000);
  if (sec < 60) return sec + 's ago';
  var min = Math.floor(sec / 60);
  if (min < 60) return min + 'm ago';
  var hr = Math.floor(min / 60);
  if (hr < 24) return hr + 'h ago';
  var d = Math.floor(hr / 24);
  if (d < 30) return d + 'd ago';
  return Math.floor(d / 30) + 'mo ago';
}

function toggleAgentsPanel() {
  var el = document.getElementById('agentsOverlay');
  el.classList.toggle('hidden');
  if (!el.classList.contains('hidden')) loadAgentsData();
}

function loadAgentsData() {
  fetch('/api/agents')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _agentsData = data.agents;
      renderAgentsList();
    })
    .catch(function(e) { console.error('Failed to load agents:', e); });
}

function showAgentTab(tab, btn) {
  _agentsActiveTab = tab;
  var tabs = document.querySelectorAll('.agents-tab');
  for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
  if (btn) btn.classList.add('active');

  // Toggle create buttons
  var createAgentBtn = document.querySelector('.agents-create-btn');
  var createTeamBtn = document.getElementById('createTeamBtn');
  if (tab === 'teams') {
    if (createAgentBtn) createAgentBtn.style.display = 'none';
    if (createTeamBtn) createTeamBtn.style.display = '';
    loadTeamsData();
  } else {
    if (createAgentBtn) createAgentBtn.style.display = '';
    if (createTeamBtn) createTeamBtn.style.display = 'none';
    renderAgentsList();
  }
}

function toggleShowRetired() {
  _showRetired = document.getElementById('showRetiredToggle').checked;
  renderAgentsList();
}

function renderAgentsList() {
  if (_agentsActiveTab === 'teams') {
    loadTeamsData();
    return;
  }
  if (!_agentsData) return;
  var list = document.getElementById('agentsList');
  var filtered = _agentsData.filter(function(a) {
    return a._type === _agentsActiveTab || (!a._type && true);
  });

  if (_agentsData.length > 0 && !_agentsData[0]._type) {
    filtered = _agentsData;
  }

  // Filter retired
  if (!_showRetired) {
    filtered = filtered.filter(function(a) { return !a.retired; });
  }

  // Sort by selected metric
  filtered.sort(function(a, b) {
    if (_agentsSortBy === 'score') return (b.avg_score || 0) - (a.avg_score || 0);
    if (_agentsSortBy === 'runs') return (b.runs || 0) - (a.runs || 0);
    if (_agentsSortBy === 'rejection') {
      var ra = (a.runs || 0) > 0 ? ((a.rejections || 0) / a.runs) : 0;
      var rb = (b.runs || 0) > 0 ? ((b.rejections || 0) / b.runs) : 0;
      return rb - ra;
    }
    return 0;
  });

  var html = '';
  if (filtered.length === 0) {
    html = '<div style="padding:12px;color:' + C.textMuted + ';font-size:9px">No agents in this category</div>';
  }

  for (var i = 0; i < filtered.length; i++) {
    var a = filtered[i];
    var scoreColor = (a.avg_score || 0) >= 7 ? '#66ffaa' : (a.avg_score || 0) >= 5 ? '#ffbb44' : '#ff5566';
    var builtinBadge = a.builtin ? '<span style="color:' + C.textMuted + ';font-size:7px;margin-left:4px">BUILTIN</span>' : '';
    var retiredBadge = a.retired ? '<span style="color:#ff5566;font-size:7px;margin-left:4px;border:1px solid #ff556644;padding:0 3px;border-radius:2px">RETIRED</span>' : '';
    var warningBadge = '';
    if (!a.retired && a.runs >= 5 && (a.avg_score || 0) < 4) {
      warningBadge = '<span style="color:#ffaa00;font-size:7px;margin-left:4px" title="Low performance: avg score < 4 over 5+ tasks">&#x26A0; LOW PERF</span>';
    }
    var trustedBadge = '';
    if (a.runs >= 5 && (a.avg_score || 0) > 8) {
      trustedBadge = '<span style="color:#66ffaa;font-size:7px;margin-left:4px;border:1px solid #66ffaa44;padding:0 3px;border-radius:2px">TRUSTED</span>';
    }

    var opacity = a.retired ? 'opacity:0.5;' : '';
    html += '<div class="agent-item" style="' + opacity + '" onclick="showAgentRegistryDetail(\'' + escapeHtml(a.name) + '\',\'' + (_agentsActiveTab) + '\')">';
    html += '<div class="agent-item-header">';
    html += '<span class="agent-item-name">' + escapeHtml(a.title || a.name) + builtinBadge + retiredBadge + warningBadge + trustedBadge + '</span>';
    html += '<span class="agent-item-score" style="color:' + scoreColor + '">';
    if (a.runs > 0) {
      html += a.avg_score.toFixed(1) + ' <span style="color:' + C.textMuted + '">(' + a.runs + ' runs)</span>';
    } else {
      html += '<span style="color:' + C.textMuted + '">new</span>';
    }
    html += '</span></div>';
    html += '<div class="agent-item-desc">' + escapeHtml((a.description || '').slice(0, 80)) + '</div>';
    if (a.tags && a.tags.length > 0) {
      html += '<div class="agent-item-tags">';
      for (var t = 0; t < Math.min(a.tags.length, 5); t++) {
        html += '<span class="agent-tag">' + escapeHtml(a.tags[t]) + '</span>';
      }
      html += '</div>';
    }

    // Performance stats
    if (a.runs > 0) {
      var rejPct = a.runs > 0 ? Math.round(((a.rejections || 0) / a.runs) * 100) : 0;
      var rejColor = rejPct < 20 ? '#66ffaa' : rejPct <= 40 ? '#ffbb44' : '#ff5566';
      var scorePct = Math.min(((a.avg_score || 0) / 10) * 100, 100);
      var passCount = (a.runs || 0) - (a.rejections || 0);

      html += '<div class="agent-perf">';
      html += '<div class="agent-perf-row">';
      html += '<span class="agent-perf-val" style="color:' + scoreColor + '">SCORE ' + (a.avg_score || 0).toFixed(1) + '/10</span>';
      html += '<div class="agent-perf-bar"><div class="agent-perf-fill" style="width:' + scorePct + '%;background:' + scoreColor + '"></div></div>';
      html += '</div>';
      html += '<div class="agent-perf-row">';
      html += '<span class="agent-perf-val" style="color:' + rejColor + '">REJECT ' + rejPct + '%</span>';
      html += '<div class="agent-perf-bar"><div class="agent-perf-fill" style="width:' + rejPct + '%;background:' + rejColor + '"></div></div>';
      html += '</div>';
      html += '<div class="agent-perf-row">';
      html += '<span class="agent-perf-val">' + passCount + ' pass / ' + (a.rejections || 0) + ' fail / ' + a.runs + ' total</span>';
      html += '<span class="agent-perf-val">' + _relativeTime(a.last_used) + '</span>';
      html += '</div>';
      html += '</div>';
    } else {
      html += '<div class="agent-perf"><div class="agent-perf-empty">No performance data yet</div></div>';
    }

    html += '</div>';
  }

  list.innerHTML = html;
}

function showAgentRegistryDetail(name, agentType) {
  fetch('/api/agents/' + agentType + '/' + name)
    .then(function(r) { return r.json(); })
    .then(function(agent) {
      var form = document.getElementById('agentFormArea');
      form.classList.remove('hidden');

      var html = '<div class="agent-edit-form">';
      html += '<h3 style="color:' + C.textPrimary + ';margin:0 0 8px 0;font-size:11px">' + escapeHtml(agent.title || agent.name) + '</h3>';

      // Version history dropdown
      var versions = agent.versions || [];
      if (versions.length > 0) {
        html += '<div style="margin-bottom:8px">';
        html += '<label>Version History:</label>';
        html += '<select id="agentVersionSelect" class="agent-input" style="margin-bottom:4px">';
        html += '<option value="">Current version</option>';
        for (var v = versions.length - 1; v >= 0; v--) {
          var ver = versions[v];
          var ts = ver.timestamp ? new Date(ver.timestamp).toLocaleString() : '';
          html += '<option value="' + ver.version + '">v' + ver.version + ' — ' + escapeHtml(ts) + '</option>';
        }
        html += '</select>';
        html += '<button class="agent-save-btn" style="font-size:7px;padding:2px 8px" onclick="doRollbackAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\')">ROLLBACK</button>';
        html += '</div>';
      }

      html += '<label>Description:</label>';
      html += '<textarea id="agentEditDesc" class="agent-input" rows="3">' + escapeHtml(agent.description || '') + '</textarea>';
      html += '<label>Tags (comma-separated):</label>';
      html += '<input id="agentEditTags" class="agent-input" value="' + escapeHtml((agent.tags || []).join(', ')) + '">';

      if (agentType === 'sub_agents') {
        html += '<label>Tools (comma-separated):</label>';
        html += '<input id="agentEditTools" class="agent-input" value="' + escapeHtml((agent.tools || []).join(', ')) + '">';
      }

      html += '<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">';
      html += '<button class="agent-save-btn" onclick="saveAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\')">SAVE</button>';
      html += '<button class="agent-save-btn" style="background:rgba(100,200,255,0.15);border-color:rgba(100,200,255,0.3);color:#88ddff" onclick="doCloneAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\')">CLONE</button>';

      if (agent.retired) {
        html += '<button class="agent-save-btn" style="background:rgba(102,255,170,0.15);border-color:rgba(102,255,170,0.3)" onclick="doRetireAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\', false)">UNRETIRE</button>';
      } else {
        html += '<button class="agent-delete-btn" style="background:rgba(255,170,0,0.15);border-color:rgba(255,170,0,0.3);color:#ffaa00" onclick="doRetireAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\', true)">RETIRE</button>';
      }

      if (!agent.builtin) {
        html += '<button class="agent-delete-btn" onclick="deleteAgent(\'' + agentType + '\',\'' + escapeHtml(agent.name) + '\')">DELETE</button>';
      }
      html += '<button class="agent-cancel-btn" onclick="document.getElementById(\'agentFormArea\').classList.add(\'hidden\')">CANCEL</button>';
      html += '</div></div>';

      form.innerHTML = html;
    });
}

function saveAgent(agentType, name) {
  var desc = document.getElementById('agentEditDesc').value;
  var tags = _csvTags(document.getElementById('agentEditTags').value);
  var updates = { description: desc, tags: tags };

  var toolsEl = document.getElementById('agentEditTools');
  if (toolsEl) {
    updates.tools = _csvTags(toolsEl.value);
  }

  fetch('/api/agents/' + agentType + '/' + name, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates)
  }).then(function() {
    document.getElementById('agentFormArea').classList.add('hidden');
    loadAgentsData();
  });
}

function deleteAgent(agentType, name) {
  if (!confirm('Delete agent ' + name + '?')) return;
  fetch('/api/agents/' + agentType + '/' + name, { method: 'DELETE' })
    .then(function() {
      document.getElementById('agentFormArea').classList.add('hidden');
      loadAgentsData();
    });
}

function doCloneAgent(agentType, name) {
  var newName = prompt('Clone name (leave blank for ' + name + '_copy):', name + '_copy');
  if (newName === null) return;
  cloneAgent(agentType, name, newName || null).then(function(result) {
    if (result && result.ok) {
      document.getElementById('agentFormArea').classList.add('hidden');
      loadAgentsData();
    } else {
      alert('Clone failed: ' + (result && result.detail ? result.detail : 'unknown error'));
    }
  });
}

function doRetireAgent(agentType, name, retired) {
  retireAgent(agentType, name, retired).then(function(result) {
    if (result && result.ok) {
      document.getElementById('agentFormArea').classList.add('hidden');
      loadAgentsData();
    }
  });
}

function doRollbackAgent(agentType, name) {
  var sel = document.getElementById('agentVersionSelect');
  if (!sel || !sel.value) { alert('Select a version to rollback to'); return; }
  if (!confirm('Rollback ' + name + ' to version ' + sel.value + '?')) return;
  rollbackAgent(agentType, name, parseInt(sel.value)).then(function(result) {
    if (result && result.ok) {
      showAgentRegistryDetail(name, agentType);
      loadAgentsData();
    } else {
      alert('Rollback failed');
    }
  });
}

function showCreateAgentForm() {
  var form = document.getElementById('agentFormArea');
  form.classList.remove('hidden');

  var html = '<div class="agent-edit-form">';
  html += '<h3 style="color:' + C.textPrimary + ';margin:0 0 8px 0;font-size:11px">CREATE NEW AGENT</h3>';
  html += '<label>Type:</label>';
  html += '<select id="newAgentType" class="agent-input"><option value="sub_agents">Sub-Agent</option><option value="managers">Manager</option><option value="reviewers">Reviewer</option></select>';
  html += '<label>Name (lowercase, underscores):</label>';
  html += '<input id="newAgentName" class="agent-input" placeholder="e.g. python_expert">';
  html += '<label>Title:</label>';
  html += '<input id="newAgentTitle" class="agent-input" placeholder="e.g. Senior Python Developer">';
  html += '<label>Description:</label>';
  html += '<textarea id="newAgentDesc" class="agent-input" rows="3" placeholder="What this agent does..."></textarea>';
  html += '<label>Tags (comma-separated):</label>';
  html += '<input id="newAgentTags" class="agent-input" placeholder="e.g. python, backend, api">';
  html += '<label>Tools (comma-separated, for sub-agents):</label>';
  html += '<input id="newAgentTools" class="agent-input" placeholder="e.g. write_file, execute_code">';
  html += '<div style="margin-top:8px;display:flex;gap:6px">';
  html += '<button class="agent-save-btn" onclick="createNewAgent()">CREATE</button>';
  html += '<button class="agent-cancel-btn" onclick="document.getElementById(\'agentFormArea\').classList.add(\'hidden\')">CANCEL</button>';
  html += '</div></div>';

  form.innerHTML = html;
}

function createNewAgent() {
  var agentType = document.getElementById('newAgentType').value;
  var spec = {
    name: _slugify(document.getElementById('newAgentName').value),
    title: document.getElementById('newAgentTitle').value.trim(),
    description: document.getElementById('newAgentDesc').value.trim(),
    tags: _csvTags(document.getElementById('newAgentTags').value),
    tools: _csvTags(document.getElementById('newAgentTools').value),
  };

  if (agentType === 'managers') {
    spec.expertise_blend = spec.tags.slice();
    spec.oversees = [];
  }
  if (agentType === 'reviewers') {
    spec.focus_areas = spec.tags.slice();
    spec.applies_to = spec.tags.slice();
  }

  fetch('/api/agents/' + agentType, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec)
  }).then(function(r) { return r.json(); })
    .then(function() {
      document.getElementById('agentFormArea').classList.add('hidden');
      loadAgentsData();
    })
    .catch(function(e) { alert('Error: ' + e); });
}

// ── Teams Tab ──────────────────────────────────────────────────────────────

var _presetsData = null;

function loadTeamsData() {
  Promise.all([fetchTeams(), fetchTeamPresets()]).then(function(results) {
    _teamsData = results[0];
    _presetsData = results[1];
    renderTeamsList();
  });
}

function renderTeamsList() {
  var list = document.getElementById('agentsList');
  var html = '';

  // ── Built-in Presets section ──
  if (_presetsData && _presetsData.length > 0) {
    html += '<div style="font-size:7px;color:' + C.textDim + ';letter-spacing:1.5px;padding:8px 8px 4px;border-bottom:1px solid rgba(100,200,255,0.08)">BUILT-IN PRESETS</div>';
    for (var p = 0; p < _presetsData.length; p++) {
      var preset = _presetsData[p];
      var agentCount = (preset.agents || []).length;
      html += '<div class="agent-item" style="border-left:2px solid rgba(100,200,255,0.25);margin-left:4px">';
      html += '<div class="agent-item-header">';
      html += '<span class="agent-item-name">' + escapeHtml(preset.title || preset.name) + ' <span style="color:' + C.textMuted + ';font-size:7px;margin-left:4px">PRESET</span></span>';
      html += '<span class="agent-item-score" style="color:' + C.textMuted + '">' + agentCount + ' agents</span>';
      html += '</div>';
      html += '<div class="agent-item-desc">' + escapeHtml((preset.description || '').slice(0, 100)) + '</div>';
      if (preset.agents && preset.agents.length > 0) {
        html += '<div class="agent-item-tags">';
        for (var a = 0; a < preset.agents.length; a++) {
          html += '<span class="agent-tag">' + escapeHtml(preset.agents[a].name || preset.agents[a]) + '</span>';
        }
        html += '</div>';
      }
      html += '<div style="margin-top:4px">';
      html += '<button class="agent-save-btn" style="font-size:7px;padding:2px 8px" onclick="event.stopPropagation(); promptApplyPreset(\'' + escapeHtml(preset.name) + '\')">APPLY</button>';
      html += '</div>';
      html += '</div>';
    }
  }

  // ── Custom Teams section ──
  html += '<div style="font-size:7px;color:' + C.textDim + ';letter-spacing:1.5px;padding:8px 8px 4px;border-bottom:1px solid rgba(100,200,255,0.08)">CUSTOM TEAMS</div>';
  if (!_teamsData || _teamsData.length === 0) {
    html += '<div style="padding:12px;color:' + C.textMuted + ';font-size:9px">No custom teams defined</div>';
  } else {
    for (var i = 0; i < _teamsData.length; i++) {
      var team = _teamsData[i];
      var teamAgentCount = (team.agents || []).length;
      html += '<div class="agent-item" onclick="showTeamDetail(\'' + escapeHtml(team.name) + '\')">';
      html += '<div class="agent-item-header">';
      html += '<span class="agent-item-name">' + escapeHtml(team.title || team.name) + '</span>';
      html += '<span class="agent-item-score" style="color:' + C.textMuted + '">' + teamAgentCount + ' agents</span>';
      html += '</div>';
      html += '<div class="agent-item-desc">' + escapeHtml((team.description || '').slice(0, 80)) + '</div>';
      if (team.agents && team.agents.length > 0) {
        html += '<div class="agent-item-tags">';
        for (var ta = 0; ta < Math.min(team.agents.length, 6); ta++) {
          html += '<span class="agent-tag">' + escapeHtml(team.agents[ta].name || team.agents[ta]) + '</span>';
        }
        if (team.agents.length > 6) html += '<span class="agent-tag">+' + (team.agents.length - 6) + '</span>';
        html += '</div>';
      }
      html += '</div>';
    }
  }

  list.innerHTML = html;
}

function promptApplyPreset(presetName) {
  var teamName = prompt('Team name for this preset:', presetName + '-team');
  if (!teamName) return;
  teamName = _slugify(teamName);
  applyTeamPreset(presetName, teamName).then(function(result) {
    if (result && result.ok) {
      loadTeamsData();
    } else {
      alert('Failed to apply preset: ' + (result && result.detail ? result.detail : 'unknown error'));
    }
  });
}

function showTeamDetail(name) {
  var team = _teamsData.find(function(t) { return t.name === name; });
  if (!team) return;
  var form = document.getElementById('agentFormArea');
  form.classList.remove('hidden');

  var html = '<div class="agent-edit-form">';
  html += '<h3 style="color:' + C.textPrimary + ';margin:0 0 8px 0;font-size:11px">' + escapeHtml(team.title || team.name) + '</h3>';
  html += '<div style="font-size:8px;color:' + C.textDim + ';margin-bottom:8px">' + escapeHtml(team.description || '') + '</div>';
  html += '<div style="font-size:8px;color:' + C.textDim + ';letter-spacing:1px;margin-bottom:4px">AGENTS IN TEAM</div>';
  var agents = team.agents || [];
  for (var i = 0; i < agents.length; i++) {
    var ag = agents[i];
    html += '<div style="font-size:9px;color:#aaeeff;padding:2px 0">' + escapeHtml(ag.name || ag) + ' <span style="color:' + C.textMuted + '">(' + escapeHtml(ag.type || '?') + ')</span></div>';
  }
  html += '<div style="margin-top:8px;display:flex;gap:6px">';
  html += '<button class="agent-delete-btn" onclick="doDeleteTeam(\'' + escapeHtml(team.name) + '\')">DELETE TEAM</button>';
  html += '<button class="agent-cancel-btn" onclick="document.getElementById(\'agentFormArea\').classList.add(\'hidden\')">CLOSE</button>';
  html += '</div></div>';

  form.innerHTML = html;
}

function doDeleteTeam(name) {
  if (!confirm('Delete team ' + name + '?')) return;
  deleteTeam(name).then(function() {
    document.getElementById('agentFormArea').classList.add('hidden');
    loadTeamsData();
  });
}

function showCreateTeamForm() {
  // Load all agents to build checkboxes
  fetch('/api/agents')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var allAgents = data.agents || [];
      var form = document.getElementById('agentFormArea');
      form.classList.remove('hidden');

      var html = '<div class="agent-edit-form">';
      html += '<h3 style="color:' + C.textPrimary + ';margin:0 0 8px 0;font-size:11px">CREATE TEAM PRESET</h3>';
      html += '<label>Name:</label>';
      html += '<input id="newTeamName" class="agent-input" placeholder="e.g. web_development">';
      html += '<label>Title:</label>';
      html += '<input id="newTeamTitle" class="agent-input" placeholder="e.g. Web Development Team">';
      html += '<label>Description:</label>';
      html += '<textarea id="newTeamDesc" class="agent-input" rows="2" placeholder="Team purpose..."></textarea>';
      html += '<label>Select Agents:</label>';
      html += '<div id="teamAgentCheckboxes" style="max-height:150px;overflow-y:auto;border:1px solid rgba(100,200,255,0.1);border-radius:4px;padding:4px">';

      var types = ['sub_agents', 'managers', 'reviewers'];
      for (var ti = 0; ti < types.length; ti++) {
        var typeAgents = allAgents.filter(function(a) { return a._type === types[ti] && !a.retired; });
        if (typeAgents.length === 0) continue;
        html += '<div style="font-size:7px;color:' + C.textDim + ';letter-spacing:1px;margin:4px 0 2px">' + types[ti].toUpperCase().replace('_', ' ') + '</div>';
        for (var ai = 0; ai < typeAgents.length; ai++) {
          var ag = typeAgents[ai];
          html += '<label style="display:block;font-size:8px;color:#aaeeff;padding:1px 0;cursor:pointer">';
          html += '<input type="checkbox" class="team-agent-cb" data-type="' + types[ti] + '" data-name="' + escapeHtml(ag.name) + '"> ';
          html += escapeHtml(ag.title || ag.name);
          html += '</label>';
        }
      }
      html += '</div>';

      html += '<div style="margin-top:8px;display:flex;gap:6px">';
      html += '<button class="agent-save-btn" onclick="createNewTeam()">CREATE</button>';
      html += '<button class="agent-cancel-btn" onclick="document.getElementById(\'agentFormArea\').classList.add(\'hidden\')">CANCEL</button>';
      html += '</div></div>';

      form.innerHTML = html;
    });
}

function createNewTeam() {
  var name = _slugify(document.getElementById('newTeamName').value);
  var title = document.getElementById('newTeamTitle').value.trim();
  var desc = document.getElementById('newTeamDesc').value.trim();
  var checkboxes = document.querySelectorAll('.team-agent-cb:checked');
  var agents = [];
  for (var i = 0; i < checkboxes.length; i++) {
    agents.push({ type: checkboxes[i].dataset.type, name: checkboxes[i].dataset.name });
  }
  if (!name) { alert('Team name is required'); return; }
  createTeam({ name: name, title: title || name, description: desc, agents: agents }).then(function(result) {
    if (result && result.ok) {
      document.getElementById('agentFormArea').classList.add('hidden');
      loadTeamsData();
    } else {
      alert('Error creating team');
    }
  });
}

// ── Import/Export ──────────────────────────────────────────────────────────

function exportAgents() {
  exportAgentsData().then(function(data) {
    if (!data) { alert('Export failed'); return; }
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'nort_agents_export.json';
    a.click();
    URL.revokeObjectURL(url);
  });
}

function showImportAgents() {
  var area = document.getElementById('importArea');
  area.classList.toggle('hidden');
  document.getElementById('importJson').value = '';
  document.getElementById('importResult').textContent = '';
}

function importAgents() {
  var jsonStr = document.getElementById('importJson').value.trim();
  var overwrite = document.getElementById('importOverwrite').checked;
  var resultEl = document.getElementById('importResult');
  if (!jsonStr) { resultEl.textContent = 'Paste JSON data first'; resultEl.style.color = '#ff5566'; return; }
  var data;
  try { data = JSON.parse(jsonStr); } catch (e) { resultEl.textContent = 'Invalid JSON'; resultEl.style.color = '#ff5566'; return; }
  importAgentsData(data, overwrite).then(function(result) {
    if (result && result.ok) {
      var s = result.summary;
      resultEl.style.color = 'rgba(102,255,170,0.8)';
      resultEl.textContent = 'Created: ' + (s.created || []).length + ', Skipped: ' + (s.skipped || []).length + ', Overwritten: ' + (s.overwritten || []).length;
      loadAgentsData();
    } else {
      resultEl.style.color = '#ff5566';
      resultEl.textContent = 'Import failed: ' + (result && result.detail ? result.detail : 'unknown');
    }
  });
}

// ── Review Tolerance Panel ────────────────────────────────────────────────

function _tolerancePreset(val) {
  if (val >= 8) return { label: 'LENIENT', color: '#00ff88' };
  if (val >= 5) return { label: 'NORMAL',  color: '#ffaa00' };
  return { label: 'STRICT', color: '#ff4444' };
}

function renderToleranceConfig(data) {
  var overlay = document.getElementById('toleranceConfigOverlay');
  if (overlay) overlay.classList.remove('hidden');

  var globalSlider = document.getElementById('globalToleranceSlider');
  var globalLabel = document.getElementById('globalToleranceValue');
  var globalPreset = document.getElementById('globalTolerancePreset');
  var defTol = data.default_tolerance || 6;
  if (globalSlider) globalSlider.value = defTol;
  if (globalLabel) globalLabel.textContent = defTol;
  if (globalPreset) {
    var gp = _tolerancePreset(defTol);
    globalPreset.textContent = gp.label;
    globalPreset.style.color = gp.color;
  }

  // Highlight active preset
  if (data.active_preset) {
    highlightActivePreset(data.active_preset);
  } else {
    highlightActivePreset('');
  }

  var body = document.getElementById('toleranceConfigBody');
  if (!body) return;

  var agents = data.agents || [];
  body.innerHTML = agents.map(function(a) {
    var eff = a.effective || 6;
    var p = _tolerancePreset(eff);
    var isCustom = a.tolerance !== null && a.tolerance !== undefined;
    return '<div class="tolerance-agent-row">' +
      '<span class="tolerance-agent-name">' + escapeHtml(a.name) +
        (isCustom ? '' : ' <span style="color:rgba(150,180,200,0.4);font-size:8px">(default)</span>') +
      '</span>' +
      '<input type="range" class="tolerance-slider" min="1" max="10" value="' + eff + '"' +
        ' data-agent="' + escapeHtml(a.name) + '"' +
        ' oninput="updateAgentToleranceLabel(this)"' +
        ' onchange="saveToleranceAgent(\'' + escapeHtml(a.name) + '\', this.value)">' +
      '<span class="tolerance-value" data-agent-label="' + escapeHtml(a.name) + '">' + eff + '</span>' +
      '<span class="tolerance-preset" data-agent-preset="' + escapeHtml(a.name) + '" style="color:' + p.color + '">' + p.label + '</span>' +
      '</div>';
  }).join('');
}

function updateToleranceLabel(el) {
  var label = document.getElementById('globalToleranceValue');
  var preset = document.getElementById('globalTolerancePreset');
  if (label) label.textContent = el.value;
  if (preset) {
    var p = _tolerancePreset(parseInt(el.value));
    preset.textContent = p.label;
    preset.style.color = p.color;
  }
}

function updateAgentToleranceLabel(el) {
  var name = el.dataset.agent;
  var label = document.querySelector('[data-agent-label="' + name + '"]');
  var preset = document.querySelector('[data-agent-preset="' + name + '"]');
  if (label) label.textContent = el.value;
  if (preset) {
    var p = _tolerancePreset(parseInt(el.value));
    preset.textContent = p.label;
    preset.style.color = p.color;
  }
}

// ── Output Browser ──────────────────────────────────────────────────────────

var _outputPlanId = null;
var _outputSource = 'artifacts'; // 'artifacts' or 'output'

function _updateSourceToggle() {
  var togArt = document.getElementById('srcArtifacts');
  var togOut = document.getElementById('srcOutput');
  if (togArt) togArt.classList.toggle('active', _outputSource === 'artifacts');
  if (togOut) togOut.classList.toggle('active', _outputSource === 'output');
}

function switchOutputSource(source) {
  _outputSource = source;
  _updateSourceToggle();
  if (_outputPlanId) loadOutputTree(_outputPlanId);
}

function showOutputBrowserForPlan(planId) {
  _outputPlanId = planId;
  _outputSource = 'output';
  var overlay = document.getElementById('outputBrowserOverlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');
  _updateSourceToggle();
  loadOutputTree(planId);
}

function _downloadBlob(resp, filename, btn, okLabel, failLabel) {
  if (!resp.ok) {
    if (btn) { btn.textContent = failLabel || '\u2717'; setTimeout(function() { btn.disabled = false; btn.textContent = okLabel || '\u2B07'; }, 2000); }
    return;
  }
  resp.blob().then(function(blob) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    var disp = resp.headers.get('Content-Disposition') || '';
    var match = disp.match(/filename="?([^"]+)"?/);
    a.download = match ? match[1] : filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (btn) { btn.disabled = false; btn.textContent = okLabel || '\u2B07'; }
  });
}

function downloadOutputZip(planId, btnEl) {
  if (btnEl) { btnEl.disabled = true; btnEl.textContent = '...'; }
  fetch('/output/' + planId + '/download')
    .then(function(resp) { return resp.ok ? resp : fetch('/api/artifacts/' + planId + '/download'); })
    .then(function(resp) { _downloadBlob(resp, planId + '_output.zip', btnEl, '\u2B07', '\u2717'); });
}

function showOutputBrowser() {
  var overlay = document.getElementById('outputBrowserOverlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');
  // Determine plan_id from active session or queue
  var planId = _outputPlanId || _activeSessionId || '';
  if (!planId) {
    // Try to get from queue
    var queueItems = document.querySelectorAll('.queue-item[data-id]');
    if (queueItems.length) planId = queueItems[queueItems.length - 1].dataset.id;
  }
  if (planId) loadOutputTree(planId);
}

function loadOutputTree(planId) {
  _outputPlanId = planId;
  var treeEl = document.getElementById('outputFileTree');
  if (treeEl) treeEl.innerHTML = '<div style="color:var(--text-dim);padding:12px">Loading...</div>';
  var endpoint = _outputSource === 'output'
    ? '/api/output/' + planId + '/files'
    : '/api/artifacts/' + planId;
  fetch(endpoint)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      renderFileTree(data.tree || {}, data.files || []);
    })
    .catch(function() {
      document.getElementById('outputFileTree').innerHTML = '<div style="color:var(--text-dim);padding:12px">No artifacts found</div>';
    });
}

var _FILE_ICONS = {
  py: '\u{1F40D}', js: '\u{1F4DC}', html: '\u{1F310}', css: '\u{1F3A8}',
  json: '\u{1F4CB}', md: '\u{1F4DD}', txt: '\u{1F4C4}', yaml: '\u2699',
  yml: '\u2699', sh: '\u{1F4DF}', sql: '\u{1F5C3}', xml: '\u{1F4C3}',
  ts: '\u{1F4DC}', jsx: '\u{1F4DC}', tsx: '\u{1F4DC}', svg: '\u{1F5BC}',
  png: '\u{1F5BC}', jpg: '\u{1F5BC}', gif: '\u{1F5BC}',
};

function _fileIcon(ext) {
  return _FILE_ICONS[ext] || '\u{1F4C4}';
}

function renderFileTree(tree, files) {
  var el = document.getElementById('outputFileTree');
  if (!el) return;
  if (!files.length) {
    el.innerHTML = '<div style="color:var(--text-dim);padding:12px">No artifacts</div>';
    return;
  }
  el.innerHTML = _buildTreeHTML(tree, 0);
}

function _buildTreeHTML(node, depth) {
  var html = '';
  var keys = Object.keys(node).sort(function(a, b) {
    var aIsFile = node[a] && node[a].file;
    var bIsFile = node[b] && node[b].file;
    if (aIsFile !== bIsFile) return aIsFile ? 1 : -1;
    return a.localeCompare(b);
  });
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var val = node[key];
    var indent = depth * 16;
    if (val && val.file) {
      var icon = _fileIcon(val.ext || '');
      var sizeStr = val.size > 1024 ? (val.size / 1024).toFixed(1) + 'K' : val.size + 'B';
      html += '<div class="tree-file" style="padding-left:' + indent + 'px" data-path="' + escapeHtml(val.path) + '" onclick="previewArtifact(\'' + escapeHtml(val.path) + '\')">';
      html += '<span class="tree-icon">' + icon + '</span>';
      html += '<span class="tree-name">' + escapeHtml(key) + '</span>';
      html += '<span class="tree-size">' + sizeStr + '</span>';
      html += '</div>';
    } else {
      html += '<div class="tree-folder" style="padding-left:' + indent + 'px">';
      html += '<span class="tree-icon">\u{1F4C1}</span>';
      html += '<span class="tree-name" style="font-weight:600">' + escapeHtml(key) + '</span>';
      html += '</div>';
      html += _buildTreeHTML(val, depth + 1);
    }
  }
  return html;
}

function previewArtifact(filePath) {
  var parts = filePath.split('/');
  var planId = parts[0];
  var relPath = parts.slice(1).join('/');
  var preview = document.getElementById('outputFilePreview');
  if (!preview) return;
  var treeItems = document.querySelectorAll('.tree-file');
  for (var ti = 0; ti < treeItems.length; ti++) treeItems[ti].classList.remove('active');
  var clicked = document.querySelector('.tree-file[data-path="' + filePath.replace(/"/g, '\\"') + '"]');
  if (clicked) clicked.classList.add('active');
  preview.innerHTML = '<div style="color:var(--text-dim);padding:12px">Loading...</div>';

  var taskId = parts.length > 1 ? parts[1] : '';

  fetch('/api/artifacts/' + planId + '/file?path=' + encodeURIComponent(relPath))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.binary) {
        preview.innerHTML = '<div style="padding:12px;color:var(--text-dim)">Binary file (' + (data.size / 1024).toFixed(1) + ' KB)</div>';
        return;
      }
      var ext = relPath.split('.').pop() || '';
      var highlighted = syntaxHighlight(data.content || '', ext);
      var revHtml = '';
      if (taskId) {
        revHtml = '<div style="margin-bottom:8px"><select id="revisionSelect" class="revision-select" onchange="loadRevisionFile(this.value, \'' + escapeHtml(relPath) + '\')">' +
          '<option value="">Current</option></select> ' +
          '<button class="btn-sm" onclick="loadRevisions(\'' + escapeHtml(planId) + '\', \'' + escapeHtml(taskId) + '\', \'' + escapeHtml(relPath) + '\')" style="font-size:8px;padding:2px 6px">LOAD HISTORY</button></div>';
      }
      preview.innerHTML = revHtml +
        '<div class="preview-header">' + escapeHtml(relPath) + ' <span style="color:var(--text-dim)">(' + (data.size / 1024).toFixed(1) + ' KB)</span></div>' +
        '<pre class="syntax-pre">' + highlighted + '</pre>';
    })
    .catch(function() {
      preview.innerHTML = '<div style="color:var(--state-error);padding:12px">Failed to load file</div>';
    });
}

function loadRevisions(planId, taskId, currentPath) {
  fetch('/api/artifacts/' + planId + '/revisions/' + taskId)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var sel = document.getElementById('revisionSelect');
      if (!sel) return;
      sel.innerHTML = '<option value="">Current</option>';
      (data.revisions || []).forEach(function(rev) {
        sel.innerHTML += '<option value="' + rev.revision + '">' + rev.revision + ' (' + rev.file_count + ' files)</option>';
      });
    });
}

function loadRevisionFile(revision, relPath) {
  if (!revision || !_outputPlanId) return;
  var parts = relPath.split('/');
  var taskId = parts[0];
  var filePart = parts.slice(1).join('/');
  var revPath = taskId + '/revisions/' + revision + '/' + filePart;
  previewArtifact(_outputPlanId + '/' + revPath);
}

function syntaxHighlight(code, ext) {
  var escaped = escapeHtml(code);
  var tokens = [];
  function stash(match, style) {
    var idx = tokens.length;
    tokens.push('<span style="' + style + '">' + match + '</span>');
    return '\x00T' + idx + 'T\x00';
  }
  // 1. Comments first
  if (ext === 'py') {
    escaped = escaped.replace(/(#[^\n]*)/g, function(m) { return stash(m, 'color:var(--text-dim)'); });
  } else if (ext === 'js' || ext === 'ts' || ext === 'jsx' || ext === 'tsx' || ext === 'css') {
    escaped = escaped.replace(/(\/\/[^\n]*)/g, function(m) { return stash(m, 'color:var(--text-dim)'); });
  }
  // 2. Strings
  if (ext !== 'html' && ext !== 'json') {
    escaped = escaped.replace(/(&#39;[^&#]*&#39;|&quot;[^&]*&quot;)/g, function(m) { return stash(m, 'color:#66ffaa'); });
  }
  // 3. Language-specific
  var keywords = [];
  if (ext === 'py') {
    keywords = ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'else:', 'elif ',
                'for ', 'while ', 'try:', 'except ', 'finally:', 'with ', 'as ', 'raise ',
                'yield ', 'async ', 'await ', 'True', 'False', 'None', 'self'];
  } else if (ext === 'js' || ext === 'ts' || ext === 'jsx' || ext === 'tsx') {
    keywords = ['function ', 'const ', 'let ', 'var ', 'return ', 'if ', 'else ',
                'for ', 'while ', 'class ', 'import ', 'export ', 'from ', 'async ',
                'await ', 'new ', 'this', 'true', 'false', 'null', 'undefined'];
  } else if (ext === 'html') {
    escaped = escaped.replace(/(&lt;\/?[a-zA-Z][a-zA-Z0-9]*)/g, function(m) { return stash(m, 'color:#cc88ff'); });
    escaped = escaped.replace(/(&gt;)/g, function(m) { return stash(m, 'color:#cc88ff'); });
  } else if (ext === 'css') {
    escaped = escaped.replace(/([\w-]+)\s*:/g, function(m, p1) { return stash(p1, 'color:#66ccff') + ':'; });
  } else if (ext === 'json') {
    escaped = escaped.replace(/(&quot;[^&]*&quot;)\s*:/g, function(m, p1) { return stash(p1, 'color:#66ccff') + ':'; });
    escaped = escaped.replace(/:\s*(&quot;[^&]*&quot;)/g, function(m, p1) { return ': ' + stash(p1, 'color:#66ffaa'); });
  }
  // 4. Keywords
  for (var i = 0; i < keywords.length; i++) {
    var kw = keywords[i];
    var re = new RegExp('\\b' + kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
    escaped = escaped.replace(re, function(m) { return stash(m, 'color:#cc88ff'); });
  }
  // Restore tokens
  escaped = escaped.replace(/\x00T(\d+)T\x00/g, function(m, idx) { return tokens[parseInt(idx)]; });
  return escaped;
}

function downloadArtifacts() {
  var planId = _outputPlanId || _activeSessionId || '';
  if (!planId) return;
  var btn = document.getElementById('outputDownloadBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'PREPARING...'; }
  fetch('/api/artifacts/' + planId + '/download')
    .then(function(resp) { _downloadBlob(resp, planId + '_artifacts.zip', btn, 'DOWNLOAD ZIP', 'NO ARTIFACTS'); })
    .catch(function() { if (btn) { btn.textContent = 'DOWNLOAD FAILED'; setTimeout(function() { btn.disabled = false; btn.textContent = 'DOWNLOAD ZIP'; }, 2000); } });
}

// ── Review Analytics Panel ──────────────────────────────────────────────

function renderReviewAnalytics(data) {
  var overlay = document.getElementById('reviewAnalyticsOverlay');
  if (overlay) overlay.classList.remove('hidden');

  var body = document.getElementById('reviewAnalyticsBody');
  if (!body) return;

  var reviewers = data.by_reviewer || [];
  if (reviewers.length === 0) {
    body.innerHTML = '<div style="color:var(--text-dim);font-size:11px;text-align:center;padding:20px">No review data yet</div>';
    return;
  }

  body.innerHTML = reviewers.map(function(r) {
    var passRate = r.total_reviews > 0 ? ((r.passes / r.total_reviews) * 100).toFixed(0) : 0;
    var failRate = r.total_reviews > 0 ? ((r.failures / r.total_reviews) * 100).toFixed(0) : 0;
    var avgScore = (r.avg_score || 0).toFixed(1);
    var overrides = r.override_count || 0;

    return '<div class="review-analytics-row">' +
      '<div class="review-analytics-name">' + escapeHtml(r.reviewer) + '</div>' +
      '<div class="review-analytics-stats">' +
        '<span class="review-stat pass">' + passRate + '% pass</span>' +
        '<span class="review-stat fail">' + failRate + '% fail</span>' +
        '<span class="review-stat avg">avg ' + avgScore + '</span>' +
        (overrides > 0 ? '<span class="review-stat override">' + overrides + ' overrides</span>' : '') +
      '</div>' +
      '<div class="review-analytics-bar-container">' +
        '<div class="review-analytics-bar pass-bar" style="width:' + passRate + '%"></div>' +
        '<div class="review-analytics-bar fail-bar" style="width:' + failRate + '%"></div>' +
      '</div>' +
    '</div>';
  }).join('');
}

// ── Tolerance Presets ────────────────────────────────────────────────

async function applyTolerancePreset(presetName) {
  try {
    await fetch('/api/tolerance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: presetName }),
    });
    loadTolerance();
  } catch (e) {
    console.error('applyTolerancePreset:', e);
  }
}

function highlightActivePreset(presetName) {
  var buttons = document.querySelectorAll('.tolerance-preset-btn');
  buttons.forEach(function(btn) {
    if (btn.dataset.preset === presetName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}
