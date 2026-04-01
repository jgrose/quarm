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

// ── Agent Detail Card ───────────────────────────────────────────────────────

var selectedNode = null;

function showAgentDetail(node) {
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

// ── Agent List Panel ────────────────────────────────────────────────────────

function toggleAgentList() {
  var panel = document.getElementById('agentListPanel');
  var btn = document.getElementById('agentListToggle');
  if (!panel) return;
  panel.classList.toggle('collapsed');
  if (btn) btn.innerHTML = panel.classList.contains('collapsed') ? '&#x25B6;' : '&#x25C0;';
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

    // Build categorized agent lists
    var managers = d.managers || [];
    var subAgents = d.sub_agents || [];
    var reviewers = d.reviewers || [];
    var tasks = d.tasks || [];

    // Managers (sentinels)
    for (var i = 0; i < managers.length; i++) {
      html += _renderAgentRow(managers[i], 'sentinel', tasks);
    }
    // Sub-agents (drones)
    for (var i = 0; i < subAgents.length; i++) {
      html += _renderAgentRow(subAgents[i], 'drone', tasks);
    }
    // Reviewers (probes)
    for (var i = 0; i < reviewers.length; i++) {
      html += _renderAgentRow(reviewers[i], 'probe', tasks);
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
  var name = agent.title || agent.name.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });

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
  var panel = document.getElementById('eventLog');
  var btn = document.getElementById('chatToggle');
  if (!panel) return;
  panel.classList.toggle('collapsed');
  if (btn) btn.innerHTML = panel.classList.contains('collapsed') ? '&#x25B6;' : '&#x25C0;';
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
    if (all[i].name === name) return all[i].title || name.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
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
  var prevAgent = null;

  for (var i = 0; i < groups.length; i++) {
    var g = groups[i];

    // System / master: centered text
    if (g.type === 'system' || g.type === 'master') {
      var sysCls = g.type === 'master' ? 'chat-system master' : 'chat-system';
      html += '<div class="' + sysCls + '">' + escapeHtml(g.messages.join(' ')) + '</div>';
      prevAgent = null;
      continue;
    }

    // Agent message bubble
    var agentKey = g.agent;
    var color = getChatColor(agentKey);
    var tier = getAgentTier(agentKey);
    var tierInfo = tier ? TIERS[tier] : null;
    var icon = tierInfo ? tierInfo.icon : '\u25CF';
    var displayName = getAgentTitle(agentKey);
    var isCont = (prevAgent === agentKey);

    if (isCont) {
      html += '<div class="chat-group continuation">';
    } else {
      html += '<div class="chat-group">';
      html += '<div class="chat-avatar" style="border-color:' + color +
              ';background:' + hexToRgba(color, 0.12) + ';color:' + color + '">' +
              icon + '</div>';
    }

    html += '<div class="chat-body">';

    if (!isCont) {
      html += '<div class="chat-name" style="color:' + color + '">' +
              escapeHtml(displayName) + '</div>';
    }

    // Bubble(s)
    for (var j = 0; j < g.messages.length; j++) {
      var msg = g.messages[j];
      var verdictMatch = msg.match(/^(\S+):\s*(PASS|FAIL|FLAG)\s*\((\d+)\/10\)/);

      html += '<div class="chat-bubble" style="border-left-color:' + color +
              ';background:' + hexToRgba(color, 0.06) + '">';
      if (verdictMatch) {
        var vCls = verdictMatch[2].toLowerCase();
        html += '<span class="chat-verdict ' + vCls + '">' + verdictMatch[2] +
                '</span> ' + escapeHtml(verdictMatch[1]) +
                ' <span style="color:' + C.textDim + '">(' + verdictMatch[3] + '/10)</span>';
        // Render remaining text after the verdict prefix
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
        html += '<div class="chat-issue">' + escapeHtml(meta.content) + '</div>';
      } else {
        html += '<div class="chat-meta">' + escapeHtml((meta.agent ? '[' + meta.agent + '] ' : '') + meta.content) + '</div>';
      }
    }

    html += '</div></div>'; // .chat-body, .chat-group
    prevAgent = agentKey;
  }

  return html;
}

function renderEventLog(lines, project, done, total) {
  var progressEl = document.getElementById('eventLogProgress');
  var bodyEl = document.getElementById('eventLogBody');
  if (!bodyEl) return;

  // Progress bar (pinned header)
  var pct = total ? Math.round(done / total * 100) : 0;
  var bar = '\u2588'.repeat(Math.floor(pct / 5)) + '\u2591'.repeat(20 - Math.floor(pct / 5));
  if (progressEl) {
    progressEl.innerHTML = '<div class="chat-progress">' +
      escapeHtml(project) + ' ' + bar + ' ' + pct + '% ' + done + '/' + total + '</div>';
  }

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
  document.querySelectorAll('.cfg-toggle').forEach(function(el) {
    var key = el.dataset.key;
    if (key && key in config) {
      if (config[key]) el.classList.add('on');
      else el.classList.remove('on');
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

function switchLedgerTab(tab, btn) {
  document.querySelectorAll('.ledger-tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('.ledger-section').forEach(function(s) { s.classList.remove('active'); });
  if (btn) btn.classList.add('active');
  var section = document.getElementById('ledger-' + tab);
  if (section) section.classList.add('active');
}

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

function renderModelConfig(data) {
  var overlay = document.getElementById('modelConfigOverlay');
  if (overlay) overlay.classList.remove('hidden');
  var body = document.getElementById('modelConfigBody');
  if (!body) return;

  _modelConfigData = data.models || [];

  body.innerHTML = _modelConfigData.map(function(m) {
    var tierClass = 'tier-' + (m.tier || 'mid');
    return '<div class="config-row">' +
      '<span class="config-label">' + escapeHtml(m.id) + ' <span style="color:' + C.textMuted + '">[' + (m.tier || '?') + ']</span></span>' +
      '<div class="cfg-toggle ' + (m.enabled ? 'on' : '') + '" data-model="' + escapeHtml(m.id) + '" onclick="toggleModel(this)"></div>' +
      '</div>';
  }).join('');
}

function toggleModel(el) {
  el.classList.toggle('on');
  var allToggles = document.querySelectorAll('#modelConfigBody .cfg-toggle');
  var enabledToggles = document.querySelectorAll('#modelConfigBody .cfg-toggle.on');
  var allEnabled = allToggles.length === enabledToggles.length;
  var enabledModels = allEnabled ? null : Array.from(enabledToggles).map(function(e) { return e.dataset.model; });
  saveModels(enabledModels);
}

// ── Utility ─────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Agent Registry Panel ───────────────────────────────────────────────────

var _agentsActiveTab = 'sub_agents';
var _agentsData = null;
var _teamsData = null;
var _showRetired = false;

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

  var html = '';
  if (filtered.length === 0) {
    html = '<div style="padding:12px;color:' + C.textMuted + ';font-size:9px">No agents in this category</div>';
  }

  for (var i = 0; i < filtered.length; i++) {
    var a = filtered[i];
    var scoreColor = (a.avg_score || 0) >= 7 ? '#66ffaa' : (a.avg_score || 0) >= 4 ? '#ffbb44' : '#ff5566';
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
  var tags = document.getElementById('agentEditTags').value.split(',').map(function(t) { return t.trim(); }).filter(Boolean);
  var updates = { description: desc, tags: tags };

  var toolsEl = document.getElementById('agentEditTools');
  if (toolsEl) {
    updates.tools = toolsEl.value.split(',').map(function(t) { return t.trim(); }).filter(Boolean);
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
    name: document.getElementById('newAgentName').value.trim().toLowerCase().replace(/\s+/g, '_'),
    title: document.getElementById('newAgentTitle').value.trim(),
    description: document.getElementById('newAgentDesc').value.trim(),
    tags: document.getElementById('newAgentTags').value.split(',').map(function(t) { return t.trim(); }).filter(Boolean),
    tools: document.getElementById('newAgentTools').value.split(',').map(function(t) { return t.trim(); }).filter(Boolean),
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

function loadTeamsData() {
  fetchTeams().then(function(teams) {
    _teamsData = teams;
    renderTeamsList();
  });
}

function renderTeamsList() {
  var list = document.getElementById('agentsList');
  if (!_teamsData || _teamsData.length === 0) {
    list.innerHTML = '<div style="padding:12px;color:' + C.textMuted + ';font-size:9px">No team presets defined</div>';
    return;
  }

  var html = '';
  for (var i = 0; i < _teamsData.length; i++) {
    var team = _teamsData[i];
    var agentCount = (team.agents || []).length;
    html += '<div class="agent-item" onclick="showTeamDetail(\'' + escapeHtml(team.name) + '\')">';
    html += '<div class="agent-item-header">';
    html += '<span class="agent-item-name">' + escapeHtml(team.title || team.name) + '</span>';
    html += '<span class="agent-item-score" style="color:' + C.textMuted + '">' + agentCount + ' agents</span>';
    html += '</div>';
    html += '<div class="agent-item-desc">' + escapeHtml((team.description || '').slice(0, 80)) + '</div>';
    if (team.agents && team.agents.length > 0) {
      html += '<div class="agent-item-tags">';
      for (var a = 0; a < Math.min(team.agents.length, 6); a++) {
        html += '<span class="agent-tag">' + escapeHtml(team.agents[a].name || team.agents[a]) + '</span>';
      }
      if (team.agents.length > 6) html += '<span class="agent-tag">+' + (team.agents.length - 6) + '</span>';
      html += '</div>';
    }
    html += '</div>';
  }
  list.innerHTML = html;
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
  var name = document.getElementById('newTeamName').value.trim().toLowerCase().replace(/\s+/g, '_');
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
    var presetBtns = document.querySelectorAll('.tolerance-preset-btn');
    presetBtns.forEach(function(btn) { btn.classList.remove('active'); });
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
  fetch('/api/artifacts/' + planId)
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
      html += '<div class="tree-file" style="padding-left:' + indent + 'px" onclick="previewArtifact(\'' + escapeHtml(val.path) + '\')">';
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
  preview.innerHTML = '<div style="color:var(--text-dim);padding:12px">Loading...</div>';

  // Load revision history for the task
  var taskId = parts.length > 1 ? parts[1] : '';
  var revDropdown = '';

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
    keywords = [];
    // Highlight tags
    escaped = escaped.replace(/(&lt;\/?[a-zA-Z][a-zA-Z0-9]*)/g, '<span style="color:#cc88ff">$1</span>');
    escaped = escaped.replace(/(&gt;)/g, '<span style="color:#cc88ff">$1</span>');
  } else if (ext === 'css') {
    keywords = [];
    escaped = escaped.replace(/([\w-]+)\s*:/g, '<span style="color:#66ccff">$1</span>:');
  } else if (ext === 'json') {
    escaped = escaped.replace(/(&quot;[^&]*&quot;)\s*:/g, '<span style="color:#66ccff">$1</span>:');
    escaped = escaped.replace(/:\s*(&quot;[^&]*&quot;)/g, ': <span style="color:#66ffaa">$1</span>');
  }
  // Apply keyword highlighting
  for (var i = 0; i < keywords.length; i++) {
    var kw = keywords[i];
    var re = new RegExp('\\b' + kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
    escaped = escaped.replace(re, '<span style="color:#cc88ff">' + kw + '</span>');
  }
  // Highlight strings
  if (ext !== 'html' && ext !== 'json') {
    escaped = escaped.replace(/(&#39;[^&#]*&#39;|&quot;[^&]*&quot;)/g, '<span style="color:#66ffaa">$1</span>');
  }
  // Highlight comments
  if (ext === 'py') {
    escaped = escaped.replace(/(#[^\n]*)/g, '<span style="color:var(--text-dim)">$1</span>');
  } else if (ext === 'js' || ext === 'ts' || ext === 'css') {
    escaped = escaped.replace(/(\/\/[^\n]*)/g, '<span style="color:var(--text-dim)">$1</span>');
  }
  return escaped;
}

function downloadArtifacts() {
  var planId = _outputPlanId || _activeSessionId || '';
  if (!planId) return;
  window.location.href = '/api/artifacts/' + planId + '/download';
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

  var maxTotal = Math.max.apply(null, reviewers.map(function(r) { return r.total_reviews; }));

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
    highlightActivePreset(presetName);
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
