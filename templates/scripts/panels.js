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

// ── Event Log ───────────────────────────────────────────────────────────────

function renderEventLog(lines, project, done, total) {
  var el = document.getElementById('eventLogBody');
  if (!el) return;
  var pct = total ? Math.round(done / total * 100) : 0;
  var bar = '\u2588'.repeat(Math.floor(pct / 5)) + '\u2591'.repeat(20 - Math.floor(pct / 5));

  var rows = lines.slice(-30).map(function(l) {
    var cls = '';
    if (l.includes('[MASTER]')) cls = 'log-master';
    else if (l.includes('PASS') || l.includes('\u2713')) cls = 'log-done';
    else if (l.includes('FLAG') || l.includes('FAIL') || l.includes('revis')) cls = 'log-review';
    else if (l.includes('Dispatch') || l.includes('Execut')) cls = 'log-execute';
    else if (/SECURITY|UX|USER_TEST/i.test(l)) cls = 'log-dispatch';
    return '<div class="log-entry ' + cls + '">' + escapeHtml(l) + '</div>';
  }).join('');

  el.innerHTML = '<div class="log-entry log-master">' + escapeHtml(project) + ' ' +
    bar + ' ' + pct + '% ' + done + '/' + total + '</div>' + rows;
  el.scrollTop = el.scrollHeight;
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
