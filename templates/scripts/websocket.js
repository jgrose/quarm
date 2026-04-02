// ═══ NORT WEBSOCKET & STATE MANAGEMENT ═══

var WS_URL = 'ws://' + location.host + '/ws';
var ws = null;
var wsConnected = false;
var agentNodeMap = {};
var liveStartedAt = null;

// ── Multi-session tracking ──────────────────────────────────────────────────
var _sessions = {};           // session_id → { data }
var _activeSessionId = null;  // which session drives the canvas
var _chatFilterId = 'all';        // chat filter: 'all' or a session_id
var _chatUnread = {};             // session_id → true when unseen messages arrive
var _chatLastSeenCount = {};      // session_id → last known log line count

// ── Reconnect backoff ───────────────────────────────────────────────────────
var _reconnectAttempts = 0;
var _reconnectDelay = 1000;       // base delay ms
var _maxReconnectDelay = 30000;   // cap at 30s

// ── Heartbeat ping/pong ─────────────────────────────────────────────────────
var _lastPong = Date.now();
var _pingInterval = null;

// ── Connection status ────────────────────────────────────────────────────────
var _connectionStatus = 'disconnected'; // 'connected' | 'connecting' | 'disconnected'

function getConnectionStatus() {
  return { status: _connectionStatus, attempts: _reconnectAttempts };
}

function connectWS() {
  _connectionStatus = 'connecting';
  ws = new WebSocket(WS_URL);
  ws.onopen = function() {
    wsConnected = true;
    _connectionStatus = 'connected';
    _reconnectAttempts = 0;
    _isReplay = true;  // suppress spawn effects for replayed state
    updateConnectionStatus(true);
    // Clear replay flag after a short delay (server sends replay immediately)
    setTimeout(function() { _isReplay = false; }, 500);

    // Start heartbeat ping interval
    _lastPong = Date.now();
    if (_pingInterval) clearInterval(_pingInterval);
    _pingInterval = setInterval(function() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      // Check for stale connection (no pong in 10s)
      if (Date.now() - _lastPong > 10000) {
        console.warn('[WS] No pong received in 10s, reconnecting');
        ws.close();
        return;
      }
      ws.send(JSON.stringify({ type: 'ping' }));
    }, 30000);

    // Restore persisted city state after server replay
    _restorePersistedState();
  };
  ws.onclose = function() {
    wsConnected = false;
    _connectionStatus = 'disconnected';
    updateConnectionStatus(false);

    // Clear heartbeat interval
    if (_pingInterval) { clearInterval(_pingInterval); _pingInterval = null; }

    // Exponential backoff with jitter
    var delay = Math.min(_reconnectDelay * Math.pow(2, _reconnectAttempts), _maxReconnectDelay);
    delay += Math.random() * 1000;
    _reconnectAttempts++;
    console.warn('[WS] Connection closed. Reconnecting in ' + Math.round(delay) + 'ms (attempt ' + _reconnectAttempts + ')');
    setTimeout(connectWS, delay);
  };
  ws.onerror = function(e) {
    console.error('[WS] Error:', e);
    ws.close();
  };
  ws.onmessage = function(evt) {
    var rawData = evt.data;
    try {
      var data = JSON.parse(rawData);
      // Handle pong messages for heartbeat
      if (data.type === 'pong') {
        _lastPong = Date.now();
        return;
      }
      handleMessage(data);
    } catch(e) { console.warn('[WS] Parse error:', e.message, rawData.substring(0, 200)); }
  };
}

// ── Persisted State Recovery ────────────────────────────────────────────────

function _restorePersistedState() {
  try {
    var raw = localStorage.getItem('nort_city_state');
    if (!raw) return;
    var saved = JSON.parse(raw);
    // Check expiry (24 hours)
    if (saved.savedAt) {
      var age = Date.now() - new Date(saved.savedAt).getTime();
      if (age > 24 * 60 * 60 * 1000) {
        localStorage.removeItem('nort_city_state');
        return;
      }
    }
    // Restore visual state after a delay to let server replay populate nodes
    setTimeout(function() {
      if (saved.nodes && typeof deserializeCityState === 'function') {
        deserializeCityState(saved);
      }
      if (saved.buildings && typeof deserializeBuildingState === 'function') {
        deserializeBuildingState(saved);
      }
    }, 600);
  } catch(e) {
    console.warn('[WS] Failed to restore persisted state:', e.message);
  }
}

// ── Message Router ──────────────────────────────────────────────────────────

function handleMessage(data) {
  if (data.type === 'queue') {
    renderQueue(data.plans || []);
    return;
  }
  if (data.type === 'plan_event') {
    handlePlanEvent(data);
    return;
  }
  if (data.type === 'approval_request') {
    showApproval(data);
    return;
  }
  if (data.type === 'approval_resolved') {
    hideApproval();
    return;
  }
  // Default: orchestrator status update
  applyStatus(data);
}

// ── Core Status Application ─────────────────────────────────────────────────

function applyStatus(data) {
  // ── Multi-session: store data and gate canvas updates ──
  var sid = data.session_id || 'default';
  _sessions[sid] = _sessions[sid] || {};
  _sessions[sid].data = data;

  // Auto-select first session
  if (!_activeSessionId) {
    _activeSessionId = sid;
  }

  // Track unread chat: mark if this session isn't the current chat view
  if (data.log) {
    var prevCount = _chatLastSeenCount[sid] || 0;
    if (data.log.length > prevCount) {
      if (_chatFilterId !== 'all' && _chatFilterId !== sid) {
        _chatUnread[sid] = true;
      }
      _chatLastSeenCount[sid] = data.log.length;
    }
  }

  // Always update the agent list panel (shows all sessions)
  if (typeof renderAgentList === 'function') renderAgentList();

  // Always update chat tabs and filtered chat (decoupled from canvas)
  if (typeof renderChatTabs === 'function') renderChatTabs();
  if (data.log && (_chatFilterId === 'all' || _chatFilterId === sid)) {
    if (typeof _renderFilteredChat === 'function') _renderFilteredChat();
  }

  // Only update canvas/effects for the active session
  if (sid !== _activeSessionId) return;

  // 1. Build node graph from rosters (first time or on roster change)
  if (data.sub_agents || data.managers || data.reviewers) {
    rebuildNodes(data);
    if (typeof syncProgramsToRoster === 'function') syncProgramsToRoster(data);
    if (typeof updateChatRosters === 'function') updateChatRosters(data);
  }

  // Track session start
  if (!liveStartedAt && data.updated_at) liveStartedAt = data.updated_at;

  // 2. Update node states from tasks
  var tasks = data.tasks || [];
  for (var i = 0; i < tasks.length; i++) {
    var task = tasks[i];
    var node = getNodeByAgent(task.agent);
    if (!node) continue;

    var prevState = node.state;
    node.state = task.status;
    node.taskId = task.id;
    node.taskTitle = task.title;
    node.model = task.current_model || '';
    node.tokens = task.task_tokens || 0;
    node.toolCalls = (task.status === 'done' || task.status === 'failed') ? [] : (task.tool_calls || []);
    node.resultPreview = task.result_preview || '';
    node.revisionCount = task.revision_count || 0;
    node.lastScore = task.last_score || 0;
    node.dependsOn = task.depends_on || [];

    // Spawn effects, audio, and bubbles on state transitions
    if (prevState !== task.status && !_isReplay) {
      // Find program matching this agent and force re-route
      for (var pi = 0; pi < ambientPrograms.length; pi++) {
        var prog = ambientPrograms[pi];
        if (prog.agentName === task.agent) {
          prog.assignedTask = { id: task.id, status: task.status, title: task.title };
          if (prog.bunkerState === 'inside') {
            prog.bunkerState = 'exiting';
            prog.exitProgress = 0;
            prog.visible = true;
          } else if (prog.bunkerState === 'walking' || prog.bunkerState === 'leaving_door') {
            _releaseLocation(prog);
            _pickTarget(prog, 0, 0);
          }
          break;
        }
      }
      if (task.status === 'in_progress' && prevState === 'pending') {
        spawnEffect(node.x, node.y, getStateColor('in_progress'));
        if (nodes.has('nexus')) spawnParticle('nexus', node.id, C.dispatch, task.title);
        if (config.sound && typeof playAgentSpawn === 'function') playAgentSpawn();
        if (typeof addBubble === 'function') addBubble('nexus', 'nexus', 'Dispatching: ' + (task.title || task.id));
      }
      if (task.status === 'done') {
        spawnCompleteEffect(node.x, node.y, C.done);
        if (nodes.has('nexus')) spawnParticle(node.id, 'nexus', C.returnEdge, '\u2713');
        if (config.sound && typeof playAgentComplete === 'function') playAgentComplete();
        if (typeof addBubble === 'function') addBubble(node.id, 'drone', 'Task complete');
      }
      if (task.status === 'in_manager_review') {
        if (config.sound && typeof playToolStart === 'function') playToolStart();
      }
      if (task.status === 'in_specialist_review') {
        if (config.sound && typeof playToolStart === 'function') playToolStart();
      }
      if (task.status === 'failed') {
        effects.push({ type: 'error', x: node.x, y: node.y, color: C.failed, age: 0, duration: 0.6 });
        if (config.sound && typeof playError === 'function') playError();
      }
    }

    // Add review bubbles
    if (!_isReplay && prevState !== task.status) {
      if (task.status === 'in_manager_review' && task.result_preview) {
        if (typeof addBubble === 'function') addBubble(node.id, 'drone', (task.result_preview || '').slice(0, 120));
      }
    }

    // Feature 4+5: On task completion, increment building completion counter
    if (prevState !== task.status && task.status === 'done' && !_isReplay) {
      if (typeof incrementBuildingCompletion === 'function') {
        incrementBuildingCompletion('data_vault');
      }
      // Feature 5: Add bubble for ambient programs when no real nodes
      if (typeof addBubble === 'function' && nodes.size === 0) {
        addBubble('nexus', 'nexus', 'Task done: ' + (task.title || task.id));
      }
    }
  }

  // 2b. Route ambient programs to work locations
  if (typeof routeProgramsToTasks === 'function') routeProgramsToTasks(tasks);

  // 3. Update edges based on active task flow
  rebuildEdges(data);

  // 4. Update active reviewer
  if (data.active_reviewer) {
    var probeNode = getNodeByAgent(data.active_reviewer);
    if (probeNode) {
      probeNode.state = 'in_specialist_review';
    }
  }
  // Dim non-active reviewers
  for (var rName in agentNodeMap) {
    var nId = agentNodeMap[rName];
    var n = nodes.get(nId);
    if (n && n.tier === 'probe' && rName !== data.active_reviewer) {
      if (n.state === 'in_specialist_review') n.state = 'pending';
    }
  }

  // 5. Update session stats
  updateSessionStats(data.tokens_used || 0, data.results_count || 0, data.total_tasks || 0);

  // 6. Event log rendering is now handled above (chat tab filter)

  // 7. Handle verdict popup + bubble
  if (data.last_verdict && config.completionFx && !_isReplay) {
    showVerdict(data.last_verdict);
    var vNode = getNodeByAgent(data.last_verdict.agent || data.last_verdict.reviewer);
    if (vNode && typeof addBubble === 'function') {
      var vRole = vNode.tier === 'probe' ? 'probe' : 'sentinel';
      addBubble(vNode.id, vRole, data.last_verdict.verdict + ' (' + data.last_verdict.score + '/10)');
    }
  }

  // Store plan ID for output browser
  if (data.session_id && typeof _outputPlanId !== 'undefined') {
    _outputPlanId = data.session_id;
  }

  // 8. Handle synthesis/completion
  if (data.synthesis_report && data.phase === 'done') {
    showCompletion(data);
    if (typeof revertToIdlePrograms === 'function') {
      revertToIdlePrograms();
    }
  }

  // 9. Update heartbeat
  updateHeartbeat(data.phase || 'idle');
}

// ── Session Switching ───────────────────────────────────────────────────────

function switchSession(sessionId) {
  if (sessionId === _activeSessionId) return;
  _activeSessionId = sessionId;

  var sess = _sessions[sessionId];
  if (!sess || !sess.data) return;
  var d = sess.data;

  // Clear and rebuild canvas for this session
  nodes.clear();
  edges.length = 0;
  agentNodeMap = {};
  if (typeof ambientPrograms !== 'undefined') ambientPrograms.length = 0;

  // Suppress effects during switch
  var wasReplay = _isReplay;
  _isReplay = true;

  // Rebuild from session data
  if (d.sub_agents || d.managers || d.reviewers) {
    rebuildNodes(d);
    if (typeof syncProgramsToRoster === 'function') syncProgramsToRoster(d);
    if (typeof updateChatRosters === 'function') updateChatRosters(d);
  }

  // Apply task states
  var tasks = d.tasks || [];
  for (var i = 0; i < tasks.length; i++) {
    var task = tasks[i];
    var node = getNodeByAgent(task.agent);
    if (!node) continue;
    node.state = task.status;
    node.taskId = task.id;
    node.taskTitle = task.title;
    node.model = task.current_model || '';
    node.tokens = task.task_tokens || 0;
    node.toolCalls = (task.status === 'done' || task.status === 'failed') ? [] : (task.tool_calls || []);
    node.resultPreview = task.result_preview || '';
    node.revisionCount = task.revision_count || 0;
    node.lastScore = task.last_score || 0;
    node.dependsOn = task.depends_on || [];
  }

  rebuildEdges(d);

  // Update chat + stats (chat follows its own filter, not canvas session)
  if (typeof _renderFilteredChat === 'function') _renderFilteredChat();
  updateSessionStats(d.tokens_used || 0, d.results_count || 0, d.total_tasks || 0);
  updateHeartbeat(d.phase || 'idle');

  _isReplay = wasReplay;

  // Refresh agent list highlighting
  if (typeof renderAgentList === 'function') renderAgentList();
}

// ── Roster → Node Graph ─────────────────────────────────────────────────────

function rebuildNodes(data) {
  var subAgents = data.sub_agents || [];
  var managers  = data.managers   || [];
  var reviewers = data.reviewers  || [];

  // Always have NEXUS
  if (!nodes.has('nexus')) {
    addNode('nexus', 'master', 'nexus', 'NEXUS', null);
  }

  agentNodeMap = {};

  // SENTINEL nodes from managers
  managers.forEach(function(m, i) {
    var id = 'sentinel_' + i;
    if (!nodes.has(id)) {
      addNode(id, m.name, 'sentinel', m.title || prettify(m.name), 'nexus');
    }
    agentNodeMap[m.name] = id;
  });

  // DRONE nodes from sub-agents
  subAgents.forEach(function(a, i) {
    var id = 'drone_' + i;
    if (!nodes.has(id)) {
      var parentId = findOverseer(a.name, managers) || 'nexus';
      addNode(id, a.name, 'drone', a.title || prettify(a.name), parentId);
    }
    agentNodeMap[a.name] = id;
  });

  // PROBE nodes from reviewers
  reviewers.forEach(function(r, i) {
    var id = 'probe_' + i;
    if (!nodes.has(id)) {
      addNode(id, r.name, 'probe', r.title || prettify(r.name), null);
    }
    agentNodeMap[r.name] = id;
  });
}

function findOverseer(agentName, managers) {
  return managers.length > 0 ? 'sentinel_0' : 'nexus';
}

// ── Edge Rebuilding ─────────────────────────────────────────────────────────

function rebuildEdges(data) {
  edges.length = 0;

  // Always show nexus -> sentinel connections
  for (var entry of nodes) {
    var node = entry[1];
    if (node.tier === 'sentinel') {
      edges.push({
        from: 'nexus',
        to: node.id,
        color: C.dispatch,
        activity: 0.3,
        label: '',
      });
    }
  }

  // Show sentinel -> drone for active tasks
  var tasks = data.tasks || [];
  for (var i = 0; i < tasks.length; i++) {
    var task = tasks[i];
    var droneNode = getNodeByAgent(task.agent);
    if (!droneNode) continue;

    if (task.status === 'in_progress' || task.status === 'revision') {
      if (droneNode.parentId) {
        edges.push({
          from: droneNode.parentId,
          to: droneNode.id,
          color: C.dispatch,
          activity: 0.8,
          label: task.title || '',
        });
      }
    }

    // Show drone -> probe during specialist review
    if (task.status === 'in_specialist_review' && data.active_reviewer) {
      var probeNode = getNodeByAgent(data.active_reviewer);
      if (probeNode && droneNode) {
        edges.push({
          from: droneNode.id,
          to: probeNode.id,
          color: C.review,
          activity: 0.9,
          label: 'REVIEW',
        });
      }
    }
  }
}

// ── Utilities ───────────────────────────────────────────────────────────────

function prettify(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
}

function updateConnectionStatus(connected) {
  var dot = document.querySelector('.status-dot');
  var text = document.getElementById('connectionText');
  if (dot) { if (connected) dot.classList.remove('disconnected'); else dot.classList.add('disconnected'); }
  if (text) text.textContent = connected ? 'CONNECTED' : 'OFFLINE';
}

function updateSessionStats(tokens, done, total) {
  var el = document.getElementById('sessionStats');
  if (el) {
    el.textContent = done + '/' + total + ' TASKS \u00B7 ' + (tokens / 1000).toFixed(0) + 'K TOKENS';
  }
}

function updateHeartbeat(phase) {
  var dot = document.querySelector('.heartbeat-dot');
  var text = document.getElementById('heartbeatText');
  if (dot) {
    if (phase !== 'idle' && phase !== 'done') dot.classList.add('running');
    else dot.classList.remove('running');
  }
  if (text) {
    text.textContent = phase.toUpperCase();
  }
}
