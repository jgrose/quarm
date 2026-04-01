// ═══ NORT WEBSOCKET & STATE MANAGEMENT ═══

var WS_URL = 'ws://' + location.host + '/ws';
var ws = null;
var wsConnected = false;
var agentNodeMap = {};
var liveStartedAt = null;

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = function() {
    wsConnected = true;
    _isReplay = true;  // suppress spawn effects for replayed state
    updateConnectionStatus(true);
    // Clear replay flag after a short delay (server sends replay immediately)
    setTimeout(function() { _isReplay = false; }, 500);
  };
  ws.onclose = function() {
    wsConnected = false;
    updateConnectionStatus(false);
    setTimeout(connectWS, 2000);
  };
  ws.onerror = function() { ws.close(); };
  ws.onmessage = function(evt) {
    try {
      var data = JSON.parse(evt.data);
      handleMessage(data);
    } catch(e) { /* ignore parse errors */ }
  };
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
  // 1. Build node graph from rosters (first time or on roster change)
  if (data.sub_agents || data.managers || data.reviewers) {
    rebuildNodes(data);
    if (typeof syncProgramsToRoster === 'function') syncProgramsToRoster(data);
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
    node.toolCalls = task.tool_calls || [];
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

  // 6. Render event log
  if (data.log) renderEventLog(data.log, data.project || 'NORT', data.results_count || 0, data.total_tasks || 0);

  // 7. Handle verdict popup + bubble
  if (data.last_verdict && config.completionFx && !_isReplay) {
    showVerdict(data.last_verdict);
    var vNode = getNodeByAgent(data.last_verdict.agent || data.last_verdict.reviewer);
    if (vNode && typeof addBubble === 'function') {
      var vRole = vNode.tier === 'probe' ? 'probe' : 'sentinel';
      addBubble(vNode.id, vRole, data.last_verdict.verdict + ' (' + data.last_verdict.score + '/10)');
    }
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
