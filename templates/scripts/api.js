// ═══ NORT API CALLS ═══

async function generatePlan() {
  var input = document.getElementById('planInput');
  var btn = document.getElementById('btnGenerate');
  var desc = input.value.trim();
  if (!desc) return;

  if (btn) { btn.textContent = 'GENERATING...'; btn.disabled = true; }

  try {
    var res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: desc }),
    });
    if (!res.ok) throw new Error(await res.text());
    input.value = '';
    var panel = document.getElementById('queuePanel');
    if (panel && panel.classList.contains('hidden')) toggleQueue();
    else refreshQueue();
  } catch (e) {
    console.error('generatePlan:', e);
  } finally {
    if (btn) { btn.textContent = 'GENERATE'; btn.disabled = false; }
  }
}

async function refreshQueue() {
  try {
    var res = await fetch('/api/plans');
    if (!res.ok) return;
    var plans = await res.json();
    renderQueue(plans);
  } catch (e) {
    console.error('refreshQueue:', e);
  }
}

async function runPlan(id) {
  try {
    await fetch('/api/plans/' + id + '/run', { method: 'POST' });
    refreshQueue();
  } catch (e) {
    console.error('runPlan:', e);
  }
}

async function deletePlan(id) {
  try {
    await fetch('/api/plans/' + id, { method: 'DELETE' });
    refreshQueue();
  } catch (e) {
    console.error('deletePlan:', e);
  }
}

async function stopPlan(id) {
  try {
    await fetch('/api/plans/' + id + '/stop', { method: 'POST' });
    refreshQueue();
  } catch (e) {
    console.error('stopPlan:', e);
  }
}

async function reorderQueue(orderedIds) {
  try {
    await fetch('/api/plans/reorder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ order: orderedIds }),
    });
  } catch (e) {
    console.error('reorderQueue:', e);
  }
}

async function viewPlan(id) {
  try {
    var res = await fetch('/api/plans/' + id);
    if (!res.ok) return;
    var plan = await res.json();
    showPlanViewer(plan.content || '(empty)', plan.title || 'PLAN');
  } catch (e) {
    console.error('viewPlan:', e);
  }
}

async function loadLedgerData() {
  try {
    var results = await Promise.all([
      fetch('/api/analytics/costs').then(function(r) { return r.json(); }),
      fetch('/api/analytics/scores').then(function(r) { return r.json(); }),
    ]);
    renderLedger(results[0], results[1]);
  } catch (e) {
    console.error('loadLedgerData:', e);
  }
}

async function loadModels() {
  try {
    var res = await fetch('/api/models');
    if (!res.ok) throw new Error(await res.text());
    var data = await res.json();
    renderModelConfig(data);
  } catch (e) {
    console.error('loadModels:', e);
  }
}

async function saveModels(allowedModels) {
  try {
    await fetch('/api/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ allowed_models: allowedModels }),
    });
  } catch (e) {
    console.error('saveModels:', e);
  }
}

async function saveWebhook(url) {
  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ webhook_url: url }),
    });
  } catch (e) {
    console.error('saveWebhook:', e);
  }
}

async function testWebhook() {
  try {
    var resp = await fetch('/api/webhook/test', { method: 'POST' });
    var data = await resp.json();
    return !!data.ok;
  } catch (e) {
    return false;
  }
}

async function approveToolCall(toolCallId, approved) {
  try {
    await fetch('/api/approvals/' + toolCallId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved: approved }),
    });
    hideApproval();
  } catch (e) {
    console.error('approveToolCall:', e);
  }
}

async function checkPendingApprovals() {
  try {
    var res = await fetch('/api/approvals');
    if (!res.ok) return;
    var data = await res.json();
    if (data.pending && data.pending.length > 0) {
      showApproval(data.pending[0]);
    }
  } catch (e) { /* ignore */ }
}

async function pollHeartbeat() {
  var dot = document.querySelector('.heartbeat-dot');
  var label = document.getElementById('heartbeatText');
  try {
    var res = await fetch('/api/health');
    if (!res.ok) return;
    var data = await res.json();
    var status = data.status || 'idle';
    if (dot) { dot.className = 'heartbeat-dot ' + status; }
    if (label) {
      if (status === 'running') {
        var count = data.running_count || 1;
        var since = data.seconds_since_update || 0;
        var prefix = count > 1 ? count + ' PLANS' : 'WORKING';
        if (since < 10) label.textContent = prefix + '...';
        else if (since < 60) label.textContent = prefix + ' (' + since + 's)';
        else label.textContent = prefix + ' (' + Math.floor(since / 60) + 'm)';
      } else if (status === 'stuck') {
        label.textContent = 'STUCK';
      } else {
        label.textContent = 'IDLE';
      }
    }
  } catch (e) {
    if (dot) dot.className = 'heartbeat-dot idle';
    if (label) label.textContent = 'OFFLINE';
  }
}

async function loadTolerance() {
  try {
    var res = await fetch('/api/tolerance');
    if (!res.ok) throw new Error(await res.text());
    var data = await res.json();
    renderToleranceConfig(data);
  } catch (e) {
    console.error('loadTolerance:', e);
  }
}

async function saveToleranceGlobal(value) {
  try {
    await fetch('/api/tolerance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ default_tolerance: value }),
    });
    // Clear preset highlight on manual change
    if (typeof highlightActivePreset === 'function') highlightActivePreset('');
  } catch (e) {
    console.error('saveToleranceGlobal:', e);
  }
}

var _saveToleranceAgentTimers = {};

function saveToleranceAgent(name, value) {
  if (_saveToleranceAgentTimers[name]) clearTimeout(_saveToleranceAgentTimers[name]);
  _saveToleranceAgentTimers[name] = setTimeout(async function() {
    delete _saveToleranceAgentTimers[name];
    try {
      var res = await fetch('/api/tolerance');
      var data = await res.json();
      var overrides = data.overrides || {};
      var intVal = parseInt(value);
      if (intVal === data.default_tolerance) {
        delete overrides[name];
      } else {
        overrides[name] = intVal;
      }
      await fetch('/api/tolerance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tolerance_overrides: overrides }),
      });
      if (typeof highlightActivePreset === 'function') highlightActivePreset('');
    } catch (e) {
      console.error('saveToleranceAgent:', e);
    }
  }, 300);
}

async function loadReviewStats() {
  try {
    var res = await fetch('/api/review-stats');
    if (!res.ok) throw new Error(await res.text());
    var data = await res.json();
    renderReviewAnalytics(data);
  } catch (e) {
    console.error('loadReviewStats:', e);
    var body = document.getElementById('reviewAnalyticsBody');
    if (body) body.innerHTML = '<div style="color:var(--state-error);font-size:11px;text-align:center;padding:20px">Failed to load review data</div>';
  }
}

// ── Agent Registry API ────────────────────────────────────────────────────

async function fetchAgentVersions(agentType, name) {
  try {
    var res = await fetch('/api/agents/' + agentType + '/' + name + '/versions');
    if (!res.ok) return [];
    var data = await res.json();
    return data.versions || [];
  } catch (e) {
    console.error('fetchAgentVersions:', e);
    return [];
  }
}

async function rollbackAgent(agentType, name, version) {
  try {
    var res = await fetch('/api/agents/' + agentType + '/' + name + '/rollback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version: version })
    });
    return await res.json();
  } catch (e) {
    console.error('rollbackAgent:', e);
    return null;
  }
}

async function cloneAgent(agentType, name, newName) {
  try {
    var body = newName ? { new_name: newName } : {};
    var res = await fetch('/api/agents/' + agentType + '/' + name + '/clone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await res.json();
  } catch (e) {
    console.error('cloneAgent:', e);
    return null;
  }
}

async function retireAgent(agentType, name, retired) {
  try {
    var res = await fetch('/api/agents/' + agentType + '/' + name + '/retire', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ retired: retired })
    });
    return await res.json();
  } catch (e) {
    console.error('retireAgent:', e);
    return null;
  }
}

async function fetchTeams() {
  try {
    var res = await fetch('/api/teams');
    if (!res.ok) return [];
    var data = await res.json();
    return data.teams || [];
  } catch (e) {
    console.error('fetchTeams:', e);
    return [];
  }
}

async function createTeam(spec) {
  try {
    var res = await fetch('/api/teams', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec)
    });
    return await res.json();
  } catch (e) {
    console.error('createTeam:', e);
    return null;
  }
}

async function deleteTeam(name) {
  try {
    await fetch('/api/teams/' + name, { method: 'DELETE' });
  } catch (e) {
    console.error('deleteTeam:', e);
  }
}

async function exportAgentsData() {
  try {
    var res = await fetch('/api/agents/export');
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  } catch (e) {
    console.error('exportAgentsData:', e);
    return null;
  }
}

async function importAgentsData(data, overwrite) {
  try {
    var res = await fetch('/api/agents/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: data, overwrite: overwrite })
    });
    return await res.json();
  } catch (e) {
    console.error('importAgentsData:', e);
    return null;
  }
}
