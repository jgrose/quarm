// ═══ NORT API CALLS ═══

// Helpers to reduce boilerplate
async function _apiGet(url, name) {
  try { var r = await fetch(url); if (!r.ok) return null; return await r.json(); }
  catch (e) { console.error(name + ':', e); return null; }
}
async function _apiPost(url, body, name) {
  try { var opts = { method: 'POST' }; if (body !== undefined) { opts.headers = { 'Content-Type': 'application/json' }; opts.body = JSON.stringify(body); } await fetch(url, opts); }
  catch (e) { console.error(name + ':', e); }
}
async function _apiPostJson(url, body, name) {
  try { var r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); return await r.json(); }
  catch (e) { console.error(name + ':', e); return null; }
}
async function _apiDelete(url, name) {
  try { await fetch(url, { method: 'DELETE' }); } catch (e) { console.error(name + ':', e); }
}

async function generatePlan() {
  var input = document.getElementById('planInput');
  var btn = document.getElementById('btnGenerate');
  var desc = input.value.trim();
  if (!desc) return;
  if (btn) { btn.textContent = 'GENERATING...'; btn.disabled = true; }
  try {
    var res = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ description: desc }) });
    if (!res.ok) throw new Error(await res.text());
    input.value = '';
    var panel = document.getElementById('queuePanel');
    if (panel && panel.classList.contains('hidden')) toggleQueue();
    else refreshQueue();
  } catch (e) { console.error('generatePlan:', e); }
  finally { if (btn) { btn.textContent = 'GENERATE'; btn.disabled = false; } }
}

async function refreshQueue() { var plans = await _apiGet('/api/plans', 'refreshQueue'); if (plans) renderQueue(plans); }
async function runPlan(id) { await _apiPost('/api/plans/' + id + '/run', undefined, 'runPlan'); refreshQueue(); }
async function deletePlan(id) { await _apiDelete('/api/plans/' + id, 'deletePlan'); refreshQueue(); }
async function stopPlan(id) { await _apiPost('/api/plans/' + id + '/stop', undefined, 'stopPlan'); refreshQueue(); }
async function reorderQueue(orderedIds) { await _apiPost('/api/plans/reorder', { order: orderedIds }, 'reorderQueue'); }

async function viewPlan(id) {
  var plan = await _apiGet('/api/plans/' + id, 'viewPlan');
  if (plan) showPlanViewer(plan.content || '(empty)', plan.title || 'PLAN');
}

async function loadLedgerData() {
  try {
    var results = await Promise.all([
      fetch('/api/analytics/costs').then(function(r) { return r.json(); }),
      fetch('/api/analytics/scores').then(function(r) { return r.json(); }),
    ]);
    renderLedger(results[0], results[1]);
  } catch (e) { console.error('loadLedgerData:', e); }
}

async function loadModels() { var data = await _apiGet('/api/models', 'loadModels'); if (data) renderModelConfig(data); }
async function saveModels(allowedModels) { await _apiPost('/api/models', { allowed_models: allowedModels }, 'saveModels'); }
async function saveWebhook(url) { await _apiPost('/api/config', { webhook_url: url }, 'saveWebhook'); }

async function testWebhook() {
  try { var resp = await fetch('/api/webhook/test', { method: 'POST' }); var data = await resp.json(); return !!data.ok; }
  catch (e) { return false; }
}

async function approveToolCall(toolCallId, approved) {
  await _apiPost('/api/approvals/' + toolCallId, { approved: approved }, 'approveToolCall');
  hideApproval();
}

async function checkPendingApprovals() {
  var data = await _apiGet('/api/approvals', 'checkPendingApprovals');
  if (data && data.pending && data.pending.length > 0) showApproval(data.pending[0]);
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

async function loadTolerance() { var data = await _apiGet('/api/tolerance', 'loadTolerance'); if (data) renderToleranceConfig(data); }

async function saveToleranceGlobal(value) {
  await _apiPost('/api/tolerance', { default_tolerance: value }, 'saveToleranceGlobal');
  if (typeof highlightActivePreset === 'function') highlightActivePreset('');
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

async function fetchTeamPresets() {
  try {
    var res = await fetch('/api/teams/presets');
    if (!res.ok) return [];
    var data = await res.json();
    return data.presets || [];
  } catch (e) {
    console.error('fetchTeamPresets:', e);
    return [];
  }
}

async function applyTeamPreset(presetName, teamName) {
  try {
    var res = await fetch('/api/teams/presets/' + presetName + '/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team_name: teamName })
    });
    return await res.json();
  } catch (e) {
    console.error('applyTeamPreset:', e);
    return null;
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
