// ═══ NORT INITIALIZATION ═══

// ── City State Persistence Helpers ──────────────────────────────────────────

function _loadPersistedCityState() {
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
    // Restore building state (locations are initialized before this runs)
    if (saved.buildings && typeof deserializeBuildingState === 'function') {
      deserializeBuildingState(saved);
    }
    // Restore node positions (nodes may not exist yet at boot, handled by websocket _restorePersistedState)
    if (saved.nodes && typeof deserializeCityState === 'function') {
      deserializeCityState(saved);
    }
  } catch(e) {
    console.warn('[Init] Failed to load persisted city state:', e.message);
  }
}

var _saveCityStateTimer = null;

function _saveCityState() {
  // Debounce: clear any pending save
  if (_saveCityStateTimer) { clearTimeout(_saveCityStateTimer); _saveCityStateTimer = null; }
  try {
    var cityData = (typeof serializeCityState === 'function') ? serializeCityState() : { nodes: {} };
    var buildingData = (typeof serializeBuildingState === 'function') ? serializeBuildingState() : { buildings: {} };
    var merged = {
      nodes: cityData.nodes || {},
      buildings: buildingData.buildings || {},
      savedAt: new Date().toISOString()
    };
    localStorage.setItem('nort_city_state', JSON.stringify(merged));
  } catch(e) {
    console.warn('[Init] Failed to save city state:', e.message);
  }
}

document.addEventListener('DOMContentLoaded', function() {
  initCanvas();
  if (typeof initAudio === 'function') initAudio();
  if (typeof initRoster === 'function') initRoster();
  if (typeof initWeather === 'function') initWeather();

  // ── Load persisted city state from localStorage ──
  _loadPersistedCityState();

  connectWS();
  refreshQueue();
  checkPendingApprovals();
  pollHeartbeat();
  setInterval(pollHeartbeat, 3000);

  // ── Periodic city state save (every 30 seconds) ──
  setInterval(_saveCityState, 30000);

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      hideAgentDetail();
      hidePlanViewer();
      hideCompletion();
      var cfgOverlay = document.getElementById('configOverlay');
      if (cfgOverlay) cfgOverlay.classList.remove('visible');
      var ledger = document.getElementById('ledgerOverlay');
      if (ledger) ledger.classList.add('hidden');
      var modelCfg = document.getElementById('modelConfigOverlay');
      if (modelCfg) modelCfg.classList.add('hidden');
      var agentsOvl = document.getElementById('agentsOverlay');
      if (agentsOvl) agentsOvl.classList.add('hidden');
    }
    if (e.key === 'q' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      toggleQueue();
    }
    if (e.key === 'r' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      if (typeof toggleRoster === 'function') toggleRoster();
    }
    if (e.key === 'm' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      config.minimap = !config.minimap;
    }
    if (e.key === 'a' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      if (typeof toggleAgentsPanel === 'function') toggleAgentsPanel();
    }
    if (e.key === 'c' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      if (typeof toggleChat === 'function') toggleChat();
    }
    if (e.key === 'l' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      if (typeof toggleAgentList === 'function') toggleAgentList();
    }
  });

  // Enter key on plan input
  var planInput = document.getElementById('planInput');
  if (planInput) {
    planInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') generatePlan();
    });
  }

  // Click-outside dismissal for overlays
  var planViewer = document.getElementById('planViewer');
  if (planViewer) {
    planViewer.addEventListener('click', function(e) {
      if (e.target === planViewer) hidePlanViewer();
    });
  }
  var modelOverlay = document.getElementById('modelConfigOverlay');
  if (modelOverlay) {
    modelOverlay.addEventListener('click', function(e) {
      if (e.target === modelOverlay) modelOverlay.classList.add('hidden');
    });
  }
  var ledgerOverlay = document.getElementById('ledgerOverlay');
  if (ledgerOverlay) {
    ledgerOverlay.addEventListener('click', function(e) {
      if (e.target === ledgerOverlay) ledgerOverlay.classList.add('hidden');
    });
  }
  var completionOverlay = document.getElementById('completionOverlay');
  if (completionOverlay) {
    completionOverlay.addEventListener('click', function(e) {
      if (e.target === completionOverlay) hideCompletion();
    });
  }
});
