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

  // ── Building detail card live updates ──
  if (typeof _updateBuildingDetailIfNeeded === 'function') {
    setInterval(_updateBuildingDetailIfNeeded, 500);
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      // Close ONE panel per Escape press, highest z-index first.
      // Each check returns early after closing the topmost visible panel.

      // z:200 — Approval banner (highest interactive layer)
      // (Approval banner has no close-on-escape — it requires explicit action)

      // z:150 — Modal overlays (completion, plan viewer)
      var completionOvl = document.getElementById('completionOverlay');
      if (completionOvl && !completionOvl.classList.contains('hidden')) {
        hideCompletion(); return;
      }
      var planViewerOvl = document.getElementById('planViewer');
      if (planViewerOvl && !planViewerOvl.classList.contains('hidden')) {
        hidePlanViewer(); return;
      }

      // z:150 — Config overlays (config, model config, ledger, cost, agents, tolerance, review analytics, output browser)
      var cfgOverlay = document.getElementById('configOverlay');
      if (cfgOverlay && cfgOverlay.classList.contains('visible')) {
        cfgOverlay.classList.remove('visible'); return;
      }
      var modelCfg = document.getElementById('modelConfigOverlay');
      if (modelCfg && modelCfg.classList.contains('visible')) {
        modelCfg.classList.remove('visible'); return;
      }
      var ledger = document.getElementById('ledgerOverlay');
      if (ledger && !ledger.classList.contains('hidden')) {
        ledger.classList.add('hidden'); return;
      }
      var costPanel = document.getElementById('costPanelOverlay');
      if (costPanel && !costPanel.classList.contains('hidden')) {
        if (typeof toggleCostPanel === 'function') toggleCostPanel();
        else costPanel.classList.add('hidden');
        return;
      }
      var agentsOvl = document.getElementById('agentsOverlay');
      if (agentsOvl && !agentsOvl.classList.contains('hidden')) {
        agentsOvl.classList.add('hidden'); return;
      }
      var dagPanel = document.getElementById('dagPanelOverlay');
      if (dagPanel && !dagPanel.classList.contains('hidden')) {
        dagPanel.classList.add('hidden'); return;
      }
      var toleranceCfg = document.getElementById('toleranceConfigOverlay');
      if (toleranceCfg && !toleranceCfg.classList.contains('hidden')) {
        toleranceCfg.classList.add('hidden'); return;
      }
      var reviewAnalytics = document.getElementById('reviewAnalyticsOverlay');
      if (reviewAnalytics && !reviewAnalytics.classList.contains('hidden')) {
        reviewAnalytics.classList.add('hidden'); return;
      }
      var outputBrowser = document.getElementById('outputBrowserOverlay');
      if (outputBrowser && !outputBrowser.classList.contains('hidden')) {
        outputBrowser.classList.add('hidden'); return;
      }

      // z:100 — Agent detail card / Building detail card
      var agentCard = document.getElementById('agentDetailCard');
      if (agentCard && !agentCard.classList.contains('hidden')) {
        hideAgentDetail(); return;
      }
      var buildingCard = document.getElementById('buildingDetailCard');
      if (buildingCard && !buildingCard.classList.contains('hidden')) {
        if (typeof hideBuildingDetail === 'function') hideBuildingDetail(); return;
      }

      // z:80 — Perf overlay
      var perfPanel = document.getElementById('perfPanel');
      if (perfPanel && !perfPanel.classList.contains('hidden')) {
        if (typeof togglePerfOverlay === 'function') togglePerfOverlay();
        else perfPanel.classList.add('hidden');
        return;
      }

      // z:40 — Floating panels (queue, thinking, roster, file attention, transcript, timeline)
      var queuePanel = document.getElementById('queuePanel');
      if (queuePanel && !queuePanel.classList.contains('hidden')) {
        toggleQueue(); return;
      }
      var thinkingPanel = document.getElementById('thinkingPanel');
      if (thinkingPanel && !thinkingPanel.classList.contains('hidden')) {
        hideThinkingPanel(); return;
      }
      var rosterPanel = document.getElementById('rosterPanel');
      if (rosterPanel && !rosterPanel.classList.contains('hidden')) {
        if (typeof toggleRoster === 'function') toggleRoster();
        else rosterPanel.classList.add('hidden');
        return;
      }
      var fileAttn = document.getElementById('fileAttention');
      if (fileAttn && !fileAttn.classList.contains('hidden')) {
        fileAttn.classList.add('hidden'); return;
      }
      var transcript = document.getElementById('transcriptPanel');
      if (transcript && !transcript.classList.contains('hidden')) {
        transcript.classList.add('hidden'); return;
      }
      var timeline = document.getElementById('timelinePanel');
      if (timeline && !timeline.classList.contains('hidden')) {
        timeline.classList.add('hidden'); return;
      }

      // z:30 — Side panels (event log, agent list) — close if open
      var eventLog = document.getElementById('eventLog');
      if (eventLog && !eventLog.classList.contains('collapsed')) {
        toggleChat(); return;
      }
      var agentList = document.getElementById('agentListPanel');
      if (agentList && !agentList.classList.contains('collapsed')) {
        toggleAgentList(); return;
      }
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
    if (e.key === 'd' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
      config.dependencies = !config.dependencies;
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
      if (e.target === modelOverlay) modelOverlay.classList.remove('visible');
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
  var reviewOverlay = document.getElementById('reviewAnalyticsOverlay');
  if (reviewOverlay) {
    reviewOverlay.addEventListener('click', function(e) {
      if (e.target === reviewOverlay) reviewOverlay.classList.add('hidden');
    });
  }
  var tolCfgOverlay = document.getElementById('toleranceConfigOverlay');
  if (tolCfgOverlay) {
    tolCfgOverlay.addEventListener('click', function(e) {
      if (e.target === tolCfgOverlay) tolCfgOverlay.classList.add('hidden');
    });
  }
  var outputBrowserOvl = document.getElementById('outputBrowserOverlay');
  if (outputBrowserOvl) {
    outputBrowserOvl.addEventListener('click', function(e) {
      if (e.target === outputBrowserOvl) outputBrowserOvl.classList.add('hidden');
    });
  }
});
