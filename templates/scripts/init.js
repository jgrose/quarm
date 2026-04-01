// ═══ NORT INITIALIZATION ═══

document.addEventListener('DOMContentLoaded', function() {
  initCanvas();
  if (typeof initAudio === 'function') initAudio();
  if (typeof initRoster === 'function') initRoster();
  if (typeof initWeather === 'function') initWeather();
  connectWS();
  refreshQueue();
  checkPendingApprovals();
  pollHeartbeat();
  setInterval(pollHeartbeat, 3000);

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
      if (modelCfg) modelCfg.classList.remove('visible');
      var agentsOvl = document.getElementById('agentsOverlay');
      if (agentsOvl) agentsOvl.classList.add('hidden');
      var reviewAnalytics = document.getElementById('reviewAnalyticsOverlay');
      if (reviewAnalytics) reviewAnalytics.classList.add('hidden');
      var toleranceCfg = document.getElementById('toleranceConfigOverlay');
      if (toleranceCfg) toleranceCfg.classList.add('hidden');
      var outputBrowser = document.getElementById('outputBrowserOverlay');
      if (outputBrowser) outputBrowser.classList.add('hidden');
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
