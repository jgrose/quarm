// ═══ NORT FLOW VIEW — INITIALIZATION ═══

(function() {
  // Mark flow mode for force.js and draw_agents.js
  config.flowMode = true;

  // Override city-specific config defaults
  config.hexGrid = false;      // no isometric grid
  config.hexNodes = false;     // we draw agents differently in flow
  config.locations = false;    // no buildings
  config.weather = false;      // no weather
  config.daynight = false;     // no day/night cycle
  config.minimap = false;      // no minimap
  config.roster = false;       // no roster panel
  config.taskArrows = false;   // we use dependency lines instead

  // Initialize canvas
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  var wrap = document.getElementById('canvasWrap');
  dpr = window.devicePixelRatio || 1;

  function resize() {
    var rect = wrap.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    initFlowParticles(rect.width, rect.height);
  }
  new ResizeObserver(resize).observe(wrap);
  resize();
  initCamera(canvas);

  // Click handler (agent nodes)
  canvas.addEventListener('click', function(e) {
    var node = getNodeAt(e);
    if (node) {
      if (typeof showAgentDetail === 'function') showAgentDetail(node);
      return;
    }
    if (typeof hideAgentDetail === 'function') hideAgentDetail();
  });

  // Repurpose the view toggle button for flow → city navigation
  var viewBtn = document.getElementById('viewToggleBtn');
  if (viewBtn) {
    viewBtn.textContent = 'CITY';
    viewBtn.title = 'Switch to city view';
    viewBtn.onclick = function() { window.location.href = '/'; };
  }

  // Connect WebSocket and start API polling
  connectWS();
  refreshQueue();
  checkPendingApprovals();
  pollHeartbeat();
  setInterval(pollHeartbeat, 3000);

  // Keyboard shortcuts (subset relevant to flow view)
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      if (typeof hideAgentDetail === 'function') hideAgentDetail();
      var cfgOverlay = document.getElementById('configOverlay');
      if (cfgOverlay) cfgOverlay.classList.remove('visible');
      var costPanel = document.getElementById('costPanelOverlay');
      if (costPanel) costPanel.classList.add('hidden');
    }
    if (e.ctrlKey || e.metaKey || _isTyping()) return;
    if (e.key === 'c' && typeof toggleChat === 'function') toggleChat();
    if (e.key === 'l' && typeof toggleAgentList === 'function') toggleAgentList();
    if (e.key === 'a' && typeof toggleAgentsPanel === 'function') toggleAgentsPanel();
  });

  // Enter key on plan input
  var planInput = document.getElementById('planInput');
  if (planInput) {
    planInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') generatePlan();
    });
  }

  // Start render loop
  _canvas = canvas;
  _ctx = ctx;
  requestAnimationFrame(flowRender);
})();
