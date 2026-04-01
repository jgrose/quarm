// ═══ NORT RENDER LOOP ═══

var lastFrame = 0;
var currentTime = 0;
var dpr = 1;
var bloomRenderer = new BloomRenderer();
var _canvas = null;
var _ctx = null;

function initCanvas() {
  _canvas = document.getElementById('canvas');
  _ctx = _canvas.getContext('2d');
  var wrap = document.getElementById('canvasWrap');
  dpr = window.devicePixelRatio || 1;

  function resize() {
    var rect = wrap.getBoundingClientRect();
    _canvas.width = rect.width * dpr;
    _canvas.height = rect.height * dpr;
    _canvas.style.width = rect.width + 'px';
    _canvas.style.height = rect.height + 'px';
  }

  new ResizeObserver(resize).observe(wrap);
  resize();
  initCamera(_canvas);

  _canvas.addEventListener('click', function(e) {
    // Minimap click-to-pan
    if (typeof handleMinimapClick === 'function' && config.minimap) {
      if (handleMinimapClick(e)) return;
    }
    // Agent node click
    var node = getNodeAt(e);
    if (node) {
      if (typeof showAgentDetail === 'function') showAgentDetail(node);
      return;
    }
    // Location click (bunker inspect)
    if (typeof getLocationAt === 'function') {
      var loc = getLocationAt(e);
      if (loc) {
        // Show location occupant info in agent detail card
        if (typeof showAgentDetail === 'function') {
          var info = { name: loc.name, tier: 'location', state: loc.category, taskId: loc.taskState || 'idle',
            model: loc.occupants.length + '/' + loc.capacity + ' programs', tokens: loc.taskCompletions || 0,
            toolCalls: [], resultPreview: loc.occupants.map(function(o) { return o.glow; }).join(', '),
            radius: 20, icon: '', label: loc.name };
          showAgentDetail(info);
        }
        return;
      }
    }
    if (typeof hideAgentDetail === 'function') hideAgentDetail();
  });

  requestAnimationFrame(render);
}

function render(timestamp) {
  var dt = Math.min((timestamp - lastFrame) / 1000, ANIM.maxDt) || ANIM.defaultDt;
  lastFrame = timestamp;
  currentTime += dt;

  var canvas = _canvas;
  var ctx = _ctx;
  var W = canvas.width / dpr;
  var H = canvas.height / dpr;

  // Day/night atmosphere (mutates C.void before clear)
  if (config.daynight && typeof applyAtmosphere === 'function') {
    applyAtmosphere(ctx, W, H, currentTime);
  }
  if (typeof updateAtmosphere === 'function') updateAtmosphere(dt);

  // Clear
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = C.void;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Atmospheric layer (screen space, no camera — parallax depth stars)
  ctx.save();
  ctx.scale(dpr, dpr);
  drawDepthParticles(ctx, W, H, currentTime);
  ctx.restore();

  // World space (camera transform — grid, locations, programs, agents)
  ctx.save();
  applyCamera(ctx, canvas);

  // Hex grid (viewport-aware)
  if (config.hexGrid) drawHexGrid(ctx, W, H, currentTime);

  // Grid locations
  drawAllLocations(ctx, W, H, currentTime, dt);

  // Ambient Tron programs
  updateAmbientPrograms(W, H, dt);
  drawAmbientPrograms(ctx, currentTime);

  // Roster level badges above programs
  if (config.roster && typeof drawRosterBadges === 'function') drawRosterBadges(ctx);

  // Force simulation
  tickForce(Array.from(nodes.values()), edges, W, H, dt);

  // Animate node fade-in
  for (var entry of nodes) {
    var n = entry[1];
    if (n.opacity < 1) n.opacity = Math.min(1, n.opacity + ANIM.agentFadeIn * dt);
    if (n.scale < 1) n.scale = Math.min(1, n.scale + ANIM.agentScaleIn * dt);
  }

  // Draw layers (back to front)
  if (config.hexNodes) {
    drawAllEdges(ctx, currentTime);
    updateParticles(dt);
    drawAllParticles(ctx, currentTime);
    drawAllAgents(ctx, currentTime);
  }
  if (config.nodeStats && typeof drawAllContextBars === 'function') drawAllContextBars(ctx, currentTime);
  if (typeof drawAllBubbles === 'function') drawAllBubbles(ctx, currentTime);
  if (config.toolCards) drawAllToolCards(ctx, currentTime);
  if (typeof drawAllDiscoveries === 'function') drawAllDiscoveries(ctx, currentTime, dt);
  drawEffects(ctx, dt);

  ctx.restore();

  // Screen-space overlays (no camera transform)
  ctx.save();
  ctx.scale(dpr, dpr);

  // Weather effects (rain, lightning — screen space)
  if (config.weather && typeof drawWeather === 'function') {
    if (typeof updateWeather === 'function') updateWeather(dt);
    drawWeather(ctx, W, H, currentTime, dt);
  }

  // Minimap
  if (config.minimap && typeof drawMinimap === 'function') drawMinimap(ctx, W, H, currentTime);

  if (typeof drawAllCostPills === 'function') drawAllCostPills(ctx, currentTime);
  if (typeof drawCostPanel === 'function') drawCostPanel(ctx, W, H);
  ctx.restore();

  // Post-processing
  if (config.bloom) bloomRenderer.apply(canvas);

  requestAnimationFrame(render);
}
