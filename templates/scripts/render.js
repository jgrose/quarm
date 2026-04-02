// ═══ NORT RENDER LOOP ═══

var lastFrame = 0;
var currentTime = 0;
var dpr = 1;
var bloomRenderer = new BloomRenderer();
var _canvas = null;
var _ctx = null;
var _nodeArr = [];

// ── Idle detection ──
var _idleFrameCount = 0;
var _isIdle = false;
var _lastRenderTime = 0;

// ── FPS counter state ──
var _frameCount = 0;
var _fpsAccum = 0;
var _fps = 0;
var _lastFpsSample = 0;

// ── Per-system timing ──
var _perfTimings = {};
var _perfFrameTotal = 0;

function _timeStart() { return performance.now(); }
function _timeEnd(name, t0) {
  var elapsed = performance.now() - t0;
  if (_perfTimings[name] === undefined) {
    _perfTimings[name] = elapsed;
  } else {
    _perfTimings[name] = _perfTimings[name] * (1 - PERF.emaAlpha) + elapsed * PERF.emaAlpha;
  }
}

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
  // Idle render throttle: ~4fps when nothing is animating
  if (config.idlePause) {
    var idle = _forceSettled &&
               particles.length === 0 &&
               effects.length === 0 &&
               !camera.dragging;
    if (idle) {
      for (var _ie of nodes) {
        if (_ie[1].opacity < 1 || _ie[1].scale < 1) { idle = false; break; }
      }
    }
    if (idle) {
      _idleFrameCount++;
      if (_idleFrameCount > 30 && timestamp - _lastRenderTime < 250) {
        _isIdle = true;
        requestAnimationFrame(render);
        return;
      }
    } else {
      _idleFrameCount = 0;
      _isIdle = false;
    }
  }
  _lastRenderTime = timestamp;

  var frameStart = performance.now();
  var dt = Math.min((timestamp - lastFrame) / 1000, ANIM.maxDt) || ANIM.defaultDt;
  lastFrame = timestamp;
  currentTime += dt;

  // FPS sampling
  _frameCount++;
  _fpsAccum += dt;
  if (_fpsAccum >= PERF.sampleInterval / 1000) {
    _fps = _frameCount / _fpsAccum;
    _frameCount = 0;
    _fpsAccum = 0;
  }

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
  if (config.hexGrid) {
    var _t0 = _timeStart();
    drawHexGrid(ctx, W, H, currentTime);
    _timeEnd('hexGrid', _t0);
  }

  // Grid locations
  var _t1 = _timeStart();
  drawAllLocations(ctx, W, H, currentTime, dt);
  _timeEnd('locations', _t1);

  // Ambient Tron programs
  var _t2 = _timeStart();
  updateAmbientPrograms(W, H, dt);
  drawAmbientPrograms(ctx, currentTime);
  _timeEnd('programs', _t2);

  // Roster level badges above programs
  if (config.roster && typeof drawRosterBadges === 'function') drawRosterBadges(ctx);

  // Force simulation
  var _t3 = _timeStart();
  _nodeArr.length = 0;
  for (var _ne of nodes) _nodeArr.push(_ne[1]);
  tickForce(_nodeArr, edges, W, H, dt);
  _timeEnd('force', _t3);

  // Animate node fade-in
  for (var entry of nodes) {
    var n = entry[1];
    if (n.opacity < 1) n.opacity = Math.min(1, n.opacity + ANIM.agentFadeIn * dt);
    if (n.scale < 1) n.scale = Math.min(1, n.scale + ANIM.agentScaleIn * dt);
  }

  // Draw layers (back to front)
  if (config.hexNodes) {
    var _t4 = _timeStart();
    drawAllEdges(ctx, currentTime, W, H);
    _timeEnd('edges', _t4);
    updateParticles(dt);
    drawAllParticles(ctx, currentTime);
    var _t5 = _timeStart();
    drawAllAgents(ctx, currentTime, W, H);
    _timeEnd('agents', _t5);
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
    var _t6 = _timeStart();
    drawWeather(ctx, W, H, currentTime, dt);
    _timeEnd('weather', _t6);
  }

  // Minimap
  if (config.minimap && typeof drawMinimap === 'function') {
    var _t7 = _timeStart();
    drawMinimap(ctx, W, H, currentTime);
    _timeEnd('minimap', _t7);
  }

  if (typeof drawAllCostPills === 'function') drawAllCostPills(ctx, currentTime);
  if (typeof drawCostPanel === 'function') drawCostPanel(ctx, W, H);
  ctx.restore();

  // Post-processing
  if (config.bloom) {
    var _t8 = _timeStart();
    bloomRenderer.apply(canvas);
    _timeEnd('bloom', _t8);
  }

  // Frame total timing
  _perfFrameTotal = _perfFrameTotal * (1 - PERF.emaAlpha) + (performance.now() - frameStart) * PERF.emaAlpha;

  // FPS badge + perf overlay (screen-space, after all rendering)
  if (config.perfOverlay) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    _drawFpsBadge(ctx, W);
    _drawPerfOverlay(ctx, W);
    ctx.restore();
  }

  requestAnimationFrame(render);
}

// ── FPS Badge (top-right corner) ──
function _drawFpsBadge(ctx, W) {
  var text = _fps.toFixed(0) + ' FPS';
  ctx.font = '600 11px Inter, system-ui, sans-serif';
  var tw = ctx.measureText(text).width;
  var px = W - tw - 16;
  var py = 14;
  // Semi-transparent background pill
  ctx.fillStyle = 'rgba(10, 15, 30, 0.65)';
  ctx.beginPath();
  ctx.roundRect(px - 6, py - 10, tw + 12, 16, 4);
  ctx.fill();
  ctx.strokeStyle = 'rgba(100, 200, 255, 0.2)';
  ctx.lineWidth = 1;
  ctx.stroke();
  // Text color shifts green->yellow->red based on fps
  if (_fps >= 55) ctx.fillStyle = '#66ffaa';
  else if (_fps >= 30) ctx.fillStyle = '#ffbb44';
  else ctx.fillStyle = '#ff5566';
  ctx.fillText(text, px, py);
}

// ── Performance Overlay (bar chart) ──
function _drawPerfOverlay(ctx, W) {
  var systems = ['hexGrid', 'locations', 'programs', 'force', 'edges', 'agents', 'weather', 'minimap', 'bloom'];
  var barW = 120;
  var barH = 8;
  var gap = 4;
  var labelW = 64;
  var panelW = labelW + barW + 40;
  var lineH = barH + gap;
  var panelH = systems.length * lineH + 36;
  var px = W - panelW - 12;
  var py = 32;

  // Panel background
  ctx.fillStyle = 'rgba(10, 15, 30, 0.75)';
  ctx.beginPath();
  ctx.roundRect(px - 8, py - 4, panelW + 16, panelH + 8, 6);
  ctx.fill();
  ctx.strokeStyle = 'rgba(100, 200, 255, 0.15)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Title
  ctx.font = '600 9px Inter, system-ui, sans-serif';
  ctx.fillStyle = 'rgba(170, 238, 255, 0.7)';
  ctx.fillText('FRAME TIMING', px, py + 8);

  // Total frame time
  var totalText = _perfFrameTotal.toFixed(1) + 'ms / ' + PERF.budgetMs.toFixed(1) + 'ms';
  ctx.font = '500 8px Inter, system-ui, sans-serif';
  ctx.fillStyle = _perfFrameTotal <= PERF.budgetMs ? 'rgba(102, 255, 170, 0.8)' : 'rgba(255, 85, 102, 0.8)';
  ctx.fillText(totalText, px + 70, py + 8);

  var startY = py + 20;
  var maxMs = Math.max(PERF.budgetMs, 2);

  for (var i = 0; i < systems.length; i++) {
    var name = systems[i];
    var ms = _perfTimings[name] || 0;
    var y = startY + i * lineH;

    // Label
    ctx.font = '400 8px Inter, system-ui, sans-serif';
    ctx.fillStyle = 'rgba(170, 238, 255, 0.5)';
    ctx.fillText(name, px, y + barH - 1);

    // Bar background
    ctx.fillStyle = 'rgba(100, 200, 255, 0.06)';
    ctx.fillRect(px + labelW, y, barW, barH);

    // Budget line at 16.6ms
    var budgetX = px + labelW + (PERF.budgetMs / maxMs) * barW;
    if (budgetX <= px + labelW + barW) {
      ctx.strokeStyle = 'rgba(255, 187, 68, 0.3)';
      ctx.beginPath();
      ctx.moveTo(budgetX, y);
      ctx.lineTo(budgetX, y + barH);
      ctx.stroke();
    }

    // Bar fill
    var fillW = Math.min(barW, (ms / maxMs) * barW);
    if (ms <= 1) ctx.fillStyle = 'rgba(102, 255, 170, 0.5)';
    else if (ms <= 4) ctx.fillStyle = 'rgba(255, 187, 68, 0.5)';
    else ctx.fillStyle = 'rgba(255, 85, 102, 0.5)';
    ctx.fillRect(px + labelW, y, fillW, barH);

    // Value text
    ctx.fillStyle = 'rgba(170, 238, 255, 0.6)';
    ctx.fillText(ms.toFixed(2) + 'ms', px + labelW + barW + 4, y + barH - 1);
  }
}
