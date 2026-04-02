// ═══ FLOW VIEW RENDER LOOP ═══
// Simplified render loop for the /flow agent-graph view.
// Draws only the node graph — no buildings, programs, weather, atmosphere, minimap.

var _flowLastFrame = 0;
var _flowTime = 0;

function flowRender(timestamp) {
  var dt = Math.min((timestamp - _flowLastFrame) / 1000, 0.1) || 0.016;
  _flowLastFrame = timestamp;
  _flowTime += dt;

  var canvas = _canvas;
  var ctx = _ctx;
  var W = canvas.width / dpr;
  var H = canvas.height / dpr;

  // Clear to void
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#050510';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Screen-space background (depth particles + hex grid, no camera)
  ctx.save();
  ctx.scale(dpr, dpr);
  if (typeof drawFlowBackground === 'function') drawFlowBackground(ctx, W, H, _flowTime);
  ctx.restore();

  // World space (camera transform)
  ctx.save();
  applyCamera(ctx, canvas);

  // Force simulation (free mode — config.flowMode skips zone attraction)
  var nodeArr = [];
  for (var entry of nodes) nodeArr.push(entry[1]);
  tickForce(nodeArr, edges, W, H, dt);

  // Animate node fade-in
  for (var entry of nodes) {
    var n = entry[1];
    if (n.opacity < 1) n.opacity = Math.min(1, n.opacity + 2.0 * dt);
    if (n.scale < 1) n.scale = Math.min(1, n.scale + 2.0 * dt);
  }

  // Draw layers (back to front)
  drawAllEdges(ctx, _flowTime, W, H);
  updateParticles(dt);
  drawAllParticles(ctx, _flowTime);
  drawAllAgents(ctx, _flowTime, W, H);

  if (typeof drawAllContextBars === 'function') drawAllContextBars(ctx, _flowTime);
  if (typeof drawAllBubbles === 'function') drawAllBubbles(ctx, _flowTime);
  if (config.toolCards) drawAllToolCards(ctx, _flowTime);
  if (typeof drawAllDiscoveries === 'function') drawAllDiscoveries(ctx, _flowTime, dt);

  // Dependencies
  if (config.dependencies && typeof drawDependencyLines === 'function') {
    drawDependencyLines(ctx, _flowTime);
  }
  if (config.dependencies && typeof drawBlockedIndicators === 'function') {
    drawBlockedIndicators(ctx, _flowTime);
  }

  drawEffects(ctx, dt);

  ctx.restore();

  // Screen-space overlays
  ctx.save();
  ctx.scale(dpr, dpr);
  if (typeof drawAllCostPills === 'function') drawAllCostPills(ctx, _flowTime);
  ctx.restore();

  // Post-processing
  if (config.bloom) bloomRenderer.apply(canvas);

  requestAnimationFrame(flowRender);
}
