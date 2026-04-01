// ═══ NORT FORCE SIMULATION ═══
// Zone-constrained force layout, no D3

var _forceSettled = false;
var _settledFrames = 0;
var _forceLastNodeCount = 0;
var _forceLastEdgeCount = 0;
var _forceLastStateHash = '';

function resetForceSettled() {
  _forceSettled = false;
  _settledFrames = 0;
}

function tickForce(nodeArr, edgeArr, W, H, dt) {
  // Auto-detect graph changes and reset settled state
  var stateHash = '';
  for (var ci = 0; ci < nodeArr.length; ci++) stateHash += nodeArr[ci].state;
  if (nodeArr.length !== _forceLastNodeCount || edgeArr.length !== _forceLastEdgeCount || stateHash !== _forceLastStateHash) {
    _forceLastNodeCount = nodeArr.length;
    _forceLastEdgeCount = edgeArr.length;
    _forceLastStateHash = stateHash;
    resetForceSettled();
  }
  if (_forceSettled) return;
  var zoneCount = 4;
  var zoneH = H / (zoneCount + 1);

  for (var i = 0; i < nodeArr.length; i++) {
    var n = nodeArr[i];

    // 1. NEXUS pinned to top center
    if (n.tier === 'nexus') {
      n.x = W / 2;
      n.y = zoneH * 0.8;
      n.vx = 0;
      n.vy = 0;
      continue;
    }

    // 2. Zone attraction — pull toward zone's Y band
    var targetY = zoneH * (n.zone + 1);
    var dy = targetY - n.y;
    n.vy += dy * FORCE_CFG.centerStrength * dt * 60;

    // Horizontal centering (mild)
    var dx = (W / 2) - n.x;
    n.vx += dx * FORCE_CFG.centerStrength * 0.3 * dt * 60;

    // 3. Repulsion from other nodes (charge force, inverse-square)
    for (var j = 0; j < nodeArr.length; j++) {
      if (i === j) continue;
      var other = nodeArr[j];
      var rx = n.x - other.x;
      var ry = n.y - other.y;
      var distSq = rx * rx + ry * ry;
      var minDist = FORCE_CFG.collide;
      if (distSq < 1) distSq = 1;
      if (distSq < minDist * minDist * 4) {
        var dist = Math.sqrt(distSq);
        var force = (FORCE_CFG.charge * dt * 60) / distSq;
        n.vx -= (rx / dist) * force;
        n.vy -= (ry / dist) * force;
      }
    }
  }

  // 4. Spring force along edges (build lookup map for O(1) access)
  var _nodeById = {};
  for (var ni = 0; ni < nodeArr.length; ni++) _nodeById[nodeArr[ni].id] = nodeArr[ni];
  for (var e = 0; e < edgeArr.length; e++) {
    var edge = edgeArr[e];
    var a = _nodeById[edge.from];
    var b = _nodeById[edge.to];
    if (!a || !b) continue;
    var ex = b.x - a.x;
    var ey = b.y - a.y;
    var eDist = Math.sqrt(ex * ex + ey * ey) || 1;
    var stretch = (eDist - FORCE_CFG.linkDist) / eDist;
    var springF = stretch * FORCE_CFG.linkStrength * dt * 60;
    if (a.tier !== 'nexus') {
      a.vx += ex * springF;
      a.vy += ey * springF;
    }
    if (b.tier !== 'nexus') {
      b.vx -= ex * springF;
      b.vy -= ey * springF;
    }
  }

  // 5. Velocity damping and bounds
  for (var m = 0; m < nodeArr.length; m++) {
    var nd = nodeArr[m];
    if (nd.tier === 'nexus') continue;
    nd.vx *= (1 - FORCE_CFG.velDecay * dt * 60);
    nd.vy *= (1 - FORCE_CFG.velDecay * dt * 60);
    nd.x += nd.vx * dt;
    nd.y += nd.vy * dt;

    // Bounds
    var pad = 60;
    if (nd.x < pad) { nd.x = pad; nd.vx *= -0.5; }
    if (nd.x > W - pad) { nd.x = W - pad; nd.vx *= -0.5; }
    if (nd.y < pad) { nd.y = pad; nd.vy *= -0.5; }
    if (nd.y > H - pad) { nd.y = H - pad; nd.vy *= -0.5; }
  }

  // Check if simulation has settled (max velocity below threshold)
  var maxVel = 0;
  for (var si = 0; si < nodeArr.length; si++) {
    var sn = nodeArr[si];
    var vel = Math.abs(sn.vx) + Math.abs(sn.vy);
    if (vel > maxVel) maxVel = vel;
  }
  if (maxVel < 0.5) {
    _settledFrames++;
    if (_settledFrames >= 60) _forceSettled = true;
  } else {
    _settledFrames = 0;
  }
}
