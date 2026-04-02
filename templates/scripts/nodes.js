// ═══ NORT NODE DATA MODEL ═══

function NortNode(id, name, tier, label, parentId) {
  this.id = id;
  this.name = name;
  this.tier = tier || 'drone';
  this.label = label || name;
  this.x = 0;
  this.y = 0;
  this.vx = 0;
  this.vy = 0;
  this.state = 'pending';
  this.opacity = 0;
  this.scale = 0;
  this.taskId = null;
  this.model = null;
  this.tokens = 0;
  this.toolCalls = [];
  this.parentId = parentId || null;
  this.spawnTime = currentTime || 0;
  this.completedTime = null;

  var t = TIERS[this.tier] || TIERS.drone;
  this.radius = t.radius;
  this.zone = t.zone;
  this.icon = t.icon;
}

var nodes = new Map();
var edges = [];
var particles = [];
var effects = [];
var config = {
  bloom: true,
  hexGrid: true,
  particles: true,
  nodeStats: false,
  locations: true,
  toolCards: true,
  edgeLabels: true,
  completionFx: true,
  sound: false,
  weather: true,
  minimap: true,
  roster: true,
  daynight: true,
  hexNodes: false,
  taskArrows: true,
  activeIndicators: true,
  // Performance settings
  qualityPreset: 'high',
  shadowQuality: 'high',
  bloomQuality: 'low',
  maxParticles: 50,
  maxEffects: 30,
  maxTrailLength: 30,
  edgeDetail: 16,
  lodEnabled: true,
  idlePause: true,
  viewportCulling: true,
};

function addNode(id, name, tier, label, parentId) {
  if (nodes.has(id)) return nodes.get(id);
  var n = new NortNode(id, name, tier, label, parentId);
  // Position near parent or canvas center
  var canvas = document.getElementById('canvas');
  var cx = canvas ? (canvas.width / (2 * (window.devicePixelRatio || 1))) : 400;
  var cy = canvas ? (canvas.height / (2 * (window.devicePixelRatio || 1))) : 300;
  var zoneH = (canvas ? canvas.height / (window.devicePixelRatio || 1) : 600) / 4;
  // Start in the correct zone band
  var targetY = zoneH * (n.zone + 0.5);
  if (parentId && nodes.has(parentId)) {
    var p = nodes.get(parentId);
    var angle = Math.random() * Math.PI * 2;
    n.x = p.x + Math.cos(angle) * 120;
    n.y = p.y + Math.sin(angle) * 120;
  } else {
    n.x = cx + (Math.random() - 0.5) * 200;
    n.y = targetY + (Math.random() - 0.5) * 40;
  }
  // On state replay (reconnect), make nodes visible immediately
  if (_isReplay) {
    n.opacity = 1;
    n.scale = 1;
  }
  nodes.set(id, n);
  if (!_isReplay) {
    spawnEffect(n.x, n.y, getStateColor(n.state));
  }
  return n;
}

// Flag to suppress spawn effects during state replay on reconnect
var _isReplay = false;

function removeNode(id) {
  var n = nodes.get(id);
  if (!n) return;
  spawnEffect(n.x, n.y, C.done);
  nodes.delete(id);
  edges = edges.filter(function(e) { return e.from !== id && e.to !== id; });
}

function getNodeByAgent(agentName) {
  for (var entry of nodes) {
    if (entry[1].name === agentName) return entry[1];
  }
  return null;
}

function spawnParticle(fromId, toId, color, label) {
  if (particles.length >= (config.maxParticles || 50)) return;
  particles.push({
    from: fromId,
    to: toId,
    color: color || C.dispatch,
    label: label || '',
    progress: 0,
    speed: ANIM.particleSpeed,
    size: 3,
  });
}

function spawnEffect(x, y, color) {
  if (effects.length >= (config.maxEffects || 30)) return;
  effects.push({
    type: 'spawn',
    x: x,
    y: y,
    color: color || C.holoBase,
    age: 0,
    duration: 0.8,
  });
}

function spawnCompleteEffect(x, y, color) {
  if (effects.length >= (config.maxEffects || 30)) return;
  effects.push({
    type: 'complete',
    x: x,
    y: y,
    color: color || C.done,
    age: 0,
    duration: 1.0,
  });
}

// ── City State Serialization ────────────────────────────────────────────────

function serializeCityState() {
  var result = { nodes: {}, savedAt: new Date().toISOString() };
  for (var entry of nodes) {
    var id = entry[0];
    var n = entry[1];
    result.nodes[id] = { x: n.x, y: n.y, state: n.state };
  }
  return result;
}

function deserializeCityState(data) {
  if (!data || !data.nodes) return;
  for (var entry of nodes) {
    var id = entry[0];
    var n = entry[1];
    var saved = data.nodes[id];
    if (saved) {
      if (typeof saved.x === 'number') n.x = saved.x;
      if (typeof saved.y === 'number') n.y = saved.y;
      // Do NOT restore state — server state is authoritative
    }
  }
}
