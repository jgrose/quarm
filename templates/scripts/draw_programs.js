// ═══ TRON LEGACY — ISOMETRIC 16-BIT AMBIENT PROGRAMS ═══
// Sim City SNES–style isometric pixel-art with 4-directional walk

var ambientPrograms = [];
var PROGRAM_COUNT = 6;
var PX = 3;

// Palette: 0=transparent 1=suit body 2=shadow 3=visor 4=circuit glow 5=highlight 6=disc 7=outline 8=circuit dim
// Sprite decoder: string of digit chars → 2D array of ints
function _ds(s, w) { var a = []; for (var i = 0; i < s.length; i += w) { var r = []; for (var j = 0; j < w; j++) r.push(s.charCodeAt(i + j) - 48); a.push(r); } return a; }

// Drone sprites (16w × 24h)
var SPRITE_SE_STAND = _ds("000007777700000000007555517000000007555551170000000733332217000000071122221700000000711112700000000007414700000000075141412700000075114141127000075171848172270007147146417217000711718481721700007707414707700000000781870000000000711111700000000741707127000000071170722700000007127072170000000741707247000000751170721270000071417072417000007114177111700000077777777700000000000000000000", 16);
var SPRITE_SE_WALK_A = _ds("000007777700000000007555517000000007555551170000000733332217000000071122221700000000711112700000000007414700000000075141412700000075114141127000751171848170000071417146417217000717018481724700007007414707270000000781870077000000711111700000000071700727000000074700007270000071700007227000071170000724700007517000721270000714170072417000071141777111700000777777777700000000000000000000", 16);
var SPRITE_SE_WALK_MID = _ds("000007777700000000007555517000000007555551170000000733332217000000071122221700000000711112700000000007414700000000075141412700000075114141127000071171848172170000747146417170000071018481027000000707414707000000000781870000000000711111700000000074171270000000007117227000000000712721700000000074172470000000075117212700000007141724170000000711411117000000007777777000000000000000000000", 16);
var SPRITE_SE_WALK_B = _ds("000007777700000000007555517000000007555551170000000733332217000000071122221700000000711112700000000007414700000000075141412700000075114141127000075171848172270007147146417217000741718481071700072707414700700007700781870000000000711111700000000717007227000000071170072270000007127000717000000741700074170000075117072117000007141707241700000711417711170000007777777770000000000000000000", 16);
var SPRITE_NE_STAND = _ds("000007777700000000007111127000000007111112270000000711112227000000075111222700000000711112700000000007414700000000075141412700000075114141127000075171868172270007147146417217000711718681721700007707414707700000000781870000000000711111700000000741707127000000071170722700000007127072170000000741707247000000751170721270000071417072417000007114177111700000077777777700000000000000000000", 16);
var SPRITE_NE_WALK_A = _ds("000007777700000000007111127000000007111112270000000711112227000000075111222700000000711112700000000007414700000000075141412700000075114141127000751171868170000071417146417217000717018681724700007007414707270000000781870077000000711111700000000071700727000000074700007270000071700007227000071170000724700007517000721270000714170072417000071141777111700000777777777700000000000000000000", 16);
var SPRITE_NE_WALK_MID = _ds("000007777700000000007111127000000007111112270000000711112227000000075111222700000000711112700000000007414700000000075141412700000075114141127000071171868172170000747146417170000071018681027000000707414707000000000781870000000000711111700000000074171270000000007117227000000000712721700000000074172470000000075117212700000007141724170000000711411117000000007777777000000000000000000000", 16);
var SPRITE_NE_WALK_B = _ds("000007777700000000007111127000000007111112270000000711112227000000075111222700000000711112700000000007414700000000075141412700000075114141127000075171868172270007147146417217000741718681071700072707414700700007700781870000000000711111700000000717007227000000071170072270000007127000717000000741700074170000075117072117000007141707241700000711417711170000007777777770000000000000000000", 16);
// Sentinel sprites (18w × 26h)
var SPRITE_SENTINEL_SE_STAND = _ds("000000777777000000000007555551700000000075555511170000000073333322170000000075122222170000000007111112700000000000741417000000007751414141277000075511414141122700755171848481722270754171464641721270071171848481721700007707414147077000000007818187000000000071111111700000000741170712270000000711170722170000000712170721170000000741170724170000007511170721217000007141170724117000007114117711117000007774848484777000000777777777770000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_SE_WALK_A = _ds("000000777777000000000007555551700000000075555511170000000073333322170000000075122222170000000007111112700000000000741417000000007751414141277000075511414141122700755171848481700000754171464641721700071701848481724700007007414147072700000007818187007700000071111111700000000071170072700000000741700007270000007117000072270000071117000072470000075117000721270000071411700724170000071141177711170000077748484847770000000777777777700000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_SE_WALK_MID = _ds("000000777777000000000007555551700000000075555511170000000073333322170000000075122222170000000007111112700000000000741417000000007751414141277000075511414141122700751171848481721700074171464641717000007101848481027000000707414147070000000007818187000000000071111111700000000074117122700000000071117221700000000071217211700000000074117241700000000751117212170000000714117241170000000711411111170000000774848484770000000077777777700000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_SE_WALK_B = _ds("000000777777000000000007555551700000000075555511170000000073333322170000000075122222170000000007111112700000000000741417000000007751414141277000075511414141122700755171848481722700754171464641721700741171848481071700727007414147007000770007818187000000000071111111700000000711700722170000000711170072270000000712170007170000000741170007417000000751117072117000000714117072417000000711411771117000000774848484777000000077777777770000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_NE_STAND = _ds("000000777777000000000007111112700000000071111122270000000071111122270000000075111222170000000007111112700000000000741417000000007751414141277000075511414141122700755171868681722270754171464641721270071171868681721700007707414147077000000007818187000000000071111111700000000741170712270000000711170722170000000712170721170000000741170724170000007511170721217000007141170724117000007114117711117000007774848484777000000777777777770000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_NE_WALK_A = _ds("000000777777000000000007111112700000000071111122270000000071111122270000000075111222170000000007111112700000000000741417000000007751414141277000075511414141122700755171868681700000754171464641721700071701868681724700007007414147072700000007818187007700000071111111700000000071170072700000000741700007270000007117000072270000071117000072470000075117000721270000071411700724170000071141177711170000077748484847770000000777777777700000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_NE_WALK_MID = _ds("000000777777000000000007111112700000000071111122270000000071111122270000000075111222170000000007111112700000000000741417000000007751414141277000075511414141122700751171868681721700074171464641717000007101868681027000000707414147070000000007818187000000000071111111700000000074117122700000000071117221700000000071217211700000000074117241700000000751117212170000000714117241170000000711411111170000000774848484770000000077777777700000000000000000000000000000000000000000", 18);
var SPRITE_SENTINEL_NE_WALK_B = _ds("000000777777000000000007111112700000000071111122270000000071111122270000000075111222170000000007111112700000000000741417000000007751414141277000075511414141122700755171868681722700754171464641721700741171868681071700727007414147007000770007818187000000000071111111700000000711700722170000000711170072270000000712170007170000000741170007417000000751117072117000000714117072417000000711411771117000000774848484777000000077777777770000000000000000000000000000000000000000", 18);
// Probe sprites (12w × 18h)
var SPRITE_PROBE_SE_STAND = _ds("000006000000000004000000000077770000000755517000000733227000000711227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000040040000000000000000", 12);
var SPRITE_PROBE_SE_WALK_A = _ds("000000000000000006000000000004000000000077770000000755517000000733227000000711227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000040040000000000000000", 12);
var SPRITE_PROBE_SE_WALK_MID = _ds("000006000000000004000000000077770000000755517000000733227000000711227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000000000000000000000000", 12);
var SPRITE_PROBE_SE_WALK_B = _ds("000006000000000004000000000077770000000755517000000733227000000711227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000040040000000000000000", 12);
var SPRITE_PROBE_NE_STAND = _ds("000006000000000004000000000077770000000711127000000711227000000751227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000040040000000000000000", 12);
var SPRITE_PROBE_NE_WALK_A = _ds("000000000000000006000000000004000000000077770000000711127000000711227000000751227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000040040000000000000000", 12);
var SPRITE_PROBE_NE_WALK_MID = _ds("000006000000000004000000000077770000000711127000000711227000000751227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000000000000000000000000", 12);
var SPRITE_PROBE_NE_WALK_B = _ds("000006000000000004000000000077770000000711127000000711227000000751227000000071170000000074470000007514412700075174472170071478872470007774477700000078870000000711117000000074470000000008800000000040040000000000000000", 12);

// ─── Direction-indexed walk frames ───
// SE/SW use SE sprites, NE/NW use NE sprites. SW/NW flip horizontally.
var WALK_FRAMES_SE = [SPRITE_SE_WALK_MID, SPRITE_SE_WALK_A, SPRITE_SE_WALK_MID, SPRITE_SE_WALK_B];
var WALK_FRAMES_NE = [SPRITE_NE_WALK_MID, SPRITE_NE_WALK_A, SPRITE_NE_WALK_MID, SPRITE_NE_WALK_B];

var WALK_FRAMES_SE_SENTINEL = [SPRITE_SENTINEL_SE_WALK_MID, SPRITE_SENTINEL_SE_WALK_A, SPRITE_SENTINEL_SE_WALK_MID, SPRITE_SENTINEL_SE_WALK_B];
var WALK_FRAMES_NE_SENTINEL = [SPRITE_SENTINEL_NE_WALK_MID, SPRITE_SENTINEL_NE_WALK_A, SPRITE_SENTINEL_NE_WALK_MID, SPRITE_SENTINEL_NE_WALK_B];

var WALK_FRAMES_SE_PROBE = [SPRITE_PROBE_SE_WALK_MID, SPRITE_PROBE_SE_WALK_A, SPRITE_PROBE_SE_WALK_MID, SPRITE_PROBE_SE_WALK_B];
var WALK_FRAMES_NE_PROBE = [SPRITE_PROBE_NE_WALK_MID, SPRITE_PROBE_NE_WALK_A, SPRITE_PROBE_NE_WALK_MID, SPRITE_PROBE_NE_WALK_B];

// Keep legacy alias for sprite cache key lookup
var WALK_FRAMES = WALK_FRAMES_SE;

// ─── Sprite cache (pre-rendered per color+frame+direction) ───
var _spriteCache = {};

function _renderSpriteToCanvas(frame, glowColor, scale) {
  // Use frame reference as part of key — check all frame arrays
  var frameKey = '_f';
  if (WALK_FRAMES_SE.indexOf(frame) >= 0) frameKey += 'se' + WALK_FRAMES_SE.indexOf(frame);
  else if (WALK_FRAMES_NE.indexOf(frame) >= 0) frameKey += 'ne' + WALK_FRAMES_NE.indexOf(frame);
  else if (WALK_FRAMES_SE_SENTINEL.indexOf(frame) >= 0) frameKey += 'sse' + WALK_FRAMES_SE_SENTINEL.indexOf(frame);
  else if (WALK_FRAMES_NE_SENTINEL.indexOf(frame) >= 0) frameKey += 'sne' + WALK_FRAMES_NE_SENTINEL.indexOf(frame);
  else if (WALK_FRAMES_SE_PROBE.indexOf(frame) >= 0) frameKey += 'pse' + WALK_FRAMES_SE_PROBE.indexOf(frame);
  else if (WALK_FRAMES_NE_PROBE.indexOf(frame) >= 0) frameKey += 'pne' + WALK_FRAMES_NE_PROBE.indexOf(frame);
  else frameKey += 'st' + frame.length + 'x' + frame[0].length;
  var key = glowColor + '_' + scale.toFixed(2) + frameKey;
  if (_spriteCache[key]) return _spriteCache[key];

  var rows = frame.length;
  var cols = frame[0].length;
  var px = PX * scale;
  var c = document.createElement('canvas');
  c.width = Math.ceil(cols * px);
  c.height = Math.ceil(rows * px);
  var cx = c.getContext('2d');
  cx.imageSmoothingEnabled = false;

  var gr = parseInt(glowColor.slice(1,3),16);
  var gg = parseInt(glowColor.slice(3,5),16);
  var gb = parseInt(glowColor.slice(5,7),16);

  for (var r = 0; r < rows; r++) {
    for (var cl = 0; cl < cols; cl++) {
      var v = frame[r][cl];
      if (v === 0) continue;
      var fr, fg, fb;
      switch (v) {
        case 7: fr = Math.floor(gr*0.08); fg = Math.floor(gg*0.08); fb = Math.floor(gb*0.08+15); break;
        case 2: fr = Math.floor(gr*0.12+8); fg = Math.floor(gg*0.12+8); fb = Math.floor(gb*0.12+18); break;
        case 1: fr = Math.floor(gr*0.18+15); fg = Math.floor(gg*0.18+15); fb = Math.floor(gb*0.18+30); break;
        case 5: fr = Math.floor(gr*0.3+25); fg = Math.floor(gg*0.3+25); fb = Math.floor(gb*0.3+45); break;
        case 8: fr = Math.floor(gr*0.5); fg = Math.floor(gg*0.5); fb = Math.floor(gb*0.5); break;
        case 4: fr = gr; fg = gg; fb = gb; break;
        case 3: fr = Math.min(255,gr+180); fg = Math.min(255,gg+180); fb = Math.min(255,gb+180); break;
        case 6: fr = Math.min(255,Math.floor(gr*0.8+100)); fg = Math.min(255,Math.floor(gg*0.8+100)); fb = Math.min(255,Math.floor(gb*0.8+100)); break;
        default: fr = gr; fg = gg; fb = gb;
      }
      cx.fillStyle = 'rgb(' + fr + ',' + fg + ',' + fb + ')';
      cx.fillRect(Math.floor(cl * px), Math.floor(r * px), Math.ceil(px), Math.ceil(px));
    }
  }

  _spriteCache[key] = c;
  return c;
}

function _getStandSprite(glowColor, scale, dir, tier) {
  var frame;
  var isNE = (dir === 'ne' || dir === 'nw');
  if (tier === 'sentinel') {
    frame = isNE ? SPRITE_SENTINEL_NE_STAND : SPRITE_SENTINEL_SE_STAND;
  } else if (tier === 'probe') {
    frame = isNE ? SPRITE_PROBE_NE_STAND : SPRITE_PROBE_SE_STAND;
  } else {
    frame = isNE ? SPRITE_NE_STAND : SPRITE_SE_STAND;
  }
  return _renderSpriteToCanvas(frame, glowColor, scale);
}

function _getWalkFrames(dir, tier) {
  var isNE = (dir === 'ne' || dir === 'nw');
  if (tier === 'sentinel') return isNE ? WALK_FRAMES_NE_SENTINEL : WALK_FRAMES_SE_SENTINEL;
  if (tier === 'probe') return isNE ? WALK_FRAMES_NE_PROBE : WALK_FRAMES_SE_PROBE;
  return isNE ? WALK_FRAMES_NE : WALK_FRAMES_SE;
}

// ─── Direction from movement vector ───

function _getDirection(dx, dy) {
  if (dx >= 0 && dy >= 0) return 'se';
  if (dx < 0 && dy >= 0) return 'sw';
  if (dx >= 0 && dy < 0) return 'ne';
  return 'nw';
}

function _needsFlip(dir) {
  return dir === 'sw' || dir === 'nw';
}

// ─── Location helpers ───

function _releaseLocation(p) {
  if (p.atLocation) {
    var idx = p.atLocation.occupants.indexOf(p);
    if (idx >= 0) p.atLocation.occupants.splice(idx, 1);
    p.atLocation = null;
    p.landingSlot = -1;
  }
}

function _assignToLocation(p, loc) {
  _releaseLocation(p);
  var slot = 0;
  for (var i = 0; i < loc.landingOffsets.length; i++) {
    var taken = false;
    for (var j = 0; j < loc.occupants.length; j++) {
      if (loc.occupants[j].landingSlot === i) { taken = true; break; }
    }
    if (!taken) { slot = i; break; }
  }
  p.locationTarget = loc;
  p.landingSlot = slot;
  p.targetX = loc.x + loc.landingOffsets[slot].dx;
  p.targetY = loc.y + loc.landingOffsets[slot].dy;
}

function _pickRandomHex(p, W, H) {
  var col = Math.floor(Math.random() * 48) + 1;
  var row = Math.floor(Math.random() * 48) + 1;
  var pos = isoToScreen(col, row);
  p.targetX = pos.x;
  p.targetY = pos.y;
  p.locationTarget = null;
}

function _pickTarget(p, W, H) {
  if (p.assignedTask) {
    var workLoc = typeof getWorkLocationForState === 'function'
      ? getWorkLocationForState(p.assignedTask.status) : null;
    if (workLoc && workLoc.occupants.length < workLoc.capacity) {
      _assignToLocation(p, workLoc);
      return;
    }
  }
  if (typeof getRandomIdleLocation === 'function') {
    var loc = getRandomIdleLocation(p.atLocation);
    if (loc) { _assignToLocation(p, loc); return; }
  }
  _pickRandomHex(p, W, H);
}

var _PROG_GLOWS = ['#66ccff','#cc88ff','#66ffaa','#ffbb44','#aaeeff','#ff8866',
                   '#ff88cc','#88ffcc','#ccff88','#88ccff','#ffcc88','#cc88ff'];

function _createProgram(glow, tier, agentName, displayName, sessionId) {
  var startPos = isoToScreen(Math.floor(Math.random() * 40) + 5, Math.floor(Math.random() * 40) + 5);
  var baseSpeed = 20 + Math.random() * 15;
  return { x: startPos.x, y: startPos.y, targetX: 0, targetY: 0, speed: baseSpeed,
    scale: 0.9 + Math.random() * 0.3, glow: glow, walkCycle: Math.random() * 10,
    idle: false, idleTimer: 0, direction: 'se', trail: [], _trailCounter: 0,
    locationTarget: null, atLocation: null, landingSlot: -1, assignedTask: null,
    programState: 'idle', emotion: { current: 'idle', target: 'idle', alpha: 1.0, transitionAge: 0 },
    tier: tier || 'drone', agentName: agentName || null,
    displayName: displayName || null, sessionId: sessionId || null,
    bunkerState: 'walking', enterProgress: 0,
    exitProgress: 0, visible: true, cycleMode: false, _baseSpeed: baseSpeed };
}

function initAmbientPrograms(W, H) {
  ambientPrograms = [];
  var tiers = ['sentinel', 'sentinel', 'drone', 'drone', 'probe', 'probe'];
  for (var i = 0; i < PROGRAM_COUNT; i++) {
    var p = _createProgram(_PROG_GLOWS[i % 6], tiers[i] || 'drone');
    ambientPrograms.push(p);
    _pickTarget(p, W, H);
  }
}

// ─── Pending-question "needs help" indicator ───

function _programHasPendingQuestion(p) {
  if (typeof _pendingQuestions === 'undefined' || !_pendingQuestions.size) return false;
  var name = (p && p.agentName) ? p.agentName : '';
  if (!name) return false;
  var hit = false;
  _pendingQuestions.forEach(function (q) {
    if (!hit && q.agent && q.agent.toLowerCase() === name.toLowerCase()) hit = true;
  });
  return hit;
}

function _drawHelpIndicator(ctx, x, y, time) {
  // Amber pulse ring around the sprite.
  var pulse = 0.5 + 0.5 * Math.sin(time / 0.9);
  ctx.save();
  ctx.globalAlpha = 0.35 + 0.45 * pulse;
  ctx.beginPath();
  ctx.arc(x, y, 18 + 4 * pulse, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,200,60,' + (0.6 + 0.3 * pulse) + ')';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  // Beacon beam.
  ctx.save();
  var grad = ctx.createLinearGradient(x, y - 12, x, y - 72);
  grad.addColorStop(0, 'rgba(95,220,255,0.55)');
  grad.addColorStop(1, 'rgba(95,220,255,0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(x - 5, y - 72, 10, 60);
  ctx.restore();

  // "?!" chip.
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#ff5f8c';
  ctx.fillRect(x - 10, y - 34, 20, 14);
  ctx.fillStyle = '#fff';
  ctx.font = '900 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('?!', x, y - 27);
  ctx.restore();
}

// ─── Pixel light trail ───

function _drawPixelTrail(ctx, trail, color, scale, cycleMode) {
  if (trail.length < 2) return;
  var px = PX * scale;
  ctx.save();
  for (var i = 1; i < trail.length; i++) {
    var t = i / trail.length;
    var alpha = t * (cycleMode ? 0.25 : 0.15);
    ctx.fillStyle = hexToRgba(color, alpha);
    var tw = cycleMode ? Math.max(Math.ceil(px * 1.5), 3) : Math.max(Math.ceil(px), 2);
    var th = Math.ceil(px * 0.5);
    ctx.fillRect(
      Math.floor(trail[i].x / px) * px,
      Math.floor(trail[i].y / px) * px,
      tw, th
    );
  }
  ctx.restore();
}

// ─── Update ───

function updateAmbientPrograms(W, H, dt) {
  if (!ambientPrograms.length) initAmbientPrograms(W, H);

  for (var i = 0; i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];

    // ─── Bunker state machine (Feature 1) ───
    if (p.bunkerState === 'entering') {
      p.enterProgress += dt * 2;
      if (p.enterProgress >= 1) {
        p.visible = false;
        p.bunkerState = 'inside';
        p.trail = [];
        p.idle = true;
        p.idleTimer = 5 + Math.random() * 8;
      }
      continue;
    }

    if (p.bunkerState === 'inside') {
      p.idleTimer -= dt;
      if (p.idleTimer <= 0) {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      }
      continue;
    }

    if (p.bunkerState === 'exiting') {
      p.exitProgress += dt * 2;
      if (p.exitProgress >= 1) {
        p.bunkerState = 'leaving_door';
        p.exitProgress = 1;
        // Set target 60px away from building
        if (p.atLocation) {
          var angle = Math.random() * Math.PI * 2;
          p._leaveX = p.atLocation.x + Math.cos(angle) * 60;
          p._leaveY = p.atLocation.y + Math.sin(angle) * 60;
        } else {
          p._leaveX = p.x + 60;
          p._leaveY = p.y;
        }
      }
      continue;
    }

    if (p.bunkerState === 'leaving_door') {
      var ldx = p._leaveX - p.x;
      var ldy = p._leaveY - p.y;
      var ldist = Math.sqrt(ldx * ldx + ldy * ldy);
      if (ldist < 5) {
        _releaseLocation(p);
        p.idle = false;
        p.bunkerState = 'walking';
        _pickTarget(p, W, H);
      } else {
        p.x += (ldx / ldist) * p.speed * dt;
        p.y += (ldy / ldist) * p.speed * dt;
        if (ldist > 3) p.direction = _getDirection(ldx, ldy);
        p.walkCycle += dt * 7;
      }
      continue;
    }

    // ─── Normal idle handling ───
    if (p.idle) {
      p.idleTimer -= dt;
      if (p.idleTimer <= 0) {
        p.idle = false;
        _releaseLocation(p);
        p.bunkerState = 'walking';
        _pickTarget(p, W, H);
      }
      continue;
    }

    var dx = p.targetX - p.x;
    var dy = p.targetY - p.y;
    var dist = Math.sqrt(dx * dx + dy * dy);

    // ─── Light cycle mode (Feature 3) ───
    if (dist > 200 && !p.cycleMode) {
      p.cycleMode = true;
      p.speed = p._baseSpeed * 2;
      p.trail = []; // clear trail on mode switch to avoid artifacts
    } else if (dist < 100 && p.cycleMode) {
      p.cycleMode = false;
      p.speed = p._baseSpeed;
      p.trail = [];
    }

    var maxTrail = p.cycleMode
      ? Math.floor((config.maxTrailLength || 30) * 1.3)
      : (config.maxTrailLength || 30);

    if (dist < 5) {
      // ─── Arrived at target: enter bunker if location ───
      if (p.locationTarget) {
        p.atLocation = p.locationTarget;
        p.atLocation.occupants.push(p);
        p.locationTarget = null;
        // Start bunker entry sequence
        p.bunkerState = 'entering';
        p.enterProgress = 0;
        p.cycleMode = false;
        p.speed = p._baseSpeed;
        p.trail = []; // clear trail when entering building
      } else {
        p.idle = true;
        p.idleTimer = 2 + Math.random() * 4;
      }
      continue;
    }

    // Move toward target (direct interpolation for iso)
    var len = dist;
    p.x += (dx / len) * p.speed * dt;
    p.y += (dy / len) * p.speed * dt;

    // Update direction from movement
    if (dist > 3) p.direction = _getDirection(dx, dy);

    p.walkCycle += dt * 7;

    // Skip trail recording at low zoom — trails invisible below 0.4 anyway
    if (!config.lodEnabled || camera.zoom >= 0.4) {
      p._trailCounter += dt;
      if (p._trailCounter > 0.08) {
        // Add slight wobble so trails curve instead of being laser-straight
        var wobble = p.cycleMode ? 1.5 : 3.0;
        p.trail.push({
          x: p.x + (Math.random() - 0.5) * wobble,
          y: p.y + (Math.random() - 0.5) * wobble * 0.5,
        });
        if (p.trail.length > maxTrail) p.trail.shift();
        p._trailCounter = 0;
      }
    } else if (p.trail.length > 0) {
      // Drain trail when zoomed out so memory is reclaimed
      p.trail.length = 0;
    }

    // Update emotion state for thought bubbles
    if (typeof updateEmotionState === 'function') updateEmotionState(p, dt);
  }
}

// ─── Draw ───

function drawAmbientPrograms(ctx, time) {
  if (!ambientPrograms.length) return;

  // Viewport culling — get visible rect for off-screen skip
  var _progCanvas = document.getElementById('canvas');
  var _progDpr = window.devicePixelRatio || 1;
  var _progW = _progCanvas ? _progCanvas.width / _progDpr : 800;
  var _progH = _progCanvas ? _progCanvas.height / _progDpr : 600;
  var progView = (typeof getVisibleRect === 'function') ? getVisibleRect(_progW, _progH) : { x: 0, y: 0, w: _progW, h: _progH };
  var progMargin = 80;

  // Merge programs + trees + buildings into one Y-sorted draw list for proper depth ordering
  var drawList = [];
  for (var pi = 0; pi < ambientPrograms.length; pi++) {
    drawList.push({ type: 'program', obj: ambientPrograms[pi], y: ambientPrograms[pi].y });
  }
  if (typeof _treeObjects !== 'undefined') {
    for (var ti = 0; ti < _treeObjects.length; ti++) {
      drawList.push({ type: 'tree', obj: _treeObjects[ti], y: _treeObjects[ti].y });
    }
  }
  if (typeof gridLocations !== 'undefined') {
    for (var li = 0; li < gridLocations.length; li++) {
      drawList.push({ type: 'building', obj: gridLocations[li], y: gridLocations[li].y });
    }
  }
  var sorted = drawList.sort(function(a, b) { return a.y - b.y; });

  ctx.save();
  ctx.imageSmoothingEnabled = false;

  for (var i = 0; i < sorted.length; i++) {
    var item = sorted[i];

    // Draw trees inline for proper depth sorting
    if (item.type === 'tree') {
      var tt = item.obj;
      if (tt.x < progView.x - progMargin || tt.x > progView.x + progView.w + progMargin ||
          tt.y < progView.y - progMargin || tt.y > progView.y + progView.h + progMargin) continue;
      if (typeof drawSingleTree === 'function') drawSingleTree(ctx, tt, time);
      continue;
    }

    // Draw buildings inline for proper depth sorting
    if (item.type === 'building') {
      var bl = item.obj;
      if (bl.x < progView.x - 120 || bl.x > progView.x + progView.w + 120 ||
          bl.y < progView.y - 120 || bl.y > progView.y + progView.h + 120) continue;
      if (typeof _drawSingleLocation === 'function') _drawSingleLocation(ctx, bl, time);
      continue;
    }

    var p = item.obj;

    // Skip invisible programs (inside bunker)
    if (p.visible === false) continue;

    // Skip off-screen programs (viewport culling)
    if (p.x < progView.x - progMargin || p.x > progView.x + progView.w + progMargin ||
        p.y < progView.y - progMargin || p.y > progView.y + progView.h + progMargin) continue;

    // Calculate bunker entry/exit scale and alpha modifiers
    var bunkerScale = 1.0;
    var bunkerAlpha = 1.0;
    if (p.bunkerState === 'entering') {
      bunkerScale = 1.0 - p.enterProgress;
      bunkerAlpha = 1.0 - p.enterProgress;
    } else if (p.bunkerState === 'exiting') {
      bunkerScale = p.exitProgress;
      bunkerAlpha = p.exitProgress;
    }

    var effectiveScale = p.scale * bunkerScale;
    if (effectiveScale < 0.01) continue;

    // Pick sprite based on direction and tier
    var frames = _getWalkFrames(p.direction, p.tier);
    var flip = _needsFlip(p.direction);
    var sprite;

    if (p.idle || p.bunkerState === 'entering' || p.bunkerState === 'exiting') {
      sprite = _getStandSprite(p.glow, effectiveScale, p.direction, p.tier);
    } else {
      var frameIdx = Math.floor(p.walkCycle) % frames.length;
      sprite = _renderSpriteToCanvas(frames[frameIdx], p.glow, effectiveScale);
    }

    // Trail (skip at low zoom for performance)
    if (!config.lodEnabled || camera.zoom >= 0.4) {
      _drawPixelTrail(ctx, p.trail, p.glow, p.scale, p.cycleMode);
    }

    var drawX = Math.floor(p.x - sprite.width / 2);
    var drawY = Math.floor(p.y - sprite.height + 6);

    // Glow pass (shadow quality aware)
    var _progSq = config.shadowQuality || 'high';
    ctx.save();
    ctx.globalAlpha = bunkerAlpha;
    if (_progSq !== 'off') {
      ctx.shadowColor = p.glow;
      ctx.shadowBlur = _progSq === 'low' ? 6 : 18;
    }

    if (_progSq !== 'off') {
      if (flip) {
        ctx.save();
        ctx.translate(drawX + sprite.width, drawY);
        ctx.scale(-1, 1);
        ctx.drawImage(sprite, 0, 0);
        ctx.restore();
      } else {
        ctx.drawImage(sprite, drawX, drawY);
      }
      ctx.shadowBlur = 0;
    }

    // Crisp pass
    if (flip) {
      ctx.save();
      ctx.translate(drawX + sprite.width, drawY);
      ctx.scale(-1, 1);
      ctx.drawImage(sprite, 0, 0);
      ctx.restore();
    } else {
      ctx.drawImage(sprite, drawX, drawY);
    }
    ctx.restore();

    // Isometric ground shadow (skip at low zoom)
    if (!config.lodEnabled || camera.zoom >= 0.4) {
      var px = PX * p.scale;
      ctx.fillStyle = hexToRgba(p.glow, 0.06 * bunkerAlpha);
      drawIsoDiamond(ctx, p.x, p.y + 4, px * 8, px * 4);
      ctx.fill();
    }

    // ─── Task status icon above head (Feature 5) ───
    if (p.assignedTask && typeof getStateColor === 'function') {
      var taskColor = getStateColor(p.assignedTask.status);
      var iconPx = PX * 4;
      var bobY = Math.sin(time * 3) * 2;
      var iconX = Math.floor(p.x - iconPx / 2);
      var iconY = Math.floor(drawY - iconPx - 4 + bobY);
      ctx.save();
      ctx.globalAlpha = bunkerAlpha * 0.9;
      if (_progSq !== 'off') {
        ctx.shadowColor = taskColor;
        ctx.shadowBlur = _progSq === 'low' ? 3 : 6;
        ctx.fillStyle = taskColor;
        ctx.fillRect(iconX, iconY, iconPx, iconPx);
        ctx.shadowBlur = 0;
      }
      ctx.fillStyle = taskColor;
      ctx.fillRect(iconX, iconY, iconPx, iconPx);
      ctx.restore();
    }

    // Floating task label above head while traveling
    if (p.assignedTask && p.visible && p.bunkerState === 'walking') {
      ctx.save();
      ctx.fillStyle = hexToRgba(getStateColor(p.assignedTask.status), 0.7);
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      var labelY = drawY - 4 + Math.sin(time * 3) * 2;
      ctx.fillText(p.assignedTask.title ? p.assignedTask.title.slice(0, 20) : p.assignedTask.id, Math.floor(p.x), Math.floor(labelY));
      ctx.restore();
    }

    // Thought bubble emotion indicator
    if (typeof drawThoughtBubble === 'function') drawThoughtBubble(ctx, p, drawY, time, bunkerAlpha);

    if (_programHasPendingQuestion(p)) {
      _drawHelpIndicator(ctx, p.x, p.y, time);
    }
  }

  ctx.restore();
}

// ─── Work assignment (called from websocket.js) ───

function routeProgramsToTasks(tasks, sessionId) {
  if (!ambientPrograms.length) return;

  // Build agent->task map
  var taskByAgent = {};
  for (var t = 0; t < tasks.length; t++) {
    if (tasks[t].status !== 'pending') {
      taskByAgent[tasks[t].agent] = tasks[t];
    }
  }

  for (var i = 0; i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];
    // Only route programs belonging to this session (when sessionId provided).
    // Programs without a sessionId (idle placeholders) still get positional fallback.
    if (sessionId && p.sessionId && p.sessionId !== sessionId) continue;

    var task = p.agentName ? taskByAgent[p.agentName] : null;

    // Also try positional fallback for idle mode (no agentName set)
    if (!task && !p.agentName && i < tasks.length && tasks[i].status !== 'pending') {
      task = tasks[i];
    }

    if (task) {
      // Check if task changed
      if (p.assignedTask && p.assignedTask.id === task.id && p.assignedTask.status === task.status) continue;

      p.assignedTask = { id: task.id, status: task.status, title: task.title,
        lastScore: task.last_score || 0, revisionCount: task.revision_count || 0,
        managerNotes: task.manager_notes || '', reviewerNotes: task.reviewer_notes || '' };
      p.programState = 'working';

      // If inside a building, trigger exit first
      if (p.bunkerState === 'inside') {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      } else if (p.bunkerState === 'walking' || p.bunkerState === 'leaving_door') {
        _releaseLocation(p);
        _pickTarget(p, 0, 0);
      }
      // If entering/exiting, let it finish then re-route naturally
    } else if (p.assignedTask) {
      // Task completed or removed -- return to idle
      p.assignedTask = null;
      p.programState = 'idle';
      if (p.bunkerState === 'inside') {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      }
    }
  }
}

// ─── Roster sync (called from websocket.js when roster data arrives) ───

function syncProgramsToRoster(data) {
  var sid = data.session_id || 'default';
  var agents = [];

  // Build agent list with tiers
  if (data.managers) {
    for (var m = 0; m < data.managers.length; m++) {
      agents.push({ name: data.managers[m].name, title: data.managers[m].title, tier: 'sentinel' });
    }
  }
  if (data.sub_agents) {
    for (var s = 0; s < data.sub_agents.length; s++) {
      agents.push({ name: data.sub_agents[s].name, title: data.sub_agents[s].title, tier: 'drone' });
    }
  }
  if (data.reviewers) {
    for (var r = 0; r < data.reviewers.length; r++) {
      agents.push({ name: data.reviewers[r].name, title: data.reviewers[r].title, tier: 'probe' });
    }
  }

  if (agents.length === 0) return;

  // If there are any programs without a session assignment (e.g. initAmbientPrograms
  // placeholders from page load), adopt them into the first incoming session so they
  // get replaced rather than stranded.
  for (var adopt = 0; adopt < ambientPrograms.length; adopt++) {
    if (ambientPrograms[adopt].sessionId == null) ambientPrograms[adopt].sessionId = sid;
  }

  // Scope to programs belonging to this session
  var sessionPrograms = ambientPrograms.filter(function(p) { return p.sessionId === sid; });

  // Check if roster changed for this session (different agent count or names)
  var changed = agents.length !== sessionPrograms.length;
  if (!changed) {
    for (var i = 0; i < agents.length; i++) {
      if (!sessionPrograms[i] || sessionPrograms[i].agentName !== agents[i].name) {
        changed = true;
        break;
      }
    }
  }

  if (!changed) return; // roster unchanged for this session, skip respawn

  // Release locations for this session's programs, then remove them from ambientPrograms
  for (var ri = 0; ri < sessionPrograms.length; ri++) {
    _releaseLocation(sessionPrograms[ri]);
  }
  ambientPrograms = ambientPrograms.filter(function(p) { return p.sessionId !== sid; });

  // Append new programs for this session's roster
  for (var ai = 0; ai < agents.length; ai++) {
    var agent = agents[ai];
    var p = _createProgram(_PROG_GLOWS[ai % _PROG_GLOWS.length], agent.tier, agent.name, agent.title, sid);
    ambientPrograms.push(p);
    _pickTarget(p, 0, 0);
  }

  // Re-init roster if available
  if (typeof initRoster === 'function') initRoster();
}

// ─── Revert to idle programs when orchestrator finishes ───

function revertToIdlePrograms() {
  for (var i = 0; i < ambientPrograms.length; i++) {
    _releaseLocation(ambientPrograms[i]);
  }
  ambientPrograms = [];
  PROGRAM_COUNT = 6; // reset for clarity
  // Re-init will happen on next updateAmbientPrograms call
}

// ─── Remove programs for a single session (e.g. when that plan completes) ───

function removeProgramsForSession(sessionId) {
  if (!sessionId) return;
  for (var i = 0; i < ambientPrograms.length; i++) {
    if (ambientPrograms[i].sessionId === sessionId) {
      _releaseLocation(ambientPrograms[i]);
    }
  }
  ambientPrograms = ambientPrograms.filter(function(p) { return p.sessionId !== sessionId; });
}
