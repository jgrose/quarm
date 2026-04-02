// ═══ NORT ATMOSPHERE ENGINE ═══
// Day/night cycle that modulates visual atmosphere over a 120-second full cycle
// Phases: dawn(0-30s) → day(30-60s) → dusk(60-90s) → night(90-120s)

var ATM_CYCLE = 120; // seconds per full cycle
var atmPhase = 'night'; // current phase name
var atmProgress = 0; // 0-1 within current phase
var atmClock = 90; // start at night phase (90s into cycle)

// Exposed globals for other draw modules to reference
var atmGlowMult = 1.0;
var atmGridAlpha = 0.35;
var atmParticleMult = 1.0;

// Phase color palettes (lerp between these)
var ATM_COLORS = {
  night: { void: '#050510', gridAlpha: 0.35, glowMult: 1.0, particleMult: 1.0 },
  dawn:  { void: '#0a0518', gridAlpha: 0.25, glowMult: 0.85, particleMult: 0.8 },
  day:   { void: '#0a0a25', gridAlpha: 0.15, glowMult: 0.7, particleMult: 0.5 },
  dusk:  { void: '#0a0515', gridAlpha: 0.3, glowMult: 0.9, particleMult: 0.9 },
};

// Phase order for cycling
var ATM_PHASE_ORDER = ['dawn', 'day', 'dusk', 'night'];
var ATM_PHASE_DURATION = ATM_CYCLE / 4; // 30s each

// ─── Color interpolation ───

function _atmParseHex(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}

function _atmToHex(r, g, b) {
  var rh = ('0' + Math.round(Math.max(0, Math.min(255, r))).toString(16)).slice(-2);
  var gh = ('0' + Math.round(Math.max(0, Math.min(255, g))).toString(16)).slice(-2);
  var bh = ('0' + Math.round(Math.max(0, Math.min(255, b))).toString(16)).slice(-2);
  return '#' + rh + gh + bh;
}

function lerpColor(hex1, hex2, t) {
  var c1 = _atmParseHex(hex1);
  var c2 = _atmParseHex(hex2);
  return _atmToHex(
    c1.r + (c2.r - c1.r) * t,
    c1.g + (c2.g - c1.g) * t,
    c1.b + (c2.b - c1.b) * t
  );
}

function _atmLerpScalar(a, b, t) {
  return a + (b - a) * t;
}

// Smooth sine-based easing: 0→1 maps to smooth S-curve
function _atmSmoothStep(t) {
  return 0.5 - 0.5 * Math.cos(t * Math.PI);
}

// ─── Phase computation ───

function _atmGetPhaseInfo(clock) {
  var wrapped = ((clock % ATM_CYCLE) + ATM_CYCLE) % ATM_CYCLE;
  var phaseIndex = Math.floor(wrapped / ATM_PHASE_DURATION);
  var progress = (wrapped - phaseIndex * ATM_PHASE_DURATION) / ATM_PHASE_DURATION;
  return {
    phase: ATM_PHASE_ORDER[phaseIndex],
    nextPhase: ATM_PHASE_ORDER[(phaseIndex + 1) % 4],
    progress: progress,
  };
}

// ─── Core update ───

function updateAtmosphere(dt) {
  atmClock = ((atmClock + dt) % ATM_CYCLE + ATM_CYCLE) % ATM_CYCLE;

  var info = _atmGetPhaseInfo(atmClock);
  atmPhase = info.phase;
  atmProgress = info.progress;

  var cur = ATM_COLORS[info.phase];
  var nxt = ATM_COLORS[info.nextPhase];
  var t = _atmSmoothStep(info.progress);

  // Interpolate void color and update global palette
  C.void = lerpColor(cur.void, nxt.void, t);

  // Expose interpolated values as globals
  atmGlowMult = _atmLerpScalar(cur.glowMult, nxt.glowMult, t);
  atmGridAlpha = _atmLerpScalar(cur.gridAlpha, nxt.gridAlpha, t);
  atmParticleMult = _atmLerpScalar(cur.particleMult, nxt.particleMult, t);
}

// ─── Phase label for UI display ───

function getAtmPhaseLabel() {
  switch (atmPhase) {
    case 'night': return '\u25CF NIGHT';
    case 'dawn':  return '\u25D0 DAWN';
    case 'day':   return '\u25CB DAY';
    case 'dusk':  return '\u25D1 DUSK';
    default:      return '\u25CF NIGHT';
  }
}

// ─── Apply atmosphere (called from render loop) ───
// Should be called BEFORE the void fill in render, or it can modify
// C.void directly (which it does via updateAtmosphere) so the
// existing fillRect handles it automatically.

function applyAtmosphere(ctx, W, H, time) {
  // Advance the cycle clock using frame delta
  // dt is derived from time progression; caller passes currentTime
  // We use a stored last-time approach for accurate dt
  if (typeof applyAtmosphere._lastTime === 'undefined') {
    applyAtmosphere._lastTime = time;
  }
  var dt = Math.min(time - applyAtmosphere._lastTime, 0.1);
  applyAtmosphere._lastTime = time;

  // Guard against first-frame zero dt
  if (dt <= 0) dt = 0.016;

  updateAtmosphere(dt);

  // Optional: apply a subtle vignette overlay that intensifies at night
  var vignetteAlpha = 0.03 + 0.07 * atmGlowMult;
  var alphaQ = Math.round(vignetteAlpha * 100);
  if (!applyAtmosphere._vigGrad || applyAtmosphere._vigW !== W || applyAtmosphere._vigH !== H || applyAtmosphere._vigA !== alphaQ) {
    applyAtmosphere._vigGrad = ctx.createRadialGradient(
      W * 0.5, H * 0.5, W * 0.2,
      W * 0.5, H * 0.5, W * 0.8
    );
    applyAtmosphere._vigGrad.addColorStop(0, 'rgba(0, 0, 0, 0)');
    applyAtmosphere._vigGrad.addColorStop(1, 'rgba(0, 0, 0, ' + vignetteAlpha + ')');
    applyAtmosphere._vigW = W;
    applyAtmosphere._vigH = H;
    applyAtmosphere._vigA = alphaQ;
  }
  ctx.fillStyle = applyAtmosphere._vigGrad;
  ctx.fillRect(0, 0, W, H);
}
