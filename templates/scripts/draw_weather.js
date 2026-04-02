// ═══ TRON WEATHER SYSTEM — DATA RAIN & LIGHTNING ═══
// Cycling weather states with matrix-style rain and electrical storms

var weatherState = 'clear';  // 'clear' | 'light_rain' | 'storm'
var weatherTimer = 0;
var weatherDuration = 60;
var rainDrops = [];
var RAIN_COUNT = 150;

// Lightning state
var _lightningBolts = [];
var _lightningFlashAlpha = 0;
var _lightningFlashFrames = 0;

// ─── Character set for data rain ───
var RAIN_CHARS = '01{}[]<>/\\|:.;~';

// ═══════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════

function initWeather() {
  rainDrops = [];
  for (var i = 0; i < RAIN_COUNT; i++) {
    rainDrops.push(_makeRainDrop(true));
  }
  weatherState = 'clear';
  weatherTimer = 0;
  weatherDuration = _randomDuration('clear');
}

function _makeRainDrop(randomY) {
  return {
    x: Math.random() * 3000 - 500,
    y: randomY ? Math.random() * 2000 - 200 : -Math.random() * 200 - 10,
    speed: 100 + Math.random() * 200,
    char: RAIN_CHARS.charAt(Math.floor(Math.random() * RAIN_CHARS.length)),
    alpha: 0,
    size: 8 + Math.random() * 6,
  };
}

function _randomDuration(state) {
  if (state === 'clear') return 30 + Math.random() * 60;
  if (state === 'light_rain') return 20 + Math.random() * 40;
  if (state === 'storm') return 15 + Math.random() * 15;
  return 60;
}

// ═══════════════════════════════════════════════════
//  STATE MACHINE
// ═══════════════════════════════════════════════════

function _nextWeatherState(current) {
  if (current === 'clear') return 'light_rain';
  if (current === 'light_rain') return 'storm';
  return 'clear';  // storm → clear
}

// ═══════════════════════════════════════════════════
//  LIGHTNING GENERATION
// ═══════════════════════════════════════════════════

function _generateLightningBolt(x1, y1, x2, y2) {
  var points = [{ x: x1, y: y1 }];
  var segments = 3 + Math.floor(Math.random() * 2); // 3-4 midpoints
  for (var i = 1; i <= segments; i++) {
    var t = i / (segments + 1);
    var mx = x1 + (x2 - x1) * t + (Math.random() - 0.5) * 120;
    var my = y1 + (y2 - y1) * t + (Math.random() - 0.5) * 80;
    points.push({ x: mx, y: my });
  }
  points.push({ x: x2, y: y2 });
  return {
    points: points,
    age: 0,
    duration: 0.3,
    alpha: 1.0,
  };
}

function _trySpawnLightning(dt) {
  // 5% chance per second → probability per frame
  if (Math.random() > 0.05 * dt) return;
  if (typeof gridLocations === 'undefined' || gridLocations.length < 2) return;

  // Pick two random distinct buildings
  var a = Math.floor(Math.random() * gridLocations.length);
  var b = a;
  var attempts = 0;
  while (b === a && attempts < 10) {
    b = Math.floor(Math.random() * gridLocations.length);
    attempts++;
  }
  if (a === b) return;

  var locA = gridLocations[a];
  var locB = gridLocations[b];

  // Convert world-space building positions to screen-space for drawing
  // We need to account for camera transform since lightning is drawn in screen-space
  var canvas = document.getElementById('canvas');
  var W = canvas ? canvas.width / dpr : 1200;
  var H = canvas ? canvas.height / dpr : 800;

  var sx1 = (locA.x + camera.x - W / 2) * camera.zoom + W / 2;
  var sy1 = (locA.y + camera.y - H / 2) * camera.zoom + H / 2;
  var sx2 = (locB.x + camera.x - W / 2) * camera.zoom + W / 2;
  var sy2 = (locB.y + camera.y - H / 2) * camera.zoom + H / 2;

  _lightningBolts.push(_generateLightningBolt(sx1, sy1, sx2, sy2));

  // Trigger screen flash
  _lightningFlashAlpha = 0.15;
  _lightningFlashFrames = 2;
}

// ═══════════════════════════════════════════════════
//  UPDATE
// ═══════════════════════════════════════════════════

function updateWeather(dt) {
  // State timer
  weatherTimer += dt;
  if (weatherTimer >= weatherDuration) {
    weatherTimer = 0;
    weatherState = _nextWeatherState(weatherState);
    weatherDuration = _randomDuration(weatherState);
  }

  // Determine target alpha range based on state
  var minAlpha = 0;
  var maxAlpha = 0;
  var isRaining = weatherState !== 'clear';

  if (weatherState === 'light_rain') {
    minAlpha = 0.1;
    maxAlpha = 0.5;
  } else if (weatherState === 'storm') {
    minAlpha = 0.2;
    maxAlpha = 0.8;
  }

  // Update rain particles
  var screenW = 3000;
  var screenH = 2000;
  var canvas = document.getElementById('canvas');
  if (canvas) {
    screenW = canvas.width / dpr + 500;
    screenH = canvas.height / dpr + 200;
  }

  // Wind during storm
  var windOffset = 0;
  if (weatherState === 'storm') {
    windOffset = Math.sin(currentTime * 0.5) * 30;
  }

  for (var i = 0; i < rainDrops.length; i++) {
    var drop = rainDrops[i];

    if (isRaining) {
      // Fade in alpha toward target
      if (drop.alpha < minAlpha) {
        drop.alpha = minAlpha + Math.random() * (maxAlpha - minAlpha);
      }

      // Move down
      drop.y += drop.speed * dt;

      // Apply wind
      drop.x += windOffset * dt;

      // Wrap around screen bounds
      if (drop.y > screenH) {
        drop.y = -Math.random() * 100 - 10;
        drop.x = Math.random() * screenW - 250;
        drop.char = RAIN_CHARS.charAt(Math.floor(Math.random() * RAIN_CHARS.length));
        drop.alpha = minAlpha + Math.random() * (maxAlpha - minAlpha);
      }
      if (drop.x > screenW) drop.x = -50;
      if (drop.x < -250) drop.x = screenW - 50;
    } else {
      // Fade out during clear
      drop.alpha = Math.max(0, drop.alpha - dt * 0.5);
    }
  }

  // Lightning (storm only)
  if (weatherState === 'storm') {
    _trySpawnLightning(dt);
  }

  // Update lightning bolt ages
  for (var j = _lightningBolts.length - 1; j >= 0; j--) {
    _lightningBolts[j].age += dt;
    if (_lightningBolts[j].age >= _lightningBolts[j].duration) {
      _lightningBolts.splice(j, 1);
    }
  }

  // Flash decay
  if (_lightningFlashFrames > 0) {
    _lightningFlashFrames--;
  } else {
    _lightningFlashAlpha = Math.max(0, _lightningFlashAlpha - dt * 2);
  }
}

// ═══════════════════════════════════════════════════
//  DRAW (screen-space)
// ═══════════════════════════════════════════════════

function drawWeather(ctx, W, H, time, dt) {
  if (!config.weather) return;
  if (!rainDrops.length) initWeather();

  ctx.save();
  ctx.scale(dpr, dpr);

  // ─── Data rain characters ───
  var isStorm = weatherState === 'storm';

  for (var i = 0; i < rainDrops.length; i++) {
    var drop = rainDrops[i];
    if (drop.alpha < 0.01) continue;

    ctx.globalAlpha = drop.alpha;

    // Color: cyan for light rain, mix cyan-green for storm
    if (isStorm) {
      // Alternate between cyan and green based on index
      var greenMix = (i % 3 === 0);
      ctx.fillStyle = greenMix ? '#44ff88' : C.holoBase;
    } else {
      ctx.fillStyle = C.holoBase;
    }

    ctx.font = Math.floor(drop.size) + 'px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(drop.char, drop.x, drop.y);
  }

  // ─── Lightning bolts ───
  for (var j = 0; j < _lightningBolts.length; j++) {
    var bolt = _lightningBolts[j];
    var boltAlpha = 1.0 - (bolt.age / bolt.duration);
    if (boltAlpha <= 0) continue;

    ctx.globalAlpha = boltAlpha;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    var _wSq = config.shadowQuality || 'high';
    if (_wSq !== 'off') { ctx.shadowColor = C.holoBase; ctx.shadowBlur = _wSq === 'low' ? 8 : 20; }

    ctx.beginPath();
    ctx.moveTo(bolt.points[0].x, bolt.points[0].y);
    for (var k = 1; k < bolt.points.length; k++) {
      ctx.lineTo(bolt.points[k].x, bolt.points[k].y);
    }
    ctx.stroke();

    // Inner bright core
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.holoBright;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(bolt.points[0].x, bolt.points[0].y);
    for (var m = 1; m < bolt.points.length; m++) {
      ctx.lineTo(bolt.points[m].x, bolt.points[m].y);
    }
    ctx.stroke();
  }

  // ─── Screen flash ───
  if (_lightningFlashAlpha > 0.001) {
    ctx.globalAlpha = _lightningFlashAlpha;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
  }

  ctx.restore();
}
