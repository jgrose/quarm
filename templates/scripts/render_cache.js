// ═══ NORT RENDER CACHE ═══
// Pre-render glow sprites and cache text measurements

var _glowCache = {};
var _textCache = {};
var _textCacheCount = 0;
var TEXT_CACHE_MAX = 2000;

function getGlowSprite(color, radius, innerAlpha, outerAlpha) {
  innerAlpha = innerAlpha || 0.4;
  outerAlpha = outerAlpha || 0;
  var key = color + '|' + radius + '|' + innerAlpha + '|' + outerAlpha;
  if (_glowCache[key]) return _glowCache[key];

  var size = Math.ceil(radius * 2);
  var c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  var ctx = c.getContext('2d');
  var grad = ctx.createRadialGradient(radius, radius, 0, radius, radius, radius);
  // Parse hex color to rgba
  var r = parseInt(color.slice(1,3), 16) || 0;
  var g = parseInt(color.slice(3,5), 16) || 0;
  var b = parseInt(color.slice(5,7), 16) || 0;
  grad.addColorStop(0, 'rgba(' + r + ',' + g + ',' + b + ',' + innerAlpha + ')');
  grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',' + outerAlpha + ')');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  _glowCache[key] = c;
  return c;
}

function getAgentGlowSprite(color, innerR, outerR, glowAlpha) {
  var key = 'ag|' + color + '|' + innerR + '|' + outerR + '|' + glowAlpha;
  if (_glowCache[key]) return _glowCache[key];

  var size = Math.ceil(outerR * 2);
  var c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  var ctx = c.getContext('2d');
  var cx = size / 2, cy = size / 2;
  var grad = ctx.createRadialGradient(cx, cy, innerR, cx, cy, outerR);
  var r = parseInt(color.slice(1,3), 16) || 0;
  var g = parseInt(color.slice(3,5), 16) || 0;
  var b = parseInt(color.slice(5,7), 16) || 0;
  grad.addColorStop(0, 'rgba(' + r + ',' + g + ',' + b + ',' + glowAlpha + ')');
  grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
  ctx.fill();
  _glowCache[key] = c;
  return c;
}

function measureTextCached(ctx, text) {
  var key = ctx.font + '|' + text;
  if (_textCache[key] !== undefined) return _textCache[key];
  if (_textCacheCount >= TEXT_CACHE_MAX) {
    _textCache = {};
    _textCacheCount = 0;
  }
  var w = ctx.measureText(text).width;
  _textCache[key] = w;
  _textCacheCount++;
  return w;
}
