// ═══ NORT THOUGHT BUBBLES ═══
// Pixel-art emotion indicators above ambient programs

// ── Emotion icon sprites (8x8, palette: 0=transparent, 1=primary, 2=bright, 3=dim) ──

var _EMOTION_SPRITES = {
  idle: [ // zzz
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,1,1,0,1,1,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,1,0],
  ],
  busy: [ // gear
    [0,0,0,1,1,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,1,0,1,1,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,0,1,0,0,1,0,1],
    [0,1,0,1,1,0,1,0],
    [0,0,1,0,0,1,0,0],
    [0,0,0,1,1,0,0,0],
  ],
  thinking: [ // question mark
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
  ],
  happy: [ // smile face
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,0,1],
    [0,1,0,0,0,0,1,0],
    [0,0,1,1,1,1,0,0],
  ],
  excited: [ // star burst
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,1,2,2,1,0,0],
    [1,1,2,2,2,2,1,1],
    [1,1,2,2,2,2,1,1],
    [0,0,1,2,2,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
  ],
  sad: [ // teardrop
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,1,0,0],
    [0,0,0,1,0,1,0,0],
    [0,0,1,0,0,0,1,0],
    [0,0,1,0,2,0,1,0],
    [0,0,0,1,0,1,0,0],
    [0,0,0,0,1,0,0,0],
  ],
  frustrated: [ // lightning bolt
    [0,0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,1,1,0,0,0,0,0],
  ],
  confused: [ // spiral
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [0,0,0,1,1,0,1,0],
    [0,0,1,0,0,1,1,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
  ],
};

var _EMOTION_COLORS = {
  idle: '#aaeeff',
  busy: '#66ccff',
  thinking: '#ffbb44',
  happy: '#66ffaa',
  excited: '#66ffaa',
  sad: '#ff5566',
  frustrated: '#ff8800',
  confused: '#cc88ff',
};

// ── Sprite cache ──

var _thoughtSpriteCache = {};

function _renderThoughtSprite(emotion, scale) {
  var key = emotion + '|' + scale;
  if (_thoughtSpriteCache[key]) return _thoughtSpriteCache[key];

  var sprite = _EMOTION_SPRITES[emotion];
  if (!sprite) return null;

  var color = _EMOTION_COLORS[emotion] || '#aaeeff';
  var px = Math.max(1, Math.round(PX * scale));
  var w = 8 * px;
  var h = 8 * px;

  var oc = document.createElement('canvas');
  oc.width = w;
  oc.height = h;
  var octx = oc.getContext('2d');

  // Parse color for palette
  var r = parseInt(color.slice(1, 3), 16);
  var g = parseInt(color.slice(3, 5), 16);
  var b = parseInt(color.slice(5, 7), 16);

  for (var row = 0; row < 8; row++) {
    for (var col = 0; col < 8; col++) {
      var v = sprite[row][col];
      if (v === 0) continue;
      if (v === 1) octx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      else if (v === 2) octx.fillStyle = 'rgb(' + Math.min(255, r + 60) + ',' + Math.min(255, g + 60) + ',' + Math.min(255, b + 60) + ')';
      else octx.fillStyle = 'rgb(' + Math.floor(r * 0.5) + ',' + Math.floor(g * 0.5) + ',' + Math.floor(b * 0.5) + ')';
      octx.fillRect(col * px, row * px, px, px);
    }
  }

  _thoughtSpriteCache[key] = oc;
  return oc;
}

// ── Emotion derivation ──

function deriveEmotion(task) {
  if (!task) return 'idle';
  var s = task.status;
  if (!s || s === 'pending') return 'idle';

  if (s === 'done') {
    var score = task.lastScore || 0;
    if (score >= 9) return 'excited';
    if (score >= 7) return 'happy';
    if (score < 5 && score > 0) return 'sad';
    return 'happy';
  }
  if (s === 'failed') return 'sad';
  if (s === 'revision') {
    return (task.revisionCount || 0) >= 2 ? 'frustrated' : 'confused';
  }
  if (s === 'in_manager_review' || s === 'in_specialist_review') return 'thinking';
  if (s === 'in_progress') {
    if (task.revisionCount > 0) return 'frustrated';
    return 'busy';
  }
  return 'idle';
}

// ── Emotion state update (call per-frame per program) ──

var _THOUGHT_FADE_DUR = 0.3;

function updateEmotionState(p, dt) {
  if (!p.emotion) return;
  var target = deriveEmotion(p.assignedTask);

  if (target !== p.emotion.current && target !== p.emotion.target) {
    p.emotion.target = target;
    p.emotion.transitionAge = 0;
  }

  if (p.emotion.target !== p.emotion.current) {
    p.emotion.transitionAge += dt;
    if (p.emotion.transitionAge < _THOUGHT_FADE_DUR) {
      // Fading out old
      p.emotion.alpha = 1.0 - (p.emotion.transitionAge / _THOUGHT_FADE_DUR);
    } else if (p.emotion.transitionAge < _THOUGHT_FADE_DUR * 2) {
      // Swap and fade in new
      if (p.emotion.current !== p.emotion.target) {
        p.emotion.current = p.emotion.target;
      }
      p.emotion.alpha = (p.emotion.transitionAge - _THOUGHT_FADE_DUR) / _THOUGHT_FADE_DUR;
    } else {
      p.emotion.current = p.emotion.target;
      p.emotion.alpha = 1.0;
    }
  } else {
    // Stable -- ensure full alpha
    if (p.emotion.alpha < 1.0) {
      p.emotion.alpha = Math.min(1.0, p.emotion.alpha + dt * 3);
    }
  }
}

// ── Draw thought bubble above a program ──

function drawThoughtBubble(ctx, p, drawY, time, bunkerAlpha) {
  if (!config.thoughtBubbles) return;
  if (!p.emotion || p.emotion.alpha <= 0.01) return;
  if (p.bunkerState === 'inside') return;

  var zoom = camera ? camera.zoom : 1;

  // LOD: skip at very low zoom
  if (zoom < 0.3) return;

  var emotion = p.emotion.current;
  var color = _EMOTION_COLORS[emotion] || '#aaeeff';
  var alpha = p.emotion.alpha * bunkerAlpha;
  if (alpha <= 0.01) return;

  // Bob offset (different frequency from task icon)
  var bobY = Math.sin(time * 1.5 + (p.x || 0) * 0.01) * 2.5;

  // Position above task status icon and label
  var offsetAbove = p.assignedTask ? -(PX * 4 + 8) : 0;
  var bubbleY = Math.floor(drawY - 18 + offsetAbove + bobY);
  var bubbleX = Math.floor(p.x);

  ctx.save();
  ctx.globalAlpha = alpha;

  // LOD 2: just a colored dot at medium zoom
  if (zoom < 0.6) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(bubbleX, bubbleY, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  // Full bubble rendering
  var spriteScale = 1.0;
  var sprite = _renderThoughtSprite(emotion, spriteScale);
  if (!sprite) { ctx.restore(); return; }

  var sw = sprite.width;
  var sh = sprite.height;
  var padX = 4;
  var padY = 3;
  var bw = sw + padX * 2;
  var bh = sh + padY * 2;
  var bx = bubbleX - bw / 2;
  var by = bubbleY - bh;

  // Bubble background
  ctx.fillStyle = 'rgba(5, 5, 16, 0.6)';
  ctx.beginPath();
  ctx.roundRect(bx, by, bw, bh, 3);
  ctx.fill();

  // Bubble border
  ctx.strokeStyle = hexToRgba(color, 0.5);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(bx, by, bw, bh, 3);
  ctx.stroke();

  // Trailing thought circles
  var cx1 = bubbleX;
  var cy1 = by + bh + 3;
  ctx.fillStyle = hexToRgba(color, 0.3);
  ctx.beginPath();
  ctx.arc(cx1, cy1, 2, 0, Math.PI * 2);
  ctx.fill();

  var cx2 = bubbleX + 2;
  var cy2 = cy1 + 4;
  ctx.beginPath();
  ctx.arc(cx2, cy2, 1.5, 0, Math.PI * 2);
  ctx.fill();

  // Draw sprite icon
  ctx.drawImage(sprite, Math.floor(bx + padX), Math.floor(by + padY));

  ctx.restore();
}

