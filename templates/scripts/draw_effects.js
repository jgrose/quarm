// ═══ NORT VISUAL EFFECTS ═══
// Enhanced spawn, complete, and error effects with scatter particles

function drawEffects(ctx, dt) {
  for (var i = effects.length - 1; i >= 0; i--) {
    var fx = effects[i];
    fx.age += dt;
    var progress = fx.age / fx.duration;
    if (progress >= 1) {
      effects.splice(i, 1);
      continue;
    }

    ctx.save();

    if (fx.type === 'spawn') {
      // ─── Spawn Effect (duration: 0.8s) ───

      var ringR = 10 + progress * 60;
      var alpha = (1 - progress) * 0.7;

      // White flash (first 30%)
      if (progress < 0.3) {
        var flashP = progress / 0.3;
        var flashAlpha = (1 - flashP) * 0.6;
        var flashR = 20 * (1 - flashP) + 5;
        var flashGrad = ctx.createRadialGradient(fx.x, fx.y, 0, fx.x, fx.y, flashR);
        flashGrad.addColorStop(0, 'rgba(255,255,255,' + flashAlpha + ')');
        flashGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = flashGrad;
        ctx.beginPath();
        ctx.arc(fx.x, fx.y, flashR, 0, Math.PI * 2);
        ctx.fill();
      }

      // Expanding hexagonal ring
      ctx.globalAlpha = alpha;
      drawHexPath(ctx, fx.x, fx.y, ringR);
      ctx.strokeStyle = fx.color;
      ctx.lineWidth = 2 * (1 - progress);
      ctx.stroke();

      // Scatter particles (8 particles flying outward)
      var scatterSeed = fx.x * 7 + fx.y * 13;
      for (var p = 0; p < 8; p++) {
        var a = (p / 8) * Math.PI * 2 + (scatterSeed % 1);
        var d = ringR * 0.8 + progress * 20;
        var px = fx.x + Math.cos(a) * d;
        var py = fx.y + Math.sin(a) * d;
        ctx.beginPath();
        ctx.fillStyle = fx.color + alphaHex(alpha * 0.8);
        ctx.arc(px, py, 1.5 * (1 - progress), 0, Math.PI * 2);
        ctx.fill();
      }

      // Sound
      if (config.sound && fx.age <= dt * 1.5) {
        playAgentSpawn();
      }
    }

    if (fx.type === 'complete') {
      // ─── Complete Effect (duration: 1.0s) ───

      var cRingR = 20 + progress * 80;
      var cAlpha = (1 - progress) * 0.6;

      // Bright white flash (first 20%)
      if (progress < 0.2) {
        var cFlashP = progress / 0.2;
        var cFlashAlpha = (1 - cFlashP) * 0.8;
        var cFlashGrad = ctx.createRadialGradient(fx.x, fx.y, 0, fx.x, fx.y, 30);
        cFlashGrad.addColorStop(0, 'rgba(255,255,255,' + cFlashAlpha + ')');
        cFlashGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = cFlashGrad;
        ctx.fillRect(fx.x - 30, fx.y - 30, 60, 60);
      }

      // Glow aura behind ring
      var auraGrad = ctx.createRadialGradient(fx.x, fx.y, cRingR - 5, fx.x, fx.y, cRingR + 10);
      auraGrad.addColorStop(0, fx.color + '00');
      auraGrad.addColorStop(0.5, fx.color + alphaHex(cAlpha * 0.4));
      auraGrad.addColorStop(1, fx.color + '00');
      ctx.fillStyle = auraGrad;
      ctx.beginPath();
      ctx.arc(fx.x, fx.y, cRingR + 10, 0, Math.PI * 2);
      ctx.fill();

      // Expanding circle ring (green glow)
      ctx.globalAlpha = cAlpha;
      ctx.beginPath();
      ctx.arc(fx.x, fx.y, cRingR, 0, Math.PI * 2);
      ctx.strokeStyle = fx.color;
      ctx.lineWidth = 3 * (1 - progress);
      ctx.stroke();

      // Sound
      if (config.sound && fx.age <= dt * 1.5) {
        playAgentComplete();
      }
    }

    if (fx.type === 'error') {
      // ─── Error Effect (duration: 0.6s) ───

      var errAlpha = (1 - progress) * 0.8;
      var pulseAmt = Math.sin(progress * Math.PI * 4) * 0.3 + 0.7;
      var errR = 10 + progress * 50;

      // Red pulsing glow expanding from center
      var errGrad = ctx.createRadialGradient(fx.x, fx.y, 0, fx.x, fx.y, errR);
      errGrad.addColorStop(0, hexToRgba(fx.color, errAlpha * pulseAmt * 0.6));
      errGrad.addColorStop(0.6, hexToRgba(fx.color, errAlpha * pulseAmt * 0.3));
      errGrad.addColorStop(1, hexToRgba(fx.color, 0));
      ctx.fillStyle = errGrad;
      ctx.beginPath();
      ctx.arc(fx.x, fx.y, errR, 0, Math.PI * 2);
      ctx.fill();

      // 3 crack lines radiating outward (60 degrees apart)
      ctx.globalAlpha = errAlpha * pulseAmt;
      ctx.strokeStyle = fx.color;
      ctx.lineWidth = 2 * (1 - progress);
      for (var c = 0; c < 3; c++) {
        var crackAngle = (c / 3) * Math.PI * 2 + 0.5;
        var crackLen = errR * 0.9;
        ctx.beginPath();
        ctx.moveTo(fx.x, fx.y);
        // Jagged crack: 3 segments with slight offsets
        var segLen = crackLen / 3;
        var cx = fx.x, cy = fx.y;
        for (var seg = 1; seg <= 3; seg++) {
          var jitter = (seg === 2) ? 5 : -3;
          cx += Math.cos(crackAngle) * segLen + Math.cos(crackAngle + Math.PI / 2) * jitter * (1 - progress);
          cy += Math.sin(crackAngle) * segLen + Math.sin(crackAngle + Math.PI / 2) * jitter * (1 - progress);
          ctx.lineTo(cx, cy);
        }
        ctx.stroke();
      }

      // Sound
      if (config.sound && fx.age <= dt * 1.5) {
        playError();
      }
    }

    ctx.restore();
  }
}
