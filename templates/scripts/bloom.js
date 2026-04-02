// ═══ NORT BLOOM POST-PROCESSING ═══
// Ported from agent-flow bloom-renderer.ts

function BloomRenderer() {
  this.canvas = document.createElement('canvas');
  this.ctx = this.canvas.getContext('2d');
  this.tempCanvas = document.createElement('canvas');
  this.tempCtx = this.tempCanvas.getContext('2d');
  this.intensity = 0.5;
}

BloomRenderer.prototype.apply = function(sourceCanvas) {
  // Determine quality: bloomQuality takes precedence, fall back to bloom boolean
  var quality = (typeof config !== 'undefined' && config.bloomQuality) ? config.bloomQuality : 'high';
  if (typeof config !== 'undefined' && !config.bloom) quality = 'off';
  if (quality === 'off') return;
  if (this.intensity <= 0) return;

  var divisor = quality === 'low' ? 4 : 2;
  var w = Math.floor(sourceCanvas.width / divisor);
  var h = Math.floor(sourceCanvas.height / divisor);
  if (w === 0 || h === 0) return;

  if (this.canvas.width !== w || this.canvas.height !== h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.tempCanvas.width = w;
    this.tempCanvas.height = h;
  }

  // Draw source at reduced resolution
  this.ctx.clearRect(0, 0, w, h);
  this.ctx.drawImage(sourceCanvas, 0, 0, w, h);

  // Blur passes: 1 for low quality, 3 for high
  if (quality === 'low') {
    this._blur(w, h, 6);
  } else {
    this._blur(w, h, 8);
    this._blur(w, h, 6);
    this._blur(w, h, 4);
  }

  // Composite bloom over source with additive blending
  var mainCtx = sourceCanvas.getContext('2d');
  mainCtx.save();
  mainCtx.globalCompositeOperation = 'lighter';
  mainCtx.globalAlpha = this.intensity;
  mainCtx.drawImage(this.canvas, 0, 0, sourceCanvas.width, sourceCanvas.height);
  mainCtx.restore();
};

BloomRenderer.prototype._blur = function(w, h, radius) {
  this.tempCtx.clearRect(0, 0, w, h);
  this.tempCtx.filter = 'blur(' + radius + 'px)';
  this.tempCtx.drawImage(this.canvas, 0, 0);
  this.tempCtx.filter = 'none';
  this.ctx.clearRect(0, 0, w, h);
  this.ctx.drawImage(this.tempCanvas, 0, 0);
};
