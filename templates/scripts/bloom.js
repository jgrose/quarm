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
  var w = Math.floor(sourceCanvas.width / 2);
  var h = Math.floor(sourceCanvas.height / 2);
  if (w === 0 || h === 0) return;

  if (this.canvas.width !== w || this.canvas.height !== h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.tempCanvas.width = w;
    this.tempCanvas.height = h;
  }

  // Draw source at half resolution
  this.ctx.clearRect(0, 0, w, h);
  this.ctx.drawImage(sourceCanvas, 0, 0, w, h);

  // 3-pass box blur using CSS filter
  this._blur(w, h, 8);
  this._blur(w, h, 6);
  this._blur(w, h, 4);

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
