// ═══ NORT AUDIO ENGINE ═══
// Procedurally synthesized sound effects via Web Audio API

var audioCtx = null;
var audioMasterGain = null;
var audioMuted = true; // default muted, toggled via config.sound
var audioVolume = 0.5;

function initAudio() {
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioMasterGain = audioCtx.createGain();
    audioMasterGain.gain.value = audioMuted ? 0 : audioVolume;
    audioMasterGain.connect(audioCtx.destination);
  } catch(e) {}
}

function resumeAudio() {
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}

function setAudioMuted(muted) {
  audioMuted = muted;
  if (audioMasterGain) audioMasterGain.gain.value = muted ? 0 : audioVolume;
}

// Helper: create a short tone
function _tone(freq, endFreq, duration, volume, type) {
  if (!audioCtx || audioMuted) return;
  resumeAudio();
  var osc = audioCtx.createOscillator();
  var gain = audioCtx.createGain();
  osc.type = type || 'sine';
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  osc.frequency.linearRampToValueAtTime(endFreq, audioCtx.currentTime + duration);
  gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(volume, audioCtx.currentTime + 0.02);
  gain.gain.linearRampToValueAtTime(0.001, audioCtx.currentTime + duration);
  osc.connect(gain);
  gain.connect(audioMasterGain);
  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

function playToolStart() {
  // Soft click: 480→240Hz sine, 25ms
  _tone(480, 240, 0.025, 0.06, 'sine');
}

function playToolEnd() {
  // Softer click: 600→300Hz sine, 25ms
  _tone(600, 300, 0.025, 0.08, 'sine');
}

function playAgentSpawn() {
  // Rising shimmer: two staggered tones (G5 + D6)
  _tone(784, 784, 0.2, 0.03, 'sine');
  setTimeout(function() { _tone(1175, 1175, 0.2, 0.03, 'sine'); }, 60);
}

function playAgentComplete() {
  // Gentle Cmaj7 arpeggio
  var notes = [261.63, 329.63, 392, 493.88];
  notes.forEach(function(freq, i) {
    setTimeout(function() { _tone(freq, freq, 0.25, 0.022, 'sine'); }, i * 70);
  });
}

function playError() {
  // Soft falling tone: 220→165Hz triangle, 250ms
  _tone(220, 165, 0.25, 0.08, 'triangle');
}

function playQuestion() {
  // Two-note attention chime: C5 then G5 (rising), sine, 180ms each
  _tone(523.25, 523.25, 0.18, 0.08, 'sine');
  setTimeout(function () { _tone(783.99, 783.99, 0.18, 0.08, 'sine'); }, 130);
}

// ═══ ENHANCED SOUND DESIGN ═══

// ─── Ambient Tron hum (continuous drone) ───
var _ambientOsc1 = null, _ambientOsc2 = null;
var _ambientGain1 = null, _ambientGain2 = null;

function startAmbientHum() {
  if (!audioCtx || _ambientOsc1) return;
  resumeAudio();

  _ambientOsc1 = audioCtx.createOscillator();
  _ambientOsc1.type = 'sine';
  _ambientOsc1.frequency.value = 60;
  _ambientGain1 = audioCtx.createGain();
  _ambientGain1.gain.value = 0.015;
  _ambientOsc1.connect(_ambientGain1).connect(audioMasterGain);
  _ambientOsc1.start();

  _ambientOsc2 = audioCtx.createOscillator();
  _ambientOsc2.type = 'sine';
  _ambientOsc2.frequency.value = 120;
  _ambientGain2 = audioCtx.createGain();
  _ambientGain2.gain.value = 0.008;
  _ambientOsc2.connect(_ambientGain2).connect(audioMasterGain);
  _ambientOsc2.start();
}

function stopAmbientHum() {
  if (_ambientOsc1) { try { _ambientOsc1.stop(); } catch(e) {} _ambientOsc1 = null; _ambientGain1 = null; }
  if (_ambientOsc2) { try { _ambientOsc2.stop(); } catch(e) {} _ambientOsc2 = null; _ambientGain2 = null; }
}

// ─── Footstep click (called per walk frame change) ───

function playFootstep(pitch) {
  if (!config.sound || audioMuted) return;
  var p = pitch || 1.0;
  _tone(800 * p, 400 * p, 0.02, 0.03, 'sine');
}

// ─── Door whoosh (white noise burst) ───

function playDoorWhoosh() {
  if (!config.sound || audioMuted) return;
  resumeAudio();
  var bufSize = audioCtx.sampleRate * 0.1;
  var buf = audioCtx.createBuffer(1, bufSize, audioCtx.sampleRate);
  var data = buf.getChannelData(0);
  for (var i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / bufSize);
  var src = audioCtx.createBufferSource();
  src.buffer = buf;
  var g = audioCtx.createGain();
  g.gain.setValueAtTime(0.06, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
  src.connect(g).connect(audioMasterGain);
  src.start();
}

// ─── Level-up arpeggio (C5 → E5 → G5 → C6) ───

function playLevelUp() {
  if (!config.sound || audioMuted) return;
  _tone(523, 523, 0.15, 0.12, 'sine');
  setTimeout(function() { _tone(659, 659, 0.15, 0.12, 'sine'); }, 120);
  setTimeout(function() { _tone(784, 784, 0.15, 0.12, 'sine'); }, 240);
  setTimeout(function() { _tone(1047, 1047, 0.25, 0.15, 'sine'); }, 360);
}

// ─── Rain ambient (white noise bed, looped) ───
var _rainNode = null;
var _rainGain = null;

function startRainAmbient() {
  if (!audioCtx || _rainNode) return;
  resumeAudio();
  var bufSize = audioCtx.sampleRate * 2;
  var buf = audioCtx.createBuffer(1, bufSize, audioCtx.sampleRate);
  var data = buf.getChannelData(0);
  for (var i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1;
  _rainNode = audioCtx.createBufferSource();
  _rainNode.buffer = buf;
  _rainNode.loop = true;
  _rainGain = audioCtx.createGain();
  _rainGain.gain.value = 0.02;
  var filt = audioCtx.createBiquadFilter();
  filt.type = 'lowpass';
  filt.frequency.value = 2000;
  _rainNode.connect(filt).connect(_rainGain).connect(audioMasterGain);
  _rainNode.start();
}

function stopRainAmbient() {
  if (_rainNode) { try { _rainNode.stop(); } catch(e) {} _rainNode = null; _rainGain = null; }
}

// ─── Thunder crack (low burst) ───

function playThunder() {
  if (!config.sound || audioMuted) return;
  _tone(80, 40, 0.4, 0.15, 'sawtooth');
  _tone(60, 30, 0.5, 0.1, 'triangle');
}
