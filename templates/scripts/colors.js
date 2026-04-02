// ═══ NORT COLOR PALETTE ═══
// Ported from agent-flow colors.ts

var C = {
  void: '#050510',
  hexGrid: '#66ccff',
  holoBase: '#66ccff',
  holoBright: '#aaeeff',
  holoHot: '#ffffff',

  // Node interior
  nodeInterior: 'rgba(10, 15, 40, 0.5)',

  // State colors (map NORT task statuses)
  pending: '#66ccff',
  in_progress: '#66ccff',
  in_manager_review: '#ffbb44',
  in_specialist_review: '#cc88ff',
  revision: '#ff8800',
  done: '#66ffaa',
  failed: '#ff5566',

  // Edge colors
  dispatch: '#cc88ff',
  returnEdge: '#66ffaa',
  review: '#ffbb44',

  // Context breakdown
  contextSystem: '#555577',
  contextUser: '#66ccff',
  contextTool: '#ffbb44',
  contextReasoning: '#cc88ff',
  contextSubagent: '#66ffaa',

  // Text
  textPrimary: '#aaeeff',
  textDim: 'rgba(102, 204, 255, 0.7)',
  textMuted: 'rgba(102, 204, 255, 0.5)',

  // Glass (for canvas-drawn panels)
  glassBg: 'rgba(10, 15, 30, 0.7)',
  glassBorder: 'rgba(100, 200, 255, 0.15)',
};

function getStateColor(state) {
  return C[state] || C.pending;
}

var _hexRgbaCache = {};
var _hexRgbaCacheSize = 0;

function hexToRgba(hex, alpha) {
  var key = hex + '|' + alpha;
  if (_hexRgbaCache[key]) return _hexRgbaCache[key];
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  var result = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  if (_hexRgbaCacheSize > 500) { _hexRgbaCache = {}; _hexRgbaCacheSize = 0; }
  _hexRgbaCache[key] = result;
  _hexRgbaCacheSize++;
  return result;
}

function alphaHex(a) {
  return ('0' + Math.round(Math.min(1, Math.max(0, a)) * 255).toString(16)).slice(-2);
}
