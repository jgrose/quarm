// ═══ NORT CONSTANTS ═══

var ANIM = {
  agentFadeIn: 3,
  agentScaleIn: 4,
  agentFadeOut: 0.4,
  agentScaleOut: 0.05,
  toolFadeIn: 4,
  toolFadeOut: 1.5,
  edgeFadeIn: 4,
  particleSpeed: 1.2,
  maxDt: 0.1,
  defaultDt: 0.016,
};

var FORCE_CFG = {
  charge: -1200,
  centerStrength: 0.03,
  collide: 140,
  linkDist: 350,
  linkStrength: 0.4,
  alphaDecay: 0.02,
  velDecay: 0.4,
};

var TIERS = {
  nexus:    { radius: 36, zone: 0, icon: '\u2B21', label: 'NEXUS' },
  sentinel: { radius: 28, zone: 1, icon: '\u25C7', label: 'SENTINEL' },
  drone:    { radius: 22, zone: 2, icon: '\u25B8', label: 'DRONE' },
  probe:    { radius: 18, zone: 3, icon: '\u25C8', label: 'PROBE' },
  shard:    { radius: 16, zone: 2.5, icon: '\u00B7', label: 'SHARD' },
};

var BUBBLE_HOLD = 10;
var BUBBLE_FADE_IN = 0.3;
var BUBBLE_FADE_OUT = 1.5;
var TOOL_CARD_W = 170;
var TOOL_CARD_H = 36;

var PERF = {
  sampleInterval: 500,
  emaAlpha: 0.1,
  budgetMs: 16.6,
};
