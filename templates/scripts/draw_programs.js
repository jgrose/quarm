// ═══ TRON LEGACY — ISOMETRIC 16-BIT AMBIENT PROGRAMS ═══
// Sim City SNES–style isometric pixel-art with 4-directional walk

var ambientPrograms = [];
var PROGRAM_COUNT = 6;
var PX = 3;

// ─── Palette indices ───
// 0=transparent  1=suit body  2=suit shadow  3=visor white
// 4=circuit glow 5=suit highlight  6=disc bright  7=outline  8=circuit dim

// ═══════════════════════════════════════════════════
//  ISOMETRIC SPRITE FRAMES — SE direction (16w × 24h)
//  SW = horizontal flip of SE
// ═══════════════════════════════════════════════════

// SE Stand / idle
var SPRITE_SE_STAND = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,7,5,5,5,5,5,1,1,7,0,0,0,0],
  [0,0,0,7,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,5,1,7,1,8,4,8,1,7,2,2,7,0,0],
  [0,7,1,4,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,1,7,1,8,4,8,1,7,2,1,7,0,0],
  [0,0,7,7,0,7,4,1,4,7,0,7,7,0,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,4,1,7,0,7,1,2,7,0,0,0,0],
  [0,0,0,7,1,1,7,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,2,7,0,7,2,1,7,0,0,0,0],
  [0,0,0,7,4,1,7,0,7,2,4,7,0,0,0,0],
  [0,0,7,5,1,1,7,0,7,2,1,2,7,0,0,0],
  [0,0,7,1,4,1,7,0,7,2,4,1,7,0,0,0],
  [0,0,7,1,1,4,1,7,7,1,1,1,7,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk frame A — left leg forward
var SPRITE_SE_WALK_A = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,7,5,5,5,5,5,1,1,7,0,0,0,0],
  [0,0,0,7,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [7,5,1,1,7,1,8,4,8,1,7,0,0,0,0,0],
  [7,1,4,1,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,7,0,1,8,4,8,1,7,2,4,7,0,0],
  [0,0,7,0,0,7,4,1,4,7,0,7,2,7,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,7,7,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,7,0,0,7,2,7,0,0,0,0],
  [0,0,0,7,4,7,0,0,0,0,7,2,7,0,0,0],
  [0,0,7,1,7,0,0,0,0,7,2,2,7,0,0,0],
  [0,7,1,1,7,0,0,0,0,7,2,4,7,0,0,0],
  [0,7,5,1,7,0,0,0,7,2,1,2,7,0,0,0],
  [0,7,1,4,1,7,0,0,7,2,4,1,7,0,0,0],
  [0,7,1,1,4,1,7,7,7,1,1,1,7,0,0,0],
  [0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk mid — legs passing
var SPRITE_SE_WALK_MID = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,7,5,5,5,5,5,1,1,7,0,0,0,0],
  [0,0,0,7,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,1,1,7,1,8,4,8,1,7,2,1,7,0,0],
  [0,0,7,4,7,1,4,6,4,1,7,1,7,0,0,0],
  [0,0,7,1,0,1,8,4,8,1,0,2,7,0,0,0],
  [0,0,0,7,0,7,4,1,4,7,0,7,0,0,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,7,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,7,2,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,2,7,2,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,7,2,4,7,0,0,0,0,0],
  [0,0,0,7,5,1,1,7,2,1,2,7,0,0,0,0],
  [0,0,0,7,1,4,1,7,2,4,1,7,0,0,0,0],
  [0,0,0,7,1,1,4,1,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk frame B — right leg forward
var SPRITE_SE_WALK_B = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,7,5,5,5,5,5,1,1,7,0,0,0,0],
  [0,0,0,7,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,5,1,7,1,8,4,8,1,7,2,2,7,0,0],
  [0,7,1,4,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,4,1,7,1,8,4,8,1,0,7,1,7,0,0],
  [0,7,2,7,0,7,4,1,4,7,0,0,7,0,0,0],
  [0,7,7,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,1,7,0,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,7,0,0,7,2,2,7,0,0,0],
  [0,0,0,7,1,2,7,0,0,0,7,1,7,0,0,0],
  [0,0,0,7,4,1,7,0,0,0,7,4,1,7,0,0],
  [0,0,0,7,5,1,1,7,0,7,2,1,1,7,0,0],
  [0,0,0,7,1,4,1,7,0,7,2,4,1,7,0,0],
  [0,0,0,7,1,1,4,1,7,7,1,1,1,7,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// ═══════════════════════════════════════════════════
//  NE direction (back view, walking up-right) — 16w × 24h
//  NW = horizontal flip of NE
// ═══════════════════════════════════════════════════

// NE Stand
var SPRITE_NE_STAND = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,7,1,1,1,1,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,7,5,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,5,1,7,1,8,6,8,1,7,2,2,7,0,0],
  [0,7,1,4,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,1,7,1,8,6,8,1,7,2,1,7,0,0],
  [0,0,7,7,0,7,4,1,4,7,0,7,7,0,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,4,1,7,0,7,1,2,7,0,0,0,0],
  [0,0,0,7,1,1,7,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,2,7,0,7,2,1,7,0,0,0,0],
  [0,0,0,7,4,1,7,0,7,2,4,7,0,0,0,0],
  [0,0,7,5,1,1,7,0,7,2,1,2,7,0,0,0],
  [0,0,7,1,4,1,7,0,7,2,4,1,7,0,0,0],
  [0,0,7,1,1,4,1,7,7,1,1,1,7,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk frame A
var SPRITE_NE_WALK_A = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,7,1,1,1,1,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,7,5,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [7,5,1,1,7,1,8,6,8,1,7,0,0,0,0,0],
  [7,1,4,1,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,7,0,1,8,6,8,1,7,2,4,7,0,0],
  [0,0,7,0,0,7,4,1,4,7,0,7,2,7,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,7,7,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,7,0,0,7,2,7,0,0,0,0],
  [0,0,0,7,4,7,0,0,0,0,7,2,7,0,0,0],
  [0,0,7,1,7,0,0,0,0,7,2,2,7,0,0,0],
  [0,7,1,1,7,0,0,0,0,7,2,4,7,0,0,0],
  [0,7,5,1,7,0,0,0,7,2,1,2,7,0,0,0],
  [0,7,1,4,1,7,0,0,7,2,4,1,7,0,0,0],
  [0,7,1,1,4,1,7,7,7,1,1,1,7,0,0,0],
  [0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk mid
var SPRITE_NE_WALK_MID = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,7,1,1,1,1,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,7,5,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,1,1,7,1,8,6,8,1,7,2,1,7,0,0],
  [0,0,7,4,7,1,4,6,4,1,7,1,7,0,0,0],
  [0,0,7,1,0,1,8,6,8,1,0,2,7,0,0,0],
  [0,0,0,7,0,7,4,1,4,7,0,7,0,0,0,0],
  [0,0,0,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,7,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,7,2,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,2,7,2,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,7,2,4,7,0,0,0,0,0],
  [0,0,0,7,5,1,1,7,2,1,2,7,0,0,0,0],
  [0,0,0,7,1,4,1,7,2,4,1,7,0,0,0,0],
  [0,0,0,7,1,1,4,1,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk frame B
var SPRITE_NE_WALK_B = [
  [0,0,0,0,0,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,7,1,1,1,1,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,7,5,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,7,4,1,4,7,0,0,0,0,0,0],
  [0,0,0,7,5,1,4,1,4,1,2,7,0,0,0,0],
  [0,0,7,5,1,1,4,1,4,1,1,2,7,0,0,0],
  [0,7,5,1,7,1,8,6,8,1,7,2,2,7,0,0],
  [0,7,1,4,7,1,4,6,4,1,7,2,1,7,0,0],
  [0,7,4,1,7,1,8,6,8,1,0,7,1,7,0,0],
  [0,7,2,7,0,7,4,1,4,7,0,0,7,0,0,0],
  [0,7,7,0,0,7,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,1,7,0,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,7,0,0,7,2,2,7,0,0,0],
  [0,0,0,7,1,2,7,0,0,0,7,1,7,0,0,0],
  [0,0,0,7,4,1,7,0,0,0,7,4,1,7,0,0],
  [0,0,0,7,5,1,1,7,0,7,2,1,1,7,0,0],
  [0,0,0,7,1,4,1,7,0,7,2,4,1,7,0,0],
  [0,0,0,7,1,1,4,1,7,7,1,1,1,7,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// ═══════════════════════════════════════════════════
//  SENTINEL SPRITES — 18w × 26h (larger, armored, cape)
// ═══════════════════════════════════════════════════

// SE Stand — sentinel, armored shoulders + cape pixels on back
var SPRITE_SENTINEL_SE_STAND = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,5,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,5,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,3,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,5,1,2,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,4,8,4,8,1,7,2,2,2,7,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,2,7,0],
  [0,7,1,1,7,1,8,4,8,4,8,1,7,2,1,7,0,0],
  [0,0,7,7,0,7,4,1,4,1,4,7,0,7,7,0,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,7,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,7,0,7,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,2,1,7,0,7,2,1,1,7,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,7,2,4,1,7,0,0,0,0],
  [0,0,7,5,1,1,1,7,0,7,2,1,2,1,7,0,0,0],
  [0,0,7,1,4,1,1,7,0,7,2,4,1,1,7,0,0,0],
  [0,0,7,1,1,4,1,1,7,7,1,1,1,1,7,0,0,0],
  [0,0,7,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk A — sentinel
var SPRITE_SENTINEL_SE_WALK_A = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,5,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,5,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,3,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,5,1,2,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,4,8,4,8,1,7,0,0,0,0,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,7,0,1,8,4,8,4,8,1,7,2,4,7,0,0],
  [0,0,7,0,0,7,4,1,4,1,4,7,0,7,2,7,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,7,7,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,7,2,7,0,0,0,0,0],
  [0,0,0,7,4,1,7,0,0,0,0,7,2,7,0,0,0,0],
  [0,0,7,1,1,7,0,0,0,0,7,2,2,7,0,0,0,0],
  [0,7,1,1,1,7,0,0,0,0,7,2,4,7,0,0,0,0],
  [0,7,5,1,1,7,0,0,0,7,2,1,2,7,0,0,0,0],
  [0,7,1,4,1,1,7,0,0,7,2,4,1,7,0,0,0,0],
  [0,7,1,1,4,1,1,7,7,7,1,1,1,7,0,0,0,0],
  [0,7,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk Mid — sentinel
var SPRITE_SENTINEL_SE_WALK_MID = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,5,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,5,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,3,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,5,1,2,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,1,1,7,1,8,4,8,4,8,1,7,2,1,7,0,0],
  [0,7,4,1,7,1,4,6,4,6,4,1,7,1,7,0,0,0],
  [0,0,7,1,0,1,8,4,8,4,8,1,0,2,7,0,0,0],
  [0,0,0,7,0,7,4,1,4,1,4,7,0,7,0,0,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,1,7,1,2,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,7,2,2,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,2,1,7,2,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,1,7,2,4,1,7,0,0,0,0,0],
  [0,0,0,7,5,1,1,1,7,2,1,2,1,7,0,0,0,0],
  [0,0,0,7,1,4,1,1,7,2,4,1,1,7,0,0,0,0],
  [0,0,0,7,1,1,4,1,1,1,1,1,1,7,0,0,0,0],
  [0,0,0,7,7,4,8,4,8,4,8,4,7,7,0,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk B — sentinel
var SPRITE_SENTINEL_SE_WALK_B = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,5,5,5,5,5,1,7,0,0,0,0,0],
  [0,0,0,0,7,5,5,5,5,5,1,1,1,7,0,0,0,0],
  [0,0,0,0,7,3,3,3,3,3,2,2,1,7,0,0,0,0],
  [0,0,0,0,7,5,1,2,2,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,4,8,4,8,1,7,2,2,7,0,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,7,0,0],
  [7,4,1,1,7,1,8,4,8,4,8,1,0,7,1,7,0,0],
  [7,2,7,0,0,7,4,1,4,1,4,7,0,0,7,0,0,0],
  [7,7,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,1,1,7,0,0,7,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,1,7,0,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,2,1,7,0,0,0,7,1,7,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,0,0,7,4,1,7,0,0,0],
  [0,0,0,7,5,1,1,1,7,0,7,2,1,1,7,0,0,0],
  [0,0,0,7,1,4,1,1,7,0,7,2,4,1,7,0,0,0],
  [0,0,0,7,1,1,4,1,1,7,7,1,1,1,7,0,0,0],
  [0,0,0,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Stand — sentinel
var SPRITE_SENTINEL_NE_STAND = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,5,1,1,1,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,6,8,6,8,1,7,2,2,2,7,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,2,7,0],
  [0,7,1,1,7,1,8,6,8,6,8,1,7,2,1,7,0,0],
  [0,0,7,7,0,7,4,1,4,1,4,7,0,7,7,0,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,7,1,2,2,7,0,0,0,0],
  [0,0,0,7,1,1,1,7,0,7,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,2,1,7,0,7,2,1,1,7,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,7,2,4,1,7,0,0,0,0],
  [0,0,7,5,1,1,1,7,0,7,2,1,2,1,7,0,0,0],
  [0,0,7,1,4,1,1,7,0,7,2,4,1,1,7,0,0,0],
  [0,0,7,1,1,4,1,1,7,7,1,1,1,1,7,0,0,0],
  [0,0,7,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk A — sentinel
var SPRITE_SENTINEL_NE_WALK_A = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,5,1,1,1,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,6,8,6,8,1,7,0,0,0,0,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,7,0,0],
  [0,7,1,7,0,1,8,6,8,6,8,1,7,2,4,7,0,0],
  [0,0,7,0,0,7,4,1,4,1,4,7,0,7,2,7,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,7,7,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,7,2,7,0,0,0,0,0],
  [0,0,0,7,4,1,7,0,0,0,0,7,2,7,0,0,0,0],
  [0,0,7,1,1,7,0,0,0,0,7,2,2,7,0,0,0,0],
  [0,7,1,1,1,7,0,0,0,0,7,2,4,7,0,0,0,0],
  [0,7,5,1,1,7,0,0,0,7,2,1,2,7,0,0,0,0],
  [0,7,1,4,1,1,7,0,0,7,2,4,1,7,0,0,0,0],
  [0,7,1,1,4,1,1,7,7,7,1,1,1,7,0,0,0,0],
  [0,7,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0,0],
  [0,0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk Mid — sentinel
var SPRITE_SENTINEL_NE_WALK_MID = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,5,1,1,1,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,1,1,7,1,8,6,8,6,8,1,7,2,1,7,0,0],
  [0,7,4,1,7,1,4,6,4,6,4,1,7,1,7,0,0,0],
  [0,0,7,1,0,1,8,6,8,6,8,1,0,2,7,0,0,0],
  [0,0,0,7,0,7,4,1,4,1,4,7,0,7,0,0,0,0],
  [0,0,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,1,7,1,2,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,7,2,2,1,7,0,0,0,0,0],
  [0,0,0,0,7,1,2,1,7,2,1,1,7,0,0,0,0,0],
  [0,0,0,0,7,4,1,1,7,2,4,1,7,0,0,0,0,0],
  [0,0,0,7,5,1,1,1,7,2,1,2,1,7,0,0,0,0],
  [0,0,0,7,1,4,1,1,7,2,4,1,1,7,0,0,0,0],
  [0,0,0,7,1,1,4,1,1,1,1,1,1,7,0,0,0,0],
  [0,0,0,7,7,4,8,4,8,4,8,4,7,7,0,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk B — sentinel
var SPRITE_SENTINEL_NE_WALK_B = [
  [0,0,0,0,0,0,7,7,7,7,7,7,0,0,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,2,2,2,7,0,0,0,0],
  [0,0,0,0,7,5,1,1,1,2,2,2,1,7,0,0,0,0],
  [0,0,0,0,0,7,1,1,1,1,1,2,7,0,0,0,0,0],
  [0,0,0,0,0,0,7,4,1,4,1,7,0,0,0,0,0,0],
  [0,0,7,7,5,1,4,1,4,1,4,1,2,7,7,0,0,0],
  [0,7,5,5,1,1,4,1,4,1,4,1,1,2,2,7,0,0],
  [7,5,5,1,7,1,8,6,8,6,8,1,7,2,2,7,0,0],
  [7,5,4,1,7,1,4,6,4,6,4,1,7,2,1,7,0,0],
  [7,4,1,1,7,1,8,6,8,6,8,1,0,7,1,7,0,0],
  [7,2,7,0,0,7,4,1,4,1,4,7,0,0,7,0,0,0],
  [7,7,0,0,0,7,8,1,8,1,8,7,0,0,0,0,0,0],
  [0,0,0,0,7,1,1,1,1,1,1,1,7,0,0,0,0,0],
  [0,0,0,7,1,1,7,0,0,7,2,2,1,7,0,0,0,0],
  [0,0,0,7,1,1,1,7,0,0,7,2,2,7,0,0,0,0],
  [0,0,0,7,1,2,1,7,0,0,0,7,1,7,0,0,0,0],
  [0,0,0,7,4,1,1,7,0,0,0,7,4,1,7,0,0,0],
  [0,0,0,7,5,1,1,1,7,0,7,2,1,1,7,0,0,0],
  [0,0,0,7,1,4,1,1,7,0,7,2,4,1,7,0,0,0],
  [0,0,0,7,1,1,4,1,1,7,7,1,1,1,7,0,0,0],
  [0,0,0,7,7,4,8,4,8,4,8,4,7,7,7,0,0,0],
  [0,0,0,0,7,7,7,7,7,7,7,7,7,7,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

// ═══════════════════════════════════════════════════
//  PROBE SPRITES — 12w × 18h (small, floating drone)
// ═══════════════════════════════════════════════════

// SE Stand — probe, antenna on head, no legs (floats)
var SPRITE_PROBE_SE_STAND = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,5,5,5,1,7,0,0,0],
  [0,0,0,7,3,3,2,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk A — probe (hover bob)
var SPRITE_PROBE_SE_WALK_A = [
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,5,5,5,1,7,0,0,0],
  [0,0,0,7,3,3,2,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk Mid — probe
var SPRITE_PROBE_SE_WALK_MID = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,5,5,5,1,7,0,0,0],
  [0,0,0,7,3,3,2,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// SE Walk B — probe (hover bob down)
var SPRITE_PROBE_SE_WALK_B = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,5,5,5,1,7,0,0,0],
  [0,0,0,7,3,3,2,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Stand — probe
var SPRITE_PROBE_NE_STAND = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,1,1,1,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,7,5,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk A — probe
var SPRITE_PROBE_NE_WALK_A = [
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,1,1,1,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,7,5,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk Mid — probe
var SPRITE_PROBE_NE_WALK_MID = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,1,1,1,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,7,5,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// NE Walk B — probe
var SPRITE_PROBE_NE_WALK_B = [
  [0,0,0,0,0,6,0,0,0,0,0,0],
  [0,0,0,0,0,4,0,0,0,0,0,0],
  [0,0,0,0,7,7,7,7,0,0,0,0],
  [0,0,0,7,1,1,1,2,7,0,0,0],
  [0,0,0,7,1,1,2,2,7,0,0,0],
  [0,0,0,7,5,1,2,2,7,0,0,0],
  [0,0,0,0,7,1,1,7,0,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,7,5,1,4,4,1,2,7,0,0],
  [0,7,5,1,7,4,4,7,2,1,7,0],
  [0,7,1,4,7,8,8,7,2,4,7,0],
  [0,0,7,7,7,4,4,7,7,7,0,0],
  [0,0,0,0,7,8,8,7,0,0,0,0],
  [0,0,0,7,1,1,1,1,7,0,0,0],
  [0,0,0,0,7,4,4,7,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0],
  [0,0,0,0,4,0,0,4,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
];

// ─── Direction-indexed walk frames ───
// SE/SW use SE sprites, NE/NW use NE sprites. SW/NW flip horizontally.
var WALK_FRAMES_SE = [SPRITE_SE_WALK_MID, SPRITE_SE_WALK_A, SPRITE_SE_WALK_MID, SPRITE_SE_WALK_B];
var WALK_FRAMES_NE = [SPRITE_NE_WALK_MID, SPRITE_NE_WALK_A, SPRITE_NE_WALK_MID, SPRITE_NE_WALK_B];

var WALK_FRAMES_SE_SENTINEL = [SPRITE_SENTINEL_SE_WALK_MID, SPRITE_SENTINEL_SE_WALK_A, SPRITE_SENTINEL_SE_WALK_MID, SPRITE_SENTINEL_SE_WALK_B];
var WALK_FRAMES_NE_SENTINEL = [SPRITE_SENTINEL_NE_WALK_MID, SPRITE_SENTINEL_NE_WALK_A, SPRITE_SENTINEL_NE_WALK_MID, SPRITE_SENTINEL_NE_WALK_B];

var WALK_FRAMES_SE_PROBE = [SPRITE_PROBE_SE_WALK_MID, SPRITE_PROBE_SE_WALK_A, SPRITE_PROBE_SE_WALK_MID, SPRITE_PROBE_SE_WALK_B];
var WALK_FRAMES_NE_PROBE = [SPRITE_PROBE_NE_WALK_MID, SPRITE_PROBE_NE_WALK_A, SPRITE_PROBE_NE_WALK_MID, SPRITE_PROBE_NE_WALK_B];

// Keep legacy alias for sprite cache key lookup
var WALK_FRAMES = WALK_FRAMES_SE;

// ─── Sprite cache (pre-rendered per color+frame+direction) ───
var _spriteCache = {};

function _renderSpriteToCanvas(frame, glowColor, scale) {
  // Use frame reference as part of key — check all frame arrays
  var frameKey = '_f';
  if (WALK_FRAMES_SE.indexOf(frame) >= 0) frameKey += 'se' + WALK_FRAMES_SE.indexOf(frame);
  else if (WALK_FRAMES_NE.indexOf(frame) >= 0) frameKey += 'ne' + WALK_FRAMES_NE.indexOf(frame);
  else if (WALK_FRAMES_SE_SENTINEL.indexOf(frame) >= 0) frameKey += 'sse' + WALK_FRAMES_SE_SENTINEL.indexOf(frame);
  else if (WALK_FRAMES_NE_SENTINEL.indexOf(frame) >= 0) frameKey += 'sne' + WALK_FRAMES_NE_SENTINEL.indexOf(frame);
  else if (WALK_FRAMES_SE_PROBE.indexOf(frame) >= 0) frameKey += 'pse' + WALK_FRAMES_SE_PROBE.indexOf(frame);
  else if (WALK_FRAMES_NE_PROBE.indexOf(frame) >= 0) frameKey += 'pne' + WALK_FRAMES_NE_PROBE.indexOf(frame);
  else frameKey += 'st' + frame.length + 'x' + frame[0].length;
  var key = glowColor + '_' + scale.toFixed(2) + frameKey;
  if (_spriteCache[key]) return _spriteCache[key];

  var rows = frame.length;
  var cols = frame[0].length;
  var px = PX * scale;
  var c = document.createElement('canvas');
  c.width = Math.ceil(cols * px);
  c.height = Math.ceil(rows * px);
  var cx = c.getContext('2d');
  cx.imageSmoothingEnabled = false;

  var gr = parseInt(glowColor.slice(1,3),16);
  var gg = parseInt(glowColor.slice(3,5),16);
  var gb = parseInt(glowColor.slice(5,7),16);

  for (var r = 0; r < rows; r++) {
    for (var cl = 0; cl < cols; cl++) {
      var v = frame[r][cl];
      if (v === 0) continue;
      var fr, fg, fb;
      switch (v) {
        case 7: fr = Math.floor(gr*0.08); fg = Math.floor(gg*0.08); fb = Math.floor(gb*0.08+15); break;
        case 2: fr = Math.floor(gr*0.12+8); fg = Math.floor(gg*0.12+8); fb = Math.floor(gb*0.12+18); break;
        case 1: fr = Math.floor(gr*0.18+15); fg = Math.floor(gg*0.18+15); fb = Math.floor(gb*0.18+30); break;
        case 5: fr = Math.floor(gr*0.3+25); fg = Math.floor(gg*0.3+25); fb = Math.floor(gb*0.3+45); break;
        case 8: fr = Math.floor(gr*0.5); fg = Math.floor(gg*0.5); fb = Math.floor(gb*0.5); break;
        case 4: fr = gr; fg = gg; fb = gb; break;
        case 3: fr = Math.min(255,gr+180); fg = Math.min(255,gg+180); fb = Math.min(255,gb+180); break;
        case 6: fr = Math.min(255,Math.floor(gr*0.8+100)); fg = Math.min(255,Math.floor(gg*0.8+100)); fb = Math.min(255,Math.floor(gb*0.8+100)); break;
        default: fr = gr; fg = gg; fb = gb;
      }
      cx.fillStyle = 'rgb(' + fr + ',' + fg + ',' + fb + ')';
      cx.fillRect(Math.floor(cl * px), Math.floor(r * px), Math.ceil(px), Math.ceil(px));
    }
  }

  _spriteCache[key] = c;
  return c;
}

function _getStandSprite(glowColor, scale, dir, tier) {
  var frame;
  var isNE = (dir === 'ne' || dir === 'nw');
  if (tier === 'sentinel') {
    frame = isNE ? SPRITE_SENTINEL_NE_STAND : SPRITE_SENTINEL_SE_STAND;
  } else if (tier === 'probe') {
    frame = isNE ? SPRITE_PROBE_NE_STAND : SPRITE_PROBE_SE_STAND;
  } else {
    frame = isNE ? SPRITE_NE_STAND : SPRITE_SE_STAND;
  }
  return _renderSpriteToCanvas(frame, glowColor, scale);
}

function _getWalkFrames(dir, tier) {
  var isNE = (dir === 'ne' || dir === 'nw');
  if (tier === 'sentinel') return isNE ? WALK_FRAMES_NE_SENTINEL : WALK_FRAMES_SE_SENTINEL;
  if (tier === 'probe') return isNE ? WALK_FRAMES_NE_PROBE : WALK_FRAMES_SE_PROBE;
  return isNE ? WALK_FRAMES_NE : WALK_FRAMES_SE;
}

// ─── Direction from movement vector ───

function _getDirection(dx, dy) {
  if (dx >= 0 && dy >= 0) return 'se';
  if (dx < 0 && dy >= 0) return 'sw';
  if (dx >= 0 && dy < 0) return 'ne';
  return 'nw';
}

function _needsFlip(dir) {
  return dir === 'sw' || dir === 'nw';
}

// ─── Location helpers ───

function _releaseLocation(p) {
  if (p.atLocation) {
    var idx = p.atLocation.occupants.indexOf(p);
    if (idx >= 0) p.atLocation.occupants.splice(idx, 1);
    p.atLocation = null;
    p.landingSlot = -1;
  }
}

function _assignToLocation(p, loc) {
  _releaseLocation(p);
  var slot = 0;
  for (var i = 0; i < loc.landingOffsets.length; i++) {
    var taken = false;
    for (var j = 0; j < loc.occupants.length; j++) {
      if (loc.occupants[j].landingSlot === i) { taken = true; break; }
    }
    if (!taken) { slot = i; break; }
  }
  p.locationTarget = loc;
  p.landingSlot = slot;
  p.targetX = loc.x + loc.landingOffsets[slot].dx;
  p.targetY = loc.y + loc.landingOffsets[slot].dy;
}

function _pickRandomHex(p, W, H) {
  var col = Math.floor(Math.random() * 48) + 1;
  var row = Math.floor(Math.random() * 48) + 1;
  var pos = isoToScreen(col, row);
  p.targetX = pos.x;
  p.targetY = pos.y;
  p.locationTarget = null;
}

function _pickTarget(p, W, H) {
  if (p.assignedTask) {
    var workLoc = typeof getWorkLocationForState === 'function'
      ? getWorkLocationForState(p.assignedTask.status) : null;
    if (workLoc && workLoc.occupants.length < workLoc.capacity) {
      _assignToLocation(p, workLoc);
      return;
    }
  }
  if (typeof getRandomIdleLocation === 'function') {
    var loc = getRandomIdleLocation(p.atLocation);
    if (loc) { _assignToLocation(p, loc); return; }
  }
  _pickRandomHex(p, W, H);
}

// ─── Init ───

function initAmbientPrograms(W, H) {
  ambientPrograms = [];
  var tiers = ['sentinel', 'sentinel', 'drone', 'drone', 'probe', 'probe'];
  for (var i = 0; i < PROGRAM_COUNT; i++) {
    var startPos = isoToScreen(Math.floor(Math.random() * 40) + 5, Math.floor(Math.random() * 40) + 5);
    ambientPrograms.push({
      x: startPos.x,
      y: startPos.y,
      targetX: 0,
      targetY: 0,
      speed: 20 + Math.random() * 15,
      scale: 0.9 + Math.random() * 0.3,
      glow: ['#66ccff','#cc88ff','#66ffaa','#ffbb44','#aaeeff','#ff8866'][i % 6],
      walkCycle: Math.random() * 10,
      idle: false,
      idleTimer: 0,
      direction: 'se',
      trail: [],
      _trailCounter: 0,
      locationTarget: null,
      atLocation: null,
      landingSlot: -1,
      assignedTask: null,
      programState: 'idle',
      // Tier (Feature 2)
      tier: tiers[i] || 'drone',
      // Agent binding (live orchestrator run)
      agentName: null,
      displayName: null,
      // Bunker entry state (Feature 1)
      bunkerState: 'walking',
      enterProgress: 0,
      exitProgress: 0,
      visible: true,
      // Light cycle mode (Feature 3)
      cycleMode: false,
      _baseSpeed: 20 + Math.random() * 15,
    });
    ambientPrograms[i].speed = ambientPrograms[i]._baseSpeed;
    _pickTarget(ambientPrograms[i], W, H);
  }
}

// ─── Pixel light trail ───

function _drawPixelTrail(ctx, trail, color, scale, cycleMode) {
  if (trail.length < 2) return;
  var px = PX * scale;
  ctx.save();
  for (var i = 1; i < trail.length; i++) {
    var t = i / trail.length;
    var alpha = t * (cycleMode ? 0.25 : 0.15);
    ctx.fillStyle = hexToRgba(color, alpha);
    var tw = cycleMode ? Math.max(Math.ceil(px * 1.5), 3) : Math.max(Math.ceil(px), 2);
    var th = Math.ceil(px * 0.5);
    ctx.fillRect(
      Math.floor(trail[i].x / px) * px,
      Math.floor(trail[i].y / px) * px,
      tw, th
    );
  }
  ctx.restore();
}

// ─── Update ───

function updateAmbientPrograms(W, H, dt) {
  if (!ambientPrograms.length) initAmbientPrograms(W, H);

  for (var i = 0; i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];

    // ─── Bunker state machine (Feature 1) ───
    if (p.bunkerState === 'entering') {
      p.enterProgress += dt * 2;
      if (p.enterProgress >= 1) {
        p.visible = false;
        p.bunkerState = 'inside';
        p.trail = [];
        p.idle = true;
        p.idleTimer = 5 + Math.random() * 8;
      }
      continue;
    }

    if (p.bunkerState === 'inside') {
      p.idleTimer -= dt;
      if (p.idleTimer <= 0) {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      }
      continue;
    }

    if (p.bunkerState === 'exiting') {
      p.exitProgress += dt * 2;
      if (p.exitProgress >= 1) {
        p.bunkerState = 'leaving_door';
        p.exitProgress = 1;
        // Set target 60px away from building
        if (p.atLocation) {
          var angle = Math.random() * Math.PI * 2;
          p._leaveX = p.atLocation.x + Math.cos(angle) * 60;
          p._leaveY = p.atLocation.y + Math.sin(angle) * 60;
        } else {
          p._leaveX = p.x + 60;
          p._leaveY = p.y;
        }
      }
      continue;
    }

    if (p.bunkerState === 'leaving_door') {
      var ldx = p._leaveX - p.x;
      var ldy = p._leaveY - p.y;
      var ldist = Math.sqrt(ldx * ldx + ldy * ldy);
      if (ldist < 5) {
        _releaseLocation(p);
        p.idle = false;
        p.bunkerState = 'walking';
        _pickTarget(p, W, H);
      } else {
        p.x += (ldx / ldist) * p.speed * dt;
        p.y += (ldy / ldist) * p.speed * dt;
        if (ldist > 3) p.direction = _getDirection(ldx, ldy);
        p.walkCycle += dt * 7;
      }
      continue;
    }

    // ─── Normal idle handling ───
    if (p.idle) {
      p.idleTimer -= dt;
      if (p.idleTimer <= 0) {
        p.idle = false;
        _releaseLocation(p);
        p.bunkerState = 'walking';
        _pickTarget(p, W, H);
      }
      continue;
    }

    var dx = p.targetX - p.x;
    var dy = p.targetY - p.y;
    var dist = Math.sqrt(dx * dx + dy * dy);

    // ─── Light cycle mode (Feature 3) ───
    if (dist > 200 && !p.cycleMode) {
      p.cycleMode = true;
      p.speed = p._baseSpeed * 2;
      p.trail = []; // clear trail on mode switch to avoid artifacts
    } else if (dist < 100 && p.cycleMode) {
      p.cycleMode = false;
      p.speed = p._baseSpeed;
      p.trail = [];
    }

    var maxTrail = p.cycleMode ? 40 : 25;

    if (dist < 5) {
      // ─── Arrived at target: enter bunker if location ───
      if (p.locationTarget) {
        p.atLocation = p.locationTarget;
        p.atLocation.occupants.push(p);
        p.locationTarget = null;
        // Start bunker entry sequence
        p.bunkerState = 'entering';
        p.enterProgress = 0;
        p.cycleMode = false;
        p.speed = p._baseSpeed;
        p.trail = []; // clear trail when entering building
      } else {
        p.idle = true;
        p.idleTimer = 2 + Math.random() * 4;
      }
      continue;
    }

    // Move toward target (direct interpolation for iso)
    var len = dist;
    p.x += (dx / len) * p.speed * dt;
    p.y += (dy / len) * p.speed * dt;

    // Update direction from movement
    if (dist > 3) p.direction = _getDirection(dx, dy);

    p.walkCycle += dt * 7;

    p._trailCounter += dt;
    if (p._trailCounter > 0.08) {
      // Add slight wobble so trails curve instead of being laser-straight
      var wobble = p.cycleMode ? 1.5 : 3.0;
      p.trail.push({
        x: p.x + (Math.random() - 0.5) * wobble,
        y: p.y + (Math.random() - 0.5) * wobble * 0.5,
      });
      if (p.trail.length > maxTrail) p.trail.shift();
      p._trailCounter = 0;
    }
  }
}

// ─── Draw ───

function drawAmbientPrograms(ctx, time) {
  if (!ambientPrograms.length) return;

  // Depth sort — draw programs with lower y first (further from camera)
  var sorted = ambientPrograms.slice().sort(function(a, b) { return a.y - b.y; });

  ctx.save();
  ctx.imageSmoothingEnabled = false;

  for (var i = 0; i < sorted.length; i++) {
    var p = sorted[i];

    // Skip invisible programs (inside bunker)
    if (p.visible === false) continue;

    // Calculate bunker entry/exit scale and alpha modifiers
    var bunkerScale = 1.0;
    var bunkerAlpha = 1.0;
    if (p.bunkerState === 'entering') {
      bunkerScale = 1.0 - p.enterProgress;
      bunkerAlpha = 1.0 - p.enterProgress;
    } else if (p.bunkerState === 'exiting') {
      bunkerScale = p.exitProgress;
      bunkerAlpha = p.exitProgress;
    }

    var effectiveScale = p.scale * bunkerScale;
    if (effectiveScale < 0.01) continue;

    // Pick sprite based on direction and tier
    var frames = _getWalkFrames(p.direction, p.tier);
    var flip = _needsFlip(p.direction);
    var sprite;

    if (p.idle || p.bunkerState === 'entering' || p.bunkerState === 'exiting') {
      sprite = _getStandSprite(p.glow, effectiveScale, p.direction, p.tier);
    } else {
      var frameIdx = Math.floor(p.walkCycle) % frames.length;
      sprite = _renderSpriteToCanvas(frames[frameIdx], p.glow, effectiveScale);
    }

    // Trail
    _drawPixelTrail(ctx, p.trail, p.glow, p.scale, p.cycleMode);

    var drawX = Math.floor(p.x - sprite.width / 2);
    var drawY = Math.floor(p.y - sprite.height + 6);

    // Glow pass
    ctx.save();
    ctx.globalAlpha = bunkerAlpha;
    ctx.shadowColor = p.glow;
    ctx.shadowBlur = 18;

    if (flip) {
      ctx.save();
      ctx.translate(drawX + sprite.width, drawY);
      ctx.scale(-1, 1);
      ctx.drawImage(sprite, 0, 0);
      ctx.restore();
    } else {
      ctx.drawImage(sprite, drawX, drawY);
    }

    // Crisp pass
    ctx.shadowBlur = 0;
    if (flip) {
      ctx.save();
      ctx.translate(drawX + sprite.width, drawY);
      ctx.scale(-1, 1);
      ctx.drawImage(sprite, 0, 0);
      ctx.restore();
    } else {
      ctx.drawImage(sprite, drawX, drawY);
    }
    ctx.restore();

    // Isometric ground shadow (diamond-shaped)
    var px = PX * p.scale;
    ctx.fillStyle = hexToRgba(p.glow, 0.06 * bunkerAlpha);
    drawIsoDiamond(ctx, p.x, p.y + 4, px * 8, px * 4);
    ctx.fill();

    // ─── Task status icon above head (Feature 5) ───
    if (p.assignedTask && typeof getStateColor === 'function') {
      var taskColor = getStateColor(p.assignedTask.status);
      var iconPx = PX * 4;
      var bobY = Math.sin(time * 3) * 2;
      var iconX = Math.floor(p.x - iconPx / 2);
      var iconY = Math.floor(drawY - iconPx - 4 + bobY);
      ctx.save();
      ctx.globalAlpha = bunkerAlpha * 0.9;
      ctx.shadowColor = taskColor;
      ctx.shadowBlur = 6;
      ctx.fillStyle = taskColor;
      ctx.fillRect(iconX, iconY, iconPx, iconPx);
      ctx.shadowBlur = 0;
      ctx.fillRect(iconX, iconY, iconPx, iconPx);
      ctx.restore();
    }

    // Floating task label above head while traveling
    if (p.assignedTask && p.visible && p.bunkerState === 'walking') {
      ctx.save();
      ctx.fillStyle = hexToRgba(getStateColor(p.assignedTask.status), 0.7);
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      var labelY = drawY - 4 + Math.sin(time * 3) * 2;
      ctx.fillText(p.assignedTask.title ? p.assignedTask.title.slice(0, 20) : p.assignedTask.id, Math.floor(p.x), Math.floor(labelY));
      ctx.restore();
    }
  }

  ctx.restore();
}

// ─── Work assignment (called from websocket.js) ───

function routeProgramsToTasks(tasks) {
  if (!ambientPrograms.length) return;

  // Build agent->task map
  var taskByAgent = {};
  for (var t = 0; t < tasks.length; t++) {
    if (tasks[t].status !== 'pending') {
      taskByAgent[tasks[t].agent] = tasks[t];
    }
  }

  for (var i = 0; i < ambientPrograms.length; i++) {
    var p = ambientPrograms[i];
    var task = p.agentName ? taskByAgent[p.agentName] : null;

    // Also try positional fallback for idle mode (no agentName set)
    if (!task && !p.agentName && i < tasks.length && tasks[i].status !== 'pending') {
      task = tasks[i];
    }

    if (task) {
      // Check if task changed
      if (p.assignedTask && p.assignedTask.id === task.id && p.assignedTask.status === task.status) continue;

      p.assignedTask = { id: task.id, status: task.status, title: task.title };
      p.programState = 'working';

      // If inside a building, trigger exit first
      if (p.bunkerState === 'inside') {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      } else if (p.bunkerState === 'walking' || p.bunkerState === 'leaving_door') {
        _releaseLocation(p);
        _pickTarget(p, 0, 0);
      }
      // If entering/exiting, let it finish then re-route naturally
    } else if (p.assignedTask) {
      // Task completed or removed -- return to idle
      p.assignedTask = null;
      p.programState = 'idle';
      if (p.bunkerState === 'inside') {
        p.bunkerState = 'exiting';
        p.exitProgress = 0;
        p.visible = true;
      }
    }
  }
}

// ─── Roster sync (called from websocket.js when roster data arrives) ───

function syncProgramsToRoster(data) {
  var agents = [];
  var glows = ['#66ccff','#cc88ff','#66ffaa','#ffbb44','#aaeeff','#ff8866',
               '#ff88cc','#88ffcc','#ccff88','#88ccff','#ffcc88','#cc88ff'];

  // Build agent list with tiers
  if (data.managers) {
    for (var m = 0; m < data.managers.length; m++) {
      agents.push({ name: data.managers[m].name, title: data.managers[m].title, tier: 'sentinel' });
    }
  }
  if (data.sub_agents) {
    for (var s = 0; s < data.sub_agents.length; s++) {
      agents.push({ name: data.sub_agents[s].name, title: data.sub_agents[s].title, tier: 'drone' });
    }
  }
  if (data.reviewers) {
    for (var r = 0; r < data.reviewers.length; r++) {
      agents.push({ name: data.reviewers[r].name, title: data.reviewers[r].title, tier: 'probe' });
    }
  }

  if (agents.length === 0) return;

  // Check if roster changed (different agent count or names)
  var changed = agents.length !== ambientPrograms.length;
  if (!changed) {
    for (var i = 0; i < agents.length; i++) {
      if (!ambientPrograms[i] || ambientPrograms[i].agentName !== agents[i].name) {
        changed = true;
        break;
      }
    }
  }

  if (!changed) return; // roster unchanged, skip respawn

  // Respawn programs matching roster
  // Save existing XP data if roster panel exists
  var savedXP = {};
  if (typeof rosterData !== 'undefined') {
    for (var rx = 0; rx < rosterData.length; rx++) {
      savedXP[rosterData[rx].name] = { xp: rosterData[rx].xp, level: rosterData[rx].level };
    }
  }

  // Release all current locations
  for (var ri = 0; ri < ambientPrograms.length; ri++) {
    _releaseLocation(ambientPrograms[ri]);
  }

  ambientPrograms = [];
  for (var ai = 0; ai < agents.length; ai++) {
    var agent = agents[ai];
    var startPos = isoToScreen(Math.floor(Math.random() * 40) + 5, Math.floor(Math.random() * 40) + 5);
    ambientPrograms.push({
      x: startPos.x,
      y: startPos.y,
      targetX: 0,
      targetY: 0,
      speed: 20 + Math.random() * 15,
      scale: 0.9 + Math.random() * 0.3,
      glow: glows[ai % glows.length],
      walkCycle: Math.random() * 10,
      idle: false,
      idleTimer: 0,
      direction: 'se',
      trail: [],
      _trailCounter: 0,
      locationTarget: null,
      atLocation: null,
      landingSlot: -1,
      assignedTask: null,
      programState: 'idle',
      bunkerState: 'walking',
      enterProgress: 0,
      exitProgress: 0,
      visible: true,
      cycleMode: false,
      _baseSpeed: 20 + Math.random() * 15,
      tier: agent.tier,
      agentName: agent.name,
      displayName: agent.title,
    });
    ambientPrograms[ai].speed = ambientPrograms[ai]._baseSpeed;
    _pickTarget(ambientPrograms[ai], 0, 0);
  }

  // Re-init roster if available
  if (typeof initRoster === 'function') initRoster();
}

// ─── Revert to idle programs when orchestrator finishes ───

function revertToIdlePrograms() {
  for (var i = 0; i < ambientPrograms.length; i++) {
    _releaseLocation(ambientPrograms[i]);
  }
  ambientPrograms = [];
  PROGRAM_COUNT = 6; // reset for clarity
  // Re-init will happen on next updateAmbientPrograms call
}
