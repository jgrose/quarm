// ═══ TRON GRID LOCATIONS — SNES 16-BIT LANDMARKS ═══
// Pixel-art buildings/structures that programs visit on the hex grid

var gridLocations = [];
var _locationSpriteCache = {};
var LOC_PX = PX * 2; // locations render at 2× program pixel size (buildings > people)

// ─── Palette (same as programs + index 9 for door/accent) ───
// 0=transparent 1=body 2=shadow 3=window/bright 4=circuit 5=highlight
// 6=accent bright 7=outline 8=circuit dim 9=door/warm accent

// Location sprites (compressed — uses _ds from draw_programs.js)
var _dsl = _ds;
var SPRITE_EOL_CLUB = _dsl("000000000000000046640000000000000000000000000000000466664000000000000000000000000000004664466400000000000000000000000000000440044000000000000000000000000000000077770000000000000000000000000000777755557777000000000000000000000077554555555422770000000000000000007755545554455542227700000000000000775555455555555554222277000000000077555554555555555555422222770000000755555545555555555555542222227000007777777777777777777777777777777700007515151515151515721212121212121700000751335141335157721212121212127000000075331545331577221332181332170000000007151541515772121331282331700000000000751545157721212121281217000000000000075141577212121212182170000000000000007545772121212121281700000000000000007545772121212121281700000000000000075141577212121212182170000000000000751545157721999921281217000000000007151541515772099012182121700000000075151545151577099021281212170000000751515141515157099012182121217000007515151545151517700721281212121700007484848444848487784784848484848700000777777777777777777777777777777000000072270000000077000000000722700000000007700000000000000000000077000000000000000000000000000000000000000000000000000000000000000000000000000000", 36);
var SPRITE_ARENA = _dsl("000000000000000000077000000000000000000000000000000000000775577000000000000000000000000000000007755555577000000000000000000000000000077554555545577000000000000000000000000775554555555455577000000000000000000007755554555555554555577000000000000000077555554555555555545555577000000000000775555554555555555555455555577000000007777777777777777777777777777777777000007515172222222222222222222222227121270000075172282244442222244442282227721270000000717282242222422242222422827721270000000007722242266224242266224227721270000000000072822422224222422224287721270000000000000722224444222224444227721270000000000000007282222222222222227721270000000000000000072222222222222227721270000000000000000000772222222222227721270000000000000000000000772222222227721270000000000000000000000000772222227721270000000000000000000000000000772227721270000000000000000000000000000000777721270000000000000000000000000000000000777770000000000000000000000000000000000000000000000000000000", 40);
var SPRITE_IO_TOWER = _dsl("000000000006600000000000000000000006600000000000000000000004400000000000000000000004400000000000000000000076670000000000000000000755227000000000000000000777777000000000000000000754217000000000000000000751227000000000000000000745827000000000000000000753217000000000000000007554222700000000000000007777777700000000000000007514821700000000000000007511221700000000000000007541821700000000000000007513212700000000000000075154221270000000000000077777777770000000000000075114821170000000000000075151212170000000000000075411812170000000000000075113221270000000000000751514821217000000000000777777777777000000000000751514821217000000000000751151212127000000000000754111811217000000000000751113221127000000000007515154821212700000000007777777777777700000000007515114812121700000000007511511211212700000000007541151812112700000000007511513221212700000000075151514821212170000000077777777777777770000000755555555222222227000007777777777777777777700007515151515721212121700007484848484784848484700007777777777777777777700000000000000000000000000000000000000000000000000", 24);
var SPRITE_DISC_RING = _dsl("00000000000000077000000000000000000000000000077557700000000000000000000000077545542770000000000000000000077545555542277000000000000000077545555555542227700000000000077545555555555542222770000000077545557777777777542222277000007545557722822228227754222227000007555772282222228227752222700000007577228226666228227722270000000007722822644446228227727000000000077222264666646222277270000000000772282646446462822772700000000007722226464464622227727000000000077228264666646282277270000000000772222264444622222772700000000075772282266662282277222700000007555772282222228227722222700000754555772282222822772222222700000775455577777777775222222770000000077545555555555542222770000000000007754555555554222770000000000000000775455555422770000000000000000000077545542770000000000000000000000007755770000000000000000000000000000770000000000000000000000000000000000000000000000000000000000000000000000000000000", 32);
var SPRITE_RECOGNIZER = _dsl("000000000000077777777770000000000000000000000000711411141170000000000000000000000007111111111117000000000000000000000075111333311157000000000000000000000071111111111117000000000000000000000007771111117770000000000000000000000000072111127000000000000000000000000000072211227000000000000000000000000000007777770000000000000000000000000000000077000000000000000000000000000000000755700000000000000000000000000000077544577000000000000000000000000007754555542770000000000000000000000775455555554227700000000000000000077545548484855422277000000000000007754554855555584542222770000000000775455485555555555845422227700000077777777777777777777777777777777770007515151515151515721212121212121700000751515451515157721212124212127000000075151515151577212121212121270000000007151451515772121241212121700000000000751515157721212121212127000000000000075451577212412121212700000000000000007515772121212121270000000000000000000777777777777777700000000000000000000000000000000000000000000000000000000000000000000000000000000000", 36);
var SPRITE_PORTAL = _dsl("000000000000066000000000000000000000000064460000000000000000000000064664600000000000000000000075455427000000000000000000075545542270000000000000000075554554222700000000000000075577777777227000000000000075570064460072270000000000007547064884607287000000000000751704866840721700000000000075470648846072870000000000007517048668407217000000000000754706488460728700000000000075170466664072170000000000007547004884007287000000000000751700066000721700000000000075470000000072870000000000007517000000007217000000000000754700000000728700000000000075170000000072170000000000007547000000007287000000000000751700000000721700000000000075470009900072870000000000007517000990007217000000000000754700099000728700000000000754847000000748427000000000754848477777748484270000000777777777777777777777700000751515151515721212121217000007515151515772121212127000000074848484778484848487000000000777777777777777777000000000007270000000000727000000000000070000000000007000000000000000000000000000000000000000000000000000000000000000", 28);
var SPRITE_CODE_FORGE = _dsl("000000000330000300003300000000000000000000003333003330033330000000000000000000000000000000000000000000000000000000000000000077000000000000000000000000000000777755777700000000000000000000000077555555552277000000000000000000007753333355333222770000000000000000775344334355343322227700000000000077533334343355334332222277000000000753343333333355333333222222700000007777777777777777777777777777770000007515151515151515721212121212170000000751514515151557721212181212700000000075151541515177212128121217000000000007515154151772121281212170000000000000751515457721212812121700000000000000075151577212121212127000000000000000007515772121212121270000000000000000007515772121212121270000000000000000075151577212121212127000000000000000751515457721212812121700000000000007515154151772121281212170000000000075151541515177212128121217000000000751514515151557721212181212700000007515151515151515721212121212170000007484848484848484784848484848470000000777777777777777777777777777700000000072700072700000727000727000000000000074700074700000747000747000000000000077700077700000777000777000000000000000000000000000000000000000000000000000000000000000000000000000000000", 36);
var SPRITE_TRIBUNAL = _dsl("000000000000000000000000000000000000000000000000000666666000000000000000000000000000007455554700000000000000000000000000075133331270000000000000000000000000751177771127000000000000000000000007777777777777700000000000000000000075545555555542270000000000000000000755455555555554227000000000000000007777777777777777777700000000000000075151515151721212121270000000000000751515151517721212121217000000000007777777777777777777777777700000000075545555555555555555542222270000000755455555555555555555554222227000007777777777777777777777777777777700007515151515151515721212121212121700000751516515151557721216121212127000000075156515151577212161212121700000000007516651515772121661212127000000000000751615157721212612121270000000000000075151577212121212121700000000000000751515457721212181212170000000000007515154151772121218121217000000000077777777777777777777777777770000000755555555555555222222222222227000007484848484848484784848484848484700007777777777777777777777777777777700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 36);
var SPRITE_ANALYSIS = _dsl("0000000000000077770000000000000000000000000777544577700000000000000000000775566556622770000000000000000775555555555522277000000000000777777777777777777777700000000007547000000000000078217000000000075170000000000000712170000000000754700000000000007821700000000007517000066666600071217000000000075470000488884000782170000000000751700000000000007121700000000007547000000000000078217000000000075170000666666000712170000000000754700004888840007821700000000007517000000000000071217000000000075470000000000000782170000000000751700000000000007121700000000007777777777777777777777000000000755555555555522222222227000000077777777777777777777777777000007515151515151572121212121217000007515145151517721218121212700000007515151515772121212121270000000007514515177212181212127000000000007515157721212121212700000000000007451772128121212170000000000000007777777777777777000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 32);
var SPRITE_RECOMPILE = _dsl("00000000000000000440000000000000000000000000000048470000000000000000000000000004111170000000000000000000000007777148700000000000000000000000000770000000000000000000000000077775577770000000000000000000777555555552277700000000000000775554555555542222770000000000777777777777777777777777000000007515172282222822721212170000000007517282266222822721217000000000007172226696222222721700000000000071728226622282227217000000000000717228222228222272170000000000075157777777777777212170000000007515151515151572121212170000000751515145151557721218121270000075151515151515772121212121270000748484848484847784848484848700000777777777777777777777777770000000727000000727000000727000000000007470000007470000007470000000000077700000077700000077700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 32);
var SPRITE_VAULT = _dsl("00000000000000077000000000000000000000000000777557770000000000000000000007775545542277700000000000000077755455555554222777000000000777554555555555555422227770000077777777777777777777777777770000751515151515151721212121212700000751515451515177212128121270000000751515151517721212121217000000000751777771772177777212700000000000717222757717222271270000000000007172772577172772712700000000000074727447774727447827000000000000717274477717274471270000000000007172772577172772712700000000000074722665774722667827000000000000717222257717222271270000000000075117777157721777721270000000007515151515177212121212170000000751515451515177212128121270000075151515151515172121212121270000748484848484848784848484848700007777777777777777777777777777000000727000000000000000007270000000007770000000000000000077700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 32);
var SPRITE_DEREZZED = _dsl("000006000000060000040000060000000000000064600006400000460000646000000000000004000000000000000000040000000000000000000000000007700000000000000000000000000000077772277770000000000000000000000077724222224227770000000000000000077727472272272472227770000000000077722472827727728274222227770000007722427242272424272242724222227700077777777777777777777777777777777770007242722724272828272427227242227700000077224728277272772827427722270000000000772742272272272247277227000000000000007727472272274727722700000000000000000077224222422772270000000000000000000000772424277227000000000000000000000000007777772270000000000000000000000000000000777700000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 36);

// ═══════════════════════════════════════════════════
//  LOCATION DEFINITIONS
// ═══════════════════════════════════════════════════

var LOCATION_DEFS = [
  // Idle locations
  { id: 'eol_club',       name: 'END OF LINE',     category: 'idle', taskState: null,                    sprite: SPRITE_EOL_CLUB,   glowColor: '#cc88ff', animType: 'pulse',   zone: 'bottom-left',  capacity: 3 },
  { id: 'arena',          name: 'CYCLE ARENA',      category: 'idle', taskState: null,                    sprite: SPRITE_ARENA,      glowColor: '#66ccff', animType: 'pulse',   zone: 'bottom-right', capacity: 3 },
  { id: 'io_tower',       name: 'I/O TOWER',        category: 'idle', taskState: null,                    sprite: SPRITE_IO_TOWER,   glowColor: '#66ffaa', animType: 'pulse',   zone: 'bottom-center',capacity: 2 },
  { id: 'disc_ring',      name: 'DISC RING',        category: 'idle', taskState: null,                    sprite: SPRITE_DISC_RING,  glowColor: '#ffbb44', animType: 'pulse',   zone: 'mid-left',     capacity: 2 },
  { id: 'recognizer_pad', name: 'RECOGNIZER PAD',   category: 'idle', taskState: null,                    sprite: SPRITE_RECOGNIZER, glowColor: '#ff8866', animType: 'flicker', zone: 'mid-right',    capacity: 2 },
  { id: 'portal',         name: 'PORTAL',           category: 'idle', taskState: null,                    sprite: SPRITE_PORTAL,     glowColor: '#aaeeff', animType: 'pulse',   zone: 'top-center',   capacity: 2 },

  // Work locations
  { id: 'code_forge',     name: 'CODE FORGE',       category: 'work', taskState: 'in_progress',           sprite: SPRITE_CODE_FORGE, glowColor: '#66ccff', animType: 'pulse',   zone: 'work-center',  capacity: 3 },
  { id: 'tribunal',       name: 'TRIBUNAL',         category: 'work', taskState: 'in_manager_review',     sprite: SPRITE_TRIBUNAL,   glowColor: '#ffbb44', animType: 'pulse',   zone: 'work-left',    capacity: 2 },
  { id: 'analysis_bay',   name: 'ANALYSIS BAY',     category: 'work', taskState: 'in_specialist_review',  sprite: SPRITE_ANALYSIS,   glowColor: '#cc88ff', animType: 'scan',    zone: 'work-right',   capacity: 2 },
  { id: 'recompile',      name: 'RECOMPILE',        category: 'work', taskState: 'revision',              sprite: SPRITE_RECOMPILE,  glowColor: '#ff8800', animType: 'flicker', zone: 'work-below',   capacity: 2 },
  { id: 'data_vault',     name: 'DATA VAULT',       category: 'work', taskState: 'done',                  sprite: SPRITE_VAULT,      glowColor: '#66ffaa', animType: 'pulse',   zone: 'work-far-right', capacity: 3 },
  { id: 'derezzed',       name: 'DEREZZED',         category: 'work', taskState: 'failed',                sprite: SPRITE_DEREZZED,   glowColor: '#ff5566', animType: 'flicker', zone: 'work-far-left',  capacity: 2 },
];

// ═══════════════════════════════════════════════════
//  COORDINATE & LAYOUT
// ═══════════════════════════════════════════════════

function _locGridToScreen(col, row) {
  return isoToScreen(col, row);
}

function initLocations(W, H) {
  var midCol = 25;
  var midRow = 25;

  // Zone → (col, row) — spread across ~50×50 grid, 10+ tiles between buildings
  var zoneMap = {
    'top-center':    { col: midCol,      row: 4 },
    'mid-left':      { col: 4,           row: midRow + 2 },
    'mid-right':     { col: 46,          row: midRow + 2 },
    'bottom-left':   { col: 8,           row: 44 },
    'bottom-center': { col: midCol,      row: 46 },
    'bottom-right':  { col: 42,          row: 44 },
    'work-center':   { col: midCol,      row: midRow - 6 },
    'work-left':     { col: midCol - 10, row: midRow - 4 },
    'work-right':    { col: midCol + 10, row: midRow - 4 },
    'work-below':    { col: midCol,      row: midRow + 6 },
    'work-far-right':{ col: midCol + 14, row: midRow + 2 },
    'work-far-left': { col: midCol - 14, row: midRow + 2 },
  };

  gridLocations = [];
  for (var i = 0; i < LOCATION_DEFS.length; i++) {
    var def = LOCATION_DEFS[i];
    var z = zoneMap[def.zone] || { col: midCol, row: midRow };
    var col = Math.max(1, Math.min(49, z.col));
    var row = Math.max(1, Math.min(49, z.row));
    var pos = _locGridToScreen(col, row);

    var spriteW = def.sprite[0].length;
    var spriteH = def.sprite.length;

    gridLocations.push({
      id: def.id,
      name: def.name,
      category: def.category,
      taskState: def.taskState,
      col: col,
      row: row,
      x: pos.x,
      y: pos.y,
      sprite: def.sprite,
      spriteW: spriteW,
      spriteH: spriteH,
      glowColor: def.glowColor,
      animType: def.animType,
      animSpeed: 1.0,
      capacity: def.capacity,
      occupants: [],
      landingOffsets: _makeLandingOffsets(spriteW, spriteH, def.capacity),
      _spriteCanvas: null,
      // Building upgrades (Feature 4)
      taskCompletions: 0,
      upgradeLevel: 0,
    });
  }
}

function _makeLandingOffsets(spriteW, spriteH, capacity) {
  var pw = spriteW * LOC_PX;
  var offsets = [];
  var spacing = pw / (capacity + 1);
  for (var i = 0; i < capacity; i++) {
    offsets.push({
      dx: Math.floor(-pw / 2 + spacing * (i + 1)),
      dy: Math.floor(spriteH * LOC_PX * 0.15),
    });
  }
  return offsets;
}

// ═══════════════════════════════════════════════════
//  CYBER ROADS — glowing paths between buildings
// ═══════════════════════════════════════════════════

var ROAD_CONNECTIONS = [
  // Work cluster interconnections
  ['code_forge', 'tribunal'],
  ['code_forge', 'analysis_bay'],
  ['code_forge', 'recompile'],
  ['tribunal', 'derezzed'],
  ['analysis_bay', 'data_vault'],
  ['data_vault', 'recompile'],
  ['derezzed', 'recompile'],
  // Idle → work corridors
  ['portal', 'code_forge'],
  ['eol_club', 'tribunal'],
  ['arena', 'analysis_bay'],
  ['disc_ring', 'derezzed'],
  ['recognizer_pad', 'data_vault'],
  ['io_tower', 'recompile'],
  // Idle ring road
  ['portal', 'disc_ring'],
  ['portal', 'recognizer_pad'],
  ['eol_club', 'io_tower'],
  ['arena', 'io_tower'],
  ['eol_club', 'disc_ring'],
  ['arena', 'recognizer_pad'],
];

// Perimeter paths around each building (extra road tiles in a ring)
var BUILDING_PERIMETERS = true; // draw 1-tile ring around each building

var _roadTiles = null;

function _buildRoadTiles() {
  if (_roadTiles) return _roadTiles;
  if (!gridLocations.length) return [];
  _roadTiles = [];
  var locMap = {};
  for (var i = 0; i < gridLocations.length; i++) {
    locMap[gridLocations[i].id] = gridLocations[i];
  }

  for (var r = 0; r < ROAD_CONNECTIONS.length; r++) {
    var fromLoc = locMap[ROAD_CONNECTIONS[r][0]];
    var toLoc = locMap[ROAD_CONNECTIONS[r][1]];
    if (!fromLoc || !toLoc) continue;

    // Bresenham-style walk on iso grid between two locations
    var c0 = fromLoc.col, r0 = fromLoc.row;
    var c1 = toLoc.col, r1 = toLoc.row;
    var dc = Math.abs(c1 - c0), dr = Math.abs(r1 - r0);
    var sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1;
    var err = dc - dr;
    var cc = c0, rr = r0;
    var steps = 0;
    var maxSteps = dc + dr + 2;

    while (steps < maxSteps) {
      // Skip tiles directly under buildings (first/last 1 tile)
      if (!(cc === c0 && rr === r0) && !(cc === c1 && rr === r1)) {
        var pos = isoToScreen(cc, rr);
        _roadTiles.push({ x: pos.x, y: pos.y, col: cc, row: rr });
      }
      if (cc === c1 && rr === r1) break;
      var e2 = 2 * err;
      if (e2 > -dr) { err -= dr; cc += sc; }
      if (e2 < dc)  { err += dc; rr += sr; }
      steps++;
    }
  }
  // Add perimeter paths (1-tile ring around each building)
  if (BUILDING_PERIMETERS) {
    var perimSet = {};
    for (var pi = 0; pi < gridLocations.length; pi++) {
      var bl = gridLocations[pi];
      var offsets = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
      for (var oi = 0; oi < offsets.length; oi++) {
        var pc = bl.col + offsets[oi][0];
        var pr = bl.row + offsets[oi][1];
        var pk = pc + '_' + pr;
        if (!perimSet[pk]) {
          perimSet[pk] = true;
          var pp = isoToScreen(pc, pr);
          _roadTiles.push({ x: pp.x, y: pp.y, col: pc, row: pr });
        }
      }
    }
  }

  return _roadTiles;
}

function drawRoads(ctx, time) {
  var tiles = _buildRoadTiles();
  if (!tiles.length) return;

  var timeSin = time * 0.8;

  ctx.save();
  for (var i = 0; i < tiles.length; i++) {
    var t = tiles[i];
    var pulse = 0.7 + Math.sin(timeSin + (t.col + t.row) * 0.3) * 0.3;

    // Road surface — dark filled diamond
    ctx.globalAlpha = 0.7 * pulse;
    ctx.fillStyle = '#08081a';
    drawIsoDiamond(ctx, t.x, t.y, TILE_W * 0.95, TILE_H * 0.95);
    ctx.fill();

    // Road edge outline
    ctx.globalAlpha = 0.5 * pulse;
    ctx.strokeStyle = C.holoBase;
    ctx.lineWidth = 0.8;
    drawIsoDiamond(ctx, t.x, t.y, TILE_W * 0.95, TILE_H * 0.95);
    ctx.stroke();

    // Circuit center line (brighter)
    ctx.globalAlpha = 0.6 * pulse;
    ctx.strokeStyle = C.holoBase;
    ctx.lineWidth = 1.2;
    drawIsoDiamond(ctx, t.x, t.y, TILE_W * 0.35, TILE_H * 0.35);
    ctx.stroke();

    // Corner glow dots at all 4 diamond vertices
    ctx.globalAlpha = 0.5 * pulse;
    ctx.fillStyle = C.holoBase;
    ctx.beginPath();
    ctx.arc(t.x, t.y - TILE_H * 0.45, 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(t.x, t.y + TILE_H * 0.45, 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(t.x + TILE_W * 0.45, t.y, 1.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(t.x - TILE_W * 0.45, t.y, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}

// ═══════════════════════════════════════════════════
//  CYBER TREES — decorative isometric foliage
// ═══════════════════════════════════════════════════

// Tree type sprites (small, isometric pixel art)
// 0=transparent 1=trunk 2=shadow 4=glow 5=highlight 6=bright 7=outline 8=dim

// Crystal spire (8w × 14h)
// Round canopy cyber-tree (10w × 12h)
var TREE_CRYSTAL = [
  [0,0,0,4,4,4,4,0,0,0],
  [0,0,4,5,5,5,5,4,0,0],
  [0,4,5,4,5,5,4,2,4,0],
  [4,5,5,5,6,6,5,2,2,4],
  [4,5,4,5,5,5,4,2,2,4],
  [4,5,5,4,5,4,2,2,2,4],
  [0,4,5,5,4,4,2,2,4,0],
  [0,0,4,4,4,4,4,4,0,0],
  [0,0,0,0,7,7,0,0,0,0],
  [0,0,0,0,7,7,0,0,0,0],
  [0,0,0,0,7,2,0,0,0,0],
  [0,0,0,0,7,7,0,0,0,0],
];

// Data bush (10w × 8h)
var TREE_BUSH = [
  [0,0,0,4,8,4,8,0,0,0],
  [0,0,4,5,1,1,1,4,0,0],
  [0,4,5,1,4,1,4,1,4,0],
  [4,5,1,4,1,1,1,4,2,4],
  [4,5,1,1,1,4,1,1,2,4],
  [0,4,1,4,1,1,4,2,4,0],
  [0,0,4,1,1,1,1,4,0,0],
  [0,0,0,7,8,7,8,0,0,0],
];

// Light pole (6w × 16h)
var TREE_POLE = [
  [0,0,6,6,0,0],
  [0,6,4,4,6,0],
  [0,0,6,6,0,0],
  [0,0,4,4,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,7,0,0],
  [0,0,7,2,0,0],
  [0,0,7,7,0,0],
];

// Marker arrays for new geometric-only types (no pixel data, just need unique refs)
var TREE_OBELISK = [[1]];
var TREE_ARC_LAMP = [[2]];

var TREE_TYPES = [TREE_CRYSTAL, TREE_BUSH, TREE_POLE, TREE_OBELISK, TREE_ARC_LAMP];
// type 0=crystal, 1=bush, 2=streetlight, 3=obelisk, 4=arc lamp

// Fixed tree placements (col, row, typeIndex, glowColor)
// type 0=crystal, 1=bush, 2=streetlight
var TREE_PLACEMENTS = [
  // ─── Streetlights along portal→forge road ───
  { col: 25, row: 8,  type: 2, glow: '#aaeeff' },
  { col: 25, row: 11, type: 2, glow: '#aaeeff' },
  { col: 25, row: 14, type: 2, glow: '#aaeeff' },
  // ─── Streetlights along forge→tribunal road ───
  { col: 20, row: 18, type: 2, glow: '#ffbb44' },
  { col: 18, row: 19, type: 2, glow: '#ffbb44' },
  // ─── Streetlights along forge→analysis road ───
  { col: 30, row: 18, type: 2, glow: '#cc88ff' },
  { col: 32, row: 19, type: 2, glow: '#cc88ff' },
  // ─── Streetlights along forge→recompile road ───
  { col: 25, row: 22, type: 2, glow: '#ff8800' },
  { col: 25, row: 26, type: 2, glow: '#ff8800' },
  // ─── Streetlights along idle→work corridors ───
  { col: 16, row: 14, type: 2, glow: '#66ccff' },
  { col: 12, row: 17, type: 2, glow: '#66ccff' },
  { col: 34, row: 14, type: 2, glow: '#66ccff' },
  { col: 38, row: 17, type: 2, glow: '#66ccff' },
  // ─── Perimeter trees around portal ───
  { col: 23, row: 3,  type: 0, glow: '#aaeeff' },
  { col: 27, row: 3,  type: 0, glow: '#aaeeff' },
  { col: 24, row: 5,  type: 1, glow: '#aaeeff' },
  { col: 26, row: 5,  type: 1, glow: '#aaeeff' },
  // ─── Perimeter around End of Line club ───
  { col: 2,  row: 26, type: 0, glow: '#cc88ff' },
  { col: 6,  row: 26, type: 0, glow: '#cc88ff' },
  { col: 3,  row: 28, type: 1, glow: '#cc88ff' },
  { col: 5,  row: 24, type: 1, glow: '#cc88ff' },
  // ─── Perimeter around arena ───
  { col: 44, row: 26, type: 0, glow: '#66ccff' },
  { col: 48, row: 26, type: 0, glow: '#66ccff' },
  { col: 45, row: 28, type: 1, glow: '#66ccff' },
  { col: 47, row: 24, type: 1, glow: '#66ccff' },
  // ─── Perimeter around work cluster buildings ───
  { col: 24, row: 17, type: 1, glow: '#66ccff' },
  { col: 26, row: 17, type: 1, glow: '#66ccff' },
  { col: 14, row: 20, type: 0, glow: '#ffbb44' },
  { col: 16, row: 22, type: 1, glow: '#ffbb44' },
  { col: 36, row: 20, type: 0, glow: '#cc88ff' },
  { col: 34, row: 22, type: 1, glow: '#cc88ff' },
  // ─── Scattered forest patches ───
  { col: 8,  row: 10, type: 0, glow: '#66ffaa' },
  { col: 9,  row: 11, type: 1, glow: '#66ffaa' },
  { col: 10, row: 10, type: 0, glow: '#66ffaa' },
  { col: 40, row: 10, type: 0, glow: '#ff8866' },
  { col: 41, row: 11, type: 1, glow: '#ff8866' },
  { col: 42, row: 10, type: 0, glow: '#ff8866' },
  // ─── South boulevard streetlights ───
  { col: 15, row: 40, type: 2, glow: '#66ffaa' },
  { col: 20, row: 42, type: 2, glow: '#66ccff' },
  { col: 25, row: 42, type: 2, glow: '#66ccff' },
  { col: 30, row: 42, type: 2, glow: '#66ffaa' },
  { col: 35, row: 40, type: 2, glow: '#66ffaa' },
  // ─── Central park (cluster between work buildings) ───
  { col: 22, row: 24, type: 0, glow: '#66ffaa' },
  { col: 23, row: 25, type: 1, glow: '#66ffaa' },
  { col: 24, row: 24, type: 0, glow: '#aaeeff' },
  { col: 26, row: 24, type: 0, glow: '#aaeeff' },
  { col: 27, row: 25, type: 1, glow: '#66ffaa' },
  { col: 28, row: 24, type: 0, glow: '#66ffaa' },
  // ─── Perimeter around I/O Tower ───
  { col: 24, row: 45, type: 1, glow: '#66ffaa' },
  { col: 26, row: 45, type: 1, glow: '#66ffaa' },
  { col: 24, row: 47, type: 0, glow: '#66ffaa' },
  { col: 26, row: 47, type: 0, glow: '#66ffaa' },
  // ─── Obelisks (type 3) — data monoliths ───
  { col: 22, row: 8,  type: 3, glow: '#aaeeff' },
  { col: 28, row: 8,  type: 3, glow: '#aaeeff' },
  { col: 13, row: 24, type: 3, glow: '#cc88ff' },
  { col: 37, row: 24, type: 3, glow: '#cc88ff' },
  { col: 20, row: 44, type: 3, glow: '#66ffaa' },
  { col: 30, row: 44, type: 3, glow: '#66ffaa' },
  { col: 6,  row: 36, type: 3, glow: '#ff8866' },
  { col: 44, row: 36, type: 3, glow: '#ff8866' },
  // ─── Arc lamps (type 4) — modern curved streetlights ───
  { col: 22, row: 18, type: 4, glow: '#66ccff' },
  { col: 28, row: 18, type: 4, glow: '#66ccff' },
  { col: 10, row: 30, type: 4, glow: '#cc88ff' },
  { col: 40, row: 30, type: 4, glow: '#cc88ff' },
  { col: 18, row: 40, type: 4, glow: '#66ffaa' },
  { col: 32, row: 40, type: 4, glow: '#66ffaa' },
  { col: 15, row: 20, type: 4, glow: '#ffbb44' },
  { col: 35, row: 20, type: 4, glow: '#ffbb44' },
];

var _treeCache = {};
var _treesInit = false;
var _treeObjects = [];
var _perimSet = {};

function _initTrees() {
  if (_treesInit) return;
  _treesInit = true;
  _treeObjects = [];
  for (var i = 0; i < TREE_PLACEMENTS.length; i++) {
    var tp = TREE_PLACEMENTS[i];
    var pos = isoToScreen(tp.col, tp.row);
    _treeObjects.push({
      x: pos.x,
      y: pos.y,
      col: tp.col,
      row: tp.row,
      sprite: TREE_TYPES[tp.type],
      glow: tp.glow,
    });
  }
}

// ─── Polygonal Tron-style tree rendering (direct canvas draw) ───

function _drawTreeCrystal(ctx, x, y, glow, time) {
  // Diamond spire — tall crystal growing from ground
  var h = 36, w = 14;
  ctx.save();
  // Spire body (diamond)
  ctx.beginPath();
  ctx.moveTo(x, y - h);           // top point
  ctx.lineTo(x + w / 2, y - h * 0.35); // right shoulder
  ctx.lineTo(x + w * 0.3, y);     // right base
  ctx.lineTo(x - w * 0.3, y);     // left base
  ctx.lineTo(x - w / 2, y - h * 0.35); // left shoulder
  ctx.closePath();
  ctx.fillStyle = hexToRgba(glow, 0.08);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(glow, 0.6);
  ctx.lineWidth = 1;
  ctx.stroke();
  // Inner edge line
  ctx.beginPath();
  ctx.moveTo(x, y - h);
  ctx.lineTo(x, y);
  ctx.strokeStyle = hexToRgba(glow, 0.2);
  ctx.lineWidth = 0.5;
  ctx.stroke();
  // Bright tip
  ctx.beginPath();
  ctx.arc(x, y - h, 2, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(glow, 0.9);
  ctx.fill();
  ctx.restore();
}

function _drawTreeBush(ctx, x, y, glow, time) {
  // Hexagonal data node — low geometric cluster
  var r = 10, h = 8;
  ctx.save();
  // Hex shape
  ctx.beginPath();
  for (var i = 0; i < 6; i++) {
    var angle = (Math.PI / 3) * i - Math.PI / 6;
    var px = x + Math.cos(angle) * r;
    var py = y - h / 2 + Math.sin(angle) * r * 0.5; // squashed for iso
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = hexToRgba(glow, 0.06);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(glow, 0.5);
  ctx.lineWidth = 1;
  ctx.stroke();
  // Center dot
  ctx.beginPath();
  ctx.arc(x, y - h / 2, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(glow, 0.7);
  ctx.fill();
  ctx.restore();
}

function _drawTreePole(ctx, x, y, glow, time) {
  // Light pole — vertical line with glowing top
  var h = 50;
  ctx.save();
  // Pole shaft
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y - h);
  ctx.strokeStyle = hexToRgba(glow, 0.3);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  // Lamp glow
  var pulse = 0.6 + Math.sin(time * 2 + x * 0.01) * 0.2;
  ctx.beginPath();
  ctx.arc(x, y - h, 3, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(glow, pulse);
  ctx.fill();
  // Light cone (subtle triangle beneath lamp)
  ctx.beginPath();
  ctx.moveTo(x - 8, y - h + 12);
  ctx.lineTo(x, y - h + 2);
  ctx.lineTo(x + 8, y - h + 12);
  ctx.closePath();
  ctx.fillStyle = hexToRgba(glow, 0.04);
  ctx.fill();
  ctx.restore();
}

function _drawTreeObelisk(ctx, x, y, glow, time) {
  // Tall narrow monolith — rectangular slab with beveled top edge
  var w = 6, h = 44;
  var pulse = 0.5 + Math.sin(time * 1.5 + y * 0.02) * 0.15;
  ctx.save();
  // Main slab
  ctx.beginPath();
  ctx.moveTo(x - w / 2, y);               // bottom-left
  ctx.lineTo(x - w / 2, y - h + 4);       // top-left below bevel
  ctx.lineTo(x, y - h);                    // apex
  ctx.lineTo(x + w / 2, y - h + 4);       // top-right below bevel
  ctx.lineTo(x + w / 2, y);               // bottom-right
  ctx.closePath();
  ctx.fillStyle = hexToRgba(glow, 0.06);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(glow, 0.5);
  ctx.lineWidth = 1;
  ctx.stroke();
  // Vertical circuit line
  ctx.beginPath();
  ctx.moveTo(x, y - 4);
  ctx.lineTo(x, y - h + 6);
  ctx.strokeStyle = hexToRgba(glow, 0.3 * pulse);
  ctx.lineWidth = 0.8;
  ctx.stroke();
  // Data dots along the circuit line
  for (var d = 0; d < 3; d++) {
    var dotY = y - 10 - d * 10;
    if (dotY < y - h + 8) break;
    ctx.beginPath();
    ctx.arc(x, dotY, 1, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(glow, 0.5 + d * 0.15);
    ctx.fill();
  }
  ctx.restore();
}

function _drawTreeArcLamp(ctx, x, y, glow, time) {
  // Modern arc lamp — curved arm with hanging light
  var h = 46, armLen = 16;
  var pulse = 0.55 + Math.sin(time * 2.5 + x * 0.02) * 0.25;
  ctx.save();
  // Vertical shaft
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y - h);
  ctx.strokeStyle = hexToRgba(glow, 0.3);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  // Curved arm (quadratic bezier arching right)
  ctx.beginPath();
  ctx.moveTo(x, y - h);
  ctx.quadraticCurveTo(x + armLen * 0.8, y - h - 4, x + armLen, y - h + 6);
  ctx.strokeStyle = hexToRgba(glow, 0.35);
  ctx.lineWidth = 1;
  ctx.stroke();
  // Hanging light at arm tip
  var lampX = x + armLen, lampY = y - h + 6;
  ctx.beginPath();
  ctx.moveTo(lampX - 3, lampY);
  ctx.lineTo(lampX, lampY + 5);
  ctx.lineTo(lampX + 3, lampY);
  ctx.closePath();
  ctx.fillStyle = hexToRgba(glow, pulse);
  ctx.fill();
  // Ground light pool (ellipse)
  ctx.beginPath();
  ctx.ellipse(lampX, y, 10, 3, 0, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(glow, 0.03);
  ctx.fill();
  // Base plate
  ctx.beginPath();
  ctx.moveTo(x - 5, y);
  ctx.lineTo(x + 5, y);
  ctx.strokeStyle = hexToRgba(glow, 0.4);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

var _treeDrawFuncs = [_drawTreeCrystal, _drawTreeBush, _drawTreePole, _drawTreeObelisk, _drawTreeArcLamp];

function drawSingleTree(ctx, t, time) {
  var typeIdx = TREE_TYPES.indexOf(t.sprite);
  var fn = _treeDrawFuncs[typeIdx] || _drawTreeCrystal;
  fn(ctx, t.x, t.y, t.glow, time);
}

function drawTrees(ctx, time) {
  _initTrees();
  if (!_treeObjects.length) return;
  // Trees are now drawn from drawAmbientPrograms for proper depth sorting.
  // This function is kept as a fallback if programs haven't initialized yet.
  if (ambientPrograms && ambientPrograms.length > 0) return;

  var treeView = (typeof getVisibleRect === 'function') ? getVisibleRect(
    (document.getElementById('canvas') || {}).width / (window.devicePixelRatio || 1) || 800,
    (document.getElementById('canvas') || {}).height / (window.devicePixelRatio || 1) || 600
  ) : null;
  var treeMargin = 60;

  for (var i = 0; i < _treeObjects.length; i++) {
    var t = _treeObjects[i];
    if (treeView && (t.x < treeView.x - treeMargin || t.x > treeView.x + treeView.w + treeMargin ||
        t.y < treeView.y - treeMargin || t.y > treeView.y + treeView.h + treeMargin)) continue;
    drawSingleTree(ctx, t, time);
  }
}

// ═══════════════════════════════════════════════════
//  SPRITE RENDERING & CACHE
// ═══════════════════════════════════════════════════

function _renderLocationSprite(loc) {
  var key = loc.id + '_' + loc.glowColor;
  if (_locationSpriteCache[key]) return _locationSpriteCache[key];

  var frame = loc.sprite;
  var rows = frame.length;
  var cols = frame[0].length;
  var px = LOC_PX;
  var c = document.createElement('canvas');
  c.width = cols * px;
  c.height = rows * px;
  var cx = c.getContext('2d');
  cx.imageSmoothingEnabled = false;

  var gr = parseInt(loc.glowColor.slice(1,3),16);
  var gg = parseInt(loc.glowColor.slice(3,5),16);
  var gb = parseInt(loc.glowColor.slice(5,7),16);

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
        case 9: fr = Math.min(255,Math.floor(gr*0.3+180)); fg = Math.min(255,Math.floor(gg*0.3+120)); fb = Math.floor(gb*0.2+40); break;
        default: fr = gr; fg = gg; fb = gb;
      }
      cx.fillStyle = 'rgb(' + fr + ',' + fg + ',' + fb + ')';
      cx.fillRect(cl * px, r * px, px, px);
    }
  }

  _locationSpriteCache[key] = c;
  return c;
}

// ═══════════════════════════════════════════════════
//  DRAWING
// ═══════════════════════════════════════════════════

function drawAllLocations(ctx, W, H, time, dt) {
  if (!config.locations) return;
  if (!gridLocations.length) initLocations(W, H);

  var alpha = 1.0;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = false;

  // 1. Roads (behind everything)
  drawRoads(ctx, time);

  // 2. Trees (behind buildings, depth-sorted by y)
  drawTrees(ctx, time);

  // 3. Buildings (depth-sorted by y — back to front, viewport culled)
  // When programs are active, buildings are drawn from drawAmbientPrograms for proper z-ordering
  if (typeof ambientPrograms === 'undefined' || !ambientPrograms.length) {
    var locView = (typeof getVisibleRect === 'function') ? getVisibleRect(W, H) : { x: 0, y: 0, w: W, h: H };
    var locMargin = 100;
    if (!drawAllLocations._sorted || drawAllLocations._sortLen !== gridLocations.length) {
      drawAllLocations._sorted = gridLocations.slice().sort(function(a, b) { return a.y - b.y; });
      drawAllLocations._sortLen = gridLocations.length;
    }
    var sorted = drawAllLocations._sorted;
    for (var i = 0; i < sorted.length; i++) {
      var loc = sorted[i];
      if (loc.x < locView.x - locMargin || loc.x > locView.x + locView.w + locMargin ||
          loc.y < locView.y - locMargin || loc.y > locView.y + locView.h + locMargin) continue;
      _drawSingleLocation(ctx, loc, time);
    }
  }

  // 4. Task flow arrows between buildings
  _updateAndDrawArrows(ctx, dt || 0.016, time);

  ctx.restore();
}

function _drawSingleLocation(ctx, loc, time) {
  var sq = config.shadowQuality || 'high';
  var sprite = _renderLocationSprite(loc);

  // Animation modulation
  var animAlpha = 1.0;
  if (loc.animType === 'pulse') {
    animAlpha = 0.85 + Math.sin(time * (loc.animSpeed || 1) * 1.5 + loc.x * 0.01) * 0.15;
  } else if (loc.animType === 'flicker') {
    animAlpha = Math.random() > 0.04 ? 1.0 : 0.5;
  } else if (loc.animType === 'scan') {
    animAlpha = 0.9 + Math.sin(time * 2.5) * 0.1;
  }

  var drawX = Math.floor(loc.x - sprite.width / 2);
  var drawY = Math.floor(loc.y - sprite.height + LOC_PX * 4);

  // Glow + crisp pass (shadow quality aware)
  ctx.save();
  ctx.globalAlpha *= animAlpha;
  if (sq !== 'off') {
    ctx.shadowColor = loc.glowColor;
    ctx.shadowBlur = sq === 'low' ? 5 : 14;
    ctx.drawImage(sprite, drawX, drawY);
    ctx.shadowBlur = 0;
  }
  ctx.drawImage(sprite, drawX, drawY);
  ctx.restore();

  // ─── Door animation (Feature 1) ───
  var doorOpen = false;
  for (var j = 0; j < loc.occupants.length; j++) {
    if (loc.occupants[j].bunkerState === 'entering' || loc.occupants[j].bunkerState === 'exiting') {
      doorOpen = true;
      break;
    }
  }
  if (doorOpen) {
    ctx.save();
    ctx.fillStyle = 'rgba(5, 5, 16, 0.9)';
    ctx.fillRect(loc.x - LOC_PX * 3, loc.y - LOC_PX * 2, LOC_PX * 6, LOC_PX * 4);
    ctx.strokeStyle = hexToRgba(loc.glowColor, 0.4);
    ctx.lineWidth = 1;
    ctx.strokeRect(loc.x - LOC_PX * 3, loc.y - LOC_PX * 2, LOC_PX * 6, LOC_PX * 4);
    ctx.restore();
  }

  // ─── Building upgrade overlays (Feature 4) ───
  if (loc.upgradeLevel >= 1) {
    // Level 1: circuit dots at building corners
    var cDots = [
      { dx: -sprite.width / 2 + 4, dy: -sprite.height + LOC_PX * 4 + 4 },
      { dx:  sprite.width / 2 - 4, dy: -sprite.height + LOC_PX * 4 + 4 },
      { dx: -sprite.width / 2 + 4, dy: LOC_PX * 2 },
      { dx:  sprite.width / 2 - 4, dy: LOC_PX * 2 },
    ];
    ctx.save();
    ctx.fillStyle = hexToRgba(loc.glowColor, 0.7);
    if (sq !== 'off') { ctx.shadowColor = loc.glowColor; ctx.shadowBlur = sq === 'low' ? 3 : 6; }
    for (var cd = 0; cd < cDots.length; cd++) {
      ctx.beginPath();
      ctx.arc(loc.x + cDots[cd].dx, loc.y + cDots[cd].dy, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  if (loc.upgradeLevel >= 2) {
    // Level 2: antenna on top of building
    var antennaX = loc.x;
    var antennaBaseY = drawY + 2;
    ctx.save();
    ctx.strokeStyle = hexToRgba(loc.glowColor, 0.6);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(antennaX, antennaBaseY);
    ctx.lineTo(antennaX, antennaBaseY - LOC_PX * 3);
    ctx.stroke();
    ctx.fillStyle = hexToRgba(loc.glowColor, 0.9);
    if (sq !== 'off') { ctx.shadowColor = loc.glowColor; ctx.shadowBlur = sq === 'low' ? 2 : 4; }
    ctx.beginPath();
    ctx.arc(antennaX, antennaBaseY - LOC_PX * 3 - 2, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  if (loc.upgradeLevel >= 3) {
    // Level 3: pulsing iso diamond ring around building base
    var ringPulse = 0.5 + Math.sin(time * 2) * 0.3;
    ctx.save();
    ctx.strokeStyle = hexToRgba(loc.glowColor, ringPulse);
    ctx.lineWidth = 1.5;
    if (sq !== 'off') { ctx.shadowColor = loc.glowColor; ctx.shadowBlur = sq === 'low' ? 4 : 8; }
    var ringW = sprite.width * 0.9;
    var ringH = sprite.width * 0.45;
    drawIsoDiamond(ctx, loc.x, loc.y + LOC_PX, ringW, ringH);
    ctx.stroke();
    ctx.restore();
  }

  // Ground shadow diamond (anchors building to iso grid)
  var groundW = sprite.width * 0.75;
  var groundH = groundW * 0.5;
  ctx.fillStyle = hexToRgba(loc.glowColor, 0.06);
  drawIsoDiamond(ctx, loc.x, loc.y + LOC_PX, groundW, groundH);
  ctx.fill();

  // Label — larger with dark background pill
  ctx.save();
  var labelSize = 13;
  if (config.lodEnabled && camera.zoom < 0.8) {
    labelSize = Math.round(13 / Math.max(camera.zoom, 0.25));
  }
  ctx.font = 'bold ' + labelSize + 'px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  var labelText = loc.name;
  var labelY = loc.y + LOC_PX * 3;
  var tw = ctx.measureText(labelText).width;
  ctx.fillStyle = 'rgba(5, 5, 16, 0.55)';
  ctx.beginPath();
  ctx.roundRect(loc.x - tw / 2 - 6, labelY - 2, tw + 12, labelSize + 6, 3);
  ctx.fill();
  ctx.fillStyle = hexToRgba(loc.glowColor, 0.85);
  ctx.fillText(labelText, loc.x, labelY);
  ctx.restore();

  // Active building indicator — pulsing ring when programs are working inside
  var activeCount = 0;
  for (var oi = 0; oi < loc.occupants.length; oi++) {
    if (loc.occupants[oi].assignedTask) activeCount++;
  }

  if (activeCount > 0 && config.activeIndicators) {
    // Pulsing activity ring (iso diamond around building base)
    var ringPulse = 0.5 + Math.sin(time * 3) * 0.3;
    ctx.save();
    ctx.globalAlpha = ringPulse;
    ctx.strokeStyle = loc.glowColor;
    ctx.lineWidth = 2;
    var ringW = sprite.width * 0.8;
    var ringH = ringW * 0.5; // iso ratio
    drawIsoDiamond(ctx, loc.x, loc.y + LOC_PX, ringW, ringH);
    ctx.stroke();
    ctx.restore();

    // Task count badge
    ctx.save();
    ctx.fillStyle = hexToRgba(loc.glowColor, 0.8);
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(activeCount + ' ACTIVE', loc.x, loc.y - sprite.height + LOC_PX * 3);
    ctx.restore();
  }
}

// ═══════════════════════════════════════════════════
//  LOCATION LOOKUPS (used by draw_programs.js)
// ═══════════════════════════════════════════════════

function getWorkLocationForState(status) {
  for (var i = 0; i < gridLocations.length; i++) {
    if (gridLocations[i].taskState === status) return gridLocations[i];
  }
  return null;
}

function getRandomIdleLocation(exclude) {
  var available = [];
  for (var i = 0; i < gridLocations.length; i++) {
    var loc = gridLocations[i];
    if (loc.category === 'idle' && loc.occupants.length < loc.capacity && loc !== exclude) {
      available.push(loc);
    }
  }
  if (available.length === 0) return null;
  return available[Math.floor(Math.random() * available.length)];
}

// ─── Building upgrade system (Feature 4) ───

function incrementBuildingCompletion(locationId) {
  for (var i = 0; i < gridLocations.length; i++) {
    var loc = gridLocations[i];
    if (loc.id === locationId) {
      loc.taskCompletions++;
      if (loc.taskCompletions >= 15) loc.upgradeLevel = 3;
      else if (loc.taskCompletions >= 7) loc.upgradeLevel = 2;
      else if (loc.taskCompletions >= 3) loc.upgradeLevel = 1;
      else loc.upgradeLevel = 0;
      return loc;
    }
  }
  return null;
}

// ─── Click-to-inspect location (Feature 1) ───

function getLocationAt(e) {
  var cvs = document.getElementById('canvas');
  var rect = cvs.getBoundingClientRect();
  var sx = e.clientX - rect.left;
  var sy = e.clientY - rect.top;
  var world = screenToWorld(sx, sy, cvs);
  for (var i = 0; i < gridLocations.length; i++) {
    var loc = gridLocations[i];
    var sprite = _renderLocationSprite(loc);
    var lx = loc.x - sprite.width / 2;
    var ly = loc.y - sprite.height + LOC_PX * 4;
    if (world.x >= lx && world.x <= lx + sprite.width && world.y >= ly && world.y <= ly + sprite.height) return loc;
  }
  return null;
}

// ═══════════════════════════════════════════════════
//  TASK FLOW ARROWS — animated arrows between buildings
// ═══════════════════════════════════════════════════

var _taskFlowArrows = []; // {fromX, fromY, toX, toY, age, duration, color}

function addTaskFlowArrow(fromLocId, toLocId, color) {
  var fromLoc = null, toLoc = null;
  for (var i = 0; i < gridLocations.length; i++) {
    if (gridLocations[i].id === fromLocId) fromLoc = gridLocations[i];
    if (gridLocations[i].id === toLocId) toLoc = gridLocations[i];
  }
  if (!fromLoc || !toLoc) return;
  _taskFlowArrows.push({
    fromX: fromLoc.x, fromY: fromLoc.y,
    toX: toLoc.x, toY: toLoc.y,
    age: 0,
    duration: 3.0,
    color: color || '#66ccff',
  });
}

function _updateAndDrawArrows(ctx, dt, time) {
  if (!config.taskArrows) return;
  for (var i = _taskFlowArrows.length - 1; i >= 0; i--) {
    var a = _taskFlowArrows[i];
    a.age += dt;
    if (a.age >= a.duration) {
      _taskFlowArrows.splice(i, 1);
      continue;
    }

    var progress = a.age / a.duration;
    var fadeAlpha = progress < 0.1 ? progress * 10 : (1 - progress) * 1.1;
    fadeAlpha = Math.max(0, Math.min(1, fadeAlpha));

    // Draw arrow line
    ctx.save();
    ctx.globalAlpha = fadeAlpha * 0.6;
    ctx.strokeStyle = a.color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.lineDashOffset = -time * 30;
    ctx.beginPath();
    ctx.moveTo(a.fromX, a.fromY);
    ctx.lineTo(a.toX, a.toY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrowhead
    var angle = Math.atan2(a.toY - a.fromY, a.toX - a.fromX);
    var headLen = 8;
    ctx.beginPath();
    ctx.moveTo(a.toX, a.toY);
    ctx.lineTo(a.toX - headLen * Math.cos(angle - 0.4), a.toY - headLen * Math.sin(angle - 0.4));
    ctx.moveTo(a.toX, a.toY);
    ctx.lineTo(a.toX - headLen * Math.cos(angle + 0.4), a.toY - headLen * Math.sin(angle + 0.4));
    ctx.stroke();

    ctx.restore();
  }
}

// ─── Location ID lookup by task state ───

function getLocationIdForState(status) {
  for (var i = 0; i < gridLocations.length; i++) {
    if (gridLocations[i].taskState === status) return gridLocations[i].id;
  }
  return null;
}

// ── Building State Serialization ────────────────────────────────────────────

function serializeBuildingState() {
  var result = { buildings: {} };
  for (var i = 0; i < gridLocations.length; i++) {
    var loc = gridLocations[i];
    result.buildings[loc.id] = {
      taskCompletions: loc.taskCompletions || 0,
      upgradeLevel: loc.upgradeLevel || 0
    };
  }
  return result;
}

function deserializeBuildingState(data) {
  if (!data || !data.buildings) return;
  for (var i = 0; i < gridLocations.length; i++) {
    var loc = gridLocations[i];
    var saved = data.buildings[loc.id];
    if (saved) {
      loc.taskCompletions = saved.taskCompletions || 0;
      // Recompute upgrade level from thresholds to stay consistent
      if (loc.taskCompletions >= 15) loc.upgradeLevel = 3;
      else if (loc.taskCompletions >= 7) loc.upgradeLevel = 2;
      else if (loc.taskCompletions >= 3) loc.upgradeLevel = 1;
      else loc.upgradeLevel = 0;
    }
  }
}
