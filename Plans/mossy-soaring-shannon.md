# Plan: Live Orchestrator Integration Overhaul

## Context

The Tron city has two completely separate visualization modes that don't interact:
- **Idle**: 6 hardcoded ambient programs walk the city
- **Active**: Programs hide entirely, floating hex nodes appear with force layout

This wastes the city. When a real orchestrator plan runs, the city should come alive — programs should represent actual agents, physically walk between buildings as tasks progress through the review chain, and the hex node system should be replaced by the city itself as the primary visualization.

---

## Core Design: Unified City Mode

**Remove the dual-mode split.** Instead of `nodes.size > 0` hiding everything:

1. When orchestrator sends roster data, **spawn programs matching the real agent roster** (dynamic count, not hardcoded 6)
2. Each program represents a real agent (drone/sentinel/probe matching agent tier)
3. Programs physically walk to buildings matching their current task state
4. Buildings stay at full alpha, gain activity indicators
5. Hex nodes become optional overlay (small floating badges), not the primary viz
6. When no orchestrator is running, fall back to idle ambient programs as before

---

## Changes by File

### Stream A: `draw_programs.js` — Dynamic roster-driven programs

**Remove hardcoded PROGRAM_COUNT = 6.** Replace with dynamic spawning:

```
When applyStatus receives roster data:
  1. Count total agents: sub_agents.length + managers.length + reviewers.length
  2. If ambientPrograms.length !== agentCount, respawn programs matching roster
  3. Each program gets: name (from roster), tier (drone/sentinel/probe), glow (from palette cycle)
  4. Sentinel programs = managers, Drone programs = sub_agents, Probe programs = reviewers
```

**Remove the three `nodes.size > 0` early-return guards:**
- `updateAmbientPrograms` — remove guard, programs always update
- `drawAmbientPrograms` — remove guard, programs always draw  
- `routeProgramsToTasks` — remove guard, always route

**Enhance `routeProgramsToTasks`:**
- Match programs to tasks by agent name (not positional index)
- When task status changes, program walks from current building to new building matching new state
- Idle programs (no task) wander between idle locations as before
- Track `program.agentName` field for roster binding

**Add task-carrying animation:**
- When program walks between work buildings (task state change), show task title as small floating text above head
- When program arrives at building, trigger bunker entry as normal

### Stream B: `draw_locations.js` — Active building indicators

**Remove alpha dimming** when nodes exist — buildings always full brightness.

**Add active building indicators:**
- When a building has programs working inside (occupants with assignedTask), draw a pulsing activity ring around it
- Show task count badge: small "2/3" text showing how many tasks are active at that building
- When a task completes at a building, trigger a completion burst effect (reuse existing `spawnCompleteEffect`)

**Add building-to-building task flow arrows:**
- When a task transitions (e.g., `in_progress` → `in_manager_review`), draw a brief animated arrow from Code Forge to Tribunal
- Arrow follows the road path between buildings
- Fades after 3 seconds

### Stream C: `websocket.js` — Roster-to-program binding

**Modify `applyStatus`:**

```javascript
// After rebuildNodes (keep nodes for data, but don't require rendering)
if (data.sub_agents || data.managers || data.reviewers) {
  rebuildNodes(data);  // keep for edge/data tracking
  syncProgramsToRoster(data);  // NEW: spawn/update city programs from roster
}
```

**New function `syncProgramsToRoster(data)`:**
- Counts total agents from roster
- If program count doesn't match, reinitializes `ambientPrograms` with correct count
- Assigns each program: `agentName`, `tier`, `displayName` from roster
- Preserves XP/level if roster hasn't changed (just task reassignment)

**Modify task state change handling:**
- On `in_progress → in_manager_review`: find program by agent name, retarget to Tribunal
- On review → revision: retarget to Recompile  
- On done: retarget to Data Vault, trigger building completion
- On failed: retarget to Derezzed

### Stream D: `render.js` + `draw_agents.js` — Optional hex node overlay

**Make hex nodes optional**, controlled by a new `config.hexNodes` flag (default: false):

```javascript
// In render loop, gate agent node rendering:
if (config.hexNodes) {
  drawAllEdges(ctx, currentTime);
  drawAllAgents(ctx, currentTime);
}
```

The city IS the visualization now. Hex nodes become a debug/overlay mode you can toggle.

Keep edges drawing optionally between buildings (as road highlights) rather than between floating nodes.

### Stream E: `draw_roster.js` — Sync roster panel with real agents

Update roster panel to show real agent names/titles during active runs instead of TRON names:
- When `syncProgramsToRoster` fires, update `rosterData` entries with real agent names
- Show task assignment in roster: "TASK-001: Build auth API"
- Show task state as status: "IN REVIEW" / "WORKING" / "REVISING"
- When orchestrator finishes, revert to Tron names

---

## Execution Plan (3 parallel subagents)

| Agent | Stream | Files | Why Together |
|-------|--------|-------|-------------|
| **Agent 1** | A+C | draw_programs.js, websocket.js | Tightly coupled — roster sync drives program spawning |
| **Agent 2** | B+D | draw_locations.js, render.js, draw_agents.js | Building indicators + render pipeline changes |
| **Agent 3** | E | draw_roster.js | Independent panel update |

After agents complete, I wire config flags into `nodes.js` and keyboard shortcuts into `init.js`.

---

## Config Additions

```javascript
// nodes.js config
config.hexNodes = false;    // show floating hex nodes (debug overlay)
config.taskArrows = true;   // show task flow arrows between buildings
config.activeIndicators = true; // show activity rings on buildings
```

---

## Verification

1. **No plan running**: 6 ambient programs walk the city as before (idle mode unchanged)
2. **Start a plan**: Programs respawn matching agent roster (e.g., 3 sub-agents + 1 manager + 2 reviewers = 6 programs with correct tiers)
3. **Task dispatched**: Program walks from idle location to Code Forge, enters building
4. **Task in review**: Program exits Code Forge, walks to Tribunal (manager review) or Analysis Bay (specialist review)
5. **Task revised**: Program walks to Recompile station
6. **Task done**: Program walks to Data Vault, building completion counter increments
7. **Task failed**: Program walks to Derezzed zone
8. **Roster panel (R key)**: Shows real agent names and task assignments during active run
9. **Hex nodes (toggle)**: Can optionally enable floating hex nodes as overlay
10. **Plan completes**: Programs revert to idle wandering with Tron names
