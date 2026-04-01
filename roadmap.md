# NORT Roadmap

## Completed

- [x] Hex grid background with pulsing glow
- [x] SNES 16-bit pixel-art Tron programs (walk animation, light trails)
- [x] 12 Tron-themed locations (6 idle, 6 work) with pixel-art sprites
- [x] Location-aware pathfinding (programs walk to buildings)
- [x] Zoom/pan camera on grid world
- [x] Isometric (Sim City SNES) conversion — diamond grid, 3/4 view buildings, 4-directional sprites
- [x] Cyber roads between buildings with glowing circuit lines
- [x] Cyber trees (round canopy, data bush, light pole) + streetlights along roads
- [x] Perimeter paths around buildings
- [x] Bunker entry — programs enter buildings, door animations
- [x] Program tier sprites — sentinel (large), drone (medium), probe (small)
- [x] Light cycle trails — thick bright trails for long-distance travel
- [x] Building upgrades — visual evolution at task completion thresholds
- [x] Live task visualization — floating task icons above working programs
- [x] Agent roster panel — collapsible sidebar with Tron names, XP, leveling
- [x] Day/night cycle — 2-minute phase rotation (dawn/day/dusk/night)
- [x] Weather effects — data rain, lightning between buildings during storms
- [x] Minimap — bottom-left overview with click-to-pan
- [x] Enhanced sound design — ambient hum, footsteps, door whoosh, level-up, thunder
- [x] Live orchestrator integration — programs represent real agents during plan runs
- [x] Dynamic roster spawning — programs match agent roster count and roles
- [x] Task flow arrows — animated dashed lines between buildings on state transitions
- [x] Active building indicators — pulsing ring + badge on busy buildings
- [x] Hex nodes optional — toggle overlay mode for debug view
- [x] Roster panel live sync — real agent names, task assignments, tier icons during runs
- [x] Dynamic agent registry — persistent agent definitions with performance tracking (in progress)

---

## In Progress

- [ ] **Agent registry & adaptive roles** — persistent JSON storage, CRUD API, plan generator integration, performance tracking, UI panel

---

## Backlog — High Priority

### Testing & Stability
- [ ] **End-to-end QA with live plan** — run full orchestrator plan, verify programs walk between buildings correctly, roster shows real names, arrows fire, building upgrades increment
- [ ] **Performance audit** — profile FPS with all 10+ draw systems active, optimize sprite cache sizes, road tile count, reduce redundant redraws
- [ ] **Error resilience** — graceful handling when WebSocket disconnects mid-run, reconnect and restore city state
- [ ] **Save/persist city state** — building upgrades, XP/levels, layout to localStorage or server so progress survives refresh

### Orchestrator Improvements
- [ ] **Plan preview mode** — when typing a plan description, ghost-highlight which buildings will activate and which agents will spawn
- [ ] **Task dependency visualization** — show dependency chains as visual connections between tasks/programs, blocked tasks shown with lock icon
- [ ] **Parallel execution indicators** — when multiple tasks run simultaneously, show visual "lightning bolt" connections between active programs
- [ ] **Review chain animation** — when task moves through review stages, show the program physically carrying a "document" sprite between buildings
- [ ] **Failure recovery visualization** — when a task fails and retries, show program walking to Derezzed Zone then back to Code Forge with "revision" badge

---

## Backlog — Medium Priority

### City & Visual Polish
- [ ] **Building interiors** — click a building to see detailed panel: occupants, task history, upgrade level, completion stats
- [ ] **Building click panel** — proper modal with occupant list, not repurposed agent detail card
- [ ] **Program click + follow** — click a program in the city to lock camera follow, show stats overlay
- [ ] **More building types** — Library (knowledge base), Server Farm (parallel tasks), Comms Tower (inter-agent messaging), Prison (failed task quarantine)
- [ ] **Building entry/exit polish** — actual door frame sprites, light spill from inside, particle burst on entry/exit
- [ ] **Procedural city generation** — auto-layout buildings based on plan complexity (more tasks = bigger city, more agents = more buildings)
- [ ] **City zones/districts** — industrial zone (work buildings), entertainment zone (idle buildings), research zone, each with distinct visual theme
- [ ] **Road traffic** — small light cycle sprites racing along road paths between buildings (decorative, separate from agent programs)
- [ ] **Recognizer flyovers** — occasional recognizer shape flies overhead casting shadow, purely atmospheric
- [ ] **Depth of field** — blur distant buildings slightly when zoomed in close

### Program Behavior
- [ ] **Program personalities** — each Tron name gets behavior traits (CLU moves fastest, RINZLER prefers combat buildings, QUORRA explores furthest)
- [ ] **Inter-program interactions** — programs stop and "talk" when they cross paths (speech bubble exchange), challenge each other at Disc Ring
- [ ] **Economy system** — programs earn credits at work locations, spend at idle locations, visible currency counter
- [ ] **Program customization** — click program to change color, name, preferred tools
- [ ] **Idle animations** — programs do context-appropriate actions when at buildings (typing at Code Forge, sparring at Disc Ring, drinking at End of Line)

### UI & Panels
- [ ] **Task history timeline** — scrubber bar showing past orchestrator runs, replay city activity like a DVR
- [ ] **Notification system** — toast popups when programs level up, buildings upgrade, weather changes, tasks complete
- [ ] **Config panel integration** — add toggles for all new features (weather, minimap, day/night, etc.) to the existing config overlay
- [ ] **Mobile/touch support** — pinch to zoom, tap to select, responsive panel layout

---

## Backlog — Exploratory / Fun

### Audio & Music
- [ ] **Procedural Tron soundtrack** — generative synth music that reacts to activity level (busier = more layers, calmer = ambient drone)
- [ ] **Spatial audio** — sounds positioned based on where they happen in the city (footsteps louder when zoomed in on a program)
- [ ] **Building ambient sounds** — each building type has unique background audio (forge = hammering, arena = crowd, club = bass)

### Advanced Visualization
- [ ] **3D isometric** — upgrade from 2D canvas to WebGL for true 3D isometric with lighting, shadows, depth
- [ ] **Particle system overhaul** — replace simple pixel trails with proper particle emitter (sparks from forge, data streams from tower, energy from portal)
- [ ] **Camera cinematics** — auto-pilot camera that slowly pans around the city, zooms into action during task execution
- [ ] **Fog of war** — unexplored areas of the grid are darkened, programs "reveal" the city as they walk

### Multi-Session & Persistence
- [ ] **Multi-session cities** — each orchestrator session gets its own city instance, switch between them like save slots
- [ ] **City export/import** — save city layout + agent roster as shareable JSON
- [ ] **Leaderboard** — track agent performance across sessions, show all-time top agents
- [ ] **Agent evolution** — agents that perform well automatically get enhanced descriptions/tools over time

### Integration
- [ ] **Webhook notifications** — push city events (task complete, program level up, building upgrade) to external webhook
- [ ] **API for external tools** — let other tools query city state (which agents are where, what tasks are running)
- [ ] **Embeddable widget** — small version of the city that can be embedded in other dashboards
- [ ] **CLI dashboard mode** — ASCII art version of the city for terminal-only environments

---

## Architecture Considerations

- **Performance**: With 10+ draw systems, monitor FPS. Consider offscreen canvas caching for static elements (roads, trees) that don't change per frame.
- **State management**: Currently all state is in global JS variables. Consider a simple event bus for state changes to reduce coupling between draw modules.
- **Code organization**: The `draw_locations.js` file is now ~1100 lines. Consider splitting into `draw_roads.js`, `draw_trees.js`, `draw_location_sprites.js`.
- **Testing**: No automated tests exist for the canvas rendering. Consider screenshot-comparison tests or at least smoke tests that verify the render loop doesn't crash.
- **Accessibility**: The dashboard is entirely visual. Consider adding screen reader descriptions for key state changes (task status, program locations).
