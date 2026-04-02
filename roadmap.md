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
- [x] Dynamic agent registry — persistent agent definitions with performance tracking
- [x] Output assembly — merge per-task artifacts into single deliverable output folder with MANIFEST.md
- [x] Artifacts in results.json — machine-readable file list per task, output_dir path
- [x] write_file sandbox fix — path traversal prevention matching read_file's existing check
- [x] Configurable review tolerance — per-reviewer/manager score threshold (1-10), dashboard UI panel with global/per-agent sliders, plan `- tolerance:` field support
- [x] Agent chat panel — group-chat-style event log with per-agent avatars, tier icons, colored bubbles, message grouping, verdict badges, and smart scroll
- [x] Collapsible drawer panels — chat and agent list panels slide in/out with toggle tabs and keyboard shortcuts
- [x] Multi-session backend — per-session state isolation via threading.local() in status_bridge, session_id wired from plan_id through orchestrator
- [x] Agent list panel — left-side drawer showing all active agents grouped by plan/session with live status, task assignments, and session switching
- [x] Session switching — click a plan in agent list to switch canvas, chat, and stats to that session; multiple concurrent plans supported
- [x] Unit & integration test suite — 30 tests covering plan parser, routing, status bridge, tolerance, review integration (pytest)
- [x] Playwright smoke tests — 6 headless browser tests verifying page load, canvas, FPS > 30, WebSocket, keyboard shortcuts, health endpoint
- [x] FPS monitoring & profiling — per-system timing with EMA smoothing, performance overlay panel (press P), canvas + HTML dual rendering
- [x] WebSocket hardening — exponential backoff (1s-30s + jitter), heartbeat ping/pong (30s/10s timeout), console.warn error logging, connection status API
- [x] City state persistence — building upgrades and node positions saved to localStorage (`nort_city_state`) with 24h TTL, restored on page load
- [x] Render optimizations — glow cache cap (500), force simulation settled-skip, building/program viewport culling, bloom conditional skip, offscreen grid canvas
- [x] Cost tracking per agent — COSTS panel with LIVE (per-agent bar chart from WebSocket) and HISTORY (drill-down by run) tabs, `$` keyboard shortcut
- [x] Task dependency visualization — animated dashed bezier lines between dependent programs, "BLOCKED 0/2" badge, `d` toggle, Settings panel row
- [x] Output browser in dashboard — file tree + syntax-highlighted preview, `/api/artifacts/{plan_id}` endpoint
- [x] Download output as zip — streaming ZIP via `/api/artifacts/{plan_id}/download`, button in output browser
- [x] Review analytics dashboard — reviewer pass/fail rates, score distribution, tolerance override frequency
- [x] Per-task tolerance — `- tolerance:` field on tasks, precedence chain (config per-agent > task-level > plan per-agent > config global > DEFAULT)
- [x] Adaptive tolerance — auto-suggest adjustments based on revision history, earned tolerance bonus (+1 for avg_score > 8 over 5+ runs)
- [x] Tolerance presets — one-click Prototype(8)/Production(5)/Audit(3) profiles in tolerance config panel
- [x] Conditional review skipping — skip specialist panel for tasks scoring 9+ at manager review (config toggle `skip_specialist_on_high_score`)
- [x] Agent flow dashboard — `/flow` route with agent-flow holographic style, void background, depth particles, pulsing hex grid, Claude spark logo, free-form force layout
- [x] Output composition agent — dedicated composition_node reviews assembled project for cross-file coherence
- [x] Artifact versioning across revisions — revision snapshots stored at `artifacts/{plan_id}/{TASK-ID}/revisions/rev_{N}/`
- [x] Post-assembly validation — linter/syntax checks on output files (`python -m py_compile`, `node --check`)
- [x] Team presets — save agent groups as teams for one-click assignment
- [x] Agent import/export — share agent definitions as JSON
- [x] Earned tolerance — agents with avg_score > 8 over 5+ tasks auto-earn +1 tolerance bonus
- [x] DAG panel — task dependency graph visualization

---

## Backlog — High Priority

### Testing & Stability
- [ ] **End-to-end QA with live plan** — run full orchestrator plan, verify programs walk between buildings correctly, roster shows real names, arrows fire, building upgrades increment

### Orchestrator Core
- [ ] **Plan preview mode** — when typing a plan description, ghost-highlight which buildings will activate and which agents will spawn
- [ ] **Parallel execution indicators** — when multiple tasks run simultaneously, show visual "lightning bolt" connections between active programs
- [ ] **Review chain animation** — when task moves through review stages, show the program physically carrying a "document" sprite between buildings
- [ ] **Failure recovery visualization** — when a task fails and retries, show program walking to Derezzed Zone then back to Code Forge with "revision" badge
- [ ] **Agent specialization learning** — after N runs, auto-enhance agent descriptions based on what tasks they scored highest on
- [ ] **Plan templates** — save successful plans as reusable templates, suggest templates when plan description matches
- [ ] **Sub-task decomposition** — allow agents to break large tasks into sub-tasks at runtime, spawning child programs
- [ ] **Inter-agent communication** — agents can request information from other agents mid-task via a message-passing building (Comms Tower)

### Review & Quality Gate Enhancements
- [ ] **Review override visualization** — surface tolerance overrides in the city (visual indicator when a review is auto-passed vs genuinely passed)
- [ ] **Reviewer calibration mode** — run a "dry review" on past task outputs to test how tolerance changes would affect outcomes before applying to live runs

### Output & Deliverables
- [ ] **Auto-generate project scaffolding** — emit a Makefile, setup script, or docker-compose that runs the assembled output based on detected file types

### Security & Sandboxing
- [ ] **Agent tool audit** — systematic review of all tool functions for injection, traversal, and privilege escalation risks
- [ ] **execute_code sandboxing** — run agent code in a container/namespace instead of a bare subprocess with 30s timeout
- [ ] **Artifact content scanning** — scan agent-written files for secrets, credentials, or known malicious patterns before including in output
- [ ] **Tool allowlist per agent** — restrict which tools each agent can use based on their role (reviewers shouldn't need write_file)

### Agent Registry Enhancements
- [ ] **Agent versioning** — track description/tool changes over time, rollback to previous versions
- [ ] **Agent cloning** — duplicate an existing agent as a starting point for a variant
- [ ] **Agent retirement** — soft-delete agents that consistently underperform (avg_score < 4 after 5+ runs)
- [ ] **Skill tree** — agents unlock new tools or capabilities after reaching certain XP/performance thresholds

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
- [ ] **Keyboard shortcut help overlay** — press "?" to see all available shortcuts (Q/R/M/A/C/L/ESC)
- [ ] **Dark/light theme toggle** — alternate color palettes (Tron Classic blue, Tron Legacy orange, Matrix green)
- [ ] **Run history panel** — list of past orchestrator runs with stats, click to see final report
- [ ] **Agent comparison view** — side-by-side stats for two agents to compare performance
- [ ] **Chat search/filter** — search agent chat by keyword, filter to specific agent or message type (reviews, tools, errors)
- [ ] **Chat export** — download agent chat log as markdown or JSON for post-mortem analysis
- [ ] **Agent chat @mentions** — clickable agent names in chat that highlight the corresponding node on the canvas
- [ ] **Chat thread view** — expand a task's full conversation thread (dispatch → execute → review → revise) as a collapsible sub-chat
- [ ] **Cross-session agent stats** — in the agent list panel, show aggregate stats (total tasks, avg score, total tokens) across all sessions
- [ ] **Session diff view** — compare two completed sessions side-by-side (same plan, different results) to identify what changed
- [ ] **Agent list drag-and-drop** — reorder or reassign agents to tasks by dragging in the agent list panel

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
- [x] **Multi-session cities** — each orchestrator session gets its own city instance, switch between them via agent list panel
- [ ] **Per-session city state preservation** — save/restore node positions, building upgrades, and ambient program state when switching sessions so the city doesn't rebuild from scratch
- [ ] **Session cleanup/archiving** — auto-archive completed sessions after N minutes, clear button per session in agent list, configurable retention
- [ ] **Session replay/DVR** — scrub through a completed session's event log chronologically, re-animating the city as if it were live
- [ ] **City export/import** — save city layout + agent roster as shareable JSON
- [ ] **Leaderboard** — track agent performance across sessions, show all-time top agents
- [ ] **Agent evolution** — agents that perform well automatically get enhanced descriptions/tools over time

### Integration
- [ ] **Webhook notifications** — push city events (task complete, program level up, building upgrade) to external webhook
- [ ] **API for external tools** — let other tools query city state (which agents are where, what tasks are running)
- [ ] **Embeddable widget** — small version of the city that can be embedded in other dashboards
- [ ] **CLI dashboard mode** — ASCII art version of the city for terminal-only environments
- [ ] **Slack/Discord integration** — post city screenshots and task completion summaries to a channel
- [ ] **GitHub integration** — auto-create PRs from orchestrator output, link tasks to issues
- [ ] **RAG knowledge graph visualization** — show the knowledge base as a network graph, highlight which knowledge chunks agents used
- [ ] **Multi-user collaborative view** — multiple users see the same city in real-time, each with their own cursor
- [ ] **LLM provider abstraction** — support Anthropic, OpenAI, Ollama, etc. as interchangeable backends with per-agent provider selection

### Game Mechanics (Gamification)
- [ ] **Achievements** — unlock badges for milestones (first plan completed, 10 tasks done, agent reaches L5, building fully upgraded)
- [ ] **Daily challenges** — auto-generated small plans to keep agents active and earning XP
- [ ] **Agent rivalries** — track which agents outperform others on similar tasks, show friendly competition stats
- [ ] **City reputation score** — aggregate metric based on avg task scores, agent levels, building upgrades
- [ ] **Seasonal events** — special weather/visual themes for holidays or milestones (100th task, etc.)
- [ ] **Prestige system** — reset agent XP for permanent stat bonuses, encouraging long-term engagement

---

## Architecture Considerations

- **Performance**: ~~With 10+ draw systems, monitor FPS.~~ FPS profiler added (press P). ~~Consider offscreen canvas caching for static elements.~~ Offscreen grid canvas implemented. Glow cache capped at 500. Force simulation auto-pauses when settled. Viewport culling active for buildings and programs.
- **State management**: Currently all state is in global JS variables. Consider a simple event bus for state changes to reduce coupling between draw modules.
- **Code organization**: The `draw_locations.js` file is now ~1300 lines. Consider splitting into `draw_roads.js`, `draw_trees.js`, `draw_location_sprites.js`.
- **Testing**: ~~No automated tests exist.~~ 78 pytest tests (unit, integration, tolerance, validation, error recovery) + 6 Playwright smoke tests. Consider screenshot-comparison tests for visual regression.
- **Accessibility**: The dashboard is entirely visual. Add ARIA live regions for event log, screen reader state descriptions.
- **Bundle size**: ~8000+ lines inlined via Jinja2 includes. Consider esbuild/rollup build step for minification.
- **TypeScript**: Original `agent-flow/` has TS sources. Migrate growing JS to TS for type safety on complex state machines.
- **Database**: File-based JSON (`registry.json`, `queue.json`). SQLite used for tracking (runs, scores, tolerance overrides). Migrate remaining JSON to SQLite if multi-user/concurrent access needed.
- **WebSocket versioning**: Add `version` field to WS messages for backward compatibility. ~~No reconnect logic.~~ Exponential backoff + heartbeat ping/pong implemented.
- **Plugin architecture**: Draw modules should register with render loop instead of hardcoded in render.js.
- **Memory**: ~~Sprite caches grow unbounded.~~ Glow cache capped at 500 with batch eviction. Text measurement cache capped at 2000. Consider LRU for other caches.
- **Worker threads**: Move force simulation and sprite rendering to Web Workers for consistent 60fps. ~~Force runs every frame.~~ Force auto-skips when settled (velocity < 0.5 for 60 frames).
- **Security**: Agent descriptions are user-editable and injected into LLM prompts — sanitize to prevent prompt injection.
- **Rate limiting**: No rate limiting on API endpoints. Add if exposed beyond localhost.
- **Multi-session memory**: `_sessions` dict in frontend and per-session dicts in `status_bridge.py` grow unbounded. Add max session count with LRU eviction for completed sessions.
- **Session state serialization**: ~~City state lost on refresh.~~ Node positions and building state persisted to localStorage with 24h TTL. Consider server-side persistence for cross-device.
- **Thread safety audit**: Per-session dicts in `status_bridge.py` use `_state_lock` but concurrent plan runs may still race on deque operations. Consider per-session locks for finer granularity.
- **Structured log migration**: Event log parsing relies on regex against `[AGENT_NAME]` brackets. Migrating to structured log entries from the backend would be more robust and enable richer chat features (timestamps, message types, threading).
