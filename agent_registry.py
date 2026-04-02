"""NORT Agent Registry — persistent agent definitions with performance tracking."""
from pathlib import Path
from datetime import datetime, timezone
import json, logging, os, shutil, tempfile

log = logging.getLogger("nort.registry")
REGISTRY_FILE = Path(__file__).parent / "agents" / "registry.json"


# ── Core I/O ─────────────────────────────────────────────────────────────────


def load_registry() -> dict:
    """Load registry from disk. Returns {"sub_agents": {}, "managers": {}, "reviewers": {}}."""
    if not REGISTRY_FILE.exists():
        seed_registry()
    try:
        return json.loads(REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to load registry: %s — reseeding", exc)
        seed_registry()
        return json.loads(REGISTRY_FILE.read_text())


def save_registry(data: dict):
    """Write registry to disk atomically.

    1. Backup existing file to .json.bak if > 100 bytes.
    2. Write to a temp file in the same directory.
    3. Flush + fsync for durability.
    4. os.replace for atomic rename.
    5. Clean up temp file on failure.
    """
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file if it's non-trivial
        if REGISTRY_FILE.exists() and REGISTRY_FILE.stat().st_size > 100:
            backup_path = REGISTRY_FILE.with_suffix(".json.bak")
            shutil.copy2(str(REGISTRY_FILE), str(backup_path))

        # Write to temp file in same directory (same filesystem for atomic rename)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(REGISTRY_FILE.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(REGISTRY_FILE))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        log.error("Failed to save registry: %s", exc)
        raise


# ── Seed ─────────────────────────────────────────────────────────────────────


def seed_registry():
    """Create initial registry with builtin reviewers and common agents."""
    now = datetime.now(timezone.utc).isoformat()

    def _base(name: str, *, builtin: bool = False, **kw) -> dict:
        return {
            "name": name,
            "tags": kw.pop("tags", []),
            "created_at": now,
            "updated_at": now,
            "runs": 0,
            "avg_score": 0,
            "total_revisions": 0,
            "tasks_passed": 0,
            "tasks_failed": 0,
            "tasks_force_accepted": 0,
            "rejection_rate": 0.0,
            "last_task_at": None,
            "builtin": builtin,
            **kw,
        }

    reviewers = {
        "security_engineer": _base(
            "security_engineer",
            builtin=True,
            title="Senior Security Engineer",
            description=(
                "Review for OWASP Top 10, broken auth, secrets exposure, "
                "input validation, least privilege. Think like an attacker."
            ),
            focus_areas=[
                "OWASP Top 10",
                "auth & secrets",
                "input validation",
                "least privilege",
                "dependency risk",
            ],
            applies_to=[
                "code", "api", "auth", "data", "config",
                "infrastructure", "backend", "security",
            ],
            tags=["security", "owasp", "auth", "infrastructure"],
        ),
        "ux_designer": _base(
            "ux_designer",
            builtin=True,
            title="Senior UX/UI Designer",
            description=(
                "Review for WCAG 2.1 AA, visual hierarchy, information "
                "architecture, cognitive load, interaction quality."
            ),
            focus_areas=[
                "WCAG accessibility",
                "visual hierarchy",
                "info architecture",
                "interaction patterns",
                "cognitive load",
            ],
            applies_to=[
                "ui", "frontend", "ux", "design", "report",
                "dashboard", "form", "user_flow",
            ],
            tags=["ux", "ui", "accessibility", "design", "frontend"],
        ),
        "user_tester": _base(
            "user_tester",
            builtin=True,
            title="End-User Representative",
            description=(
                "Review as a non-technical first-time user: clarity, plain "
                "language, workflow intuitiveness, value delivered."
            ),
            focus_areas=[
                "first-use clarity",
                "plain language",
                "workflow intuitiveness",
                "value delivered",
            ],
            applies_to=[
                "ui", "report", "documentation", "user_flow",
                "dashboard", "api", "frontend",
            ],
            tags=["usability", "clarity", "user_flow", "documentation"],
        ),
        "creative_director": _base(
            "creative_director",
            builtin=True,
            title="Creative Director",
            description=(
                "Challenge conventional thinking. Is this the obvious boring "
                "solution or something genuinely clever? Push for innovation, "
                "elegance, and delight. Ask: what would make someone say "
                "'that's brilliant'? FLAG safe, generic, or copy-paste "
                "solutions that lack originality or miss creative opportunities."
            ),
            focus_areas=[
                "innovation",
                "elegance",
                "originality",
                "lateral thinking",
                "user delight",
                "bold alternatives",
            ],
            applies_to=[
                "code", "api", "ui", "frontend", "ux", "report",
                "dashboard", "user_flow", "backend", "documentation",
            ],
            tags=["creativity", "innovation", "design", "elegance"],
        ),
        "devils_advocate": _base(
            "devils_advocate",
            builtin=True,
            title="Devil's Advocate",
            description=(
                "Assume everything is wrong. Find the hidden assumptions, "
                "logical flaws, unstated dependencies, and failure modes "
                "nobody mentioned. Ask: what happens when this breaks at 3am? "
                "What did they forget? What looks right but is subtly wrong? "
                "Be ruthless but specific — vague skepticism is useless."
            ),
            focus_areas=[
                "hidden assumptions",
                "logical flaws",
                "edge cases",
                "failure modes",
                "unstated dependencies",
                "silent failures",
            ],
            applies_to=[
                "code", "api", "auth", "data", "config",
                "infrastructure", "backend", "security",
                "ui", "frontend", "user_flow",
            ],
            tags=["critical_thinking", "edge_cases", "failure_modes", "review"],
        ),
        "performance_engineer": _base(
            "performance_engineer",
            builtin=True,
            title="Performance Engineer",
            description=(
                "Review for scalability, efficiency, and production readiness. "
                "Find N+1 queries, unbounded loops, memory leaks, missing "
                "indexes, chatty APIs, blocking calls in async paths, missing "
                "caching, and anything that will fall over at 10x traffic. "
                "Think in terms of p99 latency and cost per request."
            ),
            focus_areas=[
                "scalability",
                "N+1 queries",
                "memory management",
                "caching",
                "concurrency",
                "latency",
                "cost efficiency",
            ],
            applies_to=[
                "code", "api", "data", "backend",
                "infrastructure", "config",
            ],
            tags=["performance", "scalability", "latency", "optimization"],
        ),
    }

    sub_agents = {
        "general_developer": _base(
            "general_developer",
            title="General Developer",
            description="Versatile full-stack developer. Handles any coding task.",
            tools=["write_file", "read_file", "execute_code", "web_search"],
            tags=["code", "general", "fullstack"],
        ),
        "frontend_developer": _base(
            "frontend_developer",
            title="Frontend Developer",
            description=(
                "Expert in HTML, CSS, JavaScript, React, responsive design."
            ),
            tools=["write_file", "read_file", "web_search"],
            tags=["frontend", "ui", "html", "css", "javascript", "react"],
        ),
        "backend_developer": _base(
            "backend_developer",
            title="Backend Developer",
            description=(
                "Expert in Python, Node.js, APIs, databases, server infrastructure."
            ),
            tools=["write_file", "read_file", "execute_code", "web_search"],
            tags=["backend", "python", "api", "database", "node"],
        ),
        "technical_writer": _base(
            "technical_writer",
            title="Technical Writer",
            description=(
                "Creates clear documentation, READMEs, guides, and API docs."
            ),
            tools=["write_file", "read_file", "web_search"],
            tags=["documentation", "writing", "readme", "api_docs"],
        ),
    }

    managers = {
        "tech_lead": _base(
            "tech_lead",
            title="Technical Lead",
            description=(
                "Reviews code quality, architecture decisions, testing "
                "strategy, and production readiness."
            ),
            expertise_blend=[
                "architecture", "code_quality", "testing",
                "performance", "security",
            ],
            tags=["technical", "code", "architecture"],
        ),
        "project_manager": _base(
            "project_manager",
            title="Project Manager",
            description=(
                "Ensures deliverables meet requirements, are well-structured, "
                "and ready for stakeholders."
            ),
            expertise_blend=[
                "requirements", "delivery", "quality", "communication",
            ],
            tags=["management", "delivery", "requirements"],
        ),
    }

    data = {
        "sub_agents": sub_agents,
        "managers": managers,
        "reviewers": reviewers,
    }
    save_registry(data)
    log.info("Seeded agent registry with %d agents",
             len(sub_agents) + len(managers) + len(reviewers))


# ── Query helpers ────────────────────────────────────────────────────────────


def list_agents(agent_type: str = None) -> list[dict]:
    """List agents. If agent_type given, filter to that type."""
    reg = load_registry()
    if agent_type and agent_type in reg:
        return list(reg[agent_type].values())
    result = []
    for atype in reg:
        for agent in reg[atype].values():
            result.append({**agent, "_type": atype})
    return result


def get_agent(agent_type: str, name: str) -> dict | None:
    """Get a single agent by type and name."""
    reg = load_registry()
    return reg.get(agent_type, {}).get(name)


# ── CRUD ─────────────────────────────────────────────────────────────────────


def create_agent(agent_type: str, spec: dict) -> dict:
    """Create a new agent. Adds metadata fields."""
    reg = load_registry()
    if agent_type not in reg:
        reg[agent_type] = {}
    name = spec.get("name", "").lower().replace(" ", "_")
    if not name:
        raise ValueError("Agent must have a name")
    now = datetime.now(timezone.utc).isoformat()
    agent = {
        **spec,
        "name": name,
        "tags": spec.get("tags", []),
        "created_at": spec.get("created_at", now),
        "updated_at": now,
        "runs": spec.get("runs", 0),
        "avg_score": spec.get("avg_score", 0),
        "total_revisions": spec.get("total_revisions", 0),
        "builtin": spec.get("builtin", False),
    }
    reg[agent_type][name] = agent
    save_registry(reg)
    log.info("Created %s/%s", agent_type, name)
    return agent


def merge_agent_from_plan(agent_type: str, spec: dict) -> dict | None:
    """Merge an agent definition from a plan into an existing registry entry.

    Only merges if the incoming description is longer than the existing one
    (heuristic: more detailed spec is better). Merges tags as a union.
    Uses update_agent() which auto-snapshots versions.

    Returns the updated agent, or None if no merge was needed.
    """
    name = spec.get("name", "").lower().replace(" ", "_")
    if not name:
        return None
    agent = get_agent(agent_type, name)
    if not agent:
        return None

    incoming_desc = spec.get("description", "")
    existing_desc = agent.get("description", "")
    if len(incoming_desc) <= len(existing_desc):
        return None  # existing description is already as detailed or more

    updates = {}
    updates["description"] = incoming_desc

    incoming_title = spec.get("title", "")
    if incoming_title and incoming_title != agent.get("title", ""):
        updates["title"] = incoming_title

    incoming_tools = spec.get("tools")
    if incoming_tools is not None and incoming_tools != agent.get("tools"):
        updates["tools"] = incoming_tools

    # Merge tags as union
    existing_tags = set(agent.get("tags", []))
    incoming_tags = set(spec.get("tags", []))
    merged_tags = sorted(existing_tags | incoming_tags)
    if merged_tags != sorted(existing_tags):
        updates["tags"] = merged_tags

    if not updates:
        return None

    return update_agent(agent_type, name, updates)


def update_agent(agent_type: str, name: str, updates: dict) -> dict | None:
    """Update an existing agent. Cannot change name or builtin status."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return None
    updates.pop("name", None)
    updates.pop("builtin", None)

    # Save version before updating
    versions = agent.get("versions", [])
    snapshot = {k: v for k, v in agent.items() if k != "versions"}
    version_num = len(versions) + 1
    snapshot["version"] = version_num
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    versions.append(snapshot)
    agent["versions"] = versions

    agent.update(updates)
    agent["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg[agent_type][name] = agent
    save_registry(reg)
    return agent


def get_agent_versions(agent_type: str, name: str) -> list[dict]:
    """Return version history for an agent."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return []
    return agent.get("versions", [])


def rollback_agent(agent_type: str, name: str, version: int) -> dict | None:
    """Restore an agent to a previous version."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return None
    versions = agent.get("versions", [])
    target = next((v for v in versions if v.get("version") == version), None)
    if not target:
        return None
    # Save current state as a version first
    snapshot = {k: v for k, v in agent.items() if k != "versions"}
    snapshot["version"] = len(versions) + 1
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    versions.append(snapshot)
    # Restore fields from target (except version metadata)
    restore = {k: v for k, v in target.items() if k not in ("version", "timestamp")}
    for key in list(agent.keys()):
        if key not in ("name", "versions", "builtin"):
            agent.pop(key, None)
    agent.update(restore)
    agent["versions"] = versions
    agent["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg[agent_type][name] = agent
    save_registry(reg)
    return agent


def delete_agent(agent_type: str, name: str) -> bool:
    """Delete a non-builtin agent. Raises ValueError for builtins."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return False
    if agent.get("builtin"):
        raise ValueError(f"Cannot delete builtin agent: {name}")
    del reg[agent_type][name]
    save_registry(reg)
    log.info("Deleted %s/%s", agent_type, name)
    return True


def clone_agent(agent_type: str, name: str, new_name: str = None) -> dict | None:
    """Clone an existing agent with a new name."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return None
    new_name = new_name or f"{name}_copy"
    new_name = new_name.lower().replace(" ", "_")
    if new_name in reg.get(agent_type, {}):
        raise ValueError(f"Agent '{new_name}' already exists in {agent_type}")
    now = datetime.now(timezone.utc).isoformat()
    clone = {k: v for k, v in agent.items() if k != "versions"}
    clone["name"] = new_name
    clone["created_at"] = now
    clone["updated_at"] = now
    clone["runs"] = 0
    clone["avg_score"] = 0
    clone["total_revisions"] = 0
    clone["builtin"] = False
    clone["retired"] = False
    if "versions" in clone:
        del clone["versions"]
    reg[agent_type][new_name] = clone
    save_registry(reg)
    log.info("Cloned %s/%s → %s", agent_type, name, new_name)
    return clone


def retire_agent(agent_type: str, name: str, retired: bool = True) -> dict | None:
    """Set or unset retired status on an agent."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return None
    agent["retired"] = retired
    agent["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg[agent_type][name] = agent
    save_registry(reg)
    log.info("%s %s/%s", "Retired" if retired else "Unretired", agent_type, name)
    return agent


# ── Performance tracking ─────────────────────────────────────────────────────


def record_agent_performance(agent_type: str, name: str, score: int,
                             revisions: int = 0, verdict: str = "PASS",
                             force_accepted: bool = False):
    """Update running average score, revision count, and outcome counters after a task run."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return
    runs = agent.get("runs", 0) + 1
    old_avg = agent.get("avg_score", 0)
    new_avg = ((old_avg * (runs - 1)) + score) / runs if runs > 0 else score
    agent["runs"] = runs
    agent["avg_score"] = round(new_avg, 2)
    agent["total_revisions"] = agent.get("total_revisions", 0) + revisions

    # Outcome counter updates
    if force_accepted:
        agent["tasks_force_accepted"] = agent.get("tasks_force_accepted", 0) + 1
    elif verdict == "PASS":
        agent["tasks_passed"] = agent.get("tasks_passed", 0) + 1
    else:
        agent["tasks_failed"] = agent.get("tasks_failed", 0) + 1

    # Compute rejection rate
    passed = agent.get("tasks_passed", 0)
    failed = agent.get("tasks_failed", 0)
    forced = agent.get("tasks_force_accepted", 0)
    total_outcomes = passed + failed + forced
    agent["rejection_rate"] = round((failed + forced) / total_outcomes, 4) if total_outcomes > 0 else 0.0

    agent["last_task_at"] = datetime.now(timezone.utc).isoformat()
    agent["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg[agent_type][name] = agent
    save_registry(reg)


# ── Ranking & suggestions ────────────────────────────────────────────────────


def get_top_agents(agent_type: str, tags: list[str] = None, limit: int = 5) -> list[dict]:
    """Get top-performing agents, optionally filtered by tags."""
    agents = list_agents(agent_type)
    if tags:
        tag_set = set(t.lower() for t in tags)
        agents = [a for a in agents if tag_set & set(t.lower() for t in a.get("tags", []))]
    agents.sort(key=lambda a: (a.get("avg_score", 0), a.get("runs", 0)), reverse=True)
    return agents[:limit]


def suggest_agents_for_description(description: str) -> dict:
    """Simple keyword matching to suggest agents for a project description."""
    desc_lower = description.lower()
    result = {"sub_agents": [], "managers": [], "reviewers": []}
    for agent_type in result:
        for agent in list_agents(agent_type):
            if agent.get("retired"):
                continue
            tags = [t.lower() for t in agent.get("tags", [])]
            if any(tag in desc_lower for tag in tags):
                result[agent_type].append(agent)
            elif any(word in desc_lower for word in agent.get("title", "").lower().split()):
                result[agent_type].append(agent)
    return result


# ── Formatting ───────────────────────────────────────────────────────────────


def format_agent_catalog() -> str:
    """Format the full agent catalog as text for LLM consumption."""
    reg = load_registry()
    lines = []

    # Try to load specialization data for strength display
    spec_matrix = {}
    try:
        from specialization import get_specialization_matrix
        matrix = get_specialization_matrix()
        spec_matrix = matrix.get("agents", {})
    except Exception:
        pass

    for atype, label in [("sub_agents", "Sub-Agents"), ("managers", "Managers"), ("reviewers", "Reviewers")]:
        agents = list(reg.get(atype, {}).values())
        if not agents:
            continue
        lines.append(f"\n### Available {label}:")
        for a in sorted(agents, key=lambda x: (-x.get("avg_score", 0), -x.get("runs", 0))):
            if a.get("retired"):
                continue
            score_info = ""
            if a.get("runs", 0) > 0:
                rej = a.get("rejection_rate", 0.0)
                score_info = (
                    f" [score: {a['avg_score']:.1f}, runs: {a['runs']}, "
                    f"rej: {rej:.0%}]"
                )
            lines.append(
                f"- **{a['name']}**: {a.get('title', a['name'])} "
                f"-- {a.get('description', '')[:120]}{score_info}"
            )
            if a.get("tags"):
                lines.append(f"  tags: {', '.join(a['tags'])}")
            if atype == "sub_agents" and a.get("tools"):
                lines.append(f"  tools: {', '.join(a['tools'])}")
            # Show top 3 specialization strengths if available
            agent_spec = spec_matrix.get(a["name"], {})
            tag_data = agent_spec.get("tags", agent_spec) if isinstance(agent_spec, dict) else {}
            if tag_data:
                # Normalize: values may be floats or dicts with a "score" key
                normalized = {}
                for tag, val in tag_data.items():
                    if isinstance(val, dict):
                        normalized[tag] = val.get("score", 0)
                    elif isinstance(val, (int, float)):
                        normalized[tag] = val
                strengths = sorted(
                    normalized.items(), key=lambda x: x[1], reverse=True
                )[:3]
                if strengths:
                    strength_str = ", ".join(
                        f"{tag}({score:.1f})" for tag, score in strengths
                    )
                    lines.append(f"  strengths: {strength_str}")

    return "\n".join(lines)


# ── Built-in team presets ──────────────────────────────────────────────────

TEAM_PRESETS = {
    "security-focused": {
        "title": "Security-Focused Team",
        "description": "Backend development with strong security review. Ideal for auth systems, APIs handling sensitive data, and infrastructure.",
        "agents": [
            {"type": "sub_agents", "name": "backend_developer"},
            {"type": "reviewers", "name": "security_engineer"},
            {"type": "reviewers", "name": "performance_engineer"},
            {"type": "managers", "name": "tech_lead"},
        ],
    },
    "full-stack": {
        "title": "Full-Stack Team",
        "description": "Frontend and backend developers with DevOps support. Covers the full delivery pipeline from UI to deployment.",
        "agents": [
            {"type": "sub_agents", "name": "frontend_developer"},
            {"type": "sub_agents", "name": "backend_developer"},
            {"type": "sub_agents", "name": "general_developer"},
            {"type": "managers", "name": "tech_lead"},
            {"type": "reviewers", "name": "ux_designer"},
        ],
    },
    "review-heavy": {
        "title": "Review-Heavy Team",
        "description": "One developer backed by multiple specialist reviewers. Maximizes review coverage for high-risk deliverables.",
        "agents": [
            {"type": "sub_agents", "name": "general_developer"},
            {"type": "managers", "name": "tech_lead"},
            {"type": "reviewers", "name": "security_engineer"},
            {"type": "reviewers", "name": "performance_engineer"},
            {"type": "reviewers", "name": "devils_advocate"},
            {"type": "reviewers", "name": "creative_director"},
        ],
    },
    "minimal": {
        "title": "Minimal Team",
        "description": "Lightweight setup with one developer and one reviewer. Good for quick prototypes and small tasks.",
        "agents": [
            {"type": "sub_agents", "name": "general_developer"},
            {"type": "reviewers", "name": "user_tester"},
        ],
    },
}


def list_presets() -> list[dict]:
    """Return available built-in team presets."""
    result = []
    for key, preset in TEAM_PRESETS.items():
        result.append({
            "name": key,
            "title": preset["title"],
            "description": preset["description"],
            "agents": preset["agents"],
        })
    return result


def apply_preset(preset_name: str, team_name: str) -> dict:
    """Create a team from a built-in preset.

    Args:
        preset_name: Key in TEAM_PRESETS.
        team_name: Name for the new team (will be slugified).

    Returns:
        The created team dict.

    Raises:
        ValueError: If preset_name is unknown or team_name is empty.
    """
    if preset_name not in TEAM_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    preset = TEAM_PRESETS[preset_name]
    spec = {
        "name": team_name,
        "title": preset["title"] + " — " + team_name.replace("_", " ").title(),
        "description": preset["description"],
        "agents": list(preset["agents"]),  # copy so callers can't mutate the preset
    }
    return create_team(spec)


# ── Team CRUD ─────────────────────────────────────────────────────────────


def get_teams() -> list[dict]:
    """Return all team presets."""
    reg = load_registry()
    return list(reg.get("teams", {}).values())


def get_team(name: str) -> dict | None:
    """Get a single team preset."""
    reg = load_registry()
    return reg.get("teams", {}).get(name)


def create_team(spec: dict) -> dict:
    """Create or update a team preset."""
    reg = load_registry()
    if "teams" not in reg:
        reg["teams"] = {}
    name = spec.get("name", "").lower().replace(" ", "_")
    if not name:
        raise ValueError("Team must have a name")
    now = datetime.now(timezone.utc).isoformat()
    team = {
        "name": name,
        "title": spec.get("title", name.replace("_", " ").title()),
        "description": spec.get("description", ""),
        "agents": spec.get("agents", []),  # list of {"type": "sub_agents", "name": "agent_name"}
        "created_at": spec.get("created_at", now),
        "updated_at": now,
    }
    reg["teams"][name] = team
    save_registry(reg)
    log.info("Created team: %s", name)
    return team


def delete_team(name: str) -> bool:
    """Delete a team preset."""
    reg = load_registry()
    teams = reg.get("teams", {})
    if name not in teams:
        return False
    del teams[name]
    save_registry(reg)
    log.info("Deleted team: %s", name)
    return True


# ── Import / Export ─────────────────────────────────────────────────────────


def export_agents() -> dict:
    """Export the full registry for backup/transfer."""
    reg = load_registry()
    # Remove versions from export to keep it clean
    export = {}
    for atype in ("sub_agents", "managers", "reviewers"):
        export[atype] = {}
        for name, agent in reg.get(atype, {}).items():
            export[atype][name] = {k: v for k, v in agent.items() if k != "versions"}
    if "teams" in reg:
        export["teams"] = reg["teams"]
    return export


def import_agents(data: dict, overwrite: bool = False) -> dict:
    """Import agents from exported data. Returns summary of what happened."""
    reg = load_registry()
    summary = {"created": [], "skipped": [], "overwritten": []}
    now = datetime.now(timezone.utc).isoformat()
    for atype in ("sub_agents", "managers", "reviewers"):
        if atype not in data:
            continue
        if atype not in reg:
            reg[atype] = {}
        for name, agent in data[atype].items():
            if name in reg[atype]:
                if overwrite:
                    agent["updated_at"] = now
                    reg[atype][name] = agent
                    summary["overwritten"].append(f"{atype}/{name}")
                else:
                    summary["skipped"].append(f"{atype}/{name}")
            else:
                agent["created_at"] = now
                agent["updated_at"] = now
                reg[atype][name] = agent
                summary["created"].append(f"{atype}/{name}")
    if "teams" in data:
        if "teams" not in reg:
            reg["teams"] = {}
        for name, team in data["teams"].items():
            if name in reg["teams"] and not overwrite:
                summary["skipped"].append(f"teams/{name}")
            else:
                action = "overwritten" if name in reg["teams"] else "created"
                reg["teams"][name] = team
                summary[action].append(f"teams/{name}")
    save_registry(reg)
    return summary


def export_single_agent(agent_type: str, name: str) -> dict | None:
    """Return a self-contained export dict for a single agent."""
    agent = get_agent(agent_type, name)
    if not agent:
        return None
    clean = {k: v for k, v in agent.items() if k != "versions"}
    return {
        "nort_agent_export": True,
        "version": 1,
        "agent_type": agent_type,
        "agent": clean,
    }


def export_agent_as_claude_code(agent_type: str, name: str) -> str | None:
    """Produce a markdown file compatible with .claude/agents/*.md.

    Generates YAML frontmatter and a body with title, description,
    expertise/focus areas, performance stats, tags, and origin note.
    """
    agent = get_agent(agent_type, name)
    if not agent:
        return None

    # PascalCase name
    pascal_name = "".join(
        word.capitalize() for word in agent.get("name", name).split("_")
    )

    desc = agent.get("description", "")
    truncated_desc = desc[:120] + ("..." if len(desc) > 120 else "")

    # Map NORT tools to Claude Code permissions
    tool_map = {
        "write_file": ["Edit(*)", "Write(*)"],
        "read_file": ["Read(*)", "Grep(*)", "Glob(*)"],
        "execute_code": ["Bash(*)"],
        "web_search": ["WebSearch(*)", "WebFetch(*)"],
    }
    nort_tools = agent.get("tools", [])
    permissions = []
    for tool in nort_tools:
        permissions.extend(tool_map.get(tool, []))
    # Deduplicate while preserving order
    seen = set()
    unique_permissions = []
    for p in permissions:
        if p not in seen:
            seen.add(p)
            unique_permissions.append(p)

    # Build YAML frontmatter
    lines = ["---"]
    lines.append(f"name: {pascal_name}")
    lines.append(f"description: {truncated_desc}")
    lines.append("model: sonnet")
    if unique_permissions:
        lines.append("permissions:")
        for perm in unique_permissions:
            lines.append(f"  - {perm}")
    lines.append("---")
    lines.append("")

    # Body
    title = agent.get("title", pascal_name)
    lines.append(f"# {title}")
    lines.append("")
    lines.append(desc)
    lines.append("")

    # Expertise / focus areas
    focus = agent.get("focus_areas")
    expertise = agent.get("expertise_blend")
    if focus:
        lines.append("## Focus Areas")
        for area in focus:
            lines.append(f"- {area}")
        lines.append("")
    if expertise:
        lines.append("## Expertise")
        for area in expertise:
            lines.append(f"- {area}")
        lines.append("")

    # Performance history
    runs = agent.get("runs", 0)
    if runs > 0:
        lines.append("## Performance History")
        lines.append(f"- Runs: {runs}")
        lines.append(f"- Average Score: {agent.get('avg_score', 0):.1f}")
        lines.append(f"- Total Revisions: {agent.get('total_revisions', 0)}")
        passed = agent.get("tasks_passed", 0)
        failed = agent.get("tasks_failed", 0)
        forced = agent.get("tasks_force_accepted", 0)
        lines.append(f"- Tasks Passed: {passed}")
        lines.append(f"- Tasks Failed: {failed}")
        lines.append(f"- Tasks Force-Accepted: {forced}")
        rej = agent.get("rejection_rate", 0.0)
        lines.append(f"- Rejection Rate: {rej:.0%}")
        lines.append("")

    # Tags
    tags = agent.get("tags", [])
    if tags:
        lines.append("## Tags")
        lines.append(", ".join(tags))
        lines.append("")

    lines.append("---")
    lines.append(f"*Exported from NORT agent registry ({agent_type}/{name})*")

    return "\n".join(lines)


# ── Earned tolerance ────────────────────────────────────────────────────────


def check_earned_tolerance(agent_type: str, name: str) -> bool:
    """Check if agent qualifies for earned tolerance bonus (avg_score > 8, 5+ runs)."""
    reg = load_registry()
    agent = reg.get(agent_type, {}).get(name)
    if not agent:
        return False
    return (agent.get("runs", 0) >= 5
            and agent.get("avg_score", 0) > 8
            and agent.get("rejection_rate", 0) < 0.3)
