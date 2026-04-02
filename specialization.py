"""
specialization.py — Agent specialization learning and scoring.

Tracks which agents perform best on which types of tasks using an
exponential moving average (EMA) of scores per agent-tag pair.
Recent performance is weighted more heavily than historical scores.

Persistence: JSON file (specialization_data.json), no DB dependencies.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("nort.specialization")

DATA_FILE = os.path.join(os.path.dirname(__file__), "specialization_data.json")
DEFAULT_ALPHA = 0.3
REVISION_PENALTY_PER = 0.5  # score penalty per revision


def _load_data() -> dict:
    """Load the specialization matrix from disk."""
    try:
        with open(DATA_FILE, "r") as f:
            text = f.read().strip()
            if not text:
                return {"agents": {}, "version": 1}
            return json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"agents": {}, "version": 1}


def _save_data(data: dict):
    """Write the specialization matrix to disk."""
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        log.error("Failed to save specialization data: %s", exc)


def record_outcome(agent_name: str, tags: list[str], score: int,
                   revision_count: int, alpha: float = DEFAULT_ALPHA):
    """Update the specialization matrix after a task completes.

    Args:
        agent_name: The agent that executed the task.
        tags: Tags inferred from the task description.
        score: Final review score (1-10).
        revision_count: Number of revision cycles the task needed.
        alpha: EMA smoothing factor (0-1). Higher = more weight on recent.
    """
    if not tags or not agent_name:
        return

    # Apply revision penalty: each revision reduces effective score
    effective_score = max(1.0, score - (revision_count * REVISION_PENALTY_PER))

    data = _load_data()
    agents = data.setdefault("agents", {})
    agent = agents.setdefault(agent_name, {"tags": {}})
    agent_tags = agent.setdefault("tags", {})

    for tag in tags:
        tag = tag.lower().strip()
        if not tag:
            continue
        existing = agent_tags.get(tag)
        if existing is None:
            # First observation: initialise directly
            agent_tags[tag] = {
                "score": effective_score,
                "count": 1,
            }
        else:
            # EMA update: new = alpha * observation + (1 - alpha) * old
            old_score = existing["score"]
            new_score = alpha * effective_score + (1 - alpha) * old_score
            existing["score"] = round(new_score, 4)
            existing["count"] = existing.get("count", 0) + 1

    _save_data(data)
    log.info("Specialization updated: %s tags=%s score=%d revisions=%d",
             agent_name, tags, score, revision_count)


def suggest_specialist(tags: list[str]) -> list[dict]:
    """Return agents ranked by their specialization scores for the given tags.

    Args:
        tags: Task tags to match against.

    Returns:
        List of dicts sorted by avg_score descending:
        [{"agent_name": str, "avg_score": float, "matching_tags": list, "total_tasks": int}, ...]
    """
    if not tags:
        return []

    data = _load_data()
    agents = data.get("agents", {})
    if not agents:
        return []

    query_tags = {t.lower().strip() for t in tags if t.strip()}
    candidates = []

    for agent_name, agent_data in agents.items():
        agent_tags = agent_data.get("tags", {})
        matching = []
        total_score = 0.0
        total_tasks = 0

        for tag in query_tags:
            tag_entry = agent_tags.get(tag)
            if tag_entry:
                matching.append(tag)
                total_score += tag_entry["score"]
                total_tasks += tag_entry.get("count", 0)

        if matching:
            candidates.append({
                "agent_name": agent_name,
                "avg_score": round(total_score / len(matching), 4),
                "matching_tags": sorted(matching),
                "total_tasks": total_tasks,
            })

    candidates.sort(key=lambda c: c["avg_score"], reverse=True)
    return candidates


def get_specialization_matrix() -> dict:
    """Return the full specialization matrix for API/dashboard consumption."""
    data = _load_data()
    return {
        "agents": data.get("agents", {}),
        "version": data.get("version", 1),
    }
