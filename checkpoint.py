"""
checkpoint.py — Task-level state persistence for crash recovery.
Saves/loads OrchestratorState to JSON after each task completes.

Checkpoint is saved at the top of master_node (the dispatch boundary),
capturing all previously completed tasks and their results. On resume,
any in-flight tasks are reset to pending and re-executed.

Uses atomic write (tmp + rename) for crash safety.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone

PLANS_DIR = Path(__file__).parent / "plans"


def _checkpoint_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}_checkpoint.json"


def save_checkpoint(plan_id: str, state: dict):
    """Persist the serializable portions of OrchestratorState to disk."""
    data = {
        "plan_id": plan_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "objective": state.get("objective", ""),
        "managers": state.get("managers", []),
        "sub_agents": state.get("sub_agents", []),
        "reviewers": state.get("reviewers", []),
        "tasks": state.get("tasks", []),
        "results": state.get("results", {}),
        "tokens_used": state.get("tokens_used", 0),
        "phase": state.get("phase", "dispatch"),
        "active_task_id": state.get("active_task_id"),
        "finished": state.get("finished", False),
        "synthesis_report": state.get("synthesis_report", ""),
    }
    path = _checkpoint_path(plan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)  # atomic on POSIX


_REQUIRED_KEYS = {"tasks", "results"}


def _validate_checkpoint(data: dict) -> list[str]:
    """Check that a loaded checkpoint has all required top-level keys.
    Returns a list of error messages (empty = valid)."""
    errors = []
    for key in _REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")
    return errors


def load_checkpoint(plan_id: str) -> dict | None:
    """Load a checkpoint if one exists. Returns None if no checkpoint or if
    the checkpoint is corrupted/invalid. Corrupted files are renamed to
    .json.corrupted for debugging."""
    path = _checkpoint_path(plan_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Rename corrupted file for debugging
        corrupted = path.with_suffix(".json.corrupted")
        try:
            path.rename(corrupted)
            print(f"[WARN] Corrupted checkpoint renamed: {path} → {corrupted}")
        except OSError:
            pass
        return None

    # Validate structure
    errors = _validate_checkpoint(data)
    if errors:
        print(f"[WARN] Checkpoint {path} failed integrity check:")
        for e in errors:
            print(f"  - {e}")
        # Rename invalid checkpoint for debugging
        corrupted = path.with_suffix(".json.corrupted")
        try:
            path.rename(corrupted)
            print(f"[WARN] Invalid checkpoint renamed: {path} → {corrupted}")
        except OSError:
            pass
        return None

    return data


def has_checkpoint(plan_id: str) -> bool:
    return _checkpoint_path(plan_id).exists()


def clear_checkpoint(plan_id: str):
    path = _checkpoint_path(plan_id)
    if path.exists():
        path.unlink()
