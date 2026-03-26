"""Shared model config loader."""
import os
import json

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_allowed_models() -> list[str] | None:
    """Load the allowed model list from config.json. Returns None if all allowed."""
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f).get("allowed_models")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
