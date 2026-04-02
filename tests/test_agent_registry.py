"""Tests for agent_registry save_registry atomic write pattern."""

import json
import os
import sys
import importlib
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force-load the real agent_registry module (conftest may have stubbed it)
if "agent_registry" in sys.modules:
    del sys.modules["agent_registry"]
import agent_registry as ar


@pytest.fixture(autouse=True)
def isolate_registry(tmp_path, monkeypatch):
    """Point REGISTRY_FILE at a temp directory so tests don't touch real data."""
    reg_file = tmp_path / "agents" / "registry.json"
    monkeypatch.setattr(ar, "REGISTRY_FILE", reg_file)
    yield reg_file


class TestAtomicSave:
    """save_registry uses atomic write pattern."""

    def test_creates_file_from_scratch(self, isolate_registry):
        """save_registry creates parent dirs and writes valid JSON."""
        data = {"sub_agents": {}, "managers": {}, "reviewers": {}}
        ar.save_registry(data)
        assert isolate_registry.exists()
        assert json.loads(isolate_registry.read_text()) == data

    def test_backup_created_when_file_over_100_bytes(self, isolate_registry):
        """Existing file > 100 bytes gets a .json.bak backup."""
        # Seed a file larger than 100 bytes
        big_data = {"sub_agents": {"a": {"name": "a", "data": "x" * 200}},
                    "managers": {}, "reviewers": {}}
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text(json.dumps(big_data, indent=2))
        assert isolate_registry.stat().st_size > 100

        # Now save new data
        new_data = {"sub_agents": {}, "managers": {}, "reviewers": {}}
        ar.save_registry(new_data)

        backup = isolate_registry.with_suffix(".json.bak")
        assert backup.exists()
        assert json.loads(backup.read_text()) == big_data
        assert json.loads(isolate_registry.read_text()) == new_data

    def test_no_backup_for_small_file(self, isolate_registry):
        """Files <= 100 bytes don't get a backup."""
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text("{}")  # 2 bytes
        ar.save_registry({"sub_agents": {}, "managers": {}, "reviewers": {}})
        backup = isolate_registry.with_suffix(".json.bak")
        assert not backup.exists()

    def test_no_temp_file_left_on_success(self, isolate_registry):
        """After a successful save, no .tmp files remain."""
        ar.save_registry({"sub_agents": {}, "managers": {}, "reviewers": {}})
        tmp_files = list(isolate_registry.parent.glob("*.tmp"))
        assert tmp_files == []

    def test_temp_file_cleaned_on_failure(self, isolate_registry):
        """If json.dump raises, the temp file is removed."""
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)

        class BadEncoder:
            """Object that causes json.dump to fail."""
            def __repr__(self):
                return "BadEncoder()"

        with pytest.raises((TypeError, ValueError)):
            ar.save_registry({"bad": BadEncoder()})

        tmp_files = list(isolate_registry.parent.glob("*.tmp"))
        assert tmp_files == []

    def test_original_unchanged_on_failure(self, isolate_registry):
        """If save fails, the original file is untouched."""
        original = {"sub_agents": {"ok": True}, "managers": {}, "reviewers": {}}
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text(json.dumps(original))

        class Unserializable:
            pass

        with pytest.raises((TypeError, ValueError)):
            ar.save_registry({"bad": Unserializable()})

        assert json.loads(isolate_registry.read_text()) == original

    def test_atomic_replace_overwrites_content(self, isolate_registry):
        """Successive saves fully replace file content."""
        v1 = {"sub_agents": {"a": {}}, "managers": {}, "reviewers": {}}
        v2 = {"sub_agents": {"b": {}}, "managers": {}, "reviewers": {}}
        ar.save_registry(v1)
        ar.save_registry(v2)
        assert json.loads(isolate_registry.read_text()) == v2
