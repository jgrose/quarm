"""
Tests for the NORT agent registry module.

Covers:
  - merge_agent_from_plan: shorter desc no-merge, longer desc updates,
    tags union, tools update, title update, perf fields preserved, nonexistent
  - export_single_agent: valid export has marker, correct agent_type,
    no versions, nonexistent
  - export_agent_as_claude_code: starts with "---", permissions present,
    tool mapping, perf section when runs>0, no perf when runs=0, nonexistent
  - record_agent_performance: PASS/FAIL/force_accepted counters,
    rejection_rate, last_task_at, backward compat with (type,name,score)
"""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force-load the real agent_registry (conftest.py may have stubbed it)
if "agent_registry" in sys.modules:
    del sys.modules["agent_registry"]
import agent_registry
importlib.reload(agent_registry)


class _RegistryTestCase(unittest.TestCase):
    """Base class that isolates each test with a temp registry file."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self._tmp.close()
        self._orig_registry = agent_registry.REGISTRY_FILE
        agent_registry.REGISTRY_FILE = Path(self._tmp.name)
        # Seed a fresh registry in the temp file
        agent_registry.seed_registry()

    def tearDown(self):
        agent_registry.REGISTRY_FILE = self._orig_registry
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass


# ── merge_agent_from_plan ───────────────────────────────────────────────────


class TestMergeAgentFromPlan(_RegistryTestCase):
    """Tests for merge_agent_from_plan()."""

    def test_shorter_desc_no_merge(self):
        """A shorter incoming description should NOT trigger a merge."""
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        existing_desc = agent["description"]
        short_desc = "Short"
        self.assertGreater(len(existing_desc), len(short_desc))

        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": short_desc,
        })
        self.assertIsNone(result)

        # Verify description unchanged
        after = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(after["description"], existing_desc)

    def test_longer_desc_updates(self):
        """A longer incoming description should trigger a merge and update."""
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        existing_desc = agent["description"]
        longer_desc = existing_desc + " with extra expertise in distributed systems and cloud architecture"

        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": longer_desc,
        })
        self.assertIsNotNone(result)
        self.assertEqual(result["description"], longer_desc)

        # Verify persisted
        after = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(after["description"], longer_desc)

    def test_tags_union(self):
        """Incoming tags should be merged as a union with existing tags."""
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        existing_tags = set(agent.get("tags", []))

        new_tags = ["distributed", "cloud"]
        longer_desc = agent["description"] + " — extended description for merge trigger"

        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": longer_desc,
            "tags": list(existing_tags) + new_tags,
        })
        self.assertIsNotNone(result)
        result_tags = set(result["tags"])
        self.assertTrue(existing_tags.issubset(result_tags))
        self.assertIn("distributed", result_tags)
        self.assertIn("cloud", result_tags)

    def test_tools_update(self):
        """Incoming tools list should replace existing tools when desc is longer."""
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        longer_desc = agent["description"] + " — extended for merge"

        new_tools = ["write_file", "read_file", "execute_code", "web_search", "deploy"]
        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": longer_desc,
            "tools": new_tools,
        })
        self.assertIsNotNone(result)
        self.assertEqual(result["tools"], new_tools)

    def test_title_update(self):
        """Incoming title should be updated when different and desc is longer."""
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        longer_desc = agent["description"] + " — extended for merge"
        new_title = "Senior General Developer"

        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": longer_desc,
            "title": new_title,
        })
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], new_title)

    def test_perf_fields_preserved(self):
        """Performance fields (runs, avg_score, etc.) must survive a merge."""
        # Record some performance first
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=8, verdict="PASS"
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(agent["runs"], 1)
        self.assertEqual(agent["tasks_passed"], 1)

        longer_desc = agent["description"] + " — extended for merge to check perf preservation"
        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "general_developer",
            "description": longer_desc,
        })
        self.assertIsNotNone(result)
        self.assertEqual(result["runs"], 1)
        self.assertEqual(result["tasks_passed"], 1)
        self.assertEqual(result["avg_score"], 8)

    def test_nonexistent_returns_none(self):
        """Merge on a nonexistent agent should return None."""
        result = agent_registry.merge_agent_from_plan("sub_agents", {
            "name": "does_not_exist",
            "description": "Very long description that is certainly longer than nothing at all.",
        })
        self.assertIsNone(result)


# ── export_single_agent ─────────────────────────────────────────────────────


class TestExportSingleAgent(_RegistryTestCase):
    """Tests for export_single_agent()."""

    def test_valid_export_has_marker(self):
        """A valid export must contain the nort_agent_export marker."""
        result = agent_registry.export_single_agent("sub_agents", "general_developer")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("nort_agent_export"))

    def test_correct_agent_type(self):
        """Export must include the correct agent_type field."""
        result = agent_registry.export_single_agent("reviewers", "security_engineer")
        self.assertIsNotNone(result)
        self.assertEqual(result["agent_type"], "reviewers")

    def test_no_versions_in_export(self):
        """Exported agent dict must not contain a 'versions' key."""
        # Trigger a version by updating the agent first
        agent_registry.update_agent("sub_agents", "general_developer", {
            "description": "Updated description for version test",
        })
        result = agent_registry.export_single_agent("sub_agents", "general_developer")
        self.assertIsNotNone(result)
        self.assertNotIn("versions", result["agent"])

    def test_nonexistent_returns_none(self):
        """Export of a nonexistent agent should return None."""
        result = agent_registry.export_single_agent("sub_agents", "ghost_agent")
        self.assertIsNone(result)


# ── export_agent_as_claude_code ─────────────────────────────────────────────


class TestExportAgentAsClaudeCode(_RegistryTestCase):
    """Tests for export_agent_as_claude_code()."""

    def test_starts_with_frontmatter(self):
        """Output must start with YAML frontmatter delimiter '---'."""
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "general_developer"
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("---"))

    def test_has_permissions(self):
        """Output must contain a 'permissions:' section for agents with tools."""
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "general_developer"
        )
        self.assertIsNotNone(result)
        self.assertIn("permissions:", result)

    def test_tool_mapping_correct(self):
        """NORT tools must be mapped to Claude Code permission strings."""
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "general_developer"
        )
        self.assertIsNotNone(result)
        # general_developer has: write_file, read_file, execute_code, web_search
        self.assertIn("Edit(*)", result)
        self.assertIn("Write(*)", result)
        self.assertIn("Read(*)", result)
        self.assertIn("Grep(*)", result)
        self.assertIn("Glob(*)", result)
        self.assertIn("Bash(*)", result)
        self.assertIn("WebSearch(*)", result)
        self.assertIn("WebFetch(*)", result)

    def test_perf_section_when_runs_positive(self):
        """Performance History section must appear when runs > 0."""
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=9, verdict="PASS"
        )
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "general_developer"
        )
        self.assertIsNotNone(result)
        self.assertIn("## Performance History", result)
        self.assertIn("Runs: 1", result)

    def test_no_perf_section_when_zero_runs(self):
        """Performance History section must NOT appear when runs == 0."""
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "general_developer"
        )
        self.assertIsNotNone(result)
        self.assertNotIn("## Performance History", result)

    def test_nonexistent_returns_none(self):
        """Export of a nonexistent agent should return None."""
        result = agent_registry.export_agent_as_claude_code(
            "sub_agents", "ghost_agent"
        )
        self.assertIsNone(result)


# ── record_agent_performance ────────────────────────────────────────────────


class TestRecordAgentPerformance(_RegistryTestCase):
    """Tests for record_agent_performance()."""

    def test_pass_increments_tasks_passed(self):
        """verdict='PASS' should increment tasks_passed."""
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=8, verdict="PASS"
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(agent["tasks_passed"], 1)
        self.assertEqual(agent["tasks_failed"], 0)
        self.assertEqual(agent["tasks_force_accepted"], 0)

    def test_fail_increments_tasks_failed(self):
        """verdict='FAIL' should increment tasks_failed."""
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=3, verdict="FAIL"
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(agent["tasks_failed"], 1)
        self.assertEqual(agent["tasks_passed"], 0)

    def test_force_accepted_increments_counter(self):
        """force_accepted=True should increment tasks_force_accepted regardless of verdict."""
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=5,
            verdict="FAIL", force_accepted=True
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(agent["tasks_force_accepted"], 1)
        # force_accepted takes priority over FAIL
        self.assertEqual(agent["tasks_failed"], 0)

    def test_rejection_rate_correct(self):
        """rejection_rate should be (failed + forced) / total_outcomes."""
        # 1 PASS, 1 FAIL, 1 force_accepted => rejection = (1+1)/3 = 0.6667
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=8, verdict="PASS"
        )
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=3, verdict="FAIL"
        )
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=5,
            verdict="FAIL", force_accepted=True
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        expected = round((1 + 1) / 3, 4)
        self.assertAlmostEqual(agent["rejection_rate"], expected, places=4)

    def test_last_task_at_set(self):
        """last_task_at must be set after recording performance."""
        agent_before = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertIsNone(agent_before["last_task_at"])

        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", score=7, verdict="PASS"
        )
        agent_after = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertIsNotNone(agent_after["last_task_at"])
        self.assertIn("T", agent_after["last_task_at"])  # ISO format

    def test_backward_compat_positional_args(self):
        """Calling with just (type, name, score) should default to PASS."""
        agent_registry.record_agent_performance(
            "sub_agents", "general_developer", 7
        )
        agent = agent_registry.get_agent("sub_agents", "general_developer")
        self.assertEqual(agent["runs"], 1)
        self.assertEqual(agent["tasks_passed"], 1)
        self.assertEqual(agent["tasks_failed"], 0)
        self.assertEqual(agent["tasks_force_accepted"], 0)
        self.assertAlmostEqual(agent["avg_score"], 7.0)


import pytest


@pytest.fixture(autouse=False)
def isolate_registry(tmp_path, monkeypatch):
    """Point REGISTRY_FILE at a temp directory so tests don't touch real data."""
    reg_file = tmp_path / "agents" / "registry.json"
    monkeypatch.setattr(agent_registry, "REGISTRY_FILE", reg_file)
    yield reg_file


class TestAtomicSave:
    """save_registry uses atomic write pattern."""

    def test_creates_file_from_scratch(self, isolate_registry):
        data = {"sub_agents": {}, "managers": {}, "reviewers": {}}
        agent_registry.save_registry(data)
        assert isolate_registry.exists()
        assert json.loads(isolate_registry.read_text()) == data

    def test_backup_created_when_file_over_100_bytes(self, isolate_registry):
        big_data = {"sub_agents": {"a": {"name": "a", "data": "x" * 200}},
                    "managers": {}, "reviewers": {}}
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text(json.dumps(big_data, indent=2))
        new_data = {"sub_agents": {}, "managers": {}, "reviewers": {}}
        agent_registry.save_registry(new_data)
        backup = isolate_registry.with_suffix(".json.bak")
        assert backup.exists()
        assert json.loads(backup.read_text()) == big_data

    def test_no_backup_for_small_file(self, isolate_registry):
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text("{}")
        agent_registry.save_registry({"sub_agents": {}, "managers": {}, "reviewers": {}})
        assert not isolate_registry.with_suffix(".json.bak").exists()

    def test_no_temp_file_left_on_success(self, isolate_registry):
        agent_registry.save_registry({"sub_agents": {}, "managers": {}, "reviewers": {}})
        assert list(isolate_registry.parent.glob("*.tmp")) == []

    def test_temp_file_cleaned_on_failure(self, isolate_registry):
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        class BadObj:
            pass
        with pytest.raises((TypeError, ValueError)):
            agent_registry.save_registry({"bad": BadObj()})
        assert list(isolate_registry.parent.glob("*.tmp")) == []

    def test_original_unchanged_on_failure(self, isolate_registry):
        original = {"sub_agents": {"ok": True}, "managers": {}, "reviewers": {}}
        isolate_registry.parent.mkdir(parents=True, exist_ok=True)
        isolate_registry.write_text(json.dumps(original))
        class BadObj:
            pass
        with pytest.raises((TypeError, ValueError)):
            agent_registry.save_registry({"bad": BadObj()})
        assert json.loads(isolate_registry.read_text()) == original

    def test_atomic_replace_overwrites_content(self, isolate_registry):
        v1 = {"sub_agents": {"a": {}}, "managers": {}, "reviewers": {}}
        v2 = {"sub_agents": {"b": {}}, "managers": {}, "reviewers": {}}
        agent_registry.save_registry(v1)
        agent_registry.save_registry(v2)
        assert json.loads(isolate_registry.read_text()) == v2


if __name__ == "__main__":
    unittest.main()
