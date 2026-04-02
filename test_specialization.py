"""Tests for specialization.py — agent specialization learning and scoring."""

import json
import os
import tempfile
import pytest

# We'll patch the DATA_FILE path before importing the module
_tmpdir = tempfile.mkdtemp()
_test_data_file = os.path.join(_tmpdir, "specialization_data.json")

import specialization as spec


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Reset specialization data before each test."""
    monkeypatch.setattr(spec, "DATA_FILE", _test_data_file)
    if os.path.exists(_test_data_file):
        os.unlink(_test_data_file)
    yield
    if os.path.exists(_test_data_file):
        os.unlink(_test_data_file)


class TestRecordOutcome:
    """Tests for recording agent task outcomes."""

    def test_record_first_outcome(self):
        """First outcome for an agent+tag pair initialises the score."""
        spec.record_outcome("frontend_dev", ["ui", "react"], score=8, revision_count=0)
        data = spec._load_data()
        assert "frontend_dev" in data["agents"]
        agent = data["agents"]["frontend_dev"]
        assert "ui" in agent["tags"]
        assert "react" in agent["tags"]
        # First score with default alpha=0.3: EMA = alpha * score + (1-alpha) * 0
        # But for first entry, it should just be the score
        assert agent["tags"]["ui"]["score"] == 8.0

    def test_record_multiple_outcomes_ema(self):
        """Subsequent scores use exponential moving average."""
        spec.record_outcome("backend_dev", ["api"], score=6, revision_count=0)
        spec.record_outcome("backend_dev", ["api"], score=10, revision_count=0)
        data = spec._load_data()
        tag_data = data["agents"]["backend_dev"]["tags"]["api"]
        # First: 6.0
        # Second: 0.3 * 10 + 0.7 * 6.0 = 3.0 + 4.2 = 7.2
        assert abs(tag_data["score"] - 7.2) < 0.01

    def test_revision_penalty(self):
        """Revision count penalises the effective score."""
        spec.record_outcome("dev_a", ["python"], score=8, revision_count=0)
        first_score = spec._load_data()["agents"]["dev_a"]["tags"]["python"]["score"]
        # Reset
        os.unlink(_test_data_file)
        spec.record_outcome("dev_a", ["python"], score=8, revision_count=3)
        penalised_score = spec._load_data()["agents"]["dev_a"]["tags"]["python"]["score"]
        assert penalised_score < first_score

    def test_multiple_tags_updated(self):
        """All provided tags are updated in one call."""
        spec.record_outcome("fullstack", ["api", "frontend", "testing"], score=9, revision_count=0)
        data = spec._load_data()
        agent = data["agents"]["fullstack"]
        assert set(agent["tags"].keys()) == {"api", "frontend", "testing"}
        for tag_data in agent["tags"].values():
            assert tag_data["score"] == 9.0

    def test_count_increments(self):
        """Each outcome increments the count."""
        spec.record_outcome("dev", ["api"], score=7, revision_count=0)
        spec.record_outcome("dev", ["api"], score=8, revision_count=0)
        spec.record_outcome("dev", ["api"], score=9, revision_count=0)
        data = spec._load_data()
        assert data["agents"]["dev"]["tags"]["api"]["count"] == 3

    def test_custom_alpha(self):
        """Custom alpha value changes EMA calculation."""
        spec.record_outcome("dev", ["api"], score=6, revision_count=0, alpha=0.5)
        spec.record_outcome("dev", ["api"], score=10, revision_count=0, alpha=0.5)
        data = spec._load_data()
        # First: 6.0
        # Second: 0.5 * 10 + 0.5 * 6.0 = 5.0 + 3.0 = 8.0
        assert abs(data["agents"]["dev"]["tags"]["api"]["score"] - 8.0) < 0.01


class TestSuggestSpecialist:
    """Tests for suggesting best agents for given tags."""

    def test_suggest_returns_ranked_list(self):
        """Agents are returned ranked by their scores for the given tags."""
        spec.record_outcome("agent_a", ["api"], score=9, revision_count=0)
        spec.record_outcome("agent_b", ["api"], score=7, revision_count=0)
        spec.record_outcome("agent_c", ["api"], score=10, revision_count=0)
        result = spec.suggest_specialist(["api"])
        assert len(result) == 3
        assert result[0]["agent_name"] == "agent_c"
        assert result[1]["agent_name"] == "agent_a"
        assert result[2]["agent_name"] == "agent_b"

    def test_suggest_multi_tag_aggregation(self):
        """Multi-tag queries average the scores across tags."""
        spec.record_outcome("agent_a", ["api", "security"], score=8, revision_count=0)
        spec.record_outcome("agent_b", ["api"], score=10, revision_count=0)
        spec.record_outcome("agent_b", ["security"], score=4, revision_count=0)
        result = spec.suggest_specialist(["api", "security"])
        # agent_a: avg(8, 8) = 8.0
        # agent_b: avg(10, 4) = 7.0
        assert result[0]["agent_name"] == "agent_a"

    def test_suggest_empty_tags_returns_empty(self):
        """Empty tag list returns no suggestions."""
        result = spec.suggest_specialist([])
        assert result == []

    def test_suggest_no_data_returns_empty(self):
        """When no outcomes recorded, returns empty list."""
        result = spec.suggest_specialist(["api"])
        assert result == []

    def test_suggest_partial_tag_match(self):
        """Agents with only some matching tags still appear, scored on what matches."""
        spec.record_outcome("agent_a", ["api"], score=9, revision_count=0)
        # agent_a has api but not security
        result = spec.suggest_specialist(["api", "security"])
        assert len(result) == 1
        assert result[0]["agent_name"] == "agent_a"
        # Score should be based only on the matching tag
        assert result[0]["avg_score"] == 9.0

    def test_suggest_includes_metadata(self):
        """Each suggestion includes agent_name, avg_score, matching_tags, total_tasks."""
        spec.record_outcome("agent_a", ["api", "python"], score=8, revision_count=0)
        result = spec.suggest_specialist(["api"])
        entry = result[0]
        assert "agent_name" in entry
        assert "avg_score" in entry
        assert "matching_tags" in entry
        assert "total_tasks" in entry
        assert entry["matching_tags"] == ["api"]


class TestDataPersistence:
    """Tests for JSON file persistence."""

    def test_data_persists_across_loads(self):
        """Data survives reload from disk."""
        spec.record_outcome("dev", ["api"], score=9, revision_count=0)
        # Force reload from disk
        data = json.loads(open(_test_data_file).read())
        assert "dev" in data["agents"]

    def test_empty_file_handled_gracefully(self):
        """Corrupted or empty file doesn't crash."""
        with open(_test_data_file, "w") as f:
            f.write("")
        data = spec._load_data()
        assert data == {"agents": {}, "version": 1}

    def test_corrupted_json_handled_gracefully(self):
        """Corrupted JSON resets to empty state."""
        with open(_test_data_file, "w") as f:
            f.write("{invalid json")
        data = spec._load_data()
        assert data == {"agents": {}, "version": 1}


class TestGetSpecializationMatrix:
    """Tests for the full matrix retrieval (used by API endpoint)."""

    def test_matrix_format(self):
        """Matrix returns complete agent-tag scoring data."""
        spec.record_outcome("agent_a", ["api", "python"], score=8, revision_count=0)
        spec.record_outcome("agent_b", ["api"], score=9, revision_count=0)
        matrix = spec.get_specialization_matrix()
        assert "agents" in matrix
        assert "agent_a" in matrix["agents"]
        assert "agent_b" in matrix["agents"]

    def test_matrix_empty_when_no_data(self):
        """Empty matrix when nothing recorded."""
        matrix = spec.get_specialization_matrix()
        assert matrix["agents"] == {}
