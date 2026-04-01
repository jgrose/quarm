"""
Tests for the NORT plan parser (parse_plan in orchestrator.py).

Validates that markdown plan files are correctly parsed into
SubAgentSpec, ManagerSpec, ReviewerSpec, and TaskSpec dataclasses.
"""

import sys
from pathlib import Path

# Ensure conftest stubs are loaded first
from tests.conftest import FIXTURES_DIR

import orchestrator


class TestParseAgents:
    """Verify SubAgentSpec extraction from plan markdown."""

    def test_parse_plan_extracts_agents(self, parsed_simple_plan):
        """Agents are parsed with correct name, description, and tools list."""
        _, _, agents, _, _ = parsed_simple_plan

        assert len(agents) == 1
        agent = agents[0]

        assert agent.name == "backend_engineer"
        assert "FastAPI" in agent.description
        assert "REST APIs" in agent.description
        assert agent.tools == ["execute_code", "write_file", "read_file"]

    def test_parse_plan_extracts_multiple_agents(self, parsed_complex_plan):
        """Complex plan parses all three agents."""
        _, _, agents, _, _ = parsed_complex_plan

        agent_names = [a.name for a in agents]
        assert len(agents) == 3
        assert "backend_engineer" in agent_names
        assert "frontend_engineer" in agent_names
        assert "technical_writer" in agent_names


class TestParseManagers:
    """Verify ManagerSpec extraction from plan markdown."""

    def test_parse_plan_extracts_managers(self, parsed_simple_plan):
        """Manager is parsed with name, title, description, expertise_blend, and oversees."""
        _, managers, _, _, _ = parsed_simple_plan

        assert len(managers) == 1
        mgr = managers[0]

        assert mgr.name == "engineering_director"
        assert mgr.title == "Engineering Architecture Director"
        assert "engineering leader" in mgr.description.lower()
        assert "API_design" in mgr.expertise_blend
        assert "Python_architecture" in mgr.expertise_blend
        assert mgr.oversees == ["backend_engineer"]

    def test_parse_plan_extracts_multiple_managers(self, parsed_complex_plan):
        """Complex plan parses both managers with correct oversees lists."""
        _, managers, _, _, _ = parsed_complex_plan

        assert len(managers) == 2
        mgr_names = {m.name for m in managers}
        assert mgr_names == {"engineering_director", "product_director"}

        prod = next(m for m in managers if m.name == "product_director")
        assert "frontend_engineer" in prod.oversees
        assert "technical_writer" in prod.oversees


class TestParseTasks:
    """Verify TaskSpec extraction from plan markdown."""

    def test_parse_plan_extracts_tasks(self, parsed_simple_plan):
        """Tasks are parsed with id, title, agent, depends_on, task_type, and reviewers."""
        _, _, _, tasks, _ = parsed_simple_plan

        assert len(tasks) == 2

        t1 = next(t for t in tasks if t.id == "TASK-001")
        assert t1.title == "Build user authentication API"
        assert t1.agent == "backend_engineer"
        assert t1.depends_on == []
        assert "code" in t1.task_type
        assert "api" in t1.task_type
        assert "auth" in t1.task_type
        assert "security_engineer" in t1.reviewers

        t2 = next(t for t in tasks if t.id == "TASK-002")
        assert t2.title == "Build user profile endpoints"
        assert t2.depends_on == ["TASK-001"]

    def test_parse_plan_extracts_complex_dependencies(self, parsed_complex_plan):
        """Complex plan parses multi-dependency tasks correctly."""
        _, _, _, tasks, _ = parsed_complex_plan

        assert len(tasks) == 4

        t4 = next(t for t in tasks if t.id == "TASK-004")
        assert set(t4.depends_on) == {"TASK-002", "TASK-003"}
        assert t4.agent == "technical_writer"


class TestParseCustomReviewers:
    """Verify custom ReviewerSpec extraction and builtin override."""

    def test_parse_plan_extracts_custom_reviewers(self, parsed_complex_plan):
        """Custom reviewer overrides builtins and has correct fields."""
        _, _, _, _, all_reviewers = parsed_complex_plan

        # Find the custom reviewer
        custom = next(
            (r for r in all_reviewers if r.name == "api_standards_reviewer"),
            None
        )
        assert custom is not None, "Custom reviewer api_standards_reviewer not found"
        assert custom.title == "API Standards Reviewer"
        assert "REST" in custom.description or "RESTful" in custom.description
        assert "REST conventions" in custom.focus_areas
        assert "code" in custom.applies_to
        assert "api" in custom.applies_to

        # Builtins should still be present alongside the custom one
        builtin_names = {r.name for r in all_reviewers}
        assert "security_engineer" in builtin_names
        assert "ux_designer" in builtin_names


class TestParseToleranceField:
    """Verify tolerance field parsing on tasks and reviewers."""

    def test_parse_plan_tolerance_field(self, parsed_complex_plan):
        """Task-level tolerance is parsed from '- tolerance: 7'."""
        _, _, _, tasks, _ = parsed_complex_plan

        t1 = next(t for t in tasks if t.id == "TASK-001")
        assert t1.tolerance == 7

        # Tasks without tolerance should default to 0
        t2 = next(t for t in tasks if t.id == "TASK-002")
        assert t2.tolerance == 0

    def test_parse_plan_reviewer_tolerance(self, parsed_complex_plan):
        """Custom reviewer tolerance is parsed correctly."""
        _, _, _, _, all_reviewers = parsed_complex_plan

        custom = next(r for r in all_reviewers if r.name == "api_standards_reviewer")
        assert custom.tolerance == 5
