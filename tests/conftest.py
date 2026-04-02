"""
Shared fixtures for NORT orchestrator tests.

Mocks external dependencies (LLM, status_bridge, etc.) so unit tests
can exercise parse_plan, master_node, and review nodes in isolation.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Stub heavy-side-effect modules BEFORE importing orchestrator ─────────────
# These modules have import-time side effects (DB init, network calls, etc.)
# We replace them with inert stubs so orchestrator.py can be imported cleanly.

_bridge_stub = ModuleType("status_bridge")
_bridge_stub.write_status = MagicMock()
_bridge_stub.log_event = MagicMock()
_bridge_stub.set_project = MagicMock()
_bridge_stub.set_active_reviewer = MagicMock()
_bridge_stub.register_rosters = MagicMock()
_bridge_stub.record_file_touch = MagicMock()
_bridge_stub.set_session_id = MagicMock()
_bridge_stub.cleanup_session = MagicMock()
sys.modules.setdefault("status_bridge", _bridge_stub)

_model_cfg_stub = ModuleType("model_config")
_model_cfg_stub.load_allowed_models = MagicMock(return_value=None)
sys.modules.setdefault("model_config", _model_cfg_stub)

_tracking_stub = ModuleType("tracking")
_tracking_stub.track_run_start = MagicMock(return_value="test-run")
_tracking_stub.track_score = MagicMock()
_tracking_stub.track_run_end = MagicMock()
_tracking_stub.track_tolerance_override = MagicMock()
sys.modules.setdefault("tracking", _tracking_stub)

_tools_stub = ModuleType("tools")
_tools_stub.get_tools = MagicMock(return_value=[])
_tools_stub.execute_tool_call = MagicMock()
_tools_stub.set_tool_context = MagicMock()
_tools_stub.init_mcp_tools = MagicMock()
sys.modules.setdefault("tools", _tools_stub)

_mcp_client_stub = ModuleType("mcp_client")
_mcp_client_stub.init_mcp_from_config = MagicMock(return_value=None)
_mcp_client_stub.shutdown_mcp = MagicMock()
_mcp_client_stub.get_mcp_manager = MagicMock(return_value=MagicMock())
sys.modules.setdefault("mcp_client", _mcp_client_stub)

_mcp_wrapper_stub = ModuleType("mcp_tool_wrapper")
_mcp_wrapper_stub.register_mcp_tools_in_registry = MagicMock(return_value=0)
sys.modules.setdefault("mcp_tool_wrapper", _mcp_wrapper_stub)

_checkpoint_stub = ModuleType("checkpoint")
_checkpoint_stub.save_checkpoint = MagicMock()
_checkpoint_stub.load_checkpoint = MagicMock(return_value=None)
_checkpoint_stub.clear_checkpoint = MagicMock()
sys.modules.setdefault("checkpoint", _checkpoint_stub)

_agent_registry_stub = ModuleType("agent_registry")
_agent_registry_stub.get_agent = MagicMock(return_value=None)
_agent_registry_stub.create_agent = MagicMock()
_agent_registry_stub.record_agent_performance = MagicMock()
_agent_registry_stub.check_earned_tolerance = MagicMock(return_value=False)
_agent_registry_stub.merge_agent_from_plan = MagicMock(return_value=None)
_agent_registry_stub.export_single_agent = MagicMock(return_value=None)
_agent_registry_stub.export_agent_as_claude_code = MagicMock(return_value=None)
sys.modules.setdefault("agent_registry", _agent_registry_stub)

_rag_stub = ModuleType("rag")
_rag_stub.ingest_text = MagicMock(return_value=0)
sys.modules.setdefault("rag", _rag_stub)

# Now it is safe to import orchestrator
import orchestrator  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def simple_plan_path():
    """Path to the minimal 2-task test plan."""
    return str(FIXTURES_DIR / "simple_plan.md")


@pytest.fixture
def complex_plan_path():
    """Path to the 4-task test plan with custom reviewers and tolerances."""
    return str(FIXTURES_DIR / "complex_plan.md")


@pytest.fixture
def parsed_simple_plan(simple_plan_path):
    """Parse the simple plan and return (objective, managers, agents, tasks, reviewers)."""
    return orchestrator.parse_plan(simple_plan_path)


@pytest.fixture
def parsed_complex_plan(complex_plan_path):
    """Parse the complex plan and return (objective, managers, agents, tasks, reviewers)."""
    return orchestrator.parse_plan(complex_plan_path)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test output files."""
    return tmp_path


@pytest.fixture
def mock_llm_pass():
    """Mock ChatOpenAI that returns a PASS verdict JSON."""
    verdict = json.dumps({
        "verdict": "PASS",
        "score": 8,
        "issues": [],
        "feedback": "Looks good."
    })
    mock_resp = MagicMock()
    mock_resp.content = verdict
    mock_resp.response_metadata = {"usage": {"total_tokens": 100}}

    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_resp

    with patch.object(orchestrator, "llm", return_value=mock_model) as _:
        yield mock_model


@pytest.fixture
def mock_llm_fail():
    """Mock ChatOpenAI that returns a FAIL verdict JSON."""
    verdict = json.dumps({
        "verdict": "FAIL",
        "score": 4,
        "issues": ["Missing input validation", "No error handling"],
        "feedback": "Add validation for all user inputs."
    })
    mock_resp = MagicMock()
    mock_resp.content = verdict
    mock_resp.response_metadata = {"usage": {"total_tokens": 120}}

    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_resp

    with patch.object(orchestrator, "llm", return_value=mock_model) as _:
        yield mock_model


@pytest.fixture
def mock_llm_fail_high_score():
    """Mock ChatOpenAI that returns FAIL with a score above typical tolerance."""
    verdict = json.dumps({
        "verdict": "FAIL",
        "score": 7,
        "issues": ["Minor style inconsistency"],
        "feedback": "Consider renaming variables."
    })
    mock_resp = MagicMock()
    mock_resp.content = verdict
    mock_resp.response_metadata = {"usage": {"total_tokens": 110}}

    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_resp

    with patch.object(orchestrator, "llm", return_value=mock_model) as _:
        yield mock_model


def make_base_state(tasks, managers=None, reviewers=None):
    """Build a minimal OrchestratorState dict for testing nodes."""
    return {
        "messages": [],
        "objective": "Test objective",
        "managers": managers or [],
        "sub_agents": [{"name": "backend_engineer", "description": "test", "tools": [], "model": ""}],
        "reviewers": reviewers or [],
        "tasks": tasks,
        "active_task_id": None,
        "active_task_ids": [],
        "results": {},
        "finished": False,
        "phase": "dispatch",
        "tokens_used": 0,
        "last_verdict": None,
        "synthesis_report": "",
        "coherence_report": {},
    }
