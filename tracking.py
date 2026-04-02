"""
tracking.py — SQLite-based cost & score tracking for NORT runs.
Stores run history, per-task scores, token usage, and model choices.
"""

import sqlite3
import os
import uuid
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "nort_runs.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                plan_name TEXT,
                started_at TEXT,
                finished_at TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_revisions INTEGER DEFAULT 0,
                task_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            );
            CREATE TABLE IF NOT EXISTS task_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                task_id TEXT,
                agent TEXT,
                score INTEGER,
                verdict TEXT,
                reviewer TEXT,
                model TEXT,
                tokens INTEGER DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            CREATE TABLE IF NOT EXISTS tolerance_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                task_id TEXT,
                reviewer TEXT,
                original_verdict TEXT,
                score INTEGER,
                tolerance INTEGER,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
        """)


_init_db()


def track_run_start(plan_name: str) -> str:
    """Record a new run starting. Returns the run_id."""
    run_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO runs (id, plan_name, started_at, status) VALUES (?, ?, ?, ?)",
            (run_id, plan_name, datetime.now(timezone.utc).isoformat(), "running"),
        )
    return run_id


def track_score(run_id: str, task_id: str, agent: str, score: int,
                verdict: str, reviewer: str, model: str, tokens: int = 0):
    """Record a review score for a task."""
    with _conn() as c:
        c.execute(
            "INSERT INTO task_scores (run_id, task_id, agent, score, verdict, reviewer, model, tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task_id, agent, score, verdict, reviewer, model, tokens,
             datetime.now(timezone.utc).isoformat()),
        )


def track_tolerance_override(run_id: str, task_id: str, reviewer: str,
                              original_verdict: str, score: int, tolerance: int):
    """Record when a tolerance override changes a verdict."""
    with _conn() as c:
        c.execute(
            "INSERT INTO tolerance_overrides (run_id, task_id, reviewer, original_verdict, score, tolerance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, task_id, reviewer, original_verdict, score, tolerance,
             datetime.now(timezone.utc).isoformat()),
        )


def track_run_end(run_id: str, total_tokens: int, total_revisions: int, task_count: int, status: str = "done"):
    """Record run completion."""
    with _conn() as c:
        c.execute(
            "UPDATE runs SET finished_at=?, total_tokens=?, total_revisions=?, task_count=?, status=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), total_tokens, total_revisions, task_count, status, run_id),
        )


# ── Analytics queries ────────────────────────────────────────────────────────

def get_cost_analytics() -> dict:
    """Aggregate cost data for the dashboard."""
    with _conn() as c:
        total = c.execute("SELECT COALESCE(SUM(total_tokens),0) as t FROM runs").fetchone()["t"]
        recent = c.execute(
            "SELECT id, plan_name, total_tokens, task_count, started_at, status "
            "FROM runs ORDER BY started_at DESC LIMIT 10"
        ).fetchall()
        by_agent = c.execute(
            "SELECT agent, SUM(tokens) as total_tokens, COUNT(*) as count "
            "FROM task_scores GROUP BY agent ORDER BY total_tokens DESC"
        ).fetchall()
        by_model = c.execute(
            "SELECT model, SUM(tokens) as total_tokens, COUNT(*) as count "
            "FROM task_scores WHERE model != '' GROUP BY model ORDER BY total_tokens DESC"
        ).fetchall()
    return {
        "total_tokens": total,
        "recent_runs": [dict(r) for r in recent],
        "by_agent": [dict(r) for r in by_agent],
        "by_model": [dict(r) for r in by_model],
    }


def get_run_cost_breakdown(run_id: str) -> dict:
    """Per-agent and per-model token breakdown for a single run."""
    with _conn() as c:
        run_row = c.execute(
            "SELECT id, plan_name, total_tokens, task_count, started_at, finished_at, status "
            "FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        run_meta = dict(run_row) if run_row else {}

        by_agent = c.execute(
            "SELECT agent, SUM(tokens) as total_tokens, COUNT(*) as count "
            "FROM task_scores WHERE run_id = ? GROUP BY agent ORDER BY total_tokens DESC",
            (run_id,)
        ).fetchall()

        by_model = c.execute(
            "SELECT model, SUM(tokens) as total_tokens, COUNT(*) as count "
            "FROM task_scores WHERE run_id = ? AND model != '' GROUP BY model ORDER BY total_tokens DESC",
            (run_id,)
        ).fetchall()

        by_task = c.execute(
            "SELECT task_id, agent, model, tokens, score, verdict, reviewer "
            "FROM task_scores WHERE run_id = ? ORDER BY task_id",
            (run_id,)
        ).fetchall()

    return {
        "run": run_meta,
        "by_agent": [dict(r) for r in by_agent],
        "by_model": [dict(r) for r in by_model],
        "by_task": [dict(r) for r in by_task],
    }


def get_review_stats() -> dict:
    """Per-reviewer analytics: pass/fail rates, avg scores, override frequency."""
    with _conn() as c:
        by_reviewer = c.execute(
            "SELECT reviewer, "
            "COUNT(*) as total_reviews, "
            "AVG(score) as avg_score, "
            "SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) as passes, "
            "SUM(CASE WHEN verdict IN ('FAIL','FLAG') THEN 1 ELSE 0 END) as failures "
            "FROM task_scores WHERE reviewer != '' GROUP BY reviewer ORDER BY total_reviews DESC"
        ).fetchall()

        overrides = c.execute(
            "SELECT reviewer, COUNT(*) as override_count "
            "FROM tolerance_overrides GROUP BY reviewer ORDER BY override_count DESC"
        ).fetchall()

    override_map = {r["reviewer"]: r["override_count"] for r in overrides}

    return {
        "by_reviewer": [
            {**dict(r), "override_count": override_map.get(r["reviewer"], 0)}
            for r in by_reviewer
        ],
    }


def get_override_stats(run_id: str) -> list[dict]:
    """Get tolerance override counts per reviewer for a specific run."""
    with _conn() as c:
        rows = c.execute(
            "SELECT reviewer, COUNT(*) as count, "
            "AVG(score) as avg_score, AVG(tolerance) as avg_tolerance "
            "FROM tolerance_overrides WHERE run_id = ? GROUP BY reviewer ORDER BY count DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_score_analytics() -> dict:
    """Aggregate score data for the dashboard."""
    with _conn() as c:
        by_agent = c.execute(
            "SELECT agent, AVG(score) as avg_score, COUNT(*) as reviews, "
            "SUM(CASE WHEN verdict IN ('FAIL','FLAG') THEN 1 ELSE 0 END) as failures "
            "FROM task_scores GROUP BY agent ORDER BY avg_score DESC"
        ).fetchall()
        distribution = c.execute(
            "SELECT "
            "SUM(CASE WHEN score >= 8 THEN 1 ELSE 0 END) as high, "
            "SUM(CASE WHEN score >= 5 AND score < 8 THEN 1 ELSE 0 END) as mid, "
            "SUM(CASE WHEN score < 5 THEN 1 ELSE 0 END) as low "
            "FROM task_scores WHERE score > 0"
        ).fetchone()
        recent_scores = c.execute(
            "SELECT task_id, agent, score, verdict, reviewer, model, created_at "
            "FROM task_scores ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
    return {
        "by_agent": [dict(r) for r in by_agent],
        "distribution": dict(distribution) if distribution else {"high": 0, "mid": 0, "low": 0},
        "recent_scores": [dict(r) for r in recent_scores],
    }
