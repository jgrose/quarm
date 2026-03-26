"""
tracking.py — SQLite-based cost & score tracking for QUARM runs.
Stores run history, per-task scores, token usage, and model choices.
"""

import sqlite3
import os
import uuid
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "quarm_runs.db")


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
