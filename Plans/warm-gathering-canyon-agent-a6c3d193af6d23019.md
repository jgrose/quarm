# Plan: Add Atomic Writes to save_registry() in agent_registry.py

## Summary

Replace the non-atomic `save_registry()` function in `agent_registry.py` with an atomic write pattern (write to temp file, fsync, then `os.replace`). Also add a backup of the existing file before overwriting.

## Current State

- File: `/home/localuser/projects/quarm/.claude/worktrees/agent-a6c3d193/agent_registry.py`
- Current `save_registry()` (lines 25-32) uses `REGISTRY_FILE.write_text(json.dumps(data, indent=2))` which is NOT atomic -- a crash mid-write would corrupt the file.
- Existing imports on line 4: `import json, logging` -- needs `os` and `shutil` added.
- 84 tests currently pass.
- `save_registry()` is called from many places: `seed_registry`, `create_agent`, `update_agent`, `rollback_agent`, `delete_agent`, `clone_agent`, `retire_agent`, `record_agent_performance`, `create_team`, `delete_team`, `import_agents`.

## Steps

### Step 1: Add missing imports

Add `os` and `shutil` to the import line on line 4.

**Current:** `import json, logging`
**New:** `import json, logging, os, shutil`

### Step 2: Replace save_registry() body

Replace lines 25-32 with the atomic write pattern:

```python
def save_registry(data: dict):
    """Write registry to disk atomically (write to temp, then rename)."""
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Backup existing file before overwriting
        if REGISTRY_FILE.exists() and REGISTRY_FILE.stat().st_size > 100:
            backup_path = REGISTRY_FILE.with_suffix('.json.bak')
            shutil.copy2(str(REGISTRY_FILE), str(backup_path))
        # Atomic write: temp file in same dir, then rename
        import tempfile
        fd, tmp_path = tempfile.mkstemp(
            dir=str(REGISTRY_FILE.parent),
            prefix=".registry_",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(REGISTRY_FILE))
        except:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except OSError as exc:
        log.error("Failed to save registry: %s", exc)
        raise
```

Key properties of this pattern:
- **Backup**: copies the existing registry to `.json.bak` before writing (only if file exists and has meaningful content >100 bytes)
- **Temp file**: writes to a temp file in the same directory (same filesystem, so rename is atomic)
- **fsync**: ensures data is flushed to disk before rename
- **os.replace**: atomic rename on POSIX -- either the old file or new file exists, never a half-written state
- **Cleanup**: if writing fails, the temp file is removed; the original file is untouched

### Step 3: Run existing tests

Run `python3 -m pytest tests/ -x --tb=short -q` to verify all 84 tests still pass.

### Step 4: Commit

Commit with message describing the change. No AI attribution per project rules.

## Risk Assessment

- LOW risk. The function signature and behavior are unchanged -- it still takes a dict and writes it to disk. The only difference is HOW it writes (atomically instead of directly).
- The bare `except:` in the inner try block is intentional -- we want to clean up the temp file on ANY failure, then re-raise.
- `tempfile.mkstemp` is imported inline, which matches the task spec. Could also be at top-level, but inline keeps the import localized to where it's used.
