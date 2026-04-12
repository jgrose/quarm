# Plan: Add Atomic Writes to save_registry() in agent_registry.py

## Summary

The `save_registry()` function in `agent_registry.py` writes directly to `agents/registry.json` via `Path.write_text()`. If the process dies mid-write, the file is left as partial/corrupt JSON -- total data loss for the single source of truth. The fix is a standard atomic write pattern (write to temp in same dir, fsync, then `os.replace`) plus a single-generation `.json.bak` backup before each write.

## Current State

### agent_registry.py (lines 25-32)
```python
def save_registry(data: dict):
    """Write registry to disk."""
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_FILE.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        log.error("Failed to save registry: %s", exc)
        raise
```

- Missing `import os` and `import shutil` at the top
- No atomic write pattern
- No backup mechanism

### checkpoint.py
Already uses atomic write (tmp + rename). No changes needed -- confirmed by reading the file. The docstring even says "Uses atomic write (tmp + rename) for crash safety." It uses `path.with_suffix(".json.tmp")` followed by `tmp.rename(path)`. It's missing fsync, but that's a minor enhancement, not the focus of this task.

### Existing tests
There are no existing tests for `agent_registry.py`. Tests exist for specialization, tolerance, validation wiring, status bridge, error recovery, routing, review integration, smoke, and plan parser.

## Changes

### Step 1: Add imports to agent_registry.py

Add `import os` and `import shutil` to the import block at line 3:
```python
import json, logging, os, shutil
```

### Step 2: Replace save_registry() with atomic write + backup

Replace lines 25-32 with:
```python
def save_registry(data: dict):
    """Write registry to disk atomically (write to temp, then rename)."""
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Backup: keep one .json.bak if current file is valid and non-trivial
        if REGISTRY_FILE.exists() and REGISTRY_FILE.stat().st_size > 100:
            backup_path = REGISTRY_FILE.with_suffix('.json.bak')
            shutil.copy2(str(REGISTRY_FILE), str(backup_path))
        import tempfile
        fd, tmp_path = tempfile.mkstemp(
            dir=REGISTRY_FILE.parent,
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
            os.unlink(tmp_path)
            raise
    except OSError as exc:
        log.error("Failed to save registry: %s", exc)
        raise
```

Key design decisions:
- `tempfile.mkstemp` creates the temp in the same directory (same filesystem guarantees atomic `os.replace`)
- `f.flush()` + `os.fsync()` ensures data hits disk before rename
- `os.replace()` is atomic on POSIX -- overwrites target in one operation
- Bare `except` on the inner try is intentional -- cleanup the temp file on ANY failure (including KeyboardInterrupt)
- Backup only happens if current file exists and is > 100 bytes (avoids backing up empty/trivial files)
- Single `.json.bak` -- no accumulating backups

### Step 3: checkpoint.py -- No changes needed

checkpoint.py already implements atomic writes. It is slightly less robust (no fsync, no backup), but the task says "if it has a similar save_checkpoint() function" -- it does, and it already has the pattern. The improvement would be adding fsync, but that's outside the stated scope of "apply the same atomic write pattern."

### Step 4: Write a test for the atomic write behavior

Create `tests/test_agent_registry.py` with tests that verify:
1. `save_registry()` writes valid JSON that can be read back
2. `save_registry()` creates a `.json.bak` backup when file is > 100 bytes
3. `save_registry()` cleans up temp files on write failure
4. The backup is not created when the file is small (< 100 bytes)

### Step 5: Run existing tests

Run all tests to confirm nothing is broken by the import additions.

### Step 6: Commit

Single commit, no AI attribution per project rules.

## Risk Assessment

- **Low risk**: The change is isolated to one function. All callers of `save_registry()` are unaffected -- same signature, same behavior on success, same exception on failure.
- **Backup edge case**: If `shutil.copy2` fails (permissions, disk full), the OSError will be caught by the outer try/except and logged. The atomic write itself won't proceed, which is the correct behavior -- don't write if you can't back up.
- **Bare except**: The inner bare `except` catches everything including `BaseException`. This is correct for cleanup -- we always want to remove the temp file, then re-raise.
