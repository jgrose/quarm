# Fix: `read_file` treating directories as files

## Context

LLM sub-agents are calling `read_file("artifacts")` and `read_file("plans")` — passing directory paths instead of file paths. This produces `[Errno 21] Is a directory` warnings in the server log. The root cause is twofold:

1. **No directory guard** — `read_file` in `tools.py:231-244` calls `Path.read_text()` without first checking `target.is_file()`, so directories hit the generic `except` and produce an unhelpful error.
2. **Misleading docstring** — The docstring says *"Can read from artifacts/ or plans/ directories"*, which LLM agents interpret as "you can read the directory itself" rather than "you can read files inside those directories."

## Changes

### File: [tools.py](tools.py) (lines 230-244)

1. **Add a directory check** after the existence check (line 236-237). If `target.is_dir()`, list the directory contents and return them so the agent gets useful information instead of an error:

```python
if target.is_dir():
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return f"{path} is a directory. Contents:\n" + "\n".join(entries)
```

2. **Fix the docstring** to prevent agents from passing bare directory names:

```python
"""Read a file by path relative to the project root. If a directory path is given, lists its contents instead."""
```

## Verification

1. Run `python -c "from tools import read_file; print(read_file.invoke({'path': 'artifacts'}))"` — should list directory contents instead of erroring.
2. Run `python -c "from tools import read_file; print(read_file.invoke({'path': 'plans'}))"` — same.
3. Run `python -c "from tools import read_file; print(read_file.invoke({'path': 'orchestrator.py'}))"` — should still read the file normally.
4. Start an orchestration run and confirm the `read_file error` warnings no longer appear in the server log.
