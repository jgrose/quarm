#!/usr/bin/env python3
"""
validate_plan.py — Schema validator for QUARM plan.md files.
Checks: agent references, task ID validity, dependency integrity, reviewer names.

Usage:
    python validate_plan.py plan.example.md
    python validate_plan.py plans/*.md
"""

import re
import sys


BUILTIN_REVIEWERS = {"security_engineer", "ux_designer", "user_tester"}


def validate(path: str) -> list[str]:
    """Validate a plan file and return a list of errors (empty = valid)."""
    try:
        text = open(path).read()
    except FileNotFoundError:
        return [f"File not found: {path}"]

    errors = []

    # Extract agents
    agents = {b.group(1).lower()
              for b in re.finditer(r"### AGENT: (\S+)", text)}

    # Extract managers and their oversees lists
    managers = {}
    for b in re.finditer(r"### MANAGER: (\S+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL):
        name = b.group(1).lower()
        body = b.group(2)
        m = re.search(r"- oversees:\s*\[(.+?)\]", body)
        oversees = [x.strip().lower() for x in m.group(1).split(",")] if m else []
        managers[name] = oversees

    # Extract custom reviewers
    custom_reviewers = {b.group(1).lower()
                        for b in re.finditer(r"### REVIEWER: (\S+)", text)}
    all_reviewers = BUILTIN_REVIEWERS | custom_reviewers

    # Extract tasks
    tasks = {}
    for b in re.finditer(r"### (TASK-\w+)\s+(.*?)(?=\n###|\n##|\Z)", text, re.DOTALL):
        tid = b.group(1)
        body = b.group(2)

        # Agent
        agent_m = re.search(r"- agent:\s*(\S+)", body)
        agent = agent_m.group(1).lower() if agent_m else ""
        if agent and agent not in agents:
            errors.append(f"{tid}: agent '{agent}' not defined in Sub-Agents")

        # Reviewers
        rev_m = re.search(r"- reviewers:\s*\[(.+?)\]", body)
        if rev_m and rev_m.group(1).strip():
            for r in rev_m.group(1).split(","):
                r = r.strip().lower()
                if r and r not in all_reviewers:
                    errors.append(f"{tid}: reviewer '{r}' not recognized")

        # Depends on
        dep_m = re.search(r"- depends_on:\s*\[(.+?)\]", body)
        deps = []
        if dep_m and dep_m.group(1).strip():
            deps = [d.strip() for d in dep_m.group(1).split(",")]

        tasks[tid] = {"agent": agent, "depends_on": deps}

    # Check depends_on references
    for tid, info in tasks.items():
        for dep in info["depends_on"]:
            if dep and dep not in tasks:
                errors.append(f"{tid}: depends_on '{dep}' is not a valid task ID")

    # Check for circular dependencies
    def has_cycle(tid, visited=None, stack=None):
        if visited is None: visited = set()
        if stack is None: stack = set()
        visited.add(tid)
        stack.add(tid)
        for dep in tasks.get(tid, {}).get("depends_on", []):
            if dep in stack:
                return True
            if dep not in visited and has_cycle(dep, visited, stack):
                return True
        stack.discard(tid)
        return False

    for tid in tasks:
        if has_cycle(tid):
            errors.append(f"Circular dependency detected involving {tid}")
            break

    # Check manager oversees references
    for mname, oversees in managers.items():
        for agent in oversees:
            if agent not in agents:
                errors.append(f"Manager '{mname}': oversees '{agent}' not in Sub-Agents")

    if not tasks:
        errors.append("No tasks found in plan")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_plan.py <plan.md> [plan2.md ...]")
        sys.exit(1)

    all_ok = True
    for path in sys.argv[1:]:
        if not path.endswith(".md"):
            continue
        errs = validate(path)
        if errs:
            all_ok = False
            print(f"FAIL: {path}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK: {path}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
