#!/usr/bin/env bun
/**
 * plan-validation.hook.ts — Validate plan files on session stop
 *
 * TRIGGER: Stop
 *
 * Checks if any plan markdown files were modified (via git diff) during
 * the session. If so, runs validate_plan.py against them.
 * Only fires when plan files actually changed.
 */

import { readFileSync, existsSync } from 'fs';
import { execFileSync } from 'child_process';
import { resolve, dirname } from 'path';

let input: any;
try {
  input = JSON.parse(readFileSync(0, 'utf-8'));
} catch {
  process.exit(0);
}

const PROJECT_DIR = resolve(dirname(import.meta.dir), '..');

async function main() {
  const validateScript = resolve(PROJECT_DIR, 'validate_plan.py');
  if (!existsSync(validateScript)) return;

  // git diff HEAD covers both staged and unstaged changes
  let changedFiles: string[] = [];
  try {
    const diff = execFileSync('git', ['diff', '--name-only', 'HEAD'], {
      cwd: PROJECT_DIR,
      timeout: 5000,
      encoding: 'utf-8',
    });
    changedFiles = [...new Set(
      diff
        .trim()
        .split('\n')
        .filter(f => f.endsWith('.md') && (f.startsWith('plan') || f.startsWith('plans/')))
    )];
  } catch {
    return;
  }

  if (changedFiles.length === 0) return;

  const planFiles = changedFiles
    .map(f => resolve(PROJECT_DIR, f))
    .filter(f => existsSync(f));

  if (planFiles.length === 0) return;

  try {
    execFileSync('python3', [validateScript, ...planFiles], {
      cwd: PROJECT_DIR,
      timeout: 10000,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    console.error(`[plan-validate] ${planFiles.length} plan file(s) validated OK`);
  } catch (err: any) {
    const output = err.stdout || err.stderr || 'validation failed';
    console.error(`[plan-validate] ISSUES FOUND:\n${output.trim()}`);
  }
}

main().catch(() => {}).finally(() => {
  console.log(JSON.stringify({ continue: true }));
  process.exit(0);
});