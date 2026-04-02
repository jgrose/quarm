#!/usr/bin/env bun
/**
 * python-autotest.hook.ts — Run pytest after Python file edits
 *
 * TRIGGER: PostToolUse (Edit, Write)
 *
 * When a .py file in the project is edited, runs pytest in fail-fast mode
 * against the test suite. Skips non-Python edits silently.
 * Reports test results as a message but never blocks the tool.
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
const toolInput = input.tool_input || {};

async function main() {
  const filePath: string = toolInput.file_path || '';

  if (!filePath.endsWith('.py')) return;
  if (!filePath.startsWith(PROJECT_DIR)) return;
  if (filePath.includes('__pycache__') || filePath.includes('fixtures/')) return;

  const testsDir = resolve(PROJECT_DIR, 'tests');
  if (!existsSync(testsDir)) return;

  try {
    const result = execFileSync(
      'python3', ['-m', 'pytest', '-x', '--tb=short', '-q', 'tests/'],
      {
        cwd: PROJECT_DIR,
        timeout: 30000,
        encoding: 'utf-8',
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    );
    const lines = result.trim().split('\n');
    const summary = lines[lines.length - 1] || 'no output';
    console.error(`[autotest] ${summary}`);
  } catch (err: any) {
    const output = err.stdout || err.stderr || 'unknown error';
    const lines = output.trim().split('\n').slice(-5);
    console.error(`[autotest] FAILURES:\n${lines.join('\n')}`);
  }
}

main().catch(() => {}).finally(() => {
  console.log(JSON.stringify({ continue: true }));
  process.exit(0);
});
