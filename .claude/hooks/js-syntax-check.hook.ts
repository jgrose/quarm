#!/usr/bin/env bun
/**
 * js-syntax-check.hook.ts — Validate JS syntax after edits to templates/scripts/
 *
 * TRIGGER: PostToolUse (Edit, Write)
 *
 * Runs `node --check` on JS files in templates/scripts/ after edits.
 * Reports syntax errors as warnings but never blocks.
 * Skips non-JS files silently.
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

  if (!filePath.endsWith('.js')) return;
  if (!filePath.includes('templates/scripts/')) return;
  if (!filePath.startsWith(PROJECT_DIR)) return;
  if (!existsSync(filePath)) return;

  try {
    execFileSync('node', ['--check', filePath], {
      cwd: PROJECT_DIR,
      timeout: 5000,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    console.error(`[js-syntax] OK: ${filePath.split('/').pop()}`);
  } catch (err: any) {
    const stderr = err.stderr || '';
    console.error(`[js-syntax] SYNTAX ERROR in ${filePath.split('/').pop()}:\n${stderr.trim()}`);
  }
}

main().catch(() => {}).finally(() => {
  console.log(JSON.stringify({ continue: true }));
  process.exit(0);
});