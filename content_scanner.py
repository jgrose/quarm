"""
content_scanner.py — Scan agent-written artifacts for security issues.
"""
import re, os, stat, logging
from pathlib import Path

log = logging.getLogger("nort.scanner")

SECRET_PATTERNS = [
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
    (r'(?i)(aws_secret_access_key|aws_secret)\s*[=:]\s*\S+', "AWS Secret Key"),
    (r'ghp_[A-Za-z0-9_]{36}', "GitHub Personal Access Token"),
    (r'gho_[A-Za-z0-9_]{36}', "GitHub OAuth Token"),
    (r'sk-[A-Za-z0-9]{20,}', "OpenAI/Stripe Secret Key"),
    (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{4,}["\']', "Hardcoded Password"),
    (r'(?i)(api_key|apikey|api-key)\s*[=:]\s*["\'][^"\']{8,}["\']', "API Key Assignment"),
    (r'(?i)bearer\s+[A-Za-z0-9\-_.~+/]{20,}', "Bearer Token"),
    (r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----', "Private Key"),
]

INJECTION_PATTERNS = [
    (r'(?i)\beval\s*\(', "eval() call"),
    (r'(?i)\bexec\s*\(', "exec() call"),
    (r'(?i)subprocess\.(run|call|Popen|check_output)\s*\(', "subprocess execution"),
    (r'(?i)os\.system\s*\(', "os.system() call"),
    (r'(?i)__import__\s*\(', "dynamic import"),
    (r'(?i)\bcompile\s*\([^)]*["\']exec["\']', "compile() with exec mode"),
]

SCANNABLE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.env', '.sh', '.bash', '.md', '.txt', '.sql', '.xml', '.csv'}

def scan_file(filepath: Path) -> list[dict]:
    """Scan a single file for security issues. Returns list of findings."""
    findings = []

    if filepath.suffix.lower() not in SCANNABLE_EXTENSIONS:
        return findings

    try:
        content = filepath.read_text(errors='replace')
    except Exception:
        return findings

    for pattern, label in SECRET_PATTERNS:
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            findings.append({
                "type": "secret",
                "label": label,
                "file": str(filepath),
                "line": line_num,
                "match_preview": match.group()[:40] + "..." if len(match.group()) > 40 else match.group(),
            })

    code_exts = {'.py', '.js', '.ts', '.jsx', '.tsx', '.sh', '.bash'}
    if filepath.suffix.lower() in code_exts:
        for pattern, label in INJECTION_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    "type": "injection",
                    "label": label,
                    "file": str(filepath),
                    "line": line_num,
                    "match_preview": match.group()[:60],
                })

    try:
        mode = filepath.stat().st_mode
        if mode & stat.S_IWOTH:  # world-writable
            findings.append({
                "type": "permission",
                "label": "World-writable file",
                "file": str(filepath),
                "line": 0,
                "match_preview": oct(mode),
            })
    except Exception:
        pass

    return findings


def scan_directory(directory: Path) -> list[dict]:
    """Scan all files in a directory tree. Returns all findings."""
    all_findings = []
    if not directory.is_dir():
        return all_findings
    for filepath in sorted(directory.rglob("*")):
        if filepath.is_file():
            all_findings.extend(scan_file(filepath))
    return all_findings


def format_scan_report(findings: list[dict]) -> str:
    """Format findings into a human-readable report string."""
    if not findings:
        return ""

    by_type = {}
    for f in findings:
        by_type.setdefault(f["type"], []).append(f)

    lines = [f"CONTENT SCAN: {len(findings)} finding(s)"]
    for ftype, items in by_type.items():
        lines.append(f"\n  [{ftype.upper()}] ({len(items)} finding(s)):")
        for item in items:
            loc = f":{item['line']}" if item['line'] else ""
            lines.append(f"    - {item['label']}: {item['file']}{loc} ({item['match_preview']})")

    return "\n".join(lines)
