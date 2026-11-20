#!/usr/bin/env python3
"""Small stdlib-only SAST fallback used when Bandit/Semgrep are unavailable locally."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FINDINGS = []
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

for path in SRC.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        FINDINGS.append((path, exc.lineno or 0, f"syntax error: {exc}"))
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                FINDINGS.append((path, node.lineno, f"dangerous builtin {node.func.id}"))
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"Popen", "run", "call"}:
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        FINDINGS.append((path, node.lineno, "subprocess shell=True"))
            if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "yaml":
                    FINDINGS.append((path, node.lineno, "yaml.load instead of safe_load"))
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            FINDINGS.append((path, line, "possible hard-coded secret"))

if FINDINGS:
    for path, line, message in FINDINGS:
        print(f"{path.relative_to(ROOT)}:{line}: {message}")
    sys.exit(1)
print("SAST fallback: no high-risk patterns found")
