#!/usr/bin/env python3
"""Generate a brief conventional commit message from git diff."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Iterable


DOC_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".adoc",
}

TEST_HINTS = (
    "test",
    "tests",
    "spec",
)

CONFIG_FILES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "pre-commit-config.yaml",
}


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "git command failed"
        raise RuntimeError(stderr)
    return result.stdout


def get_untracked_files() -> list[str]:
    return [f for f in run_git(["ls-files", "--others", "--exclude-standard"]).splitlines() if f.strip()]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def get_diff(source: str) -> tuple[str, list[str], str]:
    untracked_files = get_untracked_files()

    if source == "staged":
        diff = run_git(["diff", "--staged"])
        files = [f for f in run_git(["diff", "--staged", "--name-only"]).splitlines() if f.strip()]
        return diff, files, "staged"
    if source == "unstaged":
        diff = run_git(["diff"])
        tracked_files = [f for f in run_git(["diff", "--name-only"]).splitlines() if f.strip()]
        files = dedupe_keep_order([*tracked_files, *untracked_files])
        return diff, files, "unstaged"

    staged_diff = run_git(["diff", "--staged"])
    staged_files = [f for f in run_git(["diff", "--staged", "--name-only"]).splitlines() if f.strip()]
    if staged_diff.strip() or staged_files:
        return staged_diff, staged_files, "staged"

    unstaged_diff = run_git(["diff"])
    unstaged_tracked_files = [f for f in run_git(["diff", "--name-only"]).splitlines() if f.strip()]
    unstaged_files = dedupe_keep_order([*unstaged_tracked_files, *untracked_files])
    return unstaged_diff, unstaged_files, "unstaged"


def extension(path: str) -> str:
    base = os.path.basename(path)
    _, ext = os.path.splitext(base.lower())
    return ext


def sanitize_token(token: str) -> str:
    token = token.lower().replace("_", "-")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token


def humanize_name(path: str) -> str:
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if stem.upper() == "SKILL":
        parent = os.path.basename(os.path.dirname(path))
        return parent.replace("-", " ").strip() or "skill"
    if ext in DOC_EXTENSIONS and stem.lower() == "readme":
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            return f"{parent.replace('-', ' ')} docs"
    return re.sub(r"[_-]+", " ", stem).strip() or "project"


def top_dir(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    if parts[0].startswith(".") and len(parts) > 1:
        return parts[1]
    return parts[0]


def infer_scope(files: list[str]) -> str:
    if not files:
        return ""

    if all(path.startswith(".github/skills/") for path in files):
        return "skills"

    if len(files) == 1:
        name = humanize_name(files[0])
        return sanitize_token(name.split()[0])

    dirs = {top_dir(path) for path in files}
    dirs.discard("")
    if len(dirs) == 1:
        only_dir = next(iter(dirs))
        return sanitize_token(only_dir)

    return ""


def extract_changed_lines(diff_text: str) -> list[str]:
    lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines.append(line[1:].strip())
    return lines


def is_doc_change(lines: Iterable[str]) -> bool:
    line_list = [line for line in lines if line]
    if not line_list:
        return False

    doc_patterns = (
        "#",
        '"""',
        "'''",
        "//",
        "/*",
        "*",
    )
    return all(any(line.startswith(prefix) for prefix in doc_patterns) for line in line_list)


def infer_type(files: list[str], diff_text: str) -> str:
    lowered_files = [path.lower() for path in files]
    changed_lines = extract_changed_lines(diff_text)
    text_blob = "\n".join(changed_lines).lower()

    all_docs = bool(files) and all(
        extension(path) in DOC_EXTENSIONS or "/docs/" in path.lower() or os.path.basename(path).lower().startswith("readme")
        for path in files
    )
    if all_docs or is_doc_change(changed_lines):
        return "docs"

    all_tests = bool(files) and all(any(hint in path.lower() for hint in TEST_HINTS) for path in files)
    if all_tests:
        return "test"

    all_config = bool(files) and all(
        os.path.basename(path).lower() in CONFIG_FILES
        or path.lower().startswith(".github/workflows/")
        or path.lower().startswith(".github/skills/")
        for path in files
    )
    if not diff_text.strip():
        if all_docs:
            return "docs"
        if all_tests:
            return "test"
        if all_config:
            return "chore"
        return "feat"

    if all_config:
        return "chore"

    if any(keyword in text_blob for keyword in ("perf", "optimiz", "cache", "latency", "throughput")):
        return "perf"

    if any(keyword in text_blob for keyword in ("fix", "bug", "error", "exception", "correct", "handle")):
        return "fix"

    if any(line.startswith("new file mode") for line in diff_text.lower().splitlines()):
        return "feat"

    if any(path.endswith(".test.py") or path.endswith("_test.py") for path in lowered_files):
        return "test"

    return "refactor"


def infer_target_name(diff_text: str) -> str:
    patterns = (
        r"^[+-]\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^[+-]\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^[+-]\s*function\s+([A-Za-z_][A-Za-z0-9_]*)",
    )
    for line in diff_text.splitlines():
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1)
    return ""


def infer_focus(files: list[str], diff_text: str) -> str:
    target = infer_target_name(diff_text)
    if target:
        return target

    if len(files) == 1:
        return humanize_name(files[0])

    if all(path.startswith(".github/skills/") for path in files):
        return "skills"

    dirs = {top_dir(path) for path in files}
    dirs.discard("")
    if len(dirs) == 1:
        return next(iter(dirs))

    return "project"


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_to_words(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


def build_summary(commit_type: str, focus: str, target: str) -> str:
    focus_text = compact_spaces(focus.replace("-", " "))
    target_text = compact_spaces(target)

    if commit_type == "docs" and target_text:
        return f"document {target_text}"
    if commit_type == "docs":
        return f"update {focus_text} documentation"
    if commit_type == "test":
        return f"update {focus_text} tests"
    if commit_type == "chore":
        return f"update {focus_text} tooling"
    if commit_type == "feat":
        return f"add {focus_text} support"
    if commit_type == "fix":
        return f"fix {focus_text} behavior"
    if commit_type == "perf":
        return f"improve {focus_text} performance"
    return f"refactor {focus_text}"


def generate_message(diff_text: str, files: list[str], max_len: int) -> str:
    commit_type = infer_type(files, diff_text)
    scope = infer_scope(files)
    target = infer_target_name(diff_text)
    focus = infer_focus(files, diff_text)

    summary = build_summary(commit_type, focus, target)
    summary = summary.rstrip(".")

    with_scope_prefix = f"{commit_type}({scope}): " if scope else f"{commit_type}: "
    without_scope_prefix = f"{commit_type}: "

    if scope:
        allowed = max_len - len(with_scope_prefix)
        if allowed >= 15:
            summary = truncate_to_words(summary, allowed)
            message = f"{with_scope_prefix}{summary}"
            if len(message) <= max_len:
                return message

    allowed = max_len - len(without_scope_prefix)
    summary = truncate_to_words(summary, max(10, allowed))
    return f"{without_scope_prefix}{summary}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a concise commit message from git diff.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        choices=("auto", "staged", "unstaged"),
        default="auto",
        help="Choose diff source: staged first (auto), only staged, or only unstaged.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=72,
        help="Maximum message length.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        diff_text, files, resolved_source = get_diff(args.source)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not diff_text.strip() and not files:
        print("No changes found in selected diff source.", file=sys.stderr)
        return 2

    message = generate_message(diff_text, files, max_len=max(30, args.max_len))
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
