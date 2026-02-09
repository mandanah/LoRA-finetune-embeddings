---
name: commit-message-tool
description: Generate a brief, standardized Git commit message from repository diffs. Use when asked to write or generate commit messages, especially prompts like "commit msg", "generate commit message", or "write commit msg".
---

# Commit Message Tool

Generate one clean, one-line commit message using conventional format:
`<type>(<scope>): <summary>`

If scope is unclear, use:
`<type>: <summary>`

## Command
Run:
`python .github/skills/commit-message-tool/scripts/generate_commit_message.py`

The script:
- Reads `git diff --staged` first
- Falls back to `git diff` if staged diff is empty
- Infers type and scope
- Returns one brief message line

## Options
- `--from auto|staged|unstaged` to pick diff source
- `--max-len 72` to control maximum message length

## Type Rules
- `feat`: new capability
- `fix`: bug or regression correction
- `refactor`: structure change without behavior change
- `perf`: performance improvement
- `test`: test updates
- `docs`: documentation or comments/docstrings only
- `chore`: tooling, config, dependency, or maintenance changes

## Output Rules
- Keep message brief, simple, and specific
- Avoid emojis and trailing punctuation
- Return only the final message line unless alternatives are requested
