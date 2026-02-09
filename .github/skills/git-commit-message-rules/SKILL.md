---
name: git-commit-message-rules
description: Generate brief, standardized Git commit messages from repository diffs. Use when asked to generate or write a commit message, including prompts like "generate commit message" or "write commit msg".
---

# Commit Message Writer

Generate one clean, single-line commit message from the current diff.

## Format
Use:
`<type>(<scope>): <short summary>`

If scope is unclear, omit it:
`<type>: <short summary>`

## Type Mapping
- `feat`: Add or expand user-facing behavior
- `fix`: Correct a bug or regression
- `refactor`: Restructure code without behavior change
- `perf`: Improve performance
- `test`: Add or update tests
- `docs`: Documentation-only changes
- `chore`: Tooling, config, dependency, or maintenance work

## Workflow
1. Inspect staged changes first with `git diff --staged`.
2. If staged diff is empty, inspect unstaged changes with `git diff`.
3. Identify the primary change and its affected area.
4. Select one type based on the dominant impact.
5. Write a short summary in present tense.

## Style Rules
- Keep summary at or under 72 characters.
- Keep phrasing simple, specific, and clean.
- Avoid generic summaries like `update files` or `misc fixes`.
- Do not end summary with punctuation.
- Do not use emojis.

## Output Rule
Return only the final commit message unless the user asks for alternatives.
