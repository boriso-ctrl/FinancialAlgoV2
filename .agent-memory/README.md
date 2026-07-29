# Shared Agent Memory

This directory is the canonical repo-local memory lane for Claude, Codex, and Gemini.

## Canonical Paths

- Role memory: `.agent-memory/roles/<role>/MEMORY.md`
- Proposals: `.agent-memory/proposals/`
- Maintainer config: `.agent-memory/memory-maintainer.json`
- Audit output: `.agent-memory/audit/`

## Roles

- `architect`: cross-file judgement, planning, contracts, scope control.
- `explorer`: repo mapping, dependency discovery, unfamiliar-system reconnaissance.
- `researcher`: current-source verification, external docs, standards, APIs.
- `implementer`: bounded edits, verification, contract execution.
- `reviewer`: diff review, regressions, missing tests, behavior risk.
- `auditor`: security, privacy, destructive operations, data handling risk.
- `memory-curator`: proposal triage, promotion, stale marking, policy patch review.

## Rules

- Do not promote memory without evidence: file path, command, source URL, review finding, or explicit user correction.
- Do not store secrets, customer data, raw transcripts, unsupported preferences, or one-off task state.
- Prefer the smallest durable target that changes future behavior.
- Mark stale or superseded entries instead of deleting them outright.
- Broad policy changes require review through generated patches.

## Proposal Target

Use `.agent-memory/roles/<role>/MEMORY.md` for role-specific learnings. Use `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.claude/**`, `.agents/skills/**`, or `.gemini/**` only as review-only suggested targets.
