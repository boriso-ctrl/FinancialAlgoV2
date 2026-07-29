---
name: memory-curate
description: Review agent memory proposals and promote only durable, evidence-backed learnings.
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Memory Curate

Use this workflow when reviewing `.agent-memory/proposals/` or memory proposals embedded in an agent's final output.

## Steps

1. Read the relevant proposal and evidence.
2. Verify evidence when feasible.
3. Decide the correct target: role memory, project rule, skill, hook, eval, or no-op.
4. Re-read the target file before editing.
5. Apply the smallest edit that will change future behavior.
6. Mark stale or superseded entries instead of accumulating contradictions.
7. Report promoted, rejected, and deferred proposals.

## Rejection Rules

Reject proposals that are vague, unsupported, secret-bearing, user-private, customer-specific, stale, contradicted by the repo, or merely a task summary.

## Promotion Template

Use concise entries:

```markdown
- YYYY-MM-DD [type, confidence]: durable lesson. Evidence: `path` or command/source. Review: condition/date.
```
