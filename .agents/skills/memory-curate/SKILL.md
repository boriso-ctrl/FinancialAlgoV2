---
name: memory-curate
description: Review shared agent memory proposals and promote only durable, evidence-backed learnings. Use when the user asks to curate, audit, accept, reject, or restructure memory.
---

# Memory Curate

Review `.agent-memory/proposals/` or memory proposals embedded in an agent's final output.

Workflow:

1. Read the proposal and evidence.
2. Verify evidence when feasible.
3. Reject vague, unsupported, secret-bearing, private, customer-specific, stale, contradicted, or task-summary proposals.
4. Choose the smallest correct target.
5. Use role memory for role-specific learned context.
6. Use generated patches for broad policy or adapter changes.
7. Mark stale or superseded entries instead of accumulating contradictions.
8. Report promoted, rejected, and deferred proposals.

Promotion template:

```markdown
- YYYY-MM-DD [type, confidence]: durable lesson. Evidence: `path` or command/source. Review: condition/date.
```
