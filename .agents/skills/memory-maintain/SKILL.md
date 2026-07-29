---
name: memory-maintain
description: Run the deterministic shared memory maintainer for Claude, Codex, and Gemini repo memory. Use when the user asks to audit, curate, maintain, or safely apply agent memory proposals.
---

# Memory Maintain

Run from the repository root:

```bash
python3 memory_maintainer.py --config .agent-memory/memory-maintainer.json --apply --json
```

Do not manually edit memory files as a fallback. If the command fails, report stdout, stderr, and the failing condition.

Summarize:

- report path
- applied edits
- rejected proposals
- deferred work
- generated policy patches
- warnings or errors

Safety contract:

- Auto-apply scope is only `.agent-memory/roles/**` and `.agent-memory/proposals/**`.
- Broad policy and adapter files receive proposed patches only.
- Proposals need verifiable evidence.
- Secret-like proposals are rejected or quarantined.
- Reports, backups, locks, patches, and state stay under ignored `.agent-memory/audit/`.
