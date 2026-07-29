---
name: memory-maintain
description: Run the deterministic memory maintainer to audit and safely curate repo memory.
allowed-tools: Bash, Read
---

# Memory Maintain

Run this from the repository root:

```bash
python3 memory_maintainer.py --config .agent-memory/memory-maintainer.json --apply --json
```

Safety contract:

- Auto-applies only `.agent-memory/roles/**` and `.agent-memory/proposals/**`.
- Generates proposed patches for `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.claude/**`, `.agents/skills/**`, `.codex/**`, and `.gemini/**`.
- Rejects proposals without evidence.
- Rejects or quarantines secret-like proposals.
- Writes reports, backups, patches, lock files, and state under ignored `.agent-memory/audit/`.

After running, summarize the report path, applied edits, rejected proposals, deferred work, and generated policy patches.
