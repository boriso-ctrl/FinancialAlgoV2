@AGENTS.md
@.agent-memory/README.md

## Claude Code Shared Memory Adapter

Use Claude Code's native primitives:

- `CLAUDE.md` for always-loaded Claude-specific guidance.
- `.claude/rules/` for modular or path-scoped rules.
- `.claude/agents/` for role agents.
- `.claude/skills/` for Claude-specific workflows.
- `.agent-memory/roles/<role>/MEMORY.md` for shared project-scoped role memory.
- `.agent-memory/proposals/` for unpromoted candidate learnings.

Keep this file concise. If instructions grow, move task-specific procedures to skills and path-specific instructions to `.claude/rules/`.

## Role Memory Rule

At the start of role-specific work, consult the relevant shared role memory index when available.

At the end of meaningful work, include a `memory_proposal` block only for reusable learnings. Do not propose transcript summaries or one-off task state.

Only the architect or `memory-curator` promotes proposals into durable memory.
