# Gemini Shared Memory Adapter

@./AGENTS.md
@./.agent-memory/README.md

## Gemini CLI Memory Layer

Use `GEMINI.md` as the Gemini-specific adapter. Keep durable role memory in `.agent-memory/roles/<role>/MEMORY.md`, not in separate Gemini-only memory files.

At the start of role-specific work, consult `.agent-memory/README.md` and the relevant role memory index when available.

At the end of meaningful work, include a `memory_proposal` block only for reusable learnings. Do not propose transcript summaries or one-off task state.

Only the architect or memory curator promotes proposals into durable memory.
