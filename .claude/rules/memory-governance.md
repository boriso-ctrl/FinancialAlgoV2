# Memory Governance Rule

Memory is a control surface, not a transcript.

## Durable Memory Criteria

Promote a memory only when it:

- Changes future behavior.
- Is backed by evidence.
- Has the smallest correct scope.
- Is concise enough to remain useful in context.
- Has a review or expiry condition if it can go stale.

## Correct Target

- Use `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md` for broad working agreements.
- Use `.claude/rules/` for modular or path-scoped rules.
- Use `.agent-memory/roles/<role>/MEMORY.md` for role-specific learned context.
- Use `.agent-memory/proposals/` for unpromoted candidate learnings.
- Use a tool-specific skill or command for a repeatable multi-step workflow.
- Use hooks/settings for hard enforcement.
- Use evals/checklists for regression prevention.

## Worker Agents

Worker agents should propose memory, not promote it.

Every proposal needs evidence and a suggested target. If evidence is weak, mark confidence as `observed_once` or skip the proposal.

## Memory-Curator

The curator must re-read the target file before editing it, keep `MEMORY.md` indexes short, move detail into topic files when needed, and remove or supersede stale entries.
