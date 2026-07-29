# AGENTS.md

## Memory Governance

This repo uses architect-curated, role-specific shared agent memory for Claude, Codex, and Gemini.

Worker agents may propose memory. They must not silently promote durable memory.

Durable memory must satisfy all of these:

- It changes future behavior.
- It has evidence: file path, command, source URL, review finding, or explicit user correction.
- It has a scope: user, project, local, or role.
- It has a type: convention, invariant, workflow, pitfall, source, eval, or risk.
- It has a confidence level: observed once, repeated, verified, or superseded.

Reject memory that is vague, stale, secret, customer-specific, one-off task state, or unsupported by evidence.

## Promotion Targets

- Project-wide rules belong in `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or tool adapter policy files.
- Role-specific learnings belong in `.agent-memory/roles/<role>/MEMORY.md`.
- Candidate learnings belong in `.agent-memory/proposals/`.
- Multi-step procedures belong in skills or slash commands.
- Hard enforcement belongs in hooks or settings, not prose.
- Regression risks belong in evals or review checklists.

## Proposal Format

Agents should use this format when proposing memory:

```yaml
memory_proposal:
  role: explorer | researcher | implementer | reviewer | auditor | architect
  scope: project | user | local
  type: convention | invariant | workflow | pitfall | source | eval | risk
  confidence: observed_once | repeated | verified
  evidence:
    - path, command, source URL, or review finding
  proposed_entry: concise durable memory text
  suggested_target: .agent-memory/roles/<role>/MEMORY.md
  review_after: YYYY-MM-DD or "on next related change"
```

## Self-Improvement Loop

After meaningful tasks:

1. Capture what happened.
2. Propose memory only if it will prevent repeated work or repeated mistakes.
3. Let the architect or memory-curator classify the proposal.
4. Promote the smallest durable artifact that changes future behavior.
5. Verify with a test, lint, source check, or review checklist when applicable.
6. Delete or supersede stale memory.

## Tool Adapters

- Claude reads `CLAUDE.md`, `.claude/agents/`, `.claude/rules/`, and `.claude/skills/`.
- Codex reads this `AGENTS.md` file and repo skills under `.agents/skills/`.
- Gemini reads `GEMINI.md` and project commands under `.gemini/commands/`.
- All three tools should consult `.agent-memory/README.md` and the relevant `.agent-memory/roles/<role>/MEMORY.md` before durable role work.
