# Memory Proposals

Worker agents can place candidate learnings here when a proposal is too detailed for the final response.

Use this template:

```yaml
date: YYYY-MM-DD
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

Do not store secrets, customer-specific data, raw transcripts, or one-off task state.
