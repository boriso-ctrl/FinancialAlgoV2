from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "memory_maintainer.py"

spec = importlib.util.spec_from_file_location("memory_maintainer", SCRIPT)
memory_maintainer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["memory_maintainer"] = memory_maintainer
spec.loader.exec_module(memory_maintainer)


class MemoryMaintainerTests(unittest.TestCase):
    def make_repo(self) -> Path:
        temp = Path(tempfile.mkdtemp())
        (temp / ".agent-memory" / "roles" / "implementer").mkdir(parents=True)
        (temp / ".agent-memory" / "proposals").mkdir(parents=True)
        (temp / ".claude" / "rules").mkdir(parents=True)
        (temp / ".agents" / "skills" / "memory-maintain").mkdir(parents=True)
        (temp / ".gemini" / "commands" / "memory").mkdir(parents=True)
        (temp / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").write_text(
            "# Implementer Memory\n\n## Durable Learnings\n\n- None yet.\n",
            encoding="utf-8",
        )
        (temp / "AGENTS.md").write_text("# AGENTS.md\n", encoding="utf-8")
        (temp / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
        (temp / "GEMINI.md").write_text("@./AGENTS.md\n", encoding="utf-8")
        (temp / ".agents" / "skills" / "memory-maintain" / "SKILL.md").write_text(
            "# Memory Maintain\n",
            encoding="utf-8",
        )
        (temp / ".gemini" / "commands" / "memory" / "maintain.toml").write_text(
            'prompt = "Run maintainer."\n',
            encoding="utf-8",
        )
        shutil.copy2(
            ROOT / ".agent-memory" / "memory-maintainer.json",
            temp / ".agent-memory" / "memory-maintainer.json",
        )
        return temp

    def run_cli(self, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(repo), "--now", "2026-06-03", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_parse_yamlish_proposal(self) -> None:
        data = memory_maintainer.parse_yamlish(
            """
            memory_proposal:
              role: implementer
              scope: project
              type: workflow
              confidence: verified
              evidence:
                - AGENTS.md
              proposed_entry: Run the maintainer from repo root.
              suggested_target: .agent-memory/roles/implementer/MEMORY.md
              review_after: 2026-07-01
            """
        )
        self.assertEqual(data["role"], "implementer")
        self.assertEqual(data["evidence"], ["AGENTS.md"])

    def test_apply_promotes_role_memory_and_moves_proposal(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "implementer.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: workflow
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Run `python3 memory_maintainer.py --config .agent-memory/memory-maintainer.json --apply` from the repository root.
suggested_target: .agent-memory/roles/implementer/MEMORY.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("Run `python3 memory_maintainer.py --config .agent-memory/memory-maintainer.json --apply`", memory)
        self.assertFalse(proposal.exists())
        self.assertEqual((repo / "AGENTS.md").read_text(encoding="utf-8"), "# AGENTS.md\n")
        self.assertEqual((repo / "GEMINI.md").read_text(encoding="utf-8"), "@./AGENTS.md\n")

    def test_config_cannot_widen_auto_apply_scope(self) -> None:
        repo = self.make_repo()
        config_path = repo / ".agent-memory" / "memory-maintainer.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["auto_apply_globs"] = ["**"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe memory maintainer config", result.stderr)
        self.assertEqual((repo / "AGENTS.md").read_text(encoding="utf-8"), "# AGENTS.md\n")

    def test_auto_target_must_be_matching_role_memory_file(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "wrong-role.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: workflow
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Wrong role target should not be promoted.
suggested_target: .agent-memory/roles/reviewer/MEMORY.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        reviewer = repo / ".agent-memory" / "roles" / "reviewer" / "MEMORY.md"
        self.assertFalse(reviewer.exists())
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("Wrong role target", memory)
        self.assertIn("auto-apply target must match", result.stdout)

    def test_proposal_directory_cannot_be_used_as_promotion_target(self) -> None:
        repo = self.make_repo()
        proposal_readme = repo / ".agent-memory" / "proposals" / "README.md"
        before = proposal_readme.read_text(encoding="utf-8") if proposal_readme.exists() else ""
        proposal = repo / ".agent-memory" / "proposals" / "proposal-dir-target.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: workflow
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Proposal directories are bookkeeping, not memory targets.
suggested_target: .agent-memory/proposals/README.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        after = proposal_readme.read_text(encoding="utf-8") if proposal_readme.exists() else ""
        self.assertEqual(after, before)
        self.assertIn("auto-apply target must match", result.stdout)

    def test_policy_target_generates_patch_without_editing_policy(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "policy.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: convention
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Broad policy changes require review.
suggested_target: AGENTS.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual((repo / "AGENTS.md").read_text(encoding="utf-8"), "# AGENTS.md\n")
        patches = list((repo / ".agent-memory" / "audit").rglob("*.patch"))
        self.assertTrue(patches)
        self.assertIn("Broad policy changes require review.", patches[0].read_text(encoding="utf-8"))

    def test_gemini_adapter_target_generates_patch_without_editing_adapter(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "gemini.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: convention
confidence: verified
evidence:
  - GEMINI.md
proposed_entry: Gemini adapter changes require review.
suggested_target: GEMINI.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        before = (repo / "GEMINI.md").read_text(encoding="utf-8")
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual((repo / "GEMINI.md").read_text(encoding="utf-8"), before)
        patches = list((repo / ".agent-memory" / "audit").rglob("*.patch"))
        self.assertTrue(patches)
        self.assertIn("Gemini adapter changes require review.", patches[0].read_text(encoding="utf-8"))

    def test_secret_like_proposal_is_rejected(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "secret.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: pitfall
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Use token: sk-123456789012345678901234567890
suggested_target: .agent-memory/roles/implementer/MEMORY.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("sk-123", memory)

    def test_unreachable_url_evidence_is_rejected(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "url.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: source
confidence: verified
evidence:
  - http://127.0.0.1:1/not-found
proposed_entry: Do not promote unreachable URL evidence.
suggested_target: .agent-memory/roles/implementer/MEMORY.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("unreachable URL", memory)
        self.assertIn("no verifiable evidence", result.stdout)

    def test_secret_in_review_only_adapter_generates_patch_without_editing(self) -> None:
        repo = self.make_repo()
        path = repo / ".agents" / "skills" / "memory-maintain" / "SKILL.md"
        path.write_text("# Memory Maintain\n\nUse token: sk-123456789012345678901234567890\n", encoding="utf-8")
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("sk-123", path.read_text(encoding="utf-8"))
        patches = list((repo / ".agent-memory" / "audit").rglob("*.patch"))
        self.assertTrue(patches)
        self.assertIn("[REDACTED_SECRET]", patches[0].read_text(encoding="utf-8"))

    def test_stale_marking_is_idempotent(self) -> None:
        repo = self.make_repo()
        path = repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md"
        path.write_text(
            "# Implementer Memory\n\n## Durable Learnings\n\n- 2026-01-01 [workflow, verified]: Old. Evidence: AGENTS.md. Review: 2026-02-01.\n",
            encoding="utf-8",
        )
        first = self.run_cli(repo, "--apply", "--json")
        second = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
        memory = path.read_text(encoding="utf-8")
        self.assertEqual(memory.count("[STALE 2026-06-03]"), 1)

    def test_malformed_review_date_does_not_crash_run(self) -> None:
        repo = self.make_repo()
        path = repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md"
        path.write_text(
            "# Implementer Memory\n\n## Durable Learnings\n\n"
            "- 2026-01-01 [workflow, verified]: Bad date. Evidence: AGENTS.md. Review: 2026-13-45.\n",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = path.read_text(encoding="utf-8")
        self.assertNotIn("[STALE", memory)
        self.assertIn("Review: 2026-13-45.", memory)

    def test_proposal_with_malformed_review_date_is_rejected(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "bad-date.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: workflow
confidence: verified
evidence:
  - AGENTS.md
proposed_entry: Should not be promoted with a bad review date.
suggested_target: .agent-memory/roles/implementer/MEMORY.md
review_after: 2026-13-45
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("Should not be promoted", memory)
        self.assertIn("invalid review_after date", result.stdout)

    def test_private_url_evidence_is_rejected_without_request(self) -> None:
        repo = self.make_repo()
        proposal = repo / ".agent-memory" / "proposals" / "ssrf.yaml"
        proposal.write_text(
            """
role: implementer
scope: project
type: source
confidence: verified
evidence:
  - http://169.254.169.254/latest/meta-data/
proposed_entry: SSRF target must not be treated as verifiable evidence.
suggested_target: .agent-memory/roles/implementer/MEMORY.md
review_after: 2026-07-01
""",
            encoding="utf-8",
        )
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        memory = (repo / ".agent-memory" / "roles" / "implementer" / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("SSRF target", memory)
        self.assertIn("no verifiable evidence", result.stdout)

    def test_old_audit_runs_are_pruned_to_retention(self) -> None:
        repo = self.make_repo()
        config_path = repo / ".agent-memory" / "memory-maintainer.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["audit_runs_retained"] = 3
        config_path.write_text(json.dumps(config), encoding="utf-8")
        runs_dir = repo / ".agent-memory" / "audit" / "runs"
        runs_dir.mkdir(parents=True)
        # Seed five fake older run dirs whose names sort before any real run.
        for i in range(5):
            old = runs_dir / f"20260101-00000{i}-000000"
            old.mkdir()
            (old / "report.md").write_text("old\n", encoding="utf-8")
        # A non-dir sibling must never be pruned.
        (runs_dir / "latest-report.md").write_text("keep\n", encoding="utf-8")
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        remaining = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
        # 5 seeded + 1 from this run = 6; retain 3 newest.
        self.assertEqual(len(remaining), 3)
        # The two newest seeded dirs survive; the three oldest are gone.
        self.assertNotIn("20260101-000000-000000", remaining)
        self.assertIn("20260101-000004-000000", remaining)
        self.assertTrue((runs_dir / "latest-report.md").exists())
        self.assertIn("pruned 3 old audit run(s)", result.stdout)

    def test_dry_run_does_not_prune_audit_runs(self) -> None:
        repo = self.make_repo()
        runs_dir = repo / ".agent-memory" / "audit" / "runs"
        runs_dir.mkdir(parents=True)
        for i in range(40):
            (runs_dir / f"20260101-0000{i:02d}-000000").mkdir()
        result = self.run_cli(repo, "--json")  # no --apply
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        # Dry-run still creates its own run dir but prunes nothing: 40 seeded + 1.
        self.assertEqual(len([p for p in runs_dir.iterdir() if p.is_dir()]), 41)

    def test_nested_noncanonical_memory_file_is_not_edited(self) -> None:
        repo = self.make_repo()
        path = repo / ".agent-memory" / "roles" / "implementer" / "topic" / "MEMORY.md"
        path.parent.mkdir(parents=True)
        original = "# Topic Memory\n\n- Old. Evidence: AGENTS.md. Review: 2026-02-01.\n"
        path.write_text(original, encoding="utf-8")
        result = self.run_cli(repo, "--apply", "--json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
