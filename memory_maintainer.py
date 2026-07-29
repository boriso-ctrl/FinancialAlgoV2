#!/usr/bin/env python3
"""Deterministic maintainer for shared agent role memory.

The maintainer only auto-edits the configured auto-apply paths:
`.agent-memory/roles/**` and `.agent-memory/proposals/**`.
Broad policy files are review-only and receive proposed patch files under
`.agent-memory/audit/`.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import fnmatch
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


DEFAULT_CONFIG: dict[str, Any] = {
    "audit_dir": ".agent-memory/audit",
    "auto_apply_globs": [
        ".agent-memory/roles/**",
        ".agent-memory/proposals/**",
    ],
    "review_only_globs": [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".claude/rules/**",
        ".claude/agents/**",
        ".claude/skills/**",
        ".agents/skills/**",
        ".codex/**",
        ".gemini/**",
    ],
    "proposal_dir": ".agent-memory/proposals",
    "agent_memory_dir": ".agent-memory/roles",
    "memory_index_max_lines": 200,
    "max_diff_lines_per_file": 120,
    "max_total_diff_lines": 500,
    "timezone": "Europe/Amsterdam",
    "allowed_roles": [
        "architect",
        "explorer",
        "researcher",
        "implementer",
        "reviewer",
        "auditor",
        "memory-curator",
    ],
    "allowed_scopes": ["project", "user", "local"],
    "allowed_types": [
        "convention",
        "invariant",
        "workflow",
        "pitfall",
        "source",
        "eval",
        "risk",
    ],
    "allowed_confidence": ["observed_once", "repeated", "verified"],
    "url_evidence_timeout_seconds": 5,
    "audit_runs_retained": 30,
}

CANONICAL_AGENT_MEMORY_DIR = ".agent-memory/roles"
CANONICAL_PROPOSAL_DIR = ".agent-memory/proposals"
CANONICAL_AUDIT_DIR = ".agent-memory/audit"
CANONICAL_AUTO_APPLY_GLOBS = {
    ".agent-memory/roles/**",
    ".agent-memory/proposals/**",
}

REQUIRED_FIELDS = {
    "role",
    "scope",
    "type",
    "confidence",
    "evidence",
    "proposed_entry",
    "suggested_target",
    "review_after",
}

HIGH_CONFIDENCE_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)?PRIVATE KEY-----"),
]

SECRET_LIKE_PATTERN = re.compile(
    r"(?i)\b(password|secret|api[_-]?key|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
)

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
REVIEW_DATE_RE = re.compile(r"\bReview:\s*(\d{4}-\d{2}-\d{2})\b")


@dataclasses.dataclass
class Proposal:
    path: Path
    fields: dict[str, Any]
    raw: str


@dataclasses.dataclass
class Decision:
    status: str
    proposal: Proposal
    reason: str
    target: Path | None = None
    entry: str | None = None
    patch_path: Path | None = None


@dataclasses.dataclass
class RunResult:
    run_dir: Path
    applied: list[str]
    rejected: list[str]
    deferred: list[str]
    policy_patches: list[str]
    warnings: list[str]
    errors: list[str]
    report_path: Path


class MaintenanceError(RuntimeError):
    pass


def posix_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def display_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path:
        candidates = [config_path if config_path.is_absolute() else root / config_path]
    else:
        candidates = [
            root / ".agent-memory" / "memory-maintainer.json",
            root / ".claude" / "memory-maintainer.json",
        ]
    for candidate in candidates:
        if candidate.exists():
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            config.update(loaded)
            break
    return config


def validate_config_safety(root: Path, config: dict[str, Any]) -> None:
    errors = []
    fixed_dirs = {
        "agent_memory_dir": CANONICAL_AGENT_MEMORY_DIR,
        "proposal_dir": CANONICAL_PROPOSAL_DIR,
        "audit_dir": CANONICAL_AUDIT_DIR,
    }
    for key, expected in fixed_dirs.items():
        raw = Path(str(config.get(key, "")))
        if raw.is_absolute():
            errors.append(f"{key} must be repo-relative")
            continue
        configured = (root / raw).resolve()
        expected_path = (root / expected).resolve()
        if configured != expected_path:
            errors.append(f"{key} must be {expected}")

    auto_globs = set(str(item) for item in config.get("auto_apply_globs", []))
    if auto_globs != CANONICAL_AUTO_APPLY_GLOBS:
        errors.append("auto_apply_globs must stay limited to canonical role memory and proposals")

    if errors:
        raise MaintenanceError("unsafe memory maintainer config: " + "; ".join(errors))


def today_for(config: dict[str, Any], now: str | None) -> date:
    if now:
        return date.fromisoformat(now)
    return datetime.now(ZoneInfo(config["timezone"])).date()


def matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def classify_target(path: Path, root: Path, config: dict[str, Any]) -> str:
    if not is_relative_to(path, root):
        return "outside"
    rel = posix_rel(path, root)
    if matches_any(rel, config["auto_apply_globs"]):
        return "auto"
    if matches_any(rel, config["review_only_globs"]):
        return "review"
    return "outside"


def canonical_role_memory_path(root: Path, config: dict[str, Any], role: str) -> Path:
    return (root / config["agent_memory_dir"] / role / "MEMORY.md").resolve()


def is_canonical_role_memory_target(
    path: Path, root: Path, config: dict[str, Any], role: str | None = None
) -> bool:
    if path.is_symlink() or not is_relative_to(path, root):
        return False
    base = (root / config["agent_memory_dir"]).resolve()
    try:
        parts = path.resolve().relative_to(base).parts
    except ValueError:
        return False
    if len(parts) != 2 or parts[1] != "MEMORY.md":
        return False
    if role is not None and parts[0] != role:
        return False
    return True


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains_high_confidence_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS)


def contains_secret_like_text(text: str) -> bool:
    return bool(SECRET_LIKE_PATTERN.search(text))


def redact_high_confidence_secrets(text: str) -> tuple[str, bool]:
    changed = False
    for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
        text, count = pattern.subn("[REDACTED_SECRET]", text)
        changed = changed or bool(count)
    return text, changed


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def extract_yamlish_block(text: str) -> str:
    fenced = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    return text


def parse_yamlish(text: str) -> dict[str, Any]:
    text = extract_yamlish_block(text)
    data: dict[str, Any] = {}
    current_key: str | None = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped == "memory_proposal:":
            continue
        if stripped.startswith("- ") and current_key:
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(strip_quotes(stripped[2:]))
            continue
        match = re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:\s*", stripped)
        if match:
            key, value = stripped.split(":", 1)
            key = key.strip().replace("-", "_")
            value = value.strip()
            data[key] = strip_quotes(value) if value else []
            current_key = key
            continue
        if current_key and isinstance(data.get(current_key), str):
            data[current_key] = f"{data[current_key]} {stripped}".strip()

    return data


def discover_proposal_files(root: Path, config: dict[str, Any]) -> list[Path]:
    proposal_dir = root / config["proposal_dir"]
    if not proposal_dir.exists():
        return []
    files = []
    for path in proposal_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() == "readme.md":
            continue
        files.append(path)
    return sorted(files)


def parse_proposal_file(path: Path) -> Proposal:
    raw = path.read_text(encoding="utf-8")
    fields = parse_yamlish(raw)
    return Proposal(path=path, fields=fields, raw=raw)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def url_target_is_public(url: str) -> bool:
    """Reject SSRF targets before the unattended job issues any request.

    A proposal author should not be able to make the maintainer probe
    loopback/private/link-local/reserved hosts (e.g. cloud metadata).
    """
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def url_is_reachable(url: str, timeout_seconds: int) -> bool:
    if not url_target_is_public(url):
        return False
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            return 200 <= exc.code < 400
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 400
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return False


def evidence_is_verified(item: str, root: Path, config: dict[str, Any]) -> bool:
    text = item.strip()
    if re.match(r"https?://[^\s]+", text):
        return url_is_reachable(text, int(config["url_evidence_timeout_seconds"]))
    if text.lower().startswith(("user correction:", "explicit user correction:", "review finding:")):
        return True
    if text.lower().startswith("command:") and len(text.split(":", 1)[1].strip()) > 0:
        return True

    possible = text
    if ":" in possible and not possible.startswith(("/", "./", "../")):
        prefix, rest = possible.split(":", 1)
        if prefix.lower() in {"path", "file", "source"}:
            possible = rest.strip()

    possible = possible.split(":", 1)[0].strip("` ")
    if not possible:
        return False
    candidate = (root / possible).resolve()
    return is_relative_to(candidate, root) and candidate.exists()


def validate_required_fields(proposal: Proposal, config: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(proposal.fields))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    role = str(proposal.fields.get("role", ""))
    scope = str(proposal.fields.get("scope", ""))
    memory_type = str(proposal.fields.get("type", ""))
    confidence = str(proposal.fields.get("confidence", ""))
    if role and role not in config["allowed_roles"]:
        errors.append(f"invalid role: {role}")
    if scope and scope not in config["allowed_scopes"]:
        errors.append(f"invalid scope: {scope}")
    if memory_type and memory_type not in config["allowed_types"]:
        errors.append(f"invalid type: {memory_type}")
    if confidence and confidence not in config["allowed_confidence"]:
        errors.append(f"invalid confidence: {confidence}")
    review_after = str(proposal.fields.get("review_after", "")).strip()
    date_token = DATE_RE.search(review_after)
    if date_token:
        try:
            date.fromisoformat(date_token.group(1))
        except ValueError:
            errors.append(f"invalid review_after date: {review_after}")
    return errors


def proposal_to_entry(proposal: Proposal, run_date: date) -> str:
    fields = proposal.fields
    evidence = "; ".join(parse_evidence(fields.get("evidence")))
    review_after = str(fields["review_after"]).strip()
    return (
        f"- {run_date.isoformat()} [{fields['type']}, {fields['confidence']}]: "
        f"{str(fields['proposed_entry']).strip()} "
        f"Evidence: {evidence}. Review: {review_after}."
    )


def validate_proposal(
    proposal: Proposal, root: Path, config: dict[str, Any], run_date: date
) -> Decision:
    errors = validate_required_fields(proposal, config)
    if errors:
        return Decision("rejected", proposal, "; ".join(errors))

    combined = json.dumps(proposal.fields, sort_keys=True)
    if contains_high_confidence_secret(combined) or contains_secret_like_text(combined):
        return Decision("rejected", proposal, "proposal contains secret-like text")

    evidence = parse_evidence(proposal.fields.get("evidence"))
    if not evidence:
        return Decision("rejected", proposal, "missing evidence")
    verified = [item for item in evidence if evidence_is_verified(item, root, config)]
    if not verified:
        return Decision("rejected", proposal, "no verifiable evidence")

    target = (root / str(proposal.fields["suggested_target"]).strip()).resolve()
    target_class = classify_target(target, root, config)
    if target_class == "outside":
        return Decision("rejected", proposal, "target is outside allowed memory paths")

    entry = proposal_to_entry(proposal, run_date)
    if target_class == "review":
        return Decision("policy", proposal, "review-only target", target=target, entry=entry)
    role = str(proposal.fields["role"])
    if not is_canonical_role_memory_target(target, root, config, role):
        return Decision("rejected", proposal, "auto-apply target must match the role's canonical MEMORY.md")
    return Decision("accepted", proposal, "accepted", target=target, entry=entry)


def ensure_durable_section(text: str) -> str:
    if "## Durable Learnings" in text:
        return text
    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}\n## Durable Learnings\n\n"


def append_memory_entry(existing: str, entry: str) -> tuple[str, bool]:
    normalized_entry = normalize_text(entry)
    if normalized_entry in normalize_text(existing):
        return existing, False

    text = ensure_durable_section(existing)
    lines = text.splitlines()
    output: list[str] = []
    inserted = False
    for index, line in enumerate(lines):
        output.append(line)
        if line.strip() == "## Durable Learnings" and not inserted:
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if next_line:
                output.append("")
            output.append(entry)
            inserted = True
    if not inserted:
        if output and output[-1].strip():
            output.append("")
        output.extend(["## Durable Learnings", "", entry])
    return "\n".join(output).rstrip() + "\n", True


def mark_stale_entries(text: str, run_date: date) -> tuple[str, int]:
    changed = 0
    output = []
    for line in text.splitlines():
        match = REVIEW_DATE_RE.search(line)
        review_date: date | None = None
        if match:
            try:
                review_date = date.fromisoformat(match.group(1))
            except ValueError:
                # A malformed-but-regex-matching date (e.g. 2026-13-45) must not
                # crash the unattended run; leave the line untouched.
                review_date = None
        if (
            review_date is not None
            and review_date < run_date
            and "[STALE " not in line
            and "[SUPERSEDED " not in line
        ):
            line = f"{line} [STALE {run_date.isoformat()}]"
            changed += 1
        output.append(line)
    if not changed:
        return text, 0
    return "\n".join(output).rstrip() + "\n", changed


def unified_diff_text(root: Path, path: Path, old: str, new: str) -> str:
    rel = posix_rel(path, root) if path.exists() or is_relative_to(path, root) else path.name
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def diff_line_count(diff_text: str) -> int:
    return sum(1 for line in diff_text.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def safe_name(path: Path, root: Path) -> str:
    try:
        rel = posix_rel(path, root)
    except ValueError:
        rel = path.name
    return rel.replace("/", "__").replace(" ", "_")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def backup_file(path: Path, root: Path, run_dir: Path) -> None:
    if not path.exists():
        return
    dest = run_dir / "backups" / posix_rel(path, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def write_policy_patch(
    root: Path, run_dir: Path, target: Path, entry: str, reason: str
) -> Path:
    old = target.read_text(encoding="utf-8") if target.exists() else ""
    addition = (
        "\n\n## Proposed Memory Update\n\n"
        f"{entry}\n"
        f"\n<!-- Proposed by memory_maintainer.py: {reason}. Review before applying. -->\n"
    )
    new = old.rstrip() + addition
    diff = unified_diff_text(root, target, old, new)
    patch_dir = run_dir / "proposed-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"{safe_name(target, root)}.patch"
    atomic_write(patch_path, diff)
    return patch_path


def acquire_lock(audit_dir: Path) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    lock_path = audit_dir / "memory-maintainer.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise MaintenanceError(f"another memory maintainer run is active: {lock_path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    return lock_path


def prune_audit_runs(audit_dir: Path, retain: int) -> list[str]:
    """Delete the oldest audit run dirs, keeping the newest `retain`.

    Run dir names are timestamps (`%Y%m%d-%H%M%S-%f`), so a lexical sort is
    chronological. The current run's dir is always the newest and therefore
    always survives as long as `retain >= 1`. A `retain <= 0` disables pruning.
    Only directories are considered, so sibling files like `latest-report.md`
    are never touched.
    """
    runs_dir = audit_dir / "runs"
    if retain <= 0 or not runs_dir.is_dir():
        return []
    run_dirs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()),
        key=lambda p: p.name,
    )
    pruned: list[str] = []
    for old in run_dirs[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        pruned.append(old.name)
    return pruned


def move_proposal(decision: Decision, root: Path, run_dir: Path, apply: bool) -> None:
    if not apply:
        return
    source = decision.proposal.path
    if not source.exists():
        return
    dest = run_dir / decision.status / safe_name(source, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, dest)


def memory_files(root: Path, config: dict[str, Any]) -> list[Path]:
    base = root / config["agent_memory_dir"]
    if not base.exists():
        return []
    files = []
    for role_dir in base.iterdir():
        if not role_dir.is_dir() or role_dir.is_symlink():
            continue
        path = role_dir / "MEMORY.md"
        if path.is_file() and not path.is_symlink() and is_canonical_role_memory_target(path, root, config):
            files.append(path)
    return sorted(files)


def review_only_files(root: Path, config: dict[str, Any]) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink() or not is_relative_to(path, root):
            continue
        rel = posix_rel(path, root)
        if matches_any(rel, config["review_only_globs"]):
            candidates.append(path)
    return sorted(candidates)


def scan_review_only_secrets(root: Path, config: dict[str, Any], run_dir: Path) -> list[str]:
    warnings = []
    seen: set[Path] = set()
    for path in review_only_files(root, config):
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        redacted, changed = redact_high_confidence_secrets(text)
        if changed:
            patch = write_policy_patch(root, run_dir, path, "Redact high-confidence secret.", "secret redaction")
            atomic_write(patch, unified_diff_text(root, path, text, redacted))
            warnings.append(f"secret-like content in review-only file; patch generated: {patch}")
    return warnings


def generate_report(
    root: Path,
    run_dir: Path,
    applied: list[str],
    rejected: list[str],
    deferred: list[str],
    policy_patches: list[str],
    warnings: list[str],
    errors: list[str],
    apply: bool,
) -> Path:
    def bullets(items: list[str]) -> list[str]:
        return [f"- {item}" for item in items] if items else ["- None"]

    report = [
        "# Memory Maintainer Report",
        "",
        f"Root: `{root}`",
        f"Mode: {'apply' if apply else 'dry-run'}",
        "",
        "## Applied",
        *bullets(applied),
        "",
        "## Rejected",
        *bullets(rejected),
        "",
        "## Deferred",
        *bullets(deferred),
        "",
        "## Policy Patches",
        *bullets(policy_patches),
        "",
        "## Warnings",
        *bullets(warnings),
        "",
        "## Errors",
        *bullets(errors),
        "",
    ]
    report_path = run_dir / "report.md"
    atomic_write(report_path, "\n".join(report))
    latest = run_dir.parent / "latest-report.md"
    atomic_write(latest, "\n".join(report))
    return report_path


def run(root: Path, config_path: Path | None, apply: bool, now: str | None) -> RunResult:
    root = root.resolve()
    config = load_config(root, config_path)
    validate_config_safety(root, config)
    run_date = today_for(config, now)
    audit_dir = root / config["audit_dir"]
    lock_path = acquire_lock(audit_dir)
    run_dir = audit_dir / "runs" / datetime.now(ZoneInfo(config["timezone"])).strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=True)

    applied: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    policy_patches: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    edits: dict[Path, str] = {}
    fingerprints: dict[Path, str] = {}
    decisions: list[Decision] = []

    try:
        for path in memory_files(root, config):
            old = path.read_text(encoding="utf-8")
            fingerprints[path] = sha256_file(path)
            redacted, redacted_changed = redact_high_confidence_secrets(old)
            stale_marked, stale_count = mark_stale_entries(redacted, run_date)
            if len(stale_marked.splitlines()) > int(config["memory_index_max_lines"]):
                warnings.append(f"{posix_rel(path, root)} exceeds memory_index_max_lines")
            if redacted_changed:
                applied.append(f"redacted high-confidence secret in {posix_rel(path, root)}")
            if stale_count:
                applied.append(f"marked {stale_count} stale entries in {posix_rel(path, root)}")
            if stale_marked != old:
                edits[path] = stale_marked

        for proposal_path in discover_proposal_files(root, config):
            try:
                if proposal_path.is_symlink() or not is_relative_to(proposal_path, root):
                    proposal = Proposal(proposal_path, {}, "")
                    decision = Decision("rejected", proposal, "unsafe proposal path")
                else:
                    proposal = parse_proposal_file(proposal_path)
                    decision = validate_proposal(proposal, root, config, run_date)
            except Exception as exc:  # keep malformed proposals out of memory
                raw = proposal_path.read_text(encoding="utf-8", errors="replace")
                proposal = Proposal(proposal_path, {}, raw)
                decision = Decision("rejected", proposal, f"parse failed: {exc}")
            decisions.append(decision)

            rel = display_rel(proposal_path, root)
            if decision.status == "accepted" and decision.target and decision.entry:
                old = edits.get(decision.target)
                if old is None:
                    old = decision.target.read_text(encoding="utf-8") if decision.target.exists() else ""
                    fingerprints.setdefault(decision.target, sha256_file(decision.target))
                new, changed = append_memory_entry(old, decision.entry)
                if changed:
                    edits[decision.target] = new
                    applied.append(f"promoted {rel} to {posix_rel(decision.target, root)}")
                else:
                    applied.append(f"deduplicated already-present proposal {rel}")
            elif decision.status == "policy" and decision.target and decision.entry:
                patch = write_policy_patch(root, run_dir, decision.target, decision.entry, decision.reason)
                decision.patch_path = patch
                policy_patches.append(str(patch))
            else:
                rejected.append(f"{rel}: {decision.reason}")

        warnings.extend(scan_review_only_secrets(root, config, run_dir))

        total_diff_lines = 0
        for path, new_text in edits.items():
            target_class = classify_target(path, root, config)
            if target_class != "auto":
                errors.append(f"blocked edit outside auto-apply scope: {posix_rel(path, root)}")
                continue
            old_text = path.read_text(encoding="utf-8") if path.exists() else ""
            diff = unified_diff_text(root, path, old_text, new_text)
            file_diff_lines = diff_line_count(diff)
            total_diff_lines += file_diff_lines
            if file_diff_lines > int(config["max_diff_lines_per_file"]):
                errors.append(f"blocked oversized diff for {posix_rel(path, root)}: {file_diff_lines} lines")
            if contains_high_confidence_secret(new_text):
                errors.append(f"blocked write with high-confidence secret: {posix_rel(path, root)}")
        if total_diff_lines > int(config["max_total_diff_lines"]):
            errors.append(f"blocked oversized total diff: {total_diff_lines} lines")

        if apply and not errors:
            for path, new_text in edits.items():
                if path.exists() and fingerprints.get(path, sha256_file(path)) != sha256_file(path):
                    raise MaintenanceError(f"target changed mid-run: {posix_rel(path, root)}")
                backup_file(path, root, run_dir)
                atomic_write(path, new_text)
            for decision in decisions:
                move_proposal(decision, root, run_dir, apply=True)
        elif not apply:
            deferred.append("dry-run only; no edits or proposal moves applied")
        elif errors:
            deferred.append("auto-apply aborted because safety errors were detected")

        if apply:
            pruned = prune_audit_runs(audit_dir, int(config["audit_runs_retained"]))
            if pruned:
                applied.append(
                    f"pruned {len(pruned)} old audit run(s), keeping newest "
                    f"{int(config['audit_runs_retained'])}"
                )

        report_path = generate_report(
            root,
            run_dir,
            applied,
            rejected,
            deferred,
            policy_patches,
            warnings,
            errors,
            apply,
        )
        return RunResult(run_dir, applied, rejected, deferred, policy_patches, warnings, errors, report_path)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and curate shared agent role memory.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to maintain.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON path.")
    parser.add_argument("--apply", action="store_true", help="Apply safe auto-scope edits.")
    parser.add_argument("--now", default=None, help="Override run date as YYYY-MM-DD for tests.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args.root, args.config, args.apply, args.now)
    except MaintenanceError as exc:
        print(f"memory maintainer failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "report_path": str(result.report_path),
                    "run_dir": str(result.run_dir),
                    "applied": result.applied,
                    "rejected": result.rejected,
                    "deferred": result.deferred,
                    "policy_patches": result.policy_patches,
                    "warnings": result.warnings,
                    "errors": result.errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Memory maintainer report: {result.report_path}")
        if result.errors:
            print("Safety errors detected; auto-apply was blocked.")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
