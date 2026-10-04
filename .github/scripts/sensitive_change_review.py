#!/usr/bin/env python3
"""Sensitive Change Review gate.

A PR that touches a sensitive path (the sandbox, the command floor, secret
scrubbing, redaction) needs a human approval ON THE CURRENT HEAD whose review
body carries a filled-in reasoning template. A bare "Approve" click does not
count. The gate checks format only: it cannot judge whether the reasoning is
right, it only makes sure the reviewer wrote it down.

Deterministic on purpose: no model, no sandbox run. It reads the PR, its file
list and its reviews through `gh api` and answers one question.

Usage (CI): REPO=owner/name PR=123 python3 sensitive_change_review.py
Exit codes: 0 pass / not applicable, 1 requirement not met, 2 API/read error.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from typing import Any, Callable

# Paths whose change can break every command, leak a secret, or lock a user
# out. `*` in fnmatch also matches `/`, so a directory glob covers subdirs.
# The gate's own files are listed so weakening the gate needs the same review.
SENSITIVE_GLOBS: tuple[str, ...] = (
    "src/kiro_crew/sandbox*.py",
    "src/kiro_crew/security/*",
    "src/kiro_crew/secrets/*",
    "src/kiro_crew/log_redaction.py",
    "src/kiro_crew/snapshot_redact.py",
    "src/kiro_crew/apps/scrub_sdk.py",
    "src/kiro_crew/hooks.py",
    "src/kiro_crew/security_posture.py",
    "src/kiro_crew/computer_use/gate.py",
    "src/kiro_crew/dashboard/handlers/*redaction*.py",
    ".github/workflows/sensitive-change-review.yml",
    ".github/scripts/sensitive_change_review.py",
)

HEADING = "## Sensitive change review"

# Every field the reviewer must fill. The parent "No regression:" line only
# groups the three sub-items below it, so it carries no value of its own.
REQUIRED_FIELDS: tuple[str, ...] = (
    "What rule changed",
    "Before -> after",
    "Worst case if wrong",
    "Evidence checked",
    "Tools still work",
    "Backward compatible",
    "Existing tests",
    "How to undo",
)

# A value that says nothing. "N/A" alone is empty; "N/A because ..." is not.
_PLACEHOLDERS = {"", "n/a", "na", "none", "-", "--", "tbd", "todo", ".", "ok", "yes", "no", "lgtm"}
MIN_WORDS = 3

# Only people with write access can approve. `author_association` is not
# enough: a read-only collaborator or org member also reads as COLLABORATOR /
# MEMBER, so the repository's own permission answer is asked instead.
# (`maintain` reports as `write` in this field.)
WRITE_PERMISSIONS = {"admin", "write"}

# GitHub's PR files endpoint stops at this many rows without an error.
FILES_API_CAP = 3000

_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")


class Verdict:
    """The gate's answer: pass/fail, the sensitive files, and why."""

    def __init__(
        self, ok: bool, sensitive: list[str] | None = None, messages: list[str] | None = None
    ):
        self.ok = ok
        self.sensitive = sensitive or []
        self.messages = messages or []


def sensitive_files(files: list[str]) -> list[str]:
    """The changed files that match a sensitive glob, in input order."""
    return [f for f in files if any(fnmatch.fnmatchcase(f, g) for g in SENSITIVE_GLOBS)]


def _field_value(lines: list[str], label: str) -> str | None:
    """Text after `label ...:` on its bullet, plus non-bullet continuation lines.

    Returns None when no bullet starts with the label.
    """
    want = label.lower()
    for i, line in enumerate(lines):
        m = _BULLET_RE.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        if not text.lower().startswith(want):
            continue
        colon = text.find(":", len(label))
        if colon < 0:
            return ""
        parts = [text[colon + 1 :].strip()]
        for nxt in lines[i + 1 :]:
            if not nxt.strip() or _BULLET_RE.match(nxt) or _HEADING_RE.match(nxt):
                break
            parts.append(nxt.strip())
        return " ".join(p for p in parts if p).strip()
    return None


def _is_empty(value: str) -> bool:
    stripped = value.strip().strip("`*_").strip()
    if stripped.lower() in _PLACEHOLDERS:
        return True
    return len(stripped.split()) < MIN_WORDS


def template_problems(body: str) -> list[str]:
    """Why a review body does not satisfy the template ([] when it does)."""
    if HEADING.lower() not in (body or "").lower():
        return [f"missing the `{HEADING}` heading"]
    section = body[body.lower().index(HEADING.lower()) + len(HEADING) :]
    lines = section.splitlines()
    problems: list[str] = []
    for label in REQUIRED_FIELDS:
        value = _field_value(lines, label)
        if value is None:
            problems.append(f"missing field `{label}`")
        elif _is_empty(value):
            problems.append(
                f"field `{label}` is empty or too short (need {MIN_WORDS}+ words; "
                "`N/A` needs a reason)"
            )
    return problems


def _latest_by_reviewer(reviews: list[dict]) -> dict[str, dict]:
    """Each reviewer's most recent state-changing review.

    A later CHANGES_REQUESTED or DISMISSED review replaces an earlier approval;
    a plain COMMENTED review does not change the reviewer's standing.
    """
    latest: dict[str, dict] = {}
    for review in reviews:
        if review.get("state") not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            continue
        login = (review.get("user") or {}).get("login") or ""
        if login:
            latest[login] = review
    return latest


def changed_paths(rows: list[dict]) -> list[str]:
    """Every path a PR touches, including the OLD name of a renamed file.

    Without the old name, moving a sensitive file outside its glob in the same
    PR that edits it would hide the edit.
    """
    paths: list[str] = []
    for row in rows:
        for key in ("filename", "previous_filename"):
            if row.get(key):
                paths.append(row[key])
    return paths


def file_list_problem(pr: dict, rows: list[dict]) -> str | None:
    """Why the file list cannot be trusted as complete (None when it can)."""
    expected = pr.get("changed_files")
    if len(rows) >= FILES_API_CAP:
        return f"the PR changes {FILES_API_CAP}+ files, past what the API lists"
    if not isinstance(expected, int) or len(rows) != expected:
        return f"the API listed {len(rows)} files but the PR reports {expected}"
    return None


def evaluate(
    pr: dict,
    files: list[str],
    reviews: list[dict],
    permission_of: Callable[[str], str],
) -> Verdict:
    """Decide the gate. `permission_of(login)` returns the repo permission."""
    hits = sensitive_files(files)
    if not hits:
        return Verdict(ok=True, messages=["No sensitive path changed; nothing to check."])

    head = (pr.get("head") or {}).get("sha") or ""
    author = (pr.get("user") or {}).get("login") or ""
    messages: list[str] = []
    for login, review in sorted(_latest_by_reviewer(reviews).items()):
        user = review.get("user") or {}
        why: list[str] = []
        if review.get("state") != "APPROVED":
            continue
        if user.get("type") == "Bot" or login.endswith("[bot]"):
            why.append("bots do not count")
        if login == author:
            why.append("the PR author cannot approve their own change")
        if not why and permission_of(login) not in WRITE_PERMISSIONS:
            why.append("reviewer has no write access")
        if review.get("commit_id") != head:
            why.append(
                f"approved an older commit ({(review.get('commit_id') or '?')[:12]}); re-approve on {head[:12]}"
            )
        why.extend(template_problems(review.get("body") or ""))
        if not why:
            return Verdict(
                ok=True,
                sensitive=hits,
                messages=[f"@{login} approved {head[:12]} with a complete reasoning template."],
            )
        messages.append(f"@{login}: " + "; ".join(why))

    if not messages:
        messages.append(f"No approval on {head[:12]} yet.")
    return Verdict(ok=False, sensitive=hits, messages=messages)


def _gh_json(path: str, paginate: bool = False) -> Any:
    cmd = ["gh", "api", path]
    if paginate:
        # --slurp wraps each page's array into one outer array.
        cmd += ["--paginate", "--slurp"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    data = json.loads(out)
    if paginate:
        return [item for page in data for item in page]
    return data


def _summary(verdict: Verdict) -> str:
    lines = ["### Sensitive Change Review", ""]
    if verdict.sensitive:
        lines.append("Sensitive files changed:")
        lines += [f"- `{f}`" for f in verdict.sensitive]
        lines.append("")
    lines += [f"- {m}" for m in verdict.messages]
    if not verdict.ok:
        lines += [
            "",
            "A human reviewer with write access must approve the CURRENT head and",
            "paste this template into the approval, every field filled",
            "(`N/A` needs a reason):",
            "",
            "```",
            HEADING,
            "- What rule changed:",
            "- Before -> after (what was allowed/blocked, now):",
            "- Worst case if wrong (breaks commands / leaks secret / locks user out):",
            "- Evidence checked (which sandbox result proves it):",
            "- No regression:",
            "  - Tools still work (which tools/commands were run, result):",
            "  - Backward compatible (old config / old data / old callers still work, how checked):",
            "  - Existing tests (which suites passed on this head):",
            "- How to undo:",
            "```",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    repo = os.environ.get("REPO", "")
    pr_number = os.environ.get("PR", "")
    if not repo or not pr_number.isdigit():
        print("::error::REPO and a numeric PR must be set.")
        return 2
    base = f"repos/{repo}/pulls/{pr_number}"
    try:
        pr = _gh_json(base)
        rows = _gh_json(f"{base}/files?per_page=100", paginate=True)
        reviews = _gh_json(f"{base}/reviews?per_page=100", paginate=True)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, TypeError) as exc:
        # A read failure is not a passing PR: fail closed and say why.
        print(
            f"::error::Could not read the PR from the API ({type(exc).__name__}); re-run this job."
        )
        return 2

    # An incomplete file list is not a list with no sensitive file in it.
    problem = file_list_problem(pr, rows)
    if problem:
        print(f"::error::Cannot see every changed file ({problem}); failing closed.")
        return 1

    def permission_of(login: str) -> str:
        # Fails closed: any read error counts as no access.
        try:
            data = _gh_json(f"repos/{repo}/collaborators/{login}/permission")
            return str(data.get("permission") or "none")
        except (subprocess.CalledProcessError, json.JSONDecodeError, AttributeError):
            return "none"

    verdict = evaluate(pr, changed_paths(rows), reviews, permission_of)
    text = _summary(verdict)
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text)
    if not verdict.ok:
        print("::error::Sensitive path changed without a reasoned human approval on this head.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
