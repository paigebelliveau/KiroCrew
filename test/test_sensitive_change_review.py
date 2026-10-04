"""Unit tests for .github/scripts/sensitive_change_review.py.

The gate turns red when a sensitive path changes and no trusted human approved
the CURRENT head with a filled-in reasoning template. These tests pin each way
an approval can fail to count, and the workflow wiring that keeps the checker
out of the PR's own hands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from skill_script_helpers import load_skill_script

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "sensitive_change_review.py"
WORKFLOW = ROOT / ".github" / "workflows" / "sensitive-change-review.yml"
HEAD = "a" * 40
OLD = "b" * 40

GOOD_BODY = """Looks right.

## Sensitive change review
- What rule changed: the git push floor now also blocks --mirror.
- Before -> after (what was allowed/blocked, now): --mirror was allowed, now it is blocked.
- Worst case if wrong (breaks commands / leaks secret / locks user out): a normal push could be blocked.
- Evidence checked (which sandbox result proves it): macOS Seatbelt run in the PR body, both cases.
- No regression:
  - Tools still work (which tools/commands were run, result): read, write, git status, git push all passed.
  - Backward compatible (old config / old data / old callers still work, how checked): old denied_commands.json loads unchanged.
  - Existing tests (which suites passed on this head): test_security and test_sandbox passed.
- How to undo: revert this one commit.
"""


@pytest.fixture(scope="module")
def mod():
    return load_skill_script("sensitive_change_review", SCRIPT)


def _pr(author="alice"):
    return {"head": {"sha": HEAD}, "user": {"login": author}}


def _review(body=GOOD_BODY, login="bob", state="APPROVED", commit=HEAD, typ="User"):
    return {
        "state": state,
        "body": body,
        "commit_id": commit,
        "user": {"login": login, "type": typ},
    }


def _writers(*logins):
    return lambda login: "write" if login in logins else "read"


def _eval(mod, pr, files, reviews, perm=None):
    return mod.evaluate(pr, files, reviews, perm or _writers("bob", "carol"))


SENSITIVE = ["src/kiro_crew/sandbox.py", "README.md"]


def test_no_sensitive_path_passes_without_review(mod):
    assert _eval(mod, _pr(), ["README.md", "src/kiro_crew/cli.py"], []).ok


@pytest.mark.parametrize(
    "path",
    [
        "src/kiro_crew/sandbox.py",
        "src/kiro_crew/sandbox_seatbelt.py",
        "src/kiro_crew/security/redaction.py",
        "src/kiro_crew/secrets/vault.py",
        "src/kiro_crew/log_redaction.py",
        "src/kiro_crew/apps/scrub_sdk.py",
        "src/kiro_crew/hooks.py",
        "src/kiro_crew/security_posture.py",
        "src/kiro_crew/computer_use/gate.py",
        "src/kiro_crew/dashboard/handlers/credential_redaction.py",
        ".github/scripts/sensitive_change_review.py",
        ".github/workflows/sensitive-change-review.yml",
    ],
)
def test_sensitive_globs_match(mod, path):
    assert mod.sensitive_files([path]) == [path]


def test_sensitive_globs_name_real_files(mod):
    # A glob that matches nothing tracked protects nothing.
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    tracked = out.splitlines()
    for glob in mod.SENSITIVE_GLOBS:
        assert any(mod.fnmatch.fnmatchcase(t, glob) for t in tracked), glob


def test_complete_approval_on_head_passes(mod):
    v = _eval(mod, _pr(), SENSITIVE, [_review()])
    assert v.ok, v.messages
    assert v.sensitive == ["src/kiro_crew/sandbox.py"]


def test_no_review_fails(mod):
    v = _eval(mod, _pr(), SENSITIVE, [])
    assert not v.ok
    assert "No approval" in v.messages[0]


def test_bare_approval_fails(mod):
    v = _eval(mod, _pr(), SENSITIVE, [_review(body="")])
    assert not v.ok
    assert "heading" in v.messages[0]


def test_approval_on_old_commit_fails(mod):
    v = _eval(mod, _pr(), SENSITIVE, [_review(commit=OLD)])
    assert not v.ok
    assert "older commit" in v.messages[0]


def test_bot_approval_fails(mod):
    v = _eval(mod, _pr(), SENSITIVE, [_review(login="kiro[bot]", typ="Bot")])
    assert not v.ok
    assert "bots" in v.messages[0]


def test_self_approval_fails(mod):
    v = _eval(mod, _pr(author="bob"), SENSITIVE, [_review(login="bob")])
    assert not v.ok
    assert "author" in v.messages[0]


def test_read_only_reviewer_fails(mod):
    # A read-only collaborator still reads as COLLABORATOR in author_association,
    # so the gate asks the permission API instead.
    v = _eval(mod, _pr(), SENSITIVE, [_review()], perm=_writers())
    assert not v.ok
    assert "write access" in v.messages[0]


def test_admin_reviewer_passes(mod):
    v = _eval(mod, _pr(), SENSITIVE, [_review()], perm=lambda _login: "admin")
    assert v.ok


def test_renamed_sensitive_file_counts_by_old_name(mod):
    rows = [{"filename": "src/kiro_crew/moved.py", "previous_filename": "src/kiro_crew/sandbox.py"}]
    paths = mod.changed_paths(rows)
    assert mod.sensitive_files(paths) == ["src/kiro_crew/sandbox.py"]


def test_file_list_must_be_complete(mod):
    rows = [{"filename": "a.py"}]
    assert mod.file_list_problem({"changed_files": 1}, rows) is None
    assert "reports 2" in mod.file_list_problem({"changed_files": 2}, rows)
    assert "reports None" in mod.file_list_problem({}, rows)
    big = [{"filename": f"f{i}"} for i in range(mod.FILES_API_CAP)]
    assert "3000+" in mod.file_list_problem({"changed_files": mod.FILES_API_CAP}, big)


def test_later_changes_requested_revokes_approval(mod):
    reviews = [_review(), _review(state="CHANGES_REQUESTED", body="wait")]
    assert not _eval(mod, _pr(), SENSITIVE, reviews).ok


def test_later_comment_does_not_revoke_approval(mod):
    reviews = [_review(), _review(state="COMMENTED", body="nit")]
    assert _eval(mod, _pr(), SENSITIVE, reviews).ok


def test_one_good_approval_among_bad_ones_passes(mod):
    reviews = [_review(login="carol", body="LGTM"), _review(login="bob")]
    assert _eval(mod, _pr(), SENSITIVE, reviews).ok


@pytest.mark.parametrize(
    "label", ["Tools still work", "Backward compatible", "Existing tests", "How to undo"]
)
def test_missing_field_fails(mod, label):
    body = "\n".join(line for line in GOOD_BODY.splitlines() if label not in line)
    problems = mod.template_problems(body)
    assert any(label in p and "missing" in p for p in problems), problems


@pytest.mark.parametrize("value", ["", "N/A", "n/a", "none", "LGTM", "fine", "looks ok"])
def test_empty_or_short_value_fails(mod, value):
    body = GOOD_BODY.replace("revert this one commit.", value)
    problems = mod.template_problems(body)
    assert any("How to undo" in p for p in problems), problems


def test_na_with_reason_passes(mod):
    body = GOOD_BODY.replace(
        "old denied_commands.json loads unchanged.",
        "N/A because no config or data format changed.",
    )
    assert mod.template_problems(body) == []


def test_value_on_continuation_line_counts(mod):
    body = GOOD_BODY.replace(
        "- How to undo: revert this one commit.",
        "- How to undo:\n  revert this one commit and redeploy.",
    )
    assert mod.template_problems(body) == []


def test_heading_is_case_insensitive(mod):
    body = GOOD_BODY.replace("## Sensitive change review", "## SENSITIVE CHANGE REVIEW")
    assert mod.template_problems(body) == []


def test_failure_summary_carries_the_template(mod):
    v = _eval(mod, _pr(), SENSITIVE, [])
    text = mod._summary(v)
    assert mod.HEADING in text
    for label in mod.REQUIRED_FIELDS:
        assert label in text


class TestWorkflow:
    @pytest.fixture(scope="class")
    def wf(self):
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_reruns_on_review_events(self, wf):
        on = wf.get("on") or wf[True]
        assert set(on["pull_request_review"]["types"]) == {"submitted", "edited", "dismissed"}
        assert {"synchronize", "edited"} <= set(on["pull_request"]["types"])

    def test_read_only_token(self, wf):
        assert wf["permissions"] == {"contents": "read", "pull-requests": "read"}

    def test_runs_the_default_branch_checker_first(self, wf):
        (job,) = wf["jobs"].values()
        base = job["steps"][0]["with"]
        assert base["ref"] == "${{ github.event.repository.default_branch }}"
        run = job["steps"][-1]["run"]
        assert 'script="base/.github/scripts/sensitive_change_review.py"' in run
        # The PR's own copy is never checked out or run.
        assert "head/" not in run
        assert all("head" not in str(step.get("with", {}).get("ref", "")) for step in job["steps"])

    def test_no_waiver_label(self, wf):
        assert "labels" not in WORKFLOW.read_text(encoding="utf-8")
