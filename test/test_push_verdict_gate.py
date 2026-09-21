"""Tests for the gateway-owned push verdict store and its trusted activation.

The properties under test are the ones the gate's acceptance names, and two of them are
explicit requirements: an agent-written config can neither enable nor disable the gate, and
a trusted activation survives a restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew.security import push_verdict


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    return tmp_path


def _activate(home: Path, *, enabled: bool = True) -> Path:
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")
    return leaf


def _url(name: str) -> str:
    """A synthetic remote URL. Never a real host and never a /home/<name> path."""
    return f"https://git.example.invalid/{name}.git"


def _credentialed_https(secret: str = "s3cr3t") -> str:
    """A synthetic HTTPS URL carrying an embedded ``user:secret@`` credential.

    Composed from pieces rather than spelled as a literal token URL: the point is only that a
    ``:`` precedes the ``@`` in the authority, which is the embedded-password shape.
    """
    return "https://x-access-token:" + secret + "@git.example.invalid/repo.git"


# ── Trusted activation, the conductor's two required properties ──


def test_activation_is_off_on_an_installation_nobody_activated(home: Path) -> None:
    assert push_verdict.activation_enabled() is False


def test_activation_is_on_when_the_keystone_leaf_says_so(home: Path) -> None:
    _activate(home)
    assert push_verdict.activation_enabled() is True


def test_an_agent_written_config_can_neither_enable_nor_disable_the_gate(home: Path) -> None:
    """config.json may REQUEST activation; only the keystone AUTHORIZES it."""
    config = home / "config.json"
    config.write_text(
        json.dumps({"security": {"push_verdict_required": True, "push_verdict_enabled": True}}),
        encoding="utf-8",
    )
    # Config alone does not enable it.
    assert push_verdict.activation_enabled() is False

    # And config cannot switch OFF what the keystone turned on.
    _activate(home)
    config.write_text(
        json.dumps({"security": {"push_verdict_required": False, "push_verdict_enabled": False}}),
        encoding="utf-8",
    )
    assert push_verdict.activation_enabled() is True


def test_trusted_activation_survives_a_restart(home: Path) -> None:
    """Activation is on disk, so a fresh process still sees it."""
    _activate(home)
    assert push_verdict.activation_enabled() is True


def test_a_malformed_activation_leaf_refuses_instead_of_failing_open(home: Path) -> None:
    """An activation leaf that cannot be PARSED must not read as "gating off".

    Absence and unreadability are different facts. Absence means nobody activated gating,
    and false is the honest answer. A leaf that EXISTS but is malformed means the operator's
    activation state is unknown, and answering false there would make corrupting one file a
    way to disable the gate on an installation that had turned it on -- the gate's own off
    switch, reachable by damage rather than by authorization.

    So the store raises and the floor refuses the publish rather than allowing it. An
    installation that never activated never reaches this branch: absence still returns
    false rather than raising, which the neighbouring test pins.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text("{not json", encoding="utf-8")
    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict.activation_enabled()


def test_a_non_object_activation_leaf_refuses_through_the_wrapper_too(home: Path) -> None:
    """The wrapper must not swallow what the reader raises.

    ``activation_enabled()`` is what the floor calls, so a refusal only ``activation()`` raised
    would be a fail-open at the one call site that matters. A suite that agrees with a hole
    cannot find it: while a test asserted ``[true]`` left the gate off, no mutation could be
    killed by the distinction, because both the code and the test read it the same way.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text("[true]", encoding="utf-8")
    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict.activation_enabled()


def test_activation_requires_a_real_json_true(home: Path) -> None:
    """A truthy value is not an enable, and it is not a silent disable either.

    The JSON string ``"false"`` and the number ``1`` are both truthy in Python, so a
    ``bool(...)`` read would activate the gate on either -- that half has always held. The
    other half was wrong: reading them as OFF made a corrupted leaf disable the gate on an
    installation whose operator had enabled it, so a present non-boolean now REFUSES. Only
    two shapes are an operator's: a real ``true`` and a real ``false``. An explicit ``null``
    is indistinguishable from an absent key through ``.get()``, so it keeps the absent
    reading, which is off.
    """
    leaf = _activate(home)
    for corrupted in ("false", 1, "true"):
        leaf.write_text(json.dumps({"enabled": corrupted}), encoding="utf-8")
        with pytest.raises(push_verdict.ActivationUnreadable):
            push_verdict.activation_enabled()
    leaf.write_text(json.dumps({"enabled": None}), encoding="utf-8")
    assert push_verdict.activation_enabled() is False
    leaf.write_text(json.dumps({"enabled": False}), encoding="utf-8")
    assert push_verdict.activation_enabled() is False
    leaf.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    assert push_verdict.activation_enabled() is True


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "5",
        '"true"',
        "null",
        '{"enabled": 1}',
        '{"enabled": "true"}',
        '{"enabled": "false"}',
    ],
)
def test_a_present_but_malformed_activation_refuses_rather_than_reading_off(
    home: Path, document: str
) -> None:
    """Off is the honest answer for an ABSENT leaf only.

    Unparseable bytes already raised. A document that PARSES but is not an object, or whose
    ``enabled`` is present and not a real boolean, was returning off -- a fail-open with a
    narrower entrance than the parse error: truncating the leaf to ``[]``, or writing
    ``{"enabled": 1}``, silently disabled the gate on an installation whose operator had
    turned it on. Note ``"false"`` is in here too: a STRING is a corrupted write whichever
    word it spells, and guessing the operator meant a disable is the same guess as guessing
    they meant an enable.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(document, encoding="utf-8")

    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict.activation()


def test_the_two_shapes_an_operator_actually_writes_are_honoured(home: Path) -> None:
    """The other side of the refusal above: a real boolean is read, and absence is off."""
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)

    leaf.write_text('{"enabled": false}', encoding="utf-8")
    assert push_verdict.activation().enabled is False, "an operator's disable must be honoured"

    leaf.write_text('{"enabled": true, "guard_sha256": "' + "a" * 64 + '"}', encoding="utf-8")
    assert push_verdict.activation().enabled is True

    leaf.write_text("{}", encoding="utf-8")
    assert push_verdict.activation().enabled is False, "absent enabled is a never-activated leaf"


def test_the_activation_read_is_live_on_every_publish(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activation is re-read per publish, and that is the deliberate choice.

    A reviewer reads the floor's "no expensive I/O" contract and sees an ``open()`` plus a JSON
    parse, which is a fair objection. A stat-keyed cache of the parse was written and removed:
    its key -- modification time, size and inode -- cannot tell two writes of the SAME byte
    length within one clock tick apart, and the value it would serve is an ENABLE decision, so
    the failure mode is the gate reading as off after an operator turned it on. This test is the
    pin that the cheap-but-wrong version does not come back.
    """
    leaf = _activate(home)
    parses = 0
    real_load = push_verdict.json.load

    def _counting_load(handle):
        nonlocal parses
        parses += 1
        return real_load(handle)

    monkeypatch.setattr(push_verdict.json, "load", _counting_load)

    for _ in range(3):
        assert push_verdict.activation().enabled is True
    assert parses == 3, "activation was served from a cache"

    # The property that matters: a change takes effect on the very next read, whatever its size.
    leaf.write_text(json.dumps({"enabled": False}), encoding="utf-8")
    assert push_verdict.activation().enabled is False


# ── The guard digest pin, read off the keystone ──


@pytest.mark.parametrize("pinned", ["", "abc123", "z" * 64, "A" * 63, 42, None])
def test_only_a_real_digest_counts_as_a_pin(home: Path, pinned: object) -> None:
    """A malformed pin reads as ABSENT, which is what produces the message that fixes it.

    Absence and "a digest that can never match" behave alike at the comparison, but only
    absence tells the operator to pin one. A 64-character non-hex string is the case a bare
    length check would wave through.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(json.dumps({"enabled": True, "guard_sha256": pinned}), encoding="utf-8")
    read = push_verdict.activation()
    assert read.enabled is True
    assert read.guard_sha256 == ""


def test_a_real_digest_is_read_back_lowercased(home: Path) -> None:
    """The companion, or the test above passes for an empty reader."""
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(json.dumps({"enabled": True, "guard_sha256": "A" * 64}), encoding="utf-8")
    assert push_verdict.activation().guard_sha256 == "a" * 64


# ── Finding A: the push DESTINATION is bound to an operator pin ──


def test_activation_reads_an_operator_pinned_push_url(home: Path) -> None:
    """The keystone carries the pin, and it is read back verbatim (only whitespace stripped)."""
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(
        json.dumps({"enabled": True, "pinned_push_url": "  " + _url("canonical") + "  "}),
        encoding="utf-8",
    )
    read = push_verdict.activation()
    assert read.enabled is True
    assert read.pinned_push_url == _url("canonical")


def test_an_absent_pin_leaves_the_destination_unconstrained(home: Path) -> None:
    """No pin key, and an explicit JSON null, both mean the operator pinned nothing.

    That is the never-pinned answer -- an empty string -- and the destination is unconstrained
    exactly as on an install that never activated. The pin is an opt-in, not a default.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    assert push_verdict.activation().pinned_push_url == ""
    leaf.write_text(json.dumps({"enabled": True, "pinned_push_url": None}), encoding="utf-8")
    assert push_verdict.activation().pinned_push_url == ""


@pytest.mark.parametrize("corrupt", [42, ["x"], {"a": 1}, "", "   ", "--upload-pack=/opt/evil"])
def test_a_corrupted_pin_refuses_rather_than_reading_as_absent(home: Path, corrupt: object) -> None:
    """A PRESENT-but-broken pin is corruption, and reading it as "" would be a fail-open.

    Damaging this field would silently drop the destination binding on an installation whose
    operator set one -- the same fail-open the boolean and digest checks refuse. A non-string,
    a blank/whitespace string, and a leading-dash value (git reads ``-`` as an option, never a
    repository) all RAISE. An operator who wants no pin removes the key entirely.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(json.dumps({"enabled": True, "pinned_push_url": corrupt}), encoding="utf-8")
    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict.activation()


@pytest.mark.parametrize(
    "credentialed",
    [
        _credentialed_https(),
        # scp-like ``user:secret@host:path`` (no scheme) is the SSH embedded-password shape.
        "git-user:s3cr3t@git.example.invalid:repo.git",
        # Codex F1: a USERNAME-ONLY non-SSH URL (no ``:secret``) is still an unpinnable
        # credential carrier for HTTP(S) -- a pin needs no userinfo there -- so it is refused.
        "https://x-access-token@git.example.invalid/repo.git",
        "https://someuser@git.example.invalid/repo.git",
    ],
)
def test_a_pin_carrying_an_embedded_credential_is_refused(home: Path, credentialed: str) -> None:
    """The finding this closes: the activation leaf is readable in-sandbox, so REQUIRING a
    credentialed pin to publish would force a token into a leaf any ``open()`` can read.

    A URL whose userinfo carries a PASSWORD/TOKEN (a ``:`` before the ``@`` in the authority)
    is refused with ``ActivationUnreadable``. Mutation check: on pre-fix code ``_pinned_push_url``
    returned the credentialed value verbatim (no userinfo check), so ``activation()`` did NOT
    raise and this ``pytest.raises`` block failed.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(
        json.dumps({"enabled": True, "pinned_push_url": credentialed}), encoding="utf-8"
    )
    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict.activation()


@pytest.mark.parametrize(
    "credential_free",
    [
        "https://git.example.invalid/repo.git",
        # A bare ``user@host`` username with NO ``:`` password is allowed -- it is not a secret.
        "git@git.example.invalid:repo.git",
        "ssh://git@git.example.invalid/repo.git",
    ],
)
def test_a_credential_free_pin_is_accepted(home: Path, credential_free: str) -> None:
    """A URL that names a destination without embedding a password/token is accepted verbatim.

    The gateway authenticates the publish from its OWN configured credentials, so the pin need
    only name the repository. A bare ``user@host`` (SSH username, no secret) is NOT a credential
    and is allowed.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(
        json.dumps({"enabled": True, "pinned_push_url": credential_free}), encoding="utf-8"
    )
    assert push_verdict.activation().pinned_push_url == credential_free


def test_the_embedded_credential_refusal_message_is_actionable(home: Path) -> None:
    """The refusal tells the operator what to do: pin a credential-free URL."""
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text(
        json.dumps({"enabled": True, "pinned_push_url": _credentialed_https()}), encoding="utf-8"
    )
    with pytest.raises(push_verdict.ActivationUnreadable) as excinfo:
        push_verdict.activation()
    message = str(excinfo.value)
    assert "embedded credential" in message
    assert "no embedded credentials" in message
    assert "own configured credentials" in message


# ── The URL helpers the destination pin is compared through ──


def test_unparseable_port_helper_and_credential_free_url_do_not_raise() -> None:
    """The port helper detects a bad port and the credential-free compare never raises on one."""
    assert push_verdict._url_has_unparseable_port("https://h.invalid:notaport/r.git") is True
    assert push_verdict._url_has_unparseable_port("https://h.invalid:99999/r.git") is True
    assert push_verdict._url_has_unparseable_port("https://h.invalid:443/r.git") is False
    assert push_verdict._url_has_unparseable_port("https://h.invalid/r.git") is False
    # Defensive: the pure helper must not raise even if handed a bad-port URL directly.
    push_verdict._credential_free_url("https://h.invalid:notaport/r.git")


def test_credential_free_url_keeps_distinct_ipv6_destinations_distinct() -> None:
    """Two DISTINCT IPv6 destinations must not collapse to one credential-free identity.

    ``urlsplit(...).hostname`` returns an IPv6 literal WITHOUT its ``[...]`` brackets, so the
    pre-fix helper re-appended ``:port`` to the bare address and lost the host/port boundary:
    ``https://[2001:db8::1]:443/r.git`` and ``https://[2001:db8::1:443]/r.git`` (no port) both
    recomposed to ``https://2001:db8::1:443/r.git`` and compared EQUAL. Because the destination
    pin is a plain string equality over this output, that collision let an agent-chosen IPv6
    destination match the operator pin and be published to (UNBOUNDED). Mutation check: on the
    un-fixed helper the boundary-pair assertion below is EQUAL, so this test FAILS.
    """
    a = "https://[2001:db8::1]:443/r.git"
    b = "https://[2001:db8::1:443]/r.git"  # distinct dest; collided with ``a`` pre-fix
    c = "https://[2001:db8::2]:443/r.git"  # distinct address, same port
    d = "https://[2001:db8::1]:8443/r.git"  # same address, different port
    # Distinct IPv6 destinations stay distinct after normalization.
    assert push_verdict._credential_free_url(a) != push_verdict._credential_free_url(b)
    assert push_verdict._credential_free_url(a) != push_verdict._credential_free_url(c)
    assert push_verdict._credential_free_url(a) != push_verdict._credential_free_url(d)
    # The SAME IPv6 destination still normalizes to one identity (both with and without a port).
    assert push_verdict._credential_free_url(a) == push_verdict._credential_free_url(a)
    assert push_verdict._credential_free_url(
        "https://[2001:db8::1]/r.git"
    ) == push_verdict._credential_free_url("https://[2001:db8::1]/r.git")


def test_credential_free_url_is_unchanged_for_non_ipv6() -> None:
    """HTTP(S) drops userinfo; SSH/scp keeps the username; host/port/path preserved.

    A non-IPv6 hostname never contains ``:``, so the IPv6 re-bracketing never triggers, and the
    HTTP(S) output matches what the helper produced before the F2 fix (the token username is a
    credential carrier there, not identity). The scp-like/SSH forms now KEEP the username.
    """
    # Plain URL: unchanged.
    assert (
        push_verdict._credential_free_url("https://git.example.invalid/repo.git")
        == "https://git.example.invalid/repo.git"
    )
    # scheme URL with an embedded token: userinfo stripped, host/path preserved.
    assert (
        push_verdict._credential_free_url(
            "https://x-access-token:s3cr3t@git.example.invalid/canonical.git"
        )
        == "https://git.example.invalid/canonical.git"
    )
    # Host with an explicit port: port preserved, no brackets added.
    assert (
        push_verdict._credential_free_url("https://git.example.invalid:8443/repo.git")
        == "https://git.example.invalid:8443/repo.git"
    )
    # scp-like ``user@host:path``: the username is IDENTITY (login-relative path) and is KEPT;
    # only a ``:secret`` half would be stripped, and there is none here.
    assert (
        push_verdict._credential_free_url("git@git.example.invalid:path/repo.git")
        == "git@git.example.invalid:path/repo.git"
    )
    # scp-like with an embedded ``user:secret@``: the ``:secret`` is stripped, the username KEPT.
    assert (
        push_verdict._credential_free_url("user:secret@git.example.invalid:path/repo.git")
        == "user@git.example.invalid:path/repo.git"
    )


def test_credential_free_url_keeps_ssh_usernames_distinct() -> None:
    """Two SSH accounts at one host are two identities and must NOT collapse (codex F2).

    For the scp-like ``user@host:path`` form the path is LOGIN-RELATIVE, so
    ``deploy@host:repo`` and ``staging@host:repo`` are two repositories under two accounts.
    The pre-fix helper discarded the whole ``user@`` prefix, collapsing them to one identity so
    an agent-chosen account could match a pin that authorized a different one. The username is
    now preserved for the scp-like and ``ssh://`` forms, so the two stay distinct. Mutation
    check: restoring the old whole-userinfo strip makes both sides equal and this fails.
    """
    # scp-like: two accounts, one host, one path -> two DISTINCT identities.
    assert push_verdict._credential_free_url(
        "deploy@git.example.invalid:repo.git"
    ) != push_verdict._credential_free_url("staging@git.example.invalid:repo.git")
    # ssh:// scheme: username is identity here too and is preserved (only a password stripped).
    assert (
        push_verdict._credential_free_url("ssh://deploy@git.example.invalid/repo.git")
        == "ssh://deploy@git.example.invalid/repo.git"
    )
    assert push_verdict._credential_free_url(
        "ssh://deploy@git.example.invalid/repo.git"
    ) != push_verdict._credential_free_url("ssh://staging@git.example.invalid/repo.git")
    # A password on an ssh:// URL is still a secret: stripped, username kept.
    assert (
        push_verdict._credential_free_url("ssh://deploy:pw@git.example.invalid/repo.git")
        == "ssh://deploy@git.example.invalid/repo.git"
    )


# ── The presenter renders what the route actually answers ──
#
# These cover kept code: the agent-side MCP presenter in ``mcp_tools/push_verdict.py`` only
# RENDERS the gateway route's verdict (it writes nothing and is untouched by the floor
# subtraction), so its render-every-exit contract still holds and still needs pinning.


def _render(payload: dict, monkeypatch: pytest.MonkeyPatch) -> str:
    """Drive the real presenter against one route response."""
    from kiro_crew import mcp_core
    from kiro_crew.mcp_tools import push_verdict as tool

    monkeypatch.setattr(mcp_core, "require_strict_session_key", lambda _msg: ("session-1", ""))
    monkeypatch.setattr(mcp_core, "_post", lambda *_a, **_k: payload)
    return tool.push_verdict_run("push_verdict_run", {})


def test_the_presenter_renders_each_exit_the_route_can_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every route exit, because a name the route never sends is a silent mistranslation.

    The gateway performs the push itself and answers ``"published"`` on success and
    ``"not_published"`` when the push did not land, alongside
    ``"refused"``/``"error"``/``"not_activated"``; each must render as itself and never as
    "unrecognised verdict".
    """
    published = _render({"verdict": "published", "head": "a" * 40, "base": "main"}, monkeypatch)
    assert "Published by the gateway" in published
    assert "head=" + "a" * 12 in published and "base=main" in published

    refused = _render({"verdict": "refused", "base": "main", "detail": "stale base"}, monkeypatch)
    assert "REFUSED" in refused and "main" in refused and "stale base" in refused

    errored = _render({"verdict": "error", "base": "main", "detail": "guard exploded"}, monkeypatch)
    assert "could not complete" in errored and "guard exploded" in errored

    not_activated = _render(
        {"verdict": "not_activated", "base": "main", "detail": "not gated"}, monkeypatch
    )
    assert "not activated" in not_activated and "unrecognised" not in not_activated

    not_published = _render(
        {"verdict": "not_published", "base": "main", "detail": "head_moved"}, monkeypatch
    )
    assert "NOT PUBLISHED" in not_published and "head_moved" in not_published
    assert "unrecognised" not in not_published


def test_the_presenter_refuses_to_read_an_unknown_verdict_as_a_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verdict the presenter does not know is an error, never a pass."""
    out = _render({"verdict": "probably-fine"}, monkeypatch)
    assert "unrecognised verdict" in out


def test_the_presenter_reports_a_transport_failure_as_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport failure is INDETERMINATE, not a definite failure: the gateway may still
    be mid-publish, so the presenter must not render it as a rejection the agent retries."""
    out = _render({"error": "read timed out", "transport_error": True}, monkeypatch)
    assert "INDETERMINATE" in out and "read timed out" in out


def test_the_presenter_promotes_a_not_published_body_from_a_non_2xx_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``not_published`` body arrives flattened into ``error`` at a 409/500 status; the
    presenter recovers the structured verdict rather than rendering a raw JSON blob."""
    import json as _json

    flattened = {
        "error": _json.dumps({"verdict": "not_published", "base": "main", "detail": "head_moved"})
    }
    out = _render(flattened, monkeypatch)
    assert "NOT PUBLISHED" in out and "head_moved" in out


# ── Restored: tests exercising the SURVIVING gateway handler + activation ──
#
# Re-added after an over-eager subtraction removed coverage of kept code in
# src/kiro_crew/dashboard/handlers/push_verdict.py. Every symbol referenced below still
# exists on disk; no test of removed floor/predicate behaviour is restored.


def _route_source() -> str:
    from pathlib import Path

    package = Path(push_verdict.__file__).parent.parent
    return (package / "dashboard" / "handlers" / "push_verdict.py").read_text(encoding="utf-8")


def _hook_verdict(command: str, *, session_key: str) -> "str | None":
    """The hook's refusal reason for *command*, or ``None`` when it was allowed.

    Reads ``action`` rather than any attribute named ``deny``: ``ToolHookResult.deny`` is a
    CONSTRUCTOR, so an attribute check on that name is truthy for an allowed result too and
    would make this helper report every outcome as a denial.
    """
    from kiro_crew.hooks import TOOL_DENY, HookManager, HooksConfig

    result = HookManager(HooksConfig()).on_tool_call(
        command, command=command, is_shell=True, session_key=session_key
    )
    return result.reason if result.action == TOOL_DENY else None


def _drive_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audit_raises: bool,
    activated: bool = True,
    config_answers: "dict[str, str] | None" = None,
    push_url: str = "https://example.invalid/repo.git",
    pinned_push_url: str = "https://example.invalid/repo.git",
    publish_result: object = None,
    deletions: "list | None" = None,
    advertise_head: bool = True,
):
    """Call the real route handler with its I/O stubbed, returning ``(status, payload)``.

    Everything stubbed here is I/O the handler does not own: the loopback check, session
    recognition, the slot lookup, git reads, and the guard subprocess. What is NOT stubbed
    is the ordering under test.
    """
    import asyncio
    import json as _json

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from kiro_crew.dashboard.handlers import push_verdict as route

    class _Slot:
        project = "/some/worktree"

    class _State:
        def get_slot(self, _key: str) -> "_Slot":
            return _Slot()

    def _fake_sel():
        class _Sel:
            def log_api_access(self, **kwargs: object) -> None:
                # Mirrors the REAL helper rather than being a convenient stub. Without
                # `critical=True` that helper ENQUEUES the event and returns success even
                # when the write fails; only a critical write is synchronous and re-raises.
                # A stub that raised either way would hold no one to `critical=True` -- drop
                # the keyword and a sink that is down would look like a clean audit, so the
                # unaudited verdict this file exists to forbid would be recorded with every
                # test still green.
                if audit_raises and kwargs.get("critical") is True:
                    raise OSError("audit sink is down")

        return _Sel()

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if "--absolute-git-dir" in args:
            return 0, "/some/worktree/.git"
        if "symbolic-ref" in args:
            return 0, "feature-x"
        if args[:2] == ("config", "--get"):
            answers = config_answers or {}
            key = args[2] if len(args) > 2 else ""
            return (0, answers[key]) if key in answers else (1, "")
        if args[:2] == ("remote", "get-url"):
            # Fetch and push URL agree unless a test says otherwise.
            return 0, push_url if "--push" in args else "https://example.invalid/repo.git"
        return 0, "a" * 40

    async def _fake_guard(
        _worktree: str, _base: str, *, common_dir: str = "", digest: str = "", url: str = ""
    ) -> "route._GuardRun":
        # The mirror and refs ride along because the operation PUBLISHES from them: a pass with
        # no mirror is refused rather than published, so a stub that omitted them would be
        # testing the refusal instead of the ordering.
        return route._GuardRun(
            0,
            "STATUS: SAFE TO PUSH",
            "d" * 40,
            "e" * 40,
            Path("/some/mirror"),
            route._JudgementRefs.mint(),
        )

    async def _fake_publish(**_kwargs: object) -> "route._PublishResult":
        return publish_result or route._PublishResult(True, "published", "pushed")

    async def _fake_delete_refs(*args: object, **_kwargs: object) -> None:
        if deletions is not None:
            deletions.append(args)
        return None

    async def _fake_recognize(*_args: object, **_kwargs: object) -> None:
        return None

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        # The route resolves the default branch from the remote's ADVERTISED symbolic HEAD.
        # Advertise ``main`` so the base is read over the wire, never from the agent-writable
        # local ``refs/remotes/{remote}/HEAD``. With ``advertise_head=False`` no ``ref:`` line
        # is returned, driving the fail-closed ``no_base`` path.
        if "--symref" in argv:
            if advertise_head:
                return 0, "ref: refs/heads/main\tHEAD\n" + "a" * 40 + "\tHEAD"
            return 0, "a" * 40 + "\tHEAD"
        return 0, ""

    monkeypatch.setattr(route, "is_loopback", lambda _remote: True)
    monkeypatch.setattr(route, "_recognize_session", _fake_recognize)
    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)
    monkeypatch.setattr(route, "_run_guard", _fake_guard)
    monkeypatch.setattr(route, "_publish", _fake_publish)
    monkeypatch.setattr(route, "_delete_refs", _fake_delete_refs)
    monkeypatch.setattr(route, "sel", _fake_sel)
    # An activated installation with a digest pinned. The keystone's own behaviour has its
    # own tests; what these drive is the route's ordering, which never runs without one.
    monkeypatch.setattr(
        route.push_verdict,
        "activation",
        lambda: route.push_verdict.Activation(
            enabled=activated, guard_sha256="c" * 64, pinned_push_url=pinned_push_url
        ),
    )

    app = web.Application()
    app["state"] = _State()
    request = make_mocked_request(
        "POST", "/api/push-verdict/run", headers={"X-Session-Key": "session-1"}, app=app
    )
    request["internal_auth"] = True

    response = asyncio.run(route.api_push_verdict_run(request))
    return response.status, _json.loads(response.text)


def _publish_env(
    monkeypatch, *, head_now: str, remote_now: str = "origin", base_now: str = "e" * 40
):
    """Stub the reads ``_publish`` makes, returning the argv list it launches.

    ``_publish`` reads TWO remote tips via ``ls-remote``: the base branch's (confirming
    the candidate was judged against the tip that is live now) and the source branch's (the
    lease). The fake answers ``base_now`` for ``refs/heads/main`` and the source lease
    ``b*40`` for anything else, so a test can advance the base by passing a different tip.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if args[:1] == ("rev-parse",):
            return 0, head_now
        if args[:2] == ("config", "--get"):
            key = args[2] if len(args) > 2 else ""
            return (0, remote_now) if key.endswith("pushRemote") else (1, "")
        if args[:2] == ("remote", "get-url"):
            return 0, "https://example.invalid/repo.git"
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        if "ls-remote" in argv:
            # The last operand is the ref being read: the base branch (``refs/heads/main``) for
            # F3's base-tip re-check, or the source branch for the lease.
            ref = argv[-1]
            if ref == "refs/heads/main":
                return 0, base_now + "\trefs/heads/main"
            return 0, "b" * 40 + "\trefs/heads/feature-x"
        return 0, "pushed"

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)
    return launched


async def _resolve_target(
    monkeypatch, *, push_url: str, pinned_push_url: str, activated: bool = False
):
    """Drive the real ``_effective_push_target`` with git reads stubbed."""
    from kiro_crew.dashboard.handlers import push_verdict as route

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if args[:2] == ("config", "--get"):
            return 1, ""
        if args[:2] == ("remote", "get-url"):
            # Fetch and push agree; the test varies only the push URL vs the pin.
            return 0, push_url
        return 0, ""

    monkeypatch.setattr(route, "_git", _fake_git)
    return await route._effective_push_target(
        "/some/worktree", "feature-x", activated=activated, pinned_push_url=pinned_push_url
    )


def test_the_route_is_registered_and_classified_strict() -> None:
    """Registered by the shared registrar AND strict, asserted together.

    Together because either one alone is a false reassurance: a path in the frozenset that
    no server registers is dead, and a registered path missing from the frozenset falls
    through to cookie auth, which would let a browser request stand in for the agent's
    session. Registering through ``_register_mcp_routes`` is also what stops the two
    servers from drifting, which that frozenset's own comment calls an auth bypass.
    """
    from aiohttp import web

    from kiro_crew.dashboard import server

    app = web.Application()
    server._register_mcp_routes(app)
    registered = {str(route.resource.canonical) for route in app.router.routes()}
    assert "/api/push-verdict/run" in registered
    assert server.internal_path_matches("/api/push-verdict/run", server._STRICT_INTERNAL_API_PATHS)


def test_the_handler_reasserts_loopback_and_the_internal_secret() -> None:
    """Both re-asserts must be present in the handler itself.

    Being listed in the strict frozenset does not prove the secret was checked: with the
    header absent the middleware falls through to cookie auth, and a ``local_only=False``
    deployment reclassifies strict paths as mixed. This asserts the source carries both
    guards, which is the cheapest way to catch their removal.
    """
    from pathlib import Path

    package = Path(push_verdict.__file__).parent.parent
    handler = (package / "dashboard" / "handlers" / "push_verdict.py").read_text(encoding="utf-8")
    assert "is_loopback(request.remote" in handler
    assert 'request.get("internal_auth") is not True' in handler


def test_the_route_runs_the_packaged_guard_and_it_exists() -> None:
    """The guard the route invokes must actually be there.

    A relocated or renamed script would otherwise make every request an environment error,
    which denies every publish on an activated install -- silently, since no other test
    resolves this path.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    assert route._PUSH_GUARD.is_file()
    assert route._PUSH_GUARD.name == "push_guard.py"


def test_the_route_runs_the_packaged_copy_not_the_agent_writable_one() -> None:
    """The installed skills copy is agent-writable; the packaged one is not.

    Running the installed copy would hand the gated party the very check that authorizes
    it, which is the defect this whole design exists to remove, one layer further out.
    """
    from pathlib import Path

    import kiro_crew
    from kiro_crew.dashboard.handlers import push_verdict as route

    assert route._PUSH_GUARD.is_relative_to(Path(kiro_crew.__file__).parent / "builtin_skills")


def test_the_route_pins_the_base_rather_than_letting_the_guard_detect_it() -> None:
    """The recorded base must be the base that was judged, so the gateway names it."""
    assert '"--base"' in _route_source()


def test_the_route_reads_the_guards_own_exit_contract() -> None:
    """A cross-file contract nothing else pins.

    The route decides whether to record from the guard's exit code. If the guard's contract
    changed, the route would misread a refusal as a pass with no test failing anywhere, so
    this asserts the numbers against the guard script's own documented contract.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    contract = route._PUSH_GUARD.read_text(encoding="utf-8")
    # The guard documents its exit codes as (CONSTANT, meaning) rows in EXIT_CODES; the two
    # the route reads are SAFE=0 and REFUSED=40. Assert the guard still binds those numbers to
    # those meanings so the route's constants cannot silently drift from the guard's contract.
    assert "EXIT_SAFE = 0" in contract and "EXIT_REFUSED = 40" in contract
    assert '(EXIT_SAFE, "safe' in contract and '(EXIT_REFUSED, "refused' in contract
    assert route._GUARD_SAFE == 0
    assert route._GUARD_REFUSED == 40


def test_the_route_does_not_take_the_base_from_the_branchs_upstream() -> None:
    """Staleness measured against a branch's own remote branch passes by construction.

    The base has to be the repository's default branch, and it is read from the remote's
    ADVERTISED symbolic HEAD (``ls-remote --symref``) rather than the branch's upstream OR the
    agent-writable local ``refs/remotes/{remote}/HEAD`` file. This pins the resolution against
    a regression that looks harmless and makes the check unable to fail.
    """
    source = _route_source()
    assert "ls-remote" in source and "--symref" in source
    assert "HEAD@{upstream}" not in source
    # The route must NOT read the base from the agent-writable local remote-tracking HEAD.
    assert 'symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD"' not in source


def test_the_default_branch_probe_runs_with_the_mirror_git_dir() -> None:
    """GPT 6.1: the ``ls-remote --symref`` probe must not inherit the gateway cwd repository's
    config. ``git_env`` already denies the global/system scopes, but a repository-LOCAL
    ``url.<evil>.insteadOf`` in whatever repo the gateway's cwd happens to be rewrites the
    probe URL (the gateway may have started in repo A while the session worktree is B). The
    probe therefore runs with ``GIT_DIR`` pinned to the gateway-owned mirror, excluding the cwd
    repo config.

    Mutation check: removing the ``GIT_DIR``-on-the-mirror env from the probe lets it read the
    cwd ``.git/config`` again, so these assertions on the route source fail.
    """
    source = _route_source()
    # The probe builds an env off git_env() and pins GIT_DIR to the gateway-owned mirror.
    assert "_probe_mirror = _mirror_for(common_dir)" in source
    assert '_probe_env["GIT_DIR"] = str(_probe_mirror)' in source
    # And the probe call passes that env (not the default cwd-inheriting one).
    assert "env=_probe_env" in source


def test_the_mirror_lives_inside_the_sealed_directory(home: Path) -> None:
    """The mirror's PATH is what the seal protects, so it must land inside it.

    A mirror placed anywhere else is unprotected, and an unprotected mirror is one the agent can
    plant its own base commit into. The seal and the path have to agree or the seal guards an
    empty directory.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    mirror = route._mirror_for("/some/repo/.git")
    assert mirror.parent == home / push_verdict.MIRROR_DIR
    assert mirror.suffix == ".git"
    # Keyed on the COMMON git directory: two repositories never collide, one repository is
    # stable. The route resolves ``git rev-parse --git-common-dir``, which every linked
    # ``.worktrees/*`` checkout of one repository reports identically -- so all of them key the
    # SAME mirror, where the per-worktree ``--absolute-git-dir`` would have primed one each.
    assert route._mirror_for("/other/repo/.git") != mirror
    assert route._mirror_for("/some/repo/.git") == mirror
    # The shared common dir of a repo's two worktrees is one string, hence one mirror.
    common = "/some/repo/.git"
    assert route._mirror_for(common) == route._mirror_for(common)


def test_the_guard_is_invoked_in_out_of_place_mode() -> None:
    """The three flags that make the guard judge the mirror instead of the worktree.

    Without them the guard would judge `HEAD` and `origin/<base>` in a bare mirror that has
    neither, so the run would fail rather than judge the wrong thing -- but it would fail for a
    reason no test explains. These assert the mode the whole read-only design depends on.
    """
    source = _route_source()
    assert '"--no-fetch",' in source
    assert '"--base-ref",' in source
    assert '"--candidate-ref",' in source
    assert 'env["GIT_DIR"] = str(mirror)' in source


def test_the_candidate_is_read_out_of_the_worktree_never_into_it(home: Path) -> None:
    """The fetch DIRECTION is the property that keeps the worktree unwritten.

    A fetch into the worktree fails on its own `FETCH_HEAD` once that tree is read-only, and
    before it fails it would be writing to the state being judged. So the candidate is fetched
    out of the worktree into the mirror, and the only `-C <worktree>` uses are reads.
    """
    source = _route_source()
    assert 'f"+HEAD:{refs.candidate}"' in source
    # Every worktree-scoped git call is a read: no fetch, no write verb, carries -C.
    assert '"git", "-C", worktree, *args' in source
    assert '"-C", worktree, "fetch"' not in source


def test_executable_git_configuration_is_neutralized_for_the_gateways_git() -> None:
    """The worktree is agent-writable and this git runs at gateway privilege.

    Without this, an agent writing ``core.sshCommand`` into its own ``.git/config`` has the
    gateway run that command for it, outside the agent's sandbox, just by asking for a
    verdict. Environment configuration rather than ``-c`` because it must also govern the git
    processes the guard script starts for itself.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    env = route.git_env()
    count = int(env["GIT_CONFIG_COUNT"])
    keys = {env[f"GIT_CONFIG_KEY_{index}"] for index in range(count)}
    for executable_directive in (
        "core.sshCommand",
        "core.fsmonitor",
        "core.gitProxy",
        "core.askPass",
        "core.hooksPath",
        "core.alternateRefsCommand",
        "credential.helper",
        "diff.external",
        "uploadpack.packObjectsHook",
    ):
        assert executable_directive in keys
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    # Codex F2: core.sshCommand must ignore the (agent-writable) user/system ssh config, so a
    # planted ~/.ssh/config Host block cannot run a program or key on the gateway publish.
    values = {
        env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"] for index in range(count)
    }
    assert values["core.sshCommand"] == "ssh -F none"


def test_every_git_the_gateway_launches_uses_that_environment() -> None:
    """One spawn site, and the neutralized environment is its DEFAULT.

    Asserted as the default rather than per call site: a caller that forgets to pass an
    environment gets the neutralized one, so a new git call cannot arrive unprotected by
    omission. The guard run overrides it only to add ``GIT_DIR``, on top of the same base.
    """
    source = _route_source()
    assert "env=env or git_env()" in source
    assert "env=scrubbed" in source
    assert 'env["GIT_DIR"]' in source


def test_the_gateways_spawns_stay_routed_through_the_sandbox_chokepoint() -> None:
    """Both spawns go through the chokepoint, with limits, and no bare spawn remains.

    `test/test_spawn_audit.py` is the repository-wide tripwire for this. This one fails in the
    file that owns the behaviour, so a reader who breaks it sees WHY here: the command runs
    against an agent-writable repository, and git reads executable directives out of that
    repository's own configuration, so the child has to be confined even when it does something
    the agent chose.
    """
    import ast

    source = _route_source()
    tree = ast.parse(source)
    adapter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_prepare_sandboxed_spawn"
    )
    # Read the SYNTAX, not the text: the import line and this adapter's own docstring both
    # carry these names, so counting occurrences is satisfied while the real call is replaced
    # by something that confines nothing. A mutation doing exactly that survived a text check.
    # `test/test_spawn_audit.py` reads the tree for the same reason.
    called = {
        node.func.id
        for node in ast.walk(adapter)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    referenced = {node.id for node in ast.walk(adapter) if isinstance(node, ast.Name)}
    assert "shielded_prepare_off_loop" in called
    assert "sandboxed_spawn_argv" in referenced
    # Exactly ONE spawn site in the module, which is the point rather than an accident: the
    # repository's spawn audit reads the enclosing function, so every additional site is
    # another place the routing can be forgotten. Every git this module runs goes through it.
    assert source.count("create_subprocess_limited(") == 1
    assert "asyncio.create_subprocess_exec" not in source
    # The mode is the difference between a confined child and an unconfined one, and "off" is
    # a one-word edit away from here. Routing through the chokepoint in "off" mode would
    # satisfy the repository's spawn audit while confining nothing.
    assert 'mode="standard"' in source
    assert 'mode="off"' not in source


def test_a_verdict_that_could_not_be_audited_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance clause, as behaviour: no audit record, no recorded pass.

    A pass whose only trace failed to be written is an unaudited pass, which is the class
    this design was told to remove. The audit therefore goes FIRST and its failure refuses
    the request, so the store is left exactly as it was.
    """
    status, payload = _drive_route(monkeypatch, audit_raises=True)
    assert status == 500
    assert payload["code"] == "audit_failed"


def test_an_audited_guard_pass_publishes_and_leaves_no_reusable_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: an audited guard pass publishes.

    The gateway judges and publishes inline, so a ``published`` verdict with a 200 is the whole
    observable result -- no receipt is persisted for a later command to spend.
    """
    status, payload = _drive_route(monkeypatch, audit_raises=False)
    assert status == 200
    assert payload["verdict"] == "published"


def test_an_unreadable_activation_refuses_the_publish_rather_than_allowing_it(
    home: Path,
) -> None:
    """Fail-closed, driven through the real hook.

    The store raising is only half the property; this is the half a caller sees. An
    activation leaf that cannot be parsed leaves the operator's intent UNKNOWN, and allowing
    the publish there would make damaging one file equivalent to turning the gate off -- an
    off switch reachable without authorization. So the publish is refused, and the refusal
    names the file an operator has to repair.
    """
    leaf = home / push_verdict.ACTIVATION_LEAF
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.write_text("{not json", encoding="utf-8")

    reason = _hook_verdict("git " + "push origin feature-x", session_key="session-1")
    assert reason is not None
    assert push_verdict.ACTIVATION_LEAF in reason


def test_the_guard_the_gateway_runs_is_verified_against_the_keystone_pin(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editing the packaged guard must not change what the gateway executes.

    In an editable or source install the packaged guard sits in the repository the agent
    edits, so executing it there would hand the gated party the script that authorizes it:
    print ``SAFE TO PUSH`` unconditionally and every push is recorded as checked.

    A copy taken on first use did not answer this, and that is why the pin exists. First use
    made TIMING the authority, so an edit landing before any verdict was ever requested became
    the authorized version permanently. The digest in the keystone is the authority instead,
    and it is the one statement about these bytes written outside the tree they live in.
    """
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)
    pinned = hashlib.sha256(packaged.read_bytes()).hexdigest()

    snapshot = route._guard_snapshot(pinned)
    # Inside the sealed directory, which is the whole reason the agent cannot reach it.
    assert push_verdict.MIRROR_DIR in snapshot.parts
    assert snapshot.read_bytes() == packaged.read_bytes()

    # The agent's edit: make the guard pass unconditionally. It does not match the pin, so
    # nothing runs at all -- not the edit, and not a stale copy passed off as authorized.
    packaged.write_bytes(b"import sys\nprint('STATUS: SAFE TO PUSH')\nsys.exit(0)\n")
    with pytest.raises(route._GuardUntrusted):
        route._guard_snapshot(pinned)

    # And an installation with nothing pinned runs no guard, which is what stops "activated"
    # from quietly meaning "runs whatever is in the tree".
    with pytest.raises(route._GuardUntrusted):
        route._guard_snapshot("")


def test_the_runner_executes_the_snapshot_and_not_the_packaged_path(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The snapshot must be what the RUNNER launches, not merely what a helper can produce.

    Mutation testing is why this test exists: pointing the invocation back at the packaged
    path left every other test in this file green, because the one that covers the snapshot
    calls the helper directly and the route-level test stubs the runner out. Neither can see
    which path is actually handed to the spawn, so this one reads the argv.
    """
    import asyncio
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_text("print('STATUS: SAFE TO PUSH')\n", encoding="utf-8")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *_args: str) -> tuple[int, str]:
        return 0, "https://example.invalid/repo.git"

    async def _fake_prime(*_args: object, **_kwargs: object) -> tuple[int, str]:
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        return 0, "STATUS: SAFE TO PUSH"

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_prime_mirror", _fake_prime)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)

    asyncio.run(
        route._run_guard(
            "/some/worktree",
            "main",
            common_dir="/some/worktree/.git",
            digest=hashlib.sha256(packaged.read_bytes()).hexdigest(),
            url="https://example.invalid/repo.git",
        )
    )

    assert launched, "the runner launched nothing"
    # The runner launches several commands -- it resolves the judged pair and cleans its refs
    # up -- so the guard is picked out by name rather than assumed to be the first.
    guard_argv = [argv for argv in launched if any(a.endswith("push_guard.py") for a in argv)]
    assert len(guard_argv) == 1, "the guard was not launched exactly once"
    argv = guard_argv[0]
    assert str(packaged) not in argv, "the runner executed the agent-writable packaged guard"
    guard_arg = next(a for a in argv if a.endswith("push_guard.py"))
    assert push_verdict.MIRROR_DIR in Path(guard_arg).parts


def test_each_judgement_gets_its_own_refs() -> None:
    """Fixed ref names in a shared mirror let two requests overwrite each other's candidate.

    One mirror serves one REPOSITORY, so two sessions judging two of its branches at once wrote
    the same two refs: the second fetch replaced the first's candidate and the guard then
    measured one session's branch and answered for the other's.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    first = route._JudgementRefs.mint()
    second = route._JudgementRefs.mint()
    assert first.base != second.base
    assert first.candidate != second.candidate
    assert first.base != first.candidate
    # Namespaced so they cannot collide with refs a repository already has.
    assert first.base.startswith("refs/kirocrew/")


def test_the_runner_hands_its_refs_to_the_operation_that_publishes(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refs outlive the JUDGING, because the push source is the candidate ref itself.

    Ownership sits with whoever finishes the operation: the runner hands the refs over and the
    caller removes them in its ``finally``, which the companion test drives. A runner that
    deleted its own refs would destroy the one thing the push needs, since the operation pushes
    ``candidate:refs/heads/<ref>``.
    """
    import asyncio
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *_args: str) -> tuple[int, str]:
        return 0, "https://example.invalid/repo.git"

    async def _fake_prime(*_args: object, **_kwargs: object) -> tuple[int, str]:
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        return 0, "f" * 40

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_prime_mirror", _fake_prime)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)

    run = asyncio.run(
        route._run_guard(
            "/some/worktree",
            "main",
            common_dir="/some/worktree/.git",
            digest=hashlib.sha256(packaged.read_bytes()).hexdigest(),
            url="https://example.invalid/repo.git",
        )
    )

    deleted = {argv[-1] for argv in launched if "update-ref" in argv and "-d" in argv}
    assert not deleted, "the runner deleted refs the publish still needs"
    assert run.refs is not None and run.mirror is not None, "the runner kept its refs to itself"


def test_a_publish_that_refuses_still_leaves_no_reusable_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure paths are the ones that matter for reusable authority.

    A publish refused because HEAD moved must not leave a pass behind: the next command would
    spend a receipt describing a commit the guard examined but the tree does not hold. Released
    in a ``finally``, so this holds on every way out of the operation rather than on the one the
    happy path takes.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        publish_result=route._PublishResult(False, "head_moved", "HEAD moved from d... to f..."),
    )
    assert status == 409, "a tree that moved under a judgement is a conflict, not a server error"
    assert payload["verdict"] == "not_published"
    assert payload["code"] == "head_moved"


def test_the_publish_pushes_the_judged_commit_under_a_lease_the_gateway_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two properties that close the post-snapshot gap, read off the real argv.

    The SOURCE is the candidate ref -- the one holding the commit the guard examined -- so what
    lands is what was judged however the worktree has moved since. And the lease's expected
    value is the tip THIS gateway read from the remote, never a value a caller supplied: a lease
    against an agent-supplied sha would let the agent describe a remote state that never existed,
    which is the same defect as a receipt the agent writes.
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    launched = _publish_env(monkeypatch, head_now="d" * 40)
    refs = route._JudgementRefs.mint()
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=refs,
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert result.ok and result.code == "published"

    push = next(argv for argv in launched if "push" in argv)
    assert (
        f"{refs.candidate}:refs/heads/feature-x" in push
    ), "the push source was not the judged ref"
    assert f"--force-with-lease=refs/heads/feature-x:{'b' * 40}" in push
    # The base ref is NOT leased: git enforces --force-with-lease only for a ref the push
    # updates, and this push writes the feature ref alone, so a base lease would be ignored.
    # The base is gated by the base_now re-read in _publish, never pushed.
    assert not any(arg.endswith(":refs/heads/main") for arg in push)
    assert not any("refs/heads/main:" in arg for arg in push)


def test_a_head_that_moved_after_the_guard_refuses_to_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finding's own case: a mutation the gateway never observed, then a publish."""
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    _publish_env(monkeypatch, head_now="f" * 40)
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert not result.ok and result.code == "head_moved"


def test_a_destination_that_moved_after_the_guard_refuses_to_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destination half: a ``pushRemote`` written after the judging sends it elsewhere."""
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    _publish_env(monkeypatch, head_now="d" * 40, remote_now="fork")
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert not result.ok and result.code == "target_moved"


def test_a_base_that_advanced_after_the_guard_refuses_to_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base branch tip is re-read at publish time and must still be what was judged.

    Staleness is a comparison against the base, so a base that advances after the guard ran
    means the candidate was judged stale against a tip that is gone. Publishing then
    would land a commit whose freshness was decided against a gone base -- the same window as
    HEAD or the destination moving, on the base side of the comparison.
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    # HEAD and destination are unchanged, but the base tip the gateway reads now differs from
    # the ``base_sha`` the guard measured against.
    _publish_env(monkeypatch, head_now="d" * 40, base_now="9" * 40)
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert not result.ok and result.code == "base_moved"


def test_a_detached_head_refuses_to_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty source ref (detached HEAD) has no branch to gate a verdict to.

    Refused before any push and before any remote read, because there is no ref the floor
    could hold the verdict to and nothing to publish.
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    _publish_env(monkeypatch, head_now="d" * 40)
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert not result.ok and result.code == "detached_head"


@pytest.mark.parametrize(
    # wokeignore:rule=master
    "source_ref",
    ["main", "mainline", "master", "develop-base"],  # wokeignore:rule=master
)
def test_a_protected_or_base_source_ref_refuses_to_publish(
    monkeypatch: pytest.MonkeyPatch, source_ref: str
) -> None:
    """This gate publishes a FEATURE branch, never a protected branch or the base itself.

    ``main``/``mainline``/``master`` are the argv floor's own protected set, and a source ref  # wokeignore:rule=master
    that equals the base being judged against (here ``develop-base``) would push the candidate
    onto the very branch staleness is measured from. Both are refused before any push, so the
    gateway cannot become the way around the floor's protected-push refusal.
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    # For the non-protected case, the base is set equal to the source ref so the ``== base``
    # arm of the guard is what refuses.
    base = source_ref if source_ref == "develop-base" else "main"
    _publish_env(monkeypatch, head_now="d" * 40)
    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref=source_ref,
            base=base,
            base_sha="e" * 40,
            target=route._PushTarget("origin", "https://example.invalid/repo.git"),
        )
    )
    assert not result.ok and result.code == "protected_source_ref"


def test_the_route_resolves_the_base_from_the_advertised_symbolic_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default branch comes from the remote's advertised HEAD, read over the wire.

    The route's ``_run_git`` mock advertises ``ref: refs/heads/main`` for ``ls-remote
    --symref``; nothing answers a local ``refs/remotes/{remote}/HEAD`` read. A base of ``main``
    therefore proves the resolution took the advertised path, which the agent cannot forge, and
    not the agent-writable local remote-tracking ref.
    """
    status, payload = _drive_route(monkeypatch, audit_raises=False)
    assert status == 200
    assert payload["base"] == "main"


def test_the_route_has_no_base_when_the_remote_advertises_no_symbolic_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2, fail-closed: a remote that advertises no symbolic HEAD gives no base to judge.

    The route refuses with ``no_base`` rather than falling back to a local ref an agent could
    have written.
    """
    status, payload = _drive_route(monkeypatch, audit_raises=False, advertise_head=False)
    assert status == 400
    assert payload["code"] == "no_base"


def test_a_guard_run_that_raises_still_removes_its_refs(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refs go to the caller on every ORDINARY exit, and are cleaned up on the others.

    Moving ownership to the caller left one path the caller never reaches: an exception inside
    the runner, where there is no return value to hand anything over with. Without this the
    mirror keeps a pair per crash, and nothing said so -- the mutation that removes the cleanup
    survived a whole suite until this test existed.
    """
    import asyncio
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)

    deleted: list[object] = []

    async def _fake_git(_worktree: str, *_args: str) -> tuple[int, str]:
        return 0, "https://example.invalid/repo.git"

    async def _boom(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise RuntimeError("the mirror went away mid-judgement")

    async def _record_delete(_mirror: object, refs: object) -> None:
        deleted.append(refs)

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_prime_mirror", _boom)
    monkeypatch.setattr(route, "_delete_refs", _record_delete)

    with pytest.raises(RuntimeError):
        asyncio.run(
            route._run_guard(
                "/some/worktree",
                "main",
                common_dir="/some/worktree/.git",
                digest=hashlib.sha256(packaged.read_bytes()).hexdigest(),
                url="https://example.invalid/repo.git",
            )
        )
    assert len(deleted) == 1, "a crashed judgement left its refs in the long-lived mirror"


def test_the_operation_removes_the_refs_when_it_finishes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the ownership move: nothing accumulates in the long-lived mirror."""

    removed: list = []
    status, _payload = _drive_route(monkeypatch, audit_raises=False, deletions=removed)
    assert status == 200
    assert len(removed) == 1, "the operation did not remove the judgement's refs"


def test_an_unpinned_installation_is_told_to_pin_rather_than_told_it_mismatched(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both refusals stop the run; only one of them is actionable.

    Falling through to the comparison would refuse with "these are not the bytes an operator
    authorized", which sends an operator looking for a changed file when nothing was ever
    pinned.
    """
    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)

    with pytest.raises(route._GuardUntrusted) as raised:
        route._guard_snapshot("")
    assert "no guard digest is pinned" in str(raised.value)
    assert push_verdict.ACTIVATION_LEAF in str(raised.value)


def test_a_snapshot_that_disagrees_with_the_pin_is_replaced(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale copy must not be executed just because it is the copy that exists.

    Refreshing is safe HERE and only here: the bytes were verified against the pin first, so
    what replaces the copy is authorized rather than merely newer.
    """
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)
    pinned = hashlib.sha256(packaged.read_bytes()).hexdigest()

    stale = route.data_home() / push_verdict.MIRROR_DIR / "push_guard.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"import sys\nsys.exit(0)\n")

    assert route._guard_snapshot(pinned).read_bytes() == packaged.read_bytes()


def test_two_judgements_do_not_share_refs(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner has to mint PER REQUEST, not merely be capable of minting.

    A test that calls the minting helper twice proves the helper varies. It says nothing about
    whether the runner calls it, which is where two concurrent judgements of one repository
    overwrote each other's candidate.
    """
    import asyncio
    import hashlib

    from kiro_crew.dashboard.handlers import push_verdict as route

    packaged = tmp_path / "push_guard.py"
    packaged.write_bytes(b"print('STATUS: SAFE TO PUSH')\n")
    monkeypatch.setattr(route, "_PUSH_GUARD", packaged)
    digest = hashlib.sha256(packaged.read_bytes()).hexdigest()

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *_args: str) -> tuple[int, str]:
        return 0, "https://example.invalid/repo.git"

    async def _fake_prime(*_args: object, **_kwargs: object) -> tuple[int, str]:
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        return 0, "f" * 40

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_prime_mirror", _fake_prime)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)

    for _ in range(2):
        asyncio.run(
            route._run_guard(
                "/some/worktree",
                "main",
                common_dir="/some/worktree/.git",
                digest=digest,
                url="https://example.invalid/repo.git",
            )
        )

    used = {
        argv[argv.index("--candidate-ref") + 1] for argv in launched if "--candidate-ref" in argv
    }
    assert len(used) == 2, "both judgements pointed the guard at the same candidate ref"


def test_an_unactivated_installation_banks_no_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """With gating off the floor never reads a verdict, so recording one banks a pass nobody
    asked for -- and it would be spent the moment an operator did activate."""
    from kiro_crew.dashboard.handlers import push_verdict as route

    status, payload = _drive_route(monkeypatch, audit_raises=False, activated=False)
    assert status == 200
    assert payload["verdict"] == "not_activated"
    assert route is not None


def test_a_non_activated_install_makes_no_credentialed_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX1 (F1, BLOCKING): the activation gate precedes ALL credentialed remote work.

    A never-activated (default) install is not gated at all, so it must return
    ``not_activated`` WITHOUT resolving an agent-writable push destination and WITHOUT the
    credentialed ``git ls-remote --symref`` probe against it. Before the fix the route
    resolved the destination and ran the probe FIRST and only then bailed as ``not_activated``,
    so every default install did credentialed remote work against a destination read from
    agent-writable worktree config.

    Mutation check: ``_effective_push_target`` and ``_run_git`` are stubbed to RAISE if called.
    On the un-fixed ordering (resolve + probe before the not_activated exit) the route invokes
    them and this test fails with ``AssertionError('...called on the non-activated path')``.
    On the fixed ordering the exit returns before either runs, so neither raises.
    """
    import asyncio
    import json as _json

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from kiro_crew.dashboard.handlers import push_verdict as route

    class _Slot:
        project = "/some/worktree"

    class _State:
        def get_slot(self, _key: str) -> "_Slot":
            return _Slot()

    def _fake_sel():
        class _Sel:
            def log_api_access(self, **_kwargs: object) -> None:
                return None

        return _Sel()

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        # The non-remote input validation the route keeps ABOVE the activation gate: the
        # git-dir probe and the source-ref read. Neither is a credentialed remote call.
        if "--absolute-git-dir" in args:
            return 0, "/some/worktree/.git"
        if "symbolic-ref" in args:
            return 0, "feature-x"
        return 0, "a" * 40

    async def _fake_recognize(*_args: object, **_kwargs: object) -> None:
        return None

    async def _boom_target(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "_effective_push_target was resolved on the non-activated path -- an agent-"
            "writable destination must not be read before the activation gate"
        )

    async def _boom_run_git(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise AssertionError(
            "_run_git (the ls-remote --symref probe) was called on the non-activated path -- "
            "no credentialed remote call may precede the not_activated exit"
        )

    monkeypatch.setattr(route, "is_loopback", lambda _remote: True)
    monkeypatch.setattr(route, "_recognize_session", _fake_recognize)
    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_effective_push_target", _boom_target)
    monkeypatch.setattr(route, "_run_git", _boom_run_git)
    monkeypatch.setattr(route, "sel", _fake_sel)
    monkeypatch.setattr(
        route.push_verdict,
        "activation",
        lambda: route.push_verdict.Activation(
            enabled=False, guard_sha256="c" * 64, pinned_push_url=""
        ),
    )

    app = web.Application()
    app["state"] = _State()
    request = make_mocked_request(
        "POST", "/api/push-verdict/run", headers={"X-Session-Key": "session-1"}, app=app
    )
    request["internal_auth"] = True

    response = asyncio.run(route.api_push_verdict_run(request))
    payload = _json.loads(response.text)
    assert response.status == 200
    assert payload["verdict"] == "not_activated"
    # The response shape is unchanged (the three keys the reader consumes); the base is empty
    # because it is never resolved on this path, and the reader tolerates an empty base.
    assert set(payload) == {"verdict", "base", "detail"}
    assert payload["base"] == ""


def test_a_remote_that_pushes_elsewhere_is_refused_rather_than_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway judges what it can READ; a pushurl sends the commit somewhere else.

    Answering would describe the wrong repository under an accepted remote name, so the
    request is refused and says why.
    """
    status, payload = _drive_route(
        monkeypatch, audit_raises=False, push_url="https://elsewhere.invalid/other.git"
    )
    assert status == 400
    assert payload["code"] == "push_url_differs"


def test_the_recorded_remote_is_the_one_a_publish_actually_reaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assuming ``origin`` was a gap: a bare ``git push`` names no remote to compare.

    With ``branch.<name>.pushRemote`` set, the commit landed in a repository the guard never
    looked at while the floor had nothing to hold the command to.
    """
    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        config_answers={"branch.feature-x.pushRemote": "fork"},
    )
    assert status == 200
    # And the base comes from the remote's advertised default branch (its symbolic HEAD),
    # not the branch's own upstream.
    assert payload["base"] == "main"


def test_the_resolver_refuses_a_destination_that_is_not_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finding's own case: the worktree config points the publish at another repository.

    Without the pin check this returns a resolved target and the publish lands wherever the
    agent-writable config says. FAILS on the un-fixed resolver, which had no pin parameter.
    """
    import asyncio

    target = asyncio.run(
        _resolve_target(monkeypatch, push_url=_url("attacker"), pinned_push_url=_url("canonical"))
    )
    assert target.code == "unpinned_destination"
    assert not target.url


def test_the_resolver_refuses_a_url_with_an_unparseable_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex F2: a malformed port must be refused, not crash the push request.

    ``urlsplit(...).port`` raises ValueError on a non-numeric/out-of-range port, and both the
    credential-free compare and the pin resolution read ``.port``, so an agent-writable config
    value like ``https://h.invalid:notaport/r.git`` would crash the request. The resolver now
    refuses ``unsafe_remote_url`` at its single return instead. FAILS on pre-fix code with an
    unhandled ValueError.
    """
    import asyncio

    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url="https://h.invalid:notaport/r.git",
            pinned_push_url="https://h.invalid/r.git",
            activated=True,
        )
    )
    assert target.code == "unsafe_remote_url"
    assert not target.url


def test_the_resolver_refuses_a_symlinked_local_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Codex F1: a local destination reached through a symlink is refused, not followed.

    The push URL comes from agent-writable worktree config and the worktree is agent-writable, so
    a planted symlink could repoint the pinned destination to a repository the operator never
    authorized. The resolver refuses with ``unsafe_remote_url`` at its single return, covering
    both the pre-judgement resolve and ``_publish``'s re-resolve immediately before pushing.
    """
    import asyncio
    import sys

    import pytest

    from kiro_crew.dashboard.handlers import push_verdict as route

    # POSIX-only: the threat is a POSIX agent-planted symlink and Windows CI symlink
    # privilege / realpath normalization is unreliable. The production check is cross-platform.
    if sys.platform == "win32":
        pytest.skip("symlinked-destination refusal is POSIX-only in test (Windows realpath)")
    wt = tmp_path / "wt"
    wt.mkdir()
    real = tmp_path / "real.git"
    real.mkdir()
    try:
        (wt / "link.git").symlink_to(real)  # agent-planted link inside the worktree
    except (OSError, NotImplementedError):
        pytest.skip("platform cannot create a symlink for this test")

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if args[:2] == ("config", "--get"):
            return 1, ""
        if args[:2] == ("remote", "get-url"):
            return 0, "link.git"  # a relative local remote that is a symlink
        return 0, ""

    monkeypatch.setattr(route, "_git", _fake_git)
    target = asyncio.run(
        route._effective_push_target(str(wt), "feature-x", activated=False, pinned_push_url="")
    )
    assert target.code == "unsafe_remote_url"
    assert not target.url


def test_the_resolver_allows_the_pinned_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction: a resolved URL that matches the pin resolves cleanly."""
    import asyncio

    target = asyncio.run(
        _resolve_target(monkeypatch, push_url=_url("canonical"), pinned_push_url=_url("canonical"))
    )
    assert target.code == ""
    assert target.url == _url("canonical")


def test_the_resolver_is_unconstrained_when_nothing_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pin the resolver behaves exactly as before -- any resolvable URL resolves."""
    import asyncio

    target = asyncio.run(
        _resolve_target(monkeypatch, push_url=_url("whatever"), pinned_push_url="")
    )
    assert target.code == ""
    assert target.url == _url("whatever")


def test_the_compare_strips_a_credential_from_the_resolved_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On an ACTIVATED install a resolved push URL that carries a token is REFUSED, even when it
    names the pinned repo (GPT 6.1 finding 4).

    The token stays readable in the agent-writable worktree ``.git/config``, so an approved
    script could read the URL and publish with it directly, bypassing the gateway -- the pinned
    repo match does not close that, because the embedded credential names the SAME repository. On
    an activated install the gateway owns publish authentication, so a credential-bearing worktree
    remote fails closed with ``credentialed_remote_url``.

    Mutation check: dropping the activated credential-free-remote refusal lets this resolve to
    ``code == ""`` and the assertion below flips.
    """
    import asyncio

    resolved = "https://x-access-token:s3cr3t@git.example.invalid/canonical.git"
    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url=resolved,
            pinned_push_url="https://git.example.invalid/canonical.git",
            activated=True,
        )
    )
    assert target.code == "credentialed_remote_url"


def test_the_compare_still_mismatches_a_genuinely_different_repo_after_stripping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripping userinfo does not weaken the pin: a different host/path still mismatches.

    Both sides carry a credential, but they name DIFFERENT repositories, so the credential-free
    compare still refuses -- proving the strip is a credential normalization, not a blanket
    accept.
    """
    import asyncio

    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url="https://tok@git.example.invalid/attacker.git",
            pinned_push_url="https://git.example.invalid/canonical.git",
            activated=True,
        )
    )
    assert target.code == "unpinned_destination"
    assert not target.url


def test_the_compare_mismatches_two_distinct_ssh_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: a pin for one SSH account refuses a publish resolved to another (codex F2).

    The operator pinned ``ssh://deploy@host/repo``; the agent-writable config resolves to
    ``ssh://staging@host/repo``. On the pre-fix helper both stripped to ``ssh://host/repo`` and
    compared EQUAL, so the gateway would publish through the unauthorized account. The fix keeps
    the username, so the compare refuses with ``unpinned_destination``.
    """
    import asyncio

    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url="ssh://staging@git.example.invalid/repo.git",
            pinned_push_url="ssh://deploy@git.example.invalid/repo.git",
            activated=True,
        )
    )
    assert target.code == "unpinned_destination"
    assert not target.url


def test_the_compare_mismatches_two_distinct_ipv6_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the real pin compare: distinct IPv6 destinations are refused.

    The resolved push URL and the pin name IPv6 destinations that differ only across the
    host/port boundary. On the un-fixed helper they collapsed to one credential-free identity
    and this returned a clean target (``code == ""``); the fix keeps them distinct so the
    compare refuses with ``unpinned_destination`` -- FAILS on pre-fix code.
    """
    import asyncio

    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url="https://[2001:db8::1:443]/r.git",
            pinned_push_url="https://[2001:db8::1]:443/r.git",
            activated=True,
        )
    )
    assert target.code == "unpinned_destination"
    assert not target.url


def test_the_compare_allows_the_same_ipv6_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pin still MATCHES the same IPv6 destination, so a legitimate publish is not refused."""
    import asyncio

    resolved = "https://[2001:db8::1]:443/r.git"
    target = asyncio.run(
        _resolve_target(
            monkeypatch,
            push_url=resolved,
            pinned_push_url="https://[2001:db8::1]:443/r.git",
            activated=True,
        )
    )
    assert target.code == ""
    assert target.url == resolved


def test_the_route_refuses_an_activated_publish_to_an_unpinned_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: activated + pinned, worktree config points elsewhere -> refused, nothing banked.

    The route reads the pin from the keystone BEFORE resolving the destination, so the
    agent-chosen destination is caught on the first resolution, not only on the publish-time
    re-resolve. No verdict is recorded, because the request never reaches the guard.
    """
    # Fetch and push agree (both the default URL) so this is the PIN mismatch, not
    # ``push_url_differs``: the resolved destination is fine in itself, it is just not the one
    # the operator pinned.
    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        push_url="https://example.invalid/repo.git",
        pinned_push_url=_url("canonical"),
    )
    assert status == 400
    assert payload["code"] == "unpinned_destination"


def test_the_route_publishes_to_the_pinned_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    """The companion: when the resolved destination IS the pin, the operation proceeds."""
    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        push_url="https://example.invalid/repo.git",
        pinned_push_url="https://example.invalid/repo.git",
    )
    assert status == 200
    assert payload["verdict"] == "published"


def test_the_route_refuses_an_activated_publish_when_no_destination_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: activated install, NO pin -> refused ``destination_not_pinned``, nothing banked.

    On a gated install the push destination is read from the agent-writable worktree config, so
    leaving it unconstrained is the fail-OPEN this finding closes: the credential-exempt gateway
    publish would land the judged commit at any agent-chosen same-history repository. Activation
    now REQUIRES the operator to pin a destination before publishing at all. The request never
    reaches the guard, so no verdict is recorded.

    Mutation check: on un-fixed code (no activated-no-pin refusal) the resolver returns a usable
    target, the guard runs, and the route publishes -> status 200, not the 400 asserted here.
    """
    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        activated=True,
        pinned_push_url="",
    )
    assert status == 400
    assert payload["code"] == "destination_not_pinned"


def test_a_non_activated_install_with_no_pin_still_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requirement follows ACTIVATION: a non-activated install is not gated, so no pin is
    demanded and it returns ``not_activated`` exactly as before. Guards the fail-closed against
    over-reaching onto installs that never asked to be gated."""
    status, payload = _drive_route(
        monkeypatch,
        audit_raises=False,
        activated=False,
        pinned_push_url="",
    )
    assert status == 200
    assert payload["verdict"] == "not_activated"


def test_the_resolver_fails_closed_on_activated_with_no_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unit form of the route test: the single resolver return is the enforcement point."""
    import asyncio

    target = asyncio.run(
        _resolve_target(monkeypatch, push_url=_url("agent"), pinned_push_url="", activated=True)
    )
    assert target.code == "destination_not_pinned"
    assert target.url == ""
    # And the same reads with the install NOT activated resolve normally (no pin demanded).
    target2 = asyncio.run(
        _resolve_target(monkeypatch, push_url=_url("agent"), pinned_push_url="", activated=False)
    )
    assert target2.code == "" and target2.url == _url("agent")


def test_the_publish_re_resolution_also_fails_closed_on_activated_with_no_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_publish`` re-resolves and applies the activated-no-pin latch there too, so a pin that
    is dropped (or never set) between judging and publish is caught on the re-resolve as well as
    the pre-judgement resolve. FAILS on the un-fixed ``_publish`` (no ``activated`` parameter,
    no latch), which would launch the push.
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if args[:1] == ("rev-parse",):
            return 0, "d" * 40
        if args[:2] == ("config", "--get"):
            return 1, ""
        if args[:2] == ("remote", "get-url"):
            return 0, _url("agent")
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        return 0, "pushed"

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)

    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", _url("agent")),
            activated=True,
            pinned_push_url="",
        )
    )
    assert not result.ok and result.code == "destination_not_pinned"
    assert not any("push" in argv for argv in launched)


def test_the_publish_re_resolution_also_enforces_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_publish`` re-resolves the destination and applies the pin there too.

    A ``pushurl`` / ``pushRemote`` written AFTER the judging is the destination half of the
    same class, and the re-resolve is where it is caught. Here the re-resolved push URL is not
    the pin, so the publish refuses with ``unpinned_destination`` rather than sending the judged
    commit to an unauthorized repository. FAILS on the un-fixed ``_publish`` (no pin parameter).
    """
    import asyncio

    from kiro_crew.dashboard.handlers import push_verdict as route

    launched: list[list[str]] = []

    async def _fake_git(_worktree: str, *args: str) -> tuple[int, str]:
        if args[:1] == ("rev-parse",):
            return 0, "d" * 40
        if args[:2] == ("config", "--get"):
            return 1, ""
        if args[:2] == ("remote", "get-url"):
            # The worktree now resolves to the ATTACKER URL, not the pin.
            return 0, _url("attacker")
        return 0, ""

    async def _fake_run_git(argv: list[str], **_kwargs: object) -> tuple[int, str]:
        launched.append(list(argv))
        return 0, "pushed"

    monkeypatch.setattr(route, "_git", _fake_git)
    monkeypatch.setattr(route, "_run_git", _fake_run_git)

    result = asyncio.run(
        route._publish(
            worktree="/some/worktree",
            mirror=Path("/some/mirror"),
            refs=route._JudgementRefs.mint(),
            head="d" * 40,
            source_ref="feature-x",
            base="main",
            base_sha="e" * 40,
            target=route._PushTarget("origin", _url("canonical")),
            pinned_push_url=_url("canonical"),
        )
    )
    assert not result.ok and result.code == "unpinned_destination"
    # Refused before any push argv was launched.
    assert not any("push" in argv for argv in launched)


def test_resolve_for_command_reads_nothing_on_a_non_publish_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First Principles Item 6: the activation keystone is read ONLY for a git-publish command.

    A non-publish tool call must resolve to ``None`` without touching the keystone -- that is what
    makes a never-publishing install pay no keystone read. Mutation check: a helper that resolves
    unconditionally (the pre-fix shape) calls ``activation_enabled`` here and the counter is 1.
    """
    import asyncio

    from kiro_crew.security import resolve_push_verdict_activation_for_command

    reads = {"n": 0}

    def _counting_enabled() -> bool:
        reads["n"] += 1
        return True

    monkeypatch.setattr(push_verdict, "activation_enabled", _counting_enabled)

    result = asyncio.run(resolve_push_verdict_activation_for_command("git status", "List files"))
    assert result is None, "a non-publish command resolves to None"
    assert reads["n"] == 0, "a non-publish command must not read the activation keystone"


def test_resolve_for_command_resolves_off_loop_for_a_publish_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A git-publish command resolves the keystone (so ``is_denied`` gets a pre-resolved value and
    does no on-loop read). The verb-anchored check matches ``git push`` in either the command or
    the title."""
    import asyncio

    from kiro_crew.security import resolve_push_verdict_activation_for_command

    monkeypatch.setattr(push_verdict, "activation_enabled", lambda: True)

    result = asyncio.run(
        resolve_push_verdict_activation_for_command("git push origin feature-x", "")
    )
    assert result is not None and result.enabled is True


def test_resolve_for_command_matches_a_nested_shell_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT 6.1 F1: a NESTED-shell publish (``bash -c 'git push ...'``) must resolve activation
    off-loop too. A shallow top-level check would miss it, return None, and let ``is_denied``
    discover the nested push and read the keystone synchronously on the loop. The helper uses the
    floor's own ``_shell_payload_sources`` walk, so it resolves for exactly what the gate reads.
    """
    import asyncio

    from kiro_crew.security import resolve_push_verdict_activation_for_command

    monkeypatch.setattr(push_verdict, "activation_enabled", lambda: True)

    result = asyncio.run(
        resolve_push_verdict_activation_for_command("bash -c 'git push origin feature-x'", "")
    )
    assert (
        result is not None and result.enabled is True
    ), "a nested-shell git push must be detected and resolved off-loop"


def test_pinned_push_url_malformed_url_becomes_activation_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT 6.1 F2 (fenced): a malformed pin must raise ``ActivationUnreadable``, not an uncaught
    ``ValueError`` that terminates the chat turn. ``https://[::1/repo.git`` (an unterminated IPv6
    literal) makes ``urlsplit`` raise ``ValueError``; the resolver catches only
    ``ActivationUnreadable``, so the parser error must be converted to it.
    """
    with pytest.raises(push_verdict.ActivationUnreadable):
        push_verdict._pinned_push_url("https://[::1/repo.git")


def test_worktree_remote_has_secret_flags_only_a_real_secret() -> None:
    """Opus FINDING: a credential-bearing worktree remote is flagged by an ACTUAL secret, not by
    any userinfo a URL normalizer would change. Default clone URLs that carry a bare username but
    no secret must NOT be flagged (they are legitimate on an activated install)."""
    from kiro_crew.dashboard.handlers import push_verdict as handler

    # Real secrets -> flagged.
    assert handler._worktree_remote_has_secret("https://x-access-token:s3cr3t@h/r.git") is True
    assert handler._worktree_remote_has_secret("git://u:p@h/r.git") is True
    assert handler._worktree_remote_has_secret("user:secret@host:path/r.git") is True  # scp form
    assert handler._worktree_remote_has_secret("ssh://u:p@h/r.git") is True

    # Bare username / no secret / normalization-only differences -> NOT flagged.
    assert handler._worktree_remote_has_secret("https://user@bitbucket.org/x/r.git") is False
    assert handler._worktree_remote_has_secret("https://org@dev.azure.com/x/r.git") is False
    assert handler._worktree_remote_has_secret("https://GitHub.com/x/r.git") is False
    assert handler._worktree_remote_has_secret("git+ssh://git@host/x/r.git") is False
    assert handler._worktree_remote_has_secret("git@github.com:x/r.git") is False  # scp identity
    assert handler._worktree_remote_has_secret("https://github.com/x/r.git") is False
