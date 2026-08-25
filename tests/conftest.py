"""Shared pytest fixtures.

Keeps the whole suite hermetic with respect to the features added around the
CLI's side-effects: no network for the update check, no nag from the MCP hint,
and the global ``.env`` / caches redirected into a tmp dir so tests never read
or pollute the user's real ``~/.config/mitos`` or ``~/.cache/mitos``. Tests that
exercise those features re-enable them explicitly (``monkeypatch.delenv(...)``).
"""

import os

import live_helpers

import pytest


# --- Phase 5a store-rebuild quarantine (the contained-red window) --------------
#
# Phase 5a flips the live schema (entry-001) + identity (entry-002) and rebuilds
# ``commit_parsed_entry`` over the V1a STRICT schema. That flip breaks the five
# live consumers (``sync``/``importer``/``mcp``/``cli``/``renderer``) and the
# prototype read methods **at runtime** — they bind prototype column names
# (``core_axiom``, inline ``scope``/``mechanisms``, ``edges.from_id/to_id/type``,
# the ``pending_embeddings`` drain surface) that no longer exist. This is the
# vision's *contained-red window*, not a regression to chase: the read views are
# restored in Phase 5d and the consumers reconciled in Phase 8a.
#
# To keep the substrate gate (test_identity / test_parser / test_migrations /
# test_config / test_packaging + 5a's rewritten test_store) meaningfully green
# through the 5a→8a window, the broken consumer/read test modules are quarantined
# here — a SINGLE tracked list (Decision 5) skipped via the collection hook below,
# NOT scattered per-file ``pytestmark`` skips and NOT a red CI. The list was
# derived **empirically** (flip → run the full suite → quarantine exactly the
# modules that failed *because of the flip*, not the pre-existing ``*_live.py``
# 429 flakes). Each later phase REMOVES the modules it restores;
# ``test_store_rebuild_quarantine_is_tracked`` (in tests/test_store.py) pins the
# current set so the shrink to empty is auditable.
#
# Phase 5d re-bucketed 5a's empirical labels (WIRING_LEDGER entry-003, §16): of the
# 8 modules 5a labelled "restored in 5d", only **2** were genuinely store-only
# (``test_renderer`` + ``test_adversarial_rendering`` — removed below as 5d
# restored them). The other 6 were mis-bucketed: ``test_status_readiness`` is 8×
# ``cli.cmd_status`` (gated on the **6b** cmd_status rebuild), and the remaining 5
# drive the sync consumer write path (``record_decision_entry``), the MCP/CLI
# surfaces, and/or ``amends``/``narrows`` edges (unrepresentable until V1b) — all
# **8a**'s charter. The store-level modifier (T12) + C4 (T5) proofs those would
# have given are delivered in ``tests/test_store.py`` instead.
#
# Phase 6b restored ``test_status_readiness`` (the ``cmd_status`` rebuild it gated
# on landed; the prototype-shape ``ParsedEntry`` fixture was reworked to V1a),
# leaving 12 modules — all 8a's.
#
# Phase 8a DRAINED the quarantine to EMPTY (entry-003 closed): the five live
# consumers (sync/importer/mcp_server/cli) were reconciled to the V1a substrate
# (parse_entry_stream + compute_node_id + get_node_state + the V1a drain surface),
# ``--corrects`` was wired, and every restored module was re-greened — V1b-
# unrepresentable assertions (amends/narrows modifiers, OQ parked/resolved) pared
# or deferred with a logged note (OD1; never silent-skip/coerce). The list reaching
# 0 is the contained-red window closing.
STORE_REBUILD_QUARANTINE: list[str] = []


def pytest_collection_modifyitems(config, items):
    """Skips the store-rebuild quarantine modules during the 5a→8a contained-red window.

    Applies a single skip marker to every collected item whose test module is in
    ``STORE_REBUILD_QUARANTINE`` (Phase 5a, Decision 5). The reason names the
    restoring phases so the deferral is legible in the test report; the list
    provably empties by Phase 8a.
    """
    reason = (
        "Phase 5a contained-red window: consumer methods break at runtime against "
        "the flipped V1a schema (the read views were restored in Phase 5d); "
        "restored in Phase 6b (cmd_status) / Phase 8a (consumers)."
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.path.name in STORE_REBUILD_QUARANTINE:
            item.add_marker(skip_marker)


def make_workspace(root) -> str:
    """Builds the minimal valid workspace shape and returns its canonical path.

    The shipped validity triple and nothing more: ``.mitos/`` holding a
    ``config.toml``, plus ``decisions.md``. A half-workspace is not a workspace,
    and building only the first two parts is the fixture mistake made from habit —
    it costs nothing before 5a (the handler is mocked and never sees the directory)
    and refuses to resolve after it. Deliberately no graph: a workspace is valid
    without one, which is the cloned-but-unbuilt state the escape hatch exists for.

    Lifted here at 5a from ``test_cli_selector``/``test_routing``'s byte-identical
    twins, because the flip gave it a dozen more consumers: every row that used to
    hand a mocked handler a bare ``tmp_path`` now needs a real workspace, and
    thirteen private re-spellings would be thirteen chances to write the
    half-workspace one.

    **Absolute-path form, always.** The canonical path is what a migrated row passes
    to ``-p``; ``-p .`` would resolve *pytest's* working directory — the mitos-pub
    repo, which is itself a valid workspace — so a row written that way is green
    both here and under a build that still resolved the cwd, which is the exact
    defect 5a removes.

    Args:
        root: The directory to build the workspace at (created if absent).

    Returns:
        The workspace root's canonical (``realpath``) absolute path.
    """
    os.makedirs(os.path.join(str(root), ".mitos"), exist_ok=True)
    with open(os.path.join(str(root), ".mitos", "config.toml"), "w") as f:
        f.write("# a mitos workspace\n")
    with open(os.path.join(str(root), "decisions.md"), "w") as f:
        f.write("# Decisions\n")
    return os.path.realpath(str(root))


def resolve_like_main(root: str):
    """The config ``cli.main()`` builds for a path-form selector.

    A verb never receives a hand-built config in production — ``main()`` resolves
    the selector and passes what it got, so the config carries the resolved
    *name*. The echo is exactly the field where that matters: ``mitos init``
    registers the workspace under its directory basename, so a path-form selector
    reverse-looks-up to that registered name on both surfaces. Hand-building the
    config instead has the CLI echo a path while its MCP twin echoes the name, and
    a parity row then reds on the test's shortcut rather than on a drift.

    The name is a **random per-run temp basename** for the usual
    ``mkdtemp``-then-``cmd_init`` fixture, so a row asserting the echo must
    compare against this config's ``project``, never against a literal.

    Lifted here at 5b from ``test_scopes``/``test_modifier_surfacing``'s
    byte-identical twins, for the same reason 5a lifted :func:`make_workspace`:
    the MCP flip gave it eleven more candidate consumers, and eleven private
    re-spellings would be eleven chances to hand-build the config instead.

    Args:
        root: A workspace root — the path form of a selector.

    Returns:
        The :class:`~mitos.config.MitosConfig` for the resolved workspace,
        carrying the resolved project name.
    """
    # Imported in-body: conftest is loaded before every session, and the mitos
    # import graph has no business riding that for a helper most modules never
    # call.
    from mitos import routing
    from mitos.config import MitosConfig

    target = routing.resolve_project(root)
    return MitosConfig(target.root, project=target.name)


@pytest.fixture
def workspace(tmp_path) -> str:
    """One ready-made workspace under ``tmp_path``, canonical path.

    The fixture form of :func:`make_workspace`, for the common row that needs
    exactly one. Rows needing two (or needing the directory named) call the
    function.
    """
    return make_workspace(tmp_path / "ws")


#: The credential names ``hermetic_mitos_env`` strips from every non-live test.
#: Deliberately the three LLM keys and nothing wider: ``QDRANT_URL`` stays,
#: because tests target the local instance through it, and the modules that need
#: a broader strip (``RESOLVED_ENV_KEYS``, ``QDRANT_URL``) keep their own
#: ``_keyless`` fixtures on top of this baseline.
_CREDENTIAL_NAMES: tuple[str, ...] = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


#: The live tier spends the test-tier Anthropic key where one is supplied.
#:
#: Applied at conftest **import**, which is the only placement that reaches every
#: live module: five of them hand-roll their own repo-``.env`` loader and the rest
#: resolve credentials through ``mitos.env``, so a fixture — even an autouse,
#: session-scoped one — would land after the import-time ``HAS_LIVE_KEYS`` gates
#: have already read the environment. The substitution rides ``os.environ``, tier 1
#: of that resolver, so it masks both ``.env`` files for consumers of either kind.
#:
#: Deliberately NOT gated on the live brake: applying it under
#: ``MITOS_NO_LIVE_TESTS`` costs nothing (no call is made) and keeping the two
#: mechanisms independent means a brake change cannot silently re-point the gate
#: at the usage key.
_APPLIED_TEST_CREDENTIALS = live_helpers.apply_test_credentials(
    repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@pytest.fixture(autouse=True)
def hermetic_mitos_env(request, monkeypatch, tmp_path):
    """Isolates per-test config/cache and silences the CLI's network side-effects.

    Only one quiet-switch is left: the update check. ``MITOS_NO_MCP_HINT`` was the
    second, and it retired with the per-project MCP-wiring nudge it silenced — that
    nudge advised toward a project-scope ``.mcp.json`` entry, which now *shadows*
    the machine-wide registration rather than complementing it. There is no
    remaining stderr nag for a test to suppress.

    Non-live tests also start with the three LLM credential names stripped, so a
    keyed dev box proves what keyless CI proves. The fixture used to leave the
    credentials alone, and that gap held CI red for four days (2026-07-31→08-04):
    ``test_importer``'s LLM tests read the key off ``config.env`` after 5c, the
    dev box's real key leaked in through tier 1 and kept them green locally, and
    only the keyless CI runner saw the failure. The strip is per-test and
    reversible (monkeypatch), so the import-time ``.env`` loaders in the live-ish
    modules and any row that ``setenv``s its own value are unaffected — each test
    simply starts from the CI posture instead of the box's.

    ``LIVE_MODULES`` (below) are exempt: real calls with real keys are their
    entire point, and their floor machinery already accounts for the keyless
    state honestly. ``test_adversarial_invariants.py`` needs no exemption — its
    workspace fixture re-loads the repo ``.env`` into ``os.environ`` per test.
    """
    monkeypatch.setenv("MITOS_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    if request.node.path.name not in LIVE_MODULES:
        for name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def sweep_leaked_qdrant_collections():
    """After the whole suite, delete test-ONLY collections from the shared Qdrant.

    Tests that build a workspace from ``tempfile.mkdtemp()`` create per-run
    collections named ``mitos-tmp*`` (and the adversarial suite uses
    ``mitos_adversarial_*``); not all of them clean up, so without this they
    accumulate on the shared instance. Pattern-restricted so it can NEVER touch a
    real project collection (``mitos-cartolina`` etc.), and fully best-effort:
    if Qdrant is unreachable, it simply does nothing.
    """
    yield
    url = os.environ.get("QDRANT_URL", "http://localhost:7333")
    try:
        import requests

        resp = requests.get(f"{url}/collections", timeout=2)
        if not resp.ok:
            return
        names = [c["name"] for c in resp.json().get("result", {}).get("collections", [])]
    except Exception:
        return
    for name in names:
        if name.startswith("mitos-tmp") or name.startswith("mitos_adversarial"):
            try:
                requests.delete(f"{url}/collections/{name}", timeout=5)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# The live-tier coverage floor
# --------------------------------------------------------------------------- #
#
# The live suites degrade EVERY environmental fault to a loud skip (a judge
# timeout, an embed 429, an unreachable vector store) so live-red stays
# trustworthy. That is the right call for a single fault — but it means a run
# where half the tier silently did not execute is indistinguishable from a
# healthy one, and exits 0. Observed: two identical serial runs of the same code
# gave 10 and 9 skips; under `-n 5` the same module went 5 passed → 4 → 2, green
# every time.
#
# The floor closes that. It does NOT police *whether* the tier runs — being
# switched off (MITOS_NO_LIVE_TESTS), keyless (a fork, CI), or wholly unable to
# reach a service are all honest states in which nothing pretended to check.
# It polices the PARTIAL state: the tier demonstrably worked, and some of it
# still didn't run.

#: Marker every environmental-degradation skip reason carries (live_helpers.py and
#: the live modules). A legitimate always-skip — "baseline seeding is
#: explicit-only" — deliberately does not, which is what keeps it out of the count.
_ENV_SKIP_MARKER = "not a code defect"

#: Test modules that make real Anthropic/Gemini calls. Held in sync with the set
#: consulting ``live_helpers.live_tests_disabled`` by the meta-test in
#: tests/test_live_floor.py — a new live module that skips registration fails there,
#: rather than quietly falling outside the floor. Second consumer:
#: ``hermetic_mitos_env`` exempts these modules from its credential strip, so a
#: module missing from here also runs keyless — loudly, via its own env-skips.
LIVE_MODULES: tuple[str, ...] = (
    "test_conflict_eval_live.py",
    "test_retrieval_live.py",
    "test_check_hook_recipe.py",
    "test_integration_live.py",
    "test_pathologies_live.py",
    "test_scenarios_live.py",
    "test_collection_absence_live.py",
    "test_status_overview_live.py",
)


def live_floor_verdict(live_passed: int, live_env_skipped: int) -> str | None:
    """Judges whether the live tier ran completely enough to be believed.

    Args:
        live_passed: Live-tier tests that actually executed and passed.
        live_env_skipped: Live-tier tests that skipped for an environmental cause.

    Returns:
        A failure message when the tier ran only partially, else None. Zero passes
        means the tier was off or unavailable wholesale — honest, not a hole.
    """
    if live_passed and live_env_skipped:
        total = live_passed + live_env_skipped
        return (
            f"live-tier coverage hole: {live_env_skipped} of {total} live tests "
            f"degraded to environmental skips while {live_passed} ran — so this "
            f"run checked less than it appears to, and would otherwise have "
            f"exited 0.\n"
            f"  Re-run the live tier before trusting it as a pre-push gate.\n"
            f"  Common causes: the SONNET judge's 15s ceiling under load, a Gemini "
            f"429, or parallel workers sweeping each other's Qdrant collections "
            f"(do not run the live tier under pytest-xdist).\n"
            f"  Set MITOS_STRICT_LIVE=1 to make this state a hard failure."
        )
    return None


def _is_live(nodeid: str) -> bool:
    """Reports whether a test nodeid belongs to a live-tier module."""
    return any(nodeid.split("::")[0].endswith(m) for m in LIVE_MODULES)


def pytest_sessionfinish(session, exitstatus):
    """Fails a session whose live tier ran only partially (see the floor note above)."""
    # xdist: only the controller aggregates, and only it owns the exit status.
    if hasattr(session.config, "workerinput"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    passed = sum(1 for r in reporter.stats.get("passed", []) if _is_live(r.nodeid))
    env_skipped = 0
    for r in reporter.stats.get("skipped", []):
        if not _is_live(r.nodeid):
            continue
        reason = str(getattr(r, "longrepr", "") or "")
        if _ENV_SKIP_MARKER in reason.lower():
            env_skipped += 1

    verdict = live_floor_verdict(passed, env_skipped)
    if not verdict:
        return
    # Reports by default, fails only on request. The bug this closes is
    # INVISIBILITY, not the skipping itself: a transient judge timeout is genuinely
    # environmental, and it fired on 3 of 3 serial runs the day this was written —
    # so a hard failure here would red almost every pre-push run and train the
    # bypass it exists to prevent. Loud-and-green keeps the state legible; strict
    # mode is for a release gate that must not proceed on a partial check.
    strict = bool(os.environ.get("MITOS_STRICT_LIVE"))
    reporter.write_line(f"\n[live-floor] {verdict}", red=strict, bold=True, yellow=not strict)
    if strict:
        session.exitstatus = 1
