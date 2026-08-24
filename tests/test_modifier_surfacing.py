"""Tests for reverse-relation modifier surfacing (the "amended axioms read as live" fix).

Driven by loop-Claude's most-repeated AX friction: an `amends`/`narrows` leaves its
target `active` with the ORIGINAL axiom text, and no retrieval surface signalled that a
later decision had moved on from it — so a fresh agent cited a superseded mechanism
(worst case, a relocated architecture) with full confidence. The graph always knew (the
edges exist); only the payload lied.

These pin the fix end-to-end: `GraphStore.get_modifiers[_map]` reads the incoming
modifying edges, and every read surface (MCP surface/query/list, the CLI twins,
`mitos show`, and the rendered `live_axioms.md`) stamps `superseded_by` / `amended_by` /
`narrowed_by` / `corrected_by` so the reader knows which decision to chase.

Forced fully offline (unreachable Qdrant + no keys) so the behaviour is deterministic and
never depends on running services.
"""

import json
import os
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from conftest import resolve_like_main
from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, cmd_open_questions, cmd_query, cmd_show, cmd_surface
from mitos.parser import ParsedEntry
from mitos.recall import assess_query_recall
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS
from mitos.sync import MitosSyncManager
from mitos.renderer import render_node_markdown, MitosRenderer


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode.

    `realpath` rather than the raw mkdtemp path: the `show_node` parity row names
    this workspace in path form, and resolution canonicalizes — so on a platform
    whose temp root is a symlink the two surfaces would stamp different
    `workspace` strings and parity would red on the environment.
    """
    tmp = os.path.realpath(tempfile.mkdtemp())
    config = MitosConfig(tmp)
    cmd_init(config)
    config = resolve_like_main(tmp)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _rec(m: MitosSyncManager, slug: str, scope=None, **relations) -> dict:
    res = m.record_decision_entry(
        axiom=f"Axiom for {slug}.",
        rejected_paths=f"Rejected alternative for {slug}.",
        scope=scope or [],
        slug=slug,
        **relations,
    )
    assert "error" not in res, res
    return res


# --------------------------------------------------------------------------- #
# store.get_modifiers[_map] — the source of truth
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("relation,reverse_key", [
    ("supersedes", "superseded_by"),
    ("corrects", "corrected_by"),
])
def test_get_modifiers_one_per_relation(ws, relation, reverse_key) -> None:
    """Each V1a KILL-edge surfaces under its reverse key on the (now-retired) TARGET.

    8a pared the prototype's amends/narrows cases (V1b warn-deferred) and added the
    corrects case (now authorable through the write path, K4): both V1a kill-edges
    retire the target, and the target's payload carries who moved on from it.
    """
    config, m = ws
    target = _rec(m, "target")
    _rec(m, "modifier", **{relation: "target"})
    store = GraphStore(config.db_path)
    assert store.get_modifiers(target["id"]) == {reverse_key: ["modifier"]}


def test_get_modifiers_corrects_edge_via_write_path(ws) -> None:
    """`corrects` maps to `corrected_by` and the corrected (inactive) node carries it.

    8a rewired this off the prototype's raw `INSERT INTO edges (from_id, to_id, type)`
    (dropped columns) onto the authorable `--corrects` write path (K4). The corrected
    target leaves the active view; reading it by id still stamps corrected_by (the
    store modifier seam stamps inactive nodes too — get_node, not the active surfaces).
    """
    config, m = ws
    target = _rec(m, "target")
    _rec(m, "corrector", corrects="target")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(target["id"]) == {"corrected_by": ["corrector"]}
    # The store read surface (get_node by id) stamps corrected_by on the inactive node.
    node = store.get_node(target["id"])
    assert node["corrected_by"] == ["corrector"]
    assert store.get_node_state(target["id"]) == "corrected"


def test_get_modifiers_only_on_target_not_modifier(ws) -> None:
    """The modifying decision itself is unmodified — the edge points one way."""
    config, m = ws
    _rec(m, "target")
    modifier = _rec(m, "modifier", amends="target")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(modifier["id"]) == {}


def test_get_modifiers_unmodified_is_empty(ws) -> None:
    """A solo decision returns {} — the healthy, common case stays uncluttered."""
    config, m = ws
    solo = _rec(m, "solo")
    assert GraphStore(config.db_path).get_modifiers(solo["id"]) == {}


def test_get_modifiers_accumulates_multiple_edges(ws) -> None:
    """A hub node accumulates edges: amended by one decision AND narrowed by another.

    The multi-edge case loop-Claude flagged — a long-lived hub ADR collects modifiers,
    and ALL of them must surface, not just the latest.
    """
    config, m = ws
    hub = _rec(m, "hub")
    _rec(m, "amender", amends="hub")
    _rec(m, "narrower", narrows="hub")
    mods = GraphStore(config.db_path).get_modifiers(hub["id"])
    assert mods == {"amended_by": ["amender"], "narrowed_by": ["narrower"]}


def test_get_modifiers_two_of_same_relation_sorted(ws) -> None:
    """Two decisions amending one node both appear, deterministically ordered."""
    config, m = ws
    hub = _rec(m, "hub")
    _rec(m, "zeta-amend", amends="hub")
    _rec(m, "alpha-amend", amends="hub")
    mods = GraphStore(config.db_path).get_modifiers(hub["id"])
    assert mods == {"amended_by": ["alpha-amend", "zeta-amend"]}  # slug-sorted


def test_get_modifiers_map_batches(ws) -> None:
    """The batch map keys each modified node and omits unmodified ones (V1a kill-edges).

    8a pared the prototype's amends case to the V1a corrects kill-edge: a is corrected,
    b is superseded, solo is untouched — the map keys only the two modified (now-retired)
    nodes.
    """
    config, m = ws
    a = _rec(m, "a")
    b = _rec(m, "b")
    solo = _rec(m, "solo")
    _rec(m, "a-mod", corrects="a")
    _rec(m, "b-mod", supersedes="b")
    store = GraphStore(config.db_path)
    mp = store.get_modifiers_map([a["id"], b["id"], solo["id"]])
    assert mp == {a["id"]: {"corrected_by": ["a-mod"]},
                  b["id"]: {"superseded_by": ["b-mod"]}}
    assert solo["id"] not in mp


def test_get_modifiers_map_empty_input(ws) -> None:
    """No ids → empty map, no query."""
    config, _ = ws
    assert GraphStore(config.db_path).get_modifiers_map([]) == {}


# --------------------------------------------------------------------------- #
# MCP read surfaces
# --------------------------------------------------------------------------- #

def test_mcp_query_exact_slug_surfaces_amended_by(ws) -> None:
    """THE headline case: an exact-slug read of an amended decision reads `active`
    but now also carries `amended_by` so the stale axiom can't masquerade as live."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "boundary-rule")
    _rec(m, "boundary-rule-v2", amends="boundary-rule")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("boundary-rule", project=config.workspace_dir))
    assert resp["state"] == "active"           # amends does NOT retire the parent
    assert resp["amended_by"] == ["boundary-rule-v2"]


def test_mcp_query_exact_slug_unmodified_has_no_modifier_keys(ws) -> None:
    """An unmodified decision's payload is unchanged (no spurious modifier keys)."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "clean")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("clean", project=config.workspace_dir))
    assert not any(k in resp for k in MODIFIER_EDGE_KEYS.values())


def test_mcp_list_decisions_surfaces_modifiers(ws) -> None:
    """list_decisions stamps modifier slugs onto the affected decision only."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "amended-one", scope=["z"])
    _rec(m, "amender", scope=["z"], amends="amended-one")
    _rec(m, "untouched", scope=["z"])
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="z", project=config.workspace_dir))
    by_slug = {d["slug"]: d for d in resp["decisions"]}
    assert by_slug["amended-one"]["amended_by"] == ["amender"]
    assert not any(k in by_slug["untouched"] for k in MODIFIER_EDGE_KEYS.values())


def test_mcp_surface_scope_fallback_surfaces_modifiers(ws) -> None:
    """The offline scope-fallback path on surface_decisions also stamps modifiers."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "fb-target", scope=["db"])
    _rec(m, "fb-amender", scope=["db"], amends="fb-target")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions(query="anything", scope="db",
            project=config.workspace_dir))
    target = next(d for d in resp["active_decisions"] if d["slug"] == "fb-target")
    assert target["amended_by"] == ["fb-amender"]


class _FakeEmbed:
    """Stand-in embedding provider — returns a constant vector, never calls the API."""

    def get_embedding(self, text: str, is_query: bool = False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    """Stand-in vector store — replays a fixed ranked match list.

    Records the ``limit`` it was last called with (so a test can prove ``--limit``
    threads through un-truncated) and slices its match list to ``[:limit]`` (so a
    test can prove a lowered ``--limit`` trims and a raised one does not no-op)."""

    def __init__(self, matches):
        self._matches = matches
        self.last_limit = None

    def query(self, vector, limit: int = 5):
        self.last_limit = limit
        return self._matches[:limit]


def test_mcp_surface_semantic_path_surfaces_modifiers(ws) -> None:
    """The SEMANTIC ranking loop (the AX loop's primary path) stamps modifiers too —
    not just the offline exact/scope-fallback paths. Driven by a fake vector store so
    it is deterministic without Qdrant/keys."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "sem-target", scope=["x"])
    _rec(m, "sem-amender", scope=["x"], amends="sem-target")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "sem-target", "score": 0.91}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        resp = json.loads(mcp_server.surface_decisions(query="how do we normalize rows",
            project=config.workspace_dir))
    target = next(d for d in resp["active_decisions"] if d["slug"] == "sem-target")
    assert target["amended_by"] == ["sem-amender"]


def test_mcp_query_semantic_path_surfaces_modifiers(ws) -> None:
    """query_decisions' semantic branch (claim, not slug) stamps modifiers on matches."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    _rec(m, "q-amender", scope=["x"], amends="q-target")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "q-target", "score": 0.88}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        # A spaced claim resolves to no exact slug → falls through to semantic search.
        resp = json.loads(mcp_server.query_decisions("a claim that is not any slug",
            project=config.workspace_dir))
    match = next(mt for mt in resp["matches"] if mt["slug"] == "q-target")
    assert match["amended_by"] == ["q-amender"]


@pytest.mark.skip(reason="V1a: query_decisions' exact-slug lookup is active-scoped "
                         "(get_node_by_slug returns <=1 ACTIVE node, MI-13), so a superseded "
                         "node is not reachable through this consumer surface. The kill-edge "
                         "modifier (superseded_by) on an inactive node surfaces via the store's "
                         "get_node(by id) — proven in tests/test_store.py (T12) and "
                         "test_get_modifiers_corrects_edge_via_write_path above. Deferred (K5/G4).")
def test_mcp_query_exact_slug_superseded_carries_superseded_by(ws) -> None:
    """A superseded node read by exact slug reports BOTH state and which decision replaced it."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "v1")
    _rec(m, "v2", supersedes="v1")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("v1", project=config.workspace_dir))
    assert resp["state"] == "superseded"
    assert resp["superseded_by"] == ["v2"]


def test_modifiers_survive_brief_trim(ws) -> None:
    """The staleness flag is ALWAYS attached — even on a brief payload where
    rejected_paths is trimmed. A lightweight scan that lost the heavy field still needs
    to know the axiom has been moved on from."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "hot")
    _rec(m, "hot-v2", amends="hot")
    store = GraphStore(config.db_path, read_only=True)
    node = store.get_node_by_slug("hot")

    brief = mcp_server._decision_payload(node, 1.0, brief=True, store=store)
    assert "rejected_paths" not in brief and brief["amended_by"] == ["hot-v2"]

    full = mcp_server._decision_payload(node, 1.0, brief=False, store=store)
    assert full["rejected_paths"] and full["amended_by"] == ["hot-v2"]


# --------------------------------------------------------------------------- #
# CLI read surfaces
# --------------------------------------------------------------------------- #

def test_cli_show_prints_modifier(ws, capsys) -> None:
    """`mitos show` on an amended decision prints the 'Amended by' annotation."""
    config, m = ws
    _rec(m, "shown")
    _rec(m, "shown-v2", amends="shown")
    capsys.readouterr()
    cmd_show(config, "shown")
    out = capsys.readouterr().out
    assert "Amended by" in out and "shown-v2" in out


def test_cli_show_unmodified_prints_the_negative_line(ws, capsys) -> None:
    """A node nothing has moved on from says so, rather than rendering nothing.

    The AX item: on the text surface "checked, clean" and "the stamp did not render"
    were the same screen, on the one verb whose job is telling a reader whether the
    axiom above is still the last word. `get_modifiers` returns only present keys, so
    absence is unambiguous to the caller and invisible to the reader.
    """
    config, m = ws
    _rec(m, "untouched")
    capsys.readouterr()
    cmd_show(config, "untouched")
    out = capsys.readouterr().out
    assert "Modified by:" in out and "none" in out
    # The negative line stands in for the whole family, so none of the four may
    # appear beside it — a stamp rendered next to "none" is the defect inverted.
    for label in ("Amended by", "Narrowed by", "Corrected by", "Superseded by"):
        assert label not in out


def test_cli_show_modified_omits_the_negative_line(ws, capsys) -> None:
    """The negative line is mutually exclusive with any real stamp — never both."""
    config, m = ws
    _rec(m, "moved")
    _rec(m, "moved-v2", amends="moved")
    capsys.readouterr()
    cmd_show(config, "moved")
    out = capsys.readouterr().out
    assert "Amended by" in out and "moved-v2" in out
    assert "Modified by:" not in out


def test_cli_show_negative_line_is_text_only_json_still_omits(ws, capsys) -> None:
    """The text fix does not leak into the payload.

    `show --json` shares `display.show_payload` structurally with the `show_node` MCP
    tool, so nulling absent modifier keys there would be a cross-surface payload shape
    change — out of scope for a text render item. Absent-key stays the JSON answer.
    """
    config, m = ws
    _rec(m, "clean")
    capsys.readouterr()
    cmd_show(config, "clean", as_json=True)
    out = json.loads(capsys.readouterr().out)
    for key in MODIFIER_EDGE_KEYS.values():
        assert key not in out
    assert "Modified by" not in json.dumps(out)


def test_cli_show_oq_unmodified_prints_the_negative_line(ws, capsys) -> None:
    """The line is kind-agnostic: an unmodified open question gets it too."""
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq(store, "still-open")
    capsys.readouterr()
    cmd_show(config, "still-open")
    out = capsys.readouterr().out
    assert "Modified by:" in out and "none" in out


def test_cli_show_resolves_superseded_not_reused_slug(ws, capsys) -> None:
    """The R2 trap: a superseded slug with NO active bearer (the superseder carries a
    DISTINCT slug) resolves marked-superseded instead of 404-ing — `show` as a vector."""
    config, m = ws
    _rec(m, "orig")
    _rec(m, "orig-v2", supersedes="orig")  # distinct slug → "orig" has no active bearer
    capsys.readouterr()
    cmd_show(config, "orig")
    out = capsys.readouterr().out
    assert "not found" not in out.lower()
    assert "superseded" in out.lower()
    assert "orig-v2" in out


def test_cli_show_active_slug_resolves_active(ws, capsys) -> None:
    """Active-view precedence (step 2): a slug borne by an active node resolves to the
    live node before the superseded-lineage recency step is ever consulted. (The write
    path's slug-collision guard means a slug maps to exactly one graph node, so this is
    the constructible form of active-view precedence.)"""
    config, m = ws
    _rec(m, "alive")
    capsys.readouterr()
    cmd_show(config, "alive", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "active"
    assert out["slug"] == "alive"
    assert "superseded_by" not in out


def test_cli_show_json_found_decision_letter_complete_stamped(ws, capsys) -> None:
    """`show --json` on a superseded decision: Letter-complete (axiom + rejected_paths),
    carries kind/id/state, and is modifier-stamped with `superseded_by`."""
    config, m = ws
    _rec(m, "auth")
    _rec(m, "auth-v2", supersedes="auth")
    capsys.readouterr()
    cmd_show(config, "auth", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "decision"
    assert out["id"] and out["slug"] == "auth"
    assert out["state"] == "superseded"
    assert out["axiom"]  # Letter core
    assert out["rejected_paths"]  # the anti-knowledge fence (M5)
    assert out["superseded_by"] == ["auth-v2"]  # the load-bearing stamp


def test_cli_show_json_oq_body_and_modifier_subset(ws, capsys) -> None:
    """`show --json` on an OQ emits the OQ body (topic/questions_raised/park_reason) and
    only the OQ-applicable modifier keys (the subset is structural, not filtered)."""
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq(store, "rate-policy")
    _commit_oq(store, "rate-policy-v2", amends="rate-policy")
    capsys.readouterr()
    cmd_show(config, "rate-policy", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "open_question"
    assert out["topic"] == "rate-policy"  # _oq_payload keys topic off slug
    assert out["questions_raised"]
    assert "park_reason" in out
    assert out["amended_by"] == ["rate-policy-v2"]
    assert "superseded_by" not in out and "corrected_by" not in out


def test_cli_show_json_not_found_is_object_with_hint(ws, capsys) -> None:
    """`show --json` on a genuinely-absent slug emits a JSON object (falsy found-flag +
    `mitos sync` hint), not a bare text line.

    6c gave the hint a **selector**: since the flip a bare `mitos sync` has no target
    and hard-fails, so printing one would hand a stuck reader a second wall. The
    value is the caller's own vocabulary (`config.project`), which is why this is
    asserted against the config rather than against a literal recipe.
    """
    config, m = ws
    capsys.readouterr()
    cmd_show(config, "no-such-slug", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["found"] is False
    assert out["ident"] == "no-such-slug"
    assert "mitos sync" in out["hint"].lower()
    assert f"-p {config.project!r}" in out["hint"]


# --------------------------------------------------------------------------- #
# show_node — the MCP twin of `mitos show` (5b). Same shared resolve_handle seam
# + shared show_payload builder, so resolution AND payload shape are structural,
# not test-enforced. T10 (the three resolution assertions on the MCP half) +
# T11 (the modifier stamps, notably superseded_by on a superseded dereference)
# + the CLI⇄MCP parity assertion (dict-equal over decision / OQ / not-found).
# --------------------------------------------------------------------------- #

def test_mcp_show_node_resolves_superseded_not_reused_slug(ws) -> None:
    """T10 MCP half + T11 sharpest stamp: a superseded slug with NO active bearer
    (distinct superseder slug) resolves marked-superseded and carries `superseded_by`
    — NOT the not-found object. The graveyard the active-view slug branch can't reach."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "dead-v1")
    _rec(m, "dead-v2", supersedes="dead-v1")  # distinct slug → "dead-v1" has no active bearer
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.show_node("dead-v1", project=config.workspace_dir))
    assert resp.get("found") is not False           # not the not-found object
    assert resp["kind"] == "decision"
    assert resp["slug"] == "dead-v1"
    assert resp["state"] == "superseded"
    assert resp["axiom"] and resp["rejected_paths"]  # Letter-complete (M5 fence)
    assert resp["superseded_by"] == ["dead-v2"]      # the load-bearing stamp


def test_mcp_show_node_active_slug_resolves_active(ws) -> None:
    """T10: active-view precedence — a slug borne by an active node resolves to the live
    node (the constructible form; the write-path slug-collision guard maps a slug to one
    node, so a two-nodes-one-slug active case is un-constructible)."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "alive-node")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.show_node("alive-node", project=config.workspace_dir))
    assert resp["state"] == "active"
    assert resp["slug"] == "alive-node"
    assert "superseded_by" not in resp


def test_mcp_show_node_by_id_resolves(ws) -> None:
    """T10 (id path): a content-hash id resolves regardless of state (the id branch of
    resolve_handle, preserved on the twin)."""
    from mitos import mcp_server
    config, m = ws
    res = _rec(m, "by-id")
    node_id = res["id"]
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.show_node(node_id, project=config.workspace_dir))
    assert resp["id"] == node_id
    assert resp["slug"] == "by-id"


def test_mcp_show_node_oq_body_and_modifier_subset(ws) -> None:
    """T11 (OQ): show_node on an OQ emits the OQ body (topic/questions_raised/park_reason)
    and only the OQ-applicable modifier keys (subset is structural, not filtered)."""
    from mitos import mcp_server
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq(store, "oq-policy")
    _commit_oq(store, "oq-policy-v2", amends="oq-policy")
    ro = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(ro, None, None)):
        resp = json.loads(mcp_server.show_node("oq-policy", project=config.workspace_dir))
    assert resp["kind"] == "open_question"
    assert resp["topic"] == "oq-policy"
    assert resp["questions_raised"]
    assert "park_reason" in resp
    assert resp["amended_by"] == ["oq-policy-v2"]
    assert "superseded_by" not in resp and "corrected_by" not in resp


def test_mcp_show_node_not_found_is_object_with_hint(ws) -> None:
    """T10 genuine-absence: an absent handle returns the JSON not-found object
    (`found: false` + a hint), never an error/exception.

    The hint carried `mitos sync` until 6c and now carries **no shell command at
    all** — `agent-error-surface-never-prescribes-state-creating-setup`, widened
    past its original targeting-only scope. It is not merely emptied: a dead end is
    what PATTERNS forbids, so the recovery is expressed in this surface's own
    vocabulary (two read tools) and the unsynced-draft case is stated as a fact
    rather than as a command to fix it with.
    """
    from mitos import mcp_server
    config, m = ws
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.show_node("ghost-slug", project=config.workspace_dir))
    assert resp["found"] is False
    assert resp["ident"] == "ghost-slug"
    assert "mitos " not in resp["hint"]
    assert "list_decisions" in resp["hint"] and "query_decisions" in resp["hint"]
    # Still hedged and still buffer-free: truthful for a typo AND for a draft.
    assert "decisions.md" in resp["hint"]


def test_show_node_parity_with_cli_show_json(ws, capsys) -> None:
    """CLI⇄MCP parity (the vision's load-bearing invariant): for the SAME ident over a
    real store, `json.loads(show_node(ident))` == `json.loads(mitos show --json ident)`
    — for a found decision and a found OQ. Structural via the shared resolve_handle
    seam + the shared show_payload builder; pinned here too.

    **The not-found case is checked one notch differently since 6c, deliberately.**
    It used to ride the same dict-equality, which made this row the structural
    enforcer of the hint's byte-equality across the two surfaces. 6c removed that
    single-sourcing on purpose — the CLI's hint names a selector, the MCP's may name
    no shell command at all — so keeping dict-equality here would red on exactly the
    change the phase exists to make. Scoping the row down to the found cases and
    stopping there is the move that would have quietly weakened it: the *shape*
    parity is what the invariant is about, and it still holds. So the not-found half
    asserts the **key set** is identical and **every value except `hint`** is equal —
    strictly stronger than the old row about the shape, and honest about the one
    field that must differ — plus one rule per surface, so neither hint can drift
    into the other's.

    Since both shapes now carry the corpus provenance, the MCP side must NAME this
    workspace: the fixture never chdirs, so a `project`-less call would resolve
    pytest's cwd (the mitos-pub repo — itself a valid workspace) and stamp a
    different collection than the CLI side, reading as a broken stamp when the
    truth is that the call never named a target. The path form needs no registry.
    Resolution is left to run — only `get_workspace_components` is patched — because
    a row that mocked the resolution could not observe the stamp at all.
    """
    from mitos import mcp_server
    config, m = ws
    store = GraphStore(config.db_path)
    # found decision (superseded, the most demanding shape)
    _rec(m, "par-v1")
    _rec(m, "par-v2", supersedes="par-v1")
    # found OQ (amended)
    _commit_oq(store, "par-oq")
    _commit_oq(store, "par-oq-v2", amends="par-oq")

    ro = GraphStore(config.db_path, read_only=True)
    for ident in ("par-v1", "par-oq", "nope-not-here"):
        capsys.readouterr()
        cmd_show(config, ident, as_json=True)
        cli_out = json.loads(capsys.readouterr().out)
        with patch.object(mcp_server, "get_workspace_components", return_value=(ro, None, None)):
            mcp_out = json.loads(mcp_server.show_node(ident, project=config.workspace_dir))
        if ident == "nope-not-here":
            assert cli_out.keys() == mcp_out.keys(), "not-found SHAPE drifted"
            assert ({k: v for k, v in cli_out.items() if k != "hint"}
                    == {k: v for k, v in mcp_out.items() if k != "hint"})
            # The one field that must differ, and the rule each side owes.
            assert cli_out["hint"] != mcp_out["hint"]
            assert f"-p {config.project!r}" in cli_out["hint"]   # names its selector
            assert "mitos " not in mcp_out["hint"]               # no shell command
        else:
            assert cli_out == mcp_out, f"CLI⇄MCP show parity drift on {ident!r}"
        # Both shapes stamped, including the not-found one — the half that is
        # ambiguous between "no such handle" and "wrong project", and the half a
        # found-branch-only stamp would leave bare.
        assert cli_out["project"] == config.project
        assert cli_out["collection"] == config.qdrant_collection
        assert cli_out["workspace"] == config.workspace_dir


def test_cli_list_text_marks_modified(ws, capsys) -> None:
    """`mitos list` text output flags a modified-but-live decision with ⚠."""
    config, m = ws
    _rec(m, "listed", scope=["s"])
    _rec(m, "listed-v2", scope=["s"], narrows="listed")
    capsys.readouterr()
    cmd_list(config, scope="s")
    out = capsys.readouterr().out
    assert "⚠" in out and "narrowed by" in out and "listed-v2" in out


def test_cli_list_json_carries_modifiers(ws, capsys) -> None:
    """`mitos list --json` carries the modifier key for agent consumption."""
    config, m = ws
    _rec(m, "j-target", scope=["s"])
    _rec(m, "j-amender", scope=["s"], amends="j-target")
    capsys.readouterr()
    cmd_list(config, scope="s", as_json=True)
    out = json.loads(capsys.readouterr().out)
    target = next(d for d in out["decisions"] if d["slug"] == "j-target")
    assert target["amended_by"] == ["j-amender"]


def test_cli_surface_json_carries_modifiers(ws, capsys) -> None:
    """`mitos surface --json` (scope fallback) carries the modifier key."""
    config, m = ws
    _rec(m, "s-target", scope=["db"])
    _rec(m, "s-amender", scope=["db"], amends="s-target")
    capsys.readouterr()
    cmd_surface(config, query="anything", scope="db", as_json=True)
    out = json.loads(capsys.readouterr().out)
    target = next(d for d in out["active_decisions"] if d["slug"] == "s-target")
    assert target["amended_by"] == ["s-amender"]


# --------------------------------------------------------------------------- #
# CLI `query` — superseded-filter + Letter-complete + --json + --brief (Phase 2b)
#
# `cmd_query` constructs its own MitosSyncManager(config) internally, so the fakes
# are injected by patching `mitos.cli.MitosSyncManager` to return a stub carrying
# .store (real read store), .embed_provider, .vector_store (the MCP tests patch
# get_workspace_components instead — that's MCP-only).
# --------------------------------------------------------------------------- #

class _StubManager:
    """Stub MitosSyncManager: real read store + fake embed/vector providers."""

    def __init__(self, store, embed_provider, vector_store):
        self.store = store
        self.embed_provider = embed_provider
        self.vector_store = vector_store


def test_cli_query_filters_superseded_and_stamps(ws, capsys) -> None:
    """The R2 trap, the load-bearing pin: `query` drops a superseded-not-reused match
    (active-view get_node_by_slug → None, before the state check) and stamps the survivor.

    An all-active fixture would silently miss the trap, so `dead-v1` is superseded by
    `dead-v2` with a DISTINCT slug — dropped at the `if not node` step, not the state check.
    """
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    _rec(m, "q-amender", scope=["x"], amends="q-target")
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}, {"slug": "q-target", "score": 0.8}])
    stub = _StubManager(store, _FakeEmbed(), fake_vec)

    # --json: dead-v1 absent, q-target present + stamped + fenced.
    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug", as_json=True)
    resp = json.loads(capsys.readouterr().out)
    slugs = [d["slug"] for d in resp["matches"]]
    assert "dead-v1" not in slugs
    target = next(d for d in resp["matches"] if d["slug"] == "q-target")
    assert target["amended_by"] == ["q-amender"]
    assert target["rejected_paths"]

    # text: dead-v1 omitted, q-target's ⚠ marker shown.
    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug")
    out = capsys.readouterr().out
    assert "dead-v1" not in out
    assert "q-target" in out and "⚠" in out and "amended by" in out and "q-amender" in out


def test_cli_query_json_parity_with_mcp(ws) -> None:
    """T4: `query --json` matches list is byte-equal to query_decisions' ranked matches.

    The claim MUST be a non-slug phrase — query_decisions tries exact-slug dereference
    first; a real-slug claim returns the single-decision shape, not the ranked envelope.
    """
    from mitos import mcp_server
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    _rec(m, "q-amender", scope=["x"], amends="q-target")
    claim = "a claim that is not any slug"
    matches = [{"slug": "q-target", "score": 0.88}]

    store_cli = GraphStore(config.db_path, read_only=True)
    stub = _StubManager(store_cli, _FakeEmbed(), _FakeVector(list(matches)))
    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_query(config, claim, as_json=True)
    cli_resp = json.loads(buf.getvalue())

    store_mcp = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store_mcp, _FakeEmbed(), _FakeVector(list(matches)))):
        mcp_resp = json.loads(mcp_server.query_decisions(claim, project=config.workspace_dir))

    assert cli_resp["matches"] == mcp_resp["matches"]


def test_cli_query_brief_drops_rejected_keeps_stamp(ws, capsys) -> None:
    """`--brief` sheds only rejected_paths, never a modifier stamp (M4) — text + --json."""
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    _rec(m, "q-amender", scope=["x"], amends="q-target")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "q-target", "score": 0.88}])
    stub = _StubManager(store, _FakeEmbed(), fake_vec)

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug", as_json=True, brief=True)
    target = next(d for d in json.loads(capsys.readouterr().out)["matches"] if d["slug"] == "q-target")
    assert "rejected_paths" not in target and target["amended_by"] == ["q-amender"]

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug", brief=True)
    out = capsys.readouterr().out
    assert "Rejected:" not in out and "⚠" in out and "q-amender" in out


def test_cli_query_true_miss_keeps_plain_message(ws, capsys) -> None:
    """A GENUINE miss — retrieval returned nothing — keeps the plain 2b message and a
    clean empty envelope with NO `all_superseded` field. (The true-miss leg of the
    split that 2d's blackout vector required; proves blackout ≠ miss.)

    Whole-dict equality is the point: it is the tree's only such pin on this
    envelope, and what it proves is that no stray key rides along. 3a widened it to
    carry the confidence band rather than weakening it to a subset check — the band
    is composed here rather than spelled as a literal because `config.project` is
    the caller's own vocabulary (a registered name under this module's fixture, a
    path under a hand-built config) and the CLI redirect interpolates it.
    """
    config, m = ws
    _rec(m, "unrelated", scope=["x"])
    store = GraphStore(config.db_path, read_only=True)
    stub = _StubManager(store, _FakeEmbed(), _FakeVector([]))

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug", as_json=True)
    resp = json.loads(capsys.readouterr().out)
    _, band_note = assess_query_recall(top_score=None, result_count=0,
                                       config=config, surface="cli")
    assert resp == {"query": "a claim that is not any slug", "depth_mode": "letter",
                    "matches": [],
                    "project": config.project,
                    "collection": config.qdrant_collection,
                    "workspace": config.workspace_dir,
                    "confidence": "none",
                    "note": band_note}
    assert "all_superseded" not in resp

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug")
    assert "No matching decisions found." in capsys.readouterr().out


def test_cli_query_blackout_vector(ws, capsys) -> None:
    """Blackout: retrieval returned a match but it was superseded-filtered → the
    recovery vector, NOT the bare-empty miss. `--json` gains `all_superseded` (with the
    live successor); text shows the blackout note naming the retired handle + successor."""
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}])
    stub = _StubManager(store, _FakeEmbed(), fake_vec)

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug", as_json=True)
    resp = json.loads(capsys.readouterr().out)
    assert resp["matches"] == []
    assert resp["all_superseded"] == [{"slug": "dead-v1", "state": "superseded", "superseded_by": ["dead-v2"]}]

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_query(config, "a claim that is not any slug")
    out = capsys.readouterr().out
    assert "No matching decisions found." not in out
    assert "dead-v1" in out and "dead-v2" in out and "superseded" in out


def test_cli_query_limit_threads_through(ws, capsys) -> None:
    """`--limit` SETS the top-k: a value above the default-5 must reach
    vector_store.query un-truncated (the §6 no-op-above-default trap), a low value
    trims, and out-of-range values clamp calmly to [1, RANKED_LIMIT_CEILING]."""
    from mitos.display import RANKED_LIMIT_CEILING
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    store = GraphStore(config.db_path, read_only=True)

    for requested, expected in [(20, 20), (3, 3), (999, RANKED_LIMIT_CEILING), (0, 1), (None, 5)]:
        fake_vec = _FakeVector([{"slug": "q-target", "score": 0.8}])
        stub = _StubManager(store, _FakeEmbed(), fake_vec)
        with patch("mitos.cli.MitosSyncManager", return_value=stub):
            cmd_query(config, "a claim that is not any slug", as_json=True, limit=requested)
        capsys.readouterr()
        assert fake_vec.last_limit == expected, f"--limit {requested} → {fake_vec.last_limit}, want {expected}"


def test_cli_surface_blackout_vector(ws, capsys) -> None:
    """Surface blackout: semantic ran, retrieved a match, all superseded → the recovery
    vector. `--json` gains `all_superseded`; confidence stays `none`."""
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}])
    stub = _StubManager(store, _FakeEmbed(), fake_vec)

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_surface(config, "a claim that is not any slug", as_json=True)
    resp = json.loads(capsys.readouterr().out)
    assert resp["active_decisions"] == []
    assert resp["all_superseded"] == [{"slug": "dead-v1", "state": "superseded", "superseded_by": ["dead-v2"]}]
    assert "dead-v1" in resp["note"]


def test_cli_surface_mixed_result_no_blackout(ws, capsys) -> None:
    """A mixed result (one survivor + one superseded) is byte-identical to 2b — NO
    `all_superseded` field, normal render. The superseded filter itself is unchanged."""
    config, m = ws
    _rec(m, "live-one", scope=["x"])
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}, {"slug": "live-one", "score": 0.8}])
    stub = _StubManager(store, _FakeEmbed(), fake_vec)

    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        cmd_surface(config, "a claim that is not any slug", as_json=True)
    resp = json.loads(capsys.readouterr().out)
    assert [d["slug"] for d in resp["active_decisions"]] == ["live-one"]
    assert "all_superseded" not in resp


# --------------------------------------------------------------------------- #
# Blackout + --limit MCP twins + CLI⇄MCP parity (T5)
# --------------------------------------------------------------------------- #

def test_mcp_query_blackout_vector(ws) -> None:
    """query_decisions' semantic branch fires the blackout vector when its whole
    top-k is superseded — `matches` empty, `all_superseded` populated."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        resp = json.loads(mcp_server.query_decisions("a claim that is not any slug",
            project=config.workspace_dir))
    assert resp["matches"] == []
    assert resp["all_superseded"] == [{"slug": "dead-v1", "state": "superseded", "superseded_by": ["dead-v2"]}]


def test_mcp_surface_blackout_vector(ws) -> None:
    """surface_decisions fires the blackout vector when its whole top-k is superseded."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "dead-v1", "score": 0.9}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        resp = json.loads(mcp_server.surface_decisions(query="a claim that is not any slug",
            project=config.workspace_dir))
    assert resp["active_decisions"] == []
    assert resp["all_superseded"] == [{"slug": "dead-v1", "state": "superseded", "superseded_by": ["dead-v2"]}]


def test_mcp_limit_threads_through(ws) -> None:
    """The MCP `limit` arg SETS the top-k threaded to vector_store.query; clamps calmly."""
    from mitos import mcp_server
    from mitos.display import RANKED_LIMIT_CEILING
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    store = GraphStore(config.db_path, read_only=True)

    for requested, expected in [(20, 20), (3, 3), (999, RANKED_LIMIT_CEILING), (0, 1)]:
        fake_vec = _FakeVector([{"slug": "q-target", "score": 0.8}])
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=(store, _FakeEmbed(), fake_vec)):
            mcp_server.query_decisions("a claim that is not any slug", limit=requested,
                project=config.workspace_dir)
        assert fake_vec.last_limit == expected
        fake_vec2 = _FakeVector([{"slug": "q-target", "score": 0.8}])
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=(store, _FakeEmbed(), fake_vec2)):
            mcp_server.surface_decisions(query="a claim that is not any slug", limit=requested,
                project=config.workspace_dir)
        assert fake_vec2.last_limit == expected


def test_cli_mcp_blackout_parity(ws) -> None:
    """T5: CLI⇄MCP `all_superseded` shape is identical for both query and surface."""
    from mitos import mcp_server
    import io, contextlib
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    claim = "a claim that is not any slug"
    matches = [{"slug": "dead-v1", "score": 0.9}]

    # query parity
    store_cli = GraphStore(config.db_path, read_only=True)
    stub = _StubManager(store_cli, _FakeEmbed(), _FakeVector(list(matches)))
    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_query(config, claim, as_json=True)
    cli_q = json.loads(buf.getvalue())
    store_mcp = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store_mcp, _FakeEmbed(), _FakeVector(list(matches)))):
        mcp_q = json.loads(mcp_server.query_decisions(claim, project=config.workspace_dir))
    assert cli_q["all_superseded"] == mcp_q["all_superseded"]

    # surface parity
    stub2 = _StubManager(GraphStore(config.db_path, read_only=True), _FakeEmbed(), _FakeVector(list(matches)))
    with patch("mitos.cli.MitosSyncManager", return_value=stub2):
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            cmd_surface(config, claim, as_json=True)
    cli_s = json.loads(buf2.getvalue())
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(GraphStore(config.db_path, read_only=True), _FakeEmbed(), _FakeVector(list(matches)))):
        mcp_s = json.loads(mcp_server.surface_decisions(query=claim, project=config.workspace_dir))
    assert cli_s["all_superseded"] == mcp_s["all_superseded"]


# --------------------------------------------------------------------------- #
# Renderer (live_axioms.md is a primary agent-context artifact)
# --------------------------------------------------------------------------- #

def test_render_node_markdown_emits_marker() -> None:
    """A live node with modifiers renders a ⚠ chase-it line; without, nothing extra."""
    node = {"slug": "x", "core_axiom": "Do the thing.", "scope": [], "mechanisms": []}
    plain = render_node_markdown(node)
    assert "⚠" not in plain
    marked = render_node_markdown(node, {"amended_by": ["x-v2"]})
    assert "⚠ Amended by:** x-v2" in marked and "chase" in marked


def test_render_all_writes_modifier_marker(ws) -> None:
    """render_all annotates an amended-but-active decision in live_axioms.md."""
    config, m = ws
    _rec(m, "rendered", scope=["r"])
    _rec(m, "rendered-v2", scope=["r"], amends="rendered")
    store = GraphStore(config.db_path)
    MitosRenderer(config.workspace_dir).render_all(store)
    with open(f"{config.workspace_dir}/live_axioms.md", encoding="utf-8") as f:
        content = f.read()
    assert "⚠ Amended by:** rendered-v2" in content


# --------------------------------------------------------------------------- #
# Dead-amender de-projection — the genuinely-new 2b gate (DoD #6, T6 decision side)
#
# The §4.3 FORWARD HAZARD: V1a's modifier join joins the source node only for its
# slug; it never checks the SOURCE's own liveness. Once amends/narrows commit (2a),
# an amender that is itself later superseded/corrected would otherwise ghost-stamp a
# still-live target — a dead axiom projecting onto a live node. 2b's source-liveness
# filter de-projects: the present-tense projections (amended_by/narrowed_by) drop
# when their source is killed; the historical kill-pointers (superseded_by/
# corrected_by) stay (they only ever point at already-dead targets and must stay
# consistent with get_node_state, which computes the kill state UNFILTERED).
#
# Fail-safe, not data loss: a superseded amender is superseded, NOT deleted — its
# axiom stays in the graph (recoverable via get_lineage, 3a); the target merely stops
# being projected onto. Re-amending from the successor is the author's call (smoothed
# at V3a's reconciliation surface, not 2b — 2b never auto-propagates).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kill_relation,dead_state", [
    ("supersedes", "superseded"),
    ("corrects", "corrected"),
])
def test_dead_amender_deprojects_amended_by(ws, kill_relation, dead_state) -> None:
    """A superseded/corrected amender stops stamping its still-LIVE target.

    A amended by B (A active, reads amended_by:[B]); then B is killed by C in a
    SEPARATE entry (no re-declared amends). A must read un-amended again — the dead
    amender's "go read B for the current nuance" is stale. B's axiom is NOT lost: it
    stays in the graph as a (now-inactive) node, recoverable via lineage.
    """
    config, m = ws
    a = _rec(m, "a")
    b = _rec(m, "b", amends="a")
    store = GraphStore(config.db_path)
    # Before B dies: A is amended by B.
    assert store.get_modifiers(a["id"]) == {"amended_by": ["b"]}

    # Kill B from a SEPARATE entry (stacking a kill-edge + a non-kill edge on one
    # entry trips dangling_edge — 2a gotcha).
    _rec(m, "c", **{kill_relation: "b"})
    store = GraphStore(config.db_path)

    # A de-projects: it reads active with NO amended_by key, on the store map AND on
    # an active read surface.
    assert store.get_modifiers(a["id"]) == {}
    assert store.get_node_state(a["id"]) == "active"
    a_node = store.get_node_by_slug("a")
    assert "amended_by" not in a_node and "narrowed_by" not in a_node

    # Fail-safe: B is still in the graph (its axiom recoverable), just retired.
    assert store.get_node(b["id"]) is not None
    assert store.get_node_state(b["id"]) == dead_state


def test_dead_narrower_deprojects_narrowed_by(ws) -> None:
    """The narrows mirror of de-projection — a superseded narrower stops stamping.

    Confirms the filter gates narrowed_by exactly as it gates amended_by (both are
    present-tense projections onto a live target).
    """
    config, m = ws
    a = _rec(m, "a")
    _rec(m, "b", narrows="a")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(a["id"]) == {"narrowed_by": ["b"]}

    _rec(m, "c", supersedes="b")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(a["id"]) == {}
    assert store.get_node_state(a["id"]) == "active"


def test_deletion_decision_deprojects_amender(ws) -> None:
    """§4.5 successor-less death — a bare deletion decision de-projects identically.

    A deletion decision supersedes the amender WITHOUT introducing a successor that
    re-amends A. De-projection needs no successor: the same kill-edge set
    (_KILL_EDGE_TYPES_SQL) drives it, with no separate clause for the deletion case.
    This pins that a successor-less death (the §4.5 model) correctly de-projects — we
    do NOT accidentally require a replacement to exist before dropping the stale
    projection.
    """
    config, m = ws
    a = _rec(m, "a")
    _rec(m, "amender", amends="a")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(a["id"]) == {"amended_by": ["amender"]}

    # A deletion decision: its sole role is to supersede the amender (no re-declared
    # amends to A, no replacement content).
    _rec(m, "deletion-decision", supersedes="amender")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(a["id"]) == {}
    assert store.get_node_state(a["id"]) == "active"


def test_kill_pointer_survives_source_death(ws) -> None:
    """⚠ Decision 3 guard — historical kill-pointers stay UNFILTERED (regression trap).

    A superseded by B; then B superseded by C. A's superseded_by:[B] MUST survive even
    though B is itself now dead — it is a permanent "who retired me" record, and
    get_node_state computes `superseded` UNFILTERED. A future "simplify" pass that made
    the source-liveness filter uniform would empty superseded_by while get_node_state
    still returned `superseded`, desyncing the payload from the state. This test fails
    that regression.
    """
    config, m = ws
    a = _rec(m, "a")
    _rec(m, "b", supersedes="a")
    _rec(m, "c", supersedes="b")  # B itself dies
    store = GraphStore(config.db_path)

    # The historical pointer survives B's death, and stays consistent with the state.
    assert store.get_node(a["id"])["superseded_by"] == ["b"]
    assert store.get_node_state(a["id"]) == "superseded"


def test_partial_hub_deprojects_only_dead_amender(ws) -> None:
    """Per-source de-projection, not all-or-nothing: a live amender survives a dead one.

    Hub amended by X (stays active) and Y; then Y is superseded. The hub surfaces ONLY
    the live amender X (Y dropped, slug-sorted) — the filter is correlated per modifier
    row on each edge's own source, never a blanket on/off for the whole node.
    """
    config, m = ws
    hub = _rec(m, "hub")
    _rec(m, "x-amender", amends="hub")
    _rec(m, "y-amender", amends="hub")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(hub["id"]) == {"amended_by": ["x-amender", "y-amender"]}

    _rec(m, "y-killer", supersedes="y-amender")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(hub["id"]) == {"amended_by": ["x-amender"]}
    assert store.get_node_state(hub["id"]) == "active"


# --------------------------------------------------------------------------- #
# CLI `open-questions --json` — the OQ-subset modifier stamping (Phase 2c)
#
# The "every decision-read surface stamps modifiers" rule (CLAUDE.md), applied to
# the OQ JSON read: an amended-but-active parked OQ must carry `amended_by` so it
# doesn't read as the final word — and the decision-only keys (`superseded_by` /
# `corrected_by`) must be STRUCTURALLY absent (an OQ is never the target of a
# `supersedes`/`corrects` decision edge in the OQ view), proving `_oq_modifiers`
# reuse yields the right subset without an allowlist.
# --------------------------------------------------------------------------- #

def _commit_oq(store: GraphStore, slug: str, **relations) -> dict:
    """Commits a hand-built open_question ParsedEntry through the write path."""
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = f"Topic for {slug}"
    e.questions_raised = [f"What about {slug}?"]
    e.scope = []
    for name, val in relations.items():
        setattr(e, name, val if isinstance(val, list) else [val])
    return store.commit_parsed_entry(e)


def test_cmd_open_questions_json_stamps_oq_modifier_subset(ws, capsys) -> None:
    """`open-questions --json`: an amended-but-active OQ carries `amended_by`, and the
    decision-only modifier keys are structurally absent (the subset is not filtered)."""
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq(store, "q-base")
    _commit_oq(store, "q-amender", amends="q-base")

    capsys.readouterr()
    cmd_open_questions(config, as_json=True)
    out = json.loads(capsys.readouterr().out)

    base = next(oq for oq in out["open_questions"] if oq["topic"] == "q-base")
    assert base["amended_by"] == ["q-amender"]
    assert "superseded_by" not in base
    assert "corrected_by" not in base
