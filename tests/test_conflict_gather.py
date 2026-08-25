"""Tests for the Conflict sensor's candidate-gathering stage (Phase 2a, §6.5 S1–S3).

``gather_candidates`` is the first pipeline stage: embed a proposed axiom in *document*
space → one bounded scope-blind over-fetch → per-match computed-state re-verify (keep
``active ∪ drifted``) → return every live over-fetched neighbour *un-filtered,
un-ranked, un-truncated* (S4–S6 is 2b), OR a typed ``Unavailable`` when the semantic
substrate fails. The load-bearing property under test: the **three terminal states**
(``Unavailable`` / ``[]`` / ``[Candidate…]``) never blur into each other.

Discipline (PATTERNS + scout brief): synchronous hand-rolled fakes (NOT
``unittest.mock``); a real temp ``GraphStore`` seeded via ``commit_parsed_entry`` (never
embeds → keyless + deterministic) for the graph reads; env stripped of keys so nothing
reaches a live service. ``drifted`` is not constructible on the real v0.1 store (the
channel is reserved) → its "kept as live" case uses a stub store.
"""

import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from mitos import conflict
from mitos.conflict import (
    Candidate,
    ConflictUnavailableReason,
    JUDGMENT_REASONS,
    SEMANTIC_SUBSTRATE_REASONS,
    Unavailable,
    gather_candidates,
)
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.errors import (CollectionMissingError, DatabaseError, EmbeddingError,
                          VectorStoreError)
from mitos.identity import canonical_core_string_norm, embedding_text
from mitos.parser import ParsedEntry
from mitos.store import GraphStore


# --------------------------------------------------------------------------- #
# Fixtures — offline env + a real, empty, keyless temp store
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def store() -> Iterator[GraphStore]:
    """A fresh, initialized, empty on-disk graph store (no network, no keys)."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield GraphStore(config.db_path)
    shutil.rmtree(tmp, ignore_errors=True)


def _seed_decision(store: GraphStore, slug: str, axiom: str) -> None:
    """Commits one live decision (keyless — commit_parsed_entry never embeds)."""
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = "Rejected the obvious alternative."
    entry.scope = ["cache"]
    store.commit_parsed_entry(entry)


def _seed_open_question(store: GraphStore, slug: str, topic: str) -> None:
    """Commits one open_question (keyless). Its kill-edge state is 'active' (OQ
    resolution rides the ``resolves`` edge, not a kill edge — get_node_state can't see it)."""
    entry = ParsedEntry("open_question", slug, 40, 50)
    entry.topic = topic
    entry.questions_raised = [f"How should we handle {topic}?"]
    entry.scope = ["cache"]
    store.commit_parsed_entry(entry)


def _resolve(store: GraphStore, decision_slug: str, oq_slug: str) -> None:
    """Commits a decision that resolves ``oq_slug`` (the OQ-view state becomes 'resolved',
    while get_node_state still reports 'active' — the exact production reachability)."""
    entry = ParsedEntry("decision", decision_slug, 60, 70)
    entry.axiom = f"Resolving decision for {oq_slug}."
    entry.rejected_paths = "Rejected leaving it parked."
    entry.scope = ["cache"]
    entry.resolves = [oq_slug]
    store.commit_parsed_entry(entry)


def _supersede(store: GraphStore, new_slug: str, dead_slug: str) -> None:
    """Commits a superseder so ``dead_slug`` leaves the active view (→ get_node_by_slug None)."""
    entry = ParsedEntry("decision", new_slug, 20, 30)
    entry.axiom = f"Superseding decision retiring {dead_slug}."
    entry.rejected_paths = "Rejected keeping the old policy."
    entry.scope = ["cache"]
    entry.supersedes = [dead_slug]
    store.commit_parsed_entry(entry)


# --------------------------------------------------------------------------- #
# Fakes — synchronous, hand-rolled (the project idiom)
# --------------------------------------------------------------------------- #

class _FakeEmbed:
    """Records the S1 call and either returns a fixed vector or raises."""

    def __init__(self, raises: Optional[BaseException] = None) -> None:
        self._raises = raises
        self.last_text: Optional[str] = None
        self.last_is_query: Optional[bool] = None
        self.call_count = 0

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.call_count += 1
        self.last_text = text
        self.last_is_query = is_query
        if self._raises is not None:
            raise self._raises
        return [0.1, 0.2, 0.3]


class _FakeVector:
    """Returns canned matches (or raises); records the over-fetch limit + call count."""

    def __init__(
        self,
        matches: Optional[List[Dict[str, Any]]] = None,
        raises: Optional[BaseException] = None,
    ) -> None:
        self._matches = matches if matches is not None else []
        self._raises = raises
        self.last_limit: Optional[int] = None
        self.call_count = 0

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_limit = limit
        if self._raises is not None:
            raise self._raises
        return list(self._matches)

    def upsert(self, *a: Any, **k: Any) -> None:  # pragma: no cover - never hit in 2a
        pass


class _SpyStore:
    """Wraps a real store, counting the S3 reads so a test can prove the None short-circuit."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.slug_calls: List[str] = []
        self.state_calls: List[str] = []

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        self.slug_calls.append(slug)
        return self._inner.get_node_by_slug(slug)

    def get_node_state(self, node_id: str) -> str:
        self.state_calls.append(node_id)
        return self._inner.get_node_state(node_id)


class _StubStore:
    """A fully hand-rolled store: ``get_node_by_slug`` resolves, ``get_node_state`` is scripted.

    Lets a test force ``get_node_by_slug`` (active-scoped) and ``get_node_state`` (M3
    computed) to *disagree* — the race window the real single-process store can't open —
    to prove 2a treats ``get_node_state`` as the authoritative live filter (D4a), and to
    inject a store fault that must propagate (D4).
    """

    def __init__(self, state: Optional[str] = None, state_raises: Optional[BaseException] = None) -> None:
        self._state = state
        self._state_raises = state_raises
        self.slug_calls: List[str] = []
        self.state_calls: List[str] = []

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        self.slug_calls.append(slug)
        # A live-looking hydrated node dict (carries the "id" get_node_state needs).
        return {"id": f"id-{slug}", "core_axiom": f"axiom for {slug}", "kind": "decision"}

    def get_node_state(self, node_id: str) -> str:
        self.state_calls.append(node_id)
        if self._state_raises is not None:
            raise self._state_raises
        assert self._state is not None
        return self._state


def _gather(axiom: str, embed: Any, vector: Any, store: Any) -> Any:
    """Call convenience — the three collaborators are keyword-only by contract."""
    return gather_candidates(
        axiom, embed_provider=embed, vector_store=vector, store=store
    )


# --------------------------------------------------------------------------- #
# 1. Happy path + the load-bearing 2a/2b stage boundary (the §8a golden trace)
# --------------------------------------------------------------------------- #

def test_happy_path_keeps_every_live_neighbour_and_applies_no_2b_filter(store: GraphStore) -> None:
    """The §8a trace: 2a drops ONLY the non-live row, never 2b's declared/floor drops.

    9 over-fetched neighbours; row 4 (``eviction-lru``) is superseded. 2a returns the
    other 8 in query order — crucially still carrying the declared-target/own-slug rows
    (1, 2) and the below-floor rows (8, 9), which are 2b's to drop, not 2a's. (Row 5 is
    ``drifted`` in the vision; the real v0.1 store can't produce drifted, so it is seeded
    active here — its liveness is identical for 2a's purposes; the drifted case is pinned
    separately by ``test_drifted_is_kept_as_live``.)
    """
    live = [
        ("cache-policy", 0.91),               # 1 — 2b will S4-drop (declared Supersedes: target)
        ("cache-policy-v2", 0.90),            # 2 — 2b will S4-drop (own slug)
        ("cache-ttl-fixed", 0.78),            # 3 — judged
        ("cache-invalidation-manual", 0.71),  # 5 — judged (drifted in vision; active here)
        ("metrics-naming", 0.68),             # 6 — judged
        ("global-no-cache", 0.66),            # 7 — judged
        ("db-conn-pooling", 0.41),            # 8 — 2b will S5-drop (below floor)
        ("logging-format", 0.22),             # 9 — 2b will S5-drop (below floor)
    ]
    for slug, _ in live:
        _seed_decision(store, slug, f"Axiom for {slug}.")
    # Row 4: seed then supersede so get_node_by_slug(eviction-lru) → None (the only drop).
    _seed_decision(store, "eviction-lru", "Evict least-recently-used entries.")
    _supersede(store, "eviction-fifo", "eviction-lru")

    matches = (
        [{"slug": live[0][0], "score": live[0][1]}]
        + [{"slug": live[1][0], "score": live[1][1]}]
        + [{"slug": live[2][0], "score": live[2][1]}]
        + [{"slug": "eviction-lru", "score": 0.74}]  # row 4 — dropped (superseded → None)
        + [{"slug": s, "score": sc} for s, sc in live[3:]]
    )
    result = _gather("Cache aggressively with a v2 policy.", _FakeEmbed(), _FakeVector(matches), store)

    assert isinstance(result, list)
    assert [c.slug for c in result] == [s for s, _ in live]        # order preserved, row 4 gone
    assert [c.score for c in result] == [sc for _, sc in live]     # scores carried verbatim
    assert "eviction-lru" not in {c.slug for c in result}          # the sole non-live drop
    # Stage-boundary pins: 2a does NOT apply 2b's filters.
    kept = {c.slug for c in result}
    assert {"cache-policy", "cache-policy-v2"} <= kept             # declared/own-slug NOT dropped
    assert {"db-conn-pooling", "logging-format"} <= kept          # below-floor NOT dropped
    assert all(isinstance(c, Candidate) for c in result)


# --------------------------------------------------------------------------- #
# 2. S1 — document space + normalized text (the one spot Conflict must NOT copy surface)
# --------------------------------------------------------------------------- #

def test_s1_embeds_normalized_document_space_text_not_the_raw_axiom(store: GraphStore) -> None:
    """S1 embeds ``canonical_core_string_norm(axiom)`` with ``is_query=False`` (CONF-D2/D2)."""
    embed = _FakeEmbed()
    raw = "  We cache aggressively.  "  # trailing/leading + NBSP so NFC+strip is visible
    _gather(raw, embed, _FakeVector([]), store)

    normalized = canonical_core_string_norm(raw)
    assert embed.last_is_query is False                     # document space, not query space
    assert embed.last_text == normalized                    # normalized, not raw
    assert embed.last_text == embedding_text({"kind": "decision", "axiom": raw})
    assert embed.last_text != raw                            # the raw string was NOT embedded


# --------------------------------------------------------------------------- #
# 3. S2 — a single bounded over-fetch, never iterative
# --------------------------------------------------------------------------- #

def test_s2_single_bounded_overfetch_past_top_k(store: GraphStore) -> None:
    """One query call, ``limit == CONFLICT_OVERFETCH_LIMIT`` (> TOP_K) — never a re-fetch loop."""
    vector = _FakeVector([])
    _gather("Some axiom.", _FakeEmbed(), vector, store)

    assert vector.call_count == 1
    assert vector.last_limit == conflict.CONFLICT_OVERFETCH_LIMIT
    assert vector.last_limit > conflict.CONFLICT_TOP_K


# --------------------------------------------------------------------------- #
# 4. S3 — retired / absent resolve to None (the primary filter); state NOT probed
# --------------------------------------------------------------------------- #

def test_s3_absent_and_retired_drop_at_none_without_probing_state(store: GraphStore) -> None:
    """Both an absent slug and a superseded node drop at ``get_node_by_slug is None``.

    And ``get_node_state`` is never called for either — proving the ordering short-circuit
    (probing an unresolved id would default to ``active`` and mislabel a stale vector live).
    """
    _seed_decision(store, "eviction-lru", "Evict least-recently-used entries.")
    _supersede(store, "eviction-fifo", "eviction-lru")
    spy = _SpyStore(store)
    matches = [
        {"slug": "ghost-never-committed", "score": 0.9},  # (a) absent → None
        {"slug": "eviction-lru", "score": 0.8},           # (b) superseded → None (active-scoped)
    ]
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), spy)

    assert result == []
    assert spy.slug_calls == ["ghost-never-committed", "eviction-lru"]
    assert spy.state_calls == []  # never probed — the None short-circuit held


# --------------------------------------------------------------------------- #
# 4b. S3 — a non-decision match (open_question) is screened at gather, never judged
# --------------------------------------------------------------------------- #

def test_open_question_match_is_dropped_by_kind_not_crashed_on(store: GraphStore) -> None:
    """An open_question that the KNN surfaces is filtered at gather — never reaches judgment.

    The parent conflict-sensor regression: the Qdrant collection holds OQ vectors too
    (sync embeds every committed node), get_node_by_slug is kind-agnostic, and
    get_node_state derives decision state from kill edges only — so a resolved OQ reports
    'active' and would survive to judge_input_from_node, which reads node["core_axiom"]
    an OQ lacks → KeyError mid-sweep. gather_candidates must screen kind and keep only the
    decision neighbour, with no crash.
    """
    _seed_decision(store, "cache-policy", "Cache aggressively.")
    _seed_open_question(store, "oq-cache-cadence", "cache invalidation cadence")
    _resolve(store, "cache-cadence-hourly", "oq-cache-cadence")  # OQ-view resolved; state still 'active'

    # Sanity: the OQ is live-by-get_node_state (the exact trap) and resolvable by slug.
    oq_node = store.get_node_by_slug("oq-cache-cadence")
    assert oq_node is not None and oq_node["kind"] == "open_question"
    assert store.get_node_state(oq_node["id"]) == "active"  # kill-edge state, blind to resolution
    assert "core_axiom" not in oq_node                       # the missing key that would KeyError

    matches = [
        {"slug": "oq-cache-cadence", "score": 0.95},  # nearest neighbour is the OQ — must be dropped
        {"slug": "cache-policy", "score": 0.80},      # the one real decision candidate
    ]
    result = _gather("Cache invalidation policy.", _FakeEmbed(), _FakeVector(matches), store)

    assert isinstance(result, list)                          # no crash, no Unavailable
    assert [c.slug for c in result] == ["cache-policy"]      # OQ screened, decision kept
    assert all(c.node["kind"] == "decision" for c in result)


# --------------------------------------------------------------------------- #
# 5. S3 — get_node_state is the authoritative M3 re-verify (the race-guard)
# --------------------------------------------------------------------------- #

def test_s3_drops_when_state_reverify_says_superseded_even_if_slug_resolved() -> None:
    """A node that passes the active-view slug read but whose computed state is superseded is dropped."""
    stub = _StubStore(state="superseded")
    matches = [{"slug": "raced-out", "score": 0.9}]
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), stub)

    assert result == []
    assert stub.slug_calls == ["raced-out"]
    assert stub.state_calls == ["id-raced-out"]  # the re-verify DID run and did the dropping


# --------------------------------------------------------------------------- #
# 6. S3 — drifted is live, kept (not constructible on the real store — §7 gotcha)
# --------------------------------------------------------------------------- #

def test_drifted_is_kept_as_live() -> None:
    """A ``drifted`` computed state counts as live (``active ∪ drifted``) and is kept."""
    stub = _StubStore(state="drifted")
    matches = [{"slug": "drifting", "score": 0.7}]
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), stub)

    assert isinstance(result, list) and len(result) == 1
    assert result[0].slug == "drifting"
    assert result[0].state == "drifted"


# --------------------------------------------------------------------------- #
# 7 & 8. Degraded — the two semantic-substrate faults become a typed Unavailable
# --------------------------------------------------------------------------- #

def test_embedding_failure_is_typed_unavailable_and_short_circuits_the_query(store: GraphStore) -> None:
    """``EmbeddingError`` → ``Unavailable(EMBEDDING)``; the vector store is never called."""
    embed = _FakeEmbed(raises=EmbeddingError("gemini down"))
    vector = _FakeVector([{"slug": "x", "score": 0.9}])
    result = _gather("Axiom.", embed, vector, store)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.EMBEDDING
    assert "gemini down" in result.detail          # raw exception message, for logging
    assert vector.call_count == 0                   # S2 never reached — no partial work


def test_vector_store_failure_is_typed_unavailable(store: GraphStore) -> None:
    """``VectorStoreError`` → ``Unavailable(VECTOR_STORE)``."""
    embed = _FakeEmbed()
    vector = _FakeVector(raises=VectorStoreError("qdrant unreachable"))
    result = _gather("Axiom.", embed, vector, store)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.VECTOR_STORE
    assert "qdrant unreachable" in result.detail


def test_collection_missing_is_its_own_typed_unavailable(store: GraphStore) -> None:
    """A missing collection is a THIRD reason, not the unreachable one it subclasses.

    Ordering-sensitive: ``CollectionMissingError`` is a ``VectorStoreError``, so an
    arm below the broad one would be dead and the receipt, the sync notice and the
    check token would all report "unreachable" for a running Qdrant.

    **The seeded decision is load-bearing, not scenery.** The ``store`` fixture is an
    empty graph, and over an empty active set an absent collection is the empty index
    rather than a gap (I8) — the arm returns ``[]`` and never reaches the reason this
    row is about. One live decision puts the sweep in the state where absence is a
    real hole, which is the only state where arm ordering is observable. The
    empty-graph twin lives in ``test_conflict_faults.py``.
    """
    _seed_decision(store, "seeded-precedent", "A committed precedent so absence is a gap.")
    embed = _FakeEmbed()
    vector = _FakeVector(
        raises=CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )
    )
    result = _gather("Axiom.", embed, vector, store)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.COLLECTION_MISSING
    assert "mitos-x" in result.detail


# --------------------------------------------------------------------------- #
# 9. A graph-store fault PROPAGATES — it is never masked as Unavailable/[]  (D4)
# --------------------------------------------------------------------------- #

def test_store_fault_propagates_and_is_not_masked() -> None:
    """A ``DatabaseError`` from the graph store propagates — not swallowed into a lie."""
    stub = _StubStore(state_raises=DatabaseError("db locked"))
    matches = [{"slug": "foo", "score": 0.9}]
    with pytest.raises(DatabaseError):
        _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), stub)


# --------------------------------------------------------------------------- #
# 10. Degraded ≠ empty — a healthy-but-empty substrate returns [], NOT Unavailable
# --------------------------------------------------------------------------- #

def test_healthy_empty_query_returns_empty_list_not_unavailable(store: GraphStore) -> None:
    """A healthy substrate that simply matched nothing returns ``[]`` (a list), never Unavailable."""
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector([]), store)
    assert result == []
    assert isinstance(result, list)
    assert not isinstance(result, Unavailable)


def test_all_matches_retired_returns_empty_list_not_unavailable(store: GraphStore) -> None:
    """Substrate healthy but every match retired/absent → ``[]`` (clean empty), never Unavailable."""
    _seed_decision(store, "old-policy", "The old policy.")
    _supersede(store, "new-policy", "old-policy")
    matches = [
        {"slug": "old-policy", "score": 0.9},     # superseded → None
        {"slug": "vanished", "score": 0.8},        # absent → None
    ]
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), store)
    assert result == []
    assert isinstance(result, list)


# --------------------------------------------------------------------------- #
# 11. slug=None guard — a payload missing its slug drops before any store read
# --------------------------------------------------------------------------- #

def test_none_slug_is_guarded_before_the_store_read(store: GraphStore) -> None:
    """A match with ``slug=None`` (vector_store emits it) drops with no ``get_node_by_slug`` call."""
    spy = _SpyStore(store)
    matches = [{"slug": None, "score": 0.9}]
    result = _gather("Axiom.", _FakeEmbed(), _FakeVector(matches), spy)

    assert result == []
    assert spy.slug_calls == []   # the guard fired before touching the store
    assert spy.state_calls == []


# --------------------------------------------------------------------------- #
# 12. Shared-shape structural pin — the reason enum 3b will extend
# --------------------------------------------------------------------------- #

def test_unavailable_reason_enum_exposes_the_two_substrate_reasons() -> None:
    """``ConflictUnavailableReason`` carries the 2a substrate reasons (3b adds judgment members)."""
    assert ConflictUnavailableReason.EMBEDDING.value == "embedding_unavailable"
    assert ConflictUnavailableReason.VECTOR_STORE.value == "vector_store_unavailable"
    # 1b. This string is a SHIPPED CONTRACT, not an internal label: it is also the
    # `check` degradation token, and the token is what lands in the persisted
    # `check_runs.degraded_reason` trend column. If the two ever drift, a trend
    # query and the enum disagree about the same event.
    assert ConflictUnavailableReason.COLLECTION_MISSING.value == "collection_missing"
    # The shared typed-degradation shape: reason + raw detail, frozen.
    u = Unavailable(reason=ConflictUnavailableReason.EMBEDDING, detail="boom")
    assert u.reason is ConflictUnavailableReason.EMBEDDING
    assert u.detail == "boom"
    with pytest.raises(Exception):
        u.reason = ConflictUnavailableReason.VECTOR_STORE  # frozen — no mutation


# --------------------------------------------------------------------------- #
# 13. The bucketing meta-test — a new reason must be CLASSIFIED, not just added
# --------------------------------------------------------------------------- #

def test_every_unavailable_reason_is_classified_into_exactly_one_bucket() -> None:
    """Both surfaces that bucket a reason do it as ``if semantic … else the judge``.

    That ``else`` is the hole: ``sync._notice_conflict_unavailable`` and the staged
    gate's token map would each file a new semantic member under "the judgment model
    did not respond" — silently, and confidently wrong. Nothing had ever iterated
    this enum before, so nothing could notice. Modelled on
    ``test_preflight_declares_a_disposition_for_every_store_code``: declare the
    class, then assert the declaration covers it.
    """
    semantic = set(SEMANTIC_SUBSTRATE_REASONS)
    judgment = set(JUDGMENT_REASONS)

    assert semantic | judgment == set(ConflictUnavailableReason)
    assert semantic & judgment == set()
    # Non-empty on both sides, so a future "classify everything as semantic" edit
    # cannot pass the coverage assertion by emptying the other bucket.
    assert semantic and judgment


def test_both_bucketing_surfaces_bind_the_shared_constant() -> None:
    """The two consumers read the constant rather than re-listing its members.

    A second hand-rolled ``(EMBEDDING, VECTOR_STORE)`` tuple anywhere would make the
    meta-test above pass while the surface still mis-files — the same
    grep-for-a-second-copy discipline as ``_KILL_EDGE_TYPES_SQL``.
    """
    import inspect

    from mitos import cli, sync

    for module in (sync, cli):
        source = inspect.getsource(module)
        assert "SEMANTIC_SUBSTRATE_REASONS" in source, (
            f"{module.__name__} no longer binds the shared membership constant"
        )
        assert "ConflictUnavailableReason.EMBEDDING,\n" not in source, (
            f"{module.__name__} re-lists the semantic-substrate members instead of "
            "binding SEMANTIC_SUBSTRATE_REASONS"
        )


def test_judgment_defect_environment_partition_is_exhaustive() -> None:
    """Every JUDGMENT_REASONS member is classified as either defect or environment."""
    from mitos.conflict import (
        JUDGMENT_DEFECT_REASONS,
        JUDGMENT_ENVIRONMENT_REASONS,
        JUDGMENT_REASONS,
    )

    defect = set(JUDGMENT_DEFECT_REASONS)
    environment = set(JUDGMENT_ENVIRONMENT_REASONS)
    assert defect | environment == set(JUDGMENT_REASONS)
    assert defect & environment == set()
    assert defect and environment
