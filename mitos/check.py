"""The corpus-audit engine's core — strong-edge pair screen (2a) + sweep assembly (2b).

This module is the home of ``mitos check``: the corpus-wide conflict audit that sweeps
every decision and judges the genuinely *undeclared* tensions between similar pairs. The
sync-time sensor (``conflict.py``) judges one proposal against its neighbours at write
time; the corpus audit has no privileged direction — it must read a relationship as a
fact about a *pair*, not about whichever node the sweep happened to iterate first.

Phase 2a landed the first two composition pieces: the **either-direction strong-edge
screen** that removes, id-natively, every pair the author already settled with a declared
strong relationship — so the LLM judge is never asked to second-guess a carve-out the
author was most careful to make. A relationship is stored once, as authored
(``source → target``), but "A narrows B" and "B is narrowed by A" are the same settled
decision, so the index records **both** endpoints and the screen is direction-free.

Phase 2b assembles the deterministic **corpus sweep** around that screen — the plan half
of the run engine (CHK-D2 steps 1–3): a run-start snapshot (:func:`snapshot_corpus` —
one ``get_active_decisions`` + one ``get_edges``, the index built once), a lazy per-node
sweep composing the shipped pipeline stages (:func:`iter_sweep`: gather → strong-edge
screen → ``screen_candidates``), corpus-wide exactly-once pair dedup with replayable
orientation (:func:`dedup_oriented_pairs` — the oriented proposal is the
lexicographically smaller content hash, independent of which side's sweep discovered the
pair), and judgment-sized batch grouping (:func:`group_judgment_batches`). Everything
here is deterministic and LLM-free: the run engine (2c) consumes these structures to
*plan* a run — and disclose its cost — before spending a single judgment token, and its
aggregate breaker is simply "stop consuming the generator".

The screen is **direct-edge-only, by design** (CHK-D2): it reads edges as stored, nothing
transitive. If ``B narrows A`` and then ``B' supersedes B``, the pair ``{A, B'}`` is
genuinely unexamined — the author never declared how the *new* ``B'`` relates to ``A`` —
so it must reach judgment. A lineage walk would bury that real, re-opened question under a
lapsed relationship; knowing when a relationship has lapsed is as load-bearing as knowing
when it holds.

Phase 2c composes everything above into the **run engine** — two stages split at the
load-bearing plan/execute seam: :func:`plan_corpus_check` (deterministic, zero LLM
contact, zero writes — sweep → dedup → reuse partition → the exact fresh-batch
disclosure count) strictly before :func:`execute_corpus_check` (the only spend site:
batched judgment via an injected judge, per-batch telemetry persistence, the novelty
partition). The seam is *structural*: the plan stage has no judge parameter, and the
execute stage's first argument is the :class:`CheckPlan` carrying the disclosure count —
no code path can reach the spend without holding the object that disclosed its cost
(CHK-D5). The engine renders nothing and decides no exit code — 3a maps the typed
:class:`CheckRunResult` to output + 0/1/2 through :func:`exit_code_for`.

Phase 2d adds the run's **honesty hardware** (CHK-D4) and its **memory** (CHK-D7): the
deterministic stale-index probe (:func:`probe_stale_index`) reads the pending-embeddings
Outbox at plan entry and execute exit — a backlog row below ``CHECK_STALE_RETRY_TOLERANCE``
gates the run partial (an index that never drained those nodes silently thinned the
sweep's recall), while a row at or above tolerance is a *disclosed coverage exclusion*
that never gates (the poison-row escape). One derivation site, :func:`run_degradations`,
feeds both :func:`exit_code_for` (degraded-2 dominates findings-1) and the persisted
``degraded_reason``, and :func:`check_run_row_from_result` derives every ``check_runs``
scalar from the same :class:`CheckRunResult` the report reads — the row and the report
can never disagree.

**Tier-1 leaf, same discipline as ``conflict.py``.** This module must never import a
higher-tier ``mitos`` module or a heavy dependency (``anthropic``, the Qdrant/genai
clients) at module scope — the judge arrives injected, never imported. ``check`` sits
*above* ``conflict`` in the import DAG (``check → conflict``, never the reverse); its
module-scope ``mitos`` imports are ``conflict`` plus the 2c-sanctioned sink edge
``check → telemetry`` (the sibling corpus store — ``ConflictCheckRow``/``JudgmentBatch``/
``CheckRunRow`` construction and the ``ReuseUnavailable`` fork) and the pure leaves
``errors`` (the ``DatabaseError`` half of the end-probe twin-catch — exception classes
only, zero calls), ``models`` (defensive ``model_id`` resolution) and the package
``__init__`` (``__version__`` stamping) — none of which drags a heavy dependency. A
dep-free import guard pins this (``test_importing_check_drags_no_heavy_dependency``);
the KD7 no-write lint walks this module's runtime-import closure with ``telemetry`` as
the one sanctioned boundary.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)
from uuid import uuid4

from mitos import __version__
from mitos.conflict import (
    CONFLICT_CANDIDATE_SOURCE,
    CONFLICT_PROMPT_VERSION,
    CONFLICT_SIMILARITY_FLOOR,
    CONFLICT_SURFACE_THRESHOLD,
    CONFLICT_TOP_K,
    Candidate,
    ConflictUnavailableReason,
    FIRST_ATTEMPT_JUDGMENT_REASONS,
    JudgmentExecution,
    RenderedPrompt,
    Unavailable,
    _STRONG_RELATIONSHIP_FIELDS,
    gather_candidates,
    judge_input_from_node,
    parse_judgment_response,
    render_judgment_prompt,
    screen_candidates,
)
from mitos.errors import DatabaseError
from mitos.models import get_model_id
from mitos.telemetry import (
    CheckRunRow,
    ConflictCheckRow,
    JudgmentBatch,
    ReuseIndex,
    ReuseUnavailable,
    StoredVerdict,
    TelemetryStore,
)

if TYPE_CHECKING:
    # Runtime-injected, duck-typed collaborators — annotated only for the type checker.
    # Importing ``mitos.protocols`` at runtime pulls ``parser`` + ``store`` (not a
    # leaf-cheap import), so these stay behind the guard (the conflict.py idiom).
    from mitos.protocols import EmbeddingProvider, GraphStoreProtocol, VectorStore


class StrongEdgeIndex:
    """An orientation-blind strong-edge adjacency: ``node_id → its strong partners``.

    Built once per run by :func:`build_strong_edge_index` from ``store.get_edges()`` and
    read per swept node via :meth:`partners` — the "build once, look up per node" seam
    Phase 2b threads into its run-start snapshot. Two nodes are partners iff a strong edge
    (any of the five :data:`_STRONG_RELATIONSHIP_FIELDS` types) joins them in *either*
    direction. The stored partner sets are frozen — the index is an immutable snapshot of
    the graph's strong-edge structure at build time.
    """

    def __init__(self, partners: Dict[str, "frozenset[str]"]) -> None:
        """Wraps a pre-built ``{node_id: frozenset(partner_ids)}`` adjacency.

        Args:
            partners: The fully-built adjacency; only nodes with at least one strong
                partner appear. Callers use :func:`build_strong_edge_index`, not this
                constructor directly.
        """
        self._partners = partners

    def partners(self, node_id: str) -> "frozenset[str]":
        """The strong-edge partners of ``node_id``, or ``∅`` for an unknown node.

        The empty-frozenset default is deliberate: a swept node that shares no strong
        edge (the healthy common case) is not an error, it simply has no partners. This
        keeps the lookup total so a consumer can never ``KeyError`` on a fresh node.

        Args:
            node_id: The content-hash id of the node to look up.

        Returns:
            The frozenset of partner node ids (empty if the node is absent).
        """
        return self._partners.get(node_id, frozenset())

    def __len__(self) -> int:
        """The number of nodes carrying at least one strong-edge partner."""
        return len(self._partners)


def build_strong_edge_index(edges: "List[Dict[str, str]]") -> StrongEdgeIndex:
    """One pass over the edge list → an orientation-blind strong-edge adjacency.

    For each edge whose ``edge_type`` is one of the five strong relationship types
    (:data:`_STRONG_RELATIONSHIP_FIELDS` — bound to the single source in ``conflict.py``,
    never a hand-listed lookalike), record **both** endpoints as partners of each other.
    This is what makes the screen direction-free: an edge authored ``source=X, target=Y``
    puts ``Y ∈ partners(X)`` *and* ``X ∈ partners(Y)``.

    Direct edges only — the index is built from edges verbatim, with **no** transitive or
    lineage closure (CHK-D2). A superseded declarer re-opens its pair by construction:
    ``{A, B'}`` where ``B narrows A`` and ``B' supersedes B`` has no direct strong edge, so
    it survives the screen and reaches judgment. A self-edge (``source_id == target_id``,
    which the write path rejects but a rebuilt graph could carry out-of-band) is skipped —
    a node is never its own strong partner; self-drop is ``screen_candidates``'s
    ``own_slug`` job downstream, not this screen's (mirrors ``get_contradictions``'s
    ``!= ?`` guard).

    Pure and store-free: takes a pre-fetched edge list so the logic is testable without a
    store. Phase 2b makes the single ``store.get_edges()`` call at run-start and passes the
    result here (one build, reused across the whole sweep). A ``get_edges()`` fault is the
    caller's to surface — this function does not mask it (there is no fallback empty index,
    which would silently screen nothing and mislabel every pair as fresh).

    Args:
        edges: The edge rows from ``store.get_edges()``; each a dict exposing
            ``source_id``, ``target_id``, and ``edge_type`` (extra keys ignored).

    Returns:
        The built :class:`StrongEdgeIndex`.
    """
    adjacency: Dict[str, Set[str]] = {}
    for edge in edges:
        if edge["edge_type"] not in _STRONG_RELATIONSHIP_FIELDS:
            continue
        source = edge["source_id"]
        target = edge["target_id"]
        if source == target:
            continue  # a node is never its own partner (own_slug covers self downstream)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return StrongEdgeIndex(
        {node_id: frozenset(partner_ids) for node_id, partner_ids in adjacency.items()}
    )


def screen_strong_edge_pairs(
    proposal_id: str,
    candidates: "List[Candidate]",
    index: StrongEdgeIndex,
) -> "List[Candidate]":
    """Drop every candidate that shares a strong edge with the swept node (either way).

    Given a swept node (``proposal_id``, a content hash) and its gathered candidates,
    remove each candidate whose ``node['id']`` is a strong-edge partner of ``proposal_id``
    in the index — the author already reasoned about that pair, so the audit stays silent
    and spends no judgment on it. Compares content-hash ids directly (never slugs — the
    slug is a mutable historical citation, M2).

    Order-preserving, pure, and storeless — no I/O, no state re-verify. Placed *upstream*
    of ``screen_candidates`` by the sweep (Phase 2b) so a declared strong neighbour is
    dropped before the similarity floor and the ``top_k`` truncation, and can never consume
    a slot that would shadow a genuine undeclared conflict out of the window (the CONF-D7
    drop-before-floor-before-truncate order).

    Args:
        proposal_id: The swept node's content-hash id.
        candidates: The gathered neighbours to screen (order preserved in the result).
        index: The strong-edge index built once at run-start.

    Returns:
        The candidates that share no strong edge with the swept node — the survivors that
        reach judgment (possibly empty; possibly the whole input unchanged).
    """
    partners = index.partners(proposal_id)
    return [candidate for candidate in candidates if candidate.node["id"] not in partners]


# --------------------------------------------------------------------------- #
# Phase 2b — the corpus sweep assembly (snapshot → lazy sweep → dedup → groups)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CorpusSnapshot:
    """The run-start snapshot: the sweep set + the strong-edge index, fixed together.

    One :func:`snapshot_corpus` call pins both in one place, so the whole run reads a
    single consistent view of the graph — mid-run commits are the next run's problem.
    ``len(snapshot.nodes)`` is the denominator for 2c's swept-vs-skipped accounting.

    Attributes:
        nodes: The sweep set — ``get_active_decisions(scope)``'s hydrated,
            modifier-stamped decision dicts (active ∪ drifted), in snapshot order.
            That order is a DB accident (the SELECT has no ORDER BY); nothing
            downstream may depend on it — output determinism comes from the
            hash-ordered post-passes, never from iteration order.
        edge_index: The either-direction strong-edge index, built ONCE from a single
            ``get_edges()`` pass and reused across every swept node.
    """

    nodes: Tuple[Dict[str, Any], ...]
    edge_index: StrongEdgeIndex


def snapshot_corpus(
    store: "GraphStoreProtocol", *, scope: Optional[str] = None
) -> CorpusSnapshot:
    """Snapshots the live sweep set and the strong-edge index at run start.

    Exactly two graph reads — one ``get_active_decisions(scope)`` for the sweep set,
    one ``get_edges()`` for the index — never re-read per node (the build-once seam
    the 2a handoff pins; a per-node rebuild would be O(edges) × O(nodes) for zero
    benefit). ``scope`` filters the *proposal set only* (CHK-D2/CONF-D2): a scoped
    snapshot sweeps only matching nodes, but their gathered partners may be any live
    decision — a scoped carve-out can fight a global, so candidate recall stays
    scope-blind downstream. A zero-match scope yields an empty snapshot: empty sweep,
    no pairs, no groups — healthy, not degraded (the "0 of N" wording is the CLI's).

    Store faults propagate (KD5): a failing ``get_edges()`` or
    ``get_active_decisions()`` raises here — no fallback empty index (which would
    silently screen nothing and mislabel every settled pair as fresh), no fallback
    empty sweep set (which would mislabel a broken graph as a clean corpus).

    Args:
        store: The graph store to snapshot.
        scope: An optional scope tag filtering the sweep set (passed through verbatim
            to ``get_active_decisions`` — folding is the CLI's concern).

    Returns:
        The :class:`CorpusSnapshot` the whole run iterates.
    """
    nodes = tuple(store.get_active_decisions(scope))
    edge_index = build_strong_edge_index(store.get_edges())
    return CorpusSnapshot(nodes=nodes, edge_index=edge_index)


def sweep_node(
    node: Dict[str, Any],
    *,
    edge_index: StrongEdgeIndex,
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    store: "GraphStoreProtocol",
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> "List[Candidate] | Unavailable":
    """Discovers ONE swept node's undeclared conflict candidates (the per-node stage).

    Composes the shipped pipeline stages in the pinned order — ``gather_candidates`` →
    :func:`screen_strong_edge_pairs` → ``screen_candidates`` — so the id-native
    declared-pair drop runs *upstream* of the floor and the ``top_k`` truncation
    (CONF-D7 shadowing: a declared strong neighbour must never consume a slot that
    would shadow a genuine undeclared conflict out of the window).

    The axiom passed to gather is ``node["core_axiom"]`` — a direct subscript, loud
    ``KeyError`` if the shape drifts (the phantom-empty foot-gun's loud half). Gather
    routes it through ``identity.embedding_text``, so the vector is byte-identical to
    what the outbox embedded for this node → the sweep's embeds are cache hits.
    ``declared_targets=set()`` is correct, not a gap: the strong-edge screen already
    did the declared drop id-natively, both directions; ``screen_candidates`` keeps
    the ``own_slug`` self-drop (a KNN echoing the swept node itself), the floor, and
    the rank/truncate.

    Args:
        node: The hydrated swept decision (a ``CorpusSnapshot.nodes`` element).
        edge_index: The run's strong-edge index (built once in the snapshot).
        embed_provider: The injected embedding provider.
        vector_store: The injected vector store.
        store: The injected graph store (gather's live re-verify source).
        floor: The inclusive similarity floor (default ``CONFLICT_SIMILARITY_FLOOR``).
        top_k: The cap on the surviving batch (default ``CONFLICT_TOP_K``).

    Returns:
        The surviving :class:`Candidate` list (possibly ``[]`` — healthy-empty), or
        the typed :class:`Unavailable` gather returned (degraded; passed through as a
        VALUE, verbatim — never blurred into ``[]``, never raised). Graph-store
        faults inside gather propagate (KD5) — only the two semantic-substrate faults
        degrade.
    """
    gathered = gather_candidates(
        node["core_axiom"],
        embed_provider=embed_provider,
        vector_store=vector_store,
        store=store,
    )
    if isinstance(gathered, Unavailable):
        return gathered
    undeclared = screen_strong_edge_pairs(node["id"], gathered, edge_index)
    return screen_candidates(
        undeclared,
        declared_targets=set(),
        own_slug=node["slug"],
        floor=floor,
        top_k=top_k,
    )


@dataclass(frozen=True)
class NodeSweep:
    """One swept node paired with its typed per-node outcome.

    ``result`` is a list (healthy — possibly empty) **or** the shipped
    :class:`~mitos.conflict.Unavailable` (degraded). Consumers fork on
    ``isinstance(result, Unavailable)``, never on emptiness — the same
    type-distinguished-degradation discipline as 1b's ``ReuseIndex`` vs
    ``ReuseUnavailable`` on the telemetry side.

    Attributes:
        node: The hydrated swept decision dict.
        result: ``List[Candidate]`` (healthy) or ``Unavailable`` (degraded).
    """

    node: Dict[str, Any]
    result: "List[Candidate] | Unavailable"


def iter_sweep(
    snapshot: CorpusSnapshot,
    *,
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    store: "GraphStoreProtocol",
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> "Iterator[NodeSweep]":
    """Lazily sweeps the snapshot, one typed :class:`NodeSweep` per node.

    A generator — laziness IS the breaker seam: no gather work happens for nodes
    beyond the last consumed yield, so the run engine's aggregate breaker needs no
    hook, no callback, no policy parameter. On the first :class:`Unavailable` it
    simply stops consuming and the remainder is structurally skipped — no post-trip
    node ever eats its own timeout (the one-penalty-per-run property, delivered by
    generator semantics instead of a flag). Consumed-count vs ``len(snapshot.nodes)``
    is the caller's swept-vs-skipped accounting.

    Yields in snapshot order (a DB accident — see :class:`CorpusSnapshot`); the
    hash-ordered post-passes make the plan structures order-independent.

    Args:
        snapshot: The run-start snapshot (sweep set + edge index).
        embed_provider: The injected embedding provider.
        vector_store: The injected vector store.
        store: The injected graph store.
        floor: The inclusive similarity floor, threaded to every per-node sweep.
        top_k: The per-node survivor cap, threaded likewise.

    Yields:
        One :class:`NodeSweep` per snapshot node, in snapshot order.
    """
    for node in snapshot.nodes:
        yield NodeSweep(
            node=node,
            result=sweep_node(
                node,
                edge_index=snapshot.edge_index,
                embed_provider=embed_provider,
                vector_store=vector_store,
                store=store,
                floor=floor,
                top_k=top_k,
            ),
        )


@dataclass(frozen=True)
class CorpusPair:
    """One deduped, oriented undeclared pair — the unit the judge will price and see.

    The oriented proposal is the **lexicographically smaller content hash** — a pure
    function of the pair, independent of which side's sweep discovered it, of sweep
    order, and of timestamps. That replayability is what lets verdict reuse (CHK-D3)
    key consistently run after run, and it is why the run engine stamps telemetry's
    ``judged_axiom``/``proposed_hash_if_any`` from the oriented proposal, never from
    discovery context. Both nodes ride whole (hydrated dicts): the engine needs live
    slugs for prompt rendering and report display (MI-2) plus the fed-context fields
    for telemetry — pre-projecting ``JudgeInput``\\ s here would duplicate what the
    nodes already carry.

    Attributes:
        proposal_hash: The lex-smaller content hash — THE oriented proposal.
        partner_hash: The other side's content hash.
        proposal_node: The proposal's hydrated node dict (either side may be the
            "undiscovering" one — same shape, same adapter).
        partner_node: The partner's hydrated node dict.
        score: The gathered similarity — informational context only, never an
            ordering key (two discoveries of one pair carry independently-computed
            floats; hash order is the only ordering anywhere in the plan).
    """

    proposal_hash: str
    partner_hash: str
    proposal_node: Dict[str, Any]
    partner_node: Dict[str, Any]
    score: float


def dedup_oriented_pairs(sweeps: "Iterable[NodeSweep]") -> "List[CorpusPair]":
    """Dedups the discovered pairs corpus-wide — each unordered pair exactly once.

    The dedup key is the unordered content-hash pair (``tuple(sorted((a, b)))`` — the
    same convention 1b's ``ReuseIndex`` keys on, so the engine's reuse lookup joins
    without adaptation). Ids come from ``node["id"]`` direct subscripts — real strings
    on every hydrated node, which is what keeps ``ReuseIndex.lookup``'s fail-loud
    ``None`` contract intact downstream.

    Degraded sweeps contribute zero pairs and raise nothing — but a pair the *other*
    side discovered with a degraded node still stands (partial coverage is labeled by
    the engine holding the same list, not compensated for here). When one pair is
    discovered from both sides, retention is **order-independent** (never keep-first):
    the discovery whose sweep node IS the oriented proposal wins, falling back to the
    partner-side discovery — deterministic whichever side the consumer iterated first,
    so the retained ``score`` and node dicts never depend on sweep order.

    Args:
        sweeps: The consumed :class:`NodeSweep`\\ s (a list or any iterable — the
            engine hands over exactly what it consumed before any breaker trip).

    Returns:
        The oriented pairs, sorted by pair key — a deterministic, replayable pure
        function of the healthy sweep results.
    """
    retained: Dict[Tuple[str, str], CorpusPair] = {}
    for sweep in sweeps:
        if isinstance(sweep.result, Unavailable):
            continue
        sweep_id = sweep.node["id"]
        for candidate in sweep.result:
            partner_id = candidate.node["id"]
            proposal_hash, partner_hash = sorted((sweep_id, partner_id))
            key = (proposal_hash, partner_hash)
            discovered_from_proposal = sweep_id == proposal_hash
            if not discovered_from_proposal and key in retained:
                continue  # the incumbent stands (proposal-side wins; first otherwise)
            if discovered_from_proposal:
                proposal_node, partner_node = sweep.node, candidate.node
            else:
                proposal_node, partner_node = candidate.node, sweep.node
            retained[key] = CorpusPair(
                proposal_hash=proposal_hash,
                partner_hash=partner_hash,
                proposal_node=proposal_node,
                partner_node=partner_node,
                score=candidate.score,
            )
    return [retained[key] for key in sorted(retained)]


@dataclass(frozen=True)
class JudgmentGroup:
    """One judgment-sized batch: an oriented proposal + its partner pairs (≤ top_k).

    The shape the run engine turns into one batched judgment call after the reuse
    partition removes the already-judged pairs (which is why grouping is a separate
    seam from dedup — reused pairs never enter a batch).

    Attributes:
        proposal_hash: The group's oriented proposal hash.
        proposal_node: The proposal's hydrated node dict.
        pairs: The group's :class:`CorpusPair`\\ s, partner-hash-sorted, ≤ ``top_k``.
    """

    proposal_hash: str
    proposal_node: Dict[str, Any]
    pairs: Tuple[CorpusPair, ...]


def group_judgment_batches(
    pairs: "Iterable[CorpusPair]",
    *,
    top_k: int = CONFLICT_TOP_K,
) -> "List[JudgmentGroup]":
    """Groups oriented pairs into judgment-sized batches — a pure, replayable pass.

    Groups are keyed and ordered by proposal hash; partners within a group are
    partner-hash-sorted; a group with more than ``top_k`` partners splits into
    consecutive chunks of ≤ ``top_k`` in that order. Hash order everywhere, score
    nowhere: two runs over the same snapshot and gather results produce
    byte-identical batch structures — replayability is worth more than mirroring the
    sync batch's similarity-descending cosmetic order (the judge sees the whole batch
    regardless).

    Args:
        pairs: The deduped oriented pairs (any iterable).
        top_k: The batch-size cap (default ``CONFLICT_TOP_K`` — the same window the
            sync-time judge sees).

    Returns:
        The :class:`JudgmentGroup`\\ s, proposal-hash-ordered (split chunks adjacent,
        in partner-hash order). ``len(...)`` of the post-reuse remainder is the exact
        batch count the CHK-D5 disclosure names.
    """
    by_proposal: Dict[str, List[CorpusPair]] = {}
    for pair in pairs:
        by_proposal.setdefault(pair.proposal_hash, []).append(pair)
    groups: List[JudgmentGroup] = []
    for proposal_hash in sorted(by_proposal):
        ordered = sorted(by_proposal[proposal_hash], key=lambda pair: pair.partner_hash)
        for start in range(0, len(ordered), top_k):
            chunk = tuple(ordered[start:start + top_k])
            groups.append(
                JudgmentGroup(
                    proposal_hash=proposal_hash,
                    proposal_node=chunk[0].proposal_node,
                    pairs=chunk,
                )
            )
    return groups


# --------------------------------------------------------------------------- #
# Phase 2c — the run engine (plan/execute seam, reuse partition, persistence)
# --------------------------------------------------------------------------- #

# CHK-D5 — above this many pending fresh batches an interactive run confirms before
# spending (3a's confirm flow reads this; forward wiring, unconsumed in-phase).
CHECK_CONFIRM_BATCHES = 10

# CHK-D4 — a stale-index backlog row retried at least this many times is a named
# coverage exclusion, not a transient gate (2d's probe partition reads this;
# forward wiring, unconsumed in-phase).
CHECK_STALE_RETRY_TOLERANCE = 3

# The isolation breaker: this many CONSECUTIVE first-attempt batch failures convict
# the remainder and abort the run. It is the evidence threshold that a per-batch
# failure class is in fact systematic — one truncated batch says something about that
# batch's content, three in a row say something about the prompt. Ladder-exhausted
# failures never reach it (they abort on the first one — they already carry that
# evidence, bought with ~183s of backoff).
#
# Deliberately low, because per-batch persistence makes an abort resumable: banked
# verdicts are reused on the next run, so a wrong abort costs one re-run of the
# remainder rather than the whole corpus. There is no abort-cost cliff to hedge
# against, and the hazard it bounds is real spend — 3 batches of discovery is ~$0.08
# where an unbounded continue on a systematic defect is a full corpus of judge calls
# (~$9) for zero verdicts.
_MAX_CONSECUTIVE_BATCH_FAILURES = 3


def _utc_now_iso() -> str:
    """One UTC ISO-8601 stamp — µs precision + ``+00:00`` offset (MI-10).

    A deliberate local twin of ``store._utc_now_iso`` (byte-compatible output):
    importing it would put a runtime ``check → store`` edge on the Tier-1 leaf and
    drag the graph committer into the KD7 no-write lint closure — one line is
    cheaper than either.
    """
    return datetime.now(timezone.utc).isoformat()


def _is_finding(tenable: bool, confidence: float) -> bool:
    """The ONE gate site (KD4): is a raw verdict a reportable finding?

    Applies the CONF-D4 formula — not tenable, at ``CONFLICT_SURFACE_THRESHOLD`` or
    above — identically to fresh :class:`~mitos.conflict.Judgment` values and reused
    :class:`~mitos.telemetry.StoredVerdict` rows. Finding-ness is always re-derived
    from the raw ``tenable`` + ``confidence`` at read time; the stored ``surfaced``
    flag records only what the *writing* run reported and is never read back
    (§8's widened semantics — 1b deliberately does not even project it).

    Args:
        tenable: The judge's raw verdict (fresh ``tenable_together`` or stored
            ``tenable``).
        confidence: The judge's raw self-reported confidence in ``[0, 1]``.

    Returns:
        True iff this verdict surfaces as a finding.
    """
    return (not tenable) and confidence >= CONFLICT_SURFACE_THRESHOLD


# --------------------------------------------------------------------------- #
# Phase 2d — the stale-index probe (CHK-D4 honesty hardware)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BacklogRow:
    """One pending-embeddings Outbox row, typed for the probe/report surface.

    Attributes:
        node_id: The backlogged node's content hash (== ``nodes.id`` — 3a's
            report resolves a live display slug from it, MI-2).
        queued_at: The row's UTC ISO-8601 enqueue stamp (MI-10).
        retry_count: Drain attempts so far — the partition key against
            ``CHECK_STALE_RETRY_TOLERANCE``.
    """

    node_id: str
    queued_at: str
    retry_count: int


@dataclass(frozen=True)
class StaleProbe:
    """One deterministic read of the embedding backlog, partitioned at the tolerance.

    An undrained Outbox row means the vector index never received that node — a
    reachable-but-behind index produces no typed degradation downstream
    (``gather_candidates`` succeeds and silently thins recall), so this probe is
    the only place the thinning becomes visible (CHK-D4). The partition is
    computed once, at probe time; gating is derived at read time from the raw
    rows (KD3). An empty probe (``StaleProbe((), ())``) is HEALTHY — a drained
    backlog, never a degradation.

    The probe is conservative by design: a payload-only re-sync row is
    indistinguishable from a missing re-embed in the 3-column Outbox (no intent
    discriminator), so it may over-fire — the fail-closed price, disclosed loudly
    rather than discovered as a thinned audit. A ``--scope``-narrowed run still
    gates on the corpus-global backlog: candidate recall is scope-blind (CONF-D2),
    so a backlogged node outside the scope could still have surfaced as a
    candidate.

    Attributes:
        transient: Rows with ``retry_count < CHECK_STALE_RETRY_TOLERANCE`` —
            expected to drain on the next ``mitos sync``; their presence GATES
            the run partial (the ``"stale_index"`` degradation).
        excluded: Rows at or above the tolerance — chronic failures the drain
            keeps retrying (poison rows). Disclosed as named coverage
            exclusions, they NEVER gate: without this escape one poison row
            would turn the gate red forever and a pre-commit hook would
            permanently block (the CHK-D4 escape). The durable fix (an Outbox
            dead-letter) is substrate-owned and out of this vision's scope.
    """

    transient: Tuple[BacklogRow, ...]
    excluded: Tuple[BacklogRow, ...]


@dataclass(frozen=True)
class ProbeUnavailable:
    """Typed degradation: the END-probe read itself failed — stored, never raised.

    Mirrors :class:`~mitos.telemetry.ReuseUnavailable`'s posture (check-local by
    design — ``conflict.Unavailable``'s reason enum is pipeline-substrate
    vocabulary; this leaf mints no members from outside it). Only the end probe
    degrades to this: at plan entry nothing is spent and a store fault
    propagates (KD2), but at execute exit the judgment spend has happened and
    the findings must survive. An end probe that could not read cannot certify
    completeness → the ``"probe_read"`` degradation, exit 2, findings intact,
    ``coverage_exclusions`` NULL. This degrades to *loudly-uncertifiable*, never
    to a silently-healthy empty probe — the opposite failure direction from the
    fallback the KD5-2a rule forbids.

    Attributes:
        detail: The underlying exception message — disclosure/logging only.
    """

    detail: str


def probe_stale_index(store: "GraphStoreProtocol") -> StaleProbe:
    """Reads the pending-embeddings backlog once and partitions it at the tolerance.

    One ``store.get_pending_embeddings()`` read (the GRAPH store — telemetry
    reads stay one-per-run, §6.2), partitioned on ``retry_count`` against
    ``CHECK_STALE_RETRY_TOLERANCE`` and node-id-sorted in both halves, so two
    probes over an unchanged backlog are byte-identical (DoD-3 determinism).
    Public: 3b composes it directly around the staged facade, outside this
    engine's two call sites.

    Row fields are read by direct key — a schema drift is a loud ``KeyError``,
    never a ``.get(..., 0)`` (a silently-zero ``retry_count`` would turn poison
    into transient and resurrect the permanent-red-gate bug the tolerance
    exists to kill). There is deliberately NO try/except here: the two call
    sites dispose of faults differently (KD2 — plan propagates, execute wraps),
    so the caller owns the disposition.

    Args:
        store: The graph store (the injected duck-typed collaborator; only
            ``get_pending_embeddings`` is read).

    Returns:
        The partitioned, sorted :class:`StaleProbe`. Empty backlog → healthy
        ``StaleProbe((), ())``.
    """
    transient: List[BacklogRow] = []
    excluded: List[BacklogRow] = []
    for raw in store.get_pending_embeddings():
        row = BacklogRow(
            node_id=raw["node_id"],
            queued_at=raw["queued_at"],
            retry_count=raw["retry_count"],
        )
        if row.retry_count >= CHECK_STALE_RETRY_TOLERANCE:
            excluded.append(row)
        else:
            transient.append(row)
    return StaleProbe(
        transient=tuple(sorted(transient, key=lambda row: row.node_id)),
        excluded=tuple(sorted(excluded, key=lambda row: row.node_id)),
    )


@dataclass(frozen=True)
class ReusedPair:
    """One pair whose judgment is reused from a prior verdict (CHK-D3 — zero spend).

    The verdict is the RAW stored row (no pre-derived finding-ness): the engine
    applies the one KD4 gate to it at result assembly, exactly as it does to a
    fresh judgment — so the standing report always re-derives from primary values,
    never from the historical ``surfaced`` flag.

    Attributes:
        pair: The oriented corpus pair the verdict covers.
        verdict: The latest prior verdict at the run's pins, verbatim (M8).
    """

    pair: CorpusPair
    verdict: StoredVerdict


@dataclass(frozen=True)
class CheckPlan:
    """The deterministic plan half of a corpus check — cost disclosed, nothing spent.

    Produced by :func:`plan_corpus_check` with zero LLM contact and zero writes;
    consumed by :func:`execute_corpus_check`, which cannot be reached without it
    (KD1 — the seam is a property of the type graph, not a discipline).
    ``len(fresh_groups)`` is the exact CHK-D5 disclosure count the CLI confirms.

    Attributes:
        run_id: The CHK-D7 correlation id (``uuid4().hex``) — stamped as
            ``sync_run_id`` on every row this run persists.
        started_at: The MI-10 stamp taken at plan entry (2d's ``check_runs`` seam).
        model_alias: The judge pin the reuse index was loaded at; execute re-asserts
            every execution against it (KD5). Sourcing is the caller's (KD3): 3a
            threads ``conflict_judgment._JUDGMENT_MODEL_ALIAS`` — this leaf cannot
            import the executor module.
        prompt_version: The prompt pin, ditto (defaults to the production
            ``CONFLICT_PROMPT_VERSION``).
        fresh: Whether the reuse partition was bypassed (``--fresh``). The index
            load and the novelty read are NEVER bypassed — a ``--fresh``
            re-confirmation of a standing finding stays "known".
        nodes_total: ``len(snapshot.nodes)`` — the swept-vs-skipped denominator.
        nodes_swept: Healthy sweeps consumed before any trip.
        sweep_degraded: The tripping sweep degradation, if any (KD2 — the partial
            plan still executes; the substrates are disjoint).
        pairs: Every deduped oriented pair, pair-key-sorted.
        reused: The pairs with a prior verdict at the pins (empty under ``fresh``).
        fresh_groups: The spend units — ``len()`` IS the disclosure count.
        reuse_index: The pre-run index (the CHK-D10 novelty boundary), or ``None``
            when unavailable. Loaded ONCE, before any of this run's writes exist —
            never reloaded (a mid-run reload would read this run's own rows back
            as "known").
        reuse_unavailable: The typed read degradation, if the load failed (CHK-D3/
            D10 — the run proceeds all-fresh and reports unpartitioned).
        start_probe: The run-entry backlog read (CHK-D4), taken BEFORE the corpus
            snapshot — the backlog that predates the snapshot is exactly what
            thinned the sweep the snapshot defines (KD1).
    """

    run_id: str
    started_at: str
    model_alias: str
    prompt_version: str
    fresh: bool
    nodes_total: int
    nodes_swept: int
    sweep_degraded: Optional[Unavailable]
    pairs: Tuple[CorpusPair, ...]
    reused: Tuple[ReusedPair, ...]
    fresh_groups: Tuple[JudgmentGroup, ...]
    reuse_index: Optional[ReuseIndex]
    reuse_unavailable: Optional[ReuseUnavailable]
    start_probe: StaleProbe


@dataclass(frozen=True)
class CheckFinding:
    """One reported contradiction — fresh or reused, partitioned by novelty.

    Attributes:
        proposal_hash: The oriented pair identity, proposal side (M2).
        partner_hash: The oriented pair identity, partner side.
        proposal_node: The proposal's hydrated live-at-snapshot node (3a resolves
            display slugs from here — MI-2, the stored slug is a citation).
        partner_node: The partner's hydrated node, ditto.
        score: The gathered similarity — informational context (2b's carried float).
        rationale: Fresh — this run's judgment rationale; reused — the stored
            verdict's, verbatim (M8: never a re-render).
        confidence: The raw judge confidence behind the gate.
        reused: Whether this finding rode a prior verdict (zero spend).
        source_batch_id: Fresh — this run's batch; reused — the prior row's batch
            (CHK-D3 provenance).
        source_created_at: The provenance timestamp, same split.
        novelty: ``"new"`` / ``"known"`` (CHK-D10), or ``None`` when the run could
            not tell (reuse read unavailable — unpartitioned; the exit-1 gate never
            fires on such a run, 3a exits 2).
    """

    proposal_hash: str
    partner_hash: str
    proposal_node: Dict[str, Any]
    partner_node: Dict[str, Any]
    score: float
    rationale: str
    confidence: float
    reused: bool
    source_batch_id: str
    source_created_at: str
    novelty: Optional[str]


@dataclass(frozen=True)
class CheckRunResult:
    """The typed outcome of one corpus check — counts, findings, degradations.

    The engine computes partitions and accounting; it renders nothing, colors
    nothing, maps no exit code — 3a owns presentation and the 0/1/2 contract, 2d
    the ``check_runs`` summary row (its scalars come from here; ``findings_new`` /
    ``findings_known`` derive from ``findings[i].novelty`` — no duplicated counts).

    Attributes:
        run_id: Echoed from the plan (CHK-D7).
        started_at: Echoed plan-entry stamp.
        ended_at: The MI-10 stamp taken at execute exit.
        nodes_total: Echoed plan accounting.
        nodes_swept: Echoed plan accounting.
        sweep_degraded: Echoed plan-stage trip, if any.
        findings: Every reported finding, pair-key-ordered; the novelty partition
            rides ``.novelty``.
        pairs_judged_fresh: Pairs whose fresh verdicts landed (parsed healthy).
        pairs_reused: ``len(plan.reused)``.
        batches_planned: ``len(plan.fresh_groups)``.
        batches_executed: Batches on which a judge call was fired — includes a
            batch whose execution or parse then failed (billed but unpersisted is
            a named cost, not a silent drop).
        batches_failed: Executed batches that persisted no verdicts — the isolated
            failures plus the aborting one. ``batches_executed - batches_failed``
            is the coverage numerator (:attr:`batches_judged`).
        batches_skipped: Batches never rendered or judged, because an abort-class
            failure stopped the remainder.
        judgment_failures: EVERY judgment-stage degradation this run recorded, in
            occurrence order — the isolated per-batch failures, the aborting one,
            and the judge-absent degradation. Non-empty ⇒ the run is degraded
            (:func:`run_degradations` reads it), whether or not it aborted.
        judgment_abort: The failure that stopped the remaining batches, or ``None``
            when the loop ran to the end. Always the LAST element of
            ``judgment_failures`` when set.
        reuse_unavailable: Echoed from the plan.
        telemetry_write_failures: Per-batch write-failure details (KD6 — the run
            is degraded but the judgment loop never aborts; findings still report).
        start_probe: Echoed from the plan (CHK-D4) — the run-entry backlog read.
        end_probe: The certification read, taken at execute exit: a
            :class:`StaleProbe` when readable, or :class:`ProbeUnavailable` when
            the read failed (KD2 — the paid-for findings survive, the run just
            cannot certify). Fork on the TYPE, never on emptiness — an empty
            probe is healthy. Completeness is certified only when BOTH probes
            read clean of transient rows.
    """

    run_id: str
    started_at: str
    ended_at: str
    nodes_total: int
    nodes_swept: int
    sweep_degraded: Optional[Unavailable]
    findings: Tuple[CheckFinding, ...]
    pairs_judged_fresh: int
    pairs_reused: int
    batches_planned: int
    batches_executed: int
    batches_failed: int
    batches_skipped: int
    judgment_failures: Tuple[Unavailable, ...]
    judgment_abort: Optional[Unavailable]
    reuse_unavailable: Optional[ReuseUnavailable]
    telemetry_write_failures: Tuple[str, ...]
    start_probe: StaleProbe
    end_probe: "StaleProbe | ProbeUnavailable"

    @property
    def judgment_degraded(self) -> Optional[Unavailable]:
        """The FIRST judgment failure, or ``None`` when judgment stayed healthy.

        The shipped spelling of "did the judgment stage degrade, and on what" —
        kept as a derived view once ``judgment_failures`` went plural, so a reader
        asking that question is never handed only the aborting failure (under
        isolation an earlier, isolated failure is the more informative one).
        """
        return self.judgment_failures[0] if self.judgment_failures else None

    @property
    def batches_judged(self) -> int:
        """Batches whose verdicts parsed and persisted — the coverage numerator.

        The disclosed "judged N of M batches" number (3a renders it; the ``--json``
        receipt carries both halves). Derived here rather than counted in the loop
        so it can never disagree with the executed/failed accounting.
        """
        return self.batches_executed - self.batches_failed


def _finding(
    *,
    pair: CorpusPair,
    rationale: str,
    confidence: float,
    reused: bool,
    source_batch_id: str,
    source_created_at: str,
    reuse_index: Optional[ReuseIndex],
) -> CheckFinding:
    """Assembles one finding, deriving its CHK-D10 novelty from the pre-run index.

    The partition rule (KD5): a finding is **known** iff the pre-run index holds a
    verdict for its pair that re-derives as a finding through the one KD4 gate —
    for a reused finding that lookup is its own source row, so "known" holds by
    construction; a first-ever pair, or a pair whose prior verdict was tenable/
    below-threshold, is **new**. With the index unavailable the finding is
    unpartitioned (``novelty=None``) — a run that cannot tell new from known must
    not pretend to (3a exits 2, the exit-1 gate never fires).

    Args:
        pair: The oriented pair this finding reports.
        rationale: The verdict rationale riding the finding (M8-verbatim).
        confidence: The raw confidence behind the gate.
        reused: Whether the verdict was reused (CHK-D3).
        source_batch_id: The provenance batch id (this run's or the prior row's).
        source_created_at: The provenance timestamp, same split.
        reuse_index: The plan's pre-run index, or ``None`` when unavailable.

    Returns:
        The assembled :class:`CheckFinding`.
    """
    if reuse_index is None:
        novelty: Optional[str] = None
    else:
        prior = reuse_index.lookup(pair.proposal_hash, pair.partner_hash)
        known = prior is not None and _is_finding(prior.tenable, prior.confidence)
        novelty = "known" if known else "new"
    return CheckFinding(
        proposal_hash=pair.proposal_hash,
        partner_hash=pair.partner_hash,
        proposal_node=pair.proposal_node,
        partner_node=pair.partner_node,
        score=pair.score,
        rationale=rationale,
        confidence=confidence,
        reused=reused,
        source_batch_id=source_batch_id,
        source_created_at=source_created_at,
        novelty=novelty,
    )


def plan_corpus_check(
    *,
    store: "GraphStoreProtocol",
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    telemetry: Optional[TelemetryStore],
    model_alias: str,
    prompt_version: str = CONFLICT_PROMPT_VERSION,
    scope: Optional[str] = None,
    fresh: bool = False,
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> CheckPlan:
    """Plans a corpus check — deterministic, zero LLM contact, zero writes (KD1).

    The 2b wiring contract, composed: mint the run id + the MI-10 ``started_at``
    stamp → the start probe (:func:`probe_stale_index`, BEFORE the snapshot —
    KD1; a store fault here propagates) → :func:`snapshot_corpus` → load the
    reuse index ONCE → drive
    :func:`iter_sweep`, stopping at the first :class:`Unavailable` (laziness IS the
    breaker — nothing beyond the trip is gathered, no post-trip node eats its own
    timeout) → :func:`dedup_oriented_pairs` → the reuse partition →
    :func:`group_judgment_batches`. There is no judge parameter anywhere in this
    stage — the spend site is structurally unreachable before the disclosure count
    exists.

    The one telemetry contact is the single bulk ``load_reuse_index`` at the pins
    ``{prompt_version, model_alias}`` — before any of this run's writes exist, so
    the index holds only pre-run rows and IS the CHK-D10 novelty boundary for free.
    A ``telemetry`` of ``None`` (the store never constructed) is the same typed
    read degradation as a failed load: the run proceeds all-fresh and its findings
    report unpartitioned. A sweep trip does NOT abort planning (KD2): the partial
    plan carries every pair the healthy prefix discovered, labeled by
    ``nodes_swept < nodes_total`` + ``sweep_degraded``.

    Graph-store faults propagate (2a KD5) — a broken graph must never masquerade
    as a clean corpus.

    Args:
        store: The graph store (snapshot + gather's live re-verify source).
        embed_provider: The injected embedding provider.
        vector_store: The injected vector store.
        telemetry: The sibling telemetry store, or ``None`` when it could not be
            constructed (reuse/novelty degrade typed; nothing raises).
        model_alias: The judge's family+tier alias — the reuse pin. The single
            production source stays ``conflict_judgment._JUDGMENT_MODEL_ALIAS``
            (KD3): the CLI threads it in; this leaf never imports the executor.
        prompt_version: The prompt pin (production default). A plan pinned at any
            other version cannot execute — the renderer always stamps the
            production version and execute's KD5 guard refuses the mismatch.
        scope: Optional scope tag filtering the sweep set (proposal set only —
            candidate recall stays scope-blind, CONF-D2).
        fresh: Bypass the reuse partition (every pair is judged). The index load
            and the novelty read are never bypassed.
        floor: The similarity floor threaded to every per-node sweep.
        top_k: The per-node survivor cap and the batch-size cap.

    Returns:
        The :class:`CheckPlan` — ``len(plan.fresh_groups)`` is the exact CHK-D5
        disclosure count.
    """
    run_id = uuid4().hex
    started_at = _utc_now_iso()
    # The start probe fires BEFORE the snapshot (KD1): a backlog row present here
    # predates the corpus the sweep will define, so the probe can never miss a row
    # the sweep suffers from. Faults propagate — nothing is spent yet, and a store
    # broken at entry must fail fast, not masquerade as a clean corpus (KD2).
    start_probe = probe_stale_index(store)
    snapshot = snapshot_corpus(store, scope=scope)

    if telemetry is None:
        loaded: "ReuseIndex | ReuseUnavailable" = ReuseUnavailable(
            "telemetry store unavailable (never constructed)"
        )
    else:
        loaded = telemetry.load_reuse_index(
            prompt_version=prompt_version, model_alias=model_alias
        )
    if isinstance(loaded, ReuseUnavailable):
        reuse_index: Optional[ReuseIndex] = None
        reuse_unavailable: Optional[ReuseUnavailable] = loaded
    else:
        reuse_index = loaded
        reuse_unavailable = None

    consumed: List[NodeSweep] = []
    sweep_degraded: Optional[Unavailable] = None
    for sweep in iter_sweep(
        snapshot,
        embed_provider=embed_provider,
        vector_store=vector_store,
        store=store,
        floor=floor,
        top_k=top_k,
    ):
        consumed.append(sweep)
        if isinstance(sweep.result, Unavailable):
            sweep_degraded = sweep.result
            break
    nodes_swept = len(consumed) - (1 if sweep_degraded is not None else 0)

    pairs = tuple(dedup_oriented_pairs(consumed))

    # The reuse partition (CHK-D3): a pair with a prior verdict at these exact pins
    # never re-enters a batch. ``fresh`` bypasses THIS partition only — the index
    # stays on the plan for the novelty read (a --fresh re-confirmation is "known").
    reused: List[ReusedPair] = []
    fresh_pairs: List[CorpusPair] = []
    for pair in pairs:
        verdict = None
        if reuse_index is not None and not fresh:
            verdict = reuse_index.lookup(pair.proposal_hash, pair.partner_hash)
        if verdict is not None:
            reused.append(ReusedPair(pair=pair, verdict=verdict))
        else:
            fresh_pairs.append(pair)

    return CheckPlan(
        run_id=run_id,
        started_at=started_at,
        model_alias=model_alias,
        prompt_version=prompt_version,
        fresh=fresh,
        nodes_total=len(snapshot.nodes),
        nodes_swept=nodes_swept,
        sweep_degraded=sweep_degraded,
        pairs=pairs,
        reused=tuple(reused),
        fresh_groups=tuple(group_judgment_batches(fresh_pairs, top_k=top_k)),
        reuse_index=reuse_index,
        reuse_unavailable=reuse_unavailable,
        start_probe=start_probe,
    )


def execute_corpus_check(
    plan: CheckPlan,
    *,
    judge: Optional[Callable[[RenderedPrompt], "JudgmentExecution | Unavailable"]],
    telemetry: Optional[TelemetryStore],
    store: "GraphStoreProtocol",
    env: Optional[Mapping[str, str]] = None,
) -> CheckRunResult:
    """Executes a planned corpus check — the only spend site, persisting per batch.

    Reused verdicts report first (zero spend, provenance riding the stored row).
    Then, per fresh group in plan order: build both sides' ``JudgeInput`` via the
    node adapter → render → ``judge(prompt)`` → parse → the one KD4 gate per pair →
    **persist THIS batch now** (P5 — a killed run loses nothing already judged; the
    re-run's reuse partition absorbs the persisted prefix).

    A batch-level :class:`Unavailable` — an executor error/timeout OR an
    all-or-nothing parse malformation — is dispositioned by CLASS, on how much the
    failure says about the batches after it (a malformed batch is billed but
    unpersisted either way, a named cost):

    * A reason in ``conflict.FIRST_ATTEMPT_JUDGMENT_REASONS`` is **isolated**: the
      batch is counted in ``batches_failed``, the run is degraded, and the loop
      keeps judging. The executor decided it from the one response that came back
      and never consulted its retry ladder, so it is evidence about this batch's
      content — not about the next batch. Isolating it banks the verdicts a trip
      would have discarded.
    * Any other reason **aborts** on the first occurrence: it exhausted the
      executor's retry ladder (~183s across nine attempts), which is evidence the
      cause is systematic, and 359 more batches would buy the same outcome at 359×
      the wait.
    * ``_MAX_CONSECUTIVE_BATCH_FAILURES`` consecutive isolated failures abort too —
      the threshold at which a per-batch class has shown itself to be systematic
      after all. A judged batch clears the streak.

    On an abort the remaining groups are never rendered or judged
    (``batches_skipped``), and ``judgment_abort`` names the failure that stopped
    it; ``judgment_failures`` carries every failure either way.

    A telemetry write failure degrades the RUN, never the loop (KD6): the
    failure is recorded per batch, the judgment still reports as findings, and
    later batches are still judged AND their writes attempted (each write is
    independent — a transient lock may clear).

    ``judge=None`` is lazy availability (P14): healthy when nothing fresh is
    pending (a reuse-only/empty run needs no key), a typed judgment degradation
    when fresh groups exist (zero spend, reused findings unaffected).

    After the batch loop the END probe fires (:func:`probe_stale_index` over the
    required ``store``) — the certification read: a row enqueued after the start
    probe (a commit landing mid-run) is caught here, so completeness is certified
    only when both reads come back clean (CHK-D4). Its fault disposition is the
    OPPOSITE of the start probe's (KD2): the judgment spend has happened and
    per-batch rows are on disk, so a read failure degrades typed to
    :class:`ProbeUnavailable` — catching BOTH raw ``sqlite3.Error`` (the
    unwrapped query path) and Mitos's ``DatabaseError`` (the wrapped open path);
    catch one and the other escapes with the findings. ``store`` is required, not
    defaulted: an optional probe source would let a caller silently skip
    certification — fail-closed must not be opt-in (KD1).

    KD5 join-key guards — the data-level half of the seam: every render must carry
    ``plan.prompt_version`` and every execution must stamp ``plan.model_alias``,
    else this raises ``ValueError`` before anything persists. A plan partitioned
    at one pin must never write rows at another — that mismatch would poison the
    reuse corpus for every later run, which is worse than a dead run.

    Args:
        plan: The disclosed plan (the only path here — KD1).
        judge: The injected judgment callable (the ``make_judgment_executor``
            shape), or ``None`` when unavailable. This leaf never constructs one.
        telemetry: The sibling store for per-batch persistence, or ``None`` (each
            executed batch then records one write failure — fail-closed
            disclosure, CHK-D7: an audit that cannot record its provenance is
            incomplete).
        store: The graph store the end probe reads (REQUIRED — certification is
            not opt-in). Only ``get_pending_embeddings`` is touched here.
        env: The calling workspace's resolved environment (``config.env``), used
            for the one thing this leaf cannot pre-resolve: ``execution
            .model_alias`` exists only inside the batch loop, so the *map*
            travels here rather than a resolved id. Its join to the id the judge
            actually called with is the KD5 paragraph's data-level sibling — both
            sides must resolve against the same map or the provenance column
            records a model the run never used, silently.

    Returns:
        The typed :class:`CheckRunResult` — findings pair-key-ordered, every
        degradation surfaced, accounting exact under every failure combination.

    Raises:
        ValueError: On a KD5 pin mismatch (prompt or alias) — a programming/wiring
            error, never a substrate failure.
    """
    findings: List[CheckFinding] = []
    write_failures: List[str] = []
    judgment_failures: List[Unavailable] = []
    judgment_abort: Optional[Unavailable] = None
    consecutive_failures = 0
    batches_executed = 0
    batches_failed = 0
    batches_skipped = 0
    pairs_judged_fresh = 0

    def record_batch_failure(failure: Unavailable) -> None:
        """Files one batch failure — isolating it, or tripping the abort."""
        nonlocal judgment_abort, batches_failed, consecutive_failures
        judgment_failures.append(failure)
        batches_failed += 1
        consecutive_failures += 1
        if (
            failure.reason not in FIRST_ATTEMPT_JUDGMENT_REASONS
            or consecutive_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES
        ):
            judgment_abort = failure

    # Reused verdicts first — findings at zero spend. The gate re-derives from the
    # raw stored verdict (KD4); a tenable or below-threshold prior stays silent.
    for reused_pair in plan.reused:
        verdict = reused_pair.verdict
        if not _is_finding(verdict.tenable, verdict.confidence):
            continue
        findings.append(
            _finding(
                pair=reused_pair.pair,
                rationale=verdict.rationale,
                confidence=verdict.confidence,
                reused=True,
                source_batch_id=verdict.batch_id,
                source_created_at=verdict.created_at,
                reuse_index=plan.reuse_index,
            )
        )

    if judge is None and plan.fresh_groups:
        # Not a batch failure — no batch was ever attempted, so it bypasses
        # `record_batch_failure` (nothing to isolate) and aborts outright.
        judgment_abort = Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT,
            detail=(
                f"no judge available for {len(plan.fresh_groups)} pending fresh "
                "batch(es); fresh judgment skipped, reused findings unaffected"
            ),
        )
        judgment_failures.append(judgment_abort)

    for group in plan.fresh_groups:
        if judgment_abort is not None:
            batches_skipped += 1  # never rendered, never judged — the abort holds
            continue

        proposal_input = judge_input_from_node(group.proposal_node)
        partner_slugs = [pair.partner_node["slug"] for pair in group.pairs]
        partner_inputs = [
            judge_input_from_node(pair.partner_node) for pair in group.pairs
        ]
        prompt = render_judgment_prompt(
            proposal_input, list(zip(partner_slugs, partner_inputs))
        )
        # KD5, prompt half — before any spend: the renderer always stamps the
        # production version, so a synthetic-pin plan dies HERE, judge untouched.
        if prompt.prompt_version != plan.prompt_version:
            raise ValueError(
                f"prompt_version mismatch: plan pinned {plan.prompt_version!r} but "
                f"the renderer produced {prompt.prompt_version!r} — a plan "
                "partitioned at one pin must never execute at another"
            )

        execution = judge(prompt)
        batches_executed += 1
        if isinstance(execution, Unavailable):
            record_batch_failure(execution)  # isolated, or the abort — by class
            continue
        # KD5, alias half — before parse/persist: rows at a pin the partition was
        # not computed at would poison every later run's reuse join.
        if execution.model_alias != plan.model_alias:
            raise ValueError(
                f"model_alias mismatch: plan pinned {plan.model_alias!r} but the "
                f"execution stamped {execution.model_alias!r} — refusing to "
                "persist rows at the wrong reuse pin"
            )

        judgments = parse_judgment_response(execution.raw_text, partner_slugs)
        if isinstance(judgments, Unavailable):
            record_batch_failure(judgments)  # billed but unpersisted — a named cost
            continue

        consecutive_failures = 0  # a judged batch clears the streak

        # One MI-10 stamp per batch, taken at persist time — shared by the batch's
        # rows and by its fresh findings' provenance.
        created_at = _utc_now_iso()
        try:
            model_id: Optional[str] = get_model_id(execution.model_alias, env)
        except ValueError:
            model_id = None  # provenance-only column — degrade, never lose the batch
        batch = JudgmentBatch(
            batch_id=execution.batch_id,
            model_id=model_id,
            token_input=execution.token_input,
            token_output=execution.token_output,
            token_cache_read=execution.token_cache_read,
            token_cache_creation=execution.token_cache_creation,
            elapsed_ms=execution.elapsed_ms,
            stop_reason=execution.stop_reason,
        )
        rows: List[ConflictCheckRow] = []
        for pair, partner_input, judgment in zip(
            group.pairs, partner_inputs, judgments
        ):
            surfaced = _is_finding(judgment.tenable_together, judgment.confidence)
            rows.append(
                ConflictCheckRow(
                    batch_id=execution.batch_id,
                    sync_run_id=plan.run_id,
                    # This writer IS the check surface — stamped explicitly on
                    # every row, never left to the schema DEFAULT (CHK-D7).
                    surface="check",
                    judged_axiom=proposal_input.axiom,
                    # MI-9: empty proposal rejected_paths ("") / scope ([]) is
                    # NULL, never "" (mirrors sync.py's mapping verbatim).
                    proposal_rejected_paths=proposal_input.rejected_paths or None,
                    proposal_scope=", ".join(proposal_input.scope) or None,
                    proposed_hash_if_any=group.proposal_hash,
                    candidate_slug=pair.partner_node["slug"],  # verbatim, no casefold
                    candidate_hash=pair.partner_hash,
                    # NOT NULL (M5): the raw str, even the degenerate "".
                    candidate_rejected_paths=partner_input.rejected_paths,
                    candidate_scope=", ".join(partner_input.scope) or None,
                    tenable=judgment.tenable_together,
                    confidence=judgment.confidence,
                    surfaced=surfaced,
                    candidate_source=CONFLICT_CANDIDATE_SOURCE,
                    model_alias=execution.model_alias,
                    # From the render that produced the judgment, never a
                    # re-imported literal — row provenance stays tied to it.
                    prompt_version=prompt.prompt_version,
                    mitos_version=__version__,
                    rationale=judgment.rationale,
                )
            )
            if surfaced:
                findings.append(
                    _finding(
                        pair=pair,
                        rationale=judgment.rationale,
                        confidence=judgment.confidence,
                        reused=False,
                        source_batch_id=execution.batch_id,
                        source_created_at=created_at,
                        reuse_index=plan.reuse_index,
                    )
                )
        pairs_judged_fresh += len(group.pairs)

        # Persist THIS batch now (P5) — wrapped individually (KD6): a write
        # failure is recorded and the loop continues judging and writing.
        if telemetry is None:
            write_failures.append(
                f"batch {execution.batch_id}: telemetry store unavailable "
                "(never constructed)"
            )
        else:
            try:
                telemetry.record_judged_batch(batch, rows, created_at)
            except Exception as exc:
                write_failures.append(f"batch {execution.batch_id}: {exc}")

    # The end probe — the certification read (CHK-D4), beside the ended_at stamp.
    # Twin-catch (KD2): `get_pending_embeddings` has NO internal wrapping, so a
    # query fault arrives as raw sqlite3.Error while an open failure arrives
    # wrapped as DatabaseError — catch both, and ONLY these two (a KeyError on
    # schema drift must stay loud, so never a broad Exception here).
    try:
        end_probe: "StaleProbe | ProbeUnavailable" = probe_stale_index(store)
    except (sqlite3.Error, DatabaseError) as exc:
        end_probe = ProbeUnavailable(detail=str(exc))

    findings.sort(key=lambda finding: (finding.proposal_hash, finding.partner_hash))
    return CheckRunResult(
        run_id=plan.run_id,
        started_at=plan.started_at,
        ended_at=_utc_now_iso(),
        nodes_total=plan.nodes_total,
        nodes_swept=plan.nodes_swept,
        sweep_degraded=plan.sweep_degraded,
        findings=tuple(findings),
        pairs_judged_fresh=pairs_judged_fresh,
        pairs_reused=len(plan.reused),
        batches_planned=len(plan.fresh_groups),
        batches_executed=batches_executed,
        batches_failed=batches_failed,
        batches_skipped=batches_skipped,
        judgment_failures=tuple(judgment_failures),
        judgment_abort=judgment_abort,
        reuse_unavailable=plan.reuse_unavailable,
        telemetry_write_failures=tuple(write_failures),
        start_probe=plan.start_probe,
        end_probe=end_probe,
    )


# --------------------------------------------------------------------------- #
# Phase 2d — degradation/exit derivation + the check_runs summary row (CHK-D7)
# --------------------------------------------------------------------------- #

# The stable degradation vocabulary, in declaration (= emission) order. A P18
# trend-query surface (`check_runs.degraded_reason` persists the comma-join):
# additive evolution ONLY — never rename or reorder a shipped token.
_DEGRADATION_TOKENS: Tuple[str, ...] = (
    "sweep",
    "judgment",
    "reuse_read",
    "telemetry_write",
    "stale_index",
    "probe_read",
    "collection_missing",
    "judgment_truncated",
)


def run_degradations(result: CheckRunResult) -> Tuple[str, ...]:
    """Derives the run's degradation tokens — the ONE site feeding exit and row (KD4).

    Both :func:`exit_code_for` and ``check_runs.degraded_reason`` read this tuple,
    so the 2-dominates-1 precedence and the healthy-NULL rule can never fork
    between the process exit and the persisted row (the same one-site discipline
    as :func:`_is_finding`). Tokens appear in declaration order, one per
    degradation class present:

    * ``"sweep"`` — the sweep tripped (``sweep_degraded``).
    * ``"judgment"`` — the judgment stage degraded: ANY batch failure, whether it
      was isolated or aborted the run (``judgment_failures``). Read off the full
      set, never off the aborting one — a run that isolated two bad batches and
      finished the other 358 is still partial and must not certify (CHK-C2).
    * ``"reuse_read"`` — the reuse/novelty bulk read failed (``reuse_unavailable``).
    * ``"telemetry_write"`` — at least one per-batch persist failed.
    * ``"stale_index"`` — a TRANSIENT backlog row in the start probe, or in the
      end probe when it is readable. ``excluded`` (poison) rows produce NO token:
      an only-poison backlog completes and exits on its findings — the CHK-D4
      poison-row escape (KD3).
    * ``"probe_read"`` — the end probe itself was unreadable
      (:class:`ProbeUnavailable`): the run cannot certify completeness.
    * ``"collection_missing"`` — the sweep went dark because the vector collection
      does not exist. Emitted **alongside** ``"sweep"``, never instead of it: the
      effect is unchanged (the sweep degraded) and this names the *cause*, so a
      trend query on the shipped token stays complete. Derived from the typed
      reason, not from "any vector fault" — a plain ``VectorStoreError`` still
      produces ``"sweep"`` alone.
    * ``"judgment_truncated"`` — at least one batch was truncated at
      ``max_tokens``. Emitted alongside ``"judgment"`` (same precedent as
      ``"collection_missing"`` alongside ``"sweep"``). Scanned across every
      failure, not just the first: under isolation the truncation that names the
      cause is rarely the one that stopped the run.

    Args:
        result: The typed run outcome.

    Returns:
        A deterministic tuple of stable tokens; empty == healthy.
    """
    present = {
        "sweep": result.sweep_degraded is not None,
        "judgment": bool(result.judgment_failures),
        "reuse_read": result.reuse_unavailable is not None,
        "telemetry_write": bool(result.telemetry_write_failures),
        "stale_index": bool(result.start_probe.transient)
        or (
            isinstance(result.end_probe, StaleProbe)
            and bool(result.end_probe.transient)
        ),
        "probe_read": isinstance(result.end_probe, ProbeUnavailable),
        "collection_missing": (
            result.sweep_degraded is not None
            and result.sweep_degraded.reason
            is ConflictUnavailableReason.COLLECTION_MISSING
        ),
        "judgment_truncated": any(
            failure.reason is ConflictUnavailableReason.JUDGMENT_TRUNCATED
            for failure in result.judgment_failures
        ),
    }
    return tuple(token for token in _DEGRADATION_TOKENS if present[token])


def exit_code_for(result: CheckRunResult) -> int:
    """Maps a completed run to the shipped 0/1/2 exit contract (CHK-C2).

    Degraded dominates findings: an incomplete check must not certify its
    finding set, so ANY degradation is 2 — even alongside new findings (they
    still ride the result; partial is labeled, never discarded). A healthy run
    exits 1 iff any finding is ``novelty == "new"`` (the gate consumes the
    already-derived novelty — never re-derives finding-ness, the one-gate-site
    discipline), else 0. Unpartitioned findings (``novelty is None``)
    structurally cannot reach the exit-1 branch: they only occur under
    ``reuse_unavailable``, which is already ``"reuse_read"`` → 2.

    This is the mapping for COMPLETED runs — 3a calls it on a
    :class:`CheckRunResult`. Pre-execute exits (invocation errors, headless
    refusal, boundary errors) are CLI-side exit-2 classes outside this
    function.

    Args:
        result: The typed run outcome.

    Returns:
        ``2`` if degraded, else ``1`` on any new finding, else ``0``.
    """
    if run_degradations(result):
        return 2
    if any(finding.novelty == "new" for finding in result.findings):
        return 1
    return 0


def coverage_exclusion_ids(result: CheckRunResult) -> Tuple[str, ...]:
    """Projects the named coverage exclusions — poison node ids, deduped, sorted.

    The union of ``excluded`` node ids across the start probe and (when
    readable) the end probe: the list the report/`--json` disclose by name (3a)
    and whose ``len()`` is the ``check_runs.coverage_exclusions`` count. When
    the end probe is :class:`ProbeUnavailable` the union is not fully knowable —
    the ROW stamps NULL there (:func:`check_run_row_from_result`), but this
    function still returns the start-probe half for the report: disclose what
    you know; certify nothing.

    Args:
        result: The typed run outcome.

    Returns:
        Sorted, deduplicated excluded node ids (content hashes — 3a resolves
        display slugs live, MI-2).
    """
    ids = {row.node_id for row in result.start_probe.excluded}
    if isinstance(result.end_probe, StaleProbe):
        ids.update(row.node_id for row in result.end_probe.excluded)
    return tuple(sorted(ids))


def check_run_row_from_result(
    result: CheckRunResult, *, mode: str, exit_code: int
) -> CheckRunRow:
    """Assembles the corpus-mode ``check_runs`` row from the one result object.

    The T12 law made structural: every scalar derives from the same
    :class:`CheckRunResult` the run report reads — the row and the report can
    never disagree. ``mode`` and ``exit_code`` are the two facts the engine
    doesn't own: the caller passes them (``exit_code`` from
    :func:`exit_code_for`; this builder never recounts, never re-derives
    finding-ness), and both are validated here with a loud ``ValueError`` — a
    programming error caught before any connection opens, cheaper than the
    schema CHECK's ``IntegrityError`` at write time.

    The NULL rules are FLAG-derived on purpose, including the edge where it
    costs something: a run with zero findings under ``reuse_unavailable`` stamps
    ``findings_new/known = NULL``, not ``0/0`` — 0/0 would be technically
    knowable there, but NULL keeps the rule one-flag-derivable and keeps
    degraded runs out of P18 clean-trend aggregates (such a run is exit 2 and
    must not read as a clean data point). ``pairs_reused`` under
    ``reuse_unavailable`` stays a TRUE zero: the run genuinely reused nothing.

    The seam order, pinned for 3a (KD5): compute ``exit_code_for(result)`` →
    build this row → ``TelemetryStore.record_check_run`` LAST, so a write
    failure can only move the exit toward 2 and a persisted row's ``exit_code``
    always equals the actual process exit. 3b hand-builds its staged row
    instead of calling this (staged accounting is not a :class:`CheckRunResult`).

    Args:
        result: The typed run outcome (the same object the report reads).
        mode: ``'corpus'`` or ``'staged'`` — validated against the closed set.
        exit_code: The actual process exit for this run — validated against
            ``{0, 1, 2}``; pass :func:`exit_code_for`'s value, never a recount.

    Returns:
        The assembled :class:`~mitos.telemetry.CheckRunRow`.

    Raises:
        ValueError: On a ``mode`` or ``exit_code`` outside its closed set.
    """
    if mode not in ("corpus", "staged"):
        raise ValueError(
            f"mode must be 'corpus' or 'staged', got {mode!r} — the check_runs "
            "mode set is a closed, shipped contract (CHK-C2)"
        )
    if exit_code not in (0, 1, 2):
        raise ValueError(
            f"exit_code must be 0, 1 or 2, got {exit_code!r} — pass "
            "exit_code_for(result), never a hand-derived code"
        )
    if result.reuse_unavailable is not None:
        findings_new: Optional[int] = None
        findings_known: Optional[int] = None
    else:
        findings_new = sum(1 for f in result.findings if f.novelty == "new")
        findings_known = sum(1 for f in result.findings if f.novelty == "known")
    if isinstance(result.end_probe, ProbeUnavailable):
        coverage_exclusions: Optional[int] = None  # the probe never completed
    else:
        coverage_exclusions = len(coverage_exclusion_ids(result))
    return CheckRunRow(
        run_id=result.run_id,
        mode=mode,
        started_at=result.started_at,
        ended_at=result.ended_at,
        exit_code=exit_code,
        nodes_swept=result.nodes_swept,
        pairs_judged_fresh=result.pairs_judged_fresh,
        pairs_reused=result.pairs_reused,
        findings_new=findings_new,
        findings_known=findings_known,
        coverage_exclusions=coverage_exclusions,
        degraded_reason=",".join(run_degradations(result)) or None,
        mitos_version=__version__,
    )
