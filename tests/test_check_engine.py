"""Tests for the corpus run engine (Phase 2c) — plan/execute seam, reuse, persistence.

``mitos check``'s composition layer: a deterministic ``plan_corpus_check`` (sweep →
screen → dedup → reuse partition → the exact fresh-batch disclosure count) strictly
before ``execute_corpus_check`` (batched judgment via an injected judge, per-batch
telemetry persistence, novelty partition). The load-bearing properties under test
(plan §9):

* the seam is structural — the plan stage has no judge parameter, touches telemetry
  exactly once (ONE ``load_reuse_index``), and writes nothing;
* per-batch persistence (P5) — a run killed after batch k leaves k batches durable;
* the two stages degrade independently (KD2) — a sweep trip never cancels judgment,
  a judgment trip skips the remainder, a telemetry write failure degrades the RUN
  but never the judgment loop (KD6);
* reuse & novelty at PRODUCTION pins (``"SONNET"`` + ``CONFLICT_PROMPT_VERSION`` —
  unlike 1b's synthetic-pin unit tests; the KD3 lockstep test keeps the literal
  honest against ``conflict_judgment._JUDGMENT_MODEL_ALIAS``'s source);
* stamping & join keys — ``surface='check'``, the ORIENTED proposal's
  ``judged_axiom``/``proposed_hash_if_any``, ``prompt_version`` from the render,
  the KD5 alias/prompt-pin guards.

Discipline (PATTERNS + scout brief): hand-rolled synchronous fakes (the keyed
``_conflict_helpers`` substrate + judge fakes); a real temp ``GraphStore`` seeded via
``commit_parsed_entry`` (real content hashes) and a real temp ``TelemetryStore``;
zero LLM, zero live keys. Every run-varying value (hashes, model ids, thresholds) is
computed at test time, never hardcoded. Run under ``./venv/bin/python -m pytest``.
"""

import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from mitos import __version__
from mitos.check import (
    CHECK_CONFIRM_BATCHES,
    CHECK_STALE_RETRY_TOLERANCE,
    CheckFinding,
    CheckPlan,
    CheckRunResult,
    ReusedPair,
    exit_code_for,
    execute_corpus_check,
    plan_corpus_check,
    run_degradations,
)
from mitos.conflict import (
    CONFLICT_PROMPT_VERSION,
    CONFLICT_SURFACE_THRESHOLD,
    ConflictUnavailableReason,
    Unavailable,
)
from mitos.errors import DatabaseError, VectorStoreError
from mitos.models import get_model_id
from mitos.parser import ParsedEntry
from mitos.store import GraphStore, open_connection
from mitos.telemetry import (
    ConflictCheckRow,
    JudgmentBatch,
    ReuseUnavailable,
    TelemetryStore,
)

from _conflict_helpers import (
    _RecordingJudge,
    _SequenceJudge,
    _execution,
    _keyed_substrate,
    _match,
)

# The production judge pin these run-time tests pass (the 1b handoff's discipline:
# production pins, matched by construction — synthetic-pin seeds would mask a real
# miss). The KD3 lockstep test below pins this literal against the AST-extracted
# ``_JUDGMENT_MODEL_ALIAS`` in conflict_judgment.py's SOURCE (never an import — the
# executor module drags the anthropic SDK into this keyless suite).
PRODUCTION_ALIAS = "SONNET"


# --------------------------------------------------------------------------- #
# Fixtures — offline env + real temp graph/telemetry stores
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_telemetry() -> TelemetryStore:
    """A real temp TelemetryStore booted to the telemetry ladder head."""
    tmpdir = tempfile.mkdtemp()
    store = TelemetryStore(os.path.join(tmpdir, ".mitos", "telemetry.sqlite"))
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def _commit(
    store: GraphStore,
    slug: str,
    axiom: str,
    *,
    scope: Optional[List[str]] = None,
    rejected: str = "An alternative.",
    **rels: List[str],
) -> str:
    """Commits a decision and returns its content-hash node id (real hashes for free)."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = rejected
    if scope is not None:
        entry.scope = scope
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


def _rows(telemetry: Any) -> List[Dict[str, Any]]:
    """Reads back every ``conflict_checks`` row (read-only; insertion order)."""
    conn = open_connection(telemetry.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM conflict_checks ORDER BY rowid")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _batches(telemetry: Any) -> List[Dict[str, Any]]:
    """Reads back every ``judgment_batches`` row (read-only; insertion order)."""
    conn = open_connection(telemetry.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM judgment_batches ORDER BY rowid")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _seed_verdict(
    telemetry: TelemetryStore,
    *,
    proposal_hash: str,
    candidate_hash: str,
    tenable: bool,
    confidence: float,
    batch_id: str,
    created_at: str,
    prompt_version: str = CONFLICT_PROMPT_VERSION,
    model_alias: str = PRODUCTION_ALIAS,
    rationale: str = "Seeded prior rationale.",
) -> None:
    """Seeds one prior verdict as its own batch through the REAL writer.

    1b's ``_seed_pair`` precedent, but defaulting to PRODUCTION pins (the engine
    tests' discipline) — ``surface='sync'`` on the seed row is deliberate: the
    reuse reader is surface-agnostic (sync-time rows are legitimate reuse/novelty
    sources, CHK-D10).
    """
    row = ConflictCheckRow(
        batch_id=batch_id,
        sync_run_id="seed-run",
        surface="sync",
        judged_axiom="Seeded proposal axiom.",
        proposal_rejected_paths=None,
        proposal_scope=None,
        proposed_hash_if_any=proposal_hash,
        candidate_slug="seeded-candidate",
        candidate_hash=candidate_hash,
        candidate_rejected_paths="Seeded alternative.",
        candidate_scope=None,
        tenable=tenable,
        confidence=confidence,
        surfaced=(not tenable) and confidence >= CONFLICT_SURFACE_THRESHOLD,
        candidate_source="embedding_topk",
        model_alias=model_alias,
        prompt_version=prompt_version,
        mitos_version=__version__,
        rationale=rationale,
    )
    batch = JudgmentBatch(
        batch_id=batch_id,
        model_id=None,
        token_input=1,
        token_output=1,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=1,
    )
    telemetry.record_judged_batch(batch, [row], created_at)


def _disjoint_pairs_corpus(
    store: GraphStore, n: int
) -> Tuple[List[Tuple[str, str]], Dict[str, List[Dict[str, Any]]]]:
    """Commits ``n`` disjoint (a, b) decision pairs; each a-side sweep discovers its b.

    Returns ``(sorted pair keys, neighbourhoods)`` — n pairs with n distinct oriented
    proposals ⇒ exactly n one-pair judgment groups. Every committed node has a
    neighbourhood entry (the keyed embed fails loud on unknown text).
    """
    neighbourhoods: Dict[str, List[Dict[str, Any]]] = {}
    keys: List[Tuple[str, str]] = []
    for i in range(n):
        a_axiom = f"Engine corpus axiom {i} alpha."
        b_axiom = f"Engine corpus axiom {i} beta."
        a_id = _commit(store, f"pair{i}-a", a_axiom)
        b_id = _commit(store, f"pair{i}-b", b_axiom)
        neighbourhoods[a_axiom] = [_match(f"pair{i}-b", 0.9)]
        neighbourhoods[b_axiom] = []
        keys.append(tuple(sorted((a_id, b_id))))
    return sorted(keys), neighbourhoods


def _plan(
    store: GraphStore,
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
    telemetry: Any,
    **kwargs: Any,
) -> CheckPlan:
    """Plans over the keyed substrate at production pins (overridable via kwargs)."""
    embed, vector = _keyed_substrate(neighbourhoods)
    kwargs.setdefault("model_alias", PRODUCTION_ALIAS)
    return plan_corpus_check(
        store=store,
        embed_provider=embed,
        vector_store=vector,
        telemetry=telemetry,
        **kwargs,
    )


def _canned_judge(
    plan: CheckPlan,
    *,
    tenable: bool = False,
    confidence: float = 0.9,
    batch_prefix: str = "engine-batch",
    overrides: Optional[Dict[int, Any]] = None,
) -> _SequenceJudge:
    """A ``_SequenceJudge`` with one valid canned execution per fresh group, in plan
    order — distinct ``batch_id`` per call (the ``judgment_batches`` PK gotcha).
    ``overrides`` replaces the entry at a batch index (an ``Unavailable`` return or
    a ``BaseException`` to raise — arming a trip/kill at exactly that batch)."""
    rets: List[Any] = []
    for i, group in enumerate(plan.fresh_groups):
        verdicts = [
            (pair.partner_node["slug"], tenable, confidence, f"Fresh rationale {i}.")
            for pair in group.pairs
        ]
        rets.append(_execution(verdicts, batch_id=f"{batch_prefix}-{i}"))
    for index, entry in (overrides or {}).items():
        rets[index] = entry
    return _SequenceJudge(rets)


class _SpyTelemetry:
    """Wraps a real TelemetryStore, counting reads/writes (the §9-1 seam spy —
    the ``_CountingReadsStore`` shape, never a MagicMock)."""

    def __init__(self, inner: TelemetryStore) -> None:
        self._inner = inner
        self.load_calls = 0
        self.record_calls = 0

    @property
    def telemetry_path(self) -> str:
        return self._inner.telemetry_path

    def load_reuse_index(self, *, prompt_version: str, model_alias: str) -> Any:
        self.load_calls += 1
        return self._inner.load_reuse_index(
            prompt_version=prompt_version, model_alias=model_alias
        )

    def record_judged_batch(self, batch: Any, rows: Any, created_at: str) -> None:
        self.record_calls += 1
        self._inner.record_judged_batch(batch, rows, created_at)


class _FailingWriteTelemetry:
    """A real store whose Nth write raises — the KD6 mid-run write-failure seam."""

    def __init__(self, inner: TelemetryStore, fail_on: Set[int]) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.write_attempts = 0

    @property
    def telemetry_path(self) -> str:
        return self._inner.telemetry_path

    def load_reuse_index(self, *, prompt_version: str, model_alias: str) -> Any:
        return self._inner.load_reuse_index(
            prompt_version=prompt_version, model_alias=model_alias
        )

    def record_judged_batch(self, batch: Any, rows: Any, created_at: str) -> None:
        self.write_attempts += 1
        if self.write_attempts in self._fail_on:
            raise DatabaseError("simulated telemetry write failure")
        self._inner.record_judged_batch(batch, rows, created_at)


# --------------------------------------------------------------------------- #
# §9-1 [TDD] — the plan/execute seam: one bulk read, zero writes, exact count
# --------------------------------------------------------------------------- #

def test_plan_stage_reads_reuse_once_writes_nothing_and_discloses_the_exact_count(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1 (TDD-first): planning touches telemetry EXACTLY once (one bulk
    ``load_reuse_index``), never writes, and ``len(fresh_groups)`` is the exact
    CHK-D5 disclosure count. The seam itself is structural — ``plan_corpus_check``
    has no judge parameter to fire — so the test pins the observable half."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    spy = _SpyTelemetry(temp_telemetry)

    plan = _plan(temp_store, neighbourhoods, spy)

    assert spy.load_calls == 1          # ONE bulk read per run — never per-pair probes
    assert spy.record_calls == 0        # the plan stage spends and writes nothing
    assert _rows(temp_telemetry) == [] and _batches(temp_telemetry) == []
    assert len(plan.fresh_groups) == 2  # the exact pending-batch disclosure count
    assert plan.nodes_total == 4 and plan.nodes_swept == 4
    assert [(p.proposal_hash, p.partner_hash) for p in plan.pairs] == keys
    assert plan.reused == ()
    assert plan.sweep_degraded is None and plan.reuse_unavailable is None
    assert plan.model_alias == PRODUCTION_ALIAS
    assert plan.prompt_version == CONFLICT_PROMPT_VERSION
    assert isinstance(plan.run_id, str) and plan.run_id
    assert isinstance(plan.started_at, str) and plan.started_at


def test_execute_never_reloads_the_reuse_index(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """The index read once pre-write IS the novelty boundary — execute must not
    reload it mid-run (a reload would read this run's own rows back as 'known')."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    spy = _SpyTelemetry(temp_telemetry)
    plan = _plan(temp_store, neighbourhoods, spy)

    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=spy, store=temp_store
    )

    assert spy.load_calls == 1      # still the plan-time read; execute added none
    assert spy.record_calls == 2    # one persist per batch, nothing else
    assert result.judgment_degraded is None
    assert all(f.novelty == "new" for f in result.findings)


def test_execute_on_zero_fresh_groups_with_no_judge_is_healthy(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1 tail: a reuse-only plan executes healthy with ``judge=None`` — zero judge
    contact, no degradation (lazy availability: a reuse-only run needs no key, P14)."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.9,
        batch_id="seed-batch-1",
        created_at="2026-07-01T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert plan.fresh_groups == () and len(plan.reused) == 1

    result = execute_corpus_check(
        plan, judge=None, telemetry=temp_telemetry, store=temp_store
    )

    assert result.judgment_degraded is None
    assert (result.batches_planned, result.batches_executed, result.batches_skipped) == (0, 0, 0)
    assert len(result.findings) == 1 and result.findings[0].reused
    assert result.pairs_reused == 1 and result.pairs_judged_fresh == 0


# --------------------------------------------------------------------------- #
# §9-2 [TDD] — per-batch persistence (P5): a killed run loses nothing paid for
# --------------------------------------------------------------------------- #

def test_kill_after_batch_k_of_n_leaves_k_batches_durable(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2 (TDD-first, P5): the judge dying on batch 3 of 3 propagates, and exactly
    the 2 already-judged batches + their rows are ON DISK — persistence is per
    batch as the run goes, never buffered to run end."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 3)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 3

    judge = _canned_judge(
        plan, batch_prefix="kill", overrides={2: RuntimeError("killed mid-run")}
    )
    with pytest.raises(RuntimeError, match="killed mid-run"):
        execute_corpus_check(
            plan, judge=judge, telemetry=temp_telemetry, store=temp_store
        )

    persisted = _batches(temp_telemetry)
    assert [b["batch_id"] for b in persisted] == ["kill-0", "kill-1"]
    rows = _rows(temp_telemetry)
    assert [r["batch_id"] for r in rows] == ["kill-0", "kill-1"]
    assert all(r["surface"] == "check" for r in rows)


def test_unavailable_on_batch_k_trips_the_remainder_without_raising(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2b: a typed ``Unavailable`` on batch 2 of 3 trips the rest — no raise, the
    1 healthy batch persisted, ``batches_executed``/``batches_skipped`` exact."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 3)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 3

    trip = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="judge severed"
    )
    judge = _canned_judge(plan, overrides={1: trip})
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 2                      # batch 3 never rendered or judged
    assert result.judgment_degraded is trip
    assert (result.batches_planned, result.batches_executed, result.batches_skipped) == (3, 2, 1)
    assert result.pairs_judged_fresh == 1        # only batch 1's pair landed verdicts
    assert len(_batches(temp_telemetry)) == 1    # the healthy prefix, already durable
    assert len(_rows(temp_telemetry)) == 1


# --------------------------------------------------------------------------- #
# §9-3/4 — the judgment trip: one penalty per run, malformation persists nothing
# --------------------------------------------------------------------------- #

def test_judgment_trip_on_first_batch_skips_remainder_and_keeps_reused_findings(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-3: with ≥2 fresh groups, an ``Unavailable`` on the FIRST trips the rest —
    the judge fires exactly once (the per-item-rebuild regression only a multi-item
    test catches), and the paid-for reused findings survive untouched."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 3)
    # Seed a standing prior finding for the lexicographically-first pair — it is
    # reused, so exactly 2 fresh groups remain.
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.92,
        batch_id="seed-standing",
        created_at="2026-07-01T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 2 and len(plan.reused) == 1

    trip = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="first batch died"
    )
    judge = _canned_judge(plan, overrides={0: trip})
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 1                        # one penalty per run, never N
    assert result.judgment_degraded is trip
    assert (result.batches_planned, result.batches_executed, result.batches_skipped) == (2, 1, 1)
    assert result.pairs_judged_fresh == 0
    # The reused finding — disclosed, authorized, already paid for — still reports.
    assert [f.reused for f in result.findings] == [True]
    assert result.findings[0].source_batch_id == "seed-standing"
    # Nothing new persisted beyond the seed batch.
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == ["seed-standing"]


def test_parse_malformation_isolates_the_bad_batch_and_keeps_judging(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-4: a garbage response on the first batch is ISOLATED, not a trip — billed
    but unpersisted (all-or-nothing parse), zero rows for the bad batch, and the
    SECOND batch is still judged and persisted.

    ``JUDGMENT`` is a first-attempt reason: the parse never reaches the executor's
    retry ladder, so one malformed response is evidence about that batch's content
    and says nothing about the next batch. Isolating it banks the verdicts the
    pre-0.17.3 trip discarded.
    """
    import dataclasses

    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 2

    healthy = _canned_judge(plan, batch_prefix="malformed")
    garbage = dataclasses.replace(healthy._rets[0], raw_text="utterly not json {")
    judge = _canned_judge(plan, batch_prefix="malformed", overrides={0: garbage})
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 2                      # batch 2 still fired
    assert result.judgment_abort is None         # the loop ran to the end
    assert result.judgment_degraded is not None  # …and the run is STILL degraded
    assert result.judgment_degraded.reason is ConflictUnavailableReason.JUDGMENT
    assert len(result.judgment_failures) == 1
    assert (
        result.batches_planned,
        result.batches_executed,
        result.batches_failed,
        result.batches_skipped,
    ) == (2, 2, 1, 0)
    assert result.batches_judged == 1            # the coverage numerator
    assert result.pairs_judged_fresh == 1        # only the healthy batch's pair
    # Exactly the healthy batch landed — the malformed one persisted nothing.
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == ["malformed-1"]
    assert [r["batch_id"] for r in _rows(temp_telemetry)] == ["malformed-1"]
    # Fail-closed is untouched: an isolated failure still emits the token and still
    # exits 2 — isolation banks work, it never softens partial into "mostly fine".
    # (Membership, not equality: this fixture's corpus also carries a stale-index
    # backlog, which is a different degradation class and not this row's claim.)
    assert "judgment" in run_degradations(result)
    assert exit_code_for(result) == 2


def test_scattered_isolated_failures_never_abort_and_bank_every_healthy_batch(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """P3: isolated failures separated by judged batches never reach the breaker.

    Five groups, truncation on batches 0, 2 and 4 — three first-attempt failures,
    but never three in a row, because a judged batch clears the streak. The run
    finishes, banks both healthy batches, and stays degraded.
    """
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 5)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 5

    truncated = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TRUNCATED, detail="over budget"
    )
    judge = _canned_judge(
        plan, batch_prefix="scatter",
        overrides={0: truncated, 2: truncated, 4: truncated},
    )
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 5                      # every batch attempted
    assert result.judgment_abort is None         # the breaker never tripped
    assert len(result.judgment_failures) == 3
    assert (
        result.batches_executed,
        result.batches_failed,
        result.batches_judged,
        result.batches_skipped,
    ) == (5, 3, 2, 0)
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == [
        "scatter-1", "scatter-3",
    ]
    # The truncation cause is named even though it never stopped the run — the
    # token is scanned across every failure, not read off the aborting one.
    assert "judgment_truncated" in run_degradations(result)
    assert exit_code_for(result) == 2


def test_three_consecutive_isolated_failures_trip_the_breaker_and_abort(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """P3: a first-attempt class that IS systematic is convicted after three in a row.

    Five groups, batch 0 healthy then 1/2/3 truncated. The third consecutive
    failure aborts, so batch 4 is never rendered or judged — the bound that keeps a
    systematic prompt defect from buying 360 full-price judge calls for zero
    verdicts. The healthy prefix is already banked.
    """
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 5)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 5

    truncated = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TRUNCATED, detail="over budget"
    )
    judge = _canned_judge(
        plan, batch_prefix="streak",
        overrides={1: truncated, 2: truncated, 3: truncated},
    )
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 4                      # batch 4 never fired
    assert result.judgment_abort is truncated    # the third one convicted the rest
    # The abort is always the LAST recorded failure (the CheckRunResult contract).
    assert result.judgment_failures[-1] is result.judgment_abort
    assert (
        result.batches_executed,
        result.batches_failed,
        result.batches_judged,
        result.batches_skipped,
    ) == (4, 3, 1, 1)
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == ["streak-0"]
    assert exit_code_for(result) == 2


def test_ladder_exhausted_failure_aborts_on_the_first_one_not_the_third(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """P3: the breaker is for FIRST-ATTEMPT reasons only — a ladder-exhausted one
    already carries its evidence and must not buy two more ~183s waits to confirm it.

    The discriminating row against a build that applied the consecutive counter
    uniformly: with three groups left after the failure, a uniform counter would
    fire two more batches before stopping.
    """
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 4)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 4

    exhausted = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
        detail="judgment failed after 9 attempts over 183s",
    )
    judge = _canned_judge(plan, batch_prefix="ladder", overrides={1: exhausted})
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 2                      # stopped at the failure, not after 3
    assert result.judgment_abort is exhausted
    assert (
        result.batches_executed,
        result.batches_failed,
        result.batches_judged,
        result.batches_skipped,
    ) == (2, 1, 1, 2)


# --------------------------------------------------------------------------- #
# §9-5 — a sweep trip degrades the PLAN, never cancels execution (KD2)
# --------------------------------------------------------------------------- #

def test_sweep_trip_mid_plan_carries_the_healthy_prefix_and_still_executes(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-5: a vector fault on node k stops the sweep (laziness IS the breaker) —
    the plan carries the healthy prefix's pairs, labeled swept-vs-total, and
    execute still judges them (disjoint substrates: a dead Qdrant says nothing
    about Anthropic)."""
    from mitos.check import snapshot_corpus

    for i in range(4):
        _commit(temp_store, f"sweeptrip-{i}", f"Sweep trip axiom {i}.")
    # Snapshot order is a DB accident — probe it at test time, never assume it
    # (the 2b laziness-test discipline). plan_corpus_check re-reads the same
    # store with the same SQL, so the order is identical.
    order = list(snapshot_corpus(temp_store).nodes)
    texts = [node["core_axiom"] for node in order]
    neighbourhoods: Dict[str, List[Dict[str, Any]]] = {text: [] for text in texts}
    neighbourhoods[texts[0]] = [_match(order[1]["slug"], 0.9)]

    embed, vector = _keyed_substrate(
        neighbourhoods,
        vector_raises={texts[2]: VectorStoreError("qdrant severed mid-sweep")},
    )
    plan = plan_corpus_check(
        store=temp_store,
        embed_provider=embed,
        vector_store=vector,
        telemetry=temp_telemetry,
        model_alias=PRODUCTION_ALIAS,
    )

    assert plan.nodes_total == 4 and plan.nodes_swept == 2
    assert plan.sweep_degraded is not None
    assert plan.sweep_degraded.reason is ConflictUnavailableReason.VECTOR_STORE
    expected_key = tuple(sorted((order[0]["id"], order[1]["id"])))
    assert [(p.proposal_hash, p.partner_hash) for p in plan.pairs] == [expected_key]
    assert len(plan.fresh_groups) == 1
    assert len(embed.calls) == 3  # nodes 0..2 gathered; node 3 structurally skipped

    judge = _canned_judge(plan, batch_prefix="kd2")
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 1                       # the discovered pair WAS judged
    assert result.judgment_degraded is None       # the trips are independent
    assert result.sweep_degraded is plan.sweep_degraded
    assert (result.nodes_total, result.nodes_swept) == (4, 2)
    assert len(result.findings) == 1 and not result.findings[0].reused
    assert len(_batches(temp_telemetry)) == 1


# --------------------------------------------------------------------------- #
# §9-6 / KD6 — telemetry write failure degrades the run, never the loop
# --------------------------------------------------------------------------- #

def test_telemetry_write_failure_mid_run_degrades_run_not_judgment_loop(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-6: batch 1's write fails — its judgments still report as findings, the
    failure is recorded, batch 2 is still judged AND its write attempted (each
    write independent), and no judgment trip fires."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    failing = _FailingWriteTelemetry(temp_telemetry, fail_on={1})
    plan = _plan(temp_store, neighbourhoods, failing)
    assert len(plan.fresh_groups) == 2

    judge = _canned_judge(plan, batch_prefix="wf")
    result = execute_corpus_check(
        plan, judge=judge, telemetry=failing, store=temp_store
    )

    assert judge.calls == 2                       # the loop never aborted
    assert result.judgment_degraded is None       # the corpus store is not the judge's downstream
    assert failing.write_attempts == 2            # batch 2's write WAS attempted
    assert len(result.telemetry_write_failures) == 1
    assert "wf-0" in result.telemetry_write_failures[0]
    assert len(result.findings) == 2              # judgments report despite the lost row
    assert result.pairs_judged_fresh == 2
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == ["wf-1"]


def test_execute_with_no_telemetry_records_one_write_failure_per_batch(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """KD6 tail: ``telemetry=None`` at execute (store never constructed) records one
    write failure per executed batch — disclosed-and-degraded, never silent."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)

    judge = _canned_judge(plan, batch_prefix="nt")
    result = execute_corpus_check(
        plan, judge=judge, telemetry=None, store=temp_store
    )

    assert judge.calls == 2
    assert len(result.telemetry_write_failures) == 2
    assert len(result.findings) == 2              # findings unaffected
    assert _batches(temp_telemetry) == []         # nothing landed anywhere


def test_judge_none_with_fresh_groups_is_a_typed_degradation_zero_spend(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-7: ``judge=None`` with pending fresh groups degrades typed — zero spend,
    zero writes, reused findings still present."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.9,
        batch_id="seed-jn",
        created_at="2026-07-01T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert len(plan.fresh_groups) == 1 and len(plan.reused) == 1

    result = execute_corpus_check(
        plan, judge=None, telemetry=temp_telemetry, store=temp_store
    )

    assert result.judgment_degraded is not None
    assert result.judgment_degraded.reason is ConflictUnavailableReason.JUDGMENT
    assert (result.batches_planned, result.batches_executed, result.batches_skipped) == (1, 0, 1)
    assert [f.reused for f in result.findings] == [True]
    assert result.findings[0].novelty == "known"
    assert [b["batch_id"] for b in _batches(temp_telemetry)] == ["seed-jn"]


# --------------------------------------------------------------------------- #
# §9-8..13 — reuse & novelty at production pins
# --------------------------------------------------------------------------- #

def test_reused_standing_finding_no_judge_call_known_with_prior_provenance(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-8: a seeded prior not-tenable-≥0.85 verdict for a live edge-free pair is
    reused — no judge call, the finding is ``known`` with the PRIOR row's
    batch_id/created_at provenance, and no new per-pair rows are written."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.91,
        batch_id="prior-batch",
        created_at="2026-06-30T12:00:00+00:00",
        rationale="The prior run's verbatim rationale.",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert plan.fresh_groups == () and len(plan.reused) == 1

    judge = _RecordingJudge(None)  # must never be consulted
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.called is False
    (finding,) = result.findings
    assert finding.reused is True and finding.novelty == "known"
    assert finding.source_batch_id == "prior-batch"
    assert finding.source_created_at == "2026-06-30T12:00:00+00:00"
    assert finding.rationale == "The prior run's verbatim rationale."  # M8-verbatim
    assert len(_rows(temp_telemetry)) == 1        # the seed row only — no new writes
    assert result.pairs_reused == 1 and result.pairs_judged_fresh == 0


def test_first_ever_finding_is_new(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-9a: a pair with no prior verdict judged not-tenable is a ``new`` finding."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)

    judge = _canned_judge(plan, batch_prefix="first")
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    (finding,) = result.findings
    assert finding.reused is False and finding.novelty == "new"
    assert finding.source_batch_id == "first-0"


def test_fresh_bypasses_reuse_but_never_novelty(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-10 (+ §9-9b): under ``fresh=True`` every pair is judged despite index hits
    (reuse bypassed — zero reused pairs, full spend), but novelty still reads the
    pre-run index: a re-confirmation of a standing finding stays ``known``; a flip
    of a previously-tenable pair is ``new``."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    _seed_verdict(  # a standing finding for the first pair
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.9,
        batch_id="standing",
        created_at="2026-07-01T00:00:00+00:00",
    )
    _seed_verdict(  # a previously-TENABLE verdict for the second pair
        temp_telemetry,
        proposal_hash=keys[1][0],
        candidate_hash=keys[1][1],
        tenable=True,
        confidence=0.9,
        batch_id="was-tenable",
        created_at="2026-07-01T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry, fresh=True)
    assert plan.fresh is True
    assert plan.reused == ()                      # reuse bypassed
    assert len(plan.fresh_groups) == 2            # full spend disclosed

    judge = _canned_judge(plan, batch_prefix="fresh")  # not-tenable 0.9 for both
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.calls == 2
    # Findings ride pair-key-sorted (the DoD-3 determinism half at the result).
    assert [(f.proposal_hash, f.partner_hash) for f in result.findings] == keys
    by_key = {(f.proposal_hash, f.partner_hash): f for f in result.findings}
    assert by_key[keys[0]].novelty == "known"     # re-confirmation of the standing one
    assert by_key[keys[1]].novelty == "new"       # the flip
    assert all(not f.reused for f in result.findings)
    assert result.pairs_reused == 0 and result.pairs_judged_fresh == 2


def test_corrupt_telemetry_degrades_reuse_and_findings_are_unpartitioned(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-11: a corrupt telemetry file at read time → typed ``reuse_unavailable``,
    the run proceeds all-fresh, and every finding is ``novelty=None`` — a run that
    cannot tell new from known must not pretend to (3a exits 2)."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    # Corrupt AFTER construction (the boot ladder needs a healthy file; a corrupt
    # image degrades at load_reuse_index, the 1b clobber idiom).
    with open(temp_telemetry.telemetry_path, "wb") as f:
        f.write(b"this is not a sqlite database at all")

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert isinstance(plan.reuse_unavailable, ReuseUnavailable)
    assert plan.reuse_index is None
    assert len(plan.fresh_groups) == 1            # everything fresh

    judge = _canned_judge(plan, batch_prefix="corrupt")
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert result.reuse_unavailable is plan.reuse_unavailable
    (finding,) = result.findings
    assert finding.novelty is None                # unpartitioned
    # The write against the corrupt file also fails — disclosed, not raised.
    assert len(result.telemetry_write_failures) == 1


def test_plan_with_no_telemetry_store_is_the_same_typed_degradation(
    temp_store: GraphStore,
) -> None:
    """Plan-time ``telemetry=None`` (the store never constructed) is treated as
    ``ReuseUnavailable`` — same typed fork, no raise."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    plan = _plan(temp_store, neighbourhoods, None)
    assert isinstance(plan.reuse_unavailable, ReuseUnavailable)
    assert plan.reuse_index is None and plan.reused == ()
    assert len(plan.fresh_groups) == 1


def test_reused_tenable_and_below_threshold_verdicts_yield_no_finding(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-12: the ONE gate site applies to stored verdicts — a tenable prior and a
    below-threshold not-tenable prior are both reused (no re-judgment) and both
    stay silent."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=True,
        confidence=0.95,
        batch_id="tenable-prior",
        created_at="2026-07-01T00:00:00+00:00",
    )
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[1][0],
        candidate_hash=keys[1][1],
        tenable=False,
        confidence=CONFLICT_SURFACE_THRESHOLD - 0.05,   # below the gate
        batch_id="low-confidence-prior",
        created_at="2026-07-01T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert plan.fresh_groups == () and len(plan.reused) == 2

    judge = _RecordingJudge(None)
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert judge.called is False
    assert result.findings == ()                  # silent — but judged and reused
    assert result.pairs_reused == 2


def test_latest_verdict_shadows_the_older_one(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-13: two priors for one pair — an older finding, a newer tenable — the
    newer wins (1b latest-wins consumed correctly): clean run, no finding."""
    keys, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    _seed_verdict(
        temp_telemetry,
        proposal_hash=keys[0][0],
        candidate_hash=keys[0][1],
        tenable=False,
        confidence=0.9,
        batch_id="older-finding",
        created_at="2026-07-01T00:00:00+00:00",
    )
    _seed_verdict(  # the pair was re-judged later and found tenable
        temp_telemetry,
        proposal_hash=keys[0][1],   # orientation-blind: seed the OTHER way round
        candidate_hash=keys[0][0],
        tenable=True,
        confidence=0.9,
        batch_id="newer-tenable",
        created_at="2026-07-02T00:00:00+00:00",
    )
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert plan.fresh_groups == () and len(plan.reused) == 1
    assert plan.reused[0].verdict.batch_id == "newer-tenable"

    result = execute_corpus_check(
        plan, judge=None, telemetry=temp_telemetry, store=temp_store
    )
    assert result.findings == () and result.judgment_degraded is None


# --------------------------------------------------------------------------- #
# §9-14/15 — stamping & the KD5 join-key guards
# --------------------------------------------------------------------------- #

def test_row_contract_e2e_stamps_the_oriented_proposal(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-14: the §6.4-style orientation case — the sweep discovers the pair from
    the LARGER-hash side, so the oriented proposal is the UNDISCOVERING side; the
    row stamps the oriented proposal's axiom/hash (never discovery context), plus
    surface/run_id/model/prompt/version stamps, MI-9 coercions, and the resolved
    ``model_id`` on the batch row."""
    axioms = {
        "orient-a": "Row contract axiom alpha.",
        "orient-b": "Row contract axiom beta.",
    }
    # Learn the orientation first: the content hash covers the canonical core only
    # (kind + axiom + mechanisms — the slug is NOT hashed, M2), so probing with
    # throwaway slugs tells us which AXIOM will orient as the proposal. Then build
    # the real corpus in a fresh store with role-appropriate MI-9 attributes: the
    # proposal side gets NO scope + EMPTY rejected_paths (the NULL coercions), the
    # candidate side carries both.
    probe_a = _commit(temp_store, "probe-a", axioms["orient-a"])
    probe_b = _commit(temp_store, "probe-b", axioms["orient-b"])
    smaller_axiom = axioms["orient-a"] if probe_a < probe_b else axioms["orient-b"]
    larger_axiom = axioms["orient-b"] if probe_a < probe_b else axioms["orient-a"]

    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    try:
        proposal_id = _commit(
            store, "oriented-proposal", smaller_axiom, rejected="", scope=None
        )
        partner_id = _commit(
            store,
            "discovering-partner",
            larger_axiom,
            rejected="A named alternative.",
            scope=["api"],
        )
        assert proposal_id < partner_id  # slug-free hashes → same order as the probe
        # Only the LARGER (discovering) side's sweep finds the pair.
        neighbourhoods = {
            larger_axiom: [_match("oriented-proposal", 0.88)],
            smaller_axiom: [],
        }
        plan = _plan(store, neighbourhoods, temp_telemetry)
        assert len(plan.fresh_groups) == 1
        group = plan.fresh_groups[0]
        assert group.proposal_hash == proposal_id      # the undiscovering side
        assert group.pairs[0].partner_hash == partner_id

        judge = _canned_judge(plan, batch_prefix="rowc", confidence=0.9)
        result = execute_corpus_check(
            plan, judge=judge, telemetry=temp_telemetry, store=temp_store
        )

        (row,) = _rows(temp_telemetry)
        assert row["surface"] == "check"
        assert row["sync_run_id"] == plan.run_id == result.run_id
        assert row["judged_axiom"] == smaller_axiom     # the ORIENTED proposal's
        assert row["proposed_hash_if_any"] == proposal_id
        assert row["candidate_slug"] == "discovering-partner"
        assert row["candidate_hash"] == partner_id
        assert row["proposal_rejected_paths"] is None   # MI-9: "" → NULL
        assert row["proposal_scope"] is None            # MI-9: [] → NULL
        assert row["candidate_rejected_paths"] == "A named alternative."  # raw, NOT NULL
        assert row["candidate_scope"] == "api"
        assert row["tenable"] == 0 and row["surfaced"] == 1
        assert row["confidence"] == 0.9
        assert row["candidate_source"] == "embedding_topk"
        assert row["model_alias"] == PRODUCTION_ALIAS
        assert row["prompt_version"] == CONFLICT_PROMPT_VERSION  # from the render
        assert row["mitos_version"] == __version__
        assert row["rationale"] == "Fresh rationale 0."

        (batch,) = _batches(temp_telemetry)
        assert batch["batch_id"] == row["batch_id"] == "rowc-0"
        # Read the expected id through get_model_id at test time (env overrides
        # exist) — never a hardcoded versioned id.
        assert batch["model_id"] == get_model_id(PRODUCTION_ALIAS)
        assert row["created_at"] == result.findings[0].source_created_at
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_unknown_alias_degrades_model_id_to_null_but_the_row_lands(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-14 tail: an execution whose alias ``get_model_id`` rejects still persists
    (the column is provenance-only — 1a's defensive idiom); ``model_id`` is NULL.
    The plan is pinned at the same unknown alias, so the KD5 guard passes."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry, model_alias="NOT_A_TIER")
    assert len(plan.fresh_groups) == 1

    verdicts = [
        (pair.partner_node["slug"], False, 0.9, "Unknown-alias rationale.")
        for pair in plan.fresh_groups[0].pairs
    ]
    judge = _SequenceJudge(
        [_execution(verdicts, batch_id="alias-x", model_alias="NOT_A_TIER")]
    )
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    (batch,) = _batches(temp_telemetry)
    assert batch["model_id"] is None
    (row,) = _rows(temp_telemetry)
    assert row["model_alias"] == "NOT_A_TIER"
    assert len(result.findings) == 1


def test_alias_join_key_guard_raises_loud_on_mismatch(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-15: an execution stamping a different alias than the plan's pin raises —
    no row lands at the wrong pin (a poisoned reuse corpus is worse than a dead
    run)."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)  # pinned SONNET

    verdicts = [
        (pair.partner_node["slug"], False, 0.9, "Wrong-pin rationale.")
        for pair in plan.fresh_groups[0].pairs
    ]
    judge = _SequenceJudge([_execution(verdicts, model_alias="FLASH")])
    with pytest.raises(ValueError, match="model_alias mismatch"):
        execute_corpus_check(
            plan, judge=judge, telemetry=temp_telemetry, store=temp_store
        )

    assert _rows(temp_telemetry) == [] and _batches(temp_telemetry) == []


def test_synthetic_prompt_pin_cannot_execute_and_spends_nothing(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§7 gotcha pinned: the renderer always stamps the production prompt version,
    so a plan pinned at any other version dies at the KD5 guard BEFORE the judge
    fires — synthetic-pin plans exercise the plan stage only, by design."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 1)
    plan = _plan(
        temp_store, neighbourhoods, temp_telemetry, prompt_version="synthetic-v9"
    )
    assert len(plan.fresh_groups) == 1

    judge = _RecordingJudge(None)
    with pytest.raises(ValueError, match="prompt_version mismatch"):
        execute_corpus_check(
            plan, judge=judge, telemetry=temp_telemetry, store=temp_store
        )

    assert judge.called is False                  # zero spend
    assert _rows(temp_telemetry) == []


def test_judgment_model_alias_lockstep_with_the_executor_source() -> None:
    """§9-16 (KD3): the production-pin literal these tests pass equals
    ``conflict_judgment._JUDGMENT_MODEL_ALIAS`` — AST-extracted from the module's
    SOURCE (importing it would drag the anthropic SDK into this keyless suite).
    A role-eval tier change reds this line instead of silently diverging."""
    import ast

    import mitos.conflict

    source_path = os.path.join(
        os.path.dirname(mitos.conflict.__file__), "conflict_judgment.py"
    )
    tree = ast.parse(open(source_path, encoding="utf-8").read(), filename=source_path)
    values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_JUDGMENT_MODEL_ALIAS"
            for t in node.targets
        )
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [PRODUCTION_ALIAS]


# --------------------------------------------------------------------------- #
# Determinism, accounting & the §8-catalog constants
# --------------------------------------------------------------------------- #

def test_two_consecutive_plans_over_an_unchanged_corpus_are_structurally_identical(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """DoD-3's determinism half: everything except the run identity (id + stamp) is
    byte-identical across two plans over an unchanged corpus."""
    _, neighbourhoods = _disjoint_pairs_corpus(temp_store, 2)
    plan_one = _plan(temp_store, neighbourhoods, temp_telemetry)
    plan_two = _plan(temp_store, neighbourhoods, temp_telemetry)

    assert plan_one.run_id != plan_two.run_id
    assert plan_one.pairs == plan_two.pairs
    assert plan_one.fresh_groups == plan_two.fresh_groups
    assert plan_one.reused == plan_two.reused
    assert (plan_one.nodes_total, plan_one.nodes_swept) == (
        plan_two.nodes_total,
        plan_two.nodes_swept,
    )


def test_empty_corpus_plans_empty_and_executes_healthy_with_zero_contact(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """An empty corpus is healthy, not broken: empty plan, zero embed/vector
    contact, and a judge-less execute completes clean (no key needed, P14)."""
    embed, vector = _keyed_substrate({})
    plan = plan_corpus_check(
        store=temp_store,
        embed_provider=embed,
        vector_store=vector,
        telemetry=temp_telemetry,
        model_alias=PRODUCTION_ALIAS,
    )
    assert (plan.nodes_total, plan.nodes_swept) == (0, 0)
    assert plan.pairs == () and plan.fresh_groups == () and plan.reused == ()
    assert embed.calls == [] and vector.queried == []

    result = execute_corpus_check(
        plan, judge=None, telemetry=temp_telemetry, store=temp_store
    )
    assert result.judgment_degraded is None and result.findings == ()
    assert isinstance(result.ended_at, str) and result.ended_at >= result.started_at


def test_check_constants_pin_their_catalog_values() -> None:
    """The §8 catalog entries this phase owns — value + home (importable from the
    dep-free leaf; consumed by 3a / 2d, forward wiring)."""
    assert CHECK_CONFIRM_BATCHES == 10
    assert CHECK_STALE_RETRY_TOLERANCE == 3
