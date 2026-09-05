"""Phase 5a — provoked-failure, concurrency, and posture hardening for ``mitos check``.

Every claim the vision makes about the check verb staying honest under fire —
"Qdrant dies mid-sweep and it fails closed", "the LLM errors on batch k and the
paid-for findings still print", "telemetry can't be read and the run falls back to
fresh", "two overlapping runs double-judge benignly" — is a sentence until a test
provokes the exact failure from the OUTERMOST entry point (``cli.cmd_check`` /
``cli._run_staged_check``) and watches the system right itself. This file converts
each of those hopes into a property (P10 — "a resilience claim untested is a hope";
"integration over isolation"):

* **T5 — provoked failure (driven from ``cmd_check``):** Qdrant severed mid-sweep
  (the aggregate breaker trips ONCE, the remainder is structurally skipped — the
  regression only a MULTI-node run catches), the judgment LLM failing at batch k
  (k-1 batches persisted, ``planned == executed + skipped``), telemetry corrupt at
  read (falls back to fresh, unpartitioned), telemetry write failing mid-run
  (judgments still report). Every one exits 2 with the findings labeled partial —
  "degraded (2) dominates findings (1)".
* **T8 — concurrency + attribution (deterministic, no threads):** a naive ``SUM``
  over ``judgment_batches`` counts each batch exactly once regardless of
  ``conflict_checks`` fan-out (the side-table cost property survives the second
  writer); two overlapping runs judging the same new pair both persist a legitimate,
  correctly-attributed batch (double-judge is wasteful, benign — no dedup by design).
* **T7-5a / P14 — posture:** a reuse-only run needs no Anthropic key, and the
  ``--json`` object stays machine-stable (full key set, degradation disclosed) when
  the run goes red — the CI consumer's contract holds under every degradation.
* **W12 / KD1 — the no-write fence:** a full ``cmd_check`` corpus run AND a
  ``_run_staged_check`` run over a ``_WriteSpyStore`` (every graph mutator raises)
  fire ZERO graph mutations — the behavioral proof that the thin CLI wrapper (which
  sits outside the static AST lint's closure) never writes the graph.

Discipline (scout brief / PATTERNS live-test rule): the ``offline`` + ``workspace``
fixtures from ``test_check_cli.py`` verbatim (real temp ``GraphStore`` +
``TelemetryStore`` at ``config``'s own paths), hand-rolled synchronous fakes
(``_keyed_substrate`` with ``vector_raises``, ``_SequenceJudge``, a corrupt-bytes
telemetry file), production pins (``"SONNET"`` + the default ``CONFLICT_PROMPT_VERSION``).
Zero ``AsyncMock`` (nothing in the check path is async), zero live keys. Every healthy
fixture drains its Outbox (``_drain_outbox``) so the stale-index probe does not mask the
failure under test. Run under ``./venv/bin/python -m pytest``.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos import __version__, check, cli
from mitos.config import MitosConfig
from mitos.conflict import (
    CONFLICT_PROMPT_VERSION,
    ConflictUnavailableReason,
    Unavailable,
)
from mitos.errors import CollectionMissingError, DatabaseError, VectorStoreError
from mitos.store import GraphStore, open_connection
from mitos.vector_store import QdrantVectorStore
from mitos.telemetry import ConflictCheckRow, JudgmentBatch, TelemetryStore

from _conflict_helpers import (
    _drain_outbox,
    _execution,
    _keyed_substrate,
    _match,
    _read_batch_rows,
    _read_conflict_rows,
    _SequenceJudge,
)
from test_check_probe import _canned_judge, _commit, _seed_verdict

PRODUCTION_ALIAS = "SONNET"

# The machine contract a CI consumer parses — the full §8 corpus ``--json`` key set.
# 5a proves it stays byte-stable when the run goes RED: a degraded run discloses its
# degradation without dropping or renaming a key.
#
# `test_check_cli.py::test_7_json_shape_snapshot` now IMPORTS this rather than
# re-listing it: the duplicate copy is what let a radius analysis miss five of the
# six rows here when the corpus echo added three keys. One constant, one edit.
#
# The three provenance keys are the corpus echo (§4.7): every response that resolves
# a project names it. They are asserted as MEMBERS of the exact-equality set, never
# by relaxing the comparison to a subset — the whole point of these rows is that the
# key set is machine-stable under degradation, and `>=` would gut that.
_CORPUS_JSON_KEYS = {
    "run_id", "mode", "exit_code", "started_at", "ended_at", "fresh",
    "nodes_total", "nodes_swept", "pairs_judged_fresh", "pairs_reused",
    "batches_planned", "batches_executed", "batches_judged", "batches_failed",
    "batches_skipped", "findings",
    "findings_new", "findings_known", "degradations", "coverage_exclusions",
    "index_backlog_transient", "summary_row_written",
    "project", "collection", "workspace",
}


# --------------------------------------------------------------------------- #
# Fixtures — offline env + a real temp workspace (config + graph + telemetry)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace() -> Tuple[MitosConfig, GraphStore, TelemetryStore]:
    """A temp workspace whose graph + telemetry live at ``config``'s own paths.

    ``cmd_check`` builds its own ``GraphStore(config.db_path)`` /
    ``TelemetryStore(config.telemetry_path)`` — the same files these fixtures seed.
    """
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    store = GraphStore(config.db_path)
    telemetry = TelemetryStore(config.telemetry_path)
    yield config, store, telemetry
    shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Helpers — seam wiring + telemetry readback (the 3a/3b idiom, verbatim)
# --------------------------------------------------------------------------- #

class _FakeResponse:
    """A minimal ``requests`` response — status plus a body (or a decode failure)."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.text = "hostile body"
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _pair(store: GraphStore) -> Tuple[str, str, Dict[str, List[Dict[str, Any]]]]:
    """Commits an (a, b) pair whose a-side sweep discovers b; returns ids + neighbourhoods."""
    a_axiom = "Hardening axiom alpha for the check verb."
    b_axiom = "Hardening axiom beta for the check verb."
    a_id = _commit(store, "hard-a", a_axiom)
    b_id = _commit(store, "hard-b", b_axiom)
    neighbourhoods = {a_axiom: [_match("hard-b", 0.9)], b_axiom: []}
    return a_id, b_id, neighbourhoods


def _wire_substrate(
    monkeypatch: pytest.MonkeyPatch,
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
    *,
    vector_raises: Optional[Dict[str, BaseException]] = None,
) -> Tuple[Any, Any]:
    """Monkeypatches ``cli._build_check_substrate`` to keyed fakes; returns them."""
    embed, vector = _keyed_substrate(neighbourhoods, vector_raises=vector_raises)
    monkeypatch.setattr(cli, "_build_check_substrate",
                        lambda config: (embed, vector, None))
    return embed, vector


def _wire_judge(monkeypatch: pytest.MonkeyPatch, judge: Any) -> List[bool]:
    """Monkeypatches ``cli._build_check_judge`` to return ``judge``; logs each build."""
    invoked: List[bool] = []

    def builder(config: MitosConfig) -> Any:
        invoked.append(True)
        return judge

    monkeypatch.setattr(cli, "_build_check_judge", builder)
    return invoked


def _canned_for(
    store: GraphStore, embed: Any, vector: Any, telemetry: Any, *,
    tenable: bool = False, confidence: float = 0.9,
) -> _SequenceJudge:
    """Builds a per-group canned judge from a plan matching ``cmd_check``'s internal one."""
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=telemetry,
        model_alias=PRODUCTION_ALIAS,
    )
    return _canned_judge(plan, tenable=tenable, confidence=confidence)


def _read_check_runs(config: MitosConfig) -> List[Dict[str, Any]]:
    """Reads every ``check_runs`` row back through a real read-only connection."""
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM check_runs").fetchall()]
    finally:
        conn.close()


def _decisions_md(*entries: Tuple[str, str]) -> str:
    """Renders a working-tree ``decisions.md`` from ``(slug, axiom)`` pairs (3b shape)."""
    lines = ["<!-- BEGIN ENTRIES -->\n\n"]
    for slug, axiom in entries:
        lines.append(f"## 2026-07-05 — {slug} — Title\n")
        lines.append(f"**Decided:** {axiom}\n")
        lines.append(f"**Rejected:** A rejected alternative for {slug}.\n\n")
    return "".join(lines)


def _write_decisions(config: MitosConfig, *entries: Tuple[str, str]) -> None:
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(_decisions_md(*entries))


# A pending↔committed pair whose pending sweep discovers the committed decision.
_PENDING_AXIOM = "Pending gate axiom that may conflict with the active corpus."
_ACTIVE_AXIOM = "Active corpus axiom the pending entry may contradict."


class _FailingBatchWriteTelemetry:
    """Wraps a real telemetry store; only ``record_judged_batch`` raises (per-batch fault).

    The check-CLI ``_FailingWriteTelemetry`` fails the SUMMARY write (``record_check_run``);
    this fails the PER-BATCH write mid-loop so ``execute_corpus_check`` records a
    ``telemetry_write`` degradation while the judgment still parses and reports — the KD6
    "degrade the RUN, never the loop" property driven from ``cmd_check``. Everything else
    (``load_reuse_index``, ``record_check_run``) delegates, so the reuse partition and the
    summary row are unaffected and the degradation is isolated to the batch write.
    """

    def __init__(self, inner: TelemetryStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_judged_batch(self, batch: Any, rows: Any, created_at: str) -> None:
        raise DatabaseError("provoked per-batch write fault")


class _WriteSpyStore:
    """Wraps a real ``GraphStore``; every graph/Outbox MUTATION method raises (KD1/W12).

    Reads pass through to the real store (``__getattr__`` delegation), so a full check
    run plans, sweeps, judges, and probes against the real graph. Each of the six public
    graph/Outbox mutators raises loudly if the check path ever calls it — the behavioral
    proof that the CLI check path (which sits OUTSIDE the static AST closure rooted at
    ``check.py``) writes nothing. The set is the complete public mutation surface on
    ``GraphStore`` (scout Peripheral References, verified against ``store.py``);
    ``record_decision_entry`` lives on ``MitosSyncManager``, never on the store, so it
    cannot fire through a store-wrapping spy.
    """

    _MUTATORS = frozenset({
        "write_signal",
        "note_source_reencounter",
        "commit_parsed_entry",
        "add_pending_embedding",
        "remove_pending_embedding",
        "increment_pending_attempts",
    })

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.mutations: List[str] = []

    def __getattr__(self, name: str) -> Any:
        if name in _WriteSpyStore._MUTATORS:
            def _forbidden(*args: Any, **kwargs: Any) -> Any:
                self.mutations.append(name)
                raise AssertionError(
                    f"the check path called graph mutator {name!r} — it must be read-only"
                )
            return _forbidden
        return getattr(self._inner, name)


def _conflict_row(
    *, batch_id: str, surface: str, sync_run_id: str,
    proposed_hash: str, candidate_slug: str, candidate_hash: str,
) -> ConflictCheckRow:
    """A minimal judged-pair row (surface-tagged) — the T8 attribution seed shape."""
    return ConflictCheckRow(
        batch_id=batch_id,
        sync_run_id=sync_run_id,
        surface=surface,
        judged_axiom="Seeded proposal axiom for attribution.",
        proposal_rejected_paths=None,
        proposal_scope=None,
        proposed_hash_if_any=proposed_hash,
        candidate_slug=candidate_slug,
        candidate_hash=candidate_hash,
        candidate_rejected_paths="Seeded candidate alternative.",
        candidate_scope=None,
        tenable=False,
        confidence=0.9,
        surfaced=True,
        candidate_source="embedding_topk",
        model_alias=PRODUCTION_ALIAS,
        prompt_version=CONFLICT_PROMPT_VERSION,
        mitos_version=__version__,
        rationale="Seeded attribution rationale.",
    )


def _batch(batch_id: str, *, token_input: int) -> JudgmentBatch:
    """A batch metrics row with a distinct ``token_input`` — the SUM-once probe target."""
    return JudgmentBatch(
        batch_id=batch_id,
        model_id=None,
        token_input=token_input,
        token_output=10,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=5,
    )


# =========================================================================== #
# T5 — provoked failure, driven from the outermost entry point
# =========================================================================== #

def test_t5_qdrant_severed_midsweep_partial_exit_2_breaker_trips_once(
    workspace, monkeypatch, capsys,
) -> None:
    """Qdrant dies at sweep node 2 of N → partial (exit 2), findings ride, breaker trips ONCE.

    The multi-node proof a unit test structurally cannot give: the vector store is queried
    only up to and INCLUDING the severed node — never for the nodes beyond it. A
    per-entry-rebuilt breaker (the regression) would keep querying every remaining node;
    ``iter_sweep`` laziness stops the moment the k-th gather yields ``Unavailable``.
    """
    config, store, telemetry = workspace
    finder_ax = "Finder axiom that discovers a partner before the outage."
    severed_ax = "Severed axiom whose vector query goes dark mid-run."
    target_ax = "Target axiom — the finder's partner."
    finder_id = _commit(store, "finder", finder_ax)
    _commit(store, "severed", severed_ax)
    _commit(store, "target", target_ax)
    _commit(store, "tail-1", "First tail axiom, never reached.")
    _commit(store, "tail-2", "Second tail axiom, never reached.")
    _drain_outbox(store)  # healthy Outbox — the stale-index probe must not mask the outage
    nbhds = {
        finder_ax: [_match("target", 0.9)],
        severed_ax: [],
        target_ax: [],
        "First tail axiom, never reached.": [],
        "Second tail axiom, never reached.": [],
    }
    embed, vector = _wire_substrate(
        monkeypatch, nbhds,
        vector_raises={severed_ax: VectorStoreError("qdrant down mid-run")},
    )
    # Build the canned judge from a CLEAN (non-severed) substrate so planning it does not
    # pollute the injected vector's `queried` log — the corpus yields exactly the finder's
    # one group either way (severed/target/tails discover nothing).
    clean_embed, clean_vector = _keyed_substrate(nbhds)
    judge = _canned_for(store, clean_embed, clean_vector, telemetry,
                        tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 2
    out = capsys.readouterr().out
    assert "[Conflict]" in out                         # the finder's finding survives the outage
    assert "finder" in out and "target" in out
    assert "[partial] This check could not fully run" in out
    assert "Swept 1 of 5 decisions" in out             # only node 1 completed; the 4 others uncertified
    # Breaker trips ONCE: the vector was queried for the finder and the severed node only —
    # the three tail nodes beyond the trip were never gathered.
    assert len(vector.queried) == 2


def test_t6_missing_collection_names_itself_and_its_heal_exit_2(
    workspace, monkeypatch, capsys,
) -> None:
    """T6 (corpus): an absent collection over a populated graph → exit 2, named, healed.

    The posture is unchanged from the severed-Qdrant row above — that is the point
    of re-keying the precondition onto the operation rather than rewriting it. What
    changes is that the operator is no longer told the infrastructure is down when
    it is running; they are told the collection is gone and handed the one command
    that rebuilds it.
    """
    config, store, telemetry = workspace
    axiom = "Collection-missing axiom for the corpus check."
    _commit(store, "coll-missing", axiom)
    _drain_outbox(store)
    _wire_substrate(
        monkeypatch, {axiom: []},
        vector_raises={axiom: CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )},
    )

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    out, err = capsys.readouterr()
    assert code == 2                                    # fail-closed, unchanged
    assert "[partial] This check could not fully run" in out
    assert "the vector collection does not exist" in out
    assert "mitos reconcile" in out
    assert "unreachable" not in out and "Qdrant" not in err
    assert "Traceback" not in out and "Traceback" not in err


def test_t6_missing_collection_json_carries_both_tokens(
    workspace, monkeypatch, capsys,
) -> None:
    """The machine surface gets the cause beside the effect, in declaration order.

    Corpus mode emits ``run_degradations``' declaration order; ``--staged`` sorts.
    Two surfaces, two orders — assert each to its own.
    """
    config, store, telemetry = workspace
    axiom = "Collection-missing axiom for the corpus JSON."
    _commit(store, "coll-json", axiom)
    _drain_outbox(store)
    _wire_substrate(
        monkeypatch, {axiom: []},
        vector_raises={axiom: CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )},
    )

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["degradations"] == ["sweep", "collection_missing"]
    assert obj.keys() == _CORPUS_JSON_KEYS          # no key dropped or renamed


# --------------------------------------------------------------------------- #
# §10 — the layer-removal adversarial row: the operation-time gate is now SOLE
# --------------------------------------------------------------------------- #

_HOSTILE_READS = [
    pytest.param("missing", id="404-collection-missing"),
    pytest.param("malformed-json", id="200-non-json-body"),
    pytest.param("malformed-shape", id="200-result-not-a-list"),
    pytest.param("no-result-key", id="200-without-a-result-key"),
    pytest.param("null-result", id="200-with-a-null-result"),
    pytest.param("server-error", id="500"),
    pytest.param("refused", id="connection-refused"),
]


@pytest.mark.parametrize("fault", _HOSTILE_READS)
def test_t6_hostile_read_boundary_still_fails_closed(
    workspace, monkeypatch, capsys, fault: str,
) -> None:
    """Every hostile shape at the REAL read boundary still exits 2, calmly.

    Deleting ``_build_check_substrate``'s construction-time guard makes the
    operation-time typed degradation the **only** thing standing between a missing
    collection and a ``check`` that reports clean. A latent weakness there was
    cosmetic while the guard backstopped it; it is a real defect now. So this row
    drives a genuine ``QdrantVectorStore`` with a faked ``requests`` — not an
    injected exception — so the classifier itself is under test end to end.

    The malformed 200 is the one that matters most: it is the only shape that could
    slip past a status-code-keyed classifier and reach the report as a healthy empty
    sweep, i.e. an unauditable corpus reading as clean — exactly what CHK-D5's
    fail-closed contract forbids.
    """
    import requests as _requests

    config, store, telemetry = workspace
    axiom = "Hostile-boundary axiom for the sole-gate row."
    _commit(store, "hostile", axiom)
    _drain_outbox(store)
    embed, _discard = _keyed_substrate({axiom: []})
    # A REAL store: only the transport is faked, so the 404/shape classification and
    # the typed-error plumbing are exercised rather than assumed.
    real_vector = QdrantVectorStore("http://localhost:9", "mitos-tmp-hostile")
    monkeypatch.setattr(cli, "_build_check_substrate",
                        lambda config: (embed, real_vector, None))

    if fault == "refused":
        def _post(*args: Any, **kwargs: Any) -> Any:
            raise _requests.RequestException("connection refused")
    else:
        resp = _FakeResponse(*{
            "missing": (404, {}),
            "malformed-json": (200, ValueError("Expecting value")),
            "malformed-shape": (200, {"result": "not-a-list"}),
            "no-result-key": (200, {"status": "ok"}),
            "null-result": (200, {"result": None}),
            "server-error": (500, {}),
        }[fault])

        def _post(*args: Any, **kwargs: Any) -> Any:
            return resp

    monkeypatch.setattr("mitos.vector_store.requests.post", _post)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    out, err = capsys.readouterr()
    assert code == 2                                       # fail-closed, every shape
    assert "[partial] This check could not fully run" in out
    assert "Swept 0 of 1 decisions" in out                 # never certified as clean
    assert "Traceback" not in out and "Traceback" not in err
    if fault == "missing":
        assert "mitos reconcile" in out                    # only absence gets the heal


def test_t5_judgment_fails_at_batch_k_persists_k_minus_one(
    workspace, monkeypatch, capsys,
) -> None:
    """The judge returns ``Unavailable`` at batch 2 of 3 → k-1 rows on disk, exit 2, partial.

    Fail-OPEN (scout Discrepancy #2): the fake RETURNS ``Unavailable(JUDGMENT_TIMEOUT)`` at
    the tripping batch (the real executor never raises past its seam), so it degrades typed
    rather than crashing ``execute_corpus_check``. The tripping batch is billed
    (``batches_executed`` counts it) but persists nothing; the remainder is skipped;
    ``planned == executed + skipped`` holds and exactly one batch's rows land.
    """
    config, store, telemetry = workspace
    pairs = [
        ("f1", "First finder axiom.", "t1", "First target axiom."),
        ("f2", "Second finder axiom.", "t2", "Second target axiom."),
        ("f3", "Third finder axiom.", "t3", "Third target axiom."),
    ]
    nbhds: Dict[str, List[Dict[str, Any]]] = {}
    for f_slug, f_ax, t_slug, t_ax in pairs:
        _commit(store, f_slug, f_ax)
        _commit(store, t_slug, t_ax)
        nbhds[f_ax] = [_match(t_slug, 0.9)]
        nbhds[t_ax] = []
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    # Three fresh groups, judged in plan (sorted-proposal-hash) order; batch index 1
    # (the 2nd group reached) returns a typed Unavailable — the simulated batch-k LLM fail.
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=telemetry,
        model_alias=PRODUCTION_ALIAS,
    )
    assert len(plan.fresh_groups) == 3  # the fixture yields three independent groups
    judge = _canned_judge(
        plan, tenable=False, confidence=0.9,
        overrides={1: Unavailable(reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
                                  detail="batch 2 timed out")},
    )
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS          # machine-stable under the judgment degradation
    assert obj["degradations"] == ["judgment"]
    assert judge.calls == 2                              # group 1 fired, group 2 tripped, group 3 never called
    assert obj["batches_executed"] == 2                  # the tripping batch is billed…
    assert obj["batches_skipped"] == 1                   # …and the remainder skipped…
    assert obj["batches_planned"] == obj["batches_executed"] + obj["batches_skipped"]
    assert len(obj["findings"]) >= 1                     # the first (healthy) batch's finding rides, partial
    # Per-batch persistence: exactly the ONE healthy batch's rows landed (k-1 = 1).
    assert len(_read_batch_rows(config)) == 1


def _wire_isolated_truncation(workspace, monkeypatch):
    """Wires a 3-group corpus whose SECOND batch truncates — the isolation scenario.

    ``JUDGMENT_TRUNCATED`` is a first-attempt reason, so the engine isolates that
    batch and keeps judging: 3 executed, 1 failed, 2 judged, 0 skipped. Shared by
    the receipt row and the text-report row so the two cannot drift apart on which
    scenario they describe.
    """
    config, store, telemetry = workspace
    pairs = [
        ("g1", "First finder axiom.", "h1", "First target axiom."),
        ("g2", "Second finder axiom.", "h2", "Second target axiom."),
        ("g3", "Third finder axiom.", "h3", "Third target axiom."),
    ]
    nbhds: Dict[str, List[Dict[str, Any]]] = {}
    for f_slug, f_ax, t_slug, t_ax in pairs:
        _commit(store, f_slug, f_ax)
        _commit(store, t_slug, t_ax)
        nbhds[f_ax] = [_match(t_slug, 0.9)]
        nbhds[t_ax] = []
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=telemetry,
        model_alias=PRODUCTION_ALIAS,
    )
    assert len(plan.fresh_groups) == 3
    judge = _canned_judge(
        plan, tenable=False, confidence=0.9,
        overrides={1: Unavailable(reason=ConflictUnavailableReason.JUDGMENT_TRUNCATED,
                                  detail="batch 2 truncated at max_tokens")},
    )
    _wire_judge(monkeypatch, judge)
    return config, judge


def test_t5_isolated_batch_failure_states_coverage_on_the_text_report(
    workspace, monkeypatch, capsys,
) -> None:
    """The coverage number is on the HUMAN surface too, not only in ``--json``.

    A partial audit that says only "[partial]" leaves the reader unable to tell one
    bad batch from a dead run — the number is what makes the difference legible.
    """
    config, judge = _wire_isolated_truncation(workspace, monkeypatch)

    code = cli.cmd_check(
        config, scope=None, fresh=False, assume_yes=False, as_json=False
    )
    out = capsys.readouterr().out

    assert code == 2
    assert "[partial] This check could not fully run" in out
    assert "Judged 2 of 3 judgment batches (1 failed)." in out
    assert "Traceback" not in out


def test_t5_isolated_batch_failure_reports_coverage_and_still_exits_2(
    workspace, monkeypatch, capsys,
) -> None:
    """P3 at the surface: a truncated batch 2 of 3 is ISOLATED — batch 3 is still
    judged and persisted, the receipt states coverage as a NUMBER, and the run is
    still fail-closed at exit 2 with the ``judgment`` token.

    The pair to ``test_t5_judgment_fails_at_batch_k_persists_k_minus_one``, which
    keeps the aborting (ladder-exhausted) case. Together they pin both sides of the
    class split at the surface a caller actually reads.
    """
    config, judge = _wire_isolated_truncation(workspace, monkeypatch)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2                                     # fail-closed is untouched
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS          # machine-stable under isolation
    assert obj["degradations"] == ["judgment", "judgment_truncated"]
    assert judge.calls == 3                              # batch 3 fired — the isolation
    assert obj["batches_executed"] == 3
    assert obj["batches_failed"] == 1                    # the truncated one, billed…
    assert obj["batches_judged"] == 2                    # …and coverage IS a number
    assert obj["batches_skipped"] == 0                   # nothing was discarded
    assert obj["batches_planned"] == obj["batches_executed"] + obj["batches_skipped"]
    # TWO batches' rows landed — the second healthy batch is exactly what the
    # pre-0.17.3 trip threw away.
    assert len(_read_batch_rows(config)) == 2


def test_t5_corrupt_telemetry_read_falls_back_to_fresh_unpartitioned_exit_2(
    workspace, monkeypatch, capsys,
) -> None:
    """A genuinely-unreadable telemetry file → the run falls back to fresh, unpartitioned, exit 2.

    A real ``TelemetryStore`` is constructed, then its file bytes are corrupted, so its next
    ``load_reuse_index`` opens the corrupt image and raises ``sqlite3.DatabaseError`` at query
    time → a typed ``ReuseUnavailable`` (broken read ≠ empty index). The plan proceeds
    all-fresh: the fresh group IS judged (not vacuous), the finding reports UNPARTITIONED
    (``novelty is None`` — a run that cannot tell new from known must not pretend to).
    """
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    # Construct the store (schema migrates cleanly), THEN corrupt the file so the NEXT read
    # fails — a real broken read through the shipped load_reuse_index, not a `None` stand-in.
    corrupt_tele = TelemetryStore(config.telemetry_path)
    with open(config.telemetry_path, "wb") as f:
        f.write(b"this is not a sqlite database image at all")
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: corrupt_tele)
    judge = _canned_for(store, embed, vector, None, tenable=False, confidence=0.9)
    invoked = _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS          # machine-stable under the reuse-read degradation
    assert "reuse_read" in obj["degradations"]
    assert invoked == [True]                              # the fresh group was actually judged
    assert obj["pairs_judged_fresh"] == 1                 # …not vacuous over an empty fresh set
    assert len(obj["findings"]) == 1
    assert obj["findings"][0]["novelty"] is None          # unpartitioned — cannot tell new from known
    assert obj["findings_new"] is None and obj["findings_known"] is None


def test_t5_per_batch_write_failure_reports_findings_exit_2(
    workspace, monkeypatch, capsys,
) -> None:
    """A per-batch telemetry write failing mid-run → judgments still report, telemetry_write, exit 2.

    The write degrades the RUN, never the loop (KD6): ``record_judged_batch`` raises, the
    engine records the failure and keeps the judgment, the finding still prints/JSONs, and
    no ``judgment_batches`` row lands for the failed batch. Driven through ``cmd_check`` via
    the ``_build_check_telemetry`` seam, not the engine directly.
    """
    config, store, telemetry = workspace
    _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(
        monkeypatch,
        {"Hardening axiom alpha for the check verb.": [_match("hard-b", 0.9)],
         "Hardening axiom beta for the check verb.": []},
    )
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    monkeypatch.setattr(
        cli, "_build_check_telemetry",
        lambda config: _FailingBatchWriteTelemetry(TelemetryStore(config.telemetry_path)),
    )

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS          # machine-stable under the telemetry-write degradation
    assert "telemetry_write" in obj["degradations"]
    assert len(obj["findings"]) == 1                     # the judgment still reports…
    assert obj["findings"][0]["novelty"] == "new"
    assert _read_batch_rows(config) == []                # …but the failed batch persisted no row


# =========================================================================== #
# T8 — concurrency + exactly-once cost attribution (deterministic, no threads)
# =========================================================================== #

def test_t8_mixed_surface_batches_attribute_cost_exactly_once(workspace) -> None:
    """A naive ``SUM`` over ``judgment_batches`` counts each batch once regardless of fan-out.

    The side-table cost property (KD4), re-asserted over the SECOND writer: seed one
    ``surface='sync'`` batch with TWO judged pairs and one ``surface='check'`` batch with
    one — so summing tokens through the ``conflict_checks`` fan-out would double-count the
    sync batch, but the ``judgment_batches`` side table holds each batch's cost exactly
    once. Per-surface partitioning over ``DISTINCT batch_id`` attributes each batch to its
    one surface with no double count.
    """
    config, store, telemetry = workspace
    # One sync batch (2 judged pairs → 2 conflict_checks rows, one batch), one check batch.
    telemetry.record_judged_batch(
        _batch("batch-sync", token_input=100),
        [
            _conflict_row(batch_id="batch-sync", surface="sync", sync_run_id="sync-run",
                          proposed_hash="p-sync", candidate_slug="c1", candidate_hash="h1"),
            _conflict_row(batch_id="batch-sync", surface="sync", sync_run_id="sync-run",
                          proposed_hash="p-sync", candidate_slug="c2", candidate_hash="h2"),
        ],
        "2026-07-01T00:00:00+00:00",
    )
    telemetry.record_judged_batch(
        _batch("batch-check", token_input=200),
        [_conflict_row(batch_id="batch-check", surface="check", sync_run_id="check-run",
                       proposed_hash="p-check", candidate_slug="c3", candidate_hash="h3")],
        "2026-07-02T00:00:00+00:00",
    )

    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        # The honest cost total: each batch counted once from the side table.
        total = conn.execute("SELECT SUM(token_input) FROM judgment_batches").fetchone()[0]
        assert total == 300
        assert conn.execute("SELECT COUNT(*) FROM judgment_batches").fetchone()[0] == 2
        # The fan-out that would corrupt a naive join-and-sum: 3 candidate rows, 2 for sync.
        assert conn.execute("SELECT COUNT(*) FROM conflict_checks").fetchone()[0] == 3
        # Per-surface cost partitions over DISTINCT batches — each batch attributed once.
        by_surface = dict(conn.execute(
            "SELECT cc.surface, SUM(jb.token_input) "
            "FROM judgment_batches jb "
            "JOIN (SELECT DISTINCT batch_id, surface FROM conflict_checks) cc "
            "ON jb.batch_id = cc.batch_id GROUP BY cc.surface"
        ).fetchall())
        assert by_surface == {"sync": 100, "check": 200}
    finally:
        conn.close()


def test_t8_overlapping_runs_double_judge_the_same_pair_benignly(workspace) -> None:
    """Two overlapping check runs judging the SAME new pair both persist, correctly attributed.

    Append-only whole-row writes make a concurrent double-judge benign: no dedup is applied
    (by design — §6.2), each run persists its own batch + candidate row + summary row, and
    the rows attribute cleanly by ``run_id`` with no torn/lost row. The writes are issued in
    INTERLEAVED order (both batches, then both summaries) to model the concurrent interleave
    deterministically — the outcome is order-independent because every row lands whole.
    """
    config, store, telemetry = workspace
    proposed, candidate = "shared-proposal-hash", "shared-candidate-hash"
    # Interleave: run-1 batch, run-2 batch, run-1 summary, run-2 summary.
    telemetry.record_judged_batch(
        _batch("run1-batch", token_input=50),
        [_conflict_row(batch_id="run1-batch", surface="check", sync_run_id="run-1",
                       proposed_hash=proposed, candidate_slug="cand", candidate_hash=candidate)],
        "2026-07-03T00:00:00+00:00",
    )
    telemetry.record_judged_batch(
        _batch("run2-batch", token_input=60),
        [_conflict_row(batch_id="run2-batch", surface="check", sync_run_id="run-2",
                       proposed_hash=proposed, candidate_slug="cand", candidate_hash=candidate)],
        "2026-07-03T00:00:01+00:00",
    )
    for run_id, exit_code in (("run-1", 1), ("run-2", 1)):
        telemetry.record_check_run(check.CheckRunRow(
            run_id=run_id, mode="corpus",
            started_at="2026-07-03T00:00:00+00:00", ended_at="2026-07-03T00:00:02+00:00",
            exit_code=exit_code, nodes_swept=1, pairs_judged_fresh=1, pairs_reused=0,
            findings_new=1, findings_known=0, coverage_exclusions=0,
            degraded_reason=None, mitos_version=__version__,
        ))

    rows = _read_conflict_rows(config)
    # Double-judge: the SAME pair judged twice → two legitimate rows, no dedup.
    same_pair = [r for r in rows
                 if r["proposed_hash_if_any"] == proposed and r["candidate_hash"] == candidate]
    assert len(same_pair) == 2
    assert {r["sync_run_id"] for r in same_pair} == {"run-1", "run-2"}
    # Each run's whole row set landed intact and attributes by run_id.
    assert len(_read_batch_rows(config)) == 2
    runs = {r["run_id"]: r for r in _read_check_runs(config)}
    assert set(runs) == {"run-1", "run-2"}
    assert all(runs[r]["nodes_swept"] == 1 for r in runs)


# =========================================================================== #
# T7-5a / P14 — posture: keyless reuse-only + machine-stable --json under degradation
# =========================================================================== #

def test_t7_keyless_reuse_only_run_needs_no_key_json_stable(
    workspace, monkeypatch, capsys,
) -> None:
    """A reuse-only run (zero fresh groups) completes keyless; its ``--json`` stays the full shape.

    P14 keyless posture: a run whose every pair reuses a stored verdict never constructs an
    Anthropic client (the judge builder is never reached), so no ``ANTHROPIC_API_KEY`` is
    needed. The ``offline`` fixture strips the key; the run still exits 0 and emits the whole
    §8 key set.
    """
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(telemetry, proposal_hash=a_id, candidate_hash=b_id,
                  tenable=False, confidence=0.9, batch_id="prior-batch",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 0
    assert invoked == []                                  # keyless — the judge is never built
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS
    assert obj["degradations"] == [] and obj["findings_known"] == 1


def test_t7_json_shape_machine_stable_under_sweep_degradation(
    workspace, monkeypatch, capsys,
) -> None:
    """The ``--json`` object keeps the full key set when a sweep degradation reddens the run.

    A CI consumer parsing ``--json`` must not hit a KeyError the day the substrate dies:
    every §8 key is present and the ``sweep`` degradation is disclosed in ``degradations``.
    """
    config, store, telemetry = workspace
    finder_ax = "JSON-stable finder axiom."
    severed_ax = "JSON-stable severed axiom."
    _commit(store, "js-finder", finder_ax)
    _commit(store, "js-severed", severed_ax)
    _commit(store, "js-target", "JSON-stable target axiom.")
    _drain_outbox(store)
    nbhds = {finder_ax: [_match("js-target", 0.9)], severed_ax: [],
             "JSON-stable target axiom.": []}
    embed, vector = _wire_substrate(
        monkeypatch, nbhds, vector_raises={severed_ax: VectorStoreError("qdrant down")},
    )
    clean_embed, clean_vector = _keyed_substrate(nbhds)
    judge = _canned_for(store, clean_embed, clean_vector, telemetry)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == _CORPUS_JSON_KEYS           # full contract even when red
    assert "sweep" in obj["degradations"]
    assert obj["nodes_swept"] < obj["nodes_total"]        # partial, not silently truncated


# =========================================================================== #
# W12 / KD1 — the no-write fence: cmd_check + _run_staged_check write zero graph rows
# =========================================================================== #

def test_w12_cmd_check_corpus_writes_zero_graph_mutations(
    workspace, monkeypatch, capsys,
) -> None:
    """A full ``cmd_check`` corpus run over a ``_WriteSpyStore`` fires ZERO graph mutations.

    The behavioral half of W12/KD1: the CLI check path sits outside the static AST lint's
    closure (rooted at ``check.py``), so its write-freedom is proven by exercising it —
    plan → confirm → execute → probe → summary — over a store whose every mutator raises.
    A finding is produced (the path is fully driven), and no mutator fired.
    """
    config, store, telemetry = workspace
    _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(
        monkeypatch,
        {"Hardening axiom alpha for the check verb.": [_match("hard-b", 0.9)],
         "Hardening axiom beta for the check verb.": []},
    )
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    spies: List[_WriteSpyStore] = []

    def _spy(path: str) -> _WriteSpyStore:
        spy = _WriteSpyStore(GraphStore(path))
        spies.append(spy)
        return spy

    monkeypatch.setattr(cli, "GraphStore", _spy)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 1                                      # a finding — the path ran to completion
    assert "[Conflict]" in capsys.readouterr().out
    assert len(spies) == 1 and spies[0].mutations == []   # zero graph mutations fired


def test_w12_run_staged_check_writes_zero_graph_mutations(
    workspace, monkeypatch, capsys,
) -> None:
    """A ``_run_staged_check`` gate run over a ``_WriteSpyStore`` fires ZERO graph mutations.

    The staged half of the same fence: the pure-read pending predicate, the probe, the
    facade sweep, and the ``surface='check'`` persistence all run against the spy — none is
    a graph write (the persistence is telemetry-only). Extends ``test_1c`` (the predicate's
    pure-read pin) to the whole staged orchestration.
    """
    config, store, telemetry = workspace
    _commit(store, "active-q", _ACTIVE_AXIOM)
    _drain_outbox(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _SequenceJudge([
        _execution([("active-q", False, 0.9, "They cannot both stand.")], batch_id="staged-b0"),
    ]))
    spies: List[_WriteSpyStore] = []

    def _spy(path: str) -> _WriteSpyStore:
        spy = _WriteSpyStore(GraphStore(path))
        spies.append(spy)
        return spy

    monkeypatch.setattr(cli, "GraphStore", _spy)

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 1                                      # the pending contradiction gates
    assert "[Conflict]" in capsys.readouterr().out
    assert len(spies) == 1 and spies[0].mutations == []   # zero graph mutations fired
