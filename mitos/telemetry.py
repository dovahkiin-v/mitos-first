"""Sibling telemetry store for the Conflict sensor's non-rebuildable judgments.

The Conflict sensor produces telemetry that **cannot be regenerated** from
``decisions.md``: a SONNET tenability judgment fired once, at temperature, against
a specific prompt + model generation — replay it later and a *different* rationale
comes back, so the one that fired is lost forever unless it is written down whole
(CONF-D8, M8). That corpus is the training set the future ML conflict-classifier
graduates on (OD3), and the observability surface that turns the vision's cost/P95
budgets from assertions into measurements (P15/P16).

Because Mitos's recovery model treats the whole graph as disposable — ``rm
graph.sqlite`` / ``mitos rebuild`` rebuilds it from ``decisions.md`` — a corpus
living *inside* the graph would be silently destroyed on recovery. So it lives in
its **own** file, a sibling ``.mitos/telemetry.sqlite`` fenced off from the
rebuildable graph (P7 bulkhead): the swap/backup machinery in ``cutover.py`` only
ever touches ``graph.sqlite``-derived paths, so a different basename survives the
truth-rebuild untouched (CONF-D8; the T8 guarantee).

This module pairs two append-only whole-row **writers** (``record_judged_batch``,
the per-batch judgment grain; ``record_check_run``, the per-run summary grain —
the CHK-D7 run memory) with one non-mutating bulk **reader** —
``load_reuse_index``, the corpus's first in-tool consumer (verdict
reuse + novelty; CHK-D3/D10). The reader opens the sibling DB ``mode=ro`` so it
*physically* cannot mutate the corpus, and it never chains derived state (it reads
the primary judged rows verbatim, M8). There is still no retention, decay, or
pruning (P4 deferred), and the full CONF-C1 ``edges ⋈ conflict_checks`` corpus join
still belongs to a future vision. Every judged row lands **whole and uncapped** —
the longest, thorniest contradiction is exactly the example the future classifier
will most need (CONF-D8 verbatim-and-whole).

Tier 2 (logic/store): it reuses ``store.open_connection`` (the MI-8 connection
chokepoint) and ``migrations.run_migrations`` (the ladder primitive) rather than
re-implementing either, and defines its **own** ``TELEMETRY_MIGRATION_STEPS`` in a
separate ``user_version`` space so it never touches the graph's ladder. Nothing
here imports the sync/CLI/MCP orchestration layers; the sync surface (Phase 5b)
imports *this*, never the reverse — so ``telemetry -> {store, migrations, errors}``
stays acyclic.
"""

import contextlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mitos.errors import DatabaseError
from mitos.migrations import MigrationStep, run_migrations
from mitos.store import open_connection

# --- Boundary-crossing row shapes ---------------------------------------------
#
# Three frozen dataclasses cross the persistence boundary (Python -> parameterized
# INSERT -> STRICT columns -> read-back). Each carries a ``to_params()`` producing
# a value tuple in a *fixed column order* that must match the module-level column
# tuples below, so a single INSERT is fully parameterized (P8) and column/param
# order cannot drift (the per-tuple length is lockstep-pinned by test).


@dataclass(frozen=True)
class ConflictCheckRow:
    """One judged candidate pair — the per-pair grain of the sensor's telemetry.

    Every field is stored **verbatim and uncapped**: the fed contexts
    (``judged_axiom``, both sides' ``rejected_paths``/``scope``) are the primary
    source of what the sensor actually judged (M8), not recoverable from a content
    hash, so nothing is truncated or normalized on the way in. The multi-valued
    ``rejected_paths``/``scope`` arrive **pre-serialized as TEXT** exactly as the
    judge saw them — this writer does not re-encode them (5b/3b own the fed-context
    rendering); no tuples are persisted.

    The additive, overcount-prone batch metrics (``token_*``/``elapsed_ms``) live
    on :class:`JudgmentBatch`, NOT here — copied per-row they would multiply true
    spend by the batch size under a naive ``SUM`` (RF-2/D1). Non-additive
    provenance (``model_alias``/``prompt_version``/``mitos_version``) stays per-row:
    it is never summed, so per-row duplication is harmless.

    Attributes:
        batch_id: Join key to :class:`JudgmentBatch` (a plain column, NOT an FK — a
            training label must outlive graph surgery, CONF-D8). Minted by 3b/5b.
        sync_run_id: The originating command run's id — the sync run today; the
            check run's ``run_id`` once the corpus-audit surface lands (CHK-D7
            widened the semantics; the column name stays, P1 additive evolution).
            Nullable for the deferred non-sync write-time path; the sync and check
            writers always supply it.
        surface: Which command surface fired the judgment — exact strings
            ``'sync'`` (the sync-time sensor) or ``'check'`` (the corpus audit);
            ``'record'`` is reserved for the record-write-pause vision. The label
            corpus must not fork by surface (CHK-D7), so every writer stamps this
            explicitly: the field is defaultless on purpose — the schema's
            permanent ``DEFAULT 'sync'`` is only the rung-2 backfill and can never
            catch an omitting writer; this constructor can.
        judged_axiom: The proposal's axiom text as fed to the judge, verbatim.
        proposal_rejected_paths: The proposal's ``rejected_paths`` fed context, or
            ``None`` when absent on the parsed entry.
        proposal_scope: The proposal's scope fed context, or ``None`` for
            unscoped/global (MI-9: absent = zero tags, never stored as ``""``).
        proposed_hash_if_any: The proposal's content hash. Nullable for the
            deferred write-time draft path; sync rows always supply it.
        candidate_slug: The candidate decision's slug — a debugging/citation handle
            stored verbatim (no casefolding: 1b does no slug comparison).
        candidate_hash: The candidate's M2 content hash (a plain column, not an FK).
        candidate_rejected_paths: The candidate's ``rejected_paths`` fed context.
            NOT NULL — M5 makes ``rejected_paths`` required on every decision.
        candidate_scope: The candidate's scope fed context, or ``None`` for
            unscoped/global (MI-9).
        tenable: The judge's verdict (stored INTEGER 0/1).
        confidence: The judge's raw self-reported confidence in ``[0, 1]``.
        surfaced: Whether the confidence gate fired (stored INTEGER 0/1).
        candidate_source: How the candidate was gathered (``"embedding_topk"`` in
            v0.2).
        model_alias: The family+tier alias of the judging model (never a raw model
            ID — P19; 3b/5b hand it over).
        prompt_version: The judgment prompt version that produced this row.
        mitos_version: The Mitos version that fired the judgment.
        rationale: The judge's rationale — NOT NULL, non-regenerable output (M8).
    """

    batch_id: str
    sync_run_id: Optional[str]
    surface: str
    judged_axiom: str
    proposal_rejected_paths: Optional[str]
    proposal_scope: Optional[str]
    proposed_hash_if_any: Optional[str]
    candidate_slug: str
    candidate_hash: str
    candidate_rejected_paths: str
    candidate_scope: Optional[str]
    tenable: bool
    confidence: float
    surfaced: bool
    candidate_source: str
    model_alias: str
    prompt_version: str
    mitos_version: str
    rationale: str

    def to_params(self, created_at: str) -> Tuple:
        """Produces the INSERT parameter tuple in ``_CONFLICT_CHECKS_COLUMNS`` order.

        The bool -> 0/1 conversion for ``tenable``/``surfaced`` is the primary type
        guard (STRICT SQLite has no BOOLEAN affinity); ``created_at`` is stamped
        here rather than read from the wall clock, so the whole batch shares one
        caller-supplied UTC ISO-8601 timestamp (MI-10, D5).

        Args:
            created_at: The batch-wide UTC ISO-8601 timestamp to stamp on this row.

        Returns:
            A value tuple positionally matching ``_CONFLICT_CHECKS_COLUMNS``.
        """
        return (
            self.batch_id,
            self.sync_run_id,
            self.surface,
            self.judged_axiom,
            self.proposal_rejected_paths,
            self.proposal_scope,
            self.proposed_hash_if_any,
            self.candidate_slug,
            self.candidate_hash,
            self.candidate_rejected_paths,
            self.candidate_scope,
            int(self.tenable),
            self.confidence,
            int(self.surfaced),
            self.candidate_source,
            self.model_alias,
            self.prompt_version,
            self.mitos_version,
            self.rationale,
            created_at,
        )


@dataclass(frozen=True)
class JudgmentBatch:
    """The per-batch grain: the additive metrics of one batched judgment call.

    These facts describe the **single batched call**, not any individual candidate
    pair, so they live in their own side-table keyed on ``batch_id`` (PK). That
    makes exactly-once a *structural* property — one metrics row per batch — and
    keeps the append-only writer uniform (no "is-this-the-designated-row" branch
    over ``conflict_checks``). A naive ``SUM(token_input)`` over this table returns
    true spend, never ``N x`` the batch size (RF-2/D1).

    All five metrics are caller-supplied (3b measures them from the Anthropic
    response; 1b only stores) — the writer performs zero wall-clock or API reads.

    Attributes:
        batch_id: The PK; the same minted id string carried by every
            :class:`ConflictCheckRow` of the batch.
        model_id: The resolved versioned model id live at call time (what
            ``models.get_model_id`` returned for the execution's alias) — the
            CHK-D3 staleness-bound provenance. Provenance-only: never part of any
            reuse key (that pins the per-row ``model_alias``). ``None`` when
            resolution failed (and on pre-rung-2 rows); defaultless on purpose —
            ``None`` is a legal *explicit* value, silence is not.
        token_input: Input tokens billed for the batched call.
        token_output: Output tokens billed for the batched call.
        token_cache_read: Cache-read tokens billed for the batched call.
        token_cache_creation: Cache-creation tokens billed for the batched call.
        elapsed_ms: Wall-clock latency of the batched call, in milliseconds.
        stop_reason: The API ``stop_reason`` from the response (``"tool_use"``,
            ``"max_tokens"``, …). Nullable TEXT — ``None`` on pre-rung-4 rows and
            when the executor returns ``Unavailable`` before constructing this object.
    """

    batch_id: str
    model_id: Optional[str]
    token_input: int
    token_output: int
    token_cache_read: int
    token_cache_creation: int
    elapsed_ms: int
    stop_reason: Optional[str] = None

    def to_params(self) -> Tuple:
        """Produces the INSERT parameter tuple in ``_JUDGMENT_BATCHES_COLUMNS`` order.

        Returns:
            A value tuple positionally matching ``_JUDGMENT_BATCHES_COLUMNS``.
        """
        return (
            self.batch_id,
            self.model_id,
            self.token_input,
            self.token_output,
            self.token_cache_read,
            self.token_cache_creation,
            self.elapsed_ms,
            self.stop_reason,
        )


@dataclass(frozen=True)
class CommentaryAuditRow:
    """One commentary-reconcile attribution — the P8 row for an in-place graph edit.

    Records BOTH sides of the change, not just the prior values. Self-detection of a
    crash-phantom depends on both: its ``new_values`` can be compared against the graph,
    and chain consistency fingers it historically too, because a phantom's successor
    records a ``prior`` equal to the phantom's ``prior`` rather than its ``new`` — which
    holds for the double-edit and revert cases alike. Note the entry loop does NOT run
    under sync's ``FileLock`` (only the snapshot, rotation and record writes do), so two
    concurrent syncs can interleave rows: a phantom is not guaranteed to be the NEWEST
    row, and the chain comparison rather than recency is the reliable test.

    Attributes:
        audit_id: This row's id (``uuid4().hex``), minted by the caller.
        node_id: The reconciled node — a plain column, NOT an FK (the graph is a
            different, disposable file, and the attribution must outlive its node).
        slug: The node's slug, stored verbatim as a citation handle.
        fields_changed: The diverged field names.
        prior_values: The values as the GRAPH held them before the reconcile.
        new_values: The values the corpus now declares.
        mitos_version: The build that wrote the row.
    """

    audit_id: str
    node_id: Optional[str]
    slug: Optional[str]
    fields_changed: List[str]
    prior_values: Dict[str, Any]
    new_values: Dict[str, Any]
    mitos_version: str

    def to_params(self, created_at: str) -> Tuple[Any, ...]:
        """Returns the INSERT value tuple in ``_COMMENTARY_AUDIT_COLUMNS`` order.

        Args:
            created_at: The application-supplied UTC ISO-8601 stamp (MI-10).

        Returns:
            The parameter tuple for one intent row.
        """
        return (
            self.audit_id,
            self.node_id,
            self.slug,
            json.dumps(list(self.fields_changed)),
            json.dumps(self.prior_values, sort_keys=True),
            json.dumps(self.new_values, sort_keys=True),
            None,  # outcome — an intent row is OPEN until a correlated row closes it
            None,  # correlates_to — set only on an outcome row
            self.mitos_version,
            created_at,
        )


@dataclass(frozen=True)
class CheckRunRow:
    """One check run's summary — the per-run grain of the audit's memory (CHK-D7).

    A dumb boundary row: every semantic decision behind these values — the NULL
    rules, the ``degraded_reason`` token vocabulary, the finding/novelty counts —
    lives in the check engine's row builder (``check.check_run_row_from_result``),
    never here. Telemetry stores what it is handed; the DDL comments on
    ``check_runs`` (above) are the NULL-semantics authority. Defaultless on
    purpose (the 1a fence idiom): ``None`` is a legal *explicit* value for the
    nullable columns, silence is not — an omitting writer is a ``TypeError`` at
    construction, which the schema's constraints alone could never catch.

    Attributes:
        run_id: The CHK-D7 correlation id — the PK, so "one summary row per run"
            is structural; the same id rides this run's ``conflict_checks`` rows
            as ``sync_run_id`` (the one-thread-of-truth join).
        mode: ``'corpus'`` or ``'staged'`` (the schema CHECKs the closed set; the
            engine's builder validates it before any connection opens).
        started_at: Caller-supplied UTC ISO-8601 plan-entry stamp (MI-10).
        ended_at: Caller-supplied UTC ISO-8601 execute-exit stamp (MI-10).
        exit_code: The actual process exit for the run — 0, 1 or 2 (CHECKed).
        nodes_swept: Always known at run end, even degraded.
        pairs_judged_fresh: Always known at run end.
        pairs_reused: Reuse-unavailable ⇒ 0, a TRUE zero (the run genuinely
            reused nothing) — never NULLed.
        findings_new: ``None`` = novelty partition unavailable (CHK-D10), NOT a
            zero.
        findings_known: ``None`` = partition unavailable, same line.
        coverage_exclusions: The exclusion COUNT (named nodes ride the report);
            ``None`` = the probe never completed.
        degraded_reason: ``None`` = healthy; else the engine's comma-joined
            degradation tokens (the vocabulary is the engine's, P18).
        mitos_version: The Mitos version that ran the check — NOT NULL (a
            reuse-only run writes no ``conflict_checks`` rows to join for it).
    """

    run_id: str
    mode: str
    started_at: str
    ended_at: str
    exit_code: int
    nodes_swept: int
    pairs_judged_fresh: int
    pairs_reused: int
    findings_new: Optional[int]
    findings_known: Optional[int]
    coverage_exclusions: Optional[int]
    degraded_reason: Optional[str]
    mitos_version: str

    def to_params(self) -> Tuple:
        """Produces the INSERT parameter tuple in ``_CHECK_RUNS_COLUMNS`` order.

        Takes no ``created_at`` argument (unlike :meth:`ConflictCheckRow.to_params`):
        this row carries its own MI-10 stamps as fields — ``started_at``/``ended_at``
        are run facts echoed off the result, not a write-time annotation.

        Returns:
            A value tuple positionally matching ``_CHECK_RUNS_COLUMNS``.
        """
        return (
            self.run_id,
            self.mode,
            self.started_at,
            self.ended_at,
            self.exit_code,
            self.nodes_swept,
            self.pairs_judged_fresh,
            self.pairs_reused,
            self.findings_new,
            self.findings_known,
            self.coverage_exclusions,
            self.degraded_reason,
            self.mitos_version,
        )


# --- Schema (migration ladder step 1) -----------------------------------------
#
# Telemetry owns a SEPARATE migration ladder in a SEPARATE file with its OWN
# ``user_version`` space — it must NEVER reuse the graph's ``MIGRATION_STEPS`` (its
# default arg is the live graph ladder; omitting ``steps`` would build
# ``nodes``/``edges`` inside ``telemetry.sqlite``). The DDL mirrors
# ``migrations._v1_schema``'s discipline: each CREATE TABLE is a SINGLE statement
# run via ``conn.execute`` (never ``executescript`` — it force-commits and would
# split the DDL out of the runner's atomic version-bump transaction); no
# ``DEFAULT CURRENT_TIMESTAMP`` (``created_at`` is application-supplied, MI-10); all
# identifiers are code-internal literals (P8); STRICT eliminates SQLite's permissive
# typing so type discipline is a property of the tables, not of a forgettable check.

# ``judgment_batches`` — the additive batch metrics (RF-2/D1). ``batch_id`` PRIMARY
# KEY makes "one metrics row per batch" structural. STRICT has no BOOLEAN affinity;
# every metric is a NOT NULL INTEGER.
_JUDGMENT_BATCHES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS judgment_batches (
        batch_id TEXT NOT NULL,
        token_input INTEGER NOT NULL,
        token_output INTEGER NOT NULL,
        token_cache_read INTEGER NOT NULL,
        token_cache_creation INTEGER NOT NULL,
        elapsed_ms INTEGER NOT NULL,
        PRIMARY KEY (batch_id)
    ) STRICT;
"""

# ``conflict_checks`` — one row per judged candidate pair (§8 field contract). No
# PRIMARY KEY: this is an append-only log (the implicit rowid suffices), and a
# natural-key PK could wrongly reject a legitimate re-judgment. No row-size cap —
# the verbatim fed contexts land whole (CONF-D8). ``batch_id`` is a plain column,
# NOT an FK (a training label outlives graph surgery). ``tenable``/``surfaced`` are
# INTEGER (STRICT has no BOOLEAN affinity); the CHECKs are belt-and-suspenders over
# the dataclass bool->int guard (§14 latitude; literal, no interpolation).
_CONFLICT_CHECKS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS conflict_checks (
        batch_id TEXT NOT NULL,
        sync_run_id TEXT,
        judged_axiom TEXT NOT NULL,
        proposal_rejected_paths TEXT,
        proposal_scope TEXT,
        proposed_hash_if_any TEXT,
        candidate_slug TEXT NOT NULL,
        candidate_hash TEXT NOT NULL,
        candidate_rejected_paths TEXT NOT NULL,
        candidate_scope TEXT,
        tenable INTEGER NOT NULL,
        confidence REAL NOT NULL,
        surfaced INTEGER NOT NULL,
        candidate_source TEXT NOT NULL,
        model_alias TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        mitos_version TEXT NOT NULL,
        rationale TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (tenable IN (0, 1)),
        CHECK (surfaced IN (0, 1)),
        CHECK (confidence >= 0.0 AND confidence <= 1.0)
    ) STRICT;
"""

# One statement per list element — ``conn.execute`` runs only the first statement
# in a string. Order is immaterial (no FK between the two tables).
_TELEMETRY_V1_STATEMENTS: List[str] = [
    _JUDGMENT_BATCHES_SCHEMA,
    _CONFLICT_CHECKS_SCHEMA,
]


def _conflict_checks_schema(conn: sqlite3.Connection) -> None:
    """Migration step 1: create the telemetry schema (``judgment_batches`` + ``conflict_checks``).

    Issues each ``CREATE TABLE ... STRICT`` via ``conn.execute`` (one per statement,
    never ``executescript`` — that force-commits and breaks the runner's atomic
    DDL+version-bump rollback). Does NOT touch ``user_version`` or manage the
    transaction — ``run_migrations`` owns both. Idempotent by the ``IF NOT EXISTS``
    backstop (MI-3 replay).

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction
            (opened via ``store.open_connection``, MI-8).
    """
    for statement in _TELEMETRY_V1_STATEMENTS:
        conn.execute(statement)


# --- Schema (migration ladder step 2) -------------------------------------------
#
# Rung 2 (the `mitos check` vision, CHK-D3/D7): attribute every judgment row to the
# command surface that fired it, stamp the resolved model id on each batch, and give
# check runs a summary table so a sweep's story survives process exit (P18/P20 —
# unrecorded runs are unrecoverable). All three are additive: no existing column is
# renamed, retyped, or dropped (P1 interoperability-across-time). Replay safety
# rides the runner's ``user_version`` rung guard, NOT statement re-runnability — a
# bare ALTER raises "duplicate column" on a re-run, and that is fine because an
# applied rung is never re-entered (MI-3). The ALTERs target rung-1 tables; rung
# ORDER (1 before 2) is what makes a fresh install replay 1→2 cleanly — never amend
# rung 1's shipped CREATEs.

# ``conflict_checks.surface`` — which surface fired the judgment: 'sync' | 'check'
# ('record' reserved for the record-write-pause vision). SQLite requires ``ADD
# COLUMN ... NOT NULL`` to carry a DEFAULT and makes it PERMANENT, so ``DEFAULT
# 'sync'`` is the backfill for pre-rung-2 rows AND the reason the constraint can
# never catch a writer omitting the column — the fence is the defaultless dataclass
# field plus the writers' column-tuple lockstep test (CHK-D7). No value CHECK:
# 'record' is a named reserved value with a committed future consumer, and widening
# a CHECK later costs a table rebuild — CHECK closed contracts, never an evolving
# vocabulary (contrast ``check_runs.mode``/``exit_code`` below).
_CONFLICT_CHECKS_ADD_SURFACE = """
    ALTER TABLE conflict_checks ADD COLUMN surface TEXT NOT NULL DEFAULT 'sync';
"""

# ``judgment_batches.model_id`` — the resolved versioned id live at call time
# (CHK-D3: makes the alias-vs-version staleness bound observable). Nullable: NULL on
# pre-rung-2 rows and when resolution fails — provenance-only, so losing a batch
# (whose rationale is non-regenerable, M8) over a provenance lookup would invert the
# value hierarchy. STRICT requires the explicit datatype on ADD COLUMN.
_JUDGMENT_BATCHES_ADD_MODEL_ID = """
    ALTER TABLE judgment_batches ADD COLUMN model_id TEXT;
"""

# ``check_runs`` — one summary row per check run: the run-level repetition memory
# (P18: standing-count trend, time-to-resolution/nag-count per finding, the
# long-standing-unresolved FP-suspect list). Written best-effort at run end by
# ``TelemetryStore.record_check_run``, which ships and is called from both
# ``cli.cmd_check`` and ``cli._run_staged_check`` — so a workspace `mitos check` has
# run in carries a row per run, save the runs whose telemetry store could not be
# constructed at all (``cli._build_check_telemetry`` returns None: the engine's
# documented ``reuse_read`` degradation, where the KD5 seam records nothing). That
# is the honest bound on "best-effort" and it is narrow — every ordinary run writes.
# (Named by function rather than by line number: the two call sites drift with every
# phase that edits ``cli.py``.)
# Check-freshness state therefore EXISTS on disk today; the conditional
# coherence-freshness gate is deferred on what a row MEANS — a run happened, not
# that given writes are covered — never on the state being unavailable. See ADR
# ``coherence-pointer-deferral-rests-on-check-runs-semantics-not-impossibility``,
# which names this comment as the artifact a reader re-deriving the old
# impossibility claim would land on.
# Nullability is the semantic line, pinned NOW because SQLite constraints are
# rebuild-only to change: NOT NULL = the run always knows it at run end; NULL =
# "value not computable this run", deliberately distinct from a genuine zero
# (CHK-D10 — a degraded run must never masquerade as "zero findings").
# ``mode``/``exit_code`` DO get belt-and-suspenders CHECKs: those sets are pinned
# shipped APIs (CHK-C2), closed contracts — not evolving vocabularies.
_CHECK_RUNS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS check_runs (
        run_id TEXT NOT NULL,                -- CHK-D7 correlation id (uuid4().hex); also lands in conflict_checks.sync_run_id
        mode TEXT NOT NULL,                  -- 'corpus' | 'staged' (closed set, CHECKed)
        started_at TEXT NOT NULL,            -- application-supplied UTC ISO-8601, never CURRENT_TIMESTAMP (MI-10)
        ended_at TEXT NOT NULL,              -- application-supplied UTC ISO-8601 (MI-10)
        exit_code INTEGER NOT NULL,          -- the shipped 0/1/2 exit contract (CHK-C2)
        nodes_swept INTEGER NOT NULL,        -- always known at run end, even degraded
        pairs_judged_fresh INTEGER NOT NULL, -- always known at run end
        pairs_reused INTEGER NOT NULL,       -- reuse-unavailable => 0, a TRUE zero
        findings_new INTEGER,                -- NULL = partition unavailable (CHK-D10), NOT a zero
        findings_known INTEGER,              -- NULL = partition unavailable (CHK-D10), NOT a zero
        coverage_exclusions INTEGER,         -- count only (named nodes ride the report); NULL = probe never completed
        degraded_reason TEXT,                -- NULL = healthy; value vocabulary is the engine's, unconstrained here
        mitos_version TEXT NOT NULL,         -- a reuse-only run writes no conflict_checks rows to join for version
        PRIMARY KEY (run_id),
        CHECK (mode IN ('corpus', 'staged')),
        CHECK (exit_code IN (0, 1, 2))
    ) STRICT;
"""

# One statement per list element — ``conn.execute`` runs only the first statement
# in a string. The two ALTERs must not be merged with the CREATE.
_TELEMETRY_V2_STATEMENTS: List[str] = [
    _CONFLICT_CHECKS_ADD_SURFACE,
    _JUDGMENT_BATCHES_ADD_MODEL_ID,
    _CHECK_RUNS_SCHEMA,
]


def _check_attribution_schema(conn: sqlite3.Connection) -> None:
    """Migration step 2: surface attribution + model provenance + ``check_runs``.

    Adds ``conflict_checks.surface`` (existing rows backfill ``'sync'`` via the
    permanent DEFAULT — a schema operation, no row is UPDATEd),
    ``judgment_batches.model_id`` (NULL on pre-rung-2 rows), and creates the empty
    ``check_runs`` summary table. Issues each statement via ``conn.execute`` (one
    per statement, never ``executescript``); does NOT touch ``user_version`` or
    manage the transaction — ``run_migrations`` owns both. Replay-safe via the
    runner's rung guard, not statement re-runnability (a bare ALTER raises on
    re-run; an applied rung is never re-entered).

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction
            (opened via ``store.open_connection``, MI-8).
    """
    for statement in _TELEMETRY_V2_STATEMENTS:
        conn.execute(statement)


# --- Rung 3: the commentary-reconcile attribution row (P8) --------------------
#
# Every applied commentary reconcile writes one durable record of WHAT changed.
# Without it, an agent's markdown edit plus `sync --yes` is exactly as unattributable
# afterwards as mutating the graph directly would have been — `updated_at` says only
# THAT something changed, never what. P8 requires every graph mutation to be
# attributable, so this row is what makes the reconcile's argument hold rather than
# polish on top of it.
#
# Homed HERE and not in `graph.sqlite` because after a reconcile the prior values
# exist nowhere else on earth: the markdown was rewritten by the agent, the graph was
# updated by the reconcile. That is non-rebuildable data in the exact sense two
# recorded ADRs already ruled on, both the same way — the corpus must survive the
# truth-rebuild. A graph-homed table would contradict recorded precedent without
# superseding it, which in a decision-memory tool is corpus incoherence.
#
# Honest limit: the row attributes WHAT and WHEN, not WHO. `sync` has no way to know
# who edited the markdown, and the `git log decisions.md` fallback fails at home —
# this project gitignores its own corpus.
_COMMENTARY_AUDIT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS commentary_audit (
        audit_id TEXT NOT NULL,        -- uuid4().hex, minted per row (intent or outcome)
        node_id TEXT,                  -- the reconciled node; a PLAIN COLUMN, NOT an FK
                                       -- (the graph is a different, disposable file, and
                                       -- an attribution row must outlive its node —
                                       -- same precedent as batch_id above). NULL on an
                                       -- outcome row, which correlates instead.
        slug TEXT,                     -- citation handle, verbatim, for reading by eye
        fields_changed TEXT,           -- JSON array of the diverged field names
        prior_values TEXT,             -- JSON object: the values as the GRAPH held them
        new_values TEXT,               -- JSON object: the values the corpus now declares
        outcome TEXT,                  -- NULL on an intent row; a reason string on the
                                       -- correlated outcome row that CLOSES a failure
        correlates_to TEXT,            -- an outcome row's intent audit_id (plain column)
        mitos_version TEXT NOT NULL,   -- which build wrote the row
        created_at TEXT NOT NULL,      -- application-supplied UTC ISO-8601 (MI-10)
        PRIMARY KEY (audit_id)
    ) STRICT;
"""


def _commentary_audit_schema(conn: sqlite3.Connection) -> None:
    """Migration step 3: the commentary-reconcile attribution table.

    A plain additive rung — one CREATE, no ALTER, no table rebuild, nothing to
    back-fill — so reverting the release leaves one unread table in a sibling DB with
    no data lost and no contract broken. Does NOT touch ``user_version`` or manage the
    transaction; ``run_migrations`` owns both.

    **Death story (P4).** Rows live as long as the workspace. Retention is
    **explicitly deferred**, following this module's own deferred-P4 precedent for the
    judgment corpus ("no retention, decay, or pruning"): an attribution trail that
    silently forgot would be worse than none, because a reader could not tell an
    unattributed mutation from a pruned one. When retention does arrive it must be an
    explicit, recorded decision — not a default that grew in.

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction
            (opened via ``store.open_connection``, MI-8).
    """
    conn.execute(_COMMENTARY_AUDIT_SCHEMA)


# Telemetry's OWN ladder registry — a separate ``user_version`` space from the
# graph's ``migrations.MIGRATION_STEPS``. ONE binding statement, ever: a future
# telemetry migration extends this literal or ``.append((3, ...))`` after it —
# never a second binding (a rebind is invisible to a def-time-bound default arg;
# the graph ladder learned this).
def _judgment_batches_add_stop_reason(conn: "sqlite3.Connection") -> None:
    """Rung 4: ``judgment_batches.stop_reason`` — the API response's stop reason."""
    conn.execute(
        "ALTER TABLE judgment_batches ADD COLUMN stop_reason TEXT;"
    )


TELEMETRY_MIGRATION_STEPS: List[MigrationStep] = [
    (1, _conflict_checks_schema),
    (2, _check_attribution_schema),
    (3, _commentary_audit_schema),
    (4, _judgment_batches_add_stop_reason),
]


# --- Parameterized INSERTs ----------------------------------------------------
#
# Column tuples are the single source of INSERT column order; ``to_params`` returns
# values in exactly these orders. The SQL is assembled from code-internal column
# LITERALS and a ``?`` placeholder per column — every VALUE binds via ``?`` (P8);
# the only interpolation is over the whitelisted identifiers/placeholder count,
# never user/LLM data (the documented P8 carve-out for code-internal identifiers).
_CONFLICT_CHECKS_COLUMNS: Tuple[str, ...] = (
    "batch_id",
    "sync_run_id",
    "surface",
    "judged_axiom",
    "proposal_rejected_paths",
    "proposal_scope",
    "proposed_hash_if_any",
    "candidate_slug",
    "candidate_hash",
    "candidate_rejected_paths",
    "candidate_scope",
    "tenable",
    "confidence",
    "surfaced",
    "candidate_source",
    "model_alias",
    "prompt_version",
    "mitos_version",
    "rationale",
    "created_at",
)

_JUDGMENT_BATCHES_COLUMNS: Tuple[str, ...] = (
    "batch_id",
    "model_id",
    "token_input",
    "token_output",
    "token_cache_read",
    "token_cache_creation",
    "elapsed_ms",
    "stop_reason",
)

# INSERT order == the ``check_runs`` DDL order == the ``test_check_runs_column_contract``
# pin — one source of truth, lockstep-tested against ``CheckRunRow.to_params()``.
_CHECK_RUNS_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "mode",
    "started_at",
    "ended_at",
    "exit_code",
    "nodes_swept",
    "pairs_judged_fresh",
    "pairs_reused",
    "findings_new",
    "findings_known",
    "coverage_exclusions",
    "degraded_reason",
    "mitos_version",
)


def _insert_sql(table: str, columns: Tuple[str, ...]) -> str:
    """Builds a fully-parameterized INSERT over code-internal column literals.

    Args:
        table: The target table name (a code-internal literal).
        columns: The column names, in the order ``to_params`` emits values.

    Returns:
        An ``INSERT INTO table (cols...) VALUES (?, ?, ...)`` statement with one
        ``?`` per column — every value binds as a parameter (P8).
    """
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


_INSERT_CONFLICT_CHECK_SQL = _insert_sql("conflict_checks", _CONFLICT_CHECKS_COLUMNS)
_INSERT_JUDGMENT_BATCH_SQL = _insert_sql("judgment_batches", _JUDGMENT_BATCHES_COLUMNS)
_INSERT_CHECK_RUN_SQL = _insert_sql("check_runs", _CHECK_RUNS_COLUMNS)


# --- Read-side boundary shapes (verdict-reuse / novelty projection) -----------
#
# The corpus's first consumer-facing read. ``load_reuse_index`` projects
# ``conflict_checks`` into an in-memory, orientation-blind, latest-wins verdict
# index the run engine (2c) consults once per run for BOTH the CHK-D3 reuse
# partition (reuse yesterday's verdict at zero LLM spend) and the CHK-D10 novelty
# partition (tell a new tension from a standing one). Both readers want the SAME
# rows read the SAME way, so there is one read path, not two.


@dataclass(frozen=True)
class StoredVerdict:
    """One prior judged verdict, the latest for its pair at the queried pins.

    Carries only the raw judge output the engine needs — NOT a pre-derived
    ``is_finding`` bool. "Is this verdict a finding?" is ``not tenable ∧ confidence
    ≥ CONFLICT_SURFACE_THRESHOLD`` (CONF-D4), and that threshold lives in
    ``conflict.py`` where the run engine already imports it; deriving it here would
    couple the sibling store to the judgment leaf and fork the one gate formula
    (Key Decision 4). The engine applies the formula to both the reused verdict and
    the latest-prior verdict — a single gate site.

    Attributes:
        tenable: The judge's verdict (``0/1`` in SQLite → ``bool`` here).
        confidence: The judge's raw self-reported confidence in ``[0, 1]``.
        rationale: The judge's verbatim output — the reused finding's report text
            (M8: the standing-finding report reuses this, never a re-render).
        batch_id: The source batch — CHK-D3 reuse provenance.
        created_at: The row's UTC ISO-8601 stamp — reuse provenance and the
            latest-wins ordering key (MI-10).
    """

    tenable: bool
    confidence: float
    rationale: str
    batch_id: str
    created_at: str


@dataclass(frozen=True)
class ReuseUnavailable:
    """Typed degradation: the read itself failed (corrupt image, unopenable path).

    Returned, never raised — a read failure must not abort the run, and must never
    be a silent empty (the ``except: return {}`` the typed-degradation pattern
    forbids). It is a **different type** from an empty :class:`ReuseIndex`: an empty
    index means *no priors* (a healthy first-run state), this means *the read broke*.
    The engine's exit-0-vs-exit-2 fork keys on exactly that line — both partitions
    (reuse AND novelty) read the same rows in the same call, so one read failure
    takes out both, and the engine's response is run-as-``--fresh`` **and**
    report-unpartitioned (Key Decision 5).

    Telemetry-local by design: ``conflict.Unavailable``'s reasons are all
    pipeline-substrate (EMBEDDING/VECTOR_STORE/JUDGMENT/…); bolting a TELEMETRY
    member onto that enum would couple telemetry's failure vocabulary into the
    conflict leaf (the wrong direction) and force this module to import ``conflict``.

    Attributes:
        detail: The underlying exception message — logging/telemetry ONLY, never
            user-rendered (mirrors ``conflict.Unavailable.detail``).
    """

    detail: str


class ReuseIndex:
    """An in-memory, orientation-blind, latest-wins index of prior verdicts.

    Wraps ``{ tuple(sorted((h_lo, h_hi))) : StoredVerdict }`` — one entry per judged
    pair, keyed on the sorted content-hash pair so a lookup hits regardless of which
    side was the proposal when the row was written (which side was the sweep node is
    a discovery-direction accident, CHK-D2/Key Decision 3). Every entry is confined
    to rows matching the ``prompt_version`` + ``model_alias`` the index was loaded at.
    An empty index (``len == 0``) is a healthy no-priors state, distinct from
    :class:`ReuseUnavailable`.
    """

    def __init__(self, verdicts: "Dict[Tuple[str, str], StoredVerdict]") -> None:
        """Wraps the built pair→verdict map (the reader owns construction).

        Args:
            verdicts: The latest-wins map keyed on the sorted hash pair.
        """
        self._verdicts = verdicts

    def lookup(self, hash_a: str, hash_b: str) -> Optional[StoredVerdict]:
        """Returns the latest verdict for the unordered pair, or ``None`` on a miss.

        Orientation-blind: ``lookup(a, b)`` and ``lookup(b, a)`` resolve identically.
        A ``None`` means no prior verdict at these pins (the caller need not sort).

        Args:
            hash_a: One side's content hash.
            hash_b: The other side's content hash.

        Returns:
            The pair's latest :class:`StoredVerdict`, or ``None`` if unjudged.
        """
        return self._verdicts.get(tuple(sorted((hash_a, hash_b))))

    def __contains__(self, pair: "Tuple[str, str]") -> bool:
        """Orientation-blind membership for a ``(hash_a, hash_b)`` pair."""
        return tuple(sorted(pair)) in self._verdicts

    def __len__(self) -> int:
        """The count of distinct prior pairs at these pins (2c's reused-set size)."""
        return len(self._verdicts)


# The read-side projected columns — a narrow sibling of ``_CONFLICT_CHECKS_COLUMNS``
# (the 20-col write order). ``prompt_version``/``model_alias`` are pinned in the
# WHERE (per-row exact-string match), not projected; ``candidate_slug`` is NOT
# carried (MI-2: a mutable historical citation — the standing-finding report
# resolves display slugs live from nodes by hash, M8); ``surfaced`` is NOT carried
# (finding-ness is re-derived in the engine from raw ``tenable`` + ``confidence``,
# Key Decision 4).
_REUSE_INDEX_COLUMNS: Tuple[str, ...] = (
    "proposed_hash_if_any",
    "candidate_hash",
    "tenable",
    "confidence",
    "rationale",
    "batch_id",
    "created_at",
)

# One bulk read per run — never a per-pair SQL probe over the un-indexed append-only
# table (§6.2 mandate / Key Decision 1). Ascending ``created_at`` then ``rowid``
# makes last-write-wins deterministic on a shared-timestamp batch (``rowid`` is a safe
# monotonic-with-insertion proxy because the corpus is append-only and never deleted).
# ``proposed_hash_if_any IS NOT NULL`` drops rows that can form no pair key. Values
# bind via ``?`` (P8); the only interpolation is over code-internal column literals.
_SELECT_REUSE_INDEX_SQL = (
    f"SELECT {', '.join(_REUSE_INDEX_COLUMNS)} FROM conflict_checks "
    "WHERE prompt_version = ? AND model_alias = ? "
    "AND proposed_hash_if_any IS NOT NULL "
    "ORDER BY created_at ASC, rowid ASC"
)


_COMMENTARY_AUDIT_COLUMNS: Tuple[str, ...] = (
    "audit_id", "node_id", "slug", "fields_changed", "prior_values", "new_values",
    "outcome", "correlates_to", "mitos_version", "created_at",
)

_INSERT_COMMENTARY_AUDIT_SQL = (
    f"INSERT INTO commentary_audit ({', '.join(_COMMENTARY_AUDIT_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COMMENTARY_AUDIT_COLUMNS))})"
)

_SELECT_COMMENTARY_AUDIT_SQL = (
    f"SELECT {', '.join(_COMMENTARY_AUDIT_COLUMNS)} FROM commentary_audit "
    "ORDER BY created_at ASC, rowid ASC"
)


class TelemetryStore:
    """Append-only sibling store for the Conflict sensor's judged batches.

    Boots its own migration ladder on construction (mirroring
    ``GraphStore.__init__``'s boot-on-init shape) and is otherwise
    connection-stateless: every operation opens its own connection through the MI-8
    chokepoint and closes it, holding no long-lived handle. It pairs two append-only
    writers (``record_judged_batch`` — the per-batch grain; ``record_check_run`` —
    the per-run summary grain) with one read-only bulk projection
    (``load_reuse_index``, the sibling's first consumer-facing read surface) — the
    reader opens ``mode=ro``, so "no writes on the read path" is structural, not
    disciplinary.
    """

    def __init__(self, telemetry_path: str) -> None:
        """Creates/opens the sibling telemetry DB and boots its ladder.

        Args:
            telemetry_path: Filesystem path to ``telemetry.sqlite`` (typically
                ``config.telemetry_path``). Its parent directory is created if
                absent — SQLite will not create a missing directory and would raise
                "unable to open database file" (mirrors ``GraphStore.__init__``).
        """
        self.telemetry_path = telemetry_path
        # Scaffold the parent dir before any connection touches the file — SQLite
        # does not create a missing ``.mitos/`` directory (store.py:538 idiom).
        db_dir = os.path.dirname(telemetry_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Bare boot: open the MI-8 chokepoint, run telemetry's OWN ladder, close.
        # No prototype guard, no pre-ladder snapshot — a single append-only store
        # with no risky rebuild needs none of the graph's ``_boot_migrations``
        # machinery (D2). ``run_migrations`` sets this connection to autocommit; it
        # is closed here, so the write path opens a fresh deferred-isolation
        # connection of its own (see ``record_judged_batch``).
        conn = open_connection(telemetry_path)
        try:
            run_migrations(conn, TELEMETRY_MIGRATION_STEPS)
        finally:
            conn.close()

    def record_judged_batch(
        self,
        batch: JudgmentBatch,
        rows: List[ConflictCheckRow],
        created_at: str,
    ) -> None:
        """Persists one judged batch — its metrics row + N candidate rows — atomically.

        Opens its own fresh connection (deferred isolation, so ``with conn:`` frames
        a real transaction — NOT the autocommit boot connection) and issues the one
        ``judgment_batches`` INSERT and the N ``conflict_checks`` INSERTs under a
        single ``with conn:`` block. All-or-nothing: any per-row failure rolls the
        whole batch back — no orphan candidate rows without their batch, no batch
        row without its candidates. Issues only INSERTs (append-only whole-row);
        never UPDATE/DELETE.

        ``created_at`` is the caller-supplied UTC ISO-8601 stamp shared by every
        candidate row of the batch (MI-10, D5) — the writer reads no wall clock.

        Args:
            batch: The per-batch additive metrics (one ``judgment_batches`` row).
            rows: The judged candidate pairs (N ``conflict_checks`` rows). All must
                share ``batch.batch_id`` — the writer stores what it is handed; the
                single transaction is what guarantees the join key always has its
                mate.
            created_at: The batch-wide UTC ISO-8601 timestamp stamped on every row.

        Raises:
            DatabaseError: If the batch cannot be persisted (a constraint violation,
                NOT NULL breach, or any other SQLite error) — the whole batch rolls
                back before this propagates under the CLI's ``MitosError`` boundary.
        """
        conn = open_connection(self.telemetry_path)
        try:
            with conn:
                conn.execute(_INSERT_JUDGMENT_BATCH_SQL, batch.to_params())
                for row in rows:
                    conn.execute(
                        _INSERT_CONFLICT_CHECK_SQL, row.to_params(created_at)
                    )
        except sqlite3.Error as e:
            # ``with conn:`` has already rolled the transaction back; wrap the raw
            # SQLite error in the located ``DatabaseError`` vector so it renders as a
            # one-line ``Error: …`` under the CLI's ``except MitosError`` boundary
            # (§13; mint no new error class).
            raise DatabaseError(f"Failed to persist judged batch: {e}") from e
        finally:
            conn.close()

    @contextlib.contextmanager
    def _audit_connection(self):
        """Yields a write connection with ``synchronous=FULL`` for the audit row.

        Both databases run WAL with ``synchronous=NORMAL``, so their WAL syncs are
        INDEPENDENT: on power loss the graph commit can survive while the earlier
        telemetry write is lost — the one thing write-ahead ordering exists to prevent.
        `FULL` closes that window. One line, and reconciles are rare, so the fsync cost
        is paid nowhere that matters. Residual, named rather than left to be discovered:
        this narrows the window, it does not abolish it — a filesystem that lies about
        fsync lies to both databases.

        Yields:
            An open connection with the pragma applied.
        """
        conn = open_connection(self.telemetry_path)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
        finally:
            conn.close()

    def record_commentary_intent(
        self, row: CommentaryAuditRow, created_at: str
    ) -> str:
        """Writes the attribution row for a reconcile that is ABOUT to be applied.

        Write-**ahead**, deliberately, and the failure modes are asymmetric. A
        best-effort write-AFTER (the ``note_source_reencounter`` pattern) can leave a
        **mutation with no attribution** on a crash between the two — undetectable, and
        exactly what P8 forbids. Write-ahead instead leaves a **phantom row for a change
        that never landed**, and a phantom never violates P8: it forbids unattributed
        *mutations*, not attributed *non-mutations*. Losing cross-database atomicity is
        the price, and it is the cheaper failure.

        Unlike every other writer in this module this one is **mandatory**, not
        best-effort: the caller must let a failure here refuse the reconcile rather than
        proceed unaudited, or P8 holds only when the disk cooperates. That inversion of
        this module's ambient posture is the sharpest way the design fails in practice
        if left unstated.

        Args:
            row: The attribution row.
            created_at: The application-supplied UTC ISO-8601 stamp (MI-10).

        Returns:
            The written ``audit_id``, so the caller can correlate an outcome row to it.

        Raises:
            DatabaseError: If the row cannot be persisted. The caller MUST NOT swallow
                this — refusing the reconcile is the contract.
        """
        try:
            with self._audit_connection() as conn:
                with conn:
                    conn.execute(_INSERT_COMMENTARY_AUDIT_SQL, row.to_params(created_at))
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Could not write the commentary attribution row for '{row.slug}': {exc}"
            ) from exc
        return row.audit_id

    def record_commentary_outcome(
        self, *, audit_id: str, correlates_to: str, outcome: str, created_at: str,
        mitos_version: str,
    ) -> None:
        """Appends the row that CLOSES a failed reconcile's intent row.

        Append-only closure: telemetry never UPDATEs or DELETEs, so a reconcile that
        raised (``CommitError`` — MI-13 FM2's slug-collision rollback is reachable on
        this path) is recorded as a correlated outcome rather than by deleting the
        intent. The read rule follows from that: an intent row with no correlated
        failure row, over a graph matching its ``new_values``, means applied.

        A failure to write THIS row is best-effort by contrast — the mutation did not
        happen, so there is no unattributed mutation to hide.

        Args:
            audit_id: This outcome row's own id.
            correlates_to: The intent row's ``audit_id``.
            outcome: Why the reconcile did not land.
            created_at: The application-supplied UTC ISO-8601 stamp (MI-10).
            mitos_version: The writing build. Passed in rather than imported, so this
                module reads no ambient state — the same reason it takes ``created_at``
                instead of reading a clock.

        Raises:
            DatabaseError: If the row cannot be persisted.
        """
        params = (audit_id, None, None, None, None, None, outcome, correlates_to,
                  mitos_version, created_at)
        try:
            with self._audit_connection() as conn:
                with conn:
                    conn.execute(_INSERT_COMMENTARY_AUDIT_SQL, params)
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Could not write the commentary outcome row for '{correlates_to}': {exc}"
            ) from exc

    def read_commentary_audit(self) -> List[Dict[str, Any]]:
        """Reads the whole attribution trail, oldest first, JSON columns decoded.

        Opens ``mode=ro`` so "no writes on the read path" is structural rather than
        disciplinary, matching ``load_reuse_index``.

        Returns:
            One dict per row, ``fields_changed``/``prior_values``/``new_values``
            decoded from JSON (``None`` when the column is NULL, as on an outcome row).
        """
        conn = open_connection(self.telemetry_path, read_only=True)
        try:
            rows = conn.execute(_SELECT_COMMENTARY_AUDIT_SQL).fetchall()
        finally:
            conn.close()

        out: List[Dict[str, Any]] = []
        for raw in rows:
            record = {name: raw[name] for name in _COMMENTARY_AUDIT_COLUMNS}
            for json_column in ("fields_changed", "prior_values", "new_values"):
                if record[json_column] is not None:
                    record[json_column] = json.loads(record[json_column])
            out.append(record)
        return out

    def record_check_run(self, row: CheckRunRow) -> None:
        """Persists one check run's summary row — the CHK-D7 run memory (W7).

        Mirrors :meth:`record_judged_batch`'s manners exactly: its own fresh
        deferred-isolation connection, one INSERT under ``with conn:``,
        append-only (never UPDATE/DELETE), no wall-clock read — the row carries
        the caller's MI-10 stamps. ``run_id`` is the table PK, so a second write
        for the same run raises (one summary row per run is structural).

        This writer is a dumb sink (the 1b sibling-store bulkhead): the scalars'
        semantics — NULL rules, degradation tokens, novelty counts — are the
        check engine's, computed in ``check.check_run_row_from_result`` from the
        same result object the run report reads. The seam order is the CLI's
        (3a): compute the exit code → build the row → this write, LAST, so a
        failure here can only move the exit toward 2 and a persisted row's
        ``exit_code`` always equals the actual process exit.

        Args:
            row: The assembled summary row, verbatim.

        Raises:
            DatabaseError: If the row cannot be persisted (duplicate ``run_id``,
                a CHECK/NOT NULL breach, or any other SQLite error) — never
                swallowed; the caller owns the disclose-and-exit disposition.
        """
        conn = open_connection(self.telemetry_path)
        try:
            with conn:
                conn.execute(_INSERT_CHECK_RUN_SQL, row.to_params())
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to persist check run: {e}") from e
        finally:
            conn.close()

    def load_reuse_index(
        self, *, prompt_version: str, model_alias: str
    ) -> "ReuseIndex | ReuseUnavailable":
        """Projects prior verdicts into an orientation-blind latest-wins index.

        One read-only bulk projection of ``conflict_checks`` — routed through the
        MI-8 chokepoint with ``read_only=True`` so "no writes on the read path" is a
        structural property of the connection mode, not a discipline (Key Decision 2)
        — streamed into a dict keyed on the sorted content-hash pair. The scan is
        ascending by ``created_at`` then ``rowid`` and each key is overwritten, so the
        *latest* matching row per pair wins and only the winners are retained: peak
        memory is O(distinct pairs at these pins), never O(total history). Never a
        per-pair SQL probe over the un-indexed append-only table (§6.2 mandate).

        The index carries only rows whose ``prompt_version`` **and** ``model_alias``
        match the passed pins (WHERE-filtered) — a prompt or tier change invalidates
        reuse by construction (CHK-D3). Rows with a NULL ``proposed_hash_if_any`` form
        no pair key and are filtered in SQL. Serves BOTH the reuse and novelty
        partitions from the same entries; it is ``--fresh``-agnostic (2c decides
        whether to *use* a reused verdict, but always consults the index for novelty).

        A read failure (corrupt image, unopenable path, lock timeout) returns a typed
        :class:`ReuseUnavailable` — never raised, never a silent empty. An empty
        :class:`ReuseIndex` (no priors) is a *different type* from ``ReuseUnavailable``
        (a broken read); the engine's exit-0-vs-exit-2 fork keys on exactly that line
        (Key Decision 5).

        Args:
            prompt_version: The judgment prompt version to pin reuse to (exact match).
            model_alias: The family+tier model alias to pin reuse to (exact match).

        Returns:
            A :class:`ReuseIndex` of latest-wins verdicts (possibly empty), or a
            :class:`ReuseUnavailable` if the read failed. Never raises on a bad DB.
        """
        conn = None
        try:
            conn = open_connection(self.telemetry_path, read_only=True)
            verdicts: Dict[Tuple[str, str], StoredVerdict] = {}
            # Stream the cursor (never ``fetchall``); ascending order + overwrite
            # means the last row seen per pair is the newest, so only winners survive.
            for row in conn.execute(
                _SELECT_REUSE_INDEX_SQL, (prompt_version, model_alias)
            ):
                key = tuple(sorted((row["proposed_hash_if_any"], row["candidate_hash"])))
                verdicts[key] = StoredVerdict(
                    tenable=bool(row["tenable"]),
                    confidence=row["confidence"],
                    rationale=row["rationale"],
                    batch_id=row["batch_id"],
                    created_at=row["created_at"],
                )
            return ReuseIndex(verdicts)
        except (sqlite3.Error, DatabaseError) as e:
            # Both surface as a read failure: a corrupt image raises
            # ``sqlite3.DatabaseError`` at query time, while an unopenable path is
            # wrapped by ``open_connection`` into Mitos's ``DatabaseError`` *before*
            # any query runs (store.py:474) — catch BOTH or the second escapes.
            return ReuseUnavailable(str(e))
        finally:
            if conn is not None:
                conn.close()
