"""Behavioral tests for the sibling telemetry store (Phase 1b).

Proves ``mitos/telemetry.py`` end-to-end against a **real temp SQLite** (no mocks —
this is a typed observability surface, so round-trips against a real file are the
point): the ladder boots green from scratch and replays to a no-op, a long/multi-
line judged axiom round-trips verbatim and uncapped, NULL semantics bind exactly,
the ``judgment_batches`` side-table attributes cost/latency exactly once (RF-2), a
partial write rolls the whole batch back, the writer is append-only, and the
sibling store survives a real ``mitos rebuild`` swap (T8).

Dynamic values only: ``PRAGMA user_version`` expectations come from
``_pending_head(TELEMETRY_MIGRATION_STEPS)``, never a hardcoded rung literal (the
sliced-ladder ``== 1`` in the upgrade fixture is the one deliberate exception —
it pins the *pre-upgrade* rung, not the head); metric fields are fixture inputs
echoed back and asserted for round-trip equality, not a computed count.
"""

import dataclasses
import inspect
import os
import shutil
import sqlite3

import pytest

import mitos.telemetry as telemetry_module
from mitos.config import MitosConfig
from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate
from mitos.errors import DatabaseError
from mitos.migrations import _pending_head, run_migrations
from mitos.store import BUSY_TIMEOUT_MS, GraphStore, open_connection
from mitos.telemetry import (
    _CHECK_RUNS_COLUMNS,
    _CONFLICT_CHECKS_COLUMNS,
    _JUDGMENT_BATCHES_COLUMNS,
    TELEMETRY_MIGRATION_STEPS,
    CheckRunRow,
    ConflictCheckRow,
    JudgmentBatch,
    ReuseIndex,
    ReuseUnavailable,
    StoredVerdict,
    TelemetryStore,
)

CREATED_AT = "2026-07-03T01:46:29.359834+00:00"
SENTINEL = "<!-- BEGIN ENTRIES — newest first -->"


# --- builders + read helpers ---------------------------------------------------


def _row(**overrides) -> ConflictCheckRow:
    """Builds a ConflictCheckRow with sensible defaults; override any field."""
    base = dict(
        batch_id="batch-1",
        sync_run_id="sync-1",
        surface="sync",
        judged_axiom="Use SQLite for the graph store.",
        proposal_rejected_paths="Considered Postgres; rejected for portability.",
        proposal_scope="storage, persistence",
        proposed_hash_if_any="deadbeef",
        candidate_slug="graph-store-is-sqlite",
        candidate_hash="cafef00d",
        candidate_rejected_paths="Rejected a document store.",
        candidate_scope="storage",
        tenable=True,
        confidence=0.91,
        surfaced=True,
        candidate_source="embedding_topk",
        model_alias="CLAUDE_SONNET",
        prompt_version="conflict-judge-v1",
        mitos_version="0.5.21",
        rationale="Both axioms fix the storage engine and disagree.",
    )
    base.update(overrides)
    return ConflictCheckRow(**base)


def _batch(**overrides) -> JudgmentBatch:
    """Builds a JudgmentBatch with sensible defaults; override any field."""
    base = dict(
        batch_id="batch-1",
        model_id="test-model-versioned-1",
        token_input=1200,
        token_output=340,
        token_cache_read=0,
        token_cache_creation=800,
        elapsed_ms=2500,
    )
    base.update(overrides)
    return JudgmentBatch(**base)


def _check_run_row(**overrides) -> CheckRunRow:
    """Builds a CheckRunRow with sensible defaults; override any field."""
    base = dict(
        run_id="run-2d",
        mode="corpus",
        started_at=CREATED_AT,
        ended_at="2026-07-03T01:47:02.000001+00:00",
        exit_code=1,
        nodes_swept=5,
        pairs_judged_fresh=2,
        pairs_reused=3,
        findings_new=1,
        findings_known=0,
        coverage_exclusions=0,
        degraded_reason=None,
        mitos_version="0.6.0-test",
    )
    base.update(overrides)
    return CheckRunRow(**base)


def _telemetry_path(tmp_path) -> str:
    """A telemetry path whose parent ``.mitos/`` does NOT yet exist.

    Constructing the store here also exercises the makedirs-before-open guard —
    SQLite raises "unable to open database file" on a missing directory.
    """
    return str(tmp_path / ".mitos" / "telemetry.sqlite")


def _open(path: str) -> sqlite3.Connection:
    """Opens a read-back connection through the MI-8 chokepoint (row_factory=Row)."""
    return open_connection(path)


def _user_version(conn: sqlite3.Connection) -> int:
    """Reads ``PRAGMA user_version`` back through the DB (never a bare literal)."""
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Reports whether a table is present in the schema."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,)
        ).fetchone()
        is not None
    )


def _schema_snapshot(conn: sqlite3.Connection):
    """Returns the full stored DDL as comparable tuples (byte-identity check)."""
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name;"
        ).fetchall()
    ]


def _count(conn: sqlite3.Connection, table: str, batch_id: str = None) -> int:
    """Counts rows in ``table``, optionally scoped to a ``batch_id``."""
    if batch_id is None:
        return conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE batch_id=?;", (batch_id,)
    ).fetchone()[0]


# --- lockstep: to_params order matches the INSERT column tuples ----------------


def test_to_params_arity_matches_column_tuples() -> None:
    """``to_params`` emits exactly one value per INSERT column (order-lockstep pin)."""
    assert len(_row().to_params(CREATED_AT)) == len(_CONFLICT_CHECKS_COLUMNS)
    assert len(_batch().to_params()) == len(_JUDGMENT_BATCHES_COLUMNS)
    assert len(_check_run_row().to_params()) == len(_CHECK_RUNS_COLUMNS) == 13
    # Rung 2 widened both tuples — the INSERTs self-assemble from these.
    assert "surface" in _CONFLICT_CHECKS_COLUMNS
    assert "model_id" in _JUDGMENT_BATCHES_COLUMNS


def test_rung2_fields_have_no_defaults() -> None:
    """``surface``/``model_id`` are required at construction — the writers' fence.

    The schema's permanent ``DEFAULT 'sync'`` can never catch a writer omitting the
    column (an omitting INSERT silently gets ``'sync'``), so the enforcement is the
    defaultless dataclass field: omission is a ``TypeError`` at construction
    (CHK-D7). ``model_id=None`` stays a legal *explicit* value; silence is not.
    """
    row_fields = {f.name: f for f in dataclasses.fields(ConflictCheckRow)}
    surface = row_fields["surface"]
    assert surface.default is dataclasses.MISSING
    assert surface.default_factory is dataclasses.MISSING

    batch_fields = {f.name: f for f in dataclasses.fields(JudgmentBatch)}
    model_id = batch_fields["model_id"]
    assert model_id.default is dataclasses.MISSING
    assert model_id.default_factory is dataclasses.MISSING


# --- criterion 1: fresh boot ---------------------------------------------------


def test_fresh_boot_creates_schema(tmp_path) -> None:
    """Constructing on an absent path creates the file, all three tables, head version.

    Also proves the makedirs-before-open guard: the parent ``.mitos/`` does not
    exist before construction. The head expectation is dynamic
    (``_pending_head``) — never a hardcoded rung literal.
    """
    path = _telemetry_path(tmp_path)
    assert not os.path.exists(path)

    TelemetryStore(path)

    assert os.path.exists(path)
    conn = _open(path)
    try:
        assert _table_exists(conn, "conflict_checks")
        assert _table_exists(conn, "judgment_batches")
        assert _table_exists(conn, "check_runs")
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
    finally:
        conn.close()


def _table_info(conn: sqlite3.Connection, table: str):
    """Returns ``PRAGMA table_info`` as (name, type, notnull, pk) tuples."""
    return [
        (r["name"], r["type"], r["notnull"], r["pk"])
        for r in conn.execute(f"PRAGMA table_info({table});").fetchall()
    ]


def test_check_runs_column_contract(tmp_path) -> None:
    """``check_runs`` matches the §3/W1 contract exactly — 2d's writer builds on this.

    Nullability is the semantic line and is pinned NOW (SQLite constraints are
    rebuild-only to change): NOT NULL = always known at run end; NULL = "value not
    computable this run", distinct from a genuine zero (CHK-D10).
    """
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        got = _table_info(conn, "check_runs")
    finally:
        conn.close()
    assert got == [
        ("run_id", "TEXT", 1, 1),
        ("mode", "TEXT", 1, 0),
        ("started_at", "TEXT", 1, 0),
        ("ended_at", "TEXT", 1, 0),
        ("exit_code", "INTEGER", 1, 0),
        ("nodes_swept", "INTEGER", 1, 0),
        ("pairs_judged_fresh", "INTEGER", 1, 0),
        ("pairs_reused", "INTEGER", 1, 0),
        ("findings_new", "INTEGER", 0, 0),
        ("findings_known", "INTEGER", 0, 0),
        ("coverage_exclusions", "INTEGER", 0, 0),
        ("degraded_reason", "TEXT", 0, 0),
        ("mitos_version", "TEXT", 1, 0),
    ]


def test_widened_columns_shapes(tmp_path) -> None:
    """``conflict_checks.surface`` is NOT NULL DEFAULT 'sync'; ``model_id`` is nullable TEXT."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        checks = {
            r["name"]: (r["type"], r["notnull"], r["dflt_value"])
            for r in conn.execute("PRAGMA table_info(conflict_checks);").fetchall()
        }
        batches = {
            r["name"]: (r["type"], r["notnull"], r["dflt_value"])
            for r in conn.execute("PRAGMA table_info(judgment_batches);").fetchall()
        }
    finally:
        conn.close()
    # The DEFAULT is the rung-2 backfill mechanism (and permanent — SQLite requires
    # it on ADD COLUMN NOT NULL); the writers' lockstep is the real fence.
    assert checks["surface"] == ("TEXT", 1, "'sync'")
    assert batches["model_id"] == ("TEXT", 0, None)


def test_check_runs_checks_reject_out_of_contract_values(tmp_path) -> None:
    """The belt-and-suspenders CHECKs pin the closed sets: mode, exit_code, one-row-per-run."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    insert = (
        "INSERT INTO check_runs (run_id, mode, started_at, ended_at, exit_code, "
        "nodes_swept, pairs_judged_fresh, pairs_reused, findings_new, findings_known, "
        "coverage_exclusions, degraded_reason, mitos_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    good = ("run-1", "corpus", CREATED_AT, CREATED_AT, 0, 5, 2, 3, 1, 0, 0, None, "test-v")
    conn = _open(path)
    try:
        with conn:
            conn.execute(insert, good)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("run-2", "watch") + good[2:])  # mode outside the set
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("run-3", "staged", CREATED_AT, CREATED_AT, 3) + good[5:])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, good)  # duplicate run_id — one row per run is structural
    finally:
        conn.close()


# --- the check_runs summary writer (Phase 2d, W7) --------------------------------


def test_record_check_run_roundtrips_field_for_field(tmp_path) -> None:
    """One happy corpus row lands and reads back exactly — including the three
    NULL-vs-zero distinctions (0 = a partitioned/probed fact; NULL = "could not
    tell", stamped here on none of them)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    row = _check_run_row(
        findings_new=1, findings_known=2, coverage_exclusions=0, degraded_reason=None
    )

    store.record_check_run(row)

    conn = _open(path)
    try:
        got = conn.execute("SELECT * FROM check_runs;").fetchall()
    finally:
        conn.close()
    assert len(got) == 1
    persisted = {key: got[0][key] for key in got[0].keys()}
    assert persisted == {
        "run_id": row.run_id,
        "mode": "corpus",
        "started_at": row.started_at,
        "ended_at": row.ended_at,
        "exit_code": 1,
        "nodes_swept": 5,
        "pairs_judged_fresh": 2,
        "pairs_reused": 3,
        "findings_new": 1,
        "findings_known": 2,
        "coverage_exclusions": 0,  # a genuine zero, not NULL
        "degraded_reason": None,  # healthy
        "mitos_version": row.mitos_version,
    }


def test_record_check_run_null_semantics_bind_exactly(tmp_path) -> None:
    """A degraded row's NULLs land as SQL NULL, never coerced to 0 — the CHK-D10
    line (partition unavailable / probe never completed) survives the boundary."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    store.record_check_run(
        _check_run_row(
            findings_new=None,
            findings_known=None,
            coverage_exclusions=None,
            degraded_reason="reuse_read,probe_read",
            exit_code=2,
        )
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT findings_new, findings_known, coverage_exclusions, "
            "degraded_reason FROM check_runs;"
        ).fetchone()
    finally:
        conn.close()
    assert tuple(got) == (None, None, None, "reuse_read,probe_read")


def test_record_check_run_accepts_a_staged_shaped_row(tmp_path) -> None:
    """T12's stamps-mode/zeros half at the writer grain: ``mode='staged'`` with
    ``pairs_reused=0`` (staged never reuses) lands — 3b owns the semantics e2e."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_check_run(_check_run_row(run_id="run-staged", mode="staged", pairs_reused=0))

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT mode, pairs_reused FROM check_runs WHERE run_id='run-staged';"
        ).fetchone()
    finally:
        conn.close()
    assert tuple(got) == ("staged", 0)


def test_record_check_run_duplicate_run_id_raises(tmp_path) -> None:
    """One row per run is STRUCTURAL (``run_id`` PK): a second write for the same
    run raises ``DatabaseError`` — the seam calls the writer once, at the end."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    store.record_check_run(_check_run_row())

    with pytest.raises(DatabaseError):
        store.record_check_run(_check_run_row())

    conn = _open(path)
    try:
        assert _count(conn, "check_runs") == 1
    finally:
        conn.close()


def test_record_check_run_out_of_contract_mode_raises_database_error(tmp_path) -> None:
    """The row dataclass does not police the closed sets (that is the check.py
    builder's ``ValueError``); a row carrying an out-of-contract ``mode`` dies on
    the schema CHECK at write time, wrapped as ``DatabaseError`` — and no row lands."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    with pytest.raises(DatabaseError):
        store.record_check_run(_check_run_row(mode="watch"))

    conn = _open(path)
    try:
        assert _count(conn, "check_runs") == 0
    finally:
        conn.close()


def test_record_check_run_unwritable_store_raises_database_error(tmp_path) -> None:
    """§9-10: a telemetry path that cannot be opened for writing raises a loud
    ``DatabaseError`` — the writer never swallows (the caller's disclose-and-exit-2
    seam is 3a's to test; 2d pins that the failure surfaces)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    shutil.rmtree(os.path.dirname(path))  # the .mitos/ dir vanishes out from under it

    with pytest.raises(DatabaseError):
        store.record_check_run(_check_run_row())


def test_check_run_row_fields_have_no_defaults() -> None:
    """Every ``CheckRunRow`` field is required at construction (the 1a fence idiom):
    a NULL is a legal *explicit* value, silence is not — an omitting caller is a
    ``TypeError`` at construction, never a silently-defaulted column."""
    for field in dataclasses.fields(CheckRunRow):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name


# --- criterion 2: replay idempotency x2 ----------------------------------------


def test_replay_idempotency_is_noop(tmp_path) -> None:
    """Booting the ladder a second and third time changes nothing and raises nothing."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        first_schema = _schema_snapshot(conn)
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
    finally:
        conn.close()

    # Re-boot twice over an at-head file — the user_version gate skips the applied
    # rungs (rung 2's bare ALTERs are NOT re-runnable; the guard is what makes the
    # replay a no-op, MI-3).
    TelemetryStore(path)
    TelemetryStore(path)

    conn = _open(path)
    try:
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
        assert _schema_snapshot(conn) == first_schema
    finally:
        conn.close()


# --- rung-2 upgrade: 'sync' backfill + NULL model_id on legacy rows -------------


def test_upgrade_backfills_surface_and_null_model_id(tmp_path) -> None:
    """A rung-1 DB with legacy rows boots to head: ``surface='sync'``, ``model_id`` NULL.

    The post-rung-2 writer cannot author a pre-rung-2 row (its INSERT names
    ``surface``), so the legacy fixture is built by running the SLICED ladder and
    INSERTing over the rung-1 column set via raw parameterized SQL. Every other
    field must read back byte-unchanged (``ADD COLUMN … DEFAULT`` is a schema
    operation — no row is UPDATEd; the append-only corpus is untouched).
    """
    path = _telemetry_path(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = open_connection(path)
    try:
        run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:1])
        conn.execute(
            "INSERT INTO judgment_batches (batch_id, token_input, token_output, "
            "token_cache_read, token_cache_creation, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            ("legacy-batch", 1200, 340, 0, 800, 2500),
        )
        conn.execute(
            "INSERT INTO conflict_checks (batch_id, sync_run_id, judged_axiom, "
            "proposal_rejected_paths, proposal_scope, proposed_hash_if_any, "
            "candidate_slug, candidate_hash, candidate_rejected_paths, "
            "candidate_scope, tenable, confidence, surfaced, candidate_source, "
            "model_alias, prompt_version, mitos_version, rationale, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (
                "legacy-batch", "sync-legacy", "Legacy judged axiom.", None, None,
                "hash-legacy", "legacy-cand", "cafe1234", "Rejected X.", None,
                0, 0.9, 1, "embedding_topk", "SONNET", "conflict-judge-v1",
                "0.5.21", "Legacy rationale.", CREATED_AT,
            ),
        )
        row_before = dict(conn.execute("SELECT * FROM conflict_checks;").fetchone())
        batch_before = dict(conn.execute("SELECT * FROM judgment_batches;").fetchone())
        assert _user_version(conn) == 1  # the deliberate pre-upgrade rung
    finally:
        conn.close()

    TelemetryStore(path)  # advance the ladder 1 -> head

    conn = _open(path)
    try:
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
        row_after = dict(conn.execute("SELECT * FROM conflict_checks;").fetchone())
        batch_after = dict(conn.execute("SELECT * FROM judgment_batches;").fetchone())
        upgraded_schema = _schema_snapshot(conn)
    finally:
        conn.close()
    assert row_after.pop("surface") == "sync"  # the DEFAULT backfill
    assert row_after == row_before  # every pre-existing field byte-unchanged
    assert batch_after.pop("model_id") is None  # pre-rung-2 provenance is unknowable
    assert batch_after.pop("stop_reason") is None  # pre-rung-4 rows have no stop_reason
    assert batch_after == batch_before

    # Stretch pin: fresh-install and upgraded-install execute the identical DDL
    # sequence (rung 1 then rung 2), so their stored schemas are byte-identical.
    fresh_path = str(tmp_path / ".mitos" / "fresh-telemetry.sqlite")
    TelemetryStore(fresh_path)
    conn = _open(fresh_path)
    try:
        assert _schema_snapshot(conn) == upgraded_schema
    finally:
        conn.close()


# --- MI-8: every connection routes through the store.open_connection chokepoint --


def test_connections_route_through_mi8_chokepoint(tmp_path, monkeypatch) -> None:
    """Boot + write open exactly two connections, both fully PRAGMA-configured.

    Asserts the chokepoint discipline behaviourally: a spy wraps
    ``mitos.telemetry.open_connection`` (the import-time binding — patching
    ``mitos.store`` would not intercept) and records each connection's live
    ``foreign_keys``/``busy_timeout``. A bare ``sqlite3.connect`` anywhere in the
    module would bypass the spy — the source-level pin closes that gap.
    """
    path = _telemetry_path(tmp_path)
    real_open = telemetry_module.open_connection
    pragmas = []

    def spy(db_path: str, read_only: bool = False) -> sqlite3.Connection:
        conn = real_open(db_path, read_only=read_only)
        pragmas.append(
            (
                conn.execute("PRAGMA foreign_keys;").fetchone()[0],
                conn.execute("PRAGMA busy_timeout;").fetchone()[0],
            )
        )
        return conn

    monkeypatch.setattr("mitos.telemetry.open_connection", spy)
    store = TelemetryStore(path)
    store.record_judged_batch(_batch(), [_row()], CREATED_AT)

    assert len(pragmas) == 2  # one boot connection + one per-write connection
    assert all(fk == 1 for fk, _ in pragmas)
    assert all(busy == BUSY_TIMEOUT_MS for _, busy in pragmas)
    # Supplementary source pin: telemetry never opens a bare sqlite3.connect.
    assert "sqlite3.connect" not in inspect.getsource(telemetry_module)


# --- criterion 3: verbatim uncapped round-trip ---------------------------------


def test_verbatim_uncapped_roundtrip(tmp_path) -> None:
    """A long, multi-line axiom/rationale/rejected_paths reads back byte-identical."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    long_axiom = (
        "Line one of the judged axiom.\n" * 500
        + "  trailing détail — ß, Straße, and a Greek final sigma ς.\n"
    )
    multiline_rationale = "First reason.\n\nSecond reason.\n- bullet a\n- bullet b\n"
    multiline_rejected = "Rejected path A.\nRejected path B.\n\tindented C\n"

    store.record_judged_batch(
        # 'check' (not the builder's 'sync' default) proves the writer's VALUE binds
        # — a 'sync' read-back could be the schema DEFAULT masking an omitted column.
        _batch(model_id="claude-test-9"),
        [
            _row(
                judged_axiom=long_axiom,
                rationale=multiline_rationale,
                candidate_rejected_paths=multiline_rejected,
                surface="check",
            )
        ],
        CREATED_AT,
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT judged_axiom, rationale, candidate_rejected_paths, created_at, "
            "surface FROM conflict_checks;"
        ).fetchone()
        got_batch = conn.execute(
            "SELECT model_id FROM judgment_batches;"
        ).fetchone()
    finally:
        conn.close()
    assert got["judged_axiom"] == long_axiom
    assert got["rationale"] == multiline_rationale
    assert got["candidate_rejected_paths"] == multiline_rejected
    assert got["created_at"] == CREATED_AT
    assert got["surface"] == "check"
    assert got_batch["model_id"] == "claude-test-9"


# --- criterion 4: NULL semantics ------------------------------------------------


def test_null_semantics_roundtrip(tmp_path) -> None:
    """Nullable fields bind None; present nullables bind their value, exactly."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(
        # model_id=None is a legal EXPLICIT value (resolution failed) → SQL NULL,
        # never the empty string.
        _batch(model_id=None),
        [
            _row(
                proposal_scope=None,
                candidate_scope=None,
                proposal_rejected_paths=None,
                sync_run_id="sync-9",
                proposed_hash_if_any="hash-9",
            )
        ],
        CREATED_AT,
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT proposal_scope, candidate_scope, proposal_rejected_paths, "
            "sync_run_id, proposed_hash_if_any FROM conflict_checks;"
        ).fetchone()
        got_batch = conn.execute(
            "SELECT model_id FROM judgment_batches;"
        ).fetchone()
    finally:
        conn.close()
    assert got["proposal_scope"] is None
    assert got["candidate_scope"] is None
    assert got["proposal_rejected_paths"] is None
    assert got["sync_run_id"] == "sync-9"
    assert got["proposed_hash_if_any"] == "hash-9"
    assert got_batch["model_id"] is None


def test_not_null_field_fails_loudly_and_rolls_back(tmp_path) -> None:
    """Omitting the M5-required ``candidate_rejected_paths`` raises + lands nothing."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    # candidate_rejected_paths is NOT NULL (M5 requires it on every decision).
    bad = _row(candidate_rejected_paths=None)
    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), [bad], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


def test_check_constraint_rejects_out_of_range_confidence(tmp_path) -> None:
    """The belt-and-suspenders CHECK rejects a confidence outside [0, 1]."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), [_row(confidence=2.0)], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


# --- criterion 5: batch attribution (RF-2) -------------------------------------


def test_batch_attribution_exactly_once(tmp_path) -> None:
    """N candidate rows share one batch; metrics attribute exactly once (no N x)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    n = 4
    batch = _batch(batch_id="b-rf2", token_input=1500)
    rows = [
        _row(batch_id="b-rf2", candidate_slug=f"cand-{i}", candidate_hash=f"h{i}")
        for i in range(n)
    ]
    store.record_judged_batch(batch, rows, CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks", "b-rf2") == n
        assert _count(conn, "judgment_batches", "b-rf2") == 1
        # A naive SUM over the side-table returns TRUE spend, not n x it.
        total = conn.execute(
            "SELECT SUM(token_input) FROM judgment_batches;"
        ).fetchone()[0]
        assert total == 1500
    finally:
        conn.close()


# --- criterion 6: batch atomicity ----------------------------------------------


def test_batch_atomicity_rolls_back_whole_batch(tmp_path) -> None:
    """A bad row late in a batch rolls back the batch row AND every earlier row."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    good = [
        _row(candidate_slug=f"ok-{i}", candidate_hash=f"h{i}") for i in range(3)
    ]
    bad = _row(candidate_slug="bad", candidate_hash="hbad", candidate_rejected_paths=None)

    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), good + [bad], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


# --- criterion 7: append-only ---------------------------------------------------


def test_append_only_second_batch_preserves_first(tmp_path) -> None:
    """A second batch adds its rows without touching the first batch's rows."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(_batch(batch_id="b1"), [_row(batch_id="b1")], CREATED_AT)
    store.record_judged_batch(
        _batch(batch_id="b2"),
        [_row(batch_id="b2", candidate_slug="other", candidate_hash="h-other")],
        "2026-07-04T00:00:00+00:00",
    )

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 2
        assert _count(conn, "judgment_batches") == 2
        # The first batch's row is byte-for-byte as written (untouched by batch 2).
        first = conn.execute(
            "SELECT created_at, candidate_slug FROM conflict_checks WHERE batch_id=?;",
            ("b1",),
        ).fetchone()
        assert first["created_at"] == CREATED_AT
        assert first["candidate_slug"] == "graph-store-is-sqlite"
    finally:
        conn.close()


# --- bool -> INTEGER storage (STRICT has no BOOLEAN affinity) -------------------


def test_bools_stored_as_integers(tmp_path) -> None:
    """``tenable``/``surfaced`` land as 0/1 INTEGERs under STRICT typing."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(
        _batch(), [_row(tenable=True, surfaced=False)], CREATED_AT
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT tenable, surfaced FROM conflict_checks;"
        ).fetchone()
    finally:
        conn.close()
    assert got["tenable"] == 1
    assert got["surfaced"] == 0
    assert isinstance(got["tenable"], int)
    assert isinstance(got["surfaced"], int)


# --- criterion 8: rebuild-survival (T8 unit) -----------------------------------


def _decision_block(slug: str, decided: str, rejected: str = "n/a") -> str:
    """One decision entry block in spec order (mirrors test_cutover's ``_decision``)."""
    return f"### {slug}\n\n**Decided:** {decided}\n**Rejected:** {rejected}"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_telemetry_survives_rebuild_swap(tmp_path) -> None:
    """A populated ``telemetry.sqlite`` + its rows survive a real ``mitos rebuild`` swap.

    The T8 guarantee: telemetry sits OUTSIDE the graph swap/backup/sidecar set, so
    ``rebuild_and_gate`` + ``perform_swap`` (which touch only ``graph.sqlite``-derived
    paths) leave the sibling store untouched.
    """
    config = MitosConfig(str(tmp_path))

    # A real (fresh) V1a graph at config.db_path so the completeness gate has an old
    # graph to read, plus a one-entry corpus for the rebuild to replay.
    GraphStore(config.db_path)
    _write(
        config.decisions_file,
        SENTINEL + "\n\n" + _decision_block("alpha", "Alpha axiom.") + "\n",
    )

    # Populate the telemetry sibling BEFORE the swap.
    telemetry = TelemetryStore(config.telemetry_path)
    telemetry.record_judged_batch(
        _batch(batch_id="survivor"),
        [_row(batch_id="survivor", judged_axiom="A judgment that must outlive rebuild.")],
        CREATED_AT,
    )

    # Drive the REAL swap.
    aside = default_aside_db_path(config)
    rebuild_and_gate(config, aside_db_path=aside)
    perform_swap(config, aside, timestamp="20260618-120000")

    # The sibling file and its row are untouched.
    assert os.path.exists(config.telemetry_path)
    conn = _open(config.telemetry_path)
    try:
        assert _count(conn, "conflict_checks", "survivor") == 1
        assert _count(conn, "judgment_batches", "survivor") == 1
        got = conn.execute(
            "SELECT judged_axiom FROM conflict_checks WHERE batch_id=?;", ("survivor",)
        ).fetchone()
        assert got["judged_axiom"] == "A judgment that must outlive rebuild."
    finally:
        conn.close()


# --- Phase 1b: verdict-reuse + novelty bulk-read path (load_reuse_index) --------
#
# The reuse key pins {hash_lo, hash_hi, prompt_version, model_alias}. The pins are
# self-consistent TEST values, NOT production literals: the `_row()` builder's
# defaults (`"conflict-judge-v1"` / `"CLAUDE_SONNET"`) differ from what production
# stamps, so a test that pinned the production values against a default-`_row()` seed
# would MISS by accident and mask a real bug. Each test seeds and queries the SAME
# pins; PIN_B_* is a deliberately-distinct second pin for the mismatch case.
PIN_PROMPT = "prompt-vA"
PIN_ALIAS = "ALIAS_A"
PIN_PROMPT_B = "prompt-vB"
PIN_ALIAS_B = "ALIAS_B"


def _seed_pair(
    store: TelemetryStore,
    *,
    proposed,
    candidate: str,
    batch_id: str,
    created_at: str = CREATED_AT,
    prompt_version: str = PIN_PROMPT,
    model_alias: str = PIN_ALIAS,
    tenable: bool = False,
    confidence: float = 0.9,
    rationale: str = "The two axioms stand in tension.",
    **row_overrides,
) -> None:
    """Seeds one judged pair as its OWN batch (distinct ``batch_id`` per call, §6).

    ``batch_id`` is the ``judgment_batches`` PK — a seed loop reusing one id collides
    and a row silently vanishes, so every seeded pair mints its own. ``created_at`` is
    ``record_judged_batch``'s 3rd positional arg (batch-wide), which the latest-wins
    tests control directly.
    """
    store.record_judged_batch(
        _batch(batch_id=batch_id),
        [
            _row(
                batch_id=batch_id,
                proposed_hash_if_any=proposed,
                candidate_hash=candidate,
                prompt_version=prompt_version,
                model_alias=model_alias,
                tenable=tenable,
                confidence=confidence,
                rationale=rationale,
                **row_overrides,
            )
        ],
        created_at,
    )


def test_lookup_is_orientation_blind(tmp_path) -> None:
    """A stored pair resolves the same whichever side is queried first (§8-1).

    Seeds one pair as ``(proposed=A, candidate=C)`` and a second in the *reverse*
    orientation ``(proposed=E, candidate=B)``; both resolve regardless of the
    argument order — the sorted-tuple key makes hits direction-free (Key Decision 3).
    """
    store = TelemetryStore(_telemetry_path(tmp_path))
    _seed_pair(store, proposed="A", candidate="C", batch_id="b1")
    _seed_pair(store, proposed="E", candidate="B", batch_id="b2")

    index = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert isinstance(index, ReuseIndex)

    forward = index.lookup("A", "C")
    reverse = index.lookup("C", "A")
    assert forward is not None
    assert isinstance(forward, StoredVerdict)
    assert forward == reverse  # same verdict, orientation-blind

    # The reverse-orientation seed hits from either query direction too.
    assert index.lookup("B", "E") is not None
    assert index.lookup("E", "B") is not None
    assert len(index) == 2


def test_judge_pin_mismatch_is_a_miss(tmp_path) -> None:
    """Only rows matching BOTH ``prompt_version`` and ``model_alias`` enter (§8-2).

    A prompt-version change and a model-alias change each invalidate reuse *by
    construction* (CHK-D3) — the safety mechanism the scheduled role eval ships
    behind.
    """
    store = TelemetryStore(_telemetry_path(tmp_path))
    # Pair whose only rows carry a different prompt_version.
    _seed_pair(
        store, proposed="A", candidate="C", batch_id="b1", prompt_version=PIN_PROMPT_B
    )
    # Pair whose only rows carry a different model_alias.
    _seed_pair(
        store, proposed="D", candidate="F", batch_id="b2", model_alias=PIN_ALIAS_B
    )
    # A pair matching both pins.
    _seed_pair(store, proposed="G", candidate="H", batch_id="b3")

    index = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert index.lookup("A", "C") is None  # prompt_version mismatch
    assert index.lookup("D", "F") is None  # model_alias mismatch
    assert index.lookup("G", "H") is not None  # both pins match
    assert len(index) == 1


def test_latest_row_wins_per_pair(tmp_path) -> None:
    """Conflicting verdicts for one pair collapse to the newest row (§8-3).

    Two separate ``record_judged_batch`` calls (distinct ``batch_id`` + ``created_at``)
    stamp *opposite* ``tenable`` on the same pair; the index carries the newer row's
    verdict, orientation-blind (the older row wrote the reverse orientation).
    """
    store = TelemetryStore(_telemetry_path(tmp_path))
    _seed_pair(
        store,
        proposed="A",
        candidate="C",
        batch_id="old",
        created_at="2026-07-03T00:00:00.000000+00:00",
        tenable=True,
        confidence=0.70,
        rationale="Older: no tension.",
    )
    _seed_pair(
        store,
        proposed="C",  # reverse orientation on the newer write
        candidate="A",
        batch_id="new",
        created_at="2026-07-04T00:00:00.000000+00:00",
        tenable=False,
        confidence=0.95,
        rationale="Newer: they contradict.",
    )

    index = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    verdict = index.lookup("A", "C")
    assert verdict.tenable is False  # newer row's verdict
    assert verdict.confidence == 0.95
    assert verdict.rationale == "Newer: they contradict."
    assert verdict.batch_id == "new"
    assert verdict.created_at == "2026-07-04T00:00:00.000000+00:00"
    assert len(index) == 1


def test_same_created_at_tie_breaks_on_rowid(tmp_path) -> None:
    """On a ``created_at`` tie the later-inserted (higher rowid) row wins (§8-3).

    Two batches share one ``created_at``; ``ORDER BY created_at ASC, rowid ASC`` +
    last-write-wins makes the later INSERT deterministically win — safe because the
    append-only corpus never deletes, so no freed rowid is reused (§6).
    """
    store = TelemetryStore(_telemetry_path(tmp_path))
    tie = "2026-07-05T12:00:00.000000+00:00"
    _seed_pair(
        store,
        proposed="A",
        candidate="C",
        batch_id="first",
        created_at=tie,
        tenable=True,
        confidence=0.60,
    )
    _seed_pair(
        store,
        proposed="A",
        candidate="C",
        batch_id="second",
        created_at=tie,
        tenable=False,
        confidence=0.99,
    )

    index = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    verdict = index.lookup("A", "C")
    assert verdict.batch_id == "second"  # later rowid wins
    assert verdict.tenable is False
    assert verdict.confidence == 0.99
    assert len(index) == 1


def test_null_proposed_hash_is_skipped(tmp_path) -> None:
    """A row with NULL ``proposed_hash_if_any`` forms no key and never appears (§8-4)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    _seed_pair(store, proposed=None, candidate="C", batch_id="b1")  # no pair key
    _seed_pair(store, proposed="A", candidate="C", batch_id="b2")  # valid pair

    index = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert len(index) == 1
    assert index.lookup("A", "C") is not None

    # Both rows are physically written — the NULL row is filtered in SQL, not unwritten.
    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 2
    finally:
        conn.close()


def test_corrupt_db_returns_reuse_unavailable(tmp_path) -> None:
    """A corrupt image degrades to a typed ``ReuseUnavailable``, never raises (§8-5a).

    ``mode=ro`` opens the garbage file fine; the ``SELECT`` iteration raises
    ``sqlite3.DatabaseError`` — caught and returned as the typed value.
    """
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)  # boots a real schema first
    with open(path, "wb") as fh:
        fh.write(b"this is not a sqlite database at all\x00\xff\x01\x02")

    result = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert isinstance(result, ReuseUnavailable)
    assert not isinstance(result, ReuseIndex)
    assert result.detail  # a non-empty machine/log string


def test_unopenable_path_returns_reuse_unavailable(tmp_path) -> None:
    """An unopenable path (parent dir removed) degrades to ``ReuseUnavailable`` (§8-5b).

    ``open_connection`` wraps the connect-time ``sqlite3.OperationalError`` in Mitos's
    ``DatabaseError`` *before* any query runs — the reader must catch that too, or the
    degradation escapes (scout Discrepancy #1).
    """
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    shutil.rmtree(os.path.dirname(path))  # remove .mitos/ — file can't be opened

    result = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert isinstance(result, ReuseUnavailable)
    assert result.detail


def test_empty_store_returns_empty_index(tmp_path) -> None:
    """A freshly-booted store with zero rows → an empty ``ReuseIndex``, NOT unavailable (§8-6).

    The load-bearing distinction (Key Decision 5): healthy-empty and degraded are
    *different types* — 2c's exit-0-vs-exit-2 fork keys on exactly this line.
    """
    store = TelemetryStore(_telemetry_path(tmp_path))
    result = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert isinstance(result, ReuseIndex)
    assert not isinstance(result, ReuseUnavailable)
    assert len(result) == 0
    assert result.lookup("A", "C") is None


def test_load_reuse_index_does_not_mutate(tmp_path) -> None:
    """The read leaves row counts and schema byte-identical (§8-7)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    _seed_pair(store, proposed="A", candidate="C", batch_id="b1")
    _seed_pair(store, proposed="D", candidate="F", batch_id="b2")

    conn = _open(path)
    try:
        rows_before = _count(conn, "conflict_checks")
        schema_before = _schema_snapshot(conn)
    finally:
        conn.close()

    store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == rows_before
        assert _schema_snapshot(conn) == schema_before
    finally:
        conn.close()


def test_read_only_connection_physically_rejects_writes(tmp_path) -> None:
    """The ``mode=ro`` reader connection makes "no writes" structural (§8-7 stretch).

    A write on a ``read_only=True`` connection raises ``sqlite3.OperationalError``
    ("attempt to write a readonly database") — a stronger guarantee than row-count
    invariance (Key Decision 2). This proves the connection mode the reader opens with
    (asserted in the MI-8 test below) cannot mutate the corpus.
    """
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    _seed_pair(store, proposed="A", candidate="C", batch_id="b1")

    ro = open_connection(path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM conflict_checks;")
    finally:
        ro.close()


def test_load_reuse_index_routes_through_mi8_chokepoint(tmp_path, monkeypatch) -> None:
    """The read opens exactly one connection, ``read_only=True``, via the chokepoint (§8-8).

    Extends the 1a spy: the read-only path still carries ``foreign_keys=ON`` and
    ``busy_timeout`` (both live outside ``open_connection``'s write-only PRAGMA guard).
    The spy is installed AFTER boot + seed, so it captures only the read's connection.
    """
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)
    _seed_pair(store, proposed="A", candidate="C", batch_id="b1")

    real_open = telemetry_module.open_connection
    calls = []  # (read_only, foreign_keys, busy_timeout)

    def spy(db_path: str, read_only: bool = False) -> sqlite3.Connection:
        conn = real_open(db_path, read_only=read_only)
        calls.append(
            (
                read_only,
                conn.execute("PRAGMA foreign_keys;").fetchone()[0],
                conn.execute("PRAGMA busy_timeout;").fetchone()[0],
            )
        )
        return conn

    monkeypatch.setattr("mitos.telemetry.open_connection", spy)
    result = store.load_reuse_index(prompt_version=PIN_PROMPT, model_alias=PIN_ALIAS)
    assert isinstance(result, ReuseIndex)

    assert len(calls) == 1  # exactly one bulk read, no per-pair probe
    read_only, foreign_keys, busy_timeout = calls[0]
    assert read_only is True
    assert foreign_keys == 1
    assert busy_timeout == BUSY_TIMEOUT_MS


# ===========================================================================
# Rung 3 — the commentary-reconcile attribution row (P8)
# ===========================================================================
#
# Every applied commentary reconcile writes one durable record of what changed.
# Without it, an agent's markdown edit plus `sync --yes` is exactly as
# unattributable afterwards as mutating the graph directly would have been —
# `updated_at` says only THAT something changed. P8 requires every graph mutation
# to be attributable, so this row is what makes the reconcile's whole argument
# hold rather than being polish on top of it.
#
# It lives HERE, in the sibling, and not in `graph.sqlite`, because after a
# reconcile the prior values exist nowhere else on earth — the markdown was
# rewritten by the agent and the graph was updated by the reconcile. That makes the
# row non-rebuildable in the exact sense two recorded ADRs already ruled on, both
# the same way: the corpus must survive the truth-rebuild.

def test_commentary_audit_table_exists_at_rung_three(tmp_path) -> None:
    """The audit table ships as a plain additive rung on telemetry's own ladder."""
    from mitos.telemetry import TELEMETRY_MIGRATION_STEPS, TelemetryStore

    assert [rung for rung, _fn in TELEMETRY_MIGRATION_STEPS] == [1, 2, 3, 4]

    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    try:
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(commentary_audit)")}
    finally:
        conn.close()
    assert cols, "rung 3 must create `commentary_audit`"
    for column in ("audit_id", "node_id", "slug", "fields_changed", "prior_values",
                   "new_values", "outcome", "correlates_to", "mitos_version",
                   "created_at"):
        assert column in cols, f"missing column: {column}"


def test_node_id_is_a_plain_column_not_a_foreign_key(tmp_path) -> None:
    """No FK to `nodes` — the graph is a different, disposable file.

    Following the verbatim in-file precedent (`batch_id` is a plain column, NOT an
    FK — a training label outlives graph surgery). An FK here would be worse than
    useless: the table is not even in the same database, and an attribution row must
    outlive the node it attributes, which a retired-and-dropped node would otherwise
    take with it.
    """
    from mitos.telemetry import TelemetryStore

    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    try:
        # Assert the table EXISTS first: `PRAGMA foreign_key_list` on a missing table
        # returns an empty list, so the FK assertion alone passes vacuously.
        assert list(conn.execute("PRAGMA table_info(commentary_audit)")), \
            "commentary_audit must exist for this assertion to mean anything"
        fks = list(conn.execute("PRAGMA foreign_key_list(commentary_audit)"))
    finally:
        conn.close()
    assert fks == [], f"commentary_audit must carry no foreign keys, found {fks}"


def test_an_intent_row_records_both_prior_and_new_values(tmp_path) -> None:
    """The row carries BOTH sides, because self-detection depends on both.

    "Carrying the prior values" under-specifies: sync holds the FileLock, so a
    crash-phantom is necessarily the newest row and its NEW values can be compared
    against the graph to spot it. Historically, chain consistency still fingers a
    phantom, because its successor records a `prior` equal to the phantom's `prior`
    rather than its `new`.
    """
    from mitos.telemetry import CommentaryAuditRow, TelemetryStore

    store = TelemetryStore(str(tmp_path / "telemetry.sqlite"))
    audit_id = store.record_commentary_intent(
        CommentaryAuditRow(
            audit_id="a1", node_id="node-abc", slug="probe-slug",
            fields_changed=["rejected_paths"],
            prior_values={"rejected_paths": "the old reasoning"},
            new_values={"rejected_paths": "the corrected reasoning"},
            mitos_version="0.12.0",
        ),
        created_at="2026-07-27T00:00:00.000000+00:00",
    )
    assert audit_id == "a1"

    rows = store.read_commentary_audit()
    assert len(rows) == 1
    row = rows[0]
    assert row["prior_values"] == {"rejected_paths": "the old reasoning"}
    assert row["new_values"] == {"rejected_paths": "the corrected reasoning"}
    assert row["fields_changed"] == ["rejected_paths"]
    assert row["outcome"] is None, "an intent row is open until an outcome closes it"


def test_a_failure_is_closed_by_an_appended_outcome_row_never_a_delete(tmp_path) -> None:
    """Append-only failure closure — telemetry never UPDATEs or DELETEs.

    A reconcile can raise `CommitError` (MI-13 FM2: dropping a `Supersedes:` line
    resurrects a predecessor into a slug-collision rollback). Deleting the intent row
    would be the obvious cleanup and is forbidden: the discipline here is append-only,
    so the failure is recorded as a correlated outcome row. Intent row + no failure
    row + a graph matching the new values ⇒ applied.
    """
    from mitos.telemetry import CommentaryAuditRow, TelemetryStore

    store = TelemetryStore(str(tmp_path / "telemetry.sqlite"))
    store.record_commentary_intent(
        CommentaryAuditRow(
            audit_id="a1", node_id="node-abc", slug="probe-slug",
            fields_changed=["context"], prior_values={"context": "old"},
            new_values={"context": "new"}, mitos_version="0.12.0",
        ),
        created_at="2026-07-27T00:00:00.000000+00:00",
    )
    store.record_commentary_outcome(
        audit_id="a2", correlates_to="a1", outcome="failed: slug_collision",
        created_at="2026-07-27T00:00:01.000000+00:00", mitos_version="0.12.0",
    )

    rows = store.read_commentary_audit()
    assert len(rows) == 2, "the intent row must survive — closure is an APPEND"
    intent = [r for r in rows if r["audit_id"] == "a1"][0]
    closure = [r for r in rows if r["audit_id"] == "a2"][0]
    assert intent["outcome"] is None
    assert closure["correlates_to"] == "a1"
    assert closure["outcome"] == "failed: slug_collision"


def test_the_audit_connection_runs_synchronous_full(tmp_path) -> None:
    """`PRAGMA synchronous=FULL` on the audit write, closing the power-loss window.

    Both databases run WAL with `synchronous=NORMAL`, so their WAL syncs are
    INDEPENDENT: on power loss the graph commit can survive while the earlier
    telemetry write is lost — the one thing write-ahead ordering exists to prevent.
    One line, and reconciles are rare.
    """
    from mitos.telemetry import TelemetryStore

    from mitos.telemetry import CommentaryAuditRow

    store = TelemetryStore(str(tmp_path / "telemetry.sqlite"))
    with store._audit_connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2, "2 == FULL"
        # `open_connection` applies its own pragma suite, including
        # `synchronous=NORMAL` — so the value that matters is the one in force AFTER a
        # real write on the same connection, not merely at hand-out time.
        conn.execute("INSERT INTO commentary_audit (audit_id, mitos_version, created_at) "
                     "VALUES (?, ?, ?)", ("probe", "0.12.0", "2026-07-27T00:00:00+00:00"))
        conn.commit()
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2, (
            "FULL must still be in force after the INSERT"
        )
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    # And the mandatory writer itself uses that connection.
    store.record_commentary_intent(
        CommentaryAuditRow(audit_id="a1", node_id="n", slug="s", fields_changed=["context"],
                           prior_values={"context": "a"}, new_values={"context": "b"},
                           mitos_version="0.12.0"),
        created_at="2026-07-27T00:00:01.000000+00:00",
    )
    assert len(store.read_commentary_audit()) == 2


def test_audit_rows_survive_a_rebuild_and_a_graph_wipe(tmp_path) -> None:
    """The whole reason for the sibling: the row is non-rebuildable data.

    After a reconcile the prior commentary values exist nowhere else — the markdown
    was rewritten, the graph was updated. `rm graph.sqlite` is sanctioned by P6 and
    named verbatim in two recorded ADRs; a graph-homed audit table would die there
    with nothing to rebuild from.
    """
    from mitos.telemetry import CommentaryAuditRow, TelemetryStore

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    GraphStore(config.db_path)  # a real graph beside it

    store = TelemetryStore(config.telemetry_path)
    store.record_commentary_intent(
        CommentaryAuditRow(
            audit_id="a1", node_id="node-abc", slug="probe-slug",
            fields_changed=["context"], prior_values={"context": "old"},
            new_values={"context": "new"}, mitos_version="0.12.0",
        ),
        created_at="2026-07-27T00:00:00.000000+00:00",
    )

    os.remove(config.db_path)
    for sidecar in (config.db_path + "-wal", config.db_path + "-shm"):
        if os.path.exists(sidecar):
            os.remove(sidecar)

    assert len(TelemetryStore(config.telemetry_path).read_commentary_audit()) == 1
