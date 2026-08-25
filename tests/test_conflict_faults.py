"""Phase 6a — Provoked-failure integration tests for the Conflict sensor (DoD-3 / T3).

5a/5b wired the sensor's failure story into ``mitos sync``: a typed ``Unavailable`` fails
open (a loud ``[Conflict sensor unavailable]`` notice, the commit untouched, no telemetry
row) and a per-run aggregate breaker trips the sensor off for the rest of the run, so one
downstream outage costs one penalty, not N. Those are load-bearing resilience *claims*, and
P10 (vision §6.3) says "a resilience claim you have never tested is a hope." This suite
provokes the failures for real and proves the claims hold on the actual ``perform_sync``
loop.

Why this is NOT a rerun of 5b (the non-duplication line, plan D1/D2): 5b hands the loop a
pre-built ``Unavailable`` *object* at the judge seam and its lone substrate-fault test raises
a bare ``RuntimeError`` (the *generic* stderr bulkhead, no breaker). 6a injects the **raw**
fault at the substrate/executor — a fake that actually raises ``EmbeddingError`` /
``VectorStoreError`` / a real ``anthropic.APITimeoutError`` — and proves the pipeline
*converts* each raw fault into the typed ``Unavailable`` (``gather_candidates`` narrow-catch,
conflict.py:214-224; the executor's ``try/except``, conflict_judgment.py:103-116) and
disposes of it correctly (loud stdout notice + breaker + zero rows + commit), on the real
loop. That conversion-wired-end-to-end is the untested gap.

Non-masking discipline (plan D5 / PATTERNS): every fault test asserts *positive* survival —
the committed node is present, the *specific* per-surface notice printed, the downstream
counter frozen — never merely "no exception raised." The regression guard is concrete: if
``gather_candidates``' narrow catch were widened/removed, an ``EmbeddingError`` would travel
the *generic* seam (stderr ``[Warning] Conflict check failed``, no breaker) instead of the
typed stdout ``[Conflict sensor unavailable]`` + trip — so asserting the *stdout* notice
branch + the frozen counter is precisely what reds on that regression.

Discipline: keyless-deterministic — no ``ANTHROPIC_API_KEY``/``GEMINI_API_KEY``, no reachable
service (the shared ``offline`` fixture), no ``HAS_LIVE_KEYS`` gate, no ``skip`` markers (6b's
closeout skip-audit greps this vision — leave no survivor). No new dependency: ``httpx`` is a
hard transitive dep of ``anthropic`` (both already installed); it constructs the anthropic
exception objects. No ``mitos/`` source change, no ``__version__`` bump — test-only.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import httpx
import pytest

import anthropic

from mitos.conflict import ConflictUnavailableReason, Unavailable
from mitos.conflict_judgment import make_judgment_executor
from mitos.config import MitosConfig
from mitos.errors import CollectionMissingError, EmbeddingError, VectorStoreError
from mitos.sync import MitosSyncManager

from _conflict_helpers import (
    _FakeEmbed,
    _FakeVector,
    _RecordingJudge,
    _append_decision,
    _execution,
    _match,
    _read_batch_rows,
    _read_conflict_rows,
    _seed_active,
    env,
    interactive_stdin,
    offline,
)


# --------------------------------------------------------------------------- #
# Raising fakes — subclass the shared fakes (mirrors 5a's `_BoomVector(_FakeVector)`)
# --------------------------------------------------------------------------- #

class _SeveredEmbed(_FakeEmbed):
    """The S1 embedding call raises the TYPED `EmbeddingError` (not a bare RuntimeError).

    Records the attempt first so a test can prove the embed was reached, then raises — the
    facade's narrow catch (conflict.py:216-217) must convert this to `Unavailable(EMBEDDING)`.
    """

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls.append((text, is_query))
        raise EmbeddingError("gemini embedding severed")


class _SeveredVector(_FakeVector):
    """The S2 over-fetch query raises the TYPED `VectorStoreError`.

    Increments `.queries` BEFORE raising so `vector.queries == 1` after an N≥2 run is
    meaningful: entry 1 attempted the query once; entry 2 never reached `query` because the
    breaker gate returned first (the clean freeze counter, plan D3).
    """

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.queries += 1
        raise VectorStoreError("qdrant query severed")


class _CollectionlessVector(_FakeVector):
    """The S2 over-fetch query raises the `CollectionMissingError` SUBCLASS.

    Qdrant is up; the collection is not there. The distinction the whole 1b phase
    turns on, injected at the same seam as its sibling above.
    """

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.queries += 1
        raise CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )


def _req() -> httpx.Request:
    """A dummy request — the anthropic exception classes need one to construct (3b recipe)."""
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _client_raising(exc: BaseException) -> MagicMock:
    """A fake anthropic client whose `with_options(...).messages.create(...)` raises `exc`.

    The SDK call is synchronous → a plain `MagicMock` (no `AsyncMock`). `with_options`
    returns a NEW client, so `create` lives on `client.with_options.return_value.messages
    .create`; its `call_count` is the clean judgment freeze counter (plan D3).
    """
    client = MagicMock()
    client.with_options.return_value.messages.create.side_effect = exc
    return client


def _wire(
    manager: MitosSyncManager,
    *,
    judge: Any,
    embed: Any,
    vector: _FakeVector,
) -> _FakeVector:
    """Wires all three seams with explicit embed/vector instances (the ⚠-1 two-seam gotcha).

    The shared `_wire_fakes` always builds a plain `_FakeVector` internally; the substrate
    legs need to inject a *raising* fake at the exact seam, so this variant takes the embed
    and vector instances directly (the manual-wiring shape 5a's graph-fault test uses).
    """
    manager.embed_provider = embed  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]
    manager._build_conflict_judge = lambda: judge  # type: ignore[assignment]
    return vector


Env = Tuple[MitosConfig, MitosSyncManager, str]


# =========================================================================== #
# Success Criterion 1 — embedding severed (S1) converts + commits + no row
# =========================================================================== #

def test_embedding_severed_converts_to_notice_and_commits(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """`get_embedding` raises `EmbeddingError` → typed `Unavailable(EMBEDDING)` on the real loop.

    The loop prints the loud `[Conflict sensor unavailable]` notice naming *semantic recall*,
    the entry still commits, no `conflict_checks` row is persisted, and the judge is never
    reached (embed raises at S1, before screening). The stdout notice branch is exactly what
    the generic seam would NOT produce — the regression guard for the narrow catch (D5).
    """
    config, manager, _ = env
    embed = _SeveredEmbed()
    judge = _RecordingJudge(_execution([]))  # wired so the seam exists; must never fire
    _wire(manager, judge=judge, embed=embed, vector=_FakeVector(matches=[_match("x", 0.9)]))

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" in out          # the typed-path notice, not stderr
    assert "semantic recall" in out.lower()                # names WHICH substrate went dark
    assert "[Conflict]" not in out                         # a degradation, not a finding
    assert manager.store.get_node_by_slug("health-public") is not None  # commit landed
    assert _read_conflict_rows(config) == []               # fail-open — no row for a degradation
    assert judge.calls == 0                                 # embed raised before the judge
    assert embed.calls                                     # the embed WAS attempted (non-masking)


# =========================================================================== #
# Success Criterion 2 — vector store severed (S2) converts + commits + no row
# =========================================================================== #

def test_vector_store_severed_converts_to_notice_and_commits(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """`vector_store.query` raises `VectorStoreError` → typed `Unavailable(VECTOR_STORE)`.

    Same "semantic recall" notice branch as the embedding leg (the notice switch groups
    EMBEDDING+VECTOR_STORE), the commit lands, no row — and the query was reached exactly once.
    """
    config, manager, _ = env
    vector = _SeveredVector(matches=[_match("x", 0.9)])
    judge = _RecordingJudge(_execution([]))
    _wire(manager, judge=judge, embed=_FakeEmbed(), vector=vector)

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" in out
    assert "semantic recall" in out.lower()
    assert "[Conflict]" not in out
    assert manager.store.get_node_by_slug("health-public") is not None
    assert _read_conflict_rows(config) == []
    assert vector.queries == 1     # S2 was reached and raised once
    assert judge.calls == 0        # the query raised before the judge


# =========================================================================== #
# Success Criterion 3 — substrate one-penalty (N≥2): the breaker trips once
# =========================================================================== #

def test_vector_severed_one_penalty_over_two_entries(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """Vector-severed two-entry run (the CLEAN counter leg, D3): notice once, `queries == 1`.

    Entry 1 raises in `query` → typed `Unavailable` → notice + breaker trip; entry 2's
    top-of-method breaker gate returns before gather, so `query` is never re-attempted. Both
    entries commit; zero rows. Extends 5b Case 6 (which proved the breaker only with a canned
    `Unavailable` at the judge seam) to the raw substrate fault.
    """
    config, manager, _ = env
    vector = _SeveredVector(matches=[_match("x", 0.9)])
    judge = _RecordingJudge(_execution([]))
    _wire(manager, judge=judge, embed=_FakeEmbed(), vector=vector)

    _append_decision(config, "first-entry", "The first decision axiom.")
    _append_decision(config, "second-entry", "The second decision axiom.")
    with patch("builtins.input", side_effect=["a", "a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert out.count("[Conflict sensor unavailable]") == 1  # one penalty, not two
    assert vector.queries == 1                              # entry 2 never re-gathered
    assert judge.calls == 0
    assert manager.store.get_node_by_slug("first-entry") is not None
    assert manager.store.get_node_by_slug("second-entry") is not None
    assert _read_conflict_rows(config) == []


def test_embedding_severed_one_penalty_observable_over_two_entries(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """Embedding-severed two-entry run proves one-penalty OBSERVABLY (D3 counter discipline).

    `embed` raises at S1 *before* `query`, so `vector.queries` stays 0 and cannot prove the
    freeze; and `embed.calls` is muddied by the post-commit `_best_effort_embed` re-embed
    (case 9). So the one-penalty proof here is the *notice count*: exactly one
    `[Conflict sensor unavailable]` over an N≥2 run (the breaker suppressed entry 2's notice),
    both entries commit, zero rows. The freeze *mechanism* is the shared top-of-method gate,
    rigorously counted once via the vector leg above.
    """
    config, manager, _ = env
    embed = _SeveredEmbed()
    judge = _RecordingJudge(_execution([]))
    _wire(manager, judge=judge, embed=embed, vector=_FakeVector(matches=[_match("x", 0.9)]))

    _append_decision(config, "first-entry", "The first decision axiom.")
    _append_decision(config, "second-entry", "The second decision axiom.")
    with patch("builtins.input", side_effect=["a", "a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert out.count("[Conflict sensor unavailable]") == 1  # entry 2's notice suppressed
    assert judge.calls == 0
    assert manager.store.get_node_by_slug("first-entry") is not None
    assert manager.store.get_node_by_slug("second-entry") is not None
    assert _read_conflict_rows(config) == []


# =========================================================================== #
# Success Criteria 4 + 5 — judgment provoked (k=1 of N=3), timeout AND broader error
# =========================================================================== #

@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(anthropic.APITimeoutError(_req()), id="APITimeoutError"),
        pytest.param(
            anthropic.RateLimitError(
                "rate limited", response=httpx.Response(429, request=_req()), body=None
            ),
            id="RateLimitError",
        ),
    ],
)
def test_judgment_provoked_one_penalty_flagship(
    env: Env, capsys: pytest.CaptureFixture, exc: BaseException
) -> None:
    """The REAL executor bound to a raising client, driven through the real sync loop (D2).

    The flagship: a seeded above-floor candidate so the judge is actually reached, then the
    executor's Anthropic call raises — an `APITimeoutError` (the "past CONFLICT_LLM_TIMEOUT_S"
    case simulated WITHOUT a wall-clock wait) or a broader `AnthropicError` subclass
    (`RateLimitError`). The executor converts each to `Unavailable(JUDGMENT_TIMEOUT)`, the
    facade propagates it, and the sync disposes: notice naming *judgment*, breaker trips.

    One-penalty over N=3, proven by BOTH clean counters frozen at their k=1 value:
    `create.call_count == 1` (the executor's Anthropic call) and `vector.queries == 1` (the
    gather). Entries 2 & 3 neither gather nor judge; all three commit; zero rows (a degradation
    persists nothing — CONF-D10). Proves the raw-fault → typed-`Unavailable` conversion wired
    end-to-end, not the executor in isolation (already unit-proven in 3b).
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    client = _client_raising(exc)
    judge = make_judgment_executor(client)
    vector = _FakeVector(matches=[_match("endpoints-auth", 0.9)])
    _wire(manager, judge=judge, embed=_FakeEmbed(), vector=vector)

    _append_decision(config, "entry-one", "The first proposed axiom.")
    _append_decision(config, "entry-two", "The second proposed axiom.")
    _append_decision(config, "entry-three", "The third proposed axiom.")
    with patch("builtins.input", side_effect=["a", "a", "a"]), \
         patch("mitos.conflict_judgment.time.sleep"):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert out.count("[Conflict sensor unavailable]") == 1  # one penalty across three entries
    assert "judgment" in out.lower()                        # names the judge, not semantic recall
    assert "semantic recall" not in out.lower()             # the substrate was healthy
    create = client.with_options.return_value.messages.create
    from mitos.conflict_judgment import _RETRY_BACKOFFS_S
    expected_attempts = 1 + len(_RETRY_BACKOFFS_S)
    assert create.call_count == expected_attempts  # entry 1 retried; entries 2 & 3 frozen
    assert vector.queries == 1     # entry 1 gathered once; entries 2 & 3 frozen (the gather)
    for slug in ("entry-one", "entry-two", "entry-three"):
        assert manager.store.get_node_by_slug(slug) is not None  # all three commit
    assert _read_conflict_rows(config) == []                # no row for a degradation
    assert _read_batch_rows(config) == []


# =========================================================================== #
# Success Criterion 6 — a degradation is behaviourally distinct from a clean-empty
# =========================================================================== #

def test_degradation_is_distinct_from_clean_empty(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """Two rowless outcomes, behaviourally distinct: degradation is LOUD, clean-empty is SILENT.

    Both write no `conflict_checks` row — the shared trait that makes them confusable. The
    distinction the vision forbids conflating (conflict.py docstring): a degradation
    (`Unavailable`) prints `[Conflict sensor unavailable]`; a clean-empty (substrate healthy,
    empty over-fetch → `execution is None`) prints nothing. Run 2 gets a fresh
    `_ConflictSyncRun`, so run 1's tripped breaker does not leak into it.
    """
    config, manager, _ = env
    judge = _RecordingJudge(_execution([]))

    # Run 1 — vector severed → a degradation (loud, rowless).
    _wire(manager, judge=judge, embed=_FakeEmbed(),
          vector=_SeveredVector(matches=[_match("x", 0.9)]))
    _append_decision(config, "entry-degraded", "A degraded-run axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    degraded = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" in degraded
    assert _read_conflict_rows(config) == []

    # Run 2 — healthy substrate, empty over-fetch → a clean-empty (silent, rowless).
    manager.vector_store = _FakeVector(matches=[])  # type: ignore[assignment]
    _append_decision(config, "entry-clean", "A clean-empty-run axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    clean = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" not in clean  # clean-empty never announces
    assert "[Conflict]" not in clean                     # nor a finding

    # Both rowless (the confusable trait) — but only the degradation was loud.
    assert _read_conflict_rows(config) == []
    assert judge.calls == 0                              # clean-empty short-circuits before the judge
    assert manager.store.get_node_by_slug("entry-degraded") is not None
    assert manager.store.get_node_by_slug("entry-clean") is not None


# =========================================================================== #
# Success Criterion 7 — structural reset across runs (the breaker never leaks)
# =========================================================================== #

def test_breaker_resets_across_separate_runs(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """Two separate `perform_sync` calls, substrate severed in both → the notice fires in BOTH.

    "A loud notice on each affected sync" (vision §6.3): the aggregate breaker is per-run
    (a fresh `_ConflictSyncRun` built once per run at the judge-build site), so a run-1 trip
    must never suppress run 2's notice. Both entries commit; no rows.
    """
    config, manager, _ = env
    judge = _RecordingJudge(_execution([]))
    _wire(manager, judge=judge, embed=_FakeEmbed(),
          vector=_SeveredVector(matches=[_match("x", 0.9)]))

    _append_decision(config, "run1-entry", "Run one axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    first = capsys.readouterr().out
    assert first.count("[Conflict sensor unavailable]") == 1

    _append_decision(config, "run2-entry", "Run two axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    second = capsys.readouterr().out
    assert second.count("[Conflict sensor unavailable]") == 1  # fresh run → breaker reset

    assert manager.store.get_node_by_slug("run1-entry") is not None
    assert manager.store.get_node_by_slug("run2-entry") is not None
    assert _read_conflict_rows(config) == []


# =========================================================================== #
# Success Criterion 8 — bulkhead: the raw cause (result.detail) never leaks to stdout
# =========================================================================== #

def test_raw_cause_detail_never_leaks_to_stdout(
    env: Env, capsys: pytest.CaptureFixture
) -> None:
    """All conflict UX wording lives in `sync.py`; the raw cause is logging-only (P7, SC8).

    A distinctive `detail` string is threaded through the raised `VectorStoreError`; the
    surface's `_notice_conflict_unavailable` takes only the typed `reason` (never `detail`),
    so the raw cause must NOT appear in stdout — while the surface-owned notice wording
    (`[Conflict sensor unavailable]` / "semantic recall" / the commit reassurance) does. Proves
    `mitos.conflict` returns a typed reason and never renders a presentation string.
    """
    config, manager, _ = env
    secret = "QDRANT-RAW-CAUSE-DO-NOT-PRINT-7f3a"

    class _DetailVector(_FakeVector):
        def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
            self.queries += 1
            raise VectorStoreError(secret)

    judge = _RecordingJudge(_execution([]))
    _wire(manager, judge=judge, embed=_FakeEmbed(),
          vector=_DetailVector(matches=[_match("x", 0.9)]))

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" in out   # the surface-owned notice fired
    assert "semantic recall" in out.lower()
    assert "commit" in out.lower()                  # the fail-open reassurance line
    assert secret not in out                        # the raw cause is NEVER rendered (the bulkhead)
    assert manager.store.get_node_by_slug("health-public") is not None


# =========================================================================== #
# Guard — the raw fault really is CONVERTED (not merely swallowed by the generic seam)
# =========================================================================== #

def test_typed_faults_produce_typed_unavailable_reasons() -> None:
    """A white-box pin that the raw substrate faults convert to the exact typed reasons.

    The integration tests above assert the *disposition* (notice branch + breaker) that the
    typed conversion drives; this pins the conversion itself at the facade so the two
    "semantic recall" legs are provably distinct reasons under the shared notice wording — a
    guard against a future refactor collapsing them or routing one through the generic seam.
    """
    from mitos.conflict import gather_candidates

    class _Store:
        def get_node_by_slug(self, slug: str) -> Any:  # never reached on these legs
            raise AssertionError("graph read must not run when the substrate is severed")

        def get_active_node_ids(self) -> set:
            # Declared explicitly, not left to the I8 gate's unreadable-graph
            # fallback: a stub missing this method degrades via `except Exception`
            # and the COLLECTION_MISSING leg below would then pass for the wrong
            # reason. A populated graph is what makes an absent collection a gap.
            return {"some-node-id"}

    embed_fault = gather_candidates(
        "Some axiom.", embed_provider=_SeveredEmbed(),
        vector_store=_FakeVector(matches=[_match("x", 0.9)]), store=_Store(),
    )
    assert isinstance(embed_fault, Unavailable)
    assert embed_fault.reason is ConflictUnavailableReason.EMBEDDING

    vector_fault = gather_candidates(
        "Some axiom.", embed_provider=_FakeEmbed(),
        vector_store=_SeveredVector(matches=[_match("x", 0.9)]), store=_Store(),
    )
    assert isinstance(vector_fault, Unavailable)
    assert vector_fault.reason is ConflictUnavailableReason.VECTOR_STORE

    # The third typed fault: a running Qdrant that simply has no such collection.
    # It subclasses the one above, so an arm ordered after it would silently make
    # this leg indistinguishable from an outage.
    missing_fault = gather_candidates(
        "Some axiom.", embed_provider=_FakeEmbed(),
        vector_store=_CollectionlessVector(matches=[_match("x", 0.9)]), store=_Store(),
    )
    assert isinstance(missing_fault, Unavailable)
    assert missing_fault.reason is ConflictUnavailableReason.COLLECTION_MISSING


@pytest.mark.parametrize(
    "active_ids, expect_degraded",
    [
        pytest.param({"a-committed-node"}, True, id="populated-graph-is-a-gap"),
        pytest.param(set(), False, id="empty-graph-is-the-empty-index"),
    ],
)
def test_absent_collection_degrades_only_when_the_graph_has_something_to_index(
    active_ids: set, expect_degraded: bool,
) -> None:
    """I8 at the sweep: absence is a gap iff `mitos reconcile` would index something.

    The predicate has two opposite outcomes, so it needs a fixture for each — a
    one-sided row passes under either behaviour. The empty-graph half is the one
    that was missing: `gather_candidates`' own contract already promises `[]` for
    "a legitimate empty: an empty corpus", and re-keying `check`'s precondition
    onto the operation broke that promise for the state where the collection has
    never been created. It bites hardest on a fresh workspace whose first commit
    runs `mitos check --staged` from a pre-commit hook: exit 2, advised to run
    `mitos reconcile`, which enqueues nothing over an empty active set and so
    cannot heal what it names — a dead end, not a vector.

    Gated on the same `get_active_node_ids` the four read surfaces use, and the
    same set `reconcile_embeddings` enqueues, so the gate and the heal agree by
    construction.
    """
    from mitos.conflict import gather_candidates

    class _Store:
        def get_active_node_ids(self) -> set:
            return active_ids

        def get_node_by_slug(self, slug: str) -> Any:  # no match survives to S3
            raise AssertionError("the query raised — S3 must never run")

    result = gather_candidates(
        "Some axiom.", embed_provider=_FakeEmbed(),
        vector_store=_CollectionlessVector(matches=[_match("x", 0.9)]), store=_Store(),
    )

    if expect_degraded:
        assert isinstance(result, Unavailable)
        assert result.reason is ConflictUnavailableReason.COLLECTION_MISSING
    else:
        assert result == []


def test_an_unreadable_graph_still_degrades_on_an_absent_collection() -> None:
    """The I8 gate's loud branch: "I could not check" is never the healthy empty.

    The carve-out above turns on a graph read, so the read's own failure must not
    inherit the quiet answer — a store that raises has not told us the corpus is
    empty, it has told us nothing.
    """
    from mitos.conflict import gather_candidates

    class _UnreadableStore:
        def get_active_node_ids(self) -> set:
            raise RuntimeError("graph unreadable")

        def get_node_by_slug(self, slug: str) -> Any:
            raise AssertionError("the query raised — S3 must never run")

    result = gather_candidates(
        "Some axiom.", embed_provider=_FakeEmbed(),
        vector_store=_CollectionlessVector(matches=[_match("x", 0.9)]),
        store=_UnreadableStore(),
    )

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.COLLECTION_MISSING
