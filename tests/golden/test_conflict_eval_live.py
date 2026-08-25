"""Layer-B conflict eval — live, integration-gated, banded (Conflict-sensor §6.3, T7 scaffold).

Drives the SHIPPED conflict facade (`mitos.conflict.run_conflict_check`) with the REAL
SONNET judge over the six golden `conflict:` fixtures against the frozen Harbor corpus,
and scores it. The conflict twin of `test_retrieval_live.py`: read-only over the shipped
pipeline, banded not exact. The tight numbers live in a metrics report + a soft baseline
diff (`_conflict_harness.py`); the only HARD asserts here are the DETERMINISTIC screening
(the declared-target drop is pure graph logic) and one lenient "sensor fundamentally
works" smoke floor — the judge is stochastic, so its verdict quality is soft-diffed, never
red (per Layer B: a semantic regression WARNS for review, never hard-fails, never
auto-seeds the baseline).

Gating (mirrors the other `*_live.py` suites):
  * skips cleanly without BOTH `GEMINI_API_KEY` (embeddings) and `ANTHROPIC_API_KEY`
    (judgment) — conflict needs both — HAS_LIVE_KEYS.
  * skips loudly if Qdrant is unreachable (`QdrantVectorStore.__init__` RAISES on refusal).
  * degrades an embed-quota 429 in populate to a loud skip via `skip_on_embed_quota`.
  * degrades ANY facade `Unavailable` (embed / vector-store / judge timeout-or-5xx) to a
    loud named skip — the executor fail-opens, so a quota-exhausted judge RETURNS
    `Unavailable(JUDGMENT_TIMEOUT)` rather than raising; we inspect the return (Warning E),
    a `try/except anthropic.*` would never fire.
  * uses a throwaway `mitos-tmp-golden-*` collection (swept by conftest even if teardown
    misses) plus its own teardown.

Baseline discipline: 4a ships NO seeded conflict baseline (the judge is live/costly, the
floor is still PROVISIONAL — a meaningful baseline is 4b's calibrated, reviewed act). The
soft-diff test skips loudly when it is absent; seeding is `MITOS_UPDATE_BASELINE=1`-only.
"""

import os
import sys
import tempfile
import uuid
import warnings

import pytest
import requests

from mitos import conflict
from mitos.conflict import CONFLICT_SIMILARITY_FLOOR, Unavailable
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.models import get_embedding_model_id
from mitos.parser import parse_entry_stream
from mitos.telemetry import TelemetryStore
from mitos.vector_store import QdrantVectorStore

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for live_helpers
from _harness import CORPUS_PATH, build_reference_graph  # noqa: E402
import _conflict_harness as CH  # noqa: E402  (pulls anthropic in — the quarantine boundary)
import _semantic_harness as H  # noqa: E402
from live_helpers import live_tests_disabled, skip_on_embed_quota  # noqa: E402
from metrics import recommend_floor  # noqa: E402

# Loose smoke floor — of the genuine contradictions that reached the judge, at least half
# must surface. NOT a calibrated per-fixture floor (the real measurement is the report +
# baseline diff): the judge is stochastic and the floor is provisional, so keep it lenient
# enough that ordinary judge jitter never reds it. Raise deliberately (reviewed) once 4b
# has measured the numbers. Mirrors `SMOKE_RECALL_FLOOR` in the retrieval suite.
SMOKE_NOT_TENABLE_RECALL_FLOOR = 0.5

# The calibration probe floor (plan D1/§7). At 0.0 the S5 gate screens NOTHING, so every
# candidate ranked in the top-K is scored and its raw similarity captured — the worst case
# for truncation (a lower floor only ADDS candidates, never removes the named one). Fixture
# 4's declared-drop (S4) is floor-independent, so it still drops here.
CALIBRATION_PROBE_FLOOR = 0.0

# Per-metric regression bands stored with a seeded baseline (4b seeds; 4a only ships the hook).
CONFLICT_BASELINE_BANDS = {
    "not_tenable_recall": 0.20,
    "not_tenable_precision": 0.20,
    "same_polarity_fp_rate": 0.20,
}


def _load_live_env() -> None:
    """Loads keys from the repo-root .env into os.environ (mirrors the live suites)."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_live_env()
# Conflict needs BOTH surfaces: Gemini embeddings (candidate gather) + Anthropic judgment.
HAS_LIVE_KEYS = (not live_tests_disabled()) and bool(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")
# 5c: `GeminiEmbeddingProvider` reads no process environment — the key is
# supplied explicitly. Read here, beside the gate that already reads it, in
# the same module-constant idiom as `QDRANT_URL` above.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY both required — the Layer-B conflict "
    "eval drives live embeddings AND the live SONNET judge.",
)


def _skip_if_unavailable(result) -> None:
    """Turns a facade `Unavailable` into a loud, named skip (environmental, not a defect).

    The conflict facade fail-opens: an embed-quota 429, a Qdrant fault, or a judge
    quota/5xx/timeout all RETURN a typed `Unavailable` (never raise past the seam), so
    the eval hands it back rather than measuring garbage. Each reason is a distinct
    environmental cause, none a code defect — skip loudly with the reason so a keys-present
    CI never reds on a live outage.

    Args:
        result: A `run_conflict_eval` return value.
    """
    if isinstance(result, Unavailable):
        from mitos.conflict import JUDGMENT_DEFECT_REASONS
        if result.reason in JUDGMENT_DEFECT_REASONS:
            pytest.fail(
                f"conflict facade returned Unavailable(reason={result.reason.value}) — "
                f"this is a CODE DEFECT, not an environmental fault. "
                f"detail: {result.detail}"
            )
        pytest.skip(
            f"conflict facade degraded to Unavailable(reason={result.reason.value}) — "
            f"environmental; NOT a code defect. detail: {result.detail}"
        )


@pytest.fixture(scope="module")
def conflict_index():
    """Builds the Harbor graph + index + the proposal entry map for the conflict eval.

    Yields:
        A tuple ``(store, provider, vstore, entries_by_slug, collection)``.
    """
    if not H.qdrant_reachable(QDRANT_URL):
        pytest.skip(
            f"Qdrant unreachable at {QDRANT_URL} — the Layer-B conflict eval needs a live "
            f"vector store. Environmental, not a code defect."
        )

    import tempfile

    collection = f"mitos-tmp-golden-{uuid.uuid4().hex[:8]}"
    tmp = tempfile.mkdtemp(prefix="mitos-golden-conflict-")
    store = build_reference_graph(os.path.join(tmp, "graph.sqlite"))

    # The proposal source: parse the SAME frozen corpus into {slug: ParsedEntry}. The
    # facade is driven with the proposal's own ParsedEntry (2a's own-slug guard drops the
    # self-match; the over-fetch leaves margin for the real candidate).
    failures: list = []
    entries = parse_entry_stream(
        open(CORPUS_PATH, encoding="utf-8").read(), "decision", failures=failures
    )
    assert not failures, f"reference corpus failed to parse cleanly: {failures}"
    entries_by_slug = {e.slug: e for e in entries}

    cache_dir = os.path.join(H.GOLDEN_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"embeddings-{get_embedding_model_id()}.sqlite")
    provider = GeminiEmbeddingProvider(cache_path, api_key=GEMINI_API_KEY)
    vstore = QdrantVectorStore(QDRANT_URL, collection)

    try:
        with skip_on_embed_quota():
            H.populate_index(store, provider, vstore)
        yield store, provider, vstore, entries_by_slug, collection
    finally:
        try:
            requests.delete(
                f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5
            )
        except requests.RequestException:
            pass  # conftest sweep catches mitos-tmp-* as a backstop


def test_conflict_declared_drop_and_smoke_floor(conflict_index):
    """Hard-asserts the deterministic declared drop + a lenient aggregate smoke floor.

    The declared-target drop (fixture 4: `harbor-sync-crdt-merge` declares
    `Contradicts: harbor-sync-last-write-wins`) is pure 2b graph logic, independent of the
    stochastic judge — so it is the one per-fixture behaviour we HARD-assert. The smoke
    floor is deliberately loose (`SMOKE_NOT_TENABLE_RECALL_FLOOR`): a "the sensor
    fundamentally judges contradictions" line, not a calibrated gate.
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)

    summary = CH.conflict_human_summary(report)
    path = H.write_report(report, "conflict-latest")
    print(f"\n{summary}\n[report] {path}")

    # (1) DETERMINISTIC: the declared-contradiction candidate never reached the judge.
    declared = [f for f in report["fixtures"] if f["kind"] == "declared-contradiction"]
    assert len(declared) == 1, "expected exactly one declared-contradiction fixture"
    dfx = declared[0]
    assert not dfx["judged"], (
        f"declared-target drop FAILED: {dfx['candidate']} reached the judge though "
        f"{dfx['proposal']} declares Contradicts it — 2b's declared screen regressed."
    )

    # (2) SMOKE: of the contradictions that WERE judged, at least half surfaced.
    recall = report["aggregate"]["not_tenable_recall"]
    assert recall >= SMOKE_NOT_TENABLE_RECALL_FLOOR, (
        f"not_tenable_recall={recall:.2f} < {SMOKE_NOT_TENABLE_RECALL_FLOOR} smoke floor "
        f"— the sensor is missing genuine contradictions it judged. See the report."
    )


def test_conflict_floor_calibration(conflict_index):
    """Probe run at floor=0.0: capture similarities, recommend the floor, HARD-assert recall.

    The 4b calibration act. Runs the eval at :data:`CALIBRATION_PROBE_FLOOR` so every named
    candidate is scored (nothing screened by S5), writes the calibration readout, and computes
    :func:`recommend_floor`. The one HARD invariant (deterministic — embeddings are stable for
    fixed text+model, so this reds CI on a genuine recall regression, not judge jitter): every
    judged contradiction fixture retrieves at a similarity **>= the landed
    ``CONFLICT_SIMILARITY_FLOOR``** — i.e. the calibrated floor never screens a real
    contradiction (recall-first). Everything judge-quality stays soft (the report + the
    soft-diff test), never red — the Layer B law.
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge(),
            floor=CALIBRATION_PROBE_FLOOR,
        )
    _skip_if_unavailable(report)

    summary = CH.conflict_human_summary(report)
    path = H.write_report(report, "conflict-calibration-probe")
    recommended = recommend_floor(report["fixtures"])
    print(
        f"\n{summary}\n"
        f"[recommended floor] {recommended}  "
        f"(landed CONFLICT_SIMILARITY_FLOOR = {CONFLICT_SIMILARITY_FLOOR})\n"
        f"[report] {path}"
    )

    # The judged contradictions (oracle expected_tenable is False) — the recall-first set.
    contradictions = [
        f for f in report["fixtures"]
        if f["expected_tenable"] is False and f["judged"]
    ]
    assert contradictions, (
        "no judged contradiction fixtures in the probe run — the corpus/oracle changed, or "
        "every contradiction was screened/truncated even at floor=0.0 (a top_k-ceiling / hub "
        "signal, not a floor issue). See the report."
    )
    for f in contradictions:
        # A judged contradiction MUST carry a similarity (its candidate reached the judge).
        assert f["similarity"] is not None, (
            f"judged contradiction {f['proposal']} ✗ {f['candidate']} ({f['kind']}) has no "
            f"similarity — inconsistent report record."
        )
        # RECALL-FIRST: the landed floor must not screen a genuine contradiction.
        assert f["similarity"] >= CONFLICT_SIMILARITY_FLOOR, (
            f"recall-first VIOLATION: {f['kind']} {f['proposal']} ✗ {f['candidate']} "
            f"retrieves at similarity={f['similarity']:.4f} < landed "
            f"CONFLICT_SIMILARITY_FLOOR={CONFLICT_SIMILARITY_FLOOR} — the calibrated floor "
            f"would screen this real contradiction before judgment. Recalibrate the floor "
            f"DOWN (min contradiction similarity − margin). See the calibration readout."
        )


def test_conflict_baseline_diff_soft_gate(conflict_index):
    """Soft gate: diff the conflict aggregate vs the seeded baseline; WARN, never hard-fail."""
    store, provider, vstore, entries_by_slug, _ = conflict_index
    baseline = CH.load_conflict_baseline()
    if baseline is None:
        pytest.skip(
            "no conflict.baseline.metrics.json yet — 4a ships the seed hook but not the "
            "baseline (the floor is provisional; seeding is 4b's calibrated act). Seed it "
            "with MITOS_UPDATE_BASELINE=1 (reviewed), then commit it."
        )
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)
    regressions = CH.conflict_baseline_diff(report, baseline)
    if regressions:
        # Layer B is banded + service-dependent + judge-stochastic — a regression flags
        # for human review, it does NOT red CI (never auto-accept, never hard-fail).
        warnings.warn(
            "Layer-B conflict regression vs baseline (REVIEW — do not rubber-stamp):\n"
            + "\n".join(
                f"  {r['metric']}: {r['baseline']:.3f} -> {r['current']:.3f} "
                f"({r['direction']}, band {r['band']})"
                for r in regressions
            ),
            UserWarning,
        )


@pytest.mark.skipif(
    os.environ.get("MITOS_UPDATE_BASELINE") != "1",
    reason="baseline seeding is explicit-only — set MITOS_UPDATE_BASELINE=1 to (re)seed.",
)
def test_seed_conflict_baseline(conflict_index):
    """Explicit-only: freezes the current conflict aggregate as the reviewed baseline.

    4a ships this hook but does NOT run it — a meaningful conflict baseline is 4b's
    calibrated, reviewed act (the floor is still provisional; the judge is live/costly).
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)
    CH.write_conflict_baseline(report, CONFLICT_BASELINE_BANDS)
    print(
        f"\n[conflict baseline seeded — REVIEW before committing]\n"
        f"{CH.conflict_human_summary(report)}"
    )


# =========================================================================== #
# Phase 5b — corpus-mode fixtures (the `mitos check` SWEEP over the frozen corpus)
#
# Where the four tests above drive the per-proposal FACADE, these drive the corpus
# ENGINE (`plan_corpus_check` → `execute_corpus_check`) over the whole store once —
# the sweep `mitos check` performs. They reuse the `conflict_index` fixture verbatim
# and derive the expected surfaced/screened sets from the same six `conflict:`
# fixtures (KD2 — "reuse, don't re-author"). HARD-assert the structural properties
# (declared-pair screen, exit contract, reuse determinism); keep judge quality a
# lenient smoke floor (the judge is stochastic; `not_tenable_precision=0.75` is a
# named v0.2 limit).
# =========================================================================== #

from mitos import check  # noqa: E402

# The two declared strong-edge pairs in the frozen corpus (verified
# decisions.reference.md:62 / :104) — the DoD-2 must-NEVER-judge pins.
_DECLARED_CONTRADICTS = ("harbor-sync-crdt-merge", "harbor-sync-last-write-wins")
_DECLARED_NARROWS = ("harbor-all-endpoints-authenticated", "harbor-health-endpoint-public")
# The DoD-1 unambiguous pin — an UNDECLARED genuine contradiction.
_DELETE_PAIR = ("harbor-delete-is-immediate-hard", "harbor-delete-is-soft-30d")
# The P9 multilingual pair.
_MULTILINGUAL_PAIR = ("harbor-duomenu-saugojimas-lietuvoje", "harbor-duomenys-gali-buti-es")


def _hash(store, slug: str) -> str:
    """Resolves an oracle slug to its live content hash (MI-2 — never a stored slug)."""
    node = store.get_node_by_slug(slug)
    assert node is not None, f"corpus/oracle drift: {slug} is not an active node"
    return node["id"]


def _pair_key(store, a_slug: str, b_slug: str) -> tuple:
    """The orientation-blind pair key over two slugs' live hashes."""
    return tuple(sorted((_hash(store, a_slug), _hash(store, b_slug))))


def _fresh_telemetry():
    """A fresh, migrated, EMPTY TelemetryStore (healthy no-priors — ReuseIndex len 0)."""
    tmp = tempfile.mkdtemp(prefix="mitos-golden-corpus-tel-")
    return TelemetryStore(os.path.join(tmp, "telemetry.sqlite"))


def _plan_only(store, provider, vstore):
    """Drains the outbox, then plans a corpus check (judge-free, zero spend, telemetry=None).

    T2 and P9 assert PLAN-stage structural properties (`plan.pairs`) that the reuse
    partition and judgment never touch — so telemetry=None (the reuse_unavailable
    fork) is harmless here: it changes the fresh/reused partition, not the pair set.
    """
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    return check.plan_corpus_check(
        store=store,
        embed_provider=provider,
        vector_store=vstore,
        telemetry=None,
        model_alias=CH._JUDGMENT_MODEL_ALIAS,
    )


def test_corpus_check_delete_pair_new_finding_exit_1(conflict_index):
    """T1/DoD-1: a full corpus sweep surfaces the undeclared delete contradiction, exit 1.

    The vision's core proof at the corpus level: an UNDECLARED genuine contradiction
    (immediate hard-delete ✗ 30-day soft-delete) that neither side declares an edge to
    is caught by a single sweep as a NEW finding — both nodes named, rationale +
    confidence carried — over a fresh (empty) telemetry (so nothing is "known"), exit 1.
    The aggregate recall is a lenient smoke floor (KD2 — the judge is imperfect;
    DoD-1 rides on the delete pair, not the whole set).
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    telemetry = _fresh_telemetry()
    with skip_on_embed_quota():
        bundle = CH.run_corpus_check_eval(
            oracle, entries_by_slug, provider, vstore, store,
            CH.make_live_judge(), telemetry,
        )
    _skip_if_unavailable(bundle)

    path = H.write_report(bundle["report"], "conflict-corpus-latest")
    print(f"\n{CH.corpus_human_summary(bundle)}\n[report] {path}")

    result = bundle["result"]
    delete_key = _pair_key(store, *_DELETE_PAIR)
    surfaced = {tuple(sorted((f.proposal_hash, f.partner_hash))) for f in result.findings}

    # (1) The delete pair surfaced as a NEW finding.
    assert delete_key in surfaced, (
        "DoD-1 FAILED: the undeclared delete contradiction did not surface in a full "
        "corpus sweep. See the report."
    )
    delete_finding = next(
        f for f in result.findings
        if tuple(sorted((f.proposal_hash, f.partner_hash))) == delete_key
    )
    assert delete_finding.novelty == "new", (
        f"delete finding novelty={delete_finding.novelty!r}, expected 'new' over an "
        f"empty telemetry — the novelty partition regressed."
    )
    # (2) Both Letter payloads + rationale + confidence ride the finding (both nodes).
    assert delete_finding.proposal_node.get("core_axiom"), "proposal axiom missing"
    assert delete_finding.partner_node.get("core_axiom"), "partner axiom missing"
    assert delete_finding.rationale, "finding carries no rationale"
    assert 0.0 <= delete_finding.confidence <= 1.0
    assert {delete_finding.proposal_node["slug"], delete_finding.partner_node["slug"]} == set(
        _DELETE_PAIR
    ), "the finding names the wrong pair of decisions"

    # (3) A new finding on a healthy run ⇒ exit 1 (2 dominates only under degradation).
    assert not result.telemetry_write_failures, result.telemetry_write_failures
    assert check.exit_code_for(result) == 1, (
        f"expected exit 1 (new finding, healthy run); got "
        f"{check.exit_code_for(result)} — degradations "
        f"{check.run_degradations(result)}."
    )

    # (4) Lenient smoke floor (KD2) — the sensor fundamentally catches contradictions.
    recall = bundle["aggregate"]["not_tenable_recall"]
    assert recall >= SMOKE_NOT_TENABLE_RECALL_FLOOR, (
        f"corpus not_tenable_recall={recall:.2f} < {SMOKE_NOT_TENABLE_RECALL_FLOOR} — "
        f"the sweep is missing genuine contradictions it judged. See the report."
    )


def test_corpus_declared_pairs_never_judged(conflict_index):
    """T2/DoD-2: both declared strong-edge pairs are screened before judgment (zero spend).

    The strong-edge screen runs inside `plan_corpus_check`, BEFORE any judgment — so
    the assertion is at the PLAN stage, keyless on the judgment axis (KD3). Each
    declared pair's orientation-blind key is absent from `plan.pairs` (and thus from
    every fresh group → the judge is never called on it → zero LLM spend). Non-vacuity
    (mandatory): the drop is the SCREEN, not a retrieval miss — proven by (a) the
    declared edge existing between the two hashes in `get_edges()`, and (b) a
    screenless `gather_candidates` over one endpoint retrieving the other. (The
    single-pair harness's `global-vs-scoped-narrows` fixture — reason `declared_drop`
    — is the standing per-proposal witness of the same retrieval-then-screen path.)
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    with skip_on_embed_quota():
        plan = _plan_only(store, provider, vstore)
    if plan.sweep_degraded is not None:
        _skip_if_unavailable(plan.sweep_degraded)

    plan_pair_keys = {
        tuple(sorted((p.proposal_hash, p.partner_hash))) for p in plan.pairs
    }
    edges = store.get_edges()
    edge_pairs = {
        tuple(sorted((e["source_id"], e["target_id"]))) for e in edges
    }

    for a_slug, b_slug in (_DECLARED_CONTRADICTS, _DECLARED_NARROWS):
        key = _pair_key(store, a_slug, b_slug)
        # Absence from the plan ⇒ never a candidate ⇒ never judged (zero spend).
        assert key not in plan_pair_keys, (
            f"DoD-2 FAILED: the declared pair {a_slug} ✗ {b_slug} reached judgment "
            f"(present in plan.pairs) — the strong-edge screen regressed."
        )
        # Non-vacuity (a): the declared edge exists for the screen to act on.
        assert key in edge_pairs, (
            f"non-vacuity broken: no declared edge between {a_slug} and {b_slug} in "
            f"get_edges() — the corpus/oracle changed."
        )
        # Non-vacuity (b): retrieval WOULD surface the partner (screenless gather).
        node = store.get_node_by_slug(a_slug)
        with skip_on_embed_quota():
            gathered = conflict.gather_candidates(
                node["core_axiom"],
                embed_provider=provider,
                vector_store=vstore,
                store=store,
            )
        _skip_if_unavailable(gathered)
        gathered_ids = {c.node["id"] for c in gathered}
        assert _hash(store, b_slug) in gathered_ids, (
            f"non-vacuity broken: {b_slug} did not co-retrieve for {a_slug} — the "
            f"absence-from-plan would be a retrieval miss, not a screen drop. "
            f"(gathered {len(gathered_ids)} candidates)"
        )

    print(
        f"\n[T2] both declared pairs screened pre-judgment over {len(plan.pairs)} "
        f"swept pairs (run {plan.run_id[:8]})"
    )


def test_corpus_multilingual_pair_exercised(conflict_index):
    """P9: the Lithuanian contradiction pair reaches judgment in corpus mode (not dropped).

    Language sovereignty (§6.3): the multilingual pair must keep being exercised by
    the corpus sweep — neither declares an edge, so both reach judgment. Asserted at
    the PLAN stage (in `plan.pairs` ⇒ judged-or-reused, never silently dropped),
    deterministic and zero-spend (the judge's verdict quality is the stochastic part,
    covered by T1's smoke floor).
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    with skip_on_embed_quota():
        plan = _plan_only(store, provider, vstore)
    if plan.sweep_degraded is not None:
        _skip_if_unavailable(plan.sweep_degraded)

    plan_pair_keys = {
        tuple(sorted((p.proposal_hash, p.partner_hash))) for p in plan.pairs
    }
    key = _pair_key(store, *_MULTILINGUAL_PAIR)
    assert key in plan_pair_keys, (
        "P9 FAILED: the Lithuanian contradiction pair did not reach judgment in the "
        "corpus sweep — the sensor is blind to non-English contradictions. (If this is "
        "a retrieval miss at the provisional floor, it is a soft calibration signal, "
        "but the pair must at least co-retrieve to be exercised.)"
    )
    print(f"\n[P9] multilingual pair reaches judgment (run {plan.run_id[:8]})")


def test_corpus_reuse_determinism_and_scalar_law(conflict_index):
    """T3/DoD-3 + T12: run 2 over run-1's telemetry judges nothing fresh, exit 0.

    One telemetry file across both runs (the reuse index loads once at plan-start from
    pre-run rows — the CHK-D10 novelty boundary). Run 1 (fresh, real judge) writes
    verdicts; run 2 (same telemetry, judge=None) reuses them: `pairs_judged_fresh==0`,
    every finding `novelty=='known'`, findings IDENTICAL to run 1, exit 0. T12 (5b
    half): the run-2 `check_runs` row scalars equal the report. A `--fresh` run
    re-judges (`pairs_judged_fresh>0`) yet a re-confirmation of the standing delete
    finding stays `known` / exit 0.
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    telemetry = _fresh_telemetry()  # ONE store, shared across all three runs

    # --- Run 1: fresh, real judge (writes verdicts) --------------------------
    spy = CH._JudgeSpy(CH.make_live_judge())
    with skip_on_embed_quota():
        run1 = CH.run_corpus_check_eval(
            oracle, entries_by_slug, provider, vstore, store, spy, telemetry,
        )
    _skip_if_unavailable(run1)
    r1 = run1["result"]
    assert spy.calls > 0, "run 1 should have fired fresh judgments"
    assert r1.pairs_judged_fresh > 0, "run 1 judged nothing fresh — corpus/oracle drift"
    assert r1.findings, "run 1 surfaced no findings — expected at least the delete pair"
    assert all(f.novelty == "new" for f in r1.findings), (
        "run 1 over empty telemetry should be all-new"
    )
    run1_surfaced = {
        tuple(sorted((f.proposal_hash, f.partner_hash))): f.rationale
        for f in r1.findings
    }

    # --- Run 2: reuse-only, judge=None (zero fresh spend) --------------------
    with skip_on_embed_quota():
        run2 = CH.run_corpus_check_eval(
            oracle, entries_by_slug, provider, vstore, store, None, telemetry,
        )
    _skip_if_unavailable(run2)
    r2 = run2["result"]
    path = H.write_report(run2["report"], "conflict-corpus-reuse")
    print(f"\n[T3 run2]\n{CH.corpus_human_summary(run2)}\n[report] {path}")

    assert r2.pairs_judged_fresh == 0, (
        f"reuse run judged {r2.pairs_judged_fresh} pairs fresh — reuse regressed."
    )
    assert r2.pairs_reused > 0, "reuse run reused nothing — the index did not carry run 1"
    assert all(f.novelty == "known" for f in r2.findings), (
        "reuse run findings must all be 'known' (standing)"
    )
    run2_surfaced = {
        tuple(sorted((f.proposal_hash, f.partner_hash))): f.rationale
        for f in r2.findings
    }
    assert run2_surfaced == run1_surfaced, (
        "reuse run findings differ from run 1 — reuse must replay the SAME verdicts "
        "(same pairs, same rationale verbatim, M8)."
    )
    assert check.exit_code_for(r2) == 0, (
        f"reuse-only run of standing findings must exit 0; got "
        f"{check.exit_code_for(r2)} — degradations {check.run_degradations(r2)}."
    )

    # --- T12 (5b half): the run-2 check_runs row scalars equal the report ----
    exit2 = check.exit_code_for(r2)
    row = check.check_run_row_from_result(r2, mode="corpus", exit_code=exit2)
    c2 = run2["report"]["corpus"]
    assert row.pairs_reused == r2.pairs_reused == c2["pairs_reused"] > 0
    assert row.pairs_judged_fresh == 0 == c2["pairs_judged_fresh"]
    assert row.findings_known == c2["findings_known"] == len(r2.findings)
    assert row.findings_new == c2["findings_new"] == 0
    assert row.degraded_reason is None, (
        f"healthy reuse run stamped degraded_reason={row.degraded_reason!r}"
    )
    assert row.exit_code == exit2 == c2["exit_code"] == 0

    # --- Run 3: --fresh re-judges, but a standing finding stays known/exit 0 --
    delete_key = _pair_key(store, *_DELETE_PAIR)
    if delete_key in run1_surfaced:  # the DoD-1 pin surfaced on run 1
        spy3 = CH._JudgeSpy(CH.make_live_judge())
        with skip_on_embed_quota():
            run3 = CH.run_corpus_check_eval(
                oracle, entries_by_slug, provider, vstore, store, spy3, telemetry,
                fresh=True,
            )
        _skip_if_unavailable(run3)
        r3 = run3["result"]
        assert r3.pairs_judged_fresh > 0, "--fresh must re-judge (bypass the reuse partition)"
        # The reuse index still loads (novelty read is never bypassed), so every
        # standing finding re-confirmed on a --fresh re-judge stays "known". But the
        # judge is stochastic at the check surface: --fresh re-judges ALL 67 pairs,
        # so it may ALSO surface a borderline pair that run 1 judged not-tenable.
        # Per the novelty rule (check.py:951-952 — "known" iff the pre-run index holds
        # a prior *finding* for the pair) such a pair is correctly "new" → exit 1.
        # So the load-bearing invariant is NOT a fixed whole-run exit code (that is
        # judge-stochastic) but: (a) the standing delete pin re-confirms as known,
        # (b) no run-1 standing finding regresses to "new" (a CHK-D10 bypass would),
        # and (c) the exit is novelty-driven — never a masked degradation.
        r3_findings = {
            tuple(sorted((f.proposal_hash, f.partner_hash))): f for f in r3.findings
        }
        assert delete_key in r3_findings, (
            "--fresh re-run dropped the standing delete contradiction"
        )
        assert r3_findings[delete_key].novelty == "known", (
            "a --fresh re-confirmation of a standing finding must stay 'known' "
            "(novelty read is not bypassed) — do not assert exit 1 here."
        )
        for key in run1_surfaced:
            if key in r3_findings:
                assert r3_findings[key].novelty == "known", (
                    "a run-1 standing finding regressed to 'new' under --fresh — "
                    "the novelty read was bypassed (CHK-D10 violation)."
                )
        assert not check.run_degradations(r3), (
            f"--fresh re-run degraded unexpectedly: {check.run_degradations(r3)} "
            f"— exit must stay novelty-driven, not masked by a degradation."
        )
        assert check.exit_code_for(r3) in (0, 1), (
            f"--fresh re-run exit must be novelty-driven (0 if no borderline pair "
            f"newly surfaced, 1 if one did); got {check.exit_code_for(r3)}."
        )
        print(
            f"[T3 run3 --fresh] re-judged {r3.pairs_judged_fresh} fresh, "
            f"delete pin stays known, exit {check.exit_code_for(r3)} "
            f"(novelty-driven, no degradation)"
        )
