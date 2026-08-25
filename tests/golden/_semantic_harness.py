"""Machinery for the Layer-B semantic (retrieval) eval — live-service, banded.

Layer B runs natural-language queries through the SHIPPED retrieval path against the
same frozen Harbor corpus Layer A uses, and scores the results. It is a *measurement*
layer: read-only over the existing embedding + vector-store surfaces, it never changes
retrieval behaviour. Unlike Layer A this touches live services (Gemini embeddings +
Qdrant), so its tests are integration-gated and its assertions are banded — the tight
measurement lives in a metrics report + a soft baseline diff, not in hard rank asserts.

This module holds the reusable, service-touching pieces (index population, the eval
loop, provenance, the baseline diff, a reachability probe). The pytest wiring and the
banded hard asserts live in `test_retrieval_live.py`; the pure metric math lives in
`metrics.py` (unit-tested in bare CI).

Index population re-derives each node's embedding input via the shipped
`mitos.sync._embedding_input_text` (the `embedding_text` column is gone in V1a — see
identity.py C2/M8). A store decision exposes its axiom as ``core_axiom`` while the
helper reads it under ``axiom``; the bridge is caller-side here. Feeding a raw store
node straight through would embed the empty string and silently score noise, so
`_derive_embedding_text` asserts non-empty output as a standing guard.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from mitos import __version__ as MITOS_VERSION
from mitos.models import get_embedding_model_id
from mitos.sync import _embedding_input_text
from mitos.vector_store import QdrantVectorStore

sys.path.insert(0, os.path.dirname(__file__))
from metrics import evaluate_fixture  # noqa: E402

GOLDEN_DIR = os.path.dirname(__file__)
SEMANTIC_ORACLE_PATH = os.path.join(GOLDEN_DIR, "oracle.semantic.json")
BASELINE_PATH = os.path.join(GOLDEN_DIR, "baseline.metrics.json")
REPORTS_DIR = os.path.join(GOLDEN_DIR, "reports")

# Metrics where a HIGHER value is better (a drop past the band is a regression).
HIGHER_IS_BETTER = ("recall_at_k", "precision_at_k", "mrr")
# Metrics where a LOWER value is better (a rise past the band is a regression).
LOWER_IS_BETTER = ("hard_negative_fp_rate",)


# ---------------------------------------------------------------------------
# Service reachability
# ---------------------------------------------------------------------------

def qdrant_reachable(qdrant_url: str, timeout: float = 2.0) -> bool:
    """Probes whether a Qdrant instance answers, for a loud skip when it is down.

    `QdrantVectorStore.__init__` RAISES `VectorStoreError` on a connection refusal
    (it eagerly ensures the collection), and `skip_on_embed_quota` only catches
    embedding errors — so without this probe a keys-present / Qdrant-down run reds
    the suite instead of skipping. The test gates on this and skips loudly.

    Args:
        qdrant_url: Base URL of the Qdrant instance (e.g. ``http://localhost:7333``).
        timeout: Per-request timeout in seconds.

    Returns:
        True if ``GET /collections`` returns a 2xx, False on any error or non-2xx.
    """
    try:
        resp = requests.get(f"{qdrant_url.rstrip('/')}/collections", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Index population
# ---------------------------------------------------------------------------

def _derive_embedding_text(node: Dict[str, Any]) -> str:
    """Re-derives a node's embedding-input string, bridging the reader-key gap.

    Routes through the shipped `mitos.sync._embedding_input_text` so the eval embeds
    byte-identically to production. Bridges ``core_axiom`` (store node) → ``axiom``
    (helper param) for decisions, and passes ``topic`` / ``questions_raised`` for
    open questions.

    Args:
        node: A hydrated store node dict from `get_active_decisions` or
            `get_open_questions`.

    Returns:
        The non-empty embedding-input text.

    Raises:
        ValueError: If the node kind is unrecognised.
        AssertionError: If the derived text is empty — the silent-corruption guard
            (an unbridged decision node would yield ``""``).
    """
    kind = node["kind"]
    if kind == "decision":
        text = _embedding_input_text(kind="decision", axiom=node["core_axiom"])
    elif kind == "open_question":
        text = _embedding_input_text(
            kind="open_question",
            topic=node["topic"],
            questions_raised=node["questions_raised"],
        )
    else:
        raise ValueError(f"unexpected node kind for embedding: {kind!r}")
    assert text.strip(), (
        f"empty embedding text for {node.get('slug')!r} (kind={kind}) — the "
        f"core_axiom→axiom bridge likely broke; see Fable #1 / identity.py C2/M8."
    )
    return text


def populate_index(store, provider, vstore: QdrantVectorStore) -> int:
    """Embeds every live corpus node and upserts it into the test vector collection.

    Indexes BOTH kinds (active decisions + live open questions) to match the
    production sync drain — `surface_decisions` surfaces both, so a decisions-only
    index would make the MCP smoke diverge and inflate precision.

    Args:
        store: A populated `GraphStore` (from `build_reference_graph`).
        provider: A `GeminiEmbeddingProvider` (model-keyed cache path).
        vstore: A `QdrantVectorStore` bound to the throwaway test collection.

    Returns:
        The number of nodes indexed.
    """
    nodes: List[Dict[str, Any]] = []
    nodes.extend(store.get_active_decisions())
    nodes.extend(store.get_open_questions())
    for node in nodes:
        text = _derive_embedding_text(node)
        vector = provider.get_embedding(text, is_query=False)
        payload = {
            "slug": node["slug"],
            "scope": node.get("scope") or [],
            "state": node.get("state", "active"),
            "kind": node["kind"],
            "embedding_text": text,
        }
        # Pass the RAW 64-hex node id; upsert() converts it to a UUID internally.
        # `may_create=True`: this loop seeds the throwaway collection with the WHOLE
        # live corpus (both kinds, above), so it is a genuinely covering write — the one
        # class of write that may bring an absent collection into existence.
        vstore.upsert(node["id"], vector, payload, may_create=True)
    return len(nodes)


# ---------------------------------------------------------------------------
# Retrieval eval
# ---------------------------------------------------------------------------

def _ranked_slugs(vstore: QdrantVectorStore, q_vector: List[float], k: int) -> List[str]:
    """Returns the top-k result slugs, best-first, for a query vector."""
    return [m["slug"] for m in vstore.query(q_vector, limit=k)]


def run_retrieval_eval(
    oracle: Dict[str, Any], provider, vstore: QdrantVectorStore, k: int = 5
) -> Dict[str, Any]:
    """Runs every retrieval fixture and computes per-fixture + aggregate metrics.

    Args:
        oracle: The parsed `oracle.semantic.json`.
        provider: A `GeminiEmbeddingProvider`.
        vstore: A `QdrantVectorStore` bound to the populated test collection.
        k: The top-k cutoff for the metrics.

    Returns:
        A report dict: ``{provenance, k, fixtures: [...], aggregate: {...}}``. Each
        fixture carries its query, ranked slugs, metrics, and ``measure_only`` flag.
        The aggregate averages each metric over the GATING fixtures only
        (``measure_only`` fixtures are measured and reported but excluded from the
        gated aggregate + baseline diff).
    """
    fixtures_out: List[Dict[str, Any]] = []
    for fx in oracle["retrieval"]:
        q_vector = provider.get_embedding(fx["query"], is_query=True)
        ranked = _ranked_slugs(vstore, q_vector, k)
        metrics = evaluate_fixture(
            ranked, fx.get("expect_relevant", []), fx.get("expect_absent", []), k
        )
        fixtures_out.append(
            {
                "query": fx["query"],
                "measure_only": bool(fx.get("measure_only", False)),
                "expect_relevant": fx.get("expect_relevant", []),
                "expect_absent": fx.get("expect_absent", []),
                "ranked": ranked,
                "metrics": metrics,
            }
        )

    gating = [f for f in fixtures_out if not f["measure_only"]]
    metric_keys = HIGHER_IS_BETTER + LOWER_IS_BETTER
    aggregate = {
        key: (sum(f["metrics"][key] for f in gating) / len(gating)) if gating else 0.0
        for key in metric_keys
    }
    return {
        "provenance": provenance(),
        "k": k,
        "fixtures": fixtures_out,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> Optional[str]:
    """Runs a git command in the repo, returning stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=GOLDEN_DIR, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def provenance(
    judgment_model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    judgment_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Stamps a run's provenance so metrics are comparable and replayable.

    Mirrors the conflict vision's CONF-D8 telemetry discipline. The retrieval eval
    calls this with no args, leaving ``judgment_model`` / ``prompt_version`` ``None``
    (retrieval has no LLM judge). The conflict eval (§6.3) passes both — the SONNET
    alias off the live ``JudgmentExecution`` and ``CONFLICT_PROMPT_VERSION`` — so its
    report/baseline stamp the judge identity the stub was built to carry. ``dirty_tree``
    flags an uncommitted working tree, which makes ``commit_sha`` a partial lie for replay.

    Args:
        judgment_model: The conflict judge's family+tier alias (e.g. ``"SONNET"``), or
            ``None`` for a judge-free run.
        prompt_version: The conflict judgment prompt version, or ``None`` for a
            judge-free run.

    Returns:
        A provenance dict. ``timestamp`` is the one intentionally non-deterministic
        field and is excluded from the baseline diff.
    """
    status = _git("status", "--porcelain")
    return {
        "embedding_model": get_embedding_model_id(),
        "mitos_version": MITOS_VERSION,
        "commit_sha": _git("rev-parse", "HEAD"),
        "dirty_tree": bool(status) if status is not None else None,
        "judgment_model": judgment_model,
        "judgment_model_id": judgment_model_id,
        "prompt_version": prompt_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Baseline diff (soft gate)
# ---------------------------------------------------------------------------

def baseline_diff(
    report: Dict[str, Any], baseline: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compares a run's aggregate metrics to a stored baseline, flagging regressions.

    A soft gate: it returns the list of regressions (empty == clean) for the caller
    to surface for human review. It NEVER hard-fails and NEVER auto-accepts a
    semantic regression. Per-metric bands are read from the baseline file's ``bands``
    block so the tolerance is versioned alongside the numbers. Direction-aware:
    higher-is-better metrics regress on a drop; ``hard_negative_fp_rate`` on a rise.

    Args:
        report: A report dict from `run_retrieval_eval`.
        baseline: The parsed `baseline.metrics.json` (``{aggregate, bands, ...}``).

    Returns:
        A list of regression dicts, each ``{metric, baseline, current, band,
        direction}``.
    """
    regressions: List[Dict[str, Any]] = []
    base_agg = baseline.get("aggregate", {})
    bands = baseline.get("bands", {})
    current = report["aggregate"]
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        if metric not in base_agg:
            continue
        band = bands.get(metric, 0.0)
        base_val = base_agg[metric]
        cur_val = current[metric]
        if metric in HIGHER_IS_BETTER:
            regressed = cur_val < base_val - band
            direction = "drop"
        else:
            regressed = cur_val > base_val + band
            direction = "rise"
        if regressed:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": base_val,
                    "current": cur_val,
                    "band": band,
                    "direction": direction,
                }
            )
    return regressions


# ---------------------------------------------------------------------------
# Oracle / baseline / report IO
# ---------------------------------------------------------------------------

def load_semantic_oracle() -> Dict[str, Any]:
    """Loads and returns the parsed `oracle.semantic.json`."""
    with open(SEMANTIC_ORACLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_baseline() -> Optional[Dict[str, Any]]:
    """Loads `baseline.metrics.json`, or None if it has not been seeded yet."""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def write_baseline(report: Dict[str, Any], bands: Dict[str, float]) -> None:
    """Freezes a run's aggregate as the reviewed baseline (explicit-flag path only).

    Called ONLY under `MITOS_UPDATE_BASELINE=1` (including the first seed) — never
    from an ordinary test run, so a quota-degraded run can never silently become
    ground truth. The caller reviews the numbers before committing the file.

    Args:
        report: A report dict from `run_retrieval_eval`.
        bands: Per-metric regression tolerances to store alongside the numbers.
    """
    payload = {
        "provenance": report["provenance"],
        "k": report["k"],
        "aggregate": report["aggregate"],
        "bands": bands,
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_report(report: Dict[str, Any], name: str) -> str:
    """Writes a run's full metrics report as JSON to the gitignored reports dir.

    Args:
        report: A report dict from `run_retrieval_eval`.
        name: A filename stem (no extension).

    Returns:
        The path written.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def human_summary(report: Dict[str, Any]) -> str:
    """Renders a short human-readable summary of a report for the test log.

    Args:
        report: A report dict from `run_retrieval_eval`.

    Returns:
        A multi-line string: provenance header, per-fixture recall/fp, aggregate.
    """
    p = report["provenance"]
    lines = [
        f"Layer-B retrieval eval — {p['embedding_model']} @ mitos {p['mitos_version']} "
        f"({p['commit_sha']}, dirty={p['dirty_tree']})",
        f"k={report['k']}",
    ]
    for f in report["fixtures"]:
        tag = " [measure-only]" if f["measure_only"] else ""
        m = f["metrics"]
        lines.append(
            f"  recall={m['recall_at_k']:.2f} mrr={m['mrr']:.2f} "
            f"fp={m['hard_negative_fp_rate']:.2f}{tag}  «{f['query'][:60]}»"
        )
    agg = report["aggregate"]
    lines.append(
        f"AGGREGATE (gating): recall={agg['recall_at_k']:.3f} "
        f"precision={agg['precision_at_k']:.3f} mrr={agg['mrr']:.3f} "
        f"hard_neg_fp={agg['hard_negative_fp_rate']:.3f}"
    )
    return "\n".join(lines)
