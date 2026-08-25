"""Sync pipeline for Mitos.

This module implements the core V3a and V3b sync loops, managing snapshotting,
concurrency file locks, LLM capture enrichment, user reviews, and content-aware
archive rotation.
"""

import os
import sys
import shutil
import re
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple, Callable
from filelock import FileLock, Timeout
from google import genai
from google.genai import types

from mitos import __version__ as MITOS_VERSION
from mitos.config import MitosConfig, hint_due
from mitos.conflict import (
    CONFLICT_CANDIDATE_SOURCE,
    CONFLICT_PROMPT_VERSION,
    CONFLICT_TOP_K,
    ConflictCheckResult,
    ConflictFinding,
    ConflictUnavailableReason,
    SEMANTIC_SUBSTRATE_REASONS,
    Unavailable,
    candidate_payload,
    gather_candidates,
    run_conflict_check,
    screen_candidates,
)
from mitos.telemetry import (CommentaryAuditRow, ConflictCheckRow, JudgmentBatch,
                            TelemetryStore)
from mitos.divergence import declared_edges, entry_divergence, is_reconcilable
from mitos.errors import (
    CollectionMissingError,
    MitosError,
    SynthesisError,
    ParseError,
    ValidationError,
    DatabaseError,
    EntryFailure,
    CommitError,
    STORE_CYCLE_VIOLATION,
    STORE_DANGLING_EDGE,
    STORE_KIND_CONSTRAINT_VIOLATION,
    STORE_MISSING_TARGET,
    STORE_SLUG_COLLISION,
)
from mitos.models import get_embedding_model_id, get_model_id
from mitos.parser import (ParsedEntry, mask_inline_code, parse_entry_stream,
                          parse_file_reversed)
from mitos.replay import commit_quarantine_fixpoint
from mitos.store import (
    GraphStore,
    CommitDelta,
    _utc_now_iso,
    _strip_citation,
    _EDGE_KIND_REQUIREMENT,
    _KILL_EDGE_FIELDS,
    edge_kind_is_legal,
)
from mitos.identity import SLUG_MAX_LEN, compute_node_id, embedding_text
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore, hash_to_uuid
from mitos.renderer import MitosRenderer, summarize_overflows


@dataclass
class _ConflictSyncRun:
    """The per-run context the sync-time Conflict sensor threads through its checks (5b).

    The one piece of mutable per-run state the sensor needs, built ONCE per
    ``perform_sync`` at the judge-build site and passed to every
    :meth:`MitosSyncManager._run_and_surface_conflict` call of the loop. Threading it
    explicitly (rather than hiding it on ``self``) makes the write-then-read aggregate
    breaker legible: ``breaker_tripped`` is set on the entry that degrades and read at
    the top of the next entry's check, so a single downstream outage costs one penalty,
    not N (P7 active-bulkhead, vision §4). Reset is structural — a fresh run means a
    fresh instance (``breaker_tripped=False``), so a prior sync's trip can never leak
    into the next; there is no explicit un-trip step to forget.

    Attributes:
        sync_run_id: The run-correlation id minted per sync (a ``uuid4().hex``), stamped
            on every ``conflict_checks`` row so a run's judgments read as one thread
            (P16 One Thread of Truth).
        telemetry: The best-effort telemetry store, or ``None`` when it could not be
            constructed — the sensor still surfaces findings, it just can't persist them.
        breaker_tripped: Set ``True`` on the first typed degradation of the run; every
            later entry's check then short-circuits to a true no-op (no gather, no
            query, no judge).
    """

    sync_run_id: str
    telemetry: Optional[TelemetryStore]
    breaker_tripped: bool = False


def run_sync_enrichment(
    client: genai.Client,
    entry: ParsedEntry,
    active_decisions: List[Dict[str, Any]],
    *,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """Calls Gemini to refine a decision, infer scopes, and suggest relationships.

    NOTE: no longer invoked by ``mitos sync`` — the sync runtime path is strict-
    deterministic and commits the human-authored buffer verbatim (ADR
    ``sync-strict-deterministic-no-llm-enrichment``). LLM refinement of genuinely-raw
    input lives in ``capture`` / ``import --llm-extract``. This function is retained
    only as the live test-suite's generative-quota probe target
    (``tests/live_helpers.py``); it is dead in the production path and a candidate for
    removal once that probe is repointed to a still-live generative call.

    Args:
        client: The Gemini client the caller constructed.
        entry: The parsed entry to enrich.
        active_decisions: The active decisions to summarise into the prompt.
        model_id: The resolved id for ``FLASH_LITE``, taken off the calling
            workspace's ``config.env``, or ``None`` for the baseline (2c).
    """
    active_summary = ""
    for d in active_decisions[:20]:  # Limit to top 20 active decisions for prompt budget
        active_summary += f"- slug: {d['slug']}\n  axiom: {d['core_axiom']}\n  scope: {','.join(d['scope'])}\n\n"

    prompt = f"""
You are the Mitos v0.1 capture enrichment agent. Your task is to refine and enrich a newly captured architectural decision.

Here are some currently ACTIVE decisions in the workspace:
{active_summary}

Here is the proposed decision entry:
Slug: {entry.slug}
Decided: {entry.axiom}
Rejected: {entry.rejected_paths}
Mechanisms: {','.join(entry.mechanisms)}
Scope: {','.join(entry.scope)}
Context: {entry.context}

Please enrich this entry. You must:
1. Verify and refine the `core_axiom` into a single, extremely precise, clear, and unambiguous sentence. If the raw axiom is already high quality, keep it verbatim or make minimal grammatical corrections.
2. Verify and refine the list of mechanism tags.
3. Suggest appropriate scope tags based on the content and existing active decisions.
4. Detect if this decision should supersede, amend, narrow, or depend on any of the active decisions listed above. If so, return their slugs in the suggested relationships.

Respond strictly in valid JSON format with the following keys:
- refined_core_axiom (string)
- refined_mechanisms (list of strings)
- refined_scope (list of strings)
- suggested_relationships (object with keys: supersedes, amends, narrows, depends_on, resolves)
"""
    resolved_model = (
        model_id if model_id is not None else get_model_id("FLASH_LITE")
    )
    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise SynthesisError(f"LLM enrichment call failed: {str(e)}")


def run_ambient_capture(
    client: genai.Client, raw_text: str, *, model_id: Optional[str] = None
) -> str:
    """Uses FLASH to convert raw conversational text into a canonical Markdown entry.

    Args:
        client: The Gemini client the caller constructed.
        raw_text: The developer's raw conversational input.
        model_id: The resolved id for ``FLASH``, taken off the calling
            workspace's ``config.env``, or ``None`` for the baseline (2c).
    """
    prompt = f"""
You are the Mitos v0.1 capture scribe. Convert the following developer conversation or thought into a canonical Mitos Decision Entry.

Input text:
"{raw_text}"

Please generate a canonical Markdown entry. Use exactly this format (do not include markdown block quotes):

### [slug]

**Decided:** [Single-sentence axiom that is true going forward]
**Rejected:**
- [alternative] — [specific reason why it was rejected, be precise and adversarial]
**Mechanisms:** [comma-separated mechanisms, or none]
**Scope:** [comma-separated scope tags, or none]
**Context:** [brief background context explaining why this decision was made]

[DECISION_TRANSCRIPT]
User: {raw_text}
[/DECISION_TRANSCRIPT]

Make sure the slug is a clean, lowercase hyphenated string that matches the decision topic.
"""
    resolved_model = model_id if model_id is not None else get_model_id("FLASH")
    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )
        return response.text.strip()
    except Exception as e:
        raise SynthesisError(f"Ambient capture synthesis failed: {str(e)}")


# --- record_decision helpers (write-half of the MCP server) ---

# The exact buffer marker, byte-for-byte identical to cmd_capture (cli.py). The
# `—` is an em dash; do not retype it.
_ENTRIES_MARKER = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"

# A content line that looks like a Mitos field header (e.g. `**Decided:**`); the
# parser would treat it as a new field and corrupt the entry (parser.py:400-409).
_FIELD_LINE_RE = re.compile(r'^\s*\*\*[A-Za-z -]+:\*\*')

# A column-0 H2/H3 heading opens a NEW entry in the parser (parser.py:308). Note
# this is deliberately narrow: a single `#` (H1), `####`+ headings, and any
# *indented* heading are all SAFE and must NOT be rejected.
_SECTION_HEADER_RE = re.compile(r'^#{2,3}(?!#)')

# Inline markers the parser reacts to: section/transcript/buffer boundaries and
# the [NOTE:]/[PARKED:] scanners that siphon content into entry.notes (parser.py:419-426).
_STRUCTURAL_MARKERS = (
    "[DECISION_PARKED:",
    "[DECISION_TRANSCRIPT]",
    "[/DECISION_TRANSCRIPT]",
    "BEGIN ENTRIES",
    "[NOTE:",
    "[PARKED:",
)

# The exact agent-facing error messages (spec §5). Each says what went wrong AND
# how to recover, interpolating the offending value.
_ERROR_MESSAGES: Dict[str, str] = {
    "not_initialized": "No Mitos workspace found here. Run 'mitos init' before recording decisions.",
    "empty_axiom": "'axiom' is empty. Provide the decision as a single clear sentence that is true going forward.",
    "empty_slug": "'slug' is empty. Provide a short, explicit, hyphenated handle (e.g. 'sqlite-wal-mode').",
    "slug_too_long": "slug '{slug}' is {length} characters — {over} over the {max}-character limit. The slug is the permanent citation handle (it is folded into the decision's identity), so it is NOT silently truncated. Pass a shorter 'slug' of at most {max} characters.",
    "missing_rejected_paths": "'rejected_paths' is required: state the alternatives you considered and why you ruled them out — this is what stops you or another agent from re-proposing them later.",
    "parse_failed": "The decision could not be serialised into a valid entry — most likely a structural token in axiom/rejected_paths/context: a line beginning with '##' or '###' (indent it or use '#'/'####' instead), a line shaped like '**Something:**', or a '[DECISION_TRANSCRIPT]' / '[DECISION_PARKED:' / 'BEGIN ENTRIES' / '[NOTE:' / '[PARKED:' marker. Remove or rephrase that line and retry — or, to mention a marker as prose, wrap it in backticks (`...`): inline-code spans are exempt.",
    "slug_collision": "A different decision already uses the slug '{slug}'. Give this one a distinct 'slug'; and if it is meant to replace the existing decision, also set supersedes='{slug}' (the new decision must still have its own slug — two decisions cannot share one).",
    "supersedes_not_found": "supersedes='{supersedes}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'supersedes' if this is a brand-new decision.",
    "supersedes_ambiguous": "supersedes='{supersedes}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "corrects_not_found": "corrects='{corrects}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'corrects' if this is a brand-new decision.",
    "corrects_ambiguous": "corrects='{corrects}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "relation_target_not_found": "{relation}='{target}' does not match any existing decision. Look it up first with surface_decisions/query_decisions to get the exact slug, or omit '{relation}' if no such link applies.",
    "relation_target_ambiguous": "{relation}='{target}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "commit_failed": "The decision validated but the commit failed and nothing was written: {reason}. Retry; if it persists, the workspace store may be locked or corrupt.",
    "derives_from_on_decision": "derives_from is not valid when recording a decision: a derives_from edge originates from an open question (open_question -> decision), so a decision can never be its source. If you mean 'this decision builds on that one', use cites instead.",
}

# The user-facing typed relations beyond `supersedes` (which is special: it changes
# computed state and has its own error codes). Each maps an agent-facing kwarg name to
# its canonical decisions.md field label. The parser and the store's commit path
# already understand all of these (format-spec.md §"Relationship Fields"); the agentic
# write path just had to serialize + validate them the way it already does supersedes.
_EXTRA_RELATIONS = (
    ("amends", "Amends"),
    ("narrows", "Narrows"),
    ("depends_on", "Depends-On"),
    ("resolves", "Resolves"),
    ("contradicts", "Contradicts"),
    ("derives_from", "Derives-From"),
    ("cites", "Cites"),
)


def _split_relation_slugs(raw: Optional[str]) -> List[str]:
    """Split a comma-separated relation argument into a list of stripped slugs.

    The decisions.md relationship fields are comma-separated multi-valued (V1b,
    format-spec.md "Relationship Fields"), and the markdown parser already tokenizes
    them so; this mirrors that split for the CLI/MCP ``record`` edge args, so a single
    ``--supersedes "a, b, c"`` (or ``supersedes="a, b, c"`` on the MCP twin) commits one
    edge per slug. A lone slug is a 1-element list, so single-valued authoring stays
    byte-for-byte unchanged — purely additive, no caller breaks.

    Args:
        raw: The relation argument as the caller passed it — ``None`` or a possibly
            comma-separated string of exact slugs.

    Returns:
        The stripped, non-empty slugs in author order (``[]`` when ``raw`` is falsy).
    """
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _record_error(code: str, **fields: Any) -> Dict[str, str]:
    """Builds a structured {error, code} dict using the canonical message for ``code``."""
    return {"error": _ERROR_MESSAGES[code].format(**fields), "code": code}


def _embedding_input_text(
    kind: str,
    axiom: Optional[str] = None,
    topic: Optional[str] = None,
    questions_raised: Optional[List[str]] = None,
) -> str:
    """Derives the V1a embedding-input string for an entry or node (C2/M8 single source).

    Routes through :func:`identity.embedding_text` so the record-time and drain-time
    embedding text are byte-identical — a node embedded inline at record time and the
    same node re-derived from the Outbox at drain time yield the same vector (the
    ``embedding_text`` column is gone in V1a, so drain re-derives from the immutable
    core, M8). Bridges the one reader-key gap: a decision's axiom is exposed as
    ``axiom`` on a :class:`ParsedEntry` but ``core_axiom`` on a store node dict —
    callers pass whichever they hold under ``axiom``.

    Args:
        kind: ``"decision"`` or ``"open_question"``.
        axiom: The decision axiom (for a decision entry/node).
        topic: The open_question topic.
        questions_raised: The open_question's questions, in authored order.

    Returns:
        The embedding-input string (normalized, M8-consistent with the hashed core).
    """
    return embedding_text({
        "kind": kind,
        "axiom": axiom,
        "topic": topic,
        "questions_raised": questions_raised or [],
    })


def _contains_structural_token(text: str) -> bool:
    """Returns True if any line of ``text`` would corrupt the flat-file parser.

    Matches the parser's own raw-line semantics (no leading-whitespace strip for
    headers, parser.py:308): only column-0 ``##``/``###`` headings, field-shaped
    lines, and the inline markers trigger.
    """
    for line in text.split("\n"):
        if _SECTION_HEADER_RE.match(line):
            return True
        if _FIELD_LINE_RE.match(line):
            return True
        # Code-span-masked, matching the parser's own scanners: a backtick-quoted
        # marker (`BEGIN ENTRIES`, `[NOTE: …]`) is prose about the marker, and the
        # parser no longer reacts to it — so the guard must not reject it either.
        masked = mask_inline_code(line)
        if any(marker in masked for marker in _STRUCTURAL_MARKERS):
            return True
    return False


# The citation-handle length cap. The slug is folded into the canonical-core identity
# (V1-D2), so it is permanent once committed — generous enough for a descriptive
# multi-word handle (real corpus handles top out ~68 chars), firm enough to keep one
# from running away. An *explicit* slug over this is REJECTED with an exact char count
# (never silently truncated — a silent trim diverges the stored handle from the one the
# author already cited: self-inflicted citation rot). Auto-derived slugs still trim to it.
# NOTE: this cap is advertised up-front to callers — the CLI `--slug` help imports this
# constant; the MCP `record_decision` slug docstring carries the number as a literal. If
# you change it, update that docstring (mcp_server.py) too.
_SLUG_MAX_LEN = SLUG_MAX_LEN  # shared source of truth in identity.py (re-exported here for cli.py + the record path)
_SLUG_MIN_LEN = 32  # auto-derive only: don't trim a word boundary back past here — hard-cap instead

# Returned on the two "exists" short-circuits, which write nothing at all. The
# no-op is deliberate (a committed canonical core is immutable, M1) — what was
# NOT deliberate is that the receipt used to report the buffer path, so a
# re-record aimed at correcting commentary, or at restoring a source block for a
# graph-only node, reported success while changing nothing. The note names the
# path that does work; `mitos sync` does not, since it re-commits nothing for a
# node already in the graph.
def _edge_state_labels(pairs: List[Dict[str, str]]) -> List[str]:
    """Renders edges as a sorted ``["kind:target", ...]`` state list for the audit row.

    Args:
        pairs: Edge dicts carrying ``kind`` and ``target``.

    Returns:
        The sorted labels, deduplicated.
    """
    return sorted({f"{p['kind']}:{p['target'].strip().casefold()}" for p in pairs
                   if p.get("target")})


_EXISTS_NO_OP_NOTE = (
    "already recorded — this call wrote nothing, to the buffer or the graph. "
    "A committed decision's axiom and mechanisms are immutable (M1): re-recording "
    "them is a no-op. To correct its commentary (rejected_paths, scope, "
    "invalidates_if, context), edit the entry in decisions.md and run `mitos sync` — "
    "it reconciles a diverged committed entry, printing the field diff first. "
    "Two states sync cannot reach: a node with no `### ` block (run "
    "`mitos restore-source --slug <slug>` first), and an entry already rotated into "
    "decisions/archive/ — sync reads only the buffer, so that one's reconciler is "
    "`mitos rebuild`, which can refuse the swap and which re-mints confirmation "
    "metadata, so correct it while it is still in the buffer if you can. "
    "To record a CHANGED decision, write a new one with --supersedes/--amends "
    "pointing at this slug."
)

# The standing coherence debt a successful write incurs, stated on every `created`
# receipt. The REGISTER is the mechanism, not the presence: `mitos check` reuses
# prior verdicts (a reused pair never enters a batch), so one deferred run after N
# writes covers the same ground as N runs for less — and `_confirm_spend` only fires
# above CHECK_CONFIRM_BATCHES fresh groups, so a caller auditing per write presents
# ~1 fresh group forever and the tree's only spend ring never fires. A line reading
# "audit this write" therefore converts one owed run into N and fragments the
# amortization (ADR `record-receipt-states-cumulative-audit-debt-not-per-write-work`).
# So: cumulative and corpus-wide, no imperative, no per-entry referent — and no
# command, because this string is an MCP-visible payload field, not a CLI line (ADR
# `receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer`). The
# recovery clause is each renderer's own; `cli._coherence_audit_hint` composes the
# CLI's. Enforced by test rather than by emphasis (tests/test_record_decision.py).
_COHERENCE_AUDIT_NOTE = (
    "Coherence audit is cumulative and corpus-wide: this corpus holds recorded "
    "decisions that no contradiction check has covered yet."
)

# A new decision at/above this document-document similarity to an existing one the
# author did NOT reference is paused for review (AX P4): the neighbour must surface
# BEFORE commit, while the author can still point an amends/supersedes/contradicts
# at it (you can't relink after commit — re-record is a no-op; the retired
# post-commit `related` echo was exactly one step too late).
# 0.80 is the floor of the strong-match band (see recall.py's calibration):
# an unlinked strong match must force an explicit confirm-or-link, because prose-only
# obsoletion is invisible to state-filtered retrieval — only edges retire a node (ADR
# `record-pause-floor-lowered-to-strong-match-band`; the routine-case pause tax stays
# low via --acknowledge-neighbors, the declared-target exemption, and transitive
# lineage suppression). Tune here.
_NEIGHBOR_REVIEW_THRESHOLD = 0.80

# The relations the pause offers for resolving a flagged neighbour — a fourth
# edge-field set beside store.py's kill/deferred/mutation trio, derivable from none
# of them (those are graph semantics; this is authoring guidance). Every surface
# that teaches the recovery renders from here: sync's needs_review message and the
# CLI's pause render join it directly, and the MCP record_decision docstring is
# pinned to it by test (a docstring can't render dynamically — @mcp.tool captures
# it at decoration). Three hand-kept copies let `narrows` go missing across three
# review rounds; one named set is the fix.
#
# Omissions are decisions, not gaps:
#   resolves     — decision→open_question only, and the pause gather keeps only
#                  live decisions, so a paused neighbour is structurally never an
#                  open question.
#   derives_from — originates from an open question; invalid on record and
#                  early-rejected before the pause can matter.
#   depends_on   — a ≥0.80 near-restatement is not a dependency; the "builds on"
#                  reading of a near-twin is what `cites` is for.
# corrects is included deliberately: it suppresses like the rest, and steering an
# author whose near-twin was WRONG (not outgrown) to `supersedes` stamps the
# target superseded_by instead of corrected_by on every future read — the same
# durable mislabel that let the missing `narrows` coerce false `amends` edges.
_PAUSE_RESOLVING_RELATIONS = (
    "amends", "narrows", "supersedes", "corrects", "contradicts", "cites",
)

# How many declared targets each pause-echo group renders before it collapses to a
# slice plus a sibling count (A2). Applied per group, independently.
#
# The bound's FIRST job is safety, not readability. The count is an exception marker,
# so a bound low enough to fire on an ordinary declaration would reconstruct the
# aggregate claim the echo is designed NOT to make ("N of your edges resolved a
# neighbour") on exactly the path meant to stay quiet. Set it where a collapse is
# pathological: the largest real declaration measured in this corpus is four, an
# ordinary six-slug declaration must not collapse, and forty repeated `--cites` still
# does. `routing.REGISTERED_NAMES_BOUND = 10` and `recall.SURFACE_TOP_SCOPES = 5`
# transfer their SHAPE (count beside slice, shown only on collapse) and deliberately
# not their magnitude — both are small because for those surfaces a collapse is the
# designed common case, which is the opposite of this one.
_DECLARED_ECHO_BOUND = 20

# Surface wording for the record receipt's degraded-check causes. The record surface
# owns this text (the core returns only the typed reason — the same core/surface split
# as _notice_conflict_unavailable, the sync-surface sibling). ``Unavailable.detail`` is
# logging-only by contract and never lands in the sentence.
_REVIEW_UNAVAILABLE_CAUSES = {
    ConflictUnavailableReason.EMBEDDING: "embedding service unavailable",
    ConflictUnavailableReason.VECTOR_STORE: "vector store unavailable",
    ConflictUnavailableReason.COLLECTION_MISSING: (
        "the vector collection is missing — run `mitos reconcile`"
    ),
}

# What the reconcile pre-flight (``_uncommittable_edges``) does about each store
# failure code — the whole class, declared, so none can be silently missed.
#
# Phase 3 shipped the pre-flight against `missing_target` alone and called it done;
# `kind_constraint_violation` then sailed through and wrote two audit rows per sync
# forever. The lesson was not "add one more check" but "the class must be enumerated
# somewhere a test can read." ``test_preflight_covers_store_codes`` asserts this table
# has an entry for every member of ``STORE_FAILURE_CODES``, so adding a code to the
# store forces a deliberate decision here rather than a silent gap.
#
# A code is CAUGHT (refused before the intent row) or NOT_CAUGHT with a reason.
# "Not caught" is a real position, not a TODO: the commit still rejects it correctly
# and the correlated outcome row still records the failure — the pre-flight's only job
# is to stop an UNBOUNDED trail behind a defect that repeats every sync.
_PREFLIGHT_CAUGHT = "caught"
_PREFLIGHT_DISPOSITIONS: Dict[str, str] = {
    STORE_MISSING_TARGET: _PREFLIGHT_CAUGHT,
    STORE_DANGLING_EDGE: _PREFLIGHT_CAUGHT,  # same probe: the citation resolves to nothing
    STORE_KIND_CONSTRAINT_VIOLATION: _PREFLIGHT_CAUGHT,
    STORE_SLUG_COLLISION: (
        "unreachable from a reconcile — the branch is entered only on a canonical-core "
        "hash MATCH against an existing node, so the committing node already owns the "
        "slug and cannot collide with itself"
    ),
    STORE_CYCLE_VIOLATION: (
        "not pre-flighted deliberately — deciding it needs the store's mutation-lineage "
        "walk, and re-implementing that in a pre-flight is exactly the duplication that "
        "drifts. It is also self-limiting rather than unbounded in practice: it needs a "
        "hand-edit that adds a kill/mutation edge closing a cycle, and the printed "
        "per-entry failure names it. Revisit if one is ever observed repeating."
    ),
}


def _review_unavailable_notice(cause: str) -> str:
    """One calm receipt sentence for a near-dup check that could not run (fail-open).

    Structure: name the cause and state that the decision committed without the
    check. "Couldn't check" must never read as "checked, clean".

    It names **no command**, and that is the boundary rule rather than a stylistic
    trim: this sentence is returned into ``result["neighbor_review_unavailable"]``,
    which ``record --json`` emits verbatim and MCP ``record_decision`` returns — so
    a selectored ``mitos check`` here would put a shell command carrying a CLI flag
    onto an agent's response. The recovery is composed per renderer instead, and on
    the CLI it rides the *unconditional* coherence line one paragraph below (which
    lands on this same ``created`` exit by construction), so the receipt names the
    verb exactly once rather than twice in two registers. ADRs
    ``receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer`` and
    ``created-receipt-names-its-recovery-once-on-the-unconditional-line``.

    One cause spelling is the deliberate exception: ``COLLECTION_MISSING`` keeps its
    ``mitos reconcile`` pointer in :data:`_REVIEW_UNAVAILABLE_CAUSES`, a separate
    string this function does not compose — a local re-embed rather than a judged
    audit, and test-pinned because the unmapped fallback cannot produce it.

    Args:
        cause: Short prose naming what failed (e.g. "embedding service unavailable").

    Returns:
        The receipt notice sentence.
    """
    return (
        f"Near-duplicate review could not run ({cause}); this decision committed "
        "without a neighbour check."
    )


def _declared_echo(
    declared_by_relation: Dict[str, Optional[str]],
    canonical_slugs: Dict[str, str],
    gathered_index: Dict[str, Tuple[str, float]],
    floor: float,
) -> Dict[str, Any]:
    """Partitions the caller's own declared targets for the pause body (A2).

    The predicate, stated once here so no reader re-derives it from three call sites.
    Let ``V`` be the targets the caller typed across all nine relation flags, ``G``
    the slugs this call's KNN sweep gathered, and ``PR``
    :data:`_PAUSE_RESOLVING_RELATIONS`:

    * **Group two** (``declared_no_near_match``) — targets in ``V`` declared through at
      least one ``PR`` relation that fall outside ``V ∩ G ∩ (score >= floor)``.
    * **Group one** (``declared``) — the complement over ``V``, keyed on all nine
      relations.

    Four properties are contract, not implementation taste:

    * The partition is over **targets**, not flag occurrences — a target declared
      through several relations appears once, and lands in group two if *any* of those
      relations is pause-resolving. So an ordinary cross-domain ``depends_on`` is
      acknowledged in group one and can never be reported as having moved nothing.
    * The inputs are the **primary sets** — the declarations, the gathered candidates
      and the floor (M8). Never ``screen_candidates``' filtered return: its S4 stage
      drops a declared target *before* the floor, so a declaration that resolved a
      strong neighbour and one that did nothing are both absent from the survivors,
      for unrelated reasons.
    * Only the caller's own half of ``declared_targets`` is echoed. The transitive
      lineage ancestors merged in for suppression are in **neither** group — the
      discriminator is *did the caller type it*, never *is it also an ancestor*.
    * An empty group renders **no key at all**, never an empty list: group one is
      genuinely reachable-empty (a lone distant ``cites``), and a ``declared: []``
      beside a populated group two would deny a declaration the call did receive.

    Group two carries ``score`` only when the target *was* gathered and fell below the
    floor; a target the sweep never saw has no candidate to read one off and renders
    scoreless. The absent key — rather than a ``null`` — is what carries that shape.

    Args:
        declared_by_relation: Relation name -> the caller's raw, unsplit argument, in
            the fixed render order (``supersedes``, ``corrects``, then
            ``_EXTRA_RELATIONS``' own order). Values may be ``None``.
        canonical_slugs: Casefolded declared target -> its stored slug, retained by the
            Phase-A validation loops. The echo prints the stored spelling for every
            target, gathered or not (a caller spelling and the casefold are both wrong
            here — ``_normalize_slug`` runs on the record path alone, so the stored
            handle genuinely differs from both).
        gathered_index: Casefolded gathered slug -> ``(stored slug, score)`` for every
            candidate this call's sweep returned, pre-screen.
        floor: The similarity floor the screen applied (``_NEIGHBOR_REVIEW_THRESHOLD``).

    Returns:
        The echo's keys, ready to merge into the pause dict — any of ``declared``,
        ``declared_total``, ``declared_no_near_match``,
        ``declared_no_near_match_total``. Empty when the caller declared nothing.
    """
    order: List[Tuple[str, str]] = []       # (casefolded, verbatim), first-seen order
    seen: set = set()
    resolving: set = set()
    for relation, raw in declared_by_relation.items():
        for target in _split_relation_slugs(raw):
            folded = target.casefold()
            if folded not in seen:
                seen.add(folded)
                order.append((folded, target))
            if relation in _PAUSE_RESOLVING_RELATIONS:
                resolving.add(folded)

    group_one: List[str] = []
    group_two: List[Dict[str, Any]] = []
    for folded, verbatim in order:
        # The stored spelling, from the validation loops. Every declared target
        # resolved there before the pause could compose, so the fallback is
        # unreachable — it exists so a future edit cannot turn the echo into a crash
        # on the write path.
        handle = canonical_slugs.get(folded, verbatim)
        gathered = gathered_index.get(folded)
        if folded in resolving and not (gathered is not None and gathered[1] >= floor):
            item: Dict[str, Any] = {"slug": handle}
            if gathered is not None:
                item["score"] = gathered[1]
            group_two.append(item)
        else:
            group_one.append(handle)

    echo: Dict[str, Any] = {}
    for key, items in (("declared", group_one),
                       ("declared_no_near_match", group_two)):
        if not items:
            continue
        if len(items) > _DECLARED_ECHO_BOUND:
            # The elision takes the TAIL: a score-keyed truncation is undefined over
            # group two's common member (the scoreless one) and, defaulted to zero,
            # would elide exactly the declarations carrying no other signal.
            echo[key] = items[:_DECLARED_ECHO_BOUND]
            echo[f"{key}_total"] = len(items)
        else:
            echo[key] = items
    return echo


def _declared_echo_lines(payload: Dict[str, Any]) -> List[str]:
    """The echo's prose, composed from the pause payload's own keys.

    Both prose renderers read this — sync's ``message`` and ``cmd_record``'s composed
    body — so what a human reads and what the two machine encodings carry cannot
    drift: the sentences are a render of the keys, not a second computation beside
    them. Returns ``[]`` when the payload carries no echo, which is the whole render
    for a call that declared nothing.

    Both lines report in the **declared** register and say nothing about whether an
    edge committed, in either direction. The pause returns above Phase B, so nothing
    was written; a first group reading as a receipt of landed edges would invite
    dropping exactly those flags from the re-record the pause exists to force. The
    count renders only on a collapse — rendered unconditionally it is the aggregate
    claim this surface is designed not to make.

    Args:
        payload: The pause dict (or the echo keys alone).

    Returns:
        Zero, one or two sentences without trailing punctuation, group one first.
    """
    lines: List[str] = []

    declared = payload.get("declared")
    if declared:
        line = "Declared: " + ", ".join(declared)
        total = payload.get("declared_total")
        if total is not None:
            line += f" ({total} total)"
        lines.append(line)

    no_match = payload.get("declared_no_near_match")
    if no_match:
        rendered = []
        for item in no_match:
            score = item.get("score")
            rendered.append(item["slug"] if score is None
                            else f"{item['slug']} ({score:.2f}, below floor)")
        line = "Declared, not a near neighbour here: " + ", ".join(rendered)
        total = payload.get("declared_no_near_match_total")
        if total is not None:
            line += f" ({total} total)"
        lines.append(line)

    return lines


def _normalize_slug(text: str) -> str:
    """Lowercases and hyphenates free text into slug characters — with no length cap.

    The character-normalisation half of :func:`_slugify`, factored out so the write
    path can validate an explicit slug's *length* without silently truncating it: an
    over-length explicit slug is the author's permanent citation handle, so the right
    move is to reject (and ask for a shorter one), not to mangle it down to fit.

    Args:
        text: Free text — an explicit slug, or an axiom for auto-derivation.

    Returns:
        The lowercase-hyphenated form, stripped of leading/trailing hyphens (``""``
        for empty/whitespace-only input). Uncapped.
    """
    if not text:
        return ""
    s = re.sub(r'[^a-z0-9]+', '-', text.lower())
    return re.sub(r'-+', '-', s).strip('-')


def _slugify(text: str) -> str:
    """Derives a deterministic, lowercase-hyphenated slug from free text, capped in length.

    Determinism keeps the human-readable handle stable: the slug is NOT part of the
    node id (V1a identity is the slug-free canonical core — ``compute_node_id``), but
    a stable auto-derived slug means the same decision presents the same handle and
    the casefold slug-collision check (V1-D4) behaves predictably.

    This is the auto-derive path: when the slug exceeds the length cap it is trimmed
    back to the last word boundary (hyphen) rather than sliced mid-word, so the handle
    stays readable (``…brazilian-portuguese``, not ``…brazilian-portug``). Still a pure
    function of the text, so determinism holds. The agentic write path does NOT trim an
    *explicit* slug — it validates length via :func:`_normalize_slug` and rejects an
    over-length one (see ``slug_too_long``).
    """
    s = _normalize_slug(text)
    if len(s) > _SLUG_MAX_LEN:
        cut = s[:_SLUG_MAX_LEN]
        boundary = cut.rfind('-')
        # Trim to the last whole word, unless that would gut the slug (one very
        # long leading token) — then fall back to the hard cap.
        if boundary >= _SLUG_MIN_LEN:
            cut = cut[:boundary]
        s = cut.rstrip('-')
    return s


# The three outcomes that SATISFY a `--reconcile-entry` target. Every other
# end-state the per-entry loop can reach is a shortfall by default, and that
# direction is the decision rather than the spelling: the loop ends an entry in
# more states than any table names, and the two directions are not symmetric. A
# failure state left unstamped exits 0 on a repair that never landed — the silent
# no-op this whole flag exists to remove — while a satisfied state left unstamped
# exits non-zero, loudly, and is caught by the re-run row. Enumerating the closed
# set is also the only direction that survives a control-flow change in a later
# vision.
_REPAIR_SATISFIED = frozenset({"reconciled", "committed", "clean"})

# The located cause each shortfall outcome renders as. Keys are the outcome tokens
# `_perform_sync_internal` stamps at its own `continue`s; an entry that reached
# none of them (a site added later and not stamped) falls through to the generic
# line below, which is a shortfall too.
_REPAIR_CAUSES = {
    "open_question": (
        "this entry is an open question, and the commentary reconcile is "
        "decisions-only — a hand-edit to it can never be applied by `sync`"
    ),
    "source_only": (
        "its only divergence is the `**Source:**` line, which a reconcile "
        "provably cannot change — restore the line to match the graph"
    ),
    "reconcile_refused": (
        "the commentary reconcile refused it; its reason is printed above"
    ),
    "collision": (
        "its slug already names another node and the entry declares no relation "
        "at it, so it was skipped before any reconcile — the collision report "
        "above says what to author"
    ),
    "pending_skipped": (
        "it is still pending, and a run with no terminal and no `--yes` skips a "
        "pending entry before it can be committed. This flag authorizes the "
        "reconcile, not the accept prompt"
    ),
    "operator_skipped": "it was skipped at the accept prompt",
    "operator_quit": "the run was stopped at its accept prompt",
    "quarantined": (
        "its commit was refused and the retry never drained it; the report above "
        "names why"
    ),
    "store_error": (
        "an unexpected store error stopped its commit; the report above names it"
    ),
}

#: The two causes that stand in for the never-seen class's absence claim when the
#: run stopped reading the buffer. Neither is a new state: the target is still a
#: shortfall and the exit is still non-zero — what changes is that a run holding no
#: evidence of absence does not assert one, and does not name `mitos rebuild` as the
#: heal for an entry that may be sitting in the buffer it stopped reading.
_REPAIR_UNREAD_LOCKED = (
    "another Mitos process holds the corpus lock, so this run never read the "
    "buffer — re-run once it is free"
)
_REPAIR_UNREAD_QUIT = (
    "the run was stopped at its accept prompt before this entry was reached, so "
    "nothing was ruled on it — re-run to reach it"
)

_REPAIR_GENERIC_CAUSE = (
    "it did not end in any of the states this flag can satisfy; the run's own "
    "report above says what happened to it"
)


class _RepairLedger:
    """Observes what the sync loop did with each entry `--reconcile-entry` named.

    Constructed once per run and inert when no targets were named, so every `note`
    is a no-op on the ordinary path. It holds no policy of its own: the loop stamps
    what it did, and the ledger renders the shortfall afterwards. Deliberately
    module-private and un-generalized — nothing else consumes it, and there is no
    second consumer to extract a shared refusal renderer for.

    Attributes:
        _handles: casefold handle → the caller's verbatim spelling. Verbatim is
            what the report renders, because the caller has to recognise what they
            typed; casefold is what matches, because every other slug lookup in
            the tree folds on both sides.
    """

    def __init__(self, targets: Optional[List[str]] = None) -> None:
        self._handles: Dict[str, str] = {}
        for raw in targets or []:
            self._handles.setdefault(raw.casefold(), raw)
        self._outcomes: Dict[str, str] = {}
        self._order: List[str] = []
        self._failure_slugs: Set[str] = set()
        self._unattributable_failures = False
        self._unread_reason: Optional[str] = None
        self._inert = False

    def authorizes(self, entry: ParsedEntry) -> bool:
        """Whether this entry was named, casefold-exactly, by the caller.

        One tier, no alias/prefix/did-you-mean resolution — the discipline every
        other slug lookup uses. Neither side is ``_normalize_slug``'d: that helper
        runs on the ``record`` path alone, and the parser keeps a header slug
        verbatim.
        """
        slug = getattr(entry, "slug", None)
        return bool(slug) and slug.casefold() in self._handles

    def note(self, entry: ParsedEntry, outcome: str) -> None:
        """Records the loop's own verdict for a named entry. A no-op otherwise.

        Later notes win, deliberately: the quarantine stamps a shortfall at the
        commit and the fixpoint upgrades the entries it drains, so the pessimistic
        stamp is the one that survives a missed correction.
        """
        if not self.authorizes(entry):
            return
        key = entry.slug.casefold()
        if key not in self._outcomes:
            self._order.append(key)
        self._outcomes[key] = outcome

    def note_parse_failures(self, failures: List[EntryFailure]) -> None:
        """Records this run's parse failures, for the never-seen classification."""
        if not self._handles:
            return
        for fail in failures:
            if fail.slug:
                self._failure_slugs.add(fail.slug.casefold())
            else:
                # A pre-header failure carries no slug, so it can be reported but
                # not attributed to a handle.
                self._unattributable_failures = True

    def mark_buffer_unread(self, reason: str) -> None:
        """Records that the run did not read the whole buffer, and why.

        Two sites reach it: a lock held by another process (nothing was read at
        all) and ``[q]uit`` at an accept prompt (everything below that entry went
        unread). It changes no verdict and no exit — an unreached handle is still a
        shortfall — but it changes what the report may CLAIM about one. "absent
        from the buffer, or already rotated into an archive" is an absence claim
        whose heal is `mitos rebuild`, and a run that stopped reading has no
        evidence for either half.

        Args:
            reason: The located cause, rendered in place of the absence claim.
        """
        self._unread_reason = reason

    def mark_keyless(self) -> None:
        """Marks the run as below the key floor — the report's one carve-out.

        ``mitos sync``'s ``GEMINI_API_KEY`` refusal returns above the per-entry
        loop, so on a keyless workspace the flag is inert: nothing was reconciled
        and nothing was looked for. Fail-loud is scoped to runs that clear the
        floor, and flipping that exit code is a separate contract break owned by
        a pass that deprecates it.
        """
        self._inert = True

    def report(self) -> List[str]:
        """Prints one line per named target that needs one; returns the shortfall.

        The channel is stderr and is not overridable: the block accompanies a
        non-zero exit on its common path, and the tree's non-zero-exit refusals
        answer there. One channel for the whole report, exit-0 line included.

        Returns:
            The caller's verbatim spellings for every target that did NOT end in
            the state its markdown describes. Empty means nothing was named, the
            run was below the key floor, or every named target is satisfied.
        """
        if self._inert or not self._handles:
            return []

        # Document order — the loop's, which is `decisions.md`'s, never the order
        # the caller named them. Handles the loop never reached have no position of
        # their own, so they follow in naming order.
        ordered = self._order + [k for k in self._handles if k not in self._outcomes]

        lines: List[str] = []
        shortfall: List[str] = []
        for key in ordered:
            handle = self._handles[key]
            outcome = self._outcomes.get(key)
            if outcome in _REPAIR_SATISFIED:
                if outcome == "clean":
                    # The one satisfied state that owes a line: nothing else in the
                    # run said anything about this entry, so without it a caller
                    # reads silence as "not applied".
                    lines.append(
                        f"[Repair] {handle!r} — nothing to reconcile: the corpus "
                        f"and graph already agree for this entry."
                    )
                continue
            shortfall.append(handle)
            lines.append(f"[Repair] {handle!r} — {self._cause(key, outcome)}.")

        if lines:
            # stdout is block-buffered under a pipe while stderr is not, so without
            # this flush the block lands above the run's own report and inverts the
            # reading. One channel and one flush for the whole block, so it stays
            # together and accompanies the non-zero exit on its common path.
            sys.stdout.flush()
            for line in lines:
                print(line, file=sys.stderr)
        return shortfall

    def _cause(self, key: str, outcome: Optional[str]) -> str:
        """Renders the located cause for one unsatisfied handle."""
        if outcome is not None:
            return _REPAIR_CAUSES.get(outcome, _REPAIR_GENERIC_CAUSE)
        # The never-seen class — stated as a property, never as a two-member list.
        if key in self._failure_slugs:
            return ("the buffer holds this entry but it failed to parse, so the "
                    "loop never reached it — fix the parse error reported above")
        if self._unread_reason is not None:
            # The run stopped reading the buffer, so it holds no evidence for the
            # absence claim below and may not make it.
            return self._unread_reason
        line = ("no entry with this slug was reached this run: it is absent from "
                "the buffer, or already rotated into an archive (`sync` reads the "
                "buffer alone, so an archived entry's reconciler is `mitos rebuild`)")
        if self._unattributable_failures:
            line += (". The buffer also holds unparsed entries this run could not "
                     "attribute to a slug")
        return line


class MitosSyncManager:
    """Manages the full parse-enrich-commit sync flow and side effects."""

    def __init__(self, config: MitosConfig) -> None:
        self.config = config
        self.lock_path = self.config.decisions_file + ".lock"
        self.lock = FileLock(self.lock_path, timeout=60)
        self.store = GraphStore(self.config.db_path)
        
        # Lazy initialize vector / embedding dependencies as best-effort (C2/P14)
        self.embed_provider: Optional[GeminiEmbeddingProvider] = None
        self.vector_store: Optional[QdrantVectorStore] = None

        # Set once an inline embed has met an absent collection it may not create, so
        # a multi-entry sync prints ONE deferral line and skips N−1 wasted Qdrant round
        # trips (and N−1 embedding calls). Per manager instance — one per command — and
        # read ONLY by ``_best_effort_embed``: a covering drain later in the same
        # ``perform_sync`` must not consult it, or a rebuild-then-sync would refuse to
        # heal because a record earlier in the run found the collection missing.
        self._collection_absent = False

        try:
            cache_path = os.path.join(self.config.mitos_dir, "embedding_cache.sqlite")
            self.embed_provider = GeminiEmbeddingProvider(
                cache_path,
                api_key=self.config.env.get("GEMINI_API_KEY"),
                model_id=get_embedding_model_id(self.config.env),
            )
            self.vector_store = QdrantVectorStore(
                self.config.qdrant_url,
                self.config.qdrant_collection
            )
        except Exception as e:
            # Let operations continue in degraded graph-only mode per S1/F2
            pass

    def auto_heal_decisions_file(self) -> None:
        """Auto-restores the decisions.md header and sample format block if modified or missing."""
        filepath = self.config.decisions_file
        if not os.path.exists(filepath):
            return

        # Load canonical format spec from package single source of truth
        from mitos.cli import load_format_spec
        try:
            format_spec_content = load_format_spec()
        except Exception:
            return

        # Extract sample block
        import re
        match = re.search(r'## 3\.\s+Sample Entry.*?\n```markdown\n(.*?)\n```', format_spec_content, re.DOTALL | re.IGNORECASE)
        sample_block = match.group(1).strip() if match else ""
        if not sample_block:
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        canonical_header = (
            "# Decisions for Mitos\n\n"
            "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
            "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
            "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
            f"{sample_block}\n\n"
        )

        marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
        if marker in content:
            parts = content.split(marker, 1)
            entries_content = parts[1]
            current_header = parts[0]
            if current_header.strip() != canonical_header.strip():
                new_content = canonical_header + marker + entries_content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                # stderr, not stdout: this method is on `record_decision_entry`'s path,
                # which the MCP write tool shares — and that transport uses stdout for
                # JSON-RPC, so a stray line here is protocol corruption, not noise. It
                # is also ahead of `restore-source --json`'s object, which a caller
                # parses. Matches this file's existing stderr discipline on the
                # embedding/render warnings below.
                print("Auto-restored decisions.md sample format header block ✓",
                      file=sys.stderr)
        else:
            if "## SAMPLE FORMAT" not in content:
                new_content = canonical_header + marker + "\n\n" + content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print("Auto-restored missing sample format header and BEGIN ENTRIES "
                      "marker ✓", file=sys.stderr)

    def perform_sync(self, auto_accept: bool = False, verbose: bool = False,
                     repair_targets: Optional[List[str]] = None) -> List[str]:
        """Executes the complete transactional sync flow.

        The repair ledger is built and reported HERE rather than inside
        ``_perform_sync_internal``, which has four returns plus a fall-through and
        two of them pull in opposite directions: the no-parseable-entries return
        carries two of the exit table's own rows (an unparseable target and one
        absent from the buffer), while the key-floor return must stay silent. One
        report site after every path, and one explicit carve-out marked at the
        floor, beats four call sites and four chances to miss the one that matters.

        Args:
            auto_accept: Whether ``--yes`` is in force.
            verbose: Emit cache statistics at the end of the run.
            repair_targets: Handles named by ``--reconcile-entry``, in the caller's
                verbatim spelling. ``None`` (no flag) and ``[]`` mean the same
                thing here — nothing was named, so the ledger is inert.

        Returns:
            The named targets that did NOT end in the state their markdown
            describes, for the caller to exit on. Empty on every ordinary run.
        """
        ledger = _RepairLedger(repair_targets)
        snapshot_path = os.path.join(self.config.mitos_dir, "sync_snapshot.md")
        # Second snapshot for steady-state questions.md ingestion (Phase 4a): taken
        # under the same lock as the decisions snapshot for read-consistency, and
        # cleaned in the same finally. Absent when questions.md is absent (healthy).
        questions_snapshot_path = os.path.join(self.config.mitos_dir, "questions_snapshot.md")
        try:
            self._perform_sync_internal(
                snapshot_path, questions_snapshot_path, auto_accept, verbose, ledger
            )
        finally:
            for path in (snapshot_path, questions_snapshot_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        return ledger.report()

    def _perform_sync_internal(self, snapshot_path: str, questions_snapshot_path: str, auto_accept: bool = False, verbose: bool = False, ledger: Optional["_RepairLedger"] = None) -> None:
        """Executes the internal transactional sync flow.

        ``ledger`` is defaulted so the sole production caller stays the only site
        that has to know about it and a direct call keeps working; an inert ledger
        makes every ``note`` below a no-op. ``is None`` rather than a truthiness
        coalesce, because an inert ledger is a real object with a real contract.
        """
        if ledger is None:
            ledger = _RepairLedger()

        # 1. Snapshot-at-sync-start under brief file lock
        questions_snapshotted = False
        try:
            with self.lock:
                if not os.path.exists(self.config.decisions_file):
                    print("No decisions.md file found. Run 'mitos init' first.")
                    return
                # Auto-heal the decisions file header/sample block under lock
                self.auto_heal_decisions_file()
                shutil.copy(self.config.decisions_file, snapshot_path)
                # Snapshot questions.md under the SAME lock (4a steady-state OQ
                # ingestion). An absent questions.md is healthy-empty (no current
                # corpus ships one). File-level bulkhead (D4/P7): a copy error in
                # the OQ buffer warns and yields zero OQ entries — it must never
                # abort decisions ingestion.
                if os.path.exists(self.config.questions_file):
                    try:
                        shutil.copy(self.config.questions_file, questions_snapshot_path)
                        questions_snapshotted = True
                    except OSError as exc:
                        print(
                            f"[Warning] Could not snapshot questions.md ({exc}); "
                            f"skipping open-question ingestion this sync."
                        )
        except Timeout:
            print("Another Mitos process holds the lock; check for stuck 'mitos sync'.")
            # Nothing was read, so a named target's line may not say it is absent.
            ledger.mark_buffer_unread(_REPAIR_UNREAD_LOCKED)
            return

        # 2. Parse from the snapshots, oldest-first within each file. Steady-state
        #    sync now ingests BOTH the decisions buffer and the questions.md
        #    open-question buffer (4a) — each parsed kind-by-file (V1-D8) into its
        #    OWN failure collector so a malformed entry in one buffer never aborts
        #    the other (D4 bulkhead). Oldest-first (parse_file_reversed) lands an
        #    older in-buffer entry before a newer one that references it — a flow
        #    heuristic, NOT correctness; the per-entry quarantine below (and 4b's
        #    fixpoint) are correctness. Collector mode isolates a malformed entry
        #    (reported + skipped) so the rest still sync (§5.2.2 per-entry isolation).
        dec_failures: List[EntryFailure] = []
        decision_entries = parse_file_reversed(snapshot_path, "decision", dec_failures)

        oq_failures: List[EntryFailure] = []
        oq_entries: List[ParsedEntry] = []
        if questions_snapshotted:
            # File-level bulkhead (D4/P7): the OQ snapshot read+parse is wrapped in
            # its OWN try/except so a file-level fault in questions.md — e.g. invalid
            # UTF-8 bytes the binary snapshot copy passed straight through, which
            # read_text_or_none then hits decoding as UTF-8 — warns and yields ZERO
            # OQ entries while decisions ingestion proceeds. The secondary buffer must
            # never abort the primary. (Per-ENTRY malformed OQs are already collector-
            # isolated into oq_failures, not raised; this catches the file-level fault
            # the collector cannot reach.)
            try:
                oq_entries = parse_file_reversed(
                    questions_snapshot_path, "open_question", oq_failures
                )
            except Exception as exc:
                oq_entries = []
                print(
                    f"[Warning] Could not parse questions.md ({exc}); "
                    f"skipping open-question ingestion this sync."
                )

        all_failures = dec_failures + oq_failures
        # Recorded before the `if not entries:` return below, because that return
        # carries two of the exit table's own rows: a named target whose block
        # failed to parse, and one genuinely absent from the buffer.
        ledger.note_parse_failures(all_failures)
        for fail in all_failures:
            msgs = "; ".join(item.message for item in fail.items) or "malformed entry"
            print(
                f"[Parse error] {msgs} (lines {fail.line_start}-{fail.line_end}). "
                f"Entry skipped — fix it and re-run sync."
            )

        # Decisions first, then open questions (D1): the host decision of an OQ's
        # typical Derives-From: forward-ref commits before the OQ, landing that
        # common case on the first pass. The opposite direction (a decision that
        # Resolves: an OQ authored above it) still quarantines on the first pass in
        # 4a and converges on the next sync (4b's fixpoint converges it in one).
        entries = decision_entries + oq_entries

        if not entries:
            if all_failures:
                print("No parseable entries to commit. Fix the reported entries above and re-run sync.")
            else:
                print("Zero pending entries found in the decisions.md / questions.md write-buffers.")
            return

        # Stale-entry detection (>14 days unprocessed) — DECISIONS ONLY (D5): the
        # vision defers OQ stale-detection; questions.md is a persistent buffer that
        # never rotates, so a >14-day open question must NOT emit a spurious
        # "remains unsynced" warning.
        for entry in decision_entries:
            if entry.date:
                try:
                    entry_dt = datetime.strptime(entry.date, "%Y-%m-%d")
                    diff = datetime.now() - entry_dt
                    if diff.days > 14:
                        print(f"[Warning] Entry '{entry.slug}' was drafted on {entry.date} (>14 days ago) and remains unsynced.")
                except Exception:
                    pass

        api_key = self.config.env.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY environment variable is not set. Sync requires API keys.")
            # The report's ONE carve-out. This refusal returns above the per-entry
            # loop, so a named target was neither reconciled nor looked for — there
            # is nothing to be loud about, and the run's exit stays 0 because
            # flipping it is a contract break under every CI job that reads it.
            # A floor on loudness, never a ceiling: a keyless workspace whose buffer
            # is empty reaches the return above this one and DOES report, correctly,
            # because on that corpus the named target genuinely is absent.
            ledger.mark_keyless()
            return
            
        renderer = MitosRenderer(self.config.workspace_dir)

        synced_blocks: List[Tuple[ParsedEntry, str]] = []

        # 4b intra-sync fixpoint: the main pass below collects every entry the store
        # rejects with a CommitError (a forward-ref whose in-corpus target has not
        # committed yet, a slug collision, a kind/cycle violation) into this set
        # instead of reporting it immediately. After the main pass, the fixpoint
        # re-attempts the set until a pass makes no further progress, so any acyclic
        # cross-file forward-ref chain converges in THIS single sync. Each tuple
        # carries the fully-prepared entry, its decisions-snapshot raw text (for
        # rotation if a decision commits in the fixpoint; "" for OQs), and its latest
        # failure (for the post-fixpoint residual report).
        quarantined: List[Tuple[ParsedEntry, str, CommitError]] = []

        # Conflict sensor (5a): build the judgment executor ONCE per run (CONF-D4/D7) —
        # one Anthropic client, one availability decision — and reference it per entry
        # inside the loop. Skipped (left None) when the sensor is inactive: `--yes`
        # (auto_accept runs no per-entry review at all) or the `conflict_check_on_sync`
        # toggle is off. `_build_conflict_judge` folds in the remaining availability gate
        # (ANTHROPIC_API_KEY + live embed/vector components) and lazy-imports `anthropic`,
        # so the SDK never rides a sync when the sensor is off.
        # 5b builds the per-run conflict context (sync_run_id + telemetry + breaker)
        # alongside the judge, ONCE per run, under the SAME availability guard — so no
        # TelemetryStore is constructed when the sensor is off, and the single
        # `_ConflictSyncRun` instance is what the write-then-read aggregate breaker
        # depends on (build it per entry and the breaker never trips). `conflict_run`
        # is non-None exactly when `conflict_judge` is.
        conflict_judge: Optional[Callable] = None
        conflict_run: Optional[_ConflictSyncRun] = None
        if not auto_accept and self.config.conflict_check_on_sync:
            conflict_judge = self._build_conflict_judge()
            if conflict_judge is not None:
                conflict_run = self._new_conflict_run()

        # 3. Process each parsed entry
        for entry in entries:
            # Read exact raw text block of this entry from snapshot for content-aware
            # rotation — DECISIONS ONLY (D5): the line range here indexes the
            # DECISIONS snapshot, so an open-question entry's range would index the
            # wrong file. OQ entries never rotate (questions.md is a persistent
            # buffer), so they need no raw-text block.
            entry_raw_text = ""
            if entry.kind == "decision":
                with open(snapshot_path, "r", encoding="utf-8") as f:
                    snap_lines = f.readlines()
                entry_raw_text = "".join(snap_lines[entry.line_start - 1 : entry.line_end])

            # Check if this node is already in the database (slug-free V1a id — V1-D2).
            node_id = compute_node_id(
                kind=entry.kind,
                axiom=entry.axiom,
                mechanism_refs=entry.mechanisms,
                topic=entry.topic,
                questions_raised=entry.questions_raised,
            )

            existing = self.store.get_node(node_id)
            if existing:
                # Idempotency short-circuit (S5). The node commit is deliberately
                # skipped — but a re-encounter from a NEW source must still emit one
                # source_reencounter audit signal (MI-4 / V1-D14). source is
                # out-of-core, so the canonical core is identical here; the
                # signal-eval is the one pass the skip must NOT swallow (§6.2
                # Lesson 13). This gate sits above the per-entry commit quarantine,
                # but note_source_reencounter is best-effort, so a signal failure
                # can never abort a sync of real decisions.
                self.store.note_source_reencounter(
                    node_id, existing["source"], entry.source or "user"
                )

                # C′ — the commentary reconcile, DIVERGENCE-GATED. Without the gate
                # this branch would reach `commit_parsed_entry` for every
                # already-committed entry: on the dogfood corpus that is 203 accept
                # prompts, 203 SONNET conflict judgments per sync, and the whole buffer
                # rotated into the archive. Gated, a clean corpus takes the same
                # `continue` as before — zero behavioural delta, and MI-3's
                # no-tick-on-a-byte-identical-recommit property stays intact.
                if entry.kind != "decision":
                    # Decisions only, matching the detector, which excludes open
                    # questions on both sides. The parser assigns `context` and
                    # `invalidates_if` kind-agnostically while the store NULLs them for
                    # an open question, so a reconcile of one can never converge: it
                    # would print "Reconciled ✓" and append a fresh attribution row on
                    # every single sync, for a mutation that provably cannot land.
                    ledger.note(entry, "open_question")
                    continue

                divergence = entry_divergence(
                    entry,
                    existing,
                    existing.get("scope") or [],
                    self.store.get_outgoing_edges(node_id),
                )
                if not is_reconcilable(divergence):
                    # `source` divergence is reported by `status` but is NOT
                    # reconcilable — MI-4 fences it out of the commentary UPDATE, so
                    # re-committing provably cannot change it.
                    #
                    # So this ONE branch covers TWO states for a named target, and
                    # the repair flag has to tell them apart: a genuinely clean
                    # entry is satisfied (the corpus and graph agree — exit 0),
                    # while one diverging ONLY in `source` is permanently
                    # unreconcilable and must not read as "not diverged". Exit 0
                    # there is the silent no-op the fail-loud property forbids, and
                    # `status`'s own rung already names the heal for it.
                    ledger.note(
                        entry,
                        "source_only" if divergence.get("source") else "clean",
                    )
                    continue

                if not self._apply_commentary_reconcile(
                    entry, existing, node_id, divergence, auto_accept,
                    authorized=ledger.authorizes(entry),
                ):
                    ledger.note(entry, "reconcile_refused")
                    continue
                # Reconciled. Deliberately falls through to `continue` rather than the
                # commit path below: no conflict judge (the canonical core is unchanged,
                # so there is no new claim to judge), no accept prompt, no confirmation
                # re-stamp, and above all NO ROTATION — rotation stays tied to a FIRST
                # commit, because an entry that leaves the buffer leaves sync's read-set
                # and its future divergence becomes invisible again.
                ledger.note(entry, "reconciled")
                continue

            # Slug collision check
            collision = self.store.get_node_by_slug(entry.slug)

            if collision:
                # A kill-edge the entry ALREADY declares at the colliding slug is the
                # supported same-slug supersession pattern (MI-13 rationale / FM1): the
                # author is evolving an axiom while keeping the citation handle. Commit
                # it exactly as declared — the override below would rewrite a declared
                # `supersedes` into `corrects` and silently contradict the author.
                # Normalize exactly as the edge resolver does (``_strip_citation`` +
                # casefold, MI-7): the deterministic parser stores `[slug]` with the
                # brackets on, the agentic write path stores a bare slug, and both are
                # the same declaration. Driving off ``_KILL_EDGE_FIELDS`` rather than a
                # re-hand-rolled pair keeps this carve-out correct if the kill-edge set
                # ever widens.
                declared_kill = {
                    _strip_citation(raw).casefold()
                    for field in _KILL_EDGE_FIELDS
                    for raw in (getattr(entry, field, None) or [])
                }
                if entry.slug.casefold() in declared_kill:
                    print(
                        f"\n[Collision] Slug '{entry.slug}' already exists in graph — "
                        "the entry declares a relation at it; committing as declared."
                    )
                else:
                    # An UNDECLARED collision is reported and skipped in BOTH modes,
                    # never auto-retired and never prompted for. Defaulting to `corrects`
                    # mints a killer node that retires a real decision on nothing but a
                    # slug match — and a canonical core can shift from any hand-edit to a
                    # `**Mechanisms:**` line, so the collision is as likely to be an
                    # accident as an intent (P5 Ironclad: automated recovery never
                    # destroys user data). A vector, not a wall: the entry stays in the
                    # buffer and the fix is one authored line.
                    #
                    # The interactive `[c]orrection / [s]upersession` prompt used to sit
                    # here, and it is retired rather than fixed. Its answer was only ever
                    # applied in memory — nothing spliced the chosen line into the buffer,
                    # and rotation archives the raw unmodified snapshot slice — so every
                    # interactively-resolved collision committed a kill-edge the gold
                    # source does not declare. Against P6/M7 (the markdown must remain the
                    # rebuildable truth): the entry replays at rebuild without its
                    # declaration, collides at the store, and becomes a permanent
                    # casualty. Sending the author to the markdown puts the declaration
                    # where a rebuild can find it again, and deleting the prompt also
                    # deletes the override below it, which wholesale-replaced both kill
                    # lists and so discarded any kill-edge the entry authored at a
                    # *different* slug.
                    print(f"\n[Collision] Slug '{entry.slug}' already exists in graph.")
                    print(f"  Existing Axiom: {collision.get('core_axiom')}")
                    print(f"  New Axiom:      {entry.axiom}")
                    print(
                        "  Skipped — nothing written. This entry declares no relation at "
                        f"'{entry.slug}', and sync will not retire a decision on a slug "
                        "match alone."
                    )
                    print(
                        f"  To supersede it, add `**Supersedes:** [{entry.slug}]` to the "
                        f"entry; to correct it, add `**Corrects:** [{entry.slug}]`. "
                        "Then re-run sync."
                    )
                    ledger.note(entry, "collision")
                    continue

            # No terminal and no `--yes`: report this pending entry and skip it, never
            # prompt. Position is the contract, not a placement detail. ABOVE the kind
            # split below, so ONE refusal dominates BOTH accept prompts — a guard
            # written inside either branch is one chance in two to ship a door still
            # dead on the other kind. Above the split also puts it above the per-entry
            # conflict sensor by construction, which is a spend contract: the crash
            # this replaces bounded the sensor's loop at one entry, so a guard placed
            # at the `input()` would let a run that can accept *nothing* sweep a paid
            # judgment across the whole pending buffer. And BELOW the collision block,
            # so every shipped per-entry diagnostic above it still prints.
            #
            # Report-and-skip, exit unchanged at 0: the entry stays in decisions.md
            # (sync never deletes from it), so `--yes` commits it next run — a skip
            # costs a turn, not an entry. It replaces an unguarded `input()` that died
            # with `EOF when reading a line` → `Fatal Unexpected Error`, exit 1, naming
            # nothing; the refusal names the entry it skipped.
            if not auto_accept and not sys.stdin.isatty():
                print(f"\n[Pending] '{entry.slug}' — skipped, stdin is not a terminal. "
                      "Re-run with `--yes` to accept pending entries non-interactively.")
                # A named target still pending is NOT satisfied: the repair flag
                # authorizes the reconcile gate, not this accept prompt, so the
                # refusal above stays true and stays unchanged.
                ledger.note(entry, "pending_skipped")
                continue

            if entry.kind == "decision":
                # Strict-deterministic sync (A): the decision commits EXACTLY as
                # authored — no LLM enrichment in the sync runtime path. Refining a
                # rough axiom, inferring scopes, or suggesting relationships lives only
                # where the input is genuinely raw (`capture`, `import --llm-extract`);
                # `sync` reads the human-authored buffer and must never silently rewrite
                # it (M7/P6 — markdown is the source of truth; ROADMAP — runtime parsing
                # is strict-deterministic; ADR sync-strict-deterministic-no-llm-enrichment).
                print(f"\nProposed Decision: {entry.slug}")
                print(f"  Core Axiom:  {entry.axiom}")
                print(f"  Rejected:    {entry.rejected_paths}")
                print(f"  Mechanisms:  {', '.join(entry.mechanisms)}")
                print(f"  Scope:       {', '.join(entry.scope)}")

                # Conflict sensor (5a): judge this decision against its undeclared close
                # neighbours and surface any high-confidence contradiction BEFORE the
                # accept prompt (CONF-D7), so the tension is named while the author can
                # still choose. `conflict_judge is not None` plus the guard above imply the
                # full gate (decision-kind here, not auto_accept, a terminal on stdin,
                # toggle on, judge available). RF-1:
                # `entry` holds the parsed declarations and nothing else — the slug-collision
                # override that used to rewrite them below is retired, so the property RF-1
                # relied on is now structural. Advisory — it prints, never blocks; the accept
                # prompt below is untouched.
                if conflict_judge is not None:
                    self._run_and_surface_conflict(entry, conflict_judge, conflict_run)

                if not auto_accept:
                    u_choice = input("Accept this decision? [a]ccept / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        ledger.note(entry, "operator_skipped")
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        ledger.note(entry, "operator_quit")
                        # The loop ends here, so every entry BELOW this one in the
                        # document went unread. A named target among them is still a
                        # shortfall, but the report may not call it absent.
                        ledger.mark_buffer_unread(_REPAIR_UNREAD_QUIT)
                        break

            else:
                # Open Question Sync
                print(f"\nProposed Open Question: {entry.slug}")
                print(f"  Questions: {', '.join(entry.questions_raised)}")
                if not auto_accept:
                    u_choice = input("Accept this open question? [a]ccept / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        ledger.note(entry, "operator_skipped")
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        ledger.note(entry, "operator_quit")
                        ledger.mark_buffer_unread(_REPAIR_UNREAD_QUIT)  # as above
                        break

            # Populate OD3 confirmation metadata
            # Strict-deterministic sync commits the authored buffer verbatim — no model
            # touches the decision here, so its provenance is the user/author, not an
            # enrichment model (A: sync-strict-deterministic-no-llm-enrichment).
            entry.confirmed_by = "user"
            # MI-10: application-supplied UTC ISO-8601 with an explicit offset — the
            # same helper `created_at` uses on the very same row. `datetime.now()` is
            # naive local time, which is what put 114 offset-less stamps beside 114
            # offset-aware ones in the live graph.
            entry.confirmed_at = _utc_now_iso()

            # Commit to graph database atomically per entry (C1 atomicity), now with
            # the per-entry commit-stage quarantine (4a floor — P5 dead-letter / P7
            # bulkhead): a store-stage rejection isolates to THIS entry while the rest
            # of the batch (decisions AND open questions) still commits — never the
            # whole-sync abort it was before. The catch keys on the CommitError CLASS,
            # not a code allowlist, so every §5.2.2 structural rejection (missing_target
            # ∪ slug_collision ∪ cycle_violation ∪ kind_constraint_violation ∪
            # dangling_edge) funnels through it uniformly (D3). 4b lifts the floor to a
            # ceiling: a quarantined entry is COLLECTED (not reported here) and re-tried
            # by the intra-sync fixpoint after this loop, so an acyclic forward-ref
            # converges in this sync rather than the next.
            try:
                delta = self.store.commit_parsed_entry(entry)
            except CommitError as exc:
                # 4b: collect (don't report yet) for the intra-sync fixpoint retry
                # after this loop. The entry is fully prepared (enriched, collision-
                # resolved, confirmed-stamped) — only the store-stage commit failed,
                # and its in-corpus target may yet commit in this same sync. Carry
                # entry_raw_text so a decision committed in the fixpoint still rotates.
                quarantined.append((entry, entry_raw_text, exc))
                # Pessimistic stamp: the fixpoint below upgrades whichever of these
                # it drains, so a missed correction leaves a shortfall rather than a
                # false satisfaction.
                ledger.note(entry, "quarantined")
                continue
            except (ValidationError, DatabaseError) as exc:
                # Defensive secondary bulkhead (Gotcha): a bypassed-parser empty core
                # or a raw SQLite failure carries NO §5.2.2 envelope, so the CommitError
                # quarantine above would miss it and it would abort the batch. Isolate
                # it per-entry too — report + skip (a genuine defect, never retry-
                # eligible). CommitError-only is the firm contract; this is the
                # defensive half.
                print(
                    f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-"
                    f"{entry.line_end}): unexpected store error — {exc}. Entry left in "
                    f"its buffer; fix and re-sync."
                )
                ledger.note(entry, "store_error")
                continue
            print(f"Committed node: {entry.slug} ✓")
            ledger.note(entry, "committed")

            # best-effort embedding upsert (C2) — applies to OQ nodes too.
            self._best_effort_embed(delta, entry)

            # Record successfully committed block for rotation — DECISIONS ONLY (D5):
            # questions.md never rotates (persistent buffer), and an OQ block would
            # carry decisions-snapshot raw text, so OQ entries must not enter the
            # rotation set or the pending_threshold rotation prompt's count.
            if entry.kind == "decision":
                synced_blocks.append((entry, entry_raw_text))

        # 3b. Intra-sync fixpoint retry (4b). The main pass committed every entry
        # whose targets were already present; re-attempt the quarantined set until a
        # pass makes no further progress, so any acyclic cross-file forward-ref chain
        # (D resolves Q, Q derives_from D', … however deep, in any authoring order)
        # converges in THIS single sync — order-independently, while every retry stays
        # an isolated per-entry commit_parsed_entry transaction (no batching, no
        # ordering — D5/MI-12). A genuinely-unresolvable reference (a never-authored
        # target, or a true A↔B mutual-reference cycle) makes zero progress, terminates
        # after one no-progress pass, and surfaces below as a loud per-entry vector —
        # never a hang, never a whole-sync abort. The fixpoint sits BEFORE rotation so
        # a decision it commits is appended to synced_blocks and rotates with the rest.
        residual = self._commit_quarantine_fixpoint(quarantined, synced_blocks)
        # The satisfied "committed" state is TWO sites, not one. A stamp placed only
        # after the main pass's `Committed node:` misses every entry the fixpoint
        # drains, so a named forward-ref target would exit non-zero on a run that
        # committed it. Keyed on identity because the residual tuples are the
        # quarantined ones.
        residual_ids = {id(entry) for entry, _raw, _exc in residual}
        for entry, _raw, _exc in quarantined:
            if id(entry) not in residual_ids:
                ledger.note(entry, "committed")
        for entry, _raw, exc in residual:
            self._report_commit_quarantine(entry, exc)

        # 4. Content-aware archive rotation under brief lock (V3b)
        if synced_blocks:
            if len(synced_blocks) >= self.config.pending_threshold and not auto_accept:
                print(f"\n[Lifecycle] Sync volume threshold reached ({len(synced_blocks)} entries pending rotation).")
                choice = input("Would you like to rotate the write-buffer to quarterly archive now? [y/n]: ").strip().lower()
                if choice != 'y':
                    synced_blocks.clear()
                    print("Archive rotation deferred. Entries remain in write-buffer.")

        if synced_blocks:
            try:
                with self.lock:
                    with open(self.config.decisions_file, "r", encoding="utf-8") as f:
                        live_content = f.read()

                    rotated_text = ""
                    for entry, raw_block in synced_blocks:
                        # Match by content block exactly and remove/modify in live file
                        if raw_block in live_content:
                            if self.config.rotation_mode == "mark":
                                # Mark mode: wrap the raw block in an HTML comment so it's ignored but preserved
                                commented_block = f"<!-- ROTATED START\n{raw_block}\nROTATED END -->"
                                live_content = live_content.replace(raw_block, commented_block)
                            else:
                                # Archive/Prune mode: remove from live buffer
                                live_content = live_content.replace(raw_block, "")
                            rotated_text += raw_block + "\n"

                    # Write back live buffer (non-destructive)
                    with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                        f.write(live_content)

                    # Only write to archive directory if in archive mode!
                    if self.config.rotation_mode == "archive":
                        quarter_file = f"{datetime.now().year}-Q{(datetime.now().month-1)//3 + 1}.md"
                        os.makedirs(self.config.archive_dir, exist_ok=True)
                        archive_path = os.path.join(self.config.archive_dir, quarter_file)
                        
                        with open(archive_path, "a", encoding="utf-8") as f:
                            f.write(rotated_text)
                        print(f"Rotated {len(synced_blocks)} entries to {archive_path} ✓")
                    elif self.config.rotation_mode == "prune":
                        print(f"Pruned {len(synced_blocks)} entries from buffer (rotation_mode=prune) ✓")
                    elif self.config.rotation_mode == "mark":
                        print(f"Marked {len(synced_blocks)} entries as rotated in buffer (rotation_mode=mark) ✓")
            except Exception as e:
                print(f"[Warning] Archive rotation failed: {str(e)}")

        # 5. Trigger renderer to statelessly regenerate files (C3)
        try:
            renderer.render_all(self.store)
            print("Regenerated live_axioms.md ✓")
        except Exception as e:
            # Degradation F4b: render failure doesn't affect graph commits
            print(f"[Warning] Failed to render active axioms: {str(e)}")

        # Temporary snapshot cleanup is handled by the perform_sync finally block

        # 6. Best-effort outbox queue drain attempt (C2)
        try:
            self.drain_pending_embeddings()
        except Exception as e:
            print(f"[Warning] Outbox queue drain failed: {str(e)}")

        # 7. Surplus hit/miss stats observability (4.D)
        if verbose and self.embed_provider:
            hits, misses, rate = self.embed_provider.get_stats()
            print(f"\n[Observability] Cache Stats: Hits: {hits}, Misses: {misses}, Hit Rate: {rate*100:.1f}%")

    def _commit_quarantine_fixpoint(
        self,
        quarantined: List[Tuple[ParsedEntry, str, CommitError]],
        synced_blocks: List[Tuple[ParsedEntry, str]],
    ) -> List[Tuple[ParsedEntry, str, CommitError]]:
        """Drains the per-entry quarantine set to a fixpoint (4b).

        Sits on 4a's per-entry quarantine *floor*: the main pass collected every
        entry the store rejected with a :class:`CommitError` (a forward-ref whose
        in-corpus target had not committed yet, plus the structural rejections that
        never self-heal). This re-attempts that set until a pass commits nothing new,
        so any acyclic cross-file forward-ref chain converges in a **single** sync,
        order-independently. A decision committed here is appended to ``synced_blocks``
        so it rotates with the main-pass commits; OQ nodes never rotate (D5).

        The convergence loop is the shared :func:`mitos.replay.commit_quarantine_fixpoint`
        primitive (the same engine the ``mitos rebuild`` corpus replay uses). This
        wrapper supplies the sync-specific embed + rotation callbacks and the loud
        convergence-observability line.

        Args:
            quarantined: The fully-prepared entries the main pass quarantined, each
                with its decisions-snapshot raw text ("" for an OQ) and its latest
                ``CommitError``.
            synced_blocks: The rotation record; a committed decision is appended
                ``(entry, raw)`` (mutated in place — the fixpoint runs before rotation
                reads it).

        Returns:
            The residual entries that never committed, each still carrying its latest
            ``CommitError`` — ``[]`` when everything converged.
        """
        def _record_decision_block(entry: ParsedEntry, raw: str) -> None:
            # Decisions rotate; OQs never do (raw is "" for an OQ, D5).
            if entry.kind == "decision":
                synced_blocks.append((entry, raw))

        committed, passes, residual = commit_quarantine_fixpoint(
            self.store,
            quarantined,
            embed_fn=self._best_effort_embed,
            on_commit=_record_decision_block,
        )

        # Convergence observability: make the fixpoint's work visible (the vision
        # values loud diagnostics) without any timing assertion — a structural signal
        # a test can read. Only printed when the quarantine set was non-empty.
        if quarantined:
            entry_word = "entry" if committed == 1 else "entries"
            pass_word = "pass" if passes == 1 else "passes"
            print(
                f"\n[Fixpoint] converged {committed} quarantined {entry_word} over "
                f"{passes} retry {pass_word}; {len(residual)} unresolved."
            )
        return residual

    def _report_commit_quarantine(self, entry: ParsedEntry, exc: CommitError) -> None:
        """Reports a residual per-entry commit failure as a guiding vector.

        The per-entry commit-stage bulkhead (P5 dead-letter / P7 Bulkhead): a single
        entry's store-stage rejection — any member of the §5.2.2 ``CommitError``
        class — is isolated to *that* entry. The entry is left in its buffer (never
        recorded for rotation), so a later sync can commit it once the operator fixes
        the reference or authors the missing target; the rest of this batch proceeds
        untouched.

        Called **post-fixpoint** (4b): by the time this fires, the intra-sync fixpoint
        has already exhausted every in-corpus retry, so a residual ``missing_target``
        no longer means "authored later in this corpus, settles next sync" (4a's
        optimistic framing) — it means the referenced target is **not present anywhere
        in this corpus**: a forward-ref to a never-authored / renamed-away target, or a
        true mutual-reference cycle where neither member can commit first. The vector
        says so honestly (D4), still a *guiding* vector rather than the bare "does not
        match any entry" wall that reads as a typo (D6). Every other code gets a calm,
        located generic surface. This mirrors the code-aware message builder in
        ``cutover._cutover_error_for_commit`` (the sync-side analogue — that one is
        cutover-specific and returns a ``CutoverError``, so it is not reused here).

        Args:
            entry: The entry the store rejected.
            exc: The ``CommitError`` carrying the §5.2.2 failure envelope.
        """
        items = exc.failure.items if exc.failure else []
        codes = {item.code for item in items}
        detail = "; ".join(item.message for item in items) or str(exc)
        if STORE_MISSING_TARGET in codes:
            print(
                f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-{entry.line_end}): "
                f"references a target that is not present anywhere in this corpus. The "
                f"intra-sync fixpoint already retried every in-corpus dependency, so this is "
                f"not a settles-on-the-next-sync forward-ref — author the missing target, fix "
                f"the reference, or break the mutual-reference cycle (in an A↔B cycle neither "
                f"member can commit first). Entry left in its buffer. ({detail})"
            )
        else:
            code_str = ", ".join(sorted(codes)) if codes else "referential violation"
            print(
                f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-{entry.line_end}): "
                f"the store rejected it ({code_str}). Entry left in its buffer for a later "
                f"sync once fixed. ({detail})"
            )

    def _build_conflict_judge(self) -> Optional[Callable]:
        """Builds the bound conflict-judgment executor for this sync run, or None (5a).

        The availability gate for the sync-time conflict sensor: the judgment needs an
        Anthropic client (``ANTHROPIC_API_KEY``, resolved for *this workspace* off
        ``config.env`` — as in ``importer.py``) AND
        the live ``embed_provider`` + ``vector_store`` the facade's candidate gather reads
        (both ``Optional``, ``None`` when Qdrant/Gemini were down at manager init). If any
        is absent the sensor cannot run, so this returns ``None`` and the per-entry hook is
        skipped for the whole run — the sensor is advisory and its absence must be invisible
        to a commit (P14/CONF-D7). The caller's kind/toggle/``--yes`` gates sit above this;
        this method owns only component + key availability.

        Built once per run (CONF-D4): one client, one availability decision — never per
        entry (that would re-read env + re-import ``anthropic`` N times). The SDK import is
        lazy so ``import anthropic`` never rides a ``mitos sync`` when the sensor is off (the
        caller skips this builder when the toggle is off; a missing key returns before the
        import here). The executor caps retries/timeout itself via ``with_options``, so the
        client needs no retry knobs (IMPL_NOTES 3b).

        Returns:
            The bound one-arg ``judge`` callable
            ``(RenderedPrompt) -> JudgmentExecution | Unavailable``, or ``None`` when the
            sensor is unavailable.
        """
        if self.embed_provider is None or self.vector_store is None:
            return None
        api_key = self.config.env.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        # Lazy import (CONF-D4/§8, load-bearing): `conflict_judgment` is the sole
        # module-scope `import anthropic` in the conflict pipeline. Importing it here keeps
        # the SDK off the `mitos sync` import path whenever the sensor is inactive.
        import anthropic
        from mitos.conflict_judgment import (_JUDGMENT_MODEL_ALIAS,
                                             make_judgment_executor)

        client = anthropic.Anthropic(api_key=api_key)
        return make_judgment_executor(
            client, model_id=get_model_id(_JUDGMENT_MODEL_ALIAS, self.config.env)
        )

    def _run_and_surface_conflict(
        self, entry: ParsedEntry, judge: Callable, run: "_ConflictSyncRun"
    ) -> None:
        """Runs the conflict check for one decision entry, surfaces + persists it (5a/5b).

        The sync-time surface of the Conflict sensor. Called under the caller's gates
        (decision-kind, ``not auto_accept``, a terminal on stdin, toggle on, judge
        available) immediately before the accept prompt, so a high-confidence contradiction
        is named at the moment the author can still choose (CONF-D7). The TTY gate is the
        loop's own report-and-skip refusal, which sits above the kind split and therefore
        above this call: a run with no terminal can accept nothing, so it judges nothing.
        Advisory only: it prints, applies no verb, writes nothing to the graph, and
        **never blocks the commit**.

        Contract (load-bearing, §8): no conflict-path outcome may abort a real decision
        commit. 5b gives the surface its memory and its failure manners:

        * **Aggregate breaker (P7, vision §4).** If a prior entry's check tripped the run's
          breaker, this returns immediately — a true no-op (no gather, no query, no judge),
          so a single downstream outage costs ONE penalty for the run, not N.
        * **Loud degradation notice.** A typed :class:`~mitos.conflict.Unavailable` (the
          sensor's downstream deps went dark) now prints one calm ``[Conflict sensor
          unavailable]`` notice and trips the breaker for the rest of the run (write-then-
          read: set here, read at the top of the next entry's check). The commit proceeds
          untouched (fail-open, CONF-D10).
        * **Best-effort telemetry.** A healthy JUDGED result (``execution is not None``)
          persists one ``judgment_batches`` row + N ``conflict_checks`` rows; the write is
          decoupled from surfacing and never aborts the sync (D5).

        A genuine local graph-store fault propagates past the facade (2a's D4
        ``DatabaseError``/``ValidationError``), so the whole call is wrapped defensively and
        any exception is logged to stderr, commit proceeds — matching this file's best-effort
        bulkhead idiom. This generic seam does NOT trip the breaker (it is a rare local fault,
        not a downstream-dep outage — IMPL_NOTES 5a, D3).

        Args:
            entry: The proposed decision entry — passed raw, so the facade reads the parsed
                declarations only (RF-1; structural since the slug-collision override that
                rewrote them was retired).
            judge: The bound judgment executor from :meth:`_build_conflict_judge`.
            run: The per-run :class:`_ConflictSyncRun` context (sync_run_id + telemetry +
                breaker), built once per sync and threaded through every entry's check.

        Returns:
            None. Output is the printed finding block or a degradation notice; nothing on a
            clean / tenable check (P9 quiet success — only a gated finding or a degradation
            speaks).
        """
        # Aggregate breaker gate (5b, §8): a prior entry already saw the sensor's downstream
        # go dark, so skip the whole check — including the disclosure below (harmless: it
        # already fired on the tripping entry or earlier). One penalty, not N (P3/P7).
        if run.breaker_tripped:
            return

        # First-fire activation disclosure (CONF-D5/P15): reached only after every gate has
        # passed, so the notice never prints when the sensor is inactive (no key, toggle
        # off, `--yes`).
        self._disclose_conflict_sensor_once()

        try:
            result = run_conflict_check(
                entry,
                embed_provider=self.embed_provider,
                vector_store=self.vector_store,
                store=self.store,
                judge=judge,
            )
        except Exception as exc:
            # Bulkhead (§8/D3): a graph-store fault propagates past the facade (2a D4). Log
            # and never block the commit — but do NOT trip the breaker (this is a local
            # fault, not a downstream-dep outage; the breaker is for the typed Unavailable).
            print(
                f"[Warning] Conflict check failed for '{entry.slug}': {exc}",
                file=sys.stderr,
            )
            return

        if isinstance(result, Unavailable):
            # Degraded check (5b): a loud one-time notice + trip the aggregate breaker so
            # every later entry skips the check. The commit proceeds exactly as if the
            # sensor were off (fail-open, CONF-D10); no row is persisted for a degradation.
            self._notice_conflict_unavailable(result.reason)
            run.breaker_tripped = True
            return

        # Healthy result: only surfaced (gated) findings speak (P9). Clean-empty or tenable
        # → `findings == []` → print nothing.
        if result.findings:
            self._print_conflict_findings(result.findings)

        # 5b persistence: a JUDGED result (the LLM fired) writes its judged pairs to the
        # telemetry corpus — the surfaced findings AND the silent-but-judged negative labels
        # (CONF-D8). Clean-empty (`execution is None`) is a healthy novel entry, not a
        # judged batch — it writes nothing (the canonical DoD-2 discriminator). Best-effort:
        # a telemetry failure never aborts the sync (D5).
        if result.execution is not None:
            self._persist_conflict_batch(result, run)

    def _new_conflict_run(self) -> "_ConflictSyncRun":
        """Mints the per-run conflict context for one sync (5b): id + telemetry + breaker.

        Called ONCE per ``perform_sync`` at the judge-build site, under the same
        availability guard as :meth:`_build_conflict_judge` — so no telemetry store is
        constructed when the sensor is off, and the single returned instance is what the
        write-then-read aggregate breaker threads through the loop (§8). The
        :class:`~mitos.telemetry.TelemetryStore` construction is best-effort: it boots its
        own ladder against ``config.telemetry_path`` and, on any failure, the run keeps a
        ``None`` telemetry handle — the sensor still surfaces findings, it simply can't
        persist them (a persistence outage must never disable the advisory surface, P7).

        Returns:
            A fresh :class:`_ConflictSyncRun` with a minted ``sync_run_id``, a (possibly
            ``None``) telemetry store, and ``breaker_tripped=False``.
        """
        sync_run_id = uuid.uuid4().hex
        telemetry: Optional[TelemetryStore] = None
        try:
            telemetry = TelemetryStore(self.config.telemetry_path)
        except Exception as exc:
            # Best-effort: a telemetry store that can't boot must not disable the surface.
            print(
                f"[Warning] Conflict telemetry unavailable ({exc}); "
                f"judged checks will not be persisted this sync.",
                file=sys.stderr,
            )
            telemetry = None
        return _ConflictSyncRun(sync_run_id=sync_run_id, telemetry=telemetry)

    def _persist_conflict_batch(
        self, result: ConflictCheckResult, run: "_ConflictSyncRun"
    ) -> None:
        """Persists one judged conflict batch to the telemetry corpus (5b, best-effort).

        Maps a healthy JUDGED :class:`~mitos.conflict.ConflictCheckResult` to its
        ``(JudgmentBatch, [ConflictCheckRow])`` and hands it to 1b's atomic writer. Stamps
        ``surface='sync'`` on every row (this writer IS the sync surface, CHK-D7) and the
        batch's ``model_id`` resolved from ``execution.model_alias`` at call time. Every
        fed-context field is read off the result's :class:`~mitos.conflict.JudgeInput`\\ s
        (``result.proposal_input`` / ``pair.candidate_input``) — exactly what the judge saw,
        VERBATIM — never a re-read of the node (which would risk drift and hit the 2a
        ``core_axiom`` key-name gotcha). Empty proposal ``rejected_paths``/scope and empty
        candidate scope serialize to ``NULL`` (MI-9); the NOT-NULL ``candidate_rejected_paths``
        is stored raw (even the degenerate ``""``).

        Best-effort bulkhead (D5): the write is a separate step after the print, wrapped so a
        telemetry failure (a wrapped ``DatabaseError`` from the writer, or a raw ``TypeError``
        a bad Python value dies on in ``to_params`` before SQLite sees it — IMPL_NOTES 1b)
        logs a stderr ``[Warning]`` and returns. It never aborts the sync and never trips the
        breaker (the breaker is for the *sensor's* downstream, not the corpus store).

        Args:
            result: A healthy judged result — the caller guards ``execution is not None``, so
                ``result.execution`` and ``result.judged_pairs`` are populated here.
            run: The per-run context carrying ``sync_run_id`` (stamped on every row) and the
                (possibly ``None``) telemetry store.
        """
        if run.telemetry is None:
            return
        try:
            execution = result.execution
            # CHK-D3: resolve the versioned model id HERE — moments after the call,
            # against the same resolved environment the call itself used
            # (``self.config.env``, 2c) — without widening the frozen executor
            # boundary. The map, not a pre-resolved id: ``execution.model_alias``
            # exists only inside this loop.
            # Deliberately narrower than the batch's best-effort wrap: an unknown
            # alias degrades to NULL (the column is provenance-only), never to a
            # lost batch whose rationale is non-regenerable (M8).
            try:
                model_id: Optional[str] = get_model_id(
                    execution.model_alias, self.config.env
                )
            except ValueError:
                model_id = None
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
            proposal = result.proposal_input
            rows: List[ConflictCheckRow] = []
            for pair in result.judged_pairs:
                candidate_input = pair.candidate_input
                rows.append(
                    ConflictCheckRow(
                        batch_id=execution.batch_id,
                        sync_run_id=run.sync_run_id,
                        # This writer IS the sync surface — stamped explicitly on
                        # every row, never left to the schema DEFAULT (CHK-D7).
                        surface="sync",
                        judged_axiom=proposal.axiom,
                        # MI-9: an empty proposal rejected_paths ("") / scope ([]) is NULL,
                        # never "" — the NULL column already expresses "nothing here".
                        proposal_rejected_paths=proposal.rejected_paths or None,
                        proposal_scope=", ".join(proposal.scope) or None,
                        proposed_hash_if_any=result.proposed_hash_if_any,
                        candidate_slug=pair.candidate.slug,  # verbatim, no casefold
                        candidate_hash=pair.candidate.node["id"],  # M2 content hash, not slug
                        # NOT NULL (M5 guarantees a committed decision has it): store the raw
                        # str, even the degenerate "" — never coerce to None here.
                        candidate_rejected_paths=candidate_input.rejected_paths,
                        candidate_scope=", ".join(candidate_input.scope) or None,
                        tenable=pair.judgment.tenable_together,
                        confidence=pair.judgment.confidence,
                        surfaced=pair.surfaced,
                        candidate_source=CONFLICT_CANDIDATE_SOURCE,
                        model_alias=execution.model_alias,
                        prompt_version=CONFLICT_PROMPT_VERSION,
                        mitos_version=MITOS_VERSION,
                        rationale=pair.judgment.rationale,
                    )
                )
            # One MI-10 UTC stamp per batch, shared by every row (D5).
            self._record_conflict_batch(run.telemetry, batch, rows)
        except Exception as exc:
            # Best-effort (D5/§8): a mapping or write failure never aborts the sync. Mirror
            # the file's `[Warning] ...: {exc}` stderr idiom (the embed/drain/rotation
            # siblings). No breaker trip — the corpus store is not the sensor's downstream.
            print(
                f"[Warning] Could not persist conflict telemetry: {exc}",
                file=sys.stderr,
            )

    def _record_conflict_batch(
        self,
        telemetry: TelemetryStore,
        batch: JudgmentBatch,
        rows: List[ConflictCheckRow],
    ) -> None:
        """Stamps one MI-10 UTC timestamp and writes the batch (the sole telemetry write).

        A thin seam over :meth:`~mitos.telemetry.TelemetryStore.record_judged_batch` that
        takes the single ``created_at`` stamp (MI-10: application-supplied UTC ISO-8601, one
        per batch, shared by every row — never ``CURRENT_TIMESTAMP``). Kept separate from
        :meth:`_persist_conflict_batch`'s mapping so a test can monkeypatch the write in
        isolation and so the one wall-clock read has a single home.

        Args:
            telemetry: The run's telemetry store (non-``None`` — the caller guards it).
            batch: The per-batch metrics row.
            rows: The judged candidate rows.
        """
        telemetry.record_judged_batch(batch, rows, _utc_now_iso())

    def _notice_conflict_unavailable(self, reason: ConflictUnavailableReason) -> None:
        """Prints the loud one-time degradation notice for a typed Unavailable (5b, §7).

        The surface's disposition of a typed :class:`~mitos.conflict.Unavailable`: a calm,
        plain-text notice that (a) names WHICH subsystem went dark — switching on ``reason``
        (embedding/vector-store → semantic recall; judgment/timeout → the judge) — (b) states
        the breaker consequence (conflict checking is off for the rest of this sync), and
        (c) reassures that the commit is unaffected (fail-open, CONF-D10). All conflict UX
        wording lives HERE in the surface (P7 core/surface bulkhead); ``mitos.conflict``
        returns the typed reason and ``result.detail`` is logging-only, NEVER rendered.

        A tag distinct from 5a's ``[Conflict]`` (finding) and ``[Conflict sensor active]``
        (disclosure) — ``[Conflict sensor unavailable]`` — so the three surfaces don't blur.
        Plain ASCII, no emoji, no required colour (P9 A11y).

        Args:
            reason: The typed :class:`~mitos.conflict.ConflictUnavailableReason` from the
                degraded result — the machine-readable discriminator the surface words.
        """
        if reason is ConflictUnavailableReason.COLLECTION_MISSING:
            # Inside the semantic-substrate bucket, but worded apart: the vector store
            # DID respond, so "did not respond" would be the missing-as-unreachable
            # conflation, on the surface an operator meets most often.
            what = (
                "Semantic recall is unavailable (the vector collection is missing — run "
                "`mitos reconcile`)"
            )
        elif reason in SEMANTIC_SUBSTRATE_REASONS:
            what = (
                "Semantic recall is unavailable (the vector store or embedding service did "
                "not respond)"
            )
        elif reason is ConflictUnavailableReason.JUDGMENT_TRUNCATED:
            what = (
                "Conflict judgment is unavailable (the judge's response was truncated at "
                "max_tokens)"
            )
        elif reason is ConflictUnavailableReason.JUDGMENT_TIMEOUT:
            what = (
                "Conflict judgment is unavailable (the judgment model did not respond in "
                "time)"
            )
        else:
            what = (
                "Conflict judgment is unavailable (the judgment batch was malformed)"
            )
        print(
            f"\n[Conflict sensor unavailable] {what}. Conflict checking is skipped for the\n"
            "  rest of this sync; your decisions still commit normally."
        )

    def _disclose_conflict_sensor_once(self) -> None:
        """Prints the conflict-sensor activation notice once per project (CONF-D5).

        The ``conflict_check_on_sync`` toggle defaults on, so an upgrade silently enables a
        feature that can spend Claude-tier tokens. Per P15 / progressive activation, disclose
        it the first time it actually fires: if the ``.mitos/.conflict_disclosed`` sentinel
        is absent, print a calm one-time notice (what it does, the tier + rough cost, how to
        turn it off) and create the sentinel. A **presence-only** marker, zero content —
        derivative state, safe to delete (the notice simply re-fires, harmlessly).
        """
        sentinel = os.path.join(self.config.mitos_dir, ".conflict_disclosed")
        if os.path.exists(sentinel):
            return
        print(
            "\n[Conflict sensor active] mitos now checks each synced decision against its\n"
            "  close neighbours in the graph and, before you accept, names any active one it\n"
            "  may contradict without linking to. It uses the Claude judgment tier (roughly a\n"
            "  few cents only for an entry with a contradiction-suspect neighbour; most cost\n"
            "  nothing — the similarity floor screens them out first). It only advises: it\n"
            "  never blocks a commit and writes nothing. To turn it off, set\n"
            "  `conflict_check_on_sync = false` in .mitos/config.toml."
        )
        try:
            with open(sentinel, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as exc:
            # A derivative marker; if it can't be written the notice re-fires next run
            # (harmless). Never let it block the sensor or the commit.
            print(
                f"[Warning] Could not write conflict-disclosure sentinel: {exc}",
                file=sys.stderr,
            )

    def _print_conflict_findings(self, findings: List[ConflictFinding]) -> None:
        """Renders surfaced conflict findings as a calm plain-text block (5a, §7).

        For each surfaced finding, prints the conflicting decision's slug + similarity, its
        Letter fields (axiom, rejected_paths, scope) and any modifier stamps, then the
        judgment rationale — the "why" the two may not both stand (the payload describes the
        *candidate*; the rationale describes the *tension*). The existing accept prompt runs
        unchanged afterwards.

        All conflict/notice wording lives HERE, in the surface (P7 core/surface bulkhead):
        :mod:`mitos.conflict` returns structured findings and never prints. Plain text, ASCII
        markers, no emoji, no required colour (P9 A11y). Wording is calm and non-accusatory
        ("may contradict"): the calibrated floor cannot screen a legitimate scoped narrows
        carve-out (4b), so a false positive is possible — and accept-anyway (the unchanged
        prompt) is the one-keystroke backstop.

        Args:
            findings: The gated (surfaced) findings from a healthy ``ConflictCheckResult``.
        """
        plural = "decisions it does not link to" if len(findings) > 1 else \
            "decision it does not link to"
        print(f"\n[Conflict] This decision may contradict an active {plural}:")
        for finding in findings:
            payload = finding.payload
            print(f"  -> {payload['slug']}   (similarity {payload['score']:.2f})")
            print(f"       Axiom:    {payload['axiom']}")
            # rejected_paths rides a non-brief finding payload; guard anyway (a brief
            # payload would omit it). Modifier stamps ride CONDITIONALLY — a key is present
            # only when non-empty (blind indexing KeyErrors on the common unmodified
            # candidate), so guard every optional key (mirrors 2b's own copy guard).
            if "rejected_paths" in payload:
                print(f"       Rejected: {payload['rejected_paths']}")
            scope = payload.get("scope") or []
            scope_text = ", ".join(scope) if scope else "(global — no scope declared)"
            print(f"       Scope:    {scope_text}")
            for key in ("superseded_by", "amended_by", "narrowed_by", "corrected_by"):
                if key in payload:
                    print(f"       ({key.replace('_', ' ')}: {', '.join(payload[key])})")
            print("     Why they may not both stand:")
            print(f"       {finding.rationale}")
        print(
            "  Accept commits this decision as-is (unlinked). To resolve, skip and add a\n"
            "  relationship (Supersedes: / Amends: / Narrows: / Contradicts:) in\n"
            "  decisions.md, then re-sync. Quit to abort."
        )

    def _best_effort_embed(self, delta: CommitDelta, entry: ParsedEntry) -> Optional[List[float]]:
        """Best-effort async embedding upsert pipeline (C2).

        The committing node is already enqueued on the ``pending_embeddings`` Outbox
        by ``commit_parsed_entry`` (``_enqueue_outbox``, 5c), so this is the inline
        fast path: with the provider up we embed + upsert immediately and DROP the
        now-redundant Outbox row (the node is indexed); with it down or the call
        failing we leave the row for the next ``sync`` drain — never a second enqueue
        (the commit already wrote one; the prototype's deferred ``add_pending_embedding``
        is retired here, 8a). Returns the document vector it computed and upserted (so
        a caller can reuse it for a neighbour query), or None if embedding was
        deferred/failed.

        This is a **single-node** write, so it may create an absent collection only when
        that one node *is* the whole active set — the fresh-project carve-out. The
        declaration comes from the graph gate
        (:meth:`~mitos.store.GraphStore.has_active_node_other_than`), the same shipped
        active-view predicate the read-side gate uses; a populated graph defers instead,
        keeping the collection absent so the honest "couldn't check" notice keeps firing
        rather than quietly becoming "checked, clean" over a one-point index.
        """
        if self._collection_absent:
            # Already reported once for this command: say nothing, spend nothing, and
            # skip the round trip. The outbox row the commit wrote is the durable
            # record of the deferral either way.
            return None

        embed_text = _embedding_input_text(
            kind=entry.kind, axiom=entry.axiom,
            topic=entry.topic, questions_raised=entry.questions_raised,
        )

        if not self.embed_provider or not self.vector_store:
            # Already enqueued by the commit; just note the deferral. stderr — this
            # path is shared with the MCP write tool, whose stdout is the JSON-RPC
            # channel (a stray stdout line there corrupts the protocol).
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': Embedding provider down.",
                  file=sys.stderr)
            return None

        # Prepare payload
        payload = {
            "slug": entry.slug,
            "scope": entry.scope,
            "state": "active",
            "kind": entry.kind,
            "embedding_text": embed_text
        }

        try:
            # Declared intent, from the graph gate: this write covers the active set iff
            # there is no other active node. Cheap enough to run unconditionally (an
            # EXISTS … LIMIT 1), and it must be computed before the write because only
            # the 404 reveals whether it matters. Deliberately INSIDE this try: the
            # helper's contract is that it swallows its own faults and leaves the outbox
            # row (that is what makes deferral cheap), and a graph read added outside it
            # would let a store fault escape a path that has never raised.
            may_create = not self.store.has_active_node_other_than(delta.node_id)
            # Check embedding provider and generate vector
            vector = self.embed_provider.get_embedding(payload["embedding_text"], is_query=False)
            self.vector_store.upsert(delta.node_id, vector, payload, may_create=may_create)
            # Indexed now — drop the Outbox row the commit enqueued.
            try:
                self.store.remove_pending_embedding(delta.node_id)
            except Exception as dbe:
                print(f"[Warning] Failed to clear outbox row: {str(dbe)}", file=sys.stderr)
            return vector
        except CollectionMissingError:
            # Named as itself, not as an outage: Qdrant is up, the index is not there,
            # and this write is not the one that may build it. One line per command
            # (the latch), and the outbox row the commit wrote stays put — so the work
            # is queued and `mitos reconcile` completes it in one pass. stderr, for the
            # same JSON-RPC reason as the provider-down twin above.
            self._collection_absent = True
            print(
                f"[Warning] Embedding upsert deferred for '{entry.slug}': the Qdrant "
                f"collection is missing and this write does not cover the active set. "
                f"Queued — run `mitos reconcile` to rebuild the index.",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            # The commit already enqueued this node (C2); leave the row for the next
            # drain. stderr — shared with the MCP write tool's JSON-RPC stdout channel.
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': {str(e)}", file=sys.stderr)
            return None

    def drain_pending_embeddings(self) -> None:
        """Drains the pending embeddings outbox queue (C2).

        Claims a batch of pending embeddings atomically to prevent concurrent
        drainers from double-processing rows, processes them, and removes resolved entries.

        **Whether this drain may create an absent collection is not a property of the
        running call site.** The drain has three callers (``mitos sync --embed-only``,
        the sync commit path, and ``reconcile_embeddings``), and ``--embed-only`` is a
        drain with no buffer at all — its covering-ness is entirely a property of outbox
        state that a *prior* command deliberately established, which a call-site flag
        structurally cannot express. So the knowledge travels to the substrate: the
        ``embedding_seed`` marker, written by ``rebuild``/``cutover``'s prune (the act
        that makes the outbox equal the active set) and by ``reconcile``'s enqueue pass.
        This is the marker's only consumer. It is read **once at entry** — not per
        claim-batch, so a concurrent second drainer cannot flip it mid-run — and cleared
        the moment the outbox empties, because that is the moment coverage has been
        delivered and the claim stops being true. A drain that stops early (the
        zero-progress guard, or a refusal) **keeps** the marker: rows remain, so the next
        drain is still covering.

        It is emphatically *not* an inferred comparison of outbox size against the active
        set — that is not a completeness measure (the outbox holds rows for
        no-longer-active nodes), it is more expensive than the signals it would replace,
        and it decays silently once a migration hands every project a clean collection
        holding only active points.
        """
        if not self.embed_provider or not self.vector_store:
            print("Cannot drain outbox: Embedding provider or vector store down.")
            return

        import uuid
        drainer_id = f"drainer-{uuid.uuid4()}"

        # Declared intent, from the state a prior command established (read once).
        may_create = self.store.embedding_seed() is not None

        printed_header = False
        total_drained = 0
        # Set by the refusal arm below; the refusal must escape BOTH loops (the per-item
        # `for` and the enclosing claim-batch `while`), and a bare `break` leaves only
        # the inner one.
        refused = False
        try:
            # Drain the outbox in claim-batches until it is EMPTY. A single call must
            # not stop after one `limit`-sized batch: a corpus with more than `limit`
            # unembedded nodes — the common state right after `mitos rebuild`/`cutover`,
            # which re-seed the outbox with the whole active set — would otherwise be
            # left with `vectors < nodes` until the operator happened to run `sync`
            # again (the documented single-`sync` re-embed silently under-delivering).
            #
            # Progress guard: a batch that resolves ZERO rows (every claimed row errored
            # — e.g. an embedding-provider 429/outage) breaks the loop rather than
            # re-claiming the same failing rows forever. Those rows keep their
            # incremented `retry_count` and drain on a later sync — fail-fast in
            # aggregate: one outage costs one batch of attempts, not a hot spin. An
            # orphan row (node gone from the graph) counts as resolved (it LEAVES the
            # outbox), so a batch of orphans still makes progress and the loop continues.
            while True:
                try:
                    pending = self.store.claim_pending_embeddings(drainer_id, limit=10)
                except Exception as e:
                    print(f"[Warning] Failed to claim outbox queue: {str(e)}")
                    return

                if not pending:
                    # The outbox is empty: any coverage the marker claimed has been
                    # delivered (or there was nothing to deliver), so the claim is spent
                    # and must stop authorizing creation. Clearing here covers both the
                    # drained-to-empty exit and an already-empty outbox on entry.
                    self._clear_embedding_seed_best_effort()
                    break

                if not printed_header:
                    print("Draining pending embeddings queue ...")
                    printed_header = True

                batch_resolved = 0  # rows that LEFT the outbox this batch (embedded or orphan-pruned)
                for item in pending:
                    node_id = item["node_id"]

                    # Fetch node details from graph for Qdrant payload
                    node = self.store.get_node(node_id)
                    if not node:
                        # Node has been deleted from graph; remove from queue
                        try:
                            self.store.remove_pending_embedding(node_id)
                            batch_resolved += 1
                        except Exception:
                            pass
                        continue

                    # Re-derive the embedding text from the node's immutable core (the
                    # Outbox row no longer carries it — C2/M8); byte-identical to what the
                    # inline record-time embed used for the same node.
                    embed_text = _embedding_input_text(
                        kind=node["kind"], axiom=node.get("core_axiom"),
                        topic=node.get("topic"), questions_raised=node.get("questions_raised"),
                    )

                    payload = {
                        "slug": node["slug"],
                        "scope": node["scope"],
                        "state": "active",
                        "kind": node["kind"],
                        "embedding_text": embed_text
                    }

                    try:
                        # 1. Fetch embedding vector
                        vector = self.embed_provider.get_embedding(embed_text, is_query=False)
                        # 2. Upsert to Qdrant
                        self.vector_store.upsert(
                            node_id, vector, payload, may_create=may_create
                        )
                        # 3. Clean up queue row on success
                        self.store.remove_pending_embedding(node_id)
                        batch_resolved += 1
                        total_drained += 1
                        print(f"Successfully drained embedding for '{node['slug']}' ✓")
                    except CollectionMissingError:
                        # Ahead of the generic arm below, and deliberately quiet: a
                        # refusal is a property of the whole drain, not of this row, so
                        # it costs ONE calm line and stops immediately rather than
                        # repeating per node. And NO `increment_pending_attempts` — the
                        # row was not rejected, the drain declined; inflating a counter
                        # that exists to describe genuine failure would poison it.
                        print(
                            "Stopping the drain: the Qdrant collection is missing and no "
                            "covering re-embed has been requested, so writing now would "
                            "index only part of the corpus. Run `mitos reconcile` to "
                            "rebuild it in one pass."
                        )
                        refused = True
                        break
                    except Exception as e:
                        # Increment retry count on failure (which also releases this row)
                        try:
                            self.store.increment_pending_attempts(node_id)
                        except Exception:
                            pass
                        print(f"[Warning] Failed to drain embedding for '{node['slug']}': {str(e)}")

                # The refusal escapes the claim-batch loop too — the inner `break` only
                # left the per-item one, and re-claiming would meet the same absent
                # collection. The marker (if any) is deliberately RETAINED: rows remain,
                # so the coverage claim still holds for the next drain.
                if refused:
                    break

                # Nothing left the outbox this batch — every claimed row errored (a
                # dead/erroring provider). Stop; re-claiming would return the same rows.
                if batch_resolved == 0:
                    break
        finally:
            # Clean up: release any remaining locks held by this specific drainer
            try:
                self.store.release_pending_embeddings(drainer_id)
            except Exception:
                pass

    def _clear_embedding_seed_best_effort(self) -> None:
        """Clears the coverage marker, reporting rather than raising on failure.

        The clear is bookkeeping on a delivered claim, not part of the drain's product,
        so a graph fault here must not turn a fully-drained outbox into a failed
        ``sync``. Failing to clear is also the *safe* direction — the marker only ever
        authorizes a covering drain, and the next one re-reads the (now empty) outbox
        and clears it again.
        """
        try:
            self.store.clear_embedding_seed()
        except Exception as e:
            print(f"[Warning] Failed to clear the embedding seed marker: {str(e)}")

    def reconcile_embeddings(self) -> Dict[str, int]:
        """Re-embeds active nodes missing from Qdrant, then drains the outbox.

        Heals the gap Fix 1's drain cannot reach: when Qdrant is wiped directly
        (a bare ``curl -X DELETE`` of the collection) WITHOUT a ``rebuild``/
        ``cutover`` — which re-seed the outbox — the graph has nodes, Qdrant has
        none, and the outbox is empty, so ``sync`` drains nothing. This pass
        diffs the graph's ACTIVE node set against Qdrant's actual point ids,
        enqueues every missing active node, and drains.

        Only the active surface is reconciled (``get_active_node_ids``): dead and
        superseded nodes are intentionally not re-embedded (retrieval filters
        them via ``has_id`` and never returns them — see the
        ``cutover-bounds-embedding-seed-to-active`` decision). Bounded to one
        paginated scroll + a set difference (no per-node probes), so it stays
        cheap at the P11 horizon. Idempotent: a second run finds nothing missing
        and enqueues nothing. Re-queued nodes hit the embedding cache, so the
        heal costs no embedding-API calls when the text is unchanged.

        Returns:
            Counts ``{"active", "present", "enqueued"}`` — active nodes, points
            already in Qdrant, and nodes enqueued for re-embedding.

        An **absent collection reads as empty**, not as an error: it is the very
        state this heal exists for (a wipe, or a rename that left the old name
        behind), so every active node is missing and every one gets enqueued — the
        drain's first upsert then creates the collection. Only genuine
        unreachability raises.

        Raises:
            VectorStoreError: If Qdrant is unreachable while scrolling point ids.
                A *missing collection* does not raise (see above).
        """
        if not self.embed_provider or not self.vector_store:
            # stderr, not stdout: `mitos reconcile --json` promises stdout carries
            # exactly one JSON object, and this line is `as_json`-blind — it broke
            # `json.loads(stdout)` on every provider-down workspace. Threading a
            # presentation flag down into the manager was the alternative; a
            # diagnostic on the diagnostic channel keeps the routing decision at
            # the boundary, where the rest of it already lives.
            print(
                "Cannot reconcile: Embedding provider or vector store down.",
                file=sys.stderr,
            )
            return {"active": 0, "present": 0, "enqueued": 0}

        active_ids = self.store.get_active_node_ids()
        try:
            present = self.vector_store.list_point_ids()
        except CollectionMissingError:
            # Narrow by design: a connection-level VectorStoreError must keep
            # propagating so `mitos reconcile` can still report Qdrant down.
            present = set()
        missing = [nid for nid in active_ids if hash_to_uuid(nid) not in present]

        for node_id in missing:
            self.store.add_pending_embedding(node_id)

        if missing:
            # The enqueue above is the act that makes the outbox cover the active set,
            # so record it before draining: the drain reads the marker, not a flag, and
            # stamping it durably buys crash-safety for free — a reconcile that dies
            # mid-drain leaves the marker standing, and the operator's next `sync`
            # completes the heal instead of refusing it. This is one of only three
            # producers, all operator-invoked: nothing implicit ever stamps it, which is
            # what keeps the heal operator-invoked rather than automatic.
            self.store.stamp_embedding_seed("reconcile")
            self.drain_pending_embeddings()

        return {
            "active": len(active_ids),
            "present": len(present),
            "enqueued": len(missing),
        }

    # --- record_decision: the write half of the MCP server (Fork A) ---

    def _exact_slug_node(self, slug: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Resolves a slug to an EXACT-match node.

        ``resolve_slug`` is now single-tier casefold-exact (no fuzzy prefix tier), so
        every id it returns already shares the casefolded slug; the per-node re-filter
        below is a defensive guard on that contract (and folds with ``str.casefold()``,
        never ``str.lower()`` — the two diverge on ``ß``/Greek, MI-9).

        Returns:
            A (node_id, node) tuple for the exact-slug node, or (None, None).
        """
        for node_id in self.store.resolve_slug(slug):
            node = self.store.get_node(node_id)
            if node and node.get("slug", "").casefold() == slug.casefold():
                return node_id, node
        return None, None

    def _validate_relation_target(
        self, relation: str, target: str, *,
        canonical_slugs: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Validates a typed relation's target is a unique, EXACT-match decision.

        Mirrors the supersedes check (``resolve_slug`` is casefold-exact; the re-filter
        defends its contract), keeping every recorded edge pointed at a real, unambiguous
        node. Runs in Phase A — a failure returns a structured error and writes nothing.

        Args:
            relation: The relation kwarg name (for the error message), e.g. "amends".
            target: The slug the agent passed as that relation's target.
            canonical_slugs: Optional casefolded-target -> stored-slug map to record the
                resolved node's own spelling into. This is the ONLY place an
                *ungathered* declared target's canonical handle is ever computed — the
                node is already fetched here and its slug discarded — and the pause echo
                prints stored spellings, never caller ones. Absent on the callers that
                do not compose an echo.

        Returns:
            None if valid, else a structured ``{error, code}`` dict.
        """
        ids = self.store.resolve_slug(target)
        if not ids:
            return _record_error("relation_target_not_found", relation=relation, target=target)
        if len(ids) > 1:
            return _record_error("relation_target_ambiguous", relation=relation, target=target)
        node = self.store.get_node(ids[0])
        if not node or node.get("slug", "").casefold() != target.casefold():
            return _record_error("relation_target_not_found", relation=relation, target=target)
        if canonical_slugs is not None:
            canonical_slugs[target.casefold()] = node["slug"]
        return None

    def _review_neighbors(self, entry: ParsedEntry, declared_targets: set, *,
                          gathered_index: Optional[Dict[str, Tuple[str, float]]] = None,
                          ) -> "List[Dict[str, Any]] | Unavailable":
        """Pre-commit: existing live decisions too similar to ``entry`` to ignore (P4).

        Composes the Conflict sensor's candidate stages (the same discovery `mitos
        check` uses): :func:`gather_candidates` embeds the axiom in document space,
        over-fetches one bounded KNN window, and keeps only live *decisions* — an
        open question in the window is screened by kind, never crashed on — then
        :func:`screen_candidates` drops declared/self, floors at
        ``_NEIGHBOR_REVIEW_THRESHOLD``, and truncates to ``CONFLICT_TOP_K``.
        Surfacing survivors BEFORE the write is the whole point: after the commit the
        author can no longer point an amends/supersedes at them (a re-record is a no-op).

        Degradation is typed, never silent: a failed embed or KNN query returns the
        gather's :class:`Unavailable` untouched (the caller words the receipt notice
        from its reason), and graph-store faults (``DatabaseError``/``ValidationError``)
        propagate — the record call site owns the fail-open disposition. Only the
        *structural* no-providers state (a graph-only workspace, first-class healthy)
        returns a clean ``[]`` here.

        Args:
            entry: The parsed entry about to be committed.
            declared_targets: Casefolded slugs the entry already links to — its declared
                relation targets plus their transitive mutation lineage — excluded so a
                linked neighbour is not re-flagged.
            gathered_index: Optional sink the caller passes to keep this call's *raw*
                gathered set, ``{casefolded slug: (stored slug, score)}``. The pause
                echo partitions the caller's declarations against the gathered
                candidates and the floor — the primary sets (M8) — and the screened
                return cannot answer for them: S4 drops a declared target *before* the
                floor, so a declaration that resolved a strong neighbour and one that
                fell short are indistinguishable there. A sink rather than a widened
                return type, so the eight ``patch.object`` sites binding this method by
                string stay green.

        Returns:
            A list of :func:`~mitos.conflict.candidate_payload` dicts for
            unreferenced high-similarity live neighbours, most similar first
            (possibly empty) — the enriched Letter shape (``slug`` / ``axiom`` /
            ``scope`` / ``score`` / ``rejected_paths``) plus any present
            ``amended_by``/``narrowed_by`` modifier stamps (this is a
            decision-read surface; an amended-but-active neighbour must not read
            as the final word) — or the :class:`Unavailable` the gather returned
            when the semantic substrate was unreachable.
        """
        if not self.embed_provider or not self.vector_store:
            return []
        gathered = gather_candidates(
            entry.axiom,
            embed_provider=self.embed_provider,
            vector_store=self.vector_store,
            store=self.store,
        )
        if isinstance(gathered, Unavailable):
            return gathered
        if gathered_index is not None:
            # Read the handle off the hydrated node, not off ``Candidate.slug``: the
            # latter is the vector-store payload's spelling (they agree in production
            # because the payload was written from the node, but they are two sources).
            gathered_index.update(
                {c.node["slug"].casefold(): (c.node["slug"], c.score) for c in gathered}
            )
        screened = screen_candidates(
            gathered,
            declared_targets=declared_targets,
            own_slug=entry.slug,
            floor=_NEIGHBOR_REVIEW_THRESHOLD,
            top_k=CONFLICT_TOP_K,
        )
        return [candidate_payload(c) for c in screened]

    def _lineage_suppression_slugs(
        self, mutation_target_slugs: List[Optional[str]]
    ) -> set[str]:
        """Casefolded slugs of the transitive mutation ancestors of declared targets (3b).

        Transitive-lineage near-duplicate suppression. When an author declares an
        ``amends``/``narrows``/``supersedes`` edge to the HEAD of a multi-link mutation
        chain, the near-dup gate (:meth:`_review_neighbors`) must not pause on a near-dup
        OLDER member reachable *through* that chain — by naming one node in a lineage the
        author has acknowledged the whole lineage, not just the node they happened to
        declare. This helper resolves each declared mutation target to its node id and
        walks :meth:`GraphStore.get_lineage` (the transitive ``supersedes`` ∪ ``amends``
        ∪ ``narrows`` ancestor walk, Phase 3a) from it, returning every ancestor's
        casefolded slug to merge into ``declared_targets`` — the suppression set
        :meth:`_review_neighbors` passes to ``screen_candidates``, whose S4 stage drops
        candidates by casefolded membership. Both sides casefold, matching that filter
        exactly. Merging
        into the set only ever *grows* suppression, so V1a's DIRECT suppression of all
        nine relation types is preserved by construction (must-not-regress, DoD #13b).

        Seed ONLY the mutation three — ``supersedes`` ∪ ``amends`` ∪ ``narrows``, which
        equals ``store._MUTATION_EDGE_FIELDS`` (the exact edge set ``get_lineage``
        walks). ``corrects`` and the other five declared relations still get DIRECT
        suppression via ``declared_targets`` but NO transitive extension: walking
        ``get_lineage`` from, say, a ``cites`` target would traverse a mutation graph
        that edge has nothing to do with — semantically wrong and over-suppressing.

        Sibling-cluster shape — design-AWARE, deliberately NOT auto-suppressed
        (Decision 4, vision §6.2). When N decisions each amend one shared root R, a new
        sibling that also amends R is near-dup to its CO-CHILDREN (R's other amenders),
        not only to R. Those co-children are R's *descendants*, NOT in ``get_lineage(R)``
        (R is *their* ancestor), so this walk does not — and must not — suppress them.
        Auto-suppressing them would mean "declaring ``amends R`` silences the gate for
        R's entire descendant cluster", hiding a GENUINE duplicate of one sibling behind
        an unrelated declaration. The vision defers the cluster's ergonomics to a later
        granular-acknowledge layer. The seam that layer would add — a SYMMETRIC
        per-neighbour check, ``any(t_id in {a["node_id"] for a in
        get_lineage(neighbour_id)} for t_id in declared_mutation_target_ids)`` gated
        behind explicit per-neighbour acknowledgement — is named here so it is
        discoverable, built later, never rediscovered as a "bug".

        Consumes ``get_lineage``'s loud-but-non-fatal partial-on-cycle tolerance as-is
        (Decision 5): over a corrupt graph the suppression degrades gracefully (it
        suppresses what was walked before the cycle bound fired, and ``get_lineage``
        emits the loud ``logger.warning``) and never hangs the hot ``record`` path. This
        helper adds no cycle handling of its own — that is the read-side "tolerate" half
        of 3a's one-walk-two-call-sites split.

        Args:
            mutation_target_slugs: The declared ``supersedes``/``amends``/``narrows``
                target slugs; any element may be ``None`` or empty (skipped).

        Returns:
            The casefolded slugs of every transitive mutation ancestor of every
            resolvable declared target; an empty set when no mutation edge was declared
            (or no target resolves to exactly one node).
        """
        suppressed: set[str] = set()
        for slug in mutation_target_slugs:
            if not slug or not slug.strip():
                continue
            ids = self.store.resolve_slug(slug)
            # Each declared target was pre-validated unique-and-existing before the gate
            # runs (supersedes at the supersedes fast-fail, amends/narrows via
            # _validate_relation_target), so resolve_slug returns exactly one id on the
            # supported path. Defensive skip on 0/>1 keeps the helper robust if reused:
            # a non-unique seed cannot meaningfully seed a single lineage walk.
            if len(ids) != 1:
                continue
            for ancestor in self.store.get_lineage(ids[0]):
                suppressed.add(ancestor["slug"].casefold())
        return suppressed

    def _node_state(self, node_id: str) -> str:
        """Returns the computed state of a node ('active'/'superseded'/'corrected'/'drifted')."""
        return self.store.get_node_state(node_id)

    def _embedding_status(self, node_id: str) -> str:
        """Reports whether a node's embedding is queued in the outbox ('pending') or done."""
        try:
            for row in self.store.get_pending_embeddings():
                if row.get("node_id") == node_id:
                    return "pending"
        except Exception:
            pass
        return "indexed"

    def _exists_receipt_extras(
        self, entry: ParsedEntry, existing: Dict[str, Any], node_id: str
    ) -> Dict[str, Any]:
        """Builds the `exists` receipt's divergence hint.

        AX round 10's actual ask, verbatim: *say what it ignored*. A re-record aimed at
        correcting commentary got a clean `(exists) ✓` and no indication that the values
        it carried differed from the stored ones — so the caller had to go and check by
        hand to learn that nothing had changed.

        Args:
            entry: The parsed entry this call built.
            existing: The committed node.
            node_id: The node's id.

        Returns:
            ``{"differs": [field, ...]}`` when the call carried different commentary
            than the graph holds, else ``{}`` — an empty list would read as a
            measurement rather than an absence.
        """
        try:
            report = entry_divergence(
                entry,
                existing,
                existing.get("scope") or [],
                self.store.get_outgoing_edges(node_id),
            )
        except Exception:
            return {}  # a receipt hint must never fail the receipt
        differs = list(report.get("commentary") or [])
        if report.get("scope"):
            differs.append("scope")
        if report.get("edges"):
            differs.append("edges")
        return {"differs": sorted(differs)} if differs else {}

    def _uncommittable_edges(self, entry: ParsedEntry,
                             divergence: Dict[str, Any]) -> List[str]:
        """Returns why each newly-declared edge provably cannot commit, if any.

        The reconcile pre-flight. A reconcile whose commit is FORECLOSED must be
        refused before the write-ahead intent row, because the commit failure then
        writes a correlated outcome row too — so every sync of an uncommittable entry
        appends two audit rows, forever, and retention is deliberately deferred.

        **Scoped to the class, not an instance.** Phase 3 shipped this as a
        `missing_target`-only check and was too narrow: a kind-illegal edge has a
        perfectly resolvable target, sailed through, and repeated indefinitely
        (measured on cartolina: 19 such edges, 6 audit rows after 3 syncs). The
        disposition of EVERY store failure code is now declared in
        ``_PREFLIGHT_DISPOSITIONS`` and pinned by ``test_preflight_covers_store_codes``,
        so a code added later cannot be silently missed the same way again.

        Refusal is deliberately conservative: it fires only when the commit is
        foreclosed for EVERY node the citation could bind to. A false positive here
        silently skips a legitimate reconcile, which is worse than an extra audit row.

        Args:
            entry: The parsed corpus entry (its ``kind`` is the edge source kind).
            divergence: The ``entry_divergence`` report.

        Returns:
            Human-readable reasons, in report order. Empty means nothing forecloses
            the commit — not that it will necessarily succeed.
        """
        added = (divergence.get("edges") or {}).get("added") or []
        reasons: List[str] = []
        for spec in added:
            edge_type, _, target = spec.partition(":")
            if not target:
                continue
            # `missing_target` / `dangling_edge` — the citation names nothing.
            target_kinds = self.store.resolve_slug_kinds(target)
            if not target_kinds:
                reasons.append(
                    f"'{target}' does not name any entry in the graph — fix the "
                    "citation in decisions.md, or restore the cited entry with "
                    "`mitos restore-source`"
                )
                continue
            # `kind_constraint_violation` — the citation resolves, but the `edges`
            # CHECK can never admit the edge. Refuse only if EVERY candidate kind is
            # illegal (a slug can name a whole supersession lineage, MI-13).
            if not any(
                edge_kind_is_legal(edge_type, entry.kind, target_kind)
                for target_kind in target_kinds
            ):
                requirement = _EDGE_KIND_REQUIREMENT.get(
                    edge_type, "satisfy the kind matrix"
                )
                reasons.append(
                    f"'{target}' has an incompatible kind for a {edge_type} edge — "
                    f"{edge_type} edges must {requirement}. Re-author the relation "
                    "(a decision citing a precedent wants `Cites:`) or remove the line"
                )
        return reasons

    def _apply_commentary_reconcile(
        self,
        entry: ParsedEntry,
        existing: Dict[str, Any],
        node_id: str,
        divergence: Dict[str, Any],
        auto_accept: bool,
        *,
        authorized: bool = False,
    ) -> bool:
        """Applies one commentary reconcile, or reports why it was skipped.

        The store has always been able to update commentary in place on a matching
        canonical core; all four callers gated the branch away, so it was reachable only
        from unit tests. This routes one caller to it — which is why Phase 3 is the
        first production consumer of MI-4 and MI-5.

        Args:
            entry: The parsed corpus entry.
            existing: The committed node, as stored.
            node_id: The node's id.
            divergence: The ``entry_divergence`` report.
            auto_accept: Whether ``--yes`` is in force.
            authorized: Whether the caller named THIS entry with
                ``--reconcile-entry``. It is an additional way to satisfy the gate
                below, never a second apply path: an authorized entry skips both
                authorization branches and falls through to the same foreclosure
                check, the same write-ahead attribution row and the same
                ``commit_parsed_entry`` a TTY ``[r]`` already reaches, so a fault
                mid-reconcile leaves exactly the state a confirmed reconcile leaves
                today. The named-target test lives at the call site, so this
                function stays ignorant of the flag's vocabulary.

        Returns:
            True if the reconcile was applied.
        """
        edges = divergence.get("edges") or {}
        removals = edges.get("removed") or []

        print(f"\n[Divergence] '{existing.get('slug')}' — the corpus and graph disagree.")
        for field in divergence.get("commentary") or []:
            print(f"  {field}:")
            print(f"    graph:    {existing.get(field)!r}")
            print(f"    markdown: {getattr(entry, field, None)!r}")
        if divergence.get("scope"):
            print(f"  scope:      graph {divergence['scope']['graph']} → "
                  f"markdown {divergence['scope']['markdown']}")
        for added in edges.get("added") or []:
            print(f"  edge ADDED:   {added}")
        for removed in removals:
            # Printed explicitly, and named as a deletion, because a re-commit mirrors
            # edges declaratively: a line removed from the markdown DELETES that edge.
            print(f"  edge DELETED: {removed}")

        # `authorized` satisfies BOTH branches below for the entry the caller named,
        # and nothing else: the fall-through is the shipped one, not a second route.
        if not authorized:
            if auto_accept and removals:
                # `--yes` widens sync from append-only to mutate, but never to delete
                # an edge unattended. And because `commit_parsed_entry` mirrors edges
                # declaratively, an entry's commentary cannot be applied while
                # withholding its edge state — so a MIXED entry is skipped whole. That
                # holds for every entry a run was NOT authorized for; naming the entry
                # is the other way to satisfy this gate, and it applies the whole
                # reconcile rather than the deletion alone.
                print("  Skipped — this run authorized no repair for this entry, and "
                      "an edge DELETION cannot be applied under --yes alone (an "
                      "entry's commentary cannot be applied while withholding its "
                      "edge state, so a mixed entry is skipped whole). To apply it, "
                      "name the entry:")
                # `entry.slug`, never `existing['slug']`: the door matches the handle
                # against the MARKDOWN slug the loop is iterating, so on a hand-edited
                # RENAME the graph's slug is the one string this recipe must not carry
                # — a caller copying it would name an entry no parsed block matches and
                # get the never-seen refusal on a target sitting in front of them.
                print(f"    mitos sync -p {self.config.project!r} "
                      f"--reconcile-entry {entry.slug!r}")
                print("  Or restore the relation line in decisions.md to reconcile "
                      "the commentary alone.")
                return False

            if not auto_accept:
                if not sys.stdin.isatty():
                    # No TTY, no `--yes` and no authorization for this entry: report
                    # and skip, never prompt. This gate fires on a corpus state that
                    # used to produce ZERO prompts (the old code `continue`d
                    # unconditionally), so prompting here turned every non-interactive
                    # `mitos sync` — an agent, a CI job, a cron, a piped invocation —
                    # into a fatal `EOF when reading a line`. Skipping is the
                    # fail-closed choice: the absence of a terminal is not an
                    # authorization to mutate. What IS one is `--yes` (for an entry
                    # carrying no edge deletion) or naming this entry.
                    print("  Skipped — no terminal to confirm on, and this run "
                          "authorized no repair for this entry. To apply it "
                          "non-interactively, name it:")
                    # The markdown slug, for the reason the sibling refusal states.
                    print(f"    mitos sync -p {self.config.project!r} "
                          f"--reconcile-entry {entry.slug!r}")
                    return False
                choice = input("Reconcile the graph to the markdown? "
                               "[r]econcile / [s]kip: ").strip().lower()
                if choice != "r":
                    print("  Skipped.")
                    return False

        uncommittable = self._uncommittable_edges(entry, divergence)
        if uncommittable:
            print("  Skipped — cannot reconcile; this commit is foreclosed, so no "
                  "attribution row was written:", file=sys.stderr)
            for reason in uncommittable:
                print(f"    - {reason}.", file=sys.stderr)
            return False

        # Write-AHEAD: the attribution row lands BEFORE the mutation. A best-effort
        # write-after can leave a mutation with no attribution, which is undetectable
        # and exactly what P8 forbids; write-ahead leaves at worst a phantom row for a
        # change that never landed, and a phantom never violates P8 — it forbids
        # unattributed mutations, not attributed non-mutations.
        stored_edges = self.store.get_outgoing_edges(node_id)
        prior, new_values = self._reconcile_value_pair(
            entry, existing, divergence, stored_edges
        )
        audit_id = uuid.uuid4().hex
        try:
            telemetry = TelemetryStore(self.config.telemetry_path)
            telemetry.record_commentary_intent(
                CommentaryAuditRow(
                    audit_id=audit_id,
                    node_id=node_id,
                    slug=existing.get("slug"),
                    fields_changed=sorted(new_values),
                    prior_values=prior,
                    new_values=new_values,
                    mitos_version=MITOS_VERSION,
                ),
                created_at=_utc_now_iso(),
            )
        except Exception as exc:
            # MANDATORY writer, inverting this store's ambient best-effort posture: a
            # telemetry failure must never abort a sync of real decisions, but a
            # P8-mandatory row cannot inherit that. An unauditable reconcile is REFUSED,
            # not applied unaudited — otherwise P8 holds only when the disk cooperates.
            print(f"  Skipped — cannot write the attribution row ({exc}); "
                  "refusing to reconcile unaudited.", file=sys.stderr)
            return False

        # Carry the STORED confirmation pair forward. Both columns sit in the store's
        # commentary UPDATE SET, and the parsed entry carries `confirmed_by=None` here
        # (sync's "user" stamp happens further down the commit path) — so a naive
        # reconcile NULLs them and one reusing sync's stamping rewrites them to "user".
        # Measured: either destroys 114 stamps on a change that touched only prose.
        entry.confirmed_by = existing.get("confirmed_by")
        entry.confirmed_at = existing.get("confirmed_at")

        # Withhold the transcript. `commit_parsed_entry` UPDATEs `transcripts` whenever
        # the incoming text is non-None and differs — but the transcript is NOT in the
        # divergence set, so it appears in no printed diff, no `fields_changed`, and no
        # `prior_values`. Left attached it RIDES ALONG on any reconcile triggered by
        # another species: an agent that edits `**Rejected:**` and rewrites the
        # transcript block gets both applied, the operator pressing `[r]` sees only
        # `rejected_paths` named, and the prior transcript then exists nowhere — markdown
        # rewritten, graph updated, audit row silent. That is exactly the unattributed
        # graph mutation P8 forbids, inside the feature that adds the attribution row.
        # `None` means no-change to the store, which restores the documented
        # write-once-preserved semantics; transcript reconciliation is homed in the
        # future `amend-commentary` verb along with its own attribution.
        entry.transcript = None

        try:
            self.store.commit_parsed_entry(entry)
        except CommitError as exc:
            # Reported per-entry and moved past — NEVER routed to the quarantine
            # fixpoint. A retry cannot help: nothing later in this pass changes the
            # outcome, and MI-13's FM2 (dropping a `Supersedes:` line resurrects a
            # predecessor into a slug_collision rollback) would re-fail every pass.
            reason = "; ".join(i.message for i in exc.failure.items) if exc.failure else str(exc)
            print(f"  Reconcile FAILED for '{existing.get('slug')}': {reason}",
                  file=sys.stderr)
            # Append-only closure: the intent row stays and a correlated outcome row
            # records the failure. Telemetry never UPDATEs or DELETEs, and this write
            # IS best-effort — the mutation did not happen, so there is no unattributed
            # mutation to hide.
            try:
                telemetry.record_commentary_outcome(
                    audit_id=uuid.uuid4().hex,
                    correlates_to=audit_id,
                    outcome=f"failed: {reason}",
                    created_at=_utc_now_iso(),
                    mitos_version=MITOS_VERSION,
                )
            except Exception:
                pass
            return False

        print(f"  Reconciled '{existing.get('slug')}' ✓")
        return True

    @staticmethod
    def _reconcile_value_pair(
        entry: ParsedEntry,
        existing: Dict[str, Any],
        divergence: Dict[str, Any],
        stored_edges: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Builds the audit row's ``(prior, new)`` value pair.

        Both sides, not just the prior: self-detection of a crash-phantom compares the
        NEW values against the graph, and chain consistency compares a successor's
        ``prior`` against its predecessor's ``new``.

        Args:
            entry: The parsed corpus entry.
            existing: The committed node.
            divergence: The ``entry_divergence`` report.
            stored_edges: The node's outgoing edges as stored, so the edge columns can
                record full state rather than a delta.

        Returns:
            The ``(prior_values, new_values)`` pair, keyed identically.
        """
        prior: Dict[str, Any] = {}
        new_values: Dict[str, Any] = {}
        for field in divergence.get("commentary") or []:
            prior[field] = existing.get(field)
            new_values[field] = getattr(entry, field, None)
        if divergence.get("scope"):
            prior["scope"] = divergence["scope"]["graph"]
            new_values["scope"] = divergence["scope"]["markdown"]
        if divergence.get("edges"):
            # FULL STATE on both sides, not the delta. `prior["edges"] = removed` reads
            # as "the graph held exactly these", which is false whenever the node keeps
            # an edge — and the documented read rule (an intent row over a graph
            # matching its `new_values` means applied) would then mismatch and report a
            # successfully-applied reconcile as NOT applied, inverting the P8 verdict.
            stored = _edge_state_labels(stored_edges)
            declared = _edge_state_labels(declared_edges(entry))
            prior["edges"] = stored
            new_values["edges"] = declared
        return prior, new_values

    def splice_buffer(
        self,
        transform: Callable[[str], str],
        *,
        after_write: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Rewrites `decisions.md` under the lock, rolling back on any failure.

        The buffer-surgery primitive: **lock → auto-heal → read → splice → write →
        verify → roll back on failure.** Extracted rather than inlined at its first
        caller because it is exactly what a future ``amend-commentary`` verb consumes
        — the P20 Retrofit Test allows deferring that verb only on the condition that
        its wiring is not left to be retrofitted.

        Modelled on ``record_decision_entry``'s buffer-first + rollback contract,
        which this project treats as sacred. That method is deliberately NOT refactored
        to route through here: it interleaves the graph commit with the buffer write in
        a way this generic seam would have to special-case, and rewriting a sacred
        contract to serve a new caller is the wrong direction of dependency. The
        equivalence is pinned by test instead.

        Args:
            transform: Maps the current buffer text to its replacement. Runs before the
                splice write, so a raise here writes no entry — though note the
                auto-heal above it may already have restored a drifted header, which is
                an idempotent repair and deliberately outside the rollback. This is the
                one ordering difference from ``record_decision_entry``, which runs its
                gates BEFORE auto-heal precisely so a rejected record leaves the buffer
                byte-for-byte untouched; here the transform NEEDS the healed marker to
                splice against.
            after_write: Optional verification invoked with the written text. A raise
                rolls the buffer back — this is where a fidelity check belongs, so a
                splice that would corrupt a neighbour never survives.

        Returns:
            The buffer text as written.

        Raises:
            MitosError: If the rollback ITSELF fails, naming the state the file is in —
                the one case where silence would leave an operator guessing.
            Exception: Whatever ``transform`` or ``after_write`` raised, re-raised after
                a successful rollback.
        """
        with self.lock:
            self.auto_heal_decisions_file()
            with open(self.config.decisions_file, "r", encoding="utf-8") as fh:
                original = fh.read()

            # Computed before the write, so a transform failure is a pure no-op.
            new_content = transform(original)

            try:
                with open(self.config.decisions_file, "w", encoding="utf-8") as fh:
                    fh.write(new_content)
                if after_write is not None:
                    after_write(new_content)
            except Exception:
                try:
                    with open(self.config.decisions_file, "w", encoding="utf-8") as fh:
                        fh.write(original)
                except Exception as restore_exc:
                    raise MitosError(
                        "The splice failed AND decisions.md could not be rolled back "
                        f"(rollback error: {restore_exc}). The file may hold a partial "
                        "edit — check it before running `mitos sync`."
                    ) from restore_exc
                raise
            return new_content

    def record_decision_entry(
        self,
        axiom: str,
        rejected_paths: str,
        scope: List[str],
        mechanisms: Optional[List[str]] = None,
        context: Optional[str] = None,
        supersedes: Optional[str] = None,
        corrects: Optional[str] = None,
        amends: Optional[str] = None,
        narrows: Optional[str] = None,
        depends_on: Optional[str] = None,
        resolves: Optional[str] = None,
        contradicts: Optional[str] = None,
        derives_from: Optional[str] = None,
        cites: Optional[str] = None,
        slug: Optional[str] = None,
        actor: str = "agent",
        acknowledge_neighbors: bool = False,
    ) -> Dict[str, Any]:
        """Records a single decision into the buffer and graph, non-interactively.

        The agentic write half of Mitos: persists one deliberate, pre-structured
        decision (and the alternatives it rejected) the moment an agent makes it,
        without LLM enrichment and without calling ``perform_sync``. Composes the
        existing primitives only — ``commit_parsed_entry`` for the graph and
        ``_best_effort_embed`` for the vector (preserves M7).

        Validation runs entirely in memory FIRST (Phase A); the buffer append and
        the graph commit happen together, last, under a single lock, with the
        buffer rolled back if the commit fails (Phase B). The contract: on any
        error code, ``decisions.md`` is byte-for-byte unchanged and nothing is
        committed.

        Args:
            axiom: The decision as a single clear sentence true going forward (M1).
            rejected_paths: The alternatives considered and rejected, and why (M5, required).
            scope: Area tags (may be empty).
            mechanisms: Concrete technologies/entities involved (M6).
            context: Optional background on why this was decided.
            supersedes: Optional exact slug of a prior decision this one replaces.
            corrects: Optional exact slug of a prior decision this one corrects (a
                kill-edge twin of supersedes — the target leaves the active view).
            amends: Optional exact slug of a decision this one amends.
            narrows: Optional exact slug of a decision this one narrows.
            depends_on: Optional exact slug of a decision this one depends on.
            resolves: Optional exact slug of an open question this resolves (the
                ``resolves`` edge is decision→open_question only).
            contradicts: Optional exact slug of a decision this one contradicts.
            derives_from: Optional exact slug of a decision this one derives from.
            cites: Optional exact slug of a decision this one cites.
            slug: Optional explicit slug; derived deterministically from axiom if None.
            actor: Provenance, stored in ``confirmed_by``.
            acknowledge_neighbors: Skip the pre-commit near-duplicate review and record
                even when a highly-similar unreferenced decision exists (P4). Pass True
                to commit a genuinely independent decision past the pause.

        Returns:
            A success dict ``{slug, id, state, embedding, status}`` (status
            "created"|"exists"); on the "created" path it also carries
            ``edges_created`` (the edges the commit actually wired, each
            ``{kind, target}`` — write facts read back from the store, not an
            echo of the input args), the resolved ``scope`` and ``mechanisms``
            as committed, plus an optional
            ``neighbor_review_unavailable`` notice when the pre-commit near-dup
            check could not run (the record fails open — the commit proceeds
            unchecked; the notice names the cause and no command, each surface
            composing its own recovery), plus an always-present
            ``coherence_audit`` statement of the corpus's standing, cumulative
            contradiction-check debt; OR, when a highly-similar unreferenced decision exists and
            ``acknowledge_neighbors`` is False, a ``{status: "needs_review", code:
            "similar_decision_exists", slug, neighbors, message}`` pause that wrote
            NOTHING —
            each ``neighbors`` element is an enriched, modifier-stamped decision-read
            payload (:func:`~mitos.conflict.candidate_payload`: ``slug`` / ``axiom`` /
            ``scope`` / ``score`` / ``rejected_paths`` plus any ``amended_by``/
            ``narrowed_by`` stamps) the authoring agent judges tenability from. That
            pause also echoes the caller's own declared relation targets, partitioned
            by :func:`_declared_echo` and present only when non-empty: ``declared``
            (every target it typed, canonical spellings) and
            ``declared_no_near_match`` (``{slug[, score]}`` for the pause-resolving
            declarations that moved nothing on this call), each with a
            ``*_total`` sibling count when the group collapsed at
            :data:`_DECLARED_ECHO_BOUND`;
            OR a structured ``{error, code}`` failure (see spec §5).
        """
        # === Phase A — validate everything in memory (no writes) ===

        # 1. Preconditions. os.path.exists, NOT manager construction (which creates the db).
        if not os.path.exists(self.config.decisions_file):
            return _record_error("not_initialized")

        # 2. Normalise CRLF, then validate. A stray \r perturbs the field regex and
        #    the canonical-core hash (same decision would hash differently across
        #    environments — compute_node_id normalizes, but normalize at the boundary).
        axiom = (axiom or "").replace("\r\n", "\n").replace("\r", "\n")
        rejected_paths = (rejected_paths or "").replace("\r\n", "\n").replace("\r", "\n")
        if context is not None:
            context = context.replace("\r\n", "\n").replace("\r", "\n")

        if not axiom.strip():
            return _record_error("empty_axiom")
        if not rejected_paths.strip():
            return _record_error("missing_rejected_paths")

        # 3. Reject (do NOT sanitise) content fields carrying structural tokens.
        #    Check the NON-stripped values: a leading-whitespace `  ## heading` is
        #    safe (the parser only splits on column-0 `##`), and stripping it first
        #    would falsely promote it to column 0. Storage stripping happens after.
        for field_text in (axiom, rejected_paths, context):
            if field_text and _contains_structural_token(field_text):
                return _record_error("parse_failed")

        axiom = axiom.strip()
        rejected_paths = rejected_paths.strip()
        context = context.strip() if context and context.strip() else None
        mechanisms = [m.strip() for m in mechanisms if m and m.strip()] if mechanisms else []
        scope = [s.strip() for s in scope if s and s.strip()] if scope else []
        if supersedes is not None:
            supersedes = supersedes.strip() or None
        if corrects is not None:
            corrects = corrects.strip() or None

        # Normalise the other typed relations into a stable-ordered map (supersedes and
        # corrects are handled separately — both are kill-edges that change computed
        # state and carry bespoke error codes).
        _provided = {
            "amends": amends, "narrows": narrows, "depends_on": depends_on,
            "resolves": resolves, "contradicts": contradicts,
            "derives_from": derives_from, "cites": cites,
        }
        extra_relations: Dict[str, str] = {}
        for _name, _label in _EXTRA_RELATIONS:
            _val = _provided.get(_name)
            if _val and _val.strip():
                extra_relations[_name] = _val.strip()

        # record_decision always mints a `decision`, but a `derives_from` edge must
        # ORIGINATE from an open question (open_question -> decision) — a decision can
        # never be its source, so this relation is always invalid here. Reject it in
        # the validate phase with a redirect, rather than letting it pass validation
        # and fail only at the store's kind CHECK ("validated but the commit failed").
        if "derives_from" in extra_relations:
            return _record_error("derives_from_on_decision")

        # 4. Slug — validate, don't mangle. The slug is now mandatory and explicit, and
        #    it is folded into the canonical-core identity (V1-D2), so it is permanent
        #    once committed. Normalise case/separators, but REJECT an over-length slug
        #    with an exact char count rather than silently truncating it — a silent trim
        #    would diverge the stored handle from the one the author already cited
        #    (self-inflicted citation rot, the exact failure the handle subsystem exists
        #    to prevent).
        slug = _normalize_slug(slug)
        if not slug:
            return _record_error("empty_slug")
        if len(slug) > _SLUG_MAX_LEN:
            return _record_error(
                "slug_too_long",
                slug=slug,
                length=len(slug),
                over=len(slug) - _SLUG_MAX_LEN,
                max=_SLUG_MAX_LEN,
            )

        # 5. Serialise to the canonical format (in memory only).
        lines = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected_paths}"]
        if mechanisms:
            lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
        if scope:
            lines.append(f"**Scope:** {', '.join(scope)}")
        if context:
            lines.append(f"**Context:** {context}")
        if supersedes:
            lines.append(f"**Supersedes:** {supersedes}")
        if corrects:
            lines.append(f"**Corrects:** {corrects}")
        for _name, _label in _EXTRA_RELATIONS:
            if _name in extra_relations:
                lines.append(f"**{_label}:** {extra_relations[_name]}")
        entry_text = "\n".join(lines) + "\n"

        # 6. Parse our entry back through the V1a tokenizer (sets .axiom/.topic +
        #    the relationship attrs), then run the graph-level checks as a read-only
        #    fast-fail. STRICT mode (no collector): a malformed self-serialized entry
        #    raises ParseError, which we map to the structured parse_failed code (G2).
        try:
            parsed = parse_entry_stream(entry_text, "decision")
        except ParseError:
            return _record_error("parse_failed")
        if len(parsed) != 1:
            return _record_error("parse_failed")
        entry = parsed[0]

        # Pre-validate supersedes with an EXACT match — comma-separated for a
        # multi-target supersede (each slug resolved independently; a lone slug is the
        # 1-element common case). Phase A read-only fast-fail: a miss on ANY target
        # returns an error naming that slug and writes nothing.
        # Each loop below also retains the resolved node's OWN slug spelling, keyed on
        # the casefolded caller spelling. That is the pause echo's canonical handle for
        # every declared target — including the ones the neighbour sweep never gathers,
        # whose stored spelling is computed nowhere else — and it costs a dict write on
        # a node already fetched.
        canonical_slugs: Dict[str, str] = {}
        supersedes_slugs = _split_relation_slugs(supersedes)
        for _sup in supersedes_slugs:
            ids = self.store.resolve_slug(_sup)
            if not ids:
                return _record_error("supersedes_not_found", supersedes=_sup)
            if len(ids) > 1:
                return _record_error("supersedes_ambiguous", supersedes=_sup)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != _sup.casefold():
                return _record_error("supersedes_not_found", supersedes=_sup)
            canonical_slugs[_sup.casefold()] = target["slug"]
        if supersedes_slugs:
            entry.supersedes = supersedes_slugs  # List[str] shape (V1b multi-valued)

        # Pre-validate corrects with an EXACT match — the kill-edge twin of supersedes
        # (V1a's second kill-edge; the target leaves the active view), comma-separated
        # for multiple. Same Phase-A read-only fast-fail shape, so a miss writes nothing.
        corrects_slugs = _split_relation_slugs(corrects)
        for _cor in corrects_slugs:
            ids = self.store.resolve_slug(_cor)
            if not ids:
                return _record_error("corrects_not_found", corrects=_cor)
            if len(ids) > 1:
                return _record_error("corrects_ambiguous", corrects=_cor)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != _cor.casefold():
                return _record_error("corrects_not_found", corrects=_cor)
            canonical_slugs[_cor.casefold()] = target["slug"]
        if corrects_slugs:
            entry.corrects = corrects_slugs  # List[str] shape (V1b multi-valued)

        # Validate every other typed relation EXACTLY like supersedes — each must point
        # at a real, unambiguous decision; each is comma-separated multi-valued too.
        # Still Phase A: a miss on any target returns an error and writes nothing, so
        # the buffer stays byte-for-byte unchanged.
        for _name, _raw in extra_relations.items():
            _targets = _split_relation_slugs(_raw)
            for _t in _targets:
                err = self._validate_relation_target(
                    _name, _t, canonical_slugs=canonical_slugs)
                if err:
                    return err
            if _targets:
                setattr(entry, _name, _targets)  # List[str] shape (V1b multi-valued)

        # Identity (slug-free canonical-core hash — V1-D2). Computed over the SAME
        # fields commit_parsed_entry hashes, so this pre-commit idempotency id equals
        # the commit id: a same-core re-record with a new --slug is an in-place UPDATE
        # (slug rename), never a spurious slug_collision (G3, V1-D16).
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )

        # Idempotency (M2) fast-fail.
        existing = self.store.get_node(node_id)
        if existing:
            # Cross-source re-encounter audit (MI-4 / V1-D14) before the exists
            # short-circuit returns. record authors no **Source:** line, so on this
            # path entry.source is None → "user". Best-effort (note_source_reencounter
            # never raises), so the agentic exists-path can never crash on the audit.
            self.store.note_source_reencounter(
                node_id, existing["source"], entry.source or "user"
            )
            return {
                "slug": existing["slug"],
                "id": node_id,
                "state": self._node_state(node_id),
                "embedding": self._embedding_status(node_id),
                "status": "exists",
                # `path` stays: it points at where the existing entry LIVES, which is
                # the useful answer to "already recorded — so where is it?" (#5b).
                # What was wrong is that the renderer labelled it `Written:` under a
                # `Recorded ✓` headline, so a re-record aimed at correcting commentary
                # — or at restoring a source block on a graph-only node — read as a
                # successful write while nothing had changed. The claim moves to
                # no_op_reason; the pointer stays.
                "path": self.config.decisions_file,
                "no_op_reason": _EXISTS_NO_OP_NOTE,
                **self._exists_receipt_extras(entry, existing, node_id),
            }

        # Slug-collision fast-fail (exact match only).
        coll_id, _coll_node = self._exact_slug_node(entry.slug)
        if coll_id and coll_id != node_id:
            return _record_error("slug_collision", slug=entry.slug)

        # Near-duplicate review (P4) — still Phase A, so a pause writes NOTHING
        # (buffer byte-for-byte unchanged). The neighbours payload is a stamped
        # decision-read surface (candidate_payload) the agent judges from. Surfacing
        # BEFORE commit is the point: after commit the author can no longer point a
        # relation at it (a re-record is a no-op). Offline-safe
        # (no embeddings → no pause) and bypassable with acknowledge_neighbors=True.
        # The record surface fails OPEN on any pause-read fault: the check degrading
        # must never block the commit, but "couldn't check" must not read as "checked,
        # clean" either — the threaded notice lands on the created receipt (step 10).
        review_unavailable: Optional[str] = None
        if not acknowledge_neighbors:
            neighbors: List[Dict[str, Any]] = []
            # The caller's own declarations, per relation, for the pause echo — built
            # BESIDE the declared_targets comprehension below, which folds the relation
            # name away. Purely additive: nothing that Phase B reads moves in here.
            # `test_mixed_neighbors_declared_edge_survives_acknowledge_bypass` documents
            # why that matters — this whole block is skipped under acknowledge_neighbors
            # while Phase B still writes edges from the raw relation args, so relocating
            # any normalization into it drops declared edges when both flags travel
            # together, with every other row green.
            declared_by_relation: Dict[str, Optional[str]] = {
                "supersedes": supersedes, "corrects": corrects, **extra_relations,
            }
            # This call's raw gathered set, filled by _review_neighbors. Primary-set
            # input to the partition (M8); empty when the sweep never ran.
            gathered_index: Dict[str, Tuple[str, float]] = {}
            try:
                declared_targets = {
                    t.casefold()
                    for raw in ([supersedes, corrects] + list(extra_relations.values()))
                    for t in _split_relation_slugs(raw)
                }
                # Transitive-lineage suppression (3b): also suppress the pause for the
                # OLDER members of an amends/narrows/supersedes chain reachable through a
                # declared mutation edge — by naming the chain head the author has
                # acknowledged the whole lineage (consuming get_lineage from 3a). Seed only
                # the mutation three (corrects + the other five stay direct-only). Skip the
                # walk when the gate is a no-op (offline → _review_neighbors returns []);
                # the augmentation only ever GROWS the set, so direct suppression of all
                # nine types is preserved (DoD #13b). The walk is suppression-only: it does
                # NOT re-declare amends/narrows (that serves modifier stamping, a different
                # consumer — §6.2), so a bridged predecessor still reads un-amended.
                if self.embed_provider and self.vector_store:
                    declared_targets |= self._lineage_suppression_slugs(
                        _split_relation_slugs(supersedes)
                        + _split_relation_slugs(extra_relations.get("amends"))
                        + _split_relation_slugs(extra_relations.get("narrows"))
                    )
                reviewed = self._review_neighbors(entry, declared_targets,
                                                  gathered_index=gathered_index)
            except (DatabaseError, ValidationError) as exc:
                # A graph-store fault during the pause read (gather's node reads or the
                # lineage walk). Fail open — a fault this severe fails Phase B anyway,
                # where commit-arbitration belongs; a pre-commit crash never does.
                # Exactly these two classes: anything else is a real bug and propagates.
                review_unavailable = _review_unavailable_notice("graph read failed")
                # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
                print(f"[Warning] Neighbor review skipped for '{entry.slug}': {exc}",
                      file=sys.stderr)
            else:
                if isinstance(reviewed, Unavailable):
                    # Embed/vector-store degradation, typed by the gather. The reason
                    # words the notice; `.detail` is logging-only, never rendered.
                    review_unavailable = _review_unavailable_notice(
                        _REVIEW_UNAVAILABLE_CAUSES.get(
                            reviewed.reason, reviewed.reason.value.replace("_", " ")
                        )
                    )
                else:
                    neighbors = reviewed
            if neighbors:
                # The declared-edge echo (A2). mitos is stateless across calls, so a
                # pause on a different node cannot say "since your last attempt" — and
                # a caller that cannot tell whether its declaration registered mints a
                # second, false edge into the gold source. The one thing that answers
                # it is what this call already computed and threw away.
                echo = _declared_echo(declared_by_relation, canonical_slugs,
                                      gathered_index, _NEIGHBOR_REVIEW_THRESHOLD)
                # Ahead of the closing sentence, so the commitment retraction still
                # closes the body.
                echo_prose = "".join(f"{line}. " for line in _declared_echo_lines(echo))
                return {
                    "status": "needs_review",
                    "code": "similar_decision_exists",
                    "slug": entry.slug,
                    "neighbors": neighbors,
                    "message": (
                        f"Paused: '{entry.slug}' is ≥{_NEIGHBOR_REVIEW_THRESHOLD:.2f} "
                        f"similar to {len(neighbors)} existing decision(s) you did not "
                        "reference. Judge each neighbour from its axiom, "
                        "rejected_paths, scope, and modifier stamps, then re-record "
                        "with that judgment: a relation arg "
                        f"({'/'.join(_PAUSE_RESOLVING_RELATIONS)}) pointing at any "
                        "neighbour this decision genuinely relates to, "
                        "acknowledge_neighbors=True for neighbours that stand "
                        "independently alongside it — or both at once for a mixed "
                        "set. An amended_by/narrowed_by stamp means the neighbour "
                        f"has moved on — dereference that slug before linking. {echo_prose}"
                        "Nothing was written."
                    ),
                    **echo,
                }

        # === Phase B — the only writes, fully serialised under one lock ===
        try:
            with self.lock:
                # Re-run the authoritative state checks INSIDE the lock, BEFORE any
                # buffer write (closes the TOCTOU window: a racer that committed
                # since Phase A is now seen — and a rejection here still leaves the
                # buffer byte-for-byte unchanged, since auto-heal hasn't run yet).
                existing = self.store.get_node(node_id)
                if existing:
                    # TOCTOU re-check: a racer committed this node between Phase A
                    # and here. Same cross-source audit as gate 3 (rare; INSERT OR
                    # IGNORE makes any Phase-A/Phase-B overlap a harmless no-op).
                    # Best-effort, and the threading lock here is not a DB txn, so
                    # write_signal's own connection is fine.
                    self.store.note_source_reencounter(
                        node_id, existing["source"], entry.source or "user"
                    )
                    return {
                        "slug": entry.slug,
                        "id": node_id,
                        "state": self._node_state(node_id),
                        "embedding": self._embedding_status(node_id),
                        "status": "exists",
                        # See the gate-3 twin above: the path points, it does not claim.
                        "path": self.config.decisions_file,
                        "no_op_reason": _EXISTS_NO_OP_NOTE,
                        **self._exists_receipt_extras(entry, existing, node_id),
                    }
                coll_id, _coll_node = self._exact_slug_node(entry.slug)
                if coll_id and coll_id != node_id:
                    return _record_error("slug_collision", slug=entry.slug)

                # Guarantee header + marker, then anchor the buffer for rollback.
                self.auto_heal_decisions_file()
                with open(self.config.decisions_file, "r", encoding="utf-8") as f:
                    original_content = f.read()

                # Compute the new buffer (newest-first, replacing ONLY the first marker).
                if _ENTRIES_MARKER in original_content:
                    new_content = original_content.replace(
                        _ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{entry_text}", 1
                    )
                else:
                    new_content = original_content.rstrip("\n") + f"\n\n{entry_text}"

                # Provenance (mirror perform_sync).
                entry.confirmed_by = actor
                entry.confirmed_at = _utc_now_iso()  # MI-10, as in perform_sync above

                # Write the buffer, then commit the graph. On ANY failure of either
                # — including an OSError on the write itself — roll the buffer back
                # so a failure leaves NO orphan entry, and return JSON (never raise).
                try:
                    with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    delta = self.store.commit_parsed_entry(entry)
                except (ValidationError, DatabaseError, OSError, CommitError) as commit_exc:
                    # Roll the buffer back so a failed write/commit leaves NO orphan
                    # (the sacred buffer-first + rollback contract). ``CommitError``
                    # carries the store-stage failure envelope — as of V1b a declared
                    # relation can fail at commit (e.g. a decision-source ``resolves``/
                    # ``derives_from`` violates the edge kind matrix → a loud
                    # ``kind_constraint_violation``, where V1a silently warn-deferred
                    # it). Surface the per-item messages so the agent sees the actionable
                    # field (P3 vector error), not a generic wall.
                    try:
                        with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                            f.write(original_content)
                    except Exception as restore_exc:
                        return {
                            "error": (
                                "The commit failed AND decisions.md may still hold the "
                                f"uncommitted entry (rollback error: {restore_exc}) — run "
                                f"'mitos sync' to reconcile. Underlying commit error: {commit_exc}."
                            ),
                            "code": "commit_failed",
                        }
                    reason = str(commit_exc)
                    if isinstance(commit_exc, CommitError) and commit_exc.failure:
                        reason = "; ".join(
                            i.message for i in commit_exc.failure.items
                        ) or reason
                    return _record_error("commit_failed", reason=reason)
        except Timeout:
            return _record_error(
                "commit_failed", reason="another Mitos process holds the decisions.md lock"
            )

        # 8. Embed best-effort (queues to the outbox if Gemini/Qdrant are down).
        try:
            self._best_effort_embed(delta, entry)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Embedding step failed for '{entry.slug}': {str(e)}", file=sys.stderr)

        # 9. Re-render live_axioms.md (a render failure must not fail the commit).
        #    The renderer records size-ceiling overflows on `.overflows` instead of
        #    printing them, so we can attach ONE debounced summary to the result below
        #    — after the success receipt — rather than burying it under a per-file wall.
        overflow_summary: Optional[str] = None
        try:
            renderer = MitosRenderer(self.config.workspace_dir)
            renderer.render_all(self.store)
            if renderer.overflows and hint_due(
                "scope_overflow_hint.json", self.config.workspace_dir, 24 * 60 * 60
            ):
                overflow_summary = summarize_overflows(renderer.overflows)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Failed to render active axioms: {str(e)}", file=sys.stderr)

        # 10. Return. A freshly recorded decision is always active. Everything below is
        #     post-commit read-back — the write contract is untouched (the commit
        #     already succeeded above).
        result: Dict[str, Any] = {
            "slug": entry.slug,
            "id": node_id,
            "state": "active",
            "embedding": self._embedding_status(node_id),
            "status": "created",
            "path": self.config.decisions_file,
            # Write FACTS read back from the committed node — what the commit actually
            # wired/stored, never re-derived from the author's input args. Deliberately
            # unstamped: write facts are not a decision-read surface — a reader acting on
            # a target dereferences it by slug. (Contrast the pause `neighbors`, which IS
            # stamped: ADR `record-pause-neighbors-are-stamped-decision-read-surface`.)
            "edges_created": self.store.get_outgoing_edges(node_id),
            "scope": entry.scope,
            "mechanisms": entry.mechanisms,
            # Unconditional, and only here: a write that landed incurred coherence
            # debt, while `exists`/`needs_review`/every error wrote nothing and owe
            # nothing. Carrying the FACT on the field (rather than leaving the CLI
            # renderer to decide) is what keeps the shared text tail from leaking
            # the line onto a no-op — the renderer gates on the key's presence.
            "coherence_audit": _COHERENCE_AUDIT_NOTE,
        }
        # Honest degradation (KDD-4): when the pre-commit near-dup check could not run,
        # say so on the receipt — one calm sentence, only on a genuinely failed check
        # (never on clean-empty, unconfigured, or acknowledged-past records).
        if review_unavailable:
            result["neighbor_review_unavailable"] = review_unavailable
        # Debounced, presentation-only: a one-line "N files over the size ceiling — run
        # `mitos status`" nudge, never on the burying-the-receipt critical path. Shared
        # by both surfaces (CLI prints it after the receipt; MCP returns it structured).
        if overflow_summary:
            result["scope_overflow"] = overflow_summary
        return result
