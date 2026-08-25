"""CLI entry point for Mitos.

This module implements the command-line interface for Mitos, coordinating
initialization, sync, ambient capture, querying, list, render, import, and
MCP serving.
"""

import sys
import os
import re
import time
import json
import uuid
import sqlite3
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Callable, List, Mapping, Optional, Dict, Any, Set, Tuple
from google import genai

from mitos import __version__
from mitos import check
from mitos import overview
from mitos import registry
from mitos import routing
from mitos.display import (
    apply_stdout_text_safety,
    blackout_note,
    clamp_limit,
    dumps_display,
    letter_payload,
    oneline_axiom,
    oneline_payload,
    order_scope_counts,
    projects_payload,
    truncate_words,
    resolve_display_ensure_ascii,
    show_payload,
)
from mitos.config import (
    MitosConfig,
    CONFIG_DEFAULTS,
    default_collection_name,
    global_env_path,
    toml_scalar,
)
from mitos.errors import (
    MitosError, ParseError, ValidationError, DatabaseError, ConfigError,
    VectorStoreError, CollectionMissingError, EmbeddingError,
    ProjectTargetingError, RegistryError,
    EXEMPT_CREATES_REGISTRATION, EXEMPT_EXPLICITLY_GLOBAL, EXEMPT_NO_WORKSPACE,
    TARGET_EXEMPT_VERB, TARGET_MISSING, TARGET_PATH_NOT_A_WORKSPACE,
    TARGET_RELATIVE_PATH, TARGET_UNKNOWN_NAME,
)
from mitos.divergence import corpus_graph_divergence, divergence_total
from mitos.env import resolve_key
from mitos.vector_store import scroll_point_ids, hash_to_uuid, QdrantVectorStore
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.telemetry import TelemetryStore, ConflictCheckRow, JudgmentBatch
from mitos.identity import compute_node_id
from mitos.models import get_embedding_model_id, get_model_id
from mitos.parser import (ParsedEntry, corpus_has_entries, parse_entry_stream,
                          read_text_or_none)
from mitos.conflict import (run_conflict_check, ConflictUnavailableReason,
                            SEMANTIC_SUBSTRATE_REASONS)
from mitos.migrations import is_pre_v1a_schema
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS, open_connection
from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate
from mitos.lexical import degraded_reason_from_error, lexical_fallback
from mitos.recall import (assess_query_recall, assess_surface_recall,
                          corpus_provenance, missing_graph_is_a_gap,
                          missing_graph_note, missing_index_is_a_gap,
                          provenance_line, scope_filter_recovery)
from mitos.sync import (MitosSyncManager, run_ambient_capture, _SLUG_MAX_LEN,
                        _ENTRIES_MARKER, _PAUSE_RESOLVING_RELATIONS,
                        _declared_echo_lines, _split_relation_slugs)
from mitos._agent_block import agent_block, agent_block_drift, AGENT_GUIDE_VERSION
from mitos.renderer import MitosRenderer, overflow_report
from mitos.importer import MitosProseImporter


# Worked-examples block rendered at the foot of `mitos --help` / `mitos -h`.
# Lazy by design (§6 description economy): zero context cost until --help is
# invoked, so it can afford the expansive examples the eager MCP descriptions
# can't. Teaches the surface→record reflex, the scope vocabulary, workspace
# targeting (-p), and the relation-edge guidance. Every flag/verb shown here is
# real CLI surface — keep it runnable, never invent a flag (there is no
# --retired edge). Rendered verbatim via RawDescriptionHelpFormatter.
#
# EVERY worked example names its project, because since 5a every verb below
# requires one. This block is 5,000 lines from the parser that renders it, so an
# audit scoped to "help text" misses it — grep the string, not the parser.
_EPILOG = """\
Examples:
  # Before deciding: surface precedent, then record the outcome you chose
  # (`-p` names the project: a registered name — see `mitos projects` — or an
  #  absolute path. It is accepted on either side of the verb.)
  mitos -p myproject surface "cache invalidation strategy"
  mitos -p myproject record "Write-through cache for session data" \\
    --rejected "write-back: data loss on crash" --scope cache --slug write-through-sessions

  # Discover the scope vocabulary before you invent a near-duplicate tag
  mitos -p myproject scopes

  # What does this machine have? (the only workspace-free report)
  mitos status

  # Operate on a workspace by path, from anywhere
  mitos -p /path/to/repo list --scope auth

Relating a decision to a prior (pass the prior's EXACT slug):
  --supersedes a,b   priors you've outgrown / evolved past (comma-separated for several)
  --corrects   slug  a prior that was WRONG (not merely outgrown)
  (no "retired" edge type exists — a decision dies by being superseded or corrected)
"""


def _emit_json(obj: Any, *, indent: Optional[int] = 2) -> None:
    """Prints a display payload as adaptive-``ensure_ascii`` JSON to stdout.

    The single CLI display-JSON emit path: it resolves ``ensure_ascii`` against
    the *live* ``sys.stdout`` at call time (a pytest capture, a pipe, or a real
    terminal), so raw glyphs emit on a UTF-8 stdout and fall back to ``\\uXXXX``
    escapes on a non-UTF-8 one — never a ``UnicodeEncodeError``. Centralizing the
    resolution here keeps all CLI sites uniform and the CLI⇄MCP seam single.

    Args:
        obj: A JSON-native display payload.
        indent: Pretty-print indent; ``None`` for single-line output.

    Returns:
        None.
    """
    print(dumps_display(obj, ensure_ascii=resolve_display_ensure_ascii(sys.stdout), indent=indent))


def _echo_corpus(config: MitosConfig, *, file=None) -> None:
    """Prints the standard corpus echo, leading a response on its own channel (§4.7).

    One spelling for every text site: ``recall.provenance_line`` renders the three
    fields the ``--json`` payloads carry as keys, so the two channels can never
    disagree about which corpus answered. The value is read off the config the
    caller's selector resolved — never re-derived here (``recall``'s docstrings
    state why: a stamp-time reverse lookup silently echoes a path for a registered
    project on a symlinked route).

    ``file`` exists because several handlers answer on stderr in some branches and
    stdout in others: an echo pinned to stdout is invisible to a caller reading the
    stderr answer, which is exactly the agent-facing path on ``record``'s pause. The
    echo leads the response *on the response's own channel*.

    **It flushes, and that is load-bearing rather than tidy.** Several handlers take
    this echo at the top, *before* the store construction that can raise — and a
    raise there is rendered by ``main()``'s boundary on stderr, which is unbuffered
    while a piped stdout is not. Measured without the flush:
    ``mitos scopes | cat`` on an unopenable graph printed ``Error: …`` *above* the
    echo, so the reader's opening line was a refusal for a command that had already
    named its corpus. Flushing here fixes the whole class in one place instead of at
    every leading site; a handler that then prints more before a stderr write still
    owes its own flush (``cmd_restore_source`` does).

    Args:
        config: The resolved workspace config (carries ``project`` /
            ``qdrant_collection`` / ``workspace_dir``).
        file: The stream the response rides; ``None`` ⇒ stdout.

    Returns:
        None.
    """
    stream = file or sys.stdout
    print(provenance_line(config), file=stream)
    stream.flush()


# Shared route-to-cutover guidance. `mitos init` raises it (DatabaseError),
# `mitos status` reports it (both the pre-V1a check-line and the next-steps line)
# when a prototype graph is detected, so every operator surface points the same
# direction and names the same verb. Mirrors the substance of the
# GraphStore.__init__ boot-guard message (store.py) + vision §2.1; the one-time
# `mitos cutover` verb itself is implemented below (cmd_cutover).
_CUTOVER_GUIDANCE = (
    "This graph predates the V1a schema (a prototype layout was detected). "
    "Mitos will not migrate it in place — run the one-time cutover (`mitos "
    "cutover`) to rebuild it into the V1a store (see SETUP.md → Cutover). "
    "Meanwhile the markdown gold source still answers: `mitos surface`/`query` "
    "fall back to a text match over decisions.md, and `grep decisions.md` "
    "always works — nothing is lost."
)


def _modifier_marker(payload: Dict[str, Any]) -> str:
    """Builds a one-line staleness marker from a payload's modifier keys.

    Reads the reverse-relation keys (``superseded_by``/``amended_by``/… set by
    :meth:`GraphStore.get_modifiers`) off an already-shaped payload and renders a
    compact ``⚠ amended by: <slug>`` marker so a human scanning text output sees
    that a still-live axiom has been moved on from. Empty when the node is unmodified.

    Args:
        payload: A decision payload that may carry reverse-relation modifier keys.

    Returns:
        A ``⚠ …`` marker string, or ``""`` when there are no modifiers.
    """
    parts = []
    for key in MODIFIER_EDGE_KEYS.values():
        slugs = payload.get(key)
        if slugs:
            parts.append(f"{key.replace('_', ' ')}: {', '.join(slugs)}")
    return ("⚠ " + "; ".join(parts)) if parts else ""


def _oq_modifiers(oq: Dict[str, Any]) -> Dict[str, List[str]]:
    """Lifts the reverse-relation modifier keys already stamped on an OQ dict.

    ``GraphStore.get_open_questions`` routes through the 2b modifier chokepoint, so
    a still-active OQ that a later ``amends`` / ``narrows`` has moved on from already
    carries ``amended_by`` / ``narrowed_by``. This returns the present (non-empty)
    modifier keys so the user-facing OQ output carries them too — the OQ analogue of
    the decision-side ``item.update(modifiers.get(d["id"], {}))``, read straight off
    the stamped payload (no separate ``get_modifiers_map`` call), so an amended OQ
    never reads as the final word.

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        A dict of the present reverse-relation keys to their slug lists (empty when
        the OQ is unmodified).
    """
    return {key: oq[key] for key in MODIFIER_EDGE_KEYS.values() if oq.get(key)}


def _oq_payload(oq: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the machine-readable per-OQ dict shared by every OQ ``--json`` surface.

    The single source of the open-question JSON shape: ``cmd_list``'s
    ``open_questions[]`` array and ``cmd_open_questions --json`` both emit this so an
    agent sees one OQ schema across both verbs. The present modifier keys ride via
    ``_oq_modifiers`` (an amended-but-active OQ carries ``amended_by``/``narrowed_by``
    so it never reads as the final word); the decision-only keys
    (``superseded_by``/``corrected_by``) never appear because ``get_open_questions``
    never stamps them on an OQ — the subset is structural, not filtered here.

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        A JSON-native dict with ``topic``, ``questions_raised``, ``park_reason``, and
        any present reverse-relation modifier keys.
    """
    return {"topic": oq["slug"], "questions_raised": oq["questions_raised"],
            "park_reason": oq.get("park_reason"), **_oq_modifiers(oq)}


def load_format_spec() -> str:
    """Loads the canonical format specification from the package's single source of truth."""
    spec_path = os.path.join(os.path.dirname(__file__), "format-spec.md")
    with open(spec_path, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_gitignore_entry(gitignore_path: str, entry: str) -> None:
    """Ensures ``entry`` is present in ``.gitignore``, creating the file if needed.

    Keeps a scaffolded ``.env`` (which will hold real API keys) out of version
    control. Idempotent — a no-op when the entry is already a line in the file.

    Args:
        gitignore_path: Path to the workspace ``.gitignore``.
        entry: The line to ensure is present (e.g. ``".env"``).
    """
    existing = ""
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError:
            return
        if entry in existing.splitlines():
            return
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    try:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write(f"{sep}{entry}\n")
    except OSError:
        pass


def _extract_sample_block(spec: str, header: str) -> str:
    """Extracts the fenced markdown sample under a ``## N`` header from format-spec.md.

    The spec carries one worked sample per kind inside a ```` ```markdown ```` fence:
    ``## 3. Sample Entry`` (decisions) and ``## 4. Open Question Sample`` (questions).
    ``mitos init`` lifts each into the matching buffer's preamble so a fresh
    ``decisions.md`` / ``questions.md`` shows the author the canonical shape. Only
    these two sections carry a fenced sample; the ``## 1`` / ``## 2`` field-definition
    sections do not, so this helper serves exactly those two callers.

    Args:
        spec: The full ``format-spec.md`` content.
        header: The section header to match (e.g. ``"## 3. Sample Entry"``).

    Returns:
        The sample block's inner text (stripped), or ``""`` if no fenced sample
        follows the header.
    """
    match = re.search(
        rf"{re.escape(header)}.*?\n```markdown\n(.*?)\n```",
        spec,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _registration_line(outcome: registry.RegistrationOutcome) -> str:
    """Renders ``init``'s one-line registration confirmation.

    Names the registered name and the resolved path — the whole truth available at
    registration time, and the pair a later name-targeted command resolves through.
    """
    if outcome.action == "reasserted":
        return f'Already registered as "{outcome.name}" → {outcome.path}'
    if outcome.action == "repointed":
        return (
            f'Repointed "{outcome.name}" → {outcome.path} '
            f"(was {outcome.previous_path})"
        )
    return f'Registered as "{outcome.name}" → {outcome.path}'


def _status_subject(workspace_dir: str, project: Optional[str]) -> str:
    """Names the project a `mitos status` report is about, in the caller's vocabulary.

    An operator who addressed a project by its registered name should read that
    name back — a report headed by a path alone leaves them matching directories by
    eye to confirm the tool heard them. The directory stays present regardless:
    which directory answered is the other half of the question, and 5a's flip makes
    the named form the ordinary one.

    The name is rendered through ``repr`` (1d's ``_inert_pin_note`` idiom). It is
    the one registry field ``registry.load`` validates nothing about — the read gate
    checks the *value* is a string and absolute, never the *key* — so a hand-edited
    name can carry a newline or an ESC and reach a terminal intact. It was rendered
    raw until 6c and deliberately so, because ``mitos projects`` and the 4a overview
    table also rendered it raw and a unilateral divergence would give one listing
    several spellings; the closed set was **three** surfaces, and 6c fixed all three
    in one edit rather than leaving the audit closable with one still raw.

    The **directory** stays raw, which is not an oversight either: it keeps this
    branch byte-identical to the shipped path-only form below, and a path is rendered
    raw by every other report surface in the tree, so escaping it here alone would
    re-create exactly the several-spellings drift the name fix removes.

    Args:
        workspace_dir: The absolute directory the report is about.
        project: The registered name the caller used, or ``None`` when they named
            a path (or nothing).

    Returns:
        The header subject — byte-identical to the shipped path-only form when no
        name was supplied.
    """
    if project is None:
        return workspace_dir
    return f"{project!r} ({workspace_dir})"


def _inert_pin_note(config: MitosConfig, *, offer_deletion: bool = False) -> Optional[str]:
    """Renders the one-line notice that a retired ``qdrant_collection`` pin survives.

    ``qdrant_collection`` used to be a config-file override, and ``mitos init`` used
    to materialize the derived name into ``.mitos/config.toml``. Both are gone: the
    collection is now derived from the workspace path on every construction and the
    file key is retired, so a surviving line is inert. But it is *visible*, and a
    reader who finds a name in the config file and a different name on this report
    deserves to be told which one is real rather than left to guess.

    One renderer, both surfaces (``status``'s collection row and ``init``'s echo), so
    the two wordings cannot drift. It lives here rather than in ``display.py``
    because that leaf is the CLI⇄MCP parity seam and the server has no surface for a
    workspace's config file — putting it there would widen a leaf for a consumer
    that does not exist.

    Only ``qdrant_collection`` is reported, though ``config.inert_file_keys`` carries
    every retired key the file holds: a note belongs beside the resolved value it
    claims to set, and this is the only retired key whose value is printed anywhere.
    Naming ``pending_threshold`` — which sits in every pre-V1a-seeded file — would
    re-import exactly the false-alarm-on-every-invocation noise that the retired-key
    silence exists to prevent.

    Args:
        config: The workspace config, already loaded (so ``inert_file_keys`` is
            populated).
        offer_deletion: Append the recovery clause. ``status`` sets it — a diagnostic
            surface owes a way forward, and deleting a line nothing reads creates no
            state. ``init``'s echo stays a bare receipt.

    Returns:
        The note, or ``None`` when the workspace carries no such line — which is
        every workspace ``init`` has scaffolded since the key was retired.
    """
    if "qdrant_collection" not in config.inert_file_keys:
        return None
    pinned = config.inert_file_keys["qdrant_collection"]
    # `!r`, not `{pinned}`, and not only for type honesty (an `int` pin renders as
    # `123`, a `str` quoted). The value is UNTRUSTED author-supplied text, and a TOML
    # basic string can carry ``\n`` and ``\u001b`` escapes — `repr` escapes both, so a
    # pinned value cannot break this single line or smuggle an ANSI sequence onto the
    # terminal. Tidying this to a plain interpolation would reopen that.
    note = (
        f"config.toml pins qdrant_collection = {pinned!r} — inert legacy config, "
        f"ignored; this workspace uses '{config.qdrant_collection}', derived from "
        f"its path"
    )
    if offer_deletion:
        note += ". The line can be deleted; nothing reads it"
    return note


#: The one wording for "your vectors may be under a different name" — rendered on
#: both carry-over triggers, so the two cannot drift into two half-truths. It is a
#: statement of fact plus the named heal, never a diagnosis: `init` must work on a
#: fresh, service-less machine, so it makes NO network call and therefore cannot
#: know whether the new collection is actually empty. `mitos reconcile` ships today
#: (it re-embeds the active set against whatever collection is in force), so the
#: pointer names a capability this release has, not one a later phase brings.
_COLLECTION_CARRY_OVER_POINTER = (
    "If this workspace's vectors were built under the old name, "
    "`mitos reconcile` rebuilds them here."
)


def _collection_echo_lines(config: MitosConfig,
                           outcome: registry.RegistrationOutcome) -> List[str]:
    """Renders ``init``'s collection line, plus the carry-over pointer when it applies.

    ``init`` is the moment a project *becomes* resolvable, and the only place a
    human meets its registered name, its path and its derived collection together —
    which matters more since the collection name became a path hash, deliberately
    less legible than the old basename. This is the line that pays that cost down.

    Composed from the **registration outcome**, never from
    :func:`~mitos.recall.provenance_line`: ``init`` is selector-exempt, so ``main()``
    resolves no target and ``config.project`` is the workspace *path*. The standard
    echo would therefore print a path at the one surface whose whole point is the
    *name*, and every existing row would stay green.

    Two triggers, both computable with no network probe:

    * a ``--force`` repoint — the previous path derived a different collection;
    * a surviving legacy ``qdrant_collection`` pin — ``_inert_pin_note`` has already
      printed the pinned and derived values, so only the pointer is added here (one
      renderer for the old → new information; saying it twice is how the two drift).

    Args:
        config: The workspace config (supplies the derived collection and the
            retired-key set).
        outcome: What ``registry.register`` did, and to which path.

    Returns:
        The lines to print, in order, beneath the registration line.
    """
    collection = f"collection: {config.qdrant_collection}"
    repointed = (outcome.action == "repointed" and outcome.previous_path is not None)
    previous = (default_collection_name(outcome.previous_path)
                if repointed else None)
    # `default_collection_name` realpaths its own input, and `outcome.path` is
    # already canonical, so this agrees with `config.qdrant_collection` by
    # construction — even on a symlinked route. No canonicalization needed here.
    if previous and previous != config.qdrant_collection:
        collection += f"  (was {previous})"
    lines = [collection]
    if previous or "qdrant_collection" in config.inert_file_keys:
        lines.append(_COLLECTION_CARRY_OVER_POINTER)
    return lines


def cmd_init(config: MitosConfig, name: Optional[str] = None, force: bool = False) -> None:
    """Initializes (or idempotently re-initializes) the Mitos workspace.

    Scaffolds the V1a ``.mitos/`` layout: the graph boots at the migration-ladder
    head, ``config.toml`` is seeded from the single-source ``CONFIG_DEFAULTS``,
    ``format-spec.md`` is installed from the package (refresh-on-mismatch), and the
    ``decisions.md`` / ``questions.md`` buffers are seeded only when absent. A
    re-run is idempotent: present config/buffers are left untouched, a deleted
    buffer is re-seeded, the ladder re-runs as a no-op (§5.2.7). A pre-V1a
    (prototype) graph is refused **before any file mutation** with route-to-cutover
    guidance, never ladder-advanced into a hybrid.

    ``init`` is also how a project introduces itself to Mitos *globally*: its final
    step registers the workspace's name and absolute path in the machine-local
    registry, so later commands can target it by name instead of inferring it from
    a working directory. An existing workspace joins by re-running ``init`` — that
    is the sole registration path, and it is idempotent.

    Registration is deliberately **last**, and its failure does not unwind the
    scaffold: an unwritable or malformed registry leaves a fully valid workspace
    that is merely unregistered (still reachable by its path, registrable by a
    later re-run). Rolling working files back to undo a routing entry would destroy
    user-visible state to fix a bookkeeping one.

    Args:
        config: The workspace configuration to initialize.
        name: Register under this name instead of the workspace directory's name.
        force: Repoint an existing registration of ``name`` at this workspace. It
            waives only the name collision, never the path-uniqueness guard.

    Raises:
        DatabaseError: If a pre-V1a (prototype) graph is detected — the workspace
            is left in its pre-init state; route the operator to the cutover.
        RegistryError: If registration fails (illegal name, either guard, or an
            unusable registry file). The workspace is fully scaffolded and valid,
            and simply not registered.
    """
    # 0. Refuse a pre-V1a (prototype) graph BEFORE any file mutation. The RW
    #    GraphStore boot guard would also refuse it, but only at the very end —
    #    after config/.env/skill/buffers were written. §5.2.7 requires
    #    abort-before-partial-mutation, so probe explicitly up front (read-only;
    #    open_connection's mode=ro needs the file to exist, hence the guard) and
    #    raise with route-to-cutover guidance, leaving the directory untouched.
    if os.path.exists(config.db_path):
        probe_conn = open_connection(config.db_path, read_only=True)
        try:
            if is_pre_v1a_schema(probe_conn):
                raise DatabaseError(_CUTOVER_GUIDANCE)
        finally:
            probe_conn.close()

    os.makedirs(config.mitos_dir, exist_ok=True)

    # 1. Install format-spec.md from the package — the C5 single source of truth.
    #    Refresh-on-mismatch (V1-D7 / §5.2.7): absent -> install; present but drifted
    #    from the shipped copy -> overwrite with a calm one-line warning naming both
    #    short hashes (never a silent overwrite, never a silent stale-skip). skill.md
    #    embeds the spec, so it is regenerated every init (below) and stays in lockstep.
    format_spec_path = os.path.join(config.workspace_dir, "format-spec.md")
    format_spec_content = load_format_spec()
    if not os.path.exists(format_spec_path):
        with open(format_spec_path, "w", encoding="utf-8") as f:
            f.write(format_spec_content)
    else:
        with open(format_spec_path, "r", encoding="utf-8") as f:
            on_disk_spec = f.read()
        if on_disk_spec != format_spec_content:
            old_hash = hashlib.sha256(on_disk_spec.encode("utf-8")).hexdigest()[:12]
            new_hash = hashlib.sha256(format_spec_content.encode("utf-8")).hexdigest()[:12]
            with open(format_spec_path, "w", encoding="utf-8") as f:
                f.write(format_spec_content)
            print(
                f"Refreshed format-spec.md to match the installed Mitos package "
                f"({old_hash} → {new_hash})."
            )

    # Extract the canonical sample for each buffer from the spec (one helper, both
    # kinds): the ## 3 decision sample and the ## 4 open-question sample.
    decision_sample = _extract_sample_block(format_spec_content, "## 3. Sample Entry")
    question_sample = _extract_sample_block(format_spec_content, "## 4. Open Question Sample")

    # 1a. Seed config.toml when absent — from the single-source CONFIG_DEFAULTS map
    #     (P11 / WIRING_LEDGER entry-004), NOT hand-copied literals, so a seeded file
    #     and the loader's deleted-key fallback can never diverge. The eight static
    #     keys serialize in CONFIG_DEFAULTS order; the dynamic qdrant_url line follows
    #     (an env-derived default, computed in MitosConfig.__init__).
    #     NO pending_threshold line — it left the v0.1 file schema (the loader would
    #     warn-tolerate it on every command).
    #     NO qdrant_collection line either, and that absence is load-bearing rather
    #     than tidiness: a materialized collection name travels with a `cp -r` or a
    #     `git clone` of a committed .mitos/, binding the copy to the ORIGINAL's
    #     vectors. The name is derived from the workspace path on every construction
    #     and is not overridable, so there is nothing here to write.
    config_path = os.path.join(config.mitos_dir, "config.toml")
    if not os.path.exists(config_path):
        lines = ["# Mitos Workspace Configuration"]
        for key, default in CONFIG_DEFAULTS.items():
            lines.append(f"{key} = {toml_scalar(default)}")
        lines += [
            "# Qdrant REST endpoint. Defaults to Mitos's dedicated :7333 (not the",
            "# standard :6333) so Mitos never co-locates its collections in another",
            "# Qdrant you run. Set QDRANT_URL before `init` or edit this line.",
            f"qdrant_url = {toml_scalar(config.qdrant_url)}",
        ]
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # 1b. Scaffold a gitignored .env with the credential slots, so a human or
    #     LLM setting Mitos up knows exactly where to drop keys (empty by default).
    env_path = os.path.join(config.workspace_dir, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(
                "# ============================================================\n"
                "# Mitos API keys — fill in the value(s), then run `mitos sync`.\n"
                "# This file is gitignored; never commit real keys.\n"
                "# ============================================================\n\n"
                "# Google Gemini API key — REQUIRED (unless already set globally).\n"
                "# One key covers BOTH embeddings (semantic surface/query) AND\n"
                "# decision synthesis (sync/capture).\n"
                "# Tip: set it ONCE for every project with\n"
                "#   mitos set-key --global <KEY>   (stored in ~/.config/mitos/.env)\n"
                "# or drop a project-specific key on the line below to override it.\n"
                "# Get one: https://aistudio.google.com/app/apikey\n"
                "GEMINI_API_KEY=\n\n"
                "# Anthropic (Claude) API key — OPTIONAL. Only used by\n"
                "# `mitos import --llm-extract` to convert legacy prose ADRs.\n"
                "# Get one: https://console.anthropic.com/settings/keys\n"
                "ANTHROPIC_API_KEY=\n"
            )
    # Never let the .env (real keys) get committed.
    _ensure_gitignore_entry(os.path.join(config.workspace_dir, ".gitignore"), ".env")

    # 2. Always write/overwrite skill.md (by inclusion of format-spec.md)
    skill_path = os.path.join(config.mitos_dir, "skill.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(
            "# Mitos Architecture Skill\n\n"
            "You are operating in a workspace governed by Mitos, an architectural decision graph.\n"
            "When you make an architectural decision or change a foundational pattern, you MUST record it in `decisions.md`.\n\n"
            "(If `mitos` itself is ever `command not found`, it was uninstalled after setup — reinstall it (pipx) or flag it to the human; don't silently drop decision-recording.)\n\n"
            "## Canonical Format Specification\n"
            "Your entries MUST adhere EXACTLY to the following markdown format (loaded from format-spec.md):\n\n"
            f"{format_spec_content}\n\n"
            "## Setup — API Keys\n"
            "Mitos reads keys from a `.env` file at the workspace root (`mitos init` scaffolds it with empty slots; it is gitignored). Set exactly one required key:\n"
            "- **`GEMINI_API_KEY`** (Google Gemini) — REQUIRED for semantic `surface_decisions`/`query_decisions` and for `mitos sync -p .`. One key covers both embeddings and synthesis.\n"
            "- `ANTHROPIC_API_KEY` — strongly recommended: it powers the LLM-judged layer (the `mitos check -p .` conflict audit, the sync-time conflict notice, `mitos import -p . --llm-extract`). Mitos runs without it, but only as a basic record-and-search store; with it, the corpus is audited for decisions that silently contradict each other. Degrades calmly when absent.\n"
            "Without `GEMINI_API_KEY`, `record_decision` still works (it commits to the local graph; the embedding is queued and drains on the next `mitos sync -p .` once the key is set), but semantic surface/query are unavailable. If a tool reports a missing key, tell the human to put it in `.env`.\n\n"
            "Mitos uses its own Qdrant on **:7333** (not the standard :6333), started with `docker compose up -d`. If semantic tools report Qdrant unreachable, tell the human to start it; `record_decision` still works meanwhile (embeddings queue and drain once it's up).\n\n"
            "## Addressing — every call names the project it is for\n"
            "One mitos install serves every project on this machine, so **every call must name its target**; a call that names none is refused rather than aimed at a guess. Your entries still land in THIS project's own decision graph and its own Qdrant collection — the separation is by naming, not by inability, which is exactly why the naming is worth getting right.\n"
            "- **MCP tools:** pass `project` on every call, as the **absolute path of the workspace directory this file's `.mitos/` sits in** — you know that path, because you just read this file from it. (A registered project name works too when the human gives you one. A relative path is refused: the server has no working directory to resolve it against.)\n"
            "- **CLI:** `-p .` on either side of the verb — `mitos surface -p . \"…\"` — when your shell is at the workspace root, or `-p <that absolute path>` from anywhere. `mitos status` and `mitos agent-block` take the same selector as a positional (`mitos status .`).\n"
            "Do not paste a project *name* into any file this repo commits, this one included: names are machine-local, and a name that means this project here can name a different real project on someone else's machine — which is a write into the wrong corpus rather than an error you would notice.\n"
            "Every answer echoes the corpus it acted on: `project · collection · workspace`. Read that line. Now that a call *can* reach another project, the echo is what makes a mis-aimed one visible instead of silently plausible.\n\n"
            "## Recording & recall — MCP tools (preferred) or CLI fallback\n"
            "If the Mitos MCP server is wired into your agent, call these tools directly — best experience: structured args, no shell-quoting. If it is NOT wired, each maps to a CLI verb (and the CLI also accepts the long names as aliases, e.g. `mitos record_decision -p .`):\n"
            "- `record_decision`  (CLI: `mitos record -p .`) — the moment you commit to a foundational choice (a schema, a library, a pattern, a path you're abandoning), persist it WITH the alternatives you rejected and why, so future sessions inherit it instead of relitigating. Recording rich prose via the CLI? Use `--axiom-file -` / `--rejected-file -` / `--context-file -` to read from stdin and avoid shell-quoting.\n"
            "- `surface_decisions` (CLI: `mitos surface -p .`) — surface active precedents for a claim/scope BEFORE you decide, so you don't relitigate a settled call. This is the recall loop — use it first. Every hit carries its full `rejected_paths`; pass `brief=True` (CLI `--brief`) for an axiom-only scan.\n"
            "- `query_decisions`   (CLI: `mitos query -p .`) — the TARGETED lookup: a slug you are carrying, or a pointed claim. Its confidence band rates how well the ranking matched what you named, not whether precedent exists — `surface_decisions` answers that one.\n"
            "- `list_decisions`    (CLI: `mitos list -p .`) — the EXHAUSTIVE recall path. surface/query are semantic and capped at the top few matches; this returns EVERY decision in a scope, deterministically, so a completeness pass or audit doesn't miss anything below the relevance cliff. Needs no key or Qdrant.\n\n"
            "## When to record — the capture trigger (YOUR judgement; Mitos stores, it does not decide what is worth storing)\n"
            "Recall is easy to ask for; knowing WHAT is worth recording is the real call, and it falls to you. Record a decision when it:\n"
            "- sets a pattern future work must follow, or\n"
            "- forecloses a real alternative you weighed and rejected (capture WHY in `rejected_paths` — that is what stops the next agent re-proposing it), or\n"
            "- is structural or costly to reverse, or\n"
            "- reverses or supersedes a prior decision, or\n"
            "- has cross-cutting blast radius (touches many areas).\n"
            "Skip the local, easily-reversible, or already-settled choice. A quick self-test at any fork: *would the next agent waste time re-deriving or re-litigating this?* If yes, record it. When unsure, `surface_decisions` first — if nothing is there and it clears the bar, record it.\n\n"
            "## Linking decisions\n"
            "When a decision relates to an existing one, pass that one's EXACT slug to the matching relation arg so the graph stays connected instead of accumulating silent tension: `supersedes` (replaces it), `amends`, `narrows`, `depends_on`, `resolves`, `contradicts`, `cites`. On `record_decision` these are args; on the CLI they are flags (`--supersedes`, `--depends-on`, …). Look the target up first to get its exact slug.\n"
        )

    # 3. Seed the decisions.md buffer when absent (with the extracted ## 3 sample).
    if not os.path.exists(config.decisions_file):
        with open(config.decisions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Decisions for Mitos\n\n"
                "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
                "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
                "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
                f"{decision_sample}\n\n"
                "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
            )

    # 4. Seed the questions.md buffer when absent — the open-question authoring
    #    file (ADR open-questions-authored-in-separate-questions-md-file), parallel
    #    to decisions.md. The load-bearing parts are the BEGIN ENTRIES sentinel (the
    #    parser splits the preamble on that substring) and the ## 4 sample sitting in
    #    the preamble (it yields zero graph state on the first sync).
    if not os.path.exists(config.questions_file):
        with open(config.questions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Open Questions for Mitos\n\n"
                "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
                "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
                "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
                f"{question_sample}\n\n"
                "<!-- BEGIN ENTRIES — new questions go directly below this line, newest first -->\n"
            )

    # Touch database to initialize — boots the V1a STRICT schema via the migration
    # ladder (fresh -> user_version=1; an existing V1a graph re-runs as a no-op). A
    # pre-V1a graph was already refused by the early probe above, so this never
    # ladder-advances a prototype into a hybrid.
    GraphStore(config.db_path)
    print(f"Initialized Mitos workspace at {config.workspace_dir} ✓")

    # 4a. If this workspace's config.toml carries a legacy `qdrant_collection` pin,
    #     name it as inert — `init` is one of the two places a human is already
    #     asking about this workspace's setup, and `init` never rewrites the file
    #     (D3: `.mitos/config.toml` is read-only to this vision), so the line stays
    #     and would otherwise silently disagree with the collection in force.
    inert_pin = _inert_pin_note(config)
    if inert_pin:
        print(inert_pin)

    # 5. Introduce the project to Mitos globally — the LAST mutation, after the
    #    scaffold exists and after the line above has already told the operator it
    #    does. A registry fault raises RegistryError from here, which main()'s
    #    `except MitosError` boundary renders as one calm `Error: …` line: the
    #    workspace stays valid and unregistered, and nothing is rolled back.
    #    The path registered is the CANONICAL (symlink-resolved) one, which is not
    #    necessarily `config.workspace_dir`'s abspath spelling.
    #    Flush stdout first — the same shape as the post-receipt nudges in
    #    `cmd_record`: a refusal here goes to stderr (unbuffered) while the success
    #    line above sits in a block-buffered pipe, so without this the `Error:` line
    #    overtakes it and the reader's opening line is a refusal for a job that in
    #    fact finished. The no-unwind posture is only legible in the right order.
    sys.stdout.flush()
    outcome = registry.register(config.workspace_dir, name=name, force=force)
    print(_registration_line(outcome))
    # 6. Name → path → collection, the trio a human only ever meets together here
    #    (§4.7). Composed from the outcome, not from `provenance_line` — see
    #    `_collection_echo_lines`. Placed AFTER the register call, never between
    #    the flush and it: a line in that gap re-opens the stdout/stderr inversion
    #    the flush exists to close. Flushed again for the same reason, because the
    #    entry-point notices (`update_notice`, the MCP hint) write stderr after
    #    every verb returns.
    for line in _collection_echo_lines(config, outcome):
        print(line)
    sys.stdout.flush()


def cmd_sync(config: MitosConfig, auto_accept: bool = False, embed_only: bool = False,
             verbose: bool = False, repair_targets: Optional[List[str]] = None) -> None:
    """Synchronizes the decisions write buffer with the graph store.

    The report itself is ``perform_sync``'s, so the corpus echo leads its stdout
    report from here. What the handler prints on its own is refusals — the two
    aborts and the two ``--reconcile-entry`` compositions below — and each carries
    its own echo on stderr, because an echo pinned to stdout is invisible to a
    caller reading the refusal.

    The shortfall exit is the handler's too: exiting from inside the loop would
    truncate the run before ``render_all`` and the outbox drain, so ``perform_sync``
    returns the unsatisfied targets and this is where they become an exit code.

    Args:
        config: The resolved workspace config.
        auto_accept: ``--yes``.
        embed_only: ``--embed-only`` — drains the outbox and runs no sync.
        verbose: ``--verbose``.
        repair_targets: ``--reconcile-entry`` handles, already split. ``None``
            means the flag was absent; ``[]`` means it was supplied and named
            nothing, which is a refusal rather than a fallback — the same rule the
            selector follows (``-p ""`` renders, it does not fall back to cwd).
    """
    _echo_corpus(config)
    # Refused here rather than at the parser. `add_mutually_exclusive_group` would
    # answer with argparse's terse `not allowed with argument` at exit 2, and would
    # be the only place in the tree where a `mitos sync` flag COMBINATION is judged;
    # a handler-side refusal keeps every door refusal in one register, one channel
    # and one exit code. Above the `if embed_only:` branch, because that branch
    # short-circuits to the outbox drain and never calls `perform_sync` — so the
    # flag would otherwise be discarded silently and the run would exit 0, which is
    # the silent no-op the fail-loud property forbids.
    if repair_targets is not None:
        if embed_only:
            sys.stdout.flush()
            _echo_corpus(config, file=sys.stderr)
            print("--reconcile-entry needs a full sync: --embed-only drains the "
                  "pending embeddings queue and reads no buffer entries. Re-run "
                  "without --embed-only.", file=sys.stderr)
            sys.exit(1)
        if not repair_targets:
            sys.stdout.flush()
            _echo_corpus(config, file=sys.stderr)
            print("--reconcile-entry named no entry. Give it the `### ` slug of a "
                  "diverged buffer entry, as `mitos status` reports it.",
                  file=sys.stderr)
            sys.exit(1)
    manager = MitosSyncManager(config)
    if embed_only:
        manager.drain_pending_embeddings()
    else:
        try:
            shortfall = manager.perform_sync(
                auto_accept=auto_accept, verbose=verbose,
                repair_targets=repair_targets,
            )
        except ParseError as e:
            sys.stdout.flush()
            _echo_corpus(config, file=sys.stderr)
            print(f"Sync Aborted: Parse error in write-buffer.\n{str(e)}", file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            sys.stdout.flush()
            _echo_corpus(config, file=sys.stderr)
            print(f"Sync Aborted: Validation error.\n{str(e)}", file=sys.stderr)
            sys.exit(1)
        if shortfall:
            # 1, matching the verb's two aborts — no new code vocabulary. The
            # contract binds only the caller who typed the flag; a bare `mitos sync`
            # returns an empty list and is untouched. This exit prints nothing
            # itself (the report's lines came from `sync.py`), so it owes no echo.
            sys.exit(1)


def cmd_reconcile(config: MitosConfig, as_json: bool = False) -> int:
    """Re-embeds active nodes missing from Qdrant, healing a direct vector wipe.

    The one-command heal for the gap ``sync`` cannot reach: a bare Qdrant wipe
    (``curl -X DELETE`` of the collection, no ``rebuild``/``cutover``) leaves the
    graph populated, Qdrant empty, and the outbox empty — so ``sync`` drains
    nothing. Reconcile diffs the ACTIVE node set against Qdrant's actual point
    ids, enqueues the missing nodes, and drains. Idempotent.

    Args:
        config: The active workspace config.
        as_json: Whether to emit the result as a JSON object.

    Returns:
        Process exit code (0 on success, 1 if Qdrant/embedding provider is down).
    """
    manager = MitosSyncManager(config)
    # Ahead of the call, not ahead of the report: `reconcile_embeddings` writes its
    # own provider-down line (to stderr, since 5c) before returning, so an echo
    # placed beside the report below would not be the leading line the reader sees.
    if not as_json:
        _echo_corpus(config)
    try:
        result = manager.reconcile_embeddings()
    except VectorStoreError as e:
        msg = f"Reconcile unavailable — Qdrant or embedding provider down: {str(e)}"
        if as_json:
            print(json.dumps({"error": msg, **corpus_provenance(config)}))
        else:
            sys.stdout.flush()
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps({**result, **corpus_provenance(config)}))
    else:
        print(
            f"Reconciled: {result['active']} active node(s), "
            f"{result['present']} point(s) already indexed, "
            f"{result['enqueued']} re-embedded."
        )
    return 0


def cmd_capture(config: MitosConfig, text: str) -> None:
    """Captures a raw architectural thought and appends it to decisions.md.

    Every branch — the keyless refusal included — answers on stdout, so one leading
    echo covers the whole verb.
    """
    _echo_corpus(config)
    api_key = config.env.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable is not set. Capture requires it.")
        return
        
    client = genai.Client(api_key=api_key)
    print("Synthesizing canonical decision entry ...")
    
    try:
        entry_text = run_ambient_capture(
            client, text, model_id=get_model_id("FLASH", config.env)
        )
    except Exception as e:
        print(f"Ambient capture failed: {str(e)}")
        return

    # Append below BEGIN ENTRIES line under advisory lock
    manager = MitosSyncManager(config)
    try:
        with manager.lock:
            with open(config.decisions_file, "r", encoding="utf-8") as f:
                content = f.read()

            marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
            if marker in content:
                content = content.replace(marker, f"{marker}\n\n{entry_text}\n")
            else:
                content += f"\n\n{entry_text}\n"

            with open(config.decisions_file, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"Appended synthesized decision to decisions.md buffer ✓")
    except Exception as e:
        print(f"Failed to append captured entry: {str(e)}")


def _retired_handle(store: GraphStore, slug: str) -> Optional[Dict[str, Any]]:
    """Builds a retired-handle pointer for a superseded-filtered ranked match.

    A match dropped by the active-view filter (``get_node_by_slug`` → ``None`` or a
    non-``active``/``drifted`` computed state) is not noise — it is a genuine retired
    handle the agent can chase (V1-D16: a vector-store slug always resolves to *some*
    node; nodes are never deleted). This returns ``{"slug", "state"}`` — and, when the
    graph knows it, the live successor under ``superseded_by`` — so the blackout vector
    hands the agent a pointer, not a payload. The state is read authoritatively from the
    *computed* ``get_node_state`` (the vector payload's ``state`` is stale-at-embed-time
    and absent under test), via the state-agnostic ``resolve_slug``.

    Calm degradation (P9): if the slug fails to resolve at all, returns ``None`` (the
    caller omits it) rather than crash; if the state read fails, falls back to
    ``"superseded"``.

    Args:
        store: The graph store to resolve the slug and read state/modifiers from.
        slug: The slug of the superseded-filtered match.

    Returns:
        The retired-handle dict, or ``None`` if the slug does not resolve.
    """
    try:
        node_ids = store.resolve_slug(slug)
    except Exception:
        return None
    if not node_ids:
        return None
    node_id = node_ids[0]
    try:
        state = store.get_node_state(node_id)
    except Exception:
        state = "superseded"
    handle: Dict[str, Any] = {"slug": slug, "state": state}
    try:
        successors = store.get_modifiers(node_id).get("superseded_by")
        if successors:
            handle["superseded_by"] = successors
    except Exception:
        pass
    return handle


def _emit_lexical_degraded(config: MitosConfig, query: str, *, reason: str,
                           store: Optional[GraphStore], as_json: bool,
                           brief: bool, limit: Optional[int],
                           open_questions: Optional[List[Dict[str, Any]]] = None) -> None:
    """Runs the deterministic lexical fallback and renders it on the CLI.

    The shared degraded exit for ``surface``/``query`` (ADR
    ``read-verbs-degrade-to-lexical-decisions-md-fallback``): one calm header
    naming the cause, then a term-match over decisions.md — never the raw
    provider blob, never the clean-empty header. Exit code stays 0 (deliberate:
    the JSON ``degraded`` marker + changed header already disambiguate).

    Args:
        config: The active workspace configuration (supplies decisions.md path).
        query: The claim/topic the caller was trying to recall.
        reason: One-line cause phrase (see ``degraded_reason_from_error``).
        store: A readable graph store for active-filtering + modifier stamps,
            or None when the graph itself is down (pre-V1a).
        as_json: Emit the degraded JSON envelope instead of text.
        brief: Omit ``rejected_paths`` from each match.
        limit: Max matches; None ⇒ the lexical default.
        open_questions: An already-computed scoped parked-OQ list to carry on
            the envelope (present-if-scanned semantics — None means omitted).
    """
    envelope = lexical_fallback(
        query, config.decisions_file, reason=reason, store=store,
        limit=limit, brief=brief,
    )
    envelope["query"] = query
    envelope.update(corpus_provenance(config))
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    if as_json:
        _emit_json(envelope)
        return
    print(provenance_line(config))
    print(envelope["note"])
    for i, d in enumerate(envelope["matches"], start=1):
        print(f"{i}. {d['slug']}")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    for oq in envelope.get("open_questions", []):
        print(f"[open question in scope] {oq['topic']}")


def cmd_query(config: MitosConfig, query_text: str, depth: str = "letter",
              as_json: bool = False, brief: bool = False,
              limit: Optional[int] = None) -> None:
    """Queries the vector store semantically for similar decisions — the CLI twin
    of the MCP ``query_decisions`` tool's *ranked* branch.

    Brings the CLI verb up to its MCP twin's bar: it filters superseded matches
    (state not in ``active``/``drifted``), carries a modifier-stamped,
    Letter-complete per-match payload (``core_axiom`` + ``rejected_paths`` fence)
    built via the shared :func:`letter_payload`, and emits either text or, with
    ``as_json``, the same ranked envelope ``query_decisions`` returns. The text
    render is a *renderer over the same payload list* the ``--json`` path emits, so
    the two can never disagree on what was filtered or stamped (kernel M5 + M3).

    Unlike its MCP twin, the CLI verb stays semantic-only — there is no exact-slug
    dereference branch (that is ``show``'s job, ADR
    ``cli-query-stays-semantic-not-dereference-twin``).

    Args:
        config: The active workspace configuration.
        query_text: The assertion or subsystem claim to find precedents for.
        depth: The retrieval depth; v0.1 enforces ``letter``.
        as_json: Emit the machine-readable ranked JSON envelope instead of text.
        brief: Omit ``rejected_paths`` (axiom-only) — never sheds a modifier stamp.
        limit: Ranked top-k to retrieve; ``None`` ⇒ the default 5. SETS the count
            (raises or lowers it), clamped to ``[1, RANKED_LIMIT_CEILING]`` — not a
            ``min(default, N)`` truncation.
    """
    if depth != "letter":
        msg = f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."
        if as_json:
            # Inside the handler, so it stamps (the locus rule). Its text twin one
            # line below deliberately does not: `raise` renders through `main()`'s
            # generic boundary, i.e. outside any handler, where no resolved config
            # is in scope and the response is not this verb's to shape.
            _emit_json({"error": msg, **corpus_provenance(config)}, indent=None)
            return
        raise ValueError(msg)

    # A pre-V1a graph raises at store construction — the SQLite graph is unusable,
    # so the fallback parses decisions.md directly and must not touch the graph.
    try:
        manager = MitosSyncManager(config)
    except Exception as e:
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(e),
            store=None, as_json=as_json, brief=brief, limit=limit,
        )
        return

    if not manager.embed_provider or not manager.vector_store:
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(None),
            store=manager.store, as_json=as_json, brief=brief, limit=limit,
        )
        return

    store = manager.store
    top_k = clamp_limit(limit)
    # Initialized BEFORE the `try` for the reason the CollectionMissingError arm's own
    # comment gives below: that arm constructs the empty result rather than falling
    # through, so a `top_score` bound inside the loop would be an unbound local on
    # exactly the state (empty graph + absent collection) nobody reaches by hand.
    top_score: Optional[float] = None
    try:
        q_vector = manager.embed_provider.get_embedding(query_text, is_query=True)
        raw_matches = manager.vector_store.query(q_vector, limit=top_k)

        # Filter superseded first, then stamp + Letter — mirrors the ranked loop in
        # mcp_server.query_decisions byte-for-byte (T4 parity). A superseded-not-reused
        # slug is dropped at the active-view get_node_by_slug → None step, closing the
        # M3 leak where a superseded node would otherwise read as live. Each dropped
        # match is a retired handle the blackout vector points the agent at.
        matches = []
        retired: List[Dict[str, Any]] = []
        for m in raw_matches:
            node = store.get_node_by_slug(m["slug"])
            if not node:
                handle = _retired_handle(store, m["slug"])
                if handle:
                    retired.append(handle)
                continue
            node_state = store.get_node_state(node["id"])
            if node_state not in ("active", "drifted"):
                handle = _retired_handle(store, m["slug"])
                if handle:
                    retired.append(handle)
                continue
            match = letter_payload(
                node,
                brief=brief,
                extras={"state": node_state, "score": m["score"], "depth_mode": "letter"},
            )
            match.update(store.get_modifiers(node["id"]))
            matches.append(match)
            # Off the SURFACED list, after the append — never off `raw_matches`,
            # which still holds the superseded and unresolvable nodes this loop
            # just dropped. A band read off the raw return rates a match the
            # caller never saw.
            if top_score is None or m["score"] > top_score:
                top_score = m["score"]
    except CollectionMissingError as e:
        # I8 — an absent collection over an EMPTY active set IS the empty index, and
        # a fresh project must not read as broken; over a populated graph it is a
        # real hole and the degraded header names it plus `mitos reconcile`.
        # The empty result is constructed HERE, not fallen through to: `matches` and
        # `retired` are assigned inside the `try`, after the query that raised, so a
        # bare fall-through would hit an unbound local — on exactly the state (empty
        # graph + absent collection) nobody reaches by hand.
        if missing_index_is_a_gap(store):
            _emit_lexical_degraded(
                config, query_text, reason=degraded_reason_from_error(e),
                store=store, as_json=as_json, brief=brief, limit=limit,
            )
            return
        matches = []
        retired = []
    except Exception as e:
        # Embedding/Qdrant failure mid-query (e.g. a 429): never the raw
        # provider blob — one calm cause line + the deterministic fallback.
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(e),
            store=store, as_json=as_json, brief=brief, limit=limit,
        )
        return

    # Blackout: retrieval returned matches but every one was superseded-filtered
    # (displayed == 0, retrieved > 0). That is NOT a true miss — surfacing it as one
    # makes the agent assume novelty and re-derive a settled contradiction. Emit the
    # retired handles + a distinct note instead. `retired` is non-empty only when the
    # filter dropped something, so `not matches and retired` is exactly the blackout.
    blackout = not matches and bool(retired)

    # W31 — the unbuilt graph. Unlike its I8 sibling this one is consulted on the
    # ORDINARY empty path rather than inside an `except`: an empty graph raises
    # nothing, it simply answers empty, which is precisely why the answer is
    # indistinguishable from "no precedent" without it. Mutually exclusive with
    # `blackout` by construction (a retired handle is a node, and this fires only
    # over a graph with none), so the two can never contradict each other.
    unbuilt = not matches and missing_graph_is_a_gap(
        store, config, corpus_has_entries=corpus_has_entries
    )

    # The confidence band, in the `query` register: it describes how this lookup's
    # ranking did, never what the corpus holds. One call serves both encodings.
    # `assess_query_recall` never returns None, so there is no `is not None` guard
    # here — copying `cmd_surface`'s would be a branch that can never be False.
    confidence, note = assess_query_recall(
        top_score=top_score,
        result_count=len(matches),
        config=config,
        surface="cli",
    )

    # Build the per-match list once, then branch the two renderings over it.
    if as_json:
        envelope: Dict[str, Any] = {"query": query_text, "depth_mode": "letter", "matches": matches}
        envelope.update(corpus_provenance(config))
        # Band note FIRST, then the overrides reassign it — the precedence is
        # confidence note < blackout < unbuilt, and it is the NOTE that yields:
        # `confidence` stands on every one of these exits (it is a fact about the
        # ranking that an override about the graph does not contradict).
        envelope["confidence"] = confidence
        envelope["note"] = note
        if blackout:
            envelope["all_superseded"] = retired
            # Closes the text/JSON divergence: the text branch below has printed
            # this note since 2d while `--json` carried the handles alone.
            envelope["note"] = blackout_note(retired)
        if unbuilt:
            envelope["note"] = missing_graph_note("cli")
        _emit_json(envelope)
        return

    if blackout:
        # Stamped for the same reason as the miss below: "everything here is
        # superseded" and "you asked the wrong project" are the two readings this
        # line separates. `cmd_surface`'s blackout already names its corpus (it
        # falls through the stamped empty header); this branch returns early and
        # was the one text exit of the read verbs that named none.
        _echo_corpus(config)
        print(blackout_note(retired))
        return

    # Genuine miss — nothing was retrieved (or nothing resolved). The provenance
    # line disambiguates "no precedent" from "wrong workspace" — the miss is
    # exactly where that ambiguity bites.
    if not matches:
        print(provenance_line(config))
        print("No matching decisions found.")
        # This branch APPENDS where the envelope above assigns, so the override is
        # written rather than inherited: left to the copy, a miss over an unbuilt
        # graph would print the band note *and* the pointer, and the band note's
        # redirect sends the caller to `surface`, which answers just as empty over
        # that same unbuilt graph — a turn spent one line above the correct heal.
        if unbuilt:
            print(f"→ {missing_graph_note('cli')}")
        else:
            band_line = _query_band_line(confidence)
            if band_line:
                print(band_line)
            print(f"→ {note}")
        return

    print(f"\nQuery matches for: '{query_text}'  [{provenance_line(config)}]")
    band_line = _query_band_line(confidence)
    if band_line:
        print(band_line)
    print("-" * 60)
    for i, d in enumerate(matches, start=1):
        print(f"{i}. {d['slug']}  (score {d['score']:.3f})")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    print(f"→ {note}")


def _query_band_line(confidence: str) -> Optional[str]:
    """The one-line band label for ``mitos query``'s text renderings, or None.

    Two branches, not three: ``strong`` prints nothing, exactly as on ``surface``.
    A label-axis reading of "the band renders" would compose a ``strong`` line
    carrying ``⚠`` over a good result, which reads as a warning about an answer
    that is fine. The ``⚠`` glyph is the shipped status legend (``✓``/``⚠``/``✗``),
    not emphasis — a status line without it reads as a defect, not as calm.

    The line carries the **label**; the note carries the legend and the register,
    and stays legend-complete on its own (on MCP the label arrives as the typed
    ``confidence`` field and there is no line at all).

    The ``none`` line is one sentence true of **both** of that band's states —
    matches that all ranked off-axis, and nothing ranked at all. ``surface``'s
    shipped twin ("the scope is populated, but nothing matches your query") is
    false twice over here, which is why it is not copied.

    Args:
        confidence: The band from ``assess_query_recall``.

    Returns:
        The line to print, or None when the band prints none.
    """
    if confidence == "weak":
        return "⚠ confidence: weak — top matches are close but may not be this lookup's handle."
    if confidence == "none":
        return "⚠ confidence: none — nothing ranked as a real match for this lookup."
    return None


def _show_not_found_hint(config: MitosConfig) -> str:
    """Composes the CLI's ``show`` not-found hint — the recovery, naming a selector.

    Static and hedged in the sense that matters: it reads **no buffer**, so it is
    truthful for a typo and for an authored-but-unsynced draft alike and never
    asserts presence (``show-not-found-hint-static-hedged-not-buffer-read``). Only
    the *recovery clause* is composed, and only from the config already in hand.

    It names the selector because the recipe is otherwise a hard failure: since the
    flip, a bare ``mitos sync`` has no target and errors out, so a hint that printed
    one would hand a stuck reader a second wall. The value is the caller's own
    vocabulary (``config.project`` — the registered name, or the workspace path when
    the target was not registered), rendered through ``repr`` for the same reason
    every other registry name on this surface is: a name is hand-editable text, and
    the quoted form is what a shell would want around a path with a space anyway.

    It lives here rather than in ``display.py`` because the MCP twin's hint may name
    no shell command at all, so the two cannot share a body — and because a
    selectored string in that leaf reds its own FORBIDDEN_SYNTAX literal sweep.

    Args:
        config: The resolved workspace config, carrying ``project``.

    Returns:
        The hint clause, without a trailing period — both call sites supply their
        own punctuation.
    """
    return (f"not in graph — if you just authored it in decisions.md/questions.md, "
            f"run `mitos sync -p {config.project!r}`")


def _coherence_audit_hint(config: MitosConfig) -> str:
    """Composes the CLI's coherence-audit recovery clause — this boundary's alone.

    The *fact* (a standing, corpus-wide contradiction-check debt) lives once, on
    ``sync._COHERENCE_AUDIT_NOTE``, and reaches all three encodings on the shared
    receipt dict. Only the *recovery* is composed here, for the same reason
    :func:`_show_not_found_hint` is: MCP may name no shell command (an agent handed
    one runs it), the CLI reader's actual repair *is* ``mitos check``, and a shared
    body could only be the intersection — which is empty of recovery.

    It names the selector because a bare ``mitos check`` has had no target since the
    0.15.0 flip, and it renders ``config.project`` through ``repr`` for the reason
    every other registry name on this surface does: a name is hand-editable text,
    and the quoted form is what a shell wants around a path with a space anyway.

    Unlike its sibling this returns a **complete sentence with its own terminal
    punctuation** — it has exactly one call site, where it is appended to a shared
    note that is itself a full sentence, so caller-supplied punctuation would only
    split one string across two files.

    The register stays the note's: this states what the audit *is*, one pass over
    the whole corpus, and never frames it as work owed on the entry just written.

    Args:
        config: The resolved workspace config, carrying ``project``.

    Returns:
        The recovery sentence, naming ``mitos check`` exactly once.
    """
    return f"The audit is `mitos check -p {config.project!r}` — one pass, whole corpus."


def cmd_show(config: MitosConfig, ident: str, as_json: bool = False) -> None:
    """Shows full details of a specific node by ID or slug.

    Dereferences a single handle state-agnostically via ``GraphStore.resolve_handle``
    — active-first, else the most-recent superseded node in the casefolded-slug
    lineage (marked superseded) — so a moved-on node still answers to its own slug
    instead of 404-ing. Only a genuinely-absent identifier reaches the not-found
    branch, whose hedged ``mitos sync`` pointer reads no buffer (truthful for both a
    typo and an authored-but-unsynced draft) and names the caller's own selector, so
    the recipe it prints is one a post-flip shell will actually run.
    With ``as_json`` it emits a
    Letter-complete, modifier-stamped JSON object (the not-found case a JSON object
    too, never a bare text print) — both shapes carrying the trailing
    ``project``/``collection``/``workspace`` provenance, as does the ``show_node``
    MCP twin, key for key.

    Args:
        config: The active workspace config (supplies the graph db path).
        ident: A content-hash id or a slug (case-insensitive).
        as_json: When True, emit a machine-readable JSON object instead of text.

    Returns:
        None.
    """
    # Both text exits (the dereferenced node and the not-found pointer) answer on
    # stdout, so one gated leading echo covers them; the two `--json` branches
    # carry the same three fields as keys and must not also carry a text line.
    if not as_json:
        _echo_corpus(config)
    store = GraphStore(config.db_path)

    # State-agnostic resolution: id → active slug → most-recent in lineage → None.
    # The one seam 5b's `show_node` reuses, so resolution parity is structural.
    node = store.resolve_handle(ident)

    if not node:
        # Genuine absence: a typo, or an authored-but-unsynced draft. The hint is
        # hedged and reads no buffer, so it never asserts presence to a typo. Its
        # WORDING is this surface's own since 6c — the MCP twin's carries no shell
        # command — while the not-found object's SHAPE stays shared and pinned.
        hint = _show_not_found_hint(config)
        if as_json:
            # Provenance last, exactly as the `show_node` twin stamps it: the
            # absent-handle answer is the one most in need of naming its corpus
            # ("no such handle here" vs "you asked the wrong project").
            missing = {"found": False, "ident": ident, "hint": hint}
            missing.update(corpus_provenance(config))
            _emit_json(missing)
            return
        print(f"Node with ID or Slug '{ident}' not found — {hint}.")
        return

    # Compute current active/superseded state (single-node V1a derivation, 8a)
    state = store.get_node_state(node["id"])

    # One stamp source for both the text and the --json branch (kind-agnostic — an OQ
    # carries only amended_by/narrowed_by). A superseded show that omits its modifier
    # keys reads as the final word ("amended axioms read as live" trap), and surfacing
    # superseded nodes is exactly this verb's new job — so stamping is load-bearing.
    modifiers = store.get_modifiers(node["id"])

    if as_json:
        # The dereference payload shape is single-sourced in display.show_payload
        # so the `show_node` MCP twin produces a byte-identical dict (parity is
        # structural, not test-enforced). The 5a --json regression pins prove this
        # extraction is byte-identical to the prior inline builder.
        payload = show_payload(node, state=state, modifiers=modifiers)
        payload.update(corpus_provenance(config))
        _emit_json(payload)
        return

    print(f"\n[{node['kind'].upper()}] {node['slug']}")
    print(f"ID:           {node['id']}")
    print(f"State:        {state}")
    # The negative line is the item: a node nothing has moved on from rendered
    # NOTHING here, so "checked, clean" and "the stamp did not render" were the same
    # screen — and this verb's whole job is telling a reader whether the axiom above
    # is still the last word. `get_modifiers` returns only the keys that are present,
    # so absence is unambiguous to the caller but invisible to the reader.
    #
    # `Modified by:` rather than the ledger's `Modifiers:` — it negates the four
    # labels this loop prints (`Amended by`, `Narrowed by`, `Corrected by`,
    # `Superseded by`) in their own words, and `Modifiers:` would sit two lines from
    # `Mechanisms:` as a same-length, same-shaped confusable on a surface whose
    # reported defects are already about misread lines.
    #
    # Text only. The `--json` arm omits absent keys rather than nulling them, and
    # matching it here would be a payload shape change on `show_payload` — shared
    # structurally with the `show_node` MCP tool, so not this item's to make.
    stamped = False
    for key in MODIFIER_EDGE_KEYS.values():
        if modifiers.get(key):
            stamped = True
            print(f"{(key.replace('_', ' ').capitalize() + ':'):14}{', '.join(modifiers[key])}")
    if not stamped:
        print(f"{'Modified by:':14}none")
    if node.get("date"):
        print(f"Date:         {node['date']}")
    if node.get("title"):
        print(f"Title:        {node['title']}")
        
    if node["kind"] == "decision":
        print(f"Decided:      {node['core_axiom']}")
        print(f"Rejected:     {node['rejected_paths']}")
        print(f"Mechanisms:   {', '.join(node['mechanisms'])}")
        print(f"Scope:        {', '.join(node['scope'])}")
        if node.get("invalidates_if"):
            print(f"Invalidates:  {node['invalidates_if']}")
        if node.get("context"):
            print(f"Context:      {node['context']}")
    else:
        print(f"Park Reason:  {node.get('park_reason') or 'None'}")
        print("Questions Raised:")
        for q in node["questions_raised"]:
            print(f"  - {q}")

    if node.get("transcript"):
        print("\n[Transcript]")
        print(node["transcript"])
    print()


def cmd_list(config: MitosConfig, scope: Optional[str] = None,
             state_filter: Optional[str] = None, as_json: bool = False,
             brief: bool = False, oneline: bool = False) -> None:
    """Enumerates the complete set of decisions (+ parked open questions) for a scope.

    The CLI twin of the MCP ``list_decisions`` tool — the exhaustive, deterministic
    counterpart to the ranked, capped ``surface``/``query`` recall path. Use it for a
    completeness pass: every settled call in a scope, nothing hidden below a relevance
    cliff. Needs no API key or Qdrant (it is a pure graph read).

    Args:
        config: The active workspace configuration.
        scope: Optional scope tag filter; omit for the whole project.
        state_filter: ``"active"`` (the default view) = the live set (active +
            drifted); ``"all"`` = every decision regardless of state; any other value
            = an exact computed-state match (e.g. "superseded").
        as_json: Emit a machine-readable JSON report (for agents) instead of text.
        brief: Axiom-only (omit ``rejected_paths``) — the M4 opt-out. Mutually
            exclusive with ``oneline`` (argparse enforces it on the CLI surface).
        oneline: One row per decision (slug + word-boundary-truncated axiom) — the
            orientation/table-of-contents tier below ``brief`` for big scopes.
            Modifier markers survive (stamps ride every thinner tier).
    """
    store = GraphStore(config.db_path)
    # Default the view to the live set; an absent filter must not dump superseded
    # decisions into what an agent reads as a completeness pass.
    effective_state = state_filter or "active"
    decisions = store.get_decisions(scope=scope, state=effective_state)
    modifiers = store.get_modifiers_map([d["id"] for d in decisions])
    parked = [oq for oq in store.get_open_questions(scope=scope)
              if oq["state"] == "parked"]

    def _list_item(d):
        # The oneline tier swaps the Letter core for the minimal {slug,
        # axiom_oneline, state} object; modifier stamps ride either shape
        # (stamps survive every thinner tier).
        if oneline:
            item = oneline_payload(d)
        else:
            item = letter_payload(d, brief=brief, extras={"state": d["computed_state"]})
        item.update(modifiers.get(d["id"], {}))
        return item

    # On an empty scoped read, distinguish a genuinely-fresh scope from a misspelled
    # one: an absent-from-live scope gets the same bounded self-correction vector the
    # surface verbs use (3d). Computed once, before the as_json split, so the text and
    # JSON emit points don't drift — and only on the miss path (guarded on emptiness),
    # so the hot non-empty path never pays the get_scope_counts() read. The recovery
    # payload carries no node id, so there is nothing to modifier-stamp here.
    recovery = None
    if scope and not decisions and not parked:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="cli"
        )

    if as_json:
        payload = {
            "decisions": [_list_item(d) for d in decisions],
            "open_questions": [_oq_payload(oq) for oq in parked],
            "total": len(decisions),
            "scope": scope,
            "state": effective_state,
            **corpus_provenance(config),
        }
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]
        _emit_json(payload)
        return

    if not decisions and not parked:
        print(provenance_line(config))
        if not store.get_all_nodes():
            # Empty-graph precedence wins over the unused-scope vector: a graph with no
            # nodes has an empty vocabulary, but "run sync" is the truer nudge.
            print("Graph database is empty. Run 'mitos sync' to ingest entries.")
        elif recovery:
            print(recovery["note"])
        else:
            print("No decisions match the given filters.")
        return

    scope_note = f"  (scope: {scope})" if scope else ""
    print(f"\nDecisions ({len(decisions)} found, state={effective_state}){scope_note}  "
          f"[{provenance_line(config)}]:")
    print("-" * 80)
    for d in decisions:
        marker = _modifier_marker(modifiers.get(d["id"], {}))
        if oneline:
            # One row per decision: slug + word-boundary-truncated axiom (the
            # orientation tier); a compact modifier marker rides the same row.
            row = f"{d['slug']}  {oneline_axiom(d)}"
            print(f"{row}  {marker}" if marker else row)
            continue
        scopes = f"[{', '.join(d['scope'])}]" if d["scope"] else ""
        print(f"{d['computed_state']:11} | {d['slug']:30} {scopes}")
        axiom_snip = truncate_words(d.get("core_axiom", ""), 66)
        if axiom_snip:
            print(f"              {axiom_snip}")
        if marker:
            print(f"              {marker}")
    if parked:
        print(f"\nParked open questions ({len(parked)}):")
        for oq in parked:
            print(f"  ? {oq['slug']}")
            marker = _modifier_marker(oq)
            if marker:
                print(f"        {marker}")
    print()


def cmd_open_questions(config: MitosConfig, scope: Optional[str] = None,
                       as_json: bool = False) -> None:
    """Lists all parked open questions.

    Args:
        config: The active workspace configuration.
        scope: Optional scope tag filter; omit for the whole project.
        as_json: Emit a machine-readable JSON map (the parked OQ set, each carrying
            its ``amended_by``/``narrowed_by`` modifier subset) instead of text.
    """
    # All three text branches (the listing, the empty line, the scope-recovery
    # vector) answer on stdout — one leading echo, gated off the `--json` path.
    if not as_json:
        _echo_corpus(config)
    store = GraphStore(config.db_path)
    oqs = store.get_open_questions(scope=scope)

    parked = [q for q in oqs if q["state"] == "parked"]

    # On an empty scoped read, an absent-from-live scope gets the bounded self-correction
    # vector (3d) instead of a silent "zero parked" line. Only the miss path pays the
    # get_scope_counts() read. No empty-graph precedence here (CLI asymmetry vs cmd_list):
    # on an empty graph a scoped OQ read trips the vector whose static `mitos sync` hedge
    # already covers the "just authored" case. The payload carries no node id — nothing
    # to modifier-stamp.
    recovery = None
    if scope and not parked:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="cli"
        )

    if as_json:
        # Honest-empty envelope on an empty/unmatched scope (never an error — empty is
        # first-class). An absent-from-live scope rides the additive recovery fields (3d).
        payload = {
            "open_questions": [_oq_payload(q) for q in parked],
            "total": len(parked),
            "scope": scope,
            **corpus_provenance(config),
        }
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]
        _emit_json(payload)
        return

    if not parked:
        if recovery:
            print(recovery["note"])
        else:
            print("Zero parked open questions found.")
        return

    print(f"\nParked Open Questions ({len(parked)} found):")
    print("-" * 80)
    for q in parked:
        reason = f"({q['park_reason']})" if q.get("park_reason") else ""
        print(f"Topic: {q['slug']} {reason}")
        for question in q["questions_raised"]:
            print(f"  - {question}")
        marker = _modifier_marker(q)
        if marker:
            print(f"  {marker}")
    print()


def cmd_scopes(config: MitosConfig, as_json: bool = False, archived: bool = False) -> None:
    """Enumerates the scope-tag vocabulary with each domain's live-node counts.

    The discovery surface for the project's scope vocabulary — the CLI twin of the
    MCP ``list_scopes`` tool. An agent landing in a project can already *record*
    into a scope and *recall* from one, but this is how it *sees the map*: every
    scope tag that carries a live node, ranked busiest-domain-first (total active
    decisions + parked open questions, descending; ties alphabetical), so the
    domains that matter most read first. Use it before recording or recalling, to
    learn the project's vocabulary instead of guessing it. A pure graph read — no
    API key or Qdrant needed.

    This returns a tag→counts *aggregate*, not a decision payload: there is no node
    ``id`` to stamp, so the "every decision-read surface stamps modifiers" rule does
    **not** apply here (no modifier seam — that is correct, not a missing stamp).

    Args:
        config: The active workspace configuration.
        as_json: Emit the machine-readable ``{scopes, project, collection,
            workspace}`` envelope (for agents) instead of the text table —
            ``scopes`` being the ordered ``{scope: {active_decisions,
            parked_open_questions}}`` map, the rest naming the corpus it came
            from. The byte-identical twin of the MCP ``list_scopes`` payload.
        archived: Include fully-dead domains (every scope present in the graph at a
            ``0/0`` floor) — the scope-level parallel of ``list --state all``.
            Omit for the live vocabulary only.

    Returns:
        None.
    """
    # Both text branches (the table and the empty-vocabulary line) are stdout; the
    # `--json` envelope already carries the same three fields as keys.
    if not as_json:
        _echo_corpus(config)
    store = GraphStore(config.db_path)
    counts = order_scope_counts(store.get_scope_counts(include_archived=archived))

    if as_json:
        # Same construction as the `list_scopes` twin, provenance last, so the
        # serialized bodies match byte for byte. The vocabulary nests under
        # `scopes` because scope tags are user-authored: a tag literally named
        # `project`/`collection`/`workspace` must not be overwritten by the stamp.
        envelope = {"scopes": counts}
        envelope.update(corpus_provenance(config))
        _emit_json(envelope)
        return

    if not counts:
        # Empty/fresh is first-class: an empty vocabulary IS the healthy empty state,
        # never an error. A just-initialised project simply has no scopes yet.
        print("No scopes yet — record a decision with --scope to start the vocabulary.")
        return

    name_w = max(len("scope"), max(len(s) for s in counts))
    print(f"\nScopes ({len(counts)} found, busiest first):")
    print("-" * (name_w + 30))
    print(f"{'scope':{name_w}}   {'active':>6}  {'parked':>6}  {'total':>6}")
    for scope, c in counts.items():
        active = c["active_decisions"]
        parked = c["parked_open_questions"]
        print(f"{scope:{name_w}}   {active:>6}  {parked:>6}  {active + parked:>6}")
    print()


def cmd_projects(as_json: bool = False) -> None:
    """Lists the Mitos projects registered on this machine.

    The discovery read over the global registry: which project names exist and
    which workspace each one reaches. A **global** verb — it takes no workspace
    config and works from anywhere, including a machine with no Mitos workspace at
    all, which is the state it is most likely to be met in.

    It reports registrations, not health. Whether a registered path still holds a
    valid workspace is ``mitos status``'s question; answering it here too would be
    a second source that drifts from the first.

    Args:
        as_json: Emit the machine-readable payload (``registry_path``, ``count``,
            ``projects``) instead of the text table.

    Returns:
        None.

    Raises:
        RegistryError: If the registry file exists but is unusable.
    """
    # The payload shape is `display.projects_payload`'s, not this function's —
    # lifted to the shared leaf at its second consumer (the `list_projects` MCP
    # twin), so the two surfaces cannot drift. Built once and read by BOTH
    # branches: routing only the `--json` branch through the leaf would fork the
    # text table off a second, hand-built list.
    payload = projects_payload(registry.load(), registry.registry_path())
    path = payload["registry_path"]
    projects = payload["projects"]

    if as_json:
        _emit_json(payload)
        return

    if not projects:
        # Empty/fresh is first-class: a machine with nothing registered yet is
        # healthy, and an absent registry file is the same healthy state.
        print(f"No projects registered yet — `mitos init` introduces one ({path}).")
        return

    # File order throughout, never sorted: it is the order a reverse lookup
    # resolves its first match in, so the listing must show what actually decides.
    #
    # The name goes through `repr` (1d's `_inert_pin_note` idiom): `registry.load`
    # validates the VALUE (a string, absolute) and never the KEY, so a hand-edited
    # name is the one registry field that reaches a terminal unchecked and can carry
    # a newline that breaks this table or an ESC that paints it. The column width is
    # therefore computed off the repr'd length — measuring the raw name here and
    # printing the escaped one misaligns every row.
    name_w = max(len("project"), max(len(repr(p["name"])) for p in projects))
    print(f"\nProjects ({len(projects)} registered, in registry order):")
    print(f"Registry: {path}")
    print("-" * (name_w + 30))
    print(f"{'project':{name_w}}   workspace")
    for project in projects:
        print(f"{project['name']!r:{name_w}}   {project['path']}")
    print()


def cmd_import(config: MitosConfig, filepath: str, use_llm_extract: bool = False) -> None:
    """Imports legacy prose ADR file.

    The handler prints nothing itself (``MitosProseImporter`` does), so the echo
    leads its report from here — and it is the one verb whose positional is a
    cwd-rooted *file* while its target is a selector, which is exactly the pair a
    reader needs named.
    """
    _echo_corpus(config)
    importer = MitosProseImporter(config)
    importer.import_from_file(filepath, use_llm_extract)


def cmd_render(config: MitosConfig, scope: Optional[str] = None, render_format: str = "live-axioms") -> None:
    """Statelessly regenerates live_axioms.md and scope axioms."""
    _echo_corpus(config)
    if render_format != "live-axioms":
        print(f"Warning: format '{render_format}' is not supported in v0.1. Falling back to live-axioms.")
    store = GraphStore(config.db_path)
    renderer = MitosRenderer(config.workspace_dir)
    rendered = renderer.render_all(store, scope)
    print("Render complete. Generated files:")
    for path in rendered:
        print(f"  - {path}")


def cmd_record(
    config: MitosConfig,
    axiom: str,
    rejected: str,
    scope: Optional[List[str]] = None,
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
    *,
    slug: str,
    acknowledge_neighbors: bool = False,
    as_json: bool = False,
) -> None:
    """Records a decision directly to the write-buffer and graph (thin wrapper).

    Under ``as_json``, every outcome — created/exists, the ``needs_review`` pause, and
    error — is emitted as the ``record_decision_entry`` receipt dict plus the trailing
    ``project``/``collection``/``workspace`` provenance naming the corpus written to
    (the same shape the MCP ``record_decision`` tool serializes) on **stdout**, never a
    stderr wall a ``--json`` consumer would miss. The existing exit codes are preserved (0
    created/exists, 2 needs_review, 1 error): exit code is the shell's signal, the JSON
    object is the agent's.
    """
    manager = MitosSyncManager(config)
    result = manager.record_decision_entry(
        axiom=axiom,
        rejected_paths=rejected,
        scope=scope or [],
        mechanisms=mechanisms,
        context=context,
        supersedes=supersedes,
        corrects=corrects,
        amends=amends,
        narrows=narrows,
        depends_on=depends_on,
        resolves=resolves,
        contradicts=contradicts,
        derives_from=derives_from,
        cites=cites,
        slug=slug,
        acknowledge_neighbors=acknowledge_neighbors,
    )

    if as_json:
        # Every outcome speaks JSON on stdout (no stderr walls); exit codes ride along.
        # The receipt is already the structured dict — emit it verbatim, no reshaping
        # (the pause's `neighbors` is a stamped decision-read surface: each entry is
        # the enriched candidate_payload, modifier stamps included; the created
        # receipt's write-facts stay unstamped write results). scope_overflow, when
        # present, is already inside `result`. The one addition is the trailing
        # provenance, naming the corpus the write landed in — stamped here at the
        # boundary, never inside `record_decision_entry`, so the buffer-first +
        # rollback contract carries no routing concern.
        result.update(corpus_provenance(config))
        _emit_json(result)
        if "error" in result:
            sys.exit(1)
        if result.get("status") == "needs_review":
            sys.exit(2)
        return

    # Three text outcomes on two channels — D3's sharpest case. The pause and the
    # error answer on stderr, and that is the agent-facing path: an echo pinned to
    # stdout would be missing from exactly the response a caller reads when the
    # write did NOT land.
    if "error" in result:
        _echo_corpus(config, file=sys.stderr)
        print(f"Record failed [{result['code']}]: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if result.get("status") == "needs_review":
        # P4 pause — nothing was written. Render each neighbour's enrichment (axiom,
        # rejected_paths, scope, modifier stamps) so the author can judge and link
        # without a dereference round-trip. Enrichment keys via .get(): production
        # always sends the full candidate_payload shape, but leaner dicts reach this
        # render from canned fixtures.
        #
        # Then the caller's own declared targets, partitioned (A2) — after the
        # neighbour blocks and before the recovery menu. In front of the payload it
        # would be a wall between the author and the anti-knowledge it judges
        # tenability from; the sentences are `_declared_echo_lines` over this same
        # dict's keys, so this render and the two machine encodings cannot disagree.
        _echo_corpus(config, file=sys.stderr)
        print(f"⚠ Paused — '{result['slug']}' looks like an existing decision. Nothing written.",
              file=sys.stderr)
        for n in result.get("neighbors", []):
            score = n.get("score")
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            stamps = "".join(
                f"  [{key.replace('_', ' ')}: {', '.join(n[key])}]"
                for key in ("amended_by", "narrowed_by") if n.get(key))
            print(f"  ↔ {n['slug']}  ({score_s}){stamps}", file=sys.stderr)
            if n.get("axiom"):
                print(f"      {truncate_words(n['axiom'], 60)}", file=sys.stderr)
            if n.get("rejected_paths"):
                print(f"      rejected: {truncate_words(n['rejected_paths'], 80)}",
                      file=sys.stderr)
            if n.get("scope"):
                print(f"      scope: {', '.join(n['scope'])}", file=sys.stderr)
        for line in _declared_echo_lines(result):
            print(f"  {line}", file=sys.stderr)
        # Same co-equal framing as the shared needs_review message (one constant,
        # two spellings — the CLI renders flags, the message bare names).
        menu = "/".join(f"--{r}" for r in _PAUSE_RESOLVING_RELATIONS)
        print(f"  → Judge each neighbour, then re-record with that judgment: {menu} "
              "<slug> at any neighbour this decision relates to, "
              "--acknowledge-neighbors for neighbours that stand independently — or "
              "both at once for a mixed set.", file=sys.stderr)
        sys.exit(2)

    # The "exists" short-circuit writes nothing, so it must not borrow the
    # success headline — a no-op reported as "Recorded ✓" is the same
    # couldn't-do-it-reads-as-done inversion the degraded-check notice below
    # exists to prevent, on the write itself rather than on the review.
    _echo_corpus(config)
    no_op = result.get("no_op_reason")
    if no_op:
        print(f"Decision '{result['slug']}' {no_op}")
    else:
        print(f"Recorded decision '{result['slug']}' ({result['status']}) ✓")
    print(f"  ID:        {result['id']}")
    print(f"  State:     {result['state']}")
    print(f"  Embedding: {result['embedding']}")
    if result.get("path"):
        # On the no-op branch the path still points at where the entry lives (#5b) —
        # it just must not be labelled as an action this call performed.
        if no_op:
            print(f"  Entry:     {result['path']}  (where the existing entry lives)")
        else:
            print(f"  Written:   {result['path']}  (the human-readable entry — eyeball it)")
    differs = result.get("differs")
    if differs:
        # AX round 10's ask, verbatim: *say what it ignored*. Named BEFORE the handle
        # line, because on a no-op this is the actionable part of the receipt.
        print(f"  Ignored:   this call carried different {', '.join(differs)} than the "
              f"graph holds — run `mitos sync` to reconcile the entry in decisions.md.")
    print(f"  Handle:    '{result['slug']}' — pass this to --supersedes/--amends/--depends-on/… to link future decisions.")
    # Write facts read back from the committed node (NOT an echo of the flags):
    # the edges the commit actually wired, and scope/mechanisms as stored. Lines
    # are omitted when empty — a bare decision keeps a bare receipt.
    edges = result.get("edges_created")
    if edges:
        edges_s = ", ".join(f"{e['kind']} → {e['target']}" for e in edges)
        print(f"  Edges:     {edges_s}")
    if result.get("scope"):
        print(f"  Scope:     {', '.join(result['scope'])}")
    if result.get("mechanisms"):
        print(f"  Mechanisms: {', '.join(result['mechanisms'])}")
    # Debounced size-ceiling nudge — AFTER the receipt, on stderr (an ancillary health
    # hint, never the receipt itself), so a healthy growing corpus can't bury "Recorded ✓".
    # Flush stdout first so the receipt lands before the nudge even when stdout is piped
    # (block-buffered) while stderr is unbuffered — otherwise the streams can interleave.
    overflow = result.get("scope_overflow")
    if overflow:
        sys.stdout.flush()
        print(f"\n{overflow}", file=sys.stderr)
    # Degraded-check notice — same post-receipt stderr shape: the commit succeeded,
    # but the near-dup review could not run, and silence would read as "checked, clean".
    review_notice = result.get("neighbor_review_unavailable")
    if review_notice:
        sys.stdout.flush()
        print(f"\n{review_notice}", file=sys.stderr)
    # The standing coherence debt, last — so it reads as the answer to the notice
    # above it, which after the split carries no recovery of its own. Gated on the
    # FIELD, which sync sets on the `created` return alone: this text tail is shared
    # with the `exists` exit (it branches only on the headline and the path label),
    # and a no-op incurs no audit debt — an unconditional print would also put a
    # second recipe on a receipt that already carries the `mitos sync` one. Same
    # flush-first shape as the two riders above; the `--json` branch returned long
    # ago, which is why the recovery clause is text-only by construction rather than
    # by a condition.
    coherence = result.get("coherence_audit")
    if coherence:
        sys.stdout.flush()
        print(f"\n{coherence} {_coherence_audit_hint(config)}", file=sys.stderr)


def _read_text_arg(inline: Optional[str], file_path: Optional[str]) -> Optional[str]:
    """Resolves a text argument from an inline value or a file.

    Lets agents pass multi-sentence prose without fighting shell quoting: a
    ``--*-file`` path (or ``-`` for stdin) sidesteps apostrophe/quote escaping
    that would otherwise force the prose to be degraded to satisfy bash.

    Args:
        inline: The value passed directly on the command line, if any.
        file_path: A file path to read instead, or ``"-"`` for stdin.

    Returns:
        The resolved text, or None if neither source was provided.
    """
    if file_path is not None:
        if file_path == "-":
            return sys.stdin.read()
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return inline


def _join_relation_flag(values: Optional[List[str]]) -> Optional[str]:
    """Comma-joins a repeatable relation flag back into the shared string form.

    The nine ``record`` relation flags accumulate their repeats (``action="append"``),
    while ``cmd_record``, ``record_decision_entry`` and the MCP ``record_decision``
    tool all take one comma-separated string — the form the markdown relationship
    fields already use. Joining here, at main()'s single call site, keeps those three
    signatures identical (so CLI⇄MCP parity is structural rather than policed) and
    sits above every consumer of the declared set, all of which read it through
    ``_split_relation_slugs``.

    The separator is ``", "``, not ``","``: the raw string is serialized straight into
    ``decisions.md`` as ``**Cites:** a, b, c``, and that file is the gold source a
    human reads. Both spellings parse identically downstream.

    Args:
        values: The accumulated occurrences, or None when the flag was not supplied.

    Returns:
        The comma-joined string, or None when the flag was absent (absent and empty
        are distinguished downstream).
    """
    return None if values is None else ", ".join(values)


def cmd_surface(config: MitosConfig, query: str, scope: Optional[str] = None,
                as_json: bool = False, brief: bool = False,
                limit: Optional[int] = None) -> None:
    """Surfaces active decisions relevant to a query — the CLI twin of the MCP
    ``surface_decisions`` tool (the precedent-recall half of Mitos).

    Mirrors ``mcp_server.surface_decisions`` so a CLI-only agent (or a human) can
    run the recall loop without the MCP wired. The semantic match is scope-blind;
    ``scope`` only narrows the parked open questions and the recall note (plus the
    degraded fallback when semantic recall is down). For scope-RESTRICTED retrieval
    use ``mitos list --scope`` — the only surface that hard-filters by scope. (Both
    surfaces return full ``rejected_paths``; pass ``--brief`` for a lighter scan.)

    Args:
        config: The active workspace configuration.
        query: The claim or topic to find precedents for.
        scope: Optional scope hint — does NOT filter the semantic search; scopes the
            open-questions scan and recall note only. Use ``mitos list --scope`` to
            hard-filter by scope.
        as_json: Emit a machine-readable JSON report (for agents) instead of text.
        brief: Omit ``rejected_paths`` (axiom-only — a quick "anything nearby?" scan).
        limit: Ranked top-k to retrieve; ``None`` ⇒ the default 5. SETS the count,
            clamped to ``[1, RANKED_LIMIT_CEILING]`` — not a ``min(default, N)`` clamp.
    """
    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        manager = MitosSyncManager(config)
    except Exception as e:
        _emit_lexical_degraded(
            config, query, reason=degraded_reason_from_error(e),
            store=None, as_json=as_json, brief=brief, limit=limit,
        )
        return

    store = manager.store
    top_k = clamp_limit(limit)

    def _shape(node, score):
        d = letter_payload(node, brief=brief, extras={"score": score})
        d.update(store.get_modifiers(node["id"]))
        return d

    results: Dict[str, Any] = {"active_decisions": []}
    results.update(corpus_provenance(config))
    semantic_ran = False
    top_score: Optional[float] = None
    retired: List[Dict[str, Any]] = []
    degraded_error: Optional[Exception] = None

    if manager.embed_provider and manager.vector_store:
        try:
            q_vector = manager.embed_provider.get_embedding(query, is_query=True)
            matches = manager.vector_store.query(q_vector, limit=top_k)
            semantic_ran = True
            for m in matches:
                node = store.get_node_by_slug(m["slug"])
                if not node:
                    handle = _retired_handle(store, m["slug"])
                    if handle:
                        retired.append(handle)
                    continue
                state = store.get_node_state(node["id"])
                if state not in ("active", "drifted"):
                    handle = _retired_handle(store, m["slug"])
                    if handle:
                        retired.append(handle)
                    continue
                results["active_decisions"].append(_shape(node, m["score"]))
                if top_score is None or m["score"] > top_score:
                    top_score = m["score"]
        except CollectionMissingError as e:
            # I8 — see cmd_query. Over an empty active set the absent collection IS
            # the empty index, so `semantic_ran` stays True: the read renders "ran
            # and found nothing", which is also what correctly suppresses the
            # degraded-only unranked scope dump below.
            if missing_index_is_a_gap(store):
                semantic_ran = False
                degraded_error = e
            else:
                semantic_ran = True
        except Exception as e:
            semantic_ran = False
            degraded_error = e

    # Scope listing fallback ONLY in degraded mode (mirrors the MCP tool, P5): a
    # semantic run that found nothing must not masquerade as an unranked scope dump.
    if not semantic_ran and not results["active_decisions"] and scope:
        try:
            for d in store.get_active_decisions(scope=scope)[:5]:
                results["active_decisions"].append(_shape(d, 1.0))
        except Exception:
            pass

    # Open questions only when a scope was given (absent = not scanned, [] = none here).
    if scope:
        open_questions = []
        try:
            for oq in store.get_open_questions(scope=scope):
                if oq["state"] == "parked":
                    open_questions.append({
                        "topic": oq["slug"], "questions_raised": oq["questions_raised"],
                        "park_reason": oq.get("park_reason"), **_oq_modifiers(oq),
                    })
        except Exception:
            pass
        results["open_questions"] = open_questions

    # Degraded and empty-handed on decisions: route into the deterministic
    # lexical fallback (ADR read-verbs-degrade-to-lexical-decisions-md-fallback)
    # instead of the self-contradicting "No active precedents found" +
    # unavailable note. The scoped parked-OQ scan (a pure graph read that
    # survived) rides along on the degraded output.
    if not semantic_ran and not results["active_decisions"]:
        _emit_lexical_degraded(
            config, query, reason=degraded_reason_from_error(degraded_error),
            store=store, as_json=as_json, brief=brief, limit=limit,
            open_questions=results.get("open_questions"),
        )
        return

    # Confidence signal — distinguish a settled precedent from loose neighbours / no
    # match (AX P5). Shared policy with the MCP tool via mitos.recall. The live
    # scope-count map (busiest-first) is the unused-scope oracle + did-you-mean / top-K
    # source; calm-degrade to None on error.
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None
    if scope:
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
    confidence, note = assess_surface_recall(
        semantic_ran=semantic_ran,
        top_score=top_score,
        result_count=len(results["active_decisions"]),
        scope=scope,
        scope_counts=scope_counts,
        surface="cli",
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["note"] = note

    # Blackout: semantic ranking ran and retrieved precedents, but every one was
    # superseded-filtered (no active match). Override the note with the recovery
    # vector and attach the retired handles — distinct from a true miss (where
    # `retired` is empty). Fires regardless of any parked open questions (the
    # all_superseded vector must not be suppressed by a non-empty open_questions).
    blackout = semantic_ran and not results["active_decisions"] and bool(retired)
    if blackout:
        results["note"] = blackout_note(retired)
        results["all_superseded"] = retired
        note = results["note"]

    # W31 — the unbuilt graph, consulted on the ordinary empty path (see cmd_query).
    # Placed after the blackout override so the graph note wins if both ever applied:
    # with no graph, a pointer at the collection or the graveyard names the wrong
    # heal. They cannot both apply in fact — a retired handle is a node — but the
    # precedence is written down rather than left to that coincidence.
    if not results["active_decisions"] and missing_graph_is_a_gap(
        store, config, corpus_has_entries=corpus_has_entries
    ):
        results["note"] = missing_graph_note("cli")
        note = results["note"]

    if as_json:
        _emit_json(results)
        return

    ad, oqs = results["active_decisions"], results.get("open_questions", [])
    conf = results.get("confidence")
    if not ad and not oqs:
        # The clean-empty header asserts "checked, none found" — it must never
        # co-occur with a degraded/unavailable note ("couldn't check"). With the
        # lexical fallback routing above this branch only fires when semantic
        # ran (confidence is set); the guard is belt-and-braces against any
        # future path that reaches here degraded.
        if conf is not None:
            scope_note = f" (scope: {scope})" if scope else ""
            print(provenance_line(config))
            print(f"No active precedents found for: '{query}'{scope_note}")
        print(f"→ {note}")
        return
    print(f"\nPrecedents for: '{query}'" + (f"  (scope: {scope})" if scope else "")
          + f"  [{provenance_line(config)}]")
    if conf == "weak":
        print("⚠ confidence: weak — twilight zone: matches are close but may not settle this.")
    elif conf == "none":
        print("⚠ confidence: likely off-axis — the scope is populated, but nothing matches your query.")
    print("-" * 60)
    for i, d in enumerate(ad, start=1):
        print(f"{i}. {d['slug']}  (score {d['score']:.3f})")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    for oq in oqs:
        print(f"[open question in scope] {oq['topic']}")
        marker = _modifier_marker(oq)
        if marker:
            print(f"   {marker}")
    print(f"\n→ {note}")


def cmd_serve() -> None:
    """Starts the FastMCP server over standard stdio."""
    # Importing mcp instance from mcp_server inside the function prevents early execution issues
    from mitos.mcp_server import mcp
    print("Starting Mitos MCP Server on stdio ...")
    mcp.run()


def _check_qdrant(qdrant_url: str, collection: str) -> Dict[str, Any]:
    """Probes Qdrant reachability and the project's collection (best-effort).

    Args:
        qdrant_url: The configured Qdrant REST endpoint.
        collection: The project's collection name.

    Returns:
        ``{reachable, collection_exists, points}`` — ``collection_exists`` and
        ``points`` are ``None`` when Qdrant is unreachable.
    """
    import requests
    out: Dict[str, Any] = {"reachable": False, "collection_exists": None, "points": None}
    try:
        r = requests.get(
            f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=3
        )
        out["reachable"] = True
        if r.status_code == 200:
            out["collection_exists"] = True
            out["points"] = r.json().get("result", {}).get("points_count")
        elif r.status_code == 404:
            out["collection_exists"] = False
    except Exception:
        pass
    return out


def _gemini_key_source(workspace_dir: str) -> Optional[str]:
    """Reports which tier GEMINI_API_KEY came from, for the target workspace.

    One report shape for the whole tree: this reads the same
    :func:`~mitos.env.resolve_key` the resolution path itself uses, so the tier
    the status line names is the tier that actually won. The retired
    implementation scanned the two files with a second hand-rolled parse before
    consulting the environment — deliberately, because ``main()`` used to pour
    both files into ``os.environ``, and file-first was the only way to keep the
    distinction visible through that.

    That inversion is no longer needed, and the coarseness it was compensating
    for is gone with it: since 5c deleted the entry-time dotenv load, nothing
    promotes a file's key into the process environment, so a key living in a
    ``.env`` now reports the file it lives in. ``"environment"`` means someone
    exported it.

    The answer is keyed on the **value**, not on the tier: an exported-empty
    variable resolves to ``ResolvedValue("", "environment")``, and reporting a
    tier for it would make ``env GEMINI_API_KEY= mitos status`` claim a key is
    present.

    Args:
        workspace_dir: The project directory to resolve for.

    Returns:
        ``"project .env"``, ``"global .env"``, ``"environment"``, or None.
    """
    resolved = resolve_key("GEMINI_API_KEY", workspace_dir, global_env_path())
    return resolved.tier if resolved.value else None


def _gemini_key_present(workspace_dir: str) -> bool:
    """True if GEMINI_API_KEY is available — env, project .env, or global .env."""
    return _gemini_key_source(workspace_dir) is not None


def _mcp_project_entry(workspace_dir: str) -> bool:
    """True if a project-scoped ``.mcp.json`` here declares a ``mitos`` MCP server.

    Since mitos became machine-global, this is a **finding, not a readiness
    signal** — which is why the name says what it found rather than what it
    approves of. One user-scope registration serves every project; a project-scope
    entry under the same server name does not coexist with it and does not degrade
    to it — it *wins by name and erases it*, so a broken project entry leaves the
    session with no mitos tools at all and no diagnostic saying why.

    The predicate is keyed on the literal server name ``mitos`` because that is
    what the precedence is keyed on. A project registering the same server under a
    different key (``mitos-local``, say) genuinely does not shadow, and must not be
    flagged — do not widen this to "any entry whose command mentions mitos".

    **Known limit, deliberate:** this reads ``<workspace>/.mcp.json`` only, while a
    client may ancestor-walk to a parent launch root. An entry above the workspace
    shadows the same way and is invisible here. Widening the read means walking to
    ``/`` and flagging a whole tree from one file, which is a different design; a
    clean report says "nothing here", not "nothing anywhere".

    Args:
        workspace_dir: The project directory to inspect.

    Returns:
        True when the file exists, parses, and names ``mitos`` under
        ``mcpServers``. Fully fail-silent: an unreadable or malformed file reports
        no finding rather than an error, since this is one row on a report the
        caller asked about something else.
    """
    import json as _json
    path = os.path.join(workspace_dir, ".mcp.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return "mitos" in (data.get("mcpServers") or {})
    except (OSError, ValueError, AttributeError):
        return False


def _upsert_env_var(env_path: str, name: str, value: str) -> None:
    """Inserts or replaces ``name=value`` in a ``.env`` file, preserving the rest.

    Replaces an existing (possibly empty) ``name=`` line in place; otherwise
    appends one. Creates the file (and parent dirs) if absent, and tightens the
    file to ``0600`` since it holds secrets.

    **The tree's one hand-rolled** ``.env`` **write, and it matches the way
    ``env.parse_env_file`` reads** — split on the first ``=``, strip, compare. A
    ``startswith(f"{name}=")`` test (what this did until 5c) cannot see a
    hand-spaced ``GEMINI_API_KEY = x``, so ``set-key`` appended a *second*
    assignment and left a file whose writer and reader disagreed about which line
    was the key. Two deliberate agreements with the reader: a line carrying no
    ``=`` is not an assignment and is left alone (the reader skips it too, so no
    duplicate is created from its point of view), and a commented-out line does
    not match (its key parses as ``#NAME``). One deliberate asymmetry: the reader
    ignores an *empty* assignment while this replaces it — that is the scaffolded
    ``GEMINI_API_KEY=`` slot, and after the write it is non-empty anyway.

    Every matching line is rewritten, not only the first. The reader is
    first-non-empty-wins, so a file carrying the key twice would otherwise keep a
    stale second copy that a later hand-edit could promote; rewriting all matches
    makes them the same string and the two orders agree.

    Args:
        env_path: Path to the ``.env`` file to write.
        name: The variable name (e.g. ``GEMINI_API_KEY``).
        value: The value to store.
    """
    os.makedirs(os.path.dirname(os.path.abspath(env_path)), exist_ok=True)
    lines: List[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and line.split("=", 1)[0].strip() == name:
                    lines.append(f"{name}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{name}={value}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass


def cmd_set_key(value: str, *, workspace_dir: Optional[str],
                name: str = "GEMINI_API_KEY", is_global: bool = False) -> None:
    """Stores an API key in the global or project ``.env``.

    **This function cannot resolve a working directory, and that is the point.**
    ``set-key``'s project form *is* the bare invocation, so a selector that failed
    to reach here would have landed a **credential** in the launch directory's
    ``.env`` while the caller named another project — the vision's §1 hazard on the
    one payload where it is worst. ``workspace_dir`` is therefore required and
    ``None`` means *"there is no project"*, not *"use cwd"*: it stays ``Optional``
    only because ``--global`` legitimately has no workspace, and the guard for the
    mismatch is a **raise**, never a default.

    Args:
        value: The API key value to store.
        workspace_dir: The project to write into, or ``None`` for the ``--global``
            form. Required — an unnamed project form is unconstructible rather than
            silently cwd-rooted.
        name: The env var name (default ``GEMINI_API_KEY``).
        is_global: If True, write the shared ``~/.config/mitos/.env`` (serves
            every project); otherwise write the project's own ``.env``.

    Raises:
        MitosError: If the project form was called with no workspace. A programming
            fault rather than a user one — ``main()`` refuses a selectorless
            ``set-key`` at the boundary with the teaching anatomy — so it is a plain
            boundary error, not a seventh targeting discriminator.
    """
    if not is_global and workspace_dir is None:
        raise MitosError(
            "set-key needs the project to write into. Name it with `--project` "
            "(a registered name or an absolute path), or pass `--global` to write "
            "the machine-wide .env shared by every project.")
    if is_global:
        env_path = global_env_path()
    else:
        env_path = os.path.join(workspace_dir, ".env")
    _upsert_env_var(env_path, name, value)
    scope = "globally (all projects)" if is_global else "for this project"
    print(f"Stored {name} {scope} → {env_path}")
    if not is_global:
        _ensure_gitignore_entry(os.path.join(workspace_dir, ".gitignore"), ".env")


def _fmt_k(n: int) -> str:
    """Formats a token count with a 'k' suffix for readability (e.g. 14237 → '~14k')."""
    return f"~{round(n / 1000)}k" if n >= 1000 else f"~{n}"


def _print_overflow_detail(overflows: List[Dict[str, Any]], *,
                           verbose: bool = False) -> None:
    """Prints the size-ceiling breakdown for over-budget context files (status surface).

    The detailed counterpart to the one-line nudge the write path shows: per file, its
    char/estimated-token size and — under ``verbose`` — the largest decisions in it, so
    an author knows what to re-scope. Informational only — never a readiness blocker.

    **Why the per-file breakdown is gated and the per-file size line is not.** The
    ceiling is a corpus-growth fact, so past some size every run of the report carries
    it, and unwrapping each over-ceiling file into its largest decisions buries the
    readiness verdict the report exists to give. (Measured on mitos-pub at 0.16.0: 8
    files over, 48 lines of report; the count only ever grows, which is the point —
    a figure stated as current here would be stale by the next release.) The size line
    per file is the signal — it names the file and how far over it is, which is what a
    routine or cron read is checking. The slug-level breakdown is what you want exactly
    once, when you sit down to re-scope, and ``-v`` is the moment you say so.

    The withheld detail is **announced, never silent**: an unmentioned ``-v`` is a
    capability the surface has and does not admit to, which is the same defect class
    the size ceiling itself was reported as. The JSON encoding is deliberately not
    gated — ``scope_overflow`` carries every record with its ``top_decisions`` on both
    verbosities, because a machine payload that changes shape with a text-verbosity
    flag is a payload no consumer can rely on.

    Args:
        overflows: Overflow records from ``overflow_report`` (largest file first).
        verbose: Render each file's largest-decisions breakdown. Off by default.
    """
    n = len(overflows)
    noun = "file" if n == 1 else "files"
    print(f"\n  ⚠ {n} rendered axiom {noun} over the size ceiling "
          f"(informational — not a readiness blocker):")
    withheld = False
    for o in overflows:
        print(f"      - {o['name']}: {o['chars']:,} chars "
              f"({_fmt_k(o['est_tokens'])} tokens, ceiling {o['threshold_chars']:,})")
        top = o.get("top_decisions", [])
        if not top:
            continue
        if verbose:
            print("          largest decisions:")
            for d in top:
                print(f"            • {d['slug']}  ({d['chars']:,} chars)")
        else:
            withheld = True
    if withheld:
        # Named on the default path only: under `-v` the breakdown is already above,
        # and re-offering the flag that produced it reads as a failed render.
        print("    Re-run with `-v` for the largest decisions in each file.")
    print("    These context files grow with the corpus — re-scope the largest "
          "decisions in them, or split a broad scope.")


def _graph_behind_buffer(db_path: str) -> bool:
    """Detects a graph migrated to the V1b schema in place but never re-committed.

    Cheap, **graph-only** signal (no buffer parse, no false positives): ``True`` iff
    the ``mechanisms`` registry is empty while decision nodes still carry mechanism
    refs — the signature of a corpus whose V1b catalog (the seven non-kill edge types
    + the first-seen-wins mechanism registry) was never committed because the schema
    migration only widened the DDL. A ``mitos rebuild`` populates them. Any read
    failure (a pre-mechanisms V1a-schema graph, an absent/locked DB) is a safe
    ``False`` — never a spurious nudge.

    Args:
        db_path: The live graph path.

    Returns:
        ``True`` if the graph is behind its buffer's catalog, else ``False``.
    """
    try:
        conn = open_connection(db_path, read_only=True)
    except Exception:
        return False
    try:
        if conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0] > 0:
            return False
        carries_refs = conn.execute(
            "SELECT 1 FROM nodes WHERE mechanism_refs_json IS NOT NULL "
            "AND mechanism_refs_json NOT IN ('', '[]') LIMIT 1"
        ).fetchone()
        return carries_refs is not None
    except Exception:
        return False
    finally:
        conn.close()


#: The state column's mark. ``✓`` healthy, ``✗`` a finding, ``—`` **unknown** —
#: `status`'s own vocabulary, and the third mark is the load-bearing one: an
#: unresponsive entry is not a missing one, so it must not be rendered as a finding.
_OVERVIEW_MARKS = {
    overview.STATE_OK: "✓",
    overview.STATE_MISSING: "✗",
    overview.STATE_NOT_A_WORKSPACE: "✗",
    overview.STATE_ERROR: "✗",
    overview.STATE_UNRESPONSIVE: "—",
}


def _overview_notes(project: Dict[str, Any], payload: Dict[str, Any], *,
                    corpus_scan: Callable[[str], bool] = corpus_has_entries) -> List[str]:
    """Words one project's findings — the sentences the leaf deliberately does not carry.

    Order is by what the reader most needs: where they are standing, then why the
    row is not ``ok``, then the vector cross-reference, then the registry finding.
    A fully healthy project on a healthy instance produces **no** lines at all, so a
    calm machine reads as a calm table (P9).

    **The collection flag is gated on the corpus, and it prescribes nothing.** I8
    says the missing-collection message must be gated — *"a brand-new project with
    zero decisions also has no collection, and sending that user to `mitos
    reconcile` would be a wall on the healthy-and-empty state this project treats as
    first-class"* — and until the zero-arg dispatch flip this surface was reachable
    by nobody, so its ungated flag was false about a state no user could read. Two
    changes make it true in all three states it meets:

    * **Gated on the corpus** (the injected scan), not on the graph the way 1b's
      read surfaces and 4b's deep report are. A graph gate is the sharper predicate
      and this sweep may not have it: ``mitos/overview.py``'s cost contract is three
      stats plus a config read and its docstring forbids opening SQLite,
      permanently. The corpus is the honest proxy this surface can afford, and it
      separates the fresh project (flagged before, healthy now) from a populated one.
    * **No prescription.** The corpus gate cannot separate the clone whose graph was
      never built (heal: ``mitos sync``) from the project whose collection was swept
      (heal: ``mitos reconcile``), and for the clone ``reconcile`` is the heal 4b
      calls *"one word away and worse than silence"* — it diffs an empty active set
      against an absent collection, enqueues nothing, and reports success on a
      workspace it did not touch. So the note points at ``mitos status <name>``,
      which reads the graph and names the right heal. That is the *"projects
      discovers, status diagnoses"* division the vision already states.

    The scan arrives as a keyword with a default rather than as an import inside the
    leaf: ``mitos/overview.py`` stays Tier 2 with its import closure untouched, and
    the predicate is injectable for a row that must not touch a filesystem. It runs
    only on an ``ok`` entry — every other state carries ``collection: None`` — so the
    probe has already reached that workspace within its wall-clock budget. The read
    itself is not inside that budget, which is the residual cost of answering this
    at the surface rather than in the sweep; it is one short-circuiting stream over a
    file the probe just stat'd.

    Args:
        project: One entry of the payload's ``projects`` list.
        payload: The whole payload — read only for ``cwd_project`` and for the
            document order that decides a shared path's resolving name.
        corpus_scan: The corpus-population predicate, ``parser.corpus_has_entries``
            by default. Takes the corpus path; a missing or unreadable file is
            ``False``.

    Returns:
        Zero or more note lines, already indented.
    """
    notes: List[str] = []
    # Every registry NAME on this surface renders through `repr`, matching the table
    # one screen down and `mitos projects`. The read gate validates the registry's
    # values and never its keys, so a name is unchecked hand-edited text; here it also
    # lands inside printed `mitos status <name>` recipes, where the quoted form is the
    # one a shell would accept anyway. `name` is bound to the escaped spelling once so
    # a later note cannot reintroduce the raw one.
    name = repr(project["name"])
    if payload["cwd_project"] == project["name"]:
        # The overview closes its own discovery loop: a human standing in their
        # project who expected a project report learns the form at a glance.
        notes.append(
            f"→ you are here — `mitos status {name}` for this project's full report")

    state = project["state"]
    if state == overview.STATE_MISSING:
        notes.append(
            f"nothing at {project['path']} — the project moved or was deleted; "
            f"re-run `mitos init` there, or edit {payload['registry_path']}")
    elif state == overview.STATE_NOT_A_WORKSPACE:
        notes.append(
            f"{project['path']} exists but holds no Mitos workspace (.mitos/config.toml "
            f"+ decisions.md) — run `mitos init` there, or edit "
            f"{payload['registry_path']}")
    elif state == overview.STATE_UNRESPONSIVE:
        notes.append(
            f"the check did not answer within {overview.LOCAL_PROBE_BUDGET:g}s — an "
            f"unresponsive mount, not a missing project; nothing was concluded about it")
    elif state == overview.STATE_ERROR:
        notes.append(f"the check failed: {project['error']}")

    if project["collection"] is not None:
        # `decisions.md` beside `.mitos/` is the shipped validity triple `is_workspace`
        # just proved for this entry and `MitosConfig` derives without a setting, so the
        # join is a literal here rather than a config construction the sweep refuses to
        # repeat.
        if (project["collection_present"] is False
                and corpus_scan(os.path.join(project["path"], "decisions.md"))):
            # A pointer, never a diagnosis and never a heal: the overview reads no
            # graph, so it can neither price what re-embedding would cost nor tell a
            # swept collection from an unbuilt one — and those two want opposite
            # commands. Presence would not imply population anyway.
            notes.append(
                f"⚠ no vector collection {project['collection']} on "
                f"{project['qdrant_url']} — `mitos status {name}` for the heal")
        elif project["collection_present"] is None:
            notes.append(
                f"vector collection unknown — {project['qdrant_url']} did not answer "
                f"with a usable listing (see below)")

    if project["shares_path_with"]:
        # A finding, never a fault — both names reach the same workspace, so nothing
        # can be corrupted. The actionable half is which one decides: a reverse
        # lookup answers with the first in document order, which is the first of them
        # in the listing above.
        others = ", ".join(repr(other) for other in project["shares_path_with"])
        deciding = repr(next(
            other["name"] for other in payload["projects"]
            if other["path"] == project["path"]))
        notes.append(
            f"also registered as {others} — every echo names {deciding} for this "
            f"workspace (first in registry order wins)")
    return [f"   {note}" for note in notes]


def _render_overview(payload: Dict[str, Any]) -> None:
    """Prints the global overview as a text table.

    The wording half of the sweep — every user-facing sentence lives here, following
    ``_render_targeting_error``'s precedent, while ``mitos/overview.py`` carries only
    typed data.

    **It flushes, and that is load-bearing.** ``main()``'s ``except MitosError``
    boundary writes unbuffered stderr while a piped stdout is not, so without the
    flush an error raised after this table can land *above* the output it annotates,
    with the whole suite green (``capsys`` keeps the streams apart and cannot see the
    inversion).

    Args:
        payload: ``overview.build_overview``'s payload.

    Returns:
        None.
    """
    projects = payload["projects"]
    if not projects:
        # Empty/fresh is first-class: a machine with nothing registered yet is
        # healthy, and an absent registry file is the same healthy state.
        print(f"No projects registered yet — `mitos init` introduces one "
              f"({payload['registry_path']}).")
        sys.stdout.flush()
        return

    # File order throughout, never sorted — the order a reverse lookup resolves its
    # first match in, matching `mitos projects`.
    #
    # `repr` on the name, and the width measured off the repr'd length, for the same
    # reason as `cmd_projects`' table: the registry's read gate validates values and
    # never keys, so the name is hand-editable text arriving unchecked. The two
    # tables move together — that lockstep is why 6c fixed all three surfaces at once.
    name_w = max(len("project"), max(len(repr(p["name"])) for p in projects))
    state_w = max(len("state"), max(len(p["state"]) for p in projects) + 2)
    print(f"\nMITOS PROJECTS ({payload['count']} registered, in registry order)")
    print(f"Registry: {payload['registry_path']}")
    print("-" * (name_w + state_w + 20))
    print(f"{'project':{name_w}}   {'state':{state_w}}   workspace")
    for project in projects:
        # `.get` rather than `[...]` on purpose, and safe only because the table is
        # fenced against the leaf's vocabulary by a key-set row: a renderer fallback
        # over an unfenced map is this vision's most-repeated defect shape, and a
        # KeyError here would take the whole table down over one unknown state, which
        # is the opposite of what every other line in this phase is for.
        mark = _OVERVIEW_MARKS.get(project["state"], "—")
        state = f"{mark} {project['state']}"
        print(f"{project['name']!r:{name_w}}   {state:{state_w}}   {project['path']}")
        for note in _overview_notes(project, payload):
            print(note)

    instances = payload["instances"]
    if instances:
        print("\nQdrant:")
        for instance in instances:
            if instance["listing_available"]:
                detail = f"up · {instance['collection_count']} collection(s)"
            elif instance["reachable"]:
                detail = ("up, but its collection listing is unavailable · vector "
                          "checks unknown")
            else:
                detail = "no answer · vector checks unknown"
            print(f"  {instance['url']}   {detail}")
            if instance["error"]:
                print(f"     {instance['error']}")
    print()
    sys.stdout.flush()


def cmd_status_overview(as_json: bool = False) -> int:
    """Reports what this machine has: every registered project and its health.

    The **global** half of ``status`` — explicitly global, so no silent cwd
    resolution exists to mis-aim it. It is what a **selectorless** ``mitos status``
    renders; ``mitos status <project>`` keeps the per-project deep report. The
    dispatch decides between them in ``main()``, before any ``MitosConfig`` is
    built, so a broken workspace underfoot cannot take down a report about the
    whole machine.

    **The exit contract, stated where the report is built.** It returns **0 whenever
    it can render, and non-zero only when it cannot**. An empty registry is the
    healthy fresh state, a stale entry is flagged-not-fatal, and an unreachable
    Qdrant is a timed-out cell — none of the three is an error. Only a registry that
    cannot be parsed fails, through the calm ``RegistryError`` the boundary renders.
    The readiness *gate* keeps its shipped ``0``/``1`` mapping and stays with the
    per-project report, where a readiness verdict still means something: ``ready`` is
    undefined for a report about N projects.

    Args:
        as_json: Emit the machine-readable payload instead of the text table.

    Returns:
        ``0``.

    Raises:
        RegistryError: If the registry file exists and is unusable.
    """
    # The vision's ONE display-only cwd read, and it lives here rather than in the
    # leaf so the sweep structurally has no cwd branch. Guarded on `_render_targeting_error`'s
    # precedent (a deleted working directory raises `OSError`): losing an optional
    # marker is the right degradation, losing the whole table is not. Narrower than
    # that precedent's `(RegistryError, OSError)` on purpose — the registry read
    # happens inside `build_overview`, where its failure IS the answer and must not
    # be swallowed into a table that pretends nothing is registered.
    try:
        cwd: Optional[str] = os.getcwd()
    except OSError:
        cwd = None

    payload = overview.build_overview(cwd=cwd)
    if as_json:
        _emit_json(payload)
        return 0
    _render_overview(payload)
    return 0


def cmd_status(workspace_dir: str, as_json: bool = False, *,
               project: Optional[str] = None, verbose: bool = False) -> int:
    """Reports whether Mitos is set up for a project, and what (if anything) is missing.

    Designed to be run by a human OR an LLM in a new project: it answers "is Mitos
    ready here?" with a clear ✓/⚠/✗ report, a one-line verdict, and an exit code
    (0 = fully ready, 1 = needs attention / not set up). When not ready it prints
    concise next steps and points at the full SETUP walkthrough.

    **Exit contract for ``mitos status <project>``: 0 = fully ready, 1 = needs
    attention / not set up, on every branch including the malformed-config one.**
    SETUP.md's agent loop is built on that mapping, so it is stated here rather
    than left to be read off the returns.

    ``project`` arrives as an argument because it cannot be recovered here.
    ``config.project`` on the config this function rebuilds is a *path* (the
    constructor's ``project or self.workspace_dir``), and the alternative —
    reverse-looking-up the path against the registry — is the build 3d rejected by
    name: it misses on a symlinked route whose registry entry is hand-written
    non-canonically, quietly printing a path for a registered project with every
    test green. ``main()`` already resolved the name; it passes it in, so this
    function structurally cannot re-derive it wrong.

    Args:
        workspace_dir: The project directory to inspect.
        as_json: Emit a machine-readable JSON report instead of the text report.
        project: The registered name the caller addressed this workspace by, or
            ``None`` for an unregistered path / no selector. Keyword-only and
            defaulted, so the direct call sites (the suite's, and any future one)
            keep passing a bare path.

    Returns:
        ``0`` if fully ready, ``1`` otherwise.
    """
    workspace_dir = os.path.abspath(workspace_dir)

    # `status` is the "is this set up?" probe, so a malformed config.toml is exactly
    # what it should surface — calmly, as not-ready, never a traceback. The main()
    # boundary would render only a generic `Error: …`; status owes its caller the
    # contextual "config malformed → not ready" report (Lesson 45 / entry-004).
    try:
        config = MitosConfig(workspace_dir, project=project)
    except ConfigError as e:
        if as_json:
            _emit_json({
                # Which payload this is. `status --json` returns the project report
                # today and will return the global overview too once zero-arg
                # `status` is flipped, so it announces itself rather than leaving a
                # consumer to sniff a shape — and it does so on BOTH emission sites,
                # because a discriminator present on one is detectable only by
                # absence, which is sniffing wearing a discriminator's clothes.
                "report": "project",
                # Same value the constructor would have computed had it succeeded
                # (`project or self.workspace_dir`), so a consumer never has to
                # branch on which arm produced the payload. Note this key is the
                # *targeting* identity and is unrelated to the three-field corpus
                # echo `corpus_provenance` emits elsewhere — `status` carries no
                # echo (3e's in-body carve-out), so the two never share a payload.
                "project": project or workspace_dir,
                "workspace": workspace_dir,
                "ready": False,
                "initialized": False,
                "config_error": str(e),
            })
        else:
            print(f"\nMITOS STATUS for {_status_subject(workspace_dir, project)} "
                  f"— NOT SET UP ✗\n")
            print(f"  ✗ config.toml malformed: {e}")
            print("      → fix it or re-run `mitos init`")
            print()
        return 1

    # Pre-V1a (prototype) graph detection — mirrors `init`'s early probe so the two
    # surfaces stay coherent (§5.2.7). A read-only GraphStore SKIPS the boot guard
    # (RO can't migrate), so a prototype graph would open fine and only fail deep in
    # get_all_nodes() (swallowed below) → status must run its OWN probe and force
    # not-ready. is_pre_v1a_schema is False for an absent/empty or V1a-or-later DB,
    # so a freshly-init'ed empty graph stays healthy (empty-is-healthy, P5).
    pre_v1a = False
    if os.path.exists(config.db_path):
        try:
            probe_conn = open_connection(config.db_path, read_only=True)
            try:
                pre_v1a = is_pre_v1a_schema(probe_conn)
            finally:
                probe_conn.close()
        except Exception:
            pass  # best-effort, like the RO read below: a probe failure leaves it False

    mitos_dir_ok = os.path.isdir(config.mitos_dir) and os.path.exists(
        os.path.join(config.mitos_dir, "config.toml")
    )
    decisions_ok = os.path.exists(config.decisions_file)
    spec_ok = os.path.exists(os.path.join(workspace_dir, "format-spec.md"))
    key_source = _gemini_key_source(workspace_dir)
    key_ok = key_source is not None
    q = _check_qdrant(config.qdrant_url, config.qdrant_collection)
    mcp_project_entry = _mcp_project_entry(workspace_dir)
    # Best-effort: is the pasted agent-file mitos note out of date? A recommendation,
    # never a readiness blocker — like the shadowing-entry finding.
    agent_drift = agent_block_drift(workspace_dir)

    graph_nodes = None
    active_nodes = None
    active_ids: Set[str] = set()
    id_to_slug: Dict[str, str] = {}
    # Read-only size-ceiling report for the generated context files. This is the
    # health surface the write-path overflow nudge points at — the detailed breakdown
    # (which files, which decisions to re-scope) lives here, not on every `record`.
    overflows: List[Dict[str, Any]] = []
    graph_behind = False
    embedding_seed: Optional[Dict[str, str]] = None
    # The store the unbuilt-graph predicate answers on, or None when there is no
    # graph file at all (the clone). Bound immediately after construction so a
    # later read failure still hands the predicate a store — it then answers on its
    # "could not check" arm, the loud direction, rather than on the absent-file one.
    gap_store: Optional[GraphStore] = None
    if os.path.exists(config.db_path) and not pre_v1a:
        try:
            ro_store = GraphStore(config.db_path, read_only=True)
            gap_store = ro_store
            all_nodes = ro_store.get_all_nodes()
            graph_nodes = len(all_nodes)
            id_to_slug = {n["id"]: n["slug"] for n in all_nodes}
            active_ids = ro_store.get_active_node_ids()
            active_nodes = len(active_ids)
            overflows = overflow_report(ro_store)
            # The coverage marker, if one stands: a `rebuild`/`cutover`/`reconcile`
            # bounded the outbox to the active set and has not drained yet. Purely
            # informational — it tells the operator that a plain `mitos sync` restores
            # search, which is otherwise a fact only the drain knows. Never a readiness
            # gate: an unseeded queue is an ordinary state, not a fault.
            embedding_seed = ro_store.embedding_seed()
        except Exception:
            pass  # both reads are best-effort; a failure leaves the safe defaults
        graph_behind = _graph_behind_buffer(config.db_path)

    # Vector-index completeness by EXACT id-diff, not a count proxy. `mitos status`
    # is the SENSOR (reconcile is the heal), so it must catch a shortfall the count
    # `points >= active` structurally can't — dead-vector slack in the graveyard
    # (superseded vectors that linger, never GC'd) inflates the point total past the
    # active threshold and hides genuinely-missing active vectors (the live-corpus
    # incident: 181 >= 178 read healthy over 12 invisible active nodes). We scroll
    # Qdrant's actual point ids and diff. `status` is a read-only sensor and creates
    # nothing — the module-level scroll keeps it independent of a store entirely.
    #
    # `missing_active` = active nodes with no vector (invisible to semantic search —
    #   the warned state). `orphan_points` = points with no active node — the
    #   graveyard substrate the all-superseded blackout vector consumes, per
    #   graveyard-vectors-now-consumed-by-blackout-best-effort: reported neutrally,
    #   never a warning, never deleted here. `None` means "could not verify" (scroll
    #   failed) — distinct from `0` ("verified complete"); we never fall back to the
    #   count proxy we just declared structurally blind.
    missing_active_slugs: Optional[List[str]] = None
    orphan_points: Optional[int] = None
    scroll_failed = False
    if q["reachable"] and q["collection_exists"] and active_nodes is not None:
        try:
            present = scroll_point_ids(config.qdrant_url, config.qdrant_collection)
            missing_active_slugs = sorted(
                id_to_slug.get(nid, nid) for nid in active_ids if hash_to_uuid(nid) not in present
            )
            active_uuids = {hash_to_uuid(nid) for nid in active_ids}
            orphan_points = len(present - active_uuids)
        except VectorStoreError:
            scroll_failed = True  # unknown — do not fabricate "complete"
    elif q["reachable"] and not q["collection_exists"] and active_ids:
        # Collection deleted OUTRIGHT (not just its points) while the graph holds
        # active nodes — the whole active surface is missing (the diff against an
        # empty collection). This is a full wipe, NOT the healthy fresh-project empty
        # state; absence of the collection must not read as calm health next to a
        # populated graph. `mitos reconcile` re-creates the collection and re-embeds.
        missing_active_slugs = sorted(id_to_slug.get(nid, nid) for nid in active_ids)
        orphan_points = 0

    # Corpus↔graph divergence (B′). Purely informational — a corpus mid-edit is a
    # normal state, so gating readiness on it would make an ordinary hand edit read
    # as breakage. Best-effort like every other read here: a fault leaves the safe
    # `None`, which prints nothing, rather than taking `status` down.
    divergence_report = None
    if os.path.exists(config.db_path) and not pre_v1a:
        try:
            ro_store = GraphStore(config.db_path, read_only=True)
            divergence_report = corpus_graph_divergence(ro_store, config)
        except Exception:
            pass

    initialized = mitos_dir_ok and decisions_ok
    # The unbuilt graph (W31): a corpus holding entries over a graph holding no
    # nodes — the clone that carries the committed `.mitos/config.toml` and
    # `decisions.md` but not the gitignored `*.sqlite`. The same predicate the four
    # semantic read surfaces consult, so this report and those answers cannot drift
    # into two opinions about one state.
    #
    # TWO guards, and each removes a heal that would be wrong on a report that is
    # already not-ready. `pre_v1a`: a prototype graph leaves `gap_store` None (the
    # two reads above skip it) while being, by definition, POPULATED — unguarded it
    # would be told "run `mitos sync`" beside the `mitos cutover` line it already
    # gets. `initialized`: a directory holding a `decisions.md` and no `.mitos/`
    # cannot be synced at all, and its report already leads with `mitos init` — the
    # rung there would be a second, unreachable instruction (measured by hand on a
    # real directory, not reasoned about). Neither guard touches the target state:
    # a clone carries the committed `.mitos/config.toml`, so it is `initialized`.
    graph_unbuilt = False
    if initialized and not pre_v1a:
        graph_unbuilt = missing_graph_is_a_gap(
            gap_store, config, corpus_has_entries=corpus_has_entries
        )

    # An absent (or empty) collection is a normal ready state, NOT a blocker: a
    # project with .mitos/, a key, and a reachable Qdrant is ready to record its
    # first decision. Absence is not a readiness question because the GRAPH tells
    # the two cases apart, and the collection row below already does that — over an
    # empty graph it reads "none recorded yet", over a populated one it warns and
    # points at `mitos reconcile`. Only an unreachable Qdrant degrades semantic
    # surface/query. A pre-V1a (prototype) graph is never ready — it must be routed
    # through the one-time cutover first (§5.2.7).
    #
    # An unbuilt graph over a populated corpus is likewise never ready, and unlike
    # the informational rungs (vector completeness, divergence) this one is a gate:
    # those report PARTIAL degradation over a working graph — the lexical fallback
    # still answers, `list`/`show` still serve — while this state has no working
    # graph at all, does not self-heal (nothing builds it but an explicit `mitos
    # sync`, unlike an absent collection, which the first covering write creates),
    # and is read by an agent setup loop whose next move on a `0` is to trust an
    # empty answer. A gate that cannot stop that is not a gate. The 0/1 mapping is
    # unchanged and no new verdict appears: `initialized` is still True on a clone,
    # so the shipped middle value (NEEDS ATTENTION ⚠) already fits.
    ready = (initialized and key_ok and q["reachable"]
             and not pre_v1a and not graph_unbuilt)

    if as_json:
        _emit_json({
            # The payload discriminator — see the early ConfigError branch above.
            "report": "project",
            "project": config.project,
            "workspace": workspace_dir,
            "ready": ready,
            "initialized": initialized,
            "pre_v1a": pre_v1a,
            "qdrant_url": config.qdrant_url,
            "collection": config.qdrant_collection,
            # The four resolved corpus locations, flat beside the other
            # resolved-identity values rather than in a `paths` sub-map: `checks` is
            # the payload's only nested map and it is a homogeneous verdict map,
            # which these explicitly are not (a path passes and fails nothing). The
            # key names mirror `MitosConfig.to_dict()`'s spellings so a consumer
            # reading both finds the same words — a naming convention, deliberately
            # NOT a code coupling: bound as attributes, so a future rename inside a
            # debug mapping cannot propagate into a shipped payload contract
            # (additive-only, data in the wild). Absolute by construction
            # (`__init__` abspath's the workspace), so nothing here re-`abspath`s.
            "decisions_file": config.decisions_file,
            "questions_file": config.questions_file,
            "archive_dir": config.archive_dir,
            "db_path": config.db_path,
            "checks": {
                "mitos_workspace": mitos_dir_ok,
                "decisions_buffer": decisions_ok,
                "format_spec": spec_ok,
                "gemini_api_key": key_ok,
                "qdrant_reachable": q["reachable"],
                "collection_exists": q["collection_exists"],
                "collection_points": q["points"],
                "graph_nodes": graph_nodes,
                "active_nodes": active_nodes,
                "missing_active_vectors": (
                    None if missing_active_slugs is None else len(missing_active_slugs)
                ),
                "missing_active_slugs": missing_active_slugs,
                "orphan_points": orphan_points,
                "graph_unbuilt": graph_unbuilt,
                # Renamed from `mcp_wired` in 6a, and the rename is the point: the
                # same file read now means the opposite thing. A consumer reading
                # `mcp_wired: true` for "good" would read a hazard as an
                # endorsement, and no test can catch a lie of that shape — a
                # missing key reds loudly on the first read.
                "mcp_project_entry": mcp_project_entry,
            },
            # The RAW pinned value, never the rendered note: prose belongs to the
            # text surface and typed data to the payload, or the two channels drift
            # into two claims. It may be a `str`, an `int`, or a list — the loader
            # records whatever the file held (1d's retirement rows parametrize all
            # three) and `_emit_json` serializes each fine. No `repr()`: that
            # escaping is `_inert_pin_note`'s, and it is a terminal concern. The
            # gate is spelled as the membership test `_inert_pin_note` uses, not as
            # a `.get()`: one condition, two channels, or the phase has shipped two
            # predicates that can disagree.
            "inert_collection_pin": (
                config.inert_file_keys["qdrant_collection"]
                if "qdrant_collection" in config.inert_file_keys else None
            ),
            "graph_behind_buffer": graph_behind,
            "corpus_divergence": divergence_report,
            "scope_overflow": overflows,
            "agent_guide_version": AGENT_GUIDE_VERSION,
            "agent_files": agent_drift["files"],
        })
        return 0 if ready else 1

    verdict = "READY ✓" if ready else ("NEEDS ATTENTION ⚠" if initialized else "NOT SET UP ✗")
    mark = lambda ok: "✓" if ok is True else ("✗" if ok is False else "—")
    # An absent collection on a reachable Qdrant is never a readiness ✗ — it is
    # either the fresh state (nothing recorded yet) or a wipe over a populated
    # graph, and the two branches below say which. Neutral "—" plus a note, so the
    # mark can't contradict an otherwise-READY verdict while the note still carries
    # the heal.
    if not q["reachable"]:
        coll_mark, coll_hint = None, "needs Qdrant up (see above)"
    elif q["collection_exists"]:
        coll_mark, coll_hint = True, None
    elif active_ids:
        # Absent collection but a populated graph = a full wipe, not a fresh project.
        # Say so accurately rather than the calm "none recorded yet" (the warning
        # below carries the detail); still neutral, never a readiness ✗.
        coll_mark, coll_hint = (
            None,
            f"missing — {len(active_ids)} active node(s) have no vectors; run `mitos reconcile`",
        )
    elif graph_unbuilt:
        # The calm fresh-project sentence below is FALSE over a corpus of hundreds,
        # and it is part of the same lie the rung exists to stop: an absent
        # collection here means nothing has been indexed because nothing has been
        # BUILT, not because nothing has been decided. Defer to the graph rung
        # rather than contradict it — with no graph, the collection is downstream.
        coll_mark, coll_hint = None, "nothing indexed yet — the graph is unbuilt (see below)"
    else:
        coll_mark, coll_hint = None, "auto-created on first record — none recorded yet"
    checks = [
        ("workspace (.mitos/ + config.toml)", mitos_dir_ok, "run `mitos init`"),
        # The path rides INLINE in the label, not on a line elsewhere: the row's
        # own job is to say which file it is about, and a reader must never have to
        # correlate a `✗` row with a path listed further down to learn that. The
        # parenthetical is the report's shipped idiom (`Qdrant reachable (…)`,
        # `GEMINI_API_KEY (from …)`). The `mitos init` hint stays bare because
        # `init` is selector-exempt — a bare `mitos init` is runnable, not a wall.
        (f"decisions.md buffer ({config.decisions_file})", decisions_ok,
         "created by `mitos init`"),
        # Reference copy for humans/agents — the parser reads the spec from the
        # installed package, so a missing workspace copy never gates readiness:
        # neutral "—", never a ✗ under a READY ✓ verdict (✗ is for real blockers).
        ("format-spec.md", True if spec_ok else None,
         "restore the reference copy: re-run `mitos init` (non-destructive)"),
        ("GEMINI_API_KEY" + (f" (from {key_source})" if key_source else ""), key_ok,
         "set it once for all projects: `mitos set-key --global <KEY>`"),
        (f"Qdrant reachable ({config.qdrant_url})", q["reachable"],
         "start it: `docker compose up -d` in the mitos repo"),
        # The 4th slot is an ALWAYS-printed follow-up line for this row (see the
        # print loop). The inert-pin note has to be one: it is a config fact, not a
        # service fact, so it must render in all four coll_mark branches above —
        # `coll_hint` renders only when the mark is not ✓, and a pinned workspace
        # whose collection exists is exactly a case where the two names disagree
        # visibly. Nesting it in the `collection_exists` arm would drop it from the
        # unreachable/absent branches, which are the MOST likely to be confused
        # about which name is in force.
        (f"collection '{config.qdrant_collection}'", coll_mark, coll_hint,
         _inert_pin_note(config, offer_deletion=True)),
        # No MCP-wiring row lives here any more. Wiring is a one-time machine-global
        # act since the registry landed, so it is not a per-project readiness rung
        # in either direction — and its replacement is a *finding* that must be
        # ABSENT on a healthy project, while every `checks` row renders
        # unconditionally. It prints below, beside the agent-file drift note.
    ]
    # A pre-V1a (prototype) graph is the dominant blocker — surface it prominently,
    # right after the workspace line, with the same route-to-cutover guidance `init`
    # raises. Never `READY ✓` for a graph `init` would refuse (§5.2.7).
    if pre_v1a:
        # Route through the shared constant (single source) so this check-line hint
        # and the next-steps line below can never re-diverge — both name `mitos
        # cutover`. (The store.py boot-guard message stays its own deeper-internal
        # phrasing; it is not an operator-primary surface.)
        checks.insert(1, ("graph schema (V1a)", False, _CUTOVER_GUIDANCE))
    print(f"\nMITOS STATUS for {_status_subject(workspace_dir, project)} — {verdict}\n")
    # `*rest` tolerates both widths, so the 3-tuple rows (including the pre-V1a row
    # inserted above) need no change and a future row can add a follow-up without
    # touching the others. A hint is conditional on the mark; a follow-up note is not.
    for label, ok, hint, *rest in checks:
        line = f"  {mark(ok)} {label}"
        if ok is not True and hint:
            line += f"   → {hint}"
        print(line)
        if rest and rest[0]:
            print(f"      {rest[0]}")
    if q["reachable"] and q["collection_exists"] and q["points"] is not None:
        print(f"      ({q['points']} vector(s) indexed)")
    # The three corpus locations that have no check row to hang off, as neutral `•`
    # facts in the idiom the report already speaks (`• graph holds …`, `• {n}
    # graveyard point(s)`) — a resolved path passes and fails nothing, so it is not
    # a rung in any glyph and never joins `checks`. Unconditional by design: the two
    # sessions that grepped `.mitos/decisions.md` and got confident zero-hit answers
    # were working in HEALTHY workspaces where every row read ✓, so a path shown
    # only on failure is absent from exactly the state that produced this item. The
    # archive is the load-bearing member — `<root>/decisions/archive` follows
    # neither the root convention `decisions.md` teaches nor the `.mitos/` one
    # everything else does, so a reader told the other three still cannot derive it.
    # Rendered, never inspected: no `stat`, no glob, no entry count. The archive
    # listing is one import away and already on this verb's path
    # (`corpus_graph_divergence`), and a count beside a path would be a second,
    # weaker report of what the divergence rung already covers — drifting from it by
    # construction, and the only thing here that is not O(1) in corpus size.
    print(f"  • questions.md buffer: {config.questions_file}")
    print(f"  • decisions archive: {config.archive_dir}")
    print(f"  • graph: {config.db_path}")
    if graph_nodes is not None:
        print(f"  • graph holds {graph_nodes} node(s)")
    # The unbuilt graph (W31). A blocker, not a rung readers learn to skip: every
    # semantic read over this workspace answers cleanly empty while the corpus holds
    # entries, so the caller is told "no precedent" for a project that has them.
    #
    # The heal is `mitos sync` and EMPHATICALLY not `mitos reconcile`: reconcile
    # diffs an empty active set against an absent collection, finds nothing to
    # enqueue, and reports success on a workspace it did not touch — converting a
    # recoverable state into one the operator believes they already fixed. That is
    # one word away from being the wrong answer, which is why a row asserts the
    # word's absence from this rung.
    if graph_unbuilt:
        print(
            "\n  ⚠ the graph is unbuilt — decisions.md holds entries but the graph "
            "holds no nodes, so every read answers empty and reads as 'no precedent'. "
            "Run `mitos sync` to build it (usually a clone: the graph is gitignored, "
            "the corpus is not)."
        )
    # Vector-completeness verdict from the exact id-diff computed above (not a
    # count). Three outcomes:
    #   • scroll failed (missing_active_slugs is None) → we could not verify; say so
    #     and never fall back to the count proxy (structurally blind, per
    #     status-vector-completeness-by-id-diff-not-count-proxy).
    #   • missing_active_slugs non-empty → the warned state: active nodes with no
    #     vector, invisible to semantic surface/query (names slugs at small N).
    #   • empty → verified complete; stay quiet.
    # Orphan (graveyard) points are reported neutrally, never as a warning — they
    # are the blackout vector's substrate (graveyard-vectors-now-consumed-by-blackout).
    # Each branch is self-guarding on the diff outcome above — no collection_exists
    # gate here, so a full collection wipe with a populated graph (which sets
    # missing_active_slugs to the whole active set) warns just like a points wipe.
    if scroll_failed:
        print(
            "\n  ⚠ could not verify vector completeness — Qdrant scroll failed; "
            "run `mitos status` again when Qdrant is reachable."
        )
    elif missing_active_slugs:
        n = len(missing_active_slugs)
        print(
            f"\n  ⚠ vector index incomplete — {n} active node(s) have no vector "
            f"and are invisible to semantic surface/query. Run `mitos reconcile` "
            f"to re-embed them (or `mitos sync` if the outbox is non-empty) — "
            f"informational, not a readiness blocker."
        )
        if embedding_seed:
            # Which of the two heals applies is knowable here, so say it rather than
            # leaving the operator to infer it from "if the outbox is non-empty".
            print(
                f"      the embedding queue was seeded by `{embedding_seed['established_by']}` "
                f"at {embedding_seed['established_at']} — `mitos sync` restores search"
            )
        if n <= 5:
            for slug in missing_active_slugs:
                print(f"      • {slug}")
    if orphan_points:
        print(
            f"  • {orphan_points} graveyard point(s) belong to inactive/removed "
            f"nodes — retained (they power all-superseded blackout recovery), "
            f"not an error."
        )
    if overflows:
        _print_overflow_detail(overflows, verbose=verbose)
    if divergence_report is not None:
        _print_divergence_rung(divergence_report, project=config.project)
    if graph_behind:
        print(
            "\n  ⚠ graph is behind your buffer — the V1b edge catalog + mechanism "
            "registry were never committed for this corpus (a schema upgrade widens "
            "the DDL but does not re-commit). Run `mitos rebuild` to populate them "
            "(informational — not a readiness blocker; no decisions are at risk)."
        )
    if mcp_project_entry:
        # A finding, not a check — so it prints only when there is something to
        # say, and a project with no `.mcp.json` reads clean. The wording states
        # the fact rather than prescribing a fix, because mitos cannot tell a
        # STALE entry from one deliberately kept identical to the machine-wide
        # registration, and both are legitimate: "delete it" would be wrong advice
        # half the time. Note also that this reads THIS directory only — an entry
        # at a parent launch root shadows the same way and is not visible here.
        print(
            "\n  ⚠ this project declares its own `mitos` MCP server in `.mcp.json`. "
            "A project-scope entry takes precedence over the machine-wide one by "
            "name — it does not fall back to it — so this file is what your "
            "sessions here get, and a broken entry leaves them with no mitos tools "
            "at all. Keep it in sync with your machine-wide registration, or remove "
            "it and let that one serve this project."
        )
    if agent_drift["stale"]:
        stale_files = ", ".join(
            f["file"] for f in agent_drift["files"]
            if f["status"] in ("outdated", "unversioned")
        )
        print(f"  ⚠ agent-file mitos note out of date ({stale_files}) "
              f"— refresh with `mitos agent-block`")
    print()
    if not ready:
        print("Next steps:")
        n = 1
        if pre_v1a:
            print(f"  {n}. {_CUTOVER_GUIDANCE}"); n += 1
        if not initialized:
            print(f"  {n}. `mitos init` here (creates .mitos/, decisions.md, scaffolds .env)"); n += 1
        if not key_ok:
            print(f"  {n}. Set your GEMINI_API_KEY once for all projects: "
                  f"`mitos set-key --global <KEY>` (or per-project: `mitos set-key <KEY>`)"); n += 1
        if not q["reachable"]:
            print(f"  {n}. Start Mitos's Qdrant: `docker compose up -d` from the mitos repo"); n += 1
        if graph_unbuilt:
            print(f"  {n}. Build the graph from your corpus: `mitos sync` "
                  f"(the graph is derivative — decisions.md is the source)"); n += 1
        print("  Full walkthrough → SETUP.md "
              "(https://github.com/dovahkiin-v/mitos/blob/main/SETUP.md)")
        print()
    return 0 if ready else 1


def _print_divergence_rung(report: Dict[str, Any], *, project: str) -> None:
    """Prints the corpus↔graph divergence rung — informational, never a blocker.

    A corpus mid-edit is a normal state, so gating readiness here would make an
    ordinary hand edit read as breakage; and a clean corpus must print nothing at all,
    since a rung that speaks on healthy projects is a rung readers learn to skip.

    Phrased like the vector-completeness rung above it, and for the same reason: this
    is a SENSOR, and the repair verbs (`mitos sync`, `mitos restore-source`) are named
    so the reader has somewhere to go rather than only something to worry about.

    Every key is read with ``.get``: the report may have come from the sidecar cache,
    written by a build whose species set differed, and a ``KeyError`` raised from here
    would take down the one command an operator runs to find out what is wrong. The
    cache key carries a schema version so this should not arise — belt and braces,
    because the cost of the braces is one method call.

    Args:
        report: A ``corpus_graph_divergence`` result.
        project: The caller's own vocabulary for this workspace — ``config.project``,
            i.e. the registered name for a registered target and the workspace path
            otherwise. Required and keyword-only, matching ``cmd_status``'s own
            idiom: the one recipe this function composes for a repair the reader can
            run NOW has to carry a selector (since the selector flip a bare
            ``mitos sync`` has no target), and it is passed in rather than
            re-derived here for the reason 3d rejected by name — a reverse lookup
            misses on a symlinked route whose registry entry is hand-written
            non-canonically, printing a path for a registered project with every
            other row green.
    """
    if report.get("skipped"):
        # Never a verdict from a read we could not take. "corpus busy" is the normal
        # case — another process holds the lock — and is not worth a ⚠.
        if report["skipped"] == "corpus busy":
            print("\n  • divergence check skipped — corpus busy (another mitos "
                  "process holds the lock); re-run when it finishes.")
        return

    total = divergence_total(report)
    if total == 0:
        return

    print(f"\n  ⚠ corpus and graph disagree in {total} place(s) — informational, "
          f"not a readiness blocker.")

    commentary, scope = report.get("commentary") or [], report.get("scope") or []
    if commentary:
        print(f"      • {len(commentary)} entry(s) whose commentary text differs "
              f"(the graph serves the stale value to every read)")
        for row in commentary[:5]:
            print(f"          - {row['slug']}: {', '.join(row['fields'])}")
    if scope:
        print(f"      • {len(scope)} entry(s) whose scope differs — a FINDABILITY "
              f"defect: scope-filtered reads and `mitos scopes` miss them")
        for row in scope[:5]:
            print(f"          - {row['slug']}: graph {row['graph']} vs "
                  f"markdown {row['markdown']}")
    if report.get("edges"):
        print(f"      • {len(report['edges'])} entry(s) whose declared relations "
              f"differ from the stored edges")
        # A verdict, not a count. Detection was never the gap — what a reader needs is
        # which of these a verb can actually repair, because the generic "run `mitos
        # sync`" below is wrong for the illegal ones and they will never drain.
        verdicts = report.get("edge_verdicts") or {}
        if verdicts.get("repairable"):
            print(f"          - {verdicts['repairable']} declared edge(s) whose target "
                  f"is active and legal — `mitos rebuild` replays them")
        if verdicts.get("target_retired"):
            print(f"          - {verdicts['target_retired']} point at a since-retired "
                  f"target — legal, but a replay must reach them in COMMIT order "
                  f"(citations resolve against the active view)")
        if verdicts.get("unresolvable"):
            print(f"          - {verdicts['unresolvable']} name no entry in the graph "
                  f"— fix the citation, or `mitos restore-source` if its block went "
                  f"missing")
        if verdicts.get("illegal"):
            offenders = report.get("illegal_edge_types") or []
            named = f" ({', '.join(sorted(offenders))})" if offenders else ""
            print(f"          - {verdicts['illegal']} can NEVER commit{named} — the "
                  f"kind matrix forbids them where they are declared. No verb repairs "
                  f"these: re-author the relation (a decision citing a precedent wants "
                  f"`Cites:`), or remove the line.")
    if report.get("source"):
        print(f"      • {len(report['source'])} entry(s) whose `**Source:**` line "
              f"differs from the stored provenance — a rebuild would adopt the "
              f"markdown value; restore the line to match the graph")
    if report.get("graph_only"):
        active = sum(1 for row in report["graph_only"] if row.get("active"))
        print(f"      • {len(report['graph_only'])} node(s) have NO `### ` block in "
              f"the corpus ({active} active) — `mitos rebuild` cannot reconstruct "
              f"them, so its completeness gate refuses. Run "
              f"`mitos restore-source --all-graph-only --dry-run` to review.")

    reconcilable = report.get("reconcilable") or 0
    if reconcilable:
        print(f"      → {reconcilable} of these can be repaired now: `mitos sync -p "
              f"{project!r}` reconciles a diverged buffer entry, printing the field "
              f"diff first (add `--yes` to apply without prompting, or "
              f"`--reconcile-entry <slug>` to apply one named entry's whole "
              f"reconcile — the only way to apply an edge DELETION unattended).")
    if report.get("archived_drift"):
        print(f"      ({report['archived_drift']} of these sit in an ARCHIVE file — "
              f"`sync` reads only the buffer, so their reconciler is `mitos rebuild`.)")


def cmd_restore_source(
    config: MitosConfig,
    *,
    slug: Optional[str] = None,
    all_graph_only: bool = False,
    dry_run: bool = False,
    as_json: bool = False,
) -> int:
    """Re-materializes the `### slug` source block of a node the corpus has lost.

    A graph-only node is invisible to `mitos rebuild`, which replays only what the
    markdown holds — so the node is dropped and the completeness gate refuses the
    swap, disabling the repair path on exactly the corpus that needs it. The graph
    already carries every field the parser reads, so this is a derivation rather than
    an authoring act, and it refuses to write anything whose round trip it cannot
    prove.

    Restored into the BUFFER, never an archive: archives are quarter-partitioned and
    `created_at` is stamped at commit time, so choosing a quarter would put a
    fabricated date in the gold source.

    Args:
        config: The workspace config.
        slug: Restore one node by slug.
        all_graph_only: Restore every node with no source block.
        dry_run: Print what would be written and write nothing.
        as_json: Emit one machine-readable object.

    Returns:
        ``0`` on success or a clean no-op, ``1`` when something was refused.
    """
    from mitos.divergence import corpus_graph_divergence
    from mitos.restore import (
        RestoreError,
        render_source_block,
        verify_block_in_isolation,
        verify_whole_buffer,
    )

    # Every refusal below answers on stderr and every report on stdout, so the echo
    # rides each branch rather than leading the handler. This verb writes
    # `config.decisions_file` — the user-authored gold source P6 makes immutable —
    # so naming the corpus is not decoration here: a mis-aimed
    # `--all-graph-only` rewrites *another* project's input in bulk.
    if bool(slug) == bool(all_graph_only):
        msg = "restore-source needs exactly one of --slug or --all-graph-only."
        if as_json:
            _emit_json({"error": msg, "code": "ambiguous_target",
                        **corpus_provenance(config)})
        else:
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 1

    if not os.path.exists(config.db_path):
        msg = "No graph found — nothing to restore from."
        if as_json:
            _emit_json({"error": msg, "code": "no_graph", **corpus_provenance(config)})
        else:
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 1

    store = GraphStore(config.db_path, read_only=True)
    report = corpus_graph_divergence(store, config)
    if report.get("skipped"):
        msg = f"Cannot restore — {report['skipped']}."
        if as_json:
            _emit_json({"error": msg, "code": "unavailable", **corpus_provenance(config)})
        else:
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 1

    orphan_slugs = [row["slug"] for row in report["graph_only"]]
    if slug is not None:
        if slug not in orphan_slugs:
            msg = (f"'{slug}' is not a graph-only node — it already has a `### ` block "
                   f"in the corpus, or no such node exists.")
            if as_json:
                _emit_json({"error": msg, "code": "not_graph_only",
                            **corpus_provenance(config)})
            else:
                _echo_corpus(config, file=sys.stderr)
                print(msg, file=sys.stderr)
            return 1
        targets = [slug]
    else:
        # Emit in COMMIT order, not the detector's report order (which is actives
        # alphabetically, then retireds). Order decides whether the restored SET
        # replays at all: every citation resolves against the active view, so a
        # supersession emitted before the entry that amends its victim retires the
        # target early and the amend is rejected. Measured on the live corpus — report
        # order produced 24 missing cores and 31 casualties from three such roots, all
        # of which vanished in commit order. `rowid` order is provably legal because it
        # is the sequence in which those commits already succeeded.
        slug_of = {n["id"]: n["slug"] for n in store.get_all_nodes()}
        rank_of_slug = {}
        for position, node_id in enumerate(store.node_ids_in_commit_order()):
            slug = slug_of.get(node_id)
            if slug is not None:
                rank_of_slug[slug] = position
        targets = sorted(orphan_slugs,
                         key=lambda s: rank_of_slug.get(s, len(rank_of_slug)))

    if not targets:
        if as_json:
            _emit_json({"restored": [], "refused": [], "dry_run": dry_run,
                        "written": False, **corpus_provenance(config)})
        else:
            _echo_corpus(config)
            print("No graph-only nodes — every node has a source block. ✓")
        return 0

    nodes_by_slug = {n["slug"]: n for n in store.get_all_nodes()}
    blocks: List[Tuple[str, str]] = []
    refused: List[Dict[str, str]] = []
    for target in targets:
        node = nodes_by_slug.get(target)
        if node is None:
            refused.append({"slug": target, "reason": "node not found"})
            continue
        try:
            block = render_source_block(
                node,
                store.get_outgoing_edges(node["id"]),
                store.get_transcript(node["id"]),
            )
            verify_block_in_isolation(block, node)
        except RestoreError as exc:
            refused.append({"slug": target, "reason": str(exc)})
            continue
        blocks.append((target, block))

    written = False
    if blocks and not dry_run:
        manager = MitosSyncManager(config)

        def _splice(original: str) -> str:
            """Inserts every block just below the entries marker, newest-first."""
            baseline["before"] = original
            # REVERSED: the buffer is authored newest-first, so the oldest-committed
            # block must land lowest — `parse_file_reversed` flips the file back to
            # oldest-first for replay, which is the order the edges need.
            payload = "\n\n".join(b.rstrip("\n") for _s, b in reversed(blocks))
            if _ENTRIES_MARKER in original:
                return original.replace(
                    _ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{payload}\n", 1
                )
            return original.rstrip("\n") + f"\n\n{payload}\n"

        # Captured INSIDE `transform`, which `splice_buffer` calls under the lock and
        # after auto-heal. Reading it out here instead would compare the splice against
        # a pre-lock snapshot, so an entry appended by a concurrent `record` would
        # surface as "the splice disturbed a neighbouring entry" — a fabricated
        # diagnosis for a correct splice, and a refusal the operator cannot act on.
        baseline: Dict[str, str] = {}

        def _verify(after_text: str) -> None:
            """Whole-buffer fidelity: isolation cannot prove neighbour safety."""
            verify_whole_buffer(baseline["before"], after_text, added=len(blocks))

        try:
            manager.splice_buffer(_splice, after_write=_verify)
            written = True
        except (RestoreError, MitosError, OSError) as exc:
            # RestoreError = the fidelity check refused (buffer already rolled back).
            # OSError = an unwritable workspace or an unavailable lock file.
            # MitosError = splice_buffer's both-write-AND-rollback-failed case, whose
            # message names the state the file is in. All three must land in the
            # structured report rather than escaping to `main()`, which would print a
            # bare stderr line and leave `--json` with EMPTY stdout — a caller parsing
            # this verb would see no object at all.
            refused.extend({"slug": s, "reason": str(exc)} for s, _b in blocks)
            blocks = []

    restored = [s for s, _b in blocks]
    if as_json:
        _emit_json({
            "restored": restored,
            "refused": refused,
            "dry_run": dry_run,
            "written": written,
            "path": config.decisions_file,
            **corpus_provenance(config),
        })
        return 1 if refused else 0

    # One run can answer on BOTH channels (a partial write reports on stdout and
    # refuses on stderr), so the echo leads each channel that carries an answer
    # rather than picking one — the same shape `cmd_sync` and `cmd_reconcile` take.
    if dry_run or written:
        _echo_corpus(config)

    if dry_run:
        print(f"\nWould restore {len(blocks)} source block(s) to "
              f"{config.decisions_file} — nothing written (--dry-run).\n")
        for target, block in blocks:
            print(f"--- {target} " + "-" * max(0, 60 - len(target)))
            print(block)
    elif written:
        print(f"\nRestored {len(restored)} source block(s) to {config.decisions_file} ✓")
        for target in restored:
            print(f"  • {target}")
        print("\nReview the entries, then `mitos rebuild --json` to confirm the "
              "completeness gate passes.")
    if refused:
        # Flush first: stderr is unbuffered while a piped stdout is not, so without
        # this the refusals overtake the report they annotate (the inversion
        # `cmd_init` and `cmd_record` already guard against — this branch had no
        # such flush before, and the leading echo above makes the order load-bearing).
        sys.stdout.flush()
        _echo_corpus(config, file=sys.stderr)
    for row in refused:
        print(f"  ✗ refused: {row['slug']} — {row['reason']}", file=sys.stderr)
    if refused and not written and not dry_run:
        print("\nNothing was written.", file=sys.stderr)
    print()
    return 1 if refused else 0


def cmd_agent_block(workspace_dir: str, check: bool = False) -> int:
    """Prints the canonical agent-file block, or checks pasted copies for drift.

    The block is the thin, versioned pointer a project pastes into its agent files
    (``AGENTS.md`` / ``CLAUDE.md`` / ``GEMINI.md`` / ``.cursorrules``) so the next
    agent knows mitos is set up here. Without ``--check`` it prints the current block
    to stdout, paste-ready; with ``--check`` it scans the project's agent files and
    reports which carry an out-of-date or unversioned mitos note.

    Args:
        workspace_dir: The project root (only used by ``--check``).
        check: Report drift in the project's agent files instead of printing the block.

    Returns:
        ``0`` on a plain print, or when ``--check`` finds no stale copy; ``1`` when
        ``--check`` finds an outdated/unversioned mitos note to refresh.
    """
    if not check:
        print(agent_block())
        return 0

    workspace_dir = os.path.abspath(workspace_dir)
    report = agent_block_drift(workspace_dir)
    files = report["files"]
    print(f"\nAgent-file mitos note (current guide: v{AGENT_GUIDE_VERSION}) for {workspace_dir}\n")
    if not files:
        print("  — no agent file references mitos yet.")
        print("    Paste `mitos agent-block` into your AGENTS.md / CLAUDE.md / GEMINI.md so")
        print("    the next agent knows mitos is set up here.\n")
        return 0
    for f in files:
        if f["status"] == "current":
            print(f"  ✓ {f['file']}  (guide v{f['marker_version']})")
        elif f["status"] == "outdated":
            print(f"  ⚠ {f['file']}  (guide v{f['marker_version']} → v{AGENT_GUIDE_VERSION}) "
                  f"— refresh with `mitos agent-block`")
        else:  # unversioned
            print(f"  ⚠ {f['file']}  (mitos note with no version marker) "
                  f"— refresh with `mitos agent-block`")
    print()
    return 1 if report["stale"] else 0


def cmd_cutover(
    config: MitosConfig, *, allow_drops: bool, assume_yes: bool, as_json: bool
) -> int:
    """Runs the one-time prototype→V1a cutover (the destructive migration).

    Orchestrates 7a's verdict surface into an operator-runnable verb: probe →
    rebuild + gate → present the verdict → confirm (or override a shortfall with
    ``--allow-drops``) → atomic swap → print the post-swap runbook. The load-bearing
    correctness lives in :func:`~mitos.cutover.perform_swap`; this is the thin
    interactive orchestrator (K1).

    Only a genuine **prototype** graph proceeds — an already-V1a, empty, or absent
    graph is a cheap no-op (G7), which also makes a post-success or post-crash
    re-run idempotent. A **corpus defect** raises ``CutoverError`` (propagated to
    ``main()``'s boundary, rendered one-line, exit 1) and is never overridable; a
    completeness **shortfall** is overridable with ``--allow-drops`` (P6 — the
    markdown is authoritative, a drop may be a deliberate purge).

    Args:
        config: The active workspace config.
        allow_drops: Proceed past a completeness shortfall (active cores absent from
            the rebuild). Never overrides a corpus defect.
        assume_yes: Skip the interactive swap confirmation (automation / non-TTY).
        as_json: Emit a machine-readable JSON report instead of the human runbook.

    Returns:
        ``0`` on a successful swap (or a no-op non-prototype graph), ``1``
        otherwise (absent graph, refused shortfall, declined/missing confirmation).

    Raises:
        CutoverError: On a corpus defect during the rebuild (caught at the
            ``main()`` boundary).
    """
    # 0. The echo leads every text branch — including the two early returns below
    #    and `rebuild_and_gate`'s own stdout — so one call covers all 28 sites. All
    #    of them are stdout; this verb has no stderr answer.
    if not as_json:
        _echo_corpus(config)

    # 1. Up-front prototype probe (G7) — mirrors the cmd_init / cmd_status RO-probe
    #    shape. An absent / already-V1a / empty graph is a cheap no-op: no rebuild,
    #    no swap, no Qdrant churn (and a post-success re-run is idempotent).
    if not os.path.exists(config.db_path):
        if as_json:
            # The stamp overwrites the hand-built `workspace` key with the same
            # value (`corpus_provenance` reads `config.workspace_dir` too), so it
            # is additive in effect on these four early returns. The `to_dict()`
            # shapes below carry no `workspace` key at all — `aside_db_path` is a
            # *different* directory and nothing here touches it.
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "no_graph",
                        **corpus_provenance(config)})
        else:
            print("No graph found at this workspace — run `mitos init` for a fresh "
                  "V1a workspace (nothing to cut over).")
        return 1
    probe_conn = open_connection(config.db_path, read_only=True)
    try:
        is_prototype = is_pre_v1a_schema(probe_conn)
    finally:
        probe_conn.close()
    if not is_prototype:
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "not_a_prototype",
                        **corpus_provenance(config)})
        else:
            print("Graph is already on the V1a schema (or empty) — nothing to "
                  "cut over.")
        return 0

    # 2. Rebuild + gate (7a). A corpus defect raises CutoverError, which propagates
    #    to main()'s `except MitosError` boundary (one-line error, exit 1) — never
    #    overridable here, it is malformed markdown the operator must fix.
    aside_db_path = default_aside_db_path(config)
    result = rebuild_and_gate(config, aside_db_path=aside_db_path, quiet=as_json)

    qdrant_wipe_cmd = (
        f"curl -X DELETE {config.qdrant_url}/collections/{config.qdrant_collection}"
    )

    # 3. Present the verdict.
    if not as_json:
        print("\nCutover rebuild verdict:")
        print(f"  decisions committed:       {result.decisions_committed}")
        print(f"  open questions committed:  {result.open_questions_committed}")
        print(f"  active cores (old graph):  {result.reference_active_count}")
        print(f"  active cores (rebuild):    {result.reconstructed_active_count}")

    if not result.gate_passed:
        n = len(result.missing_cores)
        if not as_json:
            print(f"\n⚠ {n} active core(s) from the prototype are ABSENT from the "
                  f"rebuild:")
            for mc in result.missing_cores:
                print(f"    - '{mc.slug}' [{mc.kind}]: {mc.axiom_excerpt}")
        if not allow_drops:
            if as_json:
                _emit_json({**result.to_dict(), "swapped": False,
                            "reason": "shortfall_refused",
                            "qdrant_wipe_cmd": qdrant_wipe_cmd,
                            **corpus_provenance(config)})
            else:
                print(f"\nRefusing to swap: {n} active core(s) would be dropped. "
                      f"Review the offenders above. If this purge is intentional "
                      f"(they were deliberately removed from the corpus), re-run "
                      f"with --allow-drops. Otherwise restore them in "
                      f"{os.path.basename(config.decisions_file)} and re-run.")
            return 1
        if not as_json:
            print(f"\n--allow-drops set: proceeding despite the {n} dropped "
                  f"core(s), treating the corpus as authoritative (P6).")

    # 4. Confirm the destructive swap (K5/G5 — never call input() on a no-TTY).
    if not assume_yes:
        if as_json:
            # JSON mode is for automation: never prompt; require an explicit --yes.
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "confirmation_required",
                        "qdrant_wipe_cmd": qdrant_wipe_cmd,
                        **corpus_provenance(config)})
            return 1
        if sys.stdin.isatty():
            answer = input("\nProceed with the cutover swap? This replaces the "
                           "live graph. [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted — no changes made.")
                return 1
        else:
            print("\nRefusing to prompt: this is a destructive operation and stdin "
                  "is not a TTY. Re-run with --yes to proceed non-interactively.")
            return 1

    # 5. Swap — the single atomic instant. The timestamp is pinned by the caller
    #    (G8) so perform_swap stays wall-clock-free and fixture-deterministic.
    bak_path = perform_swap(
        config, result.aside_db_path, timestamp=time.strftime("%Y%m%d-%H%M%S")
    )

    # 6. Print the post-swap runbook (the operator must not have to remember it).
    if as_json:
        _emit_json({**result.to_dict(), "swapped": True,
                    "bak_path": bak_path,
                    "qdrant_wipe_cmd": qdrant_wipe_cmd,
                    **corpus_provenance(config)})
        return 0

    print(f"\n✓ Cutover complete — the V1a graph is live at {config.db_path}.")
    if bak_path:
        print(f"  Old prototype graph backed up to: {bak_path}")
    print("\nFinish the cutover (it is not fully done until these run):")
    print("  1. Wipe the stale Qdrant collection (its vectors are keyed on the old")
    print("     prototype ids — it auto-recreates on the next sync):")
    print(f"       {qdrant_wipe_cmd}")
    print("  2. Re-embed the V1a active set:  mitos sync   (or: mitos sync --embed-only)")
    print("     Semantic surface/query stay degraded until the queue drains;")
    print("     graph-only `mitos list` works throughout.")
    print("     (If you ever wipe Qdrant later with the outbox already empty,")
    print("      `mitos reconcile` re-embeds the active set in one pass.)")
    print("  3. If `mitos serve` was running, restart it.")
    print("  4. Verify:  mitos status   → expect READY ✓")
    if bak_path:
        print(f"  5. Once satisfied, remove the backup:  rm {bak_path}")
    print("  Full runbook → SETUP.md → Cutover.")
    return 0


def _print_rebuild_remediation(casualties, missing_cores, decisions_basename: str) -> None:
    """Prints reassuring, per-class remediation when a rebuild is refused.

    The upgrade-path UX (no stranger's experience is broken): a user who hits a stale
    citation must learn three things at once — their decisions are SAFE, exactly WHAT
    to do per failure class, and that ``--allow-drops`` is a safe escape — never a
    bare ``refused`` wall.

    Args:
        casualties: The :class:`~mitos.cutover.Casualty` punch-list (each carries
            ``codes`` + a ``detail`` that already names any superseding successor).
        missing_cores: Active decisions absent from the rebuild (a corpus removal,
            not a citation defect) — guided separately.
        decisions_basename: The buffer filename to point edits at (e.g.
            ``decisions.md``).
    """
    print(
        f"\nRefusing to swap — the live graph is untouched and nothing is lost: "
        f"{decisions_basename} (plus the archives) is the source of truth, and every "
        f"entry below stays there. Here is how to clear each one:"
    )
    codes = {code for c in casualties for code in c.codes}
    if "dangling_edge" in codes:
        print(
            "  • dangling_edge — the entry cites a decision that has since been "
            "superseded. Re-point that citation to the active successor named in the "
            "detail above (or delete the citation line), then re-run `mitos rebuild`."
        )
    if "missing_target" in codes:
        print(
            "  • missing_target — the entry cites a slug that no longer exists "
            "(renamed away, or a typo). Fix or remove the citation, then re-run."
        )
    other = sorted(codes - {"dangling_edge", "missing_target"})
    if other:
        print(
            f"  • {', '.join(other)} — see the detail above; fix the entry in "
            f"{decisions_basename} and re-run."
        )
    if missing_cores:
        print(
            "  • Some active decisions are absent from the corpus entirely (a removal, "
            "not a citation defect). If that is intentional, --allow-drops accepts it; "
            "otherwise restore them in the buffer."
        )
    print(
        "\nOr re-run `mitos rebuild --allow-drops` to proceed now — the listed entries "
        "remain in your markdown and re-enter the graph the moment you fix the "
        "citation and rebuild again."
    )


def cmd_rebuild(
    config: MitosConfig, *, allow_drops: bool, assume_yes: bool, as_json: bool
) -> int:
    """Rebuilds the graph from the full corpus through the current catalog.

    The recurring twin of :func:`cmd_cutover`: re-commits every decision and open
    question oldest-first (archives then buffer) into a build-aside graph and
    atomically swaps it in, so a graph upgraded in place (the V1b schema on pre-V1b
    data — the catalog flip's edges and the mechanism registry never re-committed)
    gains the full catalog. Unlike cutover it runs on a **current** (V1a/V1b) graph
    and is **resilient**: an entry the catalog now rejects (a citation to a since-
    superseded or never-authored node) is a surfaced casualty, not an abort. No ADRs
    are at risk — the markdown (buffer + archives) is the source of truth (M7/P6) and
    the swap backs up the old graph.

    A graph **format** defect still raises ``CutoverError`` (propagated to ``main()``).
    A **casualty** (an entry that cannot commit) or a completeness **shortfall** (an
    active decision the rebuild would drop) blocks the swap unless ``--allow-drops``.

    Args:
        config: The active workspace config.
        allow_drops: Proceed past casualties / a shortfall (the dropped entries stay
            in the markdown; fix their citations and re-run to re-include them).
        assume_yes: Skip the interactive swap confirmation (automation / non-TTY).
        as_json: Emit a machine-readable JSON report instead of the human summary.

    Returns:
        ``0`` on a successful swap, ``1`` otherwise (absent/prototype graph, refused
        casualties/shortfall, declined/missing confirmation).

    Raises:
        CutoverError: On a corpus format defect during the rebuild (caught at the
            ``main()`` boundary).
    """
    # 0. One leading echo for all 21 text sites — every one of them is stdout,
    #    including the two early returns and `rebuild_and_gate`'s own output.
    if not as_json:
        _echo_corpus(config)

    # 1. Probe: rebuild runs on a CURRENT graph. Absent → init; prototype → the
    #    one-time cutover owns it (don't double-handle).
    if not os.path.exists(config.db_path):
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "no_graph",
                        **corpus_provenance(config)})
        else:
            print("No graph found at this workspace — run `mitos init` first "
                  "(nothing to rebuild).")
        return 1
    probe_conn = open_connection(config.db_path, read_only=True)
    try:
        is_prototype = is_pre_v1a_schema(probe_conn)
    finally:
        probe_conn.close()
    if is_prototype:
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "prototype_graph",
                        **corpus_provenance(config)})
        else:
            print("Graph is a pre-V1a prototype — run `mitos cutover` (the one-time "
                  "migration) instead of `mitos rebuild`.")
        return 1

    # 2. Rebuild + gate (resilient: casualties are surfaced, not raised). A corpus
    #    FORMAT defect still raises CutoverError → main()'s boundary (exit 1).
    aside_db_path = default_aside_db_path(config)
    result = rebuild_and_gate(config, aside_db_path=aside_db_path, strict=False,
                              quiet=as_json)

    # 3. Present the verdict.
    if not as_json:
        print("\nRebuild verdict:")
        print(f"  decisions committed:       {result.decisions_committed}")
        print(f"  open questions committed:  {result.open_questions_committed}")
        print(f"  active cores (live graph): {result.reference_active_count}")
        print(f"  active cores (rebuild):    {result.reconstructed_active_count}")

    casualties = result.residual_casualties
    if casualties and not as_json:
        noun = "entry" if len(casualties) == 1 else "entries"
        print(f"\n⚠ {len(casualties)} {noun} could not be rebuilt (left in the buffer "
              f"— fix the citation to re-include):")
        for c in casualties:
            code_str = ", ".join(c.codes) if c.codes else "rejected"
            print(f"    - '{c.slug}' (lines {c.line_start}-{c.line_end}) "
                  f"[{code_str}]: {c.detail}")

    if not result.gate_passed and not as_json:
        n = len(result.missing_cores)
        print(f"\n⚠ {n} active decision(s) in the live graph would be DROPPED by this "
              f"rebuild:")
        for mc in result.missing_cores:
            print(f"    - '{mc.slug}' [{mc.kind}]: {mc.axiom_excerpt}")

    blocked = bool(casualties) or not result.gate_passed
    if blocked and not allow_drops:
        if as_json:
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "casualties_or_shortfall_refused",
                        **corpus_provenance(config)})
        else:
            _print_rebuild_remediation(
                casualties, result.missing_cores, os.path.basename(config.decisions_file)
            )
        return 1
    if blocked and not as_json:
        print("\n--allow-drops set: proceeding despite the dropped content, treating "
              "the corpus as authoritative (P6). Dropped entries remain in the markdown.")

    # 4. Confirm the destructive swap (never call input() on a no-TTY).
    if not assume_yes:
        if as_json:
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "confirmation_required",
                        **corpus_provenance(config)})
            return 1
        if sys.stdin.isatty():
            answer = input("\nProceed with the rebuild swap? This replaces the live "
                           "graph (a backup is kept). [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted — no changes made.")
                return 1
        else:
            print("\nRefusing to prompt: this replaces the live graph and stdin is "
                  "not a TTY. Re-run with --yes to proceed non-interactively.")
            return 1

    # 5. Swap — the single atomic instant (timestamp pinned by the caller, G8).
    bak_path = perform_swap(
        config, result.aside_db_path, timestamp=time.strftime("%Y%m%d-%H%M%S")
    )

    # 6. Post-swap guidance.
    if as_json:
        _emit_json({**result.to_dict(), "swapped": True,
                    "bak_path": bak_path,
                    **corpus_provenance(config)})
        return 0

    print(f"\n✓ Rebuild complete — the graph at {config.db_path} now reflects the "
          f"full catalog from your corpus.")
    if bak_path:
        print(f"  Old graph backed up to: {bak_path}")
    print("\nNext:")
    print("  - Re-embed so semantic surface/query reflect the rebuild:  mitos sync")
    print("    (Or, if Qdrant was wiped directly and the outbox is empty:  mitos reconcile)")
    print("  - Verify:  mitos status   → expect READY ✓ (the rebuild nudge clears)")
    if bak_path:
        print(f"  - Once satisfied, remove the backup:  rm {bak_path}")
    return 0


# =========================================================================== #
# Phase 3a — `mitos check`: the read-only corpus conflict audit / CI gate.
#
# Presentation + disposition only: the engine (mitos/check.py) computes every
# partition and count; `cmd_check` maps a typed CheckRunResult to human/JSON
# output and the shipped 0/1/2 exit contract (CHK-C2). The load-bearing rules —
# the exit table, the no-row-on-refusal rule (KD4), the plan→confirm→execute→
# row seam order (KD5), the `_emit_json`-only discipline — live in the phase plan.
# =========================================================================== #

# The parent's P15 per-check token budget estimate, per judged batch — used only
# to size the TTY confirm's disclosure (a rough figure, not a billed number).
_CHECK_TOKENS_PER_BATCH_ESTIMATE = 3000

# The four reverse-relation modifier stamp keys copied off a hydrated finding node
# (the `candidate_payload` manner, conflict.py). Single-sourced from the store's
# canonical map so a new modifier edge type never drifts this surface.
_CHECK_MODIFIER_STAMP_KEYS: Tuple[str, ...] = tuple(MODIFIER_EDGE_KEYS.values())


def _confirm_spend(n: int, *, assume_yes: bool, as_json: bool) -> Optional[int]:
    """The shared CHK-D5 spend confirm — corpus + staged (KD4).

    The single gate both ``check`` modes pass ``n`` (corpus: fresh judgment groups;
    staged: pending decision entries) through before any judge call, so the ``>``
    comparison and the three refusal surfaces can never fork between the two. Fires
    strictly ``n > check.CHECK_CONFIRM_BATCHES`` (read as a module attribute so a test
    monkeypatch is seen); at/below the threshold or with ``assume_yes`` it returns
    ``None`` (proceed) without prompting. All three refusals return exit ``2``, zero
    spend: ``--json`` emits an error object (automation never prompts), a non-TTY
    prints the vector message, an interactive decline prints "nothing spent".

    Args:
        n: The disclosure unit — the count of pending judgment batches.
        assume_yes: Waive the confirm (the ``--yes`` opt-in).
        as_json: Automation surface — emit an error object instead of prompting.

    Returns:
        A refusal exit code (``2``) when the spend is declined, or ``None`` to proceed.
    """
    if n <= check.CHECK_CONFIRM_BATCHES or assume_yes:
        return None
    if as_json:
        # Automation never prompts (a prompt would also corrupt the object).
        _emit_json({
            "error": (f"{n} judgment batches pending — re-run with --yes to "
                      f"authorize the spend."),
            "code": "confirmation_required",
            "batches_planned": n,
        })
        return 2
    if not sys.stdin.isatty():
        print(f"{n} judgment batches pending — re-run with --yes to authorize "
              f"the spend.", file=sys.stderr)
        return 2
    estimate = n * _CHECK_TOKENS_PER_BATCH_ESTIMATE
    print(f"{n} judgment batches pending (≈{estimate:,} tokens) — this run "
          f"will call the judge model.")
    if input("Proceed with the spend? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted — nothing spent.")
        return 2
    return None


def _build_check_substrate(
    config: MitosConfig,
) -> Tuple[Optional[GeminiEmbeddingProvider], QdrantVectorStore, Optional[str]]:
    """Constructs the two external substrate providers — one of them best-effort (KD2).

    ``GeminiEmbeddingProvider`` RAISES at construction when ``GEMINI_API_KEY`` is
    unset; that is caught NARROWLY (its own typed error, never a blanket
    ``except``) and degraded to ``None`` + a kept detail string, so an unexpected
    error still propagates. The disposition (refuse iff the run has sweep work) is
    the caller's — this only reports availability. Separated into a module-level
    helper so tests inject keyed fakes at this seam.

    ``QdrantVectorStore`` has **no construction-time failure mode**: its
    ``__init__`` contacts no network at all, so it is returned unconditionally.
    Both faults it used to report here — a missing collection and an unreachable
    Qdrant — now arrive at the *operation*, typed, and reach exit 2 through the
    sweep's ``Unavailable`` (ADR
    ``check-precondition-re-keys-from-store-construction-to-operation``). The
    fail-closed posture is unchanged; only the classification and the wording are.

    Args:
        config: The active workspace config (paths + Qdrant coordinates).

    Returns:
        ``(embed, vector, embed_detail)`` — the embedding provider or ``None`` with
        its failure message, and the vector store.
    """
    embed: Optional[GeminiEmbeddingProvider] = None
    embed_detail: Optional[str] = None
    try:
        embed = GeminiEmbeddingProvider(
            os.path.join(config.mitos_dir, "embedding_cache.sqlite"),
            api_key=config.env.get("GEMINI_API_KEY"),
            model_id=get_embedding_model_id(config.env),
        )
    except EmbeddingError as exc:
        embed_detail = str(exc)
    vector = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    return embed, vector, embed_detail


def _build_check_telemetry(config: MitosConfig) -> Optional[TelemetryStore]:
    """Constructs the sibling telemetry store best-effort (KD2), or ``None``.

    A telemetry-construction failure is the engine's documented ``reuse_read``
    degradation (the run proceeds all-fresh, reports unpartitioned, and the KD5
    seam records no row) — it must never crash a read-only audit. Mirrors
    ``_new_conflict_run``'s best-effort posture (sync.py). A module-level seam so
    tests inject ``None`` or a failing-write wrapper here.

    Args:
        config: The active workspace config.

    Returns:
        The :class:`TelemetryStore`, or ``None`` when it could not be constructed.
    """
    try:
        return TelemetryStore(config.telemetry_path)
    except (sqlite3.Error, DatabaseError, MitosError):
        return None


def _build_check_judge(config: MitosConfig) -> Optional[Callable]:
    """Builds the bound conflict-judgment executor, or ``None`` when keyless (KD6).

    The ``_build_conflict_judge`` shape with the OPPOSITE disposition: it does NOT
    couple to embed/vector presence (check's candidate gather already ran at plan
    time), and ``None`` means "let the engine degrade typed" (``judge=None`` + fresh
    groups → a typed judgment degradation, exit 2, zero spend), not "skip the
    surface". The Anthropic SDK import is lazy so no other verb drags ``anthropic``
    onto its import path (Tier discipline). Built only after the confirm passes and
    only when fresh groups exist, so a reuse-only/clean run never constructs a client.

    ``config`` is REQUIRED rather than defaulted, and this is the one breaking
    signature in 2c: the seam is monkeypatched by name in four test modules, so a
    defaulted parameter would let a stale zero-arg fake keep passing while
    production resolved its key and its model from somewhere else.

    Args:
        config: The target workspace's config — the key and the judgment model id
            both come off its resolved ``env``, never the process environment.

    Returns:
        The bound one-arg ``judge`` callable, or ``None`` when ``ANTHROPIC_API_KEY``
        is absent.
    """
    api_key = config.env.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    from mitos.conflict_judgment import (_JUDGMENT_MODEL_ALIAS,
                                         make_judgment_executor)

    return make_judgment_executor(
        anthropic.Anthropic(api_key=api_key),
        model_id=get_model_id(_JUDGMENT_MODEL_ALIAS, config.env),
    )


def _check_finding_side(node: Dict[str, Any]) -> Dict[str, Any]:
    """Shapes one finding side as ``id`` + the Letter core + non-empty stamps (C4).

    The node is already hydrated + modifier-stamped (2b snapshot / ``Candidate.node``)
    — slugs and stamps ride free, no per-finding store read. Reuses
    :func:`display.letter_payload` for the Letter core (never a raw node), then copies
    only the present modifier stamps (the ``candidate_payload`` conditional-copy
    manner — blind indexing would KeyError on the common unmodified node).

    Args:
        node: A finding's hydrated ``proposal_node`` / ``partner_node`` dict.

    Returns:
        The JSON-native finding-side object (``id``, ``slug``, ``axiom``, ``scope``,
        ``rejected_paths``, plus any non-empty modifier stamps).
    """
    side: Dict[str, Any] = {"id": node["id"]}
    side.update(letter_payload(node, brief=False))
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in node:
            side[key] = node[key]
    return side


def _check_finding_json(finding: "check.CheckFinding") -> Dict[str, Any]:
    """Renders one :class:`~mitos.check.CheckFinding` as its flat JSON object (§8/KD7)."""
    return {
        "novelty": finding.novelty,
        "confidence": finding.confidence,
        "rationale": finding.rationale,
        "score": finding.score,
        "reused": finding.reused,
        "source_batch_id": finding.source_batch_id,
        "source_created_at": finding.source_created_at,
        "proposal": _check_finding_side(finding.proposal_node),
        "partner": _check_finding_side(finding.partner_node),
    }


def _resolve_exclusion_display(
    store: GraphStore, ids: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    """Resolves display slugs for coverage-exclusion node ids, best-effort live (MI-2).

    ``coverage_exclusion_ids`` returns content hashes; ``get_node`` is
    state-agnostic (a since-superseded node still resolves), so the raw-id fallback
    covers only a genuinely absent node (``None`` → ``slug`` ``None``). These reads
    sit in the display-model build, before the write seam.
    """
    out: List[Dict[str, Any]] = []
    for node_id in ids:
        node = store.get_node(node_id)
        out.append({"id": node_id, "slug": node.get("slug") if node else None})
    return out


def _check_transient_count(result: "check.CheckRunResult") -> int:
    """Distinct transient-backlog node count across the readable probes (§8)."""
    ids = {row.node_id for row in result.start_probe.transient}
    if isinstance(result.end_probe, check.StaleProbe):
        ids |= {row.node_id for row in result.end_probe.transient}
    return len(ids)


def _check_json_object(
    result: "check.CheckRunResult",
    row: "check.CheckRunRow",
    *,
    exclusions: List[Dict[str, Any]],
    exit_code: int,
    row_written: bool,
    scope: Optional[str],
    fresh: bool,
    transient_count: int,
) -> Dict[str, Any]:
    """Assembles the single §8 ``--json`` object (a shipped API — additive only).

    ``findings_new`` / ``findings_known`` are read off ``row`` (built via
    :func:`check.check_run_row_from_result`) so the JSON and the ``check_runs`` row
    can never disagree on the NULL-when-unpartitioned rule. ``scope`` is ABSENT when
    unset (MI-9: never ``""``). Emission is deferred to :func:`_emit_json` by the
    caller (the ONLY JSON path).
    """
    obj: Dict[str, Any] = {
        "run_id": result.run_id,
        "mode": "corpus",
        "exit_code": exit_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "fresh": fresh,
    }
    if scope is not None:
        obj["scope"] = scope
    obj.update({
        "nodes_total": result.nodes_total,
        "nodes_swept": result.nodes_swept,
        "pairs_judged_fresh": result.pairs_judged_fresh,
        "pairs_reused": result.pairs_reused,
        "batches_planned": result.batches_planned,
        "batches_executed": result.batches_executed,
        "batches_skipped": result.batches_skipped,
        "findings": [_check_finding_json(f) for f in result.findings],
        "findings_new": row.findings_new,
        "findings_known": row.findings_known,
        "degradations": list(check.run_degradations(result)),
        "coverage_exclusions": exclusions,
        "index_backlog_transient": transient_count,
        "summary_row_written": row_written,
    })
    return obj


def _print_check_finding_side(node: Dict[str, Any]) -> None:
    """Prints one finding side's Letter fields as a calm plain-text block (P9)."""
    print(f"    {node['slug']}")
    print(f"      Axiom:    {node['core_axiom']}")
    scope = node.get("scope") or []
    scope_text = ", ".join(scope) if scope else "(global — no scope declared)"
    print(f"      Scope:    {scope_text}")
    rejected = node.get("rejected_paths")
    if rejected:
        print(f"      Rejected: {rejected}")
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in node:
            print(f"      ({key.replace('_', ' ')}: {', '.join(node[key])})")


def _print_full_finding(finding: "check.CheckFinding") -> None:
    """Prints both sides of a finding plus its rationale (the full new-finding block)."""
    _print_check_finding_side(finding.proposal_node)
    _print_check_finding_side(finding.partner_node)
    print("      Why they may not both stand:")
    print(f"        {finding.rationale}   (confidence {finding.confidence:.2f})")


def _check_degradation_summary(degradations: Tuple[str, ...]) -> str:
    """Renders the degradation tokens as calm human wording (KD4 — from tokens, never
    by re-parsing ``degraded_reason``)."""
    words = {
        "sweep": "the corpus sweep degraded mid-run",
        "judgment": "the judgment stage could not complete",
        "reuse_read": "prior-verdict history was unreadable (findings shown unpartitioned)",
        "telemetry_write": "some per-batch results could not be recorded",
        "stale_index": "the vector index is behind (recall may be thinned)",
        "probe_read": "completeness could not be certified (the index probe was unreadable)",
        "collection_missing": (
            "the vector collection does not exist — run `mitos reconcile` to rebuild it"
        ),
        "judgment_truncated": (
            "the judge's response was truncated (max_tokens exceeded)"
        ),
    }
    return "; ".join(words[token] for token in degradations)


def _print_check_report(
    result: "check.CheckRunResult",
    *,
    exclusions: List[Dict[str, Any]],
    denominator: Optional[int],
    scope: Optional[str],
    row_written: bool,
    transient_count: int,
) -> None:
    """Renders the human report to stdout (findings + disposition), calm ASCII (P9).

    All wording lives HERE (the surface); the engine renders nothing. Findings are
    partitioned by the already-derived ``novelty`` (never re-derived): new findings
    print in full with the resolution pointer, standing (known) findings ride a
    compact section under the index-pinned ``standing (previously reported)`` label,
    and unpartitioned findings (novelty unknown — only under a reuse-read failure)
    get their own labeled section rather than reading as new.
    """
    degradations = check.run_degradations(result)
    new = [f for f in result.findings if f.novelty == "new"]
    known = [f for f in result.findings if f.novelty == "known"]
    unpartitioned = [f for f in result.findings if f.novelty is None]

    if new:
        noun = "contradiction" if len(new) == 1 else "contradictions"
        print(f"\n[Conflict] {len(new)} new {noun} — these decisions may not both stand:")
        for finding in new:
            _print_full_finding(finding)
        print("  Resolve by declaring a relationship in decisions.md "
              "(Supersedes: / Amends: / Narrows: / Contradicts:), then re-sync.")

    if unpartitioned:
        print("\nfindings (history unavailable — unpartitioned):")
        for finding in unpartitioned:
            _print_full_finding(finding)

    if known:
        print("\nstanding (previously reported):")
        for finding in known:
            a, b = finding.proposal_node["slug"], finding.partner_node["slug"]
            print(f"  {a} — {b}   (confidence {finding.confidence:.2f}, "
                  f"first reported {finding.source_created_at})")

    if degradations:
        print(f"\n[partial] This check could not fully run "
              f"({_check_degradation_summary(degradations)}).")
        print(f"  Swept {result.nodes_swept} of {result.nodes_total} decisions; any "
              f"findings above are labeled partial, not certified complete.")

    if not row_written:
        print("  Note: this run was not recorded to check history "
              "(the summary row could not be written).")

    if exclusions:
        print("\nCoverage exclusions (chronically un-embedded — NOT audited):")
        for item in exclusions:
            print(f"  - {item['slug'] or item['id']}")
        print("  These keep failing to embed; the durable fix is outbox quarantine "
              "(substrate-owned). Re-run `mitos sync` to retry.")

    if transient_count:
        print(f"\n{transient_count} decision(s) are behind the vector index — recall "
              f"may be thinned. Run `mitos sync` to catch up.")

    if not result.findings and not degradations:
        if scope is not None and result.nodes_total == 0:
            print(f"0 of {denominator} live decisions match scope '{scope}' — "
                  f"nothing audited.")
        elif result.nodes_total == 0:
            print("No decisions to audit — the corpus is empty.")
        else:
            noun = "decision" if result.nodes_swept == 1 else "decisions"
            print(f"Corpus coherent — {result.nodes_swept} {noun} audited, "
                  f"no contradictions found.")


def cmd_check(
    config: MitosConfig,
    *,
    staged: bool = False,
    scope: Optional[str],
    fresh: bool,
    assume_yes: bool,
    as_json: bool,
) -> int:
    """Audits the live corpus for undeclared contradictions (read-only) → exit 0/1/2.

    The one sequence (each step's contract in the phase plan §4): build substrate →
    provider-absent disposition (KD2) → ``plan_corpus_check`` → CHK-D5 confirm (KD3)
    → build judge iff fresh groups (KD6) → ``execute_corpus_check`` → build the full
    display model → the run-end seam (``exit_code_for`` → row → ``record_check_run``
    LAST, KD5) → emit. ``cmd_check`` owns its error boundary (KD1a): store faults
    around plan/execute/display map to a calm exit-2 vector message, never a traceback
    read by CI as "new findings".

    Args:
        config: The active workspace config.
        staged: Gate the pending buffer instead of sweeping the live corpus (the
            pre-commit / CI gate mode — a self-contained sequence, Phase 3b).
        scope: Optional scope tag filtering the audited (proposal) set (candidate
            recall stays scope-blind, CONF-D2).
        fresh: Re-judge every pair, bypassing verdict reuse (never the novelty read).
        assume_yes: Waive the CHK-D5 spend confirm (the opt-in on every surface).
        as_json: Emit one machine-readable object via :func:`_emit_json` (never prompts).

    Returns:
        ``0`` clean or known-only, ``1`` a NEW contradiction, ``2`` degraded, refused,
        or could-not-run.
    """
    # Flag-combo guard (staged §4 step 1) — pure, pre-store: staged never reuses
    # (nothing to bypass) and always gates the whole pending buffer (no proposal-set
    # filter), so `--scope`/`--fresh` are invocation errors, rejected before any store
    # contact. argparse can't express this (both flags are valid alone).
    if staged and (scope is not None or fresh):
        msg = ("check --staged cannot combine with --scope or --fresh — the gate always "
               "checks the whole pending buffer and never reuses verdicts.")
        if as_json:
            _emit_json({"error": msg, "code": "invalid_flags",
                        **corpus_provenance(config)})
        else:
            # Inside the handler, so it echoes (D5's locus rule). An argument-shape
            # refusal that `main()` renders would not — but this one has a reader
            # and a resolved config, and the two are what the rule turns on.
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 2
    # The gate is a self-contained sequence (its own error boundary); 3a's corpus body
    # below is untouched so its exit contract cannot regress.
    if staged:
        return _run_staged_check(config, assume_yes=assume_yes, as_json=as_json)

    # Lazy at entry (KD6): the alias is needed at PLAN time, but importing
    # `conflict_judgment` module-scope would drag `anthropic` onto every other verb.
    from mitos.conflict_judgment import _JUDGMENT_MODEL_ALIAS

    try:
        store = GraphStore(config.db_path)
        embed, vector, embed_detail = _build_check_substrate(config)
        telemetry = _build_check_telemetry(config)

        # KD2 — provider-absent disposition keys on whether the run has sweep work.
        # Only the embedding provider can be absent at this point: the vector store
        # constructs without touching the network, so a missing collection or an
        # unreachable Qdrant is discovered at the first sweep operation and carried
        # to the same exit 2 as a typed degradation.
        if embed is None:
            active = store.get_active_decisions(scope)
            if active:
                msg = (f"check could not run: cannot audit {len(active)} live "
                       f"decision(s) — embeddings ({embed_detail}) unavailable.")
                if as_json:
                    _emit_json({"error": msg, "code": "substrate_unavailable",
                                **corpus_provenance(config)})
                else:
                    _echo_corpus(config, file=sys.stderr)
                    print(msg, file=sys.stderr)
                return 2
            # Empty snapshot → the providers are never touched (iter_sweep is lazy
            # over zero nodes); fall through to the one healthy-empty engine path.

        plan = check.plan_corpus_check(
            store=store,
            embed_provider=embed,
            vector_store=vector,
            telemetry=telemetry,
            model_alias=_JUDGMENT_MODEL_ALIAS,
            scope=scope,
            fresh=fresh,
        )

        # CHK-D5 confirm (KD3) — strictly above the threshold; all refusals exit 2,
        # zero spend, no row. The shared `_confirm_spend` helper (KD4) is byte-identical
        # for corpus and staged, so the `>` comparison can never fork between the modes.
        refusal = _confirm_spend(
            len(plan.fresh_groups), assume_yes=assume_yes, as_json=as_json
        )
        if refusal is not None:
            return refusal

        # Build the judge only after the confirm passes and only when there is fresh
        # work (KD6): a reuse-only/clean run never constructs a client.
        judge = _build_check_judge(config) if plan.fresh_groups else None
        result = check.execute_corpus_check(
            plan, judge=judge, telemetry=telemetry, store=store, env=config.env
        )

        # The full display model — every remaining store read happens HERE, before
        # the write seam (KD5).
        exclusions = _resolve_exclusion_display(
            store, check.coverage_exclusion_ids(result)
        )
        transient_count = _check_transient_count(result)
        denominator: Optional[int] = None
        if scope is not None and plan.nodes_total == 0:
            # The zero-match denominator needs the unscoped live count the plan does
            # not carry (plan.nodes_total is already scope-filtered).
            denominator = len(store.get_active_decisions())

        # Run-end seam (KD5): exit → row → write LAST. Build the row unconditionally
        # (pure) so the JSON scalars derive from the one source; write it only when
        # telemetry exists.
        exit_code = check.exit_code_for(result)
        row = check.check_run_row_from_result(result, mode="corpus", exit_code=exit_code)
        row_written = False
        if telemetry is not None:
            try:
                telemetry.record_check_run(row)
                row_written = True
            except DatabaseError:
                # The write is the last fallible act: a failure only moves toward 2.
                exit_code = 2
        else:
            # telemetry None is already exit 2 via reuse_read; the no-row disclosure
            # is additive, not a second exit driver.
            exit_code = 2
    except (sqlite3.Error, DatabaseError, MitosError) as exc:
        # KD1a — the verb owns its boundary: a store fault is exit 2 with a calm
        # vector message, never a traceback CI would read as "new findings".
        msg = f"check could not run: {exc}"
        if as_json:
            _emit_json({"error": msg, "code": "check_faulted",
                        **corpus_provenance(config)})
        else:
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 2

    # Emission is pure (out of the write contract): one JSON object or the report.
    if as_json:
        # Stamped at the CALL SITE, never by threading `config` into
        # `_check_json_object`: that assembler is a shipped API whose signature has
        # no business gaining a routing parameter (the same argument D1 makes
        # against stamping inside `_emit_json`).
        _emit_json({**_check_json_object(
            result, row, exclusions=exclusions, exit_code=exit_code,
            row_written=row_written, scope=scope, fresh=fresh,
            transient_count=transient_count,
        ), **corpus_provenance(config)})
    else:
        _echo_corpus(config)
        _print_check_report(
            result, exclusions=exclusions, denominator=denominator, scope=scope,
            row_written=row_written, transient_count=transient_count,
        )
    return exit_code


# =========================================================================== #
# Phase 3b — `mitos check --staged`: the pre-commit / CI gate mode.
#
# The proactive half of the `check` verb: it gates the PENDING (not-yet-committed)
# decision entries of the working-tree `decisions.md` and fails CLOSED — a pending
# undeclared contradiction blocks the commit (exit 1), a clean buffer passes (0),
# a gate that cannot run says so (exit 2) rather than a silent pass. Self-contained
# (its own sequence + error boundary) so 3a's corpus contract cannot regress. The
# load-bearing rules — the exit table (§3), the pure-read predicate (KD1: no graph
# write), the no-row-unless-judged rule (KD2/KD8), the `surface='check'` attribution
# (KD7), `_emit_json`-only — live in the phase plan.
# =========================================================================== #


def _pending_decision_entries(
    store: GraphStore, entries: List[ParsedEntry]
) -> List[ParsedEntry]:
    """Selects the pending decision entries via the sync idempotency predicate (KD1).

    A pure READ: an entry is pending iff its slug-free canonical-core content hash is
    not yet a committed node. Replicates sync.py:650-656's ``compute_node_id`` call and
    the ``get_node`` test — but deliberately NOT the ``note_source_reencounter`` write
    two lines below it (sync.py:668): a gate that mutates the graph while gating it is a
    contradiction of its own. OQ entries are skipped (the facade is decision-only).

    Args:
        store: The graph store — touched only through ``get_node`` (read).
        entries: The parsed working-tree entries (already all ``decision`` kind when
            they come from ``parse_entry_stream(text, "decision")``; the filter is a
            harmless safety belt).

    Returns:
        The pending decision entries, in parse order.
    """
    pending: List[ParsedEntry] = []
    for entry in entries:
        if entry.kind != "decision":
            continue
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )
        if store.get_node(node_id) is None:
            pending.append(entry)
    return pending


def _persist_staged_batch(
    telemetry: Optional[TelemetryStore],
    result: "Any",
    *,
    run_id: str,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Persists one judged staged batch with ``surface='check'`` (KD7), best-effort.

    Mirrors ``sync._persist_conflict_batch``'s ``ConflictCheckResult`` → ``(JudgmentBatch,
    [ConflictCheckRow])`` mapping (sync.py:1196-1239) verbatim EXCEPT for two data-level
    values: ``surface='check'`` (this is the check surface, not sync) and ``sync_run_id``
    = this run's id. Every fed-context field is read off the result's ``JudgeInput``\\ s
    (what the judge saw), never a node re-read. The MI-9 ``""→None`` proposal/candidate
    scope + rejected coercions are load-bearing.

    Args:
        telemetry: The run's telemetry store, or ``None`` (a judged run needs it — a
            ``None`` store is a write failure the caller degrades on, KD7).
        result: A judged :class:`~mitos.conflict.ConflictCheckResult` (``execution`` set;
            the caller guards ``execution is not None``).
        run_id: This run's id, stamped as ``sync_run_id`` on every row (the one-thread-of-
            truth join to the ``check_runs`` PK).
        env: The target workspace's resolved environment (``config.env``), against
            which ``execution.model_alias`` resolves — the same map the judge's own
            model id came from, so the provenance column records the model the run
            actually used.

    Returns:
        ``None`` on a clean write, or a write-failure detail string (the caller marks the
        run degraded and reports it) — never raising, so one bad batch never crashes the gate.
    """
    if telemetry is None:
        return "telemetry store unavailable"
    try:
        execution = result.execution
        # CHK-D3: resolve the versioned model id here — moments after the call, against
        # the same resolved environment the call used (2c); an unknown alias degrades to
        # NULL (provenance-only), never a lost row.
        try:
            model_id: Optional[str] = get_model_id(execution.model_alias, env)
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
                    sync_run_id=run_id,
                    # The staged difference from the sync mapper: this IS the check
                    # surface, stamped explicitly (CHK-D7), never the schema DEFAULT.
                    surface="check",
                    judged_axiom=proposal.axiom,
                    proposal_rejected_paths=proposal.rejected_paths or None,
                    proposal_scope=", ".join(proposal.scope) or None,
                    proposed_hash_if_any=result.proposed_hash_if_any,
                    candidate_slug=pair.candidate.slug,
                    candidate_hash=pair.candidate.node["id"],
                    candidate_rejected_paths=candidate_input.rejected_paths,
                    candidate_scope=", ".join(candidate_input.scope) or None,
                    tenable=pair.judgment.tenable_together,
                    confidence=pair.judgment.confidence,
                    surfaced=pair.surfaced,
                    candidate_source=check.CONFLICT_CANDIDATE_SOURCE,
                    model_alias=execution.model_alias,
                    prompt_version=check.CONFLICT_PROMPT_VERSION,
                    mitos_version=__version__,
                    rationale=pair.judgment.rationale,
                )
            )
        telemetry.record_judged_batch(
            batch, rows, datetime.now(timezone.utc).isoformat()
        )
        return None
    except (sqlite3.Error, DatabaseError, TypeError, ValueError) as exc:
        # Best-effort (KD7): a mapping/write failure degrades the run, never crashes it.
        return str(exc)


# The staged degradation vocabulary — a subset of the corpus tokens that a gate can
# reach (KD8). Rendered to calm human wording; the raw tokens ride the `--json`.
_STAGED_DEGRADATION_WORDS = {
    "stale_index": "the vector index is behind (recall may be thinned)",
    "sweep": "the semantic substrate went dark mid-run (findings shown are partial)",
    "judgment": "the judge became unavailable mid-run (findings shown are partial)",
    "telemetry_write": "some results could not be recorded",
    "collection_missing": (
        "the vector collection does not exist — run `mitos reconcile` to rebuild it"
    ),
    "judgment_truncated": (
        "the judge's response was truncated (max_tokens exceeded)"
    ),
}


def _staged_finding_json(
    entry: ParsedEntry, proposed_hash: str, finding: "Any", partner_hash: Optional[str]
) -> Dict[str, Any]:
    """Renders one staged finding as its §8 JSON object — both sides named (KD9).

    Every staged finding is ``novelty:"new"`` (the gate never partitions, CHK-D10). The
    proposal side is the pending entry (not yet a node — no id-in-graph, no modifier
    stamps); the partner side is the facade's candidate ``payload`` (a Letter render) plus
    the candidate content hash resolved from ``judged_pairs``.
    """
    payload = finding.payload
    partner: Dict[str, Any] = {"id": partner_hash}
    for key in ("slug", "axiom", "scope", "rejected_paths"):
        if key in payload:
            partner[key] = payload[key]
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in payload:
            partner[key] = payload[key]
    return {
        "novelty": "new",
        "confidence": finding.confidence,
        "rationale": finding.rationale,
        "score": payload.get("score"),
        "proposal": {
            "id": proposed_hash,
            "slug": entry.slug,
            "axiom": entry.axiom,
            "scope": list(entry.scope),
            "rejected_paths": entry.rejected_paths,
        },
        "partner": partner,
    }


def _print_staged_finding(entry: ParsedEntry, finding: "Any") -> None:
    """Prints both sides of one staged finding plus its rationale (calm ASCII, P9)."""
    print(f"  Pending entry '{entry.slug}':")
    print(f"      Axiom:    {entry.axiom}")
    scope_text = ", ".join(entry.scope) if entry.scope else "(global — no scope declared)"
    print(f"      Scope:    {scope_text}")
    if entry.rejected_paths:
        print(f"      Rejected: {entry.rejected_paths}")
    payload = finding.payload
    print(f"  conflicts with active decision '{payload['slug']}'   "
          f"(similarity {payload['score']:.2f}):")
    print(f"      Axiom:    {payload['axiom']}")
    p_scope = payload.get("scope") or []
    p_scope_text = ", ".join(p_scope) if p_scope else "(global — no scope declared)"
    print(f"      Scope:    {p_scope_text}")
    if "rejected_paths" in payload:
        print(f"      Rejected: {payload['rejected_paths']}")
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in payload:
            print(f"      ({key.replace('_', ' ')}: {', '.join(payload[key])})")
    print("      Why they may not both stand:")
    print(f"        {finding.rationale}   (confidence {finding.confidence:.2f})")


def _print_staged_report(
    findings: List[Tuple[ParsedEntry, str, "Any", Optional[str]]],
    *,
    nodes_swept: int,
    nodes_total: int,
    degraded: "Set[str]",
    exclusions: List[Dict[str, Any]],
    transient_count: int,
) -> None:
    """Renders the human gate report to stdout — calm, both sides named (P9)."""
    if findings:
        noun = "contradiction" if len(findings) == 1 else "contradictions"
        print(f"\n[Conflict] {len(findings)} pending {noun} — these decisions may "
              f"not both stand:")
        for entry, _hash, finding, _pid in findings:
            _print_staged_finding(entry, finding)
        print("  Resolve by declaring a relationship in decisions.md "
              "(Supersedes: / Amends: / Narrows: / Contradicts:) before committing, "
              "or `git commit --no-verify` to bypass the gate deliberately.")

    if degraded:
        summary = "; ".join(
            _STAGED_DEGRADATION_WORDS[t] for t in sorted(degraded)
            if t in _STAGED_DEGRADATION_WORDS
        )
        print(f"\n[partial] This gate could not fully run ({summary}).")
        print(f"  Checked {nodes_swept} of {nodes_total} pending decision(s); any "
              f"findings above are partial, not certified complete.")

    if exclusions:
        print("\nCoverage exclusions (chronically un-embedded — NOT audited):")
        for item in exclusions:
            print(f"  - {item['slug'] or item['id']}")
        print("  These keep failing to embed; run `mitos sync` to retry (the durable "
              "fix is outbox quarantine, substrate-owned).")

    if transient_count:
        print(f"\n{transient_count} decision(s) are behind the vector index — recall "
              f"may be thinned. Run `mitos sync` to catch up.")

    if not findings and not degraded:
        noun = "decision" if nodes_swept == 1 else "decisions"
        print(f"Gate clear — {nodes_swept} pending {noun} checked, "
              f"no contradictions found.")


def _run_staged_check(
    config: MitosConfig, *, assume_yes: bool, as_json: bool
) -> int:
    """Gates the pending decision buffer, fail-closed → exit 0/1/2 (Phase 3b).

    The one sequence (§4): parse the working-tree ``decisions.md`` → select pending via
    the pure-read predicate (KD1) → no-pending short-circuit (exit 0, no probe/substrate/
    row, KD2) → build substrate (absent + pending ⇒ fail-closed exit 2) → start probe
    (KD3) → CHK-D5 confirm (KD4) → build judge (absent + pending ⇒ fail-closed exit 2,
    KD5) → per-entry facade loop with the aggregate breaker (KD6) → exit derivation +
    the hand-built ``mode='staged'`` row written LAST, only when a judgment fired (KD8).
    Owns its own error boundary: a store/parse fault is exit 2 with a calm vector message,
    never a traceback CI reads as "new findings".

    Args:
        config: The active workspace config (paths).
        assume_yes: Waive the CHK-D5 spend confirm.
        as_json: Emit one machine-readable object via :func:`_emit_json` (never prompts).

    Returns:
        ``0`` clean / no pending, ``1`` a pending contradiction, ``2`` degraded, refused,
        or could-not-gate.
    """
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        store = GraphStore(config.db_path)
        # Parse the WORKING-TREE decisions.md (git-agnostic; absent file → no pending).
        # `parse_entry_stream` STRICT mode raises ParseError(MitosError) on a malformed
        # buffer → the boundary below maps it to exit 2 (fail-closed). Never pass a
        # `failures=` collector (that would silently isolate a bad entry).
        text = read_text_or_none(config.decisions_file)
        entries = parse_entry_stream(text, "decision") if text else []
        pending = _pending_decision_entries(store, entries)

        # KD2 — no-pending short-circuit: exit 0 with zero LLM contact, no probe, no
        # substrate build, no row. The overwhelmingly common commit is free.
        if not pending:
            ended_at = datetime.now(timezone.utc).isoformat()
            if as_json:
                # The `--json` branch of the free path is NOT carved out: it already
                # emits a full object, so a key costs no shipped silence. Only the
                # one-line text branch below is the carve-out, and the two must not
                # be "harmonized" into one rule later.
                _emit_json({**_staged_json_object(
                    run_id=run_id, started_at=started_at, ended_at=ended_at,
                    exit_code=0, nodes_total=0, nodes_swept=0, batches_executed=0,
                    pairs_judged_fresh=0, finding_objs=[], degradations=[],
                    exclusions=[], transient_count=0, row_written=False,
                ), **corpus_provenance(config)})
            else:
                # THE OBLIGATION CARVE-OUT (§4.7): the echo rides output the surface
                # already emits and never converts a near-silent success into
                # output. This is a pre-commit hook's dominant path — SETUP.md sells
                # it as effectively free, in noise as much as in spend — and its
                # target is a literal in a committed hook (`-p .`), so it cannot
                # drift. Measured at 8523259: this prints exactly ONE line (the
                # vision's "printed nothing at all" is false; the conclusion is
                # not). Tripling it to name a target that cannot drift is the trade
                # this branch declines. Distinct from `agent-block`'s carve-out,
                # which is about the CHANNEL, not the obligation — collapsing the
                # two into "check and agent-block are special" loses both arguments.
                print("Gate clear — no pending decisions to check.")
            return 0

        embed, vector, embed_detail = _build_check_substrate(config)
        # The embedding provider absent WITH pending work → fail-closed (the hook
        # precondition), no row. A missing collection or an unreachable Qdrant is no
        # longer visible here — it trips the breaker in the entry loop below and
        # lands on the same exit 2, named rather than blamed on construction.
        if embed is None:
            msg = (f"check --staged could not gate {len(pending)} pending decision(s) — "
                   f"embeddings ({embed_detail}) unavailable.")
            if as_json:
                _emit_json({"error": msg, "code": "substrate_unavailable",
                            **corpus_provenance(config)})
            else:
                # The fail-closed refusal DOES echo — the carve-out below is the
                # free short-circuit alone, and this branch is already speaking.
                _echo_corpus(config, file=sys.stderr)
                print(msg, file=sys.stderr)
            return 2

        telemetry = _build_check_telemetry(config)

        # KD3 — start probe only (no end probe: staged judges a fixed buffer, not a
        # sweep). A probe fault propagates to the boundary. A transient backlog gates
        # partial (exit 2) but does NOT skip the judgment; over-tolerance rows are
        # disclosed coverage exclusions that never gate (the poison-row escape).
        start_probe = check.probe_stale_index(store)
        transient_count = len({row.node_id for row in start_probe.transient})
        degraded: Set[str] = set()
        if transient_count:
            degraded.add("stale_index")

        # KD4 — the shared confirm on the pending count. All refusals exit 2, no row.
        refusal = _confirm_spend(len(pending), assume_yes=assume_yes, as_json=as_json)
        if refusal is not None:
            return refusal

        # KD5 — the judge is required to gate real work: `run_conflict_check` CALLS it
        # (unlike corpus, which absorbs `judge=None` as a typed degradation), so a
        # missing key with pending entries is fail-closed exit 2, no row. `--no-verify`
        # is the deliberate human bypass (documented in 4b).
        judge = _build_check_judge(config)
        if judge is None:
            msg = (f"check --staged could not gate {len(pending)} pending decision(s) — "
                   f"ANTHROPIC_API_KEY is not set (the judge is required to gate).")
            if as_json:
                _emit_json({"error": msg, "code": "judge_unavailable",
                            **corpus_provenance(config)})
            else:
                _echo_corpus(config, file=sys.stderr)
                print(msg, file=sys.stderr)
            return 2

        # KD6 — per-entry facade loop, verbatim + sequential, aggregate breaker on the
        # first typed `Unavailable` (one penalty, not N). A genuine local store fault
        # RAISES past the facade → the boundary below (prior findings lost, the rare case).
        findings: List[Tuple[ParsedEntry, str, "Any", Optional[str]]] = []
        nodes_swept = 0
        pairs_judged_fresh = 0
        batches_executed = 0
        for entry in pending:
            result = run_conflict_check(
                entry, embed_provider=embed, vector_store=vector, store=store, judge=judge
            )
            if isinstance(result, check.Unavailable):
                # Trip the breaker: stop calling the facade for the remaining entries.
                # The token is faithful to WHICH downstream went dark (aligning with the
                # corpus P18 vocabulary): the semantic substrate reads as `sweep`, the
                # judge as `judgment`. Bound to the shared membership constant, because
                # an `else` here means "the judge" and would mis-file a new semantic
                # member silently and confidently.
                if result.reason in SEMANTIC_SUBSTRATE_REASONS:
                    degraded.add("sweep")
                    # Additive, never substitutive — `sweep` is the shipped effect
                    # token, this names the cause (and the heal, via the word map).
                    if result.reason is ConflictUnavailableReason.COLLECTION_MISSING:
                        degraded.add("collection_missing")
                else:
                    degraded.add("judgment")
                    if result.reason is ConflictUnavailableReason.JUDGMENT_TRUNCATED:
                        degraded.add("judgment_truncated")
                break
            nodes_swept += 1
            pairs_judged_fresh += len(result.judged_pairs)
            if result.execution is not None:
                batches_executed += 1
                detail = _persist_staged_batch(
                    telemetry, result, run_id=run_id, env=config.env
                )
                if detail is not None:
                    degraded.add("telemetry_write")
            # Map each surfaced finding to its candidate content hash (from judged_pairs).
            hash_by_slug = {
                pair.candidate.slug: pair.candidate.node["id"]
                for pair in result.judged_pairs
            }
            for finding in result.findings:
                findings.append((
                    entry, result.proposed_hash_if_any, finding,
                    hash_by_slug.get(finding.slug),
                ))

        exclusions = _resolve_exclusion_display(
            store, tuple(row.node_id for row in start_probe.excluded)
        )

        # KD8 — exit derivation (degraded 2 dominates finding 1 dominates clean 0), then
        # the hand-built row, written LAST and only when a judgment actually fired.
        exit_code = 2 if degraded else (1 if findings else 0)
        ended_at = datetime.now(timezone.utc).isoformat()
        row_written = False
        if batches_executed > 0:
            row = check.CheckRunRow(
                run_id=run_id,
                mode="staged",
                started_at=started_at,
                ended_at=ended_at,
                exit_code=exit_code,
                nodes_swept=nodes_swept,
                pairs_judged_fresh=pairs_judged_fresh,
                pairs_reused=0,
                findings_new=len(findings),
                findings_known=0,
                coverage_exclusions=len(start_probe.excluded),
                degraded_reason=",".join(sorted(degraded)) or None,
                mitos_version=__version__,
            )
            if telemetry is not None:
                try:
                    telemetry.record_check_run(row)
                    row_written = True
                except DatabaseError:
                    # The write is the last fallible act: a failure only moves toward 2.
                    degraded.add("telemetry_write")
                    exit_code = 2
            else:
                # A judged run whose telemetry could not be built cannot record — degrade.
                degraded.add("telemetry_write")
                exit_code = 2
    except (sqlite3.Error, DatabaseError, MitosError) as exc:
        # The gate's error boundary (mirrors 3a's): a store/parse/probe/facade fault is
        # exit 2 with a calm vector message, never a traceback CI reads as "new findings".
        msg = f"check --staged could not run: {exc}"
        if as_json:
            _emit_json({"error": msg, "code": "check_faulted",
                        **corpus_provenance(config)})
        else:
            _echo_corpus(config, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 2

    # Emission is pure (out of the write contract): one JSON object or the report.
    if as_json:
        # Stamped at the call site, like the corpus twin — `_staged_json_object` is
        # a shipped assembler and stays signature-stable.
        _emit_json({**_staged_json_object(
            run_id=run_id, started_at=started_at, ended_at=ended_at,
            exit_code=exit_code, nodes_total=len(pending), nodes_swept=nodes_swept,
            batches_executed=batches_executed, pairs_judged_fresh=pairs_judged_fresh,
            finding_objs=[_staged_finding_json(e, h, f, p) for e, h, f, p in findings],
            degradations=sorted(degraded), exclusions=exclusions,
            transient_count=transient_count, row_written=row_written,
        ), **corpus_provenance(config)})
    else:
        _echo_corpus(config)
        _print_staged_report(
            findings, nodes_swept=nodes_swept, nodes_total=len(pending),
            degraded=degraded, exclusions=exclusions, transient_count=transient_count,
        )
    return exit_code


def _staged_json_object(
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    exit_code: int,
    nodes_total: int,
    nodes_swept: int,
    batches_executed: int,
    pairs_judged_fresh: int,
    finding_objs: List[Dict[str, Any]],
    degradations: List[str],
    exclusions: List[Dict[str, Any]],
    transient_count: int,
    row_written: bool,
) -> Dict[str, Any]:
    """Assembles the single §8 staged ``--json`` object (a shipped API — additive only).

    The same key set as 3a §8 with staged values (KD9): ``mode:"staged"``, ``scope`` key
    ABSENT (staged never scopes), ``fresh:false``, ``pairs_reused:0``, every finding
    ``novelty:"new"``, ``findings_known:0`` — so a CI consumer's cross-surface invariant
    ``exit_code == 1 ⟺ findings_new > 0`` holds on staged exactly as on corpus. The
    ``batches_*`` accounting reads: planned = one potential batch per pending entry;
    executed = entries that fired the judge; skipped = clean-empty or breaker-skipped
    (so ``planned == executed + skipped`` holds).
    """
    return {
        "run_id": run_id,
        "mode": "staged",
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "fresh": False,
        "nodes_total": nodes_total,
        "nodes_swept": nodes_swept,
        "pairs_judged_fresh": pairs_judged_fresh,
        "pairs_reused": 0,
        "batches_planned": nodes_total,
        "batches_executed": batches_executed,
        "batches_skipped": nodes_total - batches_executed,
        "findings": finding_objs,
        "findings_new": len(finding_objs),
        "findings_known": 0,
        "degradations": degradations,
        "coverage_exclusions": exclusions,
        "index_backlog_transient": transient_count,
        "summary_row_written": row_written,
    }


def _enter_target_directory(directory: Optional[str]) -> None:
    """chdir into a -C/--directory target, or no-op when none was given.

    Git's ``-C`` semantics at the CLI boundary: the chdir runs once at process
    entry, before any env load, config construction, or arg-driven file open, so
    the whole workspace (graph, collection, ``.env``/keys, relative path args)
    retargets at once — each downstream site derives from the process CWD.

    Args:
        directory: The ``-C``/``--directory`` value, or None when the flag was
            absent (then this is a no-op and the launch CWD is unchanged).

    Raises:
        MitosError: When ``directory`` is given but is not an existing directory
            (a clean P3 error — never a raw OSError traceback). Existence is
            checked against the launch CWD, so a relative ``-C ./sub`` resolves
            where mitos was started.
    """
    if directory is None:
        return
    if not os.path.isdir(directory):
        raise MitosError(f"directory not found: {directory}")
    os.chdir(directory)


def _warn_deprecated_rotation_mode(config: MitosConfig) -> None:
    """Prints one calm stderr line when the workspace configures a deprecated mode.

    Epoch 1 of narrowing ``rotation_mode`` to ``archive``: the value is accepted and
    pinned by the loader, and this is what keeps that coercion from being *silent* —
    which is the thing ``config-loader-rotation-mode-enum-hard-fail`` forbids.

    Called at verb dispatch and **only** there. A once-per-invocation guard inside the
    loader would need module-level state, which makes "warns once" tests
    order-dependent inside one pytest process; dispatch needs no flag, because the CLI
    is a fresh process per invocation. It also keeps MCP TOOL CALLS silent:
    ``mcp_server`` builds a ``MitosConfig`` per call over a stdio JSON-RPC channel, so
    a warning there would be per-call spam — and on stdout, protocol corruption. No
    config author is present on that surface anyway. (``mitos serve`` itself routes
    through ``main`` and so warns once at startup, on stderr, before the protocol
    opens — harmless, and the operator launching the server is a fair audience.)

    Args:
        config: The dispatch-time configuration.
    """
    mode = getattr(config, "deprecated_rotation_mode", None)
    if not mode:
        return
    print(
        f"Warning: rotation_mode = '{mode}' is deprecated and now behaves as "
        "'archive'. Rotated entries are written to decisions/archive/ instead of "
        f"{'being wrapped in place' if mode == 'mark' else 'being discarded'}; "
        "update .mitos/config.toml to silence this.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# The project selector at the CLI boundary.
#
# One resolution site, in `main()`, feeding the single `MitosConfig` every verb
# already receives — so ~20 verbs retarget with no per-verb edit. A per-verb
# thread would be twenty chances to miss one, and the one missed would fail
# *silently*: a call that names a project and writes to the working directory.
# ---------------------------------------------------------------------------

# Verbs that target no single workspace, and why. The reason keys are the shared
# vocabulary (`errors.EXEMPT_*`); the wording below is this surface's alone.
# `status` is deliberately ABSENT: zero-arg `status` is a global overview, but
# `status <project>` is a supported targeting form, so `-p X status` is the flag
# spelling of something the tool already does.
_SELECTOR_EXEMPT_VERBS: Dict[str, str] = {
    "init": EXEMPT_CREATES_REGISTRATION,
    "serve": EXEMPT_NO_WORKSPACE,
    "projects": EXEMPT_EXPLICITLY_GLOBAL,
}

# Why each of them is global, in this surface's words. Keyed by VERB rather than
# by reason because two verbs share `explicitly_global` and owe different
# recoveries — `set-key` only ever lands here through `--global`, which is the
# form its note names.
_EXEMPT_VERB_NOTES: Dict[str, str] = {
    "init": ("`init` creates a registration rather than reaching one — run it in "
             "the workspace you mean (`mitos -C <dir> init`, or cd there first)."),
    "serve": ("`serve` starts the MCP server and binds no workspace of its own — "
              "launch it plainly: `mitos serve`."),
    "projects": ("`projects` reads the machine-wide registry and targets no single "
                 "workspace — it works from anywhere."),
    "set-key": ("`set-key --global` writes the machine-wide .env shared by every "
                "project — drop `--global` to write one project's own .env."),
}

# Verbs whose positional argument DENOTES a workspace, so it is a selector source.
# `import`'s positional is a source markdown *file* and stays cwd-rooted — the
# discriminator is what the positional denotes, never that it is spelled `path`.
_POSITIONAL_SELECTOR_VERBS: frozenset = frozenset({"status", "agent-block"})

# The two verbs that answer questions *about* a directory rather than acting on
# its corpus, so "there is no workspace here" is their ANSWER, not their error.
# `mitos status /some/dir` must keep printing the NOT SET UP report with its next
# steps — SETUP.md's agent loop is built on that report — and `agent-block`'s
# plain form prints a workspace-independent block, so refusing a pre-`init` repo
# would block a legitimate use.
_WORKSPACE_OPTIONAL_VERBS: frozenset = frozenset({"status", "agent-block"})


def _answer_workspace_optional_verb(args: argparse.Namespace,
                                    target: Optional[routing.ResolvedProject]) -> int:
    """Runs a ``_WORKSPACE_OPTIONAL_VERBS`` verb when its config could not be built.

    These two verbs answer questions *about* a directory rather than acting on its
    corpus, so a malformed ``.mitos/config.toml`` is their **answer**, not their
    error — the same argument that put them in ``_WORKSPACE_OPTIONAL_VERBS`` in the
    first place (3b's D8), extended from "there is no workspace here" to "the
    workspace here is broken". Measured before this was written: ``mitos status
    --json <malformed-workspace>`` printed a one-line ``Error:`` and **not one byte
    of JSON**, because the boundary's ``MitosConfig`` construction raised before
    dispatch and ``cmd_status``'s own contextual report never ran. On ``main`` the
    path form still worked — ``main()`` built its config from cwd while
    ``cmd_status`` got ``args.path`` — and 3b collapsed the two onto one
    boundary-resolved config with the exit code unchanged either way, so nothing
    went red.

    Both members are covered, not just ``status``: ``cmd_agent_block`` reads no
    config at all (verified against its body), so refusing it over a config it
    never opens is the same defect. The cost is stated rather than hidden — the
    ``agent-block`` dispatch's stderr corpus echo is **absent** on this path, since
    there is no config to name a corpus from, and inventing a fallback would make
    the echo claim a collection nobody resolved.

    Since 5a ``target`` is always a real resolution on this path: neither verb is
    exempt, a selectorless ``status`` routed to the global overview one statement
    earlier, and a selectorless ``agent-block`` raised. The parameter keeps its
    ``Optional`` type because that is ``_resolve_selector``'s declared return on the
    exempt path, not because ``None`` is reachable here.

    Args:
        args: The parsed namespace.
        target: The resolved project.

    Returns:
        The verb's exit code.
    """
    # `abspath` rather than a working-directory read — the canonical root is already
    # in hand, and the tree keeps exactly the two display-only reads entry-005
    # enumerates (this comment is worded to keep that standing grep honest, 2a's
    # precedent).
    root = os.path.abspath(target.root)
    if args.command == "status":
        # `getattr`, unlike main()'s status-only branch: this helper also answers
        # `agent-block`, whose namespace carries no `verbose`.
        return cmd_status(root, as_json=args.as_json, project=target.name,
                          verbose=getattr(args, "verbose", False))
    return cmd_agent_block(root, check=args.check)


def _selector_from_args(args: argparse.Namespace) -> Optional[str]:
    """Coalesces the three spellings of the project selector into one value.

    ``--project`` is accepted on both sides of the verb, into **two** destinations
    (``project_pre`` / ``project_post``). That is not redundancy: one shared
    ``dest`` lets the subparser's ``None`` default overwrite what the top-level
    parser stored, silently discarding a selector the caller did supply (measured
    on this parser), and the ``argparse.SUPPRESS`` repair fixes the discard while
    making the both-positions case indistinguishable from the post-verb one. Two
    destinations make the trap unconstructible — there is no shared slot to clobber
    — and the ambiguity stays visible enough to refuse.

    Naming the target twice is refused **even when the two values are identical**:
    a rule with an equality exception is one nobody can predict at the call site.

    Args:
        args: The parsed namespace.

    Returns:
        The selector as the caller typed it, or ``None`` when none was supplied.
        An empty string is a *supplied* selector carrying no target, and is
        returned as such — the gate everywhere below is ``is not None``, never
        truthiness, or `-p ""` would silently fall back to the working directory.

    Raises:
        MitosError: If the target was named twice. A plain boundary error, not a
            seventh targeting discriminator: nothing was resolved and nothing is
            unknown, so it is a CLI-local usage fault.
    """
    pre = getattr(args, "project_pre", None)
    post = getattr(args, "project_post", None)
    if pre is not None and post is not None:
        raise MitosError(
            f"the project was named twice: `--project {pre}` before the verb and "
            f"`--project {post}` after it. Pass one, on either side."
        )
    flag = pre if pre is not None else post

    positional = (getattr(args, "path", None)
                  if args.command in _POSITIONAL_SELECTOR_VERBS else None)
    if flag is not None and positional is not None:
        raise MitosError(
            f"the project was named twice: `--project {flag}` and the positional "
            f"{positional!r}. Pass one — `mitos {args.command} {positional}` or "
            f"`mitos {args.command} --project {flag}`."
        )
    return flag if flag is not None else positional


def _exempt_reason(args: argparse.Namespace) -> Optional[str]:
    """Answers "does this call target no single workspace?", once, for both readers.

    Two sides of the boundary need the same rule and would otherwise spell it
    twice: the refusal below (*a selector on this verb is a fault*) and the flip in
    ``main()`` (*the absence of a selector on this verb is fine*). A second
    hand-written ``args.command in _SELECTOR_EXEMPT_VERBS or (args.command ==
    "set-key" and args.is_global)`` is the drift seam — ``set-key --global`` would
    keep working while ``mitos -p x set-key --global`` answered a different
    question, or vice versa, and no row would see it.

    ``status`` is deliberately **not** here. Its optionality is a different rule — a
    selector is *legal* on ``status`` and merely routes elsewhere when absent — and
    folding the two together would make ``mitos -p mitos status`` refusable by
    accident.

    Args:
        args: The parsed namespace.

    Returns:
        One of ``errors.EXEMPT_*`` — the shared vocabulary that also keys
        ``routing.exempt_verb_error`` and ``_EXEMPT_VERB_NOTES`` — or ``None`` when
        the call targets a workspace. Never a boolean: a fourth spelling of the
        reason is what forks the renderer's vocabulary.
    """
    reason = _SELECTOR_EXEMPT_VERBS.get(args.command)
    if reason is None and args.command == "set-key" and getattr(args, "is_global", False):
        # Conditional membership: `set-key`'s *project* form is the bare
        # invocation, so only `--global` makes the verb global. `getattr` with a
        # default because `is_global` exists only on the `set-key` namespace.
        reason = EXEMPT_EXPLICITLY_GLOBAL
    return reason


def _refuse_selector_on_exempt_verb(args: argparse.Namespace,
                                    selector: Optional[str]) -> None:
    """Refuses a selector handed to a verb that targets no project.

    Runs **before** resolution, and the ordering is the point: ``mitos -p nosuch
    init`` must answer *"`init` takes no project selector"*, not *"unknown
    project"*. The fault is the verb, and resolving first answers the wrong
    question — then teaches a recovery (register the name) that would still leave
    the call malformed.

    Args:
        args: The parsed namespace.
        selector: The coalesced selector, or None.

    Raises:
        ProjectTargetingError: With the ``exempt_verb`` discriminator.
    """
    if selector is None:
        return
    reason = _exempt_reason(args)
    if reason is not None:
        raise routing.exempt_verb_error(args.command, reason)


def _resolve_selector(selector: Optional[str],
                      command: str) -> routing.ResolvedProject:
    """Turns a selector into a validated workspace — or raises. There is no absence.

    Absolutizes an explicitly-typed relative path (``.``, ``./x``, ``../x``,
    ``x/y``) so the resolver only ever receives a name or an absolute path — and
    runs **after** ``-C``'s process-entry chdir, so ``mitos -C /a -p ./b`` means
    ``/a/b``. Canonicalizing before the chdir would change what a relative selector
    means, with every absolute-path test still green.

    ``expanduser`` is deliberately not used: ``os.path.abspath("~/x")`` returns
    ``<cwd>/~/x``, which **is** absolute, so an unguarded absolutize would send a
    nonsense path to the resolver and answer with "no workspace at …" naming a
    directory nobody meant. With the guard, a ``~``-leading selector reaches the
    resolver path-shaped and non-absolute and lands on the relative-path class,
    whose message names the actual rule and both valid forms.

    **An absent selector is carried straight through to
    ``routing.resolve_project``, which raises the missing-class error.** That is
    5a's flip: this function used to return ``None`` for it and the caller used to
    read that as *"resolve the working directory"*. A boundary-local ``if selector
    is None: raise ProjectTargetingError(missing, …)`` would have worked and would
    have duplicated the constructor's required-field bookkeeping at the one place
    2a deliberately kept free of it — one raise site, one wording, no fifth
    spelling invented here. ``resolve_project``'s own ``not selector`` guard runs
    before any shape test, so ``None`` never reaches ``is_path_shaped``.

    The caller reaches this function **only** for a verb that targets a workspace:
    ``_exempt_reason`` short-circuits ``init``/``serve``/``projects``/``set-key
    --global`` (whose ``target`` is ``None``, and it is ``None`` exactly then), and
    the ``status`` fork routes a selectorless call to the global overview one
    statement earlier.

    Args:
        selector: The coalesced selector, or None when none was supplied.
        command: ``args.command``, for the ``_WORKSPACE_OPTIONAL_VERBS`` carve-out.

    Returns:
        The resolution. Never ``None``.

    Raises:
        ProjectTargetingError: On every resolution failure the carve-out below
            does not cover — including a wholly absent selector.
        RegistryError: If the registry file itself is unusable — propagated
            unwrapped, because there is no registered vocabulary to teach when the
            file holding it cannot be read.
    """
    sel = selector
    if (sel is not None and routing.is_path_shaped(sel) and not sel.startswith("~")
            and not os.path.isabs(sel)):
        sel = os.path.abspath(sel)
    try:
        return routing.resolve_project(sel)
    except ProjectTargetingError as err:
        # The `status`/`agent-block` carve-out (see `_WORKSPACE_OPTIONAL_VERBS`).
        # Only the PATH form: a *name* is a claim about the registry, so an unknown
        # name and a registered-but-vanished one both keep their own error — a NOT
        # SET UP report about a path the caller never typed is a worse answer than
        # either. `is_path_shaped` is checked here as well as implied by the
        # discriminator, so a later resolver change cannot widen this to names by
        # accident. `err.path` is the canonical probed root for this class (as
        # opposed to the registry's recorded string on `registered_unreachable`),
        # which is exactly the value `cmd_status` wants.
        if (command in _WORKSPACE_OPTIONAL_VERBS
                and err.discriminator == TARGET_PATH_NOT_A_WORKSPACE
                and routing.is_path_shaped(sel)):
            # A `ResolvedProject` for a directory that is not (yet) a workspace
            # stretches the dataclass's "one successful resolution" wording. It is
            # deliberate: for these two verbs the report IS the successful answer,
            # and both fields are literally true (a path was named; no registration
            # covers it). The alternative — a second return shape — would fork
            # every downstream read for a case that differs only in whether the
            # directory is populated.
            return routing.ResolvedProject(root=err.path, name=None, via="path")
        raise


def _registered_projects_line(bounded: routing.BoundedNames) -> str:
    """Renders the registered-name vocabulary, respecting the enumeration bound.

    Above ``routing.REGISTERED_NAMES_BOUND`` the enumeration collapses to the
    close matches plus a count plus the discovery pointer — and when there are no
    close matches, to the count and pointer **alone**. An empty ``names`` with
    ``collapsed=True`` is the honest answer, so this must never be spelled
    ``bounded.names or [...]``: that would undo the distinction the policy exists
    to make, in the one place nobody would look for it.

    Args:
        bounded: The policy verdict from ``routing.bounded_registered_names``.

    Returns:
        One indented line for the error body.
    """
    if bounded.total == 0:
        # Empty is first-class, and the CLI is the surface allowed to prescribe
        # the setup act that fills it.
        return "  No projects are registered yet — `mitos init` introduces one."
    if not bounded.collapsed:
        return (f"  Registered projects: {', '.join(bounded.names)} "
                f"(list with `mitos projects`).")
    if bounded.names:
        return (f"  {bounded.total} projects registered, closest: "
                f"{', '.join(bounded.names)} (list them all with `mitos projects`).")
    return (f"  {bounded.total} projects registered — list them with "
            f"`mitos projects`.")


def _render_targeting_error(err: ProjectTargetingError) -> str:
    """Composes the CLI's teaching anatomy for a targeting failure.

    The vision's §4.5 parts: what is wrong, a concrete example, a discovery
    pointer, and a did-you-mean or cwd hint where one exists. Wording lives here —
    at the surface — because it is the difference *between* the surfaces, not
    shared policy: ``routing`` holds what both must agree on (the bound, the
    did-you-mean rule, the ancestor predicate), and this renderer is allowed to
    name ``mitos init`` and ``--project`` precisely because the MCP renderer must
    never. Homing it in ``display.py`` would put it one import from an agent.

    It never calls ``str(err)``: that is the terse discriminator-level fallback
    for an unrendered path, and it is fenced by a tripwire forbidding exactly the
    strings this function emits.

    Args:
        err: The typed error, carrying structured data only.

    Returns:
        The message body, without the boundary's own ``Error: `` prefix. Multi-line;
        every line after the first is indented as a recovery, not a new failure.
    """
    if err.discriminator == TARGET_EXEMPT_VERB:
        note = _EXEMPT_VERB_NOTES.get(
            err.verb, "it targets no single workspace.")
        return f"the `{err.verb}` verb takes no project selector.\n  {note}"

    bounded = routing.bounded_registered_names(err.registered_names, err.close_matches)

    if err.discriminator == TARGET_MISSING:
        lines = [
            "no project selector was supplied.",
            "  Pass a registered name or an absolute path — `mitos --project <name> "
            "<verb>` or `mitos <verb> --project <name>`.",
            _registered_projects_line(bounded),
        ]
    elif err.discriminator == TARGET_UNKNOWN_NAME:
        lines = [f"unknown project {err.selector!r}."]
        if err.close_matches:
            # Never truncated here: `close_project_matches` expands each folded
            # match to every original that folds onto it, so a registry holding
            # several case variants of one name legitimately returns more than
            # `PROJECT_DIDYOUMEAN_MAX` — and dropping one would hide the very
            # distinction the caller needs to see.
            lines.append(f"  Did you mean: {', '.join(err.close_matches)}")
        lines.append(_registered_projects_line(bounded))
    elif err.discriminator == TARGET_RELATIVE_PATH:
        lines = [
            f"the project selector {err.selector!r} is not an absolute path.",
            "  A selector is a registered name or an absolute path — "
            "`--project mitos`, or `--project /home/you/code/mitos`.",
        ]
        if err.selector.startswith("~"):
            # Worth one line only for the shape that earns it: `~` looks absolute
            # to a human and is not, and the reason is almost always a quoted value
            # the shell therefore did not expand.
            lines.append("  (A leading `~` is not expanded here — your shell "
                         "expands it only when the value is unquoted.)")
        lines.append(_registered_projects_line(bounded))
    elif err.discriminator == TARGET_PATH_NOT_A_WORKSPACE:
        lines = [
            f"no Mitos workspace at {err.path!r}.",
            "  A workspace is a directory holding .mitos/config.toml and "
            "decisions.md — run `mitos init` there, or name a registered project.",
            _registered_projects_line(bounded),
        ]
    else:
        # TARGET_REGISTERED_UNREACHABLE — the constructor whitelists the
        # discriminator and the other five are handled above, so this is the
        # remaining class rather than a fall-through default (`errors.
        # _fallback_message` is spelled the same way, for the same reason).
        lines = [
            f"the project {err.name!r} is registered at {err.path!r}, which no "
            f"longer holds a Mitos workspace.",
            f"  Repoint it — `mitos init --name {err.name} --force` in the "
            f"workspace's current location — or edit {registry.registry_path()}.",
            _registered_projects_line(bounded),
        ]

    # The cwd hint is a guess about what the caller *meant*, so it renders only
    # where a name was at stake. On a path-form or exempt failure the caller named
    # something else entirely and a cwd nudge is noise. `os.getcwd()` is passed
    # RAW: `nearest_registered_ancestor` canonicalizes its own argument through the
    # one spelling, and a second canonicalization here would split identity.
    # This is the one registry re-read the phase permits — the hint needs the
    # name→path map, which the error does not carry, and it happens only on the
    # failure path.
    #
    # Guarded because this runs INSIDE the boundary's `except` arm, where a raise
    # escapes `main()` entirely — a sibling `except MitosError` cannot catch what
    # its neighbour handler throws — and the caller would get a traceback in place
    # of the diagnosis they need. Both calls can genuinely fail on this path and
    # only on this path: a targeting failure never constructs a `MitosConfig`, so
    # the hint may be the first cwd read of the whole run (a deleted working
    # directory is an `OSError`), and the registry is re-read here after the raise
    # site already read it (a concurrent edit is a `RegistryError`). Losing an
    # optional hint is the right degradation; losing the whole rendered error is
    # not. Deliberately NOT the guard-undoing shape 2a fenced — that one would
    # swallow a corrupt registry on the *resolution* path, where the fault is the
    # answer; here resolution already succeeded in reading it.
    if err.discriminator in (TARGET_MISSING, TARGET_UNKNOWN_NAME):
        try:
            err.cwd_hint_name = routing.nearest_registered_ancestor(
                os.getcwd(), registry.load())
        except (RegistryError, OSError):
            err.cwd_hint_name = None
        if err.cwd_hint_name:
            lines.append(
                f"  Your working directory sits inside registered project "
                f"{err.cwd_hint_name!r} — pass `--project {err.cwd_hint_name}` if "
                f"that is the target.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Split out of ``main`` so the argument grammar can be asserted on directly —
    verifying an option's spelling by running the verb costs a real workspace read
    and, on a prefix bug, a real write.

    Returns:
        The fully configured top-level parser, subparsers registered.
    """
    parser = argparse.ArgumentParser(
        description="Mitos: Architectural Decision Substrate for LLM-native workflows.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mitos {__version__}")
    parser.add_argument(
        "-C", "--directory", dest="directory", default=None, metavar="DIR",
        help="Run as if mitos were started in DIR (git's -C). Retargets relative "
             "path ARGUMENTS only — it does not name a project, so a verb still "
             "needs `--project`. Must appear BEFORE the verb: "
             "`mitos -C /ws -p /ws record … --rejected-file ./r.txt`.",
    )
    # The project selector, pre-verb half. Its twin is registered on every
    # subparser below, into a DIFFERENT dest — see `_selector_from_args` for why a
    # shared one silently discards this value. Never argparse-`required`: that
    # emits argparse's own usage error and exits 2 before mitos code runs, so none
    # of the teaching anatomy could render.
    parser.add_argument(
        "-p", "--project", dest="project_pre", default=None, metavar="SELECTOR",
        help="The project to act on — REQUIRED on every verb but `init`, `serve`, "
             "`projects` and `set-key --global`: a registered name (see `mitos "
             "projects`) or an absolute path. Accepted on either side of the verb "
             "— `mitos -p mitos list` or `mitos list -p mitos`.",
    )
    # metavar collapses the width-doubling {init,sync,query,query_decisions,…}
    # brace-list in the usage banner to a single COMMAND token (R5). This is a
    # render-only hint — it structurally cannot unregister an alias, so every
    # `aliases=[...]` verb below stays callable while absent from the banner.
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # init
    init_p = subparsers.add_parser("init", help="Initialize Mitos in current workspace.")
    init_p.add_argument(
        "--name", default=None, metavar="NAME",
        help="Register the workspace under NAME (default: this directory's name).")
    init_p.add_argument(
        "--force", action="store_true",
        help="Repoint an existing registration of NAME at this workspace.")

    # projects — the global registry read (needs no workspace; works anywhere).
    proj_p = subparsers.add_parser(
        "projects", help="List the Mitos projects registered on this machine.")
    proj_p.add_argument("--json", action="store_true", dest="as_json",
                        help="Emit machine-readable JSON.")

    # sync
    sync_p = subparsers.add_parser("sync", help="Sync buffer decisions to graph database.")
    sync_p.add_argument("--yes", action="store_true", help="Auto-accept all parsed changes.")
    sync_p.add_argument("--embed-only", action="store_true", help="Drain the pending embeddings outbox queue only.")
    sync_p.add_argument("--verbose", action="store_true", help="Show verbose cache statistics.")
    # The repair door. Same arity as the nine relation flags below: `action="append"`
    # with no `nargs`, so each occurrence stays one whole value (a bare `extend`
    # iterates the string into characters; `nargs="*"` would make the space form
    # parse and swallow the next token). `default=None` distinguishes an absent flag
    # from one supplied naming nothing, which is a refusal rather than a fallback.
    # main() splits the comma form through `_split_relation_slugs`, the tree's single
    # comma-split semantics.
    sync_p.add_argument(
        "--reconcile-entry", default=None, action="append", dest="reconcile_entry",
        metavar="SLUG",
        help="Apply the whole reconcile for the named diverged buffer entry, "
             "including an edge deletion, with no terminal and no --yes. Names the "
             "entry's `### ` slug as `mitos status` reports it (a renamed entry is "
             "named by its NEW slug — the loop reads the markdown). Repeatable and "
             "comma-separated both accumulate: `--reconcile-entry a "
             "--reconcile-entry b` == `--reconcile-entry 'a, b'`. A named target "
             "that does not end in the state your markdown describes exits 1.")

    # reconcile
    rec_p = subparsers.add_parser(
        "reconcile",
        help="Re-embed active nodes missing from Qdrant (heal a direct vector wipe).",
    )
    rec_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # capture
    cap_p = subparsers.add_parser("capture", help="Synthesize and append a decision.")
    cap_p.add_argument("text", help="Raw decision description.")

    # query (alias: query_decisions — MCP tool name)
    q_p = subparsers.add_parser("query", aliases=["query_decisions"],
                                help="Targeted lookup by slug or pointed claim (its confidence rates the ranking, not the corpus).")
    q_p.add_argument("claim", help="Assertion or subsystem query.")
    q_p.add_argument("--depth", default="letter", help="Depth (default: letter).")
    q_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")
    q_p.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — a quick scan.")
    q_p.add_argument("--limit", type=int, default=None,
                     help="Set ranked top-k to retrieve (1–50; default 5). Raises or lowers the count — a context-budget dial.")

    # surface (alias: surface_decisions — MCP tool name) — the precedent-recall loop
    surf_p = subparsers.add_parser("surface", aliases=["surface_decisions"],
                                   help="Surface active decisions relevant to a query (precedent check before deciding — its confidence rates whether precedent exists).")
    surf_p.add_argument("query", help="The claim or topic to find precedents for.")
    surf_p.add_argument("--scope", default=None, help="Optional scope hint (does NOT filter semantic recall — scopes open-questions + note only). Use `list --scope` to hard-filter by scope.")
    surf_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")
    surf_p.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — a quick scan.")
    surf_p.add_argument("--limit", type=int, default=None,
                        help="Set ranked top-k to retrieve (1–50; default 5). Raises or lowers the count — a context-budget dial.")

    # show
    show_p = subparsers.add_parser("show", help="Display details of a specific node.")
    show_p.add_argument("ident", help="Slug or ID of node.")
    show_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")

    # list (alias: list_decisions — the MCP tool name, so an agent's first instinct works)
    list_p = subparsers.add_parser("list", aliases=["list_decisions"],
                                   help="Enumerate the complete set of decisions in a scope (exhaustive recall).")
    list_p.add_argument("--scope", help="Filter by scope tag.")
    list_p.add_argument("--state", help="Computed state filter: 'active' (default, live set), 'all', or an exact state.")
    list_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")
    # brief and oneline are depth tiers below the Letter-complete default — pick one.
    list_depth = list_p.add_mutually_exclusive_group()
    list_depth.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — lighter over a big scope.")
    list_depth.add_argument("--oneline", action="store_true",
                            help="One row per decision: slug + truncated axiom — the orientation tier for big scopes (modifier markers kept).")

    # open-questions
    oq_p = subparsers.add_parser("open-questions", help="List active open questions.")
    oq_p.add_argument("--scope", help="Filter by scope tag.")
    oq_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # scopes (alias: list_scopes — the MCP tool name, so an agent's first instinct works)
    scopes_p = subparsers.add_parser("scopes", aliases=["list_scopes"],
                                     help="Enumerate the scope vocabulary with live-node counts (busiest first).")
    scopes_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")
    scopes_p.add_argument("--archived", action="store_true", dest="archived",
                          help="Include fully-dead domains at a 0/0 floor (scope-level 'list --state all').")

    # import
    imp_p = subparsers.add_parser("import", help="Import legacy prose ADR.")
    imp_p.add_argument("path", help="Path to markdown prose file.")
    imp_p.add_argument("--from", dest="import_from", default="prose", help="Import source kind.")
    imp_p.add_argument("--llm-extract", action="store_true", help="Use Sonnet compression pass.")

    # render
    ren_p = subparsers.add_parser("render", help="Regenerate rendered outputs.")
    ren_p.add_argument("--format", default="live-axioms", help="Target format.")
    ren_p.add_argument("--scope", help="Optional scope filter.")

    # record (alias: record_decision — the MCP tool name, so an agent's first instinct works)
    rec_p = subparsers.add_parser("record", aliases=["record_decision"], help="Record a decision directly to buffer and graph.")
    rec_p.add_argument("axiom", nargs="?", default=None,
                       help="The decision as a single clear sentence true going forward "
                            "(or use --axiom / --axiom-file; exactly one of the three).")
    # `--axiom` gets its OWN dest. Sharing the positional's is the cheaper-looking
    # trick and is dead on arrival: without SUPPRESS the flag form parses to None
    # (the positional's default overwrites it), and with SUPPRESS argument ORDER
    # decides which value survives, so supplying both becomes undetectable and one
    # is silently kept — the silently-keep-last defect on the field that
    # CONSTITUTES identity (a dropped axiom is a different node id, M1/M2). Plain
    # `store`, never `append`: an axiom is one sentence, not a set, so the arity the
    # nine relation flags took has no meaning here. `metavar` or the usage banner
    # leaks the internal dest as `[--axiom AXIOM_FLAG]`.
    rec_p.add_argument("--axiom", default=None, dest="axiom_flag", metavar="AXIOM",
                       help="The axiom as a flag, for callers who reach for one — "
                            "the same value as the positional (exactly one of the three).")
    rec_p.add_argument("--axiom-file", default=None, dest="axiom_file",
                       help="Read the axiom from a file ('-' = stdin) to avoid shell-quoting; "
                            "replaces the inline axiom (exactly one of the three).")
    rec_p.add_argument("--rejected", default=None, help="Alternatives considered and rejected, and why (REQUIRED — or use --rejected-file).")
    rec_p.add_argument("--rejected-file", default=None, dest="rejected_file",
                       help="Read --rejected from a file ('-' = stdin) to avoid shell-quoting long prose.")
    # `action="extend"` so BOTH spellings accumulate: `--scope a b` and `--scope a
    # --scope b`. With `nargs="*"` alone a repeated flag overwrote the destination
    # and silently kept only the last value, while the receipt echoed the truncated
    # list back as if it were the request — the direct cause of every measured scope
    # divergence on the live corpus (AX_FEEDBACK rounds 10 and 11). `extend` copies
    # the destination before extending it, so the `default=[]` list is never mutated
    # in place and a second parse in one process starts empty.
    rec_p.add_argument("--scope", nargs="*", action="extend", default=[],
                       help="Area tags. Repeatable and space-separated both accumulate: "
                            "`--scope database auth` == `--scope database --scope auth`.")
    # `extend` for the same reason as `--scope` above, and with a strictly worse
    # consequence if left out: `mechanisms` is CANONICAL CORE — it feeds
    # `compute_node_id(mechanism_refs=…)` — so a silently dropped value gives the
    # decision a different content-hash id than the author asked for, and re-recording
    # with the full list mints a second node rather than correcting the first (M1: the
    # core is immutable by construction). `default=None` is kept deliberately: absent
    # and empty are distinguished downstream, and `extend` only ever runs when the flag
    # is present, so it never sees the None.
    rec_p.add_argument("--mechanisms", nargs="*", action="extend", default=None,
                       help="Concrete technologies/entities. Repeatable and space-separated "
                            "both accumulate: `--mechanisms sqlite wal-mode`.")
    rec_p.add_argument("--context", default=None, help="Optional background on why this was decided.")
    rec_p.add_argument("--context-file", default=None, dest="context_file",
                       help="Read --context from a file ('-' = stdin).")
    # The nine relation flags accumulate repeats, so `--cites a --cites b` means
    # what `--cites "a, b"` has always meant. As plain last-wins strings they kept
    # only the final occurrence and said nothing — and on the two kill-edge flags
    # that is not a lost link but a false claim about present truth, since state is
    # computed from edges: the prior the author declared superseded stayed in the
    # active view and every read reported it as current.
    #
    # `action="append"` rather than `--scope`/`--mechanisms`' `nargs="*"` + `extend`
    # above, for two measured reasons. A bare `extend` (no `nargs`) iterates each
    # string into CHARACTERS. And `nargs="*"` would make the space form parse, which
    # here is worse than refusing it: `--cites a "My axiom"` swallows record's axiom
    # positional, so the caller supplies an axiom and is refused for missing one.
    # `append` keeps each occurrence one whole value, so neither shape is
    # constructible. `default=None` is kept on all nine — absent and empty are
    # distinguished downstream, and `append` only runs when the flag is present.
    # main() comma-joins the accumulated values back into the single string every
    # consumer already splits (`_split_relation_slugs`); see `_join_relation_flag`.
    rec_p.add_argument("--supersedes", default=None, action="append",
                       help="Exact slug(s) of prior decision(s) this one replaces. "
                            "Repeatable and comma-separated both accumulate: "
                            "`--supersedes a --supersedes b` == `--supersedes 'a, b'`.")
    rec_p.add_argument("--corrects", default=None, action="append",
                       help="Exact slug(s) of prior decision(s) this one corrects "
                            "(kill-edge twin of --supersedes). Repeatable and "
                            "comma-separated both accumulate.")
    rec_p.add_argument("--amends", default=None, action="append",
                       help="Exact slug(s) of decision(s) this one amends. Repeatable "
                            "and comma-separated both accumulate.")
    rec_p.add_argument("--narrows", default=None, action="append",
                       help="Exact slug(s) of decision(s) this one narrows. Repeatable "
                            "and comma-separated both accumulate.")
    rec_p.add_argument("--depends-on", default=None, action="append", dest="depends_on",
                       help="Exact slug(s) of decision(s) this one depends on. Repeatable "
                            "and comma-separated both accumulate.")
    rec_p.add_argument("--resolves", default=None, action="append",
                       help="Exact slug(s) of open question(s) this one resolves (resolves "
                            "is decision→open-question only). Repeatable and "
                            "comma-separated both accumulate.")
    rec_p.add_argument("--contradicts", default=None, action="append",
                       help="Exact slug(s) of decision(s) this one contradicts. Repeatable "
                            "and comma-separated both accumulate.")
    # Accumulates like its siblings — its refusal fires downstream, in the validate
    # phase — but its help stays a refusal explanation, teaching no spelling at all.
    rec_p.add_argument("--derives-from", default=None, action="append", dest="derives_from", help="Not valid when recording a decision — a derives_from edge originates from an open question (open_question -> decision), so a decision cannot be its source. Use --cites to link a decision this one builds on.")
    rec_p.add_argument("--cites", default=None, action="append",
                       help="Exact slug(s) of decision(s) this one cites. Repeatable and "
                            "comma-separated both accumulate.")
    rec_p.add_argument("--slug", required=True,
                       help=f"Explicit slug (handle) for the decision, required "
                            f"(≤{_SLUG_MAX_LEN} chars; an over-length slug is rejected, not truncated).")
    rec_p.add_argument("--acknowledge-neighbors", action="store_true", dest="acknowledge_neighbors",
                       help="Record past the near-duplicate review (the decision is genuinely "
                            "independent). Combines with the relation flags — declared edges "
                            "are still written.")
    rec_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # serve
    subparsers.add_parser("serve", help="Launch Mitos FastMCP server on stdio.")

    # status — is Mitos set up for this project? (human- and LLM-friendly check)
    status_p = subparsers.add_parser("status", help="Check whether Mitos is set up for a project.")
    status_p.add_argument("path", nargs="?", default=None,
                          help="Project to report on: a registered name (see `mitos "
                               "projects`) or an absolute path. Omit it for the "
                               "machine-wide overview of every registered project.")
    status_p.add_argument("--json", action="store_true", dest="as_json", help="Emit a machine-readable JSON report.")
    status_p.add_argument("-v", "--verbose", action="store_true",
                          help="Expand the size-ceiling breakdown to the largest "
                               "decisions in each over-ceiling file. Text only — the "
                               "`--json` payload always carries them.")

    # set-key — store an API key globally (all projects) or for this project
    sk_p = subparsers.add_parser(
        "set-key",
        help="Store an API key globally (all projects) or for one named project.")
    sk_p.add_argument("value", help="The API key value.")
    sk_p.add_argument("--name", default="GEMINI_API_KEY", help="Env var name to store (default: GEMINI_API_KEY).")
    sk_p.add_argument("--global", action="store_true", dest="is_global",
                      help="Write the global ~/.config/mitos/.env (shared by ALL "
                           "projects). Without it, `--project` is required and the "
                           "key lands in that project's own .env.")

    # cutover — the one-time prototype→V1a migration (destructive; operator-run).
    cut_p = subparsers.add_parser(
        "cutover",
        help="One-time migration of a prototype graph to the V1a store (destructive).")
    cut_p.add_argument("--allow-drops", action="store_true", dest="allow_drops",
                       help="Proceed even if active decisions would be dropped from the "
                            "rebuild (P6: a drop may be a deliberate purge).")
    cut_p.add_argument("--yes", action="store_true",
                       help="Skip the interactive confirmation (automation / non-TTY).")
    cut_p.add_argument("--json", action="store_true", dest="as_json",
                       help="Emit a machine-readable JSON report.")

    rebuild_p = subparsers.add_parser(
        "rebuild",
        help="Rebuild the graph from the full corpus through the current catalog "
             "(e.g. after a 0.3.x→0.4.0 upgrade to populate the new edges + mechanisms).")
    rebuild_p.add_argument("--allow-drops", action="store_true", dest="allow_drops",
                           help="Proceed even if entries cannot be rebuilt or active "
                                "decisions would be dropped (the markdown stays the "
                                "source of truth; a drop may be deliberate).")
    rebuild_p.add_argument("--yes", action="store_true",
                           help="Skip the interactive confirmation (automation / non-TTY).")
    rebuild_p.add_argument("--json", action="store_true", dest="as_json",
                           help="Emit a machine-readable JSON report.")

    # check — read-only corpus conflict audit / CI gate (exit 0/1/2).
    check_p = subparsers.add_parser(
        "check",
        help="Audit the live corpus for undeclared contradictions (read-only). "
             "Exit 0 = clean or known-only, 1 = a NEW contradiction, 2 = degraded, "
             "refused, or could not run.")
    check_p.add_argument("--scope", default=None,
                         help="Restrict the audited (proposal) set to one scope tag "
                              "(candidate recall stays scope-blind).")
    check_p.add_argument("--fresh", action="store_true",
                         help="Re-judge every pair, bypassing verdict reuse.")
    check_p.add_argument("--yes", action="store_true",
                         help="Authorize the LLM spend without prompting (the opt-in "
                              "on every non-interactive surface).")
    check_p.add_argument("--json", action="store_true", dest="as_json",
                         help="Emit one machine-readable JSON object (never prompts).")
    check_p.add_argument("--staged", action="store_true",
                         help="Gate the PENDING buffer of decisions.md (the pre-commit / "
                              "CI gate) instead of sweeping the live corpus. Fails closed: "
                              "exit 2 when it cannot run. Not git's staging — reads the "
                              "working-tree decisions.md. Rejects --scope/--fresh.")

    # restore-source — re-materialize a graph-only node's `### slug` block.
    rs_p = subparsers.add_parser(
        "restore-source",
        help="Re-materialize the decisions.md entry of a node whose source block is "
             "missing (so `mitos rebuild` can reconstruct it again).")
    rs_target = rs_p.add_mutually_exclusive_group(required=True)
    rs_target.add_argument("--slug", default=None, help="Restore one node by slug.")
    rs_target.add_argument("--all-graph-only", action="store_true", dest="all_graph_only",
                           help="Restore every node that has no source block.")
    rs_p.add_argument("--dry-run", action="store_true", dest="dry_run",
                      help="Print the blocks that would be written and write nothing.")
    rs_p.add_argument("--json", action="store_true", dest="as_json",
                      help="Emit a machine-readable JSON report.")

    # agent-block — print the canonical agent-file block to paste, or --check pasted copies.
    ab_p = subparsers.add_parser(
        "agent-block",
        help="Print the agent-file block to paste into AGENTS.md/CLAUDE.md/…, or --check for drift.")
    ab_p.add_argument("path", nargs="?", default=None,
                      help="The project — a registered name or an absolute path. "
                           "Required (the flag spelling `-p` works too); `--check` "
                           "scans this project's agent files.")
    ab_p.add_argument("--check", action="store_true",
                      help="Scan the project's agent files and report stale/unversioned mitos notes.")

    # Prefix abbreviation OFF on every verb. argparse defaults it ON, which made
    # `--axiom` an unambiguous prefix of `--axiom-file`: a 470-character axiom went
    # to the file reader and the command died as `[Errno 36] File name too long`
    # through the outermost boundary, as a "Fatal Unexpected Error" naming nothing
    # the caller could act on. That founding example is now HISTORICAL — `--axiom`
    # is a declared option, so it resolves exactly and no longer abbreviates
    # anything; a reader reproducing it will find it does not reproduce. The rule
    # still holds for every other flag (`--axiom-fil` is the probe that still
    # bites). Setting `allow_abbrev=False` on the top-level parser
    # does NOT propagate — an `add_parser()` child keeps its own True — so it is
    # pinned here over the whole registered set rather than as a kwarg on `record`,
    # which the next verb added would be one forgotten argument away from missing.
    # argparse reads the attribute at parse time, so assigning it after
    # registration is equivalent to passing it in.
    #
    # The post-verb `--project` is registered in the SAME loop, for the same
    # reason: a per-verb `add_argument` line is one forgotten argument away from
    # missing on the next verb added, and the miss is not a soft one —
    # `mitos record -p mitos` would be argparse's own `unrecognized arguments`,
    # exit 2 before any mitos code runs, carrying none of the anatomy.
    #
    # Deduped by `id()` because the five aliased verbs (`query`/`query_decisions`,
    # `surface`/`surface_decisions`, `list`/`list_decisions`, `scopes`/
    # `list_scopes`, `record`/`record_decision`) are ONE parser object under two
    # names: 27 names over 22 objects, and a second `add_argument` on the same
    # object raises `ArgumentError: conflicting option strings`. The
    # `allow_abbrev` assignment above needs no such guard — it is idempotent — so
    # the dedupe guards only the registration. Being one object is also what makes
    # the aliases free: covering the object once covers both names.
    _selector_registered: Set[int] = set()
    for _verb_parser in subparsers.choices.values():
        _verb_parser.allow_abbrev = False
        if id(_verb_parser) in _selector_registered:
            continue
        _selector_registered.add(id(_verb_parser))
        _verb_parser.add_argument(
            "-p", "--project", dest="project_post", default=None,
            metavar="SELECTOR",
            help="The project to act on — REQUIRED on every verb but `init`, "
                 "`serve`, `projects` and `set-key --global`: a registered name "
                 "(see `mitos projects`) or an absolute path. Also accepted "
                 "before the verb (`mitos -p NAME …`).",
        )

    return parser


def main() -> None:
    """Main CLI execution router."""
    # Make raw-text print()s crash-safe on a non-UTF-8 stdout before any verb
    # can print (R6). Inert on a UTF-8 terminal; CLI-only — the MCP transport
    # has no terminal stdout to harden (P7 bulkhead).
    apply_stdout_text_safety(sys.stdout)
    parser = _build_parser()
    args = parser.parse_args()

    try:
        # Constructed INSIDE the try so a strict-loader ConfigError on a malformed
        # `.mitos/config.toml` is caught by `except MitosError` below and rendered
        # as a one-line `Error: …` — not a raw traceback (the 6a raising-loader owns
        # this boundary). The `finally`'s only config read (config.workspace_dir for
        # the MCP hint) is already wrapped in its own `except Exception: pass`, so an
        # unbound `config` after a construction failure stays silent.
        #
        # -C/--directory runs FIRST (before the selector and the config) so the
        # whole workspace retargets together: chdir into the target, and every
        # relative path arg, the cwd hint and an argless convenience default all
        # read the post-chdir location. An absent -C target raises MitosError here
        # and renders through the `except MitosError` boundary as a clean one-line
        # error.
        #
        # No `.env` is loaded here, and that absence is the paradigm (phase 5c):
        # keys are never promoted into `os.environ`, they are resolved per call by
        # `env.resolve_values` for the workspace the call named, and hung on
        # `MitosConfig.env` (config.py). Precedence is unchanged — real env >
        # project `.env` > global `.env` — but it is now computed for the *target*
        # rather than baked in from the *launch* directory.
        _enter_target_directory(args.directory)
        # The project selector, resolved ONCE, here. Order is contract: the -C
        # chdir above runs first (so a relative selector and the cwd hint both read
        # the post-chdir location), the exempt check runs before resolution (so a
        # selector on `init` is answered by naming the verb, not by looking the
        # name up), and only then does a target exist.
        selector = _selector_from_args(args)
        _refuse_selector_on_exempt_verb(args, selector)
        if args.command == "status" and selector is None:
            # `status` is the one selector-OPTIONAL verb: a selector is legal and
            # routes to the deep report, its absence routes to the global overview.
            # A plain condition, deliberately NOT folded into `_exempt_reason` — a
            # selector on `status` is not refused, and merging the two rules would
            # make `mitos -p mitos status` refusable by accident.
            #
            # Placed BEFORE the `MitosConfig` construction, and the ordering is
            # contract: the overview is global by definition, so a malformed
            # `.mitos/config.toml` in whatever directory the caller happens to be
            # standing in must not take down a report about the whole machine. (4b's
            # `ConfigError` carve-out would catch it and route to
            # `_answer_workspace_optional_verb` → the *deep* report: silently the
            # wrong answer, exit 1 either way, nothing red.) The early `sys.exit` is
            # safe past the `finally` unconditionally: since 6a retired the MCP
            # nudge, the `finally` reads no `config` at all (its one surviving arm,
            # the update notice, binds only module-level names), so no unbound local
            # on this path can become a swallowed `NameError`. It used to be safe
            # only *conditionally* — because `status` was not in the frozenset
            # gating the nudge — and that weaker argument went with the nudge.
            sys.exit(cmd_status_overview(as_json=args.as_json))
        # `target` is `None` EXACTLY when the verb targets no workspace — the same
        # predicate the refusal above used, called once more rather than
        # hand-spelled a second time. For every other verb `_resolve_selector`
        # either returns a validated workspace or raises the §4.5 teaching error:
        # since 5a there is no third outcome, and in particular no working-directory
        # fallback. A caller who omits the selector gets a calm error naming the
        # recovery, not a silent write into whatever directory the process started
        # in.
        target = (None if _exempt_reason(args) is not None
                  else _resolve_selector(selector, args.command))
        # `project=` carries the caller's own vocabulary onto the config so every
        # echo below names the target the way the caller addressed it.
        # `target.name` is already the registered name for both selector forms
        # and `None` for an unregistered path, which the constructor resolves to
        # the canonical path. The exempt arm builds an explicit `MitosConfig(".")`
        # — same resolution as the zero-arg default it replaced, but an argument
        # rather than a default, which is what 5d removed the default behind
        # (`os.path.abspath` inside the constructor, never a working-directory read,
        # so entry-005's standing grep stays honest).
        try:
            config = (MitosConfig(target.root, project=target.name) if target
                      else MitosConfig("."))
        except ConfigError:
            # The existing carve-out, extended by one error class rather than a new
            # mechanism: for the two verbs that answer ABOUT a directory, a broken
            # config.toml is the answer they exist to give. Every other verb keeps
            # today's calm one-line boundary error.
            if args.command not in _WORKSPACE_OPTIONAL_VERBS:
                raise
            sys.exit(_answer_workspace_optional_verb(args, target))
        # Warn about the workspace the VERB will act on, not merely the CWD. Since
        # `status`/`agent-block`'s positional is now a selector source feeding the
        # same `config`, that is simply `config` — there is no second target to
        # build, and no `args.path` left for a flag-spelled call to fall back from.
        _warn_deprecated_rotation_mode(config)
        if args.command == "init":
            # `config` stays the first POSITIONAL argument: the suite asserts on
            # `mock_init.call_args.args[0]` to check which workspace a -C run
            # targeted, and a keyword here would empty that tuple.
            cmd_init(config, name=args.name, force=args.force)
        elif args.command == "projects":
            cmd_projects(as_json=args.as_json)
        elif args.command == "sync":
            # The split lives here, at the call site, exactly as 1a's join does:
            # `None` (flag absent) survives as `None` so the handler can tell it
            # from a flag supplied naming nothing. Handles reach `cmd_sync` in the
            # caller's VERBATIM spelling — never `_normalize_slug`'d, which runs on
            # the `record` path alone while the parser keeps a header slug verbatim.
            cmd_sync(config, auto_accept=args.yes, embed_only=args.embed_only,
                     verbose=args.verbose,
                     repair_targets=(
                         None if args.reconcile_entry is None
                         else [h for v in args.reconcile_entry
                               for h in _split_relation_slugs(v)]
                     ))
        elif args.command == "reconcile":
            sys.exit(cmd_reconcile(config, as_json=args.as_json))
        elif args.command == "capture":
            cmd_capture(config, args.text)
        elif args.command in ("query", "query_decisions"):
            cmd_query(config, args.claim, depth=args.depth, as_json=args.as_json, brief=args.brief, limit=args.limit)
        elif args.command in ("surface", "surface_decisions"):
            cmd_surface(config, args.query, scope=args.scope, as_json=args.as_json, brief=args.brief, limit=args.limit)
        elif args.command == "show":
            cmd_show(config, args.ident, as_json=args.as_json)
        elif args.command in ("list", "list_decisions"):
            cmd_list(config, scope=args.scope, state_filter=args.state, as_json=args.as_json,
                     brief=args.brief, oneline=args.oneline)
        elif args.command == "open-questions":
            cmd_open_questions(config, scope=args.scope, as_json=args.as_json)
        elif args.command in ("scopes", "list_scopes"):
            cmd_scopes(config, as_json=args.as_json, archived=args.archived)
        elif args.command == "import":
            cmd_import(config, args.path, use_llm_extract=args.llm_extract)
        elif args.command == "render":
            cmd_render(config, scope=args.scope, render_format=args.format)
        elif args.command in ("record", "record_decision"):
            # Exactly one axiom source: the positional, --axiom, or --axiom-file
            # (the quoting-safe twin of --rejected-file). Same JSON-aware dead-end
            # shape as the missing-rejected check below.
            #
            # The COUNT feeds both the guard and the --json code, deliberately. The
            # previous chooser read one dest — `"ambiguous" if args.axiom is not
            # None else "missing"` — an exact discriminator at arity two and wrong
            # at three: `--axiom X --axiom-file f` leaves the positional None and
            # reported `missing_axiom` to a caller who had supplied two. One value,
            # so the two cannot drift apart again. Gates are `is not None`, never
            # truthiness: an empty axiom is a SUPPLIED source that the write path
            # then refuses on its own terms.
            _axiom_sources = [source for source
                              in (args.axiom, args.axiom_flag, args.axiom_file)
                              if source is not None]
            if len(_axiom_sources) != 1:
                msg = ("record requires exactly one axiom source: the positional "
                       "axiom, --axiom, or --axiom-file ('-' = stdin) — one of the "
                       f"three, and {len(_axiom_sources)} were supplied.")
                if args.as_json:
                    _emit_json({"error": msg,
                                "code": ("ambiguous_axiom_source" if _axiom_sources
                                         else "missing_axiom")})
                else:
                    print(msg, file=sys.stderr)
                sys.exit(2)
            # Only one argument can read stdin — the first reader drains it and the
            # rest come back empty. Left unchecked that surfaced as a downstream
            # "record requires --rejected" wall, which names a flag the caller
            # already passed and sends them to add it twice.
            _stdin_args = [f"--{n}-file" for n in ("axiom", "rejected", "context")
                           if getattr(args, f"{n}_file", None) == "-"]
            if len(_stdin_args) > 1:
                msg = ("only one argument can read from stdin — these ask for it: "
                       f"{', '.join(_stdin_args)}. Pass a file path for all but one.")
                if args.as_json:
                    _emit_json({"error": msg, "code": "multiple_stdin_args"})
                else:
                    print(msg, file=sys.stderr)
                sys.exit(2)
            # The guard above proved at most one inline source is set, so coalescing
            # them is safe and `_read_text_arg`'s file-wins precedence never fires on
            # a caller who supplied both.
            _inline_axiom = args.axiom if args.axiom is not None else args.axiom_flag
            axiom = _read_text_arg(_inline_axiom, args.axiom_file)
            if args.axiom_file is not None and axiom.endswith("\n"):
                axiom = axiom[:-1]  # strip the single trailing newline files/heredocs add
            rejected = _read_text_arg(args.rejected, args.rejected_file)
            if not (rejected and rejected.strip()):
                msg = ("record requires --rejected or --rejected-file "
                       "(the rejected alternatives are mandatory).")
                if args.as_json:
                    # No stderr walls under --json: the dead-end speaks a structured
                    # object on stdout, with the exit code preserved (2).
                    _emit_json({"error": msg, "code": "missing_rejected"})
                else:
                    print(msg, file=sys.stderr)
                sys.exit(2)
            context = _read_text_arg(args.context, args.context_file)
            cmd_record(
                config,
                axiom=axiom,
                rejected=rejected,
                scope=args.scope,
                mechanisms=args.mechanisms,
                context=context,
                supersedes=_join_relation_flag(args.supersedes),
                corrects=_join_relation_flag(args.corrects),
                amends=_join_relation_flag(args.amends),
                narrows=_join_relation_flag(args.narrows),
                depends_on=_join_relation_flag(args.depends_on),
                resolves=_join_relation_flag(args.resolves),
                contradicts=_join_relation_flag(args.contradicts),
                derives_from=_join_relation_flag(args.derives_from),
                cites=_join_relation_flag(args.cites),
                slug=args.slug,
                acknowledge_neighbors=args.acknowledge_neighbors,
                as_json=args.as_json,
            )
        elif args.command == "serve":
            cmd_serve()
        elif args.command == "status":
            # `target.name` — the registered name, already resolved once above —
            # rather than anything `cmd_status` could recover from a path. Read at
            # the boundary, passed into the callee (2a's `cwd_hint_name`, 4a's D5),
            # so the report cannot name the project wrong on a symlinked route.
            sys.exit(cmd_status(config.workspace_dir, as_json=args.as_json,
                                project=target.name, verbose=args.verbose))
        elif args.command == "restore-source":
            sys.exit(cmd_restore_source(
                config, slug=args.slug, all_graph_only=args.all_graph_only,
                dry_run=args.dry_run, as_json=args.as_json))
        elif args.command == "agent-block":
            # THE CHANNEL CARVE-OUT (§4.7) — not an obligation carve-out; the two
            # are different arguments and must stay distinct. `agent-block`'s plain
            # form prints a paste-ready block whose own text tells the reader to
            # paste it into AGENTS.md / CLAUDE.md, so an echo on stdout would land a
            # resolved name and a path-hashed collection INSIDE a committed,
            # travelling artifact — the persisted machine-specific identity §4.3
            # forbids. So it answers out of band, on stderr, for BOTH forms: the
            # `--check` report is a diagnostic and could safely take stdout, but two
            # channels for one verb is a drift seam for no gain.
            #
            # Emitted here rather than in the handler on purpose: `cmd_agent_block`
            # gains no echo code at all, so "the block is byte-identical across
            # machines" stays a structural fact rather than a discipline someone
            # could later tidy onto stdout.
            _echo_corpus(config, file=sys.stderr)
            sys.exit(cmd_agent_block(config.workspace_dir, check=args.check))
        elif args.command == "set-key":
            # The other dispatch-site echo, for the same reason: `cmd_set_key` takes
            # a bare path, not a config, and `main()` is the only place that holds
            # both. Project form only — `--global` is selector-exempt and writes the
            # machine-wide .env, which names no corpus.
            if not args.is_global:
                _echo_corpus(config)
            cmd_set_key(args.value, name=args.name, is_global=args.is_global,
                        workspace_dir=target.root if target else None)
        elif args.command == "cutover":
            sys.exit(cmd_cutover(config, allow_drops=args.allow_drops,
                                 assume_yes=args.yes, as_json=args.as_json))
        elif args.command == "rebuild":
            sys.exit(cmd_rebuild(config, allow_drops=args.allow_drops,
                                 assume_yes=args.yes, as_json=args.as_json))
        elif args.command == "check":
            sys.exit(cmd_check(config, staged=args.staged, scope=args.scope,
                               fresh=args.fresh, assume_yes=args.yes,
                               as_json=args.as_json))
    except ProjectTargetingError as e:
        # ABOVE `except MitosError`, which it subclasses: without this arm the
        # fallback `str(e)` would render — a terse discriminator-level sentence
        # with no example, no vocabulary and no recovery, on exactly the failure an
        # agent has to act on in one turn. Same stderr channel and same exit
        # mapping as every other boundary fault, `--json` included: the shipped
        # boundary answers on stderr regardless, so a JSON envelope for one error
        # class would be a new asymmetry.
        print(f"Error: {_render_targeting_error(e)}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    except RegistryError as e:
        # Also above `except MitosError`, and for the mirror of the reason above:
        # `registry.load()`'s body is the located cause only — it names no `mitos`
        # command, because since 5b every MCP tool call reaches it and that surface
        # may not be handed a state-creating one. The recovery clause is each
        # boundary's own, and THIS boundary is the one a human reads, so it is the
        # one allowed to name the setup act (`mcp_server._render_registry_error`
        # names the absolute-path escape hatch instead). Appended here rather than
        # folded back into the leaf so the two wordings cannot re-merge by accident.
        #
        # Gated on `file_unusable` because this arm catches the whole class: a
        # *registration refusal* (`--name` already taken, path already registered)
        # is not repaired by re-running the thing that just refused, and it already
        # carries its own `--name`/`--force` recovery. Widening the nudge to it
        # would trade one wrong prescription for another.
        detail = str(e)
        if e.file_unusable:
            # "fixed or removed", not "readable": only one of the five refusals is a
            # readability fault. The other four are a file that opens perfectly and
            # says something illegal, and each located cause above already ends in
            # `Fix or remove …` — so this clause completes that sentence rather than
            # renaming the fault. The flag is `file_unusable` for the same reason.
            detail += ("\n  `mitos init` in a project re-registers it once the "
                       "file is fixed or removed.")
        print(f"Error: {detail}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    except MitosError as e:
        # KD1: `check` maps every pre-verb/boundary failure (bad -C, ConfigError, an
        # escaped store fault) to exit 2 — for CI, "could not run" is one routing
        # class with the verb's own exit-2 refusals; no other verb's contract moves.
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    except Exception as e:
        # The generic arm needs the same conditional: an unexpected crash under
        # `check` must not read as exit-1 "findings" to a CI consumer.
        print(f"Fatal Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    finally:
        # Best-effort 'new version available' nudge, AFTER the command's own
        # output (the finally runs even on the sys.exit paths above). Skipped for
        # the long-running MCP server; fully fail-silent so it never disrupts work.
        if args.command != "serve":
            try:
                from mitos._update import update_notice
                _notice = update_notice(__version__)
                if _notice:
                    print(_notice, file=sys.stderr)
            except Exception:
                pass


if __name__ == "__main__":
    main()
