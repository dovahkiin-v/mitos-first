"""Tests for phase 2c: the key and model-override consumers ride `config.env`.

2b built the resolver and hung its answer on ``MitosConfig.env``. This phase
routes the consumers onto it: every credential and every model override reaches
its leaf as a value the config-holding orchestrator read off the map for **the
workspace the call named**, rather than off whatever directory the process
happened to be launched in.

The discipline every group-2/3/6 row here inverts from 2b's construction rows:
the sentinel goes in the **workspace's own** ``.env`` (tier 2) with **nothing
exported**, so a site that resolved from the process environment instead of the
target cannot pass by accident. 2b's four ``QDRANT_URL`` rows
(``test_env_resolution.py``, W20) set the value in tier 1 and are cited rather
than duplicated — they prove the value arrives, these prove it arrives *from the
target*.

Group 5 was 2c's live pin on the transitional fallback and is now its inversion:
5c deleted the shim and the entry-time dotenv load together, and each row there
asserts the keyless posture where it used to assert the process environment
answering. That group plus the AST sweep below are the shim's only net — 5c's own
I6/I7 suites do not cover it, because behind a routed site the fallback never
fires; what they catch is a call site that was never routed (a real key failing
to arrive). Different failures, different nets.

Like ``test_env_resolution.py``, this module writes ``os.environ`` in places —
that is the point — so it carries the same module-autouse ``_keyless`` strip and
the same ``_unset`` helper. Copied deliberately: a bare
``monkeypatch.delenv(name, raising=False)`` on an already-absent name records
nothing, so monkeypatch's undo has nothing to undo and a raw write leaks into
every module collected afterwards.
"""

import ast
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from mitos import cli, embeddings, models
from mitos.check import execute_corpus_check, plan_corpus_check
from mitos.config import MitosConfig, RESOLVED_ENV_KEYS
from mitos.conflict import RenderedPrompt
from mitos.conflict_judgment import _JUDGMENT_MODEL_ALIAS, make_judgment_executor
from mitos.env import TIER_ENVIRONMENT, TIER_GLOBAL_ENV, TIER_PROJECT_ENV
from mitos.errors import EmbeddingError
from mitos.models import MODEL_IDS, get_embedding_model_id, get_model_id
from mitos.parser import ParsedEntry
from mitos.store import GraphStore, open_connection
from mitos.telemetry import TelemetryStore

from _conflict_helpers import _keyed_substrate, _match

# The one production alias that reaches the three provenance resolves. Pinned as a
# literal beside the import so a divergence reads as a mismatch, not a rename.
PRODUCTION_ALIAS = "SONNET"


def _unset(monkeypatch, name: str) -> None:
    """Removes `name` for the test AND guarantees it stays removed at teardown.

    2b's helper verbatim (``test_env_resolution.py``). ``delenv(name,
    raising=False)`` on an already-absent name records nothing; setting it first
    forces the absence — or the real prior value — into monkeypatch's record, so
    a raw ``os.environ`` write later in the row is undone too. Correct in both
    directions, because the records unwind in reverse.
    """
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _keyless(monkeypatch) -> None:
    """Strips every name this module routes, so each row builds its own tiers.

    Six test modules pour the repo's real ``.env`` into ``os.environ`` at *import*
    time, two of them not ``*_live.py``, so in any full-suite run the credentials
    are present for everything collected afterwards regardless of
    ``MITOS_NO_LIVE_TESTS``. Every row here opts *in* to the tier it means; a row
    that assumed absence would pass alone and red in collection order.

    ``QDRANT_URL`` is pointed at a closed port rather than stripped: nothing here
    touches Qdrant, and a real endpoint inherited from the shell would let a
    construction row reach the network.
    """
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                 "QDRANT_URL", *RESOLVED_ENV_KEYS):
        _unset(monkeypatch, name)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")


def _write(path, text: str) -> str:
    """Writes a file (creating parents) and returns its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _workspace(tmp_path, name: str = "proj", env_text: str = "") -> str:
    """A real workspace whose own `.env` is the only place a value lives.

    The graph file is materialized so a store construction is a normal open
    rather than a creation, and ``.env`` is written only when asked — an absent
    file is the keyless tier-2 this module's degradation rows need.
    """
    ws = tmp_path / name
    (ws / ".mitos").mkdir(parents=True)
    if env_text:
        _write(ws / ".env", env_text)
    GraphStore(str(ws / ".mitos" / "graph.sqlite"))
    return str(ws)


@pytest.fixture
def genai_keys(monkeypatch) -> List[Optional[str]]:
    """Records the `api_key` every `GeminiEmbeddingProvider` construction passes.

    The credential is deliberately **not** stored on the provider (P8), so the
    client constructor is the only seam a routed key can be observed at. Do not
    "fix" that by adding an ``api_key`` attribute to make a row easier.
    """
    seen: List[Optional[str]] = []

    def _client(*, api_key: Optional[str] = None) -> Any:
        seen.append(api_key)
        return MagicMock()

    monkeypatch.setattr(embeddings.genai, "Client", _client)
    return seen


@pytest.fixture
def anthropic_keys(monkeypatch) -> List[Optional[str]]:
    """Records the `api_key` every judge builder passes to `anthropic.Anthropic`."""
    import anthropic

    seen: List[Optional[str]] = []

    def _client(*, api_key: Optional[str] = None) -> Any:
        seen.append(api_key)
        return MagicMock()

    monkeypatch.setattr(anthropic, "Anthropic", _client)
    return seen


# =========================================================================== #
# Group 1 — `models.py` is a pure function of the map it is handed
# =========================================================================== #

def _resolve_alias(alias: str, env: Optional[Dict[str, str]]) -> str:
    """Dispatches to whichever of the two registry functions owns `alias`."""
    if alias == "EMBEDDING":
        return get_embedding_model_id(env)
    return get_model_id(alias, env)


@pytest.mark.parametrize("alias", sorted(MODEL_IDS))
def test_every_model_override_is_read_from_the_supplied_map(alias):
    """All four overrides, built from `MODEL_IDS` — never from `MODEL_ALIASES`.

    ``MODEL_ALIASES`` omits ``EMBEDDING``, so a parametrization built from it
    looks complete while dropping the one override costliest to get wrong: the
    embedding cache keys on content hash alone, so a mis-routed embedding
    override reads as working while cached prior-generation vectors flow into a
    new-generation collection.
    """
    env = {f"MITOS_MODEL_OVERRIDE_{alias}": f"override-{alias.lower()}"}
    assert _resolve_alias(alias, env) == f"override-{alias.lower()}"


@pytest.mark.parametrize("alias", sorted(MODEL_IDS))
def test_an_empty_override_in_the_map_leaves_the_baseline_id_in_force(alias):
    """The shipped truthiness test, preserved verbatim through the reroute.

    An empty override must not blank the model id — the alternative is a call
    issued against ``model=""``.
    """
    env = {f"MITOS_MODEL_OVERRIDE_{alias}": ""}
    assert _resolve_alias(alias, env) == MODEL_IDS[alias]


@pytest.mark.parametrize("alias", sorted(MODEL_IDS))
def test_an_absent_map_yields_the_baseline_id(alias):
    """No map supplied means no override applies — not a lookup somewhere else."""
    assert _resolve_alias(alias, None) == MODEL_IDS[alias]


def test_an_unknown_alias_still_raises_with_the_shipped_message():
    """The message interpolates `MODEL_ALIASES`, not `MODEL_IDS` — unchanged."""
    with pytest.raises(ValueError) as exc:
        get_model_id("GPT", {})
    assert "Unsupported model alias: GPT" in str(exc.value)
    assert str(models.MODEL_ALIASES) in str(exc.value)


def test_the_model_registry_reads_no_process_environment(monkeypatch):
    """A process-env override does NOT reach the registry — permanently.

    Not a transitional row: ``models.py`` is a Tier-1 leaf that must not import
    ``config`` (which imports *it* — an immediate cycle) and must not read the
    process environment, because an override living in a workspace's ``.env``
    belongs to *that* workspace. An override reaches a call by being passed;
    every production reach is routed (see groups 2, 3 and 6). 5c does not touch
    this row.
    """
    for alias in MODEL_IDS:
        monkeypatch.setenv(f"MITOS_MODEL_OVERRIDE_{alias}", f"from-environ-{alias}")

    for alias in MODEL_IDS:
        assert _resolve_alias(alias, None) == MODEL_IDS[alias]
        assert _resolve_alias(alias, {}) == MODEL_IDS[alias]


def test_the_model_registry_imports_exactly_typing():
    """The exact import closure, over the AST — a stronger claim than "no `os`".

    ``import os`` went with the reads: a leaf that no longer touches the process
    environment should not be *able* to, and a prose mention of ``os.environ``
    must not satisfy a grep-shaped check. The closure is exact rather than a
    blacklist because the rule is "stdlib typing only, and nothing from
    ``mitos``, ever".
    """
    tree = ast.parse(open(models.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported == {"typing"}


# =========================================================================== #
# Group 2 — each routed consumer receives the TARGET's value
# =========================================================================== #

SENTINEL_ENV = "GEMINI_API_KEY=from-the-target\n"


def test_the_sync_manager_builds_its_provider_on_the_targets_key(
    tmp_path, genai_keys
):
    """W19 at `sync.py`'s construction site — the workspace's own key arrives.

    The construction sits inside a broad ``except Exception: pass``, so a
    signature fault here degrades to ``embed_provider = None`` rather than
    raising. Asserting the *recorded key* (and the provider's presence) is what
    keeps a swallowed error from reading as a pass.
    """
    from mitos.sync import MitosSyncManager

    manager = MitosSyncManager(MitosConfig(_workspace(tmp_path, env_text=SENTINEL_ENV)))

    assert manager.embed_provider is not None
    assert genai_keys == ["from-the-target"]


def test_the_importer_builds_its_provider_on_the_targets_key(tmp_path, genai_keys):
    """W19 at `importer.py`'s construction site (the same swallow, one worse)."""
    from mitos.importer import MitosProseImporter

    importer = MitosProseImporter(
        MitosConfig(_workspace(tmp_path, env_text=SENTINEL_ENV))
    )

    assert importer.embed_provider is not None
    assert genai_keys == ["from-the-target"]


def test_the_check_substrate_builds_its_provider_on_the_targets_key(
    tmp_path, genai_keys
):
    """W19 at `cli._build_check_substrate` — the seam tests inject keyed fakes at."""
    embed, _, embed_detail = cli._build_check_substrate(
        MitosConfig(_workspace(tmp_path, env_text=SENTINEL_ENV))
    )

    assert embed is not None and embed_detail is None
    assert genai_keys == ["from-the-target"]


def test_the_mcp_server_builds_its_provider_on_the_targets_key(
    tmp_path, genai_keys
):
    """W19 at `mcp_server.get_workspace_components`.

    It takes the workspace config as an argument (phase 3c), so the cwd read that
    used to live inside it now lives at the call site — which is what leaves this
    row proving what it always proved: the key of the workspace *given* is the key
    the provider is built on. Phase 5d removed the constructor's ``"."`` default,
    so the target is named outright and the ``chdir`` that used to supply it is
    gone. The row keeps its bite: pytest's cwd is the repo, itself a valid
    workspace carrying a real ``GEMINI_API_KEY``, so a callee that read the working
    directory instead of the config would resolve *that* key and this assertion
    would still red.
    """
    from mitos import mcp_server

    ws = _workspace(tmp_path, env_text=SENTINEL_ENV)
    _, embed_provider, _ = mcp_server.get_workspace_components(MitosConfig(ws))

    assert embed_provider is not None
    assert genai_keys == ["from-the-target"]


def test_the_check_judge_builds_its_client_on_the_targets_key(
    tmp_path, anthropic_keys
):
    """W19 at `cli._build_check_judge` — the ANTHROPIC half, keyed and keyless."""
    keyed = MitosConfig(
        _workspace(tmp_path, "keyed", "ANTHROPIC_API_KEY=from-the-target\n")
    )
    assert cli._build_check_judge(keyed) is not None
    assert anthropic_keys == ["from-the-target"]

    assert cli._build_check_judge(MitosConfig(_workspace(tmp_path, "bare"))) is None
    assert anthropic_keys == ["from-the-target"]  # no second client was built


def test_the_sync_conflict_judge_builds_its_client_on_the_targets_key(
    tmp_path, genai_keys, anthropic_keys
):
    """W19 at `sync._build_conflict_judge`.

    Its availability gate reaches the key only when ``embed_provider`` and
    ``vector_store`` are both live, so the workspace carries the Gemini key too —
    the row would otherwise return ``None`` for the wrong reason.
    """
    from mitos.sync import MitosSyncManager

    ws = _workspace(
        tmp_path,
        env_text="GEMINI_API_KEY=from-the-target\nANTHROPIC_API_KEY=anthropic-target\n",
    )
    manager = MitosSyncManager(MitosConfig(ws))

    assert manager._build_conflict_judge() is not None
    assert anthropic_keys == ["anthropic-target"]


def test_capture_with_no_key_anywhere_prints_the_shipped_line(tmp_path, capsys):
    """The keyless disposition at `cmd_capture`, byte-identical after the reroute.

    Four "…is not set" strings share a prefix and a grep returns them as one
    class; this one and sync's are the two an existing test substring-matches, so
    the reroute must move neither.
    """
    cli.cmd_capture(MitosConfig(_workspace(tmp_path)), "We will use python.")

    out = capsys.readouterr().out
    assert "GEMINI_API_KEY environment variable is not set" in out
    assert "Capture requires it" in out


def test_sync_with_no_key_anywhere_prints_the_shipped_line(tmp_path, capsys):
    """The keyless disposition at `perform_sync`'s gate.

    The gate sits below the buffer parse, so a pending entry is required — an
    empty buffer returns before the key is ever consulted.
    """
    config = MitosConfig(_workspace(tmp_path))
    cli.cmd_init(config)
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n## 2026-06-01 — routed-pending — A pending decision\n"
            "**Decided:** Some decision.\n"
            "**Rejected:** None.\n"
            "**Mechanisms:** python\n"
            "**Scope:** substrate\n"
        )
    capsys.readouterr()

    cli.cmd_sync(config, auto_accept=True)

    out = capsys.readouterr().out
    assert "GEMINI_API_KEY environment variable is not set" in out
    assert "Sync requires API keys" in out


def test_llm_import_with_no_key_anywhere_prints_the_shipped_line(tmp_path, capsys):
    """The keyless disposition at `import_from_file`'s ANTHROPIC gate.

    The key is read unconditionally but gated only under ``use_llm_extract``, so a
    keyless non-LLM import stays a normal path — this drives the gated half.
    """
    from mitos.importer import MitosProseImporter

    config = MitosConfig(_workspace(tmp_path))
    source = _write(tmp_path / "legacy.md", "## A legacy ADR\n\nWe chose python.\n")

    MitosProseImporter(config).import_from_file(source, use_llm_extract=True)

    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY environment variable is not set" in out
    assert "Import --llm-extract requires it" in out


def test_an_exported_empty_key_still_refuses_at_the_provider(tmp_path, genai_keys):
    """`""` is a supplied answer that must keep masking the file tiers (D4).

    ``env GEMINI_API_KEY= ANTHROPIC_API_KEY= mitos …`` is the shipped idiom for a
    keyless run on a key-bearing box, and it works by *masking*: the workspace
    ``.env`` here carries a real key and must lose to the empty export.

    The second half is what makes this a pin rather than a decoration. ``""``
    and ``None`` behave alike while ``config.env``'s tier 1 *is* ``os.environ``,
    so the two spellings only diverge once the process environment has moved on
    from what the config captured — which is precisely the shape ~104 test sites
    already have. A supplied ``""`` must survive a *real* key appearing beside
    it: spelling the fallback ``x or os.environ.get(...)`` instead of
    ``if x is None:`` turns a deliberately keyless run into a keyed, billed one.
    """
    os.environ["GEMINI_API_KEY"] = ""  # raw, deliberately — `_unset` undoes it
    config = MitosConfig(_workspace(tmp_path, env_text=SENTINEL_ENV))
    cache = os.path.join(config.mitos_dir, "embedding_cache.sqlite")

    assert config.env["GEMINI_API_KEY"] == ""
    with pytest.raises(EmbeddingError):
        embeddings.GeminiEmbeddingProvider(
            cache, api_key=config.env.get("GEMINI_API_KEY")
        )

    os.environ["GEMINI_API_KEY"] = "a-real-key-that-must-not-win"
    with pytest.raises(EmbeddingError):
        embeddings.GeminiEmbeddingProvider(
            cache, api_key=config.env.get("GEMINI_API_KEY")
        )
    assert genai_keys == []


# =========================================================================== #
# Group 3 — the override reaches the call, and the provenance stamp agrees
# =========================================================================== #

def test_the_embedding_override_from_the_targets_env_reaches_the_provider(
    tmp_path, genai_keys
):
    """The row this phase exists for on the override side.

    ``compute_content_hash`` is ``sha256(text)`` and nothing else — the cache key
    carries no model id — so a mis-routed embedding override does not fail. It
    reads as working while cached prior-generation vectors flow into a
    new-generation collection, and nobody finds out for months.
    """
    from mitos.sync import MitosSyncManager

    ws = _workspace(
        tmp_path,
        env_text=SENTINEL_ENV + "MITOS_MODEL_OVERRIDE_EMBEDDING=gemini-embedding-next\n",
    )
    manager = MitosSyncManager(MitosConfig(ws))

    assert manager.embed_provider is not None
    assert manager.embed_provider.model_id == "gemini-embedding-next"


def _fake_message(text: str) -> MagicMock:
    """A fake `messages.create` return with a tool_use block carrying parsed verdicts."""
    import json as _json
    verdicts = _json.loads(text)
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"verdicts": verdicts}
    msg = MagicMock()
    msg.content = [tool_block]
    msg.stop_reason = "tool_use"
    msg.usage = MagicMock(
        input_tokens=120, output_tokens=45,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    return msg


def _client_returning(message: MagicMock) -> MagicMock:
    """A fake client whose `with_options(...).messages.create(...)` returns `message`."""
    client = MagicMock()
    client.with_options.return_value.messages.create.return_value = message
    return client


def _created_model(client: MagicMock) -> str:
    """The `model` kwarg the bound judge actually issued."""
    create = client.with_options.return_value.messages.create
    create.assert_called_once()
    return create.call_args.kwargs["model"]


def test_the_bound_judge_issues_the_model_id_it_was_built_with():
    """`make_judgment_executor(client, model_id=…)` binds the id into the call.

    The id rides the closure rather than the facade, so the facade's ``judge``
    stays a one-arg function of a ``RenderedPrompt`` — the frozen executor
    boundary is threaded, not widened.
    """
    client = _client_returning(_fake_message("[]"))
    judge = make_judgment_executor(client, model_id="claude-sonnet-from-target")

    judge(RenderedPrompt(system="S", user="U", prompt_version="conflict-tenability-v1"))

    assert _created_model(client) == "claude-sonnet-from-target"


def _batches(telemetry: TelemetryStore) -> List[Dict[str, Any]]:
    """Reads back every ``judgment_batches`` row (read-only; insertion order)."""
    import sqlite3

    conn = open_connection(telemetry.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in
                conn.execute("SELECT * FROM judgment_batches ORDER BY rowid")]
    finally:
        conn.close()


def _commit(store: GraphStore, slug: str, axiom: str) -> str:
    """Commits a decision and returns its content-hash node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "An alternative."
    return store.commit_parsed_entry(entry).node_id


def test_the_judged_model_id_and_the_persisted_one_are_the_same_string(tmp_path):
    """The join-key row: what the call used and what provenance records must agree.

    With ``MITOS_MODEL_OVERRIDE_SONNET`` in a workspace ``.env`` **only**, two
    independent resolutions have to land on it — the judge's, bound at build
    time from ``config.env``, and ``execute_corpus_check``'s, taken inside the
    batch loop from the ``env`` map it is handed (``execution.model_alias`` does
    not exist before then, which is why the map travels rather than an id).

    This is the row that catches a half-threaded phase. The call using the
    override while the telemetry column records the baseline raises nothing,
    fails no other assertion, and is a silently wrong provenance column forever.
    """
    override = "claude-sonnet-from-target"
    ws = _workspace(
        tmp_path, env_text=f"MITOS_MODEL_OVERRIDE_{PRODUCTION_ALIAS}={override}\n"
    )
    config = MitosConfig(ws)
    store = GraphStore(config.db_path)
    telemetry = TelemetryStore(config.telemetry_path)

    a_id = _commit(store, "routed-a", "The first axiom under judgment.")
    _commit(store, "routed-b", "The second axiom under judgment.")
    nodes = {
        node["core_axiom"]: node
        for node in store.get_decisions(state="active")
    }
    partner = next(n for n in nodes.values() if n["id"] != a_id)
    neighbourhoods = {axiom: [] for axiom in nodes}
    neighbourhoods["The first axiom under judgment."] = [_match(partner["slug"], 0.9)]
    embed, vector = _keyed_substrate(neighbourhoods)

    plan = plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector,
        telemetry=telemetry, model_alias=PRODUCTION_ALIAS,
    )
    assert len(plan.fresh_groups) == 1

    # The batch's partner slugs come off the group, never from the fixture's own
    # idea of which node became the proposal — grouping orients by hash, which is a
    # DB accident. A guessed slug parses to `Unavailable` and the row would red on
    # the wrong thing.
    verdicts = json.dumps([
        {"slug": pair.partner_node["slug"], "rationale": "They coexist.",
         "tenable_together": True, "confidence": 0.9}
        for pair in plan.fresh_groups[0].pairs
    ])
    client = _client_returning(_fake_message(verdicts))
    judge = make_judgment_executor(
        client, model_id=get_model_id(_JUDGMENT_MODEL_ALIAS, config.env)
    )

    result = execute_corpus_check(
        plan, judge=judge, telemetry=telemetry, store=store, env=config.env
    )

    assert result.judgment_degraded is None
    batches = _batches(telemetry)
    assert len(batches) == 1
    assert _created_model(client) == override
    assert batches[0]["model_id"] == override


# =========================================================================== #
# Group 4 — `_gemini_key_source` on the resolver's tier report
# =========================================================================== #

def test_the_key_source_reports_the_project_env_tier(tmp_path):
    """Tier 2 wins when nothing is exported."""
    ws = _workspace(tmp_path, env_text="GEMINI_API_KEY=PROJKEY\n")
    assert cli._gemini_key_source(ws) == TIER_PROJECT_ENV


def test_the_key_source_reports_the_global_env_tier(tmp_path, monkeypatch):
    """Tier 3 wins when the workspace has only the scaffolded empty slot.

    The shape ``mitos init`` produces and its own README recommends: an empty
    ``GEMINI_API_KEY=`` line under a comment telling the user to set the key once
    globally. A resolver testing key *presence* at tier 2 would answer
    ``project .env`` here and every project following the tool's advice would
    lose its key.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    cli.cmd_set_key("GLOBALKEY", workspace_dir=None, is_global=True)
    ws = _workspace(tmp_path, env_text="GEMINI_API_KEY=\n")
    assert cli._gemini_key_source(ws) == TIER_GLOBAL_ENV


def test_the_key_source_reports_the_environment_tier(tmp_path, monkeypatch):
    """Tier 1 wins over both files."""
    monkeypatch.setenv("GEMINI_API_KEY", "ENVKEY")
    ws = _workspace(tmp_path, env_text="GEMINI_API_KEY=PROJKEY\n")
    assert cli._gemini_key_source(ws) == TIER_ENVIRONMENT


def test_the_key_source_is_none_when_nothing_carries_the_key(tmp_path):
    """No tier answered — and `_gemini_key_present` (its dead wrapper) agrees."""
    ws = _workspace(tmp_path)
    assert cli._gemini_key_source(ws) is None
    assert cli._gemini_key_present(ws) is False


def test_an_exported_empty_key_is_reported_as_no_key(tmp_path, monkeypatch):
    """The report is keyed on the VALUE, never on the tier being non-None.

    An exported-empty variable resolves to ``ResolvedValue("", "environment")``
    — a real answer with a real tier — so keying on ``.tier is not None`` would
    make ``env GEMINI_API_KEY= mitos status`` claim a key is present while every
    consumer refuses to run.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")
    ws = _workspace(tmp_path, env_text="GEMINI_API_KEY=PROJKEY\n")
    assert cli._gemini_key_source(ws) is None


def test_status_attributes_a_project_env_key_to_its_own_file(
    tmp_path, monkeypatch, capsys
):
    """The inversion 2c wrote this row for: `project .env`, never `environment`.

    Driven through ``cli.main()`` on a workspace whose ``.env`` carries the key
    with nothing exported. Until 5c, ``main()`` poured that ``.env`` into
    ``os.environ`` before anything resolved, so the env-first resolver answered
    ``environment`` for a key whose durable home was the file — less specific,
    never false, since ``main()`` genuinely had put it there. 5c deleted the
    entry-time load, and with nothing promoting a file's key into tier 1 the
    attribution names the file.

    That is 2c's stated cost paid off (ADR
    ``key-source-attribution-reports-the-tier-that-won-not-the-durable-home``),
    and it is also this module's cheapest net against the entry load coming
    back: restore either ``load_dotenv_file`` call and this row reds.

    Bound to the tier **constants**, never to literals: the three strings are the
    join key between ``env.resolve_key`` and ``status``'s attribution line, so a
    rename lands as one failing import rather than as a green test asserting a
    dead string.
    """
    ws = _workspace(tmp_path, env_text="GEMINI_API_KEY=PROJKEY\n")
    monkeypatch.chdir(tmp_path)  # restores the cwd `main`'s -C chdir moves
    # `-p .` after the `-C` chdir: post-5a a selectorless `status` is the machine-wide
    # overview, which reports no key attribution for anyone.
    monkeypatch.setattr(sys, "argv", ["mitos", "-C", ws, "-p", ".", "status"])

    # `status` exits non-zero on this bare workspace (never initialized, no Qdrant)
    # — irrelevant to the attribution, which prints on every branch.
    with pytest.raises(SystemExit):
        cli.main()

    out = capsys.readouterr().out
    assert f"GEMINI_API_KEY (from {TIER_PROJECT_ENV})" in out
    assert f"GEMINI_API_KEY (from {TIER_ENVIRONMENT})" not in out


# =========================================================================== #
# Group 5 — the keyless posture, where the fallback used to answer
# =========================================================================== #

class TestTheKeylessPostureWhereTheFallbackUsedToAnswer:
    """The shim is gone (5c): a key the config did not resolve reaches nobody.

    Until 5c, ``env.transitional_env_fallback`` answered from ``os.environ``
    whenever a caller supplied nothing — a compatibility shim that let 2c's
    routing diff be purely additive. It could not lie while it lived, because
    tier 1 of the resolution it backed up **is** ``os.environ``. It stopped being
    harmless the moment 5c deleted the entry-time dotenv load: nothing promotes a
    workspace's keys into the process environment any more, so a site that forgot
    to pass would silently resolve nothing — or the launch directory's residue,
    which is the defect this whole vision exists to close.

    Every row below is the *inversion* of a row 2c wrote as a live pin, and the
    shape is deliberate: a real key sits in ``os.environ``, the config resolved
    **nothing**, and the consumer must take its keyless branch anyway. 5c's own
    I6/I7 suites cannot catch a surviving fallback — behind a routed site it
    never fires, so they pass with it fully intact; what they detect is an
    *unrouted* site. These rows plus the AST sweep below are what would notice
    the shim growing back.

    Five of the six key-consumption families the index names had a row of this
    class at 2c. The sixth — importer extraction — did not, and is the last row
    here, so the set is now complete rather than nearly so.
    """

    def test_the_provider_without_an_api_key_refuses_instead_of_reading_env(
        self, tmp_path, monkeypatch, genai_keys
    ):
        """2c's row inverted: `EmbeddingError`, with a real key in the environment.

        The seven bare ``GeminiEmbeddingProvider(cache_path)`` constructions the
        shim kept green were migrated to an explicit ``api_key=`` in the same
        commit — five of them in live-tier modules CI cannot see.

        ``genai_keys`` is asserted **empty** rather than merely unused: the
        refusal has to happen before a client is constructed, so an edit that
        built the client first and validated after cannot pass this row.

        ``api_key=None`` is spelled out because 5d made the keyword required; the
        claim is unchanged, because ``None`` was always a *supplied* answer and
        takes the same refusal branch the absent default used to. What 5d closed
        is the neighbouring case — a call site that forgot the keyword entirely —
        and that one is a ``TypeError``, pinned in
        ``tests/test_workspace_root_discipline.py``.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "from-the-process")

        with pytest.raises(EmbeddingError):
            embeddings.GeminiEmbeddingProvider(
                str(tmp_path / "cache.sqlite"), api_key=None
            )

        assert genai_keys == []

    def test_sync_without_a_resolved_key_refuses_instead_of_reading_env(
        self, tmp_path, monkeypatch, capsys
    ):
        """2c's row inverted: the "not set" line prints WITH the key in `os.environ`.

        The dominant test shape in this tree is config-first, key-second: a
        fixture builds ``MitosConfig`` and each row then writes
        ``os.environ["GEMINI_API_KEY"]`` raw. ``config.env`` is captured at
        construction, so ~113 sites across 21 files used to reach their routed
        consumer through the shim. This is that shape, and it now refuses.
        """
        config = MitosConfig(_workspace(tmp_path))
        cli.cmd_init(config)
        with open(config.decisions_file, "a", encoding="utf-8") as f:
            f.write(
                "\n## 2026-06-01 — fallback-pending — A pending decision\n"
                "**Decided:** Some decision.\n"
                "**Rejected:** None.\n"
                "**Mechanisms:** python\n"
                "**Scope:** substrate\n"
            )
        assert "GEMINI_API_KEY" not in config.env
        monkeypatch.setenv("GEMINI_API_KEY", "from-the-process")
        capsys.readouterr()

        cli.cmd_sync(config, auto_accept=True)

        assert "Sync requires API keys" in capsys.readouterr().out

    def test_the_check_judge_without_a_resolved_key_refuses_instead_of_reading_env(
        self, tmp_path, monkeypatch, anthropic_keys
    ):
        """2c's row inverted: `_build_check_judge(config)` returns `None`."""
        config = MitosConfig(_workspace(tmp_path))
        assert "ANTHROPIC_API_KEY" not in config.env
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-process")

        assert cli._build_check_judge(config) is None
        assert anthropic_keys == []

    def test_capture_without_a_resolved_key_refuses_instead_of_reading_env(
        self, tmp_path, monkeypatch, capsys
    ):
        """2c's row inverted: the "not set" line prints WITH the key in `os.environ`."""
        config = MitosConfig(_workspace(tmp_path))
        assert "GEMINI_API_KEY" not in config.env
        monkeypatch.setenv("GEMINI_API_KEY", "from-the-process")

        cli.cmd_capture(config, "We will use python.")

        assert "Capture requires it" in capsys.readouterr().out

    def test_the_importer_without_a_resolved_key_refuses_instead_of_reading_env(
        self, tmp_path, monkeypatch, anthropic_keys, capsys
    ):
        """The sixth family, which had no row of this class before 5c.

        ``import_from_file`` reads ``ANTHROPIC_API_KEY`` unconditionally and gates
        it under ``use_llm_extract``, so the LLM-extract form is the one that can
        observe an unrouted key. ``anthropic_keys`` empty is the sharp half: the
        SDK client is constructed one line below the gate, so a key arriving from
        anywhere other than ``config.env`` would show up there.
        """
        from mitos.importer import MitosProseImporter

        config = MitosConfig(_workspace(tmp_path))
        source = _write(tmp_path / "legacy.md",
                        "## A legacy ADR\n\nWe chose python.\n")
        assert "ANTHROPIC_API_KEY" not in config.env
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-process")
        capsys.readouterr()

        MitosProseImporter(config).import_from_file(source, use_llm_extract=True)

        assert "Import --llm-extract requires it" in capsys.readouterr().out
        assert anthropic_keys == []

# --- the structural net ----------------------------------------------------

# EVERY `os.environ` read in `mitos/`, as `(module, function)`. **Five keys, six
# reads** since 6a — `MITOS_NO_MCP_HINT` retired with the per-project MCP-wiring
# nudge, so `cli.py` now reads the process environment nowhere at all. After 5c
# there is no second dict beside this one either: the transitional shim's single
# read is gone, and so is `cli.load_dotenv_file`'s pair — which were also the
# tree's only `os.environ` WRITE. Keyed on the enclosing function rather than a
# line number: the set is exact either way, and this spelling does not churn when
# a file above it moves.
PERMANENT_ENV_READS = {
    ("env.py", "_resolve"): 2,          # the resolver's own tier 1 — the legitimate one
    ("config.py", "_hint_cache_path"): 1,   # XDG_CACHE_HOME — genuinely process-scoped
    ("config.py", "config_home"): 1,        # XDG_CONFIG_HOME — likewise
    ("_update.py", "_cache_path"): 1,       # XDG
    ("_update.py", "update_notice"): 1,     # MITOS_NO_UPDATE_CHECK quiet-switch
}


def _environ_reads(path: str) -> List[str]:
    """Every `os.environ` READ in a module, as the name of its enclosing function.

    Four shapes, because the tree uses all four: ``os.environ.get(...)``,
    ``os.environ.setdefault/pop(...)``, ``name in os.environ``, and a
    ``os.environ[name]`` subscript in either context. Swept over the AST rather
    than the text — several of these modules discuss ``os.environ`` in prose, it
    being the thing this design stopped consulting.
    """
    found: List[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def _scoped(self, node: ast.AST) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _scoped
        visit_AsyncFunctionDef = _scoped
        visit_ClassDef = _scoped

        def _hit(self) -> None:
            found.append(".".join(self.stack) or "<module>")

        @staticmethod
        def _is_environ(node: ast.AST) -> bool:
            return isinstance(node, ast.Attribute) and node.attr == "environ"

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr in ("get", "setdefault", "pop")
                    and self._is_environ(func.value)):
                self._hit()
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if any(self._is_environ(c) for c in node.comparators):
                self._hit()
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if self._is_environ(node.value):
                self._hit()
            self.generic_visit(node)

    Visitor().visit(ast.parse(open(path, encoding="utf-8").read()))
    return found


def test_the_process_environment_is_read_only_at_the_declared_sites():
    """The exact set of `os.environ` reads in `mitos/` — the checklist, not a comment.

    **One** declared set since 5c. It was two, because they died on different
    days: ``PERMANENT_ENV_READS`` (XDG resolution, two quiet-switches, the
    resolver's own tier 1 — and, until 5c, ``load_dotenv_file``'s guard-plus-
    write pair) beside ``TRANSITIONAL_ENV_READS``, the compatibility shim's one
    site. Both of 5c's members were deleted outright and the second dict went
    with its only member — an empty dict left behind is a hook for the next
    shim.

    **Five keys, six reads** since 6a, which retired ``MITOS_NO_MCP_HINT`` along
    with the nudge it silenced — leaving ``cli.py`` reading the process
    environment nowhere. The prior count is pinned independently outside this
    file: the ADR recorded at 2b says *"nine permanent … and exactly one
    transitional … 5c shrinks the first to seven and the second to zero"* —
    reads, not keys — and 6a takes that seven to six by deleting a reader, not by
    re-routing one. Two sources agreeing is the check; any other number means one
    of them is wrong.

    This is the row that catches a bare read growing back — including the
    asymmetric case the behavioural rows cannot see: route every credential but
    leave ``models.py`` reading the process environment, and only this reds.
    """
    counted: Dict[Any, int] = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(models.__file__), "*.py"))):
        module = os.path.basename(path)
        for func in _environ_reads(path):
            counted[(module, func)] = counted.get((module, func), 0) + 1

    assert counted == PERMANENT_ENV_READS
    assert sum(counted.values()) == 6


def test_the_process_environment_is_written_nowhere_in_mitos():
    """The other half, and the one 5c makes absolute: `mitos` mutates no environment.

    The read set above tolerates six legitimate consultations (five keys). There is no
    legitimate **write**: a program that writes its own environment cannot answer
    the same question twice about two different projects, which is the property
    this vision exists to establish. Until 5c the tree had exactly one writer,
    ``cli.load_dotenv_file``'s two statements; it was deleted rather than merely
    unhooked, so the claim is structural rather than policed.

    Swept over the parsed code for every mutating shape — a subscript **store**
    (plain and augmented), a rebind of the attribute itself, a ``del``, and the
    four mutating dict methods — because the read sweep above counts an
    ``os.environ[name]`` subscript without caring which side of an assignment it
    sits on, so it would pass a write as a read.
    """
    class WriteVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.hits: List[str] = []

        @staticmethod
        def _is_environ_sub(node: ast.AST) -> bool:
            return (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "environ")

        @staticmethod
        def _is_environ_itself(node: ast.AST) -> bool:
            # `os.environ = {...}` — a rebind rather than a mutation, and it would
            # be invisible to a subscript-only sweep while doing strictly more.
            return isinstance(node, ast.Attribute) and node.attr == "environ"

        def _targets(self, node, targets) -> None:
            for t in targets:
                if self._is_environ_sub(t) or self._is_environ_itself(t):
                    self.hits.append(ast.dump(t))
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            self._targets(node, node.targets)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self._targets(node, [node.target])

        def visit_Delete(self, node: ast.Delete) -> None:
            self._targets(node, node.targets)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr in ("setdefault", "pop", "update", "clear")
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "environ"):
                self.hits.append(func.attr)
            self.generic_visit(node)

    offenders: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(models.__file__), "*.py"))):
        v = WriteVisitor()
        v.visit(ast.parse(open(path, encoding="utf-8").read()))
        if v.hits:
            offenders[os.path.basename(path)] = v.hits

    assert offenders == {}


def test_the_entry_dotenv_loader_does_not_exist():
    """A name-absence claim, not a call-site grep — D1's structural form.

    The index asks for *"no entry path calls ``load_dotenv_file``"*. Asserting
    that as a call-site sweep leaves a live, tested, importable ``os.environ``
    writer one line away from any future verb, and invites someone to satisfy the
    claim by adding a "just for ``init``" call somewhere the sweep does not look.
    So the function was deleted, and this asserts the *symbol* is gone: there is
    no writer to call.

    Named ``mitos.cli`` explicitly rather than swept, because the failure message
    should say which symbol came back.
    """
    import mitos.cli
    import mitos.env

    assert not hasattr(mitos.cli, "load_dotenv_file")
    assert not hasattr(mitos.env, "transitional_env_fallback")


# Every model-registry call in `mitos/` that resolves WITHOUT a map, as
# `(module, function)`. All five are the leaf fallbacks a caller supplying
# nothing lands on; every other call site is routed, and the row below is what
# keeps that true.
UNROUTED_MODEL_RESOLVES = {
    ("embeddings.py", "GeminiEmbeddingProvider.__init__"),
    ("sync.py", "run_sync_enrichment"),
    ("sync.py", "run_ambient_capture"),
    ("importer.py", "run_llm_prose_compression"),
    ("conflict_judgment.py", "execute_judgment"),
}


def test_every_model_resolve_outside_a_leaf_fallback_is_handed_a_map():
    """The override half of the routing, swept structurally.

    A dropped ``env`` argument at a provenance resolve raises nothing, fails no
    other assertion, and writes a model the run never used into a column nobody
    reads until they need it — group 3's join-key row is the only *behavioural*
    net, and it covers one of the three resolves. This covers all of them, and
    catches the same omission at the two model-id resolves that never reach
    telemetry at all.
    """
    unrouted = set()
    for path in sorted(glob.glob(os.path.join(os.path.dirname(models.__file__), "*.py"))):
        tree = ast.parse(open(path, encoding="utf-8").read())
        stack: List[str] = []

        class Visitor(ast.NodeVisitor):
            def _scoped(self, node: ast.AST) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_FunctionDef = _scoped
            visit_AsyncFunctionDef = _scoped
            visit_ClassDef = _scoped

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                name = (func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else None)
                needed = {"get_model_id": 2, "get_embedding_model_id": 1}.get(name)
                if needed is not None and len(node.args) < needed:
                    unrouted.add((os.path.basename(path), ".".join(stack)))
                self.generic_visit(node)

        Visitor().visit(tree)

    assert unrouted == UNROUTED_MODEL_RESOLVES


def test_the_two_orchestrators_hand_their_map_down():
    """`cmd_check`'s and `_run_staged_check`'s calls carry `env=`.

    The row above proves each resolve *takes* a map; this proves the two CLI
    sites that own one actually pass it. A resolve reading ``env=None`` because
    its caller forgot the keyword is indistinguishable, at every other seam,
    from a workspace that simply set no override.
    """
    tree = ast.parse(open(cli.__file__, encoding="utf-8").read())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) == "execute_corpus_check"
             or getattr(node.func, "id", None) == "_persist_staged_batch")
    ]
    assert len(calls) == 2, "the two hand-down sites moved — re-read §10 item 16/17"
    for call in calls:
        assert "env" in [kw.arg for kw in call.keywords]


# =========================================================================== #
# Group 6 — T10's offline half: a cross-directory call resolves the TARGET's key
# =========================================================================== #

def test_a_cross_directory_cli_call_resolves_the_targets_key(
    tmp_path, monkeypatch, genai_keys
):
    """I7 on the CLI surface: cwd in A, config for B, B's key arrives.

    Nothing is exported, and A's ``.env`` carries a *different* sentinel — so a
    site resolving from the working directory fails loudly rather than passing on
    an empty tier. The ``QDRANT_URL`` companion is already proven by 2b's four
    construction rows (W20, ``test_env_resolution.py``); it is cited, not
    duplicated.
    """
    a = _workspace(tmp_path, "a", "GEMINI_API_KEY=key-of-a\n")
    b = _workspace(tmp_path, "b", "GEMINI_API_KEY=key-of-b\n")
    monkeypatch.chdir(a)

    embed, _, _ = cli._build_check_substrate(MitosConfig(b))

    assert embed is not None
    assert genai_keys == ["key-of-b"]


def test_a_cross_directory_mcp_call_resolves_the_targets_key(
    tmp_path, monkeypatch, genai_keys
):
    """I7 on the MCP surface, in-process.

    ``get_workspace_components`` takes the workspace config as an argument (phase
    3c), so the cross-directory claim this row makes is made at the call site: the
    process sits in A, the config names B, and B's key — not A's — is the one the
    provider was built on. The ``chdir`` stays because *this* row's subject is the
    directory move itself, unlike W19's above, whose subject is the config given
    and which dropped its ``chdir`` with 5d's constructor default. What a *stale*
    target surviving a move looks like at the process level is no longer askable
    here at all — there is no cwd-derived target left to go stale — and the
    process-level version of that hazard is 5c's **real `mitos serve` subprocess**
    row on 3a's harness
    (``test_one_session_answers_from_the_graph_file_on_disk_at_each_call``); an
    in-process approximation cannot observe the entry path or process-owned env,
    which is exactly where I6's hazards live, so it is deliberately not attempted
    here.
    """
    from mitos import mcp_server

    a = _workspace(tmp_path, "a", "GEMINI_API_KEY=key-of-a\n")
    b = _workspace(tmp_path, "b", "GEMINI_API_KEY=key-of-b\n")
    monkeypatch.chdir(a)

    _, embed_provider, _ = mcp_server.get_workspace_components(MitosConfig(b))

    assert embed_provider is not None
    assert genai_keys == ["key-of-b"]
