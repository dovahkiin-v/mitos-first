# Mitos
<!-- mcp-name: io.github.dovahkiin-v/mitos -->
<!-- ^ MCP Registry ownership proof. The registry reads this out of the PyPI
     long_description, so it must survive into the published package and must
     byte-match `name` in server.json. Enforced server-side: a publish without it
     is rejected 400. tests/test_packaging.py locks the pair. -->


![Status: Alpha](https://img.shields.io/badge/status-alpha-orange) ![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue) ![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

> 🔧 **Early release** — actively developed

When you build software with AI assistants over months, the *reasoning* behind your decisions gets lost. The assistant forgets why you chose one approach, re-suggests options you already rejected, and your design notes drift out of sync with what was actually decided. Mitos is a memory layer for those decisions: it records each decision, the alternatives you ruled out, and how later decisions replace earlier ones — then feeds that history back to your AI assistant in a compact, trustworthy form.

The result: your AI collaborator stays consistent with the calls you've actually made — it stops contradicting a past decision or re-opening a settled question, and your decision record never silently rots.

Under the hood: markdown for humans (`decisions.md` is the source of truth you can always read and grep), a typed graph for the agents (SQLite + a local Qdrant for semantic recall), and an MCP server so agents check precedent before deciding and record decisions as they make them.

---

## Fastest install: hand it to your agent

If you work with an AI coding agent (Claude Code, Cursor, Gemini CLI, …), the easiest path is to let it do the setup. In the project you want mitos in, give your agent:

```text
Read https://github.com/dovahkiin-v/mitos/blob/main/SETUP.md and set up mitos
for this project. When done, run `mitos status .` from the project directory
and report the result.
```

What your agent will end up doing — the same steps a human follows, all in [SETUP.md](SETUP.md) where you can read them first:

- install the `mitos` CLI via pipx, from this repository;
- start a local Qdrant container (`qdrant/qdrant` on port `7333`, isolated from any Qdrant you already run);
- register the MCP server once for the whole machine, if it isn't registered already;
- initialize the project workspace, which also registers the project by name;
- ask you to set your API keys yourself (`mitos set-key`) — a Gemini key (required), and an Anthropic key for the conflict-audit layer (strongly recommended); the setup guide tells agents not to handle key values.

How much your agent asks along the way is governed by your own agent's settings, not by this prompt.

## Manual setup

The same steps by hand — full detail in **[SETUP.md](SETUP.md)**:

1. **Install** (once per machine): `pipx install git+https://github.com/dovahkiin-v/mitos`
2. **Start Qdrant** (once per machine, shared by all projects): `docker compose up -d` from this repo — mitos runs its own instance on `:7333`, so it never touches a Qdrant you use for other work.
3. **Register the MCP server** (once per machine, recommended for agents): `claude mcp add --scope user mitos -- mitos serve`. One registration serves every project — see [SETUP.md](SETUP.md) for what it costs, for other harnesses, and for why a leftover per-project `.mcp.json` entry has to go.
4. **Per project**: `mitos init` from the project root, then `mitos set-key --global <your-Gemini-key>` (one key covers everything; get it at <https://aistudio.google.com/app/apikey>). Gemini is the tested embedding provider today; a multi-provider abstraction is on the roadmap.
5. **Verify**: `mitos status .` → `READY ✓`.

`mitos status .` is the compass throughout: it says exactly what's done, what's missing, and what to do next for that project. With no project named, `mitos status` answers the other question — what does this machine have — listing every registered project and checking Qdrant.

**Every command names its project.** There is no default target: `mitos init` registers the project by name, and from then on each verb takes `-p <name>`, `-p <absolute path>`, or `-p .` from the project root (agents pass the same thing as a `project` argument). `mitos projects` lists what's registered. That is what lets one install and one MCP server serve every project on the machine without a call ever landing in the wrong corpus.

## How it runs

Mitos is **per-project** — each project gets its own decision graph and its own Qdrant collection. Day to day, three verbs carry the loop (as MCP tools for agents, with identical CLI twins):

| Verb | When |
|---|---|
| `surface_decisions` (`mitos surface`) | *Before* deciding — is there precedent? Every hit carries the alternatives that were already rejected and why. |
| `record_decision` (`mitos record`) | The moment something is settled — the decision, the rejected paths, and how it relates to prior decisions (supersedes, amends, …). |
| `query_decisions` (`mitos query`) | Looking something up — by meaning or by exact handle. |

A few properties worth knowing:

- **The markdown is the source of truth.** Every decision lands in `decisions.md`, human-readable and greppable; the graph and the search index are derived from it and can always be rebuilt (`mitos rebuild`).
- **Decisions are never edited or deleted — they're superseded.** State (active / superseded / amended) is computed from typed relations between decisions, so the history of *why* always survives.
- **It fails safe.** If the search index or the embedding API is down, recording still works and search degrades to an honest text-match over the markdown — nothing blocks, nothing is lost, and degraded output says it's degraded.
- **It audits itself.** The corpus sweep (`mitos check -p .`) finds decisions that silently contradict each other, and `--staged` gates new entries as a pre-commit or CI step — see [SETUP.md](SETUP.md) for the hook, CI and cron recipes, which name their project three different ways.

Explore the rest with `mitos --help` — the help text doubles as the API reference.

## Why it exists

Building software through intensive LLM design reviews produces architectural decisions faster than a person can track. One month of that working style produced close to 900 decision records in a single markdown file — no longer greppable, readable, or manageable by hand. Existing ADR tools are built for human teams logging the occasional decision; mitos is built for a solo developer whose AI assistants generate and consume decisions continuously.

If that's your way of working, project size doesn't matter much — the higher the decision volume, the faster mitos moves from comfort to necessity.

## Development

```bash
pip install -e '.[test]'
MITOS_NO_LIVE_TESTS=1 pytest -m "not packaging" -n auto   # offline suite, parallel (~50s)
pytest -m "not packaging"                                 # adds the live tier — serial only
pytest -m packaging                                       # real-install check: fresh venv + pip install
```

`-n auto` is safe for the offline suite and **not** for the live tier: the test-collection
sweep is session-scoped, so parallel workers delete each other's Qdrant collections and
the affected tests degrade to skips rather than failures.

The `*_live.py` suites and golden Layer B make **real Gemini and Anthropic API calls**
against your own keys, and need Qdrant on `:7333`. They skip when no key is resolvable,
so a fresh clone runs the fast path by default.

Keys resolve from the environment, a repo-root `.env`, or `~/.config/mitos/.env` — so if
you already *use* mitos, a test run can pick up your personal key and spend against it.
Opt out explicitly:

```bash
MITOS_NO_LIVE_TESTS=1 PYTHONPATH=. pytest -m "not packaging"
```

The canonical decision format lives in [`mitos/format-spec.md`](mitos/format-spec.md). License: [Apache 2.0](LICENSE).
