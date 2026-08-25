# Adding Mitos to a project

Mitos is **per-project**: each project gets its own decision graph and its own
Qdrant collection, sharing one Qdrant instance. Setting it up for a new project
takes a couple of minutes.

Two commands are worth knowing before you start, because they answer different
questions:

- **`mitos status .`** — *is this one project ready?* Run it from the project
  root at any time; it names what is done, what is left, and what to do next, and
  it exits `0` when ready, `1` otherwise.
- **`mitos status`** with no project — *what does this machine have?* It lists
  every registered project, flags the broken ones, and checks Qdrant. It exits
  `0` whenever it can render that report, so it is a survey, not a verdict.

> **For LLM agents setting mitos up:** run `mitos status .` from the project
> directory first — if it already says `READY ✓`, there is nothing to install.
> Otherwise work through this guide top to bottom; everything you will install is
> listed under *Prerequisites* (the `mitos` CLI via pipx from this repository, and
> one Docker container: `qdrant/qdrant` on port `7333`). Two rules:
> - **Never handle the API key value.** When you reach step 2, ask the user to
>   run the `mitos set-key` command themselves and wait; do not ask them to
>   paste the key to you, and do not read or edit key values in any `.env`.
> - **If a step fails, stop and report** — `mitos status .` names what is missing
>   and what to do next; fix only that. Finish with step 4 (the agent-block, so
>   the next agent inherits the setup), then re-run `mitos status .` and report
>   the result to the user.
>
> If Docker is unavailable and you cannot start Qdrant, say so and finish the
> rest anyway — recording works without it (embeddings queue for later); only
> semantic search waits.

---

## Prerequisites (once per machine)

- **Mitos installed** — the recommended global install is **pipx**: `pipx install mitos-adr` (isolated, on PATH). Or `pip install mitos-adr` into a venv; or `pip install -e .` from a clone if you're hacking on Mitos.
  - **Updating:** `pipx install --force git+https://github.com/dovahkiin-v/mitos` (use `--force`, not `pipx upgrade` — a git install can otherwise no-op). Mitos checks for a newer version at most once a day and prints a one-line nudge on stderr when one exists; silence it with `MITOS_NO_UPDATE_CHECK=1` in the shell environment.
- **Docker** — for Mitos's Qdrant.
- **A Google Gemini API key** — <https://aistudio.google.com/app/apikey> — required; it covers embeddings and synthesis.
- **An Anthropic API key** — <https://console.anthropic.com/settings/keys> — strongly recommended: it powers the LLM-judged layer (the `mitos check` conflict audit, the sync-time conflict notice, and prose import with `--llm-extract`). Mitos runs without it, but only as a basic record-and-search store — the conflict sensing that keeps a growing decision corpus honest is the part worth having.

### Start Mitos's Qdrant (once per machine, shared by all projects)
From a clone of this repo:
```bash
docker compose up -d        # → mitos-qdrant on :7333
```
Or without a clone (identical result; safe to re-run — if the container already
exists, `docker start mitos-qdrant` instead):
```bash
docker run -d --name mitos-qdrant --restart unless-stopped \
  -p 7333:6333 -p 7334:6334 \
  -v mitos-qdrant-storage:/qdrant/storage qdrant/qdrant
```
Mitos uses its **own** Qdrant on `:7333` — *not* the standard `:6333` — so it
never lands in another Qdrant you run for other work. It fails safe: if Qdrant
isn't up, `mitos record` still commits to the graph and queues embeddings; only
semantic search pauses until you start it.

### Register the MCP server (once per machine, serves every project)

**This is the recommended interface for any agent working in any of your
projects.** It gives the best AX: ambient `surface_decisions` / `query_decisions`
/ `record_decision` tools, **structured arguments** (no shell-quoting — multi-
sentence prose with apostrophes survives intact), and the tool names match the
ones the docs use. The CLI works without it (see the capability map below), but
the MCP is how an agent actually *lives* in the decision loop.

`mitos serve` is a stdio MCP server, and it is **not** per-project: every tool
call names the project it acts on, so one registration serves all of them.

**Claude Code:**
```bash
claude mcp add --scope user mitos -- mitos serve
```

Verified on 2026-07-30 against `claude` 2.1.220: with that one registration, a
non-interactive session loads the mitos tools from any working directory —
inside a project, in a parent workspace, in a directory that is not a mitos
workspace at all.

Two things to know before you run it:

- **It costs context in every session on the machine, whether or not you use
  mitos.** Measured on `0.15.1` (2026-08-06): seven tools carrying 3,296
  characters of JSON schema and 17,291 characters of description. That is the
  price of loading everywhere, and it is the trade this recipe makes
  deliberately. The figure is pinned to a version because the description half
  moves whenever a tool's prose is rewritten — `0.15.1` cut it by ~1,300
  characters — so read it as a measurement of that release, not of whatever
  build you are running.
- **`claude mcp add` has no `alwaysLoad` flag**, and whether that field does
  anything at user scope is unmeasured — so don't count on the tools being
  exempt from an MCP tool-search step.

**A per-project `.mcp.json` entry shadows this one, and its failure is
terminal.** Measured: a project-scope server named `mitos` wins by name over the
machine-wide one and does **not** fall back to it — if the project entry cannot
start, the session gets no mitos tools at all, and nothing in the session says
why. So if a project of yours still carries a `mitos` entry in its `.mcp.json`
from an earlier setup, **remove it** (or make it identical to the machine-wide
registration). `mitos status <project>` reports one when it finds it — but only
in that project's own directory; an entry at a parent launch root shadows the
same way and is not visible to that check.

**Environment for a long-lived server.** Mitos reads `MITOS_NO_UPDATE_CHECK`,
`HTTPS_PROXY`, `SSL_CERT_FILE` and friends from the real process environment
only — a project `.env` does not promote them. A user-scope server has no shell
to inherit from, so pass what it needs on the registration:
```bash
claude mcp add --scope user mitos -e HTTPS_PROXY=http://proxy:3128 -- mitos serve
```
API keys are the exception: those resolve per project through `.env` files (see
step 2), so they do not belong here.

**Other harnesses** (Cursor, Gemini CLI, custom): register `mitos serve` as an
MCP server however that client registers one. The working directory no longer
matters — every call names its project. The `.mitos/skill.md` generated by `init`
is the integration prompt — load it (or reference it from your agent
instructions) so the agent uses Mitos correctly.

---

## Per-project setup

Four steps, and none of them wires an MCP server — that is machine-wide now
(above). `init` is the introduction; the server was already there.

### 1. Initialize the workspace
From the project root:
```bash
mitos init
```
This creates `.mitos/` (graph + config + skill), `decisions.md`, `format-spec.md`,
and scaffolds a **gitignored `.env`** with empty key slots. It also **registers**
the project on this machine, so you can reach it by name from anywhere. It prints
what it registered:

```
Initialized Mitos workspace at /home/you/projects/harbor ✓
Registered as "harbor" → /home/you/projects/harbor
collection: mitos-harbor-4f2a91c3
```

The name defaults to the directory's basename; `mitos init --name <name>` picks
another, and `--force` repoints an existing name at this workspace. The
collection name is derived from the workspace's *path*, so a copied or cloned
workspace never writes into the original's vectors. If a repoint changed it,
`init` says so and points at `mitos reconcile` — it makes no network call, so
that is a statement of fact plus the named heal, not a diagnosis.

### 2. Add your key
Mitos resolves `GEMINI_API_KEY` with this precedence: **shell environment → the
target project's `.env` → a shared global `~/.config/mitos/.env`**. On a
single-user machine, set it **once for every project**:
```bash
mitos set-key --global <your-key>     # writes ~/.config/mitos/.env (mode 600)
```
To override it for one project, store a project-local key instead:
```bash
mitos set-key -p . <your-key>         # writes that project's ./.env (gitignored)
```
One key covers embeddings *and* synthesis. `mitos status .` shows which source it
found (`from global .env` / `from project .env` / `from environment`).

For the strongly-recommended Anthropic key (the conflict-audit layer), same
mechanism:
```bash
mitos set-key --global --name ANTHROPIC_API_KEY <your-key>
```

### 3. Verify
```bash
mitos status .      # expect: READY ✓
```

Optional, if the project already has a prose ADR log — import it once:
```bash
mitos import -p . --from prose --llm-extract path/to/DECISIONS.md   # needs ANTHROPIC_API_KEY
```
(The file path is read relative to your working directory; `-p` names the project
it lands in.)

### 4. Tell the next agent — point your project's agent files at this guide
So **any** agent that later opens this project knows Mitos is here and how to use
it, paste the canonical pointer block into whichever agent-instruction files the
project uses (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, …). Generate
it with:

```bash
mitos agent-block .          # prints the current block to paste
mitos agent-block . --check  # later: flags any pasted copy that's gone stale
```

The block is deliberately **thin** — four durable pointers: name this project on
every call (with the rule for computing the path, and the `project · collection ·
workspace` echo that makes a mis-aimed call visible), `mitos status .`, the
record-and-surface discipline, and `mitos check -p .`. Everything volatile stays
in the always-fresh surfaces (the MCP tool schemas + this guide) instead of the
pasted copy, which is what stops the copy from going stale.

It carries a hidden version marker (`<!-- mitos-agent-guide: vN -->`). On the rare
release that changes the block, `mitos status .` (and `mitos agent-block . --check`)
will notice an older pasted copy and nudge you to refresh it. That is what makes
Mitos-awareness travel with the project across agents and sessions *without*
silently rotting.

---

## Naming the project on every call

Mitos has no default target. Every verb that touches a workspace — read or write,
CLI or MCP — names the project it acts on, and a call that names none is refused
with a teaching error rather than guessed at. That is what lets one server and
one install serve every project, and what stops a decision about one project
landing in another's `decisions.md`.

**On the CLI**, the selector is `-p` / `--project`, accepted on either side of the
verb:

```bash
mitos check -p harbor            # a registered name
mitos check -p /home/you/harbor  # an absolute path
mitos check -p .                 # this directory (shorthand for its path)
```

`mitos status` and `mitos agent-block` also take the selector as a positional
(`mitos status .`, `mitos agent-block harbor`).

**On the MCP surface**, every tool takes a `project` argument in the same two
forms — a registered name, or an **absolute path**. Relative paths, including
`.`, are refused there: an MCP server has no meaningful working directory.

**Exempt** — these four take no selector, because they act on the machine or on
the current directory: `mitos init`, `mitos serve`, `mitos projects`, and
`mitos set-key --global`. `mitos status` is the fifth case and a different one:
with no selector it does not refuse, it answers about the machine.

**Discovery:** `mitos projects` lists what is registered on this machine (the MCP
twin is `list_projects`). The same list is included in every targeting error, so
a mistyped name comes back with did-you-mean matches rather than a dead end.

**Answers echo what they resolved** — `project · collection · workspace`, on the
same channel as the answer itself — so a mis-aimed call is visible in its own
output rather than three commits later. (`mitos status` names the project in its
header instead, `mitos agent-block` echoes on stderr so its stdout stays
paste-ready, and `mitos projects` carries none: it answers for the machine.)

**Which form to use where:**

| Form | Use it when | Wrong when |
|---|---|---|
| `-p .` (or an explicit path) | your working directory **is** the workspace root when the command runs — a git hook, a CI job, you at a terminal | anything launched from elsewhere: cron, a scheduler, a service |
| `-p <registered-name>` | machine-local state that never travels — a crontab, a shell alias, an interactive habit | anything committed to a repo: the same name means a different project on someone else's machine |
| `-p <absolute path>` | an artifact that must travel and cannot use `.` — the `project` argument in a committed agent template | never wrong, only verbose |

---

## CLI vs MCP — which surface does what

Mitos has two surfaces over **one** workspace. The **CLI** is the substrate
(setup, ops, inspection) and a complete fallback; the **MCP** is the recommended
decision interface for agents. They are not either/or — a typical agent setup
uses the MCP for the decision loop and the CLI for setup/ops.

| Task | CLI | MCP |
|------|-----|-----|
| Setup & ops — `init`, `status`, `set-key`, `sync`, `import`, `render` | ✅ (only here) | — |
| Inspect open questions — `open-questions` | ✅ (only here) | — |
| **Record a decision** (+ typed relations: `supersedes`/`amends`/`depends_on`/…) | `mitos record` | `record_decision` ★ |
| **Surface precedents (the recall loop)** | `mitos surface` | `surface_decisions` ★ |
| **Look up by slug/claim** | `mitos query` | `query_decisions` |
| **Enumerate the full set in a scope (exhaustive recall)** | `mitos list` | `list_decisions` |
| **Enumerate the scope vocabulary** | `mitos scopes` | `list_scopes` |
| **Show one node by slug** | `mitos show` | `show_node` |
| **List the registered projects** | `mitos projects` | `list_projects` |

★ **Prefer the MCP for recording and surfacing**: structured arguments mean no
shell-quoting (long prose with apostrophes/quotes survives), and the tool names
are exactly these. The CLI mirrors them as a fallback (and for humans).

**Name map** (the docs use the MCP names; the CLI accepts five of them as aliases
too):

| Documented / MCP name | CLI verb | CLI alias |
|-----------------------|----------|-----------|
| `record_decision`   | `mitos record`  | `mitos record_decision` |
| `surface_decisions` | `mitos surface` | `mitos surface_decisions` |
| `query_decisions`   | `mitos query`   | `mitos query_decisions` |
| `list_decisions`    | `mitos list`    | `mitos list_decisions` |
| `list_scopes`       | `mitos scopes`  | `mitos list_scopes` |
| `show_node`         | `mitos show`    | — |
| `list_projects`     | `mitos projects`| — |

So `mitos record_decision -p . …` works (an agent's first instinct), and for long
prose pass `--rejected-file -` / `--context-file -` to read from stdin instead of
fighting the shell.

---

## When to record a decision (the capture trigger)

Recall is easy to ask for; the judgement call is knowing **what is worth
recording** — and that is on the agent, not the tool (Mitos is a memory, not a
judge). Record a decision when it:

- sets a pattern future work must follow,
- forecloses a real alternative you weighed and rejected (capture **why** in
  `rejected_paths` — that is what stops the next agent re-proposing it),
- is structural or costly to reverse,
- reverses or supersedes a prior decision, or
- has cross-cutting blast radius.

Skip the local, easily-reversible, or already-settled choice. A quick self-test
at any fork: *would the next agent waste time re-deriving or re-litigating this?*
If yes, record it — and `surface_decisions` first when unsure.

**Link related decisions.** When a new decision relates to an existing one, pass
that one's exact slug to the matching relation argument so the graph stays
connected instead of accumulating silent tension: `supersedes` (replaces it),
`amends`, `narrows`, `depends_on`, `resolves`, `contradicts`, `cites`. A record
that is strongly similar (≥0.80) to an existing decision it does not reference
pauses instead of committing (`needs_review` — nothing is written) and surfaces
each neighbour's axiom, `rejected_paths`, `scope`, and modifier stamps. Judge
each neighbour from that payload: if the new decision
amends/supersedes/contradicts/cites one, re-record with that relation pointing
at its slug; if it is genuinely independent, re-record with
`acknowledge_neighbors=True`. An
`amended_by`/`narrowed_by` stamp means the neighbour has moved on — dereference
that slug before linking.

---

## Gating commits with `mitos check` (pre-commit / CI / cron)

`mitos check` audits the corpus for undeclared contradictions. Two modes: the
default **corpus sweep** audits every active decision; **`--staged`** gates just
the pending buffer of `decisions.md` before it lands. The exit contract is
scriptable — `0` clean or known-only, `1` a NEW contradiction, `2` degraded /
refused / could-not-run (a check that cannot certify never returns `0`/`1`).

Each recipe below names its project, and the three do not name it the same way.
The discriminator is *where the job runs*: a hook and a CI job run **in** the
checkout, so `.` is the workspace root; a cron job runs from `$HOME`, so it needs
a name.

### Pre-commit hook

Wire the staged gate into `.git/hooks/pre-commit` (or your hook manager). Guard
the divergence first — `mitos check --staged` reads the **working tree**, but git
commits the **index**; if they differ, the gate checks the wrong bytes:

```sh
# Fail loudly if decisions.md differs between the index and the working tree —
# `mitos check --staged` reads the WORKING TREE, git commits the INDEX; a divergence
# would gate the wrong bytes (a bad entry fixed-but-not-restaged slips the gate).
if ! git diff --quiet -- decisions.md; then
    echo "decisions.md has unstaged changes — stage or stash them before committing" >&2
    exit 1
fi
mitos check --staged -p .
```

`-p .` is correct here specifically because git runs hooks from the worktree top
level, so `.` **is** the workspace root at run time.

A commit that touches no pending decision entries short-circuits to exit `0` with
**zero LLM contact** — the hook is effectively free on the overwhelming majority
of commits.

### CI job

CI runs in its checkout, so the same form works:

```sh
mitos check --yes -p .
```

### Scheduled corpus sweep (cron)

**cron runs jobs from `$HOME`, not from your project** — so `-p .` there resolves
to a non-workspace and the sweep exits `2`, which in an exit-code branch is
indistinguishable from a real substrate outage. Name the project instead. A
crontab never leaves the machine holding the registry, so a registered name is
safe there in a way it is never safe in a committed file:

```sh
# Nightly corpus audit — non-interactive, so --yes authorizes the spend.
# Scope-bind where spend matters. cron starts in $HOME, so name the project.
mitos check --yes -p harbor                 # full corpus
mitos check --yes -p harbor --scope <tag>   # narrow to one scope
```

`--yes` is **required** on any non-interactive surface once the pending-batch
count exceeds the confirm threshold: without a TTY and without `--yes` above
threshold, the run refuses and exits `2` (it never prompts and never spends). In
a schedule, branch on the exit code: `0` clean, `1` a new finding (fail the job /
open an issue), `2` degraded (alert — the audit couldn't certify).

### Keys, Qdrant, and the secretless-CI consequence

`--staged` with pending entries needs a **Gemini key + Anthropic key + reachable
Qdrant** in the hook's environment. A secretless CI runner **cannot** pass the
gate on a buffer with pending entries — it exits `2` (fail-closed: a gate that
cannot check must not pass). This is correct behavior, not a bug.

Two sanctioned ways to live with it:

- Run the `--staged` gate **only in the keyed dev environment** (local
  pre-commit), where the keys and Qdrant are already present.
- In **secretless CI**, run the **corpus sweep** on a **keyed schedule**
  instead of a per-commit staged gate — the scheduled form above, with the
  project named.

`git commit --no-verify` is the deliberate human bypass — the intended escape
hatch when you need to commit past the gate on purpose.

### Latency

- **No pending decision entries** → exit `0`, zero LLM contact, effectively
  instant (the common commit).
- **N pending entries** → N sequential judgment calls at ~**5s P95 each**, so a
  5-entry buffer ≈ **25s** of hook time. That is expected and acceptable for a
  decisions-file commit — the first slow commit is the gate working, not a hang.

### First-run sticker shock

The first corpus sweep judges every fresh pair, so its preflight budget estimate
(~3K tokens/batch) can print a large one-time number. It does not recur: `--scope`
narrows the sweep, and verdict **reuse** means subsequent runs re-judge only
genuinely-changed pairs. The big number on run 1 is expected, not alarming.

---

## Cutover (migrating a prototype graph to V1a)

A graph created by an **older (pre-V1a) Mitos** uses a different node identity, so
it cannot be migrated in place — `mitos init`/`status` will refuse it and route
you here. Because the graph is a *derivative* projection of your markdown corpus,
the fix is to **rebuild it from `decisions.md` (+ `questions.md` + archives) and
atomically swap it in**. This is a **one-time, destructive** operation — distinct
from the safe, repeatable `mitos init` — so it has its own verb: **`mitos
cutover`**.

You only need this if `mitos status .` reports a *prototype graph*. A fresh or
already-V1a workspace never does (running `cutover` there is a harmless no-op).

The swap itself is crash-safe by construction: the old graph is backed up first
and the new graph lands in a single atomic rename, so a crash at any instant
leaves a workspace that simply re-runs clean — no manual restore. Run these steps
**in order**, from the project root:

1. **Quiesce the workspace.** Stop `mitos serve`, and finish or abort any
   in-flight sync, *before* you start. Mitos is single-writer; the cutover
   reads the old graph assuming no concurrent writer.
2. **Run the cutover from a build that has the verb.** Use the
   current/editable install (`./venv/bin/mitos cutover -p .`, or a freshly-built
   checkout). **Do not** `pipx install --force` the global install yet — a
   reinstalled Mitos refuses a prototype graph, so the global upgrade is the
   *last* step (#9), unblocked only once the cutover has landed.
3. **Review the verdict.** The command re-parses the corpus, replays it into a
   build-aside graph, and prints a completeness verdict. A **corpus defect**
   (malformed markdown) aborts with a one-line error — fix the markdown and
   re-run; it is never overridable. A **completeness shortfall** (an active
   decision present in the old graph but absent from the rebuild) refuses the
   swap and surfaces the offenders. Inspect them: re-run with **`--allow-drops`**
   only if the removal was intentional (your `decisions.md` is authoritative — a
   drop may be a deliberate purge). Use **`--yes`** to skip the interactive
   confirmation in automation, and **`--json`** for a machine-readable report.
4. **Wipe the stale Qdrant collection.** Its vectors are keyed on the old
   prototype ids, so they no longer match. Delete it — it auto-recreates on the
   next sync:
   ```bash
   curl -X DELETE <qdrant_url>/collections/<collection>
   ```
   (`mitos cutover` prints the exact command with your URL + collection filled in.)
5. **Re-embed the V1a active set.** Run `mitos sync -p .` (or `mitos sync -p .
   --embed-only`) to drain the embedding queue. Until it finishes there is a
   **bounded semantic-surface outage**: `surface`/`query` are degraded, but
   graph-only `mitos list -p .` works throughout.

   > **Healing a bare Qdrant wipe.** If you ever delete the Qdrant collection
   > *without* a `cutover`/`rebuild` (which re-seed the embedding queue), the
   > outbox is empty and a sync has nothing to drain. Run **`mitos reconcile
   > -p .`**: it diffs the active node set against Qdrant's actual points,
   > re-queues the missing active nodes, and drains them in one pass (idempotent;
   > re-embeds hit the cache, so no embedding-API spend when text is unchanged).
   > `mitos status .` flags this state as `⚠ vector index incomplete`.
6. **Restart `mitos serve`** if it was running.
7. **Verify.** `mitos status .` → expect `READY ✓`.
8. **Remove the backup.** Once you're satisfied, delete the
   `graph.sqlite.bak_<timestamp>` the cutover left in `.mitos/`.
9. **Upgrade the global install.** *Now* run `pipx install --force
   git+https://github.com/dovahkiin-v/mitos` — the operational carry is resolved,
   and the global Mitos will accept the V1a graph.

---

## When the corpus and the graph disagree

The graph is a derivative of `decisions.md`, but nothing stops the two drifting apart:
a hand-edit to an already-committed entry only reaches the graph when `mitos sync`
is authorized to apply it (it prints the field diff and asks; `--yes` applies
everything but an edge *deletion*), and an entry that leaves the corpus altogether
leaves its node behind with no source block. `mitos status .` reports both as an
informational rung — never a readiness blocker, because a corpus mid-edit is a normal
state, not breakage:

```
  ⚠ corpus and graph disagree in 4 place(s) — informational, not a readiness blocker.
      • 1 entry(s) whose commentary text differs (the graph serves the stale value to every read)
      • 3 node(s) have NO `### ` block in the corpus (2 active) — `mitos rebuild` cannot
        reconstruct them, so its completeness gate refuses.
```

**A node with no source block is the one that matters**, because it is what makes the
completeness shortfall in step 3 above unfixable: `rebuild` replays only what the
markdown holds, so the node is dropped, the gate refuses, and the repair path is
disabled on exactly the corpus that needs it. Do **not** reach for `--allow-drops` —
that discards the decisions rather than restoring them.

Instead, re-materialize the entries. The graph already holds every field the parser
reads, so this is a derivation, not re-authoring. **Name the project on all three** —
the middle command *writes into `decisions.md`*, and a mis-aimed run splices blocks
into another project's gold source:

```bash
mitos restore-source -p . --all-graph-only --dry-run   # review the blocks, writes nothing
mitos restore-source -p . --all-graph-only             # splice them into decisions.md
mitos rebuild -p . --json                              # expect gate_passed: true
```

It refuses rather than guesses. Every block is re-parsed before anything is written —
it must come back as exactly one entry, hashing to the same node id, with every
commentary field byte-identical — and the whole buffer is re-parsed after the splice to
prove no neighbouring entry was disturbed. Anything short of that is reported and
skipped, and the file is rolled back byte-for-byte.

Restored entries land in the **buffer**, not an archive: archives are
quarter-partitioned and a node's `created_at` is stamped at commit time, so dating a
restored entry would mean inventing a date in your gold source.

`--slug <name>` restores one node; `--json` emits a machine-readable report.

> **One more reason to restore rather than live with it.** `mitos rebuild` carries
> `confirmed_by`, `confirmed_at` and `created_at` forward from the graph it replaces —
> those three have no markdown home, so a rebuild replaying only the corpus would
> otherwise re-mint all three and leave you with a corpus claiming it was decided the
> day it was rebuilt. But carry-forward can only reach nodes the rebuild actually
> reconstructs, so **a node with no source block loses its provenance along with
> itself.** Restoring the blocks first is what lets the rebuild reach them.

> **Commentary that differs is a separate case.** `mitos sync` propagates a hand-edit
> to a committed entry: it prints the field diff and reconciles on confirmation, or
> unattended under `--yes`. The one thing `--yes` will not do unattended is *delete* an
> edge, so an entry whose relation line you removed needs
> `mitos sync -p <project> --reconcile-entry <slug>`, which applies that one named
> entry's whole reconcile and exits non-zero if it did not land. An entry that has
> already rotated into an archive is out of `sync`'s reach either way — `sync` reads
> the buffer alone, so that one's reconciler is still `mitos rebuild`. Scope drift is
> worth acting on first — it is a *findability* defect, so a wrong value hides the
> decision from every scope-filtered read.
