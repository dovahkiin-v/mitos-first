"""The Conflict sensor's judgment executor — the one live Anthropic SONNET call (Phase 3b).

This is the **single module** in the Conflict pipeline that imports ``anthropic`` at
module scope, deliberately quarantined here so the Tier-1 leaf ``mitos.conflict`` stays
dependency-free (the dep-free subprocess guard in ``test_conflict_constants.py`` asserts
``anthropic`` never lands in ``sys.modules`` on ``import mitos.conflict``). The facade
(:func:`mitos.conflict.run_conflict_check`) receives the executor as an injected ``judge``
callable and names this module nowhere — the only real import edge is
``conflict_judgment → conflict`` (this module imports the boundary types + constants FROM
the leaf), never the reverse (plan D1).

**Tier 2 (logic).** Imports Tier-1 (`conflict`, `models`) + `anthropic`. Imported by
Tier-3 orchestration (5a's sync surface), never by the leaf.

The executor's job is narrow (plan D2): make the one batched tenability call via tool-use
(``tool_choice=tool``), cap it hard, measure it, and hand back a
:class:`~mitos.conflict.JudgmentExecution` (serialized verdict array + batch_id + usage +
elapsed) — or a typed :class:`~mitos.conflict.Unavailable` on a timeout, any Anthropic
error, or a truncated response (**fail-open**: it never raises past the seam, never blocks
a commit). The verdict array is serialized into ``raw_text`` as JSON, so both
``parse_judgment_response`` consumers are untouched.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Optional
from uuid import uuid4

import anthropic

from mitos.conflict import (
    CONFLICT_JUDGMENT_TEMPERATURE,
    CONFLICT_LLM_TIMEOUT_S,
    ConflictUnavailableReason,
    JudgmentExecution,
    RenderedPrompt,
    Unavailable,
)
from mitos.models import get_model_id

# The model family+tier alias (P19 — never a raw versioned id). Rides on every
# ``JudgmentExecution`` so 5b stamps each telemetry row's ``model_alias``.
_JUDGMENT_MODEL_ALIAS = "SONNET"

# Defence-in-depth budget for the tool-use response. The tool schema bounds shape;
# this bounds length. Measured: tool-use verdicts emit ~160 tokens/verdict (812 for 5),
# so 2000 accommodates the schema output with margin. The parser still rejects a
# truncated response — the tool_choice=tool constraint does not guarantee completeness
# when `stop_reason` is `max_tokens`.
_JUDGMENT_MAX_TOKENS = 2000

# The tool schema the judge is forced to call. Property order preserves the CONF-D3
# chain-of-thought lever: rationale BEFORE tenable_together, so the model reasons
# before it rules.
_VERDICT_TOOL = {
    "name": "record_verdicts",
    "description": (
        "Record the tenability verdicts for every candidate in this batch."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "rationale": {"type": "string"},
                        "tenable_together": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "slug",
                        "rationale",
                        "tenable_together",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["verdicts"],
    },
}


def execute_judgment(
    prompt: RenderedPrompt,
    *,
    client: "anthropic.Anthropic",
    timeout_s: float = CONFLICT_LLM_TIMEOUT_S,
    model_id: Optional[str] = None,
) -> "JudgmentExecution | Unavailable":
    """Runs one batched SONNET tenability call via tool-use; returns verdicts + metrics, or a typed failure.

    Forces the model to call the ``record_verdicts`` tool (``tool_choice=tool``), reads
    the verdict array from the tool_use block's ``input``, and serializes it into
    ``raw_text`` as a JSON array — so both ``parse_judgment_response`` consumers (the
    corpus path in ``check.py`` and the sync path in ``conflict.py``) are untouched.

    Retries are disabled via ``client.with_options(max_retries=0, timeout=timeout_s)``
    so ``CONFLICT_LLM_TIMEOUT_S`` is a **true** wall-clock ceiling. 5b's aggregate
    breaker owns the retry-vs-trip policy, not the SDK.

    Fail-open (plan D4): an :class:`~anthropic.APITimeoutError` and the broader
    :class:`~anthropic.AnthropicError` (rate-limit, 5xx, connection) both map to
    ``Unavailable(JUDGMENT_TIMEOUT)``. A ``stop_reason='max_tokens'`` maps to
    ``Unavailable(JUDGMENT)`` — checked BEFORE touching ``message.content``, since a
    truncated forced-tool response can carry an incomplete or absent ``tool_use``
    block. The executor never raises past this seam and never blocks the commit.

    Args:
        prompt: The rendered judgment prompt (from 3a's ``render_judgment_prompt``); its
            ``system`` is passed as the cache-anchored prefix, its ``user`` as the single
            user-message content.
        client: The injected Anthropic client (5a constructs the real one). Keyword-only.
        timeout_s: The hard per-call wall-clock cap in seconds (default
            ``CONFLICT_LLM_TIMEOUT_S``). Keyword-only.
        model_id: The resolved versioned id for ``_JUDGMENT_MODEL_ALIAS``, taken
            off the calling workspace's ``config.env`` by whichever orchestrator
            bound the client (2c). ``None`` falls back to the baseline for the
            alias — an override reaches this call only by being passed, because
            the model registry reads no process environment. Keyword-only.

    Returns:
        A :class:`~mitos.conflict.JudgmentExecution` (raw text + batch_id + usage + elapsed)
        on success, or an :class:`~mitos.conflict.Unavailable` with
        ``reason=JUDGMENT_TIMEOUT`` on a timeout or any Anthropic error.
    """
    # Mint the batch id up front (W8) — one per batched call, shared by every
    # ``conflict_checks`` row 5b writes for this batch. A plain unique ``str``.
    batch_id = uuid4().hex

    started = time.perf_counter()
    try:
        message = client.with_options(
            max_retries=0, timeout=timeout_s
        ).messages.create(
            model=(
                model_id if model_id is not None
                else get_model_id(_JUDGMENT_MODEL_ALIAS)
            ),
            max_tokens=_JUDGMENT_MAX_TOKENS,
            temperature=CONFLICT_JUDGMENT_TEMPERATURE,
            system=prompt.system,  # static cache-anchored prefix; cache_control OFF (RF-3).
            messages=[{"role": "user", "content": prompt.user}],
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdicts"},
        )
    except anthropic.APITimeoutError as exc:
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
            detail=f"judgment call timed out after {timeout_s}s: {exc}",
        )
    except anthropic.AnthropicError as exc:
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
            detail=f"anthropic error: {exc}",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Truncation check BEFORE touching content — a forced-tool response truncated at
    # max_tokens can carry an incomplete or absent tool_use block.
    if message.stop_reason == "max_tokens":
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TRUNCATED,
            detail=(
                f"judgment truncated: stop_reason='max_tokens' "
                f"(budget={_JUDGMENT_MAX_TOKENS})"
            ),
        )

    # Extract the tool_use block by type, not by position.
    tool_block = None
    for block in message.content:
        if block.type == "tool_use":
            tool_block = block
            break
    if tool_block is None:
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT,
            detail="no tool_use block in response",
        )

    # Serialize the verdicts array into raw_text so both parse_judgment_response
    # consumers (check.py corpus path AND conflict.py sync path) are untouched.
    raw_text = json.dumps(tool_block.input["verdicts"])

    usage = message.usage
    return JudgmentExecution(
        raw_text=raw_text,
        batch_id=batch_id,
        model_alias=_JUDGMENT_MODEL_ALIAS,
        token_input=getattr(usage, "input_tokens", 0) or 0,
        token_output=getattr(usage, "output_tokens", 0) or 0,
        token_cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        token_cache_creation=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        elapsed_ms=elapsed_ms,
        stop_reason=message.stop_reason,
    )


def make_judgment_executor(
    client: "anthropic.Anthropic",
    *,
    model_id: Optional[str] = None,
) -> "Callable[[RenderedPrompt], JudgmentExecution | Unavailable]":
    """Binds a client into the one-arg ``judge`` callable the facade expects (the 5a seam).

    5a calls this once with the constructed Anthropic client and passes the returned callable
    as ``run_conflict_check(..., judge=...)``. The closure keeps the facade's ``judge`` a
    clean one-arg function of a :class:`~mitos.conflict.RenderedPrompt`, so the facade never
    imports this module or touches the SDK (plan D1) — and stays trivially testable with a
    plain fake function (no SDK mock).

    Args:
        client: The Anthropic client to bind (5a constructs it, e.g. with ``max_retries=0``).
        model_id: The resolved versioned model id to bind alongside it (2c) — the
            id the calling workspace's ``config.env`` resolved for
            ``_JUDGMENT_MODEL_ALIAS``, or ``None`` for the baseline. It rides the
            closure rather than the facade, so the facade's ``judge`` stays a
            one-arg function of a ``RenderedPrompt``.

    Returns:
        A one-arg callable ``(RenderedPrompt) -> JudgmentExecution | Unavailable`` that drives
        :func:`execute_judgment` with the bound client.
    """

    def judge(prompt: "RenderedPrompt") -> "JudgmentExecution | Unavailable":
        return execute_judgment(prompt, client=client, model_id=model_id)

    return judge
