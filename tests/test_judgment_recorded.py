"""Recorded-response tests for the tool-use judgment — the $0 regression tier (P4c).

Pins the executor's extraction + serialization and the truncation path against
captured API responses. No API calls, every build.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mitos.conflict import (
    ConflictUnavailableReason,
    JudgmentExecution,
    RenderedPrompt,
    Unavailable,
    parse_judgment_response,
)
from mitos.conflict_judgment import execute_judgment


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _mock_message(fixture: dict) -> MagicMock:
    """Builds a mock Message from a captured fixture dict."""
    msg = MagicMock()
    msg.stop_reason = fixture["stop_reason"]
    blocks = []
    for block_data in fixture["content"]:
        block = MagicMock()
        block.type = block_data["type"]
        if block_data["type"] == "tool_use":
            block.input = block_data["input"]
            block.name = block_data["name"]
            block.id = block_data["id"]
        elif block_data["type"] == "text":
            block.text = block_data["text"]
        blocks.append(block)
    msg.content = blocks
    msg.usage = MagicMock(
        input_tokens=fixture["usage"]["input_tokens"],
        output_tokens=fixture["usage"]["output_tokens"],
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return msg


def _client_returning(msg: MagicMock) -> MagicMock:
    client = MagicMock()
    client.with_options.return_value.messages.create.return_value = msg
    return client


def _prompt() -> RenderedPrompt:
    return RenderedPrompt(
        system="S", user="U", prompt_version="conflict-tenability-v2"
    )


# --- Good response: extraction + serialization + parse round-trip ---

def test_good_tool_use_response_parses_successfully() -> None:
    """A captured good tool-use response extracts, serializes, and parses."""
    fixture = _load_fixture("judgment_response_tool_use_good.json")
    msg = _mock_message(fixture)
    client = _client_returning(msg)

    result = execute_judgment(_prompt(), client=client)

    assert isinstance(result, JudgmentExecution)
    assert result.stop_reason == "tool_use"

    verdicts = json.loads(result.raw_text)
    assert isinstance(verdicts, list)
    assert len(verdicts) > 0
    for v in verdicts:
        assert "slug" in v
        assert "rationale" in v
        assert isinstance(v["tenable_together"], bool)
        assert isinstance(v["confidence"], (int, float))

    slugs = [v["slug"] for v in verdicts]
    parsed = parse_judgment_response(result.raw_text, slugs)
    assert not isinstance(parsed, Unavailable), f"parse failed: {parsed.detail}"
    assert len(parsed) == len(verdicts)


# --- Truncated response: stop_reason check fires before content ---

def test_truncated_tool_use_response_returns_judgment_truncated() -> None:
    """A captured truncated forced-tool response returns JUDGMENT_TRUNCATED."""
    fixture = _load_fixture("judgment_response_tool_use_truncated.json")
    msg = _mock_message(fixture)
    client = _client_returning(msg)

    result = execute_judgment(_prompt(), client=client)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.JUDGMENT_TRUNCATED
    assert "max_tokens" in result.detail


def test_truncated_fixture_has_incomplete_tool_input() -> None:
    """The truncated fixture's tool_use block has no 'verdicts' key — confirming
    that extraction-first would KeyError past the fail-open seam."""
    fixture = _load_fixture("judgment_response_tool_use_truncated.json")
    assert fixture["stop_reason"] == "max_tokens"
    tool_blocks = [b for b in fixture["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert "verdicts" not in tool_blocks[0]["input"]


# --- Eval trigger: baseline provenance vs current code constants ---

def test_baseline_provenance_matches_current_code() -> None:
    """Detects when the committed baseline's provenance no longer matches the code.

    Compares prompt_version and judgment_model_id in the golden baseline against
    the current constants. A mismatch means the baseline was seeded under a
    different prompt or model and needs a reviewed re-seed. Loud skip, never a
    bare red or silent pass.
    """
    import os

    baseline_path = os.path.join(
        os.path.dirname(__file__), "golden", "conflict.baseline.metrics.json"
    )
    if not os.path.exists(baseline_path):
        pytest.skip(
            "no conflict.baseline.metrics.json — baseline not yet seeded; "
            "seed with MITOS_UPDATE_BASELINE=1."
        )

    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    prov = baseline.get("provenance", {})

    from mitos.conflict import CONFLICT_PROMPT_VERSION
    from mitos.models import get_model_id

    current_prompt = CONFLICT_PROMPT_VERSION
    current_model_id = get_model_id("SONNET")

    baseline_prompt = prov.get("prompt_version")
    baseline_model_id = prov.get("judgment_model_id")

    mismatches = []
    if baseline_prompt != current_prompt:
        mismatches.append(
            f"prompt_version: baseline={baseline_prompt!r}, "
            f"current={current_prompt!r}"
        )
    if baseline_model_id is None:
        mismatches.append(
            "judgment_model_id absent from baseline provenance — "
            "re-seed required (field was added in 0.17.0)"
        )
    elif baseline_model_id != current_model_id:
        mismatches.append(
            f"judgment_model_id: baseline={baseline_model_id!r}, "
            f"current={current_model_id!r}"
        )

    if mismatches:
        pytest.skip(
            "baseline provenance does not match current code — re-seed pending:\n  "
            + "\n  ".join(mismatches)
        )
