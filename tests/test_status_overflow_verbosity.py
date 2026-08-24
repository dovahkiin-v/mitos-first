"""`mitos status`'s size-ceiling breakdown, and the flag that unfolds it.

The AX item behind these rows: the ceiling is a corpus-*growth* fact, so once a
corpus matures every single run of the report carries it. The mitos-pub corpus
prints eight over-ceiling files, and unfolding each into its five largest decisions
put forty lines between the caller and the readiness verdict the report exists to
give — on the one command a routine or a cron read runs unattended.

So the split these pin: the per-file **size line** is the signal and stays on the
default path; the per-file **slug breakdown** is what you want once, when you sit
down to re-scope, and `-v` is where you say so. Two properties guard the seam —
the withheld detail is announced rather than silently dropped (an unmentioned flag
is a capability the surface has and will not admit to, which is the same defect
class this whole ledger is made of), and the `--json` encoding is **not** gated,
because a machine payload that changes shape with a text-verbosity flag is one no
consumer can rely on.
"""

import json
import os
from typing import Any, Dict, List

import pytest
from unittest.mock import patch

from conftest import make_workspace
from mitos import cli
from mitos.cli import _print_overflow_detail, main
from mitos.store import GraphStore


def _overflows(n_files: int = 2, n_top: int = 3) -> List[Dict[str, Any]]:
    """Builds overflow records in `overflow_report`'s shape (largest file first)."""
    return [
        {
            "name": f"scope{i}.md",
            "chars": 90_000 - (i * 10_000),
            "est_tokens": 22_000 - (i * 2_000),
            "threshold_chars": 20_000,
            "top_decisions": [
                {"slug": f"file{i}-decision-{j}", "chars": 5_000 - (j * 100)}
                for j in range(n_top)
            ],
        }
        for i in range(n_files)
    ]


class TestTheDefaultPath:
    """What an unattended read gets: the count, the files, and no slug wall."""

    def test_the_per_file_size_line_survives(self, capsys) -> None:
        """The signal the gate must not take: which file, and how far over."""
        _print_overflow_detail(_overflows())
        out = capsys.readouterr().out
        assert "2 rendered axiom files over the size ceiling" in out
        for name in ("scope0.md", "scope1.md"):
            assert name in out
        assert "90,000 chars" in out and "ceiling 20,000" in out

    def test_the_slug_breakdown_is_withheld(self, capsys) -> None:
        """The forty lines. No slug, and no header promising them."""
        _print_overflow_detail(_overflows())
        out = capsys.readouterr().out
        assert "largest decisions:" not in out
        assert "file0-decision-0" not in out

    def test_the_withholding_is_announced(self, capsys) -> None:
        """Silence about a flag that exists is the defect this ledger is made of."""
        _print_overflow_detail(_overflows())
        assert "`-v`" in capsys.readouterr().out

    def test_the_closing_advice_does_not_point_at_a_list_it_withheld(
            self, capsys) -> None:
        """The shipped wording said "re-scope the largest decisions **above**".

        Gating the breakdown makes that word false on the default path — a line
        describing a render that no longer happened. Pinned because it is the
        exact way a gate quietly turns working text into a lie.
        """
        _print_overflow_detail(_overflows())
        out = capsys.readouterr().out
        assert "re-scope the largest" in out
        assert "decisions above" not in out


class TestTheVerbosePath:
    """What `-v` adds, and the one thing it must stop saying."""

    def test_every_file_unfolds_to_its_slugs(self, capsys) -> None:
        _print_overflow_detail(_overflows(), verbose=True)
        out = capsys.readouterr().out
        assert out.count("largest decisions:") == 2
        for i in range(2):
            for j in range(3):
                assert f"file{i}-decision-{j}" in out

    def test_the_flag_hint_is_gone(self, capsys) -> None:
        """Re-offering the flag whose output is already on screen reads as a
        failed render, not as help."""
        _print_overflow_detail(_overflows(), verbose=True)
        assert "`-v`" not in capsys.readouterr().out


class TestTheEdges:
    """Shapes that must not fall through the new branch."""

    def test_a_file_with_no_top_decisions_announces_nothing(self, capsys) -> None:
        """`top_decisions` can be absent or empty; there is then no detail being
        withheld, so offering the flag would promise output `-v` cannot produce."""
        _print_overflow_detail(_overflows(n_files=1, n_top=0))
        out = capsys.readouterr().out
        assert "scope0.md" in out
        assert "`-v`" not in out

    def test_one_file_is_singular_on_both_verbosities(self, capsys) -> None:
        for verbose in (False, True):
            _print_overflow_detail(_overflows(n_files=1), verbose=verbose)
            assert "1 rendered axiom file over" in capsys.readouterr().out


def _workspace_with_a_graph(tmp_path) -> str:
    """A workspace whose graph EXISTS.

    `make_workspace` deliberately ships none — a workspace is valid without one —
    but `cmd_status` reads the overflow report only inside the branch guarded on
    `os.path.exists(config.db_path)`, so a graphless fixture never reaches the call
    these rows patch and every assertion would pass against an empty list.
    """
    root = make_workspace(tmp_path / "ws")
    GraphStore(os.path.join(root, ".mitos", "graph.sqlite"))
    return root


class TestTheWiring:
    """That the flag reaches the printer, and that the payload never feels it."""

    @pytest.mark.parametrize("argv_tail, expect_verbose", [
        ([], False),
        (["-v"], True),
        (["--verbose"], True),
    ])
    def test_the_flag_reaches_the_printer(
            self, tmp_path, monkeypatch, argv_tail, expect_verbose) -> None:
        root = _workspace_with_a_graph(tmp_path)
        seen: Dict[str, Any] = {}

        def _spy(overflows, *, verbose=False):
            seen["verbose"] = verbose

        monkeypatch.setattr(cli, "overflow_report", lambda store: _overflows())
        monkeypatch.setattr(cli, "_print_overflow_detail", _spy)
        monkeypatch.setattr("sys.argv", ["mitos", "status", root] + argv_tail)
        with pytest.raises(SystemExit):
            main()
        assert seen.get("verbose") is expect_verbose

    @pytest.mark.parametrize("argv_tail", [[], ["-v"]])
    def test_the_json_payload_carries_every_slug_on_both_verbosities(
            self, tmp_path, monkeypatch, capsys, argv_tail) -> None:
        """`scope_overflow` is a machine field. It does not move with a text flag."""
        root = _workspace_with_a_graph(tmp_path)
        monkeypatch.setattr(cli, "overflow_report", lambda store: _overflows())
        monkeypatch.setattr("sys.argv",
                            ["mitos", "status", root, "--json"] + argv_tail)
        with pytest.raises(SystemExit):
            main()
        payload = json.loads(capsys.readouterr().out)
        recorded = payload["scope_overflow"]
        assert len(recorded) == 2
        assert [d["slug"] for d in recorded[0]["top_decisions"]] == [
            "file0-decision-0", "file0-decision-1", "file0-decision-2"]
