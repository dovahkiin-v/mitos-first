"""Packaging integration tests — proves the wheel ships what a real install needs.

These are deliberately heavy: they build a real wheel and install it into a
*fresh, throwaway* virtualenv (non-editable). That is the whole point — an
editable install reads ``format-spec.md`` from the source tree and would hide a
missing-``package-data`` bug. ``mitos/format-spec.md`` is read from the installed
package dir in two places (``mitos.cli.load_format_spec`` at ``mitos init`` time
and ``mitos.parser`` at *import* time), so a wheel missing it doesn't just break
``init`` — it breaks ``import mitos.parser`` outright. Only a non-editable install
exercises that real read path (vision V1-D7 / §6.2).

Both tests carry the ``packaging`` marker: they are slow (venv + network build
isolation pulls setuptools/wheel) and testmon will not select them on its own, so
CI (Phase 1b) runs them explicitly with ``-m packaging`` while the fast suite
skips them with ``-m 'not packaging'``.

Offline-tolerant: build isolation needs PyPI for setuptools/wheel and the wheel
install needs it for the five runtime deps. When PyPI is unreachable the tests
skip rather than fail, so offline dev stays clean. CI has network and always runs.
"""

import os
import socket
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import mitos as _mitos

# Single source of truth, read at test time — never hardcode "0.1.13" (the release
# ritual bumps it on every shipped change; a literal here would rot silently).
EXPECTED_VERSION = _mitos.__version__

# Project root = the source tree pip builds from (tests/ -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse the host's real pip cache so repeat runs (and warm-cache offline runs) are
# fast and don't re-download setuptools/wheel/deps every time. The hermetic
# autouse fixture points XDG_CACHE_HOME at a tmp dir; PIP_CACHE_DIR overrides that
# for pip specifically without touching the update-check hermeticity.
HOST_PIP_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "pip")

pytestmark = pytest.mark.packaging


def _pypi_reachable() -> bool:
    """Returns True if pypi.org:443 accepts a TCP connection within a short timeout."""
    try:
        with socket.create_connection(("pypi.org", 443), timeout=3):
            return True
    except OSError:
        return False


def _pip_env() -> dict:
    """Builds a subprocess env for pip with a warm, host-backed cache."""
    return {**os.environ, "PIP_CACHE_DIR": HOST_PIP_CACHE}


def _mitos_env(tmp_path) -> dict:
    """Builds a hermetic env for invoking the installed `mitos` binary.

    The autouse ``hermetic_mitos_env`` fixture only patches the *parent* process;
    a subprocess that the install test spawns must carry these keys explicitly so
    it never hits the network update path or pollutes the real ``~/.config/mitos``.
    """
    return {
        **os.environ,
        "MITOS_NO_UPDATE_CHECK": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
    }


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Builds the project wheel once (non-editable), shared by the packaging tests.

    Skips the whole packaging suite when PyPI is unreachable — build isolation
    needs it for setuptools/wheel. A build failure while *online* is a real
    failure (asserted), not a skip.
    """
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable — build isolation needs setuptools/wheel from PyPI")

    wheelhouse = tmp_path_factory.mktemp("wheelhouse")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         str(PROJECT_ROOT), "-w", str(wheelhouse)],
        env=_pip_env(), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(wheelhouse.glob("mitos_adr-*.whl"))
    assert len(wheels) == 1, f"expected exactly one mitos_adr wheel, got {wheels}"
    return wheels[0]


def test_format_spec_ships_in_wheel(built_wheel):
    """The wheel carries ``mitos/format-spec.md`` as package data (fast namelist check).

    The cheap complement to the full install test: a missing-``package-data``
    regression shows up here in milliseconds, before paying for a venv install.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
    assert "mitos/format-spec.md" in names, (
        f"format-spec.md missing from the wheel — package-data is broken. Wheel contents: {names}"
    )


def test_non_editable_install_real_read_path(built_wheel, tmp_path):
    """A fresh venv + non-editable install exercises the real package-dir read path.

    Proves the load-bearing chain: the installed wheel reports the single-source
    version (no static drift), ``import mitos.parser`` succeeds (its import-time
    ``FIELD_MAP`` reads the bundled spec), the installed ``format-spec.md`` matches
    the source byte-for-byte, and ``mitos init`` reads the spec from the *installed*
    package dir.
    """
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                   check=True, capture_output=True, text=True, timeout=120)
    venv_python = venv_dir / "bin" / "python"
    venv_mitos = venv_dir / "bin" / "mitos"

    # Non-editable install of the built wheel (deps resolved from PyPI).
    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", str(built_wheel)],
        env=_pip_env(), capture_output=True, text=True, timeout=600,
    )
    assert install.returncode == 0, f"wheel install failed:\n{install.stdout}\n{install.stderr}"
    assert venv_mitos.exists(), "console script `mitos` was not installed"

    # `mitos --version` reports the single-source version (no static pyproject drift).
    version = subprocess.run([str(venv_mitos), "--version"], env=_mitos_env(tmp_path),
                             capture_output=True, text=True, timeout=60)
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"mitos {EXPECTED_VERSION}", version.stdout

    # `import mitos.parser` runs FIELD_MAP = load_dynamic_field_map() at import time,
    # which reads the bundled spec — a wheel missing it fails here outright.
    parser_import = subprocess.run([str(venv_python), "-c", "import mitos.parser"],
                                   env=_mitos_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert parser_import.returncode == 0, (
        f"`import mitos.parser` failed in the installed venv — "
        f"format-spec.md likely missing from the wheel:\n{parser_import.stderr}"
    )

    # The spec ships in the installed package dir and matches the source exactly.
    installed_spec = subprocess.run(
        [str(venv_python), "-c",
         "import os, mitos; print(os.path.join(os.path.dirname(mitos.__file__), 'format-spec.md'))"],
        env=_mitos_env(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert installed_spec.returncode == 0, installed_spec.stderr
    installed_spec_path = Path(installed_spec.stdout.strip())
    assert installed_spec_path.is_file(), f"installed spec not found at {installed_spec_path}"
    assert installed_spec_path.read_bytes() == (PROJECT_ROOT / "mitos" / "format-spec.md").read_bytes(), (
        "installed format-spec.md differs from the package source"
    )

    # The registry leaf ships. `pyproject.toml` declares `packages = ["mitos"]` — an
    # explicit list, not `find_packages()` — so a *subpackage* would be silently
    # excluded from the wheel while the editable dev install resolved it happily.
    # `registry.py` is flat in `mitos/`, on the right side of that trap; this is the
    # net that catches a future move to the wrong side.
    registry_import = subprocess.run([str(venv_python), "-c", "import mitos.registry"],
                                     env=_mitos_env(tmp_path), capture_output=True,
                                     text=True, timeout=60)
    assert registry_import.returncode == 0, (
        f"`import mitos.registry` failed in the installed venv — the module is "
        f"missing from the wheel:\n{registry_import.stderr}"
    )

    # The routing leaf ships too, for the same reason and against the same trap.
    # It is the module every workspace-targeting call routes through once the
    # boundaries land, so a wheel missing it is a console script that dies on
    # import while the editable dev install stays perfectly green.
    routing_import = subprocess.run([str(venv_python), "-c", "import mitos.routing"],
                                    env=_mitos_env(tmp_path), capture_output=True,
                                    text=True, timeout=60)
    assert routing_import.returncode == 0, (
        f"`import mitos.routing` failed in the installed venv — the module is "
        f"missing from the wheel:\n{routing_import.stderr}"
    )

    # And the env leaf, same trap. It is imported by `mitos.config`, which every
    # verb constructs, so a wheel missing it is not a degraded feature — it is an
    # `ImportError` on the very first thing the console script does.
    env_import = subprocess.run([str(venv_python), "-c", "import mitos.env"],
                                env=_mitos_env(tmp_path), capture_output=True,
                                text=True, timeout=60)
    assert env_import.returncode == 0, (
        f"`import mitos.env` failed in the installed venv — the module is "
        f"missing from the wheel:\n{env_import.stderr}"
    )

    # And the overview leaf, same trap. A flat module is on the right side of it, so
    # this is cheap insurance rather than a known gap — but it is the module the
    # zero-arg `mitos status` will route to, i.e. the first thing a human runs when
    # something feels wrong, on a machine where nothing else has told them yet.
    overview_import = subprocess.run([str(venv_python), "-c", "import mitos.overview"],
                                     env=_mitos_env(tmp_path), capture_output=True,
                                     text=True, timeout=60)
    assert overview_import.returncode == 0, (
        f"`import mitos.overview` failed in the installed venv — the module is "
        f"missing from the wheel:\n{overview_import.stderr}"
    )

    # And the MCP server module — a DIFFERENT trap from the four above. It is not
    # about what the wheel contains but about what a fresh dependency resolve
    # *installs beside it*: `mcp` 2.0.0 renamed `mcp.server.fastmcp` (the surface
    # `mcp_server.py:9` imports at module scope) to `mcp.server.mcpserver`, so an
    # unceilinged `mcp>=…` floor resolves 2.x and `mitos serve` — the recommended
    # agent interface — dies with `No module named 'mcp.server.fastmcp'` on a real
    # install while the whole editable dev tree stays green. This row is the only
    # place in the suite where deps resolve from PyPI rather than from the venv
    # someone pinned by age, which is why the ceiling is gated here.
    mcp_server_import = subprocess.run([str(venv_python), "-c", "import mitos.mcp_server"],
                                       env=_mitos_env(tmp_path), capture_output=True,
                                       text=True, timeout=60)
    assert mcp_server_import.returncode == 0, (
        f"`import mitos.mcp_server` failed in the installed venv — `mitos serve` "
        f"is unstartable from a fresh install. Check the `mcp` version this "
        f"resolve picked against the ceiling in pyproject.toml:\n"
        f"{mcp_server_import.stderr}"
    )

    # `mitos init` reads the spec from the installed package dir and scaffolds a workspace.
    workspace = tmp_path / "proj"
    workspace.mkdir()
    init = subprocess.run([str(venv_mitos), "init"], cwd=workspace, env=_mitos_env(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert init.returncode == 0, f"`mitos init` failed:\n{init.stdout}\n{init.stderr}"
    assert (workspace / "format-spec.md").is_file(), "`mitos init` did not seed format-spec.md"

    # …and registers the project it just scaffolded, into the redirected config root
    # (`_mitos_env` carries XDG_CONFIG_HOME explicitly — the autouse parent-process
    # fixture does not reach a subprocess). The real-install path is the only place
    # the whole chain runs against a non-editable package.
    registry_file = tmp_path / "xdg_config" / "mitos" / "registry.toml"
    assert registry_file.is_file(), "`mitos init` did not write the project registry"
    assert str(workspace.resolve()) in registry_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static metadata guards — no build, no network, fast.
# ---------------------------------------------------------------------------

def _load_pyproject() -> dict:
    return tomllib.loads(PROJECT_ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_dynamic_from_dunder() -> None:
    """K1: version is sourced dynamically from mitos.__version__, never a static literal."""
    project = _load_pyproject()["project"]
    assert "version" not in project, (
        "version must be dynamic — a static literal silently kills the update nudge"
    )
    assert "version" in project.get("dynamic", [])
    dynamic = _load_pyproject()["tool"]["setuptools"]["dynamic"]
    assert dynamic["version"] == {"attr": "mitos.__version__"}
    assert _mitos.__version__ != "0.0.0"


def test_project_urls_point_at_personal_repo() -> None:
    """K2: URLs are the personal dovahkiin-v GitHub, never skyforge-sh."""
    urls = _load_pyproject()["project"]["urls"]
    assert urls, "expected a [project.urls] block"
    joined = " ".join(urls.values()).lower()
    assert "dovahkiin-v" in joined
    assert "skyforge" not in joined


def test_keywords_present() -> None:
    """K4: keywords aid discovery; must be non-empty."""
    keywords = _load_pyproject()["project"].get("keywords", [])
    assert keywords, "expected a non-empty keywords list"


def test_console_script_entry_point_intact() -> None:
    scripts = _load_pyproject()["project"]["scripts"]
    assert scripts.get("mitos") == "mitos.cli:main"


def test_requires_python_floor() -> None:
    assert _load_pyproject()["project"]["requires-python"] == ">=3.13"


def test_license_is_spdx_expression() -> None:
    """K3: PEP 639 SPDX string form; no deprecated License :: classifier."""
    project = _load_pyproject()["project"]
    assert project["license"] == "Apache-2.0"
    classifiers = project.get("classifiers", [])
    assert not any(c.startswith("License ::") for c in classifiers)


def test_description_present() -> None:
    assert _load_pyproject()["project"].get("description"), "expected a non-empty description"


def test_readme_wired() -> None:
    assert _load_pyproject()["project"].get("readme") == "README.md"


def test_development_status_matches_readme_badge() -> None:
    classifiers = _load_pyproject()["project"].get("classifiers", [])
    status = [c for c in classifiers if c.startswith("Development Status")]
    assert status == ["Development Status :: 3 - Alpha"]
