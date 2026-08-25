"""Static packaging + MCP-Registry metadata guards — no build, no network, fast.

Split out of ``test_packaging.py`` deliberately. That module sets
``pytestmark = pytest.mark.packaging`` at module scope, which the fast suite
skips with ``-m 'not packaging'`` — so these guards, despite being static and
cheap, never ran outside the slow tier. A metadata guard that only executes in a
tier you routinely skip is not guarding anything, and every check here protects
something that fails *after* a release is cut and its version permanently burned.
They live here, unmarked, so they run on every invocation.

Two families:
  * pyproject invariants (K1-K4) — dynamic version wiring, personal repo URLs,
    keywords, entry point, Python floor, SPDX license form.
  * MCP Registry invariants — ``server.json`` is a third home for the release
    version alongside ``mitos.__version__``, and the README ``mcp-name:`` marker
    is a cross-file join key whose consumer is the *rendered* PyPI description.
    Measured 2026-08-25: the registry rejects a publish without that marker with
    HTTP 400 "ownership validation failed".
"""

import json
import re
import tomllib
from pathlib import Path

import mitos as _mitos

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SERVER_JSON = PROJECT_ROOT / "server.json"
_README = PROJECT_ROOT / "README.md"


def _load_server_json() -> dict:
    return json.loads(_SERVER_JSON.read_text(encoding="utf-8"))



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


# ── MCP Registry metadata ────────────────────────────────────────────────────
# ``server.json`` is a third home for the release version, alongside
# ``mitos.__version__`` (the source of truth) and the dynamic pyproject read, so
# it gets the same drift guard as the rest of this section. The README marker is
# a cross-file join key whose consumer is a *rendered artifact*: the registry
# reads it out of the PyPI long_description. Both fail quietly and remotely — a
# missed bump points the registry at a PyPI version that does not exist, and a
# renamed marker is rejected server-side (measured 2026-08-25: HTTP 400,
# "ownership validation failed") only after the release is cut and burned.


def test_server_json_version_tracks_dunder() -> None:
    server = _load_server_json()
    assert server["version"] == _mitos.__version__
    assert server["packages"][0]["version"] == _mitos.__version__


def test_readme_carries_the_registry_ownership_marker() -> None:
    found = re.search(
        r"mcp-name:\s*(?P<name>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)",
        _README.read_text(encoding="utf-8"),
    )
    assert found is not None, (
        "README.md is missing the `mcp-name:` marker — the MCP Registry proves "
        "PyPI ownership by finding it in the package description, and rejects "
        "the publish with HTTP 400 without it"
    )
    assert found.group("name") == _load_server_json()["name"]


def test_server_json_package_identifier_is_the_dist_name() -> None:
    # The registry resolves this against pypi.org, so it must be the
    # distribution name (`mitos-adr`) — NOT the import package (`mitos`) and NOT
    # the repo name (also `mitos`). Three names for one project; the wrong one
    # here fails only at publish time.
    package = _load_server_json()["packages"][0]
    assert package["registryType"] == "pypi"
    assert package["identifier"] == "mitos-adr"


def test_server_name_is_in_the_github_namespace() -> None:
    # GitHub-authenticated publishing permits only names under
    # io.github.<username>/ — anything else is a permissions error at publish.
    assert _load_server_json()["name"].startswith("io.github.dovahkiin-v/")
