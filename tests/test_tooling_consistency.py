"""Tool versions pinned in several files must agree.

Each tool is pinned where it runs: the uv lock file, pre-commit hooks, mise, the
Dockerfile and the CI workflows. These checks turn the "keep in sync" comments into a
failing test, so a partial bump (by hand or by Dependabot) is caught in CI.
"""

from __future__ import annotations

import re
import tomllib

from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest

from helpers import REPO_ROOT

PRE_COMMIT = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
DOCKERFILE = (REPO_ROOT / "demo" / "Dockerfile").read_text(encoding="utf-8")
LINT_WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "lint.yml").read_text(
    encoding="utf-8"
)
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
MISE = tomllib.loads((REPO_ROOT / "mise.toml").read_text(encoding="utf-8"))
LOCK = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
PYTHON_VERSION = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()


def hook_rev(repo: str) -> str:
    """Return the ``rev`` pinned for a pre-commit hook repository."""
    match = re.search(rf"repo: {re.escape(repo)}\s+rev: v?([\w.]+)", PRE_COMMIT)
    assert match, f"{repo} is not in .pre-commit-config.yaml"
    return match.group(1)


def locked(package: str) -> str:
    """Return the version of a package in uv.lock."""
    versions = [p["version"] for p in LOCK["package"] if p["name"] == package]
    assert versions, f"{package} is not in uv.lock"
    return versions[0]


def test_ruff_matches_between_pre_commit_and_the_lock_file() -> None:
    assert hook_rev("https://github.com/astral-sh/ruff-pre-commit") == locked("ruff")
    assert f"ruff=={locked('ruff')}" in PYPROJECT["dependency-groups"]["lint"]


def test_uv_is_the_same_everywhere() -> None:
    uv = MISE["tools"]["uv"]
    assert hook_rev("https://github.com/astral-sh/uv-pre-commit") == uv
    assert f"ghcr.io/astral-sh/uv:{uv} " in DOCKERFILE
    assert Version(uv) in SpecifierSet(PYPROJECT["tool"]["uv"]["required-version"])
    (build_requirement,) = PYPROJECT["build-system"]["requires"]
    assert Version(uv) in SpecifierSet(build_requirement.removeprefix("uv_build"))


def test_python_is_the_same_in_the_project_and_the_image() -> None:
    assert f"FROM python:{PYTHON_VERSION}-slim " in DOCKERFILE
    assert Version(PYTHON_VERSION) in SpecifierSet(
        PYPROJECT["project"]["requires-python"]
    )


@pytest.mark.parametrize("key", ["PRE_COMMIT_VERSION", "PYTHON_VERSION"])
def test_lint_workflow_matches_the_local_setup(key: str) -> None:
    match = re.search(rf'{key}: "([\d.]+)"', LINT_WORKFLOW)
    assert match, f"{key} is not set in lint.yml"
    if key == "PRE_COMMIT_VERSION":
        minimum = re.search(r'minimum_pre_commit_version: "([\d.]+)"', PRE_COMMIT)
        assert minimum
        assert match.group(1) == minimum.group(1)
    else:
        assert PYTHON_VERSION.startswith(match.group(1) + ".")
