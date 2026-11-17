"""The Hugging Face Space tree: what deploy-space.yml uploads and the Space builds."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from helpers import REPO_ROOT
from pollenformer import artifacts
import stage_space

ROOT_FILES = {"Dockerfile", ".dockerignore", "README.md", "pyproject.toml", "uv.lock"}


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("space")
    stage_space.stage(out)
    return out


def file_names(tree: Path) -> set[str]:
    return {p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()}


def test_tree_is_the_build_context_with_the_space_files_at_the_root(tree: Path) -> None:
    names = file_names(tree)
    assert ROOT_FILES | {"LICENSE"} <= names
    assert {"demo/main.py", "demo/.streamlit/config.toml"} <= names
    assert set(stage_space.tracked_files(REPO_ROOT, "src")) <= names


def test_tree_leaves_out_the_docs_and_the_files_moved_to_the_root(tree: Path) -> None:
    names = file_names(tree)
    assert not names & {"demo/Dockerfile", "demo/Dockerfile.dockerignore"}
    assert not [name for name in names if name.endswith(".md") and "/" in name]
    assert not [name for name in names if "__pycache__" in name]


def test_every_file_reaches_the_space(tree: Path) -> None:
    # hub-sync always skips root `.git*` files and any `.git/` or `.github/` folder.
    names = file_names(tree)
    assert not [n for n in names if n.startswith(".git") or "/.git" in f"/{n}"]


def test_dockerfile_copies_only_what_the_tree_holds(tree: Path) -> None:
    dockerfile = (tree / "Dockerfile").read_text(encoding="utf-8")
    copies = re.findall(r"^COPY (?!--from)(.+) \S+$", dockerfile, re.MULTILINE)
    sources = {source for line in copies for source in line.split()}
    assert sources, "the Dockerfile copies nothing from the build context"
    assert all((tree / source).exists() for source in sources), sources


def test_space_card_configures_the_docker_space_and_links_the_model(
    tree: Path,
) -> None:
    card = (tree / "README.md").read_text(encoding="utf-8")
    front_matter = card.split("---")[1]
    assert re.search(r"^sdk: docker$", front_matter, re.MULTILINE)
    port = re.search(r"^app_port: (\d+)$", front_matter, re.MULTILINE)
    assert port, "the Space card sets no app_port"
    assert f"EXPOSE {port.group(1)}" in (tree / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert f"  - {artifacts.ARTIFACTS_REPO_ID}" in front_matter.splitlines()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_only_tracked_files_are_staged(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sources = [
        *stage_space.FILES,
        "src/pkg/__init__.py",
        "demo/main.py",
        "demo/gone.py",
    ]
    for name in sources:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(name, encoding="utf-8")
    git = [shutil.which("git") or "git", "-C", str(repo)]
    # Without the GIT_* variables a git hook exports, git cannot touch this repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for args in (["init", "--quiet"], ["add", "--all"]):
        subprocess.run([*git, *args], check=True, env=env)  # noqa: S603
    # Untracked local files, and a tracked file deleted in the working tree.
    (repo / "demo/.streamlit").mkdir()
    (repo / "demo/.streamlit/secrets.toml").write_text("token", encoding="utf-8")
    (repo / "src/pkg/__pycache__").mkdir()
    (repo / "src/pkg/__pycache__/pkg.pyc").write_bytes(b"")
    (repo / "demo/gone.py").unlink()
    out = tmp_path / "space"
    (out / "stale").mkdir(parents=True)
    (out / "stale/file.txt").write_text("previous deploy", encoding="utf-8")

    staged = {p.as_posix() for p in stage_space.stage(out, root=repo)}

    assert staged == set(stage_space.FILES.values()) | {
        "src/pkg/__init__.py",
        "demo/main.py",
    }
    assert (out / "README.md").read_text(encoding="utf-8") == "demo/space-card.md"


def test_main_lists_the_staged_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stage_space.main(["--out", str(tmp_path / "space")])
    output = capsys.readouterr().out
    assert output.startswith("Staged ")
    assert "  Dockerfile\n" in output
