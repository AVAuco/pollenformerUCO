"""Assemble the Hugging Face Space tree from the repository.

The Space builds ``demo/Dockerfile`` with the same relative paths as the repository, so
the tree is the image's build context plus the Dockerfile, its ignore file and the Space
card at the root:

    python scripts/stage_space.py --out build/space

Only files tracked by git are staged, so the Space holds exactly what the repository
holds: caches, local settings and other untracked files never reach it.
``.github/workflows/deploy-space.yml`` uploads the tree with ``huggingface/hub-sync``,
and ``mise run docker:demo:build`` builds it locally the way the Space does.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent

#: Single files copied as ``source -> destination`` (relative to the repository / tree
#: root).
FILES = {
    "demo/Dockerfile": "Dockerfile",
    "demo/Dockerfile.dockerignore": ".dockerignore",
    "demo/space-card.md": "README.md",
    "pyproject.toml": "pyproject.toml",
    "uv.lock": "uv.lock",
    "LICENSE": "LICENSE",
}
#: Directories whose tracked files are copied with their paths.
TREES = ("src", "demo")
#: Tracked files left out of those directories: the ones placed at the root above and
#: the documentation, which is not part of the image.
EXCLUDE = ("demo/Dockerfile", "demo/Dockerfile.dockerignore", "*.md")


def tracked_files(root: Path, *paths: str) -> list[str]:
    """Return the files git tracks under some paths of a repository.

    Args:
        root: Repository root.
        paths: Directories to list, relative to ``root``.

    Returns:
        POSIX paths relative to ``root``.

    Raises:
        SystemExit: If git is not installed.
    """
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git is required to list the files of the repository")
    # Git hooks export GIT_DIR and friends, which would point git at another repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Fixed arguments; only the repository's own paths are passed.
    result = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z", "--", *paths],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.split("\0") if name]


def stage(out: Path, root: Path = ROOT) -> list[Path]:
    """Write the Space tree into ``out``, replacing any previous content.

    Args:
        out: Destination directory.
        root: Repository root.

    Returns:
        The staged files, relative to ``out``, sorted.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for source, destination in FILES.items():
        shutil.copy2(root / source, out / destination)
    for name in tracked_files(root, *TREES):
        path = PurePosixPath(name)
        excluded = any(path.match(pattern) for pattern in EXCLUDE)
        # A tracked file deleted in the working tree is left out, as a commit would.
        if excluded or not (root / name).is_file():
            continue
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, out / path)
    return sorted(path.relative_to(out) for path in out.rglob("*") if path.is_file())


def main(argv: list[str] | None = None) -> None:
    """Stage the Space tree and list what it contains.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    args = parser.parse_args(argv)
    files = stage(args.out)
    print(f"Staged {len(files)} files in {args.out}:")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
