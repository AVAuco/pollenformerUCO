"""Versioned model artifacts: the checkpoint and the k-NN gallery published with each release.

Files are downloaded once into a local cache and checked against their SHA-256 before use, so a
fresh clone can run a real prediction without training anything. An environment variable per
artifact (``MODEL_PATH``, ``INDEX_PATH``, ...) points to an existing local copy instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from pollenformer.config import env_path

logger = logging.getLogger(__name__)

#: Release whose assets hold the artifacts below. Bump it together with the checksums.
ARTIFACTS_RELEASE = "v0.1.0"
DEFAULT_BASE_URL = (
    f"https://github.com/AVAuco/pollenformerUCO/releases/download/{ARTIFACTS_RELEASE}"
)
_ALLOWED_SCHEMES = ("https://", "file://")
_CHUNK_BYTES = 1 << 20


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, cannot be downloaded or fails verification."""


@dataclass(frozen=True)
class Artifact:
    """A release asset and the checksum its local copy must match.

    Attributes:
        filename: Asset name in the release and in the local cache.
        sha256: Expected hex digest of the file.
        env_var: Environment variable that overrides the download with a local path.
    """

    filename: str
    sha256: str
    env_var: str


MODEL = Artifact(
    "hybrid_128_dino_classifier.pth",
    "ac08ca25dc201034d29d2286b000b1b8c283f1a685fac9b7e65c0ecdb288dbc0",
    "MODEL_PATH",
)
GALLERY_INDEX = Artifact(
    "hyb128_index.npz",
    "54f2130db1b6e75bfb4860013104443a2856adf7d327a49f713c1d8daedb983f",
    "INDEX_PATH",
)
GALLERY_META = Artifact(
    "hyb128_index.meta.json",
    "47edf64838afc2cd0d9a420b6c080bba4ec61f07c6684544367d4e8e4380e43a",
    "META_PATH",
)
GALLERY_UMAP = Artifact(
    "umap_xy.npz",
    "a04e694e0d614d31eb8696bfd0dff6aabe7a30849dcdcb56192a7e91f055ee46",
    "UMAP_XY_CACHE",
)
ALL_ARTIFACTS = (MODEL, GALLERY_INDEX, GALLERY_META, GALLERY_UMAP)


def cache_dir() -> Path:
    """Return the directory that holds downloaded artifacts for the current release.

    Returns:
        ``$POLLENFORMER_HOME/<release>``, defaulting to ``~/.cache/pollenformer/<release>``.
    """
    default_root = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "pollenformer"
    return env_path("POLLENFORMER_HOME", default_root) / ARTIFACTS_RELEASE


def thumbs_dir() -> Path:
    """Return the optional directory of gallery thumbnails named ``{index}.jpg``.

    Returns:
        ``THUMBS_DIR`` if set, else ``gallery_thumbs`` inside :func:`cache_dir`.
    """
    return env_path("THUMBS_DIR", cache_dir() / "gallery_thumbs")


def sha256sum(path: Path) -> str:
    """Compute the SHA-256 of a file without loading it whole into memory.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(artifact: Artifact, *, base_url: str | None = None, dest_dir: Path | None = None) -> Path:
    """Return a verified local copy of an artifact, downloading it when needed.

    Args:
        artifact: The artifact to fetch.
        base_url: Location of the release assets. Defaults to ``POLLENFORMER_BASE_URL`` or
            :data:`DEFAULT_BASE_URL`. Only ``https://`` and ``file://`` are accepted.
        dest_dir: Cache directory. Defaults to :func:`cache_dir`.

    Returns:
        Path to the verified file.

    Raises:
        ArtifactError: If the URL scheme is not allowed, the download fails or the checksum
            does not match.
    """
    target_dir = Path(dest_dir) if dest_dir is not None else cache_dir()
    target = target_dir / artifact.filename
    if target.is_file() and sha256sum(target) == artifact.sha256:
        return target

    base = (base_url or os.getenv("POLLENFORMER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}/{artifact.filename}"
    if not url.startswith(_ALLOWED_SCHEMES):
        raise ArtifactError(f"Refusing to download from {url}: use https:// or file://")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", artifact.filename, url)
    with tempfile.NamedTemporaryFile(dir=target_dir, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            # The scheme is restricted to https/file just above.
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                shutil.copyfileobj(response, tmp, _CHUNK_BYTES)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise ArtifactError(f"Could not download {url}: {exc}") from exc

    digest = sha256sum(tmp_path)
    if digest != artifact.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ArtifactError(
            f"Checksum mismatch for {artifact.filename}: expected {artifact.sha256}, got {digest}"
        )
    tmp_path.replace(target)
    return target


def resolve(artifact: Artifact) -> Path:
    """Return the local path of an artifact.

    Args:
        artifact: The artifact to locate.

    Returns:
        The path in the artifact's environment variable if set, else the verified download.

    Raises:
        ArtifactError: If the override points to a missing file, or the download fails.
    """
    override = os.getenv(artifact.env_var)
    if override and override.strip():
        path = Path(override)
        if not path.is_file():
            raise ArtifactError(f"{artifact.env_var} points to a missing file: {path}")
        return path
    return fetch(artifact)
