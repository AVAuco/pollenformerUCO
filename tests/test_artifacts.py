import hashlib
from pathlib import Path
import re

import pytest

from pollenformer import artifacts
from pollenformer.artifacts import Artifact, ArtifactError

PAYLOAD = b"pollen grains\n" * 64


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """A local "release" holding one asset, served through a file:// URL."""
    directory = tmp_path / "release"
    directory.mkdir()
    (directory / "asset.bin").write_bytes(PAYLOAD)
    return directory


@pytest.fixture
def asset() -> Artifact:
    return Artifact("asset.bin", hashlib.sha256(PAYLOAD).hexdigest(), "PF_TEST_ASSET")


def test_sha256sum(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(PAYLOAD)
    assert artifacts.sha256sum(path) == hashlib.sha256(PAYLOAD).hexdigest()


def test_fetch_downloads_and_verifies(
    tmp_path: Path, release_dir: Path, asset: Artifact
) -> None:
    cache = tmp_path / "cache"
    path = artifacts.fetch(asset, base_url=release_dir.as_uri(), dest_dir=cache)
    assert path == cache / "asset.bin"
    assert path.read_bytes() == PAYLOAD
    assert list(cache.glob("*.part")) == []


def test_fetch_reuses_a_valid_cached_copy(tmp_path: Path, asset: Artifact) -> None:
    (tmp_path / "asset.bin").write_bytes(PAYLOAD)
    # An unusable URL proves nothing is downloaded.
    path = artifacts.fetch(asset, base_url="https://invalid.invalid", dest_dir=tmp_path)
    assert path.read_bytes() == PAYLOAD


def test_fetch_rejects_a_checksum_mismatch(tmp_path: Path, release_dir: Path) -> None:
    wrong = Artifact("asset.bin", "0" * 64, "PF_TEST_ASSET")
    cache = tmp_path / "cache"
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        artifacts.fetch(wrong, base_url=release_dir.as_uri(), dest_dir=cache)
    assert list(cache.iterdir()) == []


def test_fetch_replaces_a_corrupt_cached_copy(
    tmp_path: Path, release_dir: Path, asset: Artifact
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "asset.bin").write_bytes(b"corrupt")
    path = artifacts.fetch(asset, base_url=release_dir.as_uri(), dest_dir=cache)
    assert path.read_bytes() == PAYLOAD


def test_fetch_reports_a_missing_asset(tmp_path: Path, asset: Artifact) -> None:
    with pytest.raises(ArtifactError, match="Could not download"):
        artifacts.fetch(
            asset, base_url=(tmp_path / "nowhere").as_uri(), dest_dir=tmp_path
        )


def test_fetch_refuses_plain_http(tmp_path: Path, asset: Artifact) -> None:
    with pytest.raises(ArtifactError, match="Refusing"):
        artifacts.fetch(asset, base_url="http://example.org/release", dest_dir=tmp_path)


def test_resolve_prefers_the_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, asset: Artifact
) -> None:
    local = tmp_path / "local.bin"
    local.write_bytes(b"anything")
    monkeypatch.setenv("PF_TEST_ASSET", str(local))
    assert artifacts.resolve(asset) == local


def test_resolve_rejects_a_missing_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, asset: Artifact
) -> None:
    monkeypatch.setenv("PF_TEST_ASSET", str(tmp_path / "missing.bin"))
    with pytest.raises(ArtifactError, match="missing file"):
        artifacts.resolve(asset)


def test_resolve_downloads_without_override(
    monkeypatch: pytest.MonkeyPatch, release_dir: Path, asset: Artifact
) -> None:
    monkeypatch.delenv("PF_TEST_ASSET", raising=False)
    monkeypatch.setenv("POLLENFORMER_BASE_URL", release_dir.as_uri())
    path = artifacts.resolve(asset)
    assert path.parent == artifacts.cache_dir()
    assert path.read_bytes() == PAYLOAD


def test_cache_dir_is_versioned_under_pollenformer_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("POLLENFORMER_HOME", str(tmp_path))
    assert artifacts.cache_dir() == tmp_path / artifacts.ARTIFACTS_RELEASE
    monkeypatch.delenv("THUMBS_DIR", raising=False)
    assert (
        artifacts.thumbs_dir()
        == tmp_path / artifacts.ARTIFACTS_RELEASE / "gallery_thumbs"
    )


def test_manifest_is_well_formed() -> None:
    filenames = [a.filename for a in artifacts.ALL_ARTIFACTS]
    assert len(set(filenames)) == len(filenames)
    assert all(re.fullmatch(r"[0-9a-f]{64}", a.sha256) for a in artifacts.ALL_ARTIFACTS)
    assert artifacts.DEFAULT_BASE_URL.endswith(
        f"/download/{artifacts.ARTIFACTS_RELEASE}"
    )
