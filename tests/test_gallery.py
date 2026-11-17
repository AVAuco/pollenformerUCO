import json
from pathlib import Path

import numpy as np
import pytest

from pollenformer.gallery import IndexLoadError, l2_normalize_rows, load_gallery_index


def write_index(
    tmp_path: Path,
    feats: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: tuple[str, ...] = ("a", "b"),
    normalized: bool = True,
    paths: list[str] | None = None,
) -> tuple[Path, Path]:
    index_path, meta_path = tmp_path / "index.npz", tmp_path / "index.meta.json"
    np.savez(index_path, feats=feats, labels=labels)
    meta = {"class_names": list(class_names), "normalized": normalized}
    if paths is not None:
        meta["paths"] = paths
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return index_path, meta_path


def test_l2_normalize_rows_keeps_zero_rows() -> None:
    out = l2_normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


def test_load_round_trip(tmp_path: Path) -> None:
    feats = l2_normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    index, meta = write_index(
        tmp_path, feats, np.array([0, 1, 1]), paths=["x", "y", "z"]
    )
    gallery = load_gallery_index(index, meta, emb_dim=2)
    assert len(gallery) == 3
    assert gallery.class_names == ("a", "b")
    assert gallery.paths == ("x", "y", "z")
    assert gallery.feats.dtype == np.float32
    assert gallery.labels.dtype == np.int64


def test_load_normalizes_when_metadata_says_so(tmp_path: Path) -> None:
    index, meta = write_index(
        tmp_path, np.array([[3.0, 4.0]]), np.array([0]), normalized=False
    )
    gallery = load_gallery_index(index, meta, emb_dim=2)
    np.testing.assert_allclose(np.linalg.norm(gallery.feats, axis=1), [1.0], rtol=1e-6)


def test_load_warns_on_embedding_dim_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    index, meta = write_index(tmp_path, np.eye(2, dtype=np.float32), np.array([0, 1]))
    load_gallery_index(index, meta, emb_dim=128)
    assert "Embedding dim mismatch" in caplog.text


def test_missing_files_raise(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError, match="not found"):
        load_gallery_index(tmp_path / "nope.npz", tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("feats", "labels", "message"),
    [
        (np.zeros(4, dtype=np.float32), np.array([0]), "must be 2D"),
        (np.eye(2, dtype=np.float32), np.array([0]), "do not match"),
        (np.eye(2, dtype=np.float32), np.array([0, 5]), "outside"),
    ],
)
def test_inconsistent_index_raises(
    tmp_path: Path, feats: np.ndarray, labels: np.ndarray, message: str
) -> None:
    index, meta = write_index(tmp_path, feats, labels)
    with pytest.raises(IndexLoadError, match=message):
        load_gallery_index(index, meta, emb_dim=2)


def test_corrupt_files_raise(tmp_path: Path) -> None:
    index, meta = write_index(tmp_path, np.eye(2, dtype=np.float32), np.array([0, 1]))
    meta.write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexLoadError, match="Failed to read meta"):
        load_gallery_index(index, meta, emb_dim=2)
    index.write_bytes(b"not an npz")
    with pytest.raises(IndexLoadError, match="Failed to load index"):
        load_gallery_index(index, meta, emb_dim=2)
