"""The k-NN gallery: training embeddings with their labels and class names."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from pollenformer.config import EMB_DIM

logger = logging.getLogger(__name__)

_NORM_EPS = 1e-12


class IndexLoadError(RuntimeError):
    """Raised when the k-NN index or its metadata cannot be loaded."""


@dataclass(frozen=True)
class GalleryIndex:
    """Reference embeddings searched by cosine k-NN.

    Attributes:
        feats: ``(N, D)`` float32 embeddings, L2-normalized row-wise.
        labels: ``(N,)`` int64 class ids, indexing ``class_names``.
        class_names: Name of each class id.
        paths: Optional source image path of each embedding.
    """

    feats: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    paths: tuple[str, ...] | None = None

    def __len__(self) -> int:
        """Return the number of reference embeddings."""
        return int(self.labels.shape[0])


def l2_normalize_rows(feats: np.ndarray) -> np.ndarray:
    """L2-normalize each row, leaving all-zero rows at zero.

    Args:
        feats: ``(N, D)`` array.

    Returns:
        A float32 copy with unit-norm rows.
    """
    feats = np.asarray(feats, dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, _NORM_EPS)


def load_gallery_index(
    index_path: Path, meta_path: Path, *, emb_dim: int = EMB_DIM
) -> GalleryIndex:
    """Load the gallery embeddings (``.npz``) and their metadata (``.json``).

    Args:
        index_path: ``.npz`` with ``feats`` ``(N, D)`` and ``labels`` ``(N,)``.
        meta_path: JSON with ``class_names``, and optionally ``paths`` and ``normalized``.
        emb_dim: Expected embedding size; a mismatch is logged, not raised.

    Returns:
        The loaded gallery, normalized when the metadata says it is not.

    Raises:
        IndexLoadError: If a file is missing or malformed, or labels and features disagree.
    """
    index_path, meta_path = Path(index_path), Path(meta_path)
    for path, kind in ((index_path, "Index npz"), (meta_path, "Meta json")):
        if not path.is_file():
            raise IndexLoadError(f"{kind} not found: {path}")

    try:
        with np.load(index_path, allow_pickle=False) as data:
            feats = np.asarray(data["feats"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise IndexLoadError(f"Failed to load index npz {index_path}: {exc}") from exc

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IndexLoadError(f"Failed to read meta {meta_path}: {exc}") from exc

    class_names = tuple(meta.get("class_names", ()))
    if feats.ndim != 2:
        raise IndexLoadError(f"feats must be 2D (N, D), got shape {feats.shape}")
    if labels.shape != (feats.shape[0],):
        raise IndexLoadError(f"labels {labels.shape} do not match feats {feats.shape}")
    if labels.size and (labels.min() < 0 or labels.max() >= len(class_names)):
        raise IndexLoadError(f"labels fall outside the {len(class_names)} class names")
    if feats.shape[1] != emb_dim:
        logger.warning("Embedding dim mismatch: file D=%d, expected %d", feats.shape[1], emb_dim)

    if not meta.get("normalized", False):
        feats = l2_normalize_rows(feats)

    paths = meta.get("paths")
    return GalleryIndex(
        feats=feats,
        labels=labels,
        class_names=class_names,
        paths=tuple(paths) if paths is not None else None,
    )
