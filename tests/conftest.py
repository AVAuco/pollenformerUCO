"""Shared fixtures: a tiny DINOv2-like model, a synthetic gallery and test images."""

from __future__ import annotations

from collections.abc import Iterator
import os

import numpy as np
from PIL import Image
import pytest
import torch

from helpers import CLASS_NAMES, DIM, EMB_DIM, FakeDinoBackbone, encode
from pollenformer import pipeline
from pollenformer.gallery import GalleryIndex, l2_normalize_rows
from pollenformer.model import DinoClassifierWithEmbeddings


@pytest.fixture
def tiny_model() -> DinoClassifierWithEmbeddings:
    torch.manual_seed(0)
    model = DinoClassifierWithEmbeddings(
        FakeDinoBackbone(), num_classes=len(CLASS_NAMES), feat_dim=DIM, emb_dim=EMB_DIM
    )
    return model.eval()


@pytest.fixture
def gallery() -> GalleryIndex:
    """Nine unit embeddings, three per class, clustered around one axis per class."""
    rng = np.random.default_rng(0)
    centers = np.eye(EMB_DIM, dtype=np.float32)[: len(CLASS_NAMES)]
    feats = np.repeat(centers, 3, axis=0) + 0.05 * rng.standard_normal((9, EMB_DIM))
    return GalleryIndex(
        feats=l2_normalize_rows(feats),
        labels=np.repeat(np.arange(len(CLASS_NAMES)), 3).astype(np.int64),
        class_names=CLASS_NAMES,
    )


@pytest.fixture
def rgb_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 256, (40, 60, 3), dtype=np.uint8), mode="RGB"
    )


@pytest.fixture
def png_bytes(rgb_image: Image.Image) -> bytes:
    return encode(rgb_image)


@pytest.fixture(autouse=True)
def _isolate_pipeline_cache() -> Iterator[None]:
    """Keep the lazily loaded model and gallery from leaking between tests."""
    pipeline.get_model.cache_clear()
    pipeline.get_gallery.cache_clear()
    yield
    pipeline.get_model.cache_clear()
    pipeline.get_gallery.cache_clear()


@pytest.fixture(autouse=True)
def _no_artifact_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
):
    """Point the artifact cache at an empty temp dir and drop local overrides.

    Tests marked ``artifacts`` keep the real environment, so they can use local copies.
    """
    if os.getenv("POLLENFORMER_ARTIFACTS") == "1":
        return
    monkeypatch.setenv(
        "POLLENFORMER_HOME", str(tmp_path_factory.mktemp("pollenformer-home"))
    )
    for name in (
        "MODEL_PATH",
        "INDEX_PATH",
        "META_PATH",
        "UMAP_XY_CACHE",
        "THUMBS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests that need the released artifacts unless POLLENFORMER_ARTIFACTS=1."""
    if os.getenv("POLLENFORMER_ARTIFACTS") == "1":
        return
    skip = pytest.mark.skip(
        reason="needs the released artifacts (POLLENFORMER_ARTIFACTS=1)"
    )
    for item in items:
        if "artifacts" in item.keywords:
            item.add_marker(skip)
