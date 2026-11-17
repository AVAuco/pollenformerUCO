from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from helpers import CLASS_NAMES
from pollenformer import artifacts, pipeline
from pollenformer.gallery import GalleryIndex, l2_normalize_rows
from pollenformer.model import DinoClassifierWithEmbeddings

CPU = torch.device("cpu")


class Upload:
    """Minimal stand-in for a Streamlit ``UploadedFile``."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def self_gallery(
    tiny_model: DinoClassifierWithEmbeddings, rgb_image: Image.Image
) -> GalleryIndex:
    """A gallery whose first entry is the embedding of ``rgb_image`` itself."""
    rng = np.random.default_rng(1)
    own = pipeline.embed_vector(rgb_image, model=tiny_model, device=CPU)
    feats = np.vstack([own, rng.standard_normal((5, own.shape[0]))]).astype(np.float32)
    return GalleryIndex(
        feats=l2_normalize_rows(feats),
        labels=np.array([1, 0, 0, 2, 2, 0]),
        class_names=CLASS_NAMES,
    )


def test_safe_normalize() -> None:
    np.testing.assert_allclose(
        pipeline.safe_normalize(np.array([3.0, 4.0])), [0.6, 0.8]
    )
    assert not pipeline.safe_normalize(np.zeros(3)).any()
    assert not pipeline.safe_normalize(np.array([np.nan, np.inf, -np.inf])).any()


def test_predict_image_finds_the_image_itself(
    tiny_model: DinoClassifierWithEmbeddings,
    self_gallery: GalleryIndex,
    rgb_image: Image.Image,
) -> None:
    response = pipeline.predict_image(
        rgb_image, topk=3, model=tiny_model, gallery=self_gallery, device=CPU
    )
    assert response.pred_class_name == CLASS_NAMES[1]
    assert response.neighbors[0].index == 0
    assert response.neighbors[0].cosine_similarity == pytest.approx(1.0, abs=1e-5)


def test_run_prediction_uses_the_lazily_loaded_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model: DinoClassifierWithEmbeddings,
    self_gallery: GalleryIndex,
    rgb_image: Image.Image,
    png_bytes: bytes,
) -> None:
    monkeypatch.setattr(pipeline, "get_model", lambda: tiny_model)
    monkeypatch.setattr(pipeline, "get_gallery", lambda: self_gallery)
    monkeypatch.setattr(pipeline, "DEVICE", CPU)
    image, response = pipeline.run_prediction(Upload(png_bytes), topk=2, k_vote=1)
    assert image.size == rgb_image.size
    assert len(response.neighbors) == 2
    assert response.neighbors[0].index == 0


def test_get_model_and_gallery_load_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"model": 0, "gallery": 0}

    def fake_load_model(path: Path, *, device: torch.device) -> str:
        calls["model"] += 1
        return f"model from {path.name}"

    def fake_load_gallery(index: Path, meta: Path) -> str:
        calls["gallery"] += 1
        return f"gallery from {index.name}+{meta.name}"

    monkeypatch.setattr(
        artifacts, "resolve", lambda artifact: tmp_path / artifact.filename
    )
    monkeypatch.setattr(pipeline, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline, "load_gallery_index", fake_load_gallery)

    assert (
        pipeline.get_model()
        == pipeline.get_model()
        == f"model from {artifacts.MODEL.filename}"
    )
    assert pipeline.get_gallery() == pipeline.get_gallery()
    assert calls == {"model": 1, "gallery": 1}


@pytest.mark.parametrize("head_fuse", ["mean", "max", 0, 7])
def test_attention_map_is_normalized_and_aligned(
    tiny_model: DinoClassifierWithEmbeddings,
    rgb_image: Image.Image,
    head_fuse: str | int,
) -> None:
    result = pipeline.compute_attention_map(
        tiny_model, rgb_image, CPU, head_fuse=head_fuse
    )
    attention, image = result["attention"], result["image"]
    assert attention.shape == (224, 224)
    assert image.shape == (224, 224, 3)
    assert attention.min() == pytest.approx(0.0, abs=1e-6)
    assert attention.max() == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert tiny_model._attn_hooks == []  # hooks are removed after the forward pass


def test_run_attention_defaults_to_the_lazy_model(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model: DinoClassifierWithEmbeddings,
    png_bytes: bytes,
) -> None:
    monkeypatch.setattr(pipeline, "get_model", lambda: tiny_model)
    monkeypatch.setattr(pipeline, "DEVICE", CPU)
    assert pipeline.run_attention(png_bytes, head_fuse="mean")["attention"].shape == (
        224,
        224,
    )


def test_attention_requires_captured_maps(
    tiny_model: DinoClassifierWithEmbeddings,
    rgb_image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tiny_model, "register_attn_hooks", lambda: None)
    monkeypatch.setattr(tiny_model, "get_attn_maps", list)
    with pytest.raises(RuntimeError, match="No attention maps"):
        pipeline.compute_attention_map(tiny_model, rgb_image, CPU)
