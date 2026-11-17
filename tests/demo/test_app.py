"""Smoke tests: the Streamlit app runs end to end on a tiny model and gallery."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest
import torch

from helpers import DEMO_DIR
from pollenformer import artifacts, pipeline
from pollenformer.gallery import GalleryIndex
from pollenformer.model import DinoClassifierWithEmbeddings

MAIN = str(DEMO_DIR / "main.py")


@pytest.fixture(autouse=True)
def _fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model: DinoClassifierWithEmbeddings,
    gallery: GalleryIndex,
) -> None:
    """Serve the tiny model and gallery; make the UMAP artifact unavailable."""
    import ui.common
    import ui.embedding
    import ui.results

    def no_umap(artifact: artifacts.Artifact) -> None:
        raise artifacts.ArtifactError(f"{artifact.filename} is not available in tests")

    monkeypatch.setattr(pipeline, "get_model", lambda: tiny_model)
    monkeypatch.setattr(pipeline, "get_gallery", lambda: gallery)
    monkeypatch.setattr(pipeline, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(ui.common, "get_gallery", lambda: gallery)
    monkeypatch.setattr(ui.results, "get_gallery", lambda: gallery)
    monkeypatch.setattr(artifacts, "resolve", no_umap)
    ui.embedding.load_gallery_xy.clear()


def predict_on_first_sample(app: AppTest, prefix: str) -> AppTest:
    app.radio(key=f"{prefix}_src").set_value("Sample image").run()
    next(b for b in app.button if "Predict" in b.label).click().run()
    return app


def test_soft_mode_lists_the_classes_on_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UI_MODE", raising=False)
    app = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not app.exception
    assert "PollenFormerUCO" in app.title[0].value
    captions = " ".join(c.value for c in app.sidebar.caption)
    assert "Olive" in captions
    assert "NoPollen" in captions


def test_soft_mode_predicts_a_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UI_MODE", raising=False)
    app = predict_on_first_sample(
        AppTest.from_file(MAIN, default_timeout=60).run(), "soft"
    )
    assert not app.exception
    assert not app.error
    assert any("🏷️" in m.value for m in app.markdown)
    assert any("Attention overlay" in m.value for m in app.markdown)


def test_hard_mode_predicts_a_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UI_MODE", "HARD")
    app = predict_on_first_sample(
        AppTest.from_file(MAIN, default_timeout=60).run(), "hard"
    )
    assert not app.exception
    assert not app.error
    assert any("Vote breakdown" in m.value for m in app.markdown)
