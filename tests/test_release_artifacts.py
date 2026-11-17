"""End-to-end checks with the released model and gallery.

Skipped by default. Run them with ``mise run test:e2e`` once the artifacts are
available, either downloaded from the release or pointed to with ``MODEL_PATH``,
``INDEX_PATH`` and ``META_PATH``.
"""

from __future__ import annotations

import json

import pytest

from helpers import DEMO_DIR
from pollenformer import cli, pipeline

pytestmark = pytest.mark.artifacts

SAMPLES = DEMO_DIR / "demo_samples"


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("Brassicaceae.png", "Brassicaceae"),
        ("brassicaceae2.png", "Brassicaceae"),
        ("castanea.png", "Castanea"),
        ("nopollen.png", "NoPollen"),
    ],
)
def test_released_model_classifies_the_demo_samples(sample: str, expected: str) -> None:
    response = pipeline.predict_image(SAMPLES / sample, topk=5, k_vote=1)
    assert response.pred_class_name == expected
    assert response.neighbors[0].cosine_similarity > 0.8


def test_released_gallery_matches_the_paper_setup() -> None:
    gallery = pipeline.get_gallery()
    assert len(gallery.class_names) == 18
    assert "NoPollen" in gallery.class_names
    assert gallery.feats.shape == (34257, 128)


def test_cli_predict_prints_clean_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["predict", str(SAMPLES / "castanea.png"), "--topk", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["pred_class_name"] == "Castanea"


def test_demo_app_predicts_with_the_released_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("UI_MODE", raising=False)
    app = AppTest.from_file(str(DEMO_DIR / "main.py"), default_timeout=300).run()
    app.radio(key="soft_src").set_value(
        "Sample image"
    ).run()  # first sample: Brassicaceae.png
    next(b for b in app.button if "Predict" in b.label).click().run()
    assert not app.exception
    assert not app.error
    assert any("Brassicaceae" in m.value for m in app.markdown)
