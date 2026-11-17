"""Pure UI helpers: vote breakdown, confidence, head parsing and upload checks."""

import io

import numpy as np
import pytest

from pollenformer.config import SIM_CONF_HIGH, SIM_CONF_LOW
from pollenformer.schemas import Neighbor, PredictResponse


def neighbor(rank: int, label: str, cos: float, index: int = 0) -> Neighbor:
    return Neighbor(
        rank=rank, index=index, label_id=0, label_name=label, cosine_similarity=cos
    )


def test_vote_breakdown_counts_and_weights() -> None:
    from ui.results import compute_vote_breakdown

    neighbors = [neighbor(1, "a", 0.9), neighbor(2, "b", 0.5), neighbor(3, "b", 0.3)]
    unweighted = compute_vote_breakdown(neighbors, k_vote=3, weighted=False)
    assert [row[0] for row in unweighted] == ["b", "a"]
    assert unweighted[0][2] == pytest.approx(2 / 3)

    weighted = compute_vote_breakdown(neighbors, k_vote=3, weighted=True)
    assert [row[0] for row in weighted] == ["a", "b"]
    assert weighted[0][1] == pytest.approx(0.9)

    single = compute_vote_breakdown(neighbors, k_vote=0, weighted=False)
    assert single == [("a", 1.0, 1.0, 1)]


@pytest.mark.parametrize(
    ("cos", "label"),
    [(SIM_CONF_HIGH, "High"), (SIM_CONF_LOW, "Moderate"), (SIM_CONF_LOW - 0.01, "Low")],
)
def test_similarity_confidence(cos: float, label: str) -> None:
    from ui.results import similarity_confidence

    assert similarity_confidence(cos)[1] == label


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("mean", "mean"),
        ("max", "max"),
        ("head 7", 7),
        ("head x", "mean"),
        ("other", "mean"),
    ],
)
def test_parse_head_fuse(label: str, expected: str | int) -> None:
    from ui.common import parse_head_fuse

    assert parse_head_fuse(label) == expected


def test_blend_attention() -> None:
    from ui.common import blend_attention

    base = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = blend_attention(base, np.ones((2, 2), dtype=np.float32), alpha=1.0)
    assert out.dtype == np.uint8
    assert out[0, 0].tolist() == [255, 0, 0]


class FakeUpload(io.BytesIO):
    def __init__(self, data: bytes, mime: str) -> None:
        super().__init__(data)
        self.type = mime


def test_upload_validation() -> None:
    from ui.common import MAX_UPLOAD_BYTES, upload_is_valid

    assert upload_is_valid(FakeUpload(b"x", "image/png"))
    assert not upload_is_valid(FakeUpload(b"x", "application/pdf"))
    assert not upload_is_valid(FakeUpload(b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg"))


def test_query_position_is_the_weighted_mean_of_neighbours() -> None:
    from ui.embedding import query_position

    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    resp = PredictResponse(
        pred_class_id=0,
        pred_class_name="a",
        neighbors=[neighbor(1, "a", 0.75, index=1), neighbor(2, "a", 0.25, index=2)],
    )
    assert query_position(xy, resp) == pytest.approx((7.5, 2.5))
    outside = resp.model_copy(update={"neighbors": [neighbor(1, "a", 0.5, index=99)]})
    assert query_position(xy, outside) is None


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, "SOFT"),
        (["--mode", "hard"], None, "HARD"),
        ([], "HARD", "HARD"),
        ([], "x", "SOFT"),
    ],
)
def test_resolve_mode(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], env: str | None, expected: str
) -> None:
    from main import resolve_mode

    if env is None:
        monkeypatch.delenv("UI_MODE", raising=False)
    else:
        monkeypatch.setenv("UI_MODE", env)
    assert resolve_mode(argv) == expected
