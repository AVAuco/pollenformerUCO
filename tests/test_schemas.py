from pydantic import ValidationError
import pytest

from pollenformer.schemas import PREDICT_RESPONSE_EXAMPLE, Neighbor, PredictResponse


def test_example_round_trips_through_json() -> None:
    response = PredictResponse.model_validate(PREDICT_RESPONSE_EXAMPLE)
    assert PredictResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "override",
    [
        {"rank": 0},
        {"cosine_similarity": 1.5},
        {"cosine_similarity": -1.01},
        {"extra": "field"},
    ],
)
def test_neighbor_rejects_invalid_values(override: dict) -> None:
    data = {
        "rank": 1,
        "index": 0,
        "label_id": 0,
        "label_name": "a",
        "cosine_similarity": 0.5,
    }
    with pytest.raises(ValidationError):
        Neighbor(**{**data, **override})
