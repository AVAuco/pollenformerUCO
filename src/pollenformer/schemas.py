"""Pydantic response schemas for the demo prediction API shape."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

NEIGHBOR_EXAMPLE = {
    "rank": 1,
    "index": 18184,
    "label_id": 8,
    "label_name": "Olive",
    "cosine_similarity": 0.8283524513244629,
}

PREDICT_RESPONSE_EXAMPLE = {
    "pred_class_id": 0,
    "pred_class_name": "Brassicaceae",
    "neighbors": [
        NEIGHBOR_EXAMPLE,
        {
            "rank": 2,
            "index": 61,
            "label_id": 0,
            "label_name": "Brassicaceae",
            "cosine_similarity": 0.8066779375076294,
        },
        {
            "rank": 3,
            "index": 166,
            "label_id": 0,
            "label_name": "Brassicaceae",
            "cosine_similarity": 0.7715726494789124,
        },
        {
            "rank": 4,
            "index": 15362,
            "label_id": 8,
            "label_name": "Olive",
            "cosine_similarity": 0.7688477039337158,
        },
        {
            "rank": 5,
            "index": 57,
            "label_id": 0,
            "label_name": "Brassicaceae",
            "cosine_similarity": 0.7651835680007935,
        },
    ],
}


class Neighbor(BaseModel):
    """Single nearest neighbor with metadata and similarity score."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": NEIGHBOR_EXAMPLE},
    )

    rank: PositiveInt = Field(..., description="1-based neighbor rank")
    index: int = Field(..., description="Index of the neighbor in the database")
    label_id: int = Field(..., description="Numeric class ID of the neighbor")
    label_name: str = Field(
        ..., description="Human-readable class name of the neighbor"
    )
    cosine_similarity: float = Field(
        ..., ge=-1.0, le=1.0, description="Cosine similarity with query embedding"
    )


class PredictResponse(BaseModel):
    """Response for image prediction including top neighbors."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": PREDICT_RESPONSE_EXAMPLE},
    )

    pred_class_id: int = Field(..., description="Predicted class ID")
    pred_class_name: str = Field(..., description="Predicted class name")
    neighbors: list[Neighbor] = Field(..., description="Top-k nearest neighbors")
