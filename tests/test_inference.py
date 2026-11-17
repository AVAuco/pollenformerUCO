import numpy as np
from PIL import Image
import pytest
import torch

from helpers import EMB_DIM
from pollenformer.gallery import GalleryIndex
from pollenformer.inference import embed_image, knn_predict
from pollenformer.model import DinoClassifierWithEmbeddings


def axis(i: int, scale: float = 1.0) -> np.ndarray:
    vec = np.zeros(EMB_DIM, dtype=np.float32)
    vec[i] = scale
    return vec


def test_nearest_neighbour_wins_with_one_vote(gallery: GalleryIndex) -> None:
    response = knn_predict(axis(1), gallery, topk=5, k_vote=1)
    assert response.pred_class_name == "Pinus"
    assert response.neighbors[0].label_name == "Pinus"


def test_neighbours_are_ranked_by_similarity(gallery: GalleryIndex) -> None:
    response = knn_predict(axis(0), gallery, topk=4)
    sims = [n.cosine_similarity for n in response.neighbors]
    assert [n.rank for n in response.neighbors] == [1, 2, 3, 4]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_topk_is_capped_at_the_gallery_size(gallery: GalleryIndex) -> None:
    assert len(knn_predict(axis(0), gallery, topk=100).neighbors) == len(gallery)


def test_query_scale_does_not_change_the_result(gallery: GalleryIndex) -> None:
    small = knn_predict(axis(2, 0.01), gallery, topk=3)
    large = knn_predict(axis(2, 50.0), gallery, topk=3)
    assert small.pred_class_id == large.pred_class_id
    assert [n.index for n in small.neighbors] == [n.index for n in large.neighbors]
    assert [n.cosine_similarity for n in small.neighbors] == pytest.approx(
        [n.cosine_similarity for n in large.neighbors]
    )


def test_query_is_not_modified_in_place(gallery: GalleryIndex) -> None:
    query = axis(0, 3.0)
    knn_predict(query, gallery)
    assert query[0] == 3.0


def test_majority_vote_beats_the_single_nearest_neighbour() -> None:
    feats = np.array([[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]], dtype=np.float32)
    gallery = GalleryIndex(
        feats=feats, labels=np.array([0, 1, 1]), class_names=("a", "b")
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert knn_predict(query, gallery, topk=3, k_vote=1).pred_class_name == "a"
    assert knn_predict(query, gallery, topk=3, k_vote=3).pred_class_name == "b"


def test_weighted_vote_follows_similarity() -> None:
    # One very close "a" against two distant "b": weighting tips the vote to "a".
    feats = np.array([[1.0, 0.0], [0.1, 0.995], [0.1, -0.995]], dtype=np.float32)
    gallery = GalleryIndex(
        feats=feats, labels=np.array([0, 1, 1]), class_names=("a", "b")
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert knn_predict(query, gallery, topk=3, k_vote=3).pred_class_name == "b"
    assert (
        knn_predict(query, gallery, topk=3, k_vote=3, weighted=True).pred_class_name
        == "a"
    )


def test_weighted_vote_with_only_negative_similarities_counts_votes() -> None:
    feats = np.array([[-1.0, 0.0], [-0.8, 0.6], [-0.8, -0.6]], dtype=np.float32)
    gallery = GalleryIndex(
        feats=feats, labels=np.array([0, 1, 1]), class_names=("a", "b")
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    response = knn_predict(query, gallery, topk=3, k_vote=3, weighted=True)
    assert response.pred_class_name == "b"


@pytest.mark.parametrize("topk", [0, -1])
def test_invalid_topk_raises(gallery: GalleryIndex, topk: int) -> None:
    with pytest.raises(ValueError, match="topk"):
        knn_predict(axis(0), gallery, topk=topk)


def test_embed_image_returns_a_unit_vector(
    tiny_model: DinoClassifierWithEmbeddings, rgb_image: Image.Image
) -> None:
    emb = embed_image(tiny_model, rgb_image, torch.device("cpu"))
    assert tuple(emb.shape) == (1, EMB_DIM)
    assert torch.linalg.vector_norm(emb).item() == pytest.approx(1.0, abs=1e-5)
