"""Embedding of a single image and cosine k-NN classification against the gallery."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from pollenformer.config import IMG_TRANSFORM, mps_sync
from pollenformer.gallery import GalleryIndex
from pollenformer.imaging import ImageInput, to_pil
from pollenformer.schemas import Neighbor, PredictResponse

_NORM_EPS = 1e-12


@torch.no_grad()
def embed_image(
    model: nn.Module,
    image: ImageInput,
    device: torch.device,
    transform: transforms.Compose = IMG_TRANSFORM,
) -> torch.Tensor:
    """Embed one image into the model's L2-normalized projection space.

    Args:
        model: Module returning ``(logits, embeddings)``.
        image: Any input accepted by :func:`pollenformer.imaging.to_pil`.
        device: Device the model lives on.
        transform: Preprocessing from a Pillow image to a ``(3, H, W)`` tensor.

    Returns:
        A ``(1, D)`` float tensor on ``device``.
    """
    batch = transform(to_pil(image)).unsqueeze(0).to(device, non_blocking=True)
    model.eval()
    _, emb = model(batch)
    emb = F.normalize(emb, p=2, dim=1)
    mps_sync(device)
    return emb


def knn_predict(
    query: np.ndarray,
    gallery: GalleryIndex,
    *,
    topk: int = 5,
    k_vote: int = 1,
    weighted: bool = False,
) -> PredictResponse:
    """Classify a query embedding by cosine k-NN over the gallery.

    Args:
        query: ``(D,)`` embedding; it is L2-normalized here.
        gallery: Reference embeddings, normalized row-wise.
        topk: Number of neighbours returned.
        k_vote: Number of nearest neighbours that vote (capped at ``topk``).
        weighted: Weight each vote by ``max(cosine, 0)`` instead of counting it once.

    Returns:
        The predicted class and the ``topk`` neighbours, most similar first.

    Raises:
        ValueError: If ``topk`` is below 1 or the gallery is empty.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + _NORM_EPS)
    sims = gallery.feats @ q  # (N,)

    k = min(topk, sims.shape[0])
    if k <= 0:
        raise ValueError("topk must be >= 1 and the gallery must be non-empty.")
    if k < sims.shape[0]:
        candidates = np.argpartition(-sims, kth=k - 1)[:k]
        order = candidates[np.argsort(-sims[candidates], kind="stable")]
    else:
        order = np.argsort(-sims, kind="stable")

    voters = order[: max(1, min(k_vote, k))]
    n_classes = len(gallery.class_names)
    if weighted and len(voters) > 1:
        weights = np.maximum(sims[voters], 0.0)
        if weights.sum() == 0.0:
            weights = np.ones_like(weights)
        scores = np.bincount(
            gallery.labels[voters], weights=weights, minlength=n_classes
        )
    else:
        scores = np.bincount(gallery.labels[voters], minlength=n_classes)
    pred_id = int(scores.argmax())

    neighbors = [
        Neighbor(
            rank=rank,
            index=int(idx),
            label_id=int(gallery.labels[idx]),
            label_name=gallery.class_names[int(gallery.labels[idx])],
            cosine_similarity=float(np.clip(sims[idx], -1.0, 1.0)),
        )
        for rank, idx in enumerate(order, start=1)
    ]
    return PredictResponse(
        pred_class_id=pred_id,
        pred_class_name=gallery.class_names[pred_id],
        neighbors=neighbors,
    )
