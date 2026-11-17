"""End-to-end prediction and attention maps.

The model and the gallery are loaded on first use and then reused, so importing this
module needs neither the network nor the artifacts.
"""

from __future__ import annotations

import functools
import logging
from typing import Protocol, TypedDict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from pollenformer import artifacts
from pollenformer.config import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    mps_sync,
)
from pollenformer.gallery import GalleryIndex, load_gallery_index
from pollenformer.imaging import ImageInput, load_image_bytes, to_pil
from pollenformer.inference import embed_image, knn_predict
from pollenformer.model import DinoClassifierWithEmbeddings, load_model
from pollenformer.schemas import PredictResponse

logger = logging.getLogger(__name__)

HeadFuse = str | int


class UploadedImage(Protocol):
    """Anything exposing the encoded bytes, such as a Streamlit ``UploadedFile``."""

    def getvalue(self) -> bytes:
        """Return the encoded image bytes."""
        ...


class AttentionResult(TypedDict):
    """Attention map and the preprocessed image it is aligned with."""

    attention: np.ndarray
    image: np.ndarray


@functools.cache
def get_model() -> DinoClassifierWithEmbeddings:
    """Return the PollenFormerUCO model, loading it on the first call.

    Returns:
        The model in evaluation mode on :data:`pollenformer.config.DEVICE`.
    """
    return load_model(artifacts.resolve(artifacts.MODEL), device=DEVICE)


@functools.cache
def get_gallery() -> GalleryIndex:
    """Return the k-NN gallery, loading it on the first call.

    Returns:
        The reference embeddings with their labels and class names.
    """
    return load_gallery_index(
        artifacts.resolve(artifacts.GALLERY_INDEX),
        artifacts.resolve(artifacts.GALLERY_META),
    )


def safe_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector, mapping non-finite values and near-zero vectors to zeros.

    Args:
        vec: Input vector.
        eps: Norm below which the result is all zeros.

    Returns:
        A float32 unit vector, or zeros.
    """
    vec = np.nan_to_num(
        np.asarray(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def embed_vector(
    image: ImageInput,
    *,
    model: DinoClassifierWithEmbeddings | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Embed an image into a one-dimensional float32 vector.

    Args:
        image: Any input accepted by :func:`pollenformer.imaging.to_pil`.
        model: Model to use; defaults to :func:`get_model`.
        device: Device the model lives on; defaults to :data:`DEVICE` at call time.

    Returns:
        The ``(D,)`` embedding.
    """
    model = model if model is not None else get_model()
    device = device if device is not None else DEVICE
    vec = embed_image(model, image, device).squeeze(0).cpu().numpy().astype(np.float32)
    if vec.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape: {vec.shape}")
    return vec


def predict_image(
    image: ImageInput,
    *,
    topk: int = 5,
    k_vote: int = 1,
    weighted: bool = False,
    model: DinoClassifierWithEmbeddings | None = None,
    gallery: GalleryIndex | None = None,
    device: torch.device | None = None,
) -> PredictResponse:
    """Classify one image by cosine k-NN over the gallery.

    Args:
        image: Any input accepted by :func:`pollenformer.imaging.to_pil`.
        topk: Number of neighbours returned.
        k_vote: Number of neighbours that vote; ``1`` is the setting reported in
            the paper.
        weighted: Weight votes by cosine similarity.
        model: Model to use; defaults to :func:`get_model`.
        gallery: Gallery to search; defaults to :func:`get_gallery`.
        device: Device the model lives on; defaults to :data:`DEVICE` at call
            time.

    Returns:
        The predicted class and nearest neighbours.
    """
    query = safe_normalize(embed_vector(image, model=model, device=device))
    gallery = gallery if gallery is not None else get_gallery()
    return knn_predict(query, gallery, topk=topk, k_vote=k_vote, weighted=weighted)


def run_prediction(
    uploaded_file: UploadedImage, topk: int = 5, k_vote: int = 3, weighted: bool = False
) -> tuple[Image.Image, PredictResponse]:
    """Decode an uploaded file and classify it (the demo's upload-to-prediction flow).

    Args:
        uploaded_file: Object exposing ``getvalue()`` with the encoded image.
        topk: Number of neighbours returned.
        k_vote: Number of neighbours that vote.
        weighted: Weight votes by cosine similarity.

    Returns:
        The decoded image and the prediction.
    """
    image = load_image_bytes(uploaded_file.getvalue())
    return image, predict_image(image, topk=topk, k_vote=k_vote, weighted=weighted)


def attention_transform(image: Image.Image) -> torch.Tensor:
    """Preprocess an image for the attention overlay (bicubic resize + ImageNet stats).

    Args:
        image: RGB image.

    Returns:
        ``(3, H, W)`` normalized tensor.
    """
    _, height, width = INPUT_SIZE
    tfm = transforms.Compose(
        [
            transforms.Resize(
                (height, width), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(image)


def _fuse_heads(cls_rows: torch.Tensor, head_fuse: HeadFuse) -> torch.Tensor:
    """Reduce ``(heads, patches)`` CLS attention to ``(patches,)``."""
    if isinstance(head_fuse, int):
        return cls_rows[head_fuse]
    if head_fuse == "max":
        return cls_rows.max(dim=0).values
    return cls_rows.mean(dim=0)


@torch.no_grad()
def compute_attention_map(
    model: DinoClassifierWithEmbeddings,
    image: ImageInput,
    device: torch.device,
    head_fuse: HeadFuse = "mean",
) -> AttentionResult:
    """Compute the last block's CLS-to-patch attention, upsampled to the input size.

    Args:
        model: Model with attention hooks (see ``register_attn_hooks``).
        image: Any input accepted by :func:`pollenformer.imaging.to_pil`.
        device: Device the model lives on.
        head_fuse: ``"mean"``, ``"max"`` or the index of a single head.

    Returns:
        The ``(H, W)`` attention in ``[0, 1]`` and the ``(H, W, 3)`` de-normalized
        input.
    """
    input_tensor = (
        attention_transform(to_pil(image)).unsqueeze(0).to(device, non_blocking=True)
    )

    model.eval()
    model.register_attn_hooks()
    try:
        model(input_tensor)
        attn_maps = model.get_attn_maps()
    finally:
        model.remove_attn_hooks()
    if not attn_maps:
        raise RuntimeError(
            "No attention maps captured; check the hooks and forward path."
        )

    attn_last = attn_maps[-1]  # (B, heads, tokens, tokens)
    if attn_last.ndim != 4 or attn_last.size(0) != 1:
        raise RuntimeError(f"Unexpected attention shape: {tuple(attn_last.shape)}")

    cls_patches = _fuse_heads(attn_last[0][:, 0, 1:], head_fuse)  # CLS -> patches
    grid_h = int(np.floor(np.sqrt(cls_patches.numel())))
    grid = cls_patches.reshape(1, 1, grid_h, cls_patches.numel() // grid_h)
    height, width = input_tensor.shape[-2:]
    attn = F.interpolate(
        grid, size=(height, width), mode="bilinear", align_corners=False
    )
    attn_np = attn.squeeze().cpu().numpy().astype(np.float32)
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-12)

    img_np = input_tensor[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img_np = np.clip(
        img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0.0, 1.0
    )
    mps_sync(device)
    return {"attention": attn_np, "image": img_np.astype(np.float32)}


def run_attention(
    image: ImageInput,
    head_fuse: HeadFuse = "mean",
    *,
    model: DinoClassifierWithEmbeddings | None = None,
) -> AttentionResult:
    """Compute the attention overlay for an image (the demo's attention flow).

    Args:
        image: Any input accepted by :func:`pollenformer.imaging.to_pil`.
        head_fuse: ``"mean"``, ``"max"`` or the index of a single head.
        model: Model to use; defaults to :func:`get_model`.

    Returns:
        The attention map and the image it is aligned with.
    """
    model = model if model is not None else get_model()
    return compute_attention_map(model, image, device=DEVICE, head_fuse=head_fuse)
