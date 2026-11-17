"""Image decoding: turn bytes, paths, arrays or tensors into RGB Pillow images."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

ImageInput = (
    Image.Image
    | bytes
    | bytearray
    | memoryview
    | str
    | Path
    | np.ndarray
    | torch.Tensor
)


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode and verify encoded image bytes, honouring the EXIF orientation.

    Args:
        data: Encoded image (JPEG, PNG, WEBP, BMP, ...).

    Returns:
        The decoded RGB image.

    Raises:
        ValueError: If the bytes are empty, corrupt or not a supported image.
    """
    if not data:
        raise ValueError("Empty file.")
    buffer = io.BytesIO(memoryview(data))
    try:
        with Image.open(buffer) as raw:
            raw.verify()
        buffer.seek(0)
        with Image.open(buffer) as checked:
            checked.load()
            return ImageOps.exif_transpose(checked).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"Unsupported or corrupt image: {exc}") from exc


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an ``HxWxC`` or ``CxHxW`` RGB(A)/grayscale array to an RGB image."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError("Array must be HxW, HxWx{1,3,4} or {1,3,4}xHxW.")

    arr = arr[..., :3] if arr.shape[2] == 4 else arr
    arr = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")


def to_pil(image: ImageInput) -> Image.Image:
    """Convert any supported image input to an RGB Pillow image.

    Args:
        image: A Pillow image, encoded bytes, a file path, an RGB array (``uint8`` in
            ``[0, 255]`` or float in ``[0, 1]``) or a ``CxHxW`` tensor in ``[0, 1]``.

    Returns:
        The RGB image, with the EXIF orientation applied where present.

    Raises:
        ValueError: If the input cannot be decoded.
        TypeError: If the input type is not supported.
    """
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, bytes | bytearray | memoryview):
        return load_image_bytes(bytes(image))
    if isinstance(image, str | Path):
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clamp(0, 1).numpy()
    if isinstance(image, np.ndarray):
        return _array_to_pil(image)
    raise TypeError(f"Unsupported image input: {type(image).__name__}")
