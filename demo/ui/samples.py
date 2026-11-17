"""Shared image input: upload a file or pick a bundled demo sample.

Both UI modes get their input from :func:`render_image_input`, so the upload-or-sample logic
lives in one place. Samples are wrapped in :class:`SampleUpload`, which mimics Streamlit's
``UploadedFile`` (``getvalue``, ``getbuffer``, ``name``, ``type``) so the prediction flow treats
uploads and samples alike.
"""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import streamlit as st

from pollenformer.artifacts import thumbs_dir

# Resolved from this file, not the working directory, so it works locally and in Docker.
DEMO_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "demo_samples"

_SAMPLE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

# Column ratios that centre the upload preview at about half the content width, matching the
# attention-overlay tiles below it.
_PREVIEW_COLS = (1, 2, 1)

# Session keys produced by the prediction flow; cleared when the input image changes so a stale
# result never lingers under a newly selected image.
_PREDICTION_STATE_KEYS = (
    "pred_response",
    "pil_img",
    "base_img_np",
    "attn_map",
    "attn_head_fuse",
    "attn_grid",
    "json_str",
    "inference_time",
)


class SampleUpload(io.BytesIO):
    """In-memory stand-in for a Streamlit ``UploadedFile``."""

    def __init__(self, data: bytes, name: str, mime: str) -> None:
        """Store the image bytes with their display name and MIME type.

        Args:
            data: Encoded image bytes.
            name: File name shown in the UI.
            mime: MIME type used by the allowed-type check, e.g. ``image/png``.
        """
        super().__init__(data)
        self.name = name
        self.type = mime


def list_samples() -> list[Path]:
    """Return the bundled sample images sorted by file name.

    Returns:
        Image paths in :data:`DEMO_SAMPLES_DIR`, empty when the folder is missing.
    """
    if not DEMO_SAMPLES_DIR.is_dir():
        return []
    paths = [path for pattern in _SAMPLE_GLOBS for path in DEMO_SAMPLES_DIR.glob(pattern)]
    return sorted(paths, key=lambda path: path.name.lower())


def _clear_stale_prediction(key_prefix: str, identity: str | None) -> None:
    """Reset the prediction state when the selected input image changes.

    Args:
        key_prefix: Per-mode prefix of the tracking key.
        identity: Stable id of the current input, or ``None`` when nothing is selected.
    """
    tracker = f"{key_prefix}_last_input"
    if st.session_state.get(tracker) != identity:
        st.session_state[tracker] = identity
        for key in _PREDICTION_STATE_KEYS:
            st.session_state[key] = None


def render_image_input(*, key_prefix: str) -> Any:
    """Render the image-source selector and return an upload-compatible object.

    Args:
        key_prefix: Unique prefix (``"soft"`` or ``"hard"``) for widget and session keys.

    Returns:
        An ``UploadedFile`` (upload) or :class:`SampleUpload` (sample), or ``None`` when
        nothing is selected.
    """
    source = st.radio(
        "Image source", options=["Upload", "Sample image"], horizontal=True, key=f"{key_prefix}_src"
    )

    if source == "Upload":
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
            key=f"{key_prefix}_uploader",
        )
        _clear_stale_prediction(key_prefix, f"upload:{uploaded.name}" if uploaded else None)
        return uploaded

    samples = list_samples()
    if not samples:
        st.warning("No demo samples are available.")
        _clear_stale_prediction(key_prefix, None)
        return None

    # Small square preview on the left, sample selector on the right.
    col_img, col_sel = st.columns([1, 3], vertical_alignment="center")
    choice = col_sel.selectbox(
        "Choose a sample", options=[p.name for p in samples], key=f"{key_prefix}_sample"
    )
    path = next(p for p in samples if p.name == choice)
    col_img.image(square_thumb(path), width="stretch")
    _clear_stale_prediction(key_prefix, f"sample:{path.name}")

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return SampleUpload(path.read_bytes(), name=path.name, mime=mime)


def square_thumb(source: Path | io.BytesIO, size: int = 160) -> Image.Image:
    """Return a centre-cropped square thumbnail of an image.

    Args:
        source: Image path or file-like object.
        size: Edge length in pixels.

    Returns:
        A ``size`` by ``size`` RGB image.
    """
    with Image.open(source) as image:
        return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)


def gallery_thumb_path(index: int) -> Path | None:
    """Return the gallery thumbnail of a neighbour, if the thumbnail pack is present.

    Args:
        index: Gallery index of a neighbour.

    Returns:
        Path to ``{index}.jpg`` in the thumbnails directory, or ``None`` if it does not exist.
    """
    path = thumbs_dir() / f"{index}.jpg"
    return path if path.exists() else None


def render_preview(uploaded: Any) -> None:
    """Show an uploaded image centred at about half the content width.

    Samples already have a thumbnail next to their selector, so only uploads are previewed.

    Args:
        uploaded: ``UploadedFile`` or :class:`SampleUpload`.
    """
    if isinstance(uploaded, SampleUpload):
        return
    _, center, _ = st.columns(_PREVIEW_COLS)
    with center:
        st.image(
            square_thumb(io.BytesIO(uploaded.getvalue()), size=512),
            caption=f"Preview: {uploaded.name}",
            width="stretch",
        )
