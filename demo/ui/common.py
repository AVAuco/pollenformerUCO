"""Upload checks, session state and the predict/attention flow shared by both UI modes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import streamlit as st

from pollenformer.artifacts import ArtifactError
from pollenformer.gallery import IndexLoadError
from pollenformer.pipeline import HeadFuse, get_gallery, run_attention, run_prediction

ALLOWED_IMAGE_CT = frozenset({"image/jpeg", "image/png", "image/webp", "image/bmp"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB

# Session keys of the prediction flow, with their initial values.
_SESSION_DEFAULTS: dict[str, Any] = {
    "pred_response": None,
    "pil_img": None,
    "base_img_np": None,
    "attn_map": None,
    "attn_head_fuse": None,
    "json_str": None,
    "inference_time": None,
}


def init_session_state() -> None:
    """Create the prediction-flow session keys that do not exist yet."""
    for key, value in _SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def class_names() -> tuple[str, ...]:
    """Return the gallery class names, or show an error and return none if unavailable.

    Returns:
        The class names, empty when the gallery cannot be loaded.
    """
    try:
        return get_gallery().class_names
    except (ArtifactError, IndexLoadError) as exc:
        st.error(f"The k-NN gallery is unavailable: {exc}")
        return ()


def upload_is_valid(uploaded: Any) -> bool:
    """Check the upload's MIME type and size, showing an error when it is rejected.

    Args:
        uploaded: Object with ``type`` and ``getbuffer()``, like a Streamlit ``UploadedFile``.

    Returns:
        Whether the upload can be processed.
    """
    content_type = getattr(uploaded, "type", None) or ""
    if content_type not in ALLOWED_IMAGE_CT:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_CT))
        st.error(f"Unsupported file type: {content_type or 'unknown'}. Allowed: {allowed}")
        return False
    if uploaded.getbuffer().nbytes > MAX_UPLOAD_BYTES:
        st.error(f"File too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB)")
        return False
    return True


def predict_and_store(uploaded: Any, *, topk: int, k_vote: int, weighted: bool) -> None:
    """Run the prediction and keep it in session state across reruns.

    Args:
        uploaded: Upload-like object exposing ``getvalue()``.
        topk: Number of neighbours returned.
        k_vote: Number of neighbours that vote.
        weighted: Weight votes by cosine similarity.
    """
    try:
        with st.spinner("Embedding and searching…"):
            start = time.perf_counter()
            image, response = run_prediction(uploaded, topk=topk, k_vote=k_vote, weighted=weighted)
            elapsed = time.perf_counter() - start
    except Exception as exc:  # shown in the page instead of crashing the app
        st.error(f"Prediction failed: {exc}")
        st.exception(exc)
        return
    st.session_state.update(
        pred_response=response,
        pil_img=image,
        json_str=response.model_dump_json(indent=2),
        inference_time=elapsed,
        base_img_np=None,
        attn_map=None,
        attn_head_fuse=None,
    )


def parse_head_fuse(label: str) -> HeadFuse:
    """Map a head selector label to the pipeline's ``head_fuse`` argument.

    Args:
        label: ``"mean"``, ``"max"`` or ``"head <n>"``.

    Returns:
        ``"mean"``, ``"max"`` or the head index; ``"mean"`` for unparsable labels.
    """
    if label.startswith("head "):
        try:
            return int(label.split()[-1])
        except ValueError:
            return "mean"
    return label if label in {"mean", "max"} else "mean"


def blend_attention(base: np.ndarray, attention: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay the attention map in red on the image.

    Args:
        base: ``(H, W, 3)`` image in ``[0, 1]``.
        attention: ``(H, W)`` attention in ``[0, 1]``.
        alpha: Overlay opacity in ``[0, 1]``.

    Returns:
        The ``(H, W, 3)`` uint8 overlay.
    """
    red = np.zeros_like(base, dtype=np.float32)
    red[..., 0] = attention
    blended = np.clip((1.0 - alpha) * base + alpha * red, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def ensure_attention(head_fuse: HeadFuse) -> None:
    """Compute the attention overlay unless the stored one matches the image and head choice.

    Args:
        head_fuse: Head selection passed to the pipeline.
    """
    state = st.session_state
    if (
        state.get("attn_map") is not None
        and state.get("base_img_np") is not None
        and state.get("attn_head_fuse") == head_fuse
    ):
        return
    try:
        vis = run_attention(state["pil_img"], head_fuse=head_fuse)
    except Exception as exc:  # shown in the page instead of crashing the app
        st.error(f"Attention overlay failed: {exc}")
        st.exception(exc)
        return
    state.update(attn_map=vis["attention"], base_img_np=vis["image"], attn_head_fuse=head_fuse)


def render_attention(
    *, alpha: float, blend_on_image: bool, head_label: str, overlay_caption: str
) -> None:
    """Show the preprocessed input next to its attention overlay or heatmap.

    Args:
        alpha: Overlay opacity.
        blend_on_image: Blend over the image; otherwise show the bare heatmap.
        head_label: Selected head, shown in the heatmap caption.
        overlay_caption: Caption of the blended overlay.
    """
    attention = st.session_state.get("attn_map")
    base = st.session_state.get("base_img_np")
    if attention is None or base is None:
        return

    st.markdown("#### Attention overlay")
    col_input, col_attention = st.columns(2)
    with col_input:
        st.caption("Original (preprocessed)")
        st.image((base * 255).astype(np.uint8), width="stretch")
    with col_attention:
        if blend_on_image:
            st.caption(overlay_caption)
            st.image(blend_attention(base, attention, alpha), width="stretch")
        else:
            st.caption(f"Heatmap (fuse={head_label})")
            st.image(np.stack([attention] * 3, axis=-1), clamp=True, width="stretch")


def render_json_download(*, expanded: bool) -> None:
    """Show the raw JSON response and a button to download it.

    Args:
        expanded: Whether the JSON expander starts open.
    """
    with st.expander("Raw JSON response", expanded=expanded):
        st.code(st.session_state["json_str"], language="json")
    st.download_button(
        label="Download JSON",
        data=st.session_state["json_str"].encode("utf-8"),
        file_name="prediction.json",
        mime="application/json",
        key="download_json",
    )
