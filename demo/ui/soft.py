"""SOFT mode: the public-facing UI with fixed k-NN and attention settings."""

from __future__ import annotations

import streamlit as st

from ui.common import (
    class_names,
    ensure_attention,
    init_session_state,
    parse_head_fuse,
    predict_and_store,
    render_attention,
    render_json_download,
    upload_is_valid,
)
from ui.embedding import render_embedding_scatter
from ui.results import (
    render_how_it_works,
    render_model_card,
    render_neighbors,
    render_prediction_card,
)
from ui.samples import render_image_input, render_preview

TOPK = 5
K_VOTE = 3
WEIGHTED = False
HEAD_LABEL = "head 7"
ALPHA = 0.5


def _render_sidebar() -> None:
    """List the recognised taxa and the accepted uploads."""
    with st.sidebar:
        st.subheader("Pollen taxonomies 🔬")
        for index, name in enumerate(class_names(), start=1):
            st.caption(f"- **Class {index}**: {name}")
        st.markdown("---")
        st.caption("Allowed types: JPG, PNG, WEBP, BMP · Max 10 MiB")


def _render_results() -> None:
    """Show the stored prediction with neighbours, embedding map and model card."""
    response = st.session_state["pred_response"]
    if response is None:
        return
    render_prediction_card(
        response,
        k_vote=K_VOTE,
        weighted=WEIGHTED,
        elapsed=st.session_state.get("inference_time"),
    )
    st.markdown("#### Nearest neighbours")
    render_neighbors(response)
    render_how_it_works()
    with st.expander("Embedding map", expanded=False):
        render_embedding_scatter(response)
    render_model_card()
    render_json_download(expanded=False)


def render_ui() -> None:
    """Render the SOFT interface."""
    _render_sidebar()
    uploaded = render_image_input(key_prefix="soft")
    init_session_state()

    if uploaded is None:
        st.info("Drop an image above to get started.")
        return
    if not upload_is_valid(uploaded):
        return

    render_preview(uploaded)
    if st.button("🔵 Predict", width="stretch"):
        predict_and_store(uploaded, topk=TOPK, k_vote=K_VOTE, weighted=WEIGHTED)
    _render_results()

    if st.session_state["pil_img"] is None:
        st.info("Click **Predict** to generate the attention overlay.")
        return
    ensure_attention(parse_head_fuse(HEAD_LABEL))
    render_attention(
        alpha=ALPHA,
        blend_on_image=True,
        head_label=HEAD_LABEL,
        overlay_caption="Attention Overlay",
    )
