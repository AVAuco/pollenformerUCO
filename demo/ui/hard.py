"""HARD mode: the expert UI with k-NN, voting and attention-head controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from pollenformer.artifacts import ARTIFACTS_RELEASE, cache_dir
from pollenformer.config import DEVICE, EMB_DIM, INPUT_SIZE, NUM_CLASSES
from pollenformer.pipeline import HeadFuse, run_attention
from ui.common import (
    blend_attention,
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
    compute_vote_breakdown,
    render_how_it_works,
    render_model_card,
    render_neighbors,
    render_prediction_card,
    render_vote_chart,
)
from ui.samples import render_image_input, render_preview

HEAD_LABELS = ["mean", "max", *(f"head {i}" for i in range(8))]
GRID_HEADS: list[HeadFuse] = [*range(8), "mean"]


@dataclass(frozen=True)
class Settings:
    """Values chosen in the sidebar."""

    topk: int
    k_vote: int
    weighted: bool
    show_attention: bool
    head_label: str
    alpha: float
    blend_on_image: bool


def _render_sidebar() -> Settings:
    """Render the k-NN, attention and runtime panels.

    Returns:
        The selected settings.
    """
    with st.sidebar:
        st.header("KNN Settings")
        topk = st.number_input(
            "Top-K neighbors", min_value=1, value=5, step=1, key="ui_topk"
        )
        k_vote = st.number_input(
            "k for voting", min_value=1, value=3, step=1, key="ui_kvote"
        )
        weighted = st.checkbox(
            "Similarity-weighted voting (when k>1)", value=False, key="ui_weighted"
        )

        st.markdown("---")
        st.header("Attention Overlay")
        show_attention = st.checkbox(
            "Show attention on Predict", value=True, key="ui_show_attn"
        )
        head_label = st.selectbox(
            "Head fuse",
            options=HEAD_LABELS,
            index=len(HEAD_LABELS) - 1,
            key="ui_head_fuse",
        )
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05, key="ui_alpha")
        blend_on_image = st.checkbox(
            "Blend over image (otherwise heatmap only)", value=True, key="ui_blend"
        )

        st.markdown("---")
        st.subheader("Runtime & Metadata")
        st.caption(f"⚙️ Device: **{DEVICE}**")
        st.caption("🗂️ Gallery: **UCOPollen train split** (k-NN index)")
        st.caption(f"📦 Artifacts: **{ARTIFACTS_RELEASE}** · `{cache_dir()}`")
        st.caption(
            f"Input size (C,H,W): {INPUT_SIZE} · classes={NUM_CLASSES} · "
            f"emb_dim={EMB_DIM}"
        )
        st.markdown("---")
        st.caption("Allowed types: JPG, PNG, WEBP, BMP · Max 10 MiB")

    return Settings(
        topk=int(topk),
        k_vote=int(k_vote),
        weighted=bool(weighted),
        show_attention=bool(show_attention),
        head_label=str(head_label),
        alpha=float(alpha),
        blend_on_image=bool(blend_on_image),
    )


def _render_results(settings: Settings) -> None:
    """Show the stored prediction: neighbours, vote chart and embedding map."""
    response = st.session_state["pred_response"]
    if response is None:
        return
    render_prediction_card(
        response,
        k_vote=settings.k_vote,
        weighted=settings.weighted,
        elapsed=st.session_state.get("inference_time"),
    )
    col_neighbors, col_votes = st.columns([3, 2])
    with col_neighbors:
        st.markdown("#### Nearest neighbours")
        render_neighbors(response, thumbs=True)
    with col_votes:
        st.markdown("#### Vote breakdown")
        render_vote_chart(
            compute_vote_breakdown(
                response.neighbors, k_vote=settings.k_vote, weighted=settings.weighted
            )
        )
    st.markdown("#### Embedding map")
    render_embedding_scatter(response)
    render_how_it_works()
    render_model_card()
    render_json_download(expanded=True)


def _render_head_grid(alpha: float) -> None:
    """Render the attention overlay of every head on demand, three per row."""
    st.markdown("#### Attention · all heads")
    if st.button("Render all-head grid", key="attn_grid_btn"):
        grid: list[tuple[str, object]] = []
        with st.spinner("Computing attention for every head…"):
            for head in GRID_HEADS:
                try:
                    vis = run_attention(st.session_state["pil_img"], head_fuse=head)
                except Exception:  # noqa: S112 - skip a failing head, render the rest
                    continue
                label = "mean" if head == "mean" else f"head {head}"
                grid.append(
                    (label, blend_attention(vis["image"], vis["attention"], alpha))
                )
        st.session_state["attn_grid"] = grid

    grid = st.session_state.get("attn_grid")
    if not grid:
        return
    columns = st.columns(3)
    for i, (label, image) in enumerate(grid):
        with columns[i % 3]:
            st.caption(label)
            st.image(image, width="stretch")


def render_ui() -> None:
    """Render the HARD interface."""
    settings = _render_sidebar()
    uploaded = render_image_input(key_prefix="hard")
    init_session_state()

    if uploaded is None:
        st.info("Drop an image above to get started.")
        return
    if not upload_is_valid(uploaded):
        return

    render_preview(uploaded)
    if st.button("🔵 Predict", width="stretch"):
        predict_and_store(
            uploaded,
            topk=settings.topk,
            k_vote=settings.k_vote,
            weighted=settings.weighted,
        )
    _render_results(settings)

    if not settings.show_attention:
        return
    if st.session_state["pil_img"] is None:
        st.info("Click **Predict** to generate the attention overlay.")
        return
    ensure_attention(parse_head_fuse(settings.head_label))
    render_attention(
        alpha=settings.alpha,
        blend_on_image=settings.blend_on_image,
        head_label=settings.head_label,
        overlay_caption=(
            f"Overlay (alpha={settings.alpha:.2f}, fuse={settings.head_label})"
        ),
    )
    _render_head_grid(settings.alpha)
