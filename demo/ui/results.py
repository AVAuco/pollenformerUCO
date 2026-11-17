"""Result panels: prediction card, neighbours, vote breakdown and model card.

They turn a ``PredictResponse`` into a readable view of the model's reasoning. The k-NN
vote is recomputed from ``response.neighbors`` with the same rule as
``pollenformer.inference.knn_predict``, so the pipeline needs no extra outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import pandas as pd
import streamlit as st

from pollenformer.config import (
    DEVICE,
    EMB_DIM,
    INPUT_SIZE,
    NUM_CLASSES,
    SIM_CONF_HIGH,
    SIM_CONF_LOW,
)
from pollenformer.pipeline import get_gallery
from pollenformer.schemas import Neighbor, PredictResponse
from ui.samples import gallery_thumb_path

# Matches the theme primaryColor in .streamlit/config.toml.
PRIMARY = "#264796"

BIBTEX = r"""@article{EspantaleonPerez2027PollenFormerUCO,
  author  = {Espantale{\'o}n-P{\'e}rez, Ricardo and Farooq, Qasim and
             L{\'o}pez-Orozco, Roc{\'i}o and Gonz{\'a}lez-Alonso, M{\'o}nica and
             Alashhab, Samer and Oteros, Jose and Buters, Jeroen and
             Gal{\'a}n, Carmen and Mar{\'i}n-Jim{\'e}nez, Manuel J.},
  title   = {{Vision Transformers} for pollen grains classification:
             A systematic backbone benchmark on {UCOPollen}},
  journal = {Atmospheric Environment},
  volume  = {389},
  pages   = {122455},
  year    = {2027},
  doi     = {10.1016/j.atmosenv.2026.122455}
}"""

VoteRow = tuple[str, float, float, int]


def compute_vote_breakdown(
    neighbors: Sequence[Neighbor], *, k_vote: int, weighted: bool
) -> list[VoteRow]:
    """Recompute the k-NN class vote from the returned neighbours.

    Tallies the first ``k_vote`` neighbours, weighting each by ``max(cosine, 0)`` when
    ``weighted`` is set, as ``pollenformer.inference.knn_predict`` does.

    Args:
        neighbors: Neighbours ordered from most to least similar.
        k_vote: Number of nearest neighbours that vote.
        weighted: Whether votes are weighted by cosine similarity.

    Returns:
        ``(class_name, score, fraction, count)`` rows, highest score first.
    """
    voters = neighbors[: max(1, min(int(k_vote), len(neighbors)))]
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for neighbor in voters:
        weight = max(neighbor.cosine_similarity, 0.0) if weighted else 1.0
        scores[neighbor.label_name] = scores.get(neighbor.label_name, 0.0) + weight
        counts[neighbor.label_name] = counts.get(neighbor.label_name, 0) + 1
    total = sum(scores.values()) or 1.0
    rows = [
        (name, score, score / total, counts[name]) for name, score in scores.items()
    ]
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def similarity_confidence(top_cos: float) -> tuple[str, str]:
    """Grade a query by its best neighbour's cosine similarity (traffic light).

    A high vote share can be misleading when the nearest neighbour is still far away (an
    out-of-distribution image); this adds a purely similarity-based signal.

    Args:
        top_cos: Highest cosine similarity among the neighbours.

    Returns:
        ``(emoji, label)``: 🟢 High, 🟡 Moderate or 🔴 Low.
    """
    if top_cos >= SIM_CONF_HIGH:
        return "🟢", "High"
    if top_cos >= SIM_CONF_LOW:
        return "🟡", "Moderate"
    return "🔴", "Low"


def render_prediction_card(
    resp: PredictResponse, *, k_vote: int, weighted: bool, elapsed: float | None = None
) -> None:
    """Render the predicted taxon with its vote and similarity confidence.

    Args:
        resp: Predicted class and neighbours.
        k_vote: Number of voting neighbours.
        weighted: Whether the vote is similarity-weighted.
        elapsed: Inference time in seconds, if measured.
    """
    breakdown = compute_vote_breakdown(resp.neighbors, k_vote=k_vote, weighted=weighted)
    winner = next(
        (row for row in breakdown if row[0] == resp.pred_class_name), breakdown[0]
    )
    top_cos = max((n.cosine_similarity for n in resp.neighbors), default=0.0)
    sem_emoji, sem_label = similarity_confidence(top_cos)

    with st.container(border=True):
        st.markdown(f"### 🏷️ {resp.pred_class_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Vote confidence",
            f"{winner[2] * 100:.0f}%",
            help="Share of the k voting neighbours that chose this class — a majority "
            "signal, NOT a probability and NOT a measure of how close the match is.",
        )
        c2.metric(
            "Match quality",
            f"{sem_emoji} {sem_label}",
            help=(
                f"Graded from the best neighbour cosine = {top_cos:.3f}. "
                f"High >= {SIM_CONF_HIGH:.2f}, Moderate >= {SIM_CONF_LOW:.2f}, "
                "else Low (the query may be out-of-distribution)."
            ),
        )
        if elapsed is not None:
            c3.metric("Inference", f"{elapsed * 1000:.0f} ms")

        if top_cos < SIM_CONF_LOW:
            st.warning(
                f"Low similarity to the gallery (best cosine {top_cos:.3f}). The image "
                "may be out-of-distribution — treat this prediction with caution.",
                icon="⚠️",
            )
        if len(breakdown) > 1:
            chips = " · ".join(
                f"**{name}** {frac * 100:.0f}%" for name, _, frac, _ in breakdown[:3]
            )
            st.caption(f"Top classes by k-NN vote share: {chips}")


def render_vote_chart(breakdown: list[VoteRow]) -> None:
    """Render a horizontal bar chart of each class's vote share.

    Args:
        breakdown: Rows from :func:`compute_vote_breakdown`.
    """
    df = pd.DataFrame(breakdown, columns=["class", "score", "fraction", "count"])
    chart = (
        alt.Chart(df)
        .mark_bar(color=PRIMARY)
        .encode(
            x=alt.X("fraction:Q", axis=alt.Axis(format="%"), title="Vote share"),
            y=alt.Y("class:N", sort="-x", title=None),
            tooltip=[
                "class",
                alt.Tooltip("fraction:Q", format=".0%", title="share"),
                "count",
            ],
        )
        .properties(height=max(80, 28 * len(df)))
    )
    st.altair_chart(chart, width="stretch")


def render_neighbors(resp: PredictResponse, *, thumbs: bool = False) -> None:
    """Render the nearest neighbours with similarity bars and optional thumbnails.

    Args:
        resp: Prediction whose ``neighbors`` are shown.
        thumbs: Show each neighbour's gallery thumbnail when the pack is present.
    """
    for neighbor in resp.neighbors:
        cols = iter(
            st.columns(
                [1, 2, 5, 4] if thumbs else [1, 6, 4], vertical_alignment="center"
            )
        )
        next(cols).markdown(f"**#{neighbor.rank}**")
        if thumbs:
            path = gallery_thumb_path(neighbor.index)
            col = next(cols)
            if path is not None:
                col.image(str(path), width="stretch")
            else:
                col.caption("—")
        is_match = neighbor.label_name == resp.pred_class_name
        next(cols).markdown(
            f"✅ **{neighbor.label_name}**" if is_match else neighbor.label_name
        )
        bar = min(max(neighbor.cosine_similarity, 0.0), 1.0)
        next(cols).progress(bar, text=f"cos {neighbor.cosine_similarity:.3f}")


def render_how_it_works() -> None:
    """Render a one-line pipeline strip that orients first-time viewers."""
    st.caption(
        "🖼️ Image → 🧠 DINOv2-ViT-B → 📐 128-D embedding → "
        "🔍 cosine k-NN → 🗳️ vote → 🏷️ taxon"
    )


def render_model_card() -> None:
    """Render a collapsible model and dataset card with the citation."""
    gallery = get_gallery()
    with st.expander("Model & dataset", expanded=False):
        st.markdown(
            "**Architecture** — DINOv2-ViT-Base backbone, fine-tuned with a hybrid "
            f"Cross-Entropy + Triplet loss; CLS token projected to a **{EMB_DIM}-D** "
            "embedding.\n\n"
            "**Inference** — cosine k-nearest-neighbour vote over a gallery of "
            f"**{len(gallery):,}** training embeddings.\n\n"
            f"**Classes** — {NUM_CLASSES}: {', '.join(gallery.class_names)}.\n\n"
            f"**Input** — C,H,W = {INPUT_SIZE} · **Device** — {DEVICE}."
        )
        st.success(
            "**Reported test performance (UCOPollen v1.0)** — cosine 1-NN over the "
            "128-D embedding: **98.1%** accuracy · **97.5%** macro-F1. "
            "Softmax head: **97.7%** accuracy · **97.1%** macro-F1.",
            icon="📊",
        )
        st.caption(
            "Dataset: **UCOPollen v1.0** — 48,950 microscopy images, 17 pollen taxa "
            "plus one negative (NoPollen) class. If you use this work, please cite:"
        )
        st.code(BIBTEX, language="bibtex")
