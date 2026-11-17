"""UMAP embedding scatter: where a query lands among the gallery, in 2D.

Reads the precomputed ``umap_xy.npz`` release artifact (built by
``scripts/build_umap.py``). The query is placed at the cosine-weighted mean of its
nearest neighbours' coordinates, an *approximate* position that needs no ``umap-learn``
at runtime. If the projection cannot be loaded the panel shows a one-line hint instead
of failing.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pollenformer import artifacts
from pollenformer.schemas import PredictResponse
from ui.common import class_names

# Cap plotted points so the scatter stays responsive with a ~34k-row gallery.
_MAX_POINTS = 5000


@st.cache_resource(show_spinner=False)
def load_gallery_xy() -> tuple[np.ndarray, np.ndarray] | None:
    """Load the 2D gallery projection.

    Returns:
        ``(xy (N, 2), labels (N,))``, or ``None`` if the artifact is unavailable.
    """
    try:
        path = artifacts.resolve(artifacts.GALLERY_UMAP)
    except artifacts.ArtifactError:
        return None
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["xy"], dtype=np.float32),
            np.asarray(data["labels"], dtype=np.int64),
        )


@st.cache_data(show_spinner=False)
def gallery_plot_df(names: tuple[str, ...]) -> pd.DataFrame | None:
    """Build a subsampled table of gallery points for plotting.

    Args:
        names: Class name of each label id.

    Returns:
        Columns ``x``, ``y`` and ``class``, or ``None`` if the projection is
        unavailable.
    """
    loaded = load_gallery_xy()
    if loaded is None:
        return None
    xy, labels = loaded
    if xy.shape[0] > _MAX_POINTS:
        keep = np.random.default_rng(0).choice(
            xy.shape[0], size=_MAX_POINTS, replace=False
        )
        xy, labels = xy[keep], labels[keep]
    label_names = [names[i] if 0 <= i < len(names) else str(i) for i in labels.tolist()]
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "class": label_names})


def query_position(xy: np.ndarray, resp: PredictResponse) -> tuple[float, float] | None:
    """Place the query at the cosine-weighted mean of its neighbours' 2D coordinates.

    Args:
        xy: ``(N, 2)`` gallery projection.
        resp: Prediction whose neighbour indices and similarities are used.

    Returns:
        ``(x, y)``, or ``None`` when no neighbour falls inside the projection.
    """
    keep = [
        (n.index, max(n.cosine_similarity, 0.0))
        for n in resp.neighbors
        if 0 <= n.index < len(xy)
    ]
    if not keep:
        return None
    idxs = np.array([index for index, _ in keep])
    weights = np.array([weight for _, weight in keep], dtype=np.float64)
    if weights.sum() == 0.0:
        weights[:] = 1.0
    x, y = (xy[idxs] * weights[:, None]).sum(axis=0) / weights.sum()
    return float(x), float(y)


def render_embedding_scatter(resp: PredictResponse) -> None:
    """Render the gallery UMAP with the query overlaid.

    Args:
        resp: Prediction whose neighbours place the query.
    """
    loaded = load_gallery_xy()
    df = gallery_plot_df(class_names())
    if loaded is None or df is None:
        st.info(
            "Embedding map unavailable: the UMAP projection could not be loaded.",
            icon="🗺️",
        )
        return
    position = query_position(loaded[0], resp)
    if position is None:
        st.caption("No gallery coordinates available for the neighbours.")
        return

    gallery = (
        alt.Chart(df)
        .mark_circle(size=18, opacity=0.45)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("class:N", legend=alt.Legend(title="Taxon", columns=2)),
            tooltip=["class"],
        )
    )
    query = (
        alt.Chart(
            pd.DataFrame({"x": [position[0]], "y": [position[1]], "label": ["query"]})
        )
        .mark_point(
            shape="diamond",
            size=300,
            color="black",
            filled=True,
            stroke="white",
            strokeWidth=1.5,
        )
        .encode(x="x:Q", y="y:Q", tooltip=["label"])
    )
    st.altair_chart((gallery + query).properties(height=440), width="stretch")
    st.caption(
        "UMAP of the 128-D gallery (cosine). ◆ marks the query's **approximate** "
        "position (cosine-weighted mean of its nearest neighbours)."
    )
