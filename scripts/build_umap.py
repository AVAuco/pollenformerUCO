"""Build the 2D UMAP projection consumed by the demo's embedding scatter.

Reads the k-NN gallery (embeddings + labels), fits a cosine UMAP down to two dimensions
and writes ``umap_xy.npz`` (``xy`` + ``labels``, no pickled objects). It is a build-time
tool that depends only on numpy and umap-learn, so it runs in an isolated environment
(``mise run assets:umap -- --index ... --out ...``); umap-learn is never a runtime
dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--index", type=Path, required=True, help="Gallery hyb128_index.npz."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output umap_xy.npz.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Fit the 2D UMAP projection and write the npz file."""
    args = _parse_args()
    import umap  # imported here so --help works without umap-learn installed

    with np.load(args.index, allow_pickle=False) as data:
        feats = np.asarray(data["feats"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
    print(f"Loaded gallery: feats={feats.shape} labels={labels.shape}")

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    xy = reducer.fit_transform(feats).astype(np.float32)
    print(f"UMAP projection done: xy={xy.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xy=xy, labels=labels)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
