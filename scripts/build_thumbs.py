"""Build the optional thumbnails shown next to each nearest neighbour in the demo.

The UCOPollen images are not distributed with the model, so this script makes a
downsized thumbnail per gallery index from a local copy of the dataset:

    mise run assets:thumbs -- --images-root /path/to/UCOPollen

It reads each index's source path from the gallery metadata and writes ``{index}.jpg``
into the thumbnails directory. The demo shows them once that directory exists; without
it the neighbour panel shows rank, class and similarity only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from pollenformer import artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Root that the metadata's relative paths are resolved against.",
    )
    parser.add_argument(
        "--meta", type=Path, default=None, help="Default: the release artifact."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: artifacts.thumbs_dir()."
    )
    parser.add_argument(
        "--size", type=int, default=96, help="Thumbnail edge in pixels."
    )
    return parser.parse_args()


def main() -> None:
    """Write one downsized JPEG thumbnail per gallery index."""
    args = _parse_args()
    meta_path = args.meta or artifacts.resolve(artifacts.GALLERY_META)
    out = args.out or artifacts.thumbs_dir()
    paths: list[str] | None = json.loads(meta_path.read_text(encoding="utf-8")).get(
        "paths"
    )
    if not paths:
        raise SystemExit(f"No 'paths' in {meta_path}; cannot build thumbnails.")

    out.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    for index, rel in enumerate(paths):
        src = args.images_root / rel if args.images_root is not None else Path(rel)
        if not src.exists():
            missing += 1
            continue
        try:
            with Image.open(src) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((args.size, args.size), Image.Resampling.BICUBIC)
                thumb.save(out / f"{index}.jpg", quality=85)
            written += 1
        except OSError as exc:  # keep going and report at the end
            print(f"  skip index={index} ({src}): {exc}")
            missing += 1

    print(f"Wrote {written} thumbnails to {out} ({missing} missing or failed)")


if __name__ == "__main__":
    main()
