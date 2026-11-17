"""Command line: ``pollenformer fetch`` and ``pollenformer predict IMAGE``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys

from pollenformer import __version__, artifacts


def _fetch(_args: argparse.Namespace) -> int:
    """Download and verify every release artifact."""
    for artifact in artifacts.ALL_ARTIFACTS:
        sys.stdout.write(f"{artifacts.resolve(artifact)}\n")
    return 0


def _predict(args: argparse.Namespace) -> int:
    """Classify one image and print the prediction as JSON."""
    # Imported here so `pollenformer fetch` does not pay for torch/torchvision model code.
    from pollenformer.pipeline import predict_image

    response = predict_image(args.image, topk=args.topk, k_vote=args.k_vote, weighted=args.weighted)
    sys.stdout.write(response.model_dump_json(indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="pollenformer", description="PollenFormerUCO pollen grain classifier."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log progress to stderr.")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="Download and verify the model and gallery.")
    fetch.set_defaults(handler=_fetch)

    predict = commands.add_parser("predict", help="Classify a microscopy image.")
    predict.add_argument("image", type=Path, help="Image file (JPG, PNG, WEBP, BMP).")
    predict.add_argument("--topk", type=int, default=5, help="Neighbours to report.")
    predict.add_argument(
        "--k-vote", type=int, default=1, help="Neighbours that vote (paper: cosine 1-NN)."
    )
    predict.add_argument(
        "--weighted", action="store_true", help="Weight votes by cosine similarity."
    )
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
