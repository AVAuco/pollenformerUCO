"""PollenFormerUCO: pollen grain classification with DINOv2 and k-NN inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pollenformer-uco")
except PackageNotFoundError:  # pragma: no cover - source tree without an installed dist
    __version__ = "0.0.0"

__all__ = ["__version__"]
