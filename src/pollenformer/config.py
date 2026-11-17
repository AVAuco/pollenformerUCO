"""Runtime settings: device, input preprocessing and k-NN thresholds.

Every value can be overridden with an environment variable. Importing this module is cheap: it
only reads the environment and picks a device; artifacts are loaded lazily by ``pipeline``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def env_path(name: str, default: str | Path) -> Path:
    """Read a path from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or blank.

    Returns:
        The configured path.
    """
    value = os.getenv(name)
    return Path(value) if value and value.strip() else Path(default)


def env_int(name: str, default: int) -> int:
    """Read an integer from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset, blank or not an integer.

    Returns:
        The configured integer.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default=%d", name, value, default)
        return default


def env_float(name: str, default: float) -> float:
    """Read a float from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset, blank or not a number.

    Returns:
        The configured float.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default=%s", name, value, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        ``True`` for ``1/true/yes/y/on`` (any case), ``False`` for any other set value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float_list(name: str, default: list[float]) -> list[float]:
    """Read a comma-separated list of floats from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset, blank or malformed.

    Returns:
        The configured list.
    """
    value = os.getenv(name)
    if not value:
        return default
    try:
        return [float(item) for item in value.split(",")]
    except ValueError:
        logger.warning("Invalid float list for %s=%r; using default=%r", name, value, default)
        return default


def select_device() -> torch.device:
    """Pick the compute device: ``DEVICE`` env var, else MPS, else CUDA, else CPU.

    Returns:
        The selected device.
    """
    name = os.getenv("DEVICE")
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: torch.device = select_device()


def mps_sync(device: torch.device = DEVICE) -> None:
    """Wait for pending Apple MPS work, so timings and host copies see finished results.

    Args:
        device: Device the work ran on; a no-op unless it is MPS.
    """
    if device.type == "mps":
        torch.mps.synchronize()


_TORCH_THREADS = env_int("TORCH_NUM_THREADS", 0)
if _TORCH_THREADS > 0 and DEVICE.type == "cpu":
    torch.set_num_threads(_TORCH_THREADS)

# Model / data specs.
NUM_CLASSES = env_int("NUM_CLASSES", 18)
EMB_DIM = env_int("EMB_DIM", 128)

# Thresholds on the best neighbour's cosine similarity for the match-quality "semaphore": a high
# vote share can still be unreliable when the top neighbour is far away (out-of-distribution
# query). Green >= HIGH, amber >= LOW, else red.
SIM_CONF_HIGH = env_float("SIM_CONF_HIGH", 0.80)
SIM_CONF_LOW = env_float("SIM_CONF_LOW", 0.65)

INPUT_H = env_int("INPUT_H", 224)
INPUT_W = env_int("INPUT_W", 224)
INPUT_SIZE: tuple[int, int, int] = (3, INPUT_H, INPUT_W)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_IMG_MEAN = env_float_list("IMG_MEAN", list(IMAGENET_MEAN))
_IMG_STD = env_float_list("IMG_STD", list(IMAGENET_STD))
if len(_IMG_MEAN) != 3 or len(_IMG_STD) != 3:
    logger.warning("IMG_MEAN/IMG_STD need 3 values; falling back to per-channel unit stats.")
    _IMG_MEAN, _IMG_STD = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ]
)

logger.info(
    "Using device=%s | input=%s mean=%s std=%s classes=%d emb_dim=%d",
    DEVICE,
    INPUT_SIZE,
    _IMG_MEAN,
    _IMG_STD,
    NUM_CLASSES,
    EMB_DIM,
)
