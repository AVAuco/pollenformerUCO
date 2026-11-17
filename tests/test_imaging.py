from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from helpers import encode
from pollenformer.imaging import load_image_bytes, to_pil


def test_load_image_bytes_decodes_to_rgb(png_bytes: bytes) -> None:
    image = load_image_bytes(png_bytes)
    assert image.mode == "RGB"
    assert image.size == (60, 40)


def test_load_image_bytes_applies_exif_orientation(rgb_image: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise on display
    image = load_image_bytes(encode(rgb_image, "JPEG", exif=exif))
    assert image.size == (40, 60)


@pytest.mark.parametrize("data", [b"", b"definitely not an image"])
def test_load_image_bytes_rejects_bad_input(data: bytes) -> None:
    with pytest.raises(ValueError, match=r"Empty|Unsupported"):
        load_image_bytes(data)


def test_to_pil_from_pil_drops_alpha() -> None:
    assert to_pil(Image.new("RGBA", (5, 4))).mode == "RGB"


def test_to_pil_from_bytes_and_path(png_bytes: bytes, tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes)
    from_bytes, from_path, from_str = to_pil(png_bytes), to_pil(path), to_pil(str(path))
    assert from_bytes.size == from_path.size == from_str.size == (60, 40)
    assert np.array_equal(np.asarray(from_bytes), np.asarray(from_path))


@pytest.mark.parametrize(
    ("array", "expected_size"),
    [
        (np.zeros((4, 6, 3), dtype=np.uint8), (6, 4)),  # HWC uint8
        (np.zeros((4, 6), dtype=np.uint8), (6, 4)),  # grayscale HW
        (np.zeros((4, 6, 1), dtype=np.uint8), (6, 4)),  # grayscale HWC
        (np.zeros((4, 6, 4), dtype=np.uint8), (6, 4)),  # RGBA
        (np.zeros((3, 4, 6), dtype=np.float32), (6, 4)),  # CHW float
    ],
)
def test_to_pil_from_arrays(array: np.ndarray, expected_size: tuple[int, int]) -> None:
    image = to_pil(array)
    assert image.mode == "RGB"
    assert image.size == expected_size


def test_to_pil_scales_unit_floats_and_keeps_rgb_order() -> None:
    array = np.zeros((2, 2, 3), dtype=np.float32)
    array[..., 0] = 1.0  # pure red must stay red
    pixel = np.asarray(to_pil(array))[0, 0]
    assert pixel.tolist() == [255, 0, 0]


def test_to_pil_from_chw_tensor() -> None:
    tensor = torch.zeros(3, 4, 6)
    tensor[2] = 1.0  # blue channel
    pixel = np.asarray(to_pil(tensor))[0, 0]
    assert pixel.tolist() == [0, 0, 255]


def test_to_pil_rejects_bad_shapes_and_types() -> None:
    with pytest.raises(ValueError, match="Array must be"):
        to_pil(np.zeros((2, 2, 2, 2), dtype=np.uint8))
    with pytest.raises(TypeError, match="Unsupported image input"):
        to_pil(42)  # type: ignore[arg-type]
