from pathlib import Path

import pytest
import torch

from pollenformer import config


@pytest.mark.parametrize(
    ("raw", "expected"), [("7", 7), ("-3", -3), ("", 5), ("   ", 5), ("seven", 5)]
)
def test_env_int(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    monkeypatch.setenv("PF_TEST_INT", raw)
    assert config.env_int("PF_TEST_INT", 5) == expected


def test_env_int_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PF_TEST_INT", raising=False)
    assert config.env_int("PF_TEST_INT", 5) == 5


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.25", 0.25), ("1e-3", 1e-3), ("", 0.5), ("x", 0.5)]
)
def test_env_float(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("PF_TEST_FLOAT", raw)
    assert config.env_float("PF_TEST_FLOAT", 0.5) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("0", False),
        ("no", False),
    ],
)
def test_env_bool(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("PF_TEST_BOOL", raw)
    assert config.env_bool("PF_TEST_BOOL", not expected) is expected


def test_env_bool_unset_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PF_TEST_BOOL", raising=False)
    assert config.env_bool("PF_TEST_BOOL", True) is True


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.1,0.2,0.3", [0.1, 0.2, 0.3]), ("", [1.0]), ("a,b", [1.0])]
)
def test_env_float_list(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[float]
) -> None:
    monkeypatch.setenv("PF_TEST_LIST", raw)
    assert config.env_float_list("PF_TEST_LIST", [1.0]) == pytest.approx(expected)


def test_env_path_falls_back_on_blank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PF_TEST_PATH", "  ")
    assert config.env_path("PF_TEST_PATH", "default/dir") == Path("default/dir")
    monkeypatch.setenv("PF_TEST_PATH", str(tmp_path))
    assert config.env_path("PF_TEST_PATH", "default/dir") == tmp_path


def test_select_device_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    assert config.select_device() == torch.device("cpu")


def test_input_transform_matches_input_size() -> None:
    from PIL import Image

    tensor = config.IMG_TRANSFORM(Image.new("RGB", (37, 53)))
    assert tuple(tensor.shape) == config.INPUT_SIZE
