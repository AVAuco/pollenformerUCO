import json
from pathlib import Path

import pytest

from pollenformer import __version__, artifacts, cli, pipeline
from pollenformer.schemas import PREDICT_RESPONSE_EXAMPLE, PredictResponse


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2


def test_fetch_lists_every_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        artifacts, "resolve", lambda artifact: tmp_path / artifact.filename
    )
    assert cli.main(["fetch"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == [str(tmp_path / a.filename) for a in artifacts.ALL_ARTIFACTS]


def test_predict_prints_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    received = {}

    def fake_predict_image(image: Path, **kwargs: object) -> PredictResponse:
        received.update(image=image, **kwargs)
        return PredictResponse.model_validate(PREDICT_RESPONSE_EXAMPLE)

    monkeypatch.setattr(pipeline, "predict_image", fake_predict_image)
    image = tmp_path / "grain.png"
    assert (
        cli.main(["predict", str(image), "--topk", "3", "--k-vote", "2", "--weighted"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["pred_class_name"] == "Brassicaceae"
    assert received == {"image": image, "topk": 3, "k_vote": 2, "weighted": True}
