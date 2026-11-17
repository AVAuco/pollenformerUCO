from pathlib import Path

import pytest
import torch
from torch import nn

from helpers import DIM, EMB_DIM, HEADS, FakeDinoBackbone
from pollenformer import model as model_module
from pollenformer.model import (
    DinoClassifierWithEmbeddings,
    build_classifier,
    load_model,
)


def test_forward_returns_logits_and_embeddings(
    tiny_model: DinoClassifierWithEmbeddings,
) -> None:
    logits, emb = tiny_model(torch.zeros(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 3)
    assert tuple(emb.shape) == (2, EMB_DIM)


@pytest.mark.parametrize("emb_dim", [None, DIM])
def test_no_projection_when_emb_dim_matches(emb_dim: int | None) -> None:
    model = DinoClassifierWithEmbeddings(
        FakeDinoBackbone(), 3, feat_dim=DIM, emb_dim=emb_dim
    )
    assert isinstance(model.embedding_layer, nn.Identity)
    assert model.classifier.in_features == DIM


def test_backbone_is_frozen_and_blocks_can_be_unfrozen(
    tiny_model: DinoClassifierWithEmbeddings,
) -> None:
    blocks = tiny_model.backbone.blocks
    assert not any(p.requires_grad for p in tiny_model.backbone.parameters())
    tiny_model.unfreeze_last_blocks(1)
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert not any(p.requires_grad for p in blocks[0].parameters())
    tiny_model.unfreeze_last_blocks(10)  # capped at the number of blocks
    assert all(p.requires_grad for p in blocks[0].parameters())
    tiny_model.unfreeze_last_blocks(0)  # no-op


def test_attention_hooks_capture_one_map_per_block(
    tiny_model: DinoClassifierWithEmbeddings,
) -> None:
    tiny_model.register_attn_hooks()
    tiny_model(torch.zeros(1, 3, 224, 224))
    maps = tiny_model.get_attn_maps()
    tokens = 1 + (224 // 14) ** 2
    assert len(maps) == len(tiny_model.backbone.blocks)
    assert tuple(maps[-1].shape) == (1, HEADS, tokens, tokens)
    torch.testing.assert_close(maps[-1].sum(dim=-1), torch.ones(1, HEADS, tokens))

    tiny_model.remove_attn_hooks()
    tiny_model(torch.zeros(1, 3, 224, 224))
    assert len(tiny_model.get_attn_maps()) == len(maps)  # nothing new captured


def test_build_classifier_uses_the_pinned_hub_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple, dict]] = []

    def fake_hub_load(*args: object, **kwargs: object) -> nn.Module:
        calls.append((args, kwargs))
        return FakeDinoBackbone(dim=384)

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    model = build_classifier("vits14", num_classes=5, emb_dim=EMB_DIM)
    ((args, kwargs),) = calls
    assert args == (model_module.DINOV2_HUB_REPO, "dinov2_vits14")
    assert kwargs["pretrained"] is False
    assert "@" not in model_module.DINOV2_HUB_REPO
    assert len(model_module.DINOV2_HUB_REPO.split(":")[1]) == 40  # full commit SHA
    assert model.classifier.out_features == 5


def test_build_classifier_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy_hub_load(*_args: object, **_kwargs: object) -> nn.Module:
        print('Downloading: "https://github.com/..."')  # noqa: T201 - mimics torch.hub
        return FakeDinoBackbone(dim=768)

    monkeypatch.setattr(torch.hub, "load", noisy_hub_load)
    build_classifier("vitb14", emb_dim=EMB_DIM)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Downloading" in captured.err


def test_load_model_restores_a_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_model: DinoClassifierWithEmbeddings,
) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save(tiny_model.state_dict(), checkpoint)
    torch.manual_seed(123)  # a fresh model gets different weights
    monkeypatch.setattr(
        model_module,
        "build_classifier",
        lambda variant: DinoClassifierWithEmbeddings(
            FakeDinoBackbone(), 3, DIM, EMB_DIM
        ),
    )
    loaded = load_model(checkpoint, device=torch.device("cpu"))
    assert not loaded.training
    for name, tensor in tiny_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)


def test_load_model_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_model: DinoClassifierWithEmbeddings,
) -> None:
    state = tiny_model.state_dict()
    state["unexpected.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "model.pth"
    torch.save(state, checkpoint)
    monkeypatch.setattr(
        model_module,
        "build_classifier",
        lambda variant: DinoClassifierWithEmbeddings(
            FakeDinoBackbone(), 3, DIM, EMB_DIM
        ),
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model(checkpoint, device=torch.device("cpu"))
