"""PollenFormerUCO network: DINOv2 backbone, projection embedding, classifier."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import nn

from pollenformer.config import DEVICE, EMB_DIM, NUM_CLASSES

logger = logging.getLogger(__name__)

#: DINOv2 hub code pinned to a commit: ``torch.hub`` runs this repository's Python at
#: load time.
DINOV2_HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"

DinoVariant = Literal["vitb14", "vits14"]
#: CLS-token width of each DINOv2 variant.
FEATURE_DIMS: dict[str, int] = {"vitb14": 768, "vits14": 384}


class DinoClassifierWithEmbeddings(nn.Module):
    """DINOv2 backbone whose CLS token feeds a projection and a linear classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feat_dim: int,
        emb_dim: int | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        """Wrap a DINOv2 backbone.

        Args:
            backbone: Module whose ``forward_features`` returns ``x_norm_clstoken``.
            num_classes: Number of output classes.
            feat_dim: Width of the backbone CLS token.
            emb_dim: Projection size; ``None`` or ``feat_dim`` keeps the CLS token
                as is.
            freeze_backbone: Disable gradients for the backbone parameters.
        """
        super().__init__()
        use_proj = emb_dim is not None and emb_dim != feat_dim
        self.embedding_layer: nn.Module = (
            nn.Linear(feat_dim, emb_dim) if use_proj else nn.Identity()
        )
        final_emb_dim = int(emb_dim) if use_proj else feat_dim

        self.backbone = backbone
        self.classifier = nn.Linear(final_emb_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._attn_feats: list[torch.Tensor] = []
        self._attn_hooks: list[torch.utils.hooks.RemovableHandle] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return classifier logits and projection embeddings for a batch.

        Args:
            x: ``(B, 3, H, W)`` normalized images.

        Returns:
            ``(logits, embeddings)`` of shapes ``(B, num_classes)`` and ``(B,
            emb_dim)``.
        """
        cls_token = self.backbone.forward_features(x)["x_norm_clstoken"]
        emb = self.embedding_layer(cls_token)
        return self.classifier(emb), emb

    def unfreeze_last_blocks(self, num_blocks: int) -> None:
        """Re-enable gradients for the last transformer blocks of the backbone.

        Args:
            num_blocks: How many trailing blocks to unfreeze; capped at the block count.
        """
        blocks = self.backbone.blocks
        if num_blocks <= 0:
            return
        if num_blocks > len(blocks):
            logger.warning(
                "Unfreezing all %d blocks (requested %d)", len(blocks), num_blocks
            )
            num_blocks = len(blocks)
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def _attention_hook(self) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        """Build a hook that recomputes and stores a block's softmax attention."""

        def hook(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]  # (B, N, C)
            batch, tokens, channels = x.shape
            qkv = module.qkv(x).reshape(
                batch, tokens, 3, module.num_heads, channels // module.num_heads
            )
            q, k = qkv.permute(2, 0, 3, 1, 4)[:2]  # each (B, heads, N, head_dim)
            attn = ((q @ k.transpose(-2, -1)) * module.scale).softmax(dim=-1)
            self._attn_feats.append(attn.detach())

        return hook

    def register_attn_hooks(self) -> None:
        """Capture every backbone block's attention map on the next forward pass."""
        self.remove_attn_hooks()
        self._attn_feats = []
        for block in self.backbone.blocks:
            self._attn_hooks.append(
                block.attn.register_forward_hook(self._attention_hook())
            )

    def remove_attn_hooks(self) -> None:
        """Remove the attention hooks registered by :meth:`register_attn_hooks`."""
        for handle in self._attn_hooks:
            handle.remove()
        self._attn_hooks = []

    def get_attn_maps(self) -> list[torch.Tensor]:
        """Return the ``(B, heads, N, N)`` attention maps of the last forward pass."""
        return self._attn_feats


def build_classifier(
    variant: DinoVariant = "vitb14",
    *,
    num_classes: int = NUM_CLASSES,
    emb_dim: int = EMB_DIM,
    pretrained: bool = False,
) -> DinoClassifierWithEmbeddings:
    """Create the classifier around a DINOv2 backbone from ``torch.hub``.

    Args:
        variant: DINOv2 size: ``vitb14`` (PollenFormerUCO) or ``vits14``.
        num_classes: Number of output classes.
        emb_dim: Projection size.
        pretrained: Download the self-supervised DINOv2 weights. Not needed when a full
            PollenFormerUCO checkpoint is loaded afterwards.

    Returns:
        The classifier, on CPU, with a frozen backbone.
    """
    # torch.hub prints its first-time download notice to stdout; keep stdout for program
    # output (e.g. the JSON of `pollenformer predict`).
    with contextlib.redirect_stdout(sys.stderr):
        backbone = torch.hub.load(
            DINOV2_HUB_REPO,
            f"dinov2_{variant}",
            pretrained=pretrained,
            trust_repo=True,
            skip_validation=True,
            verbose=False,
        )
    return DinoClassifierWithEmbeddings(
        backbone=backbone,
        num_classes=num_classes,
        feat_dim=FEATURE_DIMS[variant],
        emb_dim=emb_dim,
    )


def load_model(
    checkpoint_path: Path,
    *,
    device: torch.device = DEVICE,
    variant: DinoVariant = "vitb14",
) -> DinoClassifierWithEmbeddings:
    """Build the classifier and load a PollenFormerUCO checkpoint into it.

    Args:
        checkpoint_path: ``state_dict`` saved with ``torch.save``.
        device: Device to move the model to.
        variant: DINOv2 size the checkpoint was trained with.

    Returns:
        The model in evaluation mode.
    """
    model = build_classifier(variant)
    # weights_only: never unpickle arbitrary objects from a downloaded file.
    state_dict = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    logger.info("Loaded %s on %s", checkpoint_path, device)
    return model.eval()
