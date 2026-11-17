"""Test doubles and helpers: a tiny DINOv2-like backbone and image encoding."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import torch
from torch import nn

DIM = 16
HEADS = 8
EMB_DIM = 8
CLASS_NAMES = ("Olive", "Pinus", "NoPollen")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"


class FakeAttention(nn.Module):
    """Self-attention exposing ``qkv``, ``num_heads`` and ``scale`` like DINOv2."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, channels))


class FakeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = FakeAttention(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class FakeDinoBackbone(nn.Module):
    """Tiny stand-in for DINOv2: 14x14 patches, a CLS token and ``forward_features``."""

    def __init__(self, dim: int = DIM, num_heads: int = HEADS, depth: int = 2) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([FakeBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {"x_norm_clstoken": x[:, 0], "x_norm_patchtokens": x[:, 1:]}


def encode(image: Image.Image, fmt: str = "PNG", **params: object) -> bytes:
    """Encode an image to bytes in the given format."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **params)
    return buffer.getvalue()
