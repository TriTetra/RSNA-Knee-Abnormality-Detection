"""Study-level model: 2D backbone + slot/position embeddings + context across windows +
per-finding attention pooling (attention MIL).

Flow for one study:
    windows [N,3,H,W] --backbone--> features [N,D]
        + slot embedding (which plane / sequence)  + position embedding (where in the stack)
        --1-layer Transformer--> windows see each other (context across planes)
        --12-head attention pooling--> each finding attends to its own informative windows
        --> 12 logits

Why one attention head per finding: the ACL is best seen on sagittal, the MCL on coronal and PF OA on axial
images; a single average pooling would dilute these signals.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.cache import NUM_SLOTS
from src.config import NUM_CLASSES
from src.dataset import normalize_imagenet

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


class FindingAttentionPool(nn.Module):
    """Masked per-finding attention pooling + a mean-pooling residual."""

    def __init__(self, dim: int, num_classes: int = NUM_CLASSES, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, num_classes))
        self.drop = nn.Dropout(dropout)
        # One linear classifier per finding: (C, D) weights
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.mean_head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B,N,D], mask: [B,N]
        scores = self.att(x).transpose(1, 2)  # [B,C,N]
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        w = torch.softmax(scores, dim=-1).nan_to_num(0.0)  # empty study -> 0
        ctx = torch.einsum("bcn,bnd->bcd", w, x)  # [B,C,D]
        logits = (self.drop(ctx) * self.weight).sum(-1) + self.bias
        mean = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return logits + self.mean_head(self.drop(mean)), w


class KneeModel(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny.fb_in22k_ft_in1k", pretrained: bool = True,
                 num_classes: int = NUM_CLASSES, context_layers: int = 1, drop_path: float = 0.1):
        super().__init__()
        if timm is None:
            raise ImportError("timm gerekli: pip install timm")
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=drop_path)
        d = self.backbone.num_features
        self.slot_emb = nn.Embedding(NUM_SLOTS, d)
        self.pos_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, d))
        self.norm = nn.LayerNorm(d)
        nn.init.zeros_(self.slot_emb.weight)  # must not disturb the backbone features at initialisation
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)
        self.context = None
        if context_layers > 0:
            layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=d * 2, dropout=0.1,
                                               batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, context_layers, enable_nested_tensor=False)
        self.pool = FindingAttentionPool(d, num_classes)

    def forward(self, images: torch.Tensor, slots: torch.Tensor, pos: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """images [B,N,3,H,W] (0-1), slots/pos/mask [B,N]."""
        B, N = mask.shape
        feats = images.new_zeros(B, N, self.backbone.num_features)
        valid = images[mask]  # only real (unpadded) windows go through the backbone
        if len(valid):
            feats[mask] = self.backbone(normalize_imagenet(valid)).to(feats.dtype)
        x = feats + self.slot_emb(slots) + self.pos_mlp(pos[..., None])
        x = self.norm(x)
        # A study without any window (unreadable) would be fully masked and make softmax/attention return NaN;
        # keeping the first position open guarantees a finite output. Such studies are ignored at inference
        # anyway (see src/inference.py).
        mask = mask.clone()
        mask[mask.sum(1) == 0, 0] = True
        if self.context is not None:
            x = self.context(x, src_key_padding_mask=~mask)
        logits, att = self.pool(x, mask)
        return {"logits": logits, "attention": att}


def build_model(cfg: Dict, pretrained: bool = True) -> KneeModel:
    return KneeModel(backbone=cfg.get("backbone", "convnext_tiny.fb_in22k_ft_in1k"),
                     pretrained=pretrained,
                     context_layers=cfg.get("context_layers", 1),
                     drop_path=cfg.get("drop_path", 0.1))


def load_checkpoint(path, device: str = "cpu") -> KneeModel:
    """Load a checkpoint. A key mismatch raises an error (strict=True) instead of failing silently."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()
