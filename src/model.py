"""Vaka düzeyi model: 2D omurga + slot/konum gömmesi + pencereler arası bağlam + bulguya
özgü dikkat havuzlama (attention MIL).

Akış (bir vaka için):
    pencereler [N,3,H,W] --omurga--> öznitelikler [N,D]
        + slot gömmesi (hangi düzlem/sekans)  + konum gömmesi (yığının neresi)
        --1 katman Transformer--> pencereler birbirini görür (düzlemler arası bağlam)
        --12 başlı dikkat havuzlama--> her bulgu kendi önemli pencerelerine bakar
        --> 12 logit

Her bulgu için ayrı dikkat başı olmasının sebebi: ACL sagitalde, MCL koronalde,
PF OA aksiyalde en iyi görülür; tek bir ortalama havuzlama bu sinyali sulandırır.
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
    """Maskeli, bulguya özgü dikkat havuzlama + ortalama havuzlama artığı."""

    def __init__(self, dim: int, num_classes: int = NUM_CLASSES, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, num_classes))
        self.drop = nn.Dropout(dropout)
        # Bulgu başına ayrı doğrusal sınıflandırıcı: (C, D) ağırlık
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.mean_head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B,N,D], mask: [B,N]
        scores = self.att(x).transpose(1, 2)  # [B,C,N]
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        w = torch.softmax(scores, dim=-1).nan_to_num(0.0)  # boş vaka -> 0
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
        nn.init.zeros_(self.slot_emb.weight)  # başlangıçta omurga özniteliklerini bozmasın
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
        valid = images[mask]  # yalnız gerçek pencereler omurgadan geçer
        if len(valid):
            feats[mask] = self.backbone(normalize_imagenet(valid)).to(feats.dtype)
        x = feats + self.slot_emb(slots) + self.pos_mlp(pos[..., None])
        x = self.norm(x)
        # Hiç penceresi olmayan vaka (okunamamış) tamamen maskeli kalırsa softmax/attention
        # NaN üretir; ilk konumu açık tutarak sonlu bir çıktı garanti ederiz. Bu vakaların
        # tahmini inference'ta zaten yok sayılır (bkz. src/inference.py).
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
    """Checkpoint'i yükler. Anahtar uyuşmazlığı HATA verir (strict=True) - sessiz değil."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()
