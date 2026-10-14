"""Eğitim ve değerlendirme döngüleri, fold bölme ve metrikler.

Metrik: 12 bulgunun ROC AUC'sinin ortalaması (yarışma metriği). İki ayrı ölçüm raporlanır:
* `pseudo_auc`: doğrulama vakalarının rapor-türevi etiketlerine (0.5'te ikiye bölünmüş) göre.
  Bol örnekli ama hedef LLM etiketi olduğu için iyimser/yanlı olabilir.
* `gold_auc`: yalnız radyolog etiketli altın vakalarda. Az örnek (fold başına ~12) ama
  gerçek hedefe en yakın ölçü. Tüm fold'lar bitince 58 vakanın OOF'u ile hesaplanmalı.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_COLS


def make_folds(labels: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Altın vakaları ve pozitif bulgu sayısını dengeleyen fold ataması."""
    df = labels.copy()
    n_pos = (df[TARGET_COLS] > 0.5).sum(1).clip(0, 6)
    strata = df["is_gold"].astype(int).astype(str) + "_" + n_pos.astype(str)
    df["fold"] = -1
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(df, strata)):
        df.loc[df.index[va], "fold"] = f
    return df


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for k, c in enumerate(TARGET_COLS):
        yt = y_true[:, k] > 0.5
        out[c] = float(roc_auc_score(yt, y_pred[:, k])) if 0 < yt.sum() < len(yt) else float("nan")
    vals = [v for v in out.values() if not math.isnan(v)]
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.02):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _to(batch, device):
    return (batch["images"].to(device, non_blocking=True), batch["slots"].to(device),
            batch["pos"].to(device), batch["mask"].to(device))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, gold_weight: float = 2.0,
                    grad_accum: int = 1, log_every: int = 50, amp_dtype=torch.float16) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total, count, t0 = 0.0, 0, time.time()
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"
    for step, batch in enumerate(loader):
        images, slots, pos, mask = _to(batch, device)
        y = batch["targets"].to(device)
        w = torch.where(batch["is_gold"].to(device), gold_weight, 1.0)[:, None]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images, slots, pos, mask)["logits"]
        loss = (bce(logits.float(), y) * w).mean() / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total += loss.item() * grad_accum * len(y)
        count += len(y)
        if log_every and (step + 1) % log_every == 0:
            print(f"    adım {step + 1}/{len(loader)}  kayıp {total / count:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device, amp_dtype=torch.float16) -> pd.DataFrame:
    """Olasılık tablosu döndürür (StudyInstanceUID + 12 sütun)."""
    model.eval()
    ids, preds = [], []
    for batch in loader:
        images, slots, pos, mask = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images, slots, pos, mask)["logits"]
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids += batch["study_ids"]
    out = pd.DataFrame(np.concatenate(preds), columns=TARGET_COLS)
    out.insert(0, "StudyInstanceUID", ids)
    return out


def evaluate(pred: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, float]:
    m = labels.set_index("StudyInstanceUID").loc[pred["StudyInstanceUID"]]
    y, p = m[TARGET_COLS].values, pred[TARGET_COLS].values
    res = {"pseudo_auc": macro_auc(y, p)["macro"]}
    g = m["is_gold"].values.astype(bool)
    res["gold_auc"] = macro_auc(y[g], p[g])["macro"] if g.sum() >= 5 else float("nan")
    res["n_gold"] = int(g.sum())
    return res
