"""10 numaralı final blend notebook'undan submission varyantları üretir (10sa ... 10se).

Her varyant aynı Aşama A (public boru hattı, LB 0.956) + Aşama B (Raptor-80, etiket v3) yapısını
kullanır; yalnızca Raptor-80 kolları ve blend ağırlığı değişir. Varyantlar kadirov0 hesabında
çalışır; checkpoint'ler kadirov0 dataset'lerinden okunur (12v3 de kadirov0/rsna-knee-r80-lbl3-s101'e taşındı).

Kullanım: python scripts/make_blend_variants.py   ->  notebooks/10s{a..e}_blend_kaggle.ipynb
Push:     python scripts/kaggle_push.py kernel 10sa  (kaggle_push.py'deki VARIANT_KEYS)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "10_final_blend_kaggle.ipynb"

S101 = {"file": "raptor_ft_kadirov80_lbl3_s101.pt", "fallback": "raptor_ft_kadirov80_lbl3_s101_swa.pt",
        "norm": "imagenet", "w": 1.0}
# 5 fold birlikte tek tam-veri modeli kadar ağırlık alır (her biri 0.2)
FOLDS = [{"file": f"raptor_ft_kadirov80_lbl3_coatnet_f{k}.pt",
          "fallback": f"raptor_ft_kadirov80_lbl3_coatnet_f{k}_swa.pt", "norm": "imagenet", "w": 0.2}
         for k in range(5)]

VARIANTS = {
    "a": dict(arms=[S101], w=0.15, note="public + 12v3 tek model %15 (etiket v3'ün tek başına katkısı)"),
    "b": dict(arms=[S101] + FOLDS, w=0.15, note="public + Raptor-80 v3 ensemble (12v3 + 5 fold) %15"),
    "c": dict(arms=[S101] + FOLDS, w=0.25, note="public + Raptor-80 v3 ensemble %25"),
    "d": dict(arms=[S101] + FOLDS, w=0.35, note="public + Raptor-80 v3 ensemble %35"),
    "e": dict(arms=[S101] + FOLDS, w=1.0, note="yalnız Raptor-80 v3 ensemble (teşhis: tek başına LB gücü)"),
}


def make(key: str, v: dict) -> Path:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cell = nb["cells"][8]
    s = "".join(cell["source"])
    a = s.index('"arms": [{"file": "raptor_ft_kadirov80_lbl2.pt"')
    b = s.index("]},", a) + 3
    s = s[:a] + '"arms": ' + json.dumps(v["arms"], ensure_ascii=False) + "}," + s[b:]
    s = s.replace('W_GROUPS = {"r80": 0.15, "r44": 0.0}', f'W_GROUPS = {{"r80": {v["w"]}, "r44": 0.0}}')
    assert f'"r80": {v["w"]}' in s and "lbl3" in s
    cell["source"] = s.splitlines(keepends=True)
    nb["cells"][0]["source"] = [f"# 10s{key} · {v['note']}\n\n"] + nb["cells"][0]["source"]
    out = ROOT / "notebooks" / f"10s{key}_blend_kaggle.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for k, v in VARIANTS.items():
        print(make(k, v).name, "|", v["note"])
