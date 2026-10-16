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
SRC = ROOT / "notebooks" / "10_final_blend_kaggle.ipynb"          # Aşama B/C hücreleri (8-11) buradan
BASE = ROOT / "notebooks" / "_public" / "heliosli_blend_mine_0958.ipynb"  # Aşama A: heliosli 0.958 (hücre 0-8, değiştirilmeden)

S101 = {"file": "raptor_ft_kadirov80_lbl3_s101.pt", "fallback": "raptor_ft_kadirov80_lbl3_s101_swa.pt",
        "norm": "imagenet", "w": 1.0}
# 5 fold birlikte tek tam-veri modeli kadar ağırlık alır (her biri 0.2)
FOLDS = [{"file": f"raptor_ft_kadirov80_lbl3_coatnet_f{k}.pt",
          "fallback": f"raptor_ft_kadirov80_lbl3_coatnet_f{k}_swa.pt", "norm": "imagenet", "w": 0.2}
         for k in range(5)]

# Gizli testte her Raptor-80 kolu ~1 saat sürer; Aşama A da uzun olduğu için en fazla 3 kol.
BEST_FOLDS = [FOLDS[0], FOLDS[1]]   # gold'a göre seçilir (f2-f4 bitince güncellenecek)
for a in BEST_FOLDS:
    a["w"] = 0.5
# Gold (58): 12v3 0.9205; 12v3 + f0 + f1 0.9187 (fold'lar 12v3 ile r=0.96, katkı yok) -> submission'larda yalnız 12v3.
VARIANTS = {
    "a": dict(arms=[S101], w=0.0, note="heliosli 0.958 tabanı, olduğu gibi (güvenli final adayı)"),
    "b": dict(arms=[S101], w=0.10, note="0.958 + Raptor-80 v3 (12v3) %10"),
    "c": dict(arms=[S101], w=0.20, note="0.958 + Raptor-80 v3 (12v3) %20"),
    "d": dict(arms=[S101], w=0.30, note="0.958 + Raptor-80 v3 (12v3) %30"),
}


def make(key: str, v: dict) -> Path:
    ours = json.loads(SRC.read_text(encoding="utf-8"))
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"][:9] + ours["cells"][8:12]   # A (0.958) + B ayarları + B + B2 + C
    cell = nb["cells"][9]
    s = "".join(cell["source"])
    a = s.index('"arms": [{"file": "raptor_ft_kadirov80_lbl2.pt"')
    b = s.index("]},", a) + 3
    s = s[:a] + '"arms": ' + json.dumps(v["arms"], ensure_ascii=False) + "}," + s[b:]
    s = s.replace('W_GROUPS = {"r80": 0.15, "r44": 0.0}', f'W_GROUPS = {{"r80": {v["w"]}, "r44": 0.0}}')
    assert f'"r80": {v["w"]}' in s and "lbl3" in s
    cell["source"] = s.splitlines(keepends=True)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"# 10s{key} · {v['note']}\n\n",
        "Aşama A: heliosli'nin public 0.958 notebook'u (hücreler değiştirilmeden, kredileri aşağıda). ",
        "Aşama B/C: bizim Raptor-80 (etiket v3) modellerimiz ve final rank blend (bkz. notebooks/10).\n"]})
    out = ROOT / "notebooks" / f"10s{key}_blend_kaggle.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for k, v in VARIANTS.items():
        print(make(k, v).name, "|", v["note"])
