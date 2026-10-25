"""Generate submission variants: the public 0.958 recipe (stage A) + our Raptor-80 arms (stage B) + rank blend.

Stage A is heliosli's public 0.958 notebook, unchanged (cells 0-8). Stages B/C come from notebook 10
(cells 8-11). A variant only changes the Raptor-80 arms and the blend weight `w` of our group.
`w = 0` reproduces the public base; `w = 1` scores our models alone.

Usage: python scripts/make_blend_variants.py a [b c ...]   -> notebooks/submission/10s<key>_blend_kaggle.ipynb
Push:  python scripts/kaggle_push.py kernel 10sa

Leaderboard results of these variants (public / private): a 0.958/0.942, b 0.956/0.941, c 0.955/0.939,
d 0.953/0.937, e 0.955/0.939, f 0.929/0.915, g 0.926/0.911, h fell back to the base (arms did not fit the time budget).
Only 10sa is kept in the repo; the other checkpoints were deleted after the competition.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "notebooks" / "submission"
SRC = SUB / "10_final_blend_kaggle.ipynb"                     # stage B/C cells (8-11)
BASE = SUB / "_public" / "heliosli_blend_mine_0958.ipynb"     # stage A: public 0.958 (cells 0-8)


def arm(name, w=1.0):
    return {"file": f"{name}.pt", "fallback": f"{name}_swa.pt", "norm": "imagenet", "w": w}


S101 = arm("raptor_ft_kadirov80_lbl3_s101")                  # "12v3": Raptor-80, labels v3, gold 0.9205
VARIANTS = {
    "a": dict(arms=[S101], w=0.0, note="public 0.958 base, unchanged (safe final)"),
    "b": dict(arms=[S101], w=0.10, note="0.958 + Raptor-80 v3 at 10%"),
    "c": dict(arms=[S101], w=0.20, note="0.958 + Raptor-80 v3 at 20%"),
    "d": dict(arms=[S101], w=0.30, note="0.958 + Raptor-80 v3 at 30%"),
    "e": dict(arms=[arm("raptor_ft_kadirov80_lbl3_s101", 0.7), arm("raptor_ft_kadirov80_lbl3_dinov2_s202", 0.3)],
              w=0.20, note="0.958 + (Raptor-80 v3 0.7 + DINOv2 0.3) at 20%"),
    "f": dict(arms=[S101], w=1.0, note="Raptor-80 v3 alone"),
    "g": dict(arms=[arm("raptor_ft_kadirov80_lbl4_coatnet_s303")], w=1.0, note="Raptor-80 v4 alone"),
    "h": dict(arms=[S101, arm("raptor_ft_kadirov80_lbl3_coatnet_s404"), arm("raptor_ft_kadirov80_lbl3_coatnet_s505")],
              w=1.0, note="three Raptor-80 v3 seeds alone"),
}


def make(key: str, v: dict) -> Path:
    ours = json.loads(SRC.read_text(encoding="utf-8"))
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"][:9] + ours["cells"][8:12]     # A (0.958) + B settings + B + B2 + C
    cell = nb["cells"][9]
    s = "".join(cell["source"])
    a = s.index('"arms": [{"file": "raptor_ft_kadirov80_lbl2.pt"')
    b = s.index("]},", a) + 3
    s = s[:a] + '"arms": ' + json.dumps(v["arms"]) + "}," + s[b:]
    s = s.replace('W_GROUPS = {"r80": 0.15, "r44": 0.0}', f'W_GROUPS = {{"r80": {v["w"]}, "r44": 0.0}}')
    assert f'"r80": {v["w"]}' in s and "kadirov80_lbl" in s
    cell["source"] = s.splitlines(keepends=True)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"# 10s{key} · {v['note']}\n\n",
        "Stage A: heliosli's public 0.958 notebook (cells unchanged, credits below). ",
        "Stages B/C: our Raptor-80 models and the final rank blend (see notebook 10).\n"]})
    out = SUB / f"10s{key}_blend_kaggle.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["a"]):
        print(make(k, VARIANTS[k]).name, "|", VARIANTS[k]["note"])
