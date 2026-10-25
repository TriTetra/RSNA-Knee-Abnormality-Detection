"""Render the figures of the knowledge base (docs/assets/*.svg) from our recorded results.

Every number below comes from our Kaggle submission history or our training logs (see docs/07_experiment_log.md).
Usage: python scripts/make_doc_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.dpi": 110})
BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", facecolor="white")  # readable in GitHub dark mode
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Leaderboard journey: every submission, public and private --------------------------------------------
subs = [  # (date, label, public, private)
    ("10-06", "old pipeline\n(silent fallback)", 0.500, 0.500),
    ("10-13", "public 0.957 blend", 0.956, 0.940),
    ("10-14", "+ Raptor-44 20%", 0.953, 0.938),
    ("10-15", "Raptor-80 v2 alone", 0.928, 0.913),
    ("10-15", "+ Raptor-80 v2 15%", 0.955, 0.939),
    ("10-19", "public 0.958 base", 0.958, 0.942),
    ("10-19", "+ ours 10%", 0.956, 0.941),
    ("10-19", "+ ours 20%", 0.955, 0.939),
    ("10-19", "+ ours 30%", 0.953, 0.937),
    ("10-20", "Raptor-80 v3 alone", 0.929, 0.915),
    ("10-20", "Raptor-80 v4 alone", 0.926, 0.911),
    ("10-20", "0.958 + 2.5D reader", 0.958, 0.942),
]
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(subs))
pub = [s[2] for s in subs]
pri = [s[3] for s in subs]
ax.plot(x, pub, "o-", color=BLUE, label="public LB")
ax.plot(x, pri, "s--", color=ORANGE, label="private LB")
ax.set_xticks(x, [f"{s[0]}\n{s[1]}" for s in subs], rotation=60, ha="right", fontsize=8)
ax.set_ylim(0.90, 0.965)
ax.annotate("first submission: 0.500\n(off the chart)", xy=(0, 0.902), fontsize=8, color=RED)
ax.axhline(0.960, color=GREEN, lw=1, ls=":")
ax.text(len(subs) - 0.5, 0.9605, "1st place private 0.960", color=GREEN, fontsize=8, ha="right")
ax.set_ylabel("macro ROC AUC")
ax.set_title("Every submission: our own models never lifted the public base")
ax.legend(loc="lower right")
save(fig, "lb_journey.svg")

# 2. Blend weight sweep -------------------------------------------------------------------------------------
w = [0, 0.10, 0.20, 0.30]
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(w, [0.958, 0.956, 0.955, 0.953], "o-", color=BLUE, label="public")
ax.plot(w, [0.942, 0.941, 0.939, 0.937], "s--", color=ORANGE, label="private")
ax.set_xlabel("rank-blend weight of our Raptor-80 v3 model")
ax.set_ylabel("macro ROC AUC")
ax.set_title("A weaker member hurts at every weight")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "blend_weight_sweep.svg")

# 3. Blend threshold curves (binormal model) ----------------------------------------------------------------
d = lambda a: np.sqrt(2) * norm.ppf(a)  # noqa: E731


def blend_auc(a1, a2, rho, wt):
    w1, w2 = 1 - wt, wt
    return norm.cdf((w1 * d(a1) + w2 * d(a2)) / np.sqrt(2 * (w1 ** 2 + w2 ** 2 + 2 * w1 * w2 * rho)))


base = 0.958
alone = np.linspace(0.90, 0.965, 66)
fig, ax = plt.subplots(figsize=(6, 3.8))
for rho, c in zip([0.6, 0.7, 0.8, 0.86, 0.9], [GREEN, "#38a169", BLUE, ORANGE, RED]):
    gain = [max(blend_auc(base, a, rho, ww) for ww in np.linspace(0, 0.6, 61)) - base for a in alone]
    ax.plot(alone, gain, color=c, label=f"ρ = {rho}")
ax.axhline(0.002, color=GREY, ls=":")
ax.text(0.901, 0.0024, "+0.002 (about one LB step)", color=GREY, fontsize=8)
ax.axvline(0.929, color=RED, ls="--", lw=1)
ax.text(0.9295, -0.0003, "our model\n(0.929)", color=RED, fontsize=8)
ax.set_xlabel("standalone public AUC of the new member")
ax.set_ylabel("best gain over the 0.958 base")
ax.set_title("How strong must a new blend member be?")
ax.set_ylim(-0.0005, 0.008)
ax.legend(fontsize=8, ncol=2)
save(fig, "blend_threshold.svg")

# 4. Gold-58 versus leaderboard -----------------------------------------------------------------------------
pts = [  # (label, gold AUC, public LB alone)
    ("Raptor-80 v2 (2 seeds)", 0.9097, 0.928),
    ("Raptor-80 v3", 0.9205, 0.929),
    ("Raptor-80 v4", 0.9186, 0.926),
]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for lab, g, lb in pts:
    ax.scatter(g, lb, s=50, color=BLUE)
    ax.annotate(lab, (g, lb), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("gold AUC (58 expert-labelled studies)")
ax.set_ylabel("public LB, model alone")
ax.set_xlim(0.905, 0.925)
ax.set_ylim(0.922, 0.934)
ax.set_title("Gold +0.011, leaderboard +0.001: gold did not predict the LB")
ax.grid(alpha=0.3)
save(fig, "gold_vs_lb.svg")

# 5. Label quality on the 58 gold studies -------------------------------------------------------------------
labels = [
    ("Qwen2.5-7B (letters)", 0.797, GREY),
    ("Qwen3.5-27B (letters)", 0.828, GREY),
    ("pilkwang v2", 0.8704, GREY),
    ("Qwen3.5-27B (0-100 scores)", 0.881, GREY),
    ("Claude Opus 4.5 (public)", 0.8919, GREY),
    ("steven v4 (public)", 0.8927, GREY),
    ("ours v2: 3 readers", 0.9014, BLUE),
    ("ours v3: + image OOF", 0.9223, BLUE),
    ("2nd place: report-only LLM", 0.899, ORANGE),
    ("2nd place: LLM reads images", 0.942, ORANGE),
    ("7th place: 3 image-aware reads", 0.949, ORANGE),
    ("2nd place: + teacher OOF", 0.956, ORANGE),
]
fig, ax = plt.subplots(figsize=(7, 4.6))
y = np.arange(len(labels))
ax.barh(y, [v for _, v, _ in labels], color=[c for _, _, c in labels])
ax.set_yticks(y, [n for n, _, _ in labels], fontsize=8)
ax.set_xlim(0.78, 0.97)
for i, (_, v, _) in enumerate(labels):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("macro AUC against the 58 gold labels")
ax.set_title("Label quality: report readers plateau at ~0.90; image-aware readers reach 0.94-0.96")
save(fig, "label_quality.svg")
