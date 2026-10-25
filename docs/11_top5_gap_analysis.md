# 11 · Top-5 solutions versus ours: a gap analysis

[← index](README.md) · sources: write-ups of the 1st, 2nd, 3rd and 5th place teams (links at the bottom)

Our result: private 0.942 (rank ≈ 1,730). Top 5: private 0.958–0.960. The gap is ~0.017 AUC. This page breaks it down
by dimension, with the numbers each team reported.

## 1. The four solutions side by side

| | 1st · Team Kangaroo | 2nd · azamat1ch et al. | 3rd · Zhengru Li et al. | 5th · Tony Li, Rashmi B, Tucker Arrants, matheus | **us** |
|---|---|---|---|---|---|
| private / public | 0.960 / 0.967+ | 0.960 / 0.969 | 0.959 / 0.968 | 0.958 / 0.969 | **0.942 / 0.958** |
| **labels** | LLM **agents** (Codex / Claude CLI) view slices with tools; "trained guide" written from gold; Astra labels; temperature calibration | report reader 0.899 → **Astra reads ~277 images + report 0.942** → teacher OOF mix 0.956 (gold) | **Astra reviews MRI + report**; finding-specific readers refine PF OA, Lateral OA, Synovitis; quantile-mapped revisions | calibrated public report labels → Opus 5.5 reports + **gpt-6.1-sol reads 10 fat-sat slices** (effusion, synovitis, Baker's) → teacher OOF (v3, v5) | public report-only label sets + our ConvNeXt / Raptor OOF |
| label gold AUC | 0.936–0.943 (labels) | 0.956 | not stated | not stated (calibration alone: +0.010 private for CoAtNet) | 0.901 (v2) / 0.922 (v3, optimistic) |
| **where the model looks** | learned **3D ROI detector** → 112 mm crop + **ROI box pooling** | knee-centred close-ups (side / front / top) | 2.5D windows on a 140 mm FOV, 5 acquisition groups | **nnU-Net bone segmentation** → 4 anatomy crops (joint, medial, lateral, PF) as extra series, ~2× pixel density | whole field only (140 mm, 336 px) |
| backbone(s) | ConvNeXt-T (main), MambaOut-B+, MaxxViT-V2-B | ConvNeXt-T, EfficientNetV2-S | BiT ResNetV2-50, ConvNeXt-S, DINOv3 ViT-B, CoAtNet | EfficientNetV2-M, ConvNeXt-Base, ConvNeXt-Tiny | CoAtNet-RMLP-2 (same family as the public base), DINOv2, ConvNeXt-S |
| input unit | **per slice**, per series | 2.5D windows, MIL | 2.5D windows + depth attention | 2.5D windows; full image + 4 crops (60 series-type tags) | 2.5D windows, MIL |
| pretraining beyond ImageNet | — | **fastMRI / fastMRI+** (MIL reader, side classifier) | OAI expert targets mixed into training | **OAI MOAKS/WORMS 327 targets**; RadImageNet distillation (M6); TotalSegmentator-MRI init for the bone model | ImageNet only |
| auxiliary targets | **lesion-box head**, **report-embedding head** (Qwen3-Embedding-8B, 1 − cosine) | PF OA cartilage-grade head (MOAKS) | OA-coupled ordinal cartilage head; 17 tissue fields (U5) | — | none |
| training | AdamW 5e-5, 8 epochs, EMA 0.999, fp16, torch.compile | 5 seeds / 3 seeds per member, ~100 rented GPU hours (~$100) | fixed **25-epoch endpoint**, EMA 0.999, OneCycle with annealing tail; no gold-based epoch selection | 25 epochs, EMA, **site folds** (20 hospital clusters) | 8 epochs, best epoch by gold, SWA top-3 |
| validation | OOF AUC against the best labels, chosen after correlating 20–30 LB scores | gold split **24 tune / 34 check** | development gold, fixed recipes | site folds; LB to check transfer | 58 gold (all of it) |
| ensemble | 5 runs, weights 2:1:1:1:1 | 6 members, hand-set rank weights; specialists for PF OA and lateral compartment | 36 checkpoints in 3 groups; experts for the weak three findings; probability blend | probability base, then per-finding rank blends with region specialists | public base + our arms, rank blend |
| external data | — | fastMRI, OAI (MOAKS) | OAI (~9,440 knees) | OAI | none (OAI blocked: LG22) |

The 7th place (BarryZhou, private 0.957) used **no external data**: three image-aware LLM reads + teacher labels + one
2.5D label-query network with 13 zoomed regions. External data was helpful, not necessary.

## 2. Where the 0.017 went, ranked

The 5th-place ablation is the cleanest decomposition we have (EfficientNetV2-M, private):

| step | private | gain |
|---|---|---|
| calibrated report labels (B1) | 0.946 | — |
| + teacher-blended labels (v5), 25 epochs | 0.954 | **+0.008** |
| + anatomy-guided crops (M1) | 0.957 | **+0.003** |
| + ConvNeXt-Base / specialists / final blend | 0.958 | +0.001 |

And the 1st-place component table (256 px, gold AUC of the 5-fold models): per-slice ConvNeXt-T 0.9319 → + ROI box
pooling 0.9330 → + 112 mm ROI crop 0.9386 → + in-box attention 0.9396 → + calibrated labels 0.9410 → + lesion and
report heads 0.9432.

Mapping this onto us:

| gap | what we did | what the top teams did | estimated cost to us |
|---|---|---|---|
| **1. Labels from the images** | report-only label sets; image models only as OOF teachers | a multimodal LLM looked at the MRI with the report | largest: label gold 0.90 vs 0.95; 5th: +0.008 private from labels |
| **2. Label semantics** | averaged sources as they were | calibrated to the host's definitions: "not addressed" ≠ negative (27% of cells, 84% of synovitis), low-grade MCL is not positive (0/5 on gold), small effusions only 10/22, hedged "yes" ≈ 0.75 right only 38%; temperature k ≈ 0.77 | 5th: calibration alone +0.010 private on CoAtNet |
| **3. Where the model looks** | whole field | ROI detector / bone segmentation / knee-centred close-ups | 5th: +0.003 private; 1st: +0.0067 gold from box pooling + crop |
| **4. Backbone family** | CoAtNet (inside the public base) | ConvNeXt, EfficientNetV2, BiT, DINOv3 | 2nd's table: ConvNeXt-T 0.959 vs CoAtNet 0.952 public |
| **5. Validation that predicts the LB** | all 58 gold for every choice | split gold, site folds, OOF against the best labels calibrated on 20–30 LB points | made us trust v3 (+0.021 gold, +0.001 LB) |
| **6. Training length and readout** | 8 epochs, best epoch on gold | 25 epochs, fixed endpoint, EMA | part of 5th's +0.008 |
| **7. Domain pretraining** | ImageNet | fastMRI, OAI MOAKS, RadImageNet | 5th: +0.001 public in two single-model tests |
| **8. Auxiliary heads** | none | lesion boxes, report embedding, cartilage grades | 1st: +0.0022 gold |
| **9. Site / hospital effects** | random study folds | hospital-cluster folds; per-hospital synovitis correction (7th) | 5th: same-site leakage inflated local AUC by 0.019 |

## 3. The single most important miss: how the labelling was paid for

We dismissed image-aware labelling because a paid API looked unaffordable. The top teams mostly did **not** pay per
token:

* **5th place:** "API calls would be quite expensive." They ran `claude -p` and `codex exec` in headless mode under
  their subscriptions, had an agent launch hundreds of sub-agents, and let agents write the prompts for the sub-agents.
* **1st place:** one non-interactive agent session per study (`codex exec --json`, `claude -p --json-schema`), a
  workspace with slice overview images, a slice-viewing tool and a geometry tool, and an answer schema that asks for
  evidence before the probability. Median 2.5 minutes and ~15 images per study.
* **2nd place:** ~277 images per exam through a frontier vision model.

We were working inside an agentic coding tool the whole time. The same headless pattern (render slice sheets per study,
run one labelling session per study with a fixed schema, pilot on the gold tune split first) was within reach and is the
first thing to try next time.

## 4. Things the winners tried that did NOT work (so we do not repeat them)

* Showing the LLM the experts' gold answers: it copied them; a rule it wrote from them scored 0.80 blind (2nd).
* More blind one-finding passes: +0.066 on the 24 tuning exams, −0.019 on the 34 held back (2nd).
* Tuned blend weights: lost 0.002–0.007 to fixed weights on held-out exams (2nd); fixed rules or fold checks only (5th).
* A close-up alone: −0.010 on Lateral Meniscus until the whole knee was added back (2nd).
* Look-alike members: a model at 0.959 public added nothing to the blend (2nd).
* More OAI was not uniformly better; study-level, 2.5D or 3D modelling did not beat per-series modelling once ROI boxes
  existed (1st, 3rd).

## 5. Frameworks, libraries and algorithms we did not use

| category | used by the winners | purpose |
|---|---|---|
| labelling agents | Codex CLI (`codex exec`), Claude Code headless (`claude -p`), "pi" | per-study image + report labelling under a subscription |
| vision LLMs | GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, Claude Opus 5.5 | reading slices and reports |
| segmentation | **nnU-Net**, **TotalSegmentator-MRI** (initial weights), **nnInteractive** (mask drawing) | bone masks → anatomy crops |
| detection | per-slice ConvNeXt-T ROI detector with 3D fusion | ROI boxes → box pooling |
| text embedding | Qwen3-Embedding-8B | report-embedding auxiliary head |
| backbones | ConvNeXt-T/S/B, EfficientNetV2-S/M, BiT ResNetV2-50, DINOv3 ViT-B/16, MambaOut-B+, MaxxViT-V2-B | |
| pretraining data | fastMRI / fastMRI+, OAI MOAKS / WORMS, RadImageNet | domain features |
| training tricks | EMA 0.999, fixed 25-epoch endpoint, OneCycle with annealing tail, torch.compile, mass-preserving positive-weight BCE | |
| distillation | Born-Again Networks style teacher-fused labels; distilling a frozen RadImageNet ResNet-50 | |
| inference | TensorRT, C++ DICOM reading, one worker per T4, shared preprocessing for all members, per-stage time budgets | fit large ensembles in 9 h |

## 6. What we got right (for balance)

* Correct DICOM geometry, side detection and mirroring: every top team does the same.
* Rank blending with fixed weights and guarded stages.
* Teacher-student OOF mixing: every top team uses it; ours failed only because the base labels were report-only.
* The decision not to chase public-LB tuning; finals of two different kinds.

## Sources

* 1st: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution
* 2nd: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an (code: https://github.com/azamat1ch/rsna-knee-2nd-place)
* 3rd: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/3rd-place-solution-rsna-knee-abnormality-detectio
* 5th: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution
* 7th: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/7th-place-solution-with-no-external-training-data
