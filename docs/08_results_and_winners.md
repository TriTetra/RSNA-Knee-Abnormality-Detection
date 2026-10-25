# 08 · Results and winning solutions

[← index](README.md)

## Final standing

| | |
|---|---|
| final rank (preliminary private) | **≈ 1,730 / 6,425** (public 1,565) |
| finals | 10sa (0.958 recipe) and 10t (0.9586 recipe): public 0.958, private 0.942 |
| own best model alone | Raptor-80 v3: public 0.929, private 0.915 |
| 1st place private | 0.960 · 50th 0.951 · bronze cut-off about rank 642 (top 10%); rank 499 had 0.943 |

Public and private moved together for every submission. The ~1,270 teams that shipped the public 0.958 recipe all
landed around private 0.941–0.942, so private ranks inside that block were decided by small differences and timing.

## What the winners did

A dimension-by-dimension comparison with the 1st, 2nd, 3rd and 5th place is in [11 · Top-5 gap analysis](11_top5_gap_analysis.md).

| place | team | private | the decisive idea |
|---|---|---|---|
| 1 | Team Kangaroo (tascj, liushuzhi) | 0.960 | **agentic labelling**: one non-interactive CLI agent session per study with tools to view slices; outputs evidence, then probabilities, plus **3D ROI boxes** per finding (751 studies) and per-slice lesion boxes (1,800 studies). A ROI detector (ConvNeXt-T, mean IoU 0.71) drives **box pooling** in a per-slice ConvNeXt-T. Left knees mirrored, a small side classifier for missing tags. |
| 2 | azamat1ch | 0.960 | **vision-LLM soft labels**: report reader 0.899 → LLM reading every slice with the report 0.942 → fixes + teacher OOF 0.956 on gold. Six 2.5D MIL members (ConvNeXt-T, EfficientNetV2-S, fastMRI-pretrained reader), knee-centred close-ups, hand-set rank weights; two specialists for PF OA and the lateral compartment. |
| 3 | Zhengru Li et al. | 0.959 | Astra reviews MRI + report; finding-specific readers for PF OA / Lateral OA / Synovitis; 36 checkpoints (BiT, ConvNeXt, DINOv3, CoAtNet) in fixed group budgets; OAI targets. |
| 5 | Tony Li, Rashmi B, Tucker Arrants, matheus | 0.958 | calibrated report labels + vision-LLM reads + teacher OOF; nnU-Net bone segmentation for 4 anatomy crops; EfficientNetV2-M / ConvNeXt; region specialists; labelling with headless `claude -p` / `codex exec`. |
| 7 | BarryZhou | 0.957 | **no external data**. Three image-aware LLM reads (structured with 24 graded attributes, image-only, verification), per-hospital Synovitis correction, cross-model teacher OOF (gold 0.952). One 2.5D label-query network (12×12 tokens per window, 3D positions in a common knee frame, 13 zoomed regions), 4 backbones. |
| 32 | zosamu | — | "Better teachers, better places to look": teacher labels + ROI crops |
| 129 (silver) | s0chn1y | — | LLM rubric labels, slice-BiGRU ConvNeXt, public blend |
| 499 (bronze) | Jay Gautam | 0.943 | public 0.958 base + S6 ConvNeXt at 20–35%; Qwen3-4B yes/no logit probing for labels |

## Where we stood relative to them

![Label quality](assets/label_quality.svg)

| | us | winners |
|---|---|---|
| label source | report-only LLM label sets + our image-model OOF | multimodal LLM reading the **images** with the report, then teacher OOF |
| label quality (gold) | 0.901 (v2), 0.922 (v3, optimistic) | 0.947–0.956 |
| model | Raptor CoAtNet-384 (same family as the public base) | 2.5D ConvNeXt-T / EfficientNetV2-S, ROI or close-up crops |
| own model, public | 0.929 | 0.964–0.969 |

## Why we fell behind

1. **We did not have the images read for labels.** It was in our idea pool as "local VLM labelling", but paid APIs were
   not available to us and our local text-only LLM trials were weak, so we never ran an image-aware pilot. That was the
   lever.
2. **Our decisions trusted the 58 gold studies.** Labels v3 gained +0.021 on gold and +0.001 on the leaderboard.
3. **We trained the same family as the public base.** Raptor-CoAtNet was both inside the base and the weakest
   architecture in the 2nd-place team's table.
4. **Infrastructure ate GPU time.** See [06](06_infrastructure.md).

## What worked for us

* 0.500 → 0.958 by finding silent bugs and rebuilding the best public recipe correctly.
* Correct DICOM handling (order, orientation, scale, side) and guards against silent fallbacks.
* The inference-vs-training gold check that ruled out a preprocessing mismatch.
* The blend model that explained why our models could not help, before spending more submissions.
