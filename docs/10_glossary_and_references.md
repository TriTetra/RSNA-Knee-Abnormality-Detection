# 10 · Glossary and references

[← index](README.md)

## Glossary

| term | meaning |
|---|---|
| ROC AUC | how well a score ranks positives above negatives (0.5 random, 1.0 perfect); only the order matters |
| macro AUC | the mean AUC over the 12 findings; the competition metric |
| public / private LB | leaderboard on ~30% / ~70% of the hidden test set; private decides the final ranking |
| gold studies | the 58 training studies with radiologist labels |
| soft / pseudo label | a 0–1 target derived from a report or from another model's predictions |
| OOF (out-of-fold) | predictions for studies a fold model did not train on |
| teacher–student | train a model (teacher), mix its OOF predictions into the labels, train a new model (student) |
| bootstrap / CI | resample studies to estimate how much a metric moves by chance; CI = confidence interval |
| DICOM, series, slice | MR file format; one acquisition (3–14 per study); one 2D image of a series |
| sagittal / coronal / axial | side / front / top views |
| fluid-sensitive, fat suppression | PD/T2 sequences where fluid is bright; nulling the fat signal reveals marrow oedema |
| LPS coordinates | DICOM patient frame: +x left, +y posterior, +z head |
| ImagePositionPatient / ImageOrientationPatient | position and orientation of a slice in space; give the true slice order |
| laterality | right or left knee |
| FOV | field of view in mm |
| slot | a fixed place for one series type (plane × sequence) in the model input |
| 2.5D window | three adjacent slices fed as RGB channels to a 2D network |
| MIL / attention pooling | weighting many windows into one study-level prediction, here one attention head per finding |
| ROI | region of interest; a crop around the anatomy of a finding |
| backbone | the pretrained image network (ConvNeXt, CoAtNet, EfficientNet, DINOv2) |
| SWA | averaging the weights of several epochs |
| rank blend | averaging per-finding rank percentiles of several models |
| ρ (rho) | correlation of two models' errors; high ρ means they make the same mistakes |
| Raptor | dreaddevelopment's public family: CoAtNet-384 + 2.5D windows + per-finding attention MIL |
| R6 / R7 / S6 | heliosli's and nartaa's compact ConvNeXt-small bundles used in public blends |
| JEV / steven / pilkwang / Opus | public label sets (see references) |
| OAI / MOAKS | Osteoarthritis Initiative MRI data and its semi-quantitative expert MRI scores |
| code competition | the submitted notebook is re-run on the hidden test set by Kaggle (≤ 9 h, offline) |

## References

### This competition
1. RSNA Knee Abnormality Detection, competition page, data and rules; host rulings on commercial LLMs (discussion 733965)
   and external data (discussion 743416). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
2. 1st place solution (tascj, Team Kangaroo). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution
3. 1st place solution, second author (sakaku). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474
4. 2nd place solution (azamat1ch). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an
5. 7th place solution, no external data (BarryZhou). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/7th-place-solution-with-no-external-training-data
6. 499th place, bronze (Jay Gautam). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/499th-place-bronze-0-943-private-4-family-rank
7. Danial Zakaria (nartaa), "Reaching 0.949 with one CoAtNet". https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792
8. "Best single-model score" thread. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
9. hengck23, "Tricks for the competition". https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861
10. leoprovorov, "RSNA Knee 0.957 Explained" (error correlation, family ceiling, LB noise). https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl
11. varshithakolupuri, "Real scores of every public notebook" and "What the top notebooks are made of". https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook
12. heliosli, public blends 0.957 / 0.958 / 0.9586 (our finals). https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters
13. goodpjw2008, 2.5D ConvNeXt MIL reader and blend measurements. https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958
14. dreaddevelopment, Raptor training and inference code, public corpus. https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model
15. lavinwins, "Train your own model, the only lever left". https://www.kaggle.com/code/lavinwins/top-1-gold
16. Our public summary notebook. https://www.kaggle.com/code/kadirov0/rsna-knee-what-moved-the-leaderboard

### Label sources
17. stevenleehans, `rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`).
18. pilkwang, `rsna-knee-llm-labels` (`report_labels_v2.csv`).
19. NTejas-1, Claude Opus 4.5 labels (MIT). https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection
20. riadmohamed42, `jev-knee-labels` (JEV HYBRID) and "jev-knee-labels-verification".

### Related competitions and papers
21. RSNA 2024 Lumbar Spine Degenerative Classification, 1st place (Avengers): two-stage localise-then-classify,
    attention MIL, small backbones. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
22. RSNA 2023 Abdominal Trauma Detection, 1st place (Team Oxygen): 2.5D CNN + GRU, auxiliary segmentation loss.
23. RSNA 2025 Intracranial Aneurysm Detection, results: https://www.rsna.org/news/2025/november/2025-ai-challenge-winners
24. Jaiswal et al., RoS-KD: robust stochastic knowledge distillation for noisy medical imaging. https://arxiv.org/abs/2210.08388
25. Sambara et al., 3DReasonKnee (arXiv 2510.20967): ROI annotations on OAI knees, CC BY 4.0.

### Software
timm (ConvNeXt, CoAtNet, EfficientNet, DINOv2 weights), PyTorch, pydicom, pylibjpeg / openjpeg / python-gdcm,
scikit-learn, pandas, matplotlib, Kaggle CLI 2.x, Ollama.
