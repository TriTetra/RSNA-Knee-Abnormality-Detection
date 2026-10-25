# Changelog

Notable changes, newest first. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Experiment numbers are in [docs/07_experiment_log.md](docs/07_experiment_log.md).

## 2026-10-25 · Post-competition clean-up

### Changed
- The whole repository is in English: code comments and messages, notebooks (02 and 03 re-executed), documentation.
- Documentation rebuilt as a knowledge base in `docs/` (11 pages, Mermaid diagrams, figures from
  `scripts/make_doc_figures.py`). The previous Turkish report files were merged into it.
- Notebooks grouped and renamed: `notebooks/` (01–06, 18), `raptor/`, `corpus/`, `submission/`.
- Server scripts moved to `scripts/server/` and made portable (no user-specific paths, token from file or environment).

### Removed
- Superseded scripts: `prepare_corpus_npz.py`, `server_get_corpus.sh`, the one-off Kaggle clean-up script.
- Superseded notebooks: Colab training (13), Qwen-7B labelling notebook (08; `scripts/llm_label.py` remains),
  3-archive tar export (15), duplicate seed runs (07, 07c, 12b, 12c), measurement variants 10r, 10v3, 10sb–10sh
  (re-creatable with `scripts/make_blend_variants.py`).

### Added
- `notebooks/18_competition_summary_kaggle.ipynb`: public summary with reusable DICOM-ordering and blend-gain code.
- `docs/08_results_and_winners.md`: final standing and the winning solutions.

## 2026-10-20 · Standalone measurements
- Submitted our models alone and the 0.9586 public recipe: Raptor-80 v3 public 0.929 / private 0.915, v4 0.926 / 0.911,
  0.9586 recipe 0.958 / 0.942. Labels v3 gave only +0.001 on the leaderboard; training stopped.
- `notebooks/raptor/17`: the inference pipeline reproduces the training-time gold AUC (no preprocessing mismatch).
- `third_party/raptor/infer_knee.py`: `RAPTOR_TEST_ROOT` for diagnostic runs.
- Fixed: the 0.9586 recipe now reads the unchanged `-w1` copy of the R6 bundle (the original dataset was overwritten).

## 2026-10-19 · Blending on the 0.958 base
- `scripts/make_blend_variants.py` + `notebooks/submission/10s*`: public 0.958 recipe (stage A) + our Raptor-80 arms.
- Weight sweep 0 / 10 / 20 / 30%: public 0.958 / 0.956 / 0.955 / 0.953; our model hurts at every weight.
- DINOv2 ViT-B (336 px) and two more CoAtNet seeds on labels v3; `RES` setting for patch-14 backbones.

## 2026-10-16 · Labels v4 and the GPU server
- Labels v4 = 50% v2 + 25% ConvNeXt OOF + 25% Raptor-80 OOF (`build_training_labels_v4`, `scripts/build_labels_v4.py`).
- Server queues (5 folds on v3, round 2 on v4), memory-adaptive batch size, retries, upload separated from training,
  token read from `~/.kaggle/access_token`, corpus download in 30 verified archives.
- Base switched to heliosli's public 0.958 recipe.

## 2026-10-15 · Raptor-80 and labels v3
- 80-slice corpus (`notebooks/corpus/11`, 3 shards) and Raptor-80 training (12, 12v3).
- Labels v3 = 50% v2 + 50% ConvNeXt 5-fold OOF: gold 0.9014 → 0.9223.
- `scripts/kaggle_push.py`: run kernels on a teammate's account (`--token-file`, `--as`).
- Leaderboard: Raptor-80 v2 alone 0.928; public 0.85 + Raptor-80 0.15 = 0.955.

## 2026-10-14 · Raptor recipe, labels v2, final blend notebook
- `third_party/raptor/train_knee.py` / `infer_knee.py`: patched public Raptor code (fp16 on T4, time budget,
  configurable geometry, multi-shard corpus).
- Labels v2 = steven v4 + Claude Opus 4.5 + JEV HYBRID: gold 0.9014.
- `notebooks/submission/10`: public blend + our Raptor groups + ConvNeXt with fixed weights and fallbacks.
- `scripts/llm_label.py`: report labelling with any OpenAI-compatible LLM server.
- Leaderboard: public blend copy 0.956 (rank 5,896 → 723); + Raptor-44 at 20% = 0.953.

## 2026-10-13 · Rebuild
- Removed the old pipeline that scored 0.500 (file-name slice order, no laterality, checkpoint key mismatch,
  silent prior fallback).
- New modular `src/` pipeline: geometric slice order, orientation, mm resampling, laterality with mirroring,
  6-slot cache, ConvNeXt multi-view MIL, guarded inference, rank blend.
- Notebooks 01–06 with explanations; offline wheels for compressed DICOMs.
