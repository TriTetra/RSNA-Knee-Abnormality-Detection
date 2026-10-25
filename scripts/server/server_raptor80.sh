#!/usr/bin/env bash
# Train one Raptor-80 model on a local GPU server (e.g. RTX 5090) instead of Kaggle's GPU quota.
#
# One-time setup:
#   mkdir -p ~/rsna && cd ~/rsna
#   python3 -m venv --system-site-packages venv && source venv/bin/activate   # needs torch + timm
#   bash server_get_tar30.sh                      # 80-slice corpus -> ~/rsna/corpus80_npz (~20 GB)
#   token: ~/.kaggle/access_token (chmod 600) or KAGGLE_API_TOKEN in the environment
#
# Run (detached, survives a dropped SSH session):
#   cd ~/rsna && source venv/bin/activate
#   SEED=101 LABELS=train_labels_v3.csv setsid nohup bash server_raptor80.sh > train_s101.log 2>&1 &
#
# Environment: SEED, LABELS, ARCH, ARCH_SHORT, TAG, BS, EPOCHS, RES (384; 336 for patch-14 ViTs such as DINOv2),
#              EXTRA_ARGS (e.g. "--folds 5 --fold 0 --fold_file folds_v3_5.json"), KG (Kaggle CLI binary).
set -euo pipefail

SEED="${SEED:-101}"
LABELS="${LABELS:-train_labels_v3.csv}"
ARCH="${ARCH:-coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k}"
ARCH_SHORT="${ARCH_SHORT:-coatnet}"
TAG="${TAG:-kadirov80_lbl3_${ARCH_SHORT}_s${SEED}}"
OWNER="${OWNER:-kadirov0}"
BS="${BS:-8}"
EPOCHS="${EPOCHS:-8}"
ROOT="${ROOT:-$HOME/rsna}"
KG="${KG:-kaggle}"
cd "$ROOT"
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null)}"
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN is not set}"

echo "== [1/4] code + labels + train.csv"
if [ ! -d src_ds ]; then
  $KG datasets download "$OWNER/rsna-knee-src" -p src_ds --unzip
  for z in src_ds/*.zip; do [ -f "$z" ] && python -c "import shutil,sys,os;shutil.unpack_archive(sys.argv[1], sys.argv[1][:-4]);os.remove(sys.argv[1])" "$z"; done
fi
mkdir -p inputs
[ -f inputs/train.csv ] || { $KG competitions download -c rsna-knee-abnormality-detection -f train.csv -p inputs;
  for z in inputs/*.zip; do [ -f "$z" ] && python -c "import shutil,sys,os;shutil.unpack_archive(sys.argv[1],'inputs');os.remove(sys.argv[1])" "$z"; done; }

echo "== [2/4] corpus check"
python - <<'PY'
import numpy as np, glob
m = np.concatenate([np.load(f) for f in sorted(glob.glob("corpus80_npz/masks_s*.npy"))])
i = np.concatenate([np.load(f, allow_pickle=True) for f in sorted(glob.glob("corpus80_npz/ids_s*.npy"))])
np.save("corpus80_npz/masks.npy", m); np.save("corpus80_npz/ids.npy", i, allow_pickle=True)
n = len(glob.glob("corpus80_npz/[0-9]*.npz"))
print("studies:", len(m), "| npz files:", n)
assert n == len(m) == len(i) == 4407, "corpus incomplete: run server_get_tar30.sh first"
PY

echo "== [3/4] training (seed=$SEED, labels=$LABELS, arch=$ARCH)"
python - <<PY
import pandas as pd, glob
p = glob.glob("src_ds/**/$LABELS", recursive=True)[0]
pd.read_csv(p).drop(columns=["is_gold"]).to_parquet("labels_$TAG.parquet", index=False)
PY
cp "$(find src_ds -name train_knee.py | head -1)" train_knee.py
RAPTOR_NPZ_DIR="$ROOT/corpus80_npz" RAPTOR_INPUT_ROOT="$ROOT/inputs" python train_knee.py \
  --arch "$ARCH" --res "${RES:-384}" --epochs "$EPOCHS" --bs "$BS" \
  --k 12 --k_eval 24 --grad_ckpt --workers 12 --topk 3 --tag "$TAG" --labels "labels_$TAG.parquet" \
  --norm imagenet --seed "$SEED" --log_every 100 ${EXTRA_ARGS:-}

echo "== [4/4] upload to Kaggle (private dataset)"
# an upload failure must not mark the training as failed (a queue would otherwise retrain the same model)
bash "$ROOT/upload_fold.sh" "$TAG" || echo "UPLOAD FAILED: retry with bash upload_fold.sh $TAG"
