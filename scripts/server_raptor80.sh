#!/usr/bin/env bash
# Raptor-80'i yerel GPU sunucusunda eğitir (RTX 5090 vb.). Kaggle GPU kotası kullanmaz.
#
# Önkoşul (bir kez):
#   mkdir -p ~/rsna && cd ~/rsna
#   /home/bekhzod/miniconda3/envs/uzslr-signs/bin/python3 -m venv --system-site-packages venv
#   source venv/bin/activate && pip install -q -U kaggle
#   export KAGGLE_API_TOKEN=...        # kadirov0 hesabının token'ı (terminalde; dosyaya yazmayın)
#
# Çalıştırma (oturum kopsa da devam etsin diye nohup ile):
#   cd ~/rsna && source venv/bin/activate
#   SEED=101 LABELS=train_labels_v3.csv nohup bash server_raptor80.sh > train_s101.log 2>&1 &
#   tail -f train_s101.log
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
cd "$ROOT"
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN ortam değişkeni tanımlı değil}"

echo "== [1/4] kod + etiketler + train.csv"
if [ ! -d src_ds ]; then
  kaggle datasets download "$OWNER/rsna-knee-src" -p src_ds --unzip
  for z in src_ds/*.zip; do [ -f "$z" ] && python -c "import shutil,sys,os;shutil.unpack_archive(sys.argv[1], sys.argv[1][:-4]);os.remove(sys.argv[1])" "$z"; done
fi
mkdir -p inputs
[ -f inputs/train.csv ] || { kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p inputs;
  for z in inputs/*.zip; do [ -f "$z" ] && python -c "import shutil,sys,os;shutil.unpack_archive(sys.argv[1],'inputs');os.remove(sys.argv[1])" "$z"; done; }

echo "== [2/4] sıkıştırılmış corpus (~20 GB)"
mkdir -p corpus80_npz
for s in 0 1 2; do
  if [ ! -f "corpus80_npz/masks_s0${s}.npy" ]; then
    # eski Kaggle CLI'lar çıktıyı sayfa sayfa verir: "Next page token" kalmayana kadar devam et
    tok=""
    while :; do
      if [ -n "$tok" ]; then out=$(kaggle kernels output "$OWNER/rsna-knee-14-corpus80-npz-s${s}" -p "dl_s${s}" --page-token "$tok" 2>&1)
      else out=$(kaggle kernels output "$OWNER/rsna-knee-14-corpus80-npz-s${s}" -p "dl_s${s}" 2>&1); fi
      tok=$(echo "$out" | sed -n 's/^Next page token: //p' | tail -1)
      echo "parça ${s}: $(find "dl_s${s}" -name '*.npz' | wc -l) dosya"
      [ -z "$tok" ] && break
    done
    find "dl_s${s}" -name "*.npz" -exec mv -t corpus80_npz {} +
    find "dl_s${s}" -name "masks_s*.npy" -exec mv -t corpus80_npz {} +
    find "dl_s${s}" -name "ids_s*.npy" -exec mv -t corpus80_npz {} +
    rm -rf "dl_s${s}"
  fi
done
python - <<'PY'
import numpy as np, glob
m = np.concatenate([np.load(f) for f in sorted(glob.glob("corpus80_npz/masks_s*.npy"))])
i = np.concatenate([np.load(f, allow_pickle=True) for f in sorted(glob.glob("corpus80_npz/ids_s*.npy"))])
np.save("corpus80_npz/masks.npy", m); np.save("corpus80_npz/ids.npy", i, allow_pickle=True)
n = len(glob.glob("corpus80_npz/[0-9]*.npz"))
print("vaka:", len(m), "| npz dosyası:", n)
assert n == len(m) == len(i) == 4407, "corpus eksik"
PY

echo "== [3/4] eğitim (seed=$SEED, etiket=$LABELS)"
python - <<PY
import pandas as pd, glob
p = glob.glob("src_ds/**/$LABELS", recursive=True)[0]
pd.read_csv(p).drop(columns=["is_gold"]).to_parquet("labels_$TAG.parquet", index=False)
PY
cp "$(find src_ds -name train_knee.py | head -1)" train_knee.py
RAPTOR_NPZ_DIR="$ROOT/corpus80_npz" RAPTOR_INPUT_ROOT="$ROOT/inputs" python train_knee.py \
  --arch "$ARCH" --res 384 --epochs "$EPOCHS" --bs "$BS" \
  --k 12 --k_eval 24 --grad_ckpt --workers 12 --topk 3 --tag "$TAG" --labels "labels_$TAG.parquet" \
  --norm imagenet --seed "$SEED" --log_every 100

echo "== [4/4] Kaggle'a yükleme (private dataset)"
DS="$OWNER/rsna-knee-srv-${TAG//_/-}"
rm -rf "up_$TAG" && mkdir "up_$TAG"
cp raptor_ft_${TAG}*.pt raptor_ft_${TAG}.json raptor_gold_${TAG}*.npz "up_$TAG"/ 2>/dev/null || true
printf '{"title":"%s","id":"%s","licenses":[{"name":"CC0-1.0"}]}' "${DS#*/}" "$DS" > "up_$TAG/dataset-metadata.json"
kaggle datasets create -p "up_$TAG"
rm -rf "up_$TAG"
echo "BITTI: $DS"
