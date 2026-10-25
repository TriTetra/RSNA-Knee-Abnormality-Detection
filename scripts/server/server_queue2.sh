#!/usr/bin/env bash
# Overnight queue, round 2: full-data models on labels v4 with different backbones (different errors).
# Waits until labels v4 is on Kaggle (kadirov0/rsna-knee-labels-v4) and round 1 (server_queue.sh) has finished.
cd "${ROOT:-$HOME/rsna}" && source venv/bin/activate
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token)}"
KG="${KG:-kaggle}"
until [ -s src_ds/labels_v4/train_labels_v4.csv ]; do
  $KG datasets download kadirov0/rsna-knee-labels-v4 -p src_ds/labels_v4 --unzip >/dev/null 2>&1 \
    || { echo "$(date +%H:%M) waiting for labels v4"; sleep 120; }
done
while pgrep -f "bash server_queue.sh" >/dev/null; do echo "$(date +%H:%M) round 1 still running"; sleep 120; done
JOBS=(   # "timm architecture  short-name  seed  resolution"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 202 384"
  "convnext_small.fb_in22k_ft_in1k_384 convnext 202 384"
  "vit_base_patch14_dinov2.lvd142m dinov2 202 336"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 303 384"
)
for job in "${JOBS[@]}"; do
  read -r ARCH SHORT SEED RES <<< "$job"
  TAG=kadirov80_lbl4_${SHORT}_s${SEED}
  [ -f done_$TAG ] && continue
  echo "$(date +%H:%M) $TAG start"
  ARCH=$ARCH ARCH_SHORT=$SHORT SEED=$SEED RES=$RES TAG=$TAG BS=8 LABELS=train_labels_v4.csv \
    bash server_raptor80.sh > train_$TAG.log 2>&1 && touch done_$TAG
  echo "$(date +%H:%M) $TAG result: $(grep -o 'BEST GOLD macro-AUC [0-9.]*' train_$TAG.log | tail -1)"
done
echo "$(date +%H:%M) round 2 finished"
