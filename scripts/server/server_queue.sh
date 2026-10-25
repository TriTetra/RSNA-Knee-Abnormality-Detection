#!/usr/bin/env bash
# Overnight queue, round 1: Raptor-80 on labels v3 as 5 folds (out-of-fold predictions -> labels v4).
# Picks the batch size from free GPU memory (>= 26 GB: 8, >= 20 GB: 6, else 4) and retries a crashed fold
# with batch 4 at most 3 times. Finished folds leave a done_<TAG> marker and are skipped on re-runs.
cd "${ROOT:-$HOME/rsna}"
[ -f venv/bin/activate ] && source venv/bin/activate   # numpy / torch / timm come from here
FOLD_FILE="$(find src_ds -name folds_v3_5.json | head -1)"
ARCH=coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for FOLD in 0 1 2 3 4; do
  TAG=kadirov80_lbl3_coatnet_f$FOLD
  for try in 1 2 3; do
    [ -f done_$TAG ] && break
    while [ "$(free_mb)" -lt 14000 ]; do echo "$(date +%H:%M) $TAG GPU busy ($(free_mb) MB free), waiting"; sleep 300; done
    f=$(free_mb); BS=4; [ "$f" -ge 20000 ] && BS=6; [ "$f" -ge 26000 ] && BS=8; [ "$try" -gt 1 ] && BS=4
    echo "$(date +%H:%M) $TAG start (try $try, bs=$BS, free=$f MB)"
    ARCH=$ARCH ARCH_SHORT=coatnet SEED=42 TAG=$TAG BS=$BS LABELS=train_labels_v3.csv \
      EXTRA_ARGS="--folds 5 --fold $FOLD --fold_file $FOLD_FILE" \
      bash server_raptor80.sh > train_$TAG.log 2>&1 && touch done_$TAG
    echo "$(date +%H:%M) $TAG result: $(grep -o 'BEST GOLD macro-AUC [0-9.]*' train_$TAG.log | tail -1)"
  done
done
echo "$(date +%H:%M) queue finished"
