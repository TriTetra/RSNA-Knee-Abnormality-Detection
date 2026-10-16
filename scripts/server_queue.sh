#!/usr/bin/env bash
# Gece kuyruğu: Raptor-80, etiket v3, 5 fold (OOF -> etiket v4, altın OOF, test ensemble).
# Batch boyutunu boş GPU belleğine göre seçer (>=26 GB: 8, >=20 GB: 6, aksi: 4);
# çökerse aynı fold'u batch 4 ile en fazla 3 kez dener. Tamamlanan fold'lar done_<TAG> ile atlanır.
cd ~/rsna
[ -f venv/bin/activate ] && source venv/bin/activate   # numpy/torch/kaggle buradan gelir
FOLD_FILE="$(find ~/rsna/src_ds -name folds_v3_5.json | head -1)"
ARCH=coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for FOLD in 0 1 2 3 4; do
  TAG=kadirov80_lbl3_coatnet_f$FOLD
  for try in 1 2 3; do
    [ -f done_$TAG ] && break
    while [ "$(free_mb)" -lt 14000 ]; do echo "$(date +%H:%M) $TAG GPU dolu ($(free_mb) MB), bekleniyor"; sleep 300; done
    f=$(free_mb); BS=4; [ "$f" -ge 20000 ] && BS=6; [ "$f" -ge 26000 ] && BS=8; [ "$try" -gt 1 ] && BS=4
    echo "$(date +%H:%M) $TAG başlıyor (deneme $try, bs=$BS, boş=$f MB)"
    ARCH=$ARCH ARCH_SHORT=coatnet SEED=42 TAG=$TAG BS=$BS LABELS=train_labels_v3.csv \
      EXTRA_ARGS="--folds 5 --fold $FOLD --fold_file $FOLD_FILE" \
      bash server_raptor80.sh > train_$TAG.log 2>&1 && touch done_$TAG
    echo "$(date +%H:%M) $TAG sonuç: $(grep -o 'BEST GOLD macro-AUC [0-9.]*' train_$TAG.log | tail -1)"
  done
done
echo "$(date +%H:%M) kuyruk tamam"
