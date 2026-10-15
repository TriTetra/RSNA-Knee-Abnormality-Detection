#!/usr/bin/env bash
# Gece kuyruğu: eğitimleri sırayla çalıştırır; her birinden önce GPU'nun boşalmasını bekler.
# Kullanım (tmux içinde): bash server_queue.sh
#
# Raptor-80, etiket v3, 5 fold. Her fold:
#   - OOF tahmini verir (ikinci tur öğretmen-öğrenci = etiket v4 için),
#   - 58 altın vakada gerçek OOF değerlendirmesi sağlar,
#   - test ensemble'ının bir üyesi olur.
cd ~/rsna
FREE_MB_NEEDED=26000        # batch 8 CoAtNet-384 ~24 GB ister
FOLD_FILE="$(find ~/rsna/src_ds -name folds_v3_5.json | head -1)"
[ -n "$FOLD_FILE" ] || { echo "folds_v3_5.json bulunamadı"; exit 1; }
JOBS=(
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 42 0"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 42 1"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 42 2"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 42 3"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 42 4"
)
for job in "${JOBS[@]}"; do
  read -r ARCH SHORT SEED FOLD <<< "$job"
  TAG="kadirov80_lbl3_${SHORT}_f${FOLD}"
  if [ -f "done_${TAG}" ]; then echo "[$TAG] zaten bitti, atlanıyor"; continue; fi
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$FREE_MB_NEEDED" ] && break
    echo "$(date +%H:%M) [$TAG] GPU boş bellek ${free} MB < ${FREE_MB_NEEDED} MB, bekleniyor…"
    sleep 300
  done
  echo "$(date +%H:%M) [$TAG] başlıyor"
  if ARCH="$ARCH" ARCH_SHORT="$SHORT" SEED="$SEED" TAG="$TAG" BS=8 LABELS=train_labels_v3.csv \
       EXTRA_ARGS="--folds 5 --fold $FOLD --fold_file $FOLD_FILE" \
       bash server_raptor80.sh > "train_${TAG}.log" 2>&1; then
    touch "done_${TAG}"
    echo "$(date +%H:%M) [$TAG] BİTTİ: $(grep -o 'BEST GOLD macro-AUC [0-9.]*' "train_${TAG}.log" | tail -1)"
  else
    echo "$(date +%H:%M) [$TAG] HATA (log: train_${TAG}.log), sıradakine geçiliyor"
  fi
done
echo "kuyruk tamam"
