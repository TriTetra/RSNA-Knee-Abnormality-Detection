#!/usr/bin/env bash
# Gece kuyruğu: eğitimleri sırayla çalıştırır; her birinden önce GPU'nun boşalmasını bekler.
# Kullanım (tmux içinde): bash server_queue.sh
cd ~/rsna
FREE_MB_NEEDED=26000        # batch 8 CoAtNet-384 ~24 GB ister
JOBS=(
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 202"
  "convnext_small.fb_in22k_ft_in1k_384 cnxs 202"
  "tf_efficientnetv2_s.in21k_ft_in1k effv2s 202"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 303"
  "convnext_small.fb_in22k_ft_in1k_384 cnxs 303"
  "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k coatnet 404"
)
for job in "${JOBS[@]}"; do
  read -r ARCH SHORT SEED <<< "$job"
  TAG="kadirov80_lbl3_${SHORT}_s${SEED}"
  if [ -f "done_${TAG}" ]; then echo "[$TAG] zaten bitti, atlanıyor"; continue; fi
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$FREE_MB_NEEDED" ] && break
    echo "$(date +%H:%M) [$TAG] GPU boş bellek ${free} MB < ${FREE_MB_NEEDED} MB, bekleniyor…"
    sleep 300
  done
  echo "$(date +%H:%M) [$TAG] başlıyor"
  if ARCH="$ARCH" ARCH_SHORT="$SHORT" SEED="$SEED" TAG="$TAG" BS=8 LABELS=train_labels_v3.csv \
       bash server_raptor80.sh > "train_${TAG}.log" 2>&1; then
    touch "done_${TAG}"; echo "$(date +%H:%M) [$TAG] BİTTİ: $(grep -o 'BEST GOLD macro-AUC [0-9.]*' train_${TAG}.log | tail -1)"
  else
    echo "$(date +%H:%M) [$TAG] HATA (log: train_${TAG}.log), sıradakine geçiliyor"
  fi
done
echo "kuyruk tamam"
