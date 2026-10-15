#!/usr/bin/env bash
# Corpus80 parça 1 ve 2'yi 30'ar küçük tar arşivi (~330 MB) olarak indirir (notebook 16).
# Bağlantı koparsa yalnızca o küçük arşiv baştan iner. Bütünlük: tar -tf ile kontrol.
# Bitince corpus tamamsa (4407 npz) gece kuyruğunu (server_queue.sh) başlatır.
#
# Çalıştırma: cd ~/rsna && setsid nohup bash server_get_tar30.sh > get30.log 2>&1 &
cd ~/rsna
KG="${KG:-/home/bekhzod/miniconda3/envs/kgl/bin/kaggle}"
OWNER="${OWNER:-kadirov0}"
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN ortam değişkeni tanımlı değil}"
mkdir -p dl30 corpus80_npz

# parça 1'in ilk 10 küçük arşivi (p00-p09) = eski büyük part0 (zaten inmiş)
JOBS="$(for p in $(seq -w 10 29); do echo "1 $p"; done; for p in $(seq -w 0 29); do echo "2 $p"; done)"

for s in 1 2; do
  until $KG kernels status "$OWNER/rsna-knee-16-corpus80-tar30-s$s" 2>&1 | grep -q COMPLETE; do
    echo "$(date +%H:%M) kernel s$s henüz bitmedi, bekleniyor"; sleep 60
  done
  for f in masks_s0$s.npy ids_s0$s.npy; do
    [ -s "corpus80_npz/$f" ] && continue
    timeout 300 $KG kernels output "$OWNER/rsna-knee-16-corpus80-tar30-s$s" -p dl30 --file-pattern "^$f\$" >/dev/null 2>&1
    [ -s "dl30/$f" ] && mv "dl30/$f" corpus80_npz/
  done
done

echo "$JOBS" | while read -r s p; do
  f="c80_s${s}_p${p}.tar"
  [ -f "dl30/ok_$f" ] && continue
  for try in 1 2 3 4 5 6 7 8; do
    rm -f "dl30/$f"
    timeout 900 $KG kernels output "$OWNER/rsna-knee-16-corpus80-tar30-s$s" -p dl30 --file-pattern "^$f\$" >/dev/null 2>&1
    n=$(tar -tf "dl30/$f" 2>/dev/null | wc -l)
    if [ "$n" -ge 40 ]; then
      tar -xf "dl30/$f" -C corpus80_npz && rm -f "dl30/$f" && touch "dl30/ok_$f"
      echo "$(date +%H:%M) $f tamam ($n dosya) | toplam npz: $(ls corpus80_npz | grep -c '\.npz$')"
      break
    fi
    echo "$(date +%H:%M) $f eksik ($n dosya), tekrar ($try)"; sleep 20
  done
done

n=$(ls corpus80_npz | grep -c '\.npz$')
echo "$(date +%H:%M) toplam npz: $n | masks/ids: $(ls corpus80_npz | grep -cE '^(masks|ids)_s0[0-2]\.npy$')/6"
if [ "$n" -eq 4407 ]; then
  echo "corpus tamam, kuyruk başlıyor"; bash server_queue.sh > queue.log 2>&1
else
  echo "corpus eksik, log'u kontrol et"
fi
