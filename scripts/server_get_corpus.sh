#!/usr/bin/env bash
# Corpus80 tar arşivlerini dayanıklı biçimde indirir (bağlantı koparsa dosya bazında yeniden dener).
# Kaggle CLI >= 2.x gerekir (Python 3.11+): K=/path/to/kaggle bash server_get_corpus.sh
cd ~/rsna
K="${K:-kaggle}"
mkdir -p corpus80_npz
for s in 0 1 2; do
  for p in 0 1 2; do
    f=corpus80_s${s}_part${p}.tar
    [ -f "corpus80_npz/.ok_${f}" ] && { echo "$f zaten var"; continue; }
    for try in 1 2 3 4 5; do
      rm -rf dl
      $K kernels output kadirov0/rsna-knee-15-corpus80-tar-s$s -p dl --page-size 200 --file-pattern "^${f}\$" -q
      if tar -tf "dl/$f" >/dev/null 2>&1; then
        tar -xf "dl/$f" -C corpus80_npz && rm -rf dl && touch "corpus80_npz/.ok_${f}" && echo "$f tamam (deneme $try)" && break
      fi
      echo "$f bozuk/eksik, tekrar deneniyor ($try)"; sleep 20
    done
  done
  $K kernels output kadirov0/rsna-knee-15-corpus80-tar-s$s -p dl --page-size 200 --file-pattern 'npy$' -q && mv dl/*.npy corpus80_npz/ && rm -rf dl
  echo "parça $s tamam: $(ls corpus80_npz | grep -c npz) npz"
done
