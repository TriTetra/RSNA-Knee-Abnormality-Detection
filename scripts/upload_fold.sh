#!/usr/bin/env bash
# Bir eğitimin çıktılarını (checkpoint, gold ve OOF tahminleri) private Kaggle dataset'i olarak yükler.
# Kullanım: bash upload_fold.sh <TAG>   (KAGGLE_API_TOKEN ortamda tanımlı olmalı)
# Not: venv'deki kaggle 1.7.4.5 yeni token biçimiyle yükleme yapamıyor; py3.11 ortamındaki 2.x CLI kullanılır.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null)}"   # token dosyadan
cd ~/rsna; TAG=$1; KG=/home/bekhzod/miniconda3/envs/kgl/bin/kaggle
DS=kadirov0/rsna-knee-srv-${TAG//_/-}; rm -rf up_$TAG; mkdir up_$TAG
cp raptor_ft_${TAG}*.pt raptor_ft_${TAG}.json raptor_gold_${TAG}*.npz raptor_oof_${TAG}*.npz up_$TAG/ 2>/dev/null
printf '{"title":"%s","id":"%s","licenses":[{"name":"CC0-1.0"}]}' "${DS#*/}" "$DS" > up_$TAG/dataset-metadata.json
ls -la up_$TAG; $KG datasets create -p up_$TAG && rm -rf up_$TAG && echo "YUKLENDI: $DS"
