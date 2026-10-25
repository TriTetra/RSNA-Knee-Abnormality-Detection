#!/usr/bin/env bash
# Upload one training run's outputs (checkpoints, gold and OOF predictions) as a private Kaggle dataset.
# Usage: bash upload_fold.sh <TAG>
# Needs a Kaggle CLI >= 2.x (CLI 1.7 cannot authenticate with the newer KGAT_ tokens); set KG to its path.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null)}"
cd "${ROOT:-$HOME/rsna}"; TAG=$1; KG="${KG:-kaggle}"; OWNER="${OWNER:-kadirov0}"
DS=$OWNER/rsna-knee-srv-${TAG//_/-}; rm -rf up_$TAG; mkdir up_$TAG
cp raptor_ft_${TAG}*.pt raptor_ft_${TAG}.json raptor_gold_${TAG}*.npz raptor_oof_${TAG}*.npz up_$TAG/ 2>/dev/null
printf '{"title":"%s","id":"%s","licenses":[{"name":"CC0-1.0"}]}' "${DS#*/}" "$DS" > up_$TAG/dataset-metadata.json
ls -la up_$TAG; $KG datasets create -p up_$TAG && rm -rf up_$TAG && echo "UPLOADED: $DS"
