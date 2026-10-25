#!/usr/bin/env bash
# Download the 80-slice corpus from Kaggle as 3 shards x 30 small tar archives (~330 MB each, notebook 16)
# and unpack them into corpus80_npz/. A dropped connection only re-downloads one small archive;
# each archive is verified with `tar -tf` before it is unpacked. Re-running skips finished archives.
#
# Run: cd ~/rsna && setsid nohup bash server_get_tar30.sh > get30.log 2>&1 &
cd "${ROOT:-$HOME/rsna}"
KG="${KG:-kaggle}"                     # Kaggle CLI >= 2.x (supports --file-pattern)
OWNER="${OWNER:-kadirov0}"
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null)}"
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN is not set}"
mkdir -p dl30 corpus80_npz

for s in 0 1 2; do
  K="$OWNER/rsna-knee-16-corpus80-tar30-s$s"
  for f in masks_s0$s.npy ids_s0$s.npy; do
    [ -s "corpus80_npz/$f" ] && continue
    timeout 300 $KG kernels output "$K" -p dl30 --file-pattern "^$f\$" >/dev/null 2>&1
    [ -s "dl30/$f" ] && mv "dl30/$f" corpus80_npz/
  done
  for p in $(seq -w 0 29); do
    f="c80_s${s}_p${p}.tar"
    for try in 1 2 3 4 5 6 7 8; do
      [ -f "dl30/ok_$f" ] && break
      rm -f "dl30/$f"
      timeout 900 $KG kernels output "$K" -p dl30 --file-pattern "^$f\$" >/dev/null 2>&1
      n=$(tar -tf "dl30/$f" 2>/dev/null | wc -l)
      if [ "$n" -ge 40 ]; then tar -xf "dl30/$f" -C corpus80_npz && rm -f "dl30/$f" && touch "dl30/ok_$f"; fi
      echo "$(date +%H:%M) $f try $try: $n files | npz total: $(ls corpus80_npz | grep -c 'npz$')"
    done
  done
done
echo "$(date +%H:%M) npz total: $(ls corpus80_npz | grep -c 'npz$') (expected 4407)"
