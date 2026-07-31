#!/bin/bash
# Download + install official LangSplatV2 LERF pretrained weights into ./output/
set -u

PROXY="socks5h://127.0.0.1:2345"
export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
export PATH="/data/home/rh_xu30/miniforge3/envs/langsplat_v2/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE=/tmp/langsplatv2_weights/LERF
mkdir -p "$CACHE" "$ROOT/output"

declare -A IDS=(
  [figurines]=1qKtjgTgt9xIkBsufiRZH6Mjo6DZSPqc_
  [ramen]=12Zgqx_bXt7Hxi7vh3jko1AGvLedd96AX
  [teatime]=159jEgMG5SGeFbFcLoqFr7OpqW5yzctt3
  [waldo_kitchen]=18BDS9kp0ioCXrdBQrB04TO0fIaE8RowX
)

for scene in "${@:-figurines ramen waldo_kitchen}"; do
  id="${IDS[$scene]:-}"
  if [ -z "$id" ]; then echo "unknown scene: $scene"; continue; fi

  if [ -d "$ROOT/output/${scene}_0_1" ]; then
    echo "[$scene] already installed, skipping"
    continue
  fi

  tarball="$CACHE/${scene}.tar.gz"
  if [ ! -f "$tarball" ]; then
    echo "[$scene] downloading..."
    gdown "$id" -O "$tarball" --proxy "$PROXY" || { echo "[$scene] DOWNLOAD FAILED"; continue; }
  fi

  echo "[$scene] extracting..."
  tmp=$(mktemp -d -p /tmp)
  tar -xzf "$tarball" -C "$tmp" || { echo "[$scene] EXTRACT FAILED"; rm -rf "$tmp"; continue; }
  for d in "$tmp"/*/; do
    base=$(basename "$d")
    rm -rf "$ROOT/output/$base"
    mv "$d" "$ROOT/output/$base"
    echo "[$scene] installed output/$base"
  done
  rm -rf "$tmp"
  rm -f "$tarball"
  echo "[$scene] DONE"
done

echo "ALL_FETCH_DONE"
ls -d "$ROOT"/output/*/ 2>/dev/null
