#!/bin/bash
# Extract downloaded LERF teatime V2 weights into ./output/
set -e
SRC=${1:-/tmp/langsplatv2_weights/LERF/teatime.tar.gz}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f "$SRC" ]; then
  echo "Missing $SRC"
  # maybe still .part
  ls -lh /tmp/langsplatv2_weights/LERF/ || true
  exit 1
fi
mkdir -p output
tmp=$(mktemp -d)
tar -xzf "$SRC" -C "$tmp"
echo "Extracted tree:"
find "$tmp" -maxdepth 3 -type d
# Move scene folders into output/
# Expected names like teatime_0_1 / teatime_0_2 / teatime_0_3
find "$tmp" -type d -name 'teatime_*' | while read d; do
  base=$(basename "$d")
  echo "-> output/$base"
  rm -rf "output/$base"
  mv "$d" "output/$base"
done
# Also handle if tar root already has output/
if [ -d "$tmp/output" ]; then
  cp -a "$tmp/output/." output/
fi
rm -rf "$tmp"
ls -la output/ | head -30
echo "Done. Try: bash eval_lerf.sh teatime 0 10000"
