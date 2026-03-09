#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR="${1:-tests/data}"
OUT_DIR="${2:-cheat_reports}"
mkdir -p "$OUT_DIR"
for f in "$INPUT_DIR"/*; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  ./tools/fstockfish-cheat --input "$f" --out "$OUT_DIR/${base}.json"
done
printf 'wrote reports to %s\n' "$OUT_DIR"
