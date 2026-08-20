#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "$0")/.." && pwd)
exe="${1:-$root_dir/build/Release/vbsr_benchmark.exe}"
out="${2:-$root_dir/data/regime_map.csv}"
rows="${ROWS:-4096}"
reps="${REPS:-20}"
warmup="${WARMUP:-5}"
first=1
for distribution in uniform low high bimodal; do
  for locality in local random; do
    for degree in 2 4 8 16; do
      for rhs in 8 16 32 64; do
        result=$($exe --rows "$rows" --degree "$degree" --rhs "$rhs" --distribution "$distribution" --locality "$locality" --reps "$reps" --warmup "$warmup" --seed 1)
        if [[ $first == 1 ]]; then printf '%s\n' "$result" > "$out"; first=0
        else printf '%s\n' "$result" | tail -n +2 >> "$out"
        fi
      done
    done
  done
done
echo "Wrote $out"
