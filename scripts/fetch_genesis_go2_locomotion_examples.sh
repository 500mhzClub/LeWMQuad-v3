#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENESIS_REF="${GENESIS_REF:-main}"
OUT_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"
BASE_URL="https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/$GENESIS_REF/examples/locomotion"

mkdir -p "$OUT_DIR"

for file in go2_train.py go2_env.py go2_eval.py; do
  curl -fsSL "$BASE_URL/$file" -o "$OUT_DIR/$file"
done

printf 'Fetched Genesis Go2 locomotion examples from %s into %s\n' "$GENESIS_REF" "$OUT_DIR"
