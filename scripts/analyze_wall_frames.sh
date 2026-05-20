#!/usr/bin/env bash
# Render egocentric (replay-camera) frames for enclosed-maze scenes and tally
# how many are "wall in FOV" / low-information using the vision_quality gate.
#
# Usage: scripts/analyze_wall_frames.sh --scene-corpus PATH --out PATH \
#          [--families "..."] [--scenes-per-family 2] [--n-blocks 600] \
#          [--max-frames 400]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ros_env.sh"
REPO_ROOT="$(lewm_repo_root)"; cd "$REPO_ROOT"

CORPUS=""; OUT=""; SCENES=2; NBLOCKS=600; MAXF=400
FAMILIES="small_enclosed_maze medium_enclosed_maze large_enclosed_maze"
while [[ $# -gt 0 ]]; do case "$1" in
  --scene-corpus) CORPUS="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  --families) FAMILIES="$2"; shift 2;;
  --scenes-per-family) SCENES="$2"; shift 2;;
  --n-blocks) NBLOCKS="$2"; shift 2;;
  --max-frames) MAXF="$2"; shift 2;;
  *) echo "unknown $1" >&2; exit 2;;
esac; done
[[ -z "$CORPUS" || -z "$OUT" ]] && { echo "need --scene-corpus and --out"; exit 2; }
mkdir -p "$OUT"; W="$OUT/work"; mkdir -p "$W"

for FAM in $FAMILIES; do
  echo "=== $FAM ==="
  roll="$W/rollout/$FAM"
  bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" --scene-corpus "$CORPUS" \
    --split train --family "$FAM" --scene-limit "$SCENES" \
    --n-envs 4 --n-blocks "$NBLOCKS" --backend cpu --no-rgb \
    --collector-mix route_teacher=1.0 --recovery-interlock-clearance-m 0 \
    --log-progress-every-blocks 0 --out "$roll" >"$W/${FAM}_roll.log" 2>&1 || { echo "  roll FAIL"; continue; }
  for sd in "$roll"/*/; do
    sc=$(basename "$sd"); [[ -f "$sd/summary.json" ]] || continue
    raw="$W/raw/$FAM/$sc"
    bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" "$sd" --out "$raw" --quality-profile raw_pilot >/dev/null 2>&1 || { echo "  conv FAIL $sc"; continue; }
    pl="$W/plan/$FAM/$sc"
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" --raw-root "$(dirname "$raw")" --out-root "$pl" --backend cpu --camera-hz 10 >/dev/null 2>&1 || true
    plan=$(find "$pl" -name render_replay_plan.json -path "*${sc}*" | head -1)
    [[ -z "$plan" ]] && plan=$(find "$pl" -name render_replay_plan.json | head -1)
    [[ -z "$plan" ]] && { echo "  no plan $sc"; continue; }
    # Egocentric replay camera WITH depth, env 0, capped frames.
    rd="$W/render/$FAM/$sc"
    bash "$SCRIPT_DIR/render_replay_genesis.sh" "$plan" --scene-corpus "$CORPUS" \
      --backend cpu --camera-mode replay --rgb-format png --env-index 0 \
      --max-frames "$MAXF" --out "$rd" >"$W/${FAM}_${sc}_render.log" 2>&1 || { echo "  render FAIL $sc"; continue; }
    echo "  rendered $sc -> $rd/frames_rendered.jsonl"
  done
done

echo
echo "=== WALL-FRAME ANALYSIS ==="
PYTHONPATH=lewm_genesis:lewm_worlds python3 - "$W/render" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
root = Path(sys.argv[1])
agg = Counter(); per_fam = {}
total = 0
for fr in sorted(root.rglob("frames_rendered.jsonl")):
    fam = fr.parts[-3]
    c = Counter(); n = 0
    for line in fr.open():
        r = json.loads(line); n += 1; total += 1
        reasons = set(r.get("invalid_reasons", [])) | set(r.get("low_info_reasons", []))
        if "near_wall_depth" in reasons: c["near_wall_depth"] += 1; agg["near_wall_depth"] += 1
        if "low_rgb_texture" in reasons: c["low_rgb_texture"] += 1; agg["low_rgb_texture"] += 1
        if "near_forward_geometry" in reasons: c["near_forward_geometry"] += 1; agg["near_forward_geometry"] += 1
        # "wall fills FOV" = near_wall_depth OR low_rgb_texture
        if {"near_wall_depth","low_rgb_texture"} & reasons: c["wall_fov"] += 1; agg["wall_fov"] += 1
        c["frames"] += 1
    b = per_fam.setdefault(fam, Counter()); b.update(c)
print(f"{'family':24s} {'frames':>7s} {'wall_fov%':>9s} {'near_wall%':>10s} {'low_tex%':>9s} {'near_fwd%':>9s}")
for fam, b in sorted(per_fam.items()):
    n=b['frames'] or 1
    print(f"{fam:24s} {b['frames']:>7d} {b['wall_fov']/n*100:8.1f}% {b['near_wall_depth']/n*100:9.1f}% {b['low_rgb_texture']/n*100:8.1f}% {b['near_forward_geometry']/n*100:8.1f}%")
n=total or 1
print(f"{'TOTAL':24s} {total:>7d} {agg['wall_fov']/n*100:8.1f}% {agg['near_wall_depth']/n*100:9.1f}% {agg['low_rgb_texture']/n*100:8.1f}% {agg['near_forward_geometry']/n*100:8.1f}%")
PY
