#!/usr/bin/env bash
# Multi-family physics-only sweep of the route teacher.
#
# Runs `genesis_bulk_rollout.sh --no-rgb` once per family with the first
# ``--scene-limit`` scenes from the chosen corpus and route_teacher pinned
# to 100%, then prints per-family beacons_achieved/available aggregated
# over scenes and envs. Use this as the regression gate before/after a
# planner change: identical CORPUS / SCENE_LIMIT / N_ENVS / N_BLOCKS makes
# the result deterministic up to the planner itself.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
cd "$REPO_ROOT"

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/.generated/scene_corpus/production_20260518T231208Z}"
SPLIT="${SPLIT:-train}"
SCENE_LIMIT="${SCENE_LIMIT:-3}"
N_ENVS="${N_ENVS:-4}"
N_BLOCKS="${N_BLOCKS:-400}"
LABEL="${LABEL:-sweep}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/.generated/profile_route_teacher/${LABEL}_$(date -u +%Y%m%dT%H%M%SZ)}"
FAMILIES="${FAMILIES:-open_obstacle_field rough_local_dynamics visual_sensor_stress loop_alias_stress local_composite_motifs medium_enclosed_maze small_enclosed_maze large_enclosed_maze}"

mkdir -p "$OUT_ROOT"
echo "=== route_teacher family sweep ==="
echo "corpus:      $CORPUS_DIR"
echo "scene_limit: $SCENE_LIMIT  n_envs=$N_ENVS  n_blocks=$N_BLOCKS"
echo "out:         $OUT_ROOT"
echo

for fam in $FAMILIES; do
  fam_out="$OUT_ROOT/$fam"
  mkdir -p "$fam_out"
  echo "--- $fam ---"
  bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" \
    --scene-corpus "$CORPUS_DIR" \
    --split "$SPLIT" \
    --family "$fam" \
    --scene-limit "$SCENE_LIMIT" \
    --n-envs "$N_ENVS" \
    --n-blocks "$N_BLOCKS" \
    --backend cpu \
    --no-rgb \
    --collector-mix route_teacher=1.0 \
    --log-progress-every-blocks 0 \
    --recovery-interlock-clearance-m 0 \
    --out "$fam_out" >"$fam_out/rollout.log" 2>&1 || {
      echo "  FAIL: see $fam_out/rollout.log" >&2
    }
done

echo
echo "=== aggregate ==="
python3 - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
tot_a, tot_v = 0, 0
rows = []
for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    a, v = 0, 0
    for s in sorted(fam_dir.rglob("summary.json")):
        d = json.loads(s.read_text(encoding="utf-8"))
        pe = d.get("extra", {}).get("rollout_stats", {}).get("per_env_metrics", [])
        a += sum(e["beacons_achieved"] for e in pe)
        v += sum(e.get("beacons_available", 0) for e in pe)
    rows.append((fam_dir.name, a, v))
    tot_a += a; tot_v += v
width = max(len(r[0]) for r in rows)
for fam, a, v in rows:
    pct = (a / v * 100) if v else 0.0
    print(f"{fam:<{width}}  {a}/{v}  {pct:.1f}%")
print(f"{'AGG':<{width}}  {tot_a}/{tot_v}  {(tot_a/tot_v*100 if tot_v else 0):.1f}%")
PY
