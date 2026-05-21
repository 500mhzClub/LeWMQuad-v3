#!/usr/bin/env bash
# Drive the mass-datagen pipeline end-to-end with fail-fast quality gates.
#
# Layout under ``--out`` after a successful run::
#
#     <out>/rollout/<scene_id>/...   # MCAP from bulk rollout
#     <out>/raw/<scene_id>/...       # raw_rollout messages + audit summary
#     <out>/labels/<scene_id>/...    # derived labels + audit summary
#     <out>/plan/<seq>_<scene>/...   # per-scene render-replay plan
#     <out>/render/<scene_id>/...    # rendered vision frames + summary
#
# Each stage is gated:
#   1. bulk rollout produces N scenes
#   2. raw conversion: contract_audit.pass AND data_quality_audit.pass
#   3. derived labels: audit_derived_labels.py
#   4. render-replay: audit_rendered_vision.py (skipped if --no-render)
#
# Exits non-zero on the first failure with the offending scene id.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
cd "$REPO_ROOT"

CORPUS=""
OUT=""
SPLIT="train"
FAMILY=""
SCENE_LIMIT=""
SCENE_OFFSET=""
N_ENVS=4
N_BLOCKS=1000
BACKEND="cpu"
COLLECTOR_MIX=""
QUALITY_PROFILE="raw_training"
SKIP_RENDER=0
RENDER_MAX_FRAMES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-corpus) CORPUS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --scene-limit) SCENE_LIMIT="$2"; shift 2 ;;
    --scene-offset) SCENE_OFFSET="$2"; shift 2 ;;
    --n-envs) N_ENVS="$2"; shift 2 ;;
    --n-blocks) N_BLOCKS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --collector-mix) COLLECTOR_MIX="$2"; shift 2 ;;
    --quality-profile) QUALITY_PROFILE="$2"; shift 2 ;;
    --no-render) SKIP_RENDER=1; shift ;;
    --render-max-frames) RENDER_MAX_FRAMES="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -e/p' "$0" | sed 's/^# //; s/^#//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CORPUS" || -z "$OUT" ]]; then
  echo "Usage: $0 --scene-corpus PATH --out PATH [--split train] [--family X] [--scene-limit N] [--n-envs N] [--n-blocks N] [--backend cpu|gpu] [--collector-mix MIX] [--quality-profile raw_training|raw_pilot] [--no-render] [--render-max-frames N]" >&2
  exit 2
fi

mkdir -p "$OUT"
echo "[1/4] bulk rollout"
ROLLOUT_ARGS=(
  --scene-corpus "$CORPUS"
  --split "$SPLIT"
  --n-envs "$N_ENVS"
  --n-blocks "$N_BLOCKS"
  --backend "$BACKEND"
  --no-rgb
  --recovery-interlock-clearance-m 0
  --log-progress-every-blocks 0
  --out "$OUT/rollout"
)
[[ -n "$FAMILY" ]] && ROLLOUT_ARGS+=(--family "$FAMILY")
[[ -n "$SCENE_LIMIT" ]] && ROLLOUT_ARGS+=(--scene-limit "$SCENE_LIMIT")
[[ -n "$SCENE_OFFSET" ]] && ROLLOUT_ARGS+=(--scene-offset "$SCENE_OFFSET")
[[ -n "$COLLECTOR_MIX" ]] && ROLLOUT_ARGS+=(--collector-mix "$COLLECTOR_MIX")
bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" "${ROLLOUT_ARGS[@]}"

mkdir -p "$OUT/raw" "$OUT/labels"

echo "[2/4] raw conversion + audit"
for scene_dir in "$OUT"/rollout/*/; do
  scene=$(basename "$scene_dir")
  bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
    "$scene_dir" \
    --out "$OUT/raw/$scene" \
    --quality-profile "$QUALITY_PROFILE" >/dev/null
  summary="$OUT/raw/$scene/summary.json"
  contract=$(jq -r '.contract_audit.pass' "$summary")
  quality=$(jq -r '.data_quality_audit.pass' "$summary")
  if [[ "$contract" != "true" || "$quality" != "true" ]]; then
    echo "[FAIL raw] $scene contract=$contract quality=$quality" >&2
    exit 1
  fi
done

echo "[3/4] derive labels + audit"
for raw_dir in "$OUT"/raw/*/; do
  scene=$(basename "$raw_dir")
  PYTHONPATH=lewm_genesis:lewm_worlds python3 \
    "$SCRIPT_DIR/derive_raw_rollout_labels.py" \
    "$raw_dir" \
    --scene-corpus "$CORPUS" \
    --scene-id "$scene" \
    --out "$OUT/labels/$scene" >/dev/null
done
python3 "$SCRIPT_DIR/audit_derived_labels.py" "$OUT/labels"

if [[ "$SKIP_RENDER" -eq 1 ]]; then
  echo "[4/4] render skipped (--no-render)"
  exit 0
fi

echo "[4/4] render-replay + audit"
mkdir -p "$OUT/plan" "$OUT/render"
bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" \
  --raw-root "$OUT/raw" \
  --out-root "$OUT/plan" \
  --backend "$BACKEND" --camera-hz 10 >/dev/null

for plan_dir in "$OUT"/plan/*/; do
  plan="$plan_dir/render_replay_plan.json"
  [[ -f "$plan" ]] || continue
  scene=$(jq -r '.scene_id' "$plan")
  render_args=(
    "$plan"
    --scene-corpus "$CORPUS"
    --backend "$BACKEND"
    --replay-env-mode batched
    --out "$OUT/render/$scene"
  )
  [[ -n "$RENDER_MAX_FRAMES" ]] && render_args+=(--max-frames "$RENDER_MAX_FRAMES")
  bash "$SCRIPT_DIR/render_replay_genesis.sh" "${render_args[@]}" >/dev/null
done
python3 "$SCRIPT_DIR/audit_rendered_vision.py" "$OUT/render"

echo "[done] mass-datagen pipeline complete: $OUT"
