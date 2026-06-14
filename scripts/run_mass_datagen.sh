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
# Pin only the rollout when the driver supplies a core range (ROLLOUT_CPUS):
# genesis is orchestration-bound at ~2 cores, so pinning lets many streams pack
# the box. The convert/label stages below stay unpinned to fan out via JOBS.
if [[ -n "${ROLLOUT_CPUS:-}" ]]; then
  taskset -c "$ROLLOUT_CPUS" bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" "${ROLLOUT_ARGS[@]}"
else
  bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" "${ROLLOUT_ARGS[@]}"
fi

mkdir -p "$OUT/raw" "$OUT/labels"

# Per-scene conversion and label derivation are independent (distinct in/out
# dirs, no shared state), so fan them out across JOBS processes.
#   JOBS=1     (default) one scene at a time; under the resumable driver the
#              cross-chunk concurrency is the parallelism in this mode.
#   JOBS=N     fixed fan-out of N (e.g. JOBS=$(nproc) for a standalone run).
#   JOBS=auto  adaptive: nproc / (active chunks), read live from
#              $LEWM_ACTIVE_CHUNKS_FILE just before each stage, so as sibling
#              chunks finish a survivor's fan-out climbs and a lone tail chunk
#              grabs every core. No file (standalone) => active=1 => nproc.
JOBS="${JOBS:-1}"
resolve_jobs() {
  if [[ "$JOBS" != "auto" ]]; then echo "$JOBS"; return; fi
  local ncores active j
  ncores="$(nproc)"
  active=1
  if [[ -n "${LEWM_ACTIVE_CHUNKS_FILE:-}" && -r "${LEWM_ACTIVE_CHUNKS_FILE:-}" ]]; then
    active="$(cat "$LEWM_ACTIVE_CHUNKS_FILE" 2>/dev/null || echo 1)"
  fi
  [[ "$active" =~ ^[0-9]+$ && "$active" -ge 1 ]] || active=1
  j=$(( ncores / active )); (( j < 1 )) && j=1
  echo "$j"
}

J2="$(resolve_jobs)"
echo "[2/4] raw conversion + audit (jobs=$J2)"
convert_and_audit_one() {
  local scene_dir="$1"
  local scene; scene="$(basename "$scene_dir")"
  # A scene the bulk rollout aborted (e.g. rigid-solver NaN) can leave a
  # partial dir with no summary.json. Skip it instead of converting a
  # truncated MCAP and failing the whole chunk on an intentionally-dropped scene.
  if [[ ! -f "$scene_dir/summary.json" ]]; then
    echo "[skip] $scene (no rollout summary.json — partial/aborted scene)"
    return 0
  fi
  bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
    "$scene_dir" --out "$OUT/raw/$scene" \
    --quality-profile "$QUALITY_PROFILE" >/dev/null \
    || { echo "[FAIL raw] $scene (conversion error)" >&2; return 1; }
  local summary="$OUT/raw/$scene/summary.json"
  local contract quality
  contract="$(jq -r '.contract_audit.pass' "$summary")"
  quality="$(jq -r '.data_quality_audit.pass' "$summary")"
  if [[ "$contract" != "true" || "$quality" != "true" ]]; then
    echo "[FAIL raw] $scene contract=$contract quality=$quality" >&2
    return 1
  fi
}
export -f convert_and_audit_one
export SCRIPT_DIR OUT QUALITY_PROFILE
printf '%s\0' "$OUT"/rollout/*/ \
  | xargs -0 -r -P "$J2" -I {} bash -c 'convert_and_audit_one "$@"' _ {}

J3="$(resolve_jobs)"
echo "[3/4] derive labels (jobs=$J3)"
derive_one() {
  local raw_dir="$1"
  local scene; scene="$(basename "$raw_dir")"
  PYTHONPATH=lewm_genesis:lewm_worlds python3 \
    "$SCRIPT_DIR/derive_raw_rollout_labels.py" \
    "$raw_dir" --scene-corpus "$CORPUS" --scene-id "$scene" \
    --out "$OUT/labels/$scene" >/dev/null \
    || { echo "[FAIL labels] $scene (derive error)" >&2; return 1; }
}
export -f derive_one
export SCRIPT_DIR OUT CORPUS
printf '%s\0' "$OUT"/raw/*/ \
  | xargs -0 -r -P "$J3" -I {} bash -c 'derive_one "$@"' _ {}
echo "[3/4] audit derived labels"
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
