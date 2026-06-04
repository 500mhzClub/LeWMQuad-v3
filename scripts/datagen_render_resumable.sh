#!/usr/bin/env bash
# Resumable batched render over the converted raw rollouts (phase 2 of datagen).
#
# Renders all 48 parallel streams per scene in ONE camera.render() per timestep
# (--replay-env-mode batched), stores RGB at training resolution (224), and
# validates depth without persisting it. The GPU OpenGL rasterizer caps ~156
# env-fps total, and per-frame post-processing (validate/resize/encode/write)
# is CPU-bound, so a few concurrent workers parallelize post-processing up to
# that ceiling.
#
# RESUMABLE: each finished scene drops `.render_done`; on re-run after a
# freeze/reboot, done scenes are skipped and partial ones cleared+redone.
# Writes per-frame as it goes.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
REPO="$PWD"

CORPUS="${CORPUS:-$REPO/.generated/scene_corpus/minimum_tex_20260520T211541Z}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-$REPO/.generated/datagen_full/rollout}"
OUT="${OUT:-$REPO/.generated/datagen_full/render}"
WORKERS="${WORKERS:-4}"
CORES_PER_WORKER="${CORES_PER_WORKER:-8}"
CAMERA_HZ="${CAMERA_HZ:-10}"
RENDER_MAX_FRAMES="${RENDER_MAX_FRAMES:-}"   # cap frames/scene (testing only; empty = all)
RENDER_BACKEND="${RENDER_BACKEND:-cpu}"
REPLAY_ENV_MODE="${REPLAY_ENV_MODE:-batched}"
RENDER_VENV="${RENDER_VENV:-}"
EGL_DEVICE_ID="${EGL_DEVICE_ID:-}"
RESUME_PYTHON="${RENDER_RESUME_PYTHON:-python3}"
RESUME_HELPER="$SCRIPT_DIR/render_resume_markers.py"
mkdir -p "$OUT"

# 1) Build render plans per chunk (cheap, serial) — idempotent.
echo "[plan] building render plans..."
while IFS= read -r raw_dir; do
  [[ -d "$raw_dir" ]] || continue
  plandir="$(dirname "$raw_dir")/plan"
  if [[ ! -d "$plandir" ]]; then
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" --raw-root "$raw_dir" \
      --out-root "$plandir" --backend cpu --camera-hz "$CAMERA_HZ" >/dev/null 2>&1 || true
  fi
done < <(find "$ROLLOUT_ROOT" -type d -name raw 2>/dev/null | sort)

# 2) Collect all render plans into a job list.
JOBS="$OUT/.render_jobs.tsv"; : > "$JOBS"
find "$ROLLOUT_ROOT" -name render_replay_plan.json 2>/dev/null | sort | while read -r plan; do
  sid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('scene_id',''))" "$plan" 2>/dev/null)
  [[ -z "$sid" ]] && continue
  printf "%s\t%s\n" "$sid" "$plan" >> "$JOBS"
done
total=$(wc -l < "$JOBS" 2>/dev/null || echo 0)
echo "[render] $total scene plans, $WORKERS workers x $CORES_PER_WORKER cores"
echo "[render] out=$OUT backend=$RENDER_BACKEND replay_env_mode=$REPLAY_ENV_MODE render_venv=${RENDER_VENV:-default} egl_device_id=${EGL_DEVICE_ID:-default}"

render_scene() {
  local sid="$1" plan="$2" cpus="$3"
  local sd="$OUT/$sid"
  if "$RESUME_PYTHON" "$RESUME_HELPER" mark --quiet \
      --scene-dir "$sd" --plan "$plan" --scene-id "$sid" \
      --max-frames "$RENDER_MAX_FRAMES"; then
    echo "[skip] $sid"
    return 0
  fi
  rm -rf "$sd"; mkdir -p "$sd"
  local extra=()
  local render_env=()
  [[ -n "$RENDER_MAX_FRAMES" ]] && extra=(--max-frames "$RENDER_MAX_FRAMES")
  [[ -n "$RENDER_VENV" ]] && render_env+=("GENESIS_ROCM_PYTHON=$RENDER_VENV/bin/python")
  [[ -n "$EGL_DEVICE_ID" ]] && render_env+=("EGL_DEVICE_ID=$EGL_DEVICE_ID")
  [[ -n "${PYOPENGL_PLATFORM:-}" ]] && render_env+=("PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM")
  if taskset -c "$cpus" env "${render_env[@]}" bash "$SCRIPT_DIR/render_replay_genesis.sh" "$plan" \
      --scene-corpus "$CORPUS" --backend "$RENDER_BACKEND" --camera-mode replay \
      --replay-env-mode "$REPLAY_ENV_MODE" --rgb-format png \
      --store-resolution training --depth-validate-only "${extra[@]}" \
      --out "$sd" > "$sd/render.log" 2>&1; then
    if "$RESUME_PYTHON" "$RESUME_HELPER" mark --quiet \
        --scene-dir "$sd" --plan "$plan" --scene-id "$sid" \
        --max-frames "$RENDER_MAX_FRAMES"; then
      echo "[done] $sid"
    else
      echo "[FAIL] $sid (renderer exited 0 but output did not validate; see $sd/render.log)"
    fi
  else
    if "$RESUME_PYTHON" "$RESUME_HELPER" mark --quiet \
        --scene-dir "$sd" --plan "$plan" --scene-id "$sid" \
        --max-frames "$RENDER_MAX_FRAMES"; then
      echo "[done-after-nonzero] $sid"
    else
      echo "[FAIL] $sid (see $sd/render.log)"
    fi
  fi
}

worker() {
  local w="$1"
  local lo=$((w * CORES_PER_WORKER)); local hi=$((lo + CORES_PER_WORKER - 1))
  local i=0
  while IFS=$'\t' read -r sid plan; do
    if [[ $((i % WORKERS)) -eq "$w" ]]; then render_scene "$sid" "$plan" "${lo}-${hi}"; fi
    i=$((i + 1))
  done < "$JOBS"
}

for w in $(seq 0 $((WORKERS - 1))); do worker "$w" & done
wait
echo "PIPELINE_RENDER_DONE $(date -u +%FT%TZ)"
