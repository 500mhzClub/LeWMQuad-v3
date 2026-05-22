#!/usr/bin/env bash
# Resumable Vulkan render over the rollout corpus on the AMD R9700, using the
# genesis-world 0.3.14 render venv (gs.vulkan). Replays each scene's recorded
# camera trajectory single-env via scripts/render_replay_v03.py.
#
# Why a separate venv + EGL device: see docs/render_backend_amd.md. The OpenGL
# render device is selected with EGL_DEVICE_ID (1 = discrete R9700 here);
# gs.vulkan moves compute onto the GPU. ~8 workers is the throughput knee
# (16 oversubscribes). RESUMABLE: each finished scene drops `.render_done`;
# partial scenes are cleared + redone on re-run. Plans must already exist
# (built by plan_bulk_render_replay.sh during rollout).
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
REPO="$PWD"

VENV="${RENDER_VENV:-$REPO/.generated/venvs/genesis_render_vulkan}"
CORPUS="${CORPUS:-$REPO/.generated/scene_corpus/minimum_tex_20260520T211541Z}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-$REPO/.generated/datagen_full/rollout}"
OUT="${OUT:-$REPO/.generated/datagen_full/render}"
WORKERS="${WORKERS:-8}"
EGL_DEVICE_ID="${EGL_DEVICE_ID:-1}"   # 1 = AMD R9700 (discrete); 0 = iGPU
RESOLUTION="${RESOLUTION:-224}"
mkdir -p "$OUT"

JOBS="$OUT/.render_jobs.txt"
find "$ROLLOUT_ROOT" -name render_replay_plan.json 2>/dev/null | sort > "$JOBS"
total=$(wc -l < "$JOBS")
echo "render_venv=$VENV"
echo "corpus=$CORPUS"
echo "out=$OUT  egl_device_id=$EGL_DEVICE_ID (1=R9700)  workers=$WORKERS  res=$RESOLUTION"
echo "total scene plans=$total"

render_one() {
  local plan="$1"
  local sid
  sid="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['scene_id'])" "$plan" 2>/dev/null)"
  [[ -z "$sid" ]] && { echo "[skip] (no scene_id) $plan"; return 0; }
  local sd="$OUT/$sid"
  if [[ -f "$sd/.render_done" ]]; then echo "[skip] $sid"; return 0; fi
  rm -rf "$sd"; mkdir -p "$sd"
  if EGL_DEVICE_ID="$EGL_DEVICE_ID" PYOPENGL_PLATFORM=egl \
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
     "$VENV/bin/python" "$SCRIPT_DIR/render_replay_v03.py" \
       --plan "$plan" --scene-corpus "$CORPUS" --out "$sd" --resolution "$RESOLUTION" \
       > "$sd/render.log" 2>&1; then
    touch "$sd/.render_done"; echo "[done] $sid"
  else
    echo "[FAIL] $sid (see $sd/render.log)"
  fi
}
export -f render_one
export VENV SCRIPT_DIR CORPUS OUT EGL_DEVICE_ID RESOLUTION

# xargs keeps WORKERS render processes busy, pulling plans dynamically.
xargs -a "$JOBS" -r -P "$WORKERS" -I {} bash -c 'render_one "$@"' _ {}
echo "PIPELINE_RENDER_V03_DONE $(date -u +%FT%TZ)"
