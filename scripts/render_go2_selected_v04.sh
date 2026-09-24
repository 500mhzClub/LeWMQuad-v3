#!/usr/bin/env bash
# Resumable sparse v04 RGB rendering over the frozen paired-navigation rows.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TASKS="${TASKS:-$PWD/.generated/go2_render_selected_v04/render_tasks.jsonl}"
VENV="${RENDER_VENV:-$PWD/.generated/venvs/genesis_render_vulkan}"
WORKERS="${WORKERS:-4}"
EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/lewm_go2_v04_cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lewm_go2_v04_mpl}"
RENDER_HOME="${RENDER_HOME:-/tmp/lewm_go2_v04_home}"

render_one() {
  local task="$1"
  local scene_id plan corpus selection final expected
  scene_id="$(jq -r '.scene_id' <<<"$task")"
  plan="$(jq -r '.plan_path' <<<"$task")"
  corpus="$(jq -r '.scene_corpus' <<<"$task")"
  selection="$(jq -r '.frame_selection_path' <<<"$task")"
  final="$(jq -r '.render_output_dir' <<<"$task")"
  expected="$(jq -r '.expected_frame_count' <<<"$task")"
  mkdir -p "$(dirname -- "$final")"
  if [[ -d "$final" ]]; then
    if [[ -f "$final/.render_done" ]] \
        && [[ "$(find "$final/rgb" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$expected" ]] \
        && jq -e --argjson expected "$expected" \
          '.schema == "lewm_rendered_vision_v04" and .render_status == "complete" and .frame_count == $expected' \
          "$final/summary.json" >/dev/null; then
      echo "[skip] $scene_id"
      return 0
    fi
    echo "[FAIL] existing output is incomplete or inconsistent: $final" >&2
    return 1
  fi
  local temp
  temp="$(mktemp -d "/tmp/go2_v04_${scene_id}_XXXXXX")"
  if EGL_DEVICE_ID="$EGL_DEVICE_ID" PYOPENGL_PLATFORM=egl \
      XDG_CACHE_HOME="$XDG_CACHE_HOME" MPLCONFIGDIR="$MPLCONFIGDIR" \
      HOME="$RENDER_HOME" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMBA_NUM_THREADS=1 \
      "$VENV/bin/python" "$SCRIPT_DIR/render_replay_selected_v04.py" \
        --plan "$plan" --scene-corpus "$corpus" \
        --frame-selection "$selection" --out "$temp" \
        --width 224 --height 168 --textures \
        >"$temp/render.log" 2>&1 \
      && [[ "$(find "$temp/rgb" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$expected" ]] \
      && jq -e --argjson expected "$expected" \
        '.schema == "lewm_rendered_vision_v04" and .render_status == "complete" and .frame_count == $expected' \
        "$temp/summary.json" >/dev/null; then
    mv "$temp" "$final"
    echo "[done] $scene_id frames=$expected"
  else
    echo "[FAIL] $scene_id temp=$temp" >&2
    return 1
  fi
}

export -f render_one
export VENV SCRIPT_DIR EGL_DEVICE_ID XDG_CACHE_HOME MPLCONFIGDIR RENDER_HOME

echo "tasks=$TASKS workers=$WORKERS egl_device_id=$EGL_DEVICE_ID"
if ! xargs -r -d '\n' -P "$WORKERS" -I '{}' \
    bash -c 'render_one "$1"' _ '{}' < "$TASKS"; then
  echo "PIPELINE_RENDER_V04_FAILED" >&2
  exit 1
fi
echo "PIPELINE_RENDER_V04_DONE"
