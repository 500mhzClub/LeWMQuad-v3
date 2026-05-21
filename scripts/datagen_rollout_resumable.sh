#!/usr/bin/env bash
# Resumable, CPU-saturating physics-rollout + convert + labels over the full
# minimum-tier corpus (render is a separate phase).
#
# Each rollout process is Python-orchestration-bound at ~1.8 cores, and a whole
# family run is serial over its scenes, so 8 family-shards leave the box ~half
# idle and the biggest family (medium, 250 scenes) sets the wall-time. This
# driver instead splits every (split, family) into CHUNK-sized sub-shards and
# runs SLOTS workers, each pinned to 2 cores, pulling chunks round-robin — so
# ~16 chunks render concurrently and all 32 cores stay busy.
#
# RESUMABLE: every finished chunk drops a `.chunk_done` marker; on re-run after
# a freeze/reboot, done chunks are skipped and partial ones are cleared+redone.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
REPO="$PWD"

CORPUS="${CORPUS:-$REPO/.generated/scene_corpus/minimum_tex_20260520T211541Z}"
OUT="${OUT:-$REPO/.generated/datagen_full}"
N_ENVS="${N_ENVS:-48}"
N_BLOCKS="${N_BLOCKS:-200}"     # 200 blocks x 5 ticks = 1000 raw steps/stream
CHUNK="${CHUNK:-40}"
SLOTS="${SLOTS:-16}"            # 16 workers x 2 cores = 32 cores
CORES_PER_SLOT="${CORES_PER_SLOT:-2}"
SPLITS="${SPLITS:-train val test_id test_hard}"
FAMILIES="${FAMILIES:-open_obstacle_field local_composite_motifs small_enclosed_maze medium_enclosed_maze large_enclosed_maze loop_alias_stress rough_local_dynamics visual_sensor_stress}"
mkdir -p "$OUT"
JOBS="$OUT/.jobs.tsv"
: > "$JOBS"

# Build the chunk job list: split<TAB>family<TAB>offset<TAB>limit
for split in $SPLITS; do
  for fam in $FAMILIES; do
    n=$(find "$CORPUS/$split/$fam" -name manifest.json 2>/dev/null | wc -l)
    [[ "$n" -eq 0 ]] && continue
    off=0
    while [[ "$off" -lt "$n" ]]; do
      printf "%s\t%s\t%s\t%s\n" "$split" "$fam" "$off" "$CHUNK" >> "$JOBS"
      off=$((off + CHUNK))
    done
  done
done
total=$(wc -l < "$JOBS")
echo "corpus=$CORPUS"
echo "out=$OUT  n_envs=$N_ENVS n_blocks=$N_BLOCKS chunk=$CHUNK slots=$SLOTS"
echo "total chunks=$total"

run_chunk() {
  local split="$1" fam="$2" off="$3" lim="$4" cpus="$5"
  local cd="$OUT/rollout/$split/$fam/chunk_$(printf '%04d' "$off")"
  if [[ -f "$cd/.chunk_done" ]]; then echo "[skip] $split/$fam@$off"; return 0; fi
  rm -rf "$cd"; mkdir -p "$cd"
  if taskset -c "$cpus" bash "$SCRIPT_DIR/run_mass_datagen.sh" \
      --scene-corpus "$CORPUS" --split "$split" --family "$fam" \
      --scene-offset "$off" --scene-limit "$lim" \
      --n-envs "$N_ENVS" --n-blocks "$N_BLOCKS" \
      --backend cpu --no-render --out "$cd" > "$cd/chunk.log" 2>&1; then
    touch "$cd/.chunk_done"; echo "[done] $split/$fam@$off"
  else
    echo "[FAIL] $split/$fam@$off (see $cd/chunk.log)"
  fi
}

# Worker w processes job lines where (line_index % SLOTS == w), pinned to its cores.
worker() {
  local w="$1"
  local lo=$((w * CORES_PER_SLOT)); local hi=$((lo + CORES_PER_SLOT - 1))
  local cpus="${lo}-${hi}"
  local i=0
  while IFS=$'\t' read -r split fam off lim; do
    if [[ $((i % SLOTS)) -eq "$w" ]]; then
      run_chunk "$split" "$fam" "$off" "$lim" "$cpus"
    fi
    i=$((i + 1))
  done < "$JOBS"
}

for w in $(seq 0 $((SLOTS - 1))); do worker "$w" & done
wait
echo "PIPELINE_ROLLOUT_DONE $(date -u +%FT%TZ)"
