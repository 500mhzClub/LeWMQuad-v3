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
JOBLIST="$OUT/.jobs.tsv"
: > "$JOBLIST"

# Build the chunk job list: split<TAB>family<TAB>offset<TAB>limit
for split in $SPLITS; do
  for fam in $FAMILIES; do
    n=$(find "$CORPUS/$split/$fam" -name manifest.json 2>/dev/null | wc -l)
    [[ "$n" -eq 0 ]] && continue
    off=0
    while [[ "$off" -lt "$n" ]]; do
      printf "%s\t%s\t%s\t%s\n" "$split" "$fam" "$off" "$CHUNK" >> "$JOBLIST"
      off=$((off + CHUNK))
    done
  done
done
total=$(wc -l < "$JOBLIST")
echo "corpus=$CORPUS"
echo "out=$OUT  n_envs=$N_ENVS n_blocks=$N_BLOCKS chunk=$CHUNK slots=$SLOTS cores_per_slot=$CORES_PER_SLOT"
echo "total chunks=$total"

# --- Dynamic pull-queue + adaptive fan-out -----------------------------------
# The previous model statically partitioned the job list (line i -> worker
# i % SLOTS) and pinned each worker to its 2 cores for *every* stage. That
# starved the box two ways: a worker dealt a heavy slice ran long while others
# finished theirs and exited (idle cores), and the single-threaded
# convert/label stages stayed capped at 2 cores even with the rest of the box
# idle. Instead:
#   * workers pull the next chunk from a shared flock'd cursor, so none exit
#     early while chunks remain (load self-balances);
#   * only the [1/4] rollout is core-pinned (ROLLOUT_CPUS) — it is
#     orchestration-bound at ~2 cores, so SLOTS pinned streams pack the box;
#   * [2/4]/[3/4] run JOBS=auto, fanning out across nproc/(active chunks), so a
#     lone tail chunk's convert+label uses every core instead of one.
CURSOR="$OUT/.jobs.cursor"; echo 0 > "$CURSOR"
ACTIVE="$OUT/.active_chunks"; echo 0 > "$ACTIVE"
QLOCK="$OUT/.jobs.lock"; ALOCK="$OUT/.active.lock"
: > "$QLOCK"; : > "$ALOCK"

claim_next() {  # echo the next unclaimed job line atomically, or nothing at EOF
  exec 200>"$QLOCK"; flock 200
  local idx line
  idx="$(cat "$CURSOR" 2>/dev/null || echo 0)"
  line="$(sed -n "$((idx + 1))p" "$JOBLIST")"
  [[ -n "$line" ]] && echo $((idx + 1)) > "$CURSOR"
  flock -u 200
  printf '%s' "$line"
}

bump_active() {  # $1 = +1 / -1
  exec 201>"$ALOCK"; flock 201
  local n; n="$(cat "$ACTIVE" 2>/dev/null || echo 0)"
  echo $((n + $1)) > "$ACTIVE"
  flock -u 201
}

run_chunk() {
  local split="$1" fam="$2" off="$3" lim="$4" cpus="$5"
  local cd="$OUT/rollout/$split/$fam/chunk_$(printf '%04d' "$off")"
  if [[ -f "$cd/.chunk_done" ]]; then echo "[skip] $split/$fam@$off"; return 0; fi
  rm -rf "$cd"; mkdir -p "$cd"
  bump_active 1
  # Pin only the rollout (ROLLOUT_CPUS); convert/label fan out via JOBS=auto,
  # reading the live active-chunk count from LEWM_ACTIVE_CHUNKS_FILE.
  if ROLLOUT_CPUS="$cpus" JOBS=auto LEWM_ACTIVE_CHUNKS_FILE="$ACTIVE" \
     bash "$SCRIPT_DIR/run_mass_datagen.sh" \
       --scene-corpus "$CORPUS" --split "$split" --family "$fam" \
       --scene-offset "$off" --scene-limit "$lim" \
       --n-envs "$N_ENVS" --n-blocks "$N_BLOCKS" \
       --backend cpu --no-render --out "$cd" > "$cd/chunk.log" 2>&1; then
    touch "$cd/.chunk_done"; echo "[done] $split/$fam@$off"
  else
    echo "[FAIL] $split/$fam@$off (see $cd/chunk.log)"
  fi
  bump_active -1
}

# Each worker owns a fixed 2-core range (for rollout pinning) but pulls chunks
# dynamically, so a fast worker keeps grabbing work instead of idling.
worker() {
  local w="$1"
  local lo=$((w * CORES_PER_SLOT)); local hi=$((lo + CORES_PER_SLOT - 1))
  local cpus="${lo}-${hi}"
  local line split fam off lim
  while line="$(claim_next)"; [[ -n "$line" ]]; do
    IFS=$'\t' read -r split fam off lim <<< "$line"
    run_chunk "$split" "$fam" "$off" "$lim" "$cpus"
  done
}

for w in $(seq 0 $((SLOTS - 1))); do worker "$w" & done
wait
echo "PIPELINE_ROLLOUT_DONE $(date -u +%FT%TZ)"
