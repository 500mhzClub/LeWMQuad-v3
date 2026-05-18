#!/usr/bin/env bash
# Physics-only route-teacher rollout + a one-page metrics report.
#
# Runs `genesis_bulk_rollout.sh --no-rgb` for a single maze scene with the
# route teacher pinned to 100% of the collector mix, then prints per-env
# path length / cells visited / goal counts / primitive distribution from
# the scene's summary.json. Use this to iterate on planner logic without
# the Genesis renderer or mp4 encode pipeline (full re-render is ~4 min;
# this should finish in ~20 s).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
cd "$REPO_ROOT"

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/.generated/scene_corpus/acceptance}"
FAMILY="${FAMILY:-large_enclosed_maze}"
N_ENVS="${N_ENVS:-4}"
N_BLOCKS="${N_BLOCKS:-240}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/.generated/profile_route_teacher/$(date -u +%Y%m%dT%H%M%SZ)}"
FIXED_SPAWN="${FIXED_SPAWN:-1}"

mkdir -p "$OUT_ROOT"

extra=()
if [[ "$FIXED_SPAWN" == "1" ]]; then
  extra+=(--fixed-spawn)
fi
# The route teacher has its own stuck FSM that runs ahead of planning;
# disabling the rollout-level interlock prevents the two from fighting
# (interlock takes over for 16 blocks → teacher loses its state → loops).
extra+=(--recovery-interlock-clearance-m 0)

echo "=== route_teacher profiler ==="
echo "family: $FAMILY  scenes=1  n_envs=$N_ENVS  n_blocks=$N_BLOCKS"
echo "out:    $OUT_ROOT"

bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" \
  --scene-corpus "$CORPUS_DIR" \
  --split train \
  --family "$FAMILY" \
  --scene-limit 1 \
  --n-envs "$N_ENVS" \
  --n-blocks "$N_BLOCKS" \
  --backend cpu \
  --no-rgb \
  --collector-mix route_teacher=1.0 \
  --log-progress-every-blocks 0 \
  "${extra[@]}" \
  --out "$OUT_ROOT" >"$OUT_ROOT/rollout.log" 2>&1

summary=$(find "$OUT_ROOT" -maxdepth 3 -name summary.json | sort | head -n 1)
if [[ -z "$summary" ]]; then
  echo "no summary.json produced; see $OUT_ROOT/rollout.log" >&2
  exit 2
fi

python3 - "$summary" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
stats = payload.get("extra", {}).get("rollout_stats", {})
per_env = stats.get("per_env_metrics") or []
n_blocks = int(stats.get("n_blocks") or 0)
wall = float(stats.get("wall_seconds") or 0.0)
scene = payload.get("scene_id") or stats.get("scene_id") or "?"

print(f"\nscene={scene}  blocks={n_blocks}  wall={wall:.1f}s")
print("-" * 80)
hdr = (
    f"{'env':>3}  {'cells':>5}  {'goals':>5}  {'claim':>8}  {'retgt':>5}  "
    f"{'path_m':>7}  {'recovery':>8}  primitives"
)
print(hdr)
for env in per_env:
    primitives = env.get("primitive_counts", {})
    prim_str = ", ".join(
        f"{name}={count}" for name, count in sorted(primitives.items(), key=lambda kv: -kv[1])
    )
    achieved = int(env.get("beacons_achieved", 0))
    available = int(env.get("beacons_available", 0))
    claim = f"{achieved}/{available}" if available else f"{achieved}/0"
    print(
        f"{env['env_idx']:>3}  "
        f"{env['cells_visited']:>5}  "
        f"{env['unique_goals_targeted']:>5}  "
        f"{claim:>8}  "
        f"{env['goal_target_changes']:>5}  "
        f"{env['path_length_m']:>7.2f}  "
        f"{env['recovery_interlock_handoffs']:>8}  "
        f"{prim_str}"
    )

# Aggregate over envs for a quick scalar.
if per_env:
    avg_cells = sum(e["cells_visited"] for e in per_env) / len(per_env)
    avg_goals = sum(e["unique_goals_targeted"] for e in per_env) / len(per_env)
    avg_achieved = sum(e["beacons_achieved"] for e in per_env) / len(per_env)
    avg_available = sum(e.get("beacons_available", 0) for e in per_env) / len(per_env)
    avg_path = sum(e["path_length_m"] for e in per_env) / len(per_env)
    avg_recovery = sum(e["recovery_interlock_handoffs"] for e in per_env) / len(per_env)
    print("-" * 80)
    print(
        f"avg  cells={avg_cells:.1f}  goals={avg_goals:.1f}  "
        f"claimed={avg_achieved:.1f}/{avg_available:.1f}  "
        f"path={avg_path:.2f} m  interlock_handoffs={avg_recovery:.1f}"
    )
print(f"\nlog: {Path(sys.argv[1]).parent}/rollout.log -> {Path(sys.argv[1]).parent.parent}/rollout.log")
PY
