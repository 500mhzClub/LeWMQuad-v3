#!/usr/bin/env bash
# Tier B: train a Genesis-native Go2 locomotion policy via upstream
# examples/locomotion/go2_train.py on the ROCm Genesis venv.
#
# Defaults match the upstream Genesis recipe (B=4096, 101 iterations) which
# is enough to produce a walking gait on a 32 GB AMD Radeon AI Pro 9700.
# Override via env vars:
#
#   LEWM_GO2_EXP_NAME      experiment name (default: lewm-go2-<UTC timestamp>)
#   LEWM_GO2_NUM_ENVS      parallel envs (default: 4096)
#   LEWM_GO2_MAX_ITERS     PPO learning iterations (default: 101)
#   LEWM_GO2_SEED          PPO seed (default: 1)
#   HSA_OVERRIDE_GFX_VERSION  ROCm GPU arch override (default: unset; only set
#                             this on machines like Phoenix1/gfx1103 that need
#                             it. RDNA4/gfx1201 must NOT be overridden.)
#
# Logs land under .generated/upstream_genesis/locomotion/logs/<exp_name>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"
EXAMPLES_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ROCm Genesis Python not found at: $PYTHON_BIN" >&2
  echo "Run scripts/setup_genesis_rocm_training.sh first." >&2
  exit 2
fi

if [[ ! -f "$EXAMPLES_DIR/go2_train.py" || ! -f "$EXAMPLES_DIR/go2_env.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

EXP_NAME="${LEWM_GO2_EXP_NAME:-lewm-go2-$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_ENVS="${LEWM_GO2_NUM_ENVS:-4096}"
MAX_ITERS="${LEWM_GO2_MAX_ITERS:-101}"
SEED="${LEWM_GO2_SEED:-1}"

if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  export HSA_OVERRIDE_GFX_VERSION
  echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION (caller-supplied)"
else
  echo "HSA_OVERRIDE_GFX_VERSION unset (correct for RDNA4 / gfx1201)"
fi

echo "Genesis Go2 PPO training"
echo "  exp_name:       $EXP_NAME"
echo "  num_envs:       $NUM_ENVS"
echo "  max_iterations: $MAX_ITERS"
echo "  seed:           $SEED"
echo "  log dir:        $EXAMPLES_DIR/logs/$EXP_NAME"
echo

cd "$EXAMPLES_DIR"
exec "$PYTHON_BIN" go2_train.py \
  -e "$EXP_NAME" \
  -B "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed "$SEED"
