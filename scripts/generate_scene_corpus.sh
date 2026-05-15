#!/usr/bin/env bash
# Generate a deterministic LeWM scene corpus on disk.
#
# Default: writes a Phase-8 smoke tier (~10 scenes spanning every registered
# family and every split) under .generated/scene_corpus/<name>/. Use --standard
# with --train/--val/--test-id/--test-hard for a spec-aligned plan, or pass a
# plan JSON via --plan to drive arbitrary totals.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"

NAME="smoke"
MODE="smoke"
PLAN_SEED=0
OUT_DIR=""
PLAN_JSON=""
TRAIN_SCENES=200
VAL_SCENES=50
TEST_ID_SCENES=50
TEST_HARD_SCENES=50
EMIT_GENESIS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --plan-seed) PLAN_SEED="$2"; shift 2 ;;
    --smoke) MODE="smoke"; shift ;;
    --standard) MODE="standard"; shift ;;
    --plan) MODE="plan"; PLAN_JSON="$2"; shift 2 ;;
    --train) TRAIN_SCENES="$2"; shift 2 ;;
    --val) VAL_SCENES="$2"; shift 2 ;;
    --test-id) TEST_ID_SCENES="$2"; shift 2 ;;
    --test-hard) TEST_HARD_SCENES="$2"; shift 2 ;;
    --no-genesis) EMIT_GENESIS=0; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: generate_scene_corpus.sh [options]

Modes (mutually exclusive):
  --smoke               One scene per family across train/val/test_id/test_hard (default).
  --standard            Spec-aligned plan; use --train/--val/--test-id/--test-hard.
  --plan PATH           Drive totals from a JSON file: {"train": {"family": count}, ...}.

Common options:
  --name STR            Output subdirectory name (default: smoke).
  --out PATH            Absolute output path (default: .generated/scene_corpus/<name>).
  --plan-seed INT       Determines all scene seeds (default: 0).
  --no-genesis          Skip writing genesis_scene.json per scene.
USAGE
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/.generated/scene_corpus/$NAME"
fi

cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

python3 - "$MODE" "$PLAN_SEED" "$OUT_DIR" "$EMIT_GENESIS" \
  "$TRAIN_SCENES" "$VAL_SCENES" "$TEST_ID_SCENES" "$TEST_HARD_SCENES" \
  "$PLAN_JSON" <<'PY'
import json
import sys
from pathlib import Path

from lewm_worlds import (
    build_corpus,
    plan_corpus,
    plan_sha256,
    smoke_corpus_plan,
    standard_corpus_plan,
)

mode = sys.argv[1]
plan_seed = int(sys.argv[2])
out_dir = Path(sys.argv[3])
emit_genesis = bool(int(sys.argv[4]))
train_scenes = int(sys.argv[5])
val_scenes = int(sys.argv[6])
test_id_scenes = int(sys.argv[7])
test_hard_scenes = int(sys.argv[8])
plan_path = sys.argv[9]

if mode == "smoke":
    plan = smoke_corpus_plan(plan_seed=plan_seed)
elif mode == "standard":
    plan = standard_corpus_plan(
        plan_seed=plan_seed,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        test_id_scenes=test_id_scenes,
        test_hard_scenes=test_hard_scenes,
    )
elif mode == "plan":
    if not plan_path:
        raise SystemExit("--plan requires a JSON path")
    totals = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan = plan_corpus(plan_seed=plan_seed, totals=totals)
else:
    raise SystemExit(f"unknown mode: {mode}")

result = build_corpus(plan, out_dir, emit_genesis=emit_genesis)
print(f"out_dir={result.out_dir}")
print(f"scene_count={result.scene_count}")
print(f"plan_sha256={result.plan_sha256}")
print(f"splits={','.join(plan.splits)}")
counts = {}
for scene in result.scenes:
    counts[(scene.split, scene.family)] = counts.get((scene.split, scene.family), 0) + 1
for (split, family), count in sorted(counts.items()):
    print(f"  {split}/{family}: {count}")
PY
