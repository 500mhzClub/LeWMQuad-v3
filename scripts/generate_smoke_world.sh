#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
SEED=0
FAMILY="open_obstacle_field"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --family)
      FAMILY="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/.generated/worlds/${FAMILY}_${SEED}"
fi

cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

python3 - "$SEED" "$FAMILY" "$OUT_DIR" <<'PY'
from pathlib import Path
import sys

from lewm_worlds import build_scene_manifest, export_gazebo_sdf, manifest_sha256, topology_summary

seed = int(sys.argv[1])
family = sys.argv[2]
out_dir = Path(sys.argv[3])

manifest = build_scene_manifest(seed, family)
world = export_gazebo_sdf(manifest, out_dir)
summary = topology_summary(manifest)

print(f"world={world}")
print(f"manifest={out_dir / 'manifest.json'}")
print(f"scene_id={manifest.scene_id}")
print(f"manifest_sha256={manifest_sha256(manifest)}")
print(f"nodes={summary['node_count']} edges={summary['edge_count']} cycles={summary['cycle_count']}")
PY
