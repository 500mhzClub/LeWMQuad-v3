#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT/.generated/go2_memory_closed_loop/b7_strict_learned_maze_20260704}"
SEED="${SEED:-1}"
OUT="${OUT:-$ARTIFACT_DIR/medium_b7c_strict_learned_maze_seed${SEED}_result.json}"
TEMPLATE="$ARTIFACT_DIR/medium_b7c_v295_strict_learnedclaim_thr085_postv287_bluev290_redv278_seed1_result.json"

"$PYTHON" - "$ROOT" "$ARTIFACT_DIR" "$TEMPLATE" "$OUT" "$SEED" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
artifact_dir = Path(sys.argv[2]).resolve()
template = Path(sys.argv[3]).resolve()
output = Path(sys.argv[4]).resolve()
seed = str(sys.argv[5])

payload = json.loads(template.read_text(encoding="utf-8"))
argv = [str(item) for item in payload["provenance"]["argv"]]
repo_prefix = "/home/andrewknowles/Workspace/LeWMQuad-v3"
artifact_names = {path.name for path in artifact_dir.iterdir() if path.is_file()}
scratch_dirs = (
    Path("/tmp/lewm_v244_allcolor_target_20260704"),
    Path("/tmp/lewm_v244_allcolor_target_20260703"),
)

rewritten: list[str] = []
for item in argv:
    text = item.replace(repo_prefix, str(root))
    for name in artifact_names:
        for scratch_dir in scratch_dirs:
            text = text.replace(str(scratch_dir / name), str(artifact_dir / name))
    rewritten.append(text)

if rewritten:
    rewritten[0] = str(root / "scripts" / "benchmark_go2_memory_closed_loop.py")

def set_arg(args: list[str], key: str, value: str) -> None:
    if key in args:
        args[args.index(key) + 1] = value
    else:
        args.extend([key, value])

set_arg(rewritten, "--seed", seed)
set_arg(rewritten, "--output", str(output))
output.parent.mkdir(parents=True, exist_ok=True)
raise SystemExit(subprocess.run([sys.executable, *rewritten], cwd=str(root)).returncode)
PY
