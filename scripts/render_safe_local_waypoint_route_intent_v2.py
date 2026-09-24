#!/usr/bin/env python3
"""Render V2 RGB targets from replay-verified base poses only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    sys.path.insert(0, str(extra))
REPLAY = ROOT / ".generated/safe_local_waypoint_route_intent_v2/replay"
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/safe_local_waypoint_route_intent_v2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    entry = manifest["state_candidates"][args.state_index]
    sid = entry["state_id"]
    replay_path = REPLAY / f"{sid}.json"
    replay = json.loads(replay_path.read_text())

    import genesis as gs
    from lewm_genesis.scene_loader import load_scene_pack
    from lewm.oracle.go2_textured_v03_renderer import BasePose, TexturedV03Renderer

    gs.init(backend=gs.cpu)
    pack = load_scene_pack(
        entry["scene_dir"],
        platform_manifest=ROOT / "config/go2_platform_manifest.yaml",
        workspace_root=ROOT,
    )
    raw = json.loads((Path(entry["scene_dir"]) / "genesis_scene.json").read_text())
    renderer = TexturedV03Renderer(SimpleNamespace(pack=pack), gs=gs, raw_manifest=raw)
    rgb_dir = CACHE / "rgb" / sid
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for row in replay["rows"]:
        for h in (1, 2, 3):
            item = row["horizons"][str(h)]
            result = renderer.render_pose(BasePose(tuple(item["pose"]), tuple(item["quaternion_wxyz"])))
            path = rgb_dir / f"candidate_{row['candidate_index']:02d}_h{h}.png"
            Image.fromarray(result.image, mode="RGB").save(path)
            item["rgb_path"] = str(path)
            item["rgb_sha256"] = sha(path)
    replay["render_status"] = "COMPLETE"
    replay["renderer_contract_digest"] = renderer.contract_digest
    replay["raw_manifest_digest"] = renderer.raw_manifest_digest
    replay_path.write_text(json.dumps(replay, indent=2))
    print(json.dumps({"state_id": sid, "frames": 36}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
