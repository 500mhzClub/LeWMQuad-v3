#!/usr/bin/env python3
"""Run Genesis bulk rollouts into per-scene MCAP raw_rollout directories."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
from lewm_genesis.mcap_writer import GLOBAL_CLOCK_TOPIC, MCAPSceneWriter, WriterStats, topic_name
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner
from lewm_genesis.scene_builder import build_scene_from_pack
from lewm_genesis.scene_loader import (
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)


class NoWriterSceneWriter:
    """Drop-in rollout writer that counts messages without serializing MCAP."""

    def __init__(self, pack, out_dir: str | Path, *, n_envs: int) -> None:
        self.pack = pack
        self.n_envs = int(n_envs)
        self.summary_dir = Path(out_dir) / pack.scene_id
        if self.summary_dir.exists():
            raise FileExistsError(f"output directory already exists: {self.summary_dir}")
        self.summary_dir.mkdir(parents=True, exist_ok=False)
        self.stats = WriterStats(env_count=self.n_envs)

    def write_env(self, env_index: int, key: str, message: object, stamp_ns: int) -> None:
        del message
        self.stats.record(topic_name(key, env_index), int(stamp_ns))

    def write_clock(self, stamp_ns: int) -> None:
        self.stats.record(GLOBAL_CLOCK_TOPIC[0], int(stamp_ns))

    def close(self, *, extra_summary: dict[str, object] | None = None) -> Path:
        summary = {
            "schema": "lewm_genesis_no_writer_rollout_v0",
            "scene_id": self.pack.scene_id,
            "family": self.pack.family,
            "split": self.pack.split,
            "difficulty_tier": self.pack.difficulty_tier,
            "manifest_sha256": self.pack.manifest_sha256,
            "n_envs": int(self.n_envs),
            "writer": "none",
            "stats": {
                "env_count": self.stats.env_count,
                "total_messages": self.stats.total_messages,
                "per_topic_counts": dict(self.stats.per_topic_counts),
                "first_stamp_ns": self.stats.first_stamp_ns,
                "last_stamp_ns": self.stats.last_stamp_ns,
                "duration_s": (
                    None
                    if self.stats.first_stamp_ns is None or self.stats.last_stamp_ns is None
                    else (self.stats.last_stamp_ns - self.stats.first_stamp_ns) / 1e9
                ),
            },
        }
        if extra_summary:
            summary["extra"] = dict(extra_summary)
        summary_path = self.summary_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary_path

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scene-corpus", type=Path, default=None)
    parser.add_argument("--platform-manifest", type=Path, default=None)
    parser.add_argument("--primitive-registry", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default=None)
    parser.add_argument("--scene-limit", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional command-tick target; rounded up to whole command blocks.",
    )
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--cfg-path", type=Path, default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument("--out-of-bounds-pad-m", type=float, default=0.5)
    parser.add_argument("--log-progress-every-blocks", type=int, default=10)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument(
        "--foot-contact-source",
        choices=("genesis", "zero"),
        default=None,
        help=(
            "Source for foot-contact labels. Defaults to 'zero' on amdgpu to avoid "
            "Genesis contact-force HSA faults, and 'genesis' otherwise."
        ),
    )
    parser.add_argument(
        "--no-writer",
        action="store_true",
        help="Exercise rollout and ROS message construction without opening or writing an MCAP bag.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    scene_corpus = (
        args.scene_corpus.resolve()
        if args.scene_corpus is not None
        else repo_root / ".generated" / "scene_corpus" / "acceptance"
    )
    platform_path = (
        args.platform_manifest.resolve()
        if args.platform_manifest is not None
        else repo_root / "config" / "go2_platform_manifest.yaml"
    )
    registry_path = (
        args.primitive_registry.resolve()
        if args.primitive_registry is not None
        else repo_root / "config" / "go2_primitive_registry.yaml"
    )
    out_root = (
        args.out.resolve()
        if args.out is not None
        else repo_root / ".generated" / "genesis_bulk_rollouts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    foot_contact_source = args.foot_contact_source
    if foot_contact_source is None:
        foot_contact_source = "zero" if args.backend.lower() == "amdgpu" else "genesis"
        if foot_contact_source == "zero":
            print("foot_contact_source=zero (default for amdgpu; avoids Genesis contact-force reads)")

    platform = load_platform_manifest(platform_path)
    registry = PrimitiveRegistry.from_yaml(registry_path)
    safety = SafetyLimits.from_manifest(platform)
    n_blocks = int(args.n_blocks)
    if args.steps is not None:
        n_blocks = max(1, math.ceil(int(args.steps) / int(registry.block_size)))

    scene_dirs = find_scene_dirs(scene_corpus, split=args.split, family=args.family)
    if args.scene_limit:
        scene_dirs = scene_dirs[: max(0, int(args.scene_limit))]
    if not scene_dirs:
        raise SystemExit(
            f"no scenes found under {scene_corpus} for split={args.split!r} family={args.family!r}"
        )

    policy_device = args.policy_device
    if policy_device is None and args.backend.lower() == "cpu":
        policy_device = "cpu"

    if args.policy is not None:
        policy = GenesisGo2PPOPolicy(
            args.policy.resolve(),
            cfg_path=args.cfg_path.resolve() if args.cfg_path is not None else None,
            device=policy_device,
        )
    else:
        policy = GenesisGo2PPOPolicy.from_platform_manifest(
            platform,
            repo_root,
            device=policy_device,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, object] = {
        "schema": "lewm_genesis_bulk_rollout_run_v0",
        "scene_corpus": str(scene_corpus),
        "platform_manifest": str(platform_path),
        "primitive_registry": str(registry_path),
        "out": str(out_root),
        "split": args.split,
        "family": args.family,
        "scene_count": len(scene_dirs),
        "n_envs": int(args.n_envs),
        "n_blocks": n_blocks,
        "rgb": not args.no_rgb,
        "backend": args.backend,
        "writer": "none" if args.no_writer else "mcap",
        "foot_contact_source": foot_contact_source,
        "scene_summaries": [],
    }

    for scene_index, scene_dir in enumerate(scene_dirs):
        pack = load_scene_pack(
            scene_dir,
            platform_manifest=platform,
            workspace_root=repo_root,
        )
        print(
            f"[{scene_index + 1}/{len(scene_dirs)}] scene={pack.scene_id} "
            f"family={pack.family} split={pack.split}"
        )
        build = build_scene_from_pack(
            pack,
            n_envs=int(args.n_envs),
            backend=args.backend,
            show_viewer=bool(args.show_viewer),
        )
        config = RolloutConfig(
            n_blocks=n_blocks,
            fall_z_threshold_m=float(args.fall_z_threshold_m),
            out_of_bounds_pad_m=float(args.out_of_bounds_pad_m),
            rgb_capture_per_block=not args.no_rgb,
            seed=int(args.seed) + scene_index,
            log_progress_every_blocks=int(args.log_progress_every_blocks),
            foot_contact_source=foot_contact_source,
        )
        runner = RolloutRunner(build, policy, registry, safety, config=config)
        writer_cls = NoWriterSceneWriter if args.no_writer else MCAPSceneWriter
        writer = writer_cls(pack, out_root, n_envs=int(args.n_envs))
        try:
            stats = runner.run(writer)
        except BaseException:
            writer.__exit__(Exception, None, None)
            raise
        summary_path = writer.close(
            extra_summary={
                "rollout_stats": stats,
                "policy_artifact": platform.get("locomotion", {}).get("policy_artifact", {}),
                "backend": args.backend,
                "rgb": not args.no_rgb,
                "writer": "none" if args.no_writer else "mcap",
                "foot_contact_source": foot_contact_source,
            }
        )
        run_summary["scene_summaries"].append(str(summary_path))
        print(f"  wrote {summary_path}")

    run_summary_path = out_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"run_summary={run_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
