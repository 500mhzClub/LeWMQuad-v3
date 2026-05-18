#!/usr/bin/env python3
"""Export visual QA videos from rendered Genesis replay outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RenderedRun:
    summary_path: Path
    rendered_dir: Path
    scene_id: str
    scene_family: str
    split: str
    frame_count: int
    invalid_frame_count: int
    frames_rendered_jsonl: Path


@dataclass(frozen=True)
class VideoResult:
    family: str
    scene_id: str
    split: str
    source_summary: str
    output_video: str
    frame_count: int
    env_index: int | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small visual-QA video set from rendered_vision outputs, "
            "grouped by scene family."
        )
    )
    parser.add_argument(
        "--rendered-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Rendered-vision root or leaf directory. May be passed more than once. "
            "Default: .generated/rendered_vision"
        ),
    )
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=None,
        help="Optional scene corpus root; used only to report expected families with no videos.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / ".generated" / "scene_type_videos",
        help="Output directory for videos and manifest.json.",
    )
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help=(
            "Only include this env stream. Use --all-envs to interleave all envs "
            "from the rendered output."
        ),
    )
    parser.add_argument("--all-envs", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = args.rendered_root or [REPO_ROOT / ".generated" / "rendered_vision"]
    runs = _discover_rendered_runs(roots)
    selected = _select_by_family(runs, per_family=max(0, int(args.per_family)))
    if not selected:
        raise SystemExit("no rendered_vision summaries with RGB frame metadata found")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_families = _expected_families(args.scene_corpus)
    results: list[VideoResult] = []
    skipped: list[dict[str, Any]] = []

    for run in selected:
        records = _load_frame_records(
            run.frames_rendered_jsonl,
            env_index=None if args.all_envs else int(args.env_index),
            include_invalid=bool(args.include_invalid),
        )
        if len(records) < int(args.min_frames) and not args.all_envs:
            records = _load_frame_records(
                run.frames_rendered_jsonl,
                env_index=None,
                include_invalid=bool(args.include_invalid),
            )
        if args.max_frames is not None:
            records = records[: max(0, int(args.max_frames))]
        if len(records) < int(args.min_frames):
            skipped.append(
                {
                    "summary": str(run.summary_path),
                    "reason": f"only {len(records)} usable frames",
                }
            )
            continue

        env_suffix = "all_envs" if args.all_envs else f"env{int(args.env_index):02d}"
        video_name = (
            f"{_safe(run.scene_family)}__{_safe(run.scene_id)}__{env_suffix}."
            f"{args.format}"
        )
        video_path = out_dir / video_name
        if video_path.exists() and not args.overwrite:
            skipped.append(
                {
                    "summary": str(run.summary_path),
                    "output_video": str(video_path),
                    "reason": "output exists; pass --overwrite",
                }
            )
            continue

        frame_paths = [Path(record["rgb_path"]) for record in records if record.get("rgb_path")]
        if args.format == "mp4":
            _write_mp4(frame_paths, video_path, fps=float(args.fps))
        else:
            _write_gif(frame_paths, video_path, fps=float(args.fps))

        results.append(
            VideoResult(
                family=run.scene_family,
                scene_id=run.scene_id,
                split=run.split,
                source_summary=str(run.summary_path),
                output_video=str(video_path),
                frame_count=len(frame_paths),
                env_index=None if args.all_envs else int(args.env_index),
            )
        )

    covered_families = sorted({result.family for result in results})
    manifest = {
        "schema": "lewm_scene_type_video_manifest_v0",
        "rendered_roots": [str(path.resolve()) for path in roots],
        "scene_corpus": None if args.scene_corpus is None else str(args.scene_corpus.resolve()),
        "output_dir": str(out_dir),
        "format": args.format,
        "fps": float(args.fps),
        "per_family": int(args.per_family),
        "env_index": None if args.all_envs else int(args.env_index),
        "all_envs": bool(args.all_envs),
        "covered_families": covered_families,
        "missing_expected_families": sorted(expected_families - set(covered_families)),
        "videos": [result.__dict__ for result in results],
        "skipped": skipped,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "video_count": len(results),
                "covered_families": covered_families,
                "missing_expected_families": manifest["missing_expected_families"],
                "skipped_count": len(skipped),
            },
            sort_keys=True,
        )
    )
    return 0 if results else 2


def _discover_rendered_runs(roots: list[Path]) -> list[RenderedRun]:
    runs: list[RenderedRun] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        summary_paths: list[Path]
        if root.is_file() and root.name == "summary.json":
            summary_paths = [root]
        elif (root / "summary.json").is_file():
            summary_paths = [root / "summary.json"]
        else:
            summary_paths = sorted(root.rglob("summary.json")) if root.exists() else []
        for summary_path in summary_paths:
            if summary_path in seen:
                continue
            seen.add(summary_path)
            run = _load_rendered_run(summary_path)
            if run is not None:
                runs.append(run)
    return sorted(runs, key=lambda r: (r.scene_family, r.split, r.scene_id, str(r.summary_path)))


def _load_rendered_run(summary_path: Path) -> RenderedRun | None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if summary.get("schema") != "lewm_rendered_vision_v0":
        return None
    frames_path = summary.get("frames_rendered_jsonl")
    if not frames_path:
        return None
    frames_rendered_jsonl = Path(frames_path)
    if not frames_rendered_jsonl.is_absolute():
        frames_rendered_jsonl = summary_path.parent / frames_rendered_jsonl
    if not frames_rendered_jsonl.is_file():
        return None
    return RenderedRun(
        summary_path=summary_path,
        rendered_dir=summary_path.parent,
        scene_id=str(summary.get("scene_id") or summary_path.parent.name),
        scene_family=str(summary.get("scene_family") or "unknown_family"),
        split=str(summary.get("split") or "unknown_split"),
        frame_count=int(summary.get("frame_count") or 0),
        invalid_frame_count=int(summary.get("invalid_frame_count") or 0),
        frames_rendered_jsonl=frames_rendered_jsonl,
    )


def _select_by_family(runs: list[RenderedRun], *, per_family: int) -> list[RenderedRun]:
    selected: list[RenderedRun] = []
    counts: dict[str, int] = {}
    for run in runs:
        count = counts.get(run.scene_family, 0)
        if count >= per_family:
            continue
        selected.append(run)
        counts[run.scene_family] = count + 1
    return selected


def _load_frame_records(
    frames_path: Path,
    *,
    env_index: int | None,
    include_invalid: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if env_index is not None and int(record.get("env_index") or 0) != env_index:
                continue
            if not include_invalid and not bool(record.get("camera_valid", True)):
                continue
            rgb_path = record.get("rgb_path")
            if not rgb_path:
                continue
            path = Path(rgb_path)
            if not path.is_absolute():
                path = frames_path.parent / path
                record["rgb_path"] = str(path)
            if not path.is_file():
                continue
            records.append(record)
    return sorted(
        records,
        key=lambda record: (
            int(record.get("timestamp_ns") or 0),
            int(record.get("frame_index") or 0),
            int(record.get("env_index") or 0),
        ),
    )


def _write_mp4(frame_paths: list[Path], output_path: Path, *, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --format mp4; use --format gif instead")
    with tempfile.TemporaryDirectory(prefix="lewm_video_frames_") as temp_name:
        temp_dir = Path(temp_name)
        for index, frame_path in enumerate(frame_paths):
            target = temp_dir / f"frame_{index:06d}.png"
            _materialize_video_frame(frame_path, target)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            _format_fps(fps),
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


def _write_gif(frame_paths: list[Path], output_path: Path, *, fps: float) -> None:
    duration_ms = max(1, int(round(1000.0 / fps)))
    frames = [_load_rgb_image(path) for path in frame_paths]
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    finally:
        for frame in frames:
            frame.close()


def _materialize_video_frame(source_path: Path, target_png: Path) -> None:
    if source_path.suffix.lower() == ".png":
        try:
            os.symlink(source_path, target_png)
        except OSError:
            shutil.copy2(source_path, target_png)
        return
    image = _load_rgb_image(source_path)
    try:
        image.save(target_png)
    finally:
        image.close()


def _load_rgb_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return Image.open(path).convert("RGB")


def _expected_families(scene_corpus: Path | None) -> set[str]:
    if scene_corpus is None:
        return set()
    corpus_path = scene_corpus.resolve() / "corpus.json"
    if not corpus_path.is_file():
        return set()
    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    plan_families = payload.get("plan", {}).get("families")
    if isinstance(plan_families, list):
        return {str(family) for family in plan_families}
    return {str(scene["family"]) for scene in payload.get("scenes", []) if "family" in scene}


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _format_fps(fps: float) -> str:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if float(fps).is_integer():
        return str(int(fps))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
