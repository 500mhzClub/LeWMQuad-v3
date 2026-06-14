#!/usr/bin/env python3
"""Validate completed render outputs and write resume markers.

The render wrappers use ``.render_done`` as the scene-level resume marker. This
helper makes that marker conservative: a scene is considered done only when its
``summary.json`` says ``render_status=complete`` and its frame count matches the
render plan, optionally capped by ``--max-frames`` for smoke runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str
    frame_count: int | None = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _expected_frame_count(plan: dict, max_frames: int | None) -> int | None:
    raw = plan.get("frame_count")
    if raw is None and isinstance(plan.get("frames"), list):
        raw = len(plan["frames"])
    if raw is None:
        return None
    expected = int(raw)
    if max_frames is not None:
        expected = min(expected, int(max_frames))
    return expected


def check_complete(
    *,
    scene_dir: Path,
    plan_path: Path,
    scene_id: str | None,
    max_frames: int | None,
    visuals: str | None = None,
) -> CheckResult:
    summary_path = scene_dir / "summary.json"
    if not summary_path.is_file():
        return CheckResult(False, "missing summary.json")
    if not plan_path.is_file():
        return CheckResult(False, f"missing plan: {plan_path}")

    try:
        summary = _load_json(summary_path)
        plan = _load_json(plan_path)
    except Exception as exc:  # noqa: BLE001 - CLI should return the parse reason.
        return CheckResult(False, f"json parse failed: {exc}")

    if summary.get("render_status") != "complete":
        return CheckResult(False, "render_status is not complete")

    if visuals is not None and summary.get("visuals") != visuals:
        return CheckResult(
            False,
            f"visuals mismatch: summary={summary.get('visuals')!r} expected={visuals!r}",
        )

    if scene_id is not None and summary.get("scene_id") != scene_id:
        return CheckResult(
            False,
            f"scene_id mismatch: summary={summary.get('scene_id')!r} expected={scene_id!r}",
        )

    summary_plan = summary.get("plan")
    if summary_plan is not None:
        try:
            if Path(str(summary_plan)).resolve() != plan_path.resolve():
                return CheckResult(False, "summary plan path does not match job plan")
        except OSError:
            return CheckResult(False, "summary plan path could not be resolved")

    frame_count = summary.get("frame_count")
    if frame_count is None:
        return CheckResult(False, "missing summary frame_count")
    frame_count = int(frame_count)
    if frame_count <= 0:
        return CheckResult(False, "non-positive summary frame_count", frame_count)

    expected = _expected_frame_count(plan, max_frames)
    if expected is not None and frame_count != expected:
        return CheckResult(
            False,
            f"frame_count mismatch: summary={frame_count} expected={expected}",
            frame_count,
        )

    return CheckResult(True, "complete", frame_count)


def write_marker(scene_dir: Path, *, result: CheckResult, plan_path: Path) -> None:
    marker = scene_dir / ".render_done"
    marker.write_text(
        "\n".join(
            [
                "render_status=complete",
                f"frame_count={result.frame_count}",
                f"plan={plan_path.resolve()}",
                f"validated_at={datetime.now(timezone.utc).isoformat()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_max_frames(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-frames must be positive")
    return parsed


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--max-frames", default=None)
    parser.add_argument("--visuals", default=None)
    parser.add_argument("--quiet", action="store_true")


def cmd_check(args: argparse.Namespace) -> int:
    result = check_complete(
        scene_dir=args.scene_dir,
        plan_path=args.plan,
        scene_id=args.scene_id,
        max_frames=_parse_max_frames(args.max_frames),
        visuals=args.visuals,
    )
    if not args.quiet:
        status = "ok" if result.ok else "not-complete"
        print(f"{status}: {args.scene_dir} ({result.reason})")
    return 0 if result.ok else 1


def cmd_mark(args: argparse.Namespace) -> int:
    result = check_complete(
        scene_dir=args.scene_dir,
        plan_path=args.plan,
        scene_id=args.scene_id,
        max_frames=_parse_max_frames(args.max_frames),
        visuals=args.visuals,
    )
    if result.ok:
        write_marker(args.scene_dir, result=result, plan_path=args.plan)
    if not args.quiet:
        status = "marked" if result.ok else "not-complete"
        print(f"{status}: {args.scene_dir} ({result.reason})")
    return 0 if result.ok else 1


def cmd_backfill(args: argparse.Namespace) -> int:
    out_root = args.out.resolve()
    max_frames = _parse_max_frames(args.max_frames)
    checked = marked = skipped = 0
    for plan_path in sorted(args.rollout_root.rglob("render_replay_plan.json")):
        try:
            plan = _load_json(plan_path)
        except Exception:
            skipped += 1
            continue
        scene_id = plan.get("scene_id")
        if not scene_id:
            skipped += 1
            continue
        scene_dir = out_root / str(scene_id)
        checked += 1
        result = check_complete(
            scene_dir=scene_dir,
            plan_path=plan_path,
            scene_id=str(scene_id),
            max_frames=max_frames,
            visuals=args.visuals,
        )
        if result.ok:
            write_marker(scene_dir, result=result, plan_path=plan_path)
            marked += 1
    print(
        json.dumps(
            {
                "checked": checked,
                "marked": marked,
                "skipped_plans": skipped,
                "out": str(out_root),
                "rollout_root": str(args.rollout_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check")
    _add_check_args(check_p)
    check_p.set_defaults(func=cmd_check)

    mark_p = sub.add_parser("mark")
    _add_check_args(mark_p)
    mark_p.set_defaults(func=cmd_mark)

    backfill_p = sub.add_parser("backfill")
    backfill_p.add_argument("--out", required=True, type=Path)
    backfill_p.add_argument("--rollout-root", required=True, type=Path)
    backfill_p.add_argument("--max-frames", default=None)
    backfill_p.add_argument("--visuals", default=None)
    backfill_p.set_defaults(func=cmd_backfill)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
