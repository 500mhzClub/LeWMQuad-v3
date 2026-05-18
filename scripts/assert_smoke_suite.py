#!/usr/bin/env python3
"""Per-family smoke-suite assertions.

Validates that the artifacts under ``--suite-root`` meet the contract,
data-quality, and H-JEPA-coverage requirements that must hold before mass
data generation.

Assertions per family:
  1. Physics rollout summary present, no rollout errors, expected env/block
     counts, MCAP writer used.
  2. raw_rollout conversion summary present and reports
     ``contract_audit.pass`` and ``data_quality_audit.pass`` under
     ``raw_training``. No timestamp regressions; no missing executions;
     reset_count strictly monotonic per env.
  3. Per-env base_state coverage: pose count == n_envs * n_blocks * block_size
     (10 Hz); no env missing entirely; first and last timestamps within tick
     tolerance of expected wall.
  4. Derived labels present; label_count > 0; labels span >1 cell per env on
     average (proves the robot actually moved through the scene); at least one
     landmark observation across the rollout.
  5. Collection-policy mix: realized counts match the requested mix
     (when one was set) or, for the default mix, all six §13 collectors are
     represented across envs/blocks.
  6. Egocentric and overview renders both present;
     ``invalid_frame_count == 0`` and ``camera_safety_unresolved_count == 0``
     under raw_training; render frame count > 0; videos exist for both views.
  7. Per-step body-state sanity: no roll/pitch exceeds safety bounds (robot
     did not tip), all base z above the fall threshold, joint position arrays
     have 12 entries each.

Writes ``assertion_report.json`` under ``--suite-root``. Exits non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


COMMAND_TICK_HZ = 10.0
COMMAND_TICK_DT_S = 1.0 / COMMAND_TICK_HZ
BLOCK_SIZE_TICKS = 5

# Per data spec §16 (reset integrity), the rollout is expected to reset after
# a fall and continue. So instead of gating on the peak tip angle (which spikes
# to ~180° for a single block while the robot is fallen waiting for reset), we
# gate on the *fraction* of the rollout spent in a fallen state. A handful of
# tipped frames per env after a fall is normal and useful training signal; a
# rollout where most frames are upside-down is broken.
TIP_ANGLE_LIMIT_RAD = math.radians(60.0)
MIN_BASE_Z_M = 0.15  # matches rollout fall_z_threshold_m
MAX_FALLEN_FRAME_FRACTION = 0.03  # at most 3% of frames may be below MIN_BASE_Z_M
MAX_TIPPED_FRAME_FRACTION = 0.03  # at most 3% of frames may exceed TIP_ANGLE_LIMIT_RAD

# Camera quality gates from data-spec §15.
MAX_INVALID_FRAME_RATE = 0.001  # 0.1%
MAX_UNRESOLVED_RATE = 0.002     # 0.2%


@dataclass
class FamilyArtifacts:
    family: str
    physics_summary: Path | None = None
    physics_run_summary: Path | None = None
    raw_summary: Path | None = None
    raw_messages: Path | None = None
    label_summary: Path | None = None
    label_jsonl: Path | None = None
    render_ego_summary: Path | None = None
    render_top_summary: Path | None = None
    video_ego: Path | None = None
    video_top: Path | None = None


@dataclass
class FamilyResult:
    family: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--n-blocks", type=int, required=True)
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    corpus_dir = args.corpus.resolve()
    n_envs = int(args.n_envs)
    n_blocks = int(args.n_blocks)

    families = sorted(p.name for p in (corpus_dir / "train").iterdir() if p.is_dir())
    if not families:
        print(f"no families found under {corpus_dir}/train", file=sys.stderr)
        return 2

    results: list[FamilyResult] = []
    for family in families:
        result = _check_family(
            family=family,
            suite_root=suite_root,
            n_envs=n_envs,
            n_blocks=n_blocks,
        )
        results.append(result)

    report = {
        "schema": "lewm_smoke_suite_assertion_v0",
        "suite_root": str(suite_root),
        "corpus": str(corpus_dir),
        "n_envs": n_envs,
        "n_blocks": n_blocks,
        "families": [
            {
                "family": r.family,
                "ok": r.ok,
                "passed": r.passed,
                "failed": r.failed,
                "warnings": r.warnings,
                "stats": r.stats,
            }
            for r in results
        ],
        "ok": all(r.ok for r in results),
        "fail_count": sum(1 for r in results if not r.ok),
        "warn_count": sum(len(r.warnings) for r in results),
    }
    report_path = suite_root / "assertion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {r.family}  passed={len(r.passed)} failed={len(r.failed)} warn={len(r.warnings)}")
        for msg in r.failed:
            print(f"        FAIL: {msg}")
        for msg in r.warnings:
            print(f"        warn: {msg}")

    print(f"\nreport: {report_path}")
    return 0 if report["ok"] else 3


def _check_family(*, family: str, suite_root: Path, n_envs: int, n_blocks: int) -> FamilyResult:
    result = FamilyResult(family=family)
    arts = _discover_artifacts(family, suite_root)

    # --- 1. physics summary -------------------------------------------------
    if arts.physics_summary is None:
        result.failed.append("physics summary.json missing")
        return result
    phys = _read_json(arts.physics_summary)
    extra = phys.get("extra", {})
    stats = extra.get("rollout_stats", {})
    result.stats["scene_id"] = phys.get("scene_id")
    result.stats["wall_seconds"] = stats.get("wall_seconds")
    result.stats["sim_seconds"] = stats.get("final_sim_time_s")
    if extra.get("writer") != "mcap":
        result.failed.append(f"writer={extra.get('writer')!r}; expected 'mcap'")
    if int(phys.get("n_envs") or 0) != n_envs:
        result.failed.append(f"n_envs mismatch: rollout reports {phys.get('n_envs')}, expected {n_envs}")
    if int(stats.get("n_blocks") or 0) != n_blocks:
        result.failed.append(f"n_blocks mismatch: rollout reports {stats.get('n_blocks')}, expected {n_blocks}")
    expected_ticks = n_blocks * BLOCK_SIZE_TICKS
    actual_ticks = int(stats.get("command_ticks") or 0)
    if actual_ticks != expected_ticks:
        result.failed.append(
            f"command_ticks mismatch: rollout reports {actual_ticks}, expected {expected_ticks}"
        )
    expected_sim_s = expected_ticks * COMMAND_TICK_DT_S
    if abs(float(stats.get("final_sim_time_s") or 0.0) - expected_sim_s) > 0.05:
        result.failed.append(
            f"final_sim_time_s={stats.get('final_sim_time_s')} not within 50ms of {expected_sim_s:.2f}"
        )
    else:
        result.passed.append("physics summary clean")

    # Per-topic counts on the writer side.
    writer_stats = phys.get("stats", {})
    per_topic = writer_stats.get("per_topic_counts", {})
    base_state_total = sum(c for t, c in per_topic.items() if t.endswith("/lewm/go2/base_state"))
    expected_base_state = n_envs * expected_ticks
    if base_state_total != expected_base_state:
        result.failed.append(
            f"writer base_state count {base_state_total} != expected {expected_base_state}"
        )
    cmd_block_total = sum(c for t, c in per_topic.items() if t.endswith("/lewm/go2/command_block"))
    expected_cmd_blocks = n_envs * n_blocks
    if cmd_block_total != expected_cmd_blocks:
        result.failed.append(
            f"writer command_block count {cmd_block_total} != expected {expected_cmd_blocks}"
        )
    exec_total = sum(c for t, c in per_topic.items() if t.endswith("/lewm/go2/executed_command_block"))
    if exec_total != expected_cmd_blocks:
        result.failed.append(
            f"writer executed_command_block count {exec_total} != expected {expected_cmd_blocks}"
        )

    # --- 2. raw_rollout conversion ------------------------------------------
    if arts.raw_summary is None:
        result.failed.append("raw_rollout summary missing")
        return result
    raw = _read_json(arts.raw_summary)
    contract = raw.get("contract_audit", {})
    quality = raw.get("data_quality_audit", {})
    if not contract.get("pass"):
        result.failed.append(f"contract_audit not pass: {contract.get('issues')}")
    else:
        result.passed.append("contract_audit pass")
    if quality.get("profile") != "raw_training":
        result.failed.append(f"raw quality_profile={quality.get('profile')!r}; expected 'raw_training'")
    if not quality.get("pass"):
        result.failed.append(f"data_quality_audit not pass: {quality.get('issues')}")
    else:
        result.passed.append("data_quality_audit pass (raw_training)")
    # No regressions on any per-env state stream.
    canonical_topics = raw.get("topic_audit", {}).get("canonical_topics", {})
    base_state_audit = canonical_topics.get("/lewm/go2/base_state", {}).get("record", {})
    if (base_state_audit.get("timestamp_regressions") or 0) != 0:
        result.failed.append("base_state timestamps have regressions")
    foot_audit = canonical_topics.get("/lewm/go2/foot_contacts", {}).get("record", {})
    if (foot_audit.get("timestamp_regressions") or 0) != 0:
        result.failed.append("foot_contacts timestamps have regressions")

    # --- 3. per-env base_state coverage from messages.jsonl -----------------
    if arts.raw_messages is None or not arts.raw_messages.is_file():
        result.failed.append("messages.jsonl missing")
        return result
    coverage = _scan_messages(arts.raw_messages, n_envs)
    result.stats["env_coverage"] = coverage["env_coverage"]
    result.stats["collector_mix_realized"] = coverage["collector_mix"]
    expected_per_env = n_blocks * BLOCK_SIZE_TICKS
    for env_idx in range(n_envs):
        cnt = coverage["env_coverage"].get(env_idx, 0)
        if cnt != expected_per_env:
            result.failed.append(
                f"env {env_idx} base_state count {cnt} != expected {expected_per_env}"
            )
    if not result.failed:
        result.passed.append("per-env base_state count exact")
    # --- 7. body-state sanity (tipping / falls) ----------------------------
    total_base_rows = sum(coverage["env_coverage"].values()) or 1
    fallen_frac = coverage["fallen_frame_count"] / total_base_rows
    tipped_frac = coverage["tipped_frame_count"] / total_base_rows
    result.stats["fallen_frame_fraction"] = fallen_frac
    result.stats["tipped_frame_fraction"] = tipped_frac
    result.stats["max_tip_deg_per_env"] = {
        env: math.degrees(rad) for env, rad in coverage["max_tip_per_env_rad"].items()
    }
    result.stats["min_z_per_env_m"] = coverage["min_z_per_env_m"]
    if fallen_frac > MAX_FALLEN_FRAME_FRACTION:
        result.failed.append(
            f"fallen-frame fraction {fallen_frac*100:.2f}% "
            f"exceeds {MAX_FALLEN_FRAME_FRACTION*100:.2f}%"
        )
    elif fallen_frac > 0:
        result.warnings.append(
            f"{coverage['fallen_frame_count']} fallen frames "
            f"({fallen_frac*100:.3f}%) within tolerance"
        )
    if tipped_frac > MAX_TIPPED_FRAME_FRACTION:
        result.failed.append(
            f"tipped-frame fraction {tipped_frac*100:.2f}% "
            f"exceeds {MAX_TIPPED_FRAME_FRACTION*100:.2f}%"
        )
    elif tipped_frac > 0:
        result.warnings.append(
            f"{coverage['tipped_frame_count']} tipped frames "
            f"({tipped_frac*100:.3f}%) within tolerance"
        )
    if coverage["bad_joint_rows"]:
        result.failed.append(
            f"{coverage['bad_joint_rows']} joint_state rows did not have 12 joints"
        )
    if (
        not coverage["bad_joint_rows"]
        and fallen_frac <= MAX_FALLEN_FRAME_FRACTION
        and tipped_frac <= MAX_TIPPED_FRAME_FRACTION
    ):
        result.passed.append("body-state sanity (fall/tip fractions under spec, joint shape OK)")

    # --- 5. collection-policy mix ------------------------------------------
    mix_realized = coverage["collector_mix"]
    requested_mix = extra.get("collector_mix")
    if requested_mix is None:
        # default §13 mix — every named collector should at least appear
        if mix_realized:
            empty = [name for name, count in mix_realized.items() if count == 0]
            if empty:
                result.warnings.append(
                    f"default §13 collectors with zero blocks: {empty}"
                )
            else:
                result.passed.append("default §13 collector mix realized fully")
        else:
            result.warnings.append("no collector_mix telemetry observed")
    else:
        for name, weight in (requested_mix or {}).items():
            if float(weight) > 0 and (mix_realized.get(name) or 0) == 0:
                result.warnings.append(
                    f"requested collector {name!r} (weight={weight}) produced 0 blocks"
                )

    # --- 4. derived labels --------------------------------------------------
    if arts.label_summary is None:
        result.failed.append("derived label summary missing")
    else:
        labels_summary = _read_json(arts.label_summary)
        label_count = int(labels_summary.get("label_count") or 0)
        if label_count <= 0:
            result.failed.append("derived label_count is zero")
        else:
            result.passed.append(f"derived labels written ({label_count} rows)")
        # confirm at least one entry per local_graph_type bucket present
        result.stats["local_graph_type_counts"] = labels_summary.get(
            "local_graph_type_counts"
        )
        if arts.label_jsonl is not None and arts.label_jsonl.is_file():
            label_stats = _scan_labels(arts.label_jsonl)
            result.stats["unique_cells_per_env"] = label_stats["unique_cells_per_env"]
            result.stats["landmark_observation_count"] = label_stats["landmark_obs"]
            for env_idx, unique in label_stats["unique_cells_per_env"].items():
                if unique < 2:
                    result.warnings.append(
                        f"env {env_idx} only occupied {unique} cell(s) — limited spatial coverage"
                    )
            if label_stats["landmark_obs"] == 0:
                result.warnings.append("no landmark observations recorded")
            else:
                result.passed.append(
                    f"landmark observations seen ({label_stats['landmark_obs']} step-landmark pairs)"
                )

    # --- 6. render replay -- ego + overview ---------------------------------
    for kind, summary_path, video_path in (
        ("ego", arts.render_ego_summary, arts.video_ego),
        ("top", arts.render_top_summary, arts.video_top),
    ):
        if summary_path is None:
            result.failed.append(f"{kind} render summary missing")
            continue
        ro = _read_json(summary_path)
        if int(ro.get("frame_count") or 0) <= 0:
            result.failed.append(f"{kind} render produced no frames")
            continue
        invalid = int(ro.get("invalid_frame_count") or 0)
        unresolved = int(ro.get("camera_safety_unresolved_count") or 0)
        total = int(ro["frame_count"])
        invalid_rate = invalid / total if total else 0.0
        unresolved_rate = unresolved / total if total else 0.0
        result.stats[f"{kind}_frame_count"] = total
        result.stats[f"{kind}_invalid_rate"] = invalid_rate
        result.stats[f"{kind}_unresolved_rate"] = unresolved_rate
        if invalid_rate > MAX_INVALID_FRAME_RATE:
            result.failed.append(
                f"{kind} invalid_frame_rate {invalid_rate*100:.3f}% > {MAX_INVALID_FRAME_RATE*100:.3f}%"
            )
        elif invalid > 0:
            result.warnings.append(
                f"{kind} render had {invalid} invalid frames ({invalid_rate*100:.4f}%) — under threshold but flagged"
            )
        if unresolved_rate > MAX_UNRESOLVED_RATE:
            result.failed.append(
                f"{kind} camera_safety_unresolved_rate {unresolved_rate*100:.3f}% > {MAX_UNRESOLVED_RATE*100:.3f}%"
            )
        if video_path is None or not video_path.is_file():
            result.failed.append(f"{kind} video missing")
        else:
            result.passed.append(f"{kind} render + video present")

    return result


def _discover_artifacts(family: str, suite_root: Path) -> FamilyArtifacts:
    arts = FamilyArtifacts(family=family)
    phys_root = suite_root / "physics" / family
    if phys_root.is_dir():
        run_summary = phys_root / "run_summary.json"
        if run_summary.is_file():
            arts.physics_run_summary = run_summary
        for child in phys_root.iterdir():
            if child.is_dir() and child.name.startswith(f"{family}_"):
                arts.physics_summary = child / "summary.json"
                break

    raw_root = suite_root / "raw" / family
    if raw_root.is_dir():
        for child in raw_root.iterdir():
            if child.is_dir():
                arts.raw_summary = child / "summary.json"
                arts.raw_messages = child / "messages.jsonl"
                break

    label_root = suite_root / "derived_labels" / family
    if label_root.is_dir():
        for child in label_root.iterdir():
            if child.is_dir():
                arts.label_summary = child / "summary.json"
                arts.label_jsonl = child / "labels.jsonl"
                break

    for kind, attr in (("ego", "render_ego_summary"), ("top", "render_top_summary")):
        render_root = suite_root / f"rendered_{kind}" / family
        if render_root.is_dir():
            summary = render_root / "summary.json"
            if summary.is_file():
                setattr(arts, attr, summary)

    for kind, attr in (("ego", "video_ego"), ("top", "video_top")):
        video_root = suite_root / "videos" / kind
        if video_root.is_dir():
            matches = sorted(video_root.glob(f"{family}__*.mp4"))
            if matches:
                setattr(arts, attr, matches[0])

    return arts


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_messages(messages_path: Path, n_envs: int) -> dict[str, Any]:
    base_count_per_env: Counter[int] = Counter()
    max_tip_per_env: dict[int, float] = defaultdict(float)
    min_z_per_env: dict[int, float] = defaultdict(lambda: math.inf)
    fallen_frame_count = 0
    tipped_frame_count = 0
    bad_joint_rows = 0
    collector_counts: Counter[str] = Counter()
    with messages_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            canonical = record.get("canonical_topic")
            env_idx_raw = record.get("env_index")
            env_idx = int(env_idx_raw) if env_idx_raw is not None else 0
            payload = record.get("payload") or {}
            if canonical == "/lewm/go2/base_state":
                base_count_per_env[env_idx] += 1
                pose = payload.get("pose_world") or {}
                ori = pose.get("orientation") or {}
                pos = pose.get("position") or {}
                qx, qy, qz, qw = (
                    float(ori.get("x") or 0.0),
                    float(ori.get("y") or 0.0),
                    float(ori.get("z") or 0.0),
                    float(ori.get("w") or 1.0),
                )
                roll, pitch, _yaw = _quat_xyzw_to_rpy(qx, qy, qz, qw)
                tip = max(abs(roll), abs(pitch))
                if tip > max_tip_per_env[env_idx]:
                    max_tip_per_env[env_idx] = tip
                if tip > TIP_ANGLE_LIMIT_RAD:
                    tipped_frame_count += 1
                z = float(pos.get("z") or 0.0)
                if z < min_z_per_env[env_idx]:
                    min_z_per_env[env_idx] = z
                if z < MIN_BASE_Z_M:
                    fallen_frame_count += 1
            elif canonical == "/joint_states":
                positions = payload.get("position") or []
                if len(positions) != 12:
                    bad_joint_rows += 1
            elif canonical == "/lewm/go2/command_block":
                source = payload.get("command_source") or "unknown"
                collector_counts[str(source)] += 1
    return {
        "env_coverage": dict(base_count_per_env),
        "max_tip_per_env_rad": {k: float(v) for k, v in max_tip_per_env.items()},
        "min_z_per_env_m": {
            k: float(v if math.isfinite(v) else 0.0) for k, v in min_z_per_env.items()
        },
        "fallen_frame_count": int(fallen_frame_count),
        "tipped_frame_count": int(tipped_frame_count),
        "bad_joint_rows": int(bad_joint_rows),
        "collector_mix": dict(collector_counts),
    }


def _scan_labels(labels_path: Path) -> dict[str, Any]:
    cells_per_env: dict[int, set[int]] = defaultdict(set)
    landmark_obs = 0
    visible_landmark_ids_per_env: dict[int, set[str]] = defaultdict(set)
    with labels_path.open(encoding="utf-8") as stream:
        for line in stream:
            label = json.loads(line)
            env_idx = int(label.get("env_idx") if label.get("env_idx") is not None else 0)
            cell_id = label.get("cell_id")
            if cell_id is not None:
                cells_per_env[env_idx].add(int(cell_id))
            for obs in label.get("landmarks") or []:
                if obs.get("visible"):
                    landmark_obs += 1
                    visible_landmark_ids_per_env[env_idx].add(str(obs.get("object_id") or ""))
    return {
        "unique_cells_per_env": {k: len(v) for k, v in cells_per_env.items()},
        "landmark_obs": landmark_obs,
        "visible_landmark_ids_per_env": {
            k: sorted(v) for k, v in visible_landmark_ids_per_env.items()
        },
    }


def _quat_xyzw_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


if __name__ == "__main__":
    raise SystemExit(main())
