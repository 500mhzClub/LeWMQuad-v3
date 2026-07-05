#!/usr/bin/env python3
"""Synthesize runtime-safe visual-bearing post-claim rows for Go2 local policy.

The input rows provide the target-specific latent context. This utility only
overwrites observed egocentric readout slots, claimed-mask slots, active-target
slots, and metadata used by the optional visual-readout feature suffix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dim", type=int, default=1600)
    parser.add_argument("--memory-vec-start", type=int, default=512)
    parser.add_argument("--memory-conf-start", type=int, default=520)
    parser.add_argument(
        "--color-vocab",
        default="blue,green,red,yellow",
        help="Color order in the learned-local feature vector.",
    )
    parser.add_argument("--target-color", default="blue")
    parser.add_argument("--claimed-colors", default="red,green")
    parser.add_argument("--bearings", required=True, help="Comma-separated bearing radians.")
    parser.add_argument("--areas", required=True, help="Comma-separated area logits.")
    parser.add_argument("--forward-bearing", type=float, default=0.25)
    parser.add_argument("--arc-bearing", type=float, default=0.45)
    parser.add_argument("--arc-area-logit", type=float, default=0.0)
    parser.add_argument("--mem-conf", type=float, default=0.97)
    parser.add_argument(
        "--mem-confs",
        default="",
        help="Optional comma-separated memory confidences. Overrides --mem-conf.",
    )
    parser.add_argument("--read-score", type=float, default=0.0)
    parser.add_argument(
        "--read-scores",
        default="",
        help="Optional comma-separated read scores. Overrides --read-score.",
    )
    parser.add_argument(
        "--states",
        default="",
        help=(
            "Optional comma-separated runtime states to synthesize. When unset, "
            "preserves the historical state assignment."
        ),
    )
    parser.add_argument(
        "--runtime-in-cone-from-positive-area",
        action="store_true",
        help=(
            "Match benchmark runtime visual-readout semantics: in_cone is true "
            "only when the active target area logit is positive."
        ),
    )
    parser.add_argument(
        "--rewrite-primitive-outcomes",
        action="store_true",
        help=(
            "Rewrite the base primitive outcome slice so synthetic visual-target "
            "rows have coherent learned obstacle/progress context."
        ),
    )
    parser.add_argument(
        "--outcome-contexts",
        default="open",
        help=(
            "Comma-separated synthetic outcome contexts used with "
            "--rewrite-primitive-outcomes. Supported: open, forward_blocked, "
            "arc_open, translating_blocked, clearance_risk, blocked_center_escape, "
            "target_standoff_escape, target_servo_approach."
        ),
    )
    parser.add_argument(
        "--clear-last-primitive",
        action="store_true",
        help="Zero the base last-primitive one-hot slice instead of inheriting stale source context.",
    )
    parser.add_argument(
        "--online-map-reset-each-sample",
        action="store_true",
        help=(
            "Mark synthetic rows as local-map reset points for the training-time "
            "offline egomotion-map replay."
        ),
    )
    parser.add_argument(
        "--blocked-center-map-sequence",
        action="store_true",
        help=(
            "For blocked_center_escape rows labelled backward, insert an "
            "update-only stalled forward probe immediately before the supervised "
            "backward row. The training map replay uses the probe to mark the "
            "cell ahead as blocked, then drops it from the supervised dataset."
        ),
    )
    parser.add_argument(
        "--blocked-center-backward-bearing",
        type=float,
        default=0.0,
        help=(
            "Optional absolute bearing threshold for blocked_center_escape "
            "backward labels. When <=0, uses the historical narrow threshold."
        ),
    )
    parser.add_argument(
        "--guard-probe-labels",
        default="",
        help=(
            "Optional comma-separated supervised labels that should receive a "
            "preceding update-only guard-blocked probe row."
        ),
    )
    parser.add_argument(
        "--guard-probe-primitive",
        default="forward_medium",
        help="Primitive name to mark as guard-blocked for --guard-probe-labels.",
    )
    parser.add_argument("--vector-scale", type=float, default=0.18)
    parser.add_argument("--max-source-rows", type=int, default=0)
    args = parser.parse_args()

    color_vocab = [item.strip() for item in str(args.color_vocab).split(",") if item.strip()]
    primitive_vocab: list[str] | None = None
    target_color = str(args.target_color)
    if target_color not in color_vocab:
        raise SystemExit(f"--target-color {target_color!r} is not in --color-vocab")
    target_idx = color_vocab.index(target_color)
    claimed_colors = [item.strip() for item in str(args.claimed_colors).split(",") if item.strip()]
    unknown_claims = [color for color in claimed_colors if color not in color_vocab]
    if unknown_claims:
        raise SystemExit(f"unknown claimed colors: {unknown_claims}")
    bearings = _parse_float_list(args.bearings, "--bearings")
    areas = _parse_float_list(args.areas, "--areas")
    mem_confs = (
        _parse_float_list(args.mem_confs, "--mem-confs")
        if str(args.mem_confs).strip()
        else [float(args.mem_conf)]
    )
    read_scores = (
        _parse_float_list(args.read_scores, "--read-scores")
        if str(args.read_scores).strip()
        else [float(args.read_score)]
    )
    synth_states = [
        item.strip().upper()
        for item in str(args.states).split(",")
        if item.strip()
    ]
    outcome_contexts = (
        _parse_string_list(args.outcome_contexts, "--outcome-contexts")
        if bool(args.rewrite_primitive_outcomes)
        else [""]
    )
    guard_probe_labels = (
        set(_parse_string_list(args.guard_probe_labels, "--guard-probe-labels"))
        if str(args.guard_probe_labels).strip()
        else set()
    )

    row_arrays: dict[str, list[np.ndarray]] = {"features": [], "labels": [], "meta_json": []}
    static_arrays: dict[str, np.ndarray] = {}
    source_reports: list[dict[str, Any]] = []
    output_rows = 0
    source_rows = 0

    for input_path in args.input:
        with np.load(input_path, allow_pickle=False) as data:
            schema = str(data["schema"][0]) if "schema" in data else ""
            if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
                raise SystemExit(f"unsupported dataset schema in {input_path}: {schema}")
            current_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
            if primitive_vocab is None:
                primitive_vocab = current_vocab
            elif primitive_vocab != current_vocab:
                raise SystemExit(f"primitive vocab mismatch in {input_path}")
            for required in ("forward_medium", "arc_left", "arc_right", "yaw_left", "yaw_right"):
                if required not in current_vocab:
                    raise SystemExit(f"{input_path} primitive_vocab is missing {required!r}")
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            meta_rows = [json.loads(str(item)) for item in np.asarray(data["meta_json"]).tolist()]
            source_variant = _source_feature_variant_from_data(data, meta_rows=meta_rows)
            if features.ndim != 2 or features.shape[0] != labels.shape[0]:
                raise SystemExit(f"bad feature/label shape in {input_path}")
            if features.shape[0] != len(meta_rows):
                raise SystemExit(f"bad meta row count in {input_path}")
            source_rows += int(features.shape[0])
            keep = [
                idx
                for idx, row in enumerate(meta_rows)
                if str(row.get("target_color", target_color)) == target_color
            ]
            if int(args.max_source_rows) > 0:
                keep = keep[: int(args.max_source_rows)]
            if not keep:
                raise SystemExit(f"{input_path} has no template rows for target_color={target_color}")

            out_features: list[np.ndarray] = []
            out_labels: list[int] = []
            out_meta: list[str] = []
            for source_idx in keep:
                base_feature = features[source_idx]
                base_meta = meta_rows[source_idx]
                for area in areas:
                    for bearing in bearings:
                        base_label = _teacher_label(
                            bearing,
                            area=area,
                            forward_bearing=float(args.forward_bearing),
                            arc_bearing=float(args.arc_bearing),
                            arc_area_logit=float(args.arc_area_logit),
                        )
                        for mem_conf in mem_confs:
                            for read_score in read_scores:
                                in_cone = (
                                    float(area) > 0.0
                                    if bool(args.runtime_in_cone_from_positive_area)
                                    else abs(float(bearing)) <= float(args.forward_bearing)
                                )
                                for outcome_context in outcome_contexts:
                                    label = _outcome_context_label(
                                        base_label,
                                        outcome_context=outcome_context,
                                        bearing=bearing,
                                        area=area,
                                        forward_bearing=float(args.forward_bearing),
                                        arc_bearing=float(args.arc_bearing),
                                        blocked_center_backward_bearing=(
                                            float(args.blocked_center_backward_bearing)
                                        ),
                                    )
                                    row_feature = _rewrite_feature(
                                        base_feature,
                                        color_vocab=color_vocab,
                                        primitive_vocab=current_vocab,
                                        target_idx=target_idx,
                                        claimed_colors=claimed_colors,
                                        bearing=bearing,
                                        area=area,
                                        mem_conf=float(mem_conf),
                                        vector_scale=float(args.vector_scale),
                                        base_dim=int(args.base_dim),
                                        memory_vec_start=int(args.memory_vec_start),
                                        memory_conf_start=int(args.memory_conf_start),
                                        rewrite_primitive_outcomes=bool(args.rewrite_primitive_outcomes),
                                        outcome_context=outcome_context,
                                        outcome_label=label,
                                        clear_last_primitive=bool(args.clear_last_primitive),
                                    )
                                    row_feature = _rewrite_existing_visual_readout_suffix(
                                        row_feature,
                                        feature_variant=source_variant,
                                        base_dim=int(args.base_dim),
                                        area=area,
                                        bearing=bearing,
                                        mem_conf=mem_conf,
                                        read_score=read_score,
                                        in_cone=bool(in_cone),
                                        claimed_count=len(claimed_colors),
                                        color_count=len(color_vocab),
                                    )
                                    default_state = (
                                        "SEEK"
                                        if label != "forward_medium"
                                        else str(base_meta.get("state", "SEEK")).upper()
                                    )
                                    for synth_state in (synth_states or [default_state]):
                                        row_meta = dict(base_meta)
                                        row_meta.update(
                                            {
                                                "area": float(area),
                                                "bearing": float(bearing),
                                                "claimed_colors_augmented": list(claimed_colors),
                                                "claimed_count": int(len(claimed_colors)),
                                                "in_cone": bool(in_cone),
                                                "label": label,
                                                "mem_conf": float(mem_conf),
                                                "read_score": float(read_score),
                                                "relabel_source": "synthetic_visual_bearing",
                                                "relabel_source_primitive": label,
                                                "seen": True,
                                                "state": str(synth_state).upper(),
                                                "synthetic_base_label": base_label,
                                                "synthetic_outcome_context": str(outcome_context or "source"),
                                                "synthetic_primitive_outcomes": bool(args.rewrite_primitive_outcomes),
                                                "target_color": target_color,
                                                "target_index": int(target_idx),
                                                "visual_servo_area": float(area),
                                                "visual_servo_bearing": float(bearing),
                                                "visual_servo_mem_conf": float(mem_conf),
                                                "visual_bearing_synthesis": True,
                                            }
                                        )
                                        row_meta.setdefault("pose_xy", [0.0, 0.0])
                                        row_meta.setdefault("yaw_rad", 0.0)
                                        row_meta["online_map_reset"] = bool(
                                            args.online_map_reset_each_sample
                                        )
                                        inserted_guard_probe = False
                                        if (
                                            guard_probe_labels
                                            and str(label) in guard_probe_labels
                                            and str(args.guard_probe_primitive) in current_vocab
                                        ):
                                            probe_meta = dict(row_meta)
                                            probe_meta.update(
                                                {
                                                    "label": str(args.guard_probe_primitive),
                                                    "online_map_guard_blocked_primitive": str(
                                                        args.guard_probe_primitive
                                                    ),
                                                    "online_map_guard_blocked_probe": True,
                                                    "online_map_reset": True,
                                                    "online_map_update_only": True,
                                                    "relabel_source_primitive": str(
                                                        args.guard_probe_primitive
                                                    ),
                                                    "synthetic_map_role": "guard_blocked_probe",
                                                    "training_update_only": True,
                                                }
                                            )
                                            row_meta.update(
                                                {
                                                    "online_map_reset": False,
                                                    "synthetic_map_role": "guard_probe_supervised",
                                                }
                                            )
                                            out_features.append(row_feature.copy())
                                            out_labels.append(
                                                current_vocab.index(str(args.guard_probe_primitive))
                                            )
                                            out_meta.append(json.dumps(probe_meta, sort_keys=True))
                                            inserted_guard_probe = True
                                        if (
                                            bool(args.blocked_center_map_sequence)
                                            and str(outcome_context) == "blocked_center_escape"
                                            and str(label) == "backward"
                                            and "forward_medium" in current_vocab
                                            and not inserted_guard_probe
                                        ):
                                            probe_meta = dict(row_meta)
                                            probe_meta.update(
                                                {
                                                    "label": "forward_medium",
                                                    "online_map_guard_blocked_primitive": "forward_medium",
                                                    "online_map_guard_blocked_probe": True,
                                                    "online_map_reset": True,
                                                    "online_map_update_only": True,
                                                    "relabel_source_primitive": "forward_medium",
                                                    "synthetic_map_role": "blocked_center_forward_probe",
                                                    "training_update_only": True,
                                                }
                                            )
                                            row_meta.update(
                                                {
                                                    "online_map_reset": False,
                                                    "synthetic_map_role": "blocked_center_backward_escape",
                                                }
                                            )
                                            out_features.append(row_feature.copy())
                                            out_labels.append(current_vocab.index("forward_medium"))
                                            out_meta.append(json.dumps(probe_meta, sort_keys=True))
                                        out_features.append(row_feature)
                                        out_labels.append(current_vocab.index(label))
                                        out_meta.append(json.dumps(row_meta, sort_keys=True))
            row_arrays["features"].append(np.stack(out_features).astype(np.float32))
            row_arrays["labels"].append(np.asarray(out_labels, dtype=np.int64))
            row_arrays["meta_json"].append(np.asarray(out_meta))
            output_rows += len(out_labels)

            for key in data.files:
                if key in {"features", "labels", "meta_json", "result_json", "filter_report_json", "relabel_report_json"}:
                    continue
                value = np.asarray(data[key])
                if key not in static_arrays:
                    static_arrays[key] = value
                elif not np.array_equal(static_arrays[key], value):
                    raise SystemExit(f"static array mismatch for key {key!r} in {input_path}")
            source_reports.append(
                {
                    "path": str(input_path),
                    "rows": int(features.shape[0]),
                    "template_rows": int(len(keep)),
                    "feature_dim": int(features.shape[1]),
                    "feature_variant": source_variant,
                }
            )

    if primitive_vocab is None:
        raise SystemExit("no input datasets loaded")
    arrays = dict(static_arrays)
    arrays["features"] = np.concatenate(row_arrays["features"], axis=0)
    arrays["labels"] = np.concatenate(row_arrays["labels"], axis=0)
    arrays["meta_json"] = np.concatenate(row_arrays["meta_json"], axis=0)
    report = {
        "schema": "lewm_go2_visual_bearing_synthesis_v0",
        "inputs": [str(path) for path in args.input],
        "source_rows": int(source_rows),
        "output_rows": int(output_rows),
        "color_vocab": list(color_vocab),
        "target_color": target_color,
        "claimed_colors": list(claimed_colors),
        "bearings": bearings,
        "areas": areas,
        "mem_confs": mem_confs,
        "read_scores": read_scores,
        "states": list(synth_states),
        "rewrite_primitive_outcomes": bool(args.rewrite_primitive_outcomes),
        "outcome_contexts": list(outcome_contexts),
        "clear_last_primitive": bool(args.clear_last_primitive),
        "online_map_reset_each_sample": bool(args.online_map_reset_each_sample),
        "blocked_center_map_sequence": bool(args.blocked_center_map_sequence),
        "blocked_center_backward_bearing": float(args.blocked_center_backward_bearing),
        "guard_probe_labels": sorted(guard_probe_labels),
        "guard_probe_primitive": str(args.guard_probe_primitive),
        "runtime_in_cone_from_positive_area": bool(args.runtime_in_cone_from_positive_area),
        "forward_bearing": float(args.forward_bearing),
        "arc_bearing": float(args.arc_bearing),
        "arc_area_logit": float(args.arc_area_logit),
        "sources": source_reports,
        "wall_metrics": {
            "learned_local_policy_feature_variant": _source_feature_variant(args.input[0]),
            "learned_local_post_claim_policy_feature_variant": _source_feature_variant(args.input[0]),
            "visual_bearing_synthesis": True,
        },
    }
    arrays["result_json"] = np.asarray([json.dumps(report, sort_keys=True)])
    arrays["visual_bearing_synthesis_report_json"] = np.asarray(
        [json.dumps(report, sort_keys=True)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_float_list(raw: str, flag_name: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{flag_name} must contain at least one value")
    return values


def _parse_string_list(raw: str, flag_name: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{flag_name} must contain at least one value")
    return values


def _teacher_label(
    bearing: float,
    *,
    area: float,
    forward_bearing: float,
    arc_bearing: float,
    arc_area_logit: float,
) -> str:
    if abs(float(bearing)) <= float(forward_bearing):
        return "forward_medium"
    if float(area) <= float(arc_area_logit) and abs(float(bearing)) >= float(arc_bearing):
        return "arc_left" if float(bearing) > 0.0 else "arc_right"
    return "yaw_left" if float(bearing) > 0.0 else "yaw_right"


def _outcome_context_label(
    base_label: str,
    *,
    outcome_context: str,
    bearing: float,
    area: float,
    forward_bearing: float,
    arc_bearing: float,
    blocked_center_backward_bearing: float = 0.0,
) -> str:
    context = str(outcome_context or "open")
    signed_arc = "arc_left" if float(bearing) >= 0.0 else "arc_right"
    signed_yaw = "yaw_left" if float(bearing) >= 0.0 else "yaw_right"
    abs_bearing = abs(float(bearing))
    if context in {"open", "source"}:
        return str(base_label)
    if context == "forward_blocked":
        if abs_bearing <= max(float(arc_bearing) * 2.0, float(forward_bearing)):
            return signed_arc
        return signed_yaw
    if context == "arc_open":
        if abs_bearing > float(forward_bearing):
            return signed_arc
        return str(base_label)
    if context == "translating_blocked":
        return signed_yaw
    if context == "clearance_risk":
        if float(area) <= 0.0:
            return signed_yaw
        if abs_bearing <= max(float(arc_bearing) * 2.0, float(forward_bearing)):
            return signed_arc
        return signed_yaw
    if context == "blocked_center_escape":
        backward_bearing = (
            float(blocked_center_backward_bearing)
            if float(blocked_center_backward_bearing) > 0.0
            else max(float(forward_bearing) * 1.4, 0.28)
        )
        if float(area) <= 0.0 and abs_bearing <= backward_bearing:
            return "backward"
        if abs_bearing <= max(float(arc_bearing) * 2.0, float(forward_bearing)):
            return signed_arc
        return signed_yaw
    if context == "target_standoff_escape":
        backward_bearing = max(float(arc_bearing) * 1.55, float(forward_bearing) * 2.2, 0.68)
        if float(area) <= 0.25 and abs_bearing <= backward_bearing:
            return "backward"
        if abs_bearing <= max(float(arc_bearing) * 1.8, float(forward_bearing)):
            return signed_arc
        return signed_yaw
    if context == "target_servo_approach":
        if float(area) > 0.0 and abs_bearing <= max(float(forward_bearing), 0.25):
            return "forward_medium"
        if abs_bearing <= max(float(arc_bearing) * 1.4, float(forward_bearing)):
            return signed_arc
        return signed_yaw
    raise SystemExit(f"unsupported --outcome-contexts item: {outcome_context!r}")


def _rewrite_feature(
    feature: np.ndarray,
    *,
    color_vocab: list[str],
    primitive_vocab: list[str],
    target_idx: int,
    claimed_colors: list[str],
    bearing: float,
    area: float,
    mem_conf: float,
    vector_scale: float,
    base_dim: int,
    memory_vec_start: int,
    memory_conf_start: int,
    rewrite_primitive_outcomes: bool,
    outcome_context: str,
    outcome_label: str,
    clear_last_primitive: bool,
) -> np.ndarray:
    color_count = len(color_vocab)
    primitive_count = len(primitive_vocab)
    outcome_dim = primitive_count * 3
    outcome_start = int(base_dim) - outcome_dim
    last_primitive_start = outcome_start - primitive_count
    claimed_start = last_primitive_start - color_count
    active_start = claimed_start - color_count
    evidence_start = active_start - color_count * 2
    area_start = evidence_start - color_count

    out = np.asarray(feature, dtype=np.float32).copy()
    if out.shape[0] < int(base_dim):
        raise SystemExit(f"feature_dim={out.shape[0]} < base_dim={base_dim}")
    vec = np.asarray(
        [math.cos(float(bearing)) * float(vector_scale), math.sin(float(bearing)) * float(vector_scale)],
        dtype=np.float32,
    )
    out[int(memory_vec_start) + 2 * target_idx : int(memory_vec_start) + 2 * target_idx + 2] = vec
    out[int(memory_conf_start) + target_idx] = float(mem_conf)
    out[area_start + target_idx] = float(area)
    out[evidence_start + 2 * target_idx : evidence_start + 2 * target_idx + 2] = vec
    out[active_start : active_start + color_count] = 0.0
    out[active_start + target_idx] = 1.0
    claim_mask = np.asarray([1.0 if color in set(claimed_colors) else 0.0 for color in color_vocab], dtype=np.float32)
    out[claimed_start : claimed_start + color_count] = claim_mask
    if bool(clear_last_primitive):
        out[last_primitive_start : last_primitive_start + primitive_count] = 0.0
    if bool(rewrite_primitive_outcomes):
        _rewrite_primitive_outcome_slice(
            out,
            primitive_vocab=primitive_vocab,
            outcome_start=outcome_start,
            outcome_context=outcome_context,
            outcome_label=outcome_label,
            bearing=bearing,
        )
    return out


def _rewrite_primitive_outcome_slice(
    out: np.ndarray,
    *,
    primitive_vocab: list[str],
    outcome_start: int,
    outcome_context: str,
    outcome_label: str,
    bearing: float,
) -> None:
    context = str(outcome_context or "open")
    signed_arc = "arc_left" if float(bearing) >= 0.0 else "arc_right"
    opposite_arc = "arc_right" if signed_arc == "arc_left" else "arc_left"

    values: dict[str, tuple[float, float, float]] = {
        primitive: (0.72, 0.0, 0.72) for primitive in primitive_vocab
    }
    if "hold" in values:
        values["hold"] = (0.02, 0.0, 0.02)
    if "backward" in values:
        values["backward"] = (0.34, -0.02, 0.34)

    def set_if_present(name: str, blocked: float, progress: float, clearance_blocked: float) -> None:
        if name in values:
            values[name] = (float(blocked), float(progress), float(clearance_blocked))

    if context in {"open", "source"}:
        set_if_present("forward_fast", 0.12, 0.26, 0.14)
        set_if_present("forward_medium", 0.05, 0.22, 0.08)
        set_if_present("arc_left", 0.10, 0.15, 0.12)
        set_if_present("arc_right", 0.10, 0.15, 0.12)
        set_if_present("yaw_left", 0.02, 0.04, 0.02)
        set_if_present("yaw_right", 0.02, 0.04, 0.02)
    elif context == "forward_blocked":
        set_if_present("forward_fast", 0.96, 0.0, 0.94)
        set_if_present("forward_medium", 0.93, 0.01, 0.90)
        set_if_present("arc_left", 0.12, 0.12, 0.14)
        set_if_present("arc_right", 0.12, 0.12, 0.14)
        set_if_present("yaw_left", 0.02, 0.04, 0.02)
        set_if_present("yaw_right", 0.02, 0.04, 0.02)
    elif context == "arc_open":
        set_if_present("forward_fast", 0.45, 0.06, 0.48)
        set_if_present("forward_medium", 0.36, 0.08, 0.40)
        set_if_present(signed_arc, 0.04, 0.17, 0.06)
        set_if_present(opposite_arc, 0.54, 0.03, 0.58)
        set_if_present("yaw_left", 0.03, 0.04, 0.03)
        set_if_present("yaw_right", 0.03, 0.04, 0.03)
    elif context == "translating_blocked":
        for primitive in ("forward_fast", "forward_medium", "arc_left", "arc_right", "backward"):
            set_if_present(primitive, 0.93, 0.0, 0.91)
        set_if_present("yaw_left", 0.02, 0.04, 0.02)
        set_if_present("yaw_right", 0.02, 0.04, 0.02)
    elif context == "clearance_risk":
        set_if_present("forward_fast", 0.08, 0.08, 0.86)
        set_if_present("forward_medium", 0.01, 0.09, 0.82)
        set_if_present("arc_left", 0.04, 0.09, 0.87)
        set_if_present("arc_right", 0.01, 0.09, 0.87)
        set_if_present("backward", 0.01, 0.07, 0.79)
        set_if_present("yaw_left", 0.0, 0.08, 0.82)
        set_if_present("yaw_right", 0.0, 0.08, 0.84)
        set_if_present("hold", 0.0, 0.08, 0.68)
    elif context == "blocked_center_escape":
        set_if_present("forward_fast", 0.98, 0.0, 0.94)
        set_if_present("forward_medium", 0.96, 0.0, 0.91)
        set_if_present("arc_left", 0.90, 0.0, 0.88)
        set_if_present("arc_right", 0.90, 0.0, 0.88)
        set_if_present("backward", 0.02, 0.12, 0.06)
        set_if_present("yaw_left", 0.05, 0.02, 0.12)
        set_if_present("yaw_right", 0.05, 0.02, 0.12)
        set_if_present("hold", 0.0, 0.0, 0.65)
    elif context == "target_standoff_escape":
        set_if_present("forward_fast", 0.94, 0.0, 0.92)
        set_if_present("forward_medium", 0.92, 0.0, 0.88)
        set_if_present("arc_left", 0.58, 0.05, 0.60)
        set_if_present("arc_right", 0.58, 0.05, 0.60)
        set_if_present(signed_arc, 0.18, 0.10, 0.20)
        set_if_present(opposite_arc, 0.78, 0.0, 0.82)
        set_if_present("backward", 0.02, 0.12, 0.05)
        set_if_present("yaw_left", 0.04, 0.03, 0.08)
        set_if_present("yaw_right", 0.04, 0.03, 0.08)
        set_if_present("hold", 0.08, 0.0, 0.88)
    elif context == "target_servo_approach":
        set_if_present("forward_fast", 0.14, 0.26, 0.16)
        set_if_present("forward_medium", 0.03, 0.24, 0.06)
        set_if_present(signed_arc, 0.04, 0.18, 0.07)
        set_if_present(opposite_arc, 0.62, 0.02, 0.66)
        set_if_present("backward", 0.42, -0.02, 0.44)
        set_if_present("yaw_left", 0.03, 0.04, 0.06)
        set_if_present("yaw_right", 0.03, 0.04, 0.06)
        set_if_present("hold", 0.08, 0.0, 0.84)
    else:
        raise SystemExit(f"unsupported outcome context {outcome_context!r}")

    if outcome_label in values:
        if str(outcome_label).startswith("forward"):
            values[outcome_label] = (0.02, 0.24, 0.05)
        elif str(outcome_label).startswith("arc_"):
            values[outcome_label] = (0.02, 0.17, 0.05)
        elif str(outcome_label).startswith("yaw_"):
            values[outcome_label] = (0.01, 0.05, 0.01)
        elif str(outcome_label) == "backward":
            values[outcome_label] = (0.02, 0.12, 0.05)

    for primitive_idx, primitive in enumerate(primitive_vocab):
        blocked, progress, clearance_blocked = values[primitive]
        start = int(outcome_start) + int(primitive_idx) * 3
        out[start : start + 3] = np.asarray(
            [blocked, progress, clearance_blocked],
            dtype=np.float32,
        )


def _rewrite_existing_visual_readout_suffix(
    feature: np.ndarray,
    *,
    feature_variant: str,
    base_dim: int,
    area: float,
    bearing: float,
    mem_conf: float,
    read_score: float,
    in_cone: bool,
    claimed_count: int,
    color_count: int,
) -> np.ndarray:
    if "visual_readout" not in str(feature_variant):
        return feature
    visual_start = _visual_readout_start(feature_variant, base_dim=int(base_dim))
    if visual_start < 0 or visual_start + 8 > int(feature.shape[0]):
        return feature
    out = np.asarray(feature, dtype=np.float32).copy()
    out[visual_start : visual_start + 8] = np.asarray(
        [
            float(area) / 4.0,
            math.sin(float(bearing)),
            math.cos(float(bearing)),
            float(bearing) / math.pi,
            float(mem_conf),
            float(read_score),
            1.0 if bool(in_cone) else 0.0,
            float(claimed_count) / max(1.0, float(color_count)),
        ],
        dtype=np.float32,
    )
    return out


def _visual_readout_start(feature_variant: str, *, base_dim: int) -> int:
    start = int(base_dim)
    text = str(feature_variant)
    prefix = text.split("visual_readout", 1)[0]
    if "pose_topology" in prefix:
        start += 5
    if "clock" in prefix:
        start += 3
    if "state" in prefix:
        start += 4
    return start


def _source_feature_variant(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta_rows = (
                [json.loads(str(item)) for item in np.asarray(data["meta_json"]).tolist()]
                if "meta_json" in data
                else []
            )
            return _source_feature_variant_from_data(data, meta_rows=meta_rows)
    except Exception:
        return "base"
    return "base"


def _source_feature_variant_from_data(
    data: np.lib.npyio.NpzFile,
    *,
    meta_rows: list[dict[str, Any]],
) -> str:
    if "result_json" not in data or len(data["result_json"]) <= 0:
        return "base"
    report = json.loads(str(data["result_json"][0]))
    if isinstance(report, dict) and isinstance(report.get("result"), dict):
        report = report["result"]
    metrics = report.get("wall_metrics", {}) if isinstance(report, dict) else {}
    if not isinstance(metrics, dict):
        return "base"
    slots = {
        str(row.get("policy_feature_slot"))
        for row in meta_rows
        if isinstance(row, dict) and row.get("policy_feature_slot") is not None
    }
    if slots and slots <= {"post_claim"}:
        post_variant = metrics.get("learned_local_post_claim_policy_feature_variant")
        if post_variant is not None:
            return str(post_variant)
    return str(metrics.get("learned_local_policy_feature_variant", "base"))


if __name__ == "__main__":
    raise SystemExit(main())
