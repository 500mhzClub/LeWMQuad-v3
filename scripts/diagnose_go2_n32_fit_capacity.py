#!/usr/bin/env python3
"""Fit-only keyed-logit oracle for the N32 labels, controls, and gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract  # noqa: E402
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    empty_raw_accumulator,
    finalize_raw_accumulator,
    update_raw_accumulator,
)
from scripts import run_go2_dynamic_cartesian_n32 as runner  # noqa: E402


SCHEMA = "lewm_go2_n32_fit_capacity_diagnostic_v1"
CANONICAL_OUTPUT = (
    REPOSITORY_ROOT
    / ".generated/go2_n32_fit_capacity/keyed_oracle_v1/result.json"
).resolve()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--logit-margin", type=float, default=12.0)
    args = parser.parse_args(argv)
    try:
        args.output = runner._canonical_path(args.output, name="diagnostic output")
        contract.validate_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output != CANONICAL_OUTPUT:
        parser.error("fit-capacity output path is not canonical")
    if args.output.exists():
        parser.error("fit-capacity result already exists")
    if not np.isfinite(args.logit_margin) or args.logit_margin <= 0.0:
        parser.error("logit margin must be finite and positive")
    return args


def perfect_logits(labels: np.ndarray, margin: float) -> np.ndarray:
    truth = np.asarray(labels)
    if truth.ndim != 3 or truth.shape[1:] != (64, 64):
        raise ValueError("labels must have shape (N,64,64)")
    if truth.size and (truth.min() < 0 or truth.max() > 2):
        raise ValueError("labels must be UNKNOWN/FREE/OCCUPIED")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    logits = np.full((truth.shape[0], 3, 64, 64), -float(margin), dtype=np.float32)
    np.put_along_axis(logits, truth[:, None].astype(np.int64), float(margin), axis=1)
    return logits


def _control_source_indices(
    records: Sequence[Mapping[str, Any]], key: str
) -> np.ndarray:
    by_image = {str(record["image_sha256"]): index for index, record in enumerate(records)}
    if len(by_image) != len(records):
        raise ValueError("fit image identities are not unique")
    try:
        indices = [by_image[str(record[key])] for record in records]
    except KeyError as exc:
        raise ValueError(f"control source is outside the fit panel: {key}") from exc
    return np.asarray(indices, dtype=np.int64)


def _targets(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    shard_contract = sorted(
        {
            (str(record["label_shard_path"]), str(record["label_shard_sha256"]))
            for record in records
        }
    )
    with ThreadPoolExecutor(max_workers=runner.SOURCE_WORKERS) as pool:
        decoded = dict(
            pool.map(lambda item: (item[0], runner._decode_shard(*item)), shard_contract)
        )
    labels = []
    masks = []
    for record in records:
        shard = decoded[str(record["label_shard_path"])]
        side = str(record["side"])
        row = int(record["label_shard_row"])
        labels.append(np.asarray(shard[f"{side}_labels"][row], dtype=np.int64))
        masks.append(np.asarray(shard[f"{side}_supervision_mask"][row], dtype=bool))
    return np.stack(labels), np.stack(masks)


def _report(
    records: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    labels: np.ndarray,
    masks: np.ndarray,
    *,
    seed: int,
    margin: float,
) -> dict[str, Any]:
    logits = perfect_logits(labels, margin)
    source_indices = {
        "correct_rgb": np.arange(len(records), dtype=np.int64),
        "role_global_shuffled_rgb": _control_source_indices(
            records, "control_image_sha256"
        ),
        "same_scene_wrong_view_rgb": _control_source_indices(
            records, "same_scene_control_image_sha256"
        ),
    }
    aggregate = {name: empty_raw_accumulator() for name in contract.CONDITIONS}
    families = {
        family: {name: empty_raw_accumulator() for name in contract.CONDITIONS}
        for family in contract.FAMILIES
    }
    distances = runner._distance_grid()
    for condition in contract.CONDITIONS:
        values = logits[source_indices[condition]]
        update_raw_accumulator(aggregate[condition], values, labels, masks, distances)
        for family in contract.FAMILIES:
            selected = np.asarray(
                [str(record["family"]) == family for record in records], dtype=bool
            )
            update_raw_accumulator(
                families[family][condition],
                values[selected],
                labels[selected],
                masks[selected],
                distances,
            )
    report: dict[str, Any] = {
        "schema": contract.PANEL_REPORT_SCHEMA,
        "panel": "fit",
        "frame_count": contract.FRAME_COUNT,
        "target_batch_size": contract.BATCH_SIZE,
        "combined_model_batch_size": 12,
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "wrong_rgb_uses_target_attitude": True,
        "conditions": {
            name: finalize_raw_accumulator(aggregate[name])
            for name in contract.CONDITIONS
        },
        "families": {
            family: {
                "conditions": {
                    name: finalize_raw_accumulator(families[family][name])
                    for name in contract.CONDITIONS
                }
            }
            for family in contract.FAMILIES
        },
        "controls": dict(controls),
    }
    report["fit_gate"] = contract.fit_panel_gate_report(report)
    contract.validate_panel_report(
        report, seed=seed, panel="fit", require_fit_gate=True
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _panel, joined, _reference, _parity, join_audit, _support = (
        runner._load_bound_inputs()
    )
    records, controls = runner._canonical_records(
        joined["fit"], seed=args.seed, panel="fit"
    )
    labels, masks = _targets(records)
    report = _report(
        records,
        controls,
        labels,
        masks,
        seed=args.seed,
        margin=float(args.logit_margin),
    )
    core = {
        "schema": SCHEMA,
        "authoritative": False,
        "development_fit_only": True,
        "seed": args.seed,
        "logit_margin": float(args.logit_margin),
        "mechanism": "perfect_image_identity_to_target_logits_lookup",
        "learned_parameters": 0,
        "image_payload_opens": 0,
        "label_shard_opens": 20,
        "non_fit_payload_opens": 0,
        "g2_payload_opens": 0,
        "panel_join": join_audit,
        "fit_report": report,
        "decision": {
            "complete_fit_gate_passes": bool(report["fit_gate"]["passes"]),
            "labels_controls_and_gate_attainable": bool(
                report["fit_gate"]["passes"]
            ),
            "licenses_model_or_generalization_claim": False,
        },
    }
    result = {**core, "content_sha256": contract.canonical_json_sha256(core)}
    runner._publish_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_sha256": result["content_sha256"],
                "fit_gate_passes": report["fit_gate"]["passes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
