#!/usr/bin/env python3
"""Read-only diagnostic for a V4 checkpoint/result inference mismatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from scripts import train_go2_observable_camera_ray_fit_v4 as trainer


ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE_SHA256 = (
    "299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa"
)


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _diff(
    expected: object,
    actual: object,
    *,
    path: str = "$",
) -> list[dict[str, object]]:
    if type(expected) is not type(actual):
        return [
            {
                "path": path,
                "kind": "type",
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__,
            }
        ]
    if isinstance(expected, dict):
        rows: list[dict[str, object]] = []
        expected_keys = set(expected)
        actual_keys = set(actual)  # type: ignore[arg-type]
        for key in sorted(expected_keys | actual_keys):
            if key not in expected_keys or key not in actual_keys:
                rows.append(
                    {
                        "path": f"{path}.{key}",
                        "kind": "missing",
                        "expected_present": key in expected_keys,
                        "actual_present": key in actual_keys,
                    }
                )
            else:
                rows.extend(
                    _diff(
                        expected[key],
                        actual[key],  # type: ignore[index]
                        path=f"{path}.{key}",
                    )
                )
        return rows
    if isinstance(expected, list):
        if len(expected) != len(actual):  # type: ignore[arg-type]
            return [
                {
                    "path": path,
                    "kind": "length",
                    "expected": len(expected),
                    "actual": len(actual),  # type: ignore[arg-type]
                }
            ]
        rows = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):  # type: ignore[arg-type]
            rows.extend(_diff(left, right, path=f"{path}[{index}]"))
        return rows
    if isinstance(expected, float):
        if math.isnan(expected) or math.isnan(actual):  # type: ignore[arg-type]
            equal = math.isnan(expected) and math.isnan(actual)  # type: ignore[arg-type]
        else:
            equal = expected == actual
        return [] if equal else [
            {
                "path": path,
                "kind": "float",
                "expected": expected,
                "actual": actual,
                "absolute_difference": abs(expected - actual),  # type: ignore[operator]
            }
        ]
    if expected != actual:
        return [
            {
                "path": path,
                "kind": "value",
                "expected": expected,
                "actual": actual,
            }
        ]
    return []


def _metric_summary(evaluation: dict[str, Any]) -> dict[str, object]:
    matched = evaluation["matched_rgb"]["metrics"]
    wrong = evaluation["wrong_rgb_with_target_calibration"]["metrics"]
    return {
        "matched": {
            "pixel_hit_balanced_accuracy": matched["pixel_hit_no_hit"][
                "balanced_accuracy"
            ],
            "depth_median_m": matched["pixel_hit_depth"][
                "median_absolute_error_m"
            ],
            "depth_p95_m": matched["pixel_hit_depth"]["p95_absolute_error_m"],
            "ground_balanced_accuracy": matched["ground_clear"]["overall"][
                "balanced_accuracy"
            ],
            "raster_nll": matched["derived_raster"]["nll"],
            "raster_balanced_accuracy": matched["derived_raster"][
                "balanced_accuracy"
            ],
            "raster_class_recalls": matched["derived_raster"]["class_recalls"],
        },
        "wrong": {
            "pixel_hit_balanced_accuracy": wrong["pixel_hit_no_hit"][
                "balanced_accuracy"
            ],
            "depth_median_m": wrong["pixel_hit_depth"][
                "median_absolute_error_m"
            ],
            "depth_p95_m": wrong["pixel_hit_depth"]["p95_absolute_error_m"],
            "ground_balanced_accuracy": wrong["ground_clear"]["overall"][
                "balanced_accuracy"
            ],
            "raster_nll": wrong["derived_raster"]["nll"],
            "raster_balanced_accuracy": wrong["derived_raster"][
                "balanced_accuracy"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-size", type=int, default=5)
    args = parser.parse_args()

    trainer.validate_gpu0_r9700_runtime(device_text="cuda:0")
    inputs = trainer.load_exact_inputs(
        dataset_manifest_path=trainer.CANONICAL_DATASET_MANIFEST_PATH,
        dataset_manifest_file_sha256=(
            trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256
        ),
        audit_receipt_path=trainer.CANONICAL_AUDIT_RECEIPT_PATH,
        audit_receipt_file_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_FILE_SHA256,
        trainer_authorization_path=trainer.CANONICAL_TRAINER_AUTHORIZATION_PATH,
        trainer_authorization_file_sha256=(
            "d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802"
        ),
        trainer_review_record_path=trainer.CANONICAL_REVIEW_RECORD_PATH,
        trainer_review_record_file_sha256=(
            "c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea"
        ),
        fit_size=args.fit_size,
    )
    images, rgb_access = trainer.decode_selected_rgb(
        inputs.frames,
        maximum_workers=1,
        expected_trainer_source_sha256=TRAINER_SOURCE_SHA256,
    )
    checkpoint_raw = args.checkpoint.read_bytes()
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model = ObservableCameraRayEvidenceV4Model()
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    device = torch.device("cuda:0")
    model.to(device)
    matched = trainer.evaluate_v4_fit(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        batch_size=1,
        wrong_rgb=False,
    )
    wrong = trainer.evaluate_v4_fit(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        batch_size=1,
        wrong_rgb=True,
    )
    actual = {
        "matched_rgb": matched,
        "wrong_rgb_with_target_calibration": wrong,
    }
    result = _read_json(args.result)
    expected = result["evaluation"]
    differences = _diff(expected, actual)
    numeric = sorted(
        (row for row in differences if row["kind"] == "float"),
        key=lambda row: float(row["absolute_difference"]),
        reverse=True,
    )
    report = {
        "schema": "lewm_go2_v4_inference_reproduction_diagnostic_v1",
        "read_only": True,
        "result_file_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "checkpoint_file_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "rgb_access": rgb_access,
        "difference_count": len(differences),
        "numeric_difference_count": len(numeric),
        "non_numeric_differences": [
            row for row in differences if row["kind"] != "float"
        ][:50],
        "largest_numeric_differences": numeric[:100],
        "reported_summary": _metric_summary(expected),
        "recomputed_summary": _metric_summary(actual),
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

