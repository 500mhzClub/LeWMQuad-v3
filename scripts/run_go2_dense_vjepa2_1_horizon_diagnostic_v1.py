#!/usr/bin/env python3
"""Run the preregistered no-RGB dense V-JEPA horizon diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_go2_matched_branch_successor_screen_v1 as predecessor  # noqa: E402


RESULT_SCHEMA = "lewm_go2_dense_vjepa2_1_horizon_diagnostic_result_v1"
TERMINAL_SCHEMA = "lewm_go2_dense_vjepa2_1_horizon_diagnostic_terminal_v1"
AUTHORITY_SCHEMA = "lewm_go2_dense_vjepa2_1_horizon_diagnostic_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_NO_RGB_HORIZON_DIAGNOSTIC"
PREREGISTRATION = (
    REPO_ROOT
    / "docs/lewm_go2_dense_vjepa2_1_horizon_diagnostic_v1_preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = "f29d00876c50d813b60fdeea3543a41155066b4049ef2f414a92e162d21ae09b"
PREREGISTRATION_BYTE_COUNT = 7_352
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated/dev/go2_dense_vjepa2_1_horizon_diagnostic_v1/attempt_v1"
)
PREDECESSOR_ROOT = (
    REPO_ROOT / ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1"
)
PREDECESSOR_TERMINAL_REVIEW = (
    REPO_ROOT
    / "docs/lewm_go2_matched_branch_successor_screen_v1_terminal_review_2026-08-03.json"
)
PREDECESSOR_BINDINGS = {
    "feature_cache": {
        "path": str((PREDECESSOR_ROOT / "features/vjepa2_1.pt").resolve()),
        "sha256": "3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b",
        "byte_count": 604_097_648,
    },
    "feature_receipt": {
        "path": str((PREDECESSOR_ROOT / "features/vjepa2_1.json").resolve()),
        "sha256": "5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5",
        "byte_count": 1_822,
    },
    "result": {
        "path": str((PREDECESSOR_ROOT / "result.json").resolve()),
        "sha256": "a6caf2ed1950781815925ccc76b4dbbf40b0f331f4b14a5e60befc88f3aae605",
        "byte_count": 21_377,
    },
    "terminal": {
        "path": str((PREDECESSOR_ROOT / "terminal.json").resolve()),
        "sha256": "bf3bf322c2f3db877be405ebf5ca1daf9dd1a5ffd667b769d44cccab22ede758",
        "byte_count": 510,
    },
    "terminal_review": {
        "path": str(PREDECESSOR_TERMINAL_REVIEW.resolve()),
        "sha256": "c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83",
        "byte_count": 4_991,
    },
}
INDEX_SHA256 = "b740e3efead2f79fd17337a9fa10784c91989e52e837d023b2cc02a2c19d018d"
ARM = "dense_vjepa2_1"
UPDATE_800_WITNESS = {
    "matched_cosine_error": 0.06880560082693894,
    "persistence_cosine_error": 0.07507951919817263,
    "error_to_persistence_ratio": 0.9164363539053353,
    "branch_retrieval_accuracy": 0.2803819444444444,
    "cyclic_deranged_cosine_error": 0.09747234731912613,
    "action_intervention_margin": 0.028666746492187187,
}
PREDECESSOR_REPLAY_EVALUATION_UPDATES = frozenset((100, 200, 400))
SOURCE_PATHS = {
    **{
        f"predecessor_{label}": path
        for label, path in predecessor.SOURCE_PATHS.items()
    },
    "horizon_runner": Path(__file__).resolve(),
    "horizon_runner_test": (
        REPO_ROOT / "lewm/tests/test_run_go2_dense_vjepa2_1_horizon_diagnostic_v1.py"
    ),
}
SOURCE_LABELS = set(SOURCE_PATHS)


class HorizonError(RuntimeError):
    """Raised when the frozen diagnostic contract changes."""


def horizon_config_v1() -> dict[str, Any]:
    config = predecessor.screen_config_v1()
    config.update(
        {
            "arms": [ARM],
            "updates": 3_200,
            "trace_updates": [0, 800, 1_600, 2_400, 3_200],
            "futility_update": 1_600,
            "futility_maximum_error_to_persistence_ratio": 0.8582181769526677,
            "futility_minimum_branch_retrieval_accuracy": 0.3901909722222222,
        }
    )
    return config


def _json_from_bound_file(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    predecessor._require_binding(binding, label=label)  # noqa: SLF001
    try:
        value = json.loads(Path(str(binding["path"])).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HorizonError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise HorizonError(f"{label} is not a JSON object")
    return dict(value)


def _validate_source_review(
    binding: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    review = _json_from_bound_file(binding, label="source review")
    if (
        review.get("schema")
        != "lewm_go2_dense_vjepa2_1_horizon_diagnostic_source_review_v1"
        or review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("source_bindings") != source_bindings
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or not isinstance(review.get("checks"), Mapping)
        or not review["checks"]
        or any(value is not True for value in review["checks"].values())
    ):
        raise HorizonError("independent source review did not pass exactly")


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> dict[str, Any]:
    actual = predecessor.file_binding_v1(path)
    if actual["sha256"] != expected_sha256 or actual["byte_count"] != expected_byte_count:
        raise HorizonError("execution authority caller binding changed")
    document = _json_from_bound_file(actual, label="execution authority")
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_collection",
        "authorizes_rgb_access",
        "authorizes_evaluation",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "predecessor_bindings",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        set(document) != required
        or document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("citable_as_scientific_evidence") is not False
        or document.get("authorizes_collection") is not False
        or document.get("authorizes_rgb_access") is not False
        or document.get("authorizes_evaluation") is not False
        or document.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or document.get("config") != horizon_config_v1()
        or document.get("predecessor_bindings") != PREDECESSOR_BINDINGS
    ):
        raise HorizonError("execution authority contract changed")
    commit = document.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        != 0
    ):
        raise HorizonError("frozen source commit is not an ancestor of execution HEAD")
    preregistration = predecessor._require_binding(  # noqa: SLF001
        document["preregistration_binding"], label="preregistration"
    )
    if preregistration != {
        "path": str(PREREGISTRATION.resolve()),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }:
        raise HorizonError("authority does not bind the frozen preregistration")
    sources = document.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != SOURCE_LABELS:
        raise HorizonError("source closure labels changed")
    for label, expected in sources.items():
        actual_source = predecessor._require_binding(  # noqa: SLF001
            expected, label=f"source {label}"
        )
        if actual_source["path"] != str(SOURCE_PATHS[label].resolve()):
            raise HorizonError(f"source {label} path changed")
    _validate_source_review(
        document["source_review_binding"],
        preregistration_binding=preregistration,
        source_bindings=sources,
    )
    for label, binding in document["predecessor_bindings"].items():
        predecessor._require_binding(binding, label=f"predecessor {label}")  # noqa: SLF001
    environment = document.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "hip"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
    ):
        raise HorizonError("execution environment changed")
    return document


def load_bound_inputs_v1(
    authority: Mapping[str, Any],
) -> tuple[torch.Tensor, predecessor.ScreenIndexV1, dict[str, Any]]:
    bindings = authority["predecessor_bindings"]
    result = _json_from_bound_file(bindings["result"], label="predecessor result")
    terminal = _json_from_bound_file(bindings["terminal"], label="predecessor terminal")
    review = _json_from_bound_file(
        bindings["terminal_review"], label="predecessor terminal review"
    )
    receipt = _json_from_bound_file(
        bindings["feature_receipt"], label="predecessor feature receipt"
    )
    if (
        result.get("schema") != predecessor.SCHEMA
        or result.get("status") != "COMPLETE_ENGINEERING_SCREEN"
        or result.get("collection_justified") is not False
        or result.get("navigation_usefulness_established") is not False
        or terminal.get("schema") != predecessor.TERMINAL_SCHEMA
        or terminal.get("status") != "COMPLETE_COLLECTION_NOT_JUSTIFIED"
        or terminal.get("result_binding") != bindings["result"]
        or terminal.get("collection_justified") is not False
        or review.get("schema")
        != "lewm_go2_matched_branch_successor_screen_terminal_review_v1"
        or review.get("status")
        != "PASS_WITH_MINOR_NON_DECISION_RELEVANT_REPORTING_DEVIATION"
        or review.get("result_binding") != bindings["result"]
        or review.get("terminal_binding") != bindings["terminal"]
        or review.get("protected_material_opened") is not False
        or review.get("evaluation_rgb_opened") is not False
        or review.get("findings") != []
        or result.get("feature_caches", {}).get("vjepa2_1") != receipt
        or receipt.get("binding") != bindings["feature_cache"]
    ):
        raise HorizonError("predecessor evidence contract changed")
    predecessor_sources = result.get("authority", {}).get("source_bindings")
    if not isinstance(predecessor_sources, Mapping) or any(
        authority.get("source_bindings", {}).get(f"predecessor_{label}")
        != predecessor_sources.get(label)
        for label in predecessor.SOURCE_LABELS
    ):
        raise HorizonError("exact frozen predecessor source closure changed")
    bundle = predecessor.screen_data.load_bound_posthoc_bundle_v1()
    index = predecessor.build_screen_index_v1(bundle)
    if (
        index.index_sha256 != INDEX_SHA256
        or bundle.access_audit.get("rgb_leaf_open_count") != 0
        or result.get("screen_index", {}).get("index_sha256") != INDEX_SHA256
        or result.get("screen_index", {}).get("eval_rgb_leaf_open_count") != 0
        or receipt.get("eval_artifact_open_count") != 0
        or receipt.get("train_artifact_open_count") != predecessor.ARTIFACT_COUNT
    ):
        raise HorizonError("metadata-only predecessor index changed")
    features = predecessor._load_feature_cache(  # noqa: SLF001
        receipt, expected_encoder="vjepa2_1", index=index
    )
    if (
        features.shape != (predecessor.ARTIFACT_COUNT, 256, 768)
        or features.dtype != torch.float16
        or not bool(torch.isfinite(features).all())
    ):
        raise HorizonError("V-JEPA feature cache tensor changed")
    norms = torch.linalg.vector_norm(features.to(torch.float32), dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-3, rtol=1.0e-3)):
        raise HorizonError("V-JEPA feature cache is not token-normalized")
    return features, index, result


def _checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()

    def finite(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True

    if not finite(model_state) or not finite(optimizer_state):
        raise HorizonError("checkpoint state became nonfinite")
    payload = {
        "schema": "lewm_go2_dense_vjepa2_1_horizon_checkpoint_v1",
        "arm": ARM,
        "seed": int(config["seed"]),
        "update": update,
        "config": dict(config),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(payload, path)
    observed = torch.load(path, map_location="cpu", weights_only=True)
    if (
        observed.get("schema") != payload["schema"]
        or observed.get("arm") != ARM
        or observed.get("seed") != int(config["seed"])
        or observed.get("update") != update
        or observed.get("config") != dict(config)
        or not finite(observed.get("model_state_dict"))
        or not finite(observed.get("optimizer_state_dict"))
    ):
        raise HorizonError("checkpoint round-trip validation failed")
    return predecessor.file_binding_v1(path)


def _metrics_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return set(expected).issubset(actual) and all(actual[key] == value for key, value in expected.items())


def _require_update_800_witness(metrics: Mapping[str, Any]) -> None:
    if not _metrics_equal(metrics, UPDATE_800_WITNESS):
        raise HorizonError("update-800 predecessor drift witness changed")


def train_horizon_v1(
    features: torch.Tensor,
    index: predecessor.ScreenIndexV1,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    output_root: Path,
    require_update_800_witness: bool = True,
) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = predecessor._build_model(ARM, int(features.shape[-1]), config).to(device)  # noqa: SLF001
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ordering = torch.randperm(predecessor.STATE_COUNT, generator=generator)
    cursor = 0
    traces: list[dict[str, Any]] = [
        {
            "update": 0,
            **predecessor.evaluate_arm_v1(model, features, index, device=device),
        }
    ]
    started = time.perf_counter()
    nonfinite_count = 0
    checkpoint_bindings: dict[str, Any] = {}
    futility_passed: bool | None = None
    update_800_witness_passed = False
    completed_updates = 0
    for update in range(1, int(config["updates"]) + 1):
        batch_size = int(config["batch_states"])
        if cursor + batch_size > predecessor.STATE_COUNT:
            ordering = torch.randperm(predecessor.STATE_COUNT, generator=generator)
            cursor = 0
        selected = ordering[cursor : cursor + batch_size]
        cursor += batch_size
        context, history, candidates, targets = predecessor._batch_panels(  # noqa: SLF001
            features, index, selected, device
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(context, history, candidates).reshape(
            batch_size, predecessor.ACTION_COUNT, 256, features.shape[-1]
        )
        loss, components = predecessor.common_objective_v1(
            predictions,
            targets,
            temperature=float(config["cross_entropy_temperature"]),
            cross_entropy_coefficient=float(config["cross_entropy_coefficient"]),
        )
        if not bool(torch.isfinite(loss)):
            nonfinite_count += 1
            raise HorizonError("dense V-JEPA loss became nonfinite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            nonfinite_count += 1
            raise HorizonError("dense V-JEPA gradient norm became nonfinite")
        optimizer.step()
        objective = {
            "total": float(loss.detach()),
            "gradient_norm_before_clip": float(grad_norm.detach()),
            **{name: float(value.detach()) for name, value in components.items()},
        }
        completed_updates = update
        if update in PREDECESSOR_REPLAY_EVALUATION_UPDATES:
            # Preserve the predecessor's train/eval call schedule through the
            # exact update-800 drift witness without promoting these old
            # diagnostics into the successor's recorded trace.
            predecessor.evaluate_arm_v1(model, features, index, device=device)
        if update in set(config["trace_updates"]):
            metrics = predecessor.evaluate_arm_v1(
                model, features, index, device=device
            )
            traces.append(
                {
                    "update": update,
                    "objective": objective,
                    **metrics,
                }
            )
            if update == 800:
                _require_update_800_witness(metrics)
                update_800_witness_passed = True
            if update == int(config["futility_update"]):
                repeat = predecessor.evaluate_arm_v1(
                    model, features, index, device=device
                )
                if repeat != metrics:
                    raise HorizonError("update-1600 deterministic repeat changed")
                checkpoint_bindings["update_1600"] = _checkpoint(
                    output_root / "checkpoint_update_1600.pt",
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    config=config,
                )
                futility_passed = (
                    metrics["error_to_persistence_ratio"]
                    <= float(config["futility_maximum_error_to_persistence_ratio"])
                    and metrics["branch_retrieval_accuracy"]
                    >= float(config["futility_minimum_branch_retrieval_accuracy"])
                    and metrics["action_intervention_margin"] > 0.0
                    and nonfinite_count == 0
                )
                if not futility_passed:
                    break
    if require_update_800_witness and not update_800_witness_passed:
        raise HorizonError("update-800 drift witness was not reached")
    elapsed = time.perf_counter() - started
    if futility_passed is None:
        raise HorizonError("futility update was not reached")
    final_metrics = {
        key: value for key, value in traces[-1].items() if key not in {"update", "objective"}
    }
    deterministic_repeat = predecessor.evaluate_arm_v1(
        model, features, index, device=device
    )
    deterministic_repeat_passed = deterministic_repeat == final_metrics
    if not deterministic_repeat_passed:
        raise HorizonError("terminal deterministic repeat changed")
    capacity_established = (
        completed_updates == int(config["updates"])
        and final_metrics["error_to_persistence_ratio"]
        <= float(config["maximum_error_to_persistence_ratio"])
        and final_metrics["branch_retrieval_accuracy"]
        >= float(config["retrieval_threshold"])
        and final_metrics["action_intervention_margin"] > 0.0
        and nonfinite_count == 0
    )
    if completed_updates == int(config["updates"]):
        checkpoint_bindings["update_3200"] = _checkpoint(
            output_root / "checkpoint_update_3200.pt",
            model=model,
            optimizer=optimizer,
            update=completed_updates,
            config=config,
        )
    result = {
        "arm": ARM,
        "seed": seed,
        "completed_updates": completed_updates,
        "maximum_updates": int(config["updates"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed,
        "updates_per_second": completed_updates / elapsed,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "nonfinite_count": nonfinite_count,
        "update_800_drift_witness_passed": update_800_witness_passed,
        "futility_passed": futility_passed,
        "deterministic_repeat_passed": deterministic_repeat_passed,
        "training_set_capacity_established": capacity_established,
        "traces": traces,
        "final_metrics": final_metrics,
        "checkpoint_bindings": checkpoint_bindings,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _source_bindings_unchanged(authority: Mapping[str, Any]) -> None:
    for label, expected in authority["source_bindings"].items():
        if predecessor.file_binding_v1(Path(str(expected["path"]))) != expected:
            raise HorizonError(f"source {label} changed during execution")
    for label, expected in authority["predecessor_bindings"].items():
        if predecessor.file_binding_v1(Path(str(expected["path"]))) != expected:
            raise HorizonError(f"predecessor {label} changed during execution")


def execute_v1(authority: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(authority["output_root"]))
    predecessor._safe_path(  # noqa: SLF001
        output_root.parent, label="horizon output parent", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise HorizonError("the preregistered horizon diagnostic requires a CUDA/ROCm GPU")
    features, index, predecessor_result = load_bound_inputs_v1(authority)
    device = torch.device("cuda")
    arm = train_horizon_v1(
        features,
        index,
        config=authority["config"],
        device=device,
        output_root=output_root,
    )
    _source_bindings_unchanged(authority)
    if not arm["futility_passed"]:
        status = "COMPLETE_FUTILITY_STOP"
        next_route = "STOP_DENSE_VJEPA2_1_HORIZON_NO_FURTHER_TUNING"
    elif arm["training_set_capacity_established"]:
        status = "COMPLETE_TRAINING_SET_CAPACITY_ESTABLISHED"
        next_route = "REQUIRES_NEW_EXPLICIT_CONTROL_AND_COLLECTION_DECISION"
    else:
        status = "COMPLETE_CAPACITY_NOT_ESTABLISHED"
        next_route = "STOP_DENSE_VJEPA2_1_HORIZON_NO_FURTHER_TUNING"
    report = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_collection": False,
        "authorizes_rgb_access": False,
        "authorizes_evaluation": False,
        "fresh_scene_generalization_measured": False,
        "navigation_usefulness_established": False,
        "authority": dict(authority),
        "predecessor_result_binding": authority["predecessor_bindings"]["result"],
        "predecessor_collection_justified": predecessor_result["collection_justified"],
        "screen_index": {
            "states": len(index.state_ids),
            "scenes": len(set(index.scene_ids)),
            "families": len(set(index.family_ids)),
            "artifacts": len(index.artifact_ids),
            "index_sha256": index.index_sha256,
            "rgb_leaf_open_count": 0,
        },
        "device": {
            "type": str(device),
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "arm": arm,
        "training_set_capacity_established": arm["training_set_capacity_established"],
        "collection_justified": False,
        "next_route": next_route,
    }
    predecessor._write_json_exclusive(output_root / "result.json", report)  # noqa: SLF001
    result_binding = predecessor.file_binding_v1(output_root / "result.json")
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_collection": False,
        "authorizes_rgb_access": False,
        "authorizes_evaluation": False,
        "result_binding": result_binding,
        "completed_updates": arm["completed_updates"],
        "training_set_capacity_established": arm["training_set_capacity_established"],
        "collection_justified": False,
        "next_route": next_route,
    }
    predecessor._write_json_exclusive(output_root / "terminal.json", terminal)  # noqa: SLF001
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    args = parser.parse_args(argv)
    authority = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    output_existed = output_root.exists()
    try:
        report = execute_v1(authority)
    except Exception as error:
        if (
            not output_existed
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            predecessor._write_json_exclusive(  # noqa: SLF001
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
                    "citable_as_scientific_evidence": False,
                    "authorizes_collection": False,
                    "authorizes_rgb_access": False,
                    "authorizes_evaluation": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_updates": report["arm"]["completed_updates"],
                "training_set_capacity_established": report[
                    "training_set_capacity_established"
                ],
                "collection_justified": report["collection_justified"],
                "next_route": report["next_route"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
