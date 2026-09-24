#!/usr/bin/env python3
"""Run the one-shot global rigid-BEV transport joint-JEPA V1 probe."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
FROZEN_V3_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_global_rigid_bev_transport_runner_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH:
    raise PermissionError("global rigid-BEV transport runner path changed")
_V3 = _source_module(
    "_lewm_global_rigid_bev_transport_frozen_v3_runner",
    ROOT / FROZEN_V3_RUNNER_RELATIVE_PATH,
)
_V2 = _V3._V2
_BASE = _V2._V1

_tensor_state_sha256 = _V3._tensor_state_sha256
ROCM_GRID_SAMPLE_DETERMINISM_WARNING = str(
    contract.WARNING_POLICY["allowed_base_message"]
)
_CONTEXT_SUFFIX = re.compile(
    r" \(Triggered internally at /pytorch/aten/src/ATen/"
    r"Context\.cpp:[0-9]+\.\)"
)


class DeterministicWarningFailure(RuntimeError):
    """Reject a warning without discarding an already returned science value."""

    def __init__(
        self,
        message: str,
        *,
        scientific_result: Any,
        warning_receipt: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.scientific_result = scientific_result
        self.warning_receipt = dict(warning_receipt)


def canonicalize_rocm_determinism_warning(message: str) -> str | None:
    """Return the registered base only for the exact optional suffix form."""

    if type(message) is not str:
        return None
    if message == ROCM_GRID_SAMPLE_DETERMINISM_WARNING:
        return ROCM_GRID_SAMPLE_DETERMINISM_WARNING
    if not message.startswith(ROCM_GRID_SAMPLE_DETERMINISM_WARNING):
        return None
    suffix = message[len(ROCM_GRID_SAMPLE_DETERMINISM_WARNING):]
    if _CONTEXT_SUFFIX.fullmatch(suffix) is None:
        return None
    return ROCM_GRID_SAMPLE_DETERMINISM_WARNING


def _is_allowed_rocm_determinism_warning(message: str) -> bool:
    """Compatibility predicate for the inherited runner's message-only hook."""

    return canonicalize_rocm_determinism_warning(message) is not None


def _finalize_deterministic_warnings(
    scientific_result: Any,
    caught: Sequence[Any],
    *,
    scientific_callable_returned: bool = True,
) -> tuple[Any, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    canonical_messages: list[str] = []
    provenance_suffix_count = 0
    for item in caught:
        message = str(item.message)
        category = item.category
        canonical = canonicalize_rocm_determinism_warning(message)
        allowed = category is UserWarning and canonical is not None
        row = {
            "category": getattr(category, "__name__", str(category)),
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "allowed": allowed,
        }
        rows.append(row)
        if allowed:
            canonical_messages.append(str(canonical))
            if message != canonical:
                provenance_suffix_count += 1
        else:
            unexpected.append(row)
    receipt = {
        "deterministic_algorithms": True,
        "warn_only_due_to_rocm_grid_sample_backward": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "warning_count": len(rows),
        "warning_message_sha256": sorted({row["message_sha256"] for row in rows}),
        "canonical_warning_message_sha256": sorted({
            hashlib.sha256(message.encode("utf-8")).hexdigest()
            for message in canonical_messages
        }),
        "warning_provenance_suffix_count": provenance_suffix_count,
        "unexpected_warning_count": len(unexpected),
        "warning_categories": [row["category"] for row in rows],
        "scientific_callable_returned_before_warning_finalization": bool(
            scientific_callable_returned
        ),
    }
    if unexpected:
        raise DeterministicWarningFailure(
            "unexpected warning under deterministic execution: "
            f"{unexpected[0]['category']}:{unexpected[0]['message_sha256']}",
            scientific_result=scientific_result,
            warning_receipt=receipt,
        )
    return scientific_result, receipt


def _run_deterministic(runtime: Any, operation: Any) -> tuple[Any, dict[str, Any]]:
    import warnings

    torch = runtime.torch
    previous_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    previous_cudnn = bool(torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                scientific_result = operation()
            except BaseException as error:
                try:
                    _unused, receipt = _finalize_deterministic_warnings(
                        None,
                        caught,
                        scientific_callable_returned=False,
                    )
                except DeterministicWarningFailure as warning_error:
                    receipt = warning_error.warning_receipt
                error.determinism_warning_receipt = dict(receipt)
                raise
        return _finalize_deterministic_warnings(scientific_result, caught)
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms, warn_only=previous_warn_only
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn


def _canonical_root_entry(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == ROOT
    except (OSError, RuntimeError):
        return False


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, ...]:
    """Load the inherited stack and the new model under one canonical root."""

    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)

    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if not _canonical_root_entry(entry)]
        sys.path.insert(0, str(ROOT))
        if (
            sys.path[0] != str(ROOT)
            or sum(_canonical_root_entry(entry) for entry in sys.path) != 1
        ):
            raise PermissionError("canonical repository import root is not exact")
        matched = _BASE._source_module(
            "_lewm_global_rigid_bev_transport_matched_runtime",
            _BASE.MATCHED_RUNNER_PATH,
        )
        runtime = matched._load_runtime()
        schedule_adapter = _BASE._source_module(
            "_lewm_global_rigid_bev_transport_schedule_adapter",
            _BASE.SCHEDULE_ADAPTER_PATH,
        )
        model_api = _BASE._source_module(
            "lewm.models.geometry_anchored_global_action_indexed_rigid_bev_"
            "transport_joint_jepa_v1",
            ROOT / contract.MODEL_RELATIVE_PATH,
        )
    finally:
        sys.path[:] = original_path

    if sys.path != original_path:
        raise PermissionError("post-stack import did not restore sys.path")
    model_class = getattr(model_api, contract.MODEL_CLASS_NAME, None)
    if (
        model_class is None
        or getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_class
    ):
        raise PermissionError("new model compatibility class binding changed")
    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)
    return matched, runtime, schedule_adapter, model_api


_BASE_PARAMETER_RECEIPT = _BASE._parameter_receipt
_BASE_EXECUTE = _BASE._execute


def _parameter_receipt(
    model: Any,
    contract_api: Any,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    groups, receipt = _BASE_PARAMETER_RECEIPT(model, contract_api)
    names = [
        name for name, _parameter in model.named_parameters()
        if name.startswith("predictor.")
    ]
    predictor = receipt["predictor"]
    if (
        names != list(contract.PREDICTOR_ORDERED_PARAMETER_NAMES)
        or predictor["parameter_count"] != contract.PREDICTOR_PARAMETER_COUNT
        or predictor["tensor_count"] != contract.PREDICTOR_PARAMETER_TENSOR_COUNT
    ):
        raise PermissionError("rigid transport predictor inventory changed")
    return groups, receipt


def _retain_returned_science(
    progress: dict[str, Any], scientific_result: Any
) -> None:
    """Bind an already-returned probe before terminal warning failure."""

    if (
        not isinstance(scientific_result, tuple)
        or len(scientific_result) != 2
        or not isinstance(scientific_result[1], Mapping)
    ):
        return
    probe = scientific_result[1]
    progress["_probe"] = probe
    observations = probe.get("observations")
    checkpoints = probe.get("checkpoints")
    if isinstance(observations, list):
        progress["_observations"] = observations
    if isinstance(checkpoints, list):
        progress["_checkpoint_bindings"] = checkpoints
    if "training_trace" in probe:
        progress["_training_trace_binding"] = probe.get("training_trace")
    progress["stage"] = "post_callable_warning_finalization"


def _execute(
    *,
    sources: Mapping[str, str],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    """Persist deterministic-warning evidence on every terminal path."""

    try:
        return _BASE_EXECUTE(
            sources=sources,
            authorization=authorization,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except DeterministicWarningFailure as error:
        progress["_determinism"] = dict(error.warning_receipt)
        _retain_returned_science(progress, error.scientific_result)
        raise
    except BaseException as error:
        receipt = getattr(error, "determinism_warning_receipt", None)
        if isinstance(receipt, Mapping):
            progress["_determinism"] = dict(receipt)
        raise


def _rebind_inherited_runner() -> None:
    """Bind the frozen V3 body to the new model, identity, and warning policy."""

    _V3.contract = contract
    _V3.RUNNER_PATH = RUNNER_PATH
    _V3.__file__ = str(RUNNER_PATH)
    _V3._rebind_inherited_runner()
    _V2.contract = contract
    _V2.RUNNER_PATH = RUNNER_PATH
    _V2._load_post_reservation_stack = _load_post_reservation_stack
    _BASE.contract = contract
    _BASE.RUNNER_PATH = RUNNER_PATH
    _BASE.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _BASE.__file__ = str(RUNNER_PATH)
    _BASE._load_post_reservation_stack = _load_post_reservation_stack
    _BASE._tensor_state_sha256 = _tensor_state_sha256
    _BASE._parameter_receipt = _parameter_receipt
    _BASE._execute = _execute
    _BASE._run_deterministic = _run_deterministic
    _BASE._is_allowed_rocm_determinism_warning = (
        _is_allowed_rocm_determinism_warning
    )


_rebind_inherited_runner()


def run_isolated_import_preflight() -> dict[str, Any]:
    _rebind_inherited_runner()
    before_path = list(sys.path)
    sources = contract.current_source_bindings(ROOT)
    loaded = _load_post_reservation_stack(sources)
    if len(loaded) != 4 or sys.path != before_path:
        raise PermissionError("isolated rigid transport import preflight failed")
    return {
        "post_reservation_stack_imported": True,
        "new_model_class_bound": True,
        "scalar_safe_state_hash_bound": True,
        "canonical_root_count_during_lazy_import": 1,
        "sys_path_restored_exactly": True,
        "runtime_or_generated_inputs_opened": [],
        "checkpoints_tensors_traces_or_v3_outputs_opened": [],
        "accelerators_queried_or_used": [],
        "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V3.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V3.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
