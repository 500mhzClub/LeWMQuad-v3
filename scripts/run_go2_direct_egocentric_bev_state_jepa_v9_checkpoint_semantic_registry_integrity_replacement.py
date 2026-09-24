#!/usr/bin/env python3
"""Run the science-identical Direct-BEV V9 integrity replacement.

V9 retains the complete frozen V8 science and execution stack.  Its only
mechanical seams adapt the lexical checkpoint semantic registry to the V9
contract and preserve completed observation receipts if terminal publication
is reached through an exception path.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V9_"
    "CHECKPOINT_SEMANTIC_REGISTRY_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)
V9_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v9_checkpoint_semantic_registry_integrity_"
    "replacement_model_runtime"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_bev_v9_checkpoint_registry_integrity_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V9 runner identity changed")

_V8 = _source_only_module(
    "_lewm_direct_bev_v9_checkpoint_registry_frozen_v8_runner",
    ROOT / contract.FROZEN_V8_RUNNER_RELATIVE_PATH,
)
_LEAF = _V8._LEAF

_FROZEN_V8_SNAPSHOT_MODEL = _V8._v8_snapshot_model
_FROZEN_V8_EVALUATE_OBSERVATION_IMPL = _V8._v8_evaluate_observation_impl
_FROZEN_V8_TERMINAL_FAILURE = _V8._v8_terminal_failure

# One launcher process owns exactly one attempt.  Keeping these receipts here
# makes them available even when a completed observation is followed by an
# exception before the inherited training loop returns its normal result.
_V9_COMPLETED_OBSERVATION_RECEIPTS: list[dict[str, Any]] = []
_V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES: list[dict[str, Any]] = []
_V9_SNAPSHOT_ATTEMPT_RECEIPT: dict[str, Any] | None = None


def _runner_contract_owners() -> tuple[Any, ...]:
    return (
        _V8,
        _V8._V7,
        _V8._V6,
        _V8._V6._V5,
        _V8._V6._V5._V4,
        _V8._V6._V5._V4._V3,
        _V8._V6._V5._V4._V3._V2,
        _LEAF,
    )


def _determinism_state(runtime: Any) -> dict[str, bool] | None:
    torch = getattr(runtime, "torch", None)
    if torch is None:
        return None
    try:
        return {
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only_enabled": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        }
    except (AttributeError, RuntimeError):
        return None


def _strict_determinism_exact(value: Mapping[str, Any] | None) -> bool:
    return value == {
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def _error_receipt(error: BaseException) -> dict[str, str]:
    message = "".join(
        traceback.format_exception_only(type(error), error)
    ).strip()
    return {
        "module": type(error).__module__,
        "type": type(error).__name__,
        "message": message[:2_000],
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def _assert_snapshot_globals_topology() -> None:
    """Prove the one lexical contract that V9 is permitted to adapt."""

    registry = _LEAF._register_output_semantic_metadata
    if (
        _FROZEN_V8_SNAPSHOT_MODEL is not _V8._v8_snapshot_model
        or _FROZEN_V8_SNAPSHOT_MODEL.__globals__ is not vars(_V8)
        or registry is not _LEAF._BASE._register_output_semantic_metadata
        or registry.__globals__ is not vars(_LEAF._BASE)
        or _LEAF._write_exclusive is not _LEAF._BASE._write_exclusive
        or _LEAF._write_exclusive.__globals__ is not vars(_LEAF._BASE)
        or any(owner.contract is not contract for owner in _runner_contract_owners())
    ):
        raise RuntimeError("V9 checkpoint semantic-registry topology changed")


def _v9_snapshot_model(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Invoke the frozen V8 snapshot under the exact V9 registry schema."""

    global _V9_SNAPSHOT_ATTEMPT_RECEIPT
    _assert_snapshot_globals_topology()
    previous_contract = _LEAF._BASE.contract
    if previous_contract is not _LEAF._LEGACY_CONTRACT:
        raise RuntimeError("V9 lexical base contract was not the frozen legacy object")
    determinism = _determinism_state(runtime)
    _V9_SNAPSHOT_ATTEMPT_RECEIPT = {
        "stage": f"checkpoint_snapshot_update_{update}",
        "update": update,
        "entered": True,
        "completed": False,
        "raised": False,
        "active_runner_contract_owners_exact": True,
        "exact_prior_contract_was_frozen_legacy_object": True,
        "registry_contract_rebound_to_active_v9_science_identical_contract": (
            False
        ),
        "exact_prior_contract_restored": False,
        "strict_determinism_at_snapshot_entry": determinism,
        "strict_determinism_at_snapshot_entry_exact": (
            _strict_determinism_exact(determinism)
        ),
    }
    completed = False
    try:
        # This is the sole transient global mutation: the base registry's own
        # lexical contract.  Every runner-stack science contract is already V9.
        _LEAF._BASE.contract = contract
        _V9_SNAPSHOT_ATTEMPT_RECEIPT[
            "registry_contract_rebound_to_active_v9_science_identical_contract"
        ] = _LEAF._BASE.contract is contract
        result = _FROZEN_V8_SNAPSHOT_MODEL(
            runtime,
            model,
            output_root,
            update=update,
            metadata=metadata,
        )
        completed = True
        _V9_SNAPSHOT_ATTEMPT_RECEIPT["completed"] = True
        return result
    except BaseException as error:
        _V9_SNAPSHOT_ATTEMPT_RECEIPT["raised"] = True
        _V9_SNAPSHOT_ATTEMPT_RECEIPT["error"] = _error_receipt(error)
        raise
    finally:
        _LEAF._BASE.contract = previous_contract
        _V9_SNAPSHOT_ATTEMPT_RECEIPT[
            "exact_prior_contract_restored"
        ] = _LEAF._BASE.contract is previous_contract
        if completed:
            _V9_SNAPSHOT_ATTEMPT_RECEIPT = None


def _json_safe_observation_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    core = copy.deepcopy({
        "update": result["update"],
        "gate": result["gate"],
        "metrics": result["metrics"],
    })
    raw = contract.canonical_json_bytes(core)
    if json.loads(raw.decode("utf-8")) != core:
        raise TypeError("V9 completed observation is not JSON-safe")
    return {
        **core,
        "canonical_sha256": contract.canonical_json_sha256(core),
    }


def _observation_bindings(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "update": receipt["update"],
            "canonical_sha256": receipt["canonical_sha256"],
        }
        for receipt in receipts
    ]


def _progress_observation_evidence(
    progress: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Initialize empty evidence or validate the exact persisted prefix."""

    receipt_name = "completed_observation_receipts"
    binding_name = "completed_observation_receipt_bindings"
    witness_name = "completed_observation_determinism_witnesses"
    present = tuple(name in progress for name in (
        receipt_name, binding_name, witness_name
    ))
    if not any(present):
        if (
            _V9_COMPLETED_OBSERVATION_RECEIPTS
            or _V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
        ):
            raise RuntimeError("V9 completed observations were not persisted")
        progress[receipt_name] = []
        progress[binding_name] = []
        progress[witness_name] = []
    elif not all(present):
        raise RuntimeError("V9 persisted observation evidence is partial")
    receipts = progress[receipt_name]
    bindings = progress[binding_name]
    witnesses = progress[witness_name]
    if (
        type(receipts) is not list
        or type(bindings) is not list
        or type(witnesses) is not list
        or receipts != _V9_COMPLETED_OBSERVATION_RECEIPTS
        or bindings != _observation_bindings(receipts)
        or witnesses != _V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
        or len(witnesses) != len(receipts)
    ):
        raise RuntimeError("V9 persisted observation evidence changed")
    return receipts, bindings, witnesses


def _v9_evaluate_observation_impl(
    runtime: Any,
    model_api: Any,
    model: Any,
    partition: Mapping[str, Any],
    loader: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    update_zero: Mapping[str, Any] | None,
    prior_gates_passed: bool,
) -> dict[str, Any]:
    """Capture a completed V8 observation without changing its return value."""

    progress = getattr(loader, "progress", None)
    if type(progress) is not dict:
        raise RuntimeError("V9 observation loader lost its failure-progress state")
    _progress_observation_evidence(progress)
    result = _FROZEN_V8_EVALUATE_OBSERVATION_IMPL(
        runtime,
        model_api,
        model,
        partition,
        loader,
        selection_pairs,
        selection_mapping,
        device,
        update=update,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    if type(result) is not dict or result.get("update") != update:
        raise RuntimeError("V9 inherited observation identity changed")
    expected_updates = tuple(contract.OBSERVATION_UPDATES)
    index = len(_V9_COMPLETED_OBSERVATION_RECEIPTS)
    if index >= len(expected_updates) or update != expected_updates[index]:
        raise RuntimeError("V9 completed observations are not an exact prefix")
    receipt = _json_safe_observation_receipt(result)
    determinism = _determinism_state(runtime)
    _V9_COMPLETED_OBSERVATION_RECEIPTS.append(receipt)
    _V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES.append({
        "update": update,
        "state_after_completed_observation": determinism,
        "strict_determinism_exact": _strict_determinism_exact(determinism),
    })
    progress["completed_observation_receipts"] = copy.deepcopy(
        _V9_COMPLETED_OBSERVATION_RECEIPTS
    )
    progress["completed_observation_receipt_bindings"] = (
        _observation_bindings(_V9_COMPLETED_OBSERVATION_RECEIPTS)
    )
    progress["completed_observation_determinism_witnesses"] = copy.deepcopy(
        _V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
    )
    return result


def _reservation_failure_context(
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
) -> dict[str, Any]:
    """Bind available authority/science; point to inherited input ledgers."""

    required = (
        "attempt_identity",
        "independent_source_review",
        "execution_authorization",
        "reviewed_sources",
        "science_contract",
        "authority",
    )
    if any(name not in reservation for name in required):
        raise RuntimeError("V9 reservation lacks failure-context authority")
    return {
        "reservation_file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "attempt_identity": reservation["attempt_identity"],
        "independent_source_review": copy.deepcopy(
            reservation["independent_source_review"]
        ),
        "execution_authorization": copy.deepcopy(
            reservation["execution_authorization"]
        ),
        "reviewed_sources_sha256": contract.canonical_json_sha256(
            reservation["reviewed_sources"]
        ),
        "science_contract_sha256": contract.canonical_json_sha256(
            reservation["science_contract"]
        ),
        "authority_sha256": contract.canonical_json_sha256(
            reservation["authority"]
        ),
        "inherited_input_and_access_evidence_pointers": {
            "loader_access": "failure.loader_access_at_failure",
            "raw_constructor_reads": "failure.raw_constructor_read_ledger",
            "consumed_inputs": (
                "failure.consumed_input_ledger_without_payload_reopen"
            ),
        },
    }


def _v9_terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    *,
    error: BaseException,
    progress: Mapping[str, Any],
) -> None:
    """Add completed observation evidence, then publish frozen V8 failure."""

    translated = dict(progress)
    receipts, bindings, witnesses = _progress_observation_evidence(translated)
    if "v9_failure_context_binding" in translated:
        raise RuntimeError("V9 failure-context field already exists")
    if "checkpoint_snapshot_failure_receipt" in translated:
        raise RuntimeError("V9 snapshot failure-receipt field already exists")
    translated["completed_observation_receipts"] = copy.deepcopy(receipts)
    translated["completed_observation_receipt_bindings"] = copy.deepcopy(
        bindings
    )
    translated["completed_observation_determinism_witnesses"] = copy.deepcopy(
        witnesses
    )
    translated["v9_failure_context_binding"] = _reservation_failure_context(
        reservation, reservation_raw
    )
    if _V9_SNAPSHOT_ATTEMPT_RECEIPT is not None:
        snapshot = copy.deepcopy(_V9_SNAPSHOT_ATTEMPT_RECEIPT)
        snapshot["determinism_state_before_terminal_publication"] = (
            _determinism_state(progress.get("_runtime"))
        )
        snapshot[
            "frozen_determinism_finally_unwound_before_terminal_publication"
        ] = True
        snapshot["lexical_base_contract_restored_before_terminal_publication"] = (
            _LEAF._BASE.contract is _LEAF._LEGACY_CONTRACT
        )
        if snapshot.get("raised") is True:
            inherited_stage = translated.get("stage")
            if type(inherited_stage) is not str:
                raise RuntimeError("V9 inherited snapshot failure stage is absent")
            translated[
                "inherited_progress_stage_before_v9_snapshot_failure_marker"
            ] = inherited_stage
            translated["stage"] = snapshot["stage"]
        translated["checkpoint_snapshot_failure_receipt"] = snapshot
    _FROZEN_V8_TERMINAL_FAILURE(
        output_root,
        reservation,
        reservation_raw,
        error=error,
        progress=translated,
    )


_V9_SEAM_TABLE = (
    ("_evaluate_observation_impl", _v9_evaluate_observation_impl),
    ("_snapshot_model", _v9_snapshot_model),
    ("_terminal_failure", _v9_terminal_failure),
)
_V9_SEAM_NAMES = frozenset(name for name, _ in _V9_SEAM_TABLE)


def _assert_v9_seams() -> None:
    expected_v9 = dict(_V9_SEAM_TABLE)
    for name, expected_v8 in _V8._V8_SEAM_TABLE:
        expected = expected_v9.get(name, expected_v8)
        if getattr(_LEAF, name) is not expected:
            raise RuntimeError(f"V9 lost frozen runner seam: {name}")
    if set(expected_v9) != _V9_SEAM_NAMES:
        raise RuntimeError("V9 mechanical seam declaration changed")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("V9 failure-chain validator was not rebound")
    _assert_snapshot_globals_topology()
    if _LEAF._BASE.contract is not _LEAF._LEGACY_CONTRACT:
        raise RuntimeError("V9 lexical base contract leaked outside snapshot")


def _rebind_inherited_runner() -> None:
    """Complete the frozen V8 rebind, then install only V9 mechanics."""

    wrapper = Path(__file__).resolve()
    _V8.contract = contract
    _V8.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V8.V8_MODEL_RUNTIME_MODULE_NAME = V9_MODEL_RUNTIME_MODULE_NAME
    _V8.__file__ = str(wrapper)
    _V8._rebind_inherited_runner()
    for name, function in _V9_SEAM_TABLE:
        setattr(_LEAF, name, function)
    owners = _runner_contract_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V9 contract did not reach the complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V9 preflight identity did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V9 runner path did not reach runner stack")
    _assert_v9_seams()


def _assert_fresh_attempt_receipts() -> None:
    if (
        _V9_COMPLETED_OBSERVATION_RECEIPTS
        or _V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
        or _V9_SNAPSHOT_ATTEMPT_RECEIPT is not None
    ):
        raise RuntimeError("V9 execution requires a fresh process receipt registry")


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_v9_seams()
    return result


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    _assert_fresh_attempt_receipts()
    result = _LEAF.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    _assert_v9_seams()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_v9_seams()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
