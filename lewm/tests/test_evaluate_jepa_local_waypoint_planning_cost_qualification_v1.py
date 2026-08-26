from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py"
SPEC = importlib.util.spec_from_file_location("planning_cost_evaluator_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def _synthetic_child_bindings(
    phases: tuple[str, ...] = ("PREFLIGHT", "MATERIALIZE"),
) -> dict[str, dict[str, object]]:
    return {
        phase: {
            "path": f"receipts/gpu_child_{phase.lower()}.json",
            "sha256": ("a" if phase == "PREFLIGHT" else "b") * 64,
            "bytes": 123,
            "content_digest": ("c" if phase == "PREFLIGHT" else "d") * 64,
        }
        for phase in phases
    }


def _populate_current_token_schema_fields(
    value: dict[str, object], spec: dict[str, object]
) -> None:
    if "current_token_execution_amendment_binding_exact" in spec:
        value["current_token_execution_amendment_binding"] = copy.deepcopy(
            spec["current_token_execution_amendment_binding_exact"]
        )
    if "current_token_authority_policy_exact" in spec:
        value["current_token_authority_policy"] = copy.deepcopy(
            spec["current_token_authority_policy_exact"]
        )
    if "gpu_receipt_serialization_amendment_binding_exact" in spec:
        value["gpu_receipt_serialization_amendment_binding"] = copy.deepcopy(
            spec["gpu_receipt_serialization_amendment_binding_exact"]
        )
    if "gpu_child_receipt_order_amendment_binding_exact" in spec:
        value["gpu_child_receipt_order_amendment_binding"] = copy.deepcopy(
            spec["gpu_child_receipt_order_amendment_binding_exact"]
        )
    if "markdown_report_order_amendment_binding_exact" in spec:
        value["markdown_report_order_amendment_binding"] = copy.deepcopy(
            spec["markdown_report_order_amendment_binding_exact"]
        )
    if "gpu_receipt_serialization_preinference_check_exact" in spec:
        value["gpu_receipt_serialization_preinference_check"] = copy.deepcopy(
            spec["gpu_receipt_serialization_preinference_check_exact"]
        )
    phases = spec.get("gpu_child_execution_receipts_required_phase_ids")
    if phases is not None:
        value["gpu_child_execution_receipts"] = _synthetic_child_bindings(
            tuple(str(phase) for phase in phases)
        )


def test_control_history_is_exact_contiguous_applied_k_minus_one_slice() -> None:
    blocks = {
        index: np.asarray(
            [[index * 10 + tick, 0.0, -(index * 10 + tick)] for tick in range(5)],
            dtype=np.float32,
        )
        for index in (37, 38, 39, 40)
    }
    values, indices = E._control_history_from_replay_blocks(blocks)
    assert indices == list(range(184, 199))
    assert values.shape == (15, 3)
    assert values[0].tolist() == blocks[37][4].tolist()
    assert values[-1].tolist() == blocks[40][3].tolist()


def test_control_history_fails_closed_on_missing_or_wrong_blocks() -> None:
    with pytest.raises(E.QualificationError, match="block 37..40"):
        E._control_history_from_replay_blocks({})
    bad = {index: np.zeros((4, 3), np.float32) for index in (37, 38, 39, 40)}
    with pytest.raises(E.QualificationError, match="shape"):
        E._control_history_from_replay_blocks(bad)


class _GoalGraph:
    n_nodes = 21
    nav_blocked_cells = frozenset({20})
    beacon_cells_set = frozenset({20})

    def cell_center(self, cell: int) -> tuple[float, float]:
        return {
            12: (-0.8519999980926514, -2.555000066757202),
            11: (-1.7039999961853027, -2.555000066757202),
            20: (-1.7039999961853027, -1.7039999961853027),
        }[cell]

    def neighbors(self, cell: int) -> tuple[int, ...]:
        return {12: (11,), 11: (12, 20), 20: (11,)}.get(cell, ())

    def bfs_distance(
        self, start: int, goal: int, *, transit_blocked: frozenset[int]
    ) -> int | None:
        assert (start, goal) == (12, 20)
        assert transit_blocked == self.nav_blocked_cells
        return 2


def test_purpose18_beacon_endpoint_is_reachable_and_not_substituted() -> None:
    value = E._goal_cell_semantics(_GoalGraph(), [12, 11, 20])
    assert value["path_cells"] == [12, 11, 20]
    assert value["goal_cell_endpoint_reachable"] is True
    assert value["goal_cell_nav_blocked"] is True
    assert value["goal_cell_block_classification"] == "BEACON_ENDPOINT"
    assert value["goal_cell_is_beacon_endpoint"] is True
    assert value["goal_cell_is_low_clearance_transit_blocked"] is False
    assert value["goal_render_semantics"] == E.CONTRACT.GOAL_VIEW_RENDER_SEMANTICS


def test_low_clearance_endpoint_is_allowed_and_disclosed() -> None:
    graph = _GoalGraph()
    graph.beacon_cells_set = frozenset()
    value = E._goal_cell_semantics(graph, [12, 11, 20])
    assert value["goal_cell_endpoint_reachable"] is True
    assert value["goal_cell_block_classification"] == (
        "LOW_CLEARANCE_TRANSIT_BLOCKED"
    )
    assert value["goal_cell_is_low_clearance_transit_blocked"] is True


def test_goal_cell_semantics_fail_closed_on_invalid_edge_or_unreachable_endpoint() -> None:
    graph = _GoalGraph()
    graph.neighbors = lambda cell: {12: (11,), 11: ()}.get(cell, ())
    with pytest.raises(E.QualificationError, match="edges are not traversable"):
        E._goal_cell_semantics(graph, [12, 11, 20])

    graph = _GoalGraph()
    graph.bfs_distance = lambda *_args, **_kwargs: None
    with pytest.raises(E.QualificationError, match="endpoint is unreachable"):
        E._goal_cell_semantics(graph, [12, 11, 20])
    with pytest.raises(E.QualificationError, match="invalid scene-graph node"):
        E._goal_cell_semantics(graph, [12, 11, 21])


def test_static_goal_view_preflight_reproduces_exact_frozen_domain() -> None:
    validation, semantics = E._static_goal_view_semantics_and_validation()
    assert validation == E.CONTRACT.GOAL_VIEW_STATIC_VALIDATION_SUCCESS
    assert set(semantics) == {f"purpose-{index}" for index in range(48)}
    assert semantics["purpose-18"]["path_cells"] == [12, 11, 20]
    assert semantics["purpose-18"]["goal_cell_block_classification"] == (
        "BEACON_ENDPOINT"
    )


def test_cpu_worker_allocator_mitigation_preserves_32_worker_contract() -> None:
    environment = E._worker_environment()
    expected = E.CONTRACT.EXECUTION_WATCHDOGS["cpu_worker_environment"]
    assert environment["MALLOC_ARENA_MAX"] == expected["MALLOC_ARENA_MAX"] == "1"
    assert expected["workers"] == 32
    assert expected["dynamic_worker_fallback"] is False
    assert expected["scientific_semantics_change"] is False


def test_process_identity_does_not_match_parent_shell_text() -> None:
    name = "evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py"
    assert not E._argv_is_experiment(["zsh", "-lc", f"python {name} execute"])
    assert E._argv_is_experiment(["python", f"/tmp/{name}", "execute"])


def test_output_relative_reference_survives_atomic_directory_rename(tmp_path: Path) -> None:
    attempt = tmp_path / ".attempt"
    payload = attempt / "latents/x.npy"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"x")
    reference = E._artifact_reference(payload, attempt)
    assert reference == "latents/x.npy"
    canonical = tmp_path / "canonical"
    attempt.rename(canonical)
    assert E._artifact_path(reference, canonical).read_bytes() == b"x"


def test_cpu_interpreter_resolves_to_exact_frozen_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = Path(E.sys.executable).resolve()
    exact = {
        "path": str(resolved),
        "sha256": E.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    monkeypatch.setattr(E.CONTRACT, "INTERPRETER_BINARY_BINDING", exact)
    assert E._validated_interpreter_binary_binding(E.sys.executable) == exact
    monkeypatch.setattr(
        E.CONTRACT, "INTERPRETER_BINARY_BINDING", {**exact, "bytes": exact["bytes"] + 1}
    )
    with pytest.raises(E.QualificationError, match="interpreter binary binding drift"):
        E._validated_interpreter_binary_binding(E.sys.executable)


def test_complete_cpu_materialization_deadline_interrupts_postworker_work() -> None:
    previous_handler = E.signal.getsignal(E.signal.SIGALRM)
    with pytest.raises(E.QualificationError, match="global watchdog exceeded"):
        with E._cpu_materialization_global_deadline(0.02):
            # This represents post-worker hash/index publication work.  The
            # coordinator alarm, not the worker-pool polling loop, interrupts it.
            E.time.sleep(0.10)
    assert E.signal.getitimer(E.signal.ITIMER_REAL) == (0.0, 0.0)
    assert E.signal.getsignal(E.signal.SIGALRM) == previous_handler


def test_schema_validator_rejects_one_dropped_required_key() -> None:
    spec = E.CONTRACT.build_output_schema()["files"]["gpu_inference_receipt"]
    value = {key: None for key in spec["required_keys"] if key != "content_digest"}
    value["schema"] = spec["schema"]
    value["gpu_environment_revalidation"] = spec[
        "gpu_environment_revalidation_exact"
    ]
    value["gpu_watchdog_status"] = spec["gpu_watchdog_status_exact"]
    value["interpreter"] = dict(E.CONTRACT.INTERPRETER_BINARY_BINDING)
    value["interpreter_entrypoint"] = E.CONTRACT.build_contract()["execution"][
        "environments"
    ]["encoder_predictor"]["interpreter"]
    _populate_current_token_schema_fields(value, spec)
    value = E.attach_digest(value)
    E._validate_schema_value("gpu_inference_receipt", value)
    value.pop("training_steps")
    value = E.attach_digest(value)
    with pytest.raises(E.QualificationError, match="required keys"):
        E._validate_schema_value("gpu_inference_receipt", value)


def test_preexecution_schema_rejects_contract_disclosure_tamper() -> None:
    spec = E.CONTRACT.build_output_schema()["files"]["preexecution_receipt"]
    value = {key: None for key in spec["required_keys"] if key != "content_digest"}
    value["schema"] = spec["schema"]
    value["goal_view_execution_amendment_binding"] = copy.deepcopy(
        spec["goal_view_execution_amendment_binding_exact"]
    )
    value["goal_view_static_validation"] = copy.deepcopy(
        spec["goal_view_static_validation_exact"]
    )
    value["cpu_worker_environment"] = copy.deepcopy(
        spec["cpu_worker_environment_exact"]
    )
    value["preexecution_custody"] = {
        "contract_disclosure": copy.deepcopy(
            spec["preexecution_contract_disclosure_exact"]
        ),
        "live_validation": {
            **copy.deepcopy(spec["preexecution_live_validation_exact"]),
            "fixture_checks_passed": len(
                E.CONTRACT.build_fixture_receipt()["executed_checks"]
            ),
        },
    }
    _populate_current_token_schema_fields(value, spec)
    value = E.attach_digest(value)
    E._validate_schema_value("preexecution_receipt", value)
    value["preexecution_custody"]["contract_disclosure"].pop(
        next(iter(value["preexecution_custody"]["contract_disclosure"]))
    )
    value = E.attach_digest(value)
    with pytest.raises(E.QualificationError, match="contract disclosure"):
        E._validate_schema_value("preexecution_receipt", value)


def test_frozen_source_closure_is_regenerated_over_the_exact_path_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closure_path = tmp_path / "source_closure.json"
    stored = E.CONTRACT.attach_content_digest(
        {
            "schema": "synthetic_source_closure_v1",
            "experiment_id": E.CONTRACT.EXPERIMENT_ID,
            "starting_head": E.CONTRACT.STARTING_HEAD,
            "rows": [],
            "row_count": 0,
            "missing_paths": [],
            "complete": True,
            "outcome_or_result_payloads_parsed": [],
            "custody_only_result_files_hashed_without_parsing": [],
            "generated_cache_paths_traversed": [],
        }
    )
    closure_path.write_bytes(E.canonical_json_bytes(stored) + b"\n")
    monkeypatch.setattr(
        E.CONTRACT, "TRACKED_SOURCE_CLOSURE_PATH", str(closure_path)
    )
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_goal_view_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_current_token_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_receipt_serialization_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.GPU_RECEIPT_SERIALIZATION_EXECUTION_AMENDMENT
        ),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_child_receipt_order_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.GPU_CHILD_RECEIPT_ORDER_EXECUTION_AMENDMENT
        ),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_markdown_report_order_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.MARKDOWN_REPORT_ORDER_EXECUTION_AMENDMENT
        ),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "build_source_closure",
        lambda _root, require_complete=True: stored,
    )
    assert E.validate_frozen_receipts()["source_closure"] == stored

    omitted = E.CONTRACT.attach_content_digest(
        {**{key: value for key, value in stored.items() if key != "content_digest"},
         "row_count": 1}
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "build_source_closure",
        lambda _root, require_complete=True: omitted,
    )
    with pytest.raises(E.QualificationError, match="complete frozen path domain"):
        E.validate_frozen_receipts()


def test_frozen_receipts_fail_closed_when_goal_view_amendment_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )

    def absent() -> dict[str, object]:
        raise E.CONTRACT.ContractError("absent")

    monkeypatch.setattr(
        E.CONTRACT, "load_and_validate_goal_view_execution_amendment", absent
    )
    with pytest.raises(E.QualificationError, match="amendment is absent or invalid"):
        E.validate_frozen_receipts()


def test_frozen_receipts_fail_closed_when_current_token_amendment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_goal_view_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT),
    )

    def absent() -> dict[str, object]:
        raise E.CONTRACT.ContractError("absent")

    monkeypatch.setattr(
        E.CONTRACT, "load_and_validate_current_token_execution_amendment", absent
    )
    with pytest.raises(E.QualificationError, match="current-token.*absent or invalid"):
        E.validate_frozen_receipts()


def test_frozen_receipts_fail_closed_when_gpu_receipt_amendment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_goal_view_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_current_token_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT),
    )

    def absent() -> dict[str, object]:
        raise E.CONTRACT.ContractError("absent")

    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_receipt_serialization_execution_amendment",
        absent,
    )
    with pytest.raises(E.QualificationError, match="GPU receipt serialization.*absent"):
        E.validate_frozen_receipts()


def test_frozen_receipts_fail_closed_when_gpu_child_order_amendment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_goal_view_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_current_token_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_receipt_serialization_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.GPU_RECEIPT_SERIALIZATION_EXECUTION_AMENDMENT
        ),
    )

    def absent() -> dict[str, object]:
        raise E.CONTRACT.ContractError("absent")

    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_child_receipt_order_execution_amendment",
        absent,
    )
    with pytest.raises(E.QualificationError, match="GPU child receipt order.*absent"):
        E.validate_frozen_receipts()


def test_frozen_receipts_fail_closed_when_markdown_order_amendment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_contract", lambda: {})
    monkeypatch.setattr(E.CONTRACT, "load_and_validate_output_schema", lambda: {})
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_fixture_receipt",
        lambda: {"pass": True, "executed_checks": {"synthetic": True}},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_goal_view_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_current_token_execution_amendment",
        lambda: copy.deepcopy(E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_receipt_serialization_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.GPU_RECEIPT_SERIALIZATION_EXECUTION_AMENDMENT
        ),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_gpu_child_receipt_order_execution_amendment",
        lambda: copy.deepcopy(
            E.CONTRACT.GPU_CHILD_RECEIPT_ORDER_EXECUTION_AMENDMENT
        ),
    )

    def absent() -> dict[str, object]:
        raise E.CONTRACT.ContractError("absent")

    monkeypatch.setattr(
        E.CONTRACT,
        "load_and_validate_markdown_report_order_execution_amendment",
        absent,
    )
    with pytest.raises(E.QualificationError, match="Markdown report order.*absent"):
        E.validate_frozen_receipts()


def test_freeze_writes_all_amendments_before_source_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(E.CONTRACT, "write_contract", lambda *_args: calls.append("contract"))
    monkeypatch.setattr(
        E.CONTRACT, "write_output_schema", lambda *_args: calls.append("schema")
    )
    monkeypatch.setattr(
        E.CONTRACT, "write_fixture_receipt", lambda *_args: calls.append("fixture")
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_goal_view_execution_amendment",
        lambda *_args: calls.append("goal_amendment"),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_current_token_execution_amendment",
        lambda *_args: calls.append("current_amendment"),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_gpu_receipt_serialization_execution_amendment",
        lambda *_args: calls.append("gpu_receipt_amendment"),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_gpu_child_receipt_order_execution_amendment",
        lambda *_args: calls.append("gpu_child_order_amendment"),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_markdown_report_order_execution_amendment",
        lambda *_args: calls.append("markdown_order_amendment"),
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "build_source_closure",
        lambda *_args, **_kwargs: calls.append("build_closure") or {},
    )
    monkeypatch.setattr(
        E.CONTRACT,
        "write_source_closure",
        lambda *_args: calls.append("write_closure"),
    )
    monkeypatch.setattr(
        E,
        "validate_frozen_receipts",
        lambda: {
            "fixture": {"content_digest": "f"},
            "goal_view_execution_amendment": {"content_digest": "g"},
            "current_token_execution_amendment": {"content_digest": "c"},
            "gpu_receipt_serialization_execution_amendment": {
                "content_digest": "r"
            },
            "gpu_child_receipt_order_execution_amendment": {
                "content_digest": "o"
            },
            "markdown_report_order_execution_amendment": {
                "content_digest": "m"
            },
            "source_closure": {"content_digest": "s"},
        },
    )
    receipt = E.freeze(E.ROOT)
    assert calls == [
        "contract",
        "schema",
        "fixture",
        "goal_amendment",
        "current_amendment",
        "gpu_receipt_amendment",
        "gpu_child_order_amendment",
        "markdown_order_amendment",
        "build_closure",
        "write_closure",
    ]
    assert receipt["current_token_execution_amendment_content_digest"] == "c"
    assert (
        receipt[
            "gpu_receipt_serialization_execution_amendment_content_digest"
        ]
        == "r"
    )
    assert (
        receipt["gpu_child_receipt_order_execution_amendment_content_digest"]
        == "o"
    )
    assert (
        receipt["markdown_report_order_execution_amendment_content_digest"]
        == "m"
    )


def test_result_schema_validator_requires_all_nested_count_runtime_storage_keys() -> None:
    spec = E.CONTRACT.build_output_schema()["files"]["result"]
    value = {key: None for key in spec["required_keys"] if key != "result_content_sha256"}
    value["schema"] = spec["schema"]
    value["materialisation_counts"] = {
        key: 0 for key in spec["materialisation_count_required_keys"]
    }
    value["goal_view_counts"] = {
        key: 0 for key in spec["goal_view_count_required_keys"]
    }
    value["runtime_s"] = {key: 0.0 for key in spec["runtime_required_keys"]}
    value["storage"] = {key: 0 for key in spec["storage_required_keys"]}
    value["goal_view_execution_amendment_binding"] = copy.deepcopy(
        spec["goal_view_execution_amendment_binding_exact"]
    )
    value["goal_pose_semantics"] = copy.deepcopy(
        spec["goal_pose_semantics_exact"]
    )
    value["goal_cell_classification_counts"] = copy.deepcopy(
        spec["goal_cell_classification_counts_exact"]
    )
    value["goal_cell_classification_validation"] = copy.deepcopy(
        spec["goal_cell_classification_validation_exact"]
    )
    value["historical_renderer_limitations"] = dict(
        E.CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
    )
    value["reconstruction_prefix_custody"] = dict(
        E.CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
    )
    value["controller_execution_custody"] = dict(
        E.CONTRACT.CONTROLLER_EXECUTION_CUSTODY
    )
    value["execution_watchdog_status"] = dict(
        E.CONTRACT.EXECUTION_WATCHDOG_STATUS_SUCCESS
    )
    _populate_current_token_schema_fields(value, spec)
    value = E.attach_digest(value, "result_content_sha256")
    E._validate_schema_value("result", value)

    tampered = copy.deepcopy(value)
    tampered["markdown_report_order_amendment_binding"]["sha256"] = "0" * 64
    tampered = E.attach_digest(tampered, "result_content_sha256")
    with pytest.raises(E.QualificationError, match="Markdown report order.*drift"):
        E._validate_schema_value("result", tampered)

    value["storage"].pop("peak_vram_bytes")
    value = E.attach_digest(value, "result_content_sha256")
    with pytest.raises(E.QualificationError, match="storage lacks required keys"):
        E._validate_schema_value("result", value)


def test_gpu_environment_schema_requires_exact_serialization_preinference_check() -> None:
    spec = E.CONTRACT.build_output_schema()["files"]["gpu_environment_receipt"]
    value = {key: None for key in spec["required_keys"] if key != "content_digest"}
    value["schema"] = spec["schema"]
    expected_environment = E.CONTRACT.build_contract()["execution"][
        "environments"
    ]["encoder_predictor"]
    value["executable"] = expected_environment["interpreter"]
    value["interpreter_binding"] = copy.deepcopy(
        spec["interpreter_binding_exact"]
    )
    value["torch"] = expected_environment["torch"]
    value["device"] = {
        key: None for key in spec["device_required_keys"]
    }
    value["packages"] = {
        package_id: expected_environment[package_id]
        for package_id in spec["package_ids"]
    }
    value["import_closure"] = copy.deepcopy(spec["import_closure_exact"])
    value.update(copy.deepcopy(spec["preinference_exact"]))
    foundational = {}
    for package_id, package in E.CONTRACT.GPU_FOUNDATIONAL_PACKAGE_BINDINGS.items():
        package_root = Path(package["package_root"]).resolve()
        foundational[package_id] = {
            **copy.deepcopy(package),
            "package_root": str(package_root),
            "import_resolution": {
                "import_name": package["import_name"],
                "find_spec_origin": str(package_root / "__init__.py"),
                "submodule_search_locations": [str(package_root)],
                "live_module_file": str(package_root / "__init__.py"),
                "expected_package_root": str(package_root),
                "resolved_inside_frozen_package_root": True,
                "pass": True,
            },
        }
    value["foundational_package_roots"] = foundational
    value["foundational_package_closure_policy"] = copy.deepcopy(
        E.CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
    )
    _populate_current_token_schema_fields(value, spec)
    value = E.attach_digest(value)
    E._validate_schema_value("gpu_environment_receipt", value)

    missing = copy.deepcopy(value)
    missing.pop("gpu_receipt_serialization_preinference_check")
    missing = E.attach_digest(missing)
    with pytest.raises(E.QualificationError, match="lacks required keys"):
        E._validate_schema_value("gpu_environment_receipt", missing)

    tampered = copy.deepcopy(value)
    tampered["gpu_receipt_serialization_preinference_check"][
        "observed_json_type"
    ] = "pathlib.PosixPath"
    tampered = E.attach_digest(tampered)
    with pytest.raises(E.QualificationError, match="preinference check drift"):
        E._validate_schema_value("gpu_environment_receipt", tampered)


def test_gpu_child_receipt_mapping_survives_canonical_json_key_order() -> None:
    roundtripped = json.loads(
        E.canonical_json_bytes(
            {"gpu_child_execution_receipts": _synthetic_child_bindings()}
        )
    )
    assert list(roundtripped["gpu_child_execution_receipts"]) == [
        "MATERIALIZE",
        "PREFLIGHT",
    ]
    for file_id in ("result", "persistence_receipt"):
        spec = E.CONTRACT.build_output_schema()["files"][file_id]
        E._validate_gpu_child_execution_receipt_mapping(
            file_id, roundtripped, spec
        )

        missing = copy.deepcopy(roundtripped)
        missing["gpu_child_execution_receipts"].pop("MATERIALIZE")
        with pytest.raises(E.QualificationError, match="receipt bindings drift"):
            E._validate_gpu_child_execution_receipt_mapping(
                file_id, missing, spec
            )

        extra = copy.deepcopy(roundtripped)
        extra["gpu_child_execution_receipts"]["TERMINAL"] = copy.deepcopy(
            extra["gpu_child_execution_receipts"]["PREFLIGHT"]
        )
        with pytest.raises(E.QualificationError, match="receipt bindings drift"):
            E._validate_gpu_child_execution_receipt_mapping(file_id, extra, spec)

        wrong_phase = copy.deepcopy(roundtripped)
        preflight_path = wrong_phase["gpu_child_execution_receipts"]["PREFLIGHT"][
            "path"
        ]
        materialize_path = wrong_phase["gpu_child_execution_receipts"][
            "MATERIALIZE"
        ]["path"]
        wrong_phase["gpu_child_execution_receipts"]["PREFLIGHT"][
            "path"
        ] = materialize_path
        wrong_phase["gpu_child_execution_receipts"]["MATERIALIZE"][
            "path"
        ] = preflight_path
        with pytest.raises(E.QualificationError, match="receipt bindings drift"):
            E._validate_gpu_child_execution_receipt_mapping(
                file_id, wrong_phase, spec
            )


def test_result_schema_rejects_goal_cell_classification_validation_drift() -> None:
    spec = E.CONTRACT.build_output_schema()["files"]["result"]
    value = {key: None for key in spec["required_keys"] if key != "result_content_sha256"}
    value["schema"] = spec["schema"]
    value["materialisation_counts"] = {
        key: 0 for key in spec["materialisation_count_required_keys"]
    }
    value["goal_view_counts"] = {
        key: 0 for key in spec["goal_view_count_required_keys"]
    }
    value["runtime_s"] = {key: 0.0 for key in spec["runtime_required_keys"]}
    value["storage"] = {key: 0 for key in spec["storage_required_keys"]}
    value["goal_view_execution_amendment_binding"] = copy.deepcopy(
        spec["goal_view_execution_amendment_binding_exact"]
    )
    value["goal_pose_semantics"] = copy.deepcopy(spec["goal_pose_semantics_exact"])
    value["goal_cell_classification_counts"] = copy.deepcopy(
        spec["goal_cell_classification_counts_exact"]
    )
    value["goal_cell_classification_validation"] = copy.deepcopy(
        spec["goal_cell_classification_validation_exact"]
    )
    value["historical_renderer_limitations"] = copy.deepcopy(
        E.CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
    )
    value["reconstruction_prefix_custody"] = copy.deepcopy(
        E.CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
    )
    value["controller_execution_custody"] = copy.deepcopy(
        E.CONTRACT.CONTROLLER_EXECUTION_CUSTODY
    )
    value["execution_watchdog_status"] = copy.deepcopy(
        E.CONTRACT.EXECUTION_WATCHDOG_STATUS_SUCCESS
    )
    _populate_current_token_schema_fields(value, spec)
    value = E.attach_digest(value, "result_content_sha256")
    E._validate_schema_value("result", value)

    value["goal_cell_classification_validation"]["endpoint_reachable"] = 47
    value = E.attach_digest(value, "result_content_sha256")
    with pytest.raises(E.QualificationError, match="classification validation"):
        E._validate_schema_value("result", value)


def test_row_only_reducer_has_no_tensor_or_array_load_call() -> None:
    tree = ast.parse(inspect.getsource(E._reduce_candidate_rows_without_tensors))
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "_tensor_array" not in calls
    assert "np.load" not in calls


def test_row_only_reducer_does_not_open_tensor_even_on_fail_closed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = {"value": False}

    def forbidden(*_args: object, **_kwargs: object) -> object:
        opened["value"] = True
        raise AssertionError("tensor opener reached")

    monkeypatch.setattr(E, "_tensor_array", forbidden)
    with pytest.raises(E.QualificationError, match="candidate identities"):
        E._reduce_candidate_rows_without_tensors("a" * 40, [], {"records": []})
    assert opened["value"] is False


def test_cross_source_candidate_outcome_drift_fails_closed() -> None:
    base = {
        field: False for field in E._CROSS_SOURCE_CANDIDATE_INVARIANT_FIELDS
    }
    base.update(
        {
            "family": "FAMILY",
            "role": "heldout",
            "candidate_identity": "candidate-0",
            "successor_safe_action_count": 3,
            "successor_viable": True,
            "oracle_viability_admissible": True,
            "successor_nonviable": False,
        }
    )
    rows = {
        ("state-0", 0, source): {**base}
        for source in E.CONTRACT.SOURCE_IDS
    }
    E._validate_cross_source_candidate_invariants(rows)
    rows[("state-0", 0, "TWO_STEP_PREDICTED")][
        "successor_safe_action_count"
    ] = 0
    with pytest.raises(E.QualificationError, match="successor_safe_action_count"):
        E._validate_cross_source_candidate_invariants(rows)


def test_kinematic_reducer_uses_full_applied_tape_and_rejects_projection_drift() -> None:
    applied = np.zeros((3, 5, 3), dtype=np.float64)
    applied[:, :, 0] = 0.4
    applied[:, :, 1] = np.linspace(-0.25, 0.25, 15).reshape(3, 5)
    applied[:, :, 2] = -0.3
    authority = {
        "applied_action_blocks_raw_3x5x3": applied.tolist(),
        "action_blocks_raw_3x5x2": applied[:, :, (0, 2)].tolist(),
    }
    observed = E._validated_applied_action_tape(authority)
    assert np.array_equal(observed, applied)
    assert np.any(observed[:, :, 1] != 0.0)

    authority["action_blocks_raw_3x5x2"][0][0][0] = 0.5
    with pytest.raises(E.QualificationError, match="active-channel projection"):
        E._validated_applied_action_tape(authority)


def test_result_cross_binding_rejects_contract_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.json"
    closure = tmp_path / "closure.json"
    fixture.write_bytes(b"fixture\n")
    closure.write_bytes(b"closure\n")
    monkeypatch.setattr(E.CONTRACT, "TRACKED_FIXTURE_PATH", str(fixture))
    monkeypatch.setattr(E.CONTRACT, "TRACKED_SOURCE_CLOSURE_PATH", str(closure))
    monkeypatch.setattr(
        E, "_binding", lambda relative, _root: {"path": str(relative), "bound": True}
    )
    child_bindings = _synthetic_child_bindings()
    monkeypatch.setattr(
        E,
        "_gpu_child_execution_receipt_bindings",
        lambda *_args, **_kwargs: copy.deepcopy(child_bindings),
    )
    classification = {
        "true_future_gate_classification": "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
        "two_step_gate_passed": False,
        "two_step_gate_signal_or_null": None,
        "predicted_base_screens": {},
        "diagnostic_flags": {},
        "primary_classification": "RAW_LATENT_GOAL_COST_NO_GO",
        "secondary_classifications": [],
        "next_experiment": "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
    }
    reproduction = {
        "tensor_to_cost_rows_without_inference": True,
        "cost_rows_to_aggregates_without_inference": True,
        "candidate_rows": 1728,
        "selection_rows": 720,
        "paired_effect_rows": 32,
    }
    cpu_watchdog = {
        "no_progress_timeout_s": 3600,
        "global_timeout_s": 10800,
        "workers": 32,
        "completed_workers": 48,
        "no_progress_timeout_breaches": 0,
        "global_timeout_breaches": 0,
        "terminate_signals": 0,
        "kill_signals": 0,
        "automatic_retries": 0,
        "resume_used": False,
        "pass": True,
    }
    gpu_watchdog = {
        "phase": "GPU_MATERIALIZATION",
        "timeout_s": 10800,
        "timed_out": False,
        "automatic_retries": 0,
        "resume_used": False,
        "pass": True,
    }
    loaded = {
        "aggregate_metrics": {
            "classification": classification,
            "gates": {"gate": False},
            "paired_comparisons": {"comparison": {}},
        },
        "oracle_admissibility_fanout_index": {
            "states": 48,
            "current_blocks": 432,
            "successor_blocks": 3888,
            "physics_frames": 1_080_000,
            "cpu_watchdog_status": cpu_watchdog,
        },
        "gpu_inference_receipt": {"gpu_watchdog_status": gpu_watchdog},
            "latent_tensor_index": {
                "total_records": 5424,
                "counts_by_kind": {
                    "ONE_STEP_PREDICTED": 1728,
                    "TWO_STEP_PREDICTED": 1728,
                },
                "physical_encoder_materialisation_counts": {
                    "total_new_encoder_frames": 144,
                    "batches": 9,
                    "authority_current_payload_copies": 48,
                },
        },
        "goal_view_index": {
            "states": 48,
            "records": [{}] * 48,
            "failed_state_ids": [],
            "goal_pose_semantics": copy.deepcopy(E.GOAL_POSE_SEMANTICS),
            "goal_cell_classification_counts": copy.deepcopy(
                E.CONTRACT.GOAL_CELL_CLASSIFICATION_COUNTS
            ),
            "goal_cell_classification_validation": copy.deepcopy(
                E.CONTRACT.GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
            ),
        },
        "context_reconstruction_index": {
            "branch_snapshot_authority_validation": {"passed": 48}
        },
        "persistence_receipt": {
            "row_reproduction": reproduction,
                "goal_view_execution_amendment_binding": copy.deepcopy(
                    E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
                ),
                "current_token_execution_amendment_binding": copy.deepcopy(
                    E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT_BINDING
                ),
                "gpu_receipt_serialization_amendment_binding": copy.deepcopy(
                    E.CONTRACT.GPU_RECEIPT_SERIALIZATION_EXECUTION_AMENDMENT_BINDING
                ),
                "gpu_child_receipt_order_amendment_binding": copy.deepcopy(
                    E.CONTRACT.GPU_CHILD_RECEIPT_ORDER_EXECUTION_AMENDMENT_BINDING
                ),
                "markdown_report_order_amendment_binding": copy.deepcopy(
                    E.CONTRACT.MARKDOWN_REPORT_ORDER_EXECUTION_AMENDMENT_BINDING
                ),
                "current_token_authority_policy": copy.deepcopy(
                    E.CONTRACT.CURRENT_TOKEN_AUTHORITY_POLICY
                ),
                "gpu_child_execution_receipts": copy.deepcopy(child_bindings),
                "prohibition_counters": E.prohibition_counters(),
        },
    }
    execution_watchdog = E._successful_execution_watchdog_status(
        loaded["oracle_admissibility_fanout_index"],
        loaded["gpu_inference_receipt"],
    )
    loaded["persistence_receipt"]["execution_watchdog_status"] = execution_watchdog
    result = {
        "source_freeze_commit": "a" * 40,
        "head": "a" * 40,
        "experiment_id": E.CONTRACT.EXPERIMENT_ID,
        "contract_sha256": E.CONTRACT.CONTRACT_SHA256,
        "output_schema_sha256": E.CONTRACT.OUTPUT_SCHEMA_SHA256,
        "seed": E.CONTRACT.SEED,
        "fixture_sha256": E.sha256_file(fixture),
        "source_closure_sha256": E.sha256_file(closure),
            "goal_view_execution_amendment_binding": copy.deepcopy(
                E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
            ),
            "current_token_execution_amendment_binding": copy.deepcopy(
                E.CONTRACT.CURRENT_TOKEN_EXECUTION_AMENDMENT_BINDING
            ),
            "gpu_receipt_serialization_amendment_binding": copy.deepcopy(
                E.CONTRACT.GPU_RECEIPT_SERIALIZATION_EXECUTION_AMENDMENT_BINDING
            ),
            "gpu_child_receipt_order_amendment_binding": copy.deepcopy(
                E.CONTRACT.GPU_CHILD_RECEIPT_ORDER_EXECUTION_AMENDMENT_BINDING
            ),
            "markdown_report_order_amendment_binding": copy.deepcopy(
                E.CONTRACT.MARKDOWN_REPORT_ORDER_EXECUTION_AMENDMENT_BINDING
            ),
            "current_token_authority_policy": copy.deepcopy(
                E.CONTRACT.CURRENT_TOKEN_AUTHORITY_POLICY
            ),
            "gpu_child_execution_receipts": copy.deepcopy(child_bindings),
        "goal_pose_semantics": copy.deepcopy(E.GOAL_POSE_SEMANTICS),
        "goal_cell_classification_counts": copy.deepcopy(
            E.CONTRACT.GOAL_CELL_CLASSIFICATION_COUNTS
        ),
        "goal_cell_classification_validation": copy.deepcopy(
            E.CONTRACT.GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
        ),
        "metrics": {"path": str(E.AGGREGATE_REL), "bound": True},
        "gpu_inference_custody": {"path": str(E.GPU_INFERENCE_REL), "bound": True},
        "gpu_environment_receipt_binding": {
            "path": str(E.GPU_ENVIRONMENT_REL),
            "bound": True,
        },
        "cpu_runtime_input_inventory_binding": {
            "path": str(E.CPU_RUNTIME_INPUT_INVENTORY_REL),
            "bound": True,
        },
        "dense_route_replay_input_index_binding": {
            "path": str(E.DENSE_REPLAY_INPUT_INDEX_REL),
            "bound": True,
        },
        "gates": loaded["aggregate_metrics"]["gates"],
        "paired_materiality": loaded["aggregate_metrics"]["paired_comparisons"],
        "requirements_custody": E.CONTRACT.build_contract()["requirements_custody"],
        "materialisation_counts": {
            "states": 48,
            "candidates": 576,
            "current_blocks": 432,
            "successor_blocks": 3888,
            "total_oracle_blocks": 4320,
            "physics_frames": 1_080_000,
            "reconstruction_prefix_blocks": 1_920,
            "reconstruction_physics_frames": 480_000,
            "oracle_fanout_blocks": 4_320,
            "oracle_fanout_physics_frames": 1_080_000,
            "total_simulator_blocks": 6_240,
            "total_simulator_physics_frames": 1_560_000,
                "snapshot_reproductions": 48,
                "new_encoder_frames": 144,
                "new_encoder_batches": 9,
                "current_authority_payload_copies": 48,
                "current_token_reencodes": 0,
            "latent_tensors": 5424,
            "predicted_tensors": 3456,
            "candidate_evidence_rows": 1728,
            "selection_evidence_rows": 720,
            "paired_effect_evidence_rows": 32,
        },
        "goal_view_counts": {"states": 48, "views": 48, "failed": 0},
        "row_reproduction": reproduction,
        "prohibition_counters": E.prohibition_counters(),
        "historical_renderer_limitations": dict(
            E.CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        ),
        "reconstruction_prefix_custody": dict(
            E.CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
        ),
        "controller_execution_custody": dict(
            E.CONTRACT.CONTROLLER_EXECUTION_CUSTODY
        ),
        "execution_watchdog_status": execution_watchdog,
        **classification,
    }
    E._validate_result_cross_bindings(
        result,
        loaded,
        output_root=tmp_path,
        candidate_rows=[{}] * 1728,
        selection_rows=[{}] * 720,
        paired_rows=[{}] * 32,
    )
    result["contract_sha256"] = "0" * 64
    with pytest.raises(E.QualificationError, match="contract digest"):
        E._validate_result_cross_bindings(
            result,
            loaded,
            output_root=tmp_path,
            candidate_rows=[{}] * 1728,
            selection_rows=[{}] * 720,
            paired_rows=[{}] * 32,
        )


def test_persistence_manifest_binds_report_and_has_exact_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "stable.bin").write_bytes(b"stable")
    monkeypatch.setattr(E, "_binding", lambda *_args, **_kwargs: {"bound": True})
    monkeypatch.setattr(
        E,
        "_gpu_child_execution_receipt_bindings",
        lambda *_args, **_kwargs: _synthetic_child_bindings(),
    )
    monkeypatch.setattr(E, "validate_input_hashes", lambda: {})
    monkeypatch.setattr(
        E,
        "_successful_execution_watchdog_status",
        lambda *_args, **_kwargs: {"pass": True},
    )
    report = b"# report\n"
    receipt = E._build_persistence_receipt(
        "a" * 40,
        root,
        {"candidate_rows": 1728},
        report_payload=report,
        execution_watchdog_status={"pass": True},
    )
    rows = {row["path"]: row for row in receipt["artifact_manifest"]}
    assert rows["report.md"]["sha256"] == E.hashlib.sha256(report).hexdigest()
    assert receipt["artifact_manifest_exclusions"] == [
        "receipts/persistence.json",
        "result.json",
        "receipts/RUNNING.json",
    ]
    assert receipt["goal_view_execution_amendment_binding"] == (
        E.CONTRACT.GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
    )


def _install_execute_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_postpublish: bool
) -> tuple[Path, Path, Path]:
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(E, "OUTPUT_ROOT", canonical)
    tracked_result = tmp_path / "tracked-result.json"
    tracked_report = tmp_path / "tracked-report.md"
    monkeypatch.setattr(E.CONTRACT, "TRACKED_RESULT_PATH", str(tracked_result))
    monkeypatch.setattr(E.CONTRACT, "TRACKED_REPORT_PATH", str(tracked_report))
    monkeypatch.setattr(E, "validate_source_freeze_commit", lambda *_a, **_k: None)

    def fake_preflight(_commit: str, *, output_root: Path, **_kwargs: object) -> None:
        output_root.mkdir()

    def fake_evaluate(_commit: str, *, output_root: Path) -> dict[str, str]:
        (output_root / E.RESULT_REL).write_bytes(b'{"result":true}\n')
        (output_root / E.REPORT_REL).write_bytes(b"# report\n")
        return {"result_content_sha256": "d" * 64}

    calls = {"count": 0}

    def fake_check(**_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if fail_postpublish and calls["count"] == 2:
            raise E.QualificationError("postpublication failure")
        return {
            "primary_classification": "RAW_LATENT_GOAL_COST_NO_GO",
            "nothing_running": True,
        }

    monkeypatch.setattr(E, "preflight", fake_preflight)
    monkeypatch.setattr(E, "materialize", lambda *_a, **_k: None)
    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    monkeypatch.setattr(E, "check", fake_check)
    monkeypatch.setattr(E, "_active_experiment_processes", lambda: [])
    return canonical, tracked_result, tracked_report


def test_execute_atomically_publishes_and_copies_tracked_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical, tracked_result, tracked_report = _install_execute_stubs(
        tmp_path, monkeypatch, fail_postpublish=False
    )
    receipt = E.execute("a" * 40, output_root=canonical, workers=32)
    assert receipt["atomic_publication"] is True
    assert canonical.is_dir()
    assert tracked_result.read_bytes() == (canonical / E.RESULT_REL).read_bytes()
    assert tracked_report.read_bytes() == (canonical / E.REPORT_REL).read_bytes()
    assert not list(tmp_path.glob(".canonical.attempt-*"))


def test_execute_archives_and_removes_tracked_bytes_on_postpublish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical, tracked_result, tracked_report = _install_execute_stubs(
        tmp_path, monkeypatch, fail_postpublish=True
    )
    with pytest.raises(E.QualificationError, match="postpublication"):
        E.execute("a" * 40, output_root=canonical, workers=32)
    assert not canonical.exists()
    assert not tracked_result.exists()
    assert not tracked_report.exists()
    archives = list(tmp_path.glob(".*.failed-*"))
    assert len(archives) == 1
    failure = json.loads((archives[0] / "receipts/failure.json").read_text())
    E.validate_digest(failure)
    assert failure["partial_artifacts_reusable"] is False


def test_execute_preflight_failure_always_gets_self_digesting_archive_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(E, "OUTPUT_ROOT", canonical)
    monkeypatch.setattr(E, "validate_source_freeze_commit", lambda *_a, **_k: None)
    monkeypatch.setattr(E, "_active_experiment_processes", lambda: [])

    def fail_preflight(*_args: object, **_kwargs: object) -> None:
        raise E.QualificationError("synthetic preflight failure")

    monkeypatch.setattr(E, "preflight", fail_preflight)
    with pytest.raises(E.QualificationError, match="synthetic preflight"):
        E.execute("a" * 40, output_root=canonical, workers=32)
    assert not canonical.exists()
    archives = list(tmp_path.glob(".*.failed-*"))
    assert len(archives) == 1
    failure = json.loads((archives[0] / "receipts/failure.json").read_text())
    E.validate_digest(failure)
    assert failure["phase"] == "PREFLIGHT"
    assert failure["partial_artifacts_reusable"] is False
    assert failure["nothing_running"] is True


def test_execute_rejects_nonfrozen_canonical_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen-output"
    monkeypatch.setattr(E, "OUTPUT_ROOT", frozen)
    monkeypatch.setattr(E, "validate_source_freeze_commit", lambda *_a, **_k: None)
    with pytest.raises(E.QualificationError, match="differs from frozen path"):
        E.execute("a" * 40, output_root=tmp_path / "different-output", workers=32)


def test_post_result_source_validation_does_not_bind_live_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40

    def fake_git(*arguments: str) -> str:
        return commit if arguments[0] == "rev-parse" and arguments[1] != "HEAD" else "b" * 40

    monkeypatch.setattr(E, "git_output", fake_git)
    E.validate_source_freeze_commit(commit, require_live_head=False)
    with pytest.raises(E.QualificationError, match="live HEAD"):
        E.validate_source_freeze_commit(commit, require_live_head=True)


def test_contact_block_advances_every_production_boundary_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EpisodeState:
        episode_step = 200
        reset_count = 0

        def step(self) -> None:
            self.episode_step += 1

    class Runner:
        _physics_steps_per_policy = 50
        _policy_dt_ns = 100
        _policy_steps_per_command_tick = 1
        _block_size = 5
        _blocks_in_episode = 40
        _sim_time_ns = 20_000

        def __init__(self) -> None:
            self.episode_states = [EpisodeState()]
            self.build = SimpleNamespace(scene=SimpleNamespace(step=lambda: None))
            self.policy = SimpleNamespace(act=lambda _observation: np.zeros((1, 12)))
            self._step_policy_step = lambda _target: None

        def _build_observation(self, _target: np.ndarray) -> np.ndarray:
            return np.zeros((1, 1))

        def _apply_joint_targets(self, _targets: np.ndarray) -> None:
            return None

        def execute_requested_block(self, requested: np.ndarray) -> SimpleNamespace:
            for tick in range(self._block_size):
                self._step_policy_step(requested[:, tick, :])
            return SimpleNamespace(executed=requested.copy())

    runner = Runner()
    ctx = SimpleNamespace(
        runner=runner,
        ticks_executed=200,
        episode_ticks=200,
        policy_steps=200,
        last_block_executed=None,
    )
    monkeypatch.setattr(E, "_contact_links_at_step", lambda *_args: [])
    requested = np.zeros((1, 5, 3), dtype=np.float32)
    _block, executed, contacts = E._execute_contact_block(ctx, requested, {})
    assert executed.shape == (1, 5, 3)
    assert len(contacts) == 250
    assert runner.episode_states[0].episode_step == 205
    assert runner._blocks_in_episode == 41
    assert ctx.ticks_executed == 205
    assert ctx.episode_ticks == 205
    assert ctx.policy_steps == 205
    assert np.array_equal(ctx.last_block_executed, executed)


def test_raw_continuation_accepts_terminal_flags_but_rejects_nan() -> None:
    episode = SimpleNamespace(episode_step=205, reset_count=0)
    runner = SimpleNamespace(
        n_envs=1,
        _block_size=5,
        _policy_steps_per_command_tick=1,
        _policy_dt_ns=100,
        _command_dt_ns=500,
        _sim_time_ns=20_500,
        episode_states=[episode],
        _last_executed=np.zeros((1, 3), dtype=np.float64),
        _consecutive_tipped_blocks=np.asarray([2]),
        _blocks_in_episode=np.asarray([41]),
        safety=SimpleNamespace(
            min_vx_mps=-1.0,
            max_vx_mps=1.0,
            min_vy_mps=-1.0,
            max_vy_mps=1.0,
            max_yaw_rate_radps=1.0,
        ),
    )
    policy = SimpleNamespace(
        _last_actions=np.zeros((1, 2)), policy_joint_names=["a", "b"]
    )
    ctx = SimpleNamespace(
        runner=runner,
        policy=policy,
        ticks_executed=205,
        episode_ticks=205,
        policy_steps=205,
        reset_in_last_block=False,
        episode_start_reset_count=0,
        last_block_executed=np.zeros((1, 5, 3), dtype=np.float64),
    )
    flags = {"fall": True, "out_of_bounds": True, "tipped": True, "nan": False}
    v1 = SimpleNamespace(TICKS=5, _termination_flags=lambda _ctx: flags)
    boundary = E._raw_continuation_boundary(ctx, v1)
    assert boundary["terminal_flags"] == flags
    assert boundary["capture_mode"] == "RAW_CONTINUATION_AFTER_CURRENT_H1"
    v1._termination_flags = lambda _ctx: {**flags, "nan": True}
    with pytest.raises(E.QualificationError, match="NaN"):
        E._raw_continuation_boundary(ctx, v1)


def test_distribution_record_closure_validates_every_present_row(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site-packages"
    package = site / "demo"
    dist = site / "demo-1.0.dist-info"
    package.mkdir(parents=True)
    dist.mkdir()
    payload_path = package / "a.py"
    payload_path.write_bytes(b"x")
    encoded = base64.urlsafe_b64encode(hashlib.sha256(b"x").digest()).decode().rstrip("=")
    record_path = dist / "RECORD"
    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["demo/a.py", f"sha256={encoded}", "1"])
        writer.writerow(["demo-1.0.dist-info/RECORD", "", ""])
        writer.writerow(["demo/missing.pyc", "", ""])
    present = [
        {"path": "demo/a.py", "sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1},
        {
            "path": "demo-1.0.dist-info/RECORD",
            "sha256": E.sha256_file(record_path),
            "bytes": record_path.stat().st_size,
        },
    ]
    present.sort(key=lambda row: row["path"])
    expected = {
        "distribution": "demo",
        "version": "1.0",
        "package_root": str(package),
        "record_closure": {
            "record_path": str(record_path),
            "record_sha256": E.sha256_file(record_path),
            "record_bytes": record_path.stat().st_size,
            "record_entries": 3,
            "declared_hash_entries": 1,
            "present_files": 2,
            "absent_unhashed_files": 1,
            "absent_unhashed_path_list_sha256": hashlib.sha256(
                E.canonical_json_bytes(["demo/missing.pyc"])[:-1]
            ).hexdigest(),
            "present_file_bytes": sum(row["bytes"] for row in present),
            "present_file_aggregate_sha256": hashlib.sha256(
                E.canonical_json_bytes(present)[:-1]
            ).hexdigest(),
        },
    }
    receipt = E._validate_distribution_record_closure("demo", expected)
    assert receipt["pass"] is True
    payload_path.write_bytes(b"y")
    with pytest.raises(E.QualificationError, match="RECORD digest drift"):
        E._validate_distribution_record_closure("demo", expected)


def test_cpu_runtime_import_resolution_rejects_pythonpath_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_root = tmp_path / "frozen" / "demo"
    frozen_origin = frozen_root / "__init__.py"
    frozen_origin.parent.mkdir(parents=True)
    frozen_origin.write_bytes(b"# frozen\n")
    spec = SimpleNamespace(
        origin=str(frozen_origin),
        submodule_search_locations=[str(frozen_root)],
    )
    module = SimpleNamespace(__file__=str(frozen_origin))
    monkeypatch.setattr(E.importlib.util, "find_spec", lambda _name: spec)
    monkeypatch.setattr(E.importlib, "import_module", lambda _name: module)
    receipt = E._runtime_import_resolution("demo", frozen_root)
    assert receipt["expected_package_root"] == str(frozen_root.resolve())
    assert receipt["resolved_inside_frozen_package_root"] is True
    assert receipt["pass"] is True

    shadow_origin = tmp_path / "shadow" / "demo" / "__init__.py"
    shadow_origin.parent.mkdir(parents=True)
    shadow_origin.write_bytes(b"# shadow\n")
    spec.origin = str(shadow_origin)
    spec.submodule_search_locations = [str(shadow_origin.parent)]
    module.__file__ = str(shadow_origin)
    with pytest.raises(E.QualificationError, match="shadowing"):
        E._runtime_import_resolution("demo", frozen_root)


def test_prefreeze_byte_inventory_roundtrip_validates_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    path = tmp_path / "inputs/state.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"outcome":"never parsed by this gate"}\n')
    row = {
        "state_id": "purpose-0",
        "path": "inputs/state.json",
        "sha256": E.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    canonical = E.canonical_json_bytes([row])[:-1]
    binding = {
        "record_count": 1,
        "total_bytes": path.stat().st_size,
        "canonical_records_bytes": len(canonical),
        "canonical_sorted_path_sha_bytes_aggregate_sha256": hashlib.sha256(
            canonical
        ).hexdigest(),
        "record_fields": ["state_id", "path", "sha256", "bytes"],
    }
    # Persisted canonical JSON sorts object keys; field insertion order must not
    # alter the prospectively frozen row-array order or aggregate.
    roundtripped = json.loads(E.canonical_json_bytes([row]))
    validation = E._validate_prefreeze_byte_inventory(
        roundtripped, binding, states=None
    )
    assert validation["validated_before_json_parse"] is True
    assert validation["outcome_fields_parsed_before_validation"] == []
    path.write_bytes(b"drift")
    with pytest.raises(E.QualificationError, match="file drift"):
        E._validate_prefreeze_byte_inventory(roundtripped, binding, states=None)


def test_gpu_child_watchdogs_use_frozen_phase_timeouts_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        observed.append(int(kwargs["timeout"]))
        if len(observed) == 2:
            raise E.subprocess.TimeoutExpired(cmd="gpu", timeout=kwargs["timeout"])
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(E.subprocess, "run", fake_run)
    root = tmp_path / "attempt"
    root.mkdir()
    common = [
        "--source-freeze-commit",
        "a" * 40,
        "--output-root",
        str(root),
    ]
    E._run_gpu_child(["preflight", *common])
    assert observed == [E.CONTRACT.EXECUTION_WATCHDOGS["gpu_preflight_timeout_s"]]
    preflight = E.load_json(root / "receipts/gpu_child_preflight.json")
    assert preflight["phase"] == "PREFLIGHT"
    bindings = E._gpu_child_execution_receipt_bindings(
        root, "a" * 40, ("PREFLIGHT",)
    )
    assert bindings == {
        "PREFLIGHT": E._binding(Path("receipts/gpu_child_preflight.json"), root)
    }
    with pytest.raises(E.GPUChildExecutionError, match="materialize watchdog"):
        E._run_gpu_child(["materialize", *common])
    assert observed[-1] == E.CONTRACT.EXECUTION_WATCHDOGS[
        "gpu_materialization_timeout_s"
    ]
    receipt = E.load_json(root / "receipts/gpu_child_materialize.json")
    assert receipt["timed_out"] is True
    assert receipt["returncode"] is None
    assert receipt["pass"] is False
    E.validate_digest(receipt)


def test_gpu_child_failure_streams_survive_whole_attempt_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_commit = "b" * 40
    attempt = tmp_path / ".attempt"
    attempt.mkdir()
    stdout = "partial child output\n"
    stderr = "Traceback\nexact GPU failure marker\n"

    def fail(*args: object, **_kwargs: object) -> object:
        raise E.subprocess.CalledProcessError(
            7, args[0], output=stdout, stderr=stderr
        )

    monkeypatch.setattr(E.subprocess, "run", fail)
    with pytest.raises(
        E.GPUChildExecutionError, match="exact GPU failure marker"
    ) as caught:
        E._run_gpu_child(
            [
                "materialize",
                "--source-freeze-commit",
                source_commit,
                "--output-root",
                str(attempt),
            ]
        )
    child = E.load_json(attempt / "receipts/gpu_child_materialize.json")
    assert child["returncode"] == 7
    assert child["stderr"]["tail_utf8"].endswith(
        "exact GPU failure marker\n"
    )
    assert (attempt / child["stdout"]["path"]).read_text() == stdout
    assert (attempt / child["stderr"]["path"]).read_text() == stderr
    E.validate_digest(child)

    monkeypatch.setattr(E, "_active_experiment_processes", lambda: [])
    archive = E._archive_failed_attempt(
        attempt,
        source_freeze_commit=source_commit,
        phase="MATERIALIZATION",
        error=caught.value,
    )
    assert archive is not None and not attempt.exists()
    archived_failure = E.load_json(archive / "receipts/failure.json")
    binding = archived_failure["failed_child_execution_receipt"]
    assert binding == E._binding(
        E.Path("receipts/gpu_child_materialize.json"), archive
    )
    assert "exact GPU failure marker" in archived_failure["error_message"]
    assert (
        archive / "materialization/logs/gpu_materialize.stderr.log"
    ).read_text() == stderr


def test_successful_gpu_terminal_check_is_observational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "final"
    root.mkdir()
    monkeypatch.setattr(
        E.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='{"pass":true}\n', stderr="", returncode=0
        ),
    )
    result = E._run_gpu_child(
        [
            "check",
            "--source-freeze-commit",
            "c" * 40,
            "--output-root",
            str(root),
        ]
    )
    assert result.returncode == 0
    assert list(root.rglob("*")) == []


def test_report_source_contains_exact_current_token_amendment_section() -> None:
    source = inspect.getsource(E._markdown_report)
    assert "## Current-token authority amendment and BF16 cohort limitation" in source
    assert "## Markdown report order amendment and failed-attempt custody" in source
    assert "json.dumps(result['diagnostic_flags'], sort_keys=True)" in source
    assert "144 frames in nine fixed batches of 16" in source
    assert "Current-token re-encodes are zero" in source


def test_markdown_report_survives_canonical_json_key_order_roundtrip() -> None:
    population_aggregate = {
        "states": 1,
        "pairwise_accuracy": 0.0,
        "spearman_rho": 0.0,
        "normalized_regret": 0.0,
        "best_route_top3_rate": 0.0,
        "selected_progress_ratio": 0.0,
        "selected_immediate_contacts_h1": 0,
        "selected_nonviable_successors": 0,
        "selected_stuck": 0,
    }
    family_aggregate = {
        "states": 1,
        "pairwise_accuracy": 0.0,
        "normalized_regret": 0.0,
        "best_route_top3_rate": 0.0,
        "selected_route_progress_m_sum": 0.0,
        "complete_family_collapse": False,
    }
    aggregates = {
        "per_role": {
            source: {
                "heldout": {
                    "populations": {
                        population: {
                            "aggregate": copy.deepcopy(population_aggregate),
                            **(
                                {
                                    "per_family": {
                                        family: copy.deepcopy(family_aggregate)
                                        for family in E.CONTRACT.FAMILY_IDS
                                    }
                                }
                                if population == "ORACLE_VIABILITY_ADMISSIBLE"
                                else {}
                            ),
                        }
                        for population in E.CONTRACT.POPULATION_IDS
                    }
                }
            }
            for source in (
                "TRUE_FUTURE_LATENT_COST",
                "ONE_STEP_PREDICTED_LATENT_COST",
                "TWO_STEP_PREDICTED_LATENT_COST",
                "KINEMATIC_ROUTE_BASELINE",
                "RANDOM",
            )
        },
        "paired_comparisons": {
            comparison_id: {
                "mean_route_progress_gain_m": {
                    "point": 0.0,
                    "bootstrap_lower_95": 0.0,
                    "bootstrap_upper_95": 0.0,
                },
                "normalized_regret_reduction": {
                    "point": 0.0,
                    "bootstrap_lower_95": 0.0,
                    "bootstrap_upper_95": 0.0,
                },
                "material_improvement": False,
            }
            for comparison_id in E.CONTRACT.PAIRED_COMPARISON_IDS
        },
        "latent_progress_diagnostics": {
            source: {
                "monotonicity": {
                    "all": {
                        "current_to_h3_nonincrease_fraction": 0.0,
                        "overall_monotonic_trajectory_fraction": 0.0,
                    }
                }
            }
            for source in E.CONTRACT.SOURCE_IDS
        },
        "all_candidate_tendency_diagnostics": {
            source: {
                "all": {
                    outcome: {"downranking_accuracy": 0.0}
                    for outcome in (
                        "immediate_contact_h1",
                        "successor_nonviable",
                        "stuck",
                        "no_progress",
                    )
                }
            }
            for source in E.CONTRACT.SOURCE_IDS
        },
    }
    result = {
        "source_freeze_commit": "f" * 40,
        "seed": E.CONTRACT.SEED,
        "materialisation_counts": {"states": 48},
        "goal_cell_classification_counts": {"valid": 48},
        "goal_pose_semantics": {"candidate_independent": True},
        "current_token_execution_amendment_binding": {"path": "current.json"},
        "current_token_authority_policy": {
            "device_batch_cohort_limitation": "synthetic"
        },
        "gpu_receipt_serialization_amendment_binding": {"path": "gpu.json"},
        "gpu_child_receipt_order_amendment_binding": {"path": "child.json"},
        "markdown_report_order_amendment_binding": {"path": "markdown.json"},
        "historical_renderer_limitations": {"preserved": True},
        "controller_execution_custody": {"training": 0},
        "gates": {"true_future": {"pass": False}},
        "two_step_gate_passed": False,
        "primary_classification": "SYNTHETIC",
        "secondary_classifications": ["SYNTHETIC_SECONDARY"],
        # Deliberately differs from canonical JSON object-member order.
        "diagnostic_flags": {"z_last": True, "a_first": False},
        "next_experiment": "SYNTHETIC_NEXT",
        "runtime_s": {"total": 0.0},
        "storage": {"bytes": 0},
    }

    before = E._markdown_report(result, aggregates)
    roundtripped_result = json.loads(E.canonical_json_bytes(result))
    roundtripped_aggregates = json.loads(E.canonical_json_bytes(aggregates))

    assert E._markdown_report(roundtripped_result, roundtripped_aggregates) == before


def test_successful_watchdog_status_is_exact_and_rejects_partial_resume() -> None:
    cpu = {
        "no_progress_timeout_s": 3600,
        "global_timeout_s": 10800,
        "workers": 32,
        "completed_workers": 48,
        "no_progress_timeout_breaches": 0,
        "global_timeout_breaches": 0,
        "terminate_signals": 0,
        "kill_signals": 0,
        "automatic_retries": 0,
        "resume_used": False,
        "pass": True,
    }
    gpu = {
        "gpu_watchdog_status": {
            "phase": "GPU_MATERIALIZATION",
            "timeout_s": 10800,
            "timed_out": False,
            "automatic_retries": 0,
            "resume_used": False,
            "pass": True,
        }
    }
    status = E._successful_execution_watchdog_status(
        {"cpu_watchdog_status": cpu}, gpu
    )
    assert status["configuration"] == E.CONTRACT.EXECUTION_WATCHDOGS
    assert status["pass"] is True
    cpu["resume_used"] = True
    with pytest.raises(E.QualificationError, match="watchdog custody"):
        E._successful_execution_watchdog_status(
            {"cpu_watchdog_status": cpu}, gpu
        )
