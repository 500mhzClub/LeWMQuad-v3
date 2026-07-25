from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest import mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_overlapping_tokenization_v1.py"
LAUNCHER_PATH = ROOT / "scripts/launch_go2_rgb_overlapping_tokenization_v1.py"
STATIC_V3_RUNNER_PATH = (
    ROOT / "scripts/run_go2_shared_jepa_v5_multires_probe_v3.py"
)
TEMPORAL_LIFECYCLE_PATH = (
    ROOT / "scripts/run_go2_rgb_causal_temporal_perception_v1.py"
)
STATIC_EVIDENCE_PATH = (
    ROOT / "lewm/models/observable_camera_ray_evidence_v4.py"
)


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_runner_and_launcher_import_are_source_only() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"torch", "numpy", "PIL"}:
            raise AssertionError(f"source-only import loaded {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        module = _load_runner("_overlap_source_only_import")
        launcher_spec = importlib.util.spec_from_file_location(
            "_overlap_launcher_source_only_import", LAUNCHER_PATH
        )
        assert launcher_spec is not None and launcher_spec.loader is not None
        launcher = importlib.util.module_from_spec(launcher_spec)
        sys.modules[launcher_spec.name] = launcher
        launcher_spec.loader.exec_module(launcher)

    assert module.PREFLIGHT_ENVIRONMENT_KEY == (
        "LEWM_RGB_OVERLAPPING_TOKENIZATION_V1_PREFLIGHT_JSON"
    )
    assert module.contract.MODEL_FAMILY.endswith(
        "overlapping_tokenization_v1"
    )
    assert launcher.PREFLIGHT_ENVIRONMENT_KEY == (
        module.PREFLIGHT_ENVIRONMENT_KEY
    )
    assert module._BASE.contract is module.contract
    assert launcher._BASE.contract is launcher.contract


def test_inherited_determinism_warning_is_static_ground_query_policy() -> None:
    module = _load_runner("_overlap_static_determinism_policy")
    warning = "grid_sampler_2d_backward_cuda does not have a deterministic"
    static_v3_source = STATIC_V3_RUNNER_PATH.read_text(encoding="utf-8")
    inherited_source = TEMPORAL_LIFECYCLE_PATH.read_text(encoding="utf-8")
    evidence_source = STATIC_EVIDENCE_PATH.read_text(encoding="utf-8")

    assert Path(
        module._BASE._execute_after_reservation.__code__.co_filename
    ).resolve() == TEMPORAL_LIFECYCLE_PATH.resolve()
    assert warning in static_v3_source
    assert warning in inherited_source
    assert "F.grid_sample(" in evidence_source
    # The overlap adapter retains the unchanged static Camera ground-query
    # gradient policy; it does not need a temporal execution-loop fork.
    assert "_execute_after_reservation =" not in RUNNER_PATH.read_text(
        encoding="utf-8"
    )


def test_static_batch_has_only_rgb_geometry_and_supervision() -> None:
    module = _load_runner("_overlap_static_batch")
    pairs = [
        {
            "dataset_role": "train",
            "current_endpoint_sha256": "current-0",
            "next_endpoint_sha256": "next-0",
            # These schedule metadata fields must never enter the model batch.
            "primitive": "yaw_left",
            "relative_se2_current_frame": [9.0, 8.0, 7.0],
        },
        {
            "dataset_role": "train",
            "current_endpoint_sha256": "current-1",
            "next_endpoint_sha256": "next-1",
            "primitive": "forward_slow",
            "relative_se2_current_frame": [6.0, 5.0, 4.0],
        },
    ]
    opened: list[tuple[str, str, str, str]] = []

    def frame(endpoint_id: str, *, role: str, arm: str, stage: str):
        opened.append((endpoint_id, role, arm, stage))
        code = float(endpoint_id.endswith("1"))
        if endpoint_id.startswith("next"):
            code += 2.0
        return {
            "image": torch.full((3, 4, 4), code),
            "camera_origin": torch.full((3,), code),
            "camera_basis": torch.full((3, 3), code),
            "ground": torch.tensor(code),
        }

    trainer = SimpleNamespace(
        r=SimpleNamespace(torch=torch),
        inputs=SimpleNamespace(frame=frame),
        supervision=lambda frames, _device: SimpleNamespace(
            endpoint_count=len(frames)
        ),
    )
    batch = module._visual_only_batch(
        trainer,
        pairs,
        [0, 1],
        torch.device("cpu"),
        role="train",
        arm=module.INHERITED_TRAIN_ARM_NAME,
        stage="camera_gradient",
    )
    assert set(batch["forward"]) == {
        "current_image",
        "next_image",
        "current_camera_origin_body_m",
        "current_camera_basis_body_fru",
        "current_ground_plane_z_body_m",
        "next_camera_origin_body_m",
        "next_camera_basis_body_fru",
        "next_ground_plane_z_body_m",
    }
    assert batch["current_supervision"].endpoint_count == 2
    assert batch["next_supervision"].endpoint_count == 2
    assert all(
        role == "train"
        and arm == module.ARM_NAME
        and stage == "camera_gradient"
        for _endpoint, role, arm, stage in opened
    )
    module._assert_static_payload(
        batch["forward"], name="synthetic static batch"
    )


def test_static_camera_pair_uses_two_independent_forward_frame_calls() -> None:
    module = _load_runner("_overlap_static_camera_pair")
    batch_size = 2
    current_output = SimpleNamespace(
        bev=torch.zeros(batch_size, 3, 4, 4)
    )
    next_output = SimpleNamespace(
        bev=torch.ones(batch_size, 3, 4, 4)
    )

    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, ...]] = []

        def forward_frame(self, *args):
            self.calls.append(args)
            return (current_output, next_output)[len(self.calls) - 1]

        def forward_camera_pair(self, *_args, **_kwargs):
            raise AssertionError("temporal pair forward is forbidden")

    forward = {
        "current_image": torch.zeros(batch_size, 3, 4, 4),
        "next_image": torch.ones(batch_size, 3, 4, 4),
        "current_camera_origin_body_m": torch.zeros(batch_size, 3),
        "current_camera_basis_body_fru": torch.zeros(batch_size, 3, 3),
        "current_ground_plane_z_body_m": torch.zeros(batch_size),
        "next_camera_origin_body_m": torch.ones(batch_size, 3),
        "next_camera_basis_body_fru": torch.ones(batch_size, 3, 3),
        "next_ground_plane_z_body_m": torch.ones(batch_size),
    }
    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            SharedTrainingPairV5=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
    )
    model = Model()
    pair = module._camera_pair(runtime, model, {"forward": forward})

    assert len(model.calls) == 2
    assert model.calls[0] == (
        forward["current_image"],
        forward["current_camera_origin_body_m"],
        forward["current_camera_basis_body_fru"],
        forward["current_ground_plane_z_body_m"],
    )
    assert model.calls[1] == (
        forward["next_image"],
        forward["next_camera_origin_body_m"],
        forward["next_camera_basis_body_fru"],
        forward["next_ground_plane_z_body_m"],
    )
    assert pair.current is current_output
    assert pair.next is next_output
    assert pair.jepa is None


def test_runner_accepts_exact_production_migration_and_partition() -> None:
    module = _load_runner("_overlap_production_migration")
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4Model,
    )
    from lewm.models import (
        shared_observable_camera_ray_jepa_v5 as shared,
    )
    from lewm.models import (
        shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1
        as overlap,
    )

    fit = ObservableCameraRayEvidenceV4Model()
    caller_rng = torch.random.get_rng_state().clone()
    model, head, encoder, frozen, partition = module._prepare_model(
        SimpleNamespace(torch=torch, model_module=shared),
        overlap,
        fit,
        torch.device("cpu"),
    )

    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert sum(parameter.numel() for parameter in head) == (
        module.contract.EXPECTED_PARAMETER_COUNTS["evidence_head"]
    )
    assert sum(parameter.numel() for parameter in encoder) == (
        module.contract.EXPECTED_PARAMETER_COUNTS["encoder"]
    )
    assert len(head) == (
        module.contract.EXPECTED_PARAMETER_TENSOR_COUNTS["evidence_head"]
    )
    assert len(encoder) == (
        module.contract.EXPECTED_PARAMETER_TENSOR_COUNTS["encoder"]
    )
    assert frozen and not any(
        parameter.requires_grad for parameter in frozen
    )
    assert model.model_family == module.contract.MODEL_FAMILY
    assert partition["migration"]["exact_copy_state_entry_count"] == 83
    assert partition["migration"]["transformed_state_entry_count"] == 1
    assert partition["migration"][
        "retained_n320_derived_entry_count"
    ] == 84
    assert partition["architecture_contract"] == (
        module.contract.overlapping_tokenization_architecture_contract_v1()
    )
    assert partition["architecture_contract_sha256"] == (
        module.contract.ARCHITECTURE_CONTRACT_SHA256
    )
    assert (
        partition.inherited_selection_sentinel_neutralized_count == 0
    )
    assert module._INHERITED_SELECTION_SENTINEL not in partition
    partition[module._INHERITED_SELECTION_SENTINEL] = None
    serialized = module.contract.canonical_json_bytes({
        "partition": partition
    })
    assert b"temporal_population" not in serialized
    assert b"history_valid" not in serialized
    assert b"relative_se2" not in serialized


def test_synthetic_prepare_forward_evaluate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_chained_source_gate")
    from lewm.models.observable_camera_ray_evidence_v4 import (
        IMAGE_SIZE,
        ObservableCameraRayEvidenceV4Model,
    )
    from lewm.models import (
        shared_observable_camera_ray_jepa_v5 as shared,
    )
    from lewm.models import (
        shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1
        as overlap,
    )

    runtime = SimpleNamespace(torch=torch, model_module=shared)
    model, _head, _encoder, _frozen, _partition = (
        module._prepare_model(
            runtime,
            overlap,
            ObservableCameraRayEvidenceV4Model(),
            torch.device("cpu"),
        )
    )
    origin = torch.tensor([[0.326, 0.02, 0.043]])
    basis = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    ground = torch.tensor([-0.35])
    with torch.no_grad():
        pair = module._camera_pair(runtime, model, {"forward": {
            "current_image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "next_image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "current_camera_origin_body_m": origin,
            "current_camera_basis_body_fru": basis,
            "current_ground_plane_z_body_m": ground,
            "next_camera_origin_body_m": origin.clone(),
            "next_camera_basis_body_fru": basis.clone(),
            "next_ground_plane_z_body_m": ground.clone(),
        }})
    assert pair.current.evidence.pixel_first_hit_hazard_logits.shape[0] == 1
    assert pair.next.evidence.ground_clear_to_target_logits.shape[0] == 1
    assert pair.jepa is None

    physical = {scope: {} for scope in module.contract.SCOPES}
    evaluation = {
        "scope_evaluations": {
            scope: {
                "physical_margins": [-1.0] * 21,
                "passes": False,
            }
            for scope in module.contract.SCOPES
        },
        "complete_physical_scope_count": 0,
        "margin_count": 189,
        "passed_margin_count": 0,
        "total_shortfall": 189.0,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": 0.5,
            "ground_balanced_accuracy": 0.5,
            "depth_p95_m": 1.0,
        },
    }
    monkeypatch.setattr(
        module.contract,
        "evaluate_physical_scopes",
        lambda observed: evaluation if observed is physical else None,
    )
    trainer = SimpleNamespace(
        physical_metrics=lambda observed, _pairs, _device, **_kwargs: (
            physical,
            1.0,
        ) if observed is model else None
    )
    frozen_sha256 = module._BASE._subset_sha(
        runtime, model, module.contract.FROZEN_STATE_PREFIXES
    )
    metric = module._evaluate(
        runtime,
        trainer,
        model,
        ({"dataset_role": "checkpoint_selection"},),
        torch.device("cpu"),
        update=100,
        frozen_sha256=frozen_sha256,
    )
    assert module.contract.validate_provisional_metric(
        metric, update=100
    ) == metric


def test_inherited_selection_sentinel_is_validated_and_not_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_selection_sentinel")
    assert module._BASE._selection_temporal_index is (
        module._neutralize_inherited_selection_sentinel
    )
    role_counts = dict(module.contract.SELECTION_ROLE_COUNTS)
    role_counts.update({"pairs": 1, "unique_endpoints": 2})
    monkeypatch.setattr(module.contract, "SELECTION_ROLE_COUNTS", role_counts)
    observed = module._neutralize_inherited_selection_sentinel([
        {
            "dataset_role": "checkpoint_selection",
            "current_endpoint_sha256": "a",
            "next_endpoint_sha256": "b",
            "primitive": "yaw_left",
            "relative_se2_current_frame": [1.0, 2.0, 3.0],
        }
    ])
    assert observed == ((), {}, None)

    partition = module._StaticPartition({"static": True})
    partition[module._INHERITED_SELECTION_SENTINEL] = observed[2]
    assert partition == {"static": True}
    assert partition.inherited_selection_sentinel_neutralized_count == 1
    assert module._INHERITED_SELECTION_SENTINEL not in partition
    with pytest.raises(PermissionError, match="more than once"):
        partition[module._INHERITED_SELECTION_SENTINEL] = None


@pytest.mark.parametrize(
    "payload",
    [
        {"history_valid": True},
        {"nominal_delta_current_frame": [0.0, 0.0, 0.0]},
        {"nested": {"relative_se2_current_frame": [0.0, 0.0, 0.0]}},
        {"temporal_population": {}},
        {"warm_scopes_informational_only": {}},
    ],
)
def test_static_payload_guard_rejects_dynamic_condition_fields(payload) -> None:
    module = _load_runner("_overlap_static_payload_guard")
    with pytest.raises(PermissionError, match="retained dynamic field"):
        module._assert_static_payload(payload, name="synthetic payload")


def test_train_wrapper_replaces_inherited_controls_without_dynamic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_train_wrapper")
    partition = module._StaticPartition({"static": True})
    partition[module._INHERITED_SELECTION_SENTINEL] = None
    metrics = [
        {
            "update": update,
            "evaluation": {"synthetic": update},
        }
        for update in module.contract.CHECKPOINT_UPDATES
    ]
    def inherited_train(*_args):
        assert module._BASE.contract.CONTROL_PASS == (
            module._FIXED_TRAINING_COMPLETE
        )
        assert module._BASE.contract.CONTROL_FAIL == (
            module._FIXED_TRAINING_COMPLETE
        )
        return {
            "metrics": metrics,
            "controls": [
                module._inherited_fixed_flow_control(update)
                for update in module.contract.CHECKPOINT_UPDATES
            ],
            "terminal_control": {
                "action": module._FIXED_TRAINING_COMPLETE,
            },
        }

    monkeypatch.setattr(module, "_TEMPORAL_TRAIN", inherited_train)

    result = module._train(
        None,
        None,
        None,
        (),
        (),
        (),
        (),
        (),
        (),
        None,
        Path("."),
        partition,
        None,
    )
    assert result["controls"] == [
        module.contract.provisional_checkpoint_control(update)
        for update in module.contract.CHECKPOINT_UPDATES
    ]
    assert isinstance(
        result["terminal_control"], module._DeferredTerminalControl
    )
    assert module._INHERITED_SELECTION_SENTINEL not in partition
    assert module._BASE.contract is module.contract


def test_train_wrapper_rejects_preledger_pass_or_fail_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_train_wrapper_rejects_control")
    partition = module._StaticPartition({"static": True})
    partition[module._INHERITED_SELECTION_SENTINEL] = None
    metrics = [
        {"update": update, "evaluation": {"synthetic": update}}
        for update in module.contract.CHECKPOINT_UPDATES
    ]
    inherited = {
        "metrics": metrics,
        "controls": [
            {"action": module.contract.CONTROL_CONTINUE},
            {"action": module.contract.CONTROL_CONTINUE},
            {"action": module.contract.CONTROL_PASS},
        ],
        "terminal_control": {"action": module.contract.CONTROL_PASS},
    }
    monkeypatch.setattr(module, "_TEMPORAL_TRAIN", lambda *_args: inherited)

    with pytest.raises(
        PermissionError, match="preledger checkpoint evidence changed"
    ):
        module._train(
            None,
            None,
            None,
            (),
            (),
            (),
            (),
            (),
            (),
            None,
            Path("."),
            partition,
            None,
        )
    assert module._BASE.contract is module.contract
