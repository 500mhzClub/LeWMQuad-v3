from __future__ import annotations

import builtins
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest import mock

import pytest
import torch

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_temporal_v1 import (
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    SharedObservableCameraRayJepaV5MultiresTemporalV1,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_causal_temporal_perception_v1.py"
LAUNCHER_PATH = (
    ROOT / "scripts/launch_go2_rgb_causal_temporal_perception_v1.py"
)


def _load_runner(name: str = "_temporal_v1_runner_test"):
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


def _selection_rows(module) -> list[dict[str, object]]:
    # current={0..494}, next={429..923}: union=924 and overlap=66.
    families = module.contract.FAMILIES
    family_by_component = {
        index: families[index % len(families)] for index in range(429)
    }
    rows: list[dict[str, object]] = []
    for index in range(495):
        component = index if index < 429 else index - 429
        family = family_by_component[component]
        family_index = families.index(family)
        core: dict[str, object] = {
            "schema": "lewm_go2_shared_jepa_v5_raw_supervision_pair_v1",
            "dataset_role": "checkpoint_selection",
            "global_row": index,
            "scene_id": f"selection-scene-{family_index}",
            "family": family,
            "current_endpoint_sha256": f"{index:064x}",
            "next_endpoint_sha256": f"{429 + index:064x}",
            "episode_id": "episode-1",
            "env_index": 0,
            "reset_count": 1,
            "source_split": "development",
            "frames_jsonl_sha256": hashlib.sha256(
                f"frames-{family}".encode()
            ).hexdigest(),
            "scene_manifest_sha256": hashlib.sha256(
                f"manifest-{family}".encode()
            ).hexdigest(),
            "primitive": "forward_medium",
            "relative_se2_current_frame": [0.1, 0.0, 0.0],
            "label_shard_path_metadata_only": (
                f"labels/{family}/labels.npz"
            ),
            "label_shard_sha256": hashlib.sha256(
                f"labels-{family}".encode()
            ).hexdigest(),
            "label_shard_row": index,
            "sidecar_row_identity_sha256": hashlib.sha256(
                f"sidecar-{index}".encode()
            ).hexdigest(),
        }
        rows.append({
            **core,
            "content_sha256": module.contract.canonical_json_sha256(core),
        })
    return rows


def _rehash_pair(module, row: dict[str, object]) -> None:
    core = dict(row)
    core.pop("content_sha256", None)
    row["content_sha256"] = module.contract.canonical_json_sha256(core)


def test_runner_import_defers_tensor_and_payload_stacks() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"torch", "numpy", "PIL"}:
            raise AssertionError(f"source-only runner imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        module = _load_runner("_temporal_v1_source_only_import")
    assert module.PREFLIGHT_ENVIRONMENT_KEY == (
        "LEWM_CAUSAL_TEMPORAL_PERCEPTION_V1_PREFLIGHT_JSON"
    )


def test_temporal_selection_index_binds_exact_warm_cold_population() -> None:
    module = _load_runner("_temporal_v1_population")
    ids_by_family, predecessor_by_target, receipt = (
        module._selection_temporal_index(_selection_rows(module))
    )
    assert receipt == {
        "pairs": 495,
        "unique_endpoints": 924,
        "warm_endpoints": 495,
        "cold_endpoints": 429,
        "both_roles": 66,
        "ambiguous_predecessors": 0,
        "scenes": 8,
        "mapping_sha256":
            "6e794fbf88e340151a26aa8bb4696f3bd5f283a9df062dd4ae860de3"
            "57aa58ce",
        "fixed_lag_seconds": 0.5,
        "fixed_lag_ticks": 5,
        "reset_safe": True,
        "pair_content_identity_verified": True,
        "connected_endpoint_stream_consistency_verified": True,
    }
    assert len(predecessor_by_target) == 495
    assert sum(map(len, ids_by_family.values())) == 924
    assert all(len(ids) >= 2 for ids in ids_by_family.values())


def test_temporal_selection_index_rejects_ambiguous_predecessor() -> None:
    module = _load_runner("_temporal_v1_ambiguous")
    rows = _selection_rows(module)
    duplicate = dict(rows[0])
    duplicate["current_endpoint_sha256"] = "f" * 64
    duplicate["global_row"] = 495
    duplicate["label_shard_row"] = 495
    duplicate["sidecar_row_identity_sha256"] = hashlib.sha256(
        b"sidecar-495"
    ).hexdigest()
    _rehash_pair(module, duplicate)
    rows.append(duplicate)
    with pytest.raises(PermissionError, match="multiple predecessors"):
        module._selection_temporal_index(rows)


def test_temporal_selection_index_rejects_irregular_lag_override() -> None:
    module = _load_runner("_temporal_v1_irregular_lag")
    rows = _selection_rows(module)
    rows[0]["transition_duration_s"] = 0.4
    _rehash_pair(module, rows[0])
    with pytest.raises(PermissionError, match="schema fields changed"):
        module._selection_temporal_index(rows)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("env_index", 2),
        ("episode_id", "episode-after-reset"),
        ("reset_count", 2),
        ("scene_id", "other-scene"),
        (
            "frames_jsonl_sha256",
            "f" * 64,
        ),
    ),
)
def test_temporal_selection_index_rejects_connected_stream_mismatch(
    field: str,
    value: object,
) -> None:
    module = _load_runner(f"_temporal_v1_stream_mismatch_{field}")
    rows = _selection_rows(module)
    # Endpoint 429 is the target of row 0 and the predecessor of row 429.
    rows[429][field] = value
    _rehash_pair(module, rows[429])
    with pytest.raises(
        PermissionError,
        match="crossed scene, episode, reset, or stream",
    ):
        module._selection_temporal_index(rows)


def test_temporal_selection_index_rejects_pair_content_mutation() -> None:
    module = _load_runner("_temporal_v1_pair_content")
    rows = _selection_rows(module)
    rows[0]["primitive"] = "yaw_left"
    with pytest.raises(PermissionError, match="content identity changed"):
        module._selection_temporal_index(rows)


def test_camera_pair_passes_only_rgb_and_calibration_to_temporal_model() -> None:
    module = _load_runner("_temporal_v1_camera_pair")
    batch_size = 2
    previous = SimpleNamespace(bev=torch.zeros(batch_size, 3, 4, 4))
    current = SimpleNamespace(bev=torch.ones(batch_size, 3, 4, 4))

    class Model:
        received = None

        def forward_camera_pair(self, **kwargs):
            self.received = kwargs
            return previous, current

    model = Model()
    pair_type = lambda **kwargs: SimpleNamespace(**kwargs)
    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(SharedTrainingPairV5=pair_type),
    )
    sentinel = object()
    forward = {
        "current_image": torch.zeros(batch_size, 3, 112, 112),
        "next_image": torch.ones(batch_size, 3, 112, 112),
        "current_camera_origin_body_m": torch.zeros(batch_size, 3),
        "current_camera_basis_body_fru": torch.zeros(batch_size, 3, 3),
        "current_ground_plane_z_body_m": torch.zeros(batch_size),
        "next_camera_origin_body_m": torch.ones(batch_size, 3),
        "next_camera_basis_body_fru": torch.ones(batch_size, 3, 3),
        "next_ground_plane_z_body_m": torch.ones(batch_size),
        "action": sentinel,
        "realized_delta_pose_current": sentinel,
        "commanded_delta_pose_current": sentinel,
    }
    pair = module._camera_pair(runtime, model, {"forward": forward})
    assert model.received is not None
    assert set(model.received) == {
        "previous_image",
        "current_image",
        "previous_camera_origin_body_m",
        "previous_camera_basis_body_fru",
        "previous_ground_plane_z_body_m",
        "current_camera_origin_body_m",
        "current_camera_basis_body_fru",
        "current_ground_plane_z_body_m",
    }
    assert all(
        value is not sentinel for value in model.received.values()
    )
    assert pair.current is previous
    assert pair.next is current
    assert pair.jepa is None
    assert pair.commanded_overlap_mask.dtype == torch.bool
    assert bool(pair.commanded_overlap_mask.all())


def test_visual_only_training_batch_never_loads_or_transfers_forbidden_inputs(
) -> None:
    module = _load_runner("_temporal_v1_visual_only_batch")
    forbidden = {
        "primitive",
        "relative_se2_current_frame",
        "action",
        "realized_delta_pose_current",
        "commanded_delta_pose_current",
        "diagnostic_wrong_action",
        "diagnostic_wrong_action_delta_pose_current",
        "diagnostic_wrong_commanded_delta_pose_current",
    }

    class Guarded(dict):
        def __getitem__(self, key):
            if key in forbidden:
                raise AssertionError(f"forbidden input was loaded: {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in forbidden:
                raise AssertionError(f"forbidden input was loaded: {key}")
            return super().get(key, default)

    forbidden_value = object()
    pairs = [
        Guarded({
            "dataset_role": "train",
            "current_endpoint_sha256": "current-0",
            "next_endpoint_sha256": "next-0",
            "primitive": forbidden_value,
            "relative_se2_current_frame": forbidden_value,
        }),
        Guarded({
            "dataset_role": "train",
            "current_endpoint_sha256": "current-1",
            "next_endpoint_sha256": "next-1",
            "primitive": forbidden_value,
            "relative_se2_current_frame": forbidden_value,
        }),
    ]

    def frame(endpoint_id, *, role, arm, stage):
        assert role == "train"
        assert arm == "causal_temporal_perception_v1"
        assert stage == "camera_gradient"
        offset = float(endpoint_id.endswith("1"))
        if endpoint_id.startswith("next"):
            offset += 2.0
        return Guarded({
            "image": torch.full((3, 4, 4), offset),
            "camera_origin": torch.full((3,), offset),
            "camera_basis": torch.full((3, 3), offset),
            "ground": torch.tensor(offset),
            "action": forbidden_value,
            "realized_delta_pose_current": forbidden_value,
            "commanded_delta_pose_current": forbidden_value,
        })

    supervision_calls = []

    def supervision(frames, device):
        supervision_calls.append(tuple(frames))
        assert device == torch.device("cpu")
        return SimpleNamespace(frame_count=len(frames))

    trainer = SimpleNamespace(
        r=SimpleNamespace(torch=torch),
        inputs=SimpleNamespace(frame=frame),
        supervision=supervision,
        batch=lambda *args, **kwargs: pytest.fail(
            "inherited action/SE2 batch builder was called"
        ),
        commanded_table=lambda *args, **kwargs: pytest.fail(
            "inherited commanded-SE2 table builder was called"
        ),
    )
    batch = module._visual_only_batch(
        trainer,
        pairs,
        [0, 1],
        torch.device("cpu"),
        role="train",
        arm="causal_temporal_perception_v1",
        stage="camera_gradient",
    )
    assert set(batch) == {
        "forward",
        "current_supervision",
        "next_supervision",
    }
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
    assert len(supervision_calls) == 2

    previous = SimpleNamespace(bev=torch.zeros(2, 3, 4, 4))
    current = SimpleNamespace(bev=torch.ones(2, 3, 4, 4))

    class Model:
        received = None

        def forward_camera_pair(self, **kwargs):
            self.received = kwargs
            return previous, current

    model = Model()
    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            SharedTrainingPairV5=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
    )
    module._camera_pair(runtime, model, batch)
    assert model.received is not None
    assert set(model.received) == {
        "previous_image",
        "current_image",
        "previous_camera_origin_body_m",
        "previous_camera_basis_body_fru",
        "previous_ground_plane_z_body_m",
        "current_camera_origin_body_m",
        "current_camera_basis_body_fru",
        "current_ground_plane_z_body_m",
    }


def test_train_and_evaluation_sources_have_no_inherited_action_se2_seam() -> None:
    module = _load_runner("_temporal_v1_no_action_se2_seam")
    training = inspect.getsource(module._train)
    evaluation = inspect.getsource(module._temporal_physical_metrics)
    execution = inspect.getsource(module._execute_after_reservation)
    forbidden_source_fragments = {
        "trainer.batch(",
        "trainer.commanded_table(",
        "relative_se2_current_frame",
        "realized_delta_pose_current",
        "commanded_delta_pose_current",
        "diagnostic_wrong_action",
        "wrong_commanded",
    }
    assert "_visual_only_batch(" in training
    for fragment in forbidden_source_fragments:
        assert fragment not in training
        assert fragment not in evaluation
        assert fragment not in execution


def test_runner_uses_only_the_generic_prior_runtime_output_receipt() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert source.count('"prior_runtime_output_open_count": 0') == 3
    assert '"v1_runtime_output_open_count"' not in source
    assert '"v2_runtime_output_open_count"' not in source
    assert "MODEL_FILE_SHA256" not in source
    assert "_load_post_reservation_stack(sources)" in source


def test_runner_accepts_the_real_temporal_n320_migration_receipt() -> None:
    module = _load_runner("_temporal_v1_receipt")
    fit = ObservableCameraRayEvidenceV4Model()
    model, receipt = (
        SharedObservableCameraRayJepaV5MultiresTemporalV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )
    )
    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            tensor_state_dict_sha256=tensor_state_dict_sha256
        ),
    )
    normalized = module._validate_migration_receipt(
        runtime,
        sys.modules[
            SharedObservableCameraRayJepaV5MultiresTemporalV1.__module__
        ],
        model,
        fit,
        receipt,
    )
    assert normalized["copied_state_entry_count"] == 84
    assert normalized["copied_temporal_entry_count"] == 0
    assert normalized["temporal_output_projection_exact_zero"] is True


@dataclass(frozen=True)
class _BatchValue:
    tensor: torch.Tensor
    label: str


def test_warm_metric_slicing_preserves_nonbatch_metadata() -> None:
    module = _load_runner("_temporal_v1_slice")
    value = _BatchValue(torch.arange(12).reshape(3, 4), "fixed")
    selected = module._slice_batch_dataclass(value, slice(1, 2))
    assert torch.equal(selected.tensor, value.tensor[1:2])
    assert selected.label == "fixed"


def test_launcher_and_runner_use_the_same_contract_and_preflight_key() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    contract_path = (
        "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
    )
    key = "LEWM_CAUSAL_TEMPORAL_PERCEPTION_V1_PREFLIGHT_JSON"
    assert contract_path in runner and contract_path in launcher
    assert key in runner and key in launcher
    assert "go2_shared_jepa_v5_multires_probe_v3.py" not in runner
    assert "go2_shared_jepa_v5_multires_probe_v3.py" not in launcher
