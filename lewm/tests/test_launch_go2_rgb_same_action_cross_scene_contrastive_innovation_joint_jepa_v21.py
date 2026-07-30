from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = ROOT / (
    "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORTED_BEFORE = set(sys.modules)
launcher = _load("_test_v21_scene_innovation_launcher", LAUNCHER_PATH)
_IMPORTED_BY_LAUNCHER = set(sys.modules) - _IMPORTED_BEFORE


class _FakeTensor:
    def __init__(
        self,
        values=(),
        *,
        shape: tuple[int, ...],
        dtype: object,
        device: str,
    ) -> None:
        self.values = tuple(values)
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _FakeTorch:
    Tensor = _FakeTensor
    int64 = object()

    @classmethod
    def tensor(cls, values, *, dtype, device):
        return _FakeTensor(
            values,
            shape=(len(tuple(values)),),
            dtype=dtype,
            device=device,
        )


def _training_module(*inherited_keys: str) -> SimpleNamespace:
    key = launcher.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
    return SimpleNamespace(
        SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21=key,
        REQUIRED_BATCH_KEYS=tuple(inherited_keys),
        REQUIRED_BATCH_KEYS_V21=(*inherited_keys, key),
        CURRENT_RGB_KEY=inherited_keys[0],
    )


def test_import_is_source_only_and_no_argument_cli_is_denied(capsys) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_LAUNCHER
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
            "joint_jepa_v21_launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_private_v20_adapter_selects_exact_fresh_v21_lifecycle() -> None:
    launcher._assert_configured_base_v21()
    base = launcher._BASE_LAUNCHER
    assert base.MODEL_MODULE_NAME.endswith(
        "joint_jepa_v18_object_space_height_volume"
    )
    assert base.TRAINING_MODULE_NAME == (
        "scripts.run_go2_rgb_same_action_cross_scene_contrastive_innovation_"
        "joint_jepa_v21"
    )
    assert base.EXECUTOR_MODULE_NAME == (
        "scripts.execute_go2_rgb_same_action_cross_scene_contrastive_"
        "innovation_joint_jepa_v21"
    )
    assert base.EXPERIMENT_ARM_NAME == (
        "same_action_cross_scene_contrastive_innovation_v21"
    )
    assert base.SOURCE_EVIDENCE_SCHEMA_PREFIX == (
        "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
        "joint_jepa_v21"
    )
    assert base.AUTHORITY_RELATIVE_PATH.endswith(
        "joint_jepa_v21_execution_authorization_2026-07-30.json"
    )
    assert base.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "joint_jepa_v21_source_manifest_2026-07-30.json"
    )
    assert base.SOURCE_REVIEW_RELATIVE_PATH.endswith(
        "joint_jepa_v21_source_review_2026-07-30.json"
    )
    assert base.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH.endswith(
        "joint_jepa_v21_clean_export_certification_2026-07-30.json"
    )
    assert base.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH.endswith(
        "joint_jepa_v21_source_closure.py"
    )
    assert base.MAXIMUM_UPDATES == 1_000
    assert base.MAXIMUM_PRESENTATIONS == 16_000
    assert base.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert base._v12_observation_v13 is launcher._v12_observation_v19
    assert base._build_one_microbatch_v13 is launcher._build_one_microbatch_v21


def test_frozen_preregistration_and_v20_result_are_bound() -> None:
    receipt = launcher.private_launcher_adapter_receipt_v21()
    assert receipt["base_launcher"] == {
        "path": (
            "scripts/launch_go2_rgb_object_space_height_volume_executed_"
            "successor_semantic_grounding_joint_jepa_v19.py"
        ),
        "commit": "04c383183dab586bb3395acfceaaa749e55ff3ce",
        "file_sha256": (
            "be51d8da0c7f564124afa0f9f647b8f0c28e821078adc14c315fca8b91d422da"
        ),
        "byte_count": 17_196,
    }
    assert receipt["preregistration"] == {
        "path": (
            "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_"
            "innovation_joint_jepa_v21_preregistration_2026-07-30.md"
        ),
        "commit": "c2bbce067175dd980c9ed2511dc14db5a222afe4",
        "file_sha256": (
            "f4ff1453e5cb63677dad66253d568c9204bd5504b3b3871e2b0c341402b1850e"
        ),
        "byte_count": 11_594,
    }
    assert receipt["predecessor_scientific_result"] == {
        "path": (
            "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json"
        ),
        "commit": "8321d76004aa1f3c87dfa04c3b18d701267a89ec",
        "file_sha256": (
            "d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956"
        ),
        "byte_count": 17_166,
        "content_sha256": (
            "37f683c1b2a5086c92d9cb081e9ba55b4fef4ed61f8cefea99fb0e5760e5cab2"
        ),
    }
    assert receipt["negative_row_key"] == "scene_innovation_negative_row"
    assert receipt["metadata_validated_before_inherited_tensor_construction"]
    assert receipt["full_schedule_metadata_preflight_before_first_tensor"]
    assert receipt["schedule_preflight_receipt_attribute"] == (
        "scene_innovation_negative_row_preflight_v21"
    )
    assert receipt["exact_one_field_batch_extension"]
    assert receipt["numeric_comparisons_retained_without_rescoring"]
    assert receipt["update100_new_terminal_branch"] is False
    assert receipt["update400_and_update1000_gates_inherited"] is True
    assert receipt["retry_authorized"] is False
    assert receipt["resume_authorized"] is False
    assert receipt["execution_authorized"] is False


@pytest.mark.parametrize(
    ("scene_ids", "expected"),
    [
        (("a", "b", "b", "a"), (1, 3, 3, 1)),
        (("a", "a", "b", "c"), (2, 2, 3, 0)),
        (("a", "b", "c", "d"), (1, 2, 3, 0)),
    ],
)
def test_first_cyclic_different_scene_mapping_is_exact(
    scene_ids: tuple[str, ...],
    expected: tuple[int, ...],
) -> None:
    observed = launcher.first_cyclic_different_scene_rows_v21(scene_ids)
    assert observed == expected
    assert all(index != row for row, index in enumerate(observed))
    assert all(scene_ids[index] != scene_ids[row] for row, index in enumerate(observed))


def test_negative_selection_rejects_malformed_or_all_one_scene() -> None:
    with pytest.raises(PermissionError, match="no different-scene"):
        launcher.first_cyclic_different_scene_rows_v21(("a",) * 4)
    for malformed in (
        ("a", "b", "c"),
        ("a", "b", "c", ""),
        ("a", "b", "c", 4),
    ):
        with pytest.raises(ValueError, match="four nonempty strings"):
            launcher.first_cyclic_different_scene_rows_v21(malformed)


def test_selected_train_metadata_drives_local_row_indices() -> None:
    runtime = SimpleNamespace(
        pairs={
            "train": [
                {"scene_id": "a", "unused": object()},
                {"scene_id": "b"},
                {"scene_id": "unused"},
                {"scene_id": "b"},
                {"scene_id": "a"},
            ]
        }
    )
    # Selected local scene order is a,b,b,a; returned values are local rows,
    # not global schedule indices.
    assert launcher.negative_rows_from_train_metadata_v21(
        runtime, (4, 1, 3, 0)
    ) == (1, 3, 3, 1)
    with pytest.raises(PermissionError, match="escaped"):
        launcher.negative_rows_from_train_metadata_v21(runtime, (0, 1, 3, 99))
    with pytest.raises(PermissionError, match="nonnegative integers"):
        launcher.negative_rows_from_train_metadata_v21(runtime, (0, 1, 3, True))


def test_full_schedule_preflight_is_exact_and_cached(monkeypatch) -> None:
    runtime = SimpleNamespace(
        schedule=(0, 1, 2, 3) * 4_000,
        pairs={
            "train": [
                {"scene_id": "a"},
                {"scene_id": "b"},
                {"scene_id": "b"},
                {"scene_id": "a"},
            ]
        },
    )
    original = launcher.negative_rows_from_train_metadata_v21
    calls = 0

    def counted(runtime_value, indices):
        nonlocal calls
        calls += 1
        return original(runtime_value, indices)

    monkeypatch.setattr(
        launcher, "negative_rows_from_train_metadata_v21", counted
    )
    receipt = launcher.preflight_schedule_negative_rows_v21(runtime)
    assert receipt == {
        "schema": (
            "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
            "joint_jepa_v21_schedule_preflight_v1"
        ),
        "schedule_index_count": 16_000,
        "microbatch_count": 4_000,
        "negative_row_index_count": 16_000,
        "negative_row_mapping_sha256": (
            "b0a9f0b5e475d79bd12431068bc1338399c583164747cc9880be3bc048d8159a"
        ),
        "all_rows_nonself_different_scene": True,
        "passed": True,
    }
    assert calls == 4_000
    assert launcher.preflight_schedule_negative_rows_v21(runtime) is receipt
    assert calls == 4_000


def test_bad_late_schedule_chunk_fails_before_any_tensor_build(monkeypatch) -> None:
    schedule = list((0, 1, 2, 3) * 4_000)
    schedule[-4:] = (0, 0, 0, 0)
    called = False

    def forbidden_builder(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("tensor builder must not run")

    monkeypatch.setattr(
        launcher, "_ORIGINAL_BUILD_ONE_MICROBATCH_V13", forbidden_builder
    )
    runtime = SimpleNamespace(
        schedule=tuple(schedule),
        pairs={
            "train": [
                {"scene_id": "a"},
                {"scene_id": "b"},
                {"scene_id": "c"},
                {"scene_id": "d"},
            ]
        },
    )
    with pytest.raises(PermissionError, match="no different-scene"):
        launcher._build_one_microbatch_v21(
            runtime=runtime,
            indices=(0, 1, 2, 3),
            stage="train_update_0001_microbatch_0",
        )
    assert called is False
    assert not hasattr(
        runtime, launcher.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21
    )


def test_metadata_failure_precedes_inherited_tensor_construction(monkeypatch) -> None:
    called = False

    def forbidden_builder(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("tensor builder must not run")

    monkeypatch.setattr(
        launcher, "_ORIGINAL_BUILD_ONE_MICROBATCH_V13", forbidden_builder
    )
    runtime = SimpleNamespace(
        schedule=(0, 1, 2, 3) * 4_000,
        pairs={"train": [{"scene_id": "same"} for _ in range(4)]},
    )
    with pytest.raises(PermissionError, match="no different-scene"):
        launcher._build_one_microbatch_v21(
            runtime=runtime,
            indices=(0, 1, 2, 3),
            stage="train_update_0001_microbatch_0",
        )
    assert called is False


def test_batch_adapter_adds_exact_int64_device_tensor_after_inherited_build(
    monkeypatch,
) -> None:
    selected_before_build = False
    original_negative = launcher.negative_rows_from_train_metadata_v21

    def select(runtime, indices):
        nonlocal selected_before_build
        selected_before_build = True
        return original_negative(runtime, indices)

    def inherited_builder(*, runtime, indices, stage):
        assert selected_before_build
        assert indices == (0, 1, 2, 3)
        assert stage == "train_update_0001_microbatch_0"
        return {
            "current_rgb": _FakeTensor(
                shape=(4, 3, 2, 2), dtype=object(), device="cpu"
            ),
            "inherited": _FakeTensor(shape=(4,), dtype=object(), device="cpu"),
        }

    monkeypatch.setattr(launcher, "negative_rows_from_train_metadata_v21", select)
    monkeypatch.setattr(
        launcher, "_ORIGINAL_BUILD_ONE_MICROBATCH_V13", inherited_builder
    )
    runtime = SimpleNamespace(
        torch=_FakeTorch,
        training_module=_training_module("current_rgb", "inherited"),
        schedule=(0, 1, 2, 3) * 4_000,
        pairs={
            "train": [
                {"scene_id": "a"},
                {"scene_id": "b"},
                {"scene_id": "b"},
                {"scene_id": "a"},
            ]
        },
    )
    result = launcher._build_one_microbatch_v21(
        runtime=runtime,
        indices=(0, 1, 2, 3),
        stage="train_update_0001_microbatch_0",
    )
    assert tuple(result) == (
        "current_rgb",
        "inherited",
        "scene_innovation_negative_row",
    )
    negative = result["scene_innovation_negative_row"]
    assert tuple(negative.shape) == (4,)
    assert negative.dtype is _FakeTorch.int64
    assert negative.device == result["current_rgb"].device
    assert negative.values == (1, 3, 3, 1)


def test_batch_adapter_rejects_any_schema_beyond_one_suffix(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher, "preflight_schedule_negative_rows_v21", lambda runtime: {}
    )
    monkeypatch.setattr(
        launcher,
        "negative_rows_from_train_metadata_v21",
        lambda runtime, indices: (1, 0, 3, 2),
    )
    training = _training_module("current_rgb")
    training.REQUIRED_BATCH_KEYS_V21 = (
        "current_rgb",
        "extra",
        "scene_innovation_negative_row",
    )
    runtime = SimpleNamespace(torch=_FakeTorch, training_module=training)
    with pytest.raises(RuntimeError, match="exact one-field extension"):
        launcher._build_one_microbatch_v21(
            runtime=runtime,
            indices=(0, 1, 2, 3),
            stage="unused",
        )


def test_selector_or_hook_mutation_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "MODEL_MODULE_NAME",
        "lewm.models.unreviewed",
    )
    with pytest.raises(PermissionError, match="changed after import"):
        launcher._assert_configured_base_v21()


def test_terminal_exit_codes_preserve_inherited_gate_surface(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v21", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v21",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v21",
        lambda **kwargs: {"status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 2


def test_isolated_import_does_not_load_runtime_or_touch_payload() -> None:
    code = (
        "import runpy,sys;"
        f"ns=runpy.run_path({str(LAUNCHER_PATH)!r});"
        "assert ns['main']([])==4;"
        "assert 'torch' not in sys.modules;"
        "assert 'numpy' not in sys.modules;"
        "assert 'PIL' not in sys.modules"
    )
    environment = dict(os.environ)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "ZE_AFFINITY_MASK",
    ):
        environment.pop(name, None)
    environment["HIP_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
