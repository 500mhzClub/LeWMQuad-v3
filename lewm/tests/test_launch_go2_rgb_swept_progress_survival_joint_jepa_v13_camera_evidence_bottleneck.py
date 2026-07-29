from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = (
    ROOT
    / "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
_IMPORTED_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_v13_camera_evidence_bottleneck_launcher",
    LAUNCHER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)
_IMPORTED_BY_LAUNCHER = set(sys.modules) - _IMPORTED_BEFORE


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_import_and_no_argument_cli_are_source_only_and_denied(capsys: object) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_LAUNCHER
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    value = json.loads(capsys.readouterr().out)
    assert value == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
            "launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_isolated_interpreter_can_import_from_certified_source_root() -> None:
    code = (
        "import importlib,runpy,sys;"
        f"ns=runpy.run_path({str(LAUNCHER_PATH)!r});"
        "root=ns['_activate_certified_source_root_v13'](ns['ROOT']);"
        "assert sys.path[0]==str(root);"
        "importlib.import_module('scripts.execute_go2_rgb_swept_progress_survival_"
        "joint_jepa_v13_camera_evidence_bottleneck');"
        "assert 'torch' not in sys.modules"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    no_bytecode_guard = (
        "import runpy;"
        f"ns=runpy.run_path({str(LAUNCHER_PATH)!r});"
        "\ntry: ns['_activate_certified_source_root_v13'](ns['ROOT'])"
        "\nexcept PermissionError: pass"
        "\nelse: raise AssertionError('python -B was not required')"
    )
    without_b = subprocess.run(
        [sys.executable, "-I", "-c", no_bytecode_guard],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert without_b.returncode == 0, without_b.stderr


def test_authority_loader_accepts_only_fixed_canonical_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    expected = tmp_path / launcher.AUTHORITY_RELATIVE_PATH
    expected.parent.mkdir(parents=True)
    value = {"schema": "test", "status": "not_authority"}
    expected.write_bytes(_canonical(value) + b"\n")
    assert launcher._load_authority_file_v13(expected) == value

    external = tmp_path / "external.json"
    external.write_bytes(_canonical(value) + b"\n")
    with pytest.raises(PermissionError):
        launcher._load_authority_file_v13(external)

    expected.unlink()
    expected.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        launcher._load_authority_file_v13(expected)
    expected.unlink()
    expected.symlink_to(external)
    with pytest.raises(PermissionError):
        launcher._load_authority_file_v13(expected)


def test_source_evidence_rejects_a_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    value = {"schema": "evidence"}
    core = dict(value)
    value["content_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    (external / "evidence.json").write_bytes(_canonical(value) + b"\n")
    (source / "docs").symlink_to(external, target_is_directory=True)
    with pytest.raises(PermissionError):
        launcher._read_content_bound_evidence_v13(
            source,
            "docs/evidence.json",
            name="synthetic source evidence",
        )

def test_runtime_binding_requires_bound_content_digest_when_expected() -> None:
    base = {
        "path": "runtime/input.json",
        "file_sha256": "1" * 64,
        "byte_count": 10,
    }
    with pytest.raises(PermissionError):
        launcher._require_runtime_binding_v13(
            base,
            expected_path=base["path"],
            expected_sha256=base["file_sha256"],
            expected_byte_count=base["byte_count"],
            expected_content_sha256="2" * 64,
            name="input",
        )
    complete = {**base, "content_sha256": "2" * 64}
    assert launcher._require_runtime_binding_v13(
        complete,
        expected_path=base["path"],
        expected_sha256=base["file_sha256"],
        expected_byte_count=base["byte_count"],
        expected_content_sha256="2" * 64,
        name="input",
    ) == complete


def test_structural_probe_skips_duplicate_rgb_commitments() -> None:
    endpoints = {
        "a": {
            "dataset_role": "checkpoint_selection",
            "image_sha256_commitment_only": "1" * 64,
        },
        "b": {
            "dataset_role": "checkpoint_selection",
            "image_sha256_commitment_only": "1" * 64,
        },
        "c": {
            "dataset_role": "checkpoint_selection",
            "image_sha256_commitment_only": "2" * 64,
        },
    }
    assert launcher._select_distinct_structural_probe_v13(
        endpoints,
        {"a": "b", "b": "c", "c": "a"},
    ) == ("b", "c")


def test_runtime_data_root_is_canonical_and_distinct(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    authority = {"runtime_data_root": str(data)}
    assert launcher._validate_runtime_data_root_v13(source, authority) == data
    with pytest.raises(PermissionError):
        launcher._validate_runtime_data_root_v13(
            source, {"runtime_data_root": str(source)}
        )
    nested_data = source / "runtime-data"
    nested_data.mkdir()
    with pytest.raises(PermissionError):
        launcher._validate_runtime_data_root_v13(
            source, {"runtime_data_root": str(nested_data)}
        )
    outer_data = tmp_path / "outer-data"
    nested_source = outer_data / "source"
    nested_source.mkdir(parents=True)
    with pytest.raises(PermissionError):
        launcher._validate_runtime_data_root_v13(
            nested_source, {"runtime_data_root": str(outer_data)}
        )
    link = tmp_path / "data-link"
    link.symlink_to(data, target_is_directory=True)
    with pytest.raises(PermissionError):
        launcher._validate_runtime_data_root_v13(
            source, {"runtime_data_root": str(link)}
        )


def test_runtime_file_rejects_a_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    external = tmp_path / "external"
    data.mkdir()
    external.mkdir()
    (external / "frame.png").write_bytes(b"synthetic")
    (data / "images").symlink_to(external, target_is_directory=True)
    with pytest.raises(PermissionError):
        launcher._require_contained_regular_file_v13(
            data,
            data / "images/frame.png",
            name="synthetic RGB",
        )


def test_certified_source_root_rejects_an_unbound_or_git_checkout(
    tmp_path: Path,
) -> None:
    certified = tmp_path / "certified"
    other = tmp_path / "same-bytes-elsewhere"
    certified.mkdir()
    other.mkdir()
    authority = {"certified_source_root": str(certified)}
    assert (
        launcher._validate_certified_source_root_v13(certified, authority)
        == certified
    )
    with pytest.raises(PermissionError):
        launcher._validate_certified_source_root_v13(other, authority)
    (certified / ".git").mkdir()
    with pytest.raises(PermissionError):
        launcher._validate_certified_source_root_v13(certified, authority)


class _DummyRawInputs:
    def __init__(self) -> None:
        self.rehash_count = 0
        self.consumed = {
            "authority": {"roles": ["authority"]},
            "index": {"roles": ["index"]},
            "train": {"roles": ["train"]},
            "selection": {"roles": ["checkpoint_selection"]},
        }

    def rehash_consumed(self) -> dict[str, object]:
        self.rehash_count += 1
        return {
            "unique_file_count": 4,
            "all_consumed_files_rehashed": True,
            "records_sha256": "3" * 64,
            "records": [],
        }


class _DummyLoader:
    def receipt(self) -> dict[str, object]:
        return {"forbidden_semantic_counters": {"attempt_count": 0}}


def _runtime_for_access_test(tmp_path: Path) -> object:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    records = []
    for name in ("manifest.json", "train.jsonl", "selection.jsonl"):
        raw = name.encode("ascii")
        (data / name).write_bytes(raw)
        records.append(
            {
                "path": name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    runtime = object.__new__(launcher.V13ComposedRuntime)
    runtime.repository_root = source
    runtime.runtime_data_root = data
    runtime.raw_inputs = _DummyRawInputs()
    runtime.loader = _DummyLoader()
    runtime.label_access = {
        "manifest": records[0],
        "opened_roles": ["train", "checkpoint_selection"],
        "opened_role_files": records[1:],
    }
    runtime.runtime_fingerprint = {
        "executable": "synthetic",
        "python": "synthetic",
        "torch": "synthetic",
        "torch_hip": "synthetic",
        "numpy": "synthetic",
        "pillow": "synthetic",
    }
    runtime.executor_api = SimpleNamespace(
        validate_bound_sources_v13=lambda root: {
            "schema": "test",
            "validated_paths": ["one", "two"],
            "validated_path_count": 2,
            "execution_authority_granted": False,
        }
    )
    source_bindings = []
    for name in ("model.py", "launcher.py"):
        raw = name.encode("ascii")
        (source / name).write_bytes(raw)
        source_bindings.append(
            {
                "path": name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    runtime.source_evidence = {
        "certified_export_binding_count": len(source_bindings),
        "certified_source_bindings_sha256": hashlib.sha256(
            _canonical(source_bindings)
        ).hexdigest(),
        "certified_source_bindings": source_bindings,
    }
    runtime._access_consumed_count = -1
    runtime._access_opened_roles = ()
    runtime._terminal_rehash_started = False
    runtime._terminal_access_receipt = None
    return runtime


def test_per_update_access_is_in_memory_and_terminal_rehash_is_exactly_once(
    tmp_path: Path,
) -> None:
    runtime = _runtime_for_access_test(tmp_path)
    lightweight = runtime.access_receipt_v13()
    assert lightweight == {
        "forbidden_input_count": 0,
        "probability_calibration_open_count": 0,
        "opened_roles": [
            "authority",
            "index",
            "train",
            "checkpoint_selection",
        ],
    }
    assert runtime.raw_inputs.rehash_count == 0
    terminal = runtime.terminal_access_receipt_v13()
    assert terminal["terminal_full_rehash_count"] == 1
    assert terminal["raw_consumed_file_rehash_count"] == 4
    assert terminal["label_source_rehash_count"] == 3
    assert terminal["certified_source_rehash_count"] == 2
    assert terminal["certified_source_bindings"] == runtime.source_evidence[
        "certified_source_bindings"
    ]
    assert terminal["all_consumed_inputs_rehashed"] is True
    assert runtime.terminal_access_receipt_v13() == terminal
    assert runtime.raw_inputs.rehash_count == 1


def test_physical_provenance_summary_is_exact() -> None:
    families = (
        "large_enclosed_maze",
        "local_composite_motifs",
        "loop_alias_stress",
        "medium_enclosed_maze",
        "open_obstacle_field",
        "rough_local_dynamics",
        "small_enclosed_maze",
        "visual_sensor_stress",
    )
    template = {
        "model_entrypoint": "encode_online_with_evidence",
        "learned_evidence_source": "encoding.nominal_evidence",
        "semantic_probability_source": (
            "softmax(model.semantic_logits_from_latent(encoding.latent),dim=1)"
        ),
        "target_metadata_only": (
            "ground_query_in_frustum",
            "ground_target_distance_m",
        ),
        "auxiliary_logits_used": False,
        "old_camera_raster_used": False,
        "batch_size": 4,
    }
    rows = [
        {**template, "family": families[index % len(families)], "arm": arm}
        for arm in ("matched", "wrong_rgb")
        for index in range(231)
    ]
    result = launcher._summarize_physical_provenance_v13(
        rows,
        {"aggregate": {"wrong_rgb_pixel_balanced_accuracy_drop": 0.01}},
        registered_families=families,
    )
    assert result == {
        "target_endpoint_count": 924,
        "matched_nominal_call_count": 924,
        "wrong_nominal_call_count": 924,
        "qualifying_updater_call_count": 1_848,
        "qualifying_updater_name": "update_physical_accumulator_from_rgb_v13",
        "auxiliary_logits_used": False,
        "old_camera_raster_used": False,
        "target_query_identity_pass": True,
        "wrong_rgb_dependence_nonzero": True,
    }


def test_cli_exit_codes_distinguish_success_and_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v13", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v13",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v13",
        lambda **kwargs: {"status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 2
    with pytest.raises(SystemExit):
        launcher._parser().parse_args(
            [
                "--future-authority",
                "authority.json",
                "--created-utc",
                "2026-07-29T00:00:00Z",
            ]
        )


def test_reservation_precedes_deferred_runtime_and_no_legacy_runtime_is_used() -> None:
    execute_source = inspect.getsource(launcher.execute_future_authorized_v13)
    assert execute_source.index("_validate_source_evidence_v13") < execute_source.index(
        "reserve_attempt_v13"
    )
    assert execute_source.index("reserve_attempt_v13") < execute_source.index(
        "compose_runtime_v13"
    )
    compose_source = inspect.getsource(launcher.compose_runtime_v13)
    assert 'importlib.import_module("torch")' in compose_source
    assert "_load_runtime" not in compose_source
    assert ".Trainer(" not in compose_source
    assert "soft_rasterize=" not in compose_source
    assert "loss_adapter=" not in compose_source
    assert "direct._normalize_endpoint_paths" not in compose_source
    assert "_validate_runtime_payload_containment_v13" in compose_source
