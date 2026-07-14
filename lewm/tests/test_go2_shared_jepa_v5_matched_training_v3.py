from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_go2_shared_jepa_v5_matched_training_v3.py"
_LOAD_COUNT = itertools.count()


def _fresh_runner():
    name = f"_lewm_matched_v3_test_runner_{next(_LOAD_COUNT)}"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def test_scope_hashes_science_and_production_cap() -> None:
    runner = _fresh_runner()
    contract = runner.contract
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.RETRY_AUTHORIZED is False
    assert contract.AUTOMATIC_V4_AUTHORIZED is False
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("/matched_training_v3")
    assert contract.science_contract() == {
        **contract._V2_SCIENCE_CONTRACT,
        "candidate": {
            **contract._V2_SCIENCE_CONTRACT["candidate"],
            "schema": contract.PRE_G2_CHECKPOINT_SCHEMA,
        },
    }
    for path, expected in contract.V2_SOURCE_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    sources = contract.current_source_bindings()
    assert set(sources) == set(contract.SOURCE_PATHS)
    production_lines = sum(
        len((ROOT / path).read_text(encoding="utf-8").splitlines())
        for path in (contract.CONTRACT_RELATIVE_PATH, contract.RUNNER_RELATIVE_PATH)
    )
    assert production_lines <= 400


def test_isolated_import_is_read_only_and_accelerator_free() -> None:
    output = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v3"
    before = output.exists()
    code = f"""
import importlib.util, json, sys
sys.dont_write_bytecode=True
p={str(RUNNER)!r}
s=importlib.util.spec_from_file_location('_isolated_v3',p)
m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
print(json.dumps(sorted(set(sys.modules) & {{'torch','numpy','PIL','cv2'}})))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == []
    assert output.exists() is before


def test_exact_v2_then_v3_install_reinstall_and_drift_guards() -> None:
    runner = _fresh_runner()
    installed_v2 = runner.predecessor.contract.install_successor(
        runner.predecessor.base, runner.predecessor._BASE_NAMESPACE_SNAPSHOT
    )
    baseline = MappingProxyType(dict(vars(installed_v2)))
    installed = runner.contract.install_successor(
        installed_v2, baseline, runner.predecessor.contract
    )
    assert installed.RawInputs.__mro__[1].__name__ == "RawInputsV2"
    assert installed.Trainer.__mro__[1].__name__ == "TrainerV2"
    assert installed.RawInputs.__name__ == "RawInputsV3"
    assert installed.Trainer.__name__ == "TrainerV3"
    assert installed.contract.SCHEMA_PREFIX == runner.contract.SCHEMA_PREFIX
    with pytest.raises(RuntimeError, match="already installed"):
        runner.contract.install_successor(
            installed, MappingProxyType(dict(vars(installed))), runner.predecessor.contract
        )

    drifted = _fresh_runner()
    v2 = drifted.predecessor.contract.install_successor(
        drifted.predecessor.base, drifted.predecessor._BASE_NAMESPACE_SNAPSHOT
    )
    snapshot = MappingProxyType(dict(vars(v2)))
    v2.RawInputs = object
    with pytest.raises(PermissionError, match="namespace drifted"):
        drifted.contract.install_successor(v2, snapshot, drifted.predecessor.contract)


def test_v2_terminal_chain_and_lifecycle_normalization() -> None:
    runner = _fresh_runner()
    installed = runner.install()
    contract = runner.contract
    predecessor = contract.validate_predecessor(installed._read_regular)
    assert set(predecessor["artifacts"]) == set(contract.PREDECESSOR_ARTIFACT_BINDINGS)
    assert predecessor["terminal_audit"]["content_sha256"] == contract.V2_TERMINAL_AUDIT_BINDING["content_sha256"]
    assert predecessor["terminal_audit"]["authority"]["automatic_successor_authorized"] is False
    for kind, filename in (("initialization", "initialization.json"), ("schedule", "schedule.json")):
        v2 = predecessor["artifacts"][filename]
        core = dict(v2)
        core.pop("content_sha256")
        core["schema"] = {
            "initialization": contract.INITIALIZATION_SCHEMA,
            "schedule": contract.SCHEDULE_SCHEMA,
        }[kind]
        v3 = contract._v2.with_content_sha256(core)
        assert contract.normalize_lifecycle_artifact_to_v2(v3, kind=kind) == v2
        changed = dict(v3)
        changed["content_sha256"] = "0" * 64
        with pytest.raises(PermissionError):
            contract.normalize_lifecycle_artifact_to_v2(changed, kind=kind)


def test_ground_scalar_fallback_is_exact_and_rejects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _fresh_runner()
    cls = runner.install().RawInputs
    parent = cls.__mro__[1]
    state = {"error": TypeError("expected np.ndarray (got numpy.float32)")}

    def fail(*args, **kwargs):
        error = state["error"]
        raise type(error)(str(error))

    monkeypatch.setattr(parent, "_row_array", fail)
    instance = object.__new__(cls)
    instance.runtime = SimpleNamespace(np=np, torch=torch)
    relative = "shards/demo/ground_plane_z_body_m.f4"
    endpoint = {"scene_shard": "shards/demo/shard.json", "shard_row": 1}
    shard = {"endpoint_count": 2}

    instance.array_cache = {relative: np.asarray([0.1, 0.2], dtype="<f4")}
    value = instance._row_array(
        endpoint, shard, "ground_plane_z_body_m.f4", arm="promoted_jepa", stage="test"
    )
    assert value.shape == torch.Size([])
    assert value.dtype == torch.float32
    assert value.device.type == "cpu"
    assert value.item() == pytest.approx(0.2)

    state["error"] = TypeError("different failure")
    with pytest.raises(TypeError, match="different failure"):
        instance._row_array(endpoint, shard, "ground_plane_z_body_m.f4", arm="x", stage="x")
    state["error"] = TypeError("expected np.ndarray (got numpy.float32)")
    with pytest.raises(TypeError):
        instance._row_array(endpoint, shard, "camera_origin_body_m.f4", arm="x", stage="x")
    for cache in (
        np.asarray([0.1, 0.2], dtype="<f8"),
        np.asarray([[0.1], [0.2]], dtype="<f4"),
        [0.1, 0.2],
        None,
    ):
        instance.array_cache = {} if cache is None else {relative: cache}
        with pytest.raises(PermissionError, match="scalar cache"):
            instance._row_array(endpoint, shard, "ground_plane_z_body_m.f4", arm="x", stage="x")


def test_all_9460_rows_have_the_exact_eight_array_layouts() -> None:
    root = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
    manifest = _json(root / "manifest.json")
    layout = {
        item["path"]: (item["dtype"], tuple(item["trailing_shape"]))
        for item in manifest["array_layout"]
    }
    assert layout == {
        "camera_origin_body_m.f4": ("<f4", (3,)),
        "camera_basis_body_fru.f4": ("<f4", (3, 3)),
        "ground_plane_z_body_m.f4": ("<f4", ()),
        "ground_support_in_frustum.u1": ("|u1", (128, 128, 5)),
        "ground_support_clear_to_target.u1": ("|u1", (128, 128, 5)),
        "pixel_hit_mask.u1": ("|u1", (84, 112)),
        "pixel_first_hit_distance_m.f4": ("<f4", (84, 112)),
        "raster_labels.u1": ("|u1", (64, 64)),
    }
    endpoints = [json.loads(line) for line in (root / "endpoints.jsonl").read_text().splitlines()]
    shards = {
        path.relative_to(root).as_posix(): _json(path)
        for path in sorted((root / "shards").glob("*/shard.json"))
    }
    assert len(endpoints) == 9_460
    assert len(shards) == 88
    assert sum(item["endpoint_count"] for item in shards.values()) == 9_460
    rows: dict[str, list[int]] = {name: [] for name in shards}
    item_sizes = {"<f4": 4, "|u1": 1}
    for shard_path, shard in shards.items():
        records = {item["path"]: item for item in shard["files"]}
        count = shard["endpoint_count"]
        for name, (dtype, trailing) in layout.items():
            record = records[name]
            shape = tuple(record["shape"])
            binary = root / Path(shard_path).parent / name
            assert record["dtype"] == dtype
            assert shape == (count, *trailing)
            assert binary.stat().st_size == record["byte_count"]
            assert record["byte_count"] == math.prod(shape) * item_sizes[dtype]
    for endpoint in endpoints:
        shard_path = endpoint["scene_shard"]
        row = endpoint["shard_row"]
        assert 0 <= row < shards[shard_path]["endpoint_count"]
        rows[shard_path].append(row)
    assert all(sorted(value) == list(range(shards[name]["endpoint_count"])) for name, value in rows.items())
    assert Counter(item["dataset_role"] for item in endpoints) == {
        "train": 7_777,
        "checkpoint_selection": 924,
        "probability_calibration": 759,
    }
    scalar_rows = {
        name: len(endpoints) if trailing == () else 0
        for name, (_, trailing) in layout.items()
    }
    assert scalar_rows == {name: (9_460 if name == "ground_plane_z_body_m.f4" else 0) for name in layout}


def test_exact_first_b4_migrated_forward_and_joint_loss_is_nonreserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _fresh_runner()
    installed = runner.install()
    runtime = installed._load_runtime()
    authorization = _json(ROOT / runner.contract._v2.AUTHORIZATION_RELATIVE_PATH)
    fit, _, _ = installed._camera_model_after_reservation(runtime, authorization)
    output = tmp_path / "must_not_be_created"
    v3_output = ROOT / runner.contract.OUTPUT_ROOT_RELATIVE_PATH
    assert not output.exists()
    assert not v3_output.exists()
    inputs = installed.RawInputs(runtime, authorization)
    trainer = installed.Trainer(runtime, inputs, output, {})
    initial_state, receipt = trainer.initialize(fit)
    assert receipt["complete_state_sha256"] == "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"
    train_pairs = inputs.role_pairs("train")
    schedule_path = ROOT / runner.contract.PREDECESSOR_ROOT_RELATIVE_PATH / "schedule.json"
    assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == runner.contract.PREDECESSOR_ARTIFACT_BINDINGS["schedule.json"]["file_sha256"]
    indices = _json(schedule_path)["presentation_indices"][:4]
    assert indices == [1550, 2807, 3399, 1468]
    assert [train_pairs[index]["content_sha256"] for index in indices] == [
        "def9429ea275caf6aacfb74e219db2f3b24e205dd7260085811d225fe5ee5cd1",
        "93c5ca9231b0e612b99f7a2760835604a26370a09a41e0e35c0e8e4712580bf7",
        "315f8e1ee0f47cfa1f3e001ca7af73d381d8f52547b500d78f8c22b5190aace8",
        "9f1038281b9e57336d80ecf93da40f2cb4a87c9bc0565e3cf584a280e6ae418a",
    ]
    vocabulary, commanded = trainer.commanded_table(train_pairs)
    batch = trainer.batch(
        train_pairs,
        indices,
        vocabulary,
        commanded,
        runtime.torch.device("cpu"),
        role="train",
        arm="promoted_jepa",
        stage="nonreserving_cpu_preflight",
    )
    forward = batch["forward"]
    assert forward["current_image"].shape == (4, 3, 112, 112)
    assert forward["current_camera_origin_body_m"].shape == (4, 3)
    assert forward["current_camera_basis_body_fru"].shape == (4, 3, 3)
    assert forward["current_ground_plane_z_body_m"].shape == (4,)
    assert forward["next_ground_plane_z_body_m"].shape == (4,)
    assert forward["next_prediction_mask"].shape == (4, 64, 64)
    supervision = batch["current_supervision"]
    assert supervision.pixel_hit_mask.shape == (4, 84, 112)
    assert supervision.pixel_first_hit_distance_m.shape == (4, 84, 112)
    assert supervision.ground_support_in_frustum.shape == (4, 128, 128, 5)
    assert supervision.ground_support_clear_to_target.shape == (4, 128, 128, 5)
    assert supervision.target_raster_labels.shape == (4, 64, 64)
    assert all(value.dtype == torch.float32 for name, value in forward.items() if hasattr(value, "dtype") and name != "next_prediction_mask")

    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer or EMA path entered during preflight")

    monkeypatch.setattr(runtime.torch.optim, "AdamW", forbidden)
    model = runtime.model_module.SharedObservableCameraRayJepaV5().eval()
    model.load_state_dict(initial_state, strict=True)
    monkeypatch.setattr(model, "update_ema_target_after_optimizer_step", forbidden)
    before = runtime.model_module.tensor_state_dict_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    with runtime.torch.no_grad():
        pair = model.forward_training_pair(**forward)
        joint = runtime.loss_adapter.combine_joint_losses_v4(
            model, pair, batch["current_supervision"], batch["next_supervision"]
        )
    values = (joint.total, joint.established_jepa.total, joint.observable_camera_ray_v4.total)
    assert all(bool(runtime.torch.isfinite(value).item()) for value in values)
    assert all(parameter.grad is None for parameter in model.parameters())
    after = runtime.model_module.tensor_state_dict_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    assert after == before == receipt["complete_state_sha256"]
    assert not output.exists()
    assert not v3_output.exists()
