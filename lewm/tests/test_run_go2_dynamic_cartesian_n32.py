from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract
from lewm.datasets.go2_attitude_sidecar import row_identity_sha256
from scripts import run_go2_dynamic_cartesian_n32 as runner


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _panel_row(index: int) -> dict[str, object]:
    return {
        "schema": "lewm_go2_physical_micro_overfit_row_v1",
        "scene_id": f"scene-{index // 4}",
        "family": contract.FAMILIES[index % len(contract.FAMILIES)],
        "dataset_role": "train",
        "global_row": index,
        "env_index": index % 16,
        "episode_id": "1",
        "reset_count": 0,
        "current_episode_step": index * 2,
        "next_episode_step": index * 2 + 1,
        "current_frame_index": index * 20,
        "next_frame_index": index * 20 + 1,
        "current_timestamp_ns": index * 200,
        "next_timestamp_ns": index * 200 + 100,
        "primitive": "hold",
        "relative_se2_current_frame": [0.0, 0.0, 0.0],
        "label_shard_path": f"/synthetic/shard-{index // 32}.npz",
        "label_shard_sha256": _digest(f"shard-{index // 32}"),
        "label_shard_row": index % 32,
        "current_image_path": f"/synthetic/current-{index}.png",
        "current_image_sha256": _digest(f"current-{index}"),
        "next_image_path": f"/synthetic/next-{index}.png",
        "next_image_sha256": _digest(f"next-{index}"),
    }


def _sidecar_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "global_row": row["global_row"],
        "dataset_role": "train",
        "row_identity_sha256": row_identity_sha256(row),
        "scene_id_sha256": hashlib.sha256(str(row["scene_id"]).encode()).hexdigest(),
        "env_index": row["env_index"],
        "current_frame_index": row["current_frame_index"],
        "next_frame_index": row["next_frame_index"],
        "current_timestamp_ns": row["current_timestamp_ns"],
        "next_timestamp_ns": row["next_timestamp_ns"],
        "current": {"base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0], "stored_base_yaw_rad": 0.0},
        "next": {"base_quat_world_xyzw": [0.0, 0.0, 0.1, 0.99498743710662], "stored_base_yaw_rad": 0.2003348423231196},
    }


def _joined_fixture() -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    panels = {}
    sidecar = []
    for panel_index, panel in enumerate(("fit", *contract.HOLDOUT_PANELS)):
        rows = [_panel_row(panel_index * 160 + index) for index in range(160)]
        panels[panel] = rows
        sidecar.extend(_sidecar_row(row) for row in rows)
    return panels, sidecar


def test_exact_480_row_sidecar_join_and_current_next_order() -> None:
    panels, sidecar = _joined_fixture()
    joined, audit = runner._join_panel_attitudes(panels, sidecar)
    assert audit["transition_count"] == 480
    assert audit["frame_count"] == 960
    assert len(joined["fit"]) == 320
    assert [record["side"] for record in joined["fit"][:4]] == ["current", "next", "current", "next"]
    assert joined["fit"][0]["base_quat_world_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert joined["fit"][1]["stored_base_yaw_rad"] == pytest.approx(0.2003348423231196)


@pytest.mark.parametrize("field", ("dataset_role", "row_identity_sha256", "scene_id_sha256", "env_index", "current_frame_index", "next_frame_index", "current_timestamp_ns", "next_timestamp_ns"))
def test_sidecar_join_fails_closed_on_every_identity_field(field: str) -> None:
    panels, sidecar = _joined_fixture()
    sidecar[0][field] = "wrong" if isinstance(sidecar[0][field], str) else int(sidecar[0][field]) + 1
    with pytest.raises(ValueError, match="attitude join mismatch"):
        runner._join_panel_attitudes(panels, sidecar)


def test_sidecar_join_rejects_duplicate_or_cross_panel_global_rows() -> None:
    panels, sidecar = _joined_fixture()
    panels["cross_scene_holdout"][0] = dict(panels["fit"][0])
    with pytest.raises(ValueError, match="480 unique"):
        runner._join_panel_attitudes(panels, sidecar)


def _manual_record(index: int) -> dict[str, object]:
    return {
        "image_path": f"correct-{index}",
        "control_image_path": f"global-{index}",
        "same_scene_control_image_path": f"same-{index}",
        "label_shard_path": "shard",
        "label_shard_row": index,
        "side": "current",
        "base_quat_world_xyzw": [float(index), 0.0, 0.0, 1.0],
        "stored_base_yaw_rad": float(index) / 10.0,
    }


def test_wrong_rgb_batches_retain_target_attitude() -> None:
    records = [_manual_record(0), _manual_record(1)]
    dataset = runner.DynamicPanelDataset(records, "synthetic")
    for record in records:
        for key in ("image_path", "control_image_path", "same_scene_control_image_path"):
            dataset._images[str(record[key])] = torch.zeros(3, 2, 2)
    dataset._targets = {index: (torch.zeros(64, 64, dtype=torch.long), torch.ones(64, 64, dtype=torch.bool)) for index in range(2)}
    batch = dataset.evaluation_batch((0, 1))
    assert batch["base_quat_world_xyzw"][:, 0].tolist() == [0.0, 1.0]
    assert batch["stored_base_yaw_rad"].tolist() == pytest.approx([0.0, 0.1])
    assert dataset.snapshot()["image_requests"] == 6
    assert dataset.snapshot()["attitude_requests"] == 2


def test_direct_hierarchical_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(4, 3, 64, 64, requires_grad=True)
    labels = torch.arange(4 * 64 * 64).reshape(4, 64, 64).remainder(3).long()
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss = runner.direct_hierarchical_loss(logits, labels, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


class _EvalDataset:
    def __init__(self) -> None:
        self.events = {name: 0 for name in runner.EVENT_FIELDS}

    def snapshot(self):
        return dict(self.events)

    def delta(self, before):
        return {name: self.events[name] - before.get(name, 0) for name in runner.EVENT_FIELDS}

    def evaluation_batch(self, indices):
        count = len(indices)
        self.events["image_requests"] += count * 3
        self.events["target_requests"] += count
        self.events["attitude_requests"] += count
        return {
            **{condition: torch.zeros(count, 3, 2, 2) for condition in contract.CONDITIONS},
            "labels": torch.zeros(count, 64, 64, dtype=torch.long),
            "mask": torch.ones(count, 64, 64, dtype=torch.bool),
            "base_quat_world_xyzw": torch.arange(count, dtype=torch.float32)[:, None].repeat(1, 4),
            "stored_base_yaw_rad": torch.arange(count, dtype=torch.float32),
        }


class _CapturingModel:
    def __init__(self) -> None:
        self.attitudes = []

    def eval(self):
        return self

    def occupancy_logits(self, images, quaternions, yaws):
        self.attitudes.append((quaternions.clone(), yaws.clone()))
        return torch.zeros(images.shape[0], 3, 64, 64)


def test_evaluation_combines_twelve_frames_and_repeats_target_attitudes(monkeypatch) -> None:
    monkeypatch.setattr(runner, "empty_raw_accumulator", lambda: {})
    monkeypatch.setattr(runner, "update_raw_accumulator", lambda *_args: None)
    monkeypatch.setattr(runner, "finalize_raw_accumulator", lambda _value: {"synthetic": True})
    records = [{"family": contract.FAMILIES[index % len(contract.FAMILIES)]} for index in range(4)]
    model, dataset = _CapturingModel(), _EvalDataset()
    report, access = runner.evaluate_panel(model, dataset, records, device=torch.device("cpu"), panel="synthetic_holdout", controls={})
    quaternions, yaws = model.attitudes[0]
    assert quaternions.shape == (12, 4) and yaws.shape == (12,)
    assert torch.equal(quaternions[:4], quaternions[4:8])
    assert torch.equal(quaternions[:4], quaternions[8:12])
    assert report["wrong_rgb_uses_target_attitude"] is True
    assert access["model_output_frames"] == 12


def test_resource_validation_rejects_cpu_raphael_and_wrong_visibility(monkeypatch) -> None:
    with pytest.raises(ValueError, match="cuda:0"):
        runner._validate_resource_environment("cpu")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="HIP_VISIBLE_DEVICES"):
        runner._validate_resource_environment("cuda:0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    for name in runner.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "AMD Raphael Graphics")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=2 * 1024**3))
    with pytest.raises(ValueError, match="R9700"):
        runner._validate_resource_environment("cuda:0")


def test_resource_validation_accepts_only_registered_discrete_gpu(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    for name in runner.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "AMD Radeon AI PRO R9700")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=32 * 1024**3))
    device, record = runner._validate_resource_environment("cuda:0")
    assert device == torch.device("cuda:0")
    assert record["raphael_rejected"] is True


def test_atomic_publisher_is_no_replace(tmp_path: Path) -> None:
    output = (tmp_path / "result.json").resolve()
    runner._publish_json_exclusive(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        runner._publish_json_exclusive(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.staging-*"))


def test_authoritative_attempt_marker_is_one_shot_even_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = 20260710
    marker_path = (tmp_path / "seed-attempt.json").resolve()
    result_path = (tmp_path / "seed-result.json").resolve()
    monkeypatch.setattr(runner, "_canonical_attempt_marker", lambda _seed: marker_path)
    monkeypatch.setattr(runner, "_canonical_output", lambda _seed: result_path)
    monkeypatch.setattr(
        contract, "validate_attempt_marker", lambda value, *_args, **_kwargs: value
    )
    arguments = [contract.COMMAND_CONTRACT["runner"], "--seed", str(seed)]
    keyword = {
        "seed": seed,
        "invocation": arguments,
        "started_at_utc": "2026-07-11T00:00:00+00:00",
        "implementation_manifest": {"content_sha256": "a" * 64},
        "implementation_manifest_file_sha256": "b" * 64,
    }
    marker, digest = runner._claim_authoritative_attempt(**keyword)
    assert marker_path.is_file() and not result_path.exists()
    assert runner._sha256_file(marker_path) == digest
    assert marker["retry_permitted"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        runner._claim_authoritative_attempt(**keyword)


def test_source_map_covers_isolated_transitive_runtime_imports() -> None:
    assert set(runner.SOURCE_ROLES) == set(runner._source_path_contract())
    assert set(runner.SOURCE_ROLES) == set(contract.IMPLEMENTATION_SOURCE_PATHS)
    code = """
import pathlib
import sys
from scripts import run_go2_dynamic_cartesian_n32
from scripts import finalize_go2_dynamic_cartesian_n32
from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract
root = pathlib.Path(contract.REPOSITORY_ROOT).resolve()
bound = {pathlib.Path(path).resolve() for path in contract.IMPLEMENTATION_SOURCE_PATHS.values() if path.endswith('.py')}
loaded = set()
for module in tuple(sys.modules.values()):
    path = getattr(module, '__file__', None)
    if not path:
        continue
    resolved = pathlib.Path(path).resolve()
    if resolved.suffix == '.py' and resolved.is_file() and resolved.is_relative_to(root):
        loaded.add(resolved)
missing = sorted(map(str, loaded - bound))
assert not missing, missing
"""
    completed = subprocess.run(
        (sys.executable, "-c", code),
        cwd=runner.REPOSITORY_ROOT,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_parser_enforces_seed_order_and_canonical_outputs(tmp_path: Path, monkeypatch) -> None:
    outputs = {seed: (tmp_path / f"seed-{seed}.json").resolve() for seed in contract.EXPECTED_SEEDS}
    markers = {seed: (tmp_path / f"seed-{seed}-attempt.json").resolve() for seed in contract.EXPECTED_SEEDS}
    monkeypatch.setattr(runner, "_canonical_output", lambda seed: outputs[seed])
    monkeypatch.setattr(runner, "_canonical_attempt_marker", lambda seed: markers[seed])
    common = ["--implementation-manifest", str(runner.IMPLEMENTATION_MANIFEST_PATH), "--expected-implementation-manifest-sha256", "0" * 64]
    with pytest.raises(SystemExit):
        runner._parse_args(["--output", str(outputs[20260711]), "--seed", "20260711", *common])
    parsed = runner._parse_args(["--output", str(outputs[20260711]), "--seed", "20260711", "--seed-20260710-result", str(outputs[20260710]), "--expected-seed-20260710-sha256", "1" * 64, "--seed-20260710-attempt-marker", str(markers[20260710]), "--expected-seed-20260710-attempt-marker-sha256", "2" * 64, *common])
    assert parsed.seed == 20260711
    with pytest.raises(SystemExit):
        runner._parse_args(["--output", str(outputs[20260710]), "--non-authoritative-smoke", *common])


def test_model_config_is_exact_dynamic_patch7_and_inert_weight() -> None:
    assert contract.MODEL_CONFIG["bev_lift_type"] == runner.DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
    assert contract.MODEL_CONFIG["image_size"] == 112
    assert contract.MODEL_CONFIG["patch_size"] == 7
    assert contract.MODEL_CONFIG["projective_output_cell_size_m"] == 0.1
    assert contract.MODEL_CONFIG["occupancy_weight"] == 2.0
    assert contract.MODEL_CONFIG["jepa_weight"] == 0.0


def test_physical_manifest_derives_exact_bound_dynamic_query_support() -> None:
    manifest = json.loads(runner.DATASET_MANIFEST_PATH.read_text())
    support = runner.build_projective_query_support_contract(
        manifest,
        lift_type=runner.DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    )
    assert support == contract.PROJECTIVE_QUERY_SUPPORT
    assert runner.validate_projective_query_support_binding(
        model_config=contract.MODEL_CONFIG,
        projective_query_support=support,
        dataset_manifest=manifest,
        occupancy_output_contract={
            "projective_query_support_contract_sha256": support["contract_sha256"]
        },
    ) == support


def test_secure_reader_rejects_symlink_alias(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(OSError):
        runner._regular_bytes(alias.resolve().parent / alias.name, name="aliased input")


@pytest.mark.parametrize(
    "payload",
    (
        b'{"seed":20260710,"seed":20260711}',
        b'{"loss":NaN}',
    ),
)
def test_runner_strict_json_rejects_duplicate_keys_and_nonfinite_values(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError):
        runner._strict_json(payload, name="authoritative input")


def test_determinism_configuration_is_strict(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled, *, warn_only: calls.append((enabled, warn_only)))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    runner._configure_determinism(20260710)
    assert calls == [(True, False)]


def test_initial_state_seeds_immediately_before_model_construction(monkeypatch) -> None:
    observed = []

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(()))

    def construct(_device):
        model = _Tiny()
        observed.append(float(model.weight.detach()))
        return model

    monkeypatch.setattr(runner, "_new_model", construct)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    seed = 20260710
    torch.manual_seed(seed)
    expected_model = _Tiny()
    state = expected_model.state_dict()
    state_hash = runner._state_dict_sha256(state)
    state_contract, contract_hash = runner._state_contract(expected_model)
    manifest = {"model_initial_state_sha256": {str(seed): state_hash}, "model_state_contract_sha256": {str(seed): contract_hash}}
    runner._build_initial_state(torch.device("cpu"), seed, manifest)
    assert observed == pytest.approx([float(expected_model.weight.detach())])


def test_authoritative_schedule_commitments_cover_both_seeds_and_branches() -> None:
    assert set(contract.SCHEDULE_SHA256) == {(seed, branch) for seed in contract.EXPECTED_SEEDS for branch in contract.BRANCH_CONFIGS}
    for (seed, branch), digest in contract.SCHEDULE_SHA256.items():
        schedule = contract.deterministic_minibatch_schedule(seed=seed, branch=branch)
        assert contract.validate_minibatch_schedule(schedule, seed=seed, branch=branch) == digest


def test_runner_manifest_loader_checks_external_hash_and_every_source(
    tmp_path: Path, monkeypatch
) -> None:
    source_paths = {}
    entries = []
    for index, role in enumerate(runner.SOURCE_ROLES):
        path = (tmp_path / f"{role}.py").resolve()
        path.write_bytes(f"source-{index}".encode())
        source_paths[role] = path
        entries.append(
            {"role": role, "path": str(path), "sha256": runner._sha256_file(path)}
        )
    entries.sort(key=lambda item: item["role"])
    manifest_core = {
        "schema": contract.IMPLEMENTATION_MANIFEST_SCHEMA,
        "binding": {
            "path": str(source_paths["binding"]),
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": str(source_paths["preoutput_amendment"]),
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": str(source_paths["attempt_control_amendment"]),
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "sources": {
            "entries": entries,
            "entry_count": len(entries),
            "source_map_sha256": contract.canonical_json_sha256(entries),
        },
        "tests": {
            "command": contract.IMPLEMENTATION_TEST_COMMAND,
            "passed": contract.IMPLEMENTATION_TEST_PASSED,
            "all_passed": True,
        },
        "inputs": contract.INPUT_BINDINGS,
        "resource_policy": contract.RESOURCE_POLICY,
        "model_config": contract.MODEL_CONFIG,
        "objective": contract.OBJECTIVE_CONTRACT,
        "preprocessing": contract.PREPROCESSING_CONTRACT,
        "controls": contract.CONTROL_CONTRACT,
        "projective_query_support": contract.PROJECTIVE_QUERY_SUPPORT,
        "model_initial_state_sha256": {
            str(seed): _digest(f"state-{seed}") for seed in contract.EXPECTED_SEEDS
        },
        "model_state_contract_sha256": {
            str(seed): _digest(f"contract-{seed}") for seed in contract.EXPECTED_SEEDS
        },
        "schedules": contract.SCHEDULE_CONTRACT,
        "commands": contract.COMMAND_CONTRACT,
    }
    manifest = {
        **manifest_core,
        "content_sha256": contract.canonical_json_sha256(manifest_core),
    }
    manifest_path = (tmp_path / "implementation.json").resolve()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    file_hash = runner._sha256_file(manifest_path)
    monkeypatch.setattr(runner, "IMPLEMENTATION_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(runner, "_source_path_contract", lambda: source_paths)
    monkeypatch.setattr(
        contract,
        "IMPLEMENTATION_SOURCE_PATHS",
        {role: str(path) for role, path in source_paths.items()},
    )
    assert runner._validate_implementation_manifest(manifest_path, file_hash) == manifest
    source_paths["model"].write_text("changed")
    with pytest.raises(ValueError, match="source hash"):
        runner._validate_implementation_manifest(manifest_path, file_hash)
    with pytest.raises(ValueError, match="file SHA-256"):
        runner._validate_implementation_manifest(manifest_path, "0" * 64)
