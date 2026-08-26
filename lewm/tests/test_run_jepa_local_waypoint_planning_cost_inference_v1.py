from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_jepa_local_waypoint_planning_cost_inference_v1.py"
SPEC = importlib.util.spec_from_file_location("planning_cost_gpu_worker_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def _mixed_frames() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in range(48):
        state_id = f"purpose-{state}"
        for slot in range(3):
            rows.append(
                {
                    "kind": "CONTEXT",
                    "state_id": state_id,
                    "slot": slot,
                    "rgb_sha256": f"{state:02x}{slot}".ljust(64, "0"),
                }
            )
        rows.append(
            {
                "kind": "GOAL",
                "state_id": state_id,
                "rgb_sha256": f"goal{state}".ljust(64, "0"),
            }
        )
    return rows


def test_mixed_context_goal_inventory_filters_goal_rows_without_slot_keyerror() -> None:
    mapping = G._context_frame_map(_mixed_frames())
    assert len(mapping) == 144
    assert mapping[("purpose-0", 2)]["kind"] == "CONTEXT"


def test_context_inventory_fails_closed_on_missing_slot() -> None:
    rows = _mixed_frames()
    rows.pop(0)
    with pytest.raises(G.InferenceError, match="144 context slots"):
        G._context_frame_map(rows)


def test_new_encoder_inventory_excludes_current_and_forms_nine_full_batches() -> None:
    rows = _mixed_frames()
    encoded = G._new_encoder_frame_inventory(rows)
    assert len(encoded) == 144
    assert len(encoded) // G.ENCODER_BATCH == 9
    assert all(
        row["kind"] == "GOAL" or int(row["slot"]) in (0, 1)
        for row in encoded
    )
    assert sum(row["kind"] == "CONTEXT" for row in encoded) == 96
    assert sum(row["kind"] == "GOAL" for row in encoded) == 48
    assert not any(
        row["kind"] == "CONTEXT" and int(row["slot"]) == 2
        for row in encoded
    )


def test_frozen_current_is_copied_once_and_context_current_are_exact_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = (2, 3)
    monkeypatch.setattr(G, "TENSOR_SHAPE", shape)
    monkeypatch.setattr(G, "TENSOR_DTYPE", np.dtype(np.float16))
    authority: dict[str, dict[str, object]] = {}
    rows = _mixed_frames()
    for row in rows:
        if row["kind"] == "CONTEXT":
            row["family"] = "family"
            row["role"] = "heldout"
            if row["slot"] == 2:
                rgb = tmp_path / "rgb" / f"{row['state_id']}.png"
                rgb.parent.mkdir(parents=True, exist_ok=True)
                rgb.write_bytes(f"rgb-{row['state_id']}".encode())
                row["rgb_path"] = str(rgb)
                row["rgb_sha256"] = G.sha256_file(rgb)
    for state in range(48):
        state_id = f"purpose-{state}"
        value = np.asarray(
            [[state, state + 0.25, state + 0.5], [1.0, -2.0, 3.0]],
            dtype=np.float16,
        )
        path = tmp_path / "authority" / f"{state_id}.f16"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.tobytes(order="C"))
        rgb_sha = next(
            str(row["rgb_sha256"])
            for row in rows
            if row["kind"] == "CONTEXT"
            and row["state_id"] == state_id
            and row["slot"] == 2
        )
        authority[state_id] = {
            "rgb_sha256": rgb_sha,
            "token_path": str(path),
            "token_sha256": G.sha256_file(path),
        }
    monkeypatch.setattr(G, "_dense_current_authority", lambda: authority)

    context, current = G._copy_current_and_validate(rows, output_root=tmp_path)
    assert len(context) == len(current) == 48
    assert len({row["path"] for row in [*context, *current]}) == 48
    for context_row, current_row in zip(context, current, strict=True):
        assert context_row["kind"] == "CONTEXT"
        assert context_row["horizon_or_null"] == 0
        assert current_row["kind"] == "CURRENT"
        assert current_row["horizon_or_null"] is None
        assert {
            key: context_row[key] for key in ("path", "sha256", "bytes")
        } == {key: current_row[key] for key in ("path", "sha256", "bytes")}
        external = current_row["external_existing_artifact"]
        assert external["raw_payload_byte_exact"] is True
        assert external["copied_to_attempt_once"] is True
        assert external["reencoded"] is False


def test_frozen_current_copy_rejects_rgb_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _mixed_frames()
    for row in rows:
        if row["kind"] == "CONTEXT":
            row["family"] = "family"
            row["role"] = "heldout"
            if row["slot"] == 2:
                rgb = tmp_path / "rgb" / f"{row['state_id']}.png"
                rgb.parent.mkdir(parents=True, exist_ok=True)
                rgb.write_bytes(f"rgb-{row['state_id']}".encode())
                row["rgb_path"] = str(rgb)
                row["rgb_sha256"] = G.sha256_file(rgb)
    authority = {}
    for state in range(48):
        state_id = f"purpose-{state}"
        context = next(
            row
            for row in rows
            if row["kind"] == "CONTEXT"
            and row["state_id"] == state_id
            and row["slot"] == 2
        )
        authority[state_id] = {
            "rgb_sha256": context["rgb_sha256"],
            "token_path": str(tmp_path / "absent.f16"),
            "token_sha256": "0" * 64,
        }
    monkeypatch.setattr(G, "_dense_current_authority", lambda: authority)
    (tmp_path / "rgb/purpose-0.png").write_bytes(b"tampered")
    with pytest.raises(G.InferenceError, match="current RGB SHA"):
        G._copy_current_and_validate(rows, output_root=tmp_path)


def test_canonical_frame_order_is_digest_then_kind_state_slot() -> None:
    rows = _mixed_frames()
    observed = sorted(rows, key=G._canonical_frame_key)
    keys = [G._canonical_frame_key(row) for row in observed]
    assert keys == sorted(keys)


class _FakeTensor:
    def __init__(self, value: np.ndarray):
        self.value = np.ascontiguousarray(value)
        self.dtype = self.value.dtype
        self.shape = self.value.shape
        self.ndim = self.value.ndim

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def contiguous(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeModel:
    def __init__(self, value: float):
        self.value = value

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {
            "buffer": _FakeTensor(np.asarray([3], np.int64)),
            "weight": _FakeTensor(np.asarray([[self.value, 2.0]], np.float32)),
        }


def test_parameter_state_digest_is_deterministic_and_covers_buffers() -> None:
    first = G._parameter_state_digest(_FakeModel(1.0))
    assert first == G._parameter_state_digest(_FakeModel(1.0))
    assert first != G._parameter_state_digest(_FakeModel(9.0))
    assert len(first) == 64


def test_gpu_artifact_reference_survives_attempt_rename(tmp_path: Path) -> None:
    attempt = tmp_path / ".attempt"
    path = attempt / "latents/value.npy"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tensor")
    reference = G._artifact_reference(path, attempt)
    canonical = tmp_path / "canonical"
    attempt.rename(canonical)
    assert G._artifact_path(reference, canonical).read_bytes() == b"tensor"


def test_gpu_interpreter_resolves_to_exact_frozen_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = Path(G.sys.executable).resolve()
    exact = {
        "path": str(resolved),
        "sha256": G.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    monkeypatch.setattr(G.CONTRACT, "INTERPRETER_BINARY_BINDING", exact)
    assert G._validated_interpreter_binary_binding(G.sys.executable) == exact
    monkeypatch.setattr(
        G.CONTRACT, "INTERPRETER_BINARY_BINDING", {**exact, "bytes": exact["bytes"] + 1}
    )
    with pytest.raises(G.InferenceError, match="interpreter binary binding drift"):
        G._validated_interpreter_binary_binding(G.sys.executable)


def test_gpu_source_commit_live_head_gate_and_postresult_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        output = commit if arguments[-1] != "HEAD" else "b" * 40
        return SimpleNamespace(stdout=output + "\n")

    monkeypatch.setattr(G.subprocess, "run", fake_run)
    G.validate_source_commit(commit, require_live_head=False)
    with pytest.raises(G.InferenceError, match="live HEAD"):
        G.validate_source_commit(commit, require_live_head=True)


def test_gpu_worker_import_does_not_import_genesis() -> None:
    assert not any(name == "genesis" or name.startswith("genesis.") for name in G.sys.modules)


def test_gpu_import_closure_includes_transitive_local_semantic_modules() -> None:
    assert G.GPU_LOCAL_SOURCE_MODULE_IDS == (
        "scripts.dev_frozen_dense_representation_encoders_v1",
        "scripts.dev_proprio_predictor_v1",
        "scripts.run_dev_v03_temporal_action_jepa_v1",
        "scripts.build_dev_v03_proprio_action_manifest_v1",
        "scripts.dev_action_slew_reconstruction_v1",
        "scripts.dev_checkpoint_v1",
        "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract",
    )


def test_encoder_source_repository_binding_is_exact_and_fails_on_dirty_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "encoder-source"
    backbones = repository / "src/hub/backbones.py"
    backbones.parent.mkdir(parents=True)
    backbones.write_bytes(b"frozen-backbone-source\n")
    commit = "2" * 40
    expected = {
        "latent_bindings": {
            "encoder": {
                "constructor": "vjepa2_1_vit_large_384",
                "source_repository": {
                    "path": str(repository),
                    "git_commit": commit,
                    "worktree_clean_required": True,
                    "backbones_path": "src/hub/backbones.py",
                    "backbones_sha256": G.sha256_file(backbones),
                    "backbones_bytes": backbones.stat().st_size,
                },
            }
        }
    }
    monkeypatch.setattr(G.CONTRACT, "build_contract", lambda: expected)
    dirty = {"value": False}

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        if arguments[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=commit + "\n")
        if arguments[1:3] == ["status", "--porcelain=v1"]:
            return SimpleNamespace(stdout=" M src/hub/backbones.py\n" if dirty["value"] else "")
        raise AssertionError(arguments)

    monkeypatch.setattr(G.subprocess, "run", fake_run)
    receipt = G._encoder_source_repository_binding()
    assert receipt["git_commit"] == commit
    assert receipt["worktree_clean"] is True
    assert receipt["backbones_sha256"] == G.sha256_file(backbones)
    dirty["value"] = True
    with pytest.raises(G.InferenceError, match="repository binding drift"):
        G._encoder_source_repository_binding()


def test_gpu_foundational_package_roots_require_exact_version_and_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "demo"
    package_root.mkdir()
    expected = {
        "demo": {
            "distribution": "demo-dist",
            "version": "1.2.3",
            "import_name": "demo",
            "package_root": str(package_root),
        }
    }
    monkeypatch.setattr(G.CONTRACT, "GPU_FOUNDATIONAL_PACKAGE_BINDINGS", expected)
    observed_version = {"value": "1.2.3"}
    monkeypatch.setattr(
        G.importlib.metadata,
        "version",
        lambda _distribution: observed_version["value"],
    )
    monkeypatch.setattr(
        G,
        "_runtime_import_resolution",
        lambda import_name, root: {
            "import_name": import_name,
            "find_spec_origin": str(root / "__init__.py"),
            "submodule_search_locations": [str(root)],
            "live_module_file": str(root / "__init__.py"),
            "expected_package_root": str(root),
            "resolved_inside_frozen_package_root": True,
            "pass": True,
        },
    )
    receipt = G._foundational_package_roots()
    assert receipt["demo"]["version"] == "1.2.3"
    assert receipt["demo"]["import_resolution"]["pass"] is True
    observed_version["value"] = "9.9.9"
    with pytest.raises(G.InferenceError, match="foundational package drift"):
        G._foundational_package_roots()


def test_gpu_inference_revalidates_and_binds_preflight_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    roots = {
        "torch": {
            "distribution": "torch",
            "version": "gpu-torch",
            "import_name": "torch",
            "package_root": "/frozen/torch",
            "import_resolution": {"pass": True},
        }
    }
    policy = {"scope": "synthetic"}
    interpreter = Path(G.sys.executable).resolve()
    expected_environment = {
        "interpreter": str(interpreter),
        "python": "gpu-python",
        "torch": "gpu-torch",
        "numpy": "gpu-numpy",
        "scipy": "gpu-scipy",
        "pillow": "gpu-pillow",
        "pyyaml": "gpu-pyyaml",
    }
    monkeypatch.setattr(G.CONTRACT, "FOUNDATIONAL_PACKAGE_CLOSURE_POLICY", policy)
    monkeypatch.setattr(
        G.CONTRACT,
        "build_contract",
        lambda: {
            "execution": {
                "environments": {"encoder_predictor": expected_environment}
            }
        },
    )
    monkeypatch.setattr(G, "_foundational_package_roots", lambda: copy.deepcopy(roots))
    live = {
        "python": "gpu-python",
        "executable": str(interpreter),
        "interpreter_binding": {
            "path": str(interpreter),
            "sha256": G.sha256_file(interpreter),
            "bytes": interpreter.stat().st_size,
        },
        "torch": "gpu-torch",
        "packages": {
            "numpy": "gpu-numpy",
            "scipy": "gpu-scipy",
            "pillow": "gpu-pillow",
            "pyyaml": "gpu-pyyaml",
        },
        "cuda_available": True,
        "device": {
            "type": "cuda",
            "index": 0,
            "name": "AMD Radeon AI PRO R9700",
            "total_memory_bytes": 123,
            "hip": "synthetic",
        },
    }
    monkeypatch.setattr(
        G, "_live_gpu_environment_identity", lambda: copy.deepcopy(live)
    )
    monkeypatch.setattr(G, "_assert_gpu_process_separation", lambda: None)
    value = G.attach_digest(
        {
            **G.phase_core(
                "jepa_local_waypoint_planning_cost_gpu_environment_v1", commit
            ),
            "python": live["python"],
            "executable": live["executable"],
            "interpreter_binding": live["interpreter_binding"],
            "torch": live["torch"],
            "packages": live["packages"],
            "device": live["device"],
            "foundational_package_roots": copy.deepcopy(roots),
            "foundational_package_closure_policy": copy.deepcopy(policy),
            "checkpoint_tensor_open_count": 0,
            "predictor_inference_calls": 0,
            "genesis_imported": False,
            "pass": True,
        }
    )
    path = tmp_path / G.GPU_ENVIRONMENT_REL
    path.parent.mkdir(parents=True)
    G.atomic_json(path, value)
    binding = G._gpu_environment_binding_for_inference(
        commit, output_root=tmp_path
    )
    assert binding["sha256"] == G.sha256_file(path)
    assert binding["content_digest"] == value["content_digest"]

    drifted = copy.deepcopy(value)
    drifted["foundational_package_roots"]["torch"]["package_root"] = "/shadow"
    G.atomic_json(path, G.attach_digest(drifted))
    with pytest.raises(G.InferenceError, match="between preflight and inference"):
        G._gpu_environment_binding_for_inference(commit, output_root=tmp_path)
