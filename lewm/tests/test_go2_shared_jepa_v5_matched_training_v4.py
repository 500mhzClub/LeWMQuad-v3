from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import itertools
from types import MappingProxyType, SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_go2_shared_jepa_v5_matched_training_v4.py"
_LOAD_COUNT = itertools.count()
FIRST_UPDATE_INDICES = [1550, 2807, 3399, 1468, 1317, 1451, 448, 1842, 3056, 217, 429, 1601, 3965, 2124, 2875, 1382]


def _fresh_runner():
    name = f"_lewm_matched_v4_test_runner_{next(_LOAD_COUNT)}"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scope_science_hashes_and_production_cap() -> None:
    contract = _fresh_runner().contract
    assert (contract.MAXIMUM_ATTEMPTS, contract.RETRY_AUTHORIZED, contract.AUTOMATIC_V5_AUTHORIZED) == (1, False, False)
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("/matched_training_v4")
    expected = copy.deepcopy(contract._V3_SCIENCE_CONTRACT)
    expected["candidate"]["schema"] = contract.PRE_G2_CHECKPOINT_SCHEMA
    assert contract.science_contract() == expected
    for path, digest in contract.V3_SOURCE_SHA256.items():
        assert _file_sha(ROOT / path) == digest
    assert _file_sha(ROOT / contract.TERMINAL_AUDIT_RELATIVE_PATH) == contract.V3_TERMINAL_AUDIT_BINDING["file_sha256"]
    assert set(contract.current_source_bindings()) == set(contract.SOURCE_PATHS)
    production_lines = sum(len((ROOT / path).read_text().splitlines()) for path in (contract.CONTRACT_RELATIVE_PATH, contract.RUNNER_RELATIVE_PATH))
    assert production_lines <= 300


def test_isolated_import_is_accelerator_free_and_nonmutating() -> None:
    output = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4"
    before = output.exists()
    code = f"""
import importlib.util,json,sys
p={str(RUNNER)!r}; s=importlib.util.spec_from_file_location('_isolated_v4',p)
m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
print(json.dumps(sorted(set(sys.modules)&{{'torch','numpy','PIL','cv2'}})))
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    assert result.stderr == "" and json.loads(result.stdout) == []
    assert output.exists() is before


def test_exact_v3_then_v4_install_reinstall_and_drift_guards() -> None:
    runner = _fresh_runner()
    installed_v3 = runner.predecessor.install()
    installed = runner.contract.install_successor(installed_v3, MappingProxyType(dict(vars(installed_v3))), runner.predecessor.contract)
    assert installed.RawInputs.__name__ == "RawInputsV3"
    assert [item.__name__ for item in installed.Trainer.__mro__[:4]] == ["TrainerV4", "TrainerV3", "TrainerV2", "Trainer"]
    assert installed.contract.SCHEMA_PREFIX == runner.contract.SCHEMA_PREFIX
    assert installed.contract.science_contract() == runner.contract.science_contract()
    assert installed.contract.current_source_bindings(ROOT) == runner.contract.current_source_bindings(ROOT)
    with pytest.raises(RuntimeError, match="already installed"):
        runner.contract.install_successor(installed, MappingProxyType(dict(vars(installed))), runner.predecessor.contract)

    drifted = _fresh_runner()
    v3 = drifted.predecessor.install()
    snapshot = MappingProxyType(dict(vars(v3)))
    v3.RawInputs = object
    with pytest.raises(PermissionError, match="namespace drifted"):
        drifted.contract.install_successor(v3, snapshot, drifted.predecessor.contract)


def test_v3_terminal_chain_and_lifecycle_normalization() -> None:
    runner = _fresh_runner()
    installed, contract = runner.install(), runner.contract
    predecessor = contract.validate_predecessor(installed._read_regular)
    assert set(predecessor["artifacts"]) == set(contract.PREDECESSOR_ARTIFACT_BINDINGS)
    assert predecessor["terminal_audit"]["content_sha256"] == contract.V3_TERMINAL_AUDIT_BINDING["content_sha256"]
    assert predecessor["terminal_audit"]["training_boundary"]["backward_invocation_count"] == 1
    assert predecessor["terminal_audit"]["training_boundary"]["optimizer_step_count"] == 0
    current: dict[str, dict] = {}
    for kind, filename in (("initialization", "initialization.json"), ("schedule", "schedule.json")):
        core = dict(predecessor["artifacts"][filename])
        core.pop("content_sha256")
        core["schema"] = {"initialization": contract.INITIALIZATION_SCHEMA, "schedule": contract.SCHEDULE_SCHEMA}[kind]
        current[filename] = contract._v3.with_content_sha256(core)
        assert contract.normalize_lifecycle_artifact_to_v3(current[filename], kind=kind) == predecessor["artifacts"][filename]

    def read(path: Path) -> bytes:
        return contract.canonical_json_bytes(current[path.name]) + b"\n"

    initialization, schedule = contract.require_v3_initialization_and_schedule(Path("/not-opened"), predecessor, read)
    assert initialization == current["initialization.json"]
    assert schedule["presentation_indices"][:16] == FIRST_UPDATE_INDICES
    changed = dict(initialization)
    changed["content_sha256"] = "0" * 64
    with pytest.raises(PermissionError):
        contract.normalize_lifecycle_artifact_to_v3(changed, kind="initialization")


def test_warn_only_device_hook_and_exact_warning_parser() -> None:
    runner = _fresh_runner()
    installed, contract = runner.install(), runner.contract
    calls: list[tuple[bool, bool]] = []
    state = {"enabled": False, "warn_only": False}

    def use(mode: bool, *, warn_only: bool = False) -> None:
        calls.append((mode, warn_only))
        state.update(enabled=mode, warn_only=mode and warn_only)

    properties = SimpleNamespace(name="AMD Radeon AI PRO R9700", total_memory=32_000_000_000)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_properties=lambda _device: properties),
        device=lambda value: value,
        use_deterministic_algorithms=use,
        are_deterministic_algorithms_enabled=lambda: state["enabled"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False)),
        version=SimpleNamespace(hip="test-hip"),
        __version__="test-torch",
    )
    trainer = installed.Trainer(SimpleNamespace(torch=torch), SimpleNamespace(), Path("."), {})
    device, resource = trainer.device()
    assert device == "cuda:0" and calls == [(True, False), (True, True)]
    assert resource["determinism"]["warn_only"] is True
    assert resource["determinism"]["arms"] == {}

    plain = contract.normalize_determinism_warning(contract.GRID_WARNING, UserWarning)
    trailer = contract.normalize_determinism_warning(contract.GRID_WARNING + contract._CONTEXT_PREFIX + "157" + contract._CONTEXT_SUFFIX, UserWarning)
    assert plain["context_source_line"] is None and trailer["context_source_line"] == 157
    collector = contract.CompactDeterminismWarnings()
    collector(contract.GRID_WARNING, UserWarning)
    collector(contract.GRID_WARNING + contract._CONTEXT_PREFIX + "157" + contract._CONTEXT_SUFFIX, UserWarning)
    assert collector.receipt()["kernel_counts"] == {"grid_sampler_2d_backward_cuda": 2}
    with pytest.raises(RuntimeError, match="emitted no expected grid-sampler warning"):
        contract.run_with_expected_grid_warning(lambda: "mock frozen training completed")
    for message, category in ((contract.GRID_WARNING, RuntimeWarning), (contract.GRID_WARNING + " drift", UserWarning), (contract.GRID_WARNING + contract._CONTEXT_PREFIX + "0157" + contract._CONTEXT_SUFFIX, UserWarning)):
        with pytest.raises(RuntimeError, match="unexpected training warning"):
            contract.normalize_determinism_warning(message, category)


@pytest.mark.skipif(os.environ.get("LEWM_MATCHED_V4_GPU_PREFLIGHT") != "1", reason="explicit GPU0 preflight only")
def test_gpu0_exact_nonreserving_first_full_update() -> None:
    import torch

    runner = _fresh_runner()
    installed, contract = runner.install(), runner.contract
    output = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    throwaway_output = Path(f"/tmp/lewm_v4_preflight_{os.getpid()}_must_not_exist")
    assert not output.exists() and not output.is_symlink() and not throwaway_output.exists()
    predecessor_paths = [ROOT / contract.PREDECESSOR_ROOT_RELATIVE_PATH / name for name in contract.PREDECESSOR_ARTIFACT_BINDINGS]
    before = {path: _file_sha(path) for path in predecessor_paths}
    predecessor = contract.validate_predecessor(installed._read_regular)
    authorization = _json(ROOT / contract.AUTHORIZATION_RELATIVE_PATH)
    runtime = installed._load_runtime()
    fit, _, _ = installed._camera_model_after_reservation(runtime, authorization)
    inputs = installed.RawInputs(runtime, authorization)
    trainer = installed.Trainer(runtime, inputs, throwaway_output, {})
    device, resource = trainer.device()
    initial_state, initialization = trainer.initialize(fit)
    assert initialization["complete_state_sha256"] == predecessor["artifacts"]["initialization.json"]["complete_state_sha256"]
    train_pairs = inputs.role_pairs("train")
    schedule = predecessor["artifacts"]["schedule.json"]["presentation_indices"]
    assert schedule[:16] == FIRST_UPDATE_INDICES
    vocabulary, commanded_cpu = trainer.commanded_table(train_pairs)
    commanded = commanded_cpu.to(device)

    torch.manual_seed(contract.INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(contract.INITIALIZATION_SEED)
    model = runtime.model_module.SharedObservableCameraRayJepaV5().to(device)
    model.load_state_dict(initial_state, strict=True)
    state_sha = lambda: runtime.model_module.tensor_state_dict_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
    initial_sha = state_sha()
    assert initial_sha == initialization["complete_state_sha256"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=contract.learning_rate(1), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False,
    )
    optimizer.zero_grad(set_to_none=True)
    collector, backward_count, presentation_count = contract.CompactDeterminismWarnings(), 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        original_showwarning = warnings.showwarning
        warnings.showwarning = collector
        try:
            for microbatch in range(4):
                indices = schedule[microbatch * 4 : (microbatch + 1) * 4]
                batch = trainer.batch(train_pairs, indices, vocabulary, commanded, device, role="train", arm="promoted_jepa", stage="nonreserving_gpu_first_update_preflight")
                pair = model.forward_training_pair(**batch["forward"])
                joint = runtime.loss_adapter.combine_joint_losses_v4(model, pair, batch["current_supervision"], batch["next_supervision"])
                backward = trainer.backward_for_arm(joint, "promoted_jepa")
                assert bool(torch.isfinite(backward).item())
                (backward / 4).backward()
                backward_count += 1
                presentation_count += len(indices)
        finally:
            warnings.showwarning = original_showwarning
    assert (backward_count, presentation_count) == (4, 16)
    gradient_before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gradient_after = math.sqrt(sum(float(parameter.grad.detach().float().square().sum().cpu()) for parameter in model.parameters() if parameter.grad is not None))
    assert bool(torch.isfinite(gradient_before).item()) and math.isfinite(gradient_after) and gradient_after <= 1.0001
    optimizer.step()
    ema_count = 0
    model.update_ema_target_after_optimizer_step()
    ema_count += 1
    assert ema_count == 1
    steps = {int(value["step"].item()) for value in optimizer.state.values() if "step" in value}
    assert steps == {1}
    assert state_sha() != initial_sha
    assert all(bool(torch.isfinite(value).all().item()) for value in model.state_dict().values() if value.is_floating_point() or value.is_complex())
    warning_receipt = collector.receipt()
    assert warning_receipt["warning_count"] > 0
    assert warning_receipt["kernel_counts"] == {"grid_sampler_2d_backward_cuda": warning_receipt["warning_count"]}
    assert resource["determinism"]["warn_only"] is True
    assert {path: _file_sha(path) for path in predecessor_paths} == before
    assert not output.exists() and not output.is_symlink() and not throwaway_output.exists()
    del model, optimizer, fit, initial_state, commanded
    torch.cuda.empty_cache()
