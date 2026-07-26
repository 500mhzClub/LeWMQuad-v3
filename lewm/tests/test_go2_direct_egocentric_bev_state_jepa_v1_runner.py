from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"


def _load_runner():
    name = "_test_go2_direct_egocentric_bev_state_jepa_v1_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_is_source_only_and_defers_tensor_stack() -> None:
    program = f"""
import importlib.util, json, pathlib, sys
path = pathlib.Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_direct_bev_runner_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({{
    'torch': 'torch' in sys.modules,
    'numpy': 'numpy' in sys.modules,
    'PIL': 'PIL' in sys.modules,
    'has_loader': hasattr(module, 'DirectBevNarrowLoader'),
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "PIL": False,
        "has_loader": True,
        "numpy": False,
        "torch": False,
    }


def test_exact_raster_confusion_formula() -> None:
    runner = _load_runner()
    result = runner._confusion_metrics(
        [[8, 1, 1], [2, 6, 2], [0, 1, 9]],
        nll_sum=15.0,
        cell_count=30,
    )
    assert result["confusion_target_row_predicted_column"] == [
        [8, 1, 1],
        [2, 6, 2],
        [0, 1, 9],
    ]
    assert result["unknown_recall"] == 0.8
    assert result["free_recall"] == 0.6
    assert result["occupied_recall"] == 0.9
    assert result["balanced_accuracy"] == (0.8 + 0.6 + 0.9) / 3
    assert result["nll"] == 0.5


def test_action_macro_balanced_accuracy_uses_all_nine_classes() -> None:
    runner = _load_runner()
    actual = list(range(9)) * 2
    predicted = list(range(9)) + [0] * 9
    value, recalls = runner._macro_balanced_accuracy(actual, predicted)
    assert recalls[0] == 1.0
    assert recalls[1:] == [0.5] * 8
    assert value == sum(recalls) / 9


def test_source_declares_exact_caps_and_no_general_frame_call() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MAXIMUM_UPDATES" in source
    assert "MAXIMUM_PRESENTATIONS" in source
    assert "MICROBATCH_SIZE" in source
    assert ".frame(" not in source
    assert '"raster_labels.u1"' in source
    assert "_row_array(" in source
    assert "read_count_after_write\": 0" in source


def test_narrow_loader_ledgers_forbidden_supervision_before_failure() -> None:
    runner = _load_runner()
    progress: dict[str, object] = {}
    loader = runner.DirectBevNarrowLoader(
        SimpleNamespace(),
        SimpleNamespace(),
        progress=progress,
    )
    initial_access = loader.model_facing_access_counts()
    assert initial_access
    assert set(initial_access.values()) == {0}
    public_access = runner._access_counters(loader)
    assert set(public_access) == set(runner.contract.ACCESS_COUNTER_FIELDS)
    assert public_access["raw_manifest_open_count"] == 1
    assert public_access["n320_checkpoint_open_count"] == 1
    assert all(
        public_access[name] == 0
        for name in runner.contract.FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS
    )
    with pytest.raises(PermissionError, match="only raster_labels.u1"):
        loader.raster_label(
            "unused",
            role="train",
            stage="fixture",
            scope="training",
            filename="depth.f32",
        )
    receipt = loader.receipt()
    forbidden = receipt["forbidden_semantic_counters"]
    assert forbidden["other_supervision_array_open_count"] == 1
    assert receipt["raster_physical_array_open_attempt_count"] == 0
    assert receipt["rgb_physical_read_attempt_count"] == {
        "current": 0,
        "next": 0,
        "fixed_negative": 0,
        "endpoint": 0,
    }
    assert progress["direct_bev_loader_access"] == receipt
    with pytest.raises(PermissionError, match="forbidden Direct BEV loader"):
        runner._access_counters(loader)


def test_selection_endpoint_population_is_exact_and_family_bound(
    monkeypatch,
) -> None:
    runner = _load_runner()
    endpoints = {
        "a": {
            "dataset_role": "checkpoint_selection",
            "family": runner.contract.ROUGH_RASTER_FAMILY,
            "scene_id": "rough",
        },
        "b": {
            "dataset_role": "checkpoint_selection",
            "family": runner.contract.ROUGH_RASTER_FAMILY,
            "scene_id": "rough",
        },
        "c": {
            "dataset_role": "checkpoint_selection",
            "family": "other",
            "scene_id": "plain",
        },
        "d": {
            "dataset_role": "checkpoint_selection",
            "family": "other",
            "scene_id": "plain",
        },
    }
    pairs = [
        {
            "dataset_role": "checkpoint_selection",
            "family": runner.contract.ROUGH_RASTER_FAMILY,
            "scene_id": "rough",
            "current_endpoint_sha256": "a",
            "next_endpoint_sha256": "b",
        },
        {
            "dataset_role": "checkpoint_selection",
            "family": "other",
            "scene_id": "plain",
            "current_endpoint_sha256": "c",
            "next_endpoint_sha256": "d",
        },
    ]
    ordered = sorted(endpoints)
    monkeypatch.setattr(runner.contract, "AGGREGATE_RASTER_ENDPOINT_COUNT", 4)
    monkeypatch.setattr(runner.contract, "ROUGH_RASTER_ENDPOINT_COUNT", 2)
    monkeypatch.setattr(
        runner.contract,
        "AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256",
        runner.contract.canonical_json_sha256(ordered),
    )
    aggregate, rough = runner._selection_endpoint_population(
        SimpleNamespace(endpoints=endpoints), pairs
    )
    assert aggregate == ordered
    assert rough == ["a", "b"]

    bad = [dict(pairs[0], scene_id="wrong"), pairs[1]]
    with pytest.raises(PermissionError, match="metadata changed"):
        runner._selection_endpoint_population(
            SimpleNamespace(endpoints=endpoints), bad
        )


class _FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


def _tiny_parameter_fixture(runner):
    rows = [
        ("encoder.weight", _FakeParameter(2, requires_grad=True)),
        ("bev_decoder.query", _FakeParameter(3, requires_grad=True)),
        ("state_head.weight", _FakeParameter(5, requires_grad=True)),
        ("predictor.weight", _FakeParameter(7, requires_grad=True)),
        ("target_encoder.weight", _FakeParameter(11, requires_grad=False)),
        (
            "target_bev_decoder.query",
            _FakeParameter(13, requires_grad=False),
        ),
        (
            "target_state_head.weight",
            _FakeParameter(17, requires_grad=False),
        ),
    ]
    names = {
        "encoder": ["encoder.weight"],
        "decoder_state": ["bev_decoder.query", "state_head.weight"],
        "predictor": ["predictor.weight"],
        "detached_target_encoder_decoder_state": [
            "target_encoder.weight",
            "target_bev_decoder.query",
            "target_state_head.weight",
        ],
    }
    counts = {
        "encoder": 2,
        "decoder_state": 8,
        "predictor": 7,
        "detached_target_encoder_decoder_state": 41,
    }
    inventory = {
        group: {
            "parameter_count": counts[group],
            "tensor_count": len(group_names),
            "ordered_parameter_name_sha256": (
                runner.contract.canonical_json_sha256(group_names)
            ),
        }
        for group, group_names in names.items()
    }
    inventory["total"] = {
        "parameter_count": sum(counts.values()),
        "tensor_count": len(rows),
    }
    return rows, inventory


def test_parameter_partition_and_optimizer_exclude_targets(monkeypatch) -> None:
    runner = _load_runner()
    rows, inventory = _tiny_parameter_fixture(runner)
    monkeypatch.setattr(runner.contract, "MODEL_PARAMETER_INVENTORY", inventory)
    partition = runner._parameter_partition(
        SimpleNamespace(named_parameters=lambda: iter(rows))
    )
    assert partition["receipt"] == inventory

    calls: list[object] = []

    class FakeAdamW:
        def __init__(self, groups, **kwargs) -> None:
            self.param_groups = groups
            calls.append(kwargs)

    runtime = SimpleNamespace(
        torch=SimpleNamespace(
            optim=SimpleNamespace(AdamW=FakeAdamW),
        )
    )
    optimizer, receipt = runner._build_optimizer(runtime, partition)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 3e-4]
    assert calls == [{
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-4,
    }]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    targets = {
        id(parameter)
        for _, parameter in partition["groups"][
            "detached_target_encoder_decoder_state"
        ]
    }
    assert optimized.isdisjoint(targets)
    assert receipt == {
        "name": "AdamW",
        "precision": "float32",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "encoder_learning_rate": 1e-4,
        "decoder_state_predictor_learning_rate": 3e-4,
        "encoder_decoder_state_joint_clip_norm": 1.0,
        "predictor_separate_clip_norm": 1.0,
        "target_parameters_excluded": True,
        "optimizer_group_count": 2,
    }


def test_initialization_helper_checks_rng_n320_sync_and_zero_residual() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._initialize_model)
    for required in (
        "fit.encoder.state_dict()",
        "torch.random.get_rng_state()",
        "torch.cuda.get_rng_state_all()",
        "model.target_encoder",
        "model._online_modules()",
        "model._target_modules()",
        "model.predictor.net[-1].weight",
        "model.predictor.net[-1].bias",
        "_parameter_partition(model)",
    ):
        assert required in source
    assert source.index("model = model_api.DirectEgocentricBevStateJepaV1(") < (
        source.index("model = model.to(device)")
    )


def test_evaluator_freezes_raw_formulas_and_all_selection_populations() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._evaluate_observation_impl)
    assert "g_sum += _scalar(result.G) * size" in source
    assert "j_sum += _scalar(result.J) * size" in source
    assert 'candidate_count_histogram != {"10": 60, "11": 435}' in source
    assert 'binding["same_action_row_count"]' in source
    assert 'binding["non_hold_row_count"]' in source
    assert "result.next_online_state_logits" in source
    assert '"wrong_rgb_correct_logits_reused_from_isolated_O_next": True' in source


class _FakeStateValue:
    def detach(self):
        return self

    def to(self, *, device):
        assert device == "cpu"
        return self

    def contiguous(self):
        return self

    def clone(self):
        return self


def test_snapshot_is_write_only_and_binds_preregistered_update(
    monkeypatch,
) -> None:
    runner = _load_runner()
    saved: list[dict[str, object]] = []
    writes: list[tuple[Path, bytes]] = []
    registrations: list[tuple[Path, dict[str, object]]] = []

    class FakeTorch:
        @staticmethod
        def save(value, stream) -> None:
            saved.append(value)
            stream.write(b"direct-bev-snapshot-fixture")

    monkeypatch.setattr(runner, "_state_sha", lambda runtime, value: "d" * 64)
    monkeypatch.setattr(
        runner,
        "_write_exclusive",
        lambda path, raw: writes.append((path, raw)),
    )
    monkeypatch.setattr(
        runner,
        "_register_output_semantic_metadata",
        lambda path, **metadata: registrations.append((path, metadata)),
    )
    model = SimpleNamespace(
        state_dict=lambda: {"z": _FakeStateValue(), "a": _FakeStateValue()}
    )
    receipt = runner._snapshot_model(
        SimpleNamespace(torch=FakeTorch()),
        model,
        Path("/virtual/output"),
        update=100,
        metadata={"gate": "fixture"},
    )
    assert writes == [(
        Path("/virtual/output/checkpoints/update_100.pt"),
        b"direct-bev-snapshot-fixture",
    )]
    assert list(saved[0]["model_state_dict"]) == ["a", "z"]
    assert saved[0]["metadata"] == {"gate": "fixture"}
    assert receipt["state_sha256"] == "d" * 64
    assert receipt["schedule_prefix_sha256"] == (
        runner.contract.SCHEDULE_PREFIX_SHA256[100]
    )
    assert receipt["write_only"] is True
    assert receipt["read_count_after_write"] == 0
    assert registrations[0][1]["update"] == 100
    assert registrations[0][1]["phase"] == "phase_a"
    with pytest.raises(ValueError, match="not preregistered"):
        runner._snapshot_model(
            SimpleNamespace(torch=FakeTorch()),
            model,
            Path("/virtual/output"),
            update=101,
            metadata={},
        )


def test_snapshot_uses_real_inherited_semantic_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the inherited registry instead of mocking its phase check."""

    runner = _load_runner()

    class FakeTorch:
        @staticmethod
        def save(value, stream) -> None:
            stream.write(b"direct-bev-real-registry-fixture")

    monkeypatch.setattr(runner, "_state_sha", lambda runtime, value: "e" * 64)
    runner._reset_output_binding_registry(tmp_path)
    runner._snapshot_model(
        SimpleNamespace(torch=FakeTorch()),
        SimpleNamespace(
            state_dict=lambda: {"only": _FakeStateValue()}
        ),
        tmp_path,
        update=100,
        metadata={"gate": "fixture"},
    )
    inventory = runner._terminal_inventory(tmp_path)
    binding = inventory["file_bindings"][0]
    assert binding["path"] == "checkpoints/update_100.pt"
    assert binding["phase"] == "phase_a"
    assert binding["update"] == 100


def test_cli_delegates_only_exact_explicit_one_shot_hashes(monkeypatch) -> None:
    runner = _load_runner()
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        runner,
        "run_parent",
        lambda **kwargs: calls.append(kwargs) or 2,
    )
    assert runner.main([
        "--run",
        "--review-sha256",
        "a" * 64,
        "--authorization-sha256",
        "b" * 64,
    ]) == 2
    assert calls == [{
        "review_file_sha256": "a" * 64,
        "authorization_file_sha256": "b" * 64,
    }]
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--run",
            "--review-sha256",
            "not-a-digest",
            "--authorization-sha256",
            "b" * 64,
        ])


def test_mocked_parent_lifecycle_terminalizes_post_reservation_failure(
    monkeypatch,
) -> None:
    runner = _load_runner()
    review = {"content_sha256": "1" * 64}
    authorization = {"content_sha256": "2" * 64}
    sources = {"source.py": "3" * 64}
    reservation = {"attempt_identity": "4" * 64}
    order: list[str] = []
    failure = RuntimeError("mocked post-reservation failure")

    def load_authority(review_sha256, authorization_sha256):
        order.append("authority")
        assert review_sha256 == "a" * 64
        assert authorization_sha256 == "b" * 64
        return review, b"review\n", authorization, b"authorization\n", sources

    def reserve(output_root, **kwargs):
        order.append("reserve")
        assert output_root == runner.ROOT / runner.contract.OUTPUT_ROOT_RELATIVE_PATH
        assert kwargs["review"] is review
        assert kwargs["authorization"] is authorization
        assert kwargs["sources"] is sources
        return reservation, b"reservation\n"

    def execute(**kwargs):
        order.append("execute")
        progress = kwargs["progress"]
        assert progress["stage"] == "reserved"
        assert progress["updates"] == 0
        assert progress["presentations"] == 0
        progress["stage"] = "mocked_failure_stage"
        raise failure

    def terminal_failure(output_root, observed_reservation, raw, **kwargs):
        order.append("terminal_failure")
        assert output_root == runner.ROOT / runner.contract.OUTPUT_ROOT_RELATIVE_PATH
        assert observed_reservation is reservation
        assert raw == b"reservation\n"
        assert kwargs["error"] is failure
        assert kwargs["progress"]["stage"] == "mocked_failure_stage"

    monkeypatch.setattr(runner, "_load_authority_pre_reservation", load_authority)
    monkeypatch.setattr(runner, "_reserve", reserve)
    monkeypatch.setattr(runner, "_execute_after_reservation", execute)
    monkeypatch.setattr(runner, "_terminal_failure", terminal_failure)
    with pytest.raises(RuntimeError, match="mocked post-reservation failure"):
        runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
        )
    assert order == ["authority", "reserve", "execute", "terminal_failure"]


def test_normal_lifecycle_orders_authority_gpu_release_and_receipts() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._execute_after_reservation)
    markers = (
        "contract.current_source_bindings(ROOT)",
        "_run_preflight_after_reservation(",
        "_load_post_reservation_stack(sources)",
        "_construct_raw_inputs_with_progress(",
        "_load_n320_with_progress(",
        "_train_probe(",
        'stage="Direct BEV terminal model release"',
        'model.to("cpu")',
        "gpu_active_elapsed_seconds = _check_gpu_time(",
        'output_root / "metrics.json"',
        'output_root / "artifact.json"',
        "inputs.rehash_consumed()",
        'output_root / "access.json"',
        'output_root / "result.json"',
        'output_root / "completed.json"',
        "_seal_terminal_with_repair(output_root)",
    )
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "contract.validate_failure_status_chain({" in source
    assert "return 0 if passed else 2" in source


def test_loader_access_counts_keep_endpoint_and_raster_cache_layers_separate() -> None:
    runner = _load_runner()
    loader = object.__new__(runner.DirectBevNarrowLoader)
    loader._counters = {
        "rgb_request_count": {
            "current": 3,
            "next": 4,
            "fixed_negative": 2,
            "endpoint": 5,
        },
        "rgb_cache_hit_count": {
            "current": 1,
            "next": 2,
            "fixed_negative": 1,
            "endpoint": 3,
        },
        "rgb_cache_miss_count": {
            "current": 2,
            "next": 2,
            "fixed_negative": 1,
            "endpoint": 2,
        },
        "rgb_physical_read_success_count": {
            "current": 2,
            "next": 2,
            "fixed_negative": 1,
            "endpoint": 2,
        },
        "raster_row_request_count": {
            "training": 6,
            "observation": 4,
            "endpoint_observation": 5,
        },
        "raster_row_cache_hit_count": {
            "training": 4,
            "observation": 3,
            "endpoint_observation": 4,
        },
        "raster_row_cache_miss_count": {
            "training": 2,
            "observation": 1,
            "endpoint_observation": 1,
        },
        "raster_underlying_array_cache_hit_count": 3,
        "raster_underlying_array_cache_miss_count": 1,
        "raster_physical_array_open_success_count": 1,
    }

    assert loader.model_facing_access_counts() == {
        "current_rgb_row_request_count": 3,
        "next_rgb_row_request_count": 4,
        "fixed_negative_rgb_row_request_count": 2,
        "endpoint_rgb_row_request_count": 5,
        "rgb_cache_hit_count": 7,
        "rgb_cache_miss_count": 7,
        "rgb_physical_file_open_count": 7,
        "raster_label_row_request_count": 15,
        "raster_label_row_cache_hit_count": 11,
        "raster_label_row_cache_miss_count": 4,
        "raster_label_underlying_array_cache_hit_count": 3,
        "raster_label_underlying_array_cache_miss_count": 1,
        "raster_label_physical_array_open_count": 1,
    }

    loader._counters["rgb_request_count"]["endpoint"] += 1
    with pytest.raises(RuntimeError, match="access accounting changed"):
        loader.model_facing_access_counts()
