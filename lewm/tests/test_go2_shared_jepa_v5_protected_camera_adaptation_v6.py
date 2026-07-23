from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lewm.benchmarks import (
    go2_shared_jepa_v5_protected_camera_adaptation_v6 as contract,
)


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_test_protected_camera_adaptation_v6_runner", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(update: int, *, passed: int, shortfall: float) -> dict:
    return {
        "update": update,
        "path": contract.metric_sidecar_path(update),
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
    }


def _progress(
    update: int,
    *,
    passed: int,
    shortfall: float,
    worst: float = -1.0,
    loss: float = 1.0,
    all_nine: bool = False,
    baseline: dict | None = None,
) -> dict:
    return contract.control_decision_from_progress(
        update=update,
        passed_margin_count=passed,
        total_shortfall=shortfall,
        worst_margin=worst,
        aggregate_complete_v4_loss=loss,
        all_nine_physical_pass=all_nine,
        same_run_health_baseline=baseline,
    )


def _review(sources: dict[str, str]) -> dict:
    return contract.with_content_sha256(
        {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/camera_v6_roundtrip_reviewer",
            "reviewed_sources": sources,
            "predecessor": contract.predecessor_contract(),
            "science_contract": contract.science_contract(),
            "science_delta": contract.science_delta(),
            "evidence": contract.evidence_contract(),
            "visibility_preflight": contract.visibility_preflight_contract(),
            "reporting_contract": contract.reporting_contract(),
            "control_contract": contract.control_contract(),
            "source_only": True,
            "findings": [],
            "authority": dict(contract.REVIEW_AUTHORITY),
        }
    )


def test_exact_v4_loss_substitution_over_unchanged_v5_8k_science() -> None:
    expected = copy.deepcopy(contract._v5_contract.science_contract())
    expected["camera_loss"] = {
        **expected["camera_loss"],
        "source": contract.TAIL_DEPTH_LOSS_RELATIVE_PATH,
        "terms": [
            "hierarchical_first_hit_nll",
            "tail_depth_p95_cvar",
            "ground_clear_distance_state_balanced_bce",
            "derived_raster_hierarchical_bce",
            "derived_raster_cell_nll",
        ],
        "tail_depth_p95_cvar": copy.deepcopy(
            contract.TAIL_DEPTH_DEFINITION
        ),
    }
    assert contract.science_contract() == expected
    assert contract.science_contract()["schedule"] == (
        contract._v5_contract.science_contract()["schedule"]
    )
    assert contract.science_contract()["optimizer"] == (
        contract._v5_contract.science_contract()["optimizer"]
    )
    assert contract.MAXIMUM_UPDATE == 8_000
    assert contract.PRESENTATION_COUNT == 128_000
    assert contract.CHECKPOINT_UPDATES == (
        100, 400, 1_000, 4_000, 6_000, 8_000
    )
    assert contract.TRAINABLE_PARAMETER_PREFIXES == (
        "encoder.", "evidence_head."
    )
    assert contract.FROZEN_STATE_PREFIXES == (
        "bev_decoder.", "predictor.", "occupancy_head.",
        "target_encoder.", "target_bev_decoder.",
    )
    assert contract.EXPECTED_PARAMETER_COUNTS == {
        "encoder": 2_747_520,
        "evidence_head": 357_993,
    }
    assert contract.EXPECTED_PARAMETER_TENSOR_COUNTS == {
        "encoder": 78,
        "evidence_head": 14,
    }
    delta = contract.science_delta()
    assert delta["training_science_change_count"] == 1
    assert delta[
        "architecture_data_sampling_seed_initialization_optimizer_or_coefficient_changes"
    ] == []
    assert delta["other_training_science_changes"] == []


def test_same_run_health_checks_are_coarse_or_rules_with_equality() -> None:
    u100 = _baseline(100, passed=61, shortfall=100.0)
    assert _progress(
        400, passed=66, shortfall=100.0, baseline=u100
    )["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(
        400, passed=61, shortfall=90.0, baseline=u100
    )["action"] == contract.CONTROL_ACTION_CONTINUE
    stopped = _progress(
        400, passed=65, shortfall=90.0000001, baseline=u100
    )
    assert stopped["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert stopped["terminal_stage"].endswith("update_400")

    u400 = _baseline(400, passed=84, shortfall=60.0)
    assert _progress(
        1_000, passed=89, shortfall=60.0, baseline=u400
    )["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(
        1_000, passed=84, shortfall=54.0, baseline=u400
    )["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(
        1_000, passed=88, shortfall=54.1, baseline=u400
    )["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    with pytest.raises(PermissionError, match="baseline"):
        _progress(400, passed=100, shortfall=1.0)
    with pytest.raises(PermissionError, match="baseline"):
        _progress(
            1_000,
            passed=100,
            shortfall=1.0,
            baseline=_baseline(100, passed=1, shortfall=1.0),
        )


def test_informational_tail_checkpoints_and_exact_terminal_gate() -> None:
    assert _progress(
        100, passed=0, shortfall=1_000.0
    )["action"] == contract.CONTROL_ACTION_CONTINUE
    for update in (4_000, 6_000):
        assert _progress(
            update, passed=0, shortfall=1_000.0
        )["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(
        8_000, passed=188, shortfall=0.1
    )["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    qualified = _progress(
        8_000,
        passed=189,
        shortfall=0.0,
        worst=0.0,
        all_nine=True,
    )
    assert qualified["action"] == contract.CONTROL_ACTION_QUALIFY
    assert qualified["qualifies"] is True


def test_fixed_source_and_evidence_hashes_without_payload_access() -> None:
    assert contract.PREREGISTRATION_COMMIT == (
        "48712ffe5379324847f027d10c2305e82b351397"
    )
    for path, expected in contract.V5_SOURCE_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    for path, expected in contract.FIXED_EVIDENCE_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
        assert (ROOT / path).stat().st_size == (
            contract.FIXED_EVIDENCE_BYTE_COUNT[path]
        )
    assert contract.FIXED_EVIDENCE_SHA256[
        contract.TAIL_DEPTH_LOSS_RELATIVE_PATH
    ] == "6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384"
    assert set(contract.V5_SOURCE_SHA256) <= set(contract.SOURCE_PATHS)
    assert set(contract.FIXED_EVIDENCE_SHA256) <= set(contract.SOURCE_PATHS)


def test_review_authorization_and_visibility_preflight_round_trip() -> None:
    sources = {path: "c" * 64 for path in contract.SOURCE_PATHS}
    review = _review(sources)
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    parsed_review = contract.parse_canonical_json(
        review_raw, name="round-trip Camera V6 review"
    )
    assert contract.validate_review(
        parsed_review, expected_sources=sources
    ) == parsed_review
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = contract.with_content_sha256(
        {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status":
                "authorized_one_exact_camera_v6_final_fresh_update0_"
                "tail_depth_8k_attempt",
            "authorizer": "/root/camera_v6_roundtrip_authorizer",
            "independent_review": review_binding,
            "predecessor": contract.predecessor_contract(),
            "raw": contract.expected_raw_authority(),
            "camera": contract.expected_camera_authority(),
            "experiment": contract.science_contract(),
            "science_delta": contract.science_delta(),
            "evidence": contract.evidence_contract(),
            "visibility_preflight": contract.visibility_preflight_contract(),
            "reporting_contract": contract.reporting_contract(),
            "control_contract": contract.control_contract(),
            "authority": dict(contract.EXECUTION_AUTHORITY),
        }
    )
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == authorization
    changed = dict(authorization)
    changed["visibility_preflight"] = {
        **contract.visibility_preflight_contract(),
        "visible_device_count": 2,
    }
    changed.pop("content_sha256")
    with pytest.raises(PermissionError):
        contract.validate_authorization(
            contract.with_content_sha256(changed),
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_thin_runner_installs_and_restores_only_v6_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    originals = (
        runner._v5.contract,
        runner._v5._train,
        runner._v5._publish_metric_sidecar,
        runner._v5._publish_training,
        runner._v5._access_receipt,
    )

    def fail_parent(**kwargs):
        assert kwargs == {
            "review_file_sha256": "a" * 64,
            "authorization_file_sha256": "b" * 64,
        }
        assert runner._v5.contract is runner.contract
        assert runner._v5._train is runner._train
        assert (
            runner._v5._publish_metric_sidecar
            is runner._publish_metric_sidecar
        )
        assert runner._v5._publish_training is runner._publish_training
        assert runner._v5._access_receipt is runner._access_receipt
        raise RuntimeError("synthetic Camera V6 parent failure")

    monkeypatch.setattr(runner, "_BASE_V5_RUN_PARENT", fail_parent)
    with pytest.raises(RuntimeError, match="synthetic Camera V6"):
        runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
        )
    assert (
        runner._v5.contract,
        runner._v5._train,
        runner._v5._publish_metric_sidecar,
        runner._v5._publish_training,
        runner._v5._access_receipt,
    ) == originals


def test_loss_wrapper_replaces_only_v4_slot_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner()

    def old_loss(*args, **kwargs):
        return None

    def tail_loss(*args, **kwargs):
        return None

    runtime = SimpleNamespace(
        loss_adapter=SimpleNamespace(
            observable_camera_ray_v4_loss_v4=old_loss
        )
    )
    original_components = runner._v1_runner._camera_components
    monkeypatch.setattr(
        runner,
        "_load_exact_tail_loss",
        lambda: SimpleNamespace(
            observable_camera_ray_v4_tail_depth_loss_v4=tail_loss
        ),
    )

    def fake_base(*args):
        assert runtime.loss_adapter.observable_camera_ray_v4_loss_v4 is tail_loss
        assert runner._v1_runner._camera_components is runner._camera_components
        return (
            [
                {
                    "losses": {
                        "current_tail_depth_p95_cvar": 1.0,
                        "next_tail_depth_p95_cvar": 1.0,
                    }
                }
            ],
        )

    monkeypatch.setattr(runner, "_BASE_V5_TRAIN", fake_base)
    result = runner._train(
        runtime,
        object(),
        object(),
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        object(),
        object(),
        tmp_path,
    )
    assert len(result[0]) == 1
    assert runtime.loss_adapter.observable_camera_ray_v4_loss_v4 is old_loss
    assert runner._v1_runner._camera_components is original_components


def test_real_tail_loader_preserves_reviewed_package_import_and_hash() -> None:
    pytest.importorskip("torch")
    runner = _runner()
    original_sys_path = list(sys.path)
    tail = runner._load_exact_tail_loss()
    assert sys.path == original_sys_path
    expected = (ROOT / contract.TAIL_DEPTH_LOSS_RELATIVE_PATH).resolve()
    resolved = Path(tail.__file__).resolve()
    assert tail.__package__ == "lewm.models"
    assert resolved == expected
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == (
        contract.FIXED_EVIDENCE_SHA256[
            contract.TAIL_DEPTH_LOSS_RELATIVE_PATH
        ]
    )
    assert callable(tail.observable_camera_ray_v4_tail_depth_loss_v4)


def test_sidecar_overlay_delegates_atomic_publication_and_never_uses_pt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner()
    baseline = _baseline(100, passed=61, shortfall=100.0)
    monkeypatch.setattr(
        runner,
        "_same_run_health_baseline",
        lambda output_root, *, update: baseline,
    )
    observed = {}

    def fake_publish(output_root, *, update, checkpoint, metric):
        observed.update(
            {
                "output_root": output_root,
                "update": update,
                "checkpoint": checkpoint,
                "metric": dict(metric),
            }
        )
        return {"path": contract.metric_sidecar_path(update)}

    monkeypatch.setattr(
        runner, "_BASE_V3_PUBLISH_METRIC_SIDECAR", fake_publish
    )
    metric = {}
    result = runner._publish_metric_sidecar(
        tmp_path,
        update=400,
        checkpoint={"path": "checkpoints/update_400.pt"},
        metric=metric,
    )
    assert result["path"].endswith(".metrics.json")
    assert observed["metric"]["same_run_health_baseline"] == baseline
    assert "update_4000_control_baseline" not in observed["metric"]
    reporting = contract.reporting_contract()
    assert reporting["sidecar_is_only_live_checkpoint_readiness_marker"] is True
    assert reporting["checkpoint_file_existence_is_not_a_readiness_marker"] is True
    assert "chmod_0444" in reporting["publication_mechanism"]
    assert "hard_link" in reporting["publication_mechanism"]


def test_isolated_runner_import_is_accelerator_and_payload_free() -> None:
    output = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    before = output.exists()
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"""
import importlib.util,json,sys
p={str(path)!r}
s=importlib.util.spec_from_file_location('_isolated_camera_v6',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(json.dumps(sorted(set(sys.modules)&{{'torch','numpy','PIL','cv2'}})))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == []
    assert output.exists() is before
