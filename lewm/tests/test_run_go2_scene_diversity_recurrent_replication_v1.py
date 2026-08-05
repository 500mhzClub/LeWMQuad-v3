from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_go2_scene_diversity_recurrent_replication_v1 as runner


def test_runner_surface_binds_new_and_full_historical_source_closures() -> None:
    assert runner.AUTHORITY_SCHEMA == (
        "lewm_go2_scene_diversity_recurrent_replication_v1_execution_authority_v1"
    )
    assert runner.AUTHORITY_STATUS == (
        "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_V1"
    )
    assert runner.DEFAULT_COLLECTION_ROOT.parent == runner.DEFAULT_ATTEMPT_ROOT
    assert runner.SOURCE_PATHS["replication_runner"] == Path(runner.__file__).resolve()
    assert runner.SOURCE_PATHS["replication_runner_test"] == Path(__file__).resolve()
    assert {
        "recurrent_model",
        "recurrent_benchmark",
        "replication_benchmark",
        "replication_plan_builder",
        "replication_collector",
        "replication_authority_builder",
        "collection_supervisor",
        "calibration_supervisor",
    } <= set(runner.SOURCE_PATHS)
    assert len([name for name in runner.SOURCE_PATHS if name.startswith("collection_runtime_")]) > 70
    assert runner.expected_dino_v1()["checkpoint_binding"] == {
        "path": str(runner.DINO_CHECKPOINT.resolve()),
        "sha256": runner.DINO_CHECKPOINT_SHA256,
        "byte_count": runner.DINO_CHECKPOINT_BYTE_COUNT,
    }


def test_context_ledger_requires_durable_checkpoint_and_forbids_successors() -> None:
    ledger = runner.ContextOnlyLedgerV1()
    with pytest.raises(runner.SceneDiversityRunnerError, match="custody stage"):
        ledger.load_receipts("eval")
    ledger.load_receipts("train")
    ledger.open_role_index("train", "/train/index")
    ledger.open_state_receipt("train", "/train/state-0")
    ledger.open_render_receipt("train", "/train/render-0")
    ledger.open_rgb("train", "context", "train-context-0")
    with pytest.raises(runner.SceneDiversityRunnerError, match="structurally forbidden"):
        ledger.open_rgb("train", "successor", "train-successor-0")
    with pytest.raises(runner.SceneDiversityRunnerError, match="more than once"):
        ledger.open_rgb("train", "context", "train-context-0")
    ledger.checkpoint()
    with pytest.raises(runner.SceneDiversityRunnerError, match="custody stage"):
        ledger.open_rgb("train", "context", "late-train-context")
    ledger.load_receipts("eval")
    assert ledger.rgb_opens["train_successor"] == 0
    assert ledger.rgb_opens["eval_successor"] == 0


def test_context_ledger_finalizes_only_exact_32_scene_roles() -> None:
    ledger = runner.ContextOnlyLedgerV1()
    ledger.load_receipts("train")
    ledger.open_role_index("train", "/train/index")
    for index in range(128):
        ledger.open_state_receipt("train", f"/train/state-{index}")
    for index in range(32):
        ledger.open_render_receipt("train", f"/train/render-{index}")
    for index in range(384):
        ledger.open_rgb("train", "context", f"train-context-{index}")
    ledger.checkpoint()
    ledger.load_receipts("eval")
    ledger.open_role_index("eval", "/eval/index")
    for index in range(128):
        ledger.open_state_receipt("eval", f"/eval/state-{index}")
    for index in range(32):
        ledger.open_render_receipt("eval", f"/eval/render-{index}")
    for index in range(384):
        ledger.open_rgb("eval", "context", f"eval-context-{index}")

    audit = ledger.finalized()

    assert audit["state_receipt_opens"] == {"train": 128, "eval": 128}
    assert audit["render_receipt_opens"] == {"train": 32, "eval": 32}
    assert audit["unique_context_artifacts"] == 768
    assert audit["successor_rgb_open_count"] == 0


def _plan(role: str, *, overlapping: bool = False) -> SimpleNamespace:
    prefix = "shared" if overlapping else role
    states = tuple(
        SimpleNamespace(
            state_id=f"{prefix}-state-{index}",
            scene_id=f"{prefix}-scene-{index // 4}",
        )
        for index in range(128)
    )
    return SimpleNamespace(
        states=states,
        artifact_ids=tuple(f"{prefix}-artifact-{index}" for index in range(1536)),
    )


def test_role_disjointness_requires_32_scene_and_full_artifact_separation() -> None:
    report = runner.assert_role_disjointness_v1(_plan("train"), _plan("eval"))
    assert report["train_scene_count"] == 32
    assert report["eval_artifact_count"] == 1536
    with pytest.raises(runner.SceneDiversityRunnerError, match="not disjoint"):
        runner.assert_role_disjointness_v1(
            _plan("train", overlapping=True), _plan("eval", overlapping=True)
        )


def test_direct_context_reader_opens_only_admitted_rehashed_bytes(tmp_path: Path) -> None:
    raw = b"bound RGB bytes"
    rgb = tmp_path / "context.png"
    rgb.write_bytes(raw)
    frame = {
        "path": "context.png",
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    role = runner.RoleRuntimeDataV1(
        role="train",
        plan=None,
        physical_inputs=torch.empty(0),
        targets=torch.empty(0),
        history_commands=torch.empty(0),
        candidate_commands=torch.empty(0),
        relative_goals=torch.empty(0),
        dense_ranks=torch.empty(0, dtype=torch.long),
        context_artifact_ids=(),
        context_artifacts={"admitted": frame},
        collection_root=tmp_path,
        stored_rgb_bytes=len(raw),
        stored_rgb_frames=1,
        identity_sha256="a" * 64,
    )

    assert runner._read_context_rgb_v1(role, "admitted") == raw  # noqa: SLF001
    with pytest.raises(runner.SceneDiversityRunnerError, match="not admitted"):
        runner._read_context_rgb_v1(role, "successor")  # noqa: SLF001
    rgb.write_bytes(b"changed")
    with pytest.raises(runner.SceneDiversityRunnerError, match="binding changed"):
        runner._read_context_rgb_v1(role, "admitted")  # noqa: SLF001


def test_checkpoint_save_returns_the_durably_reopened_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = {
        "identity_sha256": "c" * 64,
        "state": {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)},
    }
    monkeypatch.setattr(
        runner.benchmark,
        "checkpoint_identity_v1",
        lambda value: str(value["identity_sha256"]),
    )

    reopened = runner._save_checkpoint_exclusive(  # noqa: SLF001
        tmp_path / "checkpoint.pt", checkpoint
    )

    assert reopened is not checkpoint
    assert reopened["identity_sha256"] == checkpoint["identity_sha256"]
    assert torch.equal(reopened["state"]["weight"], checkpoint["state"]["weight"])


def test_collection_preflights_before_consuming_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt_v1"
    collection = attempt / "collection"
    events: list[str] = []
    authority = {
        "attempt_root": str(attempt),
        "collection_root": str(collection),
        "plan_binding": {"path": "/plan", "sha256": "a" * 64, "byte_count": 1},
        "caps": {"wall_seconds": 10.0, "selected_device_vram_byte_ceiling": 1000},
    }
    binding = {"path": "/authority", "sha256": "b" * 64, "byte_count": 1}
    plan = {
        "execution_contract": {
            "graphics_preflight": {"vulkan_device_name": "bound-device"}
        }
    }
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda _index: "bound-device")

    monkeypatch.setattr(
        runner.calibration_supervisor,
        "_validate_python_invocation",
        lambda _plan: events.append("python") or "/python",
    )
    monkeypatch.setattr(
        runner.calibration_supervisor,
        "_child_environment",
        lambda _plan: {},
    )
    monkeypatch.setattr(
        runner.calibration_supervisor,
        "_run_graphics_preflight",
        lambda *_args, **_kwargs: events.append("graphics") or {"passed": True},
    )
    counter = tmp_path / "counter"
    counter.write_text("0")
    monkeypatch.setattr(
        runner.calibration_supervisor,
        "_selected_gpu_memory_files",
        lambda _plan: (counter, counter, "vendor", "device"),
    )

    class Sampler:
        baseline_used_bytes = 0
        peak_used_bytes = 0
        read_errors = 0
        interval_seconds = 0.02
        def start(self) -> None: pass
        def stop(self) -> dict[str, int]:
            return {"read_errors": 0, "peak_used_bytes": 0}

    monkeypatch.setattr(
        runner.calibration_supervisor, "_GlobalVramSampler", lambda *_a, **_k: Sampler()
    )
    monkeypatch.setattr(
        runner,
        "_reserve_attempt_v1",
        lambda *_a, **_k: (events.append("reserve"), attempt.mkdir()),
    )

    def collect(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("collector")
        collection.mkdir()
        (collection / "physics_result.json").write_text("{}")
        return {"exit_code": 0}

    monkeypatch.setattr(
        runner.collection_supervisor,
        "_run_collector_once_with_vram_ceiling",
        collect,
    )

    runner._collect_if_absent_v1(authority, binding, plan)  # noqa: SLF001

    assert events == ["python", "graphics", "reserve", "collector"]


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("authority_binding", {"path": "/wrong", "sha256": "f" * 64, "byte_count": 1}),
        ("source_bindings", {}),
        ("caps", {}),
        ("authorizes_retry_or_resume", True),
        ("failure", {"type": "unexpected"}),
    ),
)
def test_physics_index_requires_reviewed_lineage_and_terminal_flags(
    tmp_path: Path, field: str, changed: object
) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    plan_binding = runner.file_binding_v1(plan_file)
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "sha256": "a" * 64,
        "byte_count": 1,
    }
    sources = {"runner": {"path": "/runner", "sha256": "b" * 64, "byte_count": 1}}
    caps = {"wall_seconds": 7200.0}
    authority = {
        "collection_root": str(collection),
        "plan_binding": plan_binding,
        "source_bindings": sources,
        "caps": caps,
    }
    plan = {"states": [{"state_id": f"state-{index}"} for index in range(256)]}
    physics = {
        "schema": "lewm_go2_world_model_counterfactual_pilot_physics_result_v1",
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "failure": None,
        "authority_binding": authority_binding,
        "source_bindings": sources,
        "caps": caps,
        "expected_counts": runner.collector.EXPECTED_COUNTS,
        "observed_counts": runner.collector.EXPECTED_COUNTS,
        "plan_binding": {
            "path": plan_binding["path"],
            "file_sha256": plan_binding["sha256"],
            "byte_count": plan_binding["byte_count"],
        },
        "state_receipt_bindings": [{} for _ in range(256)],
        "render_receipt_bindings": [{} for _ in range(64)],
        "scene_metrics": [{} for _ in range(64)],
        "collection_wall_seconds": 1.0,
    }
    physics[field] = changed
    (collection / "physics_result.json").write_text(
        json.dumps(physics, sort_keys=True) + "\n"
    )

    with pytest.raises(runner.SceneDiversityRunnerError, match="contract changed"):
        runner._load_physics_index_v1(  # noqa: SLF001
            authority, authority_binding, plan
        )
