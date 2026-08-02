from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from lewm.tests.test_go2_world_model_counterfactual_pilot_v1 import _smoke_plan


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_BUILDER = (
    ROOT
    / "scripts"
    / "build_go2_world_model_counterfactual_calibration_authority_v1.py"
)
PLAN_BUILDER = (
    ROOT / "scripts" / "build_go2_world_model_counterfactual_calibration_plan_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(path: str, token: str) -> dict[str, object]:
    return {
        "path": path,
        "file_sha256": (token * 64)[:64],
        "byte_count": 1,
    }


def _plan(tmp_path: Path):
    builder = _load(PLAN_BUILDER, "calibration_plan_for_authority_test")
    builder.REPO_ROOT = tmp_path
    smoke = _smoke_plan(tmp_path / "runtime")
    scenes = []
    for family_index, family in enumerate(pilot.FAMILIES):
        scene_root = tmp_path / "scenes" / family
        scene_root.mkdir(parents=True)
        manifest = scene_root / "manifest.json"
        genesis = scene_root / "genesis_scene.json"
        target = [1.0, float(family_index)]
        manifest.write_text(json.dumps({
            "scene_id": f"calibration-{family_index}",
            "family": family,
            "landmarks": [{
                "object_id": "target-000",
                "center_xyz_m": [*target, 0.5],
            }],
        }))
        genesis.write_text("{}")
        scenes.append({
            "family": family,
            "scene_id": f"calibration-{family_index}",
            "scene_manifest_binding": pilot.file_binding(manifest),
            "scene_genesis_binding": pilot.file_binding(genesis),
            "states": [
                {
                    "state_id": f"calibration-{family_index}:{state_index}",
                    "history_action_ids": [state_index, state_index + 1],
                    "target_xy_m": target,
                }
                for state_index in range(2)
            ],
        })
    return builder.build_calibration_plan_v1(
        attempt_id="calibration-attempt-v2",
        output_root=(tmp_path / ".generated/dev/calibration-attempt-v2").resolve(),
        scene_panel={"schema": builder.SCENE_PANEL_SCHEMA, "scenes": scenes},
        runtime_contract={
            "schema": builder.RUNTIME_CONTRACT_SCHEMA,
            "runtime_bindings": smoke["runtime_bindings"],
            "execution_contract": smoke["execution_contract"],
        },
    )


def test_authority_builder_emits_exact_160_branch_one_shot(tmp_path: Path) -> None:
    builder = _load(AUTHORITY_BUILDER, "calibration_authority_builder_test")
    plan = _plan(tmp_path)
    builder.collector.CALIBRATION_SUCCESSOR_ATTEMPT_ID = plan["attempt_id"]
    builder.collector.CALIBRATION_SUCCESSOR_ROOT = plan["output_root"]
    plan_binding = _binding(str((tmp_path / "plan.json").resolve()), "a")
    predecessor_path = (
        ROOT / builder.collector.CALIBRATION_PREDECESSOR_FAILURE_RELATIVE
    )
    predecessor_binding = pilot.file_binding(predecessor_path)
    source_bindings = [
        {
            "name": name,
            "binding": (
                predecessor_binding
                if name == "predecessor_terminal_failure_result"
                else _binding(str((ROOT / relative).resolve()), "b")
            ),
        }
        for name, relative in builder.canonical_runtime_source_paths_v1().items()
    ]
    review = {
        "schema": pilot.SOURCE_REVIEW_SCHEMA,
        "status": "PASS_SOURCE_ONLY_NOT_AUTHORITY",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": "c" * 40,
        "reviewed_source_bindings": source_bindings,
        "remaining_findings": [],
        "reviewer": {
            "identity": "independent-test-reviewer",
            "independence_basis": "unit fixture",
        },
        "reviewed_at": "2026-08-02T12:00:00+00:00",
        "review_method": ["reviewed exact unit fixture"],
        "test_evidence": ["focused fixture passed"],
        "accepted_limitations": ["metadata-only unit fixture"],
    }
    authority = builder.build_authority_v1(
        plan=plan,
        plan_binding=plan_binding,
        review=review,
        review_binding=_binding(str((tmp_path / "review.json").resolve()), "d"),
        predecessor_failure_binding=predecessor_binding,
        authorizer_identity="explicit-test-authorizer",
        authorizer_basis="unit fixture only",
        issued_at="2026-08-02T12:01:00+00:00",
        terminal_reviewer="terminal-test-reviewer",
        wall_seconds=3600.0,
        platform_basis="unit fixture asserts resolved gates",
    )
    assert authority["status"] == builder.AUTHORITY_STATUS
    assert authority["caps"]["total_branches"] == 160
    assert authority["caps"]["candidate_branches"] == 144
    assert authority["caps"]["sentinel_branches"] == 16
    assert authority["attempt"]["maximum_attempts"] == 1
    assert authority["attempt"]["retry"] is False
    assert authority["predecessor_failure_binding"] == predecessor_binding
    assert authority["platform_gate_disposition"][
        "outputs_eligible_for_training_after_receipt_join"
    ] is False
    assert authority["external_supervisor"]["source_binding"]["path"].endswith(
        "run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    )

    mutations = []
    changed_plan = copy.deepcopy(authority)
    changed_plan["plan_binding"]["file_sha256"] = "e" * 64
    mutations.append((changed_plan, "selected plan"))
    changed_caps = copy.deepcopy(authority)
    changed_caps["caps"]["total_branches"] = 159
    mutations.append((changed_caps, "work caps"))
    changed_platform = copy.deepcopy(authority)
    changed_platform["platform_gate_disposition"][
        "platform_hard_gates_resolved"
    ] = False
    mutations.append((changed_platform, "platform-gate"))
    changed_supervisor = copy.deepcopy(authority)
    changed_supervisor["external_supervisor"]["source_binding"][
        "file_sha256"
    ] = "f" * 64
    mutations.append((changed_supervisor, "reviewed closure"))
    changed_predecessor = copy.deepcopy(authority)
    changed_predecessor["predecessor_failure_binding"]["file_sha256"] = "f" * 64
    mutations.append((changed_predecessor, "predecessor failure"))
    for mutated, message in mutations:
        with pytest.raises(pilot.PilotContractError, match=message):
            builder.collector._validate_non_smoke_authority(
                mutated,
                plan=plan,
                plan_binding=plan_binding,
            )

    predecessor_retry_plan = copy.deepcopy(plan)
    predecessor_retry_plan["attempt_id"] = (
        builder.collector.CALIBRATION_PREDECESSOR_ATTEMPT_ID
    )
    predecessor_retry_authority = copy.deepcopy(authority)
    predecessor_retry_authority["attempt"]["id"] = (
        builder.collector.CALIBRATION_PREDECESSOR_ATTEMPT_ID
    )
    with pytest.raises(pilot.PilotContractError, match="exact V2 successor identity"):
        builder.collector._validate_non_smoke_authority(
            predecessor_retry_authority,
            plan=pilot.validate_plan(predecessor_retry_plan),
            plan_binding=plan_binding,
        )

    predecessor_root_plan = copy.deepcopy(plan)
    predecessor_root_plan["output_root"] = (
        builder.collector.CALIBRATION_PREDECESSOR_ROOT
    )
    predecessor_root_authority = copy.deepcopy(authority)
    predecessor_root_authority["attempt"]["root"] = (
        builder.collector.CALIBRATION_PREDECESSOR_ROOT
    )
    with pytest.raises(pilot.PilotContractError, match="exact V2 successor identity"):
        builder.collector._validate_non_smoke_authority(
            predecessor_root_authority,
            plan=pilot.validate_plan(predecessor_root_plan),
            plan_binding=plan_binding,
        )

    predecessor_artifact_plan = copy.deepcopy(plan)
    runtime_name = next(iter(predecessor_artifact_plan["runtime_bindings"]))
    predecessor_artifact_plan["runtime_bindings"][runtime_name]["path"] = str(
        Path(builder.collector.CALIBRATION_PREDECESSOR_ROOT) / "artifact.bin"
    )
    with pytest.raises(pilot.PilotContractError, match="reuse V1 root artifacts"):
        builder.collector._validate_non_smoke_authority(
            authority,
            plan=predecessor_artifact_plan,
            plan_binding=plan_binding,
        )


def test_source_closure_uses_calibration_not_smoke_supervisor() -> None:
    builder = _load(AUTHORITY_BUILDER, "calibration_authority_sources_test")
    paths = builder.canonical_runtime_source_paths_v1()
    assert paths["external_supervisor"].endswith(
        "counterfactual_calibration_authorized_v1.py"
    )
    assert "counterfactual_smoke_authorized_v1.py" not in paths.values()
    assert paths["predecessor_terminal_failure_result"].endswith(
        "counterfactual_calibration_v1_terminal_failure_result_2026-08-02.json"
    )


def test_review_template_cannot_self_assert_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load(AUTHORITY_BUILDER, "calibration_review_template_test")
    monkeypatch.setattr(
        builder,
        "committed_source_bindings_v1",
        lambda _commit: [{"name": "fixture", "binding": _binding("/fixture", "a")}],
    )
    template = builder.build_source_review_template_v1(source_commit="b" * 40)
    assert template["status"] == "PENDING_INDEPENDENT_REVIEW"
    assert template["authority_granted_by_this_document"] is False
    assert template["remaining_findings"] == ["INDEPENDENT_REVIEW_REQUIRED"]
    with pytest.raises(pilot.PilotContractError):
        pilot.validate_source_review(
            template,
            authority={
                "source_commit": "b" * 40,
                "source_bindings": template["reviewed_source_bindings"],
            },
        )


def test_help_exposes_review_and_authority_subcommands() -> None:
    completed = subprocess.run(
        [sys.executable, str(AUTHORITY_BUILDER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "review" in completed.stdout
    assert "authority" in completed.stdout
    authority_help = subprocess.run(
        [sys.executable, str(AUTHORITY_BUILDER), "authority", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--predecessor-failure" in authority_help.stdout
