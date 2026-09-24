from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_go2_shared_jepa_v5_gate.py"
FINALIZER = ROOT / "scripts/finalize_go2_shared_jepa_v5_gate.py"
PUBLISHER = ROOT / "scripts/publish_go2_shared_jepa_v5_checkpoint.py"
CORE = ROOT / "scripts/go2_shared_jepa_v5_one_shot.py"
LAUNCHER = ROOT / "scripts/go2_shared_jepa_v5_launcher.py"

GATE_METRICS = {
    "g2": (
        "aggregate_physical_gate_pass_fraction",
        "per_family_physical_gate_pass_fraction",
        "jepa_health_gate_pass_fraction",
        "counterfactual_gate_pass_fraction",
    ),
    "g3": (
        "exact_morphology_equivalence_pass_fraction",
        "configuration_runtime_gate_pass_fraction",
        "safety_gate_pass_fraction",
        "task_gate_pass_fraction",
    ),
}

AUTHORITY_SCHEMA = "lewm_go2_shared_jepa_v5_stage_authority_v3"
RUNNER_REVISIONS = {"g2": "runner_g2_inputs_v2", "g3": "runner_g3_inputs_v2"}
FINALIZER_REVISIONS = {
    "g2": "finalizer_g2_evidence_v2",
    "g3": "finalizer_g3_evidence_v2",
}
PUBLISHER_REVISIONS = {
    "g2-candidate": "publisher_g2_candidate_v2",
    "full-promotion": "publisher_full_promotion_v2",
}


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content(core: dict) -> dict:
    return {**core, "content_sha256": hashlib.sha256(_bytes(core)).hexdigest()}


def _write_json(root: Path, path: Path, core: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _bytes(_content(core)) + b"\n"
    path.write_bytes(encoded)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _file_spec(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(
    tmp_path: Path,
    *,
    self_reported: bool = False,
    failing: bool = False,
) -> dict:
    root = tmp_path / "synthetic-root"
    for relative in (
        "registry/g2",
        "registry/g3",
        "outputs/g2/outcomes",
        "outputs/g3/outcomes",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    role_path = root / "artifacts/roles.json"
    roles = {
        "train": ["train-a"],
        "g2": ["g2-a", "g2-b"],
        "g3": ["g3-a"],
    }
    role_spec = _write_json(
        root,
        role_path,
        {
            "schema": "lewm_go2_shared_jepa_dataset_roles_v7",
            "protocol_generation": "cpu-synthetic-v1",
            "roles": roles,
            "scene_families": {
                "train-a": "family-a",
                "g2-a": "family-a",
                "g2-b": "family-b",
                "g3-a": "family-a",
            },
        },
    )

    checkpoint_path = root / "artifacts/checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b'{"bias":0}\n')
    checkpoint_spec = _file_spec(root, checkpoint_path)

    package_path = root / "runtime/lewm_init.py"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_text("CAPTURED_PACKAGE = True\n", encoding="ascii")
    engine_path = root / "runtime/inference.py"
    engine_path.write_text(
        "import json\n"
        "def load_checkpoint(encoded):\n"
        "    value = json.loads(encoded.decode('ascii'))\n"
        "    value['calls'] = 0\n"
        "    return value\n"
        "def infer_one(model, model_input):\n"
        "    model['calls'] += 1\n"
        "    return {'call_index': model['calls'], 'passed': bool(model_input['passed'])}\n",
        encoding="ascii",
    )

    scene_specs: dict[str, dict[str, dict[str, str]]] = {"g2": {}, "g3": {}}
    scene_counts = {"g2-a": 2, "g2-b": 1, "g3-a": 2}
    families = {"g2-a": "family-a", "g2-b": "family-b", "g3-a": "family-a"}
    for gate in ("g2", "g3"):
        for scene_id in roles[gate]:
            instances = [
                {
                    "instance_id": f"{scene_id}-instance-{index}",
                    "model_input": {"passed": True},
                    "targets": {},
                }
                for index in range(1, scene_counts[scene_id] + 1)
            ]
            if failing and gate == "g2" and scene_id == "g2-a":
                instances[0]["model_input"]["passed"] = False
            if self_reported:
                instances[0]["metric_outcomes"] = {
                    name: True for name in GATE_METRICS[gate]
                }
            path = root / f"inputs/{gate}/{scene_id}.json"
            scene_specs[gate][scene_id] = _write_json(
                root,
                path,
                {
                    "schema": "lewm_go2_shared_jepa_raw_scene_input_v1",
                    "scene_id": scene_id,
                    "family": families[scene_id],
                    "instances": instances,
                },
            )

    rules = {
        gate: {
            metric: {"output_path": ["passed"], "operator": "is_true"}
            for metric in GATE_METRICS[gate]
        }
        for gate in ("g2", "g3")
    }
    return {
        "repository_root": str(root.resolve()),
        "protocol_generation": "cpu-synthetic-v1",
        "attempt_registry_path": "registry",
        "dataset_role_manifest": role_spec,
        "evaluated_checkpoint": checkpoint_spec,
        "runtime_modules": {
            "lewm": {**_file_spec(root, package_path), "package": True},
            "lewm.synthetic_v5_inference": {
                **_file_spec(root, engine_path),
                "package": False,
            },
        },
        "inference_entry_module": "lewm.synthetic_v5_inference",
        "gate_specs": {
            gate: {
                "scene_inputs": scene_specs[gate],
                "outcome_paths": {
                    scene_id: f"outputs/{gate}/outcomes/{scene_id}.json"
                    for scene_id in roles[gate]
                },
                "ledger_path": f"outputs/{gate}/runner_ledger.json",
                "metric_rules": rules[gate],
            }
            for gate in ("g2", "g3")
        },
        "final_report_paths": {
            gate: f"outputs/{gate}/final_report.json" for gate in ("g2", "g3")
        },
        "publication_paths": {
            "g2-candidate": "outputs/g2_candidate_publication.json",
            "full-promotion": "outputs/full_promotion.json",
        },
    }


def _rewrite_authority(path: Path, authority: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = dict(authority)
    core.pop("content_sha256", None)
    path.write_bytes(_bytes(_content(core)) + b"\n")


def _command(
    program: Path,
    authority: Path,
    *args: str,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "LEWM_V5_SYNTHETIC_AUTHORITY_PATH": str(authority),
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, "-I", str(program), *args],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _stage_authority(
    fixture: dict,
    revision: str,
    fields: dict,
) -> tuple[Path, dict]:
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "lifecycle_revision": revision,
        "synthetic_only": True,
        "repository_root": fixture["repository_root"],
        "protocol_generation": fixture["protocol_generation"],
        **copy.deepcopy(fields),
    }
    root = Path(fixture["repository_root"])
    path = root / f"authorities/{revision}.json"
    _rewrite_authority(path, authority)
    return path, authority


def _runner_authority(
    fixture: dict,
    gate: str,
    *,
    g2_candidate_publisher_authority: dict[str, str] | None = None,
    g2_candidate_publication: dict[str, str] | None = None,
) -> tuple[Path, dict]:
    fields = {
        "gate": gate,
        "attempt_registry_path": fixture["attempt_registry_path"],
        "dataset_role_manifest": fixture["dataset_role_manifest"],
        "evaluated_checkpoint": fixture["evaluated_checkpoint"],
        "runtime_modules": fixture["runtime_modules"],
        "inference_entry_module": fixture["inference_entry_module"],
        "gate_spec": fixture["gate_specs"][gate],
    }
    if gate == "g3":
        if (
            g2_candidate_publisher_authority is None
            or g2_candidate_publication is None
        ):
            raise ValueError(
                "G3 runner fixture requires candidate authority and publication"
            )
        fields["g2_candidate_publisher_authority"] = (
            g2_candidate_publisher_authority
        )
        fields["g2_candidate_publication"] = g2_candidate_publication
        publication = json.loads(
            (
                Path(fixture["repository_root"])
                / g2_candidate_publication["path"]
            ).read_bytes()
        )
        fields["g2_candidate_publisher_execution_identity"] = publication[
            "publisher_execution_identity"
        ]
    return _stage_authority(fixture, RUNNER_REVISIONS[gate], fields)


def _finalizer_authority(
    fixture: dict,
    gate: str,
    runner_authority_path: Path,
    runner_authority: dict,
) -> tuple[Path, dict]:
    root = Path(fixture["repository_root"])
    gate_spec = runner_authority["gate_spec"]
    ledger = json.loads((root / gate_spec["ledger_path"]).read_bytes())
    return _stage_authority(
        fixture,
        FINALIZER_REVISIONS[gate],
        {
            "gate": gate,
            "runner_authority": _file_spec(root, runner_authority_path),
            "runner_ledger": _file_spec(root, root / gate_spec["ledger_path"]),
            "outcome_files": {
                scene_id: _file_spec(root, root / relative)
                for scene_id, relative in gate_spec["outcome_paths"].items()
            },
            "final_report_path": fixture["final_report_paths"][gate],
            "runner_execution_identity": ledger["runner_execution_identity"],
        },
    )


def _publisher_authority(
    fixture: dict,
    mode: str,
    finalized: dict[str, tuple[Path, dict]],
) -> tuple[Path, dict]:
    root = Path(fixture["repository_root"])
    bindings = {}
    for gate, (authority_path, authority) in sorted(finalized.items()):
        report_path = root / authority["final_report_path"]
        report = json.loads(report_path.read_bytes())
        bindings[gate] = {
            "finalizer_authority": _file_spec(root, authority_path),
            "fixed_final_report": _file_spec(root, report_path),
            "finalizer_execution_identity": report[
                "finalizer_execution_identity"
            ],
        }
    return _stage_authority(
        fixture,
        PUBLISHER_REVISIONS[mode],
        {
            "finalized_gates": bindings,
            "publication_path": fixture["publication_paths"][mode],
        },
    )


def _run_and_finalize(
    fixture: dict,
    gate: str,
    *,
    g2_candidate_publisher_authority: dict[str, str] | None = None,
    g2_candidate_publication: dict[str, str] | None = None,
) -> tuple[Path, dict, Path, dict]:
    runner_path, runner = _runner_authority(
        fixture,
        gate,
        g2_candidate_publisher_authority=g2_candidate_publisher_authority,
        g2_candidate_publication=g2_candidate_publication,
    )
    run = _command(RUNNER, runner_path, gate)
    assert run.returncode == 0, run.stderr
    finalizer_path, finalizer = _finalizer_authority(
        fixture,
        gate,
        runner_path,
        runner,
    )
    finalized = _command(FINALIZER, finalizer_path, gate)
    assert finalized.returncode == 0, finalized.stderr
    return runner_path, runner, finalizer_path, finalizer


def test_production_clis_fail_before_artifact_access_while_six_identities_are_unset() -> None:
    bindings = {
        RUNNER: (
            "CANONICAL_G2_RUNNER_AUTHORITY_FILE_SHA256",
            "CANONICAL_G3_RUNNER_AUTHORITY_FILE_SHA256",
        ),
        FINALIZER: (
            "CANONICAL_G2_FINALIZER_AUTHORITY_FILE_SHA256",
            "CANONICAL_G3_FINALIZER_AUTHORITY_FILE_SHA256",
        ),
        PUBLISHER: (
            "CANONICAL_G2_CANDIDATE_PUBLISHER_AUTHORITY_FILE_SHA256",
            "CANONICAL_FULL_PROMOTION_PUBLISHER_AUTHORITY_FILE_SHA256",
        ),
    }
    for program, names in bindings.items():
        source = program.read_text(encoding="utf-8")
        for name in names:
            assert f"{name}: str | None = None" in source
    environment = dict(os.environ)
    environment.pop("LEWM_V5_SYNTHETIC_AUTHORITY_PATH", None)
    for command in (
            [sys.executable, "-I", str(RUNNER), "g2"],
            [sys.executable, "-I", str(FINALIZER), "g2"],
            [sys.executable, "-I", str(PUBLISHER), "g2-candidate"],
            [sys.executable, "-I", str(PUBLISHER), "full-promotion"],
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode != 0
        assert "production lifecycle revision is pending" in result.stderr


def test_runner_performs_exactly_one_captured_inference_per_raw_instance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority = _runner_authority(fixture, "g2")
    result = _command(RUNNER, authority_path, "g2")
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["total_inference_count"] == 3
    assert receipt["synthetic_only"] is True
    assert receipt["production_authority_eligible"] is False

    root = Path(fixture["repository_root"])
    first = json.loads((root / "outputs/g2/outcomes/g2-a.json").read_text())
    expected = [
        hashlib.sha256(_bytes({"call_index": index, "passed": True})).hexdigest()
        for index in (1, 2)
    ]
    assert [row["inference_output_sha256"] for row in first["instances"]] == expected
    second = json.loads((root / "outputs/g2/outcomes/g2-b.json").read_text())
    assert second["instances"][0]["inference_output_sha256"] == hashlib.sha256(
        _bytes({"call_index": 3, "passed": True})
    ).hexdigest()
    ledger = json.loads((root / "outputs/g2/runner_ledger.json").read_text())
    assert ledger["total_instance_count"] == ledger["total_inference_count"] == 3
    assert ledger["runner_execution_identity"] == {
        "schema": "lewm_go2_shared_jepa_v5_execution_identity_v1",
        "entrypoint_wrapper": _file_spec(root=ROOT, path=RUNNER),
        "captured_launcher": _file_spec(root=ROOT, path=LAUNCHER),
        "captured_core": _file_spec(root=ROOT, path=CORE),
    }


def test_runner_rejects_precomputed_metric_outcomes_and_consumes_attempt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, self_reported=True)
    authority_path, _ = _runner_authority(fixture, "g2")
    first = _command(RUNNER, authority_path, "g2")
    assert first.returncode != 0
    assert "raw inference instance fields changed" in first.stderr
    second = _command(RUNNER, authority_path, "g2")
    assert second.returncode != 0
    assert "role-global attempt was already consumed" in second.stderr


def test_runner_revision_contains_no_future_evidence_and_writes_absent_paths(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority = _runner_authority(fixture, "g2")
    assert set(authority) == {
        "schema",
        "lifecycle_revision",
        "synthetic_only",
        "repository_root",
        "protocol_generation",
        "gate",
        "attempt_registry_path",
        "dataset_role_manifest",
        "evaluated_checkpoint",
        "runtime_modules",
        "inference_entry_module",
        "gate_spec",
    }
    assert set(authority["gate_spec"]) == {
        "scene_inputs",
        "outcome_paths",
        "ledger_path",
        "metric_rules",
    }
    encoded = authority_path.read_text(encoding="utf-8")
    for forbidden in (
        '"runner_ledger"',
        '"outcome_files"',
        '"fixed_final_report"',
        '"finalized_gates"',
    ):
        assert forbidden not in encoded
    root = Path(fixture["repository_root"])
    output_paths = list(authority["gate_spec"]["outcome_paths"].values()) + [
        authority["gate_spec"]["ledger_path"]
    ]
    assert all(not (root / relative).exists() for relative in output_paths)
    result = _command(RUNNER, authority_path, "g2")
    assert result.returncode == 0, result.stderr
    assert all((root / relative).is_file() for relative in output_paths)


def test_runner_rejects_future_output_binding_before_attempt_reservation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority = _runner_authority(fixture, "g2")
    authority["runner_ledger"] = {
        "path": "outputs/g2/runner_ledger.json",
        "file_sha256": "a" * 64,
    }
    _rewrite_authority(authority_path, authority)
    result = _command(RUNNER, authority_path, "g2")
    assert result.returncode != 0
    assert "stage authority fields changed" in result.stderr
    root = Path(fixture["repository_root"])
    assert not any((root / "registry/g2").iterdir())
    assert not any((root / "outputs/g2/outcomes").iterdir())


def test_finalizer_revision_rejects_runner_output_changed_after_freeze(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runner_path, runner = _runner_authority(fixture, "g2")
    run = _command(RUNNER, runner_path, "g2")
    assert run.returncode == 0, run.stderr
    finalizer_path, finalizer = _finalizer_authority(
        fixture,
        "g2",
        runner_path,
        runner,
    )
    root = Path(fixture["repository_root"])
    outcome = root / runner["gate_spec"]["outcome_paths"]["g2-a"]
    outcome.write_bytes(outcome.read_bytes() + b" ")
    result = _command(FINALIZER, finalizer_path, "g2")
    assert result.returncode != 0
    assert "file hash changed" in result.stderr
    assert not (root / finalizer["final_report_path"]).exists()


def test_downstream_reconstruction_rejects_execution_source_identity_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runner_path, runner = _runner_authority(fixture, "g2")
    run = _command(RUNNER, runner_path, "g2")
    assert run.returncode == 0, run.stderr
    finalizer_path, finalizer = _finalizer_authority(
        fixture,
        "g2",
        runner_path,
        runner,
    )
    finalizer["runner_execution_identity"]["entrypoint_wrapper"][
        "file_sha256"
    ] = "a" * 64
    _rewrite_authority(finalizer_path, finalizer)

    result = _command(FINALIZER, finalizer_path, "g2")

    assert result.returncode != 0
    assert "runner ledger authority changed" in result.stderr
    root = Path(fixture["repository_root"])
    assert not (root / finalizer["final_report_path"]).exists()


def test_g2_candidate_publication_precedes_g3_and_marks_it_pending(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _, _, finalizer_path, finalizer = _run_and_finalize(fixture, "g2")
    root = Path(fixture["repository_root"])
    assert not any((root / "outputs/g3/outcomes").iterdir())
    assert not (root / "outputs/g3/runner_ledger.json").exists()
    assert not (root / "outputs/g3/final_report.json").exists()
    publisher_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (finalizer_path, finalizer)},
    )
    result = _command(PUBLISHER, publisher_path, "g2-candidate")
    assert result.returncode == 0, result.stderr
    publication = json.loads(
        (root / fixture["publication_paths"]["g2-candidate"]).read_text()
    )
    assert publication["publication_kind"] == "g2_candidate"
    assert publication["satisfied_gates"] == ["g2"]
    assert publication["pending_gates"] == ["g3"]
    assert publication["full_promotion_eligible"] is False
    assert publication["synthetic_only"] is True
    assert publication["production_authority_eligible"] is False
    assert publication["dataset_role_manifest"] == {
        **fixture["dataset_role_manifest"],
        "content_sha256": json.loads(
            (root / fixture["dataset_role_manifest"]["path"]).read_bytes()
        )["content_sha256"],
        "protocol_generation": fixture["protocol_generation"],
    }


def test_g3_runner_rejects_candidate_changed_after_runner_revision_freeze(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _, _, finalizer_path, finalizer = _run_and_finalize(fixture, "g2")
    publisher_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (finalizer_path, finalizer)},
    )
    published = _command(PUBLISHER, publisher_path, "g2-candidate")
    assert published.returncode == 0, published.stderr
    root = Path(fixture["repository_root"])
    candidate_path = root / fixture["publication_paths"]["g2-candidate"]
    runner_path, _ = _runner_authority(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(root, publisher_path),
        g2_candidate_publication=_file_spec(root, candidate_path),
    )
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
    result = _command(RUNNER, runner_path, "g3")
    assert result.returncode != 0
    assert "G2 candidate publication file hash changed" in result.stderr
    assert not any((root / "outputs/g3/outcomes").iterdir())
    assert not (root / "outputs/g3/runner_ledger.json").exists()


def test_g3_runner_reconstructs_candidate_instead_of_trusting_frozen_claims(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _, _, finalizer_path, finalizer = _run_and_finalize(fixture, "g2")
    publisher_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (finalizer_path, finalizer)},
    )
    published = _command(PUBLISHER, publisher_path, "g2-candidate")
    assert published.returncode == 0, published.stderr
    root = Path(fixture["repository_root"])
    candidate_path = root / fixture["publication_paths"]["g2-candidate"]
    forged = json.loads(candidate_path.read_bytes())
    forged["final_reports"]["g2"]["content_sha256"] = "a" * 64
    forged.pop("content_sha256")
    candidate_path.write_bytes(_bytes(_content(forged)) + b"\n")
    runner_path, _ = _runner_authority(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(root, publisher_path),
        g2_candidate_publication=_file_spec(root, candidate_path),
    )
    result = _command(RUNNER, runner_path, "g3")
    assert result.returncode != 0
    assert "does not reproduce its fixed G2 evidence" in result.stderr
    assert not any((root / "outputs/g3/outcomes").iterdir())


def test_full_promotion_rejects_g2_only_finalized_inventory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _, _, finalizer_path, finalizer = _run_and_finalize(fixture, "g2")
    publisher_path, _ = _publisher_authority(
        fixture,
        "full-promotion",
        {"g2": (finalizer_path, finalizer)},
    )
    result = _command(PUBLISHER, publisher_path, "full-promotion")
    assert result.returncode != 0
    assert "finalized-gate inventory changed" in result.stderr
    root = Path(fixture["repository_root"])
    assert not (root / fixture["publication_paths"]["full-promotion"]).exists()


def test_candidate_authority_cannot_be_reused_as_full_promotion_revision(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _, _, finalizer_path, finalizer = _run_and_finalize(fixture, "g2")
    publisher_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (finalizer_path, finalizer)},
    )
    result = _command(PUBLISHER, publisher_path, "full-promotion")
    assert result.returncode != 0
    assert "authority revision or mode changed" in result.stderr


def test_runner_rejects_overlapping_exclusive_output_paths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority = _runner_authority(fixture, "g2")
    authority["gate_spec"]["ledger_path"] = authority["gate_spec"][
        "outcome_paths"
    ]["g2-a"]
    _rewrite_authority(authority_path, authority)
    result = _command(RUNNER, authority_path, "g2")
    assert result.returncode != 0
    assert "runner output paths overlap" in result.stderr
    root = Path(fixture["repository_root"])
    assert not any((root / "outputs/g2/outcomes").iterdir())


def test_runner_rejects_prebound_output_before_attempt_or_input_access(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority = _runner_authority(fixture, "g2")
    root = Path(fixture["repository_root"])
    prebound = root / authority["gate_spec"]["outcome_paths"]["g2-a"]
    prebound.write_bytes(b"attacker-prebound\n")

    result = _command(RUNNER, authority_path, "g2")

    assert result.returncode != 0
    assert "runner output already exists" in result.stderr
    assert prebound.read_bytes() == b"attacker-prebound\n"
    assert not any((root / "registry/g2").iterdir())
    assert not (root / authority["gate_spec"]["ledger_path"]).exists()
    assert not (root / authority["gate_spec"]["outcome_paths"]["g2-b"]).exists()


def test_finalizer_and_publisher_preflight_exclusive_outputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runner_path, runner = _runner_authority(fixture, "g2")
    run = _command(RUNNER, runner_path, "g2")
    assert run.returncode == 0, run.stderr
    finalizer_path, finalizer = _finalizer_authority(
        fixture,
        "g2",
        runner_path,
        runner,
    )
    root = Path(fixture["repository_root"])
    report_path = root / finalizer["final_report_path"]
    report_path.write_bytes(b"prebound-report\n")
    finalized = _command(FINALIZER, finalizer_path, "g2")
    assert finalized.returncode != 0
    assert "final report already exists" in finalized.stderr
    assert report_path.read_bytes() == b"prebound-report\n"

    report_path.unlink()
    finalized = _command(FINALIZER, finalizer_path, "g2")
    assert finalized.returncode == 0, finalized.stderr
    publisher_path, publisher = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (finalizer_path, finalizer)},
    )
    publication_path = root / publisher["publication_path"]
    publication_path.write_bytes(b"prebound-publication\n")
    published = _command(PUBLISHER, publisher_path, "g2-candidate")
    assert published.returncode != 0
    assert "publication already exists" in published.stderr
    assert publication_path.read_bytes() == b"prebound-publication\n"


def test_runner_rejects_role_manifest_protocol_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = Path(fixture["repository_root"])
    role_path = root / fixture["dataset_role_manifest"]["path"]
    manifest = json.loads(role_path.read_bytes())
    manifest["protocol_generation"] = "wrong-generation"
    core = dict(manifest)
    core.pop("content_sha256")
    role_path.write_bytes(_bytes(_content(core)) + b"\n")
    fixture["dataset_role_manifest"] = _file_spec(root, role_path)
    authority_path, _ = _runner_authority(fixture, "g2")

    result = _command(RUNNER, authority_path, "g2")

    assert result.returncode != 0
    assert "role manifest protocol generation changed" in result.stderr
    assert not any((root / "outputs/g2/outcomes").iterdir())


def test_g3_rejects_cross_manifest_reassignment_of_exact_g2_scenes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _, _, g2_finalizer_path, g2_finalizer = _run_and_finalize(fixture, "g2")
    candidate_authority_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (g2_finalizer_path, g2_finalizer)},
    )
    candidate = _command(PUBLISHER, candidate_authority_path, "g2-candidate")
    assert candidate.returncode == 0, candidate.stderr
    root = Path(fixture["repository_root"])
    candidate_path = root / fixture["publication_paths"]["g2-candidate"]

    fixture["dataset_role_manifest"] = _write_json(
        root,
        root / "artifacts/roles-reassigned.json",
        {
            "schema": "lewm_go2_shared_jepa_dataset_roles_v7",
            "protocol_generation": fixture["protocol_generation"],
            "roles": {
                "train": ["train-a"],
                "g2": ["g3-a"],
                "g3": ["g2-a", "g2-b"],
            },
            "scene_families": {
                "train-a": "family-a",
                "g3-a": "family-a",
                "g2-a": "family-a",
                "g2-b": "family-b",
            },
        },
    )
    original_g2 = fixture["gate_specs"]["g2"]
    fixture["gate_specs"]["g3"] = {
        "scene_inputs": original_g2["scene_inputs"],
        "outcome_paths": {
            "g2-a": "outputs/g3/outcomes/reassigned-g2-a.json",
            "g2-b": "outputs/g3/outcomes/reassigned-g2-b.json",
        },
        "ledger_path": "outputs/g3/runner_ledger.json",
        "metric_rules": fixture["gate_specs"]["g3"]["metric_rules"],
    }
    runner_path, _ = _runner_authority(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(
            root,
            candidate_authority_path,
        ),
        g2_candidate_publication=_file_spec(root, candidate_path),
    )

    result = _command(RUNNER, runner_path, "g3")

    assert result.returncode != 0
    assert "does not reproduce its fixed G2 evidence" in result.stderr
    assert not any((root / "outputs/g3/outcomes").iterdir())
    assert not (root / "outputs/g3/runner_ledger.json").exists()


def test_full_promotion_requires_exact_g2_predecessor_used_by_g3(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    g2_runner_path, g2_runner, g2_finalizer_path, g2_finalizer = (
        _run_and_finalize(fixture, "g2")
    )
    root = Path(fixture["repository_root"])
    candidate_authority_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (g2_finalizer_path, g2_finalizer)},
    )
    candidate = _command(PUBLISHER, candidate_authority_path, "g2-candidate")
    assert candidate.returncode == 0, candidate.stderr
    candidate_path = root / fixture["publication_paths"]["g2-candidate"]
    _, _, g3_finalizer_path, g3_finalizer = _run_and_finalize(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(
            root,
            candidate_authority_path,
        ),
        g2_candidate_publication=_file_spec(root, candidate_path),
    )

    ledger = json.loads(
        (root / g2_runner["gate_spec"]["ledger_path"]).read_bytes()
    )
    alternate_report = "outputs/g2/alternate_final_report.json"
    alternate_fields = {
        "gate": "g2",
        "runner_authority": _file_spec(root, g2_runner_path),
        "runner_ledger": _file_spec(
            root,
            root / g2_runner["gate_spec"]["ledger_path"],
        ),
        "outcome_files": {
            scene_id: _file_spec(root, root / relative)
            for scene_id, relative in g2_runner["gate_spec"]["outcome_paths"].items()
        },
        "final_report_path": alternate_report,
        "runner_execution_identity": ledger["runner_execution_identity"],
    }
    original_authority_bytes = g2_finalizer_path.read_bytes()
    temporary_path, alternate_authority = _stage_authority(
        fixture,
        FINALIZER_REVISIONS["g2"],
        alternate_fields,
    )
    alternate_bytes = temporary_path.read_bytes()
    temporary_path.write_bytes(original_authority_bytes)
    alternate_path = root / "authorities/finalizer_g2_evidence_v2_alternate.json"
    alternate_path.write_bytes(alternate_bytes)
    finalized = _command(FINALIZER, alternate_path, "g2")
    assert finalized.returncode == 0, finalized.stderr

    full_authority_path, _ = _publisher_authority(
        fixture,
        "full-promotion",
        {
            "g2": (alternate_path, alternate_authority),
            "g3": (g3_finalizer_path, g3_finalizer),
        },
    )
    published = _command(PUBLISHER, full_authority_path, "full-promotion")

    assert published.returncode != 0
    assert "G2 report differs from G3 predecessor" in published.stderr
    assert not (root / fixture["publication_paths"]["full-promotion"]).exists()


def test_copied_entrypoint_rejects_before_synthetic_authority_access(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    authority_path, _ = _runner_authority(fixture, "g2")
    copied = tmp_path / RUNNER.name
    copied.write_bytes(RUNNER.read_bytes())

    result = _command(copied, authority_path, "g2")

    assert result.returncode != 0
    assert "wrapper was executed from a copied path" in result.stderr
    root = Path(fixture["repository_root"])
    assert not any((root / "registry/g2").iterdir())


def test_entrypoint_requires_python_isolated_mode_before_authority_access(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority_path, _ = _runner_authority(fixture, "g2")
    environment = dict(os.environ)
    environment["LEWM_V5_SYNTHETIC_AUTHORITY_PATH"] = str(authority_path)

    result = subprocess.run(
        [sys.executable, str(RUNNER), "g2"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "requires isolated Python" in result.stderr
    root = Path(fixture["repository_root"])
    assert not any((root / "registry/g2").iterdir())


@pytest.mark.parametrize("tampered_source", ("launcher", "core"))
def test_captured_source_change_rejects_before_authority_access(
    tmp_path: Path,
    tampered_source: str,
) -> None:
    source_root = tmp_path / "captured-source-root"
    scripts = source_root / "scripts"
    scripts.mkdir(parents=True)
    copied_core = scripts / CORE.name
    copied_core.write_bytes(CORE.read_bytes())
    launcher_text = LAUNCHER.read_text(encoding="utf-8").replace(
        str(ROOT),
        str(source_root),
    )
    copied_launcher = scripts / LAUNCHER.name
    copied_launcher.write_text(launcher_text, encoding="ascii")
    copied_launcher_hash = hashlib.sha256(copied_launcher.read_bytes()).hexdigest()
    canonical_launcher_hash = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
    wrapper_text = (
        RUNNER.read_text(encoding="utf-8")
        .replace(str(ROOT), str(source_root))
        .replace(canonical_launcher_hash, copied_launcher_hash)
    )
    copied_wrapper = scripts / RUNNER.name
    copied_wrapper.write_text(wrapper_text, encoding="ascii")
    if tampered_source == "launcher":
        copied_launcher.write_bytes(copied_launcher.read_bytes() + b"# changed\n")
        expected_error = "captured V5 launcher source hash changed"
    else:
        copied_core.write_bytes(copied_core.read_bytes() + b"# changed\n")
        expected_error = "captured V5 core source hash changed"

    result = subprocess.run(
        [sys.executable, "-I", str(copied_wrapper), "g2"],
        cwd=ROOT,
        env={
            **os.environ,
            "LEWM_V5_SYNTHETIC_AUTHORITY_PATH": str(
                tmp_path / "must-not-be-opened.json"
            ),
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_full_three_program_synthetic_workflow_is_permanently_ineligible(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _, _, g2_finalizer_path, g2_finalizer = _run_and_finalize(fixture, "g2")
    candidate_authority_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (g2_finalizer_path, g2_finalizer)},
    )
    candidate = _command(
        PUBLISHER,
        candidate_authority_path,
        "g2-candidate",
    )
    assert candidate.returncode == 0, candidate.stderr
    candidate_receipt = json.loads(candidate.stdout)
    assert candidate_receipt["publication_kind"] == "g2_candidate"
    assert candidate_receipt["full_promotion_eligible"] is False
    root = Path(fixture["repository_root"])
    candidate_path = root / fixture["publication_paths"]["g2-candidate"]
    candidate_spec = _file_spec(root, candidate_path)

    _, _, g3_finalizer_path, g3_finalizer = _run_and_finalize(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(
            root,
            candidate_authority_path,
        ),
        g2_candidate_publication=candidate_spec,
    )
    g3_ledger = json.loads((root / "outputs/g3/runner_ledger.json").read_text())
    candidate_events = [
        event
        for event in g3_ledger["events"]
        if event["operation"] == "verify_g2_candidate_evidence"
    ]
    assert [event["artifact"] for event in candidate_events] == [
        "g2_candidate_publisher_authority",
        "g2_finalizer_authority",
        "g2_runner_authority",
        "dataset_role_manifest",
        "runner_ledger",
        "raw_outcome:g2-a",
        "raw_outcome:g2-b",
        "evaluated_checkpoint",
        "g2_final_report",
        "g2_candidate_publication",
    ]
    full_authority_path, _ = _publisher_authority(
        fixture,
        "full-promotion",
        {
            "g2": (g2_finalizer_path, g2_finalizer),
            "g3": (g3_finalizer_path, g3_finalizer),
        },
    )
    published = _command(PUBLISHER, full_authority_path, "full-promotion")
    assert published.returncode == 0, published.stderr
    receipt = json.loads(published.stdout)
    assert receipt["publication_kind"] == "full_promotion"
    assert receipt["full_promotion_eligible"] is False
    assert receipt["synthetic_only"] is True
    assert receipt["production_authority_eligible"] is False
    publication = json.loads(
        (root / fixture["publication_paths"]["full-promotion"]).read_text()
    )
    assert publication["satisfied_gates"] == ["g2", "g3"]
    assert publication["pending_gates"] == []
    assert publication["synthetic_only"] is True
    assert publication["production_authority_eligible"] is False
    assert publication["publisher_execution_identity"] == {
        "schema": "lewm_go2_shared_jepa_v5_execution_identity_v1",
        "entrypoint_wrapper": _file_spec(root=ROOT, path=PUBLISHER),
        "captured_launcher": _file_spec(root=ROOT, path=LAUNCHER),
        "captured_core": _file_spec(root=ROOT, path=CORE),
    }


def test_library_production_capabilities_are_tombstones(tmp_path: Path) -> None:
    from lewm.benchmarks import finalize_shared_observable_camera_ray_jepa_v5_g2 as g2
    from lewm.benchmarks import finalize_shared_observable_camera_ray_jepa_v5_g3 as g3
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy as runner
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_finalizer_core as core
    from lewm.models import shared_observable_camera_ray_jepa_v5 as model
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy as registry

    assert not hasattr(runner, "reopen_canonical_runner_batch")
    assert not hasattr(runner, "validated_runner_batch_payload")
    assert not hasattr(runner, "CanonicalRunnerBatchV6")
    assert not hasattr(core, "finalize_gate_records")
    assert not hasattr(g2, "finalize_g2")
    assert not hasattr(g3, "finalize_g3")
    assert not hasattr(model, "ProductionCheckpointContextV5")
    assert not hasattr(model, "load_production_checkpoint_context_v5")
    assert not hasattr(model, "build_checkpoint_v5_payload")
    assert not hasattr(model, "validate_checkpoint_v5_payload")
    assert not hasattr(model, "checkpoint_v5_weights_only_roundtrip")
    assert not hasattr(registry, "acquire_canonical_attempt")
    assert not (tmp_path / "reservation.json").exists()


def test_cli_rejects_caller_selected_registry_and_output_paths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_path, _ = _runner_authority(fixture, "g2")
    result = _command(
        RUNNER,
        authority_path,
        "g2",
        "--registry-root",
        str(tmp_path / "attacker-registry"),
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
    assert not (tmp_path / "attacker-registry").exists()


def test_reproduced_live_runner_substitution_cannot_replace_captured_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy as old_runner

    monkeypatch.setattr(
        old_runner,
        "_normalize_batch_material",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("old runner used")),
    )
    evil = tmp_path / "evil-imports/lewm"
    evil.mkdir(parents=True)
    (evil / "__init__.py").write_text(
        "raise AssertionError('uncaptured lewm package imported')\n",
        encoding="ascii",
    )
    fixture = _fixture(tmp_path / "fixture")
    authority_path, _ = _runner_authority(fixture, "g2")
    result = _command(
        RUNNER,
        authority_path,
        "g2",
        extra_environment={"PYTHONPATH": str(evil.parent)},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["total_inference_count"] == 3
    root = Path(fixture["repository_root"])
    outcome = json.loads((root / "outputs/g2/outcomes/g2-a.json").read_text())
    assert len(outcome["instances"]) == 2


def test_reproduced_finalizer_global_substitution_cannot_forge_a_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_finalizer_core as old_core

    monkeypatch.setattr(
        old_core,
        "_derive_gate_record_core",
        lambda **kwargs: {"passed": True, "metrics": {}},
    )
    fixture = _fixture(tmp_path, failing=True)
    runner_path, runner = _runner_authority(fixture, "g2")
    run = _command(RUNNER, runner_path, "g2")
    assert run.returncode == 0, run.stderr
    finalizer_path, _ = _finalizer_authority(
        fixture,
        "g2",
        runner_path,
        runner,
    )
    finalized = _command(FINALIZER, finalizer_path, "g2")
    assert finalized.returncode == 0, finalized.stderr
    receipt = json.loads(finalized.stdout)
    assert receipt["passed"] is False
    report = json.loads(
        (Path(fixture["repository_root"]) / "outputs/g2/final_report.json").read_text()
    )
    assert report["passed"] is False
    assert min(report["metrics"].values()) < 1.0


def test_finalizer_rejects_per_scene_ledger_count_shuffle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    runner_path, runner = _runner_authority(fixture, "g2")
    run = _command(RUNNER, runner_path, "g2")
    assert run.returncode == 0, run.stderr
    root = Path(fixture["repository_root"])
    ledger_path = root / runner["gate_spec"]["ledger_path"]
    ledger = json.loads(ledger_path.read_bytes())
    scene_events = [
        event
        for event in ledger["events"]
        if event["operation"] == "open_raw_scene_and_run_each_instance"
    ]
    assert [event["instance_count"] for event in scene_events] == [2, 1]
    scene_events[0]["instance_count"] = 1
    scene_events[0]["inference_count"] = 1
    scene_events[1]["instance_count"] = 2
    scene_events[1]["inference_count"] = 2
    core = dict(ledger)
    core.pop("content_sha256")
    ledger_path.write_bytes(_bytes(_content(core)) + b"\n")
    finalizer_path, _ = _finalizer_authority(
        fixture,
        "g2",
        runner_path,
        runner,
    )

    finalized = _command(FINALIZER, finalizer_path, "g2")
    assert finalized.returncode != 0
    assert "does not reproduce reopened evidence" in finalized.stderr
    assert not (root / "outputs/g2/final_report.json").exists()


def test_publisher_rejects_skeletal_passing_reports(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = Path(fixture["repository_root"])
    _, _, g2_finalizer_path, g2_finalizer = _run_and_finalize(fixture, "g2")
    candidate_authority_path, _ = _publisher_authority(
        fixture,
        "g2-candidate",
        {"g2": (g2_finalizer_path, g2_finalizer)},
    )
    candidate = _command(
        PUBLISHER,
        candidate_authority_path,
        "g2-candidate",
    )
    assert candidate.returncode == 0, candidate.stderr
    candidate_spec = _file_spec(
        root,
        root / fixture["publication_paths"]["g2-candidate"],
    )
    _, _, g3_finalizer_path, g3_finalizer = _run_and_finalize(
        fixture,
        "g3",
        g2_candidate_publisher_authority=_file_spec(
            root,
            candidate_authority_path,
        ),
        g2_candidate_publication=candidate_spec,
    )
    finalized = {
        "g2": (g2_finalizer_path, g2_finalizer),
        "g3": (g3_finalizer_path, g3_finalizer),
    }
    for gate, (_, authority) in finalized.items():
        path = root / authority["final_report_path"]
        original = json.loads(path.read_bytes())
        path.write_bytes(
            _bytes(
                _content(
                    {
                        "schema": "lewm_go2_shared_jepa_final_report_v9",
                        "gate": gate,
                            "passed": True,
                            "finalizer_execution_identity": original[
                                "finalizer_execution_identity"
                            ],
                        "synthetic_only": True,
                        "production_authority_eligible": False,
                    }
                )
            )
            + b"\n"
        )
    publisher_path, _ = _publisher_authority(
        fixture,
        "full-promotion",
        finalized,
    )
    published = _command(PUBLISHER, publisher_path, "full-promotion")
    assert published.returncode != 0
    assert "does not reproduce its fixed evidence" in published.stderr
    assert not (root / fixture["publication_paths"]["full-promotion"]).exists()


def test_registry_tombstone_cannot_be_reactivated_by_authority_monkeypatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy as registry

    target = tmp_path / "forged-registry"
    target.mkdir()
    monkeypatch.setattr(
        registry,
        "require_frozen_production_authority",
        lambda: {"attempt_registry_path": target},
    )
    with pytest.raises(PermissionError, match="one-shot runner CLI"):
        registry._removed_acquire_canonical_attempt_tombstone(
            gate="g2",
            namespace_sha256="a" * 64,
            reservation={"schema": "caller-forged"},
        )
    assert not any(target.iterdir())
