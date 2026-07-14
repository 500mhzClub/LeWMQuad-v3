from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from scripts import launch_go2_observable_camera_ray_fit_v4 as launcher


def test_ladder_v3_contract_binds_v2_root_failure_lineage_and_amendment() -> None:
    contract = launcher.EXPECTED_FIT_CONTRACT
    amendment = (
        "docs/lewm_go2_observable_camera_ray_fit_v4_"
        "ladder_v3_failure_successor_amendment_2026-07-13.md"
    )
    assert contract["ladder_contract"] == "observable_camera_ray_fit_v4_ladder_v3"
    assert contract["development_output_root"].endswith("development_fit_v2")
    assert contract["v1_failure_lineage"] == {
        "reservation_file_sha256": (
            "115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e"
        ),
        "reservation_content_sha256": (
            "ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd"
        ),
        "failure_file_sha256": (
            "6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48"
        ),
        "failure_content_sha256": (
            "7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4"
        ),
    }
    assert launcher.REQUIRED_SOURCE_ROLES[amendment] == (
        "ladder_v3_failure_successor_amendment"
    )


def _canonical_file(path: Path, core: dict[str, object]) -> tuple[str, dict[str, object]]:
    value = {
        **core,
        "content_sha256": hashlib.sha256(
            launcher._canonical_json_bytes(core)
        ).hexdigest(),
    }
    payload = launcher._canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), value


def _synthetic_source_map(root: Path) -> dict[str, object]:
    entries = []
    for index, relative in enumerate(sorted(launcher.REQUIRED_SOURCE_PATHS)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"synthetic source {relative}\n".encode("ascii")
        path.write_bytes(payload)
        entries.append(
            {
                "path": relative,
                "role": launcher.REQUIRED_SOURCE_ROLES[relative],
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "algorithm": "canonical_json_sha256_entries_v1",
        "entry_count": len(entries),
        "entries": entries,
        "source_map_sha256": hashlib.sha256(
            launcher._canonical_json_bytes(entries)
        ).hexdigest(),
    }


def test_strict_hashed_object_rejects_byte_identical_copied_path(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    digest, _value = _canonical_file(canonical, {"schema": "synthetic"})
    copied = tmp_path / "copied.json"
    copied.write_bytes(canonical.read_bytes())

    with pytest.raises(PermissionError, match="path is not canonical"):
        launcher._strict_hashed_object(
            copied,
            digest,
            name="synthetic receipt",
            canonical_path=canonical.resolve(),
        )


def test_source_map_requires_exact_path_set_and_committed_bytes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source_map = _synthetic_source_map(root)
    expected = source_map["source_map_sha256"]
    assert launcher._validate_source_map(source_map, root=root) == expected

    missing = json.loads(json.dumps(source_map))
    missing["entries"].pop()
    missing["entry_count"] = len(missing["entries"])
    missing["source_map_sha256"] = hashlib.sha256(
        launcher._canonical_json_bytes(missing["entries"])
    ).hexdigest()
    with pytest.raises(PermissionError, match="source closure changed"):
        launcher._validate_source_map(missing, root=root)

    first = source_map["entries"][0]
    (root / first["path"]).write_text("mutated\n", encoding="ascii")
    with pytest.raises(ValueError, match="source SHA-256 changed"):
        launcher._validate_source_map(source_map, root=root)


def test_pending_review_rejects_before_dataset_or_audit_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    source_map = _synthetic_source_map(root)
    dataset_path = root / "restricted" / "manifest.json"
    audit_path = root / "restricted" / "audit_result.json"
    authorization_path = root / "authorization.json"
    review_path = root / "review.json"
    upstream_path = (
        root
        / "docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json"
    )

    review_digest, review = _canonical_file(
        review_path,
        {
            "schema": launcher.REVIEW_RECORD_SCHEMA,
            "status": "pending_second_independent_review",
            "decision": "pending",
            "reviewer": None,
            "reviewed_source_map_sha256": None,
            "restricted_payload_opened": False,
            "findings": ["synthetic review remains pending"],
        },
    )
    authorization_digest, _authorization = _canonical_file(
        authorization_path,
        {
            "schema": launcher.AUTHORIZATION_SCHEMA,
            "status": "pending_independent_review",
            "dataset_binding": {
                "path": str(dataset_path),
                "file_sha256": launcher.DATASET_MANIFEST_FILE_SHA256,
                "content_sha256": launcher.DATASET_MANIFEST_CONTENT_SHA256,
                "status": "reviewed_exact_artifact",
            },
            "audit_binding": {
                "path": str(audit_path),
                "file_sha256": launcher.AUDIT_RECEIPT_FILE_SHA256,
                "content_sha256": launcher.AUDIT_RECEIPT_CONTENT_SHA256,
                "status": "reviewed_exact_artifact",
            },
            "upstream_implementation": {
                "path": str(upstream_path),
                "file_sha256": launcher.UPSTREAM_IMPLEMENTATION_FILE_SHA256,
                "content_sha256": launcher.UPSTREAM_IMPLEMENTATION_CONTENT_SHA256,
                "source_map_sha256": launcher.UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256,
            },
            "fit_contract": launcher.EXPECTED_FIT_CONTRACT,
            "allowed_fit_sizes": list(launcher.SUPPORTED_FIT_SIZES),
            "source_map": source_map,
            "authorization": {
                "development_fit": False,
                "development_checkpoint_creation_authorized": False,
                "checkpoint_use_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
            "review_record": {
                "path": str(review_path),
                "file_sha256": review_digest,
                "content_sha256": review["content_sha256"],
                "status": "pending_second_independent_review",
            },
        },
    )

    assert not dataset_path.exists()
    assert not audit_path.exists()
    strict_loader = launcher._strict_hashed_object

    def synthetic_upstream_loader(
        path: Path,
        expected_file_sha256: str,
        *,
        name: str,
        canonical_path: Path,
        require_canonical: bool = True,
    ) -> dict[str, object]:
        if path == upstream_path and name == "V4 upstream implementation manifest":
            return {
                "content_sha256": launcher.UPSTREAM_IMPLEMENTATION_CONTENT_SHA256,
                "source_map": {
                    "source_map_sha256": launcher.UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256
                },
            }
        return strict_loader(
            path,
            expected_file_sha256,
            name=name,
            canonical_path=canonical_path,
            require_canonical=require_canonical,
        )

    monkeypatch.setattr(launcher, "_strict_hashed_object", synthetic_upstream_loader)
    with pytest.raises(PermissionError, match="one-shot only"):
        launcher.preflight_exact_authorization(
            dataset_path=dataset_path,
            dataset_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
            audit_path=audit_path,
            audit_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
            authorization_path=authorization_path,
            authorization_file_sha256=authorization_digest,
            review_record_path=review_path,
            review_record_file_sha256=review_digest,
            root=root,
            canonical_dataset_path=dataset_path,
            canonical_audit_path=audit_path,
            canonical_authorization_path=authorization_path,
            canonical_review_record_path=review_path,
            upstream_implementation_path=upstream_path,
        )


def test_direct_exact_script_rejects_before_numpy_or_torch_import(tmp_path: Path) -> None:
    marker = tmp_path / "imported.txt"
    probe = (
        "import os\n"
        "with open(os.environ['LEWM_V4_IMPORT_MARKER'], 'a', encoding='ascii') as f:\n"
        "    f.write(__name__ + '\\n')\n"
        "raise RuntimeError('probe module imported')\n"
    )
    (tmp_path / "numpy.py").write_text(probe, encoding="ascii")
    (tmp_path / "torch.py").write_text(probe, encoding="ascii")
    environment = dict(os.environ)
    environment["LEWM_V4_IMPORT_MARKER"] = str(marker)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), environment.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, str(launcher.TRAINER_PATH), "--fit-size", "5"],
        cwd=launcher.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "trainer execution must use" in completed.stderr
    assert not marker.exists()


def test_same_pid_serialized_environment_receipt_cannot_grant_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "imported.txt"
    probe = (
        "import os\n"
        "with open(os.environ['LEWM_V4_IMPORT_MARKER'], 'a', encoding='ascii') as f:\n"
        "    f.write(__name__ + '\\n')\n"
        "raise RuntimeError('probe module imported')\n"
    )
    (tmp_path / "numpy.py").write_text(probe, encoding="ascii")
    (tmp_path / "torch.py").write_text(probe, encoding="ascii")
    wrapper = tmp_path / "forge_then_exec.py"
    wrapper.write_text(
        "import os,tempfile\n"
        "from scripts import launch_go2_observable_camera_ray_fit_v4 as l\n"
        "fd,name=tempfile.mkstemp(prefix='forged_v4.',suffix='.json')\n"
        "os.write(fd,b'{\"launcher_pid\":'+str(os.getpid()).encode()+b'}\\n'); os.close(fd)\n"
        "env=dict(os.environ); env['LEWM_V4_PREAUTHORIZATION_PATH']=name; env['LEWM_V4_PREAUTHORIZATION_SHA256']='0'*64\n"
        "os.execve(os.sys.executable,[os.sys.executable,str(l.TRAINER_PATH),'--fit-size','5'],env)\n",
        encoding="ascii",
    )
    environment = dict(os.environ)
    environment["LEWM_V4_IMPORT_MARKER"] = str(marker)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(launcher.ROOT), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(wrapper)],
        cwd=launcher.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "trainer execution must use" in completed.stderr
    assert not marker.exists()


def test_launcher_exports_no_context_capability_or_callable_loader() -> None:
    forbidden = {
        "VerifiedLaunchContext",
        "create_verified_launch_context",
        "validate_verified_launch_context",
        "load_verified_trainer",
        "load_content_addressed_runtime",
        "_ContentAddressedLoader",
        "_ContentAddressedFinder",
        "_ContentAddressedRuntime",
        "_load_content_addressed_runtime",
        "_capture_canonical_runtime_sources",
        "_reverify_loaded_runtime_sources",
        "_load_captured_launcher",
        "_captured_rgb_worker_dispatch",
        "RUNTIME_MODULE_PATHS",
        "ALLOWED_UNTRACKED_IMPORT_ROOTS",
    }
    assert forbidden.isdisjoint(vars(launcher))


def test_schema_only_mapping_cannot_mint_same_process_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = launcher.parse_args(_launcher_args())
    with pytest.raises(TypeError):
        launcher._execute_captured_trainer(
            args,
            {"schema": launcher.VERIFIED_CONTEXT_SCHEMA, "source_map": {}},
        )
    assert tuple(
        inspect.signature(launcher._execute_captured_trainer).parameters
    ) == ("args",)


def test_canonical_tampered_authorization_fails_before_trainer_or_torch_import(
    tmp_path: Path,
) -> None:
    review_raw = launcher.CANONICAL_REVIEW_RECORD_PATH.read_bytes()
    argv = [
            "--dataset-manifest",
            str(launcher.CANONICAL_DATASET_PATH),
            "--dataset-manifest-sha256",
            launcher.DATASET_MANIFEST_FILE_SHA256,
            "--audit-receipt",
            str(launcher.CANONICAL_AUDIT_PATH),
            "--audit-receipt-sha256",
            launcher.AUDIT_RECEIPT_FILE_SHA256,
            "--trainer-authorization",
            str(launcher.CANONICAL_AUTHORIZATION_PATH),
            "--trainer-authorization-sha256",
            "0" * 64,
            "--trainer-review-record",
            str(launcher.CANONICAL_REVIEW_RECORD_PATH),
            "--trainer-review-record-sha256",
            hashlib.sha256(review_raw).hexdigest(),
            "--fit-size",
            "5",
            "--steps",
            str(launcher.DEFAULT_STEPS[5]),
            "--seed",
            "20260710",
        ]
    marker = tmp_path / "torch_imported"
    (tmp_path / "torch.py").write_text(
        "import os; open(os.environ['LEWM_V4_TORCH_MARKER'],'w').write('imported')\n",
        encoding="ascii",
    )
    environment = dict(os.environ)
    environment["LEWM_V4_TORCH_MARKER"] = str(marker)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(launcher.ROOT), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(Path(launcher.__file__).resolve()), *argv],
        cwd=launcher.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "trainer authorization caller SHA-256 changed" in completed.stderr
    assert not marker.exists()


def test_loaded_module_code_fingerprint_detects_function_replacement() -> None:
    module = types.ModuleType("synthetic_v4_mutation_probe")
    exec("def value():\n    return 1\n", module.__dict__)
    before = launcher._module_code_sha256(module)
    exec("def value():\n    return 2\n", module.__dict__)
    assert launcher._module_code_sha256(module) != before


def test_rgb_worker_dispatch_requires_isolated_spawn() -> None:
    with pytest.raises(PermissionError, match="requires an isolated spawn"):
        launcher._rgb_worker_terminal(((), "a" * 64, "b" * 64))


def test_fixed_path_isolated_tampered_authority_exposes_no_runtime_or_torch() -> None:
    interpreter = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
    assert interpreter.is_file()
    review_raw = launcher.CANONICAL_REVIEW_RECORD_PATH.read_bytes()
    argv = [
        "--dataset-manifest",
        str(launcher.CANONICAL_DATASET_PATH),
        "--dataset-manifest-sha256",
        launcher.DATASET_MANIFEST_FILE_SHA256,
        "--audit-receipt",
        str(launcher.CANONICAL_AUDIT_PATH),
        "--audit-receipt-sha256",
        launcher.AUDIT_RECEIPT_FILE_SHA256,
        "--trainer-authorization",
        str(launcher.CANONICAL_AUTHORIZATION_PATH),
        "--trainer-authorization-sha256",
        "0" * 64,
        "--trainer-review-record",
        str(launcher.CANONICAL_REVIEW_RECORD_PATH),
        "--trainer-review-record-sha256",
        hashlib.sha256(review_raw).hexdigest(),
        "--fit-size",
        "5",
        "--steps",
        str(launcher.DEFAULT_STEPS[5]),
        "--seed",
        "20260710",
    ]
    forbidden = (
        "_ContentAddressedLoader",
        "_ContentAddressedFinder",
        "_ContentAddressedRuntime",
        "_load_content_addressed_runtime",
        "_capture_canonical_runtime_sources",
        "_reverify_loaded_runtime_sources",
        "_load_captured_launcher",
        "RUNTIME_MODULE_PATHS",
        "ALLOWED_UNTRACKED_IMPORT_ROOTS",
    )
    code = f"""
import importlib.util
import sys

path = {str(Path(launcher.__file__).resolve())!r}
spec = importlib.util.spec_from_file_location("v4_fixed_pending_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for name in {forbidden!r}:
    assert not hasattr(module, name), name
try:
    module.main({argv!r})
except (PermissionError, ValueError) as exc:
    assert "trainer authorization caller SHA-256 changed" in str(exc), str(exc)
else:
    raise AssertionError("tampered authorization unexpectedly executed")
assert "torch" not in sys.modules
assert not any(
    name == "scripts.train_go2_observable_camera_ray_fit_v4"
    or name.startswith("_lewm_v4_ca_")
    for name in sys.modules
)
print("fixed-tampered-no-runtime-trainer-torch")
"""
    completed = subprocess.run(
        [str(interpreter), "-I", "-c", code],
        cwd=launcher.ROOT,
        env={
            **dict(os.environ),
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "fixed-tampered-no-runtime-trainer-torch"


def test_library_caller_cannot_execute_captured_trainer() -> None:
    args = launcher.parse_args(_launcher_args())
    with pytest.raises(PermissionError, match="execution is one-shot only"):
        launcher._execute_captured_trainer(args)


def _launcher_args(*, fit_size: int = 5, seed: int = 20260710) -> list[str]:
    return [
        "--dataset-manifest",
        "/synthetic/manifest.json",
        "--dataset-manifest-sha256",
        "1" * 64,
        "--audit-receipt",
        "/synthetic/audit.json",
        "--audit-receipt-sha256",
        "2" * 64,
        "--trainer-authorization",
        "/synthetic/authorization.json",
        "--trainer-authorization-sha256",
        "3" * 64,
        "--trainer-review-record",
        "/synthetic/review.json",
        "--trainer-review-record-sha256",
        "4" * 64,
        "--fit-size",
        str(fit_size),
        "--steps",
        str(launcher.DEFAULT_STEPS[fit_size]),
        "--seed",
        str(seed),
    ]


def test_launcher_parser_freezes_schedule_and_ladder_dependencies() -> None:
    args = launcher.parse_args(_launcher_args())
    assert args.fit_size == 5
    assert args.steps == launcher.DEFAULT_STEPS[5]
    assert args.seed == launcher.EXPECTED_SEEDS[0]

    wrong_steps = _launcher_args()
    wrong_steps[wrong_steps.index("--steps") + 1] = "1001"
    with pytest.raises(PermissionError, match="configuration is not frozen"):
        launcher.parse_args(wrong_steps)

    with pytest.raises(PermissionError, match="predecessor gate"):
        launcher.parse_args(_launcher_args(fit_size=16))

    with pytest.raises(PermissionError, match="completed first-seed gate"):
        launcher.parse_args(_launcher_args(seed=20260711))
