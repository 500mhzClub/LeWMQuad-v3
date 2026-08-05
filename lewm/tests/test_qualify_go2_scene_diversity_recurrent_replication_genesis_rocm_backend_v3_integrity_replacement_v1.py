from __future__ import annotations

import copy
import json
from pathlib import Path
import resource
import subprocess
import sys

import pytest

from scripts import (
    qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1
    as qualifier,
)


def _isolated_json(script: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=qualifier.REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_fresh_process_reproduces_old_mismatch_and_replacement_is_order_invariant() -> None:
    repo = str(qualifier.REPO_ROOT)
    old = _isolated_json(
        f"""
import json, sys
sys.path.insert(0, {repo!r})
runner_name = 'scripts.run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3'
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3 as q
assert runner_name not in sys.modules
with q._configured_predecessor_qualifier_v3():
    paths = dict(q.predecessor._source_paths())
print(json.dumps({{'count': len(paths), 'path': str(paths['v2_rocm_preregistration_source'])}}, sort_keys=True))
"""
    )
    assert old == {
        "count": 220,
        "path": str(
            qualifier.REPO_ROOT
            / "docs/lewm_go2_scene_diversity_recurrent_replication_"
            "genesis_rocm_backend_v3_preregistration_2026-08-04.md"
        ),
    }

    runner_first = _isolated_json(
        f"""
import json, sys
sys.path.insert(0, {repo!r})
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as r
print(json.dumps({{'paths': {{k: str(v) for k, v in r.SOURCE_PATHS.items()}}, 'binding': r.v2_preregistration_source_binding()}}, sort_keys=True))
"""
    )
    qualifier_first = _isolated_json(
        f"""
import json, sys
sys.path.insert(0, {repo!r})
runner_name = 'scripts.run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1'
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as q
assert runner_name not in sys.modules
with q._configured_predecessor_qualifier_v3():
    paths = dict(q.predecessor._source_paths())
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as r
print(json.dumps({{'paths': {{k: str(v) for k, v in paths.items()}}, 'binding': r.v2_preregistration_source_binding()}}, sort_keys=True))
"""
    )
    assert qualifier_first == runner_first
    poisoned_predecessor = _isolated_json(
        f"""
import json, sys
sys.path.insert(0, {repo!r})
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3 as old_q
with old_q._configured_predecessor_qualifier_v3():
    poisoned = dict(old_q.predecessor._source_paths())
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as replacement
print(json.dumps({{'poisoned_path': str(poisoned['v2_rocm_preregistration_source']), 'replacement_path': str(replacement.SOURCE_PATHS['v2_rocm_preregistration_source']), 'replacement_count': len(replacement.SOURCE_PATHS), 'binding': replacement.v2_preregistration_source_binding()}}, sort_keys=True))
"""
    )
    assert poisoned_predecessor["poisoned_path"].endswith(
        "genesis_rocm_backend_v3_preregistration_2026-08-04.md"
    )
    assert poisoned_predecessor["replacement_path"].endswith(
        "genesis_rocm_backend_v2_preregistration_2026-08-04.md"
    )
    assert poisoned_predecessor["replacement_count"] == len(
        runner_first["paths"]
    )
    assert poisoned_predecessor["binding"] == runner_first["binding"]
    assert runner_first["binding"] == {
        "path": str(qualifier.REPO_ROOT / "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_preregistration_2026-08-04.md"),
        "sha256": "f4d2b46ddb7a0ac97f95160f55c8aadd58f22ae1e63b7ab85e500c083a86b334",
        "byte_count": 4_007,
    }


def _set_exact_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = qualifier.plan_builder.rocm_execution_environment(
        "qualification"
    )
    keys = (
        set(qualifier.collector.kernel._SANITIZED_SELECTOR_KEYS)  # noqa: SLF001
        | set(qualifier.collector.ROCM_ADDITIONAL_SANITIZED_KEYS)
        | set(expected)
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in expected.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def exact_qualification_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_environment(monkeypatch)


@pytest.fixture(scope="module")
def qualification_plan() -> dict:
    runtime = qualifier.plan_builder.build_rocm_runtime_bindings()
    frozen = copy.deepcopy(
        qualifier.plan_builder.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return qualifier.plan_builder.build_qualification_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )


def test_qualification_contract_is_science_identical_v3_replacement() -> None:
    contract = qualifier.QUALIFICATION_CONTRACT
    assert contract["probe_scene_indices_in_order"] == [12, 0]
    assert contract["required_host_home"] == "/home/andrewknowles"
    assert contract["v2_runtime_payload_reuse_authorized"] is False
    assert contract["v3_runtime_payload_reuse_authorized"] is False
    assert contract["ld_lld_driver_entrypoint"] == str(
        qualifier.plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT
    )


def test_child_environment_overwrites_home_and_removes_host_aliases(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/tmp/ambient-home")
    monkeypatch.setenv("USER", "ambient-user")
    monkeypatch.setenv("LOGNAME", "ambient-logname")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    assert child["HOME"] == qualifier.plan_builder.REQUIRED_HOST_HOME
    assert all(key not in child for key in ("USER", "LOGNAME", "LANG"))
    assert child == qualification_plan["execution_contract"]["environment"]


def _mock_run(plan: dict, *, home: str):
    identity = {
        "arch_name": "gfx1201:sramecc-:xnack-",
        "device_name": "AMD Radeon AI PRO R9700",
        "genesis_backend_symbol": "gs.amdgpu",
        "genesis_file": plan["runtime_bindings"]["genesis_init_source"]["path"],
        "genesis_version": "0.4.6",
        "home": home,
        "hsa_override_present": False,
        "numpy_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "numpy/__init__.py"
            ).resolve()
        ),
        "pillow_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "PIL/__init__.py"
            ).resolve()
        ),
        "torch_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "torch/__init__.py"
            ).resolve()
        ),
        "torch_hip_version": "7.2.0",
        "torch_version": "2.12.0+rocm7.2",
        "visible_device_count": 1,
    }

    def run(argv, **_kwargs):
        args = [str(value) for value in argv]
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                args, 0, "AMD LLD 20.0.0 exact test banner\n", ""
            )
        if Path(args[0]).name == "rocminfo":
            return subprocess.CompletedProcess(args, 0, "Name: gfx1201\n", "")
        if args[0] == plan["execution_contract"]["python_invocation_path"]:
            return subprocess.CompletedProcess(
                args, 0, json.dumps(identity) + "\n", ""
            )
        egl = (
            "Device #0:\nOpenGL renderer: llvmpipe\n"
            "Device #1:\nOpenGL renderer: AMD Radeon AI PRO R9700\n"
        )
        return subprocess.CompletedProcess(args, 2, egl, "expected stderr")

    return run


def test_identity_receipt_records_exact_home(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qualifier.subprocess,
        "run",
        _mock_run(
            qualification_plan, home=qualifier.plan_builder.REQUIRED_HOST_HOME
        ),
    )
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    result = qualifier._run_rocm_egl_preflight(  # noqa: SLF001
        qualification_plan, child_env=child
    )
    assert result["identity"]["home"] == "/home/andrewknowles"
    assert result["environment"]["HOME"] == "/home/andrewknowles"


def test_identity_receipt_rejects_mutated_home(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qualifier.subprocess,
        "run",
        _mock_run(qualification_plan, home="/tmp/wrong-home"),
    )
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    with pytest.raises(
        qualifier.GenesisRocmBackendV3IntegrityReplacementV1QualificationError,
        match="identity preflight failed",
    ):
        qualifier._run_rocm_egl_preflight(  # noqa: SLF001
            qualification_plan, child_env=child
        )


def test_actual_genesis_import_home_oracle_without_initialization(
    tmp_path: Path,
) -> None:
    query = """
import json, os, resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import genesis as gs
print(json.dumps({
    'home': os.environ.get('HOME'),
    'genesis_version': gs.__version__,
    'genesis_backend_symbol': 'gs.amdgpu' if int(gs.amdgpu) == 3 else 'unexpected',
}, sort_keys=True))
"""
    argv = [
        str(qualifier.plan_builder.ROCM_PYTHON.absolute()),
        "-I",
        "-B",
        "-c",
        query,
    ]
    base = qualifier.plan_builder.rocm_execution_environment("qualification")
    base["GS_CACHE_FILE_PATH"] = str(tmp_path / "diagnostic-cache")
    without_home = dict(base)
    without_home.pop("HOME")
    failed = subprocess.run(
        argv,
        cwd=qualifier.REPO_ROOT,
        env=without_home,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    assert failed.returncode in {-6, 134}
    assert failed.stdout.strip() == ""
    assert "home != nullptr" in failed.stderr

    with_home = subprocess.run(
        argv,
        cwd=qualifier.REPO_ROOT,
        env=base,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    assert with_home.returncode == 0
    identity = json.loads(with_home.stdout.splitlines()[-1])
    assert identity == {
        "home": "/home/andrewknowles",
        "genesis_version": "0.4.6",
        "genesis_backend_symbol": "gs.amdgpu",
    }


def test_worker_argv_is_exact_v3_qualifier_child() -> None:
    kwargs = {
        "scene_index": 12,
        "plan_path": Path("/tmp/plan"),
        "expected_plan_byte_count": 1,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority"),
        "expected_authority_byte_count": 2,
        "expected_authority_sha256": "b" * 64,
        "reservation_binding": {"byte_count": 3, "file_sha256": "c" * 64},
        "orchestrator_nonce": "d" * 64,
    }
    argv = qualifier._worker_argv_qualification(**kwargs)  # noqa: SLF001
    assert argv[0] == str(qualifier.plan_builder.ROCM_PYTHON.absolute())
    assert argv[1] == str(Path(qualifier.__file__).resolve())


def test_qualification_authority_requires_v3_review_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = {
        "predecessor_v3_qualification_terminal_review_binding": {
            **qualifier.collector._standard_v3_review_binding(),  # noqa: SLF001
            "sha256": "0" * 64,
        }
    }
    monkeypatch.setattr(
        qualifier.predecessor,
        "validate_qualification_authority",
        lambda *_args, **_kwargs: (changed, {}, {}, {}),
    )
    with pytest.raises(
        qualifier.collector.SceneProcessCollectionError,
        match="V3 terminal-review binding changed",
    ):
        qualifier.validate_qualification_authority(Path("/unused"))
