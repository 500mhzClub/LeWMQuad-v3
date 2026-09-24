from __future__ import annotations

import copy
import json
from pathlib import Path
import resource
import subprocess
import sys

import pytest

from scripts import (
    qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2
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
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as r
print(json.dumps({{'paths': {{k: str(v) for k, v in r.SOURCE_PATHS.items()}}, 'binding': r.v2_preregistration_source_binding()}}, sort_keys=True))
"""
    )
    qualifier_first = _isolated_json(
        f"""
import json, sys
sys.path.insert(0, {repo!r})
runner_name = 'scripts.run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2'
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as q
assert runner_name not in sys.modules
with q._configured_predecessor_qualifier_v3():
    paths = dict(q.predecessor._source_paths())
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as r
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
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as replacement
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
        qualifier.plan_builder.predecessor.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
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
        qualifier.GenesisRocmBackendV3IntegrityReplacementV2QualificationError,
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


def test_qualification_authority_requires_replacement_v1_review_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = {
        "predecessor_replacement_v1_qualification_terminal_review_binding": {
            **qualifier.collector._standard_replacement_v1_review_binding(),  # noqa: SLF001
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
        match="replacement V1 terminal-review binding changed",
    ):
        qualifier.validate_qualification_authority(Path("/unused"))


def test_fresh_process_validates_realistic_authority_and_reaches_reservation_seam() -> None:
    repo = str(qualifier.REPO_ROOT)
    result = _isolated_json(
        f"""
import copy, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, {repo!r})
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as q
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as r

def write(path, value):
    path.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode() + b'\\n')
    return r.file_binding_v1(path)

def config(module):
    for name in ('_configuration_overrides_v3', '_configuration_overrides_v2', '_configuration_overrides_rocm'):
        if hasattr(module, name):
            return getattr(module, name)()
    return None

def snapshot(first, count):
    rows = []
    module = first
    for _ in range(count):
        values = config(module)
        if values is not None:
            target = module.predecessor
            rows.append((target, {{name: getattr(target, name) for name in values}}))
        module = getattr(module, 'predecessor', None)
        if module is None:
            break
    return rows

def restored(rows):
    return all(getattr(target, name) is value for target, values in rows for name, value in values.items())

assert not q.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
assert not q.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
expected_env = q.plan_builder.rocm_execution_environment('qualification')
keys = set(q.collector.kernel._SANITIZED_SELECTOR_KEYS) | set(q.collector.ROCM_ADDITIONAL_SANITIZED_KEYS) | set(expected_env)
for key in keys:
    os.environ.pop(key, None)
os.environ.update(expected_env)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / 'source.py'
    source.write_text('VALUE = 1\\n')
    prereg = root / 'prereg.md'
    prereg.write_text('# Sentinel preregistration\\n')
    runtime = q.plan_builder.build_rocm_runtime_bindings()
    frozen = copy.deepcopy(q.plan_builder.predecessor.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN)
    plan = q.plan_builder.build_qualification_plan(frozen_plan=frozen, runtime_bindings=runtime)
    plan_path = root / 'plan.json'
    plan_binding = write(plan_path, plan)
    sources = {{'sentinel_source': source}}
    source_bindings = {{name: r.file_binding_v1(path) for name, path in sources.items()}}
    review = {{'status': 'PASS_INDEPENDENT_SOURCE_REVIEW', 'findings': [], 'protected_material_opened': False, 'qualification_plan_binding': plan_binding, 'source_bindings': source_bindings}}
    review_path = root / 'source_review.json'
    review_binding = write(review_path, review)
    authority = {{
        'schema': q.QUALIFICATION_AUTHORITY_SCHEMA,
        'status': q.QUALIFICATION_AUTHORITY_STATUS,
        'attempt_id': q.plan_builder.QUALIFICATION_ATTEMPT_ID,
        'attempt_root': str(q.plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve()),
        'collection_root': str(q.plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve()),
        'plan_binding': plan_binding,
        'preregistration_binding': r.file_binding_v1(prereg),
        'source_review_binding': review_binding,
        'source_bindings': source_bindings,
        'dino': {{}},
        'config': {{}},
        'caps': copy.deepcopy(q.collector.EXPECTED_CAPS),
        'permissions': copy.deepcopy(q.collector.EXPECTED_PERMISSIONS),
        'qualification_contract': copy.deepcopy(q.QUALIFICATION_CONTRACT),
        'predecessor_cpu_terminal_review_binding': r._standard_binding(q.plan_builder.CPU_TERMINAL_REVIEW_BINDING),
        'predecessor_v1_qualification_terminal_review_binding': dict(q.collector._EXACT_V1_REVIEW_BINDING),
        'predecessor_v2_qualification_terminal_review_binding': dict(q.collector._EXACT_V2_REVIEW_BINDING),
        'predecessor_v3_qualification_terminal_review_binding': dict(q.collector._EXACT_V3_REVIEW_BINDING),
        'predecessor_replacement_v1_qualification_terminal_review_binding': dict(q.collector._EXACT_REPLACEMENT_V1_REVIEW_BINDING),
    }}
    assert set(authority) == q.QUALIFICATION_AUTHORITY_FIELDS
    authority_path = root / 'authority.json'
    authority_binding = write(authority_path, authority)
    q.PREREGISTRATION = prereg
    q.SOURCE_REVIEW = review_path
    q.QUALIFICATION_AUTHORITY = authority_path
    q._source_paths = lambda: sources

    qualifier_state = snapshot(q, 4)
    collector_state = snapshot(q.collector, 5)
    validated = q.validate_qualification_authority(authority_path, expected_sha256=authority_binding['sha256'], expected_byte_count=authority_binding['byte_count'])

    def bomb(*_args, **_kwargs):
        raise AssertionError('legacy helper was invoked')
    module = r.predecessor
    for _ in range(4):
        if hasattr(module, 'predecessor_failure_bindings_rocm'):
            module.predecessor_failure_bindings_rocm = bomb
        module = getattr(module, 'predecessor', None)
        if module is None:
            break
    assert len(r.predecessor_failure_bindings_rocm()) == 12

    class ReservationSeam(RuntimeError):
        pass
    base = q.predecessor.predecessor.predecessor.predecessor
    original_reserve = base._reserve_qualification
    calls = []
    def stop(**kwargs):
        calls.append(kwargs)
        raise ReservationSeam('reservation seam reached')
    base._reserve_qualification = stop
    try:
        q.execute_qualification(validated[0], authority_binding=validated[1], plan=validated[2], plan_binding=validated[3])
    except ReservationSeam:
        pass
    finally:
        base._reserve_qualification = original_reserve

    assert len(calls) == 1
    assert restored(qualifier_state)
    assert restored(collector_state)
    assert not q.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
    assert not q.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    print(json.dumps({{'validated_fields': len(validated[0]), 'reservation_calls': len(calls), 'restored': True}}, sort_keys=True))
"""
    )
    assert result == {
        "reservation_calls": 1,
        "restored": True,
        "validated_fields": 19,
    }
