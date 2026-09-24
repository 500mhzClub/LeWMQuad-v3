"""Captured runner for the development-only G3 exact-physical audit."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.machinery
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_MANIFEST = ROOT / "config/go2_generalization_v4/development.json"
GEOMETRY_CONTRACT = ROOT / "config/go2_generalization_geometry_v2.json"
SCENE_ROOT = ROOT / ".generated/scene_corpus/go2_generalization_v4/development"
EXPECTED_DEVELOPMENT_SHA256 = (
    "563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41"
)
EXPECTED_GEOMETRY_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
SOURCE_PATHS = (
    "lewm/benchmarks/go2_g3_exact_physical_equivalence.py",
    "lewm/planning/revisioned_physical_configuration_memory.py",
    "lewm/planning/zero_inflation_exact_physical_adapter_v1.py",
    "lewm/planning/geometry_contract.py",
    "lewm_worlds/lewm_worlds/manifest.py",
    "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v1.py",
    "scripts/audit_go2_g3_exact_physical_equivalence.py",
)
CAPTURED_MODULE_PATHS: Mapping[str, str] = {
    "lewm_worlds.manifest": "lewm_worlds/lewm_worlds/manifest.py",
    "lewm.planning.geometry_contract": "lewm/planning/geometry_contract.py",
    "lewm.planning.revisioned_physical_configuration_memory": (
        "lewm/planning/revisioned_physical_configuration_memory.py"
    ),
    "lewm.planning.zero_inflation_exact_physical_adapter_v1": (
        "lewm/planning/zero_inflation_exact_physical_adapter_v1.py"
    ),
    "lewm.benchmarks.go2_g3_exact_physical_equivalence": (
        "lewm/benchmarks/go2_g3_exact_physical_equivalence.py"
    ),
}
CAPTURED_MODULE_ORDER = (
    "lewm_worlds.manifest",
    "lewm.planning.geometry_contract",
    "lewm.planning.revisioned_physical_configuration_memory",
    "lewm.planning.zero_inflation_exact_physical_adapter_v1",
    "lewm.benchmarks.go2_g3_exact_physical_equivalence",
)
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPECTED_CAPTURED_MODULE_SHA256S: Mapping[str, str] = {
    "lewm.benchmarks.go2_g3_exact_physical_equivalence": (
        "b0155968a267afb08817987c3779e61e2e59b32e60281b1116a3757ac4fa461d"
    ),
    "lewm.planning.geometry_contract": (
        "6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b"
    ),
    "lewm.planning.revisioned_physical_configuration_memory": (
        "13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add"
    ),
    "lewm.planning.zero_inflation_exact_physical_adapter_v1": (
        "2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad"
    ),
    "lewm_worlds.manifest": (
        "5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888"
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _captured_project_sources() -> dict[str, tuple[str, bytes, str]]:
    captured: dict[str, tuple[str, bytes, str]] = {}
    for module_name, relative in CAPTURED_MODULE_PATHS.items():
        path = (ROOT / relative).resolve()
        payload = path.read_bytes()
        captured[module_name] = (
            str(path),
            payload,
            hashlib.sha256(payload).hexdigest(),
        )
    actual = {name: row[2] for name, row in captured.items()}
    if actual != dict(EXPECTED_CAPTURED_MODULE_SHA256S):
        raise PermissionError("captured G3 dependency source hashes are not frozen")
    return captured


def _install_captured_project_sources(
    captured: Mapping[str, tuple[str, bytes, str]],
) -> None:
    """Install reviewed bytes into fresh canonical module identities."""

    if set(captured) != set(CAPTURED_MODULE_PATHS):
        raise RuntimeError("captured project module closure is incomplete")
    protected_names = {
        "lewm",
        "lewm.planning",
        "lewm.benchmarks",
        "lewm_worlds",
        *CAPTURED_MODULE_PATHS,
    }
    preloaded = sorted(name for name in protected_names if name in sys.modules)
    if preloaded:
        raise RuntimeError(f"canonical project module was preloaded: {preloaded}")

    for package_name, package_path in (
        ("lewm", ROOT / "lewm"),
        ("lewm.planning", ROOT / "lewm/planning"),
        ("lewm.benchmarks", ROOT / "lewm/benchmarks"),
        ("lewm_worlds", ROOT / "lewm_worlds/lewm_worlds"),
    ):
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package
        if "." in package_name:
            parent_name, _, child_name = package_name.rpartition(".")
            setattr(sys.modules[parent_name], child_name, package)

    for module_name in CAPTURED_MODULE_ORDER:
        path_text, payload, expected_sha256 = captured[module_name]
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RuntimeError(f"captured module bytes changed: {module_name}")
        module = types.ModuleType(module_name)
        module.__file__ = path_text
        module.__package__ = module_name.rpartition(".")[0]
        module.__spec__ = importlib.machinery.ModuleSpec(
            module_name,
            loader=None,
            origin=path_text,
        )
        sys.modules[module_name] = module
        parent_name, _, child_name = module_name.rpartition(".")
        setattr(sys.modules[parent_name], child_name, module)
        try:
            exec(compile(payload, path_text, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise


def _strict_json(path: Path) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains non-finite number {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def _scene_path(family: str, scene_id: str) -> Path:
    path = (SCENE_ROOT / family / scene_id / "manifest.json").resolve()
    if not path.is_relative_to(SCENE_ROOT.resolve()) or path.is_symlink():
        raise ValueError("development scene path escaped the canonical root")
    return path


def _bind_job_result(
    job: tuple[int, str, str, str, str, str],
    result: Mapping[str, object],
) -> dict[str, object]:
    (
        index,
        scene_id,
        family,
        expected_manifest_sha256,
        runner_source_sha256,
        source_graph_sha256,
    ) = job
    job_core = {
        "index": index,
        "scene_id": scene_id,
        "family": family,
        "manifest_sha256": expected_manifest_sha256,
        "runner_source_sha256": runner_source_sha256,
        "source_graph_sha256": source_graph_sha256,
    }
    return {
        **job_core,
        "job_sha256": hashlib.sha256(_canonical_bytes(job_core)).hexdigest(),
        "result_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
    }


def _evaluate_job(
    job: tuple[int, str, str, str, str, str],
) -> tuple[int, dict[str, object], dict[str, object]]:
    (
        index,
        scene_id,
        family,
        expected_manifest_sha256,
        runner_source_sha256,
        source_graph_sha256,
    ) = job
    for name in THREAD_CAPS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"worker requires {name}=1")
    from lewm.benchmarks.go2_g3_exact_physical_equivalence import (
        evaluate_exact_scene,
    )
    from lewm.planning.geometry_contract import load_geometry_contract
    from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict

    path = _scene_path(family, scene_id)
    payload = _strict_json(path)
    manifest = parse_scene_manifest_dict(dict(payload))
    if manifest.scene_id != scene_id or manifest.family != family:
        raise ValueError(f"development scene identity mismatch: {scene_id}")
    if manifest_sha256(manifest) != expected_manifest_sha256:
        raise ValueError(f"development scene semantic hash mismatch: {scene_id}")
    geometry = load_geometry_contract(
        GEOMETRY_CONTRACT,
        repository_root=ROOT,
        verify_sources=True,
    )
    result = evaluate_exact_scene(manifest, geometry).to_dict()
    return index, result, _bind_job_result(job, result)


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _worker_identity_probe(_value: int) -> tuple[int, str, str]:
    return os.getpid(), __name__, _evaluate_job.__module__


def _sealed_bootstrap_probe(expected_runner_source_sha256: str) -> dict[str, object]:
    actual = _hash_file(Path(__file__))
    if actual != expected_runner_source_sha256:
        raise PermissionError("sealed G3 runner source changed")
    captured = _captured_project_sources()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        worker_pid, worker_module, worker_evaluate_module = executor.submit(
            _worker_identity_probe, 0
        ).result()
    probe_job = (
        0,
        "synthetic-probe-scene",
        "synthetic-probe-family",
        "1" * 64,
        expected_runner_source_sha256,
        "2" * 64,
    )
    probe_result = {"synthetic_probe": True}
    return {
        "schema": "lewm_go2_g3_exact_physical_runner_bootstrap_probe_v1",
        "runner_source_sha256": actual,
        "runner_module_name": __name__,
        "evaluate_job_module_name": _evaluate_job.__module__,
        "captured_runner_executed": _evaluate_job.__module__ == __name__,
        "coordinator_pid": os.getpid(),
        "one_worker_pid": worker_pid,
        "one_worker_module_name": worker_module,
        "one_worker_evaluate_job_module_name": worker_evaluate_module,
        "one_worker_crossed_process_boundary": worker_pid != os.getpid(),
        "captured_module_sha256s": {
            name: row[2] for name, row in sorted(captured.items())
        },
        "synthetic_job_binding": _bind_job_result(probe_job, probe_result),
    }


def _sealed_run(
    *,
    output: Path,
    workers: int,
    expected_runner_source_sha256: str,
) -> dict[str, object]:
    if workers < 1 or workers > 6:
        raise ValueError("workers must be between 1 and 6")
    for name in THREAD_CAPS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"runner requires {name}=1")
    if output.exists():
        raise FileExistsError(f"exact-equivalence output already exists: {output}")
    if _hash_file(Path(__file__)) != expected_runner_source_sha256:
        raise PermissionError("sealed G3 runner source changed before execution")
    if _hash_file(DEVELOPMENT_MANIFEST) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("development manifest hash changed")
    if _hash_file(GEOMETRY_CONTRACT) != EXPECTED_GEOMETRY_SHA256:
        raise ValueError("geometry contract hash changed")
    captured = _captured_project_sources()
    source_bindings = {relative: _hash_file(ROOT / relative) for relative in SOURCE_PATHS}
    for module_name, relative in CAPTURED_MODULE_PATHS.items():
        if captured[module_name][2] != source_bindings[relative]:
            raise RuntimeError(f"captured source hash differs from disk: {module_name}")
    development = _strict_json(DEVELOPMENT_MANIFEST)
    records = development.get("validation_scenes")
    if not isinstance(records, list) or len(records) != 24:
        raise ValueError("development manifest must bind exactly 24 validation scenes")
    source_graph_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "runner_source_sha256": expected_runner_source_sha256,
                "captured_module_sha256s": dict(EXPECTED_CAPTURED_MODULE_SHA256S),
            }
        )
    ).hexdigest()
    jobs: list[tuple[int, str, str, str, str, str]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError("development scene record must be an object")
        scene_id = str(raw.get("scene_id", ""))
        family = str(raw.get("family", ""))
        manifest_sha256 = str(raw.get("manifest_sha256", ""))
        if not scene_id or not family or len(manifest_sha256) != 64:
            raise ValueError("development scene binding is incomplete")
        jobs.append(
            (
                index,
                scene_id,
                family,
                manifest_sha256,
                expected_runner_source_sha256,
                source_graph_sha256,
            )
        )
    if len({job[1] for job in jobs}) != 24:
        raise ValueError("development scene identities must be unique")

    # Even one-worker execution crosses a process boundary. Fork inherits this
    # captured private runner module; every child then installs only the frozen
    # project-source closure before evaluating its bound jobs.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_install_captured_project_sources,
        initargs=(captured,),
    ) as executor:
        indexed = list(executor.map(_evaluate_job, jobs))
    indexed.sort(key=lambda row: row[0])
    if len(indexed) != len(jobs):
        raise RuntimeError("sealed G3 runner returned an incomplete job inventory")
    for job, (index, payload, binding) in zip(jobs, indexed, strict=True):
        if index != job[0] or binding != _bind_job_result(job, payload):
            raise PermissionError("sealed G3 job/result binding changed")

    # Reconstruct the frozen result values to run the pure suite finalizer.
    _install_captured_project_sources(captured)
    from lewm.benchmarks.go2_g3_exact_physical_equivalence import (
        G3ExactSceneResult,
        summarize_exact_scenes,
    )

    fields = tuple(G3ExactSceneResult.__dataclass_fields__)
    scene_results = [
        G3ExactSceneResult(**{name: payload[name] for name in fields})
        for _index, payload, _binding in indexed
    ]
    if source_bindings != {
        relative: _hash_file(ROOT / relative) for relative in SOURCE_PATHS
    }:
        raise RuntimeError("audited source changed during execution")
    if (
        source_bindings[
            "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v1.py"
        ]
        != expected_runner_source_sha256
    ):
        raise PermissionError("sealed G3 runner hash differs from its result binding")
    result = summarize_exact_scenes(
        scene_results,
        source_bindings={
            **source_bindings,
            "config/go2_generalization_v4/development.json": (
                EXPECTED_DEVELOPMENT_SHA256
            ),
            "config/go2_generalization_geometry_v2.json": EXPECTED_GEOMETRY_SHA256,
        },
    )
    result["worker_count"] = workers
    result["threads_per_worker"] = 1
    result["development_scene_order"] = [job[1] for job in jobs]
    result["runner_source_sha256"] = expected_runner_source_sha256
    result["source_graph_sha256"] = source_graph_sha256
    result["job_bindings"] = [binding for _index, _payload, binding in indexed]
    # The pure summary hash intentionally excludes runner-only execution fields.
    result["summary_content_sha256"] = result.pop("content_sha256")
    result["content_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_atomic(output, result)
    return result
