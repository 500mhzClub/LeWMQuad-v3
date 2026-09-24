"""Captured runner for the additive two-resolution G3 V2 audit candidate."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.machinery
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import stat
import sys
import types
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_MANIFEST = ROOT / "config/go2_generalization_v4/development.json"
GEOMETRY_CONTRACT = ROOT / "config/go2_generalization_geometry_v2.json"
SCENE_ROOT = ROOT / ".generated/scene_corpus/go2_generalization_v4/development"
CANONICAL_OUTPUT = (
    ROOT / ".generated/go2_g3_exact_physical_equivalence/v2/candidate.json"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41"
)
EXPECTED_GEOMETRY_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
EXPECTED_PROFILE_SHA256 = (
    "2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314"
)
GOVERNING_DESIGN_PATH = (
    "docs/lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md"
)
EXPECTED_GOVERNING_DESIGN_SHA256 = (
    "a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1"
)
SOURCE_PATHS = (
    "lewm/benchmarks/go2_g3_exact_physical_equivalence.py",
    "lewm/benchmarks/go2_g3_exact_physical_equivalence_v2.py",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
    "lewm/planning/revisioned_physical_configuration_memory.py",
    "lewm/planning/two_resolution_configuration_projection_v2.py",
    "lewm/planning/zero_inflation_exact_physical_adapter_v1.py",
    "lewm/planning/geometry_contract.py",
    "lewm_worlds/lewm_worlds/manifest.py",
    "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py",
    "scripts/audit_go2_g3_exact_physical_equivalence_v2.py",
    "docs/lewm_go2_g3_exact_physical_equivalence_v2_amendment_2026-07-13.md",
    GOVERNING_DESIGN_PATH,
    "docs/lewm_go2_observable_camera_ray_evidence_v4_contract_2026-07-12.md",
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
    "lewm.benchmarks.go2_observable_camera_ray_evidence_v4": (
        "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
    ),
    "lewm.benchmarks.go2_g3_exact_physical_equivalence": (
        "lewm/benchmarks/go2_g3_exact_physical_equivalence.py"
    ),
    "lewm.planning.two_resolution_configuration_projection_v2": (
        "lewm/planning/two_resolution_configuration_projection_v2.py"
    ),
    "lewm.benchmarks.go2_g3_exact_physical_equivalence_v2": (
        "lewm/benchmarks/go2_g3_exact_physical_equivalence_v2.py"
    ),
}
CAPTURED_MODULE_ORDER = (
    "lewm_worlds.manifest",
    "lewm.planning.geometry_contract",
    "lewm.planning.revisioned_physical_configuration_memory",
    "lewm.planning.zero_inflation_exact_physical_adapter_v1",
    "lewm.benchmarks.go2_observable_camera_ray_evidence_v4",
    "lewm.benchmarks.go2_g3_exact_physical_equivalence",
    "lewm.planning.two_resolution_configuration_projection_v2",
    "lewm.benchmarks.go2_g3_exact_physical_equivalence_v2",
)
EXPECTED_CAPTURED_MODULE_SHA256S: Mapping[str, str] = {
    "lewm.benchmarks.go2_g3_exact_physical_equivalence": (
        "b0155968a267afb08817987c3779e61e2e59b32e60281b1116a3757ac4fa461d"
    ),
    "lewm.benchmarks.go2_g3_exact_physical_equivalence_v2": (
        "a626a726b2837c6dd8cfacd6d7be3b796278b127ea998ff3a3b894bbf7d69823"
    ),
    "lewm.benchmarks.go2_observable_camera_ray_evidence_v4": (
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
    ),
    "lewm.planning.geometry_contract": (
        "6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b"
    ),
    "lewm.planning.revisioned_physical_configuration_memory": (
        "13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add"
    ),
    "lewm.planning.two_resolution_configuration_projection_v2": (
        "3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107"
    ),
    "lewm.planning.zero_inflation_exact_physical_adapter_v1": (
        "2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad"
    ),
    "lewm_worlds.manifest": (
        "5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888"
    ),
}
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    if {name: row[2] for name, row in captured.items()} != dict(
        EXPECTED_CAPTURED_MODULE_SHA256S
    ):
        raise PermissionError("captured G3 V2 dependency hashes are not frozen")
    return captured


def _install_captured_project_sources(
    captured: Mapping[str, tuple[str, bytes, str]],
) -> None:
    if set(captured) != set(CAPTURED_MODULE_PATHS):
        raise RuntimeError("captured G3 V2 project closure is incomplete")
    protected = {
        "lewm",
        "lewm.planning",
        "lewm.benchmarks",
        "lewm_worlds",
        *CAPTURED_MODULE_PATHS,
    }
    preloaded = sorted(name for name in protected if name in sys.modules)
    if preloaded:
        raise RuntimeError(f"canonical G3 V2 module was preloaded: {preloaded}")
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
        path_text, payload, expected = captured[module_name]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f"captured G3 V2 bytes changed: {module_name}")
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
    job: tuple[int, str, str, str, str, str, str],
    result: Mapping[str, object],
) -> dict[str, object]:
    (
        index,
        scene_id,
        family,
        expected_manifest_sha256,
        runner_source_sha256,
        source_graph_sha256,
        profile_sha256,
    ) = job
    job_core = {
        "index": index,
        "scene_id": scene_id,
        "family": family,
        "manifest_sha256": expected_manifest_sha256,
        "runner_source_sha256": runner_source_sha256,
        "source_graph_sha256": source_graph_sha256,
        "profile_sha256": profile_sha256,
    }
    return {
        **job_core,
        "job_sha256": hashlib.sha256(_canonical_bytes(job_core)).hexdigest(),
        "result_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
    }


def _evaluate_job(
    job: tuple[int, str, str, str, str, str, str],
) -> tuple[int, dict[str, object], dict[str, object]]:
    index, scene_id, family, expected_manifest_sha256, *_rest = job
    for name in THREAD_CAPS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"G3 V2 worker requires {name}=1")
    from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
        evaluate_exact_scene_v2,
    )
    from lewm.planning.geometry_contract import load_geometry_contract
    from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict

    path = _scene_path(family, scene_id)
    manifest = parse_scene_manifest_dict(dict(_strict_json(path)))
    if manifest.scene_id != scene_id or manifest.family != family:
        raise ValueError(f"development scene identity mismatch: {scene_id}")
    if manifest_sha256(manifest) != expected_manifest_sha256:
        raise ValueError(f"development scene semantic hash mismatch: {scene_id}")
    geometry = load_geometry_contract(
        GEOMETRY_CONTRACT,
        repository_root=ROOT,
        verify_sources=True,
    )
    result = evaluate_exact_scene_v2(manifest, geometry).to_dict()
    return index, result, _bind_job_result(job, result)


def _open_verified_publication_parent(path: Path) -> int:
    parent = Path(os.path.abspath(path.parent))
    parent.mkdir(parents=True, exist_ok=True)
    for candidate in (parent, *parent.parents):
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("publication parent chain must be real directories")
        if candidate == Path(candidate.anchor):
            break
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_DIRECTORY", 0
    ) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(parent, flags)


def _assert_canonical_output_path(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError("G3 V2 output must be a canonical Path")
    lexical = Path(os.path.abspath(path))
    if lexical != CANONICAL_OUTPUT or path.name != "candidate.json":
        raise PermissionError("G3 V2 runner output path is not canonical")


def _assert_destination_absent(path: Path) -> None:
    directory_fd = _open_verified_publication_parent(path)
    try:
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"G3 V2 output already exists: {path}")
    finally:
        os.close(directory_fd)


def _write_atomic_no_replace(path: Path, payload: Mapping[str, object]) -> None:
    """Publish complete JSON with an atomic hard-link create, never replacement."""

    if not isinstance(path, Path) or not path.name or path.name in {".", ".."}:
        raise ValueError("publication path must have a simple filename")
    encoded = _canonical_bytes(payload) + b"\n"
    directory_fd = _open_verified_publication_parent(path)
    temporary_name = f".g3-v2-{os.getpid()}-{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing G3 V2 candidate")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"G3 V2 output was created concurrently: {path}"
            ) from None
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _worker_identity_probe(_value: int) -> tuple[int, str, str]:
    return os.getpid(), __name__, _evaluate_job.__module__


def _sealed_bootstrap_probe(expected_runner_source_sha256: str) -> dict[str, object]:
    if _hash_file(Path(__file__)) != expected_runner_source_sha256:
        raise PermissionError("sealed G3 V2 runner source changed")
    if _hash_file(ROOT / GOVERNING_DESIGN_PATH) != EXPECTED_GOVERNING_DESIGN_SHA256:
        raise PermissionError("governing G3 V2 design contract hash changed")
    captured = _captured_project_sources()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        worker_pid, worker_module, worker_evaluate_module = executor.submit(
            _worker_identity_probe,
            0,
        ).result()
    probe_job = (
        0,
        "synthetic-v2-probe-scene",
        "synthetic-v2-probe-family",
        "1" * 64,
        expected_runner_source_sha256,
        "2" * 64,
        EXPECTED_PROFILE_SHA256,
    )
    return {
        "schema": "lewm_go2_g3_exact_physical_runner_bootstrap_probe_v2",
        "runner_source_sha256": expected_runner_source_sha256,
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
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "governing_design_path": GOVERNING_DESIGN_PATH,
        "governing_design_sha256": EXPECTED_GOVERNING_DESIGN_SHA256,
        "synthetic_job_binding": _bind_job_result(
            probe_job,
            {"synthetic_probe": True, "production_promotion_authorized": False},
        ),
        "production_promotion_authorized": False,
    }


def _sealed_run(
    *,
    output: Path,
    workers: int,
    expected_runner_source_sha256: str,
) -> dict[str, object]:
    if workers < 1 or workers > 6:
        raise ValueError("workers must be between 1 and 6")
    _assert_canonical_output_path(output)
    _assert_destination_absent(output)
    for name in THREAD_CAPS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"G3 V2 runner requires {name}=1")
    if _hash_file(Path(__file__)) != expected_runner_source_sha256:
        raise PermissionError("sealed G3 V2 runner source changed before execution")
    if _hash_file(DEVELOPMENT_MANIFEST) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("development manifest hash changed")
    if _hash_file(GEOMETRY_CONTRACT) != EXPECTED_GEOMETRY_SHA256:
        raise ValueError("geometry contract hash changed")
    captured = _captured_project_sources()
    source_bindings = {relative: _hash_file(ROOT / relative) for relative in SOURCE_PATHS}
    if (
        source_bindings.get(GOVERNING_DESIGN_PATH)
        != EXPECTED_GOVERNING_DESIGN_SHA256
    ):
        raise PermissionError("governing G3 V2 design contract hash changed")
    for module_name, relative in CAPTURED_MODULE_PATHS.items():
        if captured[module_name][2] != source_bindings[relative]:
            raise RuntimeError(f"captured source differs from disk: {module_name}")
    development = _strict_json(DEVELOPMENT_MANIFEST)
    records = development.get("validation_scenes")
    if not isinstance(records, list) or len(records) != 24:
        raise ValueError("development manifest must bind exactly 24 scenes")
    source_graph_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "runner_source_sha256": expected_runner_source_sha256,
                "captured_module_sha256s": dict(EXPECTED_CAPTURED_MODULE_SHA256S),
                "profile_sha256": EXPECTED_PROFILE_SHA256,
                "governing_design_path": GOVERNING_DESIGN_PATH,
                "governing_design_sha256": EXPECTED_GOVERNING_DESIGN_SHA256,
                "source_bindings": dict(sorted(source_bindings.items())),
            }
        )
    ).hexdigest()
    jobs: list[tuple[int, str, str, str, str, str, str]] = []
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
                EXPECTED_PROFILE_SHA256,
            )
        )
    if len({job[1] for job in jobs}) != 24:
        raise ValueError("development scene identities must be unique")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_install_captured_project_sources,
        initargs=(captured,),
    ) as executor:
        indexed = list(executor.map(_evaluate_job, jobs))
    indexed.sort(key=lambda row: row[0])
    if len(indexed) != len(jobs):
        raise RuntimeError("sealed G3 V2 runner returned incomplete results")
    for job, (index, payload, binding) in zip(jobs, indexed, strict=True):
        if index != job[0] or binding != _bind_job_result(job, payload):
            raise PermissionError("sealed G3 V2 job/result binding changed")

    _install_captured_project_sources(captured)
    from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
        G3ExactSceneResultV2,
        summarize_exact_scenes_v2,
    )

    fields = tuple(G3ExactSceneResultV2.__dataclass_fields__)
    scene_results = [
        G3ExactSceneResultV2(**{name: payload[name] for name in fields})
        for _index, payload, _binding in indexed
    ]
    if source_bindings != {
        relative: _hash_file(ROOT / relative) for relative in SOURCE_PATHS
    }:
        raise RuntimeError("G3 V2 audited source changed during execution")
    result = summarize_exact_scenes_v2(
        scene_results,
        source_bindings={
            **source_bindings,
            "config/go2_generalization_v4/development.json": (
                EXPECTED_DEVELOPMENT_SHA256
            ),
            "config/go2_generalization_geometry_v2.json": EXPECTED_GEOMETRY_SHA256,
        },
    )
    result.update(
        {
            "worker_count": workers,
            "threads_per_worker": 1,
            "development_scene_order": [job[1] for job in jobs],
            "runner_source_sha256": expected_runner_source_sha256,
            "source_graph_sha256": source_graph_sha256,
            "job_bindings": [binding for _index, _payload, binding in indexed],
        }
    )
    result["summary_content_sha256"] = result.pop("content_sha256")
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    _write_atomic_no_replace(output, result)
    return result
