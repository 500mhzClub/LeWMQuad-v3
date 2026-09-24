from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from scripts import run_go2_shared_jepa_v5_matched_training_v2 as runner


contract = runner.contract
installed = contract.install_successor(runner.base, runner._BASE_NAMESPACE_SNAPSHOT)
runtime_contract = installed.contract


ROOT = Path(__file__).resolve().parents[2]


def _bound_json(binding: dict[str, object]) -> dict[str, object]:
    raw = runner.base._read_regular(
        ROOT / str(binding["path"]),
        expected_sha256=str(binding["file_sha256"]),
    )
    assert len(raw) == binding["byte_count"]
    return contract.parse_canonical_json(raw, name=str(binding["path"]))


def _v2_review(sources: dict[str, str]) -> dict[str, object]:
    original = _bound_json(contract.V1_REVIEW_BINDING)
    core = {key: value for key, value in original.items() if key != "content_sha256"}
    core.update(
        schema=contract.REVIEW_SCHEMA,
        reviewed_sources=sources,
        science_contract=contract.science_contract(),
    )
    return contract.with_content_sha256(core)


def _v2_authorization(review_binding: dict[str, object]) -> dict[str, object]:
    original = _bound_json(contract.V1_AUTHORIZATION_BINDING)
    core = {key: value for key, value in original.items() if key != "content_sha256"}
    core.update(
        schema=contract.AUTHORIZATION_SCHEMA,
        independent_review=review_binding,
        experiment=contract.science_contract(),
    )
    return contract.with_content_sha256(core)


def _copy_predecessor(root: Path) -> Path:
    predecessor = root / contract.PREDECESSOR_ROOT_RELATIVE_PATH
    predecessor.mkdir(parents=True)
    source = ROOT / contract.PREDECESSOR_ROOT_RELATIVE_PATH
    for name in contract.PREDECESSOR_ARTIFACT_BINDINGS:
        (predecessor / name).write_bytes((source / name).read_bytes())
    for binding in (
        contract.V1_REVIEW_BINDING,
        contract.V1_AUTHORIZATION_BINDING,
        contract.V1_TERMINAL_AUDIT_BINDING,
    ):
        destination = root / str(binding["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / str(binding["path"])).read_bytes())
    return predecessor


def _endpoint(
    endpoint_id: str,
    absolute: str,
    *,
    image_sha: str = "2" * 64,
    role: str = "train",
) -> dict[str, object]:
    return {
        "endpoint_identity_sha256": endpoint_id,
        "image_path_metadata_only": absolute,
        "image_sha256_commitment_only": image_sha,
        "content_sha256": "3" * 64,
        "dataset_role": role,
    }


def test_overlay_is_lean_hash_bound_and_science_identical() -> None:
    for path, expected in contract.V1_SOURCE_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected

    contract_lines = (ROOT / contract.CONTRACT_RELATIVE_PATH).read_text().count("\n")
    runner_text = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
    runner_lines = runner_text.count("\n")
    contract_text = (ROOT / contract.CONTRACT_RELATIVE_PATH).read_text()
    assert contract_lines <= 650
    assert runner_lines <= 80
    assert contract_lines + runner_lines <= 750
    for duplicated_training_operation in (
        "optimizer.step(",
        ".backward(",
        "forward_training_pair(",
        "torch.optim.AdamW(",
    ):
        assert duplicated_training_operation not in contract_text + runner_text
    assert "return super().train_arm(**kwargs)" in contract_text
    assert installed.RawInputs.__mro__[1].__name__ == "RawInputs"
    assert installed.Trainer.__mro__[1].__name__ == "Trainer"
    assert "self.bound_endpoints = dict(self.endpoints)" in contract_text

    v1_science = contract.v1_science_contract()
    v2_science = contract.science_contract()
    normalized_v1 = copy.deepcopy(v1_science)
    normalized_v2 = copy.deepcopy(v2_science)
    normalized_v1["candidate"].pop("schema")
    normalized_v2["candidate"].pop("schema")
    assert normalized_v2 == normalized_v1
    assert contract.canonical_json_sha256(v1_science) == (
        "b37769fc976de5d2b04b9ec9bac8aadb87776baa3b2c5865686b90d0233ea5cd"
    )
    assert contract.canonical_json_sha256(normalized_v2) == (
        "f9c8df78fff0585d16b9fc87eb4aa677ae30b6944b64e096d9394c52ee05bd54"
    )
    assert v2_science["maximum_attempts"] == 1
    assert v2_science["retry_authorized"] is False
    assert contract.AUTOMATIC_V3_AUTHORIZED is False
    assert contract.SOURCE_PATHS == (
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.TEST_RELATIVE_PATH,
        contract.TERMINAL_AUDIT_RELATIVE_PATH,
        *contract._v1.SOURCE_PATHS,
    )
    assert tuple(contract.current_source_bindings()) == contract.SOURCE_PATHS
    assert runtime_contract.science_contract() == v2_science


def test_lifecycle_is_distinct_and_child_reinstalls_v2_without_credentials() -> None:
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != contract.PREDECESSOR_ROOT_RELATIVE_PATH
    assert contract.REVIEW_RELATIVE_PATH != contract.V1_REVIEW_RELATIVE_PATH
    assert contract.AUTHORIZATION_RELATIVE_PATH != contract.V1_AUTHORIZATION_RELATIVE_PATH
    for schema in (
        contract.REVIEW_SCHEMA,
        contract.AUTHORIZATION_SCHEMA,
        contract.RESERVATION_SCHEMA,
        contract.SCHEDULE_SCHEMA,
        contract.SNAPSHOT_SCHEMA,
        contract.SELECTION_SCHEMA,
        contract.CALIBRATION_SCHEMA,
        contract.PRE_G2_CHECKPOINT_SCHEMA,
        contract.RESULT_SCHEMA,
        contract.VERIFICATION_SCHEMA,
        contract.COMPLETION_SCHEMA,
        contract.FAILURE_SCHEMA,
    ):
        assert "matched_training_v2" in schema
    command = installed._child_command()
    assert command[1:3] == ("-I", "-B")
    assert command[3] == str(ROOT / contract.RUNNER_RELATIVE_PATH)
    assert command[4:] == ("--internal-verify",)
    assert installed.parse_args(["--internal-verify"]).internal_verify is True
    with pytest.raises(SystemExit):
        installed.parse_args(["--internal-verify", "--review-sha256", "a" * 64])
    with pytest.raises(SystemExit):
        installed.parse_args(["--run"])
    with pytest.raises(RuntimeError, match="already installed"):
        contract.install_successor(installed, runner._BASE_NAMESPACE_SNAPSHOT)


@pytest.mark.parametrize("mutation", ("replace", "add", "delete"))
def test_installer_rejects_a_drifted_private_v1_baseline(mutation: str) -> None:
    fresh = runner._load_module(
        ROOT / contract.V1_RUNNER_RELATIVE_PATH,
        f"_lewm_go2_matched_v2_drifted_runner_test_{mutation}",
        expected_sha256=contract.V1_SOURCE_SHA256[contract.V1_RUNNER_RELATIVE_PATH],
    )
    snapshot = MappingProxyType(dict(vars(fresh)))
    if mutation == "replace":
        fresh._load_runtime = lambda: None
    elif mutation == "add":
        fresh.undeclared_runtime_hook = object()
    else:
        del fresh._load_runtime
    with pytest.raises(PermissionError, match="namespace drifted"):
        contract.install_successor(fresh, snapshot)


def test_isolated_import_loads_no_accelerator_or_image_stack() -> None:
    source = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(ROOT / contract.RUNNER_RELATIVE_PATH)!r})
spec = importlib.util.spec_from_file_location('_v2_isolated_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not {{'torch', 'numpy', 'PIL'}} & set(sys.modules)
assert module.contract.INSTALLATION_SENTINEL not in vars(module.base)
installed = module.contract.install_successor(
    module.base, module._BASE_NAMESPACE_SNAPSHOT
)
assert installed._child_command()[3] == str(path)
try:
    module.contract.install_successor(installed, module._BASE_NAMESPACE_SNAPSHOT)
except RuntimeError:
    pass
else:
    raise AssertionError('second installation was accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source],
        cwd=ROOT,
        env={"PATH": str(Path(sys.executable).parent), "HIP_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_predecessor_is_terminal_before_any_training() -> None:
    predecessor = contract.validate_predecessor(installed._read_regular)
    root = ROOT / contract.PREDECESSOR_ROOT_RELATIVE_PATH
    assert sorted(item.name for item in root.iterdir()) == sorted(
        contract.PREDECESSOR_ARTIFACT_BINDINGS
    )
    failed = predecessor["artifacts"]["failed.json"]
    assert failed["status"] == "failed_infrastructure"
    assert failed["stage"] == "promoted_training"
    assert failed["error"] == {
        "type": "ValueError",
        "message": "development RGB path escaped its root",
    }
    assert failed["retry_authorized"] is False
    assert failed["heldout_open_count"] == 0
    assert failed["g2_attempted"] is False
    counts = predecessor["terminal_audit"]["zero_training_proof"]["counts"]
    assert counts and all(value == 0 for value in counts.values())

    initialization = predecessor["artifacts"]["initialization.json"]
    schedule = predecessor["artifacts"]["schedule.json"]
    for key, expected in contract.PREDECESSOR_INITIALIZATION_IDENTITY.items():
        assert initialization[key] == expected
    for key, expected in contract.PREDECESSOR_SCHEDULE_IDENTITY.items():
        assert schedule[key] == expected
    assert schedule["presentation_count"] == len(schedule["presentation_indices"]) == 128_000

    rgb_source = inspect.getsource(installed.RawInputs.__mro__[1].read_rgb)
    train_source = inspect.getsource(installed.Trainer.__mro__[1].train_arm)
    assert rgb_source.index("safe_relative_path") < rgb_source.index("_read_regular")
    assert train_source.index("self.batch(") < train_source.index(
        "model.forward_training_pair("
    )
    assert train_source.index("model.forward_training_pair(") < train_source.index(
        ".backward()"
    ) < train_source.index("optimizer.step()")


def test_predecessor_rejects_extra_missing_mutated_and_symlinked_files(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / name for name in ("extra", "missing", "mutated", "symlink")]
    predecessor_roots = [_copy_predecessor(root) for root in roots]

    (predecessor_roots[0] / "arm_promoted.pt").write_bytes(b"forbidden")
    with pytest.raises(PermissionError, match="inventory changed"):
        contract.validate_predecessor(runner.base._read_regular, root=roots[0])

    (predecessor_roots[1] / "schedule.json").unlink()
    with pytest.raises(PermissionError, match="inventory changed"):
        contract.validate_predecessor(runner.base._read_regular, root=roots[1])

    (predecessor_roots[2] / "failed.json").write_bytes(b"changed\n")
    with pytest.raises(PermissionError, match="input hash changed"):
        contract.validate_predecessor(runner.base._read_regular, root=roots[2])

    schedule = predecessor_roots[3] / "schedule.json"
    schedule.unlink()
    schedule.symlink_to(ROOT / contract.PREDECESSOR_ROOT_RELATIVE_PATH / "schedule.json")
    with pytest.raises(PermissionError, match="not four regular files"):
        contract.validate_predecessor(runner.base._read_regular, root=roots[3])


def test_path_normalizer_accepts_only_the_exact_render_layout() -> None:
    relative = (
        ".generated/go2_render_selected_v04/scenes/"
        "scene_703b25447899b393/rgb/frame_005313_env_33.png"
    )
    absolute = f"{ROOT.as_posix()}/{relative}"
    assert contract.normalize_endpoint_rgb_path(absolute) == relative
    invalid = (
        relative,
        f"/tmp/{relative}",
        f"{ROOT.as_posix()}-collision/{relative}",
        absolute.replace("/rgb/", "/rgb//"),
        absolute.replace("/rgb/", "/./rgb/"),
        absolute.replace("/rgb/", "/other/../rgb/"),
        absolute.replace("scene_703b25447899b393", "scene_703B25447899B393"),
        absolute.replace("env_33", "env_3"),
        f"{ROOT.as_posix()}/lewm/not-an-endpoint.png",
        absolute.replace("/", "\\"),
    )
    for value in invalid:
        with pytest.raises((PermissionError, ValueError)):
            contract.normalize_endpoint_rgb_path(value)


def test_allowlist_rewrites_views_without_mutating_bound_rows() -> None:
    relative = (
        ".generated/go2_render_selected_v04/scenes/"
        "scene_703b25447899b393/rgb/frame_005313_env_33.png"
    )
    absolute = f"{ROOT.as_posix()}/{relative}"
    endpoint_id = "1" * 64
    endpoints = {endpoint_id: _endpoint(endpoint_id, absolute)}
    allowlist, normalized = contract.build_rgb_allowlist(endpoints)
    assert endpoints[endpoint_id]["image_path_metadata_only"] == absolute
    assert normalized[endpoint_id]["image_path_metadata_only"] == relative
    assert allowlist[relative] == {
        "file_sha256": "2" * 64,
        "dataset_role": "train",
        "endpoint_identity_sha256": endpoint_id,
        "endpoint_content_sha256": "3" * 64,
        "source_absolute_path": absolute,
    }

    duplicate = {
        endpoint_id: endpoints[endpoint_id],
        "4" * 64: _endpoint("4" * 64, absolute),
    }
    with pytest.raises(PermissionError, match="repeats"):
        contract.build_rgb_allowlist(duplicate)
    for damaged in (
        {endpoint_id: _endpoint("5" * 64, absolute)},
        {endpoint_id: _endpoint(endpoint_id, absolute, image_sha="bad")},
        {endpoint_id: _endpoint(endpoint_id, absolute, role="heldout")},
    ):
        with pytest.raises(PermissionError, match="authority fields"):
            contract.build_rgb_allowlist(damaged)


def test_exact_9460_endpoint_index_builds_authority_without_opening_rgb() -> None:
    manifest_raw = runner.base._read_regular(
        ROOT / contract.RAW_MANIFEST_RELATIVE_PATH,
        expected_sha256=contract.RAW_MANIFEST_FILE_SHA256,
    )
    manifest = contract.validate_raw_manifest(
        contract.parse_canonical_json(manifest_raw, name="Raw manifest")
    )
    endpoint_binding = manifest["endpoint_index"]
    endpoint_raw = runner.base._read_regular(
        ROOT / contract.RAW_ROOT_RELATIVE_PATH / endpoint_binding["path"],
        expected_sha256=endpoint_binding["file_sha256"],
    )
    rows = runner.base._parse_jsonl(endpoint_raw, name="Raw endpoints")
    endpoints = {row["endpoint_identity_sha256"]: row for row in rows}
    allowlist, normalized = contract.build_rgb_allowlist(endpoints)
    assert len(rows) == len(endpoints) == len(allowlist) == len(normalized) == 9_460
    assert all(not Path(path).is_absolute() for path in allowlist)
    assert all(Path(row["image_path_metadata_only"]).is_absolute() for row in rows)
    assert all(
        not Path(row["image_path_metadata_only"]).is_absolute()
        for row in normalized.values()
    )
    predecessor = contract.validate_predecessor(installed._read_regular)
    inputs = installed.RawInputs(object(), predecessor["v1_authorization"])
    assert len(inputs.bound_endpoints) == len(inputs.rgb_allowlist) == 9_460
    assert all(
        not path.startswith(".generated/go2_render_selected_v04/scenes/")
        for path in inputs.consumed
    )


def test_rgb_read_requires_bound_hash_role_endpoint_and_regular_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative = (
        ".generated/go2_render_selected_v04/scenes/"
        "scene_703b25447899b393/rgb/frame_005313_env_33.png"
    )
    endpoint_id = "1" * 64
    authority = {
        "file_sha256": "2" * 64,
        "dataset_role": "train",
        "endpoint_identity_sha256": endpoint_id,
    }
    inputs = object.__new__(installed.RawInputs)
    inputs.rgb_allowlist = {relative: authority}
    inputs._active_rgb_endpoint = endpoint_id
    calls: list[tuple[object, ...]] = []

    def accepted(self: object, path: str, expected: str, **kwargs: object) -> bytes:
        calls.append((self, path, expected, kwargs))
        return b"rgb"

    monkeypatch.setattr(installed.RawInputs.__mro__[1], "read_rgb", accepted)
    assert inputs.read_rgb(
        relative, "2" * 64, role="train", arm="promoted_jepa", stage="gradient"
    ) == b"rgb"
    assert len(calls) == 1

    for path, digest, role, active in (
        ("/absolute/internal.png", "2" * 64, "train", endpoint_id),
        (relative.replace("env_33", "env_34"), "2" * 64, "train", endpoint_id),
        (relative, "9" * 64, "train", endpoint_id),
        (relative, "2" * 64, "checkpoint_selection", endpoint_id),
        (relative, "2" * 64, "train", "8" * 64),
    ):
        inputs._active_rgb_endpoint = active
        with pytest.raises((PermissionError, ValueError)):
            inputs.read_rgb(path, digest, role=role, arm="a", stage="s")
    assert len(calls) == 1

    leaf = tmp_path / relative
    leaf.parent.mkdir(parents=True)
    target = tmp_path / "actual.png"
    target.write_bytes(b"rgb")
    leaf.symlink_to(target)
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    inputs._active_rgb_endpoint = endpoint_id
    with pytest.raises(PermissionError, match="symlink component"):
        inputs.read_rgb(relative, "2" * 64, role="train", arm="a", stage="s")
    assert len(calls) == 1

    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / ".generated").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(contract, "ROOT", repository)
    with pytest.raises(PermissionError, match="symlink component"):
        inputs.read_rgb(relative, "2" * 64, role="train", arm="a", stage="s")
    assert len(calls) == 1


def test_v2_initialization_and_full_schedule_must_normalize_exactly(
    tmp_path: Path,
) -> None:
    predecessor = contract.validate_predecessor(installed._read_regular)
    output = tmp_path / "attempt"
    output.mkdir()
    schemas = {
        "initialization.json": contract.INITIALIZATION_SCHEMA,
        "schedule.json": contract.SCHEDULE_SCHEMA,
    }
    for filename, schema in schemas.items():
        original = predecessor["artifacts"][filename]
        core = {key: value for key, value in original.items() if key != "content_sha256"}
        core["schema"] = schema
        value = contract.with_content_sha256(core)
        (output / filename).write_bytes(contract.canonical_json_bytes(value) + b"\n")
    initialization, schedule = contract.require_v1_initialization_and_schedule(
        output, predecessor, installed._read_regular
    )
    assert initialization["complete_state_sha256"] == (
        contract.PREDECESSOR_INITIALIZATION_IDENTITY["complete_state_sha256"]
    )
    assert schedule["indices_sha256"] == contract.PREDECESSOR_SCHEDULE_IDENTITY[
        "indices_sha256"
    ]

    core = {key: value for key, value in schedule.items() if key != "content_sha256"}
    core["presentation_indices"] = list(core["presentation_indices"])
    core["presentation_indices"][0] = 1
    changed = contract.with_content_sha256(core)
    (output / "schedule.json").write_bytes(
        contract.canonical_json_bytes(changed) + b"\n"
    )
    with pytest.raises(PermissionError, match="exact V1 identity"):
        contract.require_v1_initialization_and_schedule(
            output, predecessor, installed._read_regular
        )


def test_v2_review_and_authorization_are_new_strict_and_audit_bound() -> None:
    sources = {
        path: hashlib.sha256(f"review:{path}".encode()).hexdigest()
        for path in contract.SOURCE_PATHS
    }
    assert contract.TERMINAL_AUDIT_RELATIVE_PATH in sources
    review = _v2_review(sources)
    assert runtime_contract.validate_review(review, expected_sources=sources) == review
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = _v2_authorization(review_binding)
    assert runtime_contract.validate_authorization(
        authorization, review_binding=review_binding
    ) == authorization

    changed = copy.deepcopy(authorization)
    changed["experiment"]["retry_authorized"] = True
    core = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = contract.canonical_json_sha256(core)
    with pytest.raises(PermissionError, match="authorization"):
        runtime_contract.validate_authorization(changed, review_binding=review_binding)
