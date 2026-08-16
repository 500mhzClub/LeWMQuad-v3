"""Read-only terminal-consumer amendment for gradient localisation V1.

The completed diagnostic stored dynamic execution counters while its frozen
plan stored aggregate pass counters.  This consumer translates those two
representations and then invokes the complete frozen terminal validator with
an equality adapter accepting exactly those two equivalent mappings.  It does
not edit an artifact, execute a model, or authorise repair or training.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterator, Mapping

from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_contract as CONTRACT,
)
from scripts import diagnose_go2_attentive_readout_gradient_localisation_v1 as RUNNER


ROOT = Path(__file__).resolve().parents[2]
STATUS = "VALIDATED_GRADIENT_LOCALISATION_TERMINAL_CONSUMER_AMENDMENT"
SOURCE_BASE_COMMIT = "ccdb4de735a71760cd2683e491ce221240bcf6e4"
SOURCE_CLOSURE_SCHEMA = (
    "go2_attentive_readout_gradient_localisation_v1_terminal_consumer_source_closure_v1")
SOURCE_CLOSURE_SELF_KEY = "terminal_consumer_source_closure_digest"
RECEIPT_SCHEMA = (
    "go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment_v1")
RECEIPT_SELF_KEY = "terminal_consumer_amendment_digest"

NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment.py",
    "lewm/tests/test_go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment.py",
)

FROZEN_SOURCE_FILES = {
    "lewm/oracle/go2_attentive_readout_gradient_localisation_v1_contract.py": (
        "c6e3b08017faa09edbeb77e82ddbc7a1c972dda968913cd7962278ced0faa913",
        32_896),
    "lewm/tests/test_go2_attentive_readout_gradient_localisation_v1_contract.py": (
        "0fa93373ea10a661ef5161ac122a68b85f43ce1e7ddce143cb63e4460add544d",
        8_330),
    "scripts/diagnose_go2_attentive_readout_gradient_localisation_v1.py": (
        "17ad299d2694403ef7dbf92fd7ddf015645a6e68d289030cb2e357d8836b9b25",
        122_185),
    "lewm/tests/test_diagnose_go2_attentive_readout_gradient_localisation_v1.py": (
        "bb9daae4b9fdcd0dfe138ab7d78d774d40b654b66fbb8bed101da9faf9ceb5f1",
        8_568),
}
FROZEN_SOURCE_CLOSURE_DIGEST = (
    "3b3fd688daef66a6dda98136bc2d0b79ef132714752cb860af41e43002ea9a0b")

RUNTIME_RELATIVE = CONTRACT.DIAGNOSTIC_RUNTIME_ROOT
FROZEN_ARTIFACTS = {
    "attempt.json": {
        "sha256": "8df048a440f1cde2b362f456131c766ddc190e30967fe222556694361eed7b33",
        "byte_count": 1_611, "self_key": RUNNER.ATTEMPT_SELF_KEY,
        "self_digest": "3ab590d43085113a6709a01e4745462673f8a80ae0f9436b245beb63479fcee8",
    },
    "backend_matrix.json": {
        "sha256": "5e96b82a68b332272aa7bbc632d51ffda8365576f0401a49a9b2e626860e3a9e",
        "byte_count": 282_344, "self_key": RUNNER.GROUP_SELF_KEY,
        "self_digest": "fe195df4b644c55851923b0941c55789fb3bfcd075e21b58437b59f02ac67307",
    },
    "contract.json": {
        "sha256": "db8cab54afa2d645871a82ed22007010feb1d4be5d2909f9091cf69bc439dad3",
        "byte_count": 23_529, "self_key": CONTRACT.CONTRACT_SELF_KEY,
        "self_digest": "bc10101d8cd989b61fcdbcc235db0470bf978fe44e9f3cdd4408ae18fc7c8b71",
    },
    "exact_reproduction.json": {
        "sha256": "1276d09816b6112420a2eecd855d2b2929f83384b9f814490b6b871e6f3817a7",
        "byte_count": 58_104, "self_key": RUNNER.PASS_SELF_KEY,
        "self_digest": "ef926ed8b4d8d346b4a7ca69cdb0d4979545a0b6ed72037052e29c70e1c036e4",
    },
    "fit_only_fixture.json": {
        "sha256": "aa252abe918b400466d1a91a7af543e378a71e65e79b2f48c3512279457b3daa",
        "byte_count": 4_424, "self_key": RUNNER.FIXTURE_SELF_KEY,
        "self_digest": "f39933c02110e2b4246801c04313cdbc66494681c8bc643c104b3b4d2f818712",
    },
    "hook_inventory.json": {
        "sha256": "bd5664a5bca66128f9777e209c69c1af567443141a68138738ff00332fd3df94",
        "byte_count": 349_674, "self_key": RUNNER.PASS_SELF_KEY,
        "self_digest": "85a8abb021060ca3e29469fd84acac374634ce536ec588feeb8a64c99667f215",
    },
    "loss_isolation.json": {
        "sha256": "00a5b917835221a10c8079f6eba7f186232eedda8577feee0d3b444ebef24165",
        "byte_count": 257_624, "self_key": RUNNER.GROUP_SELF_KEY,
        "self_digest": "fbba85358b9337cac02079b83c0d37696df5b1dfbb8f0943b68f6a4be80072cb",
    },
    "terminal.json": {
        "sha256": "0afe02dd08baea3fcc1657fbb672153d0c8b07a778856565428cdaac262aa196",
        "byte_count": 46_801, "self_key": RUNNER.TERMINAL_SELF_KEY,
        "self_digest": "7ec0c9d5cd01c965568f38ca7c5e119e0f7fb74b65dc0f909bdba09f98b26187",
    },
}

EXPECTED_CLASSIFICATION = "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED"
FROZEN_CONSUMER_EXCEPTION_TYPE = "GradientLocalisationError"
FROZEN_CONSUMER_EXCEPTION_MESSAGE = "completed matrix SDPA ledger changed"
HEX64 = re.compile(r"[0-9a-f]{64}")


class TerminalConsumerAmendmentError(RuntimeError):
    """The frozen source, artifacts, translation, or classification changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalConsumerAmendmentError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or non-regular")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TerminalConsumerAmendmentError(f"{label} is invalid") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), f"{label} self digest changed")
    result[key] = recorded
    return result


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"{key} already exists")
    result[key] = digest(result)
    return result


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TerminalConsumerAmendmentError(
            f"cannot bind terminal-consumer source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    require(_git(root, "status", "--porcelain=v1") == "",
            "terminal-consumer source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_BASE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(ancestor.returncode == 0,
            "terminal-consumer source does not descend from its frozen base")
    changed = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{SOURCE_BASE_COMMIT}..{head}"
    ).splitlines())))
    require(changed == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed terminal-consumer diff is not exactly two additive paths")
    frozen = {}
    for relative, (expected_sha, expected_bytes) in FROZEN_SOURCE_FILES.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected_bytes
                and file_sha256(path) == expected_sha,
                f"frozen gradient-localisation source changed at {relative}")
        frozen[relative] = {"path": relative, "sha256": expected_sha,
                            "byte_count": expected_bytes}
    additive = {}
    for relative in NEW_SOURCE_PATHS:
        path = root / relative
        require(path.is_file() and not path.is_symlink(),
                f"terminal-consumer source is absent: {relative}")
        additive[relative] = {"path": relative, "sha256": file_sha256(path),
                              "byte_count": path.stat().st_size}
    payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "base_source_commit": SOURCE_BASE_COMMIT,
        "exact_committed_additive_path_diff": list(changed),
        "frozen_gradient_localisation_files": frozen,
        "additive_terminal_consumer_files": additive,
    }
    return {**payload, SOURCE_CLOSURE_SELF_KEY: digest(payload)}


def translate_planned_execution_counts(
        planned: Mapping[str, Any]) -> dict[str, int]:
    required = {
        "exact_reproduction", "hook_inventory", "loss_isolation",
        "backend_matrix", "fresh_model_constructions", "forwards",
        "backwards", "optimizer_constructions", "optimizer_steps",
        "gradient_clips", "fixture_validation_row_record_opens",
        "fixture_validation_latent_shard_opens", "unique_fit_row_record_files",
        "unique_fit_latent_shard_files", "pass_latent_shard_loads",
        "batch_presentations", "examples_presented",
    }
    require(set(planned) == required
            and all(isinstance(value, int) and value >= 0
                    for value in planned.values()),
            "planned execution-count schema changed")
    pass_total = (planned["exact_reproduction"] + planned["hook_inventory"]
                  + planned["loss_isolation"] + planned["backend_matrix"])
    require(pass_total == planned["fresh_model_constructions"]
            == planned["forwards"] == planned["backwards"]
            == planned["optimizer_constructions"],
            "aggregate pass budget is internally inconsistent")
    return {
        "backward_attempts": planned["backwards"],
        "batch_presentations": planned["batch_presentations"],
        "completed_backwards": planned["backwards"],
        "completed_forwards": planned["forwards"],
        "examples_presented": planned["examples_presented"],
        "fixture_validation_latent_shard_opens":
            planned["fixture_validation_latent_shard_opens"],
        "fixture_validation_row_record_opens":
            planned["fixture_validation_row_record_opens"],
        "forward_attempts": planned["forwards"],
        "fresh_model_constructions": planned["fresh_model_constructions"],
        "gradient_clips": planned["gradient_clips"],
        "optimizer_constructions": planned["optimizer_constructions"],
        "optimizer_steps": planned["optimizer_steps"],
        "pass_latent_shard_loads": planned["pass_latent_shard_loads"],
        "unique_fit_latent_shard_files": planned["unique_fit_latent_shard_files"],
        "unique_fit_row_record_files": planned["unique_fit_row_record_files"],
    }


class EquivalentExecutionCounts(dict[str, int]):
    """Equality adapter for exactly one aggregate/dynamic counter pair."""

    def __init__(self, planned: Mapping[str, int]) -> None:
        super().__init__(planned)
        self.planned = dict(planned)
        self.dynamic = translate_planned_execution_counts(planned)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and (
            dict(other) == self.planned or dict(other) == self.dynamic)


def runtime_root(root: Path = ROOT) -> Path:
    return root / RUNTIME_RELATIVE


def validate_frozen_runtime_bytes(root: Path = ROOT) -> dict[str, Any]:
    runtime = runtime_root(root)
    require(runtime.is_dir() and not runtime.is_symlink(),
            "gradient-localisation runtime root changed")
    children = list(runtime.iterdir())
    require({path.name for path in children} == set(FROZEN_ARTIFACTS)
            and all(path.is_file() and not path.is_symlink()
                    and path.stat().st_mode & 0o222 == 0 for path in children),
            "gradient-localisation runtime inventory or immutability changed")
    receipts = {}
    for name, expected in FROZEN_ARTIFACTS.items():
        path = runtime / name
        require(path.stat().st_size == expected["byte_count"]
                and file_sha256(path) == expected["sha256"],
                f"frozen runtime bytes changed at {name}")
        value = validate_signed(read_json(path, name), expected["self_key"], name)
        require(value[expected["self_key"]] == expected["self_digest"],
                f"frozen runtime self digest changed at {name}")
        receipts[name] = value
    contract = CONTRACT.validate_contract(receipts["contract.json"])
    require(contract[CONTRACT.CONTRACT_SELF_KEY]
            == FROZEN_ARTIFACTS["contract.json"]["self_digest"]
            and contract["source_closure"]["source_repository_commit"]
            == SOURCE_BASE_COMMIT
            and contract["source_closure"][CONTRACT.SOURCE_CLOSURE_SELF_KEY]
            == FROZEN_SOURCE_CLOSURE_DIGEST,
            "installed gradient-localisation contract changed")
    return {"contract": contract, "artifacts": receipts}


@contextmanager
def _translated_frozen_validator(contract: Mapping[str, Any],
                                 planned: Mapping[str, int],
                                 ) -> Iterator[None]:
    original_loader = RUNNER.load_installed_contract
    original_counts = CONTRACT.EXECUTION_COUNTS
    adapter = EquivalentExecutionCounts(planned)
    RUNNER.load_installed_contract = lambda _root=ROOT: dict(contract)
    CONTRACT.EXECUTION_COUNTS = adapter
    try:
        yield
    finally:
        CONTRACT.EXECUTION_COUNTS = original_counts
        RUNNER.load_installed_contract = original_loader


def validate_frozen_consumer_defect(root: Path,
                                    contract: Mapping[str, Any]) -> None:
    """Prove the frozen validator reaches exactly its count-schema defect."""
    original_loader = RUNNER.load_installed_contract
    RUNNER.load_installed_contract = lambda _root=ROOT: dict(contract)
    try:
        try:
            RUNNER.validate_terminal(root)
        except RUNNER.GradientLocalisationError as exc:
            require(type(exc).__name__ == FROZEN_CONSUMER_EXCEPTION_TYPE
                    and str(exc) == FROZEN_CONSUMER_EXCEPTION_MESSAGE,
                    "frozen terminal-consumer exception changed")
        else:
            raise TerminalConsumerAmendmentError(
                "frozen validator no longer reproduces its consumer defect")
    finally:
        RUNNER.load_installed_contract = original_loader


def validate_completed_terminal(root: Path = ROOT) -> dict[str, Any]:
    frozen = validate_frozen_runtime_bytes(root)
    contract = frozen["contract"]
    artifacts = frozen["artifacts"]
    planned = contract["execution_counts"]
    dynamic = translate_planned_execution_counts(planned)
    terminal = artifacts["terminal.json"]
    attempt = artifacts["attempt.json"]
    require(attempt["planned_execution_counts"] == planned
            and terminal["execution_counts"] == dynamic,
            "planned-to-dynamic execution-count translation changed")
    validate_frozen_consumer_defect(root, contract)
    with _translated_frozen_validator(contract, planned):
        fully_validated = RUNNER.validate_terminal(root)
    require(fully_validated == terminal
            and terminal["terminal_kind"]
            == "COMPLETED_MECHANISM_CLASSIFICATION"
            and terminal["mechanism_classification"] == EXPECTED_CLASSIFICATION
            and terminal["completed_passes"] == list(RUNNER.PASS_ORDER)
            and terminal["later_repair_gate"] == {
                "automatic_repair_or_training": False,
                "classification_can_support_separate_repair_decision": False,
                "repair_authorised_now": False,
                "training_authorised_now": False,
            }, "completed terminal classification changed")
    return terminal


def build_consumer_receipt(source: Mapping[str, Any],
                           terminal: Mapping[str, Any]) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_CLOSURE_SCHEMA
            and source.get(SOURCE_CLOSURE_SELF_KEY)
            == digest({key: value for key, value in source.items()
                       if key != SOURCE_CLOSURE_SELF_KEY}),
            "terminal-consumer source closure is invalid")
    require(terminal.get(RUNNER.TERMINAL_SELF_KEY)
            == FROZEN_ARTIFACTS["terminal.json"]["self_digest"]
            and terminal.get("mechanism_classification")
            == EXPECTED_CLASSIFICATION,
            "validated terminal binding changed")
    artifact_set = [{"name": name, **{
        key: value for key, value in expected.items()
        if key in ("sha256", "byte_count", "self_digest")}}
        for name, expected in sorted(FROZEN_ARTIFACTS.items())]
    return signed({
        "schema": RECEIPT_SCHEMA, "status": STATUS,
        "source_closure": dict(source),
        "frozen_runtime_artifact_set": artifact_set,
        "frozen_runtime_artifact_set_digest": digest(artifact_set),
        "installed_contract_digest":
            FROZEN_ARTIFACTS["contract.json"]["self_digest"],
        "terminal_digest": terminal[RUNNER.TERMINAL_SELF_KEY],
        "mechanism_classification": EXPECTED_CLASSIFICATION,
        "consumer_predicate_amendment": {
            "frozen_validator_exception_type":
                FROZEN_CONSUMER_EXCEPTION_TYPE,
            "frozen_validator_exception_message":
                FROZEN_CONSUMER_EXCEPTION_MESSAGE,
            "old_invalid_comparison":
                "dynamic terminal execution_counts == aggregate planned execution_counts",
            "new_exact_comparison":
                "dynamic terminal execution_counts == deterministic translation(planned)",
            "scientific_calculation_changed": False,
            "runtime_artifact_changed": False,
        },
        "authority": {
            "model_execution": False, "artifact_write": False,
            "rerun": False, "repair": False, "training": False,
            "checkpoint": False, "predictor_access": False,
        },
    }, RECEIPT_SELF_KEY)


def validate_terminal_consumer(root: Path = ROOT) -> dict[str, Any]:
    source = source_closure(root)
    terminal = validate_completed_terminal(root)
    return build_consumer_receipt(source, terminal)


__all__ = [name for name in globals() if name.isupper()] + [
    "EquivalentExecutionCounts", "TerminalConsumerAmendmentError",
    "build_consumer_receipt", "canonical_bytes", "digest", "file_sha256",
    "read_json", "runtime_root", "signed", "source_closure",
    "translate_planned_execution_counts", "validate_completed_terminal",
    "validate_frozen_consumer_defect",
    "validate_frozen_runtime_bytes", "validate_signed",
    "validate_terminal_consumer",
]
