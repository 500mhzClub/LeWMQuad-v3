"""Outcome-free custody for the interrupted small-family allocation search.

The clean a1b8952 scorer-fit state resolver had completed seven family shards
and the general/safety prefix of the eighth family when its deterministic
allocation search was stopped after 24 hours.  No small-family terminal shard,
failure receipt, branch, frame, latent, scorer run, or predictor access existed.

This module is deliberately not a scientific gate.  It performs three narrow
custody operations:

* archive the exact interrupted authorities, seven fixed shards, their exact
  request/capture transports, and the exact 12-scene small-family prefix;
* issue one immutable, non-overwriting interruption receipt; and
* reissue the seven fixed shards under successor source authority without
  rewriting the archived request/capture bytes.

The reissued shard is a distinct wrapper.  Its embedded successor payload may
change only registered source/contract/launch fields (plus its derived self
digest).  A caller-supplied semantic validator must replay the archived
transport in memory before a wrapper is issued or consumed.  Thus a top-level
lineage rewrite can never stand in for request/capture validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lewm.oracle import go2_scorer_projection_fix_interruption_v1 as PROJECTION
from lewm.oracle import (
    go2_scorer_fixed_reissue_validation_interruption_v1 as TRANSITION,
)
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as SELECTOR


ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "go2_scorer_small_search_performance_preoutcome_interruption_v1"
STATUS = "SUPERSEDED_PRE_OUTCOME_PERFORMANCE_INTERRUPTION"
V1_SCHEMA = SCHEMA
V1_STATUS = STATUS
V1_SELF_KEY = (
    "preoutcome_small_search_performance_interruption_receipt_digest"
)
V2_SCHEMA = (
    "go2_scorer_small_search_performance_preoutcome_interruption_v2"
)
V2_STATUS = (
    "SOURCE_TRANSITION_BOUND_PRE_OUTCOME_PERFORMANCE_INTERRUPTION"
)
V2_SELF_KEY = (
    "preoutcome_small_search_performance_interruption_receipt_v2_digest"
)
REISSUED_SHARD_SCHEMA = "go2_branch_corpus_v1_2_source_reissued_state_shard_v1"
REISSUED_SHARD_STATUS = "COMPLETE_SOURCE_REISSUED_OUTCOME_FREE_FIXED_STATE_SHARD"
SMALL_PREFIX_REISSUE_SCHEMA = (
    "go2_scorer_small_fixed_prefix_source_reissue_v1"
)
SMALL_PREFIX_REISSUE_STATUS = (
    "COMPLETE_SOURCE_REISSUED_OUTCOME_FREE_SMALL_FIXED_PREFIX"
)

CORPUS_ROOT_RELATIVE = Path(".generated/go2_branch_corpus_v1_2")
SCORER_ROOT_RELATIVE = Path(".generated/go2_utility_scorer_v1_2")
SCORER_FIT_RELATIVE = CORPUS_ROOT_RELATIVE / "scorer_fit"
ARCHIVE_ROOT_RELATIVE = (
    SCORER_FIT_RELATIVE / "superseded_preoutcome_small_search_performance_v1"
)
RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preoutcome_small_search_performance_interruption_receipt_v1.json"
)
V1_RECEIPT_RELATIVE_PATH = RECEIPT_RELATIVE_PATH
V2_RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preoutcome_small_search_performance_interruption_receipt_v2.json"
)
SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE / "small_fixed_prefix_source_reissue_receipt_v1.json"
)

INTERRUPTED_SOURCE_REPOSITORY_COMMIT = (
    "a1b89521bb825a0673d4663d2a9bff3f8f976a7d"
)
INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST = (
    "398ff354a96378fadbaa81a84ce7a949defcb373ab798a42ff9060d305970917"
)
INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST = (
    "dab982af7e514b02c95c6a4a6d1cbe85ec98394edac28f0bffad0de899705207"
)
INTERRUPTED_SELECTION_DIGEST = (
    "c20b4feceb865b25fb24e5534be5f84d14a5795d069ca2b0c14cd3f23d8ca9dd"
)
INTERRUPTED_SCORER_CONTRACT_DIGEST = (
    "4b9e8e2870cc5d82e6596a3833f197e2f9ff5cf7a0ad6b4d069afc5bae634e7c"
)
INTERRUPTED_SCORER_CONTRACT_ARTIFACT_DIGEST = (
    "34ab0f412dd1df05a9c7d8bc9dc9c7c7750541026e45501414b2c2d92bc2e792"
)
INTERRUPTED_CLEAN_SOURCE_LAUNCH_RECEIPT_DIGEST = (
    "24dea9398e962ccacd450974dccf6e85cd9228e7f20b50549c8a2e34ef6ade24"
)
INTERRUPTED_MIXED_DISPOSITION_DIGEST = (
    "40f56622f07f9487a8801dffb22b96e730bbb554e8fc3568c29c00faf802d099"
)
INTERRUPTED_PROJECTION_RECEIPT_DIGEST = (
    "8b0180547dd768cb351a2ced56687ebe5aee842491868e62d17d0a2a66b1b715"
)

INTERRUPTED_COMMAND = [
    ".generated/venvs/genesis_render_vulkan/bin/python",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "--pool", "scorer_fit", "--stage", "states",
    "--family", "small_enclosed_maze", "--backend", "cpu",
]
INTERRUPTED_PID = 1_204_602
CUTOFF_ELAPSED = "1-00:00:57"
CUTOFF_CPU = "1-07:46:21"
INTERRUPTED_CALL_CHAIN = [
    "ALLOC.validate_allocation_manifest",
    "_lexicographic_rotations",
    "scipy.optimize.milp",
]

FIXED_TRANSPORT_ROW_COUNT = 290
FIXED_TRANSPORT_BYTE_COUNT = 4_852_501
FIXED_TRANSPORT_ROW_SET_DIGEST = (
    "9af407320ab46c329626e5d1effde671f9f39cbadaa810ae161b56f61707f26a"
)
SMALL_PREFIX_REQUEST_COUNT = 12
SMALL_PREFIX_CAPTURE_COUNT = 12
SMALL_PREFIX_ROW_COUNT = 24
SMALL_PREFIX_BYTE_COUNT = 253_684
# Canonical rows are defined by _transport_row and sorted request-first by
# filename.  This is in addition to the historical inventory witness below.
SMALL_PREFIX_ROW_SET_DIGEST = (
    "24c506e18f51290825e445492b4c026c6dfed6a85cc1f2848735408c71d78801"
)
SMALL_PREFIX_HISTORICAL_INVENTORY_DIGEST = (
    "bd947bedab7ea7119a9eace2c9ee6ed404488dbb6f7f98136de9500c091887e5"
)
SMALL_PREFIX_HISTORICAL_COMPACT_DIGEST = (
    "649b82918e777f403ba9c1bbca6d798249fe0286e229dc53d778b1e0baadb51c"
)
SMALL_PREFIX_STATE_PROJECTION_DIGEST = (
    "167833ce1c6bf51da449917637fae1a76db8218e7f5116840b5f73f19d3ebb4d"
)
SMALL_PREFIX_CURSOR_SCENE_ID = "small_enclosed_maze_100b36b62f36"

ZERO_OUTCOME_FIELDS = {
    "candidate_outcomes_loaded": False,
    "branch_identities_created": False,
    "branches_attempted": 0,
    "frames_rendered": 0,
    "target_latents_encoded": 0,
    "scorer_training_started": False,
    "scorer_qualification_started": False,
    "predictor_checkpoints_opened": 0,
}


def _archive_path(active_path: str | Path, *, label: str) -> str:
    active = Path(active_path)
    try:
        suffix = active.relative_to(SCORER_FIT_RELATIVE)
    except ValueError:
        try:
            suffix = active.relative_to(SCORER_ROOT_RELATIVE)
        except ValueError as exc:  # pragma: no cover - constants are tested
            raise ValueError("artifact is outside a registered generated root") from exc
        return str(SCORER_ROOT_RELATIVE / "superseded_preoutcome_small_search_performance_v1" / label / suffix)
    return str(ARCHIVE_ROOT_RELATIVE / label / suffix)


INTERRUPTED_AUTHORITIES: dict[str, dict[str, Any]] = {
    "mixed_precontract_disposition": {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": str(SCORER_FIT_RELATIVE / "preserved_state_mixed_precontract_disposition_reachability_v2.json"),
        "archive_path": str(ARCHIVE_ROOT_RELATIVE / "authorities" / "preserved_state_mixed_precontract_disposition_reachability_v2.40f56622.json"),
        "self_digest_key": "mixed_precontract_disposition_receipt_digest",
        "self_digest": INTERRUPTED_MIXED_DISPOSITION_DIGEST,
        "raw_sha256": "aea7b086f6cfceebc111d50d732e41a2d017a110fb67eff919d5f8d0530d14d9",
        "byte_count": 29_403,
    },
    "scorer_contract": {
        "managed_root": str(SCORER_ROOT_RELATIVE),
        "active_path": str(SCORER_ROOT_RELATIVE / "scorer_contract_v1_2.json"),
        "archive_path": str(SCORER_ROOT_RELATIVE / "superseded_preoutcome_small_search_performance_v1" / "authorities" / "scorer_contract_v1_2.34ab0f41.json"),
        "self_digest_key": "contract_artifact_digest",
        "self_digest": INTERRUPTED_SCORER_CONTRACT_ARTIFACT_DIGEST,
        "raw_sha256": "18f7d2ef88c229c6dcbd2bdd1449e2de48e04ed487c1082f06ece5ad57dee3e6",
        "byte_count": 73_298,
    },
    "clean_source_launch": {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": str(SCORER_FIT_RELATIVE / "clean_source_launch_receipt.json"),
        "archive_path": str(ARCHIVE_ROOT_RELATIVE / "authorities" / "clean_source_launch_receipt.24dea939.json"),
        "self_digest_key": "clean_source_launch_receipt_digest",
        "self_digest": INTERRUPTED_CLEAN_SOURCE_LAUNCH_RECEIPT_DIGEST,
        "raw_sha256": "b26eba29751add8cf2052130361e6159a746b98f44b7a0ab83f0fc503ba14578",
        "byte_count": 1_983,
    },
    "projection_fix_interruption": {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": str(PROJECTION.RECEIPT_RELATIVE_PATH),
        "archive_path": str(ARCHIVE_ROOT_RELATIVE / "authorities" / "preoutcome_projection_fix_interruption_receipt_v1.8b018054.json"),
        "self_digest_key": "preoutcome_projection_fix_interruption_receipt_digest",
        "self_digest": INTERRUPTED_PROJECTION_RECEIPT_DIGEST,
        "raw_sha256": "a9404a3833a92819aaceec5c9af633c5ae3f579f09501f8f75f0de725b7b14ec",
        "byte_count": 19_061,
    },
}


def _fixed_shard(
    family: str, kind: str, active_name: str, self_digest: str,
    raw_sha256: str, byte_count: int, transport_rows: int,
    transport_bytes: int, transport_digest: str,
) -> dict[str, Any]:
    active = SCORER_FIT_RELATIVE / active_name
    return {
        "family": family,
        "kind": kind,
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": str(active),
        "archive_path": str(ARCHIVE_ROOT_RELATIVE / "fixed_state_shards" / f"{active_name}.{self_digest[:8]}.json"),
        "self_digest_key": "state_shard_digest",
        "self_digest": self_digest,
        "raw_sha256": raw_sha256,
        "byte_count": byte_count,
        "transport_row_count": transport_rows,
        "transport_byte_count": transport_bytes,
        "transport_row_set_digest": transport_digest,
    }


FIXED_STATE_SHARDS: tuple[dict[str, Any], ...] = (
    _fixed_shard("large_enclosed_maze", "mixed", "active_mixed_state_shard_large_enclosed_maze_reachability_v2.json", "a9b61d6bf3743dcc45fcfa3bd0c1046a118dfb35ba33c06067bea681c66da48b", "f5d9860da3b245ebd485c0b360bc9ca53cef07f99b37b083fae377414879c7c1", 246_046, 66, 959_269, "08a447266953ca5aa46aef72a6b38d3a6f7fa30c3e870cbed474fef1cb01230b"),
    _fixed_shard("local_composite_motifs", "mixed", "active_mixed_state_shard_local_composite_motifs_reachability_v2.json", "4fbf528ad294ae9f9bc894dea1bd927bb675c55de25da6e358a8d9ed27f56d61", "66fb58949491923daad71dbb0e302e3698e8ed5d826b49f72e4c672eabe8e4ef", 137_512, 6, 128_597, "a377c4a450b1b699ff148bd4c1b7199118f5edde9e8b83496a6d922277ae1919"),
    _fixed_shard("loop_alias_stress", "mixed", "active_mixed_state_shard_loop_alias_stress_reachability_v2.json", "6ea7bb5bad29ce76e664ac44a85a3383d5ea6ae5d6b03952f4d9a53e465e7073", "147c149b7a933d3354d826f1c825aef33058b9e92aca80fb579b59b232f9316b", 132_441, 8, 157_413, "620abfac1b40ee5c5aabfd53fdb823975473389ec8c0a04d0b7cdebf36d7cda4"),
    _fixed_shard("medium_enclosed_maze", "ordinary", "state_shard_medium_enclosed_maze.json", "106f244270be79fd4267d9fc5c77ff4c81fb9e246ac0ed0d20239687c85d36d4", "9b76f80d9a4d40e4af2b951cb07689545a751c520d34ef85ccbeeb6b5da2fcca", 273_369, 86, 1_551_613, "ca3c0521be3619acecf7b3ff83e8cdae9233a5763b08f98439dac050c4946b16"),
    _fixed_shard("open_obstacle_field", "ordinary", "state_shard_open_obstacle_field.json", "0539bb27534f01ef21e442d36a25789ecca94bcd99cdccfed2febeb17e6368ea", "0495f3a1415c811aa0bdce8cb42ea014372e91ebb9414a889b96bf3b63ec9c06", 250_827, 52, 899_686, "6700729897f6383d9d3aa80b62b4fe80f98c94141a67751157a8590a43da55df"),
    _fixed_shard("rough_local_dynamics", "ordinary", "state_shard_rough_local_dynamics.json", "a6a911fcc39054d6757e43d1139ae87005e7074f6977b30d5abeee2a371bce57", "f5bc222326272f5fe76616f246c0d00ab6aff7d622aff28e589374d9f4e2eb58", 244_910, 38, 639_186, "ae367a4ec6f06e7014ea685af84b82b55740e72100d91e0f3527bbd9fc1cc5a0"),
    _fixed_shard("visual_sensor_stress", "ordinary", "state_shard_visual_sensor_stress.json", "6d7e1115f7e8af8e007d690d15d222f1defbe68cc34756a75c8eef55ee191c3b", "c3b2a5e3e5d41c56bd0f07634452772221d905610176c31baf427f22da3e1853", 241_067, 34, 516_737, "df55209f486ad7a21e9248f1e41487ed03d36ab57750a4eb56ec1068968fc3f7"),
)

SMALL_PREFIX_ROOTS = {
    "request": str(SCORER_FIT_RELATIVE / "state_resolution_scene_requests_v1" / "small_enclosed_maze"),
    "capture": str(SCORER_FIT_RELATIVE / "state_resolution_scene_captures_v1" / "small_enclosed_maze"),
}

SUCCESSOR_LINEAGE_KEYS = (
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "scorer_contract_v1_2_digest",
    "clean_source_launch_receipt_digest",
    "mixed_precontract_disposition_receipt_digest",
)

SMALL_PREFIX_REISSUE_SELF_KEY = "small_fixed_prefix_source_reissue_receipt_digest"
SMALL_PREFIX_REQUEST_SELF_KEY = "state_resolution_scene_request_digest"
SMALL_PREFIX_CAPTURE_SELF_KEY = "state_resolution_scene_capture_digest"


class PerformanceInterruptionError(RuntimeError):
    """The performance-interruption custody or successor replay failed."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _forbidden(path: Path) -> bool:
    return any(
        part == ".." or part == "sealed" or part == "sealed_test.json"
        or part.startswith("sealed_") for part in path.parts
    )


def _assert_no_symlink(path: Path) -> None:
    if _forbidden(path):
        raise PerformanceInterruptionError("lineage path crosses inaccessible custody")
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PerformanceInterruptionError("lineage descendant contains a symlink")


def _pin_managed(
    relative_path: str | Path, *, root: Path,
    managed_root_relative: str | Path = CORPUS_ROOT_RELATIVE,
) -> Path:
    repository = Path(root)
    if not repository.is_absolute():
        repository = Path.cwd() / repository
    managed = repository / Path(managed_root_relative)
    logical = repository / Path(relative_path)
    if _forbidden(managed) or _forbidden(logical):
        raise PerformanceInterruptionError("lineage path crosses inaccessible custody")
    try:
        suffix = logical.relative_to(managed)
    except ValueError as exc:
        raise PerformanceInterruptionError("lineage path escaped managed root") from exc
    if not suffix.parts:
        raise PerformanceInterruptionError("lineage path names only managed root")
    _assert_no_symlink(managed.parent)
    if managed.is_symlink():
        raw_target = managed.readlink()
        target = raw_target if raw_target.is_absolute() else managed.parent / raw_target
        if target.name != managed.name or _forbidden(target):
            raise PerformanceInterruptionError("managed alias identity changed")
        _assert_no_symlink(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise PerformanceInterruptionError("managed root is missing") from exc
    else:
        if not managed.is_dir():
            raise PerformanceInterruptionError("managed root is missing")
        canonical_root = managed.resolve(strict=True)
    if not canonical_root.is_dir() or canonical_root.name != managed.name:
        raise PerformanceInterruptionError("managed root identity changed")
    _assert_no_symlink(canonical_root)
    pinned = canonical_root.joinpath(*suffix.parts)
    _assert_no_symlink(pinned)
    return pinned


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PerformanceInterruptionError(f"{label} is missing")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceInterruptionError(f"{label} JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise PerformanceInterruptionError(f"{label} is not an object")
    return payload


def _verify_self(payload: Mapping[str, Any], key: str, label: str) -> None:
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _digest({
        name: value for name, value in payload.items() if name != key
    }):
        raise PerformanceInterruptionError(f"{label} self binding changed")


_RECEIPT_BINDING_KEYS = {
    "path", "receipt_digest", "raw_sha256", "byte_count", "status",
}


def _validate_receipt_binding(
    binding: Mapping[str, Any], *, label: str,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Validate the common exact-file receipt binding without opening it."""

    if not isinstance(binding, Mapping) or set(binding) != _RECEIPT_BINDING_KEYS:
        raise PerformanceInterruptionError(
            f"{label} receipt binding key surface changed")
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PerformanceInterruptionError(f"{label} receipt path is invalid")
    relative = Path(path_value)
    if relative.is_absolute() or _forbidden(relative):
        raise PerformanceInterruptionError(
            f"{label} receipt path escaped managed custody")
    try:
        suffix = relative.relative_to(CORPUS_ROOT_RELATIVE)
    except ValueError as exc:
        raise PerformanceInterruptionError(
            f"{label} receipt path escaped the corpus root") from exc
    if not suffix.parts:
        raise PerformanceInterruptionError(
            f"{label} receipt path names only the corpus root")
    for key in ("receipt_digest", "raw_sha256"):
        value = binding.get(key)
        if (
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PerformanceInterruptionError(
                f"{label} receipt {key} is invalid")
    byte_count = binding.get("byte_count")
    if (
        not isinstance(byte_count, int) or isinstance(byte_count, bool)
        or byte_count <= 0
    ):
        raise PerformanceInterruptionError(
            f"{label} receipt byte count is invalid")
    status = binding.get("status")
    if not isinstance(status, str) or not status:
        raise PerformanceInterruptionError(
            f"{label} receipt status is invalid")
    if expected_status is not None and status != expected_status:
        raise PerformanceInterruptionError(
            f"{label} receipt status changed")
    return copy.deepcopy(dict(binding))


def _load_exact_bound_receipt(
    receipt: Mapping[str, Any], binding: Mapping[str, Any], *,
    self_key: str, status: str, label: str, root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open one exact receipt path and bind its payload, bytes and self digest."""

    normalized = _validate_receipt_binding(
        binding, label=label, expected_status=status)
    path = _pin_managed(normalized["path"], root=root)
    payload = _load_json(path, label)
    raw = path.read_bytes()
    if (
        payload != dict(receipt)
        or len(raw) != normalized["byte_count"]
        or _raw_sha256(raw) != normalized["raw_sha256"]
    ):
        raise PerformanceInterruptionError(
            f"{label} exact byte binding changed")
    _verify_self(payload, self_key, label)
    if (
        payload.get(self_key) != normalized["receipt_digest"]
        or payload.get("status") != status
    ):
        raise PerformanceInterruptionError(
            f"{label} declared receipt binding changed")
    return payload, normalized


def _binding_paths(binding: Mapping[str, Any], *, root: Path) -> tuple[Path, Path]:
    managed = binding.get("managed_root", str(CORPUS_ROOT_RELATIVE))
    return (
        _pin_managed(binding["active_path"], root=root,
                     managed_root_relative=managed),
        _pin_managed(binding["archive_path"], root=root,
                     managed_root_relative=managed),
    )


def _exact_file(path: Path, binding: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    raw = path.read_bytes()
    return len(raw) == binding["byte_count"] and _raw_sha256(raw) == binding["raw_sha256"]


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _select_bound_location(
    *, active: Path, archive: Path, active_exact: bool, archive_exact: bool,
    require_archived: bool, label: str, allow_same_inode_crash: bool = False,
) -> Path:
    """Select an exact location without changing either directory entry.

    A hard-link-then-unlink archive can be interrupted with both names pointing
    at the same inode.  Only the mutating issuer is allowed to accept that
    recoverable intermediate state.  Every public validation path rejects it,
    so reopening evidence can never have the side effect of finishing a move.
    """

    if active_exact and archive_exact:
        if allow_same_inode_crash and _same_file(active, archive):
            return archive
        raise PerformanceInterruptionError(f"{label} active/archive collision")
    chosen = archive if archive_exact else active if active_exact else None
    if chosen is None or (require_archived and chosen != archive):
        suffix = "archive " if require_archived else ""
        raise PerformanceInterruptionError(
            f"exact {label} {suffix}is unavailable")
    return chosen


def _locate(
    binding: Mapping[str, Any], *, root: Path, require_archived: bool,
    label: str,
) -> tuple[dict[str, Any], Path]:
    active, archive = _binding_paths(binding, root=root)
    active_exact = _exact_file(active, binding)
    archive_exact = _exact_file(archive, binding)
    chosen = _select_bound_location(
        active=active, archive=archive, active_exact=active_exact,
        archive_exact=archive_exact, require_archived=require_archived,
        label=label)
    payload = _load_json(chosen, label)
    _verify_self(payload, str(binding["self_digest_key"]), label)
    if payload.get(binding["self_digest_key"]) != binding["self_digest"]:
        raise PerformanceInterruptionError(f"{label} declared digest changed")
    return payload, chosen


def _locate_for_archive(
    binding: Mapping[str, Any], *, root: Path, label: str,
) -> tuple[dict[str, Any], Path]:
    """Read an archive source while tolerating only the recoverable hardlink state."""

    active, archive = _binding_paths(binding, root=root)
    chosen = _select_bound_location(
        active=active, archive=archive,
        active_exact=_exact_file(active, binding),
        archive_exact=_exact_file(archive, binding),
        require_archived=False, label=label, allow_same_inode_crash=True)
    payload = _load_json(chosen, label)
    _verify_self(payload, str(binding["self_digest_key"]), label)
    if payload.get(binding["self_digest_key"]) != binding["self_digest"]:
        raise PerformanceInterruptionError(f"{label} declared digest changed")
    return payload, chosen


def _validate_zero(payload: Mapping[str, Any], label: str) -> None:
    for key, expected in ZERO_OUTCOME_FIELDS.items():
        if key in payload and payload.get(key) != expected:
            raise PerformanceInterruptionError(f"{label} is not outcome-free")


def _transport_row(
    *, family: str, kind: str, path: Path, logical_path: str,
    payload: Mapping[str, Any], mixed: bool,
) -> dict[str, Any]:
    self_key = (
        f"mixed_replacement_scene_{kind}_digest" if mixed else
        f"state_resolution_scene_{kind}_digest"
    )
    _verify_self(payload, self_key, f"{family} {kind}")
    _validate_zero(payload, f"{family} {kind}")
    raw = path.read_bytes()
    return {
        "family": family,
        "kind": kind,
        "path": logical_path,
        "archive_path": _archive_path(logical_path, label="transports"),
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "self_digest": payload[self_key],
    }


def _fixed_transport_rows(
    shard: Mapping[str, Any], binding: Mapping[str, Any], *, root: Path,
    archived: bool | None,
) -> list[dict[str, Any]]:
    family = str(binding["family"])
    mixed = binding["kind"] == "mixed"
    provenance_key = (
        "mixed_replacement_scene_capture_provenance" if mixed else
        "state_resolution_scene_capture_provenance"
    )
    provenance = shard.get(provenance_key)
    if not isinstance(provenance, list):
        raise PerformanceInterruptionError(f"{family} transport provenance changed")
    rows: list[dict[str, Any]] = []
    for pair in provenance:
        if not isinstance(pair, Mapping):
            raise PerformanceInterruptionError(f"{family} transport row changed")
        for kind in ("request", "capture"):
            logical = str(pair.get(f"{kind}_path", ""))
            active = _pin_managed(logical, root=root)
            archive_logical = _archive_path(logical, label="transports")
            archive = _pin_managed(archive_logical, root=root)
            expected_sha = pair.get(f"{kind}_raw_sha256")
            expected_bytes = pair.get(f"{kind}_byte_count")
            active_exact = (active.is_file() and not active.is_symlink()
                            and active.stat().st_size == expected_bytes
                            and _raw_sha256(active.read_bytes()) == expected_sha)
            archive_exact = (archive.is_file() and not archive.is_symlink()
                             and archive.stat().st_size == expected_bytes
                             and _raw_sha256(archive.read_bytes()) == expected_sha)
            path = _select_bound_location(
                active=active, archive=archive,
                active_exact=active_exact, archive_exact=archive_exact,
                require_archived=archived is True,
                label=f"{family} {kind} transport",
                allow_same_inode_crash=archived is None)
            if archived is False and path != active:
                raise PerformanceInterruptionError(
                    f"exact {family} {kind} active transport is unavailable")
            payload = _load_json(path, f"{family} {kind} transport")
            row = _transport_row(
                family=family, kind=kind, path=path, logical_path=logical,
                payload=payload, mixed=mixed)
            if (row["raw_sha256"] != pair.get(f"{kind}_raw_sha256")
                    or row["byte_count"] != pair.get(f"{kind}_byte_count")):
                raise PerformanceInterruptionError(f"{family} transport binding changed")
            rows.append(row)
    if (len(rows) != binding["transport_row_count"]
            or sum(row["byte_count"] for row in rows)
            != binding["transport_byte_count"]
            or _digest([{key: value for key, value in row.items()
                         if key != "archive_path"} for row in rows])
            != binding["transport_row_set_digest"]):
        raise PerformanceInterruptionError(f"{family} transport set changed")
    return rows


def _small_row(kind: str, path: Path, *, root: Path) -> dict[str, Any]:
    payload = _load_json(path, f"small prefix {kind}")
    self_key = f"state_resolution_scene_{kind}_digest"
    _verify_self(payload, self_key, f"small prefix {kind}")
    _validate_zero(payload, f"small prefix {kind}")
    if kind == "request":
        scene_id = payload.get("scene", {}).get("scene_id")
        ordinal = payload.get("scene_ordinal")
        selected = None
        failure = None
    else:
        scene_id = payload.get("scene_id")
        ordinal = payload.get("request", {}).get("scene_ordinal")
        selected = payload.get("chosen_state") is not None
        failure = payload.get("worker_failure")
    raw = path.read_bytes()
    logical = str(Path(SMALL_PREFIX_ROOTS[kind]) / path.name)
    return {
        "kind": kind,
        "name": path.name,
        "path": logical,
        "archive_path": _archive_path(logical, label="small_prefix"),
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "self_digest": payload[self_key],
        "scene_id": scene_id,
        "scene_ordinal": ordinal,
        "selected": selected,
        "worker_failure": failure,
    }


def _small_prefix_rows(*, root: Path, archived: bool | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("request", "capture"):
        logical_root = Path(SMALL_PREFIX_ROOTS[kind])
        active_directory = _pin_managed(logical_root, root=root)
        archive_directory = _pin_managed(
            _archive_path(logical_root, label="small_prefix"), root=root)
        inventories: dict[Path, dict[str, Path]] = {}
        for directory in (active_directory, archive_directory):
            if directory.exists() and (not directory.is_dir() or directory.is_symlink()):
                raise PerformanceInterruptionError("small prefix transport root changed")
            inventory: dict[str, Path] = {}
            if directory.is_dir():
                for path in directory.glob("*.json"):
                    if path.is_symlink():
                        raise PerformanceInterruptionError(
                            "small prefix transport contains a symlink")
                    inventory[path.name] = path
            inventories[directory] = inventory
        active_names = inventories[active_directory]
        archive_names = inventories[archive_directory]
        common = set(active_names) & set(archive_names)
        for name in common:
            if not (archived is None and _same_file(
                    active_names[name], archive_names[name])):
                raise PerformanceInterruptionError(
                    "small prefix active/archive collision")
        if archived is True:
            chosen = archive_names
        elif archived is False:
            if archive_names:
                raise PerformanceInterruptionError(
                    "small prefix archive exists before archival")
            chosen = active_names
        else:
            chosen = dict(active_names)
            chosen.update(archive_names)
        paths: list[Path] = []
        for name in sorted(chosen):
            paths.append(chosen[name])
        expected = SMALL_PREFIX_REQUEST_COUNT if kind == "request" else SMALL_PREFIX_CAPTURE_COUNT
        if len(paths) != expected:
            raise PerformanceInterruptionError("small prefix transport inventory changed")
        rows.extend(_small_row(kind, path, root=root) for path in paths)
    digest_rows = [{key: value for key, value in row.items()
                    if key not in {"path", "archive_path"}} for row in rows]
    if (len(rows) != SMALL_PREFIX_ROW_COUNT
            or sum(row["byte_count"] for row in rows) != SMALL_PREFIX_BYTE_COUNT
            or _digest(digest_rows) != SMALL_PREFIX_ROW_SET_DIGEST):
        raise PerformanceInterruptionError("small prefix row-set binding changed")
    requests = {row["name"]: row for row in rows if row["kind"] == "request"}
    captures = [row for row in rows if row["kind"] == "capture"]
    if set(requests) != {row["name"] for row in captures}:
        raise PerformanceInterruptionError("small prefix request/capture join changed")
    if any(row["worker_failure"] is not None for row in captures):
        raise PerformanceInterruptionError("small prefix contains worker failure")
    return rows


def _load_transport_row_payload(
    row: Mapping[str, Any], *, root: Path, require_archived: bool,
    allow_same_inode_crash: bool = False,
) -> dict[str, Any]:
    binding = _dynamic_binding(row)
    loader = _locate_for_archive if allow_same_inode_crash else _locate
    if allow_same_inode_crash:
        payload, _path = loader(
            binding, root=root, label=f"small prefix {row['kind']}")
    else:
        payload, _path = loader(
            binding, root=root, require_archived=require_archived,
            label=f"small prefix {row['kind']}")
    return payload


def _small_prefix_pairs(
    rows: Sequence[Mapping[str, Any]], *, root: Path,
    require_archived: bool, allow_same_inode_crash: bool = False,
) -> list[dict[str, Any]]:
    """Reconstruct and structurally replay the exact 12-row resolver prefix."""

    requests = {str(row["name"]): row for row in rows
                if row.get("kind") == "request"}
    captures = {str(row["name"]): row for row in rows
                if row.get("kind") == "capture"}
    if (len(requests) != SMALL_PREFIX_REQUEST_COUNT
            or len(captures) != SMALL_PREFIX_CAPTURE_COUNT
            or set(requests) != set(captures)):
        raise PerformanceInterruptionError(
            "small prefix request/capture join changed")
    pairs: list[dict[str, Any]] = []
    for name in requests:
        request_row = requests[name]
        capture_row = captures[name]
        request = _load_transport_row_payload(
            request_row, root=root, require_archived=require_archived,
            allow_same_inode_crash=allow_same_inode_crash)
        capture = _load_transport_row_payload(
            capture_row, root=root, require_archived=require_archived,
            allow_same_inode_crash=allow_same_inode_crash)
        request_digest = request.get(SMALL_PREFIX_REQUEST_SELF_KEY)
        ordinal = request.get("scene_ordinal")
        scene_id = request.get("scene", {}).get("scene_id")
        if (
            not isinstance(ordinal, int) or isinstance(ordinal, bool)
            or not isinstance(scene_id, str) or not scene_id
            or name != f"{request_digest}.json"
            or request_row.get("scene_ordinal") != ordinal
            or request_row.get("scene_id") != scene_id
            or capture_row.get("scene_ordinal") != ordinal
            or capture_row.get("scene_id") != scene_id
            or capture.get("request") != request
            or capture.get(SMALL_PREFIX_REQUEST_SELF_KEY) != request_digest
            or capture.get("scene_id") != scene_id
            or capture.get("worker_failure") is not None
        ):
            raise PerformanceInterruptionError(
                "small prefix request/capture pair changed")
        pairs.append({
            "name": name,
            "scene_id": scene_id,
            "scene_ordinal": ordinal,
            "request_row": copy.deepcopy(dict(request_row)),
            "capture_row": copy.deepcopy(dict(capture_row)),
            "request": request,
            "capture": capture,
        })
    pairs.sort(key=lambda pair: int(pair["scene_ordinal"]))
    if [pair["scene_ordinal"] for pair in pairs] != list(
            range(SMALL_PREFIX_REQUEST_COUNT)):
        raise PerformanceInterruptionError(
            "small prefix scene ordinals are not the exact lexical prefix")
    if len({pair["scene_id"] for pair in pairs}) != len(pairs):
        raise PerformanceInterruptionError("small prefix repeats a scene")

    required = {
        "general": 5,
        "safety_enriched": 5,
        "completion_enriched": 0,
    }
    found = {key: 0 for key in required}
    priority = list(required)
    trace: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        request = pair["request"]
        capture = pair["capture"]
        requested = [name for name in priority
                     if found[name] < required[name]]
        if (
            request.get("required_counts") != required
            or request.get("found_before_scene") != found
            or request.get("stratum_priority") != priority
            or request.get("requested_strata_in_priority_order") != requested
            or not requested
        ):
            raise PerformanceInterruptionError(
                "small prefix dynamic quota request changed")
        chosen = capture.get("chosen_state")
        chosen_stratum = None
        chosen_digest = None
        if chosen is not None:
            if not isinstance(chosen, Mapping):
                raise PerformanceInterruptionError(
                    "small prefix chosen state is malformed")
            chosen_stratum = chosen.get("stratum")
            chosen_digest = chosen.get("state_identity_digest")
            if (
                chosen_stratum not in requested
                or chosen.get("scene_id") != pair["scene_id"]
                or not isinstance(chosen_digest, str)
                or not chosen_digest
            ):
                raise PerformanceInterruptionError(
                    "small prefix chosen state changed")
            found[str(chosen_stratum)] += 1
        quota_full = found == required
        if quota_full != (index == len(pairs) - 1):
            raise PerformanceInterruptionError(
                "small prefix does not stop at the first full quota")
        trace.append({
            "scene_ordinal": pair["scene_ordinal"],
            "scene_id": pair["scene_id"],
            "found_before_scene": request["found_before_scene"],
            "requested_strata_in_priority_order": requested,
            "chosen_stratum": chosen_stratum,
            "chosen_state_identity_digest": chosen_digest,
        })
    if found != required:
        raise PerformanceInterruptionError(
            "small prefix does not contain exact 5G/5S")
    pairs[-1]["reducer_trace"] = trace
    return pairs


def _small_prefix_projection(
    rows: Sequence[Mapping[str, Any]], *, root: Path,
    require_archived: bool, allow_same_inode_crash: bool = False,
) -> dict[str, Any]:
    pairs = _small_prefix_pairs(
        rows, root=root, require_archived=require_archived,
        allow_same_inode_crash=allow_same_inode_crash)
    states = [copy.deepcopy(pair["capture"]["chosen_state"])
              for pair in pairs
              if pair["capture"].get("chosen_state") is not None]
    projection = [{key: state[key] for key in (
        "state_id", "state_identity_digest", "scene_id", "stratum", "split_role")}
        for state in sorted(states, key=lambda value: value["state_id"])]
    cursor = str(pairs[-1]["scene_id"])
    if (_digest(projection) != SMALL_PREFIX_STATE_PROJECTION_DIGEST
            or sum(state["stratum"] == "general" for state in states) != 5
            or sum(state["stratum"] == "safety_enriched" for state in states) != 5
            or any(state["stratum"] == "completion_enriched" for state in states)
            or cursor != SMALL_PREFIX_CURSOR_SCENE_ID):
        raise PerformanceInterruptionError("small prefix state projection changed")
    trace = pairs[-1].pop("reducer_trace")
    return {
        "states": states,
        "pairs": pairs,
        "state_projection_digest": SMALL_PREFIX_STATE_PROJECTION_DIGEST,
        "resolver_cursor_scene_id": SMALL_PREFIX_CURSOR_SCENE_ID,
        "reducer_trace_digest": _digest(trace),
        "general_count": 5,
        "safety_enriched_count": 5,
        "completion_enriched_count": 0,
    }


def _validate_outcome_attestation(attestation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        SELECTOR.validate_phase1_outcome_surface_absence_attestation(attestation)
    except Exception as exc:
        raise PerformanceInterruptionError("outcome-surface absence audit failed") from exc
    return copy.deepcopy(dict(attestation))


def _archive_one(binding: Mapping[str, Any], *, root: Path, label: str) -> None:
    _payload, source = _locate_for_archive(binding, root=root, label=label)
    active, archive = _binding_paths(binding, root=root)
    if source == archive:
        if _exact_file(active, binding):
            if not _same_file(active, archive):
                raise PerformanceInterruptionError(
                    f"{label} active/archive collision")
            active.unlink()
            _fsync_directories(active.parent, archive.parent)
        return
    if source != active:
        raise PerformanceInterruptionError(f"{label} location changed")
    archive.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(archive.parent)
    if archive.exists() or archive.is_symlink():
        raise PerformanceInterruptionError(f"{label} archive collision")
    os.link(active, archive, follow_symlinks=False)
    _fsync_directories(archive.parent)
    active.unlink()
    _fsync_directories(active.parent, archive.parent)


def _dynamic_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": row["path"],
        "archive_path": row["archive_path"],
        "self_digest_key": (
            ("mixed_replacement_scene_" if "mixed_preoutcome" in row["path"] else
             "state_resolution_scene_") + row["kind"] + "_digest"
        ),
        "self_digest": row["self_digest"],
        "raw_sha256": row["raw_sha256"],
        "byte_count": row["byte_count"],
    }


def _fsync_directories(*directories: Path) -> None:
    for path in dict.fromkeys(directories):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PerformanceInterruptionError("temporary lineage path exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        installed = True
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _receipt_payload(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str, outcome_attestation: Mapping[str, Any],
    fixed_transport_rows: Sequence[Mapping[str, Any]],
    small_prefix_rows: Sequence[Mapping[str, Any]],
    small_prefix_reducer_trace_digest: str,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "record_complete": True,
        "attempt_complete": False,
        "binding_receipt": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "interrupted_source_repository_commit": INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "interrupted_clean_source_binding_digest": INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
        "interrupted_bound_implementations_digest": INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
        "interrupted_selection_digest": INTERRUPTED_SELECTION_DIGEST,
        "superseding_source_repository_commit": source_repository_commit,
        "superseding_clean_source_binding_digest": clean_source_binding_digest,
        "superseding_bound_implementations_digest": bound_implementations_digest,
        "execution": {
            "argv": INTERRUPTED_COMMAND,
            "pid": INTERRUPTED_PID,
            "cutoff_elapsed": CUTOFF_ELAPSED,
            "cutoff_cpu": CUTOFF_CPU,
            "exit": "KeyboardInterrupt",
            "call_chain": INTERRUPTED_CALL_CHAIN,
            "terminal_state_shard_issued": False,
            "terminal_failure_receipt_issued": False,
        },
        "interrupted_authorities": INTERRUPTED_AUTHORITIES,
        "fixed_state_shards": list(FIXED_STATE_SHARDS),
        "fixed_transport_rows": list(fixed_transport_rows),
        "fixed_transport_row_count": FIXED_TRANSPORT_ROW_COUNT,
        "fixed_transport_byte_count": FIXED_TRANSPORT_BYTE_COUNT,
        "fixed_transport_row_set_digest": FIXED_TRANSPORT_ROW_SET_DIGEST,
        "small_prefix_roots": SMALL_PREFIX_ROOTS,
        "small_prefix_rows": list(small_prefix_rows),
        "small_prefix_request_count": SMALL_PREFIX_REQUEST_COUNT,
        "small_prefix_capture_count": SMALL_PREFIX_CAPTURE_COUNT,
        "small_prefix_row_count": SMALL_PREFIX_ROW_COUNT,
        "small_prefix_byte_count": SMALL_PREFIX_BYTE_COUNT,
        "small_prefix_row_set_digest": SMALL_PREFIX_ROW_SET_DIGEST,
        "small_prefix_historical_inventory_digest": SMALL_PREFIX_HISTORICAL_INVENTORY_DIGEST,
        "small_prefix_historical_compact_digest": SMALL_PREFIX_HISTORICAL_COMPACT_DIGEST,
        "small_prefix_state_projection_digest": SMALL_PREFIX_STATE_PROJECTION_DIGEST,
        "small_prefix_cursor_scene_id": SMALL_PREFIX_CURSOR_SCENE_ID,
        "small_prefix_reducer_trace_digest":
            small_prefix_reducer_trace_digest,
        "small_prefix_stops_at_first_full_quota": True,
        "outcome_surface_absence_attestation": copy.deepcopy(dict(outcome_attestation)),
        **ZERO_OUTCOME_FIELDS,
        "small_family_candidate_allocation_search_complete": False,
        "fixed_state_shards_scientifically_reusable_only_after_successor_revalidation": True,
        "archived_transport_bytes_are_never_rewritten": True,
        "top_level_lineage_rewrite_is_sufficient": False,
        "projection_fix_receipt_requires_successor_reissue": True,
    }
    payload["preoutcome_small_search_performance_interruption_receipt_digest"] = _digest(payload)
    return payload


def receipt_binding(receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    on_disk = _load_json(path, "performance interruption receipt")
    _verify_self(
        on_disk,
        "preoutcome_small_search_performance_interruption_receipt_digest",
        "performance interruption receipt")
    if on_disk != dict(receipt) or on_disk.get("status") != STATUS:
        raise PerformanceInterruptionError(
            "performance interruption binding payload changed")
    raw = path.read_bytes()
    return {
        "path": str(RECEIPT_RELATIVE_PATH),
        "receipt_digest": receipt["preoutcome_small_search_performance_interruption_receipt_digest"],
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "status": STATUS,
    }


def validate_archived_fixed_state_shards(
    receipt: Mapping[str, Any], *, root: Path = ROOT,
    revalidate_predecessor: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]], Any] | None = None,
    successor_bindings: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reopen exact old shards/transports and optionally replay their semantics."""

    if receipt.get("fixed_state_shards") != list(FIXED_STATE_SHARDS):
        raise PerformanceInterruptionError("fixed-shard receipt inventory changed")
    observed_rows: list[dict[str, Any]] = []
    result: dict[str, dict[str, Any]] = {}
    for binding in FIXED_STATE_SHARDS:
        shard, _path = _locate(binding, root=root, require_archived=True,
                               label=f"fixed shard {binding['family']}")
        rows = _fixed_transport_rows(shard, binding, root=root, archived=True)
        observed_rows.extend(rows)
        if revalidate_predecessor is not None:
            verdict = revalidate_predecessor(
                copy.deepcopy(shard), copy.deepcopy(rows),
                copy.deepcopy(dict(successor_bindings or {})))
            if verdict is not True:
                raise PerformanceInterruptionError(
                    f"semantic replay did not pass for {binding['family']}")
        result[str(binding["family"])] = shard
    receipt_rows = receipt.get("fixed_transport_rows")
    if (observed_rows != receipt_rows
            or len(observed_rows) != FIXED_TRANSPORT_ROW_COUNT
            or sum(row["byte_count"] for row in observed_rows)
            != FIXED_TRANSPORT_BYTE_COUNT
            or _digest([{key: value for key, value in row.items()
                         if key != "archive_path"} for row in observed_rows])
            != FIXED_TRANSPORT_ROW_SET_DIGEST):
        raise PerformanceInterruptionError("archived fixed transport changed")
    return result


def validated_small_fixed_prefix(
    receipt: Mapping[str, Any], *, root: Path = ROOT,
    revalidate_prefix: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
) -> dict[str, Any]:
    rows = _small_prefix_rows(root=root, archived=True)
    if rows != receipt.get("small_prefix_rows"):
        raise PerformanceInterruptionError("small prefix receipt rows changed")
    projection = _small_prefix_projection(
        rows, root=root, require_archived=True)
    pairs = projection.pop("pairs")
    if revalidate_prefix is not None and revalidate_prefix(
            copy.deepcopy(pairs)) is not True:
        raise PerformanceInterruptionError(
            "small prefix scientific/reducer replay did not pass")
    return projection


def validate_performance_interruption_receipt(
    receipt: Mapping[str, Any], *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str, root: Path = ROOT,
) -> dict[str, Any]:
    self_key = "preoutcome_small_search_performance_interruption_receipt_digest"
    _verify_self(receipt, self_key, "performance interruption receipt")
    attestation = _validate_outcome_attestation(
        receipt.get("outcome_surface_absence_attestation", {}))
    if (
        receipt.get("schema") != SCHEMA or receipt.get("status") != STATUS
        or receipt.get("record_complete") is not True
        or receipt.get("attempt_complete") is not False
        or receipt.get("binding_receipt") is not False
        or receipt.get("scientific_gate_input") is not False
        or receipt.get("may_satisfy_selector_gate") is not False
        or receipt.get("interrupted_source_repository_commit") != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
        or receipt.get("interrupted_clean_source_binding_digest") != INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST
        or receipt.get("interrupted_bound_implementations_digest") != INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST
        or receipt.get("interrupted_selection_digest") != INTERRUPTED_SELECTION_DIGEST
        or receipt.get("superseding_source_repository_commit") != expected_source_repository_commit
        or receipt.get("superseding_clean_source_binding_digest") != expected_clean_source_binding_digest
        or receipt.get("superseding_bound_implementations_digest") != expected_bound_implementations_digest
        or receipt.get("execution") != {
            "argv": INTERRUPTED_COMMAND, "pid": INTERRUPTED_PID,
            "cutoff_elapsed": CUTOFF_ELAPSED, "cutoff_cpu": CUTOFF_CPU,
            "exit": "KeyboardInterrupt", "call_chain": INTERRUPTED_CALL_CHAIN,
            "terminal_state_shard_issued": False,
            "terminal_failure_receipt_issued": False,
        }
        or receipt.get("interrupted_authorities") != INTERRUPTED_AUTHORITIES
        or receipt.get("small_prefix_row_set_digest") != SMALL_PREFIX_ROW_SET_DIGEST
        or receipt.get("small_prefix_historical_inventory_digest") != SMALL_PREFIX_HISTORICAL_INVENTORY_DIGEST
        or receipt.get("small_prefix_historical_compact_digest") != SMALL_PREFIX_HISTORICAL_COMPACT_DIGEST
        or receipt.get("small_prefix_state_projection_digest") != SMALL_PREFIX_STATE_PROJECTION_DIGEST
        or receipt.get("small_prefix_cursor_scene_id") != SMALL_PREFIX_CURSOR_SCENE_ID
        or receipt.get("small_prefix_stops_at_first_full_quota") is not True
        or any(receipt.get(key) != value for key, value in ZERO_OUTCOME_FIELDS.items())
        or receipt.get("small_family_candidate_allocation_search_complete") is not False
        or receipt.get("fixed_state_shards_scientifically_reusable_only_after_successor_revalidation") is not True
        or receipt.get("archived_transport_bytes_are_never_rewritten") is not True
        or receipt.get("top_level_lineage_rewrite_is_sufficient") is not False
        or receipt.get("projection_fix_receipt_requires_successor_reissue") is not True
    ):
        raise PerformanceInterruptionError("performance interruption receipt contract changed")
    # Exact key-set and reconstruction reject a self-resigned narrative or an
    # added scientific claim.
    for label, binding in INTERRUPTED_AUTHORITIES.items():
        _locate(binding, root=root, require_archived=True,
                label=f"interrupted authority {label}")
    validate_archived_fixed_state_shards(receipt, root=root)
    small_projection = validated_small_fixed_prefix(receipt, root=root)
    if (receipt.get("small_prefix_reducer_trace_digest")
            != small_projection["reducer_trace_digest"]):
        raise PerformanceInterruptionError(
            "small prefix reducer trace binding changed")
    expected = _receipt_payload(
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        outcome_attestation=attestation,
        fixed_transport_rows=receipt.get("fixed_transport_rows", []),
        small_prefix_rows=receipt.get("small_prefix_rows", []),
        small_prefix_reducer_trace_digest=small_projection[
            "reducer_trace_digest"],
    )
    if dict(receipt) != expected:
        raise PerformanceInterruptionError("performance interruption receipt differs from exact reconstruction")
    return dict(receipt)


def load_and_validate_performance_interruption_receipt(
    *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str, root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    receipt = _load_json(path, "performance interruption receipt")
    return validate_performance_interruption_receipt(
        receipt,
        expected_source_repository_commit=expected_source_repository_commit,
        expected_clean_source_binding_digest=expected_clean_source_binding_digest,
        expected_bound_implementations_digest=expected_bound_implementations_digest,
        root=root,
    )


# Explicit V1 names keep the immutable ca09 receipt independently callable after
# V2 becomes the sole active performance authority.  The historical names above
# remain aliases for compatibility with the transition recorder.
validate_performance_interruption_receipt_v1 = \
    validate_performance_interruption_receipt
load_and_validate_performance_interruption_receipt_v1 = \
    load_and_validate_performance_interruption_receipt


def load_and_validate_archived_performance_interruption_receipt_v1(
    *, receipt: Mapping[str, Any], receipt_binding: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the exact archived ca09 V1 receipt without a current-HEAD claim."""

    binding = _validate_receipt_binding(
        receipt_binding, label="archived V1 performance interruption",
        expected_status=V1_STATUS)
    if binding["path"] == str(V1_RECEIPT_RELATIVE_PATH):
        raise PerformanceInterruptionError(
            "archived V1 performance receipt still names its active path")
    active = _pin_managed(V1_RECEIPT_RELATIVE_PATH, root=root)
    if active.exists() or active.is_symlink():
        raise PerformanceInterruptionError(
            "archived V1 performance receipt still has an active alias")
    payload, _ = _load_exact_bound_receipt(
        receipt, binding, self_key=V1_SELF_KEY, status=V1_STATUS,
        label="archived V1 performance interruption", root=root)
    return validate_performance_interruption_receipt_v1(
        payload,
        expected_source_repository_commit=str(
            payload.get("superseding_source_repository_commit", "")),
        expected_clean_source_binding_digest=str(
            payload.get("superseding_clean_source_binding_digest", "")),
        expected_bound_implementations_digest=str(
            payload.get("superseding_bound_implementations_digest", "")),
        root=root,
    )


def issue_and_archive_performance_interruption_receipt(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str,
    outcome_surface_absent: Callable[[], Mapping[str, Any]],
    revalidate_small_prefix: Callable[[Sequence[Mapping[str, Any]]], Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Archive exact a1b bytes and issue/reopen one immutable receipt.

    The successor projection-fix receipt is deterministically reissued after
    the old a1b projection receipt is archived.  This closes the transitive
    authority gap before a successor scorer contract is issued.
    """

    receipt_path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit=source_repository_commit,
            expected_clean_source_binding_digest=clean_source_binding_digest,
            expected_bound_implementations_digest=bound_implementations_digest,
            root=root,
        )
    else:
        attestation = _validate_outcome_attestation(outcome_surface_absent())
        authority_locations = {
            label: _locate_for_archive(
                binding, root=root,
                label=f"interrupted authority {label}")
            for label, binding in INTERRUPTED_AUTHORITIES.items()
        }
        fixed_rows: list[dict[str, Any]] = []
        for binding in FIXED_STATE_SHARDS:
            shard, _ = _locate_for_archive(
                binding, root=root,
                label=f"fixed shard {binding['family']}")
            fixed_rows.extend(_fixed_transport_rows(
                shard, binding, root=root, archived=None))
        if (len(fixed_rows) != FIXED_TRANSPORT_ROW_COUNT
                or sum(row["byte_count"] for row in fixed_rows) != FIXED_TRANSPORT_BYTE_COUNT
                or _digest([{key: value for key, value in row.items()
                             if key != "archive_path"} for row in fixed_rows])
                != FIXED_TRANSPORT_ROW_SET_DIGEST):
            raise PerformanceInterruptionError("complete fixed transport set changed")
        small_rows = _small_prefix_rows(root=root, archived=None)
        # Establish the exact fixed 5G/5S prefix and cursor before moving even
        # one byte.  _locate supports a partially archived crash prefix while
        # still requiring every row's exact raw/self binding.
        small_projection = _small_prefix_projection(
            small_rows, root=root, require_archived=False,
            allow_same_inode_crash=True)
        small_pairs = small_projection.pop("pairs")
        if revalidate_small_prefix(copy.deepcopy(small_pairs)) is not True:
            raise PerformanceInterruptionError(
                "small prefix scientific/reducer replay did not pass")
        expected = _receipt_payload(
            source_repository_commit=source_repository_commit,
            clean_source_binding_digest=clean_source_binding_digest,
            bound_implementations_digest=bound_implementations_digest,
            outcome_attestation=attestation,
            fixed_transport_rows=fixed_rows,
            small_prefix_rows=small_rows,
            small_prefix_reducer_trace_digest=small_projection[
                "reducer_trace_digest"],
        )
        # All bytes and the outcome-free gate have passed before the first move.
        for row in [*fixed_rows, *small_rows]:
            _archive_one(_dynamic_binding(row), root=root,
                         label=f"transport {row['path']}")
        for binding in FIXED_STATE_SHARDS:
            _archive_one(binding, root=root,
                         label=f"fixed shard {binding['family']}")
        for label, binding in INTERRUPTED_AUTHORITIES.items():
            # authority_locations is intentionally retained until here so the
            # complete pre-mutation validation cannot be optimized away.
            if label not in authority_locations:  # pragma: no cover
                raise PerformanceInterruptionError("authority prevalidation lost")
            _archive_one(binding, root=root,
                         label=f"interrupted authority {label}")
        _atomic_write(receipt_path, expected)
        receipt = validate_performance_interruption_receipt(
            expected,
            expected_source_repository_commit=source_repository_commit,
            expected_clean_source_binding_digest=clean_source_binding_digest,
            expected_bound_implementations_digest=bound_implementations_digest,
            root=root,
        )
    # Every invocation replays the exact archived prefix through the caller's
    # central scientific validator.  This is read-only after issuance.
    validated_small_fixed_prefix(
        receipt, root=root, revalidate_prefix=revalidate_small_prefix)
    # The old active projection receipt was archived above.  Its original
    # request/capture rows and older authorities remain exactly where the
    # projection module binds them, so this issues the same semantic lineage
    # under the successor source without modifying the performance receipt.
    PROJECTION.issue_and_archive_interruption_receipt(
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        root=root,
    )
    return receipt


def _validated_current_projection_binding(
    receipt: Mapping[str, Any], *, source_repository_commit: str,
    clean_source_binding_digest: str, bound_implementations_digest: str,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the supplied projection receipt to be the current exact file."""

    current = PROJECTION.load_and_validate_interruption_receipt(
        expected_source_repository_commit=source_repository_commit,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        root=root,
    )
    if current != dict(receipt):
        raise PerformanceInterruptionError(
            "supplied current projection receipt differs from its active bytes")
    binding = _validate_receipt_binding(
        PROJECTION.receipt_binding(current, root=root),
        label="current projection-fix interruption",
        expected_status=PROJECTION.STATUS,
    )
    if binding["path"] != str(PROJECTION.RECEIPT_RELATIVE_PATH):
        raise PerformanceInterruptionError(
            "current projection-fix interruption path changed")
    return current, binding


def _validated_source_transition(
    binding: Mapping[str, Any], *, source_repository_commit: str,
    clean_source_binding_digest: str, bound_implementations_digest: str,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently validate the active transition and its exact bytes."""

    supplied = _validate_receipt_binding(
        binding, label="fixed-reissue source transition",
        expected_status=TRANSITION.STATUS)
    transition = TRANSITION.load_and_validate_interruption_receipt(
        expected_source_repository_commit=source_repository_commit,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        root=root,
    )
    canonical = _validate_receipt_binding(
        TRANSITION.receipt_binding(transition, root=root),
        label="fixed-reissue source transition",
        expected_status=TRANSITION.STATUS)
    if canonical["path"] != str(TRANSITION.RECEIPT_RELATIVE_PATH):
        raise PerformanceInterruptionError(
            "fixed-reissue source-transition path changed")
    if supplied != canonical:
        raise PerformanceInterruptionError(
            "supplied fixed-reissue source-transition binding changed")
    return transition, canonical


def _validated_transition_predecessor_v1(
    transition: Mapping[str, Any], binding: Mapping[str, Any], *,
    supplied_receipt: Mapping[str, Any] | None = None,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the V1 authority named by the validated transition."""

    supplied_binding = _validate_receipt_binding(
        binding, label="archived V1 performance interruption",
        expected_status=V1_STATUS)
    canonical_binding = _validate_receipt_binding(
        TRANSITION.archived_performance_receipt_binding_v1(
            transition, root=root),
        label="archived V1 performance interruption",
        expected_status=V1_STATUS)
    if supplied_binding != canonical_binding:
        raise PerformanceInterruptionError(
            "archived V1 binding is not the transition-bound authority")
    predecessor = TRANSITION.load_archived_performance_receipt_v1(
        transition, root=root)
    if supplied_receipt is not None and dict(supplied_receipt) != predecessor:
        raise PerformanceInterruptionError(
            "supplied archived V1 receipt is not the transition-bound payload")
    validated = load_and_validate_archived_performance_interruption_receipt_v1(
        receipt=predecessor,
        receipt_binding=canonical_binding,
        root=root,
    )
    return validated, canonical_binding


def _assert_v2_preissuance_no_write(*, root: Path) -> None:
    """Reject any durable successor wrapper/prefix byte before V2 issuance."""

    for binding in FIXED_STATE_SHARDS:
        active = _pin_managed(binding["active_path"], root=root)
        if active.exists() or active.is_symlink():
            raise PerformanceInterruptionError(
                "V2 performance issuance found a fixed-shard successor write")
        if active.parent.is_dir():
            prefix = f".{active.name}.tmp-"
            if any(entry.name.startswith(prefix) for entry in active.parent.iterdir()):
                raise PerformanceInterruptionError(
                    "V2 performance issuance found a fixed-shard temporary write")
    prefix_receipt = _pin_managed(
        SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH, root=root)
    if prefix_receipt.exists() or prefix_receipt.is_symlink():
        raise PerformanceInterruptionError(
            "V2 performance issuance found a small-prefix reissue receipt")
    for kind, relative in SMALL_PREFIX_ROOTS.items():
        directory = _pin_managed(relative, root=root)
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise PerformanceInterruptionError(
                f"V2 performance {kind} prefix root changed")
        entries = list(directory.iterdir())
        if entries:
            raise PerformanceInterruptionError(
                f"V2 performance issuance found a {kind} prefix write")


def _receipt_payload_v2(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str,
    source_transition_receipt_binding: Mapping[str, Any],
    predecessor_v1_receipt_binding: Mapping[str, Any],
    current_projection_receipt_binding: Mapping[str, Any],
    outcome_attestation: Mapping[str, Any],
    fixed_transport_rows: Sequence[Mapping[str, Any]],
    small_prefix_rows: Sequence[Mapping[str, Any]],
    small_prefix_reducer_trace_digest: str,
) -> dict[str, Any]:
    """Build the current-source V2 authority from immutable V1 evidence."""

    transition = _validate_receipt_binding(
        source_transition_receipt_binding,
        label="fixed-reissue source transition")
    predecessor = _validate_receipt_binding(
        predecessor_v1_receipt_binding,
        label="archived V1 performance interruption",
        expected_status=V1_STATUS)
    projection = _validate_receipt_binding(
        current_projection_receipt_binding,
        label="current projection-fix interruption",
        expected_status=PROJECTION.STATUS)
    payload = _receipt_payload(
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        outcome_attestation=outcome_attestation,
        fixed_transport_rows=fixed_transport_rows,
        small_prefix_rows=small_prefix_rows,
        small_prefix_reducer_trace_digest=small_prefix_reducer_trace_digest,
    )
    payload.pop(V1_SELF_KEY)
    payload["schema"] = V2_SCHEMA
    payload["status"] = V2_STATUS
    payload["source_transition_receipt"] = transition
    payload["predecessor_v1_performance_interruption_receipt"] = predecessor
    payload["current_projection_fix_interruption_receipt"] = projection
    payload["predecessor_v1_receipt_validated"] = True
    payload["current_projection_fix_receipt_validated"] = True
    payload["original_archives_reopened_read_only"] = True
    payload["archive_mutation_performed"] = False
    payload["successor_fixed_wrapper_count_at_issuance"] = 0
    payload["successor_small_prefix_request_count_at_issuance"] = 0
    payload["successor_small_prefix_capture_count_at_issuance"] = 0
    payload["successor_small_prefix_reissue_receipt_issued"] = False
    payload["projection_fix_receipt_requires_successor_reissue"] = False
    payload[V2_SELF_KEY] = _digest(payload)
    return payload


def validate_performance_interruption_receipt_v2(
    receipt: Mapping[str, Any], *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str,
    expected_source_transition_receipt_binding: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Reconstruct V2 from archived V1 and the current projection read-only."""

    _verify_self(receipt, V2_SELF_KEY, "V2 performance interruption receipt")
    transition_receipt, transition = _validated_source_transition(
        expected_source_transition_receipt_binding,
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        root=root,
    )
    if receipt.get("source_transition_receipt") != transition:
        raise PerformanceInterruptionError(
            "V2 performance source-transition binding changed")
    predecessor_binding = _validate_receipt_binding(
        receipt.get("predecessor_v1_performance_interruption_receipt", {}),
        label="archived V1 performance interruption",
        expected_status=V1_STATUS)
    predecessor, predecessor_binding = _validated_transition_predecessor_v1(
        transition_receipt, predecessor_binding, root=root)
    if expected_source_repository_commit == predecessor.get(
            "superseding_source_repository_commit"):
        raise PerformanceInterruptionError(
            "V2 performance source did not advance beyond archived V1")
    _projection, projection_binding = _validated_current_projection_binding(
        _load_json(
            _pin_managed(PROJECTION.RECEIPT_RELATIVE_PATH, root=root),
            "current projection-fix interruption"),
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        root=root,
    )
    if receipt.get("current_projection_fix_interruption_receipt") != \
            projection_binding:
        raise PerformanceInterruptionError(
            "V2 performance current projection binding changed")
    attestation = _validate_outcome_attestation(
        receipt.get("outcome_surface_absence_attestation", {}))
    validate_archived_fixed_state_shards(predecessor, root=root)
    small_projection = validated_small_fixed_prefix(predecessor, root=root)
    expected = _receipt_payload_v2(
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        source_transition_receipt_binding=transition,
        predecessor_v1_receipt_binding=predecessor_binding,
        current_projection_receipt_binding=projection_binding,
        outcome_attestation=attestation,
        fixed_transport_rows=predecessor["fixed_transport_rows"],
        small_prefix_rows=predecessor["small_prefix_rows"],
        small_prefix_reducer_trace_digest=small_projection[
            "reducer_trace_digest"],
    )
    if dict(receipt) != expected:
        raise PerformanceInterruptionError(
            "V2 performance interruption receipt differs from exact reconstruction")
    return dict(receipt)


def load_and_validate_performance_interruption_receipt_v2(
    *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str,
    expected_source_transition_receipt_binding: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(V2_RECEIPT_RELATIVE_PATH, root=root)
    receipt = _load_json(path, "V2 performance interruption receipt")
    return validate_performance_interruption_receipt_v2(
        receipt,
        expected_source_repository_commit=expected_source_repository_commit,
        expected_clean_source_binding_digest=expected_clean_source_binding_digest,
        expected_bound_implementations_digest=
            expected_bound_implementations_digest,
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        root=root,
    )


def performance_interruption_receipt_binding_v2(
    receipt: Mapping[str, Any], *, root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(V2_RECEIPT_RELATIVE_PATH, root=root)
    on_disk = _load_json(path, "V2 performance interruption receipt")
    _verify_self(on_disk, V2_SELF_KEY, "V2 performance interruption receipt")
    if on_disk != dict(receipt) or on_disk.get("status") != V2_STATUS:
        raise PerformanceInterruptionError(
            "V2 performance interruption binding payload changed")
    raw = path.read_bytes()
    return {
        "path": str(V2_RECEIPT_RELATIVE_PATH),
        "receipt_digest": receipt[V2_SELF_KEY],
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "status": V2_STATUS,
    }


def issue_performance_interruption_receipt_v2(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str,
    source_transition_receipt_binding: Mapping[str, Any],
    predecessor_v1_receipt: Mapping[str, Any],
    predecessor_v1_receipt_binding: Mapping[str, Any],
    current_projection_receipt: Mapping[str, Any],
    outcome_surface_absent: Callable[[], Mapping[str, Any]],
    revalidate_small_prefix: Callable[[Sequence[Mapping[str, Any]]], Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Issue V2 from exact existing archives; never move or rewrite old bytes."""

    transition_receipt, transition = _validated_source_transition(
        source_transition_receipt_binding,
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        root=root,
    )
    predecessor, predecessor_binding = _validated_transition_predecessor_v1(
        transition_receipt,
        predecessor_v1_receipt_binding,
        supplied_receipt=predecessor_v1_receipt,
        root=root,
    )
    if source_repository_commit == predecessor.get(
            "superseding_source_repository_commit"):
        raise PerformanceInterruptionError(
            "V2 performance source did not advance beyond archived V1")
    _projection, projection_binding = _validated_current_projection_binding(
        current_projection_receipt,
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        root=root,
    )
    path = _pin_managed(V2_RECEIPT_RELATIVE_PATH, root=root)
    if path.exists() or path.is_symlink():
        receipt = load_and_validate_performance_interruption_receipt_v2(
            expected_source_repository_commit=source_repository_commit,
            expected_clean_source_binding_digest=clean_source_binding_digest,
            expected_bound_implementations_digest=
                bound_implementations_digest,
            expected_source_transition_receipt_binding=transition,
            root=root,
        )
        validated_small_fixed_prefix(
            predecessor, root=root, revalidate_prefix=revalidate_small_prefix)
        return receipt

    _assert_v2_preissuance_no_write(root=root)
    attestation = _validate_outcome_attestation(outcome_surface_absent())
    validate_archived_fixed_state_shards(predecessor, root=root)
    small_projection = validated_small_fixed_prefix(
        predecessor, root=root, revalidate_prefix=revalidate_small_prefix)
    expected = _receipt_payload_v2(
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        source_transition_receipt_binding=transition,
        predecessor_v1_receipt_binding=predecessor_binding,
        current_projection_receipt_binding=projection_binding,
        outcome_attestation=attestation,
        fixed_transport_rows=predecessor["fixed_transport_rows"],
        small_prefix_rows=predecessor["small_prefix_rows"],
        small_prefix_reducer_trace_digest=small_projection[
            "reducer_trace_digest"],
    )
    # Close the validation-to-issuance race: no successor wrapper or prefix
    # byte may appear during the potentially expensive read-only replay above.
    _assert_v2_preissuance_no_write(root=root)
    if _validate_outcome_attestation(outcome_surface_absent()) != attestation:
        raise PerformanceInterruptionError(
            "V2 performance outcome-surface absence changed before issuance")
    _atomic_write(path, expected)
    return validate_performance_interruption_receipt_v2(
        expected,
        expected_source_repository_commit=source_repository_commit,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        expected_source_transition_receipt_binding=transition,
        root=root,
    )


def validate_successor_lineage_bindings(
    successor_bindings: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the sole registered source/contract/launch delta surface."""

    if not isinstance(successor_bindings, Mapping) \
            or set(successor_bindings) != set(SUCCESSOR_LINEAGE_KEYS):
        raise PerformanceInterruptionError("successor lineage key surface changed")
    normalized = {key: successor_bindings[key]
                  for key in SUCCESSOR_LINEAGE_KEYS}
    for key, value in normalized.items():
        expected_length = 40 if key == "source_repository_commit" else 64
        if (
            not isinstance(value, str) or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PerformanceInterruptionError(
                f"successor lineage value {key} is invalid")
    return normalized


def project_successor_state_shard_bindings(
    predecessor_bindings: Mapping[str, Any],
    successor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace exactly seven nested lineage keys and preserve every other key."""

    if not isinstance(predecessor_bindings, Mapping):
        raise PerformanceInterruptionError(
            "predecessor state-shard bindings are malformed")
    successor = validate_successor_lineage_bindings(successor_bindings)
    if any(key not in predecessor_bindings for key in SUCCESSOR_LINEAGE_KEYS):
        raise PerformanceInterruptionError(
            "predecessor state-shard bindings lack a lineage key")
    projected = copy.deepcopy(dict(predecessor_bindings))
    for key in SUCCESSOR_LINEAGE_KEYS:
        projected[key] = successor[key]
    if (
        set(projected) != set(predecessor_bindings)
        or any(projected[key] != predecessor_bindings[key]
               for key in predecessor_bindings
               if key not in SUCCESSOR_LINEAGE_KEYS)
    ):
        raise PerformanceInterruptionError(
            "successor changed an unregistered state-shard binding")
    return projected


def _successor_payload(
    predecessor: Mapping[str, Any], successor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    successor_bindings = validate_successor_lineage_bindings(
        successor_bindings)
    payload = copy.deepcopy(dict(predecessor))
    for key in SUCCESSOR_LINEAGE_KEYS:
        if key not in payload:
            raise PerformanceInterruptionError(f"predecessor lacks lineage key {key}")
        payload[key] = successor_bindings[key]
    self_key = "state_shard_digest"
    payload.pop(self_key, None)
    payload[self_key] = _digest(payload)
    changed = {key for key in payload if payload.get(key) != predecessor.get(key)}
    if not changed <= {*SUCCESSOR_LINEAGE_KEYS, self_key}:
        raise PerformanceInterruptionError("successor changed scientific shard payload")
    return payload


def performance_interruption_receipt_digest(
    receipt: Mapping[str, Any],
) -> str:
    """Return the schema-specific digest without treating V1 as current V2."""

    if receipt.get("schema") == V2_SCHEMA:
        key = V2_SELF_KEY
    elif receipt.get("schema") == V1_SCHEMA:
        key = V1_SELF_KEY
    else:
        raise PerformanceInterruptionError(
            "performance interruption receipt schema is unknown")
    value = receipt.get(key)
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PerformanceInterruptionError(
            "performance interruption receipt digest is invalid")
    return value


def _reissued_wrapper(
    *, binding: Mapping[str, Any], predecessor: Mapping[str, Any],
    successor: Mapping[str, Any], successor_bindings: Mapping[str, Any],
    receipt: Mapping[str, Any], transport_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": REISSUED_SHARD_SCHEMA,
        "status": REISSUED_SHARD_STATUS,
        "complete": True,
        "binding_receipt": False,
        "active_identity_input": True,
        "changes_scientific_selection": False,
        "reuses_exact_preoutcome_identity_evidence": True,
        "family": binding["family"],
        "kind": binding["kind"],
        "active_path": binding["active_path"],
        "predecessor_shard_binding": dict(binding),
        "predecessor_state_shard_digest": predecessor["state_shard_digest"],
        "archived_transport_row_count": len(transport_rows),
        "archived_transport_byte_count": sum(row["byte_count"] for row in transport_rows),
        "archived_transport_row_set_digest": _digest([
            {key: value for key, value in row.items() if key != "archive_path"}
            for row in transport_rows]),
        "successor_lineage_bindings": copy.deepcopy(dict(successor_bindings)),
        "performance_interruption_receipt_digest":
            performance_interruption_receipt_digest(receipt),
        "semantic_replay": {
            "archived_request_capture_bytes_reopened": True,
            "old_source_contract_launch_fields_treated_as_historical_lineage_only": True,
            "scientific_and_reducer_checks_replayed_in_memory": True,
            "archived_transport_bytes_rewritten": False,
        },
        "successor_state_shard": copy.deepcopy(dict(successor)),
        **ZERO_OUTCOME_FIELDS,
    }
    payload["source_reissued_state_shard_digest"] = _digest(payload)
    return payload


def validate_reissued_fixed_state_shard(
    wrapper: Mapping[str, Any], *, receipt: Mapping[str, Any],
    expected_source_transition_receipt_binding: Mapping[str, Any],
    revalidate_predecessor: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]], Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    _verify_self(wrapper, "source_reissued_state_shard_digest",
                 "source-reissued fixed shard")
    family = wrapper.get("family")
    matches = [row for row in FIXED_STATE_SHARDS if row["family"] == family]
    if len(matches) != 1:
        raise PerformanceInterruptionError("reissued shard family changed")
    binding = matches[0]
    predecessor, _ = _locate(binding, root=root, require_archived=True,
                             label=f"fixed shard {family}")
    rows = _fixed_transport_rows(predecessor, binding, root=root, archived=True)
    successor_bindings = validate_successor_lineage_bindings(
        wrapper.get("successor_lineage_bindings", {}))
    validated_receipt = validate_performance_interruption_receipt_v2(
        receipt,
        expected_source_repository_commit=successor_bindings[
            "source_repository_commit"],
        expected_clean_source_binding_digest=successor_bindings[
            "clean_source_binding_digest"],
        expected_bound_implementations_digest=successor_bindings[
            "bound_implementations_digest"],
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        root=root)
    expected_successor = _successor_payload(predecessor, successor_bindings)
    if revalidate_predecessor(
            copy.deepcopy(predecessor), copy.deepcopy(rows),
            copy.deepcopy(dict(successor_bindings))) is not True:
        raise PerformanceInterruptionError("reissued shard semantic replay failed")
    expected = _reissued_wrapper(
        binding=binding, predecessor=predecessor,
        successor=expected_successor, successor_bindings=successor_bindings,
        receipt=validated_receipt, transport_rows=rows)
    if dict(wrapper) != expected:
        raise PerformanceInterruptionError("source-reissued fixed shard changed")
    return expected_successor


def reissue_fixed_state_shards(
    *, receipt: Mapping[str, Any],
    expected_source_transition_receipt_binding: Mapping[str, Any],
    revalidate_predecessor: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]], Any],
    build_successor_bindings: Callable[[], Mapping[str, Any]],
    outcome_surface_absent: Callable[[], Mapping[str, Any]],
    root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Issue seven successor wrappers after exact archived semantic replay."""

    bindings = validate_successor_lineage_bindings(
        build_successor_bindings())
    validated_receipt = validate_performance_interruption_receipt_v2(
        receipt,
        expected_source_repository_commit=bindings["source_repository_commit"],
        expected_clean_source_binding_digest=bindings[
            "clean_source_binding_digest"],
        expected_bound_implementations_digest=bindings[
            "bound_implementations_digest"],
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        root=root)
    _validate_outcome_attestation(outcome_surface_absent())
    if (bindings.get("source_repository_commit")
            != receipt["superseding_source_repository_commit"]
            or bindings.get("clean_source_binding_digest")
            != receipt["superseding_clean_source_binding_digest"]
            or bindings.get("bound_implementations_digest")
            != receipt["superseding_bound_implementations_digest"]):
        raise PerformanceInterruptionError("successor bindings differ from interruption receipt")
    predecessors = validate_archived_fixed_state_shards(
        validated_receipt, root=root,
        revalidate_predecessor=revalidate_predecessor,
        successor_bindings=bindings)
    outputs: dict[str, dict[str, Any]] = {}
    for binding in FIXED_STATE_SHARDS:
        family = str(binding["family"])
        predecessor = predecessors[family]
        rows = _fixed_transport_rows(predecessor, binding, root=root, archived=True)
        successor = _successor_payload(predecessor, bindings)
        expected = _reissued_wrapper(
            binding=binding, predecessor=predecessor, successor=successor,
            successor_bindings=bindings, receipt=validated_receipt,
            transport_rows=rows)
        active = _pin_managed(binding["active_path"], root=root)
        if active.exists() or active.is_symlink():
            existing = _load_json(active, f"reissued fixed shard {family}")
            validate_reissued_fixed_state_shard(
                existing, receipt=validated_receipt,
                expected_source_transition_receipt_binding=
                    expected_source_transition_receipt_binding,
                revalidate_predecessor=revalidate_predecessor, root=root)
            if existing != expected:
                raise PerformanceInterruptionError("reissued shard collision")
            outputs[family] = existing
            continue
        _atomic_write(active, expected)
        validate_reissued_fixed_state_shard(
            expected, receipt=validated_receipt,
            expected_source_transition_receipt_binding=
                expected_source_transition_receipt_binding,
            revalidate_predecessor=revalidate_predecessor, root=root)
        outputs[family] = expected
    return outputs


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _small_request_semantic_projection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    projection = copy.deepcopy(dict(payload))
    projection.pop("state_shard_bindings", None)
    projection.pop(SMALL_PREFIX_REQUEST_SELF_KEY, None)
    return projection


def _small_capture_semantic_projection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    projection = copy.deepcopy(dict(payload))
    projection.pop("request", None)
    projection.pop(SMALL_PREFIX_REQUEST_SELF_KEY, None)
    projection.pop(SMALL_PREFIX_CAPTURE_SELF_KEY, None)
    return projection


def _successor_small_row(
    kind: str, payload: Mapping[str, Any], *, logical_path: str,
) -> dict[str, Any]:
    raw = _json_bytes(payload)
    scene_id = (payload.get("scene", {}).get("scene_id")
                if kind == "request" else payload.get("scene_id"))
    ordinal = (payload.get("scene_ordinal") if kind == "request" else
               payload.get("request", {}).get("scene_ordinal"))
    self_key = (SMALL_PREFIX_REQUEST_SELF_KEY if kind == "request"
                else SMALL_PREFIX_CAPTURE_SELF_KEY)
    return {
        "kind": kind,
        "name": Path(logical_path).name,
        "path": logical_path,
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "self_digest": payload[self_key],
        "scene_id": scene_id,
        "scene_ordinal": ordinal,
        "selected": (None if kind == "request"
                     else payload.get("chosen_state") is not None),
        "worker_failure": (None if kind == "request"
                           else payload.get("worker_failure")),
    }


def _project_small_prefix_pair(
    pair: Mapping[str, Any], successor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    successor_bindings = validate_successor_lineage_bindings(
        successor_bindings)
    predecessor_request = pair.get("request")
    predecessor_capture = pair.get("capture")
    if not isinstance(predecessor_request, Mapping) \
            or not isinstance(predecessor_capture, Mapping):
        raise PerformanceInterruptionError(
            "archived small prefix pair is malformed")
    projected_bindings = project_successor_state_shard_bindings(
        predecessor_request.get("state_shard_bindings", {}),
        successor_bindings)
    request = copy.deepcopy(dict(predecessor_request))
    request["state_shard_bindings"] = projected_bindings
    request.pop(SMALL_PREFIX_REQUEST_SELF_KEY, None)
    request[SMALL_PREFIX_REQUEST_SELF_KEY] = _digest(request)
    if (_small_request_semantic_projection(request)
            != _small_request_semantic_projection(predecessor_request)):
        raise PerformanceInterruptionError(
            "small prefix request scientific projection changed")

    capture = copy.deepcopy(dict(predecessor_capture))
    capture["request"] = copy.deepcopy(request)
    capture[SMALL_PREFIX_REQUEST_SELF_KEY] = request[
        SMALL_PREFIX_REQUEST_SELF_KEY]
    capture.pop(SMALL_PREFIX_CAPTURE_SELF_KEY, None)
    capture[SMALL_PREFIX_CAPTURE_SELF_KEY] = _digest(capture)
    if (_small_capture_semantic_projection(capture)
            != _small_capture_semantic_projection(predecessor_capture)):
        raise PerformanceInterruptionError(
            "small prefix capture scientific projection changed")

    name = f"{request[SMALL_PREFIX_REQUEST_SELF_KEY]}.json"
    request_path = str(Path(SMALL_PREFIX_ROOTS["request"]) / name)
    capture_path = str(Path(SMALL_PREFIX_ROOTS["capture"]) / name)
    return {
        "scene_id": pair["scene_id"],
        "scene_ordinal": pair["scene_ordinal"],
        "request": request,
        "capture": capture,
        "request_row": _successor_small_row(
            "request", request, logical_path=request_path),
        "capture_row": _successor_small_row(
            "capture", capture, logical_path=capture_path),
    }


def _prefix_row_binding(
    row: Mapping[str, Any], *, archived: bool,
) -> dict[str, Any]:
    return {
        "path": row["archive_path"] if archived else row["path"],
        "raw_sha256": row["raw_sha256"],
        "byte_count": row["byte_count"],
        "self_digest": row["self_digest"],
    }


def _small_prefix_mapping_rows(
    archived_pairs: Sequence[Mapping[str, Any]],
    successor_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(archived_pairs) != len(successor_pairs):
        raise PerformanceInterruptionError(
            "small prefix successor pair count changed")
    mappings: list[dict[str, Any]] = []
    for predecessor, successor in zip(
            archived_pairs, successor_pairs, strict=True):
        if (predecessor["scene_id"] != successor["scene_id"]
                or predecessor["scene_ordinal"] != successor["scene_ordinal"]):
            raise PerformanceInterruptionError(
                "small prefix successor pair order changed")
        mappings.append({
            "scene_id": predecessor["scene_id"],
            "scene_ordinal": predecessor["scene_ordinal"],
            "archived_request": _prefix_row_binding(
                predecessor["request_row"], archived=True),
            "archived_capture": _prefix_row_binding(
                predecessor["capture_row"], archived=True),
            "successor_request": _prefix_row_binding(
                successor["request_row"], archived=False),
            "successor_capture": _prefix_row_binding(
                successor["capture_row"], archived=False),
            "request_semantic_projection_digest": _digest(
                _small_request_semantic_projection(predecessor["request"])),
            "capture_semantic_projection_digest": _digest(
                _small_capture_semantic_projection(predecessor["capture"])),
        })
    return mappings


def _small_prefix_reissue_payload(
    *, performance_receipt: Mapping[str, Any],
    successor_bindings: Mapping[str, Any],
    archived_projection: Mapping[str, Any],
    archived_pairs: Sequence[Mapping[str, Any]],
    successor_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = validate_successor_lineage_bindings(successor_bindings)
    mappings = _small_prefix_mapping_rows(archived_pairs, successor_pairs)
    successor_rows = [
        copy.deepcopy(pair[f"{kind}_row"])
        for kind in ("request", "capture") for pair in successor_pairs
    ]
    request_projection_digests = [
        row["request_semantic_projection_digest"] for row in mappings]
    capture_projection_digests = [
        row["capture_semantic_projection_digest"] for row in mappings]
    payload = {
        "schema": SMALL_PREFIX_REISSUE_SCHEMA,
        "status": SMALL_PREFIX_REISSUE_STATUS,
        "complete": True,
        "binding_receipt": False,
        "active_identity_input": True,
        "changes_scientific_selection": False,
        "reuses_exact_preoutcome_identity_evidence": True,
        "performance_interruption_receipt_digest":
            performance_interruption_receipt_digest(performance_receipt),
        "successor_lineage_bindings": bindings,
        "projection_allowlist": list(SUCCESSOR_LINEAGE_KEYS),
        "archived_transport_row_count": SMALL_PREFIX_ROW_COUNT,
        "archived_transport_byte_count": SMALL_PREFIX_BYTE_COUNT,
        "archived_transport_row_set_digest": SMALL_PREFIX_ROW_SET_DIGEST,
        "successor_request_count": SMALL_PREFIX_REQUEST_COUNT,
        "successor_capture_count": SMALL_PREFIX_CAPTURE_COUNT,
        "successor_transport_row_count": len(successor_rows),
        "successor_transport_byte_count": sum(
            int(row["byte_count"]) for row in successor_rows),
        "successor_transport_row_set_digest": _digest(successor_rows),
        "mapping_rows": mappings,
        "request_semantic_projection_set_digest": _digest(
            request_projection_digests),
        "capture_semantic_projection_set_digest": _digest(
            capture_projection_digests),
        "selected_state_projection_digest": archived_projection[
            "state_projection_digest"],
        "resolver_cursor_scene_id": archived_projection[
            "resolver_cursor_scene_id"],
        "reducer_trace_digest": archived_projection[
            "reducer_trace_digest"],
        "general_count": archived_projection["general_count"],
        "safety_enriched_count": archived_projection[
            "safety_enriched_count"],
        "completion_enriched_count": archived_projection[
            "completion_enriched_count"],
        "semantic_replay": {
            "archived_pairs_reopened": True,
            "successor_pairs_validated_in_memory_before_write": True,
            "dynamic_quota_and_first_full_cursor_replayed": True,
            "archived_transport_bytes_rewritten": False,
            "simulator_execution_performed": False,
        },
        **ZERO_OUTCOME_FIELDS,
    }
    payload[SMALL_PREFIX_REISSUE_SELF_KEY] = _digest(payload)
    return payload


def _load_exact_successor_pair(
    expected: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    loaded: dict[str, Any] = {
        "scene_id": expected["scene_id"],
        "scene_ordinal": expected["scene_ordinal"],
    }
    for kind in ("request", "capture"):
        row = expected[f"{kind}_row"]
        path = _pin_managed(row["path"], root=root)
        if not path.is_file() or path.is_symlink():
            raise PerformanceInterruptionError(
                f"successor small prefix {kind} is missing")
        raw = path.read_bytes()
        payload = _load_json(path, f"successor small prefix {kind}")
        self_key = (SMALL_PREFIX_REQUEST_SELF_KEY if kind == "request"
                    else SMALL_PREFIX_CAPTURE_SELF_KEY)
        _verify_self(payload, self_key, f"successor small prefix {kind}")
        if (
            raw != _json_bytes(expected[kind])
            or len(raw) != row["byte_count"]
            or _raw_sha256(raw) != row["raw_sha256"]
            or payload != expected[kind]
        ):
            raise PerformanceInterruptionError(
                f"successor small prefix {kind} changed")
        loaded[kind] = payload
        loaded[f"{kind}_row"] = copy.deepcopy(dict(row))
    return loaded


def _validate_successor_prefix_inventory(
    expected_pairs: Sequence[Mapping[str, Any]], *, root: Path,
    require_complete: bool,
) -> None:
    for kind in ("request", "capture"):
        directory = _pin_managed(SMALL_PREFIX_ROOTS[kind], root=root)
        if directory.exists() and (not directory.is_dir() or directory.is_symlink()):
            raise PerformanceInterruptionError(
                f"successor small prefix {kind} root changed")
        expected = {
            Path(pair[f"{kind}_row"]["path"]).name for pair in expected_pairs}
        observed = set()
        if directory.is_dir():
            for path in directory.glob("*.json"):
                if path.is_symlink():
                    raise PerformanceInterruptionError(
                        f"successor small prefix {kind} contains a symlink")
                observed.add(path.name)
        if (require_complete and observed != expected) or not observed <= expected:
            raise PerformanceInterruptionError(
                f"successor small prefix {kind} inventory changed")


def _install_exact_json(
    path: Path, payload: Mapping[str, Any], *, label: str,
) -> None:
    if path.exists() or path.is_symlink():
        if (not path.is_file() or path.is_symlink()
                or path.read_bytes() != _json_bytes(payload)):
            raise PerformanceInterruptionError(f"{label} collision")
        return
    _atomic_write(path, payload)


def validate_small_prefix_reissue_receipt(
    receipt: Mapping[str, Any], *,
    performance_receipt: Mapping[str, Any],
    expected_source_transition_receipt_binding: Mapping[str, Any],
    successor_bindings: Mapping[str, Any],
    revalidate_prefix: Callable[[
        Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]],
        Mapping[str, Any]], Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Reconstruct every old/new pair and replay caller-owned science checks."""

    bindings = validate_successor_lineage_bindings(successor_bindings)
    validated_performance = validate_performance_interruption_receipt_v2(
        performance_receipt,
        expected_source_repository_commit=bindings[
            "source_repository_commit"],
        expected_clean_source_binding_digest=bindings[
            "clean_source_binding_digest"],
        expected_bound_implementations_digest=bindings[
            "bound_implementations_digest"],
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        root=root)
    if (
        bindings["source_repository_commit"]
        != validated_performance["superseding_source_repository_commit"]
        or bindings["clean_source_binding_digest"]
        != validated_performance["superseding_clean_source_binding_digest"]
        or bindings["bound_implementations_digest"]
        != validated_performance["superseding_bound_implementations_digest"]
    ):
        raise PerformanceInterruptionError(
            "small prefix successor source differs from interruption receipt")
    _verify_self(receipt, SMALL_PREFIX_REISSUE_SELF_KEY,
                 "small prefix reissue receipt")
    archived_rows = _small_prefix_rows(root=root, archived=True)
    if archived_rows != validated_performance.get("small_prefix_rows"):
        raise PerformanceInterruptionError(
            "small prefix archived rows differ from interruption receipt")
    archived_projection = _small_prefix_projection(
        archived_rows, root=root, require_archived=True)
    archived_pairs = archived_projection.pop("pairs")
    expected_successors = [
        _project_small_prefix_pair(pair, bindings) for pair in archived_pairs]
    _validate_successor_prefix_inventory(
        expected_successors, root=root, require_complete=True)
    successor_pairs = [
        _load_exact_successor_pair(pair, root=root)
        for pair in expected_successors]
    if revalidate_prefix(
            copy.deepcopy(archived_pairs), copy.deepcopy(successor_pairs),
            copy.deepcopy(bindings)) is not True:
        raise PerformanceInterruptionError(
            "small prefix successor scientific/reducer replay did not pass")
    expected = _small_prefix_reissue_payload(
        performance_receipt=validated_performance,
        successor_bindings=bindings,
        archived_projection=archived_projection,
        archived_pairs=archived_pairs,
        successor_pairs=successor_pairs)
    if dict(receipt) != expected:
        raise PerformanceInterruptionError(
            "small prefix reissue receipt differs from exact reconstruction")
    return expected


def load_and_validate_small_prefix_reissue_receipt(
    *, performance_receipt: Mapping[str, Any],
    expected_source_transition_receipt_binding: Mapping[str, Any],
    successor_bindings: Mapping[str, Any],
    revalidate_prefix: Callable[[
        Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]],
        Mapping[str, Any]], Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(
        SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH, root=root)
    receipt = _load_json(path, "small prefix reissue receipt")
    return validate_small_prefix_reissue_receipt(
        receipt, performance_receipt=performance_receipt,
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        successor_bindings=successor_bindings,
        revalidate_prefix=revalidate_prefix, root=root)


def small_prefix_reissue_receipt_binding(
    receipt: Mapping[str, Any], *, root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(
        SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH, root=root)
    on_disk = _load_json(path, "small prefix reissue receipt")
    _verify_self(on_disk, SMALL_PREFIX_REISSUE_SELF_KEY,
                 "small prefix reissue receipt")
    if (on_disk != dict(receipt)
            or on_disk.get("status") != SMALL_PREFIX_REISSUE_STATUS):
        raise PerformanceInterruptionError(
            "small prefix reissue binding payload changed")
    raw = path.read_bytes()
    return {
        "path": str(SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH),
        "receipt_digest": receipt[SMALL_PREFIX_REISSUE_SELF_KEY],
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "status": SMALL_PREFIX_REISSUE_STATUS,
    }


def reissue_small_fixed_prefix(
    *, performance_receipt: Mapping[str, Any],
    expected_source_transition_receipt_binding: Mapping[str, Any],
    build_successor_bindings: Callable[[], Mapping[str, Any]],
    revalidate_prefix: Callable[[
        Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]],
        Mapping[str, Any]], Any],
    outcome_surface_absent: Callable[[], Mapping[str, Any]],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Install 12 normal-schema successor pairs without simulator execution."""

    bindings = validate_successor_lineage_bindings(
        build_successor_bindings())
    validated_performance = validate_performance_interruption_receipt_v2(
        performance_receipt,
        expected_source_repository_commit=bindings[
            "source_repository_commit"],
        expected_clean_source_binding_digest=bindings[
            "clean_source_binding_digest"],
        expected_bound_implementations_digest=bindings[
            "bound_implementations_digest"],
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        root=root)
    _validate_outcome_attestation(outcome_surface_absent())
    if (
        bindings["source_repository_commit"]
        != validated_performance["superseding_source_repository_commit"]
        or bindings["clean_source_binding_digest"]
        != validated_performance["superseding_clean_source_binding_digest"]
        or bindings["bound_implementations_digest"]
        != validated_performance["superseding_bound_implementations_digest"]
    ):
        raise PerformanceInterruptionError(
            "small prefix successor source differs from interruption receipt")
    receipt_path = _pin_managed(
        SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH, root=root)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_and_validate_small_prefix_reissue_receipt(
            performance_receipt=validated_performance,
            expected_source_transition_receipt_binding=
                expected_source_transition_receipt_binding,
            successor_bindings=bindings,
            revalidate_prefix=revalidate_prefix, root=root)

    archived_rows = _small_prefix_rows(root=root, archived=True)
    if archived_rows != validated_performance.get("small_prefix_rows"):
        raise PerformanceInterruptionError(
            "small prefix archived rows differ from interruption receipt")
    archived_projection = _small_prefix_projection(
        archived_rows, root=root, require_archived=True)
    archived_pairs = archived_projection.pop("pairs")
    successor_pairs = [
        _project_small_prefix_pair(pair, bindings) for pair in archived_pairs]
    _validate_successor_prefix_inventory(
        successor_pairs, root=root, require_complete=False)
    if revalidate_prefix(
            copy.deepcopy(archived_pairs), copy.deepcopy(successor_pairs),
            copy.deepcopy(bindings)) is not True:
        raise PerformanceInterruptionError(
            "small prefix successor scientific/reducer replay did not pass")
    # The complete old/new projection passes before the first active write.
    for pair in successor_pairs:
        for kind in ("request", "capture"):
            path = _pin_managed(pair[f"{kind}_row"]["path"], root=root)
            _install_exact_json(
                path, pair[kind], label=f"successor small prefix {kind}")
    loaded_successors = [
        _load_exact_successor_pair(pair, root=root) for pair in successor_pairs]
    _validate_successor_prefix_inventory(
        loaded_successors, root=root, require_complete=True)
    expected = _small_prefix_reissue_payload(
        performance_receipt=validated_performance,
        successor_bindings=bindings,
        archived_projection=archived_projection,
        archived_pairs=archived_pairs,
        successor_pairs=loaded_successors)
    _atomic_write(receipt_path, expected)
    return validate_small_prefix_reissue_receipt(
        expected, performance_receipt=validated_performance,
        expected_source_transition_receipt_binding=
            expected_source_transition_receipt_binding,
        successor_bindings=bindings,
        revalidate_prefix=revalidate_prefix, root=root)


def lineage_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "receipt_path": str(RECEIPT_RELATIVE_PATH),
        "interrupted_source_repository_commit": INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "execution_command": INTERRUPTED_COMMAND,
        "execution_cutoff_elapsed": CUTOFF_ELAPSED,
        "execution_cutoff_cpu": CUTOFF_CPU,
        "execution_pid": INTERRUPTED_PID,
        "execution_exit": "KeyboardInterrupt",
        "execution_call_chain": INTERRUPTED_CALL_CHAIN,
        "interrupted_authorities": INTERRUPTED_AUTHORITIES,
        "fixed_state_shards": list(FIXED_STATE_SHARDS),
        "fixed_transport_row_count": FIXED_TRANSPORT_ROW_COUNT,
        "fixed_transport_byte_count": FIXED_TRANSPORT_BYTE_COUNT,
        "fixed_transport_row_set_digest": FIXED_TRANSPORT_ROW_SET_DIGEST,
        "small_prefix_request_count": SMALL_PREFIX_REQUEST_COUNT,
        "small_prefix_capture_count": SMALL_PREFIX_CAPTURE_COUNT,
        "small_prefix_row_count": SMALL_PREFIX_ROW_COUNT,
        "small_prefix_byte_count": SMALL_PREFIX_BYTE_COUNT,
        "small_prefix_row_set_digest": SMALL_PREFIX_ROW_SET_DIGEST,
        "small_prefix_historical_inventory_digest": SMALL_PREFIX_HISTORICAL_INVENTORY_DIGEST,
        "small_prefix_state_projection_digest": SMALL_PREFIX_STATE_PROJECTION_DIGEST,
        "small_prefix_cursor_scene_id": SMALL_PREFIX_CURSOR_SCENE_ID,
        "small_prefix_stops_at_first_full_quota": True,
        "terminal_state_shard_existed": False,
        "terminal_failure_receipt_existed": False,
        "scientific_outcome_existed": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "reissued_shard_schema": REISSUED_SHARD_SCHEMA,
        "small_prefix_reissue_schema": SMALL_PREFIX_REISSUE_SCHEMA,
        "small_prefix_reissue_receipt_path": str(
            SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH),
        "successor_projection_allowlist": list(SUCCESSOR_LINEAGE_KEYS),
        "reissue_requires_archived_transport_semantic_replay": True,
        "archived_transport_bytes_are_never_rewritten": True,
    }


lineage_contract_v1 = lineage_contract


def lineage_contract_v2() -> dict[str, Any]:
    """Static current contract for the source-transition-bound V2 authority."""

    predecessor = lineage_contract_v1()
    return {
        "schema": V2_SCHEMA,
        "status": V2_STATUS,
        "receipt_path": str(V2_RECEIPT_RELATIVE_PATH),
        "predecessor_v1_schema": V1_SCHEMA,
        "predecessor_v1_status": V1_STATUS,
        "predecessor_v1_active_path": str(V1_RECEIPT_RELATIVE_PATH),
        "predecessor_v1_lineage_contract": predecessor,
        "predecessor_v1_lineage_contract_digest": _digest(predecessor),
        "source_transition_receipt_required": True,
        "source_transition_receipt_binding_surface": sorted(
            _RECEIPT_BINDING_KEYS),
        "current_source_projection_fix_receipt_required": True,
        "original_a1b_archives_reopened_read_only": True,
        "archive_mutation_permitted": False,
        "successor_outputs_required_absent_at_issuance": True,
        "scientific_outcome_required_absent_at_issuance": True,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "reissued_shard_schema": REISSUED_SHARD_SCHEMA,
        "small_prefix_reissue_schema": SMALL_PREFIX_REISSUE_SCHEMA,
        "successor_projection_allowlist": list(SUCCESSOR_LINEAGE_KEYS),
        "reissue_requires_archived_transport_semantic_replay": True,
        "archived_transport_bytes_are_never_rewritten": True,
    }
