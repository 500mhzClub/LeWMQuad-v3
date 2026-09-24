"""Pure schedule adapter for the science-identical multiresolution V2 probe.

This module performs no file, generated-input, tensor, Torch, GPU, or output
access.  The V2 lifecycle runner must durably ledger the schedule open before
passing the already-read bytes and their authorized binding to Phase A.
Phase B later joins that immutable schedule view to the actual ordered train
pair identities without reopening or regenerating the schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from lewm.benchmarks import go2_shared_jepa_v5_matched_training_v1 as matched_v1


BOUND_SCHEDULE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v4/schedule.json"
)
BOUND_SCHEDULE_FILE_SHA256 = (
    "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270"
)
BOUND_SCHEDULE_CONTENT_SHA256 = (
    "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15"
)
BOUND_SCHEDULE_BYTE_COUNT = 607_373
BOUND_V4_SCHEDULE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_matched_training_v4_schedule_v1"
)
NORMALIZED_V1_SCHEDULE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_matched_training_v1_schedule_v1"
)
NORMALIZED_V1_SCHEDULE_CONTENT_SHA256 = (
    "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d"
)
V1_SCIENCE_CONTRACT_SHA256 = (
    "e181381c00585fa5df41a71fff918b5599acc955d59283ce397ba6dd530dc23f"
)
USED_PRESENTATIONS = 16_000
FROZEN_PREFIX_SHA256 = (
    (
        1_600,
        "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    ),
    (
        6_400,
        "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    ),
    (
        16_000,
        "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    ),
)
FROZEN_SCHEDULE_IDENTITY = {
    "indices_sha256":
        "a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663",
    "ordered_pair_ids_sha256":
        "74b90f10347a89d2151c4f65f76d6fc3c6a94fb3e8caa350d2a92e934e80840a",
    "presentation_pair_ids_sha256":
        "1534dcdd85feb8421639a0dc433473913f6674556e22e0fa9f515be455b7b79a",
    "per_update_pair_ids_sha256":
        "fe4aab82bd05b5e3438e8623319211ae75220f8bf3143223f6b6e375d91d46f0",
}
_SCHEDULE_FIELDS = {
    "schema",
    "seed",
    "train_pair_count",
    "presentation_count",
    "update_count",
    "microbatch_size",
    "accumulation_steps",
    "effective_batch_size",
    "ordered_pair_ids_sha256",
    "indices_sha256",
    "presentation_pair_ids_sha256",
    "per_update_pair_ids_sha256",
    "presentation_indices",
    "content_sha256",
}


class ScheduleAdapterIntegrityError(PermissionError):
    """The bound schedule escaped the exact V2 integrity contract."""


@dataclass(frozen=True)
class _SchedulePolicy:
    path: str
    file_sha256: str
    content_sha256: str
    byte_count: int
    bound_schema: str
    normalized_schema: str
    normalized_content_sha256: str
    schedule_identity_items: tuple[tuple[str, str], ...]
    prefix_items: tuple[tuple[int, str], ...]

    def binding(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
        }

    def schedule_identity(self) -> dict[str, str]:
        return dict(self.schedule_identity_items)


_FROZEN_POLICY = _SchedulePolicy(
    path=BOUND_SCHEDULE_PATH,
    file_sha256=BOUND_SCHEDULE_FILE_SHA256,
    content_sha256=BOUND_SCHEDULE_CONTENT_SHA256,
    byte_count=BOUND_SCHEDULE_BYTE_COUNT,
    bound_schema=BOUND_V4_SCHEDULE_SCHEMA,
    normalized_schema=NORMALIZED_V1_SCHEDULE_SCHEMA,
    normalized_content_sha256=NORMALIZED_V1_SCHEDULE_CONTENT_SHA256,
    schedule_identity_items=tuple(sorted(FROZEN_SCHEDULE_IDENTITY.items())),
    prefix_items=FROZEN_PREFIX_SHA256,
)


@dataclass(frozen=True)
class ValidatedScheduleV2:
    """Immutable Phase-A state carried to the later train-identity join."""

    presentation_indices: tuple[int, ...]
    normalized_v1_raw: bytes
    binding_path: str
    binding_file_sha256: str
    binding_content_sha256: str
    binding_byte_count: int

    def original_binding(self) -> dict[str, Any]:
        return {
            "path": self.binding_path,
            "file_sha256": self.binding_file_sha256,
            "content_sha256": self.binding_content_sha256,
            "byte_count": self.binding_byte_count,
        }


def _integrity(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleAdapterIntegrityError(message)


def _parse_schedule(raw: bytes) -> dict[str, Any]:
    try:
        return matched_v1.parse_canonical_json(
            raw, name="bound matched-training V4 schedule"
        )
    except (TypeError, ValueError) as error:
        raise ScheduleAdapterIntegrityError(
            "bound schedule is not one canonical self-hashed JSON line"
        ) from error


def _validate_matched_v1_contract() -> None:
    _integrity(
        matched_v1.SCHEDULE_SCHEMA == NORMALIZED_V1_SCHEDULE_SCHEMA,
        "matched-training V1 schedule schema changed",
    )
    _integrity(
        (
            matched_v1.SCHEDULE_SEED,
            matched_v1.TRAIN_PAIR_COUNT,
            matched_v1.PRESENTATION_COUNT,
            matched_v1.UPDATE_COUNT,
            matched_v1.MICROBATCH_SIZE,
            matched_v1.ACCUMULATION_STEPS,
            matched_v1.EFFECTIVE_BATCH_SIZE,
        )
        == (20260713, 4262, 128_000, 8_000, 4, 4, 16),
        "matched-training V1 schedule dimensions changed",
    )


def _validate_phase_a_with_policy(
    *,
    raw: bytes,
    binding: Mapping[str, Any],
    policy: _SchedulePolicy,
) -> ValidatedScheduleV2:
    _validate_matched_v1_contract()
    _integrity(type(raw) is bytes, "schedule payload must be immutable bytes")
    _integrity(
        type(binding) is dict and dict(binding) == policy.binding(),
        "schedule authorization binding changed",
    )
    _integrity(
        len(raw) == policy.byte_count,
        "bound schedule byte count changed",
    )
    _integrity(
        hashlib.sha256(raw).hexdigest() == policy.file_sha256,
        "bound schedule file hash changed",
    )

    schedule = _parse_schedule(raw)
    _integrity(
        set(schedule) == _SCHEDULE_FIELDS,
        "bound schedule fields changed",
    )
    _integrity(
        schedule["content_sha256"] == policy.content_sha256,
        "bound schedule content hash changed",
    )
    _integrity(
        schedule["schema"] == policy.bound_schema,
        "bound schedule owning schema changed",
    )
    _integrity(
        (
            schedule["seed"],
            schedule["train_pair_count"],
            schedule["presentation_count"],
            schedule["update_count"],
            schedule["microbatch_size"],
            schedule["accumulation_steps"],
            schedule["effective_batch_size"],
        )
        == (20260713, 4262, 128_000, 8_000, 4, 4, 16),
        "bound schedule dimensions or seed changed",
    )
    _integrity(
        {
            key: schedule[key]
            for key in policy.schedule_identity()
        }
        == policy.schedule_identity(),
        "bound schedule identity fields changed",
    )

    indices = schedule["presentation_indices"]
    _integrity(
        type(indices) is list,
        "bound schedule indices are not a plain list",
    )
    try:
        normalized_indices = matched_v1.validate_schedule_indices(indices)
    except (TypeError, ValueError) as error:
        raise ScheduleAdapterIntegrityError(
            "bound schedule integer or permutation invariants changed"
        ) from error
    _integrity(
        matched_v1.canonical_json_sha256(list(normalized_indices))
        == schedule["indices_sha256"],
        "bound schedule index identity changed",
    )

    normalized_core = dict(schedule)
    normalized_core.pop("content_sha256")
    normalized_core["schema"] = policy.normalized_schema
    normalized = matched_v1.with_content_sha256(normalized_core)
    _integrity(
        normalized["content_sha256"]
        == policy.normalized_content_sha256,
        "schema-only V4-to-V1 normalized content changed",
    )
    _integrity(
        {
            key: value
            for key, value in schedule.items()
            if key not in {"schema", "content_sha256"}
        }
        == {
            key: value
            for key, value in normalized.items()
            if key not in {"schema", "content_sha256"}
        },
        "V4-to-V1 adapter changed a scientific schedule field",
    )

    for presentations, expected in policy.prefix_items:
        observed = matched_v1.canonical_json_sha256(
            list(normalized_indices[:presentations])
        )
        _integrity(
            observed == expected,
            f"bound schedule prefix changed at {presentations} presentations",
        )
    _integrity(
        tuple(presentations for presentations, _ in policy.prefix_items)
        == (1_600, 6_400, 16_000),
        "bound schedule prefix checkpoints changed",
    )

    normalized_raw = matched_v1.canonical_json_bytes(normalized) + b"\n"
    return ValidatedScheduleV2(
        presentation_indices=tuple(normalized_indices),
        normalized_v1_raw=normalized_raw,
        binding_path=str(binding["path"]),
        binding_file_sha256=str(binding["file_sha256"]),
        binding_content_sha256=str(binding["content_sha256"]),
        binding_byte_count=int(binding["byte_count"]),
    )


def validate_bound_schedule_phase_a(
    *,
    raw: bytes,
    binding: Mapping[str, Any],
) -> ValidatedScheduleV2:
    """Validate already-ledgered V4 bytes before any other runtime input."""

    return _validate_phase_a_with_policy(
        raw=raw,
        binding=binding,
        policy=_FROZEN_POLICY,
    )


def _finalize_with_policy(
    *,
    state: ValidatedScheduleV2,
    ordered_train_pair_ids: Sequence[str],
    policy: _SchedulePolicy,
) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    _validate_matched_v1_contract()
    _integrity(
        type(state) is ValidatedScheduleV2,
        "schedule adapter state type changed",
    )
    _integrity(
        state.original_binding() == policy.binding(),
        "schedule adapter state binding changed",
    )
    _integrity(
        not isinstance(ordered_train_pair_ids, (str, bytes)),
        "ordered train-pair identities are not a sequence",
    )
    pair_ids = list(ordered_train_pair_ids)
    normalized = _parse_schedule(state.normalized_v1_raw)
    _integrity(
        normalized["schema"] == policy.normalized_schema
        and normalized["content_sha256"]
        == policy.normalized_content_sha256,
        "normalized V1 comparison view changed",
    )
    _integrity(
        tuple(normalized["presentation_indices"])
        == state.presentation_indices,
        "schedule adapter state indices changed",
    )

    indices = list(state.presentation_indices)
    try:
        recomputed = matched_v1.with_content_sha256({
            **matched_v1.schedule_core(indices, pair_ids),
            "presentation_indices": indices,
        })
    except (TypeError, ValueError) as error:
        raise ScheduleAdapterIntegrityError(
            "actual ordered train-pair identities are invalid"
        ) from error
    _integrity(
        recomputed == normalized,
        "actual ordered train-pair identity does not match the bound schedule",
    )
    for presentations, expected in policy.prefix_items:
        _integrity(
            matched_v1.canonical_json_sha256(indices[:presentations])
            == expected,
            f"finalized schedule prefix changed at {presentations} presentations",
        )

    returned = list(indices[:USED_PRESENTATIONS])
    _integrity(
        len(returned) == USED_PRESENTATIONS
        and returned == normalized["presentation_indices"][:USED_PRESENTATIONS],
        "returned schedule prefix changed",
    )
    binding = state.original_binding()
    record_core = {
        "schema":
            "lewm_go2_shared_jepa_v5_multires_probe_v2_schedule_adapter_v1",
        "status": "PASS_EXACT_V4_BYTES_V1_SEMANTICS_AND_TRAIN_IDENTITY",
        "bound_schedule": dict(binding),
        "bound_schema": policy.bound_schema,
        "normalized_schema": policy.normalized_schema,
        "normalized_content_sha256": policy.normalized_content_sha256,
        "ordered_train_pair_ids_sha256":
            matched_v1.canonical_json_sha256(pair_ids),
        "indices_sha256": normalized["indices_sha256"],
        "prefix_sha256": {
            str(presentations): digest
            for presentations, digest in policy.prefix_items
        },
        "returned_presentations": USED_PRESENTATIONS,
        "schedule_bytes_rewritten": False,
        "schedule_reopened_or_regenerated": False,
        "indices_mutated_reordered_filtered_or_reseeded": False,
        "phase_a_complete": True,
        "phase_b_train_identity_complete": True,
    }
    record = matched_v1.with_content_sha256(record_core)
    return returned, binding, record


def finalize_train_identity(
    *,
    state: ValidatedScheduleV2,
    ordered_train_pair_ids: Sequence[str],
) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    """Join Phase-A state to actual train identities without schedule I/O."""

    return _finalize_with_policy(
        state=state,
        ordered_train_pair_ids=ordered_train_pair_ids,
        policy=_FROZEN_POLICY,
    )


def require_v1_science_contract_identity(
    science_contract: Mapping[str, Any],
) -> str:
    """Require the exact preregistered V1/V2 science payload identity."""

    _integrity(
        type(science_contract) is dict,
        "science contract must be a plain dict",
    )
    observed = matched_v1.canonical_json_sha256(science_contract)
    _integrity(
        observed == V1_SCIENCE_CONTRACT_SHA256,
        "V1/V2 science contract identity changed",
    )
    return observed


__all__ = [
    "ScheduleAdapterIntegrityError",
    "ValidatedScheduleV2",
    "finalize_train_identity",
    "require_v1_science_contract_identity",
    "validate_bound_schedule_phase_a",
]
