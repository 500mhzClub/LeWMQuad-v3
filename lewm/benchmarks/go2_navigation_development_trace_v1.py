"""Closed source-only schemas for the Go2 navigation V4 controller trace.

The records in this module are controller/broker provenance primitives.  They
perform no filesystem access, know no scene or evaluator identity, and grant no
runtime authority.  Native floating point values are deliberately excluded
from the JSON surface: finite binary64 values use one canonical big-endian hex
encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct


SCHEMA_VERSION = 1
CANONICAL_COLORS = ("red", "yellow", "blue", "green")
AUTHORITY_MODES = frozenset(
    {"synthetic_mock", "development_candidate", "separately_authorized_later"}
)
ACTORS = frozenset(
    {"launcher", "physics_broker", "controller", "observer", "evaluator", "finalizer"}
)
OPEN_PHASES = frozenset(
    {"capture", "controller", "broker", "post_seal_observer", "evaluation", "finalization"}
)
OPEN_DISPOSITIONS = frozenset(
    {
        "allowed",
        "denied_unexpected",
        "denied_duplicate",
        "denied_hash_mismatch",
        "denied_symlink",
        "denied_escape",
        "denied_wrong_phase",
        "denied_wrong_actor",
    }
)
TARGET_OUTCOME_KINDS = frozenset({"positive", "qualified_negative", "abstain"})
DECISION_KINDS = frozenset({"target", "exploration", "fault"})
ACTION_SOURCES = ("target_router", "learned_g4", "stop", "fault")
STALL_STATES = frozenset({"moving", "controller_stalled", "terminal_fault"})
TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "tick_budget_exhausted",
        "terminal_fault",
        "fall_committed",
        "physics_failure",
    }
)

CONTROLLER_EPISODE_BINDING_SCHEMA = "lewm_go2_controller_episode_binding_v1"
RESET_RECEIPT_SCHEMA = "lewm_go2_navigation_reset_receipt_v1"
NAVIGATION_TICK_RECORD_SCHEMA = "lewm_go2_navigation_tick_record_v1"
ACTUAL_OPEN_LEDGER_SCHEMA = "lewm_go2_actual_open_ledger_v1"
ACTUAL_OPEN_LEDGER_ROW_SCHEMA = "lewm_go2_actual_open_ledger_row_v1"
CONTROLLER_TRACE_SCHEMA = "lewm_go2_controller_trace_v1"

PRODUCTION_CONTROLLER_EPISODE_BINDING_V1 = None
PRODUCTION_RESET_RECEIPT_V1 = None
PRODUCTION_NAVIGATION_TRACE_V1 = None

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_BINARY64_RE = re.compile(r"[0-9a-f]{16}\Z")
_MAX_SIGNED_64 = 2**63 - 1


class NavigationTraceV1Error(ValueError):
    """Base error for closed V1 trace records."""


class NavigationTraceV1SchemaError(NavigationTraceV1Error):
    """A record is not an exact closed schema."""


class NavigationTraceV1HashError(NavigationTraceV1Error):
    """A record commitment or chain is invalid."""


def _walk_plain_json(value: object, *, name: str = "value") -> None:
    """Reject non-JSON containers, floats, and mapping subclasses."""

    if value is None or type(value) in {str, bool, int}:
        if type(value) is int and not -_MAX_SIGNED_64 - 1 <= value <= _MAX_SIGNED_64:
            raise NavigationTraceV1SchemaError(f"{name} integer is outside signed-64 range")
        return
    if type(value) is float:
        raise NavigationTraceV1SchemaError(
            f"{name} native float is forbidden; use canonical_binary64_hex"
        )
    if type(value) is list:
        for index, item in enumerate(value):
            _walk_plain_json(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise NavigationTraceV1SchemaError(f"{name} contains a nonstring key")
            _walk_plain_json(item, name=f"{name}.{key}")
        return
    raise NavigationTraceV1SchemaError(
        f"{name} must use exact JSON scalar/list/dict types"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole accepted canonical JSON representation."""

    _walk_plain_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_float(_: str) -> object:
    raise NavigationTraceV1SchemaError(
        "native JSON floats are forbidden; use canonical binary64 hex"
    )


def _reject_constant(_: str) -> object:
    raise NavigationTraceV1SchemaError("nonfinite JSON constants are forbidden")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise NavigationTraceV1SchemaError("JSON object key must be a string")
        if key in result:
            raise NavigationTraceV1SchemaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_json_bytes(raw: bytes) -> object:
    """Parse exact canonical JSON while detecting duplicate keys."""

    if type(raw) is not bytes:
        raise TypeError("raw must be exact bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise NavigationTraceV1SchemaError("canonical JSON must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, NavigationTraceV1SchemaError) as exc:
        if isinstance(exc, NavigationTraceV1SchemaError):
            raise
        raise NavigationTraceV1SchemaError("invalid JSON") from exc
    _walk_plain_json(value)
    if canonical_json_bytes(value) != raw:
        raise NavigationTraceV1SchemaError("JSON bytes are not canonical")
    return value


def canonical_binary64_hex(value: float | int) -> str:
    """Encode one finite number as canonical big-endian IEEE-754 binary64."""

    if type(value) not in {int, float}:
        raise TypeError("binary64 value must be an exact int or float")
    number = float(value)
    if not math.isfinite(number):
        raise NavigationTraceV1SchemaError("binary64 value must be finite")
    if number == 0.0:
        number = 0.0
    return struct.pack(">d", number).hex()


def decode_canonical_binary64_hex(value: str, *, name: str = "binary64") -> float:
    if type(value) is not str or _BINARY64_RE.fullmatch(value) is None:
        raise NavigationTraceV1SchemaError(f"{name} must be 16 lowercase hex digits")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(number):
        raise NavigationTraceV1SchemaError(f"{name} must encode a finite binary64")
    if number == 0.0 and value != "0000000000000000":
        raise NavigationTraceV1SchemaError(f"{name} encodes noncanonical negative zero")
    if canonical_binary64_hex(number) != value:
        raise NavigationTraceV1SchemaError(f"{name} is not canonical")
    return number


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise NavigationTraceV1SchemaError(f"{name} must be a lowercase SHA-256")
    return value


def require_optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_sha256(value, name=name)


def require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise NavigationTraceV1SchemaError(f"{name} must be a canonical identifier")
    return value


def require_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SIGNED_64:
        raise NavigationTraceV1SchemaError(
            f"{name} must be an exact nonnegative signed-64 integer"
        )
    return value


def require_exact_dict(
    value: object, *, name: str, keys: frozenset[str] | set[str]
) -> dict[str, object]:
    if type(value) is not dict:
        raise NavigationTraceV1SchemaError(f"{name} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise NavigationTraceV1SchemaError(f"{name} contains a nonstring key")
    if set(value) != set(keys):
        raise NavigationTraceV1SchemaError(f"{name} keys are not exact")
    return value


def validate_content_commitment(
    value: object,
    *,
    schema: str,
    keys: frozenset[str] | set[str],
) -> dict[str, object]:
    """Validate a closed content-committed mapping without opening anything."""

    record = require_exact_dict(value, name=schema, keys=keys)
    if record["schema"] != schema or type(record["schema"]) is not str:
        raise NavigationTraceV1SchemaError(f"{schema} schema is invalid")
    if type(record["version"]) is not int or record["version"] != SCHEMA_VERSION:
        raise NavigationTraceV1SchemaError(f"{schema} version is invalid")
    claimed = require_sha256(record["content_sha256"], name="content_sha256")
    core = dict(record)
    del core["content_sha256"]
    if canonical_json_sha256(core) != claimed:
        raise NavigationTraceV1HashError(f"{schema} content commitment mismatch")
    return record


def _tuple_of_sha256(value: object, *, name: str, length: int | None = None) -> tuple[str, ...]:
    if type(value) is not list:
        raise NavigationTraceV1SchemaError(f"{name} must be an exact list")
    if length is not None and len(value) != length:
        raise NavigationTraceV1SchemaError(f"{name} length is invalid")
    return tuple(require_sha256(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _tuple_of_ints(value: object, *, name: str, length: int | None = None) -> tuple[int, ...]:
    if type(value) is not list:
        raise NavigationTraceV1SchemaError(f"{name} must be an exact list")
    if length is not None and len(value) != length:
        raise NavigationTraceV1SchemaError(f"{name} length is invalid")
    return tuple(
        require_nonnegative_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


_BINDING_HASH_FIELDS = (
    "shared_v5_checkpoint_file_sha256",
    "shared_v5_model_state_sha256",
    "g2_report_sha256",
    "g2_candidate_publication_sha256",
    "target_head_checkpoint_sha256",
    "target_head_config_sha256",
    "target_head_calibration_sha256",
    "g4_head_checkpoint_sha256",
    "g4_head_config_sha256",
    "physical_calibration_sha256",
    "physical_thresholds_sha256",
    "geometry_profile_sha256",
    "runner_config_sha256",
    "controller_config_sha256",
    "follower_config_sha256",
    "captured_source_graph_sha256",
)


@dataclass(frozen=True)
class ControllerEpisodeBindingV1:
    shared_v5_checkpoint_file_sha256: str
    shared_v5_model_state_sha256: str
    g2_report_sha256: str
    g2_candidate_publication_sha256: str
    target_head_checkpoint_sha256: str
    target_head_config_sha256: str
    target_head_calibration_sha256: str
    g4_head_checkpoint_sha256: str
    g4_head_config_sha256: str
    physical_calibration_sha256: str
    physical_thresholds_sha256: str
    geometry_profile_sha256: str
    runner_config_sha256: str
    controller_config_sha256: str
    follower_config_sha256: str
    captured_source_graph_sha256: str
    semantic_colors: tuple[str, str, str, str]
    tick_budget: int
    execution_seed: int
    reset_id: str
    session_id: str
    authority_mode: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in _BINDING_HASH_FIELDS:
            require_sha256(getattr(self, name), name=name)
        if type(self.semantic_colors) is not tuple or self.semantic_colors != CANONICAL_COLORS:
            raise NavigationTraceV1SchemaError("semantic colors are not canonical")
        if require_nonnegative_int(self.tick_budget, name="tick_budget") == 0:
            raise NavigationTraceV1SchemaError("tick_budget must be positive")
        require_nonnegative_int(self.execution_seed, name="execution_seed")
        require_identifier(self.reset_id, name="reset_id")
        require_identifier(self.session_id, name="session_id")
        if self.reset_id == self.session_id:
            raise NavigationTraceV1SchemaError("reset and session IDs must differ")
        if type(self.authority_mode) is not str or self.authority_mode not in AUTHORITY_MODES:
            raise NavigationTraceV1SchemaError("authority_mode is invalid")
        object.__setattr__(self, "content_sha256", canonical_json_sha256(self._core_dict()))

    def _core_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": CONTROLLER_EPISODE_BINDING_SCHEMA,
            "version": SCHEMA_VERSION,
        }
        result.update({name: getattr(self, name) for name in _BINDING_HASH_FIELDS})
        result.update(
            {
                "semantic_colors": list(self.semantic_colors),
                "tick_budget": self.tick_budget,
                "execution_seed": self.execution_seed,
                "reset_id": self.reset_id,
                "session_id": self.session_id,
                "authority_mode": self.authority_mode,
            }
        )
        return result

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "content_sha256": self.content_sha256}

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ControllerEpisodeBindingV1":
        keys = frozenset(
            {"schema", "version", "content_sha256", "semantic_colors", "tick_budget", "execution_seed", "reset_id", "session_id", "authority_mode", *_BINDING_HASH_FIELDS}
        )
        record = validate_content_commitment(value, schema=CONTROLLER_EPISODE_BINDING_SCHEMA, keys=keys)
        colors = record["semantic_colors"]
        if type(colors) is not list or any(type(item) is not str for item in colors):
            raise NavigationTraceV1SchemaError("semantic_colors must be an exact string list")
        created = cls(
            **{name: record[name] for name in _BINDING_HASH_FIELDS},
            semantic_colors=tuple(colors),
            tick_budget=record["tick_budget"],
            execution_seed=record["execution_seed"],
            reset_id=record["reset_id"],
            session_id=record["session_id"],
            authority_mode=record["authority_mode"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV1HashError("binding reconstruction changed content")
        return created


ZERO_REVISION_NAMES = (
    "physical",
    "configuration",
    "view",
    "target_red",
    "target_yellow",
    "target_blue",
    "target_green",
    "router",
    "follower",
    "integration",
    "action_journal",
    "claim_journal",
    "tick_chain",
)


def zero_owner_revisions() -> tuple[tuple[str, int], ...]:
    return tuple((name, 0) for name in ZERO_REVISION_NAMES)


def _validate_revision_rows(
    value: object, *, name: str, require_zero: bool
) -> tuple[tuple[str, int], ...]:
    if type(value) is not dict:
        raise NavigationTraceV1SchemaError(f"{name} must be an exact dict")
    if set(value) != set(ZERO_REVISION_NAMES):
        raise NavigationTraceV1SchemaError(f"{name} keys are not exact")
    result = tuple(
        (key, require_nonnegative_int(value[key], name=f"{name}.{key}"))
        for key in ZERO_REVISION_NAMES
    )
    if require_zero and any(item != 0 for _, item in result):
        raise NavigationTraceV1SchemaError(f"{name} must contain exact zeros")
    return result


@dataclass(frozen=True)
class ResetReceiptV1:
    binding_sha256: str
    reset_id: str
    session_id: str
    reset_capability_id: str
    physical_memory_owner_id: str
    configuration_projection_owner_id: str
    planner_owner_id: str
    view_owner_id: str
    target_memory_owner_ids: tuple[str, str, str, str]
    router_owner_id: str
    follower_owner_id: str
    integration_owner_id: str
    action_journal_owner_id: str
    claim_journal_owner_id: str
    trace_owner_id: str
    owner_revisions: tuple[tuple[str, int], ...]
    empty_action_journal_sha256: str
    empty_claim_journal_sha256: str
    empty_tick_chain_sha256: str
    reset_clearance_certificate_sha256: str | None
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.binding_sha256, name="binding_sha256")
        id_names = (
            "reset_id",
            "session_id",
            "reset_capability_id",
            "physical_memory_owner_id",
            "configuration_projection_owner_id",
            "planner_owner_id",
            "view_owner_id",
            "router_owner_id",
            "follower_owner_id",
            "integration_owner_id",
            "action_journal_owner_id",
            "claim_journal_owner_id",
            "trace_owner_id",
        )
        identifiers = [require_identifier(getattr(self, name), name=name) for name in id_names]
        if type(self.target_memory_owner_ids) is not tuple or len(self.target_memory_owner_ids) != 4:
            raise NavigationTraceV1SchemaError("target_memory_owner_ids must be a four-tuple")
        identifiers.extend(
            require_identifier(item, name=f"target_memory_owner_ids[{index}]")
            for index, item in enumerate(self.target_memory_owner_ids)
        )
        if len(identifiers) != len(set(identifiers)):
            raise NavigationTraceV1SchemaError("reset/session/owner IDs must all be fresh and unique")
        if type(self.owner_revisions) is not tuple or self.owner_revisions != zero_owner_revisions():
            raise NavigationTraceV1SchemaError("owner_revisions must be canonical exact zeros")
        for name in (
            "empty_action_journal_sha256",
            "empty_claim_journal_sha256",
            "empty_tick_chain_sha256",
        ):
            require_sha256(getattr(self, name), name=name)
        require_optional_sha256(
            self.reset_clearance_certificate_sha256,
            name="reset_clearance_certificate_sha256",
        )
        object.__setattr__(self, "content_sha256", canonical_json_sha256(self._core_dict()))

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": RESET_RECEIPT_SCHEMA,
            "version": SCHEMA_VERSION,
            "binding_sha256": self.binding_sha256,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "reset_capability_id": self.reset_capability_id,
            "physical_memory_owner_id": self.physical_memory_owner_id,
            "configuration_projection_owner_id": self.configuration_projection_owner_id,
            "planner_owner_id": self.planner_owner_id,
            "view_owner_id": self.view_owner_id,
            "target_memory_owner_ids": list(self.target_memory_owner_ids),
            "router_owner_id": self.router_owner_id,
            "follower_owner_id": self.follower_owner_id,
            "integration_owner_id": self.integration_owner_id,
            "action_journal_owner_id": self.action_journal_owner_id,
            "claim_journal_owner_id": self.claim_journal_owner_id,
            "trace_owner_id": self.trace_owner_id,
            "owner_revisions": dict(self.owner_revisions),
            "empty_action_journal_sha256": self.empty_action_journal_sha256,
            "empty_claim_journal_sha256": self.empty_claim_journal_sha256,
            "empty_tick_chain_sha256": self.empty_tick_chain_sha256,
            "reset_clearance_certificate_sha256": self.reset_clearance_certificate_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "content_sha256": self.content_sha256}

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ResetReceiptV1":
        core_keys = {
            "binding_sha256", "reset_id", "session_id", "reset_capability_id",
            "physical_memory_owner_id", "configuration_projection_owner_id", "planner_owner_id",
            "view_owner_id", "target_memory_owner_ids", "router_owner_id", "follower_owner_id",
            "integration_owner_id", "action_journal_owner_id", "claim_journal_owner_id", "trace_owner_id",
            "owner_revisions", "empty_action_journal_sha256", "empty_claim_journal_sha256",
            "empty_tick_chain_sha256", "reset_clearance_certificate_sha256",
        }
        record = validate_content_commitment(
            value,
            schema=RESET_RECEIPT_SCHEMA,
            keys=frozenset({"schema", "version", "content_sha256", *core_keys}),
        )
        target_ids = record["target_memory_owner_ids"]
        if type(target_ids) is not list or any(type(item) is not str for item in target_ids):
            raise NavigationTraceV1SchemaError("target_memory_owner_ids must be an exact string list")
        revisions = _validate_revision_rows(record["owner_revisions"], name="owner_revisions", require_zero=True)
        created = cls(
            **{name: record[name] for name in core_keys - {"target_memory_owner_ids", "owner_revisions"}},
            target_memory_owner_ids=tuple(target_ids),
            owner_revisions=revisions,
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV1HashError("reset receipt reconstruction changed content")
        return created


@dataclass(frozen=True)
class CallCounterPanelV1:
    observation_tick_count: int
    shared_frame_outcome_count: int
    shared_v5_forward_frame_call_count: int
    vision_encoder_forward_tokens_call_count: int
    target_four_color_batch_count: int
    g4_value_head_call_count: int
    rgb_decode_call_count: int
    rgb_preprocess_call_count: int
    extra_rgb_decode_or_preprocess_count: int

    def __post_init__(self) -> None:
        for name in self.names():
            require_nonnegative_int(getattr(self, name), name=name)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return (
            "observation_tick_count",
            "shared_frame_outcome_count",
            "shared_v5_forward_frame_call_count",
            "vision_encoder_forward_tokens_call_count",
            "target_four_color_batch_count",
            "g4_value_head_call_count",
            "rgb_decode_call_count",
            "rgb_preprocess_call_count",
            "extra_rgb_decode_or_preprocess_count",
        )

    @classmethod
    def zero(cls) -> "CallCounterPanelV1":
        return cls(*(0 for _ in cls.names()))

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.names()}

    @classmethod
    def from_dict(cls, value: object) -> "CallCounterPanelV1":
        record = require_exact_dict(value, name="call counter panel", keys=set(cls.names()))
        return cls(*(record[name] for name in cls.names()))

    def plus(self, other: "CallCounterPanelV1") -> "CallCounterPanelV1":
        if type(other) is not CallCounterPanelV1:
            raise TypeError("other must be exact CallCounterPanelV1")
        return CallCounterPanelV1(
            *(getattr(self, name) + getattr(other, name) for name in self.names())
        )

    def assert_one_encode_invariants(self, *, per_tick: bool = False) -> None:
        observations = self.observation_tick_count
        equal_names = (
            "shared_frame_outcome_count",
            "shared_v5_forward_frame_call_count",
            "vision_encoder_forward_tokens_call_count",
            "target_four_color_batch_count",
            "rgb_decode_call_count",
            "rgb_preprocess_call_count",
        )
        if any(getattr(self, name) != observations for name in equal_names):
            raise NavigationTraceV1SchemaError("one-encode counter equality failed")
        if self.g4_value_head_call_count > observations:
            raise NavigationTraceV1SchemaError("G4 head count exceeds observations")
        if self.extra_rgb_decode_or_preprocess_count != 0:
            raise NavigationTraceV1SchemaError("extra RGB decode/preprocess count is nonzero")
        if per_tick and observations != 1:
            raise NavigationTraceV1SchemaError("an admitted tick must contain exactly one observation")


_TICK_HASH_FIELDS = (
    "controller_input_sha256", "inference_receipt_sha256",
    "pre_physical_content_sha256", "post_physical_content_sha256",
    "physical_transaction_sha256", "physical_retraction_sha256",
    "pre_configuration_content_sha256", "post_configuration_content_sha256",
    "configuration_snapshot_sha256", "configuration_component_sha256",
    "frontier_sha256", "tick_admission_receipt_sha256", "view_admission_sha256",
    "scheduler_rows_sha256", "waypoint_receipt_sha256", "follower_receipt_sha256",
    "requested_command_block_sha256", "executed_command_block_sha256",
    "platform_envelope_clipping_sha256", "broker_execution_sha256", "broker_fall_sha256",
)


@dataclass(frozen=True)
class NavigationTickRecordV1:
    tick_index: int
    timestamp_binary64_hex: str
    synchronization_id: str
    reset_id: str
    session_id: str
    controller_input_sha256: str
    inference_receipt_sha256: str
    per_tick_counts: CallCounterPanelV1
    cumulative_counts: CallCounterPanelV1
    pre_physical_revision: int
    post_physical_revision: int
    pre_physical_content_sha256: str
    post_physical_content_sha256: str
    physical_transaction_sha256: str
    physical_retraction_sha256: str
    pre_configuration_revision: int
    post_configuration_revision: int
    pre_configuration_content_sha256: str
    post_configuration_content_sha256: str
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    frontier_sha256: str
    tick_admission_receipt_sha256: str
    view_admission_sha256: str
    pre_view_revision: int
    post_view_revision: int
    target_outcome_kinds: tuple[str, str, str, str]
    target_outcome_receipt_sha256s: tuple[str, str, str, str]
    pre_target_revisions: tuple[int, int, int, int]
    post_target_revisions: tuple[int, int, int, int]
    posterior_sha256s: tuple[str, str, str, str]
    posterior_component_sha256s: tuple[str, str, str, str]
    posterior_ages: tuple[int, int, int, int]
    locked_color: str | None
    scheduler_rows_sha256: str
    decision_kind: str
    target_route_sha256: str | None
    g4_candidate_set_sha256: str | None
    baseline_scores_sha256: str | None
    learned_scores_sha256: str | None
    selected_row: int | None
    selected_path_sha256: str | None
    terminal_yaw_binary64_hex: str | None
    waypoint_receipt_sha256: str
    follower_receipt_sha256: str
    requested_command_block_sha256: str
    executed_command_block_sha256: str
    platform_envelope_clipping_sha256: str
    action_source: str
    claim_intent_sha256: str | None
    pre_claim_journal_revision: int
    post_claim_journal_revision: int
    controller_fault_code: str | None
    stall_state: str
    broker_execution_sha256: str
    broker_fall_sha256: str
    previous_tick_chain_sha256: str
    content_sha256: str = field(init=False)
    chain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_nonnegative_int(self.tick_index, name="tick_index")
        decode_canonical_binary64_hex(self.timestamp_binary64_hex, name="timestamp_binary64_hex")
        for name in ("synchronization_id", "reset_id", "session_id"):
            require_identifier(getattr(self, name), name=name)
        for name in _TICK_HASH_FIELDS:
            require_sha256(getattr(self, name), name=name)
        if type(self.per_tick_counts) is not CallCounterPanelV1 or type(self.cumulative_counts) is not CallCounterPanelV1:
            raise NavigationTraceV1SchemaError("counter panels must be exact CallCounterPanelV1")
        self.per_tick_counts.assert_one_encode_invariants(per_tick=True)
        self.cumulative_counts.assert_one_encode_invariants()
        for prefix in ("physical", "configuration", "view"):
            pre = require_nonnegative_int(getattr(self, f"pre_{prefix}_revision"), name=f"pre_{prefix}_revision")
            post = require_nonnegative_int(getattr(self, f"post_{prefix}_revision"), name=f"post_{prefix}_revision")
            if post != pre + 1:
                raise NavigationTraceV1SchemaError(f"{prefix} revision must advance exactly once")
        if type(self.target_outcome_kinds) is not tuple or len(self.target_outcome_kinds) != 4:
            raise NavigationTraceV1SchemaError("target outcomes must be a four-tuple")
        if any(type(item) is not str or item not in TARGET_OUTCOME_KINDS for item in self.target_outcome_kinds):
            raise NavigationTraceV1SchemaError("target outcome kind is invalid")
        for name in ("target_outcome_receipt_sha256s", "posterior_sha256s", "posterior_component_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 4:
                raise NavigationTraceV1SchemaError(f"{name} must be a four-tuple")
            for index, item in enumerate(values):
                require_sha256(item, name=f"{name}[{index}]")
        for name in ("pre_target_revisions", "post_target_revisions", "posterior_ages"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 4:
                raise NavigationTraceV1SchemaError(f"{name} must be a four-tuple")
            for index, item in enumerate(values):
                require_nonnegative_int(item, name=f"{name}[{index}]")
        if any(post != pre + 1 for pre, post in zip(self.pre_target_revisions, self.post_target_revisions)):
            raise NavigationTraceV1SchemaError("each target posterior must advance exactly once")
        if self.locked_color is not None and self.locked_color not in CANONICAL_COLORS:
            raise NavigationTraceV1SchemaError("locked_color is invalid")
        if type(self.decision_kind) is not str or self.decision_kind not in DECISION_KINDS:
            raise NavigationTraceV1SchemaError("decision_kind is invalid")
        for name in (
            "target_route_sha256", "g4_candidate_set_sha256", "baseline_scores_sha256",
            "learned_scores_sha256", "selected_path_sha256", "claim_intent_sha256",
        ):
            require_optional_sha256(getattr(self, name), name=name)
        if self.selected_row is not None:
            require_nonnegative_int(self.selected_row, name="selected_row")
        if self.terminal_yaw_binary64_hex is not None:
            decode_canonical_binary64_hex(self.terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex")
        if type(self.action_source) is not str or self.action_source not in ACTION_SOURCES:
            raise NavigationTraceV1SchemaError("action_source is invalid")
        pre_claim = require_nonnegative_int(self.pre_claim_journal_revision, name="pre_claim_journal_revision")
        post_claim = require_nonnegative_int(self.post_claim_journal_revision, name="post_claim_journal_revision")
        if post_claim != pre_claim + (1 if self.claim_intent_sha256 is not None else 0):
            raise NavigationTraceV1SchemaError("claim journal transition does not match claim intent")
        if self.controller_fault_code is not None:
            require_identifier(self.controller_fault_code, name="controller_fault_code")
        if type(self.stall_state) is not str or self.stall_state not in STALL_STATES:
            raise NavigationTraceV1SchemaError("stall_state is invalid")
        require_sha256(self.previous_tick_chain_sha256, name="previous_tick_chain_sha256")
        self._validate_decision_branch()
        content = canonical_json_sha256(self._core_dict())
        chain = canonical_json_sha256(
            {
                "schema": "lewm_go2_navigation_tick_chain_link_v1",
                "version": SCHEMA_VERSION,
                "previous_tick_chain_sha256": self.previous_tick_chain_sha256,
                "tick_content_sha256": content,
            }
        )
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "chain_sha256", chain)

    def _validate_decision_branch(self) -> None:
        g4_fields = (self.g4_candidate_set_sha256, self.baseline_scores_sha256, self.learned_scores_sha256)
        selected_fields = (self.selected_row, self.selected_path_sha256, self.terminal_yaw_binary64_hex)
        if self.decision_kind == "target":
            if self.target_route_sha256 is None or any(item is not None for item in g4_fields):
                raise NavigationTraceV1SchemaError("target decision branch bindings are invalid")
            if self.selected_path_sha256 is None or self.terminal_yaw_binary64_hex is None:
                raise NavigationTraceV1SchemaError("target decision lacks path or yaw")
            if self.per_tick_counts.g4_value_head_call_count != 0:
                raise NavigationTraceV1SchemaError("target decision called G4 head")
        elif self.decision_kind == "exploration":
            if self.target_route_sha256 is not None or any(item is None for item in g4_fields + selected_fields):
                raise NavigationTraceV1SchemaError("exploration decision branch bindings are invalid")
            if self.per_tick_counts.g4_value_head_call_count != 1:
                raise NavigationTraceV1SchemaError("exploration decision must call G4 head exactly once")
        elif self.action_source != "fault" or self.controller_fault_code is None:
            raise NavigationTraceV1SchemaError("fault branch lacks terminal fault source/code")

    def _core_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": NAVIGATION_TICK_RECORD_SCHEMA,
            "version": SCHEMA_VERSION,
            "tick_index": self.tick_index,
            "timestamp_binary64_hex": self.timestamp_binary64_hex,
            "synchronization_id": self.synchronization_id,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "per_tick_counts": self.per_tick_counts.to_dict(),
            "cumulative_counts": self.cumulative_counts.to_dict(),
            "pre_physical_revision": self.pre_physical_revision,
            "post_physical_revision": self.post_physical_revision,
            "pre_configuration_revision": self.pre_configuration_revision,
            "post_configuration_revision": self.post_configuration_revision,
            "pre_view_revision": self.pre_view_revision,
            "post_view_revision": self.post_view_revision,
            "target_outcome_kinds": list(self.target_outcome_kinds),
            "target_outcome_receipt_sha256s": list(self.target_outcome_receipt_sha256s),
            "pre_target_revisions": list(self.pre_target_revisions),
            "post_target_revisions": list(self.post_target_revisions),
            "posterior_sha256s": list(self.posterior_sha256s),
            "posterior_component_sha256s": list(self.posterior_component_sha256s),
            "posterior_ages": list(self.posterior_ages),
            "locked_color": self.locked_color,
            "decision_kind": self.decision_kind,
            "target_route_sha256": self.target_route_sha256,
            "g4_candidate_set_sha256": self.g4_candidate_set_sha256,
            "baseline_scores_sha256": self.baseline_scores_sha256,
            "learned_scores_sha256": self.learned_scores_sha256,
            "selected_row": self.selected_row,
            "selected_path_sha256": self.selected_path_sha256,
            "terminal_yaw_binary64_hex": self.terminal_yaw_binary64_hex,
            "action_source": self.action_source,
            "claim_intent_sha256": self.claim_intent_sha256,
            "pre_claim_journal_revision": self.pre_claim_journal_revision,
            "post_claim_journal_revision": self.post_claim_journal_revision,
            "controller_fault_code": self.controller_fault_code,
            "stall_state": self.stall_state,
            "previous_tick_chain_sha256": self.previous_tick_chain_sha256,
        }
        result.update({name: getattr(self, name) for name in _TICK_HASH_FIELDS})
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            **self._core_dict(),
            "content_sha256": self.content_sha256,
            "chain_sha256": self.chain_sha256,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "NavigationTickRecordV1":
        probe = require_exact_dict(value, name=NAVIGATION_TICK_RECORD_SCHEMA, keys=set(cls._serialized_keys()))
        if probe["schema"] != NAVIGATION_TICK_RECORD_SCHEMA or probe["version"] != SCHEMA_VERSION:
            raise NavigationTraceV1SchemaError("tick schema/version is invalid")
        claimed_content = require_sha256(probe["content_sha256"], name="content_sha256")
        claimed_chain = require_sha256(probe["chain_sha256"], name="chain_sha256")
        core = dict(probe)
        del core["content_sha256"]
        del core["chain_sha256"]
        if canonical_json_sha256(core) != claimed_content:
            raise NavigationTraceV1HashError("tick content commitment mismatch")
        kwargs = {key: probe[key] for key in cls._constructor_scalar_keys()}
        kwargs.update(
            {
                "per_tick_counts": CallCounterPanelV1.from_dict(probe["per_tick_counts"]),
                "cumulative_counts": CallCounterPanelV1.from_dict(probe["cumulative_counts"]),
                "target_outcome_kinds": cls._string_tuple(probe["target_outcome_kinds"], "target_outcome_kinds"),
                "target_outcome_receipt_sha256s": _tuple_of_sha256(probe["target_outcome_receipt_sha256s"], name="target_outcome_receipt_sha256s", length=4),
                "pre_target_revisions": _tuple_of_ints(probe["pre_target_revisions"], name="pre_target_revisions", length=4),
                "post_target_revisions": _tuple_of_ints(probe["post_target_revisions"], name="post_target_revisions", length=4),
                "posterior_sha256s": _tuple_of_sha256(probe["posterior_sha256s"], name="posterior_sha256s", length=4),
                "posterior_component_sha256s": _tuple_of_sha256(probe["posterior_component_sha256s"], name="posterior_component_sha256s", length=4),
                "posterior_ages": _tuple_of_ints(probe["posterior_ages"], name="posterior_ages", length=4),
            }
        )
        created = cls(**kwargs)
        if created.content_sha256 != claimed_content or created.chain_sha256 != claimed_chain:
            raise NavigationTraceV1HashError("tick reconstruction changed commitment")
        return created

    @staticmethod
    def _string_tuple(value: object, name: str) -> tuple[str, str, str, str]:
        if type(value) is not list or len(value) != 4 or any(type(item) is not str for item in value):
            raise NavigationTraceV1SchemaError(f"{name} must be an exact four-string list")
        return tuple(value)  # type: ignore[return-value]

    @classmethod
    def _constructor_scalar_keys(cls) -> tuple[str, ...]:
        excluded = {
            "per_tick_counts", "cumulative_counts", "target_outcome_kinds",
            "target_outcome_receipt_sha256s", "pre_target_revisions", "post_target_revisions",
            "posterior_sha256s", "posterior_component_sha256s", "posterior_ages",
            "content_sha256", "chain_sha256", "schema", "version",
        }
        return tuple(name for name in cls.__dataclass_fields__ if name not in excluded)

    @classmethod
    def _serialized_keys(cls) -> tuple[str, ...]:
        return (
            "schema", "version", *cls._constructor_scalar_keys(),
            "per_tick_counts", "cumulative_counts", "target_outcome_kinds",
            "target_outcome_receipt_sha256s", "pre_target_revisions", "post_target_revisions",
            "posterior_sha256s", "posterior_component_sha256s", "posterior_ages",
            "content_sha256", "chain_sha256",
        )


EMPTY_TICK_CHAIN_SHA256 = canonical_json_sha256(
    {"schema": "lewm_go2_navigation_tick_chain_empty_v1", "version": SCHEMA_VERSION}
)
EMPTY_OPEN_LEDGER_SHA256 = canonical_json_sha256(
    {"schema": "lewm_go2_actual_open_ledger_empty_v1", "version": SCHEMA_VERSION}
)


def _canonical_no_follow_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise NavigationTraceV1SchemaError("open ledger path is invalid")
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or any(part in {".", ".."} for part in path.parts):
        raise NavigationTraceV1SchemaError("open ledger path must be canonical and absolute")
    return value


@dataclass(frozen=True)
class ActualOpenLedgerRowV1:
    actor: str
    phase: str
    role: str
    no_follow_canonical_path: str
    expected_sha256: str
    actual_sha256: str
    sequence: int
    previous_row_sha256: str
    access_disposition: str
    row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.actor) is not str or self.actor not in ACTORS:
            raise NavigationTraceV1SchemaError("open actor is invalid")
        if type(self.phase) is not str or self.phase not in OPEN_PHASES:
            raise NavigationTraceV1SchemaError("open phase is invalid")
        require_identifier(self.role, name="role")
        _canonical_no_follow_path(self.no_follow_canonical_path)
        require_sha256(self.expected_sha256, name="expected_sha256")
        require_sha256(self.actual_sha256, name="actual_sha256")
        require_nonnegative_int(self.sequence, name="sequence")
        require_sha256(self.previous_row_sha256, name="previous_row_sha256")
        if type(self.access_disposition) is not str or self.access_disposition not in OPEN_DISPOSITIONS:
            raise NavigationTraceV1SchemaError("access_disposition is invalid")
        if self.access_disposition == "allowed" and self.expected_sha256 != self.actual_sha256:
            raise NavigationTraceV1SchemaError("allowed open has a hash mismatch")
        object.__setattr__(self, "row_sha256", canonical_json_sha256(self._core_dict()))

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": ACTUAL_OPEN_LEDGER_ROW_SCHEMA,
            "version": SCHEMA_VERSION,
            "actor": self.actor,
            "phase": self.phase,
            "role": self.role,
            "no_follow_canonical_path": self.no_follow_canonical_path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "sequence": self.sequence,
            "previous_row_sha256": self.previous_row_sha256,
            "access_disposition": self.access_disposition,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "row_sha256": self.row_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ActualOpenLedgerRowV1":
        keys = {
            "schema", "version", "actor", "phase", "role", "no_follow_canonical_path",
            "expected_sha256", "actual_sha256", "sequence", "previous_row_sha256",
            "access_disposition", "row_sha256",
        }
        record = require_exact_dict(value, name=ACTUAL_OPEN_LEDGER_ROW_SCHEMA, keys=keys)
        if record["schema"] != ACTUAL_OPEN_LEDGER_ROW_SCHEMA or record["version"] != SCHEMA_VERSION:
            raise NavigationTraceV1SchemaError("open row schema/version is invalid")
        claimed = require_sha256(record["row_sha256"], name="row_sha256")
        created = cls(
            **{name: record[name] for name in keys - {"schema", "version", "row_sha256"}}
        )
        if created.row_sha256 != claimed:
            raise NavigationTraceV1HashError("open row hash mismatch")
        return created


@dataclass(frozen=True)
class ActualOpenLedgerV1:
    rows: tuple[ActualOpenLedgerRowV1, ...] = ()
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or any(type(row) is not ActualOpenLedgerRowV1 for row in self.rows):
            raise NavigationTraceV1SchemaError("open ledger rows must be an exact tuple")
        previous = EMPTY_OPEN_LEDGER_SHA256
        seen: set[tuple[str, str, str, str]] = set()
        for sequence, row in enumerate(self.rows):
            if row.sequence != sequence or row.previous_row_sha256 != previous:
                raise NavigationTraceV1SchemaError("open ledger chain is not contiguous")
            key = (row.actor, row.phase, row.role, row.no_follow_canonical_path)
            if key in seen:
                raise NavigationTraceV1SchemaError("duplicate open ledger row")
            seen.add(key)
            previous = row.row_sha256
        object.__setattr__(self, "content_sha256", canonical_json_sha256(self._core_dict()))

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": ACTUAL_OPEN_LEDGER_SCHEMA,
            "version": SCHEMA_VERSION,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "content_sha256": self.content_sha256}

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def append(
        self,
        *,
        actor: str,
        phase: str,
        role: str,
        no_follow_canonical_path: str,
        expected_sha256: str,
        actual_sha256: str,
        access_disposition: str,
    ) -> "ActualOpenLedgerV1":
        previous = self.rows[-1].row_sha256 if self.rows else EMPTY_OPEN_LEDGER_SHA256
        row = ActualOpenLedgerRowV1(
            actor=actor,
            phase=phase,
            role=role,
            no_follow_canonical_path=no_follow_canonical_path,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            sequence=len(self.rows),
            previous_row_sha256=previous,
            access_disposition=access_disposition,
        )
        return ActualOpenLedgerV1((*self.rows, row))

    def controller_projection(self) -> "ActualOpenLedgerV1":
        projected = ActualOpenLedgerV1()
        for row in self.rows:
            if row.actor == "controller":
                projected = projected.append(
                    actor=row.actor,
                    phase=row.phase,
                    role=row.role,
                    no_follow_canonical_path=row.no_follow_canonical_path,
                    expected_sha256=row.expected_sha256,
                    actual_sha256=row.actual_sha256,
                    access_disposition=row.access_disposition,
                )
        return projected

    @classmethod
    def from_dict(cls, value: object) -> "ActualOpenLedgerV1":
        record = validate_content_commitment(
            value,
            schema=ACTUAL_OPEN_LEDGER_SCHEMA,
            keys={"schema", "version", "rows", "content_sha256"},
        )
        rows = record["rows"]
        if type(rows) is not list:
            raise NavigationTraceV1SchemaError("open ledger rows must be an exact list")
        created = cls(tuple(ActualOpenLedgerRowV1.from_dict(row) for row in rows))
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV1HashError("open ledger reconstruction changed content")
        return created


def _validate_named_counts(
    value: object, *, names: tuple[str, ...], name: str
) -> tuple[tuple[str, int], ...]:
    if type(value) is not dict or set(value) != set(names):
        raise NavigationTraceV1SchemaError(f"{name} must have exact canonical keys")
    return tuple(
        (key, require_nonnegative_int(value[key], name=f"{name}.{key}"))
        for key in names
    )


@dataclass(frozen=True)
class ControllerTraceV1:
    episode_binding: ControllerEpisodeBindingV1
    reset_receipt: ResetReceiptV1
    ticks: tuple[NavigationTickRecordV1, ...]
    semantic_claim_intent_sha256s: tuple[str, ...]
    action_source_counts: tuple[tuple[str, int], ...]
    final_owner_revisions: tuple[tuple[str, int], ...]
    terminal_status: str
    inference_counts: CallCounterPanelV1
    evaluator_access_count: int
    evaluator_callback_count: int
    actual_open_controller_projection: ActualOpenLedgerV1
    final_tick_chain_sha256: str = field(init=False)
    content_sha256: str = field(init=False)
    chain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.episode_binding) is not ControllerEpisodeBindingV1:
            raise NavigationTraceV1SchemaError("episode_binding type is invalid")
        if type(self.reset_receipt) is not ResetReceiptV1:
            raise NavigationTraceV1SchemaError("reset_receipt type is invalid")
        if self.reset_receipt.binding_sha256 != self.episode_binding.content_sha256:
            raise NavigationTraceV1SchemaError("reset receipt does not bind episode")
        if (
            self.reset_receipt.reset_id != self.episode_binding.reset_id
            or self.reset_receipt.session_id != self.episode_binding.session_id
        ):
            raise NavigationTraceV1SchemaError("reset/session differ from episode binding")
        if type(self.ticks) is not tuple or any(type(tick) is not NavigationTickRecordV1 for tick in self.ticks):
            raise NavigationTraceV1SchemaError("ticks must be an exact tuple")
        if len(self.ticks) > self.episode_binding.tick_budget:
            raise NavigationTraceV1SchemaError("tick count exceeds bound budget")
        previous = self.reset_receipt.empty_tick_chain_sha256
        cumulative = CallCounterPanelV1.zero()
        claims: list[str] = []
        source_counts = {name: 0 for name in ACTION_SOURCES}
        for index, tick in enumerate(self.ticks):
            if tick.tick_index != index or tick.previous_tick_chain_sha256 != previous:
                raise NavigationTraceV1SchemaError("tick chain is not contiguous")
            if tick.reset_id != self.reset_receipt.reset_id or tick.session_id != self.reset_receipt.session_id:
                raise NavigationTraceV1SchemaError("tick reset/session binding changed")
            cumulative = cumulative.plus(tick.per_tick_counts)
            if tick.cumulative_counts != cumulative:
                raise NavigationTraceV1SchemaError("tick cumulative counters are invalid")
            if tick.claim_intent_sha256 is not None:
                claims.append(tick.claim_intent_sha256)
            source_counts[tick.action_source] += 1
            previous = tick.chain_sha256
        if type(self.semantic_claim_intent_sha256s) is not tuple:
            raise NavigationTraceV1SchemaError("semantic claim intents must be an exact tuple")
        for index, item in enumerate(self.semantic_claim_intent_sha256s):
            require_sha256(item, name=f"semantic_claim_intent_sha256s[{index}]")
        if len(self.semantic_claim_intent_sha256s) > 4 or len(set(self.semantic_claim_intent_sha256s)) != len(self.semantic_claim_intent_sha256s):
            raise NavigationTraceV1SchemaError("semantic claim intents are duplicated or exceed four")
        if tuple(claims) != self.semantic_claim_intent_sha256s:
            raise NavigationTraceV1SchemaError("semantic claim intent projection is invalid")
        if type(self.action_source_counts) is not tuple or self.action_source_counts != tuple(source_counts.items()):
            raise NavigationTraceV1SchemaError("action source counts are invalid")
        if type(self.final_owner_revisions) is not tuple:
            raise NavigationTraceV1SchemaError("final owner revisions must be an exact tuple")
        if tuple(name for name, _ in self.final_owner_revisions) != ZERO_REVISION_NAMES:
            raise NavigationTraceV1SchemaError("final owner revision names/order are invalid")
        for name, value in self.final_owner_revisions:
            require_nonnegative_int(value, name=f"final_owner_revisions.{name}")
        if type(self.terminal_status) is not str or self.terminal_status not in TERMINAL_STATUSES:
            raise NavigationTraceV1SchemaError("terminal_status is invalid")
        if type(self.inference_counts) is not CallCounterPanelV1 or self.inference_counts != cumulative:
            raise NavigationTraceV1SchemaError("final inference counts are invalid")
        self.inference_counts.assert_one_encode_invariants()
        if require_nonnegative_int(self.evaluator_access_count, name="evaluator_access_count") != 0:
            raise NavigationTraceV1SchemaError("controller evaluator access must be zero")
        if require_nonnegative_int(self.evaluator_callback_count, name="evaluator_callback_count") != 0:
            raise NavigationTraceV1SchemaError("controller evaluator callbacks must be zero")
        if type(self.actual_open_controller_projection) is not ActualOpenLedgerV1:
            raise NavigationTraceV1SchemaError("open ledger projection type is invalid")
        if any(row.actor != "controller" for row in self.actual_open_controller_projection.rows):
            raise NavigationTraceV1SchemaError("controller ledger projection exposes another actor")
        object.__setattr__(self, "final_tick_chain_sha256", previous)
        content = canonical_json_sha256(self._core_dict())
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(
            self,
            "chain_sha256",
            canonical_json_sha256(
                {
                    "schema": "lewm_go2_controller_trace_chain_v1",
                    "version": SCHEMA_VERSION,
                    "controller_trace_content_sha256": content,
                    "final_tick_chain_sha256": previous,
                }
            ),
        )

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": CONTROLLER_TRACE_SCHEMA,
            "version": SCHEMA_VERSION,
            "episode_binding": self.episode_binding.to_dict(),
            "reset_receipt": self.reset_receipt.to_dict(),
            "ticks": [tick.to_dict() for tick in self.ticks],
            "semantic_claim_intent_sha256s": list(self.semantic_claim_intent_sha256s),
            "action_source_counts": dict(self.action_source_counts),
            "final_owner_revisions": dict(self.final_owner_revisions),
            "terminal_status": self.terminal_status,
            "inference_counts": self.inference_counts.to_dict(),
            "evaluator_access_count": self.evaluator_access_count,
            "evaluator_callback_count": self.evaluator_callback_count,
            "actual_open_controller_projection": self.actual_open_controller_projection.to_dict(),
            "final_tick_chain_sha256": self.final_tick_chain_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._core_dict(),
            "content_sha256": self.content_sha256,
            "chain_sha256": self.chain_sha256,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ControllerTraceV1":
        keys = {
            "schema", "version", "episode_binding", "reset_receipt", "ticks",
            "semantic_claim_intent_sha256s", "action_source_counts", "final_owner_revisions",
            "terminal_status", "inference_counts", "evaluator_access_count", "evaluator_callback_count",
            "actual_open_controller_projection", "final_tick_chain_sha256", "content_sha256", "chain_sha256",
        }
        record = require_exact_dict(value, name=CONTROLLER_TRACE_SCHEMA, keys=keys)
        if record["schema"] != CONTROLLER_TRACE_SCHEMA or record["version"] != SCHEMA_VERSION:
            raise NavigationTraceV1SchemaError("controller trace schema/version is invalid")
        claimed_content = require_sha256(record["content_sha256"], name="content_sha256")
        claimed_chain = require_sha256(record["chain_sha256"], name="chain_sha256")
        core = dict(record)
        del core["content_sha256"]
        del core["chain_sha256"]
        if canonical_json_sha256(core) != claimed_content:
            raise NavigationTraceV1HashError("controller trace content mismatch")
        ticks = record["ticks"]
        claims = record["semantic_claim_intent_sha256s"]
        if type(ticks) is not list or type(claims) is not list:
            raise NavigationTraceV1SchemaError("trace ticks/claims must be exact lists")
        created = cls(
            episode_binding=ControllerEpisodeBindingV1.from_dict(record["episode_binding"]),
            reset_receipt=ResetReceiptV1.from_dict(record["reset_receipt"]),
            ticks=tuple(NavigationTickRecordV1.from_dict(item) for item in ticks),
            semantic_claim_intent_sha256s=tuple(
                require_sha256(item, name=f"semantic_claim_intent_sha256s[{index}]")
                for index, item in enumerate(claims)
            ),
            action_source_counts=_validate_named_counts(record["action_source_counts"], names=ACTION_SOURCES, name="action_source_counts"),
            final_owner_revisions=_validate_revision_rows(record["final_owner_revisions"], name="final_owner_revisions", require_zero=False),
            terminal_status=record["terminal_status"],
            inference_counts=CallCounterPanelV1.from_dict(record["inference_counts"]),
            evaluator_access_count=record["evaluator_access_count"],
            evaluator_callback_count=record["evaluator_callback_count"],
            actual_open_controller_projection=ActualOpenLedgerV1.from_dict(record["actual_open_controller_projection"]),
        )
        if (
            created.content_sha256 != claimed_content
            or created.chain_sha256 != claimed_chain
            or created.final_tick_chain_sha256 != record["final_tick_chain_sha256"]
        ):
            raise NavigationTraceV1HashError("controller trace reconstruction changed commitment")
        return created


def parse_controller_trace_v1(raw: bytes) -> ControllerTraceV1:
    return ControllerTraceV1.from_dict(parse_canonical_json_bytes(raw))


__all__ = [
    "ACTION_SOURCES",
    "ACTORS",
    "ActualOpenLedgerRowV1",
    "ActualOpenLedgerV1",
    "CANONICAL_COLORS",
    "CallCounterPanelV1",
    "ControllerEpisodeBindingV1",
    "ControllerTraceV1",
    "EMPTY_OPEN_LEDGER_SHA256",
    "EMPTY_TICK_CHAIN_SHA256",
    "NavigationTickRecordV1",
    "NavigationTraceV1Error",
    "NavigationTraceV1HashError",
    "NavigationTraceV1SchemaError",
    "ResetReceiptV1",
    "canonical_binary64_hex",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "decode_canonical_binary64_hex",
    "parse_canonical_json_bytes",
    "parse_controller_trace_v1",
    "require_exact_dict",
    "require_identifier",
    "require_nonnegative_int",
    "require_optional_sha256",
    "require_sha256",
    "validate_content_commitment",
    "zero_owner_revisions",
]
