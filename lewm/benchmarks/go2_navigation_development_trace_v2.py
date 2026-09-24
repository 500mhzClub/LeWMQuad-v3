"""Closed evidence schemas for the Navigation V4 Foundation V2 correction.

This standard-library-only module contains immutable evidence, never live
runtime authority.  It performs no filesystem access and imports neither
blocked V1 foundation module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct


SCHEMA_VERSION_V2 = 2
CANONICAL_COLORS_V2 = ("red", "yellow", "blue", "green")
OWNER_NAMES_V2 = (
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
ACTION_SOURCES_V2 = ("target_router", "learned_g4", "stop", "fault")
OPEN_ACTORS_V2 = (
    "launcher",
    "physics_broker",
    "controller",
    "observer",
    "evaluator",
    "finalizer",
)
OPEN_PHASES_V2 = (
    "capture",
    "controller",
    "broker",
    "post_seal_observer",
    "evaluation",
    "finalization",
)
OPEN_DISPOSITIONS_V2 = (
    "allowed",
    "denied_duplicate",
    "denied_hash_mismatch",
    "denied_symlink",
    "denied_escape",
    "denied_wrong_phase",
    "denied_wrong_actor",
    "denied_unexpected",
)
FAULT_STAGES_V2 = (
    "frame_validation",
    "shared_frame",
    "target_batch",
    "physical_projection",
    "admission",
    "view",
    "target_update",
    "scheduling",
    "candidate",
    "g4",
    "follower",
    "action",
    "claim",
    "commit",
    "lease",
)

PRODUCTION_NAVIGATION_TRACE_V2 = None
PRODUCTION_RESET_RECEIPT_V2 = None
PRODUCTION_OPEN_LEDGER_V2 = None

OWNER_STATE_SCHEMA_V2 = "lewm_go2_owner_state_v2"
OWNER_STATE_BUNDLE_SCHEMA_V2 = "lewm_go2_owner_state_bundle_v2"
RESET_RECEIPT_SCHEMA_V2 = "lewm_go2_navigation_reset_receipt_v2"
TARGET_DECISION_SCHEMA_V2 = "lewm_go2_target_decision_evidence_v2"
EXPLORATION_DECISION_SCHEMA_V2 = "lewm_go2_exploration_decision_evidence_v2"
FAULT_DECISION_SCHEMA_V2 = "lewm_go2_fault_decision_evidence_v2"
FOLLOWER_COMMAND_SCHEMA_V2 = "lewm_go2_follower_command_evidence_v2"
FOLLOWER_STOP_SCHEMA_V2 = "lewm_go2_follower_stop_evidence_v2"
FAULT_STOP_SCHEMA_V2 = "lewm_go2_fault_stop_evidence_v2"
CLAIM_INTENT_SCHEMA_V2 = "lewm_go2_semantic_claim_intent_v2"
NAVIGATION_TICK_SCHEMA_V2 = "lewm_go2_navigation_tick_record_v2"
OPEN_PATH_EVIDENCE_SCHEMA_V2 = "lewm_go2_open_path_resolution_evidence_v2"
OPEN_POLICY_EVIDENCE_SCHEMA_V2 = "lewm_go2_open_policy_decision_evidence_v2"
OPEN_LEDGER_ROW_SCHEMA_V2 = "lewm_go2_actual_open_ledger_row_v2"
OPEN_LEDGER_SCHEMA_V2 = "lewm_go2_actual_open_ledger_v2"
OPEN_PROJECTION_SCHEMA_V2 = "lewm_go2_actual_open_ledger_projection_v2"
CONTROLLER_TRACE_SCHEMA_V2 = "lewm_go2_controller_trace_v2"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_BINARY64_RE = re.compile(r"[0-9a-f]{16}\Z")
_MAX_I64 = 2**63 - 1


class NavigationTraceV2Error(ValueError):
    """Base V2 evidence error."""


class NavigationTraceV2SchemaError(NavigationTraceV2Error):
    """A closed evidence schema was violated."""


class NavigationTraceV2HashError(NavigationTraceV2Error):
    """A content or chain commitment was invalid."""


def _walk_plain_json_v2(value: object, *, name: str = "value") -> None:
    if value is None or type(value) in {str, bool, int}:
        if type(value) is int and not -_MAX_I64 - 1 <= value <= _MAX_I64:
            raise NavigationTraceV2SchemaError(f"{name} is outside signed-64 range")
        return
    if type(value) is float:
        raise NavigationTraceV2SchemaError(
            f"{name} native float is forbidden; use canonical binary64 hex"
        )
    if type(value) is list:
        for index, item in enumerate(value):
            _walk_plain_json_v2(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise NavigationTraceV2SchemaError(f"{name} contains a nonstring key")
            _walk_plain_json_v2(item, name=f"{name}.{key}")
        return
    raise NavigationTraceV2SchemaError(f"{name} must use exact JSON types")


def canonical_json_bytes_v2(value: object) -> bytes:
    _walk_plain_json_v2(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256_v2(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes_v2(value)).hexdigest()


def _pairs_v2(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise NavigationTraceV2SchemaError("JSON object key must be a string")
        if key in result:
            raise NavigationTraceV2SchemaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_float_v2(_: str) -> object:
    raise NavigationTraceV2SchemaError("native JSON floats are forbidden")


def _reject_constant_v2(_: str) -> object:
    raise NavigationTraceV2SchemaError("nonfinite JSON constants are forbidden")


def parse_canonical_json_bytes_v2(raw: bytes) -> object:
    if type(raw) is not bytes:
        raise TypeError("raw must be exact bytes")
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_v2,
            parse_float=_reject_float_v2,
            parse_constant=_reject_constant_v2,
        )
    except UnicodeDecodeError as exc:
        raise NavigationTraceV2SchemaError("canonical JSON must be ASCII") from exc
    except json.JSONDecodeError as exc:
        raise NavigationTraceV2SchemaError("invalid JSON") from exc
    _walk_plain_json_v2(value)
    if canonical_json_bytes_v2(value) != raw:
        raise NavigationTraceV2SchemaError("JSON bytes are not canonical")
    return value


def canonical_binary64_hex_v2(value: int | float) -> str:
    if type(value) not in {int, float}:
        raise TypeError("binary64 value must be exact int or float")
    number = float(value)
    if not math.isfinite(number):
        raise NavigationTraceV2SchemaError("binary64 value must be finite")
    if number == 0.0:
        number = 0.0
    return struct.pack(">d", number).hex()


def decode_canonical_binary64_hex_v2(value: object, *, name: str) -> float:
    if type(value) is not str or _BINARY64_RE.fullmatch(value) is None:
        raise NavigationTraceV2SchemaError(f"{name} must be 16 lowercase hex digits")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(number):
        raise NavigationTraceV2SchemaError(f"{name} must encode finite binary64")
    if number == 0.0 and value != "0000000000000000":
        raise NavigationTraceV2SchemaError(f"{name} encodes negative zero")
    if canonical_binary64_hex_v2(number) != value:
        raise NavigationTraceV2SchemaError(f"{name} is not canonical")
    return number


def require_sha256_v2(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise NavigationTraceV2SchemaError(f"{name} must be lowercase SHA-256")
    return value


def require_optional_sha256_v2(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_sha256_v2(value, name=name)


def require_identifier_v2(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise NavigationTraceV2SchemaError(f"{name} must be a canonical identifier")
    return value


def require_nonnegative_int_v2(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise NavigationTraceV2SchemaError(
            f"{name} must be exact nonnegative signed-64 integer"
        )
    return value


def require_exact_dict_v2(
    value: object, *, name: str, keys: set[str] | frozenset[str]
) -> dict[str, object]:
    if type(value) is not dict:
        raise NavigationTraceV2SchemaError(f"{name} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise NavigationTraceV2SchemaError(f"{name} contains a nonstring key")
    if set(value) != set(keys):
        raise NavigationTraceV2SchemaError(f"{name} keys are not exact")
    return value


def _validate_content_v2(
    value: object,
    *,
    schema: str,
    keys: set[str] | frozenset[str],
) -> dict[str, object]:
    record = require_exact_dict_v2(value, name=schema, keys=keys)
    if record["schema"] != schema or type(record["schema"]) is not str:
        raise NavigationTraceV2SchemaError(f"{schema} schema changed")
    if type(record["version"]) is not int or record["version"] != SCHEMA_VERSION_V2:
        raise NavigationTraceV2SchemaError(f"{schema} version changed")
    claimed = require_sha256_v2(record["content_sha256"], name="content_sha256")
    core = dict(record)
    del core["content_sha256"]
    if canonical_json_sha256_v2(core) != claimed:
        raise NavigationTraceV2HashError(f"{schema} content mismatch")
    return record


def initial_owner_content_sha256_v2(
    *, owner_name: str, owner_id: str, reset_id: str, session_id: str
) -> str:
    if owner_name not in OWNER_NAMES_V2:
        raise NavigationTraceV2SchemaError("owner_name is invalid")
    require_identifier_v2(owner_id, name="owner_id")
    require_identifier_v2(reset_id, name="reset_id")
    require_identifier_v2(session_id, name="session_id")
    return canonical_json_sha256_v2(
        {
            "schema": "lewm_go2_fresh_empty_owner_content_v2",
            "version": SCHEMA_VERSION_V2,
            "owner_name": owner_name,
            "owner_id": owner_id,
            "reset_id": reset_id,
            "session_id": session_id,
            "empty": True,
        }
    )


@dataclass(frozen=True)
class OwnerStateV2:
    owner_name: str
    owner_id: str
    revision: int
    owner_content_sha256: str
    reset_id: str
    session_id: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.owner_name) is not str or self.owner_name not in OWNER_NAMES_V2:
            raise NavigationTraceV2SchemaError("owner_name is invalid")
        require_identifier_v2(self.owner_id, name="owner_id")
        require_nonnegative_int_v2(self.revision, name="revision")
        require_sha256_v2(self.owner_content_sha256, name="owner_content_sha256")
        require_identifier_v2(self.reset_id, name="reset_id")
        require_identifier_v2(self.session_id, name="session_id")
        if self.reset_id == self.session_id:
            raise NavigationTraceV2SchemaError("reset and session IDs must differ")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": OWNER_STATE_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "owner_name": self.owner_name,
            "owner_id": self.owner_id,
            "revision": self.revision,
            "owner_content_sha256": self.owner_content_sha256,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "OwnerStateV2":
        record = _validate_content_v2(
            value,
            schema=OWNER_STATE_SCHEMA_V2,
            keys={
                "schema", "version", "owner_name", "owner_id", "revision",
                "owner_content_sha256", "reset_id", "session_id", "content_sha256",
            },
        )
        created = cls(
            owner_name=record["owner_name"],
            owner_id=record["owner_id"],
            revision=record["revision"],
            owner_content_sha256=record["owner_content_sha256"],
            reset_id=record["reset_id"],
            session_id=record["session_id"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("owner state reconstruction mismatch")
        return created


@dataclass(frozen=True)
class OwnerStateBundleV2:
    rows: tuple[OwnerStateV2, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or len(self.rows) != len(OWNER_NAMES_V2):
            raise NavigationTraceV2SchemaError("owner bundle must contain exactly thirteen rows")
        if any(type(row) is not OwnerStateV2 for row in self.rows):
            raise NavigationTraceV2SchemaError("owner bundle row type changed")
        if tuple(row.owner_name for row in self.rows) != OWNER_NAMES_V2:
            raise NavigationTraceV2SchemaError("owner bundle order/names changed")
        ids = tuple(row.owner_id for row in self.rows)
        if len(ids) != len(set(ids)):
            raise NavigationTraceV2SchemaError("owner IDs must be unique")
        resets = {row.reset_id for row in self.rows}
        sessions = {row.session_id for row in self.rows}
        if len(resets) != 1 or len(sessions) != 1:
            raise NavigationTraceV2SchemaError("owner bundle reset/session changed")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    @property
    def reset_id(self) -> str:
        return self.rows[0].reset_id

    @property
    def session_id(self) -> str:
        return self.rows[0].session_id

    def row(self, owner_name: str) -> OwnerStateV2:
        if owner_name not in OWNER_NAMES_V2:
            raise NavigationTraceV2SchemaError("owner name is invalid")
        return self.rows[OWNER_NAMES_V2.index(owner_name)]

    def _core(self) -> dict[str, object]:
        return {
            "schema": OWNER_STATE_BUNDLE_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "OwnerStateBundleV2":
        record = _validate_content_v2(
            value,
            schema=OWNER_STATE_BUNDLE_SCHEMA_V2,
            keys={"schema", "version", "rows", "content_sha256"},
        )
        rows = record["rows"]
        if type(rows) is not list:
            raise NavigationTraceV2SchemaError("owner bundle rows must be exact list")
        created = cls(tuple(OwnerStateV2.from_dict(row) for row in rows))
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("owner bundle reconstruction mismatch")
        return created


@dataclass(frozen=True)
class ResetReceiptV2:
    authority_sequence: int
    issuer_id: str
    reset_id: str
    session_id: str
    reset_capability_id: str
    physical_projection_producer_id: str
    candidate_producer_id: str
    initial_owner_states: OwnerStateBundleV2
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if require_nonnegative_int_v2(self.authority_sequence, name="authority_sequence") == 0:
            raise NavigationTraceV2SchemaError("authority_sequence must be positive")
        for name in (
            "issuer_id", "reset_id", "session_id", "reset_capability_id",
            "physical_projection_producer_id", "candidate_producer_id",
        ):
            require_identifier_v2(getattr(self, name), name=name)
        if type(self.initial_owner_states) is not OwnerStateBundleV2:
            raise NavigationTraceV2SchemaError("initial owner bundle type changed")
        if (
            self.initial_owner_states.reset_id != self.reset_id
            or self.initial_owner_states.session_id != self.session_id
        ):
            raise NavigationTraceV2SchemaError("initial owners do not bind reset/session")
        minted_ids = (
            self.reset_id,
            self.session_id,
            self.reset_capability_id,
            self.physical_projection_producer_id,
            self.candidate_producer_id,
            *(row.owner_id for row in self.initial_owner_states.rows),
        )
        if len(minted_ids) != len(set(minted_ids)):
            raise NavigationTraceV2SchemaError("reset/owner/producer identities must be distinct")
        for row in self.initial_owner_states.rows:
            if row.revision != 0:
                raise NavigationTraceV2SchemaError("tick-zero owner revision is nonzero")
            expected = initial_owner_content_sha256_v2(
                owner_name=row.owner_name,
                owner_id=row.owner_id,
                reset_id=self.reset_id,
                session_id=self.session_id,
            )
            if row.owner_content_sha256 != expected:
                raise NavigationTraceV2SchemaError("tick-zero owner content is not fresh/empty")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": RESET_RECEIPT_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "authority_sequence": self.authority_sequence,
            "issuer_id": self.issuer_id,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "reset_capability_id": self.reset_capability_id,
            "physical_projection_producer_id": self.physical_projection_producer_id,
            "candidate_producer_id": self.candidate_producer_id,
            "initial_owner_states": self.initial_owner_states.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ResetReceiptV2":
        record = _validate_content_v2(
            value,
            schema=RESET_RECEIPT_SCHEMA_V2,
            keys={
                "schema", "version", "authority_sequence", "issuer_id", "reset_id",
                "session_id", "reset_capability_id", "physical_projection_producer_id",
                "candidate_producer_id", "initial_owner_states", "content_sha256",
            },
        )
        created = cls(
            authority_sequence=record["authority_sequence"],
            issuer_id=record["issuer_id"],
            reset_id=record["reset_id"],
            session_id=record["session_id"],
            reset_capability_id=record["reset_capability_id"],
            physical_projection_producer_id=record["physical_projection_producer_id"],
            candidate_producer_id=record["candidate_producer_id"],
            initial_owner_states=OwnerStateBundleV2.from_dict(record["initial_owner_states"]),
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("reset receipt reconstruction mismatch")
        return created


@dataclass(frozen=True)
class CallCounterPanelV2:
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
            require_nonnegative_int_v2(getattr(self, name), name=name)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return (
            "observation_tick_count", "shared_frame_outcome_count",
            "shared_v5_forward_frame_call_count", "vision_encoder_forward_tokens_call_count",
            "target_four_color_batch_count", "g4_value_head_call_count",
            "rgb_decode_call_count", "rgb_preprocess_call_count",
            "extra_rgb_decode_or_preprocess_count",
        )

    @classmethod
    def zero(cls) -> "CallCounterPanelV2":
        return cls(*(0 for _ in cls.names()))

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.names()}

    @classmethod
    def from_dict(cls, value: object) -> "CallCounterPanelV2":
        record = require_exact_dict_v2(value, name="call_counter_panel_v2", keys=set(cls.names()))
        return cls(*(record[name] for name in cls.names()))

    def plus(self, other: "CallCounterPanelV2") -> "CallCounterPanelV2":
        if type(other) is not CallCounterPanelV2:
            raise TypeError("other must be exact CallCounterPanelV2")
        return CallCounterPanelV2(
            *(getattr(self, name) + getattr(other, name) for name in self.names())
        )

    def assert_complete_observation(self, *, g4_calls: int) -> None:
        if type(g4_calls) is not int or g4_calls not in {0, 1}:
            raise NavigationTraceV2SchemaError("g4_calls must be exact zero or one")
        required_one = (
            self.observation_tick_count,
            self.shared_frame_outcome_count,
            self.shared_v5_forward_frame_call_count,
            self.vision_encoder_forward_tokens_call_count,
            self.target_four_color_batch_count,
            self.rgb_decode_call_count,
            self.rgb_preprocess_call_count,
        )
        if any(value != 1 for value in required_one):
            raise NavigationTraceV2SchemaError("one-encode complete-observation equality failed")
        if self.g4_value_head_call_count != g4_calls:
            raise NavigationTraceV2SchemaError("G4 count does not match decision")
        if self.extra_rgb_decode_or_preprocess_count != 0:
            raise NavigationTraceV2SchemaError("extra RGB decode/preprocess count is nonzero")


def advance_owner_bundle_v2(
    pre: OwnerStateBundleV2,
    *,
    advanced_owner_content_sha256: dict[str, str],
) -> OwnerStateBundleV2:
    """Evidence helper: advance only the exact named owner rows by one."""

    if type(pre) is not OwnerStateBundleV2:
        raise TypeError("pre must be exact OwnerStateBundleV2")
    if type(advanced_owner_content_sha256) is not dict:
        raise NavigationTraceV2SchemaError("advanced content must be exact dict")
    if any(type(name) is not str for name in advanced_owner_content_sha256):
        raise NavigationTraceV2SchemaError("advanced content contains nonstring owner")
    if not set(advanced_owner_content_sha256).issubset(OWNER_NAMES_V2):
        raise NavigationTraceV2SchemaError("advanced content names are invalid")
    rows = []
    for row in pre.rows:
        if row.owner_name in advanced_owner_content_sha256:
            content = require_sha256_v2(
                advanced_owner_content_sha256[row.owner_name],
                name=f"advanced_owner_content_sha256.{row.owner_name}",
            )
            if content == row.owner_content_sha256:
                raise NavigationTraceV2SchemaError("advanced owner content must change")
            rows.append(
                OwnerStateV2(
                    owner_name=row.owner_name,
                    owner_id=row.owner_id,
                    revision=row.revision + 1,
                    owner_content_sha256=content,
                    reset_id=row.reset_id,
                    session_id=row.session_id,
                )
            )
        else:
            rows.append(row)
    return OwnerStateBundleV2(tuple(rows))


@dataclass(frozen=True)
class TargetDecisionEvidenceV2:
    locked_color: str
    target_route_sha256: str
    selected_path_sha256: str
    terminal_yaw_binary64_hex: str
    content_sha256: str = field(init=False)

    tag = "target"

    def __post_init__(self) -> None:
        if type(self.locked_color) is not str or self.locked_color not in CANONICAL_COLORS_V2:
            raise NavigationTraceV2SchemaError("locked target color is invalid")
        for name in ("target_route_sha256", "selected_path_sha256"):
            require_sha256_v2(getattr(self, name), name=name)
        decode_canonical_binary64_hex_v2(
            self.terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex"
        )
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": TARGET_DECISION_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "locked_color": self.locked_color,
            "target_route_sha256": self.target_route_sha256,
            "selected_path_sha256": self.selected_path_sha256,
            "terminal_yaw_binary64_hex": self.terminal_yaw_binary64_hex,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "TargetDecisionEvidenceV2":
        record = _validate_content_v2(
            value,
            schema=TARGET_DECISION_SCHEMA_V2,
            keys={
                "schema", "version", "tag", "locked_color", "target_route_sha256",
                "selected_path_sha256", "terminal_yaw_binary64_hex", "content_sha256",
            },
        )
        if record["tag"] != "target" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("target decision tag changed")
        created = cls(
            locked_color=record["locked_color"],
            target_route_sha256=record["target_route_sha256"],
            selected_path_sha256=record["selected_path_sha256"],
            terminal_yaw_binary64_hex=record["terminal_yaw_binary64_hex"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("target decision reconstruction mismatch")
        return created


@dataclass(frozen=True)
class ExplorationDecisionEvidenceV2:
    candidate_identity: str
    candidate_admission_sha256: str
    candidate_feature_batch_sha256: str
    candidate_rows_sha256: str
    baseline_scores_sha256: str
    learned_scores_sha256: str
    selected_row: int
    selected_candidate_row_sha256: str
    selected_path_sha256: str
    terminal_yaw_binary64_hex: str
    content_sha256: str = field(init=False)

    tag = "exploration"

    def __post_init__(self) -> None:
        require_identifier_v2(self.candidate_identity, name="candidate_identity")
        for name in (
            "candidate_admission_sha256", "candidate_feature_batch_sha256",
            "candidate_rows_sha256", "baseline_scores_sha256", "learned_scores_sha256",
            "selected_candidate_row_sha256", "selected_path_sha256",
        ):
            require_sha256_v2(getattr(self, name), name=name)
        require_nonnegative_int_v2(self.selected_row, name="selected_row")
        decode_canonical_binary64_hex_v2(
            self.terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex"
        )
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": EXPLORATION_DECISION_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "candidate_identity": self.candidate_identity,
            "candidate_admission_sha256": self.candidate_admission_sha256,
            "candidate_feature_batch_sha256": self.candidate_feature_batch_sha256,
            "candidate_rows_sha256": self.candidate_rows_sha256,
            "baseline_scores_sha256": self.baseline_scores_sha256,
            "learned_scores_sha256": self.learned_scores_sha256,
            "selected_row": self.selected_row,
            "selected_candidate_row_sha256": self.selected_candidate_row_sha256,
            "selected_path_sha256": self.selected_path_sha256,
            "terminal_yaw_binary64_hex": self.terminal_yaw_binary64_hex,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ExplorationDecisionEvidenceV2":
        keys = {
            "schema", "version", "tag", "candidate_identity", "candidate_admission_sha256",
            "candidate_feature_batch_sha256", "candidate_rows_sha256", "baseline_scores_sha256",
            "learned_scores_sha256", "selected_row", "selected_candidate_row_sha256",
            "selected_path_sha256", "terminal_yaw_binary64_hex", "content_sha256",
        }
        record = _validate_content_v2(value, schema=EXPLORATION_DECISION_SCHEMA_V2, keys=keys)
        if record["tag"] != "exploration" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("exploration decision tag changed")
        created = cls(
            **{
                name: record[name]
                for name in keys - {"schema", "version", "tag", "content_sha256"}
            }
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("exploration decision reconstruction mismatch")
        return created


NormalDecisionEvidenceV2 = TargetDecisionEvidenceV2 | ExplorationDecisionEvidenceV2


def _normal_decision_from_dict_v2(value: object) -> NormalDecisionEvidenceV2:
    if type(value) is not dict:
        raise NavigationTraceV2SchemaError("normal decision must be exact dict")
    tag = value.get("tag")
    if tag == "target":
        return TargetDecisionEvidenceV2.from_dict(value)
    if tag == "exploration":
        return ExplorationDecisionEvidenceV2.from_dict(value)
    raise NavigationTraceV2SchemaError("normal decision tag is invalid")


@dataclass(frozen=True)
class FaultDecisionEvidenceV2:
    fault_code: str
    fault_stage: str
    completed_decision: NormalDecisionEvidenceV2 | None
    content_sha256: str = field(init=False)

    tag = "fault"

    def __post_init__(self) -> None:
        require_identifier_v2(self.fault_code, name="fault_code")
        if type(self.fault_stage) is not str or self.fault_stage not in FAULT_STAGES_V2:
            raise NavigationTraceV2SchemaError("fault_stage is invalid")
        if self.completed_decision is not None and type(self.completed_decision) not in {
            TargetDecisionEvidenceV2,
            ExplorationDecisionEvidenceV2,
        }:
            raise NavigationTraceV2SchemaError("fault completed decision type changed")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": FAULT_DECISION_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "fault_code": self.fault_code,
            "fault_stage": self.fault_stage,
            "completed_decision": (
                None if self.completed_decision is None else self.completed_decision.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FaultDecisionEvidenceV2":
        record = _validate_content_v2(
            value,
            schema=FAULT_DECISION_SCHEMA_V2,
            keys={
                "schema", "version", "tag", "fault_code", "fault_stage",
                "completed_decision", "content_sha256",
            },
        )
        if record["tag"] != "fault" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("fault decision tag changed")
        completed = record["completed_decision"]
        created = cls(
            fault_code=record["fault_code"],
            fault_stage=record["fault_stage"],
            completed_decision=(
                None if completed is None else _normal_decision_from_dict_v2(completed)
            ),
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("fault decision reconstruction mismatch")
        return created


DecisionEvidenceV2 = NormalDecisionEvidenceV2 | FaultDecisionEvidenceV2


def decision_from_dict_v2(value: object) -> DecisionEvidenceV2:
    if type(value) is not dict:
        raise NavigationTraceV2SchemaError("decision must be exact dict")
    tag = value.get("tag")
    if tag in {"target", "exploration"}:
        return _normal_decision_from_dict_v2(value)
    if tag == "fault":
        return FaultDecisionEvidenceV2.from_dict(value)
    raise NavigationTraceV2SchemaError("decision tag is invalid")


@dataclass(frozen=True)
class FollowerCommandEvidenceV2:
    selected_path_sha256: str
    terminal_yaw_binary64_hex: str
    follower_receipt_sha256: str
    requested_command_sha256: str
    executed_command_sha256: str
    command_source_commitment_sha256: str
    stall_state: str
    content_sha256: str = field(init=False)

    tag = "follower_command"

    def __post_init__(self) -> None:
        for name in (
            "selected_path_sha256", "follower_receipt_sha256", "requested_command_sha256",
            "executed_command_sha256", "command_source_commitment_sha256",
        ):
            require_sha256_v2(getattr(self, name), name=name)
        decode_canonical_binary64_hex_v2(
            self.terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex"
        )
        if type(self.stall_state) is not str or self.stall_state not in {"moving", "controller_stalled"}:
            raise NavigationTraceV2SchemaError("command stall_state is invalid")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": FOLLOWER_COMMAND_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "selected_path_sha256": self.selected_path_sha256,
            "terminal_yaw_binary64_hex": self.terminal_yaw_binary64_hex,
            "follower_receipt_sha256": self.follower_receipt_sha256,
            "requested_command_sha256": self.requested_command_sha256,
            "executed_command_sha256": self.executed_command_sha256,
            "command_source_commitment_sha256": self.command_source_commitment_sha256,
            "stall_state": self.stall_state,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FollowerCommandEvidenceV2":
        keys = {
            "schema", "version", "tag", "selected_path_sha256",
            "terminal_yaw_binary64_hex", "follower_receipt_sha256",
            "requested_command_sha256", "executed_command_sha256",
            "command_source_commitment_sha256", "stall_state", "content_sha256",
        }
        record = _validate_content_v2(value, schema=FOLLOWER_COMMAND_SCHEMA_V2, keys=keys)
        if record["tag"] != "follower_command" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("follower command tag changed")
        created = cls(
            **{
                name: record[name]
                for name in keys - {"schema", "version", "tag", "content_sha256"}
            }
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("follower command reconstruction mismatch")
        return created


@dataclass(frozen=True)
class FollowerStopEvidenceV2:
    selected_path_sha256: str
    terminal_yaw_binary64_hex: str
    follower_receipt_sha256: str
    stop_code: str
    command_source_commitment_sha256: str
    content_sha256: str = field(init=False)

    tag = "follower_stop"

    def __post_init__(self) -> None:
        for name in (
            "selected_path_sha256", "follower_receipt_sha256",
            "command_source_commitment_sha256",
        ):
            require_sha256_v2(getattr(self, name), name=name)
        decode_canonical_binary64_hex_v2(
            self.terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex"
        )
        require_identifier_v2(self.stop_code, name="stop_code")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": FOLLOWER_STOP_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "selected_path_sha256": self.selected_path_sha256,
            "terminal_yaw_binary64_hex": self.terminal_yaw_binary64_hex,
            "follower_receipt_sha256": self.follower_receipt_sha256,
            "stop_code": self.stop_code,
            "command_source_commitment_sha256": self.command_source_commitment_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FollowerStopEvidenceV2":
        keys = {
            "schema", "version", "tag", "selected_path_sha256",
            "terminal_yaw_binary64_hex", "follower_receipt_sha256", "stop_code",
            "command_source_commitment_sha256", "content_sha256",
        }
        record = _validate_content_v2(value, schema=FOLLOWER_STOP_SCHEMA_V2, keys=keys)
        if record["tag"] != "follower_stop" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("follower stop tag changed")
        created = cls(
            **{
                name: record[name]
                for name in keys - {"schema", "version", "tag", "content_sha256"}
            }
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("follower stop reconstruction mismatch")
        return created


@dataclass(frozen=True)
class FaultStopEvidenceV2:
    disposition: str
    stop_evidence_sha256: str | None
    requested_stop_command_sha256: str | None
    executed_stop_command_sha256: str | None
    content_sha256: str = field(init=False)

    tag = "fault_stop"

    def __post_init__(self) -> None:
        if type(self.disposition) is not str or self.disposition not in {
            "issued",
            "not_safe_to_issue",
        }:
            raise NavigationTraceV2SchemaError("fault stop disposition is invalid")
        values = (
            self.stop_evidence_sha256,
            self.requested_stop_command_sha256,
            self.executed_stop_command_sha256,
        )
        for index, value in enumerate(values):
            require_optional_sha256_v2(value, name=f"fault_stop_hash[{index}]")
        if self.disposition == "issued" and any(value is None for value in values):
            raise NavigationTraceV2SchemaError("issued fault stop lacks exact evidence")
        if self.disposition == "not_safe_to_issue" and any(value is not None for value in values):
            raise NavigationTraceV2SchemaError("unsafe fault stop carries command evidence")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": FAULT_STOP_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tag": self.tag,
            "disposition": self.disposition,
            "stop_evidence_sha256": self.stop_evidence_sha256,
            "requested_stop_command_sha256": self.requested_stop_command_sha256,
            "executed_stop_command_sha256": self.executed_stop_command_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FaultStopEvidenceV2":
        keys = {
            "schema", "version", "tag", "disposition", "stop_evidence_sha256",
            "requested_stop_command_sha256", "executed_stop_command_sha256",
            "content_sha256",
        }
        record = _validate_content_v2(value, schema=FAULT_STOP_SCHEMA_V2, keys=keys)
        if record["tag"] != "fault_stop" or type(record["tag"]) is not str:
            raise NavigationTraceV2SchemaError("fault stop tag changed")
        created = cls(
            disposition=record["disposition"],
            stop_evidence_sha256=record["stop_evidence_sha256"],
            requested_stop_command_sha256=record["requested_stop_command_sha256"],
            executed_stop_command_sha256=record["executed_stop_command_sha256"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("fault stop reconstruction mismatch")
        return created


ActionEvidenceV2 = FollowerCommandEvidenceV2 | FollowerStopEvidenceV2 | FaultStopEvidenceV2


def action_from_dict_v2(value: object) -> ActionEvidenceV2:
    if type(value) is not dict:
        raise NavigationTraceV2SchemaError("action must be exact dict")
    tag = value.get("tag")
    if tag == "follower_command":
        return FollowerCommandEvidenceV2.from_dict(value)
    if tag == "follower_stop":
        return FollowerStopEvidenceV2.from_dict(value)
    if tag == "fault_stop":
        return FaultStopEvidenceV2.from_dict(value)
    raise NavigationTraceV2SchemaError("action tag is invalid")


@dataclass(frozen=True)
class SemanticClaimIntentV2:
    color: str
    readiness_receipt_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.color) is not str or self.color not in CANONICAL_COLORS_V2:
            raise NavigationTraceV2SchemaError("claim color is invalid")
        require_sha256_v2(self.readiness_receipt_sha256, name="readiness_receipt_sha256")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": CLAIM_INTENT_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "color": self.color,
            "readiness_receipt_sha256": self.readiness_receipt_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "SemanticClaimIntentV2":
        record = _validate_content_v2(
            value,
            schema=CLAIM_INTENT_SCHEMA_V2,
            keys={
                "schema", "version", "color", "readiness_receipt_sha256",
                "content_sha256",
            },
        )
        created = cls(
            color=record["color"],
            readiness_receipt_sha256=record["readiness_receipt_sha256"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("claim reconstruction mismatch")
        return created


def derive_action_source_v2(
    decision: DecisionEvidenceV2,
    action: ActionEvidenceV2,
    claim: SemanticClaimIntentV2 | None,
) -> str:
    if type(decision) is FaultDecisionEvidenceV2:
        if type(action) is not FaultStopEvidenceV2 or claim is not None:
            raise NavigationTraceV2SchemaError("fault decision requires fault stop and no claim")
        return "fault"
    if type(action) is FaultStopEvidenceV2:
        raise NavigationTraceV2SchemaError("normal decision cannot carry fault action")
    if action.selected_path_sha256 != decision.selected_path_sha256:
        raise NavigationTraceV2SchemaError("action path differs from decision path")
    if action.terminal_yaw_binary64_hex != decision.terminal_yaw_binary64_hex:
        raise NavigationTraceV2SchemaError("action yaw differs from decision yaw")
    if claim is not None:
        if type(decision) is not TargetDecisionEvidenceV2 or claim.color != decision.locked_color:
            raise NavigationTraceV2SchemaError("claim is not bound to target decision color")
    if type(decision) is ExplorationDecisionEvidenceV2 and claim is not None:
        raise NavigationTraceV2SchemaError("exploration cannot carry claim intent")
    if type(action) is FollowerStopEvidenceV2:
        return "stop"
    if type(action) is not FollowerCommandEvidenceV2:
        raise NavigationTraceV2SchemaError("normal action type changed")
    if type(decision) is TargetDecisionEvidenceV2:
        return "target_router"
    if type(decision) is ExplorationDecisionEvidenceV2:
        return "learned_g4"
    raise NavigationTraceV2SchemaError("decision/action union is invalid")


def _validate_fault_counts_v2(counts: CallCounterPanelV2, decision: FaultDecisionEvidenceV2) -> None:
    if counts.extra_rgb_decode_or_preprocess_count != 0:
        raise NavigationTraceV2SchemaError("fault tick has extra RGB decode/preprocess")
    if any(getattr(counts, name) > 1 for name in counts.names()[:-1]):
        raise NavigationTraceV2SchemaError("fault tick per-call count exceeds one")
    if counts.shared_frame_outcome_count > counts.observation_tick_count:
        raise NavigationTraceV2SchemaError("fault count ordering is invalid")
    if counts.shared_v5_forward_frame_call_count != counts.vision_encoder_forward_tokens_call_count:
        raise NavigationTraceV2SchemaError("fault frame/encoder counts differ")
    if counts.shared_v5_forward_frame_call_count > counts.observation_tick_count:
        raise NavigationTraceV2SchemaError("fault frame count exceeds observation")
    if counts.target_four_color_batch_count > counts.shared_frame_outcome_count:
        raise NavigationTraceV2SchemaError("fault target count exceeds frame outcomes")
    if counts.g4_value_head_call_count > counts.target_four_color_batch_count:
        raise NavigationTraceV2SchemaError("fault G4 count exceeds target batches")
    completed = decision.completed_decision
    if completed is not None:
        counts.assert_complete_observation(
            g4_calls=1 if type(completed) is ExplorationDecisionEvidenceV2 else 0
        )


def _validate_owner_transition_v2(
    *,
    pre: OwnerStateBundleV2,
    post: OwnerStateBundleV2,
    decision: DecisionEvidenceV2,
    action: ActionEvidenceV2,
    claim: SemanticClaimIntentV2 | None,
    disposition: str,
) -> None:
    if pre.reset_id != post.reset_id or pre.session_id != post.session_id:
        raise NavigationTraceV2SchemaError("owner transition crossed reset/session")
    if disposition == "terminal_fault_rollback":
        if type(decision) is not FaultDecisionEvidenceV2 or type(action) is not FaultStopEvidenceV2:
            raise NavigationTraceV2SchemaError("rollback transition lacks fault union")
        advanced = {"tick_chain"}
    elif disposition == "committed":
        if type(decision) is FaultDecisionEvidenceV2 or type(action) is FaultStopEvidenceV2:
            raise NavigationTraceV2SchemaError("committed transition carries fault union")
        advanced = {
            "physical",
            "configuration",
            "view",
            "target_red",
            "target_yellow",
            "target_blue",
            "target_green",
            "follower",
            "integration",
            "action_journal",
            "tick_chain",
        }
        if type(decision) is TargetDecisionEvidenceV2:
            advanced.add("router")
        if claim is not None:
            advanced.add("claim_journal")
    else:
        raise NavigationTraceV2SchemaError("transition disposition is invalid")
    for name in OWNER_NAMES_V2:
        before = pre.row(name)
        after = post.row(name)
        if (
            before.owner_id != after.owner_id
            or before.reset_id != after.reset_id
            or before.session_id != after.session_id
        ):
            raise NavigationTraceV2SchemaError(f"{name} owner identity changed")
        if name in advanced:
            if after.revision != before.revision + 1:
                raise NavigationTraceV2SchemaError(f"{name} revision did not advance exactly once")
            if after.owner_content_sha256 == before.owner_content_sha256:
                raise NavigationTraceV2SchemaError(f"{name} advanced without content transition")
        elif after.to_dict() != before.to_dict():
            if after.revision == before.revision:
                raise NavigationTraceV2SchemaError(f"{name} content changed without revision")
            raise NavigationTraceV2SchemaError(f"{name} changed despite no transition")


@dataclass(frozen=True)
class NavigationTickRecordV2:
    tick_index: int
    timestamp_binary64_hex: str
    synchronization_id: str
    reset_id: str
    session_id: str
    controller_input_sha256: str
    inference_receipt_sha256: str
    per_tick_counts: CallCounterPanelV2
    cumulative_counts: CallCounterPanelV2
    pre_owner_states: OwnerStateBundleV2
    post_owner_states: OwnerStateBundleV2
    physical_producer_receipt_sha256: str | None
    target_batch_receipt_sha256: str | None
    tick_admission_receipt_sha256: str | None
    view_admission_sha256: str | None
    target_outcome_sha256s: tuple[str, ...]
    decision: DecisionEvidenceV2
    action: ActionEvidenceV2
    claim_intent: SemanticClaimIntentV2 | None
    transition_disposition: str
    broker_execution_sha256: str
    broker_fall_sha256: str
    previous_tick_chain_sha256: str
    action_source: str = field(init=False)
    content_sha256: str = field(init=False)
    chain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_nonnegative_int_v2(self.tick_index, name="tick_index")
        decode_canonical_binary64_hex_v2(
            self.timestamp_binary64_hex, name="timestamp_binary64_hex"
        )
        require_identifier_v2(self.synchronization_id, name="synchronization_id")
        require_identifier_v2(self.reset_id, name="reset_id")
        require_identifier_v2(self.session_id, name="session_id")
        for name in (
            "controller_input_sha256",
            "inference_receipt_sha256",
            "broker_execution_sha256",
            "broker_fall_sha256",
            "previous_tick_chain_sha256",
        ):
            require_sha256_v2(getattr(self, name), name=name)
        for name in (
            "physical_producer_receipt_sha256",
            "target_batch_receipt_sha256",
            "tick_admission_receipt_sha256",
            "view_admission_sha256",
        ):
            require_optional_sha256_v2(getattr(self, name), name=name)
        if type(self.per_tick_counts) is not CallCounterPanelV2 or type(self.cumulative_counts) is not CallCounterPanelV2:
            raise NavigationTraceV2SchemaError("tick counters must be exact V2 panels")
        if type(self.pre_owner_states) is not OwnerStateBundleV2 or type(self.post_owner_states) is not OwnerStateBundleV2:
            raise NavigationTraceV2SchemaError("tick owner bundles must be exact V2 bundles")
        if (
            self.pre_owner_states.reset_id != self.reset_id
            or self.pre_owner_states.session_id != self.session_id
            or self.post_owner_states.reset_id != self.reset_id
            or self.post_owner_states.session_id != self.session_id
        ):
            raise NavigationTraceV2SchemaError("tick owner bundle reset/session changed")
        if type(self.target_outcome_sha256s) is not tuple:
            raise NavigationTraceV2SchemaError("target outcomes must be exact tuple")
        for index, item in enumerate(self.target_outcome_sha256s):
            require_sha256_v2(item, name=f"target_outcome_sha256s[{index}]")
        if type(self.decision) not in {
            TargetDecisionEvidenceV2,
            ExplorationDecisionEvidenceV2,
            FaultDecisionEvidenceV2,
        }:
            raise NavigationTraceV2SchemaError("decision exact type changed")
        if type(self.action) not in {
            FollowerCommandEvidenceV2,
            FollowerStopEvidenceV2,
            FaultStopEvidenceV2,
        }:
            raise NavigationTraceV2SchemaError("action exact type changed")
        if self.claim_intent is not None and type(self.claim_intent) is not SemanticClaimIntentV2:
            raise NavigationTraceV2SchemaError("claim exact type changed")
        source = derive_action_source_v2(self.decision, self.action, self.claim_intent)
        if type(self.decision) is FaultDecisionEvidenceV2:
            if self.transition_disposition != "terminal_fault_rollback":
                raise NavigationTraceV2SchemaError("fault decision must roll back")
            _validate_fault_counts_v2(self.per_tick_counts, self.decision)
            if len(self.target_outcome_sha256s) not in {0, 4}:
                raise NavigationTraceV2SchemaError("fault target outcomes length is invalid")
        else:
            if self.transition_disposition != "committed":
                raise NavigationTraceV2SchemaError("normal decision must commit")
            self.per_tick_counts.assert_complete_observation(
                g4_calls=1 if type(self.decision) is ExplorationDecisionEvidenceV2 else 0
            )
            if len(self.target_outcome_sha256s) != 4:
                raise NavigationTraceV2SchemaError("committed tick needs four target outcomes")
            if any(
                value is None
                for value in (
                    self.physical_producer_receipt_sha256,
                    self.target_batch_receipt_sha256,
                    self.tick_admission_receipt_sha256,
                    self.view_admission_sha256,
                )
            ):
                raise NavigationTraceV2SchemaError("committed tick lacks admission evidence")
        _validate_owner_transition_v2(
            pre=self.pre_owner_states,
            post=self.post_owner_states,
            decision=self.decision,
            action=self.action,
            claim=self.claim_intent,
            disposition=self.transition_disposition,
        )
        object.__setattr__(self, "action_source", source)
        content = canonical_json_sha256_v2(self._core())
        chain = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_navigation_tick_chain_link_v2",
                "version": SCHEMA_VERSION_V2,
                "previous_tick_chain_sha256": self.previous_tick_chain_sha256,
                "tick_content_sha256": content,
            }
        )
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "chain_sha256", chain)

    def _core(self) -> dict[str, object]:
        return {
            "schema": NAVIGATION_TICK_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "tick_index": self.tick_index,
            "timestamp_binary64_hex": self.timestamp_binary64_hex,
            "synchronization_id": self.synchronization_id,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "controller_input_sha256": self.controller_input_sha256,
            "inference_receipt_sha256": self.inference_receipt_sha256,
            "per_tick_counts": self.per_tick_counts.to_dict(),
            "cumulative_counts": self.cumulative_counts.to_dict(),
            "pre_owner_states": self.pre_owner_states.to_dict(),
            "post_owner_states": self.post_owner_states.to_dict(),
            "physical_producer_receipt_sha256": self.physical_producer_receipt_sha256,
            "target_batch_receipt_sha256": self.target_batch_receipt_sha256,
            "tick_admission_receipt_sha256": self.tick_admission_receipt_sha256,
            "view_admission_sha256": self.view_admission_sha256,
            "target_outcome_sha256s": list(self.target_outcome_sha256s),
            "decision": self.decision.to_dict(),
            "action": self.action.to_dict(),
            "claim_intent": None if self.claim_intent is None else self.claim_intent.to_dict(),
            "transition_disposition": self.transition_disposition,
            "broker_execution_sha256": self.broker_execution_sha256,
            "broker_fall_sha256": self.broker_fall_sha256,
            "previous_tick_chain_sha256": self.previous_tick_chain_sha256,
            "action_source": self.action_source,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._core(),
            "content_sha256": self.content_sha256,
            "chain_sha256": self.chain_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NavigationTickRecordV2":
        keys = {
            "schema", "version", "tick_index", "timestamp_binary64_hex",
            "synchronization_id", "reset_id", "session_id", "controller_input_sha256",
            "inference_receipt_sha256", "per_tick_counts", "cumulative_counts",
            "pre_owner_states", "post_owner_states", "physical_producer_receipt_sha256",
            "target_batch_receipt_sha256", "tick_admission_receipt_sha256",
            "view_admission_sha256", "target_outcome_sha256s", "decision", "action",
            "claim_intent", "transition_disposition", "broker_execution_sha256",
            "broker_fall_sha256", "previous_tick_chain_sha256", "action_source",
            "content_sha256", "chain_sha256",
        }
        record = require_exact_dict_v2(value, name=NAVIGATION_TICK_SCHEMA_V2, keys=keys)
        if record["schema"] != NAVIGATION_TICK_SCHEMA_V2 or record["version"] != SCHEMA_VERSION_V2:
            raise NavigationTraceV2SchemaError("tick schema/version changed")
        claimed_content = require_sha256_v2(record["content_sha256"], name="content_sha256")
        claimed_chain = require_sha256_v2(record["chain_sha256"], name="chain_sha256")
        core = dict(record)
        del core["content_sha256"]
        del core["chain_sha256"]
        if canonical_json_sha256_v2(core) != claimed_content:
            raise NavigationTraceV2HashError("tick content mismatch")
        outcomes = record["target_outcome_sha256s"]
        if type(outcomes) is not list:
            raise NavigationTraceV2SchemaError("target outcomes must be exact list")
        claim_value = record["claim_intent"]
        created = cls(
            tick_index=record["tick_index"],
            timestamp_binary64_hex=record["timestamp_binary64_hex"],
            synchronization_id=record["synchronization_id"],
            reset_id=record["reset_id"],
            session_id=record["session_id"],
            controller_input_sha256=record["controller_input_sha256"],
            inference_receipt_sha256=record["inference_receipt_sha256"],
            per_tick_counts=CallCounterPanelV2.from_dict(record["per_tick_counts"]),
            cumulative_counts=CallCounterPanelV2.from_dict(record["cumulative_counts"]),
            pre_owner_states=OwnerStateBundleV2.from_dict(record["pre_owner_states"]),
            post_owner_states=OwnerStateBundleV2.from_dict(record["post_owner_states"]),
            physical_producer_receipt_sha256=record["physical_producer_receipt_sha256"],
            target_batch_receipt_sha256=record["target_batch_receipt_sha256"],
            tick_admission_receipt_sha256=record["tick_admission_receipt_sha256"],
            view_admission_sha256=record["view_admission_sha256"],
            target_outcome_sha256s=tuple(outcomes),
            decision=decision_from_dict_v2(record["decision"]),
            action=action_from_dict_v2(record["action"]),
            claim_intent=(
                None if claim_value is None else SemanticClaimIntentV2.from_dict(claim_value)
            ),
            transition_disposition=record["transition_disposition"],
            broker_execution_sha256=record["broker_execution_sha256"],
            broker_fall_sha256=record["broker_fall_sha256"],
            previous_tick_chain_sha256=record["previous_tick_chain_sha256"],
        )
        if record["action_source"] != created.action_source or type(record["action_source"]) is not str:
            raise NavigationTraceV2SchemaError("serialized action_source is not derived")
        if created.content_sha256 != claimed_content or created.chain_sha256 != claimed_chain:
            raise NavigationTraceV2HashError("tick reconstruction mismatch")
        return created


def _canonical_no_follow_path_v2(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise NavigationTraceV2SchemaError("canonical no-follow path is invalid")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise NavigationTraceV2SchemaError("canonical no-follow path escaped or changed")
    return value


@dataclass(frozen=True)
class OpenPathResolutionEvidenceV2:
    requested_path_sha256: str
    resolution_status: str
    canonical_no_follow_path: str | None
    resolution_evidence_sha256: str
    no_follow_enforced: bool
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256_v2(self.requested_path_sha256, name="requested_path_sha256")
        if type(self.resolution_status) is not str or self.resolution_status not in {
            "canonical",
            "symlink",
            "escape",
        }:
            raise NavigationTraceV2SchemaError("resolution_status is invalid")
        require_sha256_v2(self.resolution_evidence_sha256, name="resolution_evidence_sha256")
        if type(self.no_follow_enforced) is not bool or self.no_follow_enforced is not True:
            raise NavigationTraceV2SchemaError("no-follow must be exact true")
        if self.resolution_status == "canonical":
            _canonical_no_follow_path_v2(self.canonical_no_follow_path)
        elif self.canonical_no_follow_path is not None:
            raise NavigationTraceV2SchemaError(
                "failed path resolution cannot claim a canonical path"
            )
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": OPEN_PATH_EVIDENCE_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "requested_path_sha256": self.requested_path_sha256,
            "resolution_status": self.resolution_status,
            "canonical_no_follow_path": self.canonical_no_follow_path,
            "resolution_evidence_sha256": self.resolution_evidence_sha256,
            "no_follow_enforced": self.no_follow_enforced,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "OpenPathResolutionEvidenceV2":
        record = _validate_content_v2(
            value,
            schema=OPEN_PATH_EVIDENCE_SCHEMA_V2,
            keys={
                "schema", "version", "requested_path_sha256", "resolution_status",
                "canonical_no_follow_path", "resolution_evidence_sha256",
                "no_follow_enforced", "content_sha256",
            },
        )
        created = cls(
            requested_path_sha256=record["requested_path_sha256"],
            resolution_status=record["resolution_status"],
            canonical_no_follow_path=record["canonical_no_follow_path"],
            resolution_evidence_sha256=record["resolution_evidence_sha256"],
            no_follow_enforced=record["no_follow_enforced"],
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("path evidence reconstruction mismatch")
        return created


@dataclass(frozen=True)
class OpenPolicyDecisionEvidenceV2:
    disposition: str
    actor: str
    phase: str
    role: str
    canonical_no_follow_path: str | None
    expected_sha256: str | None
    allowlist_match_kind: str
    matched_actor: str | None
    matched_phase: str | None
    policy_source_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.disposition) is not str or self.disposition not in OPEN_DISPOSITIONS_V2:
            raise NavigationTraceV2SchemaError("policy disposition is invalid")
        if type(self.actor) is not str or self.actor not in OPEN_ACTORS_V2:
            raise NavigationTraceV2SchemaError("policy actor is invalid")
        if type(self.phase) is not str or self.phase not in OPEN_PHASES_V2:
            raise NavigationTraceV2SchemaError("policy phase is invalid")
        require_identifier_v2(self.role, name="role")
        if self.canonical_no_follow_path is not None:
            _canonical_no_follow_path_v2(self.canonical_no_follow_path)
        require_optional_sha256_v2(self.expected_sha256, name="expected_sha256")
        require_sha256_v2(self.policy_source_sha256, name="policy_source_sha256")
        if type(self.allowlist_match_kind) is not str or self.allowlist_match_kind not in {
            "exact_allowlist",
            "same_actor_other_phase",
            "same_phase_other_actor",
            "none",
            "not_evaluated",
        }:
            raise NavigationTraceV2SchemaError("allowlist_match_kind is invalid")
        if self.matched_actor is not None and self.matched_actor not in OPEN_ACTORS_V2:
            raise NavigationTraceV2SchemaError("matched_actor is invalid")
        if self.matched_phase is not None and self.matched_phase not in OPEN_PHASES_V2:
            raise NavigationTraceV2SchemaError("matched_phase is invalid")
        exact_dispositions = {"allowed", "denied_duplicate", "denied_hash_mismatch"}
        if self.disposition in exact_dispositions:
            if (
                self.allowlist_match_kind != "exact_allowlist"
                or self.matched_actor != self.actor
                or self.matched_phase != self.phase
                or self.canonical_no_follow_path is None
                or self.expected_sha256 is None
            ):
                raise NavigationTraceV2SchemaError("exact allowlist policy evidence is incoherent")
        elif self.disposition == "denied_wrong_phase":
            if (
                self.allowlist_match_kind != "same_actor_other_phase"
                or self.matched_actor != self.actor
                or self.matched_phase is None
                or self.matched_phase == self.phase
                or self.canonical_no_follow_path is None
                or self.expected_sha256 is None
            ):
                raise NavigationTraceV2SchemaError("wrong-phase policy evidence is incoherent")
        elif self.disposition == "denied_wrong_actor":
            if (
                self.allowlist_match_kind != "same_phase_other_actor"
                or self.matched_phase != self.phase
                or self.matched_actor is None
                or self.matched_actor == self.actor
                or self.canonical_no_follow_path is None
                or self.expected_sha256 is None
            ):
                raise NavigationTraceV2SchemaError("wrong-actor policy evidence is incoherent")
        elif self.disposition == "denied_unexpected":
            if (
                self.allowlist_match_kind != "none"
                or self.matched_actor is not None
                or self.matched_phase is not None
                or self.canonical_no_follow_path is None
            ):
                raise NavigationTraceV2SchemaError("unexpected policy evidence is incoherent")
        else:
            if (
                self.allowlist_match_kind != "not_evaluated"
                or self.matched_actor is not None
                or self.matched_phase is not None
            ):
                raise NavigationTraceV2SchemaError("resolution denial policy must be unevaluated")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": OPEN_POLICY_EVIDENCE_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "disposition": self.disposition,
            "actor": self.actor,
            "phase": self.phase,
            "role": self.role,
            "canonical_no_follow_path": self.canonical_no_follow_path,
            "expected_sha256": self.expected_sha256,
            "allowlist_match_kind": self.allowlist_match_kind,
            "matched_actor": self.matched_actor,
            "matched_phase": self.matched_phase,
            "policy_source_sha256": self.policy_source_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "OpenPolicyDecisionEvidenceV2":
        keys = {
            "schema", "version", "disposition", "actor", "phase", "role",
            "canonical_no_follow_path", "expected_sha256", "allowlist_match_kind",
            "matched_actor", "matched_phase", "policy_source_sha256", "content_sha256",
        }
        record = _validate_content_v2(value, schema=OPEN_POLICY_EVIDENCE_SCHEMA_V2, keys=keys)
        created = cls(
            **{name: record[name] for name in keys - {"schema", "version", "content_sha256"}}
        )
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("policy evidence reconstruction mismatch")
        return created


EMPTY_OPEN_LEDGER_CHAIN_SHA256_V2 = canonical_json_sha256_v2(
    {"schema": "lewm_go2_actual_open_ledger_empty_v2", "version": SCHEMA_VERSION_V2}
)


@dataclass(frozen=True)
class ActualOpenLedgerRowV2:
    actor: str
    phase: str
    role: str
    path_evidence: OpenPathResolutionEvidenceV2
    policy_evidence: OpenPolicyDecisionEvidenceV2
    expected_sha256: str | None
    actual_sha256: str | None
    bytes_opened: bool
    duplicate_of_row_sha256: str | None
    sequence: int
    previous_row_sha256: str
    disposition: str
    row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.actor) is not str or self.actor not in OPEN_ACTORS_V2:
            raise NavigationTraceV2SchemaError("open actor is invalid")
        if type(self.phase) is not str or self.phase not in OPEN_PHASES_V2:
            raise NavigationTraceV2SchemaError("open phase is invalid")
        require_identifier_v2(self.role, name="role")
        if type(self.path_evidence) is not OpenPathResolutionEvidenceV2:
            raise NavigationTraceV2SchemaError("path evidence exact type changed")
        if type(self.policy_evidence) is not OpenPolicyDecisionEvidenceV2:
            raise NavigationTraceV2SchemaError("policy evidence exact type changed")
        if (
            self.policy_evidence.actor != self.actor
            or self.policy_evidence.phase != self.phase
            or self.policy_evidence.role != self.role
        ):
            raise NavigationTraceV2SchemaError("policy evidence open key changed")
        require_optional_sha256_v2(self.expected_sha256, name="expected_sha256")
        require_optional_sha256_v2(self.actual_sha256, name="actual_sha256")
        require_optional_sha256_v2(
            self.duplicate_of_row_sha256, name="duplicate_of_row_sha256"
        )
        if type(self.bytes_opened) is not bool:
            raise NavigationTraceV2SchemaError("bytes_opened must be exact bool")
        require_nonnegative_int_v2(self.sequence, name="sequence")
        require_sha256_v2(self.previous_row_sha256, name="previous_row_sha256")
        if type(self.disposition) is not str or self.disposition not in OPEN_DISPOSITIONS_V2:
            raise NavigationTraceV2SchemaError("open disposition is invalid")
        if self.policy_evidence.disposition != self.disposition:
            raise NavigationTraceV2SchemaError("policy disposition differs from row")
        if self.policy_evidence.expected_sha256 != self.expected_sha256:
            raise NavigationTraceV2SchemaError("policy expected hash differs from row")
        if (
            self.policy_evidence.canonical_no_follow_path
            != self.path_evidence.canonical_no_follow_path
        ):
            raise NavigationTraceV2SchemaError("policy/path canonical path differs")
        self._validate_disposition_without_prior()
        object.__setattr__(self, "row_sha256", canonical_json_sha256_v2(self._core()))

    @property
    def canonical_no_follow_path(self) -> str | None:
        return self.path_evidence.canonical_no_follow_path

    @property
    def requested_path_sha256(self) -> str:
        return self.path_evidence.requested_path_sha256

    def _validate_disposition_without_prior(self) -> None:
        canonical = self.canonical_no_follow_path
        if self.disposition == "allowed":
            if (
                self.path_evidence.resolution_status != "canonical"
                or canonical is None
                or self.expected_sha256 is None
                or self.actual_sha256 != self.expected_sha256
                or self.bytes_opened is not True
                or self.duplicate_of_row_sha256 is not None
            ):
                raise NavigationTraceV2SchemaError("allowed open evidence is incoherent")
        elif self.disposition == "denied_duplicate":
            if (
                self.path_evidence.resolution_status != "canonical"
                or canonical is None
                or self.expected_sha256 is None
                or self.actual_sha256 is not None
                or self.bytes_opened is not False
                or self.duplicate_of_row_sha256 is None
            ):
                raise NavigationTraceV2SchemaError("duplicate denial evidence is incoherent")
        elif self.disposition == "denied_hash_mismatch":
            if (
                self.path_evidence.resolution_status != "canonical"
                or canonical is None
                or self.expected_sha256 is None
                or self.actual_sha256 is None
                or self.actual_sha256 == self.expected_sha256
                or self.bytes_opened is not True
                or self.duplicate_of_row_sha256 is not None
            ):
                raise NavigationTraceV2SchemaError("hash-mismatch evidence is incoherent")
        else:
            if (
                self.bytes_opened is not False
                or self.actual_sha256 is not None
                or self.duplicate_of_row_sha256 is not None
            ):
                raise NavigationTraceV2SchemaError("no-open denial carries opened-byte evidence")
            if self.disposition == "denied_symlink" and self.path_evidence.resolution_status != "symlink":
                raise NavigationTraceV2SchemaError("symlink denial lacks symlink resolution")
            if self.disposition == "denied_escape" and self.path_evidence.resolution_status != "escape":
                raise NavigationTraceV2SchemaError("escape denial lacks escape resolution")
            if self.disposition in {
                "denied_wrong_phase",
                "denied_wrong_actor",
                "denied_unexpected",
            } and self.path_evidence.resolution_status != "canonical":
                raise NavigationTraceV2SchemaError("policy denial lacks canonical path")

    def _core(self) -> dict[str, object]:
        return {
            "schema": OPEN_LEDGER_ROW_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "actor": self.actor,
            "phase": self.phase,
            "role": self.role,
            "requested_path_sha256": self.requested_path_sha256,
            "canonical_no_follow_path": self.canonical_no_follow_path,
            "path_resolution_evidence_sha256": self.path_evidence.content_sha256,
            "policy_decision_evidence_sha256": self.policy_evidence.content_sha256,
            "path_evidence": self.path_evidence.to_dict(),
            "policy_evidence": self.policy_evidence.to_dict(),
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "bytes_opened": self.bytes_opened,
            "duplicate_of_row_sha256": self.duplicate_of_row_sha256,
            "sequence": self.sequence,
            "previous_row_sha256": self.previous_row_sha256,
            "disposition": self.disposition,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "row_sha256": self.row_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ActualOpenLedgerRowV2":
        keys = {
            "schema", "version", "actor", "phase", "role", "requested_path_sha256",
            "canonical_no_follow_path", "path_resolution_evidence_sha256",
            "policy_decision_evidence_sha256", "path_evidence", "policy_evidence",
            "expected_sha256", "actual_sha256", "bytes_opened", "duplicate_of_row_sha256",
            "sequence", "previous_row_sha256", "disposition", "row_sha256",
        }
        record = require_exact_dict_v2(value, name=OPEN_LEDGER_ROW_SCHEMA_V2, keys=keys)
        if record["schema"] != OPEN_LEDGER_ROW_SCHEMA_V2 or record["version"] != SCHEMA_VERSION_V2:
            raise NavigationTraceV2SchemaError("open row schema/version changed")
        claimed = require_sha256_v2(record["row_sha256"], name="row_sha256")
        path = OpenPathResolutionEvidenceV2.from_dict(record["path_evidence"])
        policy = OpenPolicyDecisionEvidenceV2.from_dict(record["policy_evidence"])
        if (
            record["requested_path_sha256"] != path.requested_path_sha256
            or record["canonical_no_follow_path"] != path.canonical_no_follow_path
            or record["path_resolution_evidence_sha256"] != path.content_sha256
            or record["policy_decision_evidence_sha256"] != policy.content_sha256
        ):
            raise NavigationTraceV2SchemaError("open row nested evidence commitment differs")
        created = cls(
            actor=record["actor"],
            phase=record["phase"],
            role=record["role"],
            path_evidence=path,
            policy_evidence=policy,
            expected_sha256=record["expected_sha256"],
            actual_sha256=record["actual_sha256"],
            bytes_opened=record["bytes_opened"],
            duplicate_of_row_sha256=record["duplicate_of_row_sha256"],
            sequence=record["sequence"],
            previous_row_sha256=record["previous_row_sha256"],
            disposition=record["disposition"],
        )
        if created.row_sha256 != claimed:
            raise NavigationTraceV2HashError("open row reconstruction mismatch")
        return created


@dataclass(frozen=True)
class OpenLedgerProjectionReferenceV2:
    source_sequence: int
    source_row_sha256: str

    def __post_init__(self) -> None:
        require_nonnegative_int_v2(self.source_sequence, name="source_sequence")
        require_sha256_v2(self.source_row_sha256, name="source_row_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sequence": self.source_sequence,
            "source_row_sha256": self.source_row_sha256,
        }


@dataclass(frozen=True)
class ActualOpenLedgerProjectionV2:
    source_ledger_sha256: str
    references: tuple[OpenLedgerProjectionReferenceV2, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256_v2(self.source_ledger_sha256, name="source_ledger_sha256")
        if type(self.references) is not tuple or any(
            type(item) is not OpenLedgerProjectionReferenceV2 for item in self.references
        ):
            raise NavigationTraceV2SchemaError("projection references must be exact tuple")
        sequences = tuple(item.source_sequence for item in self.references)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise NavigationTraceV2SchemaError("projection source sequences changed or duplicated")
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": OPEN_PROJECTION_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "source_ledger_sha256": self.source_ledger_sha256,
            "references": [item.to_dict() for item in self.references],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ActualOpenLedgerProjectionV2":
        record = _validate_content_v2(
            value,
            schema=OPEN_PROJECTION_SCHEMA_V2,
            keys={"schema", "version", "source_ledger_sha256", "references", "content_sha256"},
        )
        refs = record["references"]
        if type(refs) is not list:
            raise NavigationTraceV2SchemaError("projection references must be exact list")
        parsed = []
        for item in refs:
            row = require_exact_dict_v2(
                item,
                name="open projection reference",
                keys={"source_sequence", "source_row_sha256"},
            )
            parsed.append(
                OpenLedgerProjectionReferenceV2(
                    source_sequence=row["source_sequence"],
                    source_row_sha256=row["source_row_sha256"],
                )
            )
        created = cls(record["source_ledger_sha256"], tuple(parsed))
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("projection reconstruction mismatch")
        return created


@dataclass(frozen=True)
class ActualOpenLedgerV2:
    rows: tuple[ActualOpenLedgerRowV2, ...] = ()
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or any(type(row) is not ActualOpenLedgerRowV2 for row in self.rows):
            raise NavigationTraceV2SchemaError("open ledger rows must be exact tuple")
        previous = EMPTY_OPEN_LEDGER_CHAIN_SHA256_V2
        denied_seen = False
        prior_allowed: dict[tuple[str, str, str, str, str], ActualOpenLedgerRowV2] = {}
        prior_by_hash: dict[str, ActualOpenLedgerRowV2] = {}
        for sequence, row in enumerate(self.rows):
            if denied_seen:
                raise NavigationTraceV2SchemaError("open attempt follows terminal denial")
            if row.sequence != sequence or row.previous_row_sha256 != previous:
                raise NavigationTraceV2SchemaError("open ledger sequence/chain is not contiguous")
            canonical = row.canonical_no_follow_path
            if row.disposition == "allowed":
                assert canonical is not None and row.expected_sha256 is not None
                key = (row.actor, row.phase, row.role, canonical, row.expected_sha256)
                if key in prior_allowed:
                    raise NavigationTraceV2SchemaError("repeated allowed open is invalid")
                prior_allowed[key] = row
            elif row.disposition == "denied_duplicate":
                assert canonical is not None and row.expected_sha256 is not None
                key = (row.actor, row.phase, row.role, canonical, row.expected_sha256)
                prior = prior_allowed.get(key)
                if (
                    prior is None
                    or prior.row_sha256 != row.duplicate_of_row_sha256
                    or row.duplicate_of_row_sha256 not in prior_by_hash
                ):
                    raise NavigationTraceV2SchemaError(
                        "duplicate denial does not name exact prior allowed row"
                    )
            if row.disposition != "allowed":
                denied_seen = True
            prior_by_hash[row.row_sha256] = row
            previous = row.row_sha256
        object.__setattr__(self, "content_sha256", canonical_json_sha256_v2(self._core()))

    def _core(self) -> dict[str, object]:
        return {
            "schema": OPEN_LEDGER_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core(), "content_sha256": self.content_sha256}

    def append_attempt(
        self,
        *,
        actor: str,
        phase: str,
        role: str,
        path_evidence: OpenPathResolutionEvidenceV2,
        policy_evidence: OpenPolicyDecisionEvidenceV2,
        expected_sha256: str | None,
        actual_sha256: str | None,
        bytes_opened: bool,
        duplicate_of_row_sha256: str | None,
        disposition: str,
    ) -> "ActualOpenLedgerV2":
        if self.rows and self.rows[-1].disposition != "allowed":
            raise NavigationTraceV2SchemaError("cannot append after terminal denial")
        previous = (
            self.rows[-1].row_sha256 if self.rows else EMPTY_OPEN_LEDGER_CHAIN_SHA256_V2
        )
        row = ActualOpenLedgerRowV2(
            actor=actor,
            phase=phase,
            role=role,
            path_evidence=path_evidence,
            policy_evidence=policy_evidence,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            bytes_opened=bytes_opened,
            duplicate_of_row_sha256=duplicate_of_row_sha256,
            sequence=len(self.rows),
            previous_row_sha256=previous,
            disposition=disposition,
        )
        return ActualOpenLedgerV2((*self.rows, row))

    def controller_projection(self) -> ActualOpenLedgerProjectionV2:
        return ActualOpenLedgerProjectionV2(
            source_ledger_sha256=self.content_sha256,
            references=tuple(
                OpenLedgerProjectionReferenceV2(row.sequence, row.row_sha256)
                for row in self.rows
                if row.actor == "controller"
            ),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ActualOpenLedgerV2":
        record = _validate_content_v2(
            value,
            schema=OPEN_LEDGER_SCHEMA_V2,
            keys={"schema", "version", "rows", "content_sha256"},
        )
        rows = record["rows"]
        if type(rows) is not list:
            raise NavigationTraceV2SchemaError("open ledger rows must be exact list")
        created = cls(tuple(ActualOpenLedgerRowV2.from_dict(row) for row in rows))
        if created.content_sha256 != record["content_sha256"]:
            raise NavigationTraceV2HashError("open ledger reconstruction mismatch")
        return created


@dataclass(frozen=True)
class ControllerTraceV2:
    reset_receipt: ResetReceiptV2
    ticks: tuple[NavigationTickRecordV2, ...]
    actual_open_controller_projection: ActualOpenLedgerProjectionV2
    final_owner_states: OwnerStateBundleV2 = field(init=False)
    final_tick_chain_sha256: str = field(init=False)
    semantic_claim_intent_sha256s: tuple[str, ...] = field(init=False)
    action_source_counts: tuple[tuple[str, int], ...] = field(init=False)
    terminal_status: str = field(init=False)
    inference_counts: CallCounterPanelV2 = field(init=False)
    evaluator_access_count: int = field(init=False, default=0)
    evaluator_callback_count: int = field(init=False, default=0)
    content_sha256: str = field(init=False)
    chain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reset_receipt) is not ResetReceiptV2:
            raise NavigationTraceV2SchemaError("reset receipt exact type changed")
        if type(self.ticks) is not tuple or any(
            type(tick) is not NavigationTickRecordV2 for tick in self.ticks
        ):
            raise NavigationTraceV2SchemaError("trace ticks must be exact tuple")
        if type(self.actual_open_controller_projection) is not ActualOpenLedgerProjectionV2:
            raise NavigationTraceV2SchemaError("open projection exact type changed")
        expected_pre = self.reset_receipt.initial_owner_states
        previous_chain = expected_pre.row("tick_chain").owner_content_sha256
        cumulative = CallCounterPanelV2.zero()
        claims: list[str] = []
        claim_colors: list[str] = []
        counts = {name: 0 for name in ACTION_SOURCES_V2}
        fault_seen = False
        for index, tick in enumerate(self.ticks):
            if fault_seen:
                raise NavigationTraceV2SchemaError("tick follows terminal fault rollback")
            if tick.tick_index != index:
                raise NavigationTraceV2SchemaError("trace tick indices are not contiguous")
            if (
                tick.reset_id != self.reset_receipt.reset_id
                or tick.session_id != self.reset_receipt.session_id
            ):
                raise NavigationTraceV2SchemaError("trace tick crossed reset/session")
            if tick.pre_owner_states.to_dict() != expected_pre.to_dict():
                raise NavigationTraceV2SchemaError("tick pre-owner chain is discontinuous")
            if tick.previous_tick_chain_sha256 != previous_chain:
                raise NavigationTraceV2SchemaError("tick hash chain is discontinuous")
            cumulative = cumulative.plus(tick.per_tick_counts)
            if tick.cumulative_counts != cumulative:
                raise NavigationTraceV2SchemaError("tick cumulative counters are not derived")
            if tick.claim_intent is not None:
                claims.append(tick.claim_intent.content_sha256)
                claim_colors.append(tick.claim_intent.color)
            counts[tick.action_source] += 1
            expected_pre = tick.post_owner_states
            previous_chain = tick.chain_sha256
            fault_seen = tick.transition_disposition == "terminal_fault_rollback"
        if len(claim_colors) != len(set(claim_colors)) or len(claim_colors) > 4:
            raise NavigationTraceV2SchemaError("claim projection repeats a semantic color")
        object.__setattr__(self, "final_owner_states", expected_pre)
        object.__setattr__(self, "final_tick_chain_sha256", previous_chain)
        object.__setattr__(self, "semantic_claim_intent_sha256s", tuple(claims))
        object.__setattr__(self, "action_source_counts", tuple(counts.items()))
        object.__setattr__(
            self,
            "terminal_status",
            "zero_tick_sealed" if not self.ticks else (
                "terminal_fault" if fault_seen else "completed"
            ),
        )
        object.__setattr__(self, "inference_counts", cumulative)
        content = canonical_json_sha256_v2(self._core())
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(
            self,
            "chain_sha256",
            canonical_json_sha256_v2(
                {
                    "schema": "lewm_go2_controller_trace_chain_v2",
                    "version": SCHEMA_VERSION_V2,
                    "controller_trace_content_sha256": content,
                    "final_tick_chain_sha256": previous_chain,
                }
            ),
        )

    def _core(self) -> dict[str, object]:
        return {
            "schema": CONTROLLER_TRACE_SCHEMA_V2,
            "version": SCHEMA_VERSION_V2,
            "reset_receipt": self.reset_receipt.to_dict(),
            "ticks": [tick.to_dict() for tick in self.ticks],
            "final_owner_states": self.final_owner_states.to_dict(),
            "final_tick_chain_sha256": self.final_tick_chain_sha256,
            "semantic_claim_intent_sha256s": list(self.semantic_claim_intent_sha256s),
            "action_source_counts": dict(self.action_source_counts),
            "terminal_status": self.terminal_status,
            "inference_counts": self.inference_counts.to_dict(),
            "evaluator_access_count": self.evaluator_access_count,
            "evaluator_callback_count": self.evaluator_callback_count,
            "actual_open_controller_projection": self.actual_open_controller_projection.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._core(),
            "content_sha256": self.content_sha256,
            "chain_sha256": self.chain_sha256,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes_v2(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ControllerTraceV2":
        keys = {
            "schema", "version", "reset_receipt", "ticks", "final_owner_states",
            "final_tick_chain_sha256", "semantic_claim_intent_sha256s",
            "action_source_counts", "terminal_status", "inference_counts",
            "evaluator_access_count", "evaluator_callback_count",
            "actual_open_controller_projection", "content_sha256", "chain_sha256",
        }
        record = require_exact_dict_v2(value, name=CONTROLLER_TRACE_SCHEMA_V2, keys=keys)
        if record["schema"] != CONTROLLER_TRACE_SCHEMA_V2 or record["version"] != SCHEMA_VERSION_V2:
            raise NavigationTraceV2SchemaError("controller trace schema/version changed")
        claimed_content = require_sha256_v2(record["content_sha256"], name="content_sha256")
        claimed_chain = require_sha256_v2(record["chain_sha256"], name="chain_sha256")
        core = dict(record)
        del core["content_sha256"]
        del core["chain_sha256"]
        if canonical_json_sha256_v2(core) != claimed_content:
            raise NavigationTraceV2HashError("controller trace content mismatch")
        ticks = record["ticks"]
        if type(ticks) is not list:
            raise NavigationTraceV2SchemaError("controller trace ticks must be exact list")
        created = cls(
            reset_receipt=ResetReceiptV2.from_dict(record["reset_receipt"]),
            ticks=tuple(NavigationTickRecordV2.from_dict(tick) for tick in ticks),
            actual_open_controller_projection=ActualOpenLedgerProjectionV2.from_dict(
                record["actual_open_controller_projection"]
            ),
        )
        if created.to_dict() != record:
            raise NavigationTraceV2SchemaError(
                "serialized final/projection/count/status fields are not derived"
            )
        if created.content_sha256 != claimed_content or created.chain_sha256 != claimed_chain:
            raise NavigationTraceV2HashError("controller trace reconstruction mismatch")
        return created


def parse_controller_trace_v2(raw: bytes) -> ControllerTraceV2:
    return ControllerTraceV2.from_dict(parse_canonical_json_bytes_v2(raw))


__all__ = [
    "ACTION_SOURCES_V2",
    "ActualOpenLedgerProjectionV2",
    "ActualOpenLedgerRowV2",
    "ActualOpenLedgerV2",
    "CANONICAL_COLORS_V2",
    "CallCounterPanelV2",
    "ControllerTraceV2",
    "ExplorationDecisionEvidenceV2",
    "FaultDecisionEvidenceV2",
    "FaultStopEvidenceV2",
    "FollowerCommandEvidenceV2",
    "FollowerStopEvidenceV2",
    "NavigationTickRecordV2",
    "NavigationTraceV2Error",
    "NavigationTraceV2HashError",
    "NavigationTraceV2SchemaError",
    "OpenLedgerProjectionReferenceV2",
    "OpenPathResolutionEvidenceV2",
    "OpenPolicyDecisionEvidenceV2",
    "OwnerStateBundleV2",
    "OwnerStateV2",
    "ResetReceiptV2",
    "SemanticClaimIntentV2",
    "TargetDecisionEvidenceV2",
    "advance_owner_bundle_v2",
    "canonical_binary64_hex_v2",
    "canonical_json_bytes_v2",
    "canonical_json_sha256_v2",
    "decode_canonical_binary64_hex_v2",
    "decision_from_dict_v2",
    "derive_action_source_v2",
    "initial_owner_content_sha256_v2",
    "parse_canonical_json_bytes_v2",
    "parse_controller_trace_v2",
    "require_exact_dict_v2",
    "require_identifier_v2",
    "require_nonnegative_int_v2",
    "require_optional_sha256_v2",
    "require_sha256_v2",
]
