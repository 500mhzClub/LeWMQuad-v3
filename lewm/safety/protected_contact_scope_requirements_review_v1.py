"""Prospective Stage-A requirements contract for protected Go2 contact scope.

This module is deliberately outcome-blind.  It records the frozen simulated
contact ontology, the exact protected collision-component inventory, and the
requirements evidence that is missing before any protected scope may change.
It does not load a corpus, an evaluation result, sensor evidence, or a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


EXPERIMENT_ID = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1"
SCHEMA_VERSION = "protected_contact_scope_requirements_review_v1.contract.v1"
START_COMMIT = "99cfa17cddb2aaddde69b8bdb6c3ea4a8e5ca849"
STAGE = "STAGE_A_REQUIREMENTS_ONLY"
STAGE_B_STATUS = "STAGE_B_NOT_AUTHORIZED"
PRIMARY_CLASSIFICATION = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED"
H1_TARGET = "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT"

PRESERVED_PREDECESSOR_CLASSIFICATIONS = (
    "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
    "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
    "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO",
    "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL",
    "DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED",
    "STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL",
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL",
    "REPLANNING_INTERFACE_UNRESOLVED",
    "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
    "ASSUMED_SENSOR_CONTRACT",
)
PRESERVED_EXACT_GEOMETRY_FINDING = (
    "Exact Genesis per-link geometry reconstructs immediate contact, successor "
    "contact, zero-versus-nonzero safe-action availability, and the complete "
    "two-ply viability decision."
)
PRESERVED_RANGE_FINDING = (
    "Under the current full-body, 13-link protected-contact scope, no tested "
    "one-, two-, three-, or diagnostic four-origin range arrangement reproduced "
    "that decision sufficiently."
)

HARD_PREACTION_SEPARATION = "HARD_PREACTION_SEPARATION"
CONDITIONAL_RECOVERABLE_CONTACT = "CONDITIONAL_RECOVERABLE_CONTACT"
MONITOR_AND_RECOVER = "MONITOR_AND_RECOVER"
PERMITTED_SUPPORT_OR_SELF_CONTACT = "PERMITTED_SUPPORT_OR_SELF_CONTACT"
SEVERITY_OR_REQUIREMENT_UNRESOLVED = "SEVERITY_OR_REQUIREMENT_UNRESOLVED"

REQUIREMENT_CATEGORIES = (
    HARD_PREACTION_SEPARATION,
    CONDITIONAL_RECOVERABLE_CONTACT,
    MONITOR_AND_RECOVER,
    PERMITTED_SUPPORT_OR_SELF_CONTACT,
    SEVERITY_OR_REQUIREMENT_UNRESOLVED,
)

ALLOWED_SECONDARY_CLASSIFICATIONS = (
    "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
    "DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED",
    "PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED",
    "RECOVERABILITY_REQUIREMENTS_UNRESOLVED",
    "MISSION_PROGRESS_REQUIREMENTS_PRESENT",
    "DISTRIBUTED_BODY_SENSING_CANDIDATE",
    "FULL_BODY_EXTERNAL_RANGE_SENSING_IMPLAUSIBLE",
)

PROTECTED_LINK_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
CALF_LINK_NAMES = frozenset(PROTECTED_LINK_NAMES[9:])
GROUND_CLASS = "GROUND_PLANE"

# Exact input-field names/tokens forbidden at the requirements-selection
# barrier.  Values may quote historic classification names for traceability;
# scientific values and tables are never inputs.
FORBIDDEN_OUTCOME_FIELD_NAMES = frozenset(
    {
        "frozen_contact_label",
        "native_contact",
        "exact_contact",
        "contact_outcome",
        "sensor_coverage",
        "per_link_error",
        "heldout_metrics",
        "calibration_threshold",
        "safe_action_count",
        "successor_safe_action_count",
        "h3_route_progress",
        "normalized_regret",
        "best_admissible_top_1",
        "best_admissible_top_3",
        "condition_pass",
        "gate_pass",
    }
)
FORBIDDEN_OUTCOME_KEY_TOKENS = (
    "heldout",
    "_auc",
    "average_precision",
    "false_negative_rate",
    "negative_retention",
    "safe_action",
    "route_progress",
    "normalized_regret",
    "best_admissible",
    "sensor_coverage",
    "per_link_error",
    "calibration_threshold",
    "condition_metric",
)
_BARRIER_METADATA_KEYS = frozenset(
    {"outcome_fields_used", "forbidden_outcome_fields", "forbidden_outcome_key_tokens"}
)


class ContractValidationError(ValueError):
    """Raised when a Stage-A receipt or requirements input fails closed."""


@dataclass(frozen=True)
class CollisionComponent:
    """Invariant part of one frozen Genesis collision primitive.

    ``data`` follows the frozen analytical reducer: full XYZ size for a box,
    ``(radius, length)`` for a capsule, and ``(radius,)`` for a sphere.
    Link-local transforms have small state-derived numerical variation in the
    predecessor index, so this invariant inventory binds component identity,
    primitive data, and the authoritative URDF locator instead.
    """

    geom_index: int
    link_index: int
    link_name: str
    component_id: str
    primitive: str
    data: tuple[float, ...]
    urdf_primitive: str
    urdf_lines: str
    function: str
    support_role: str
    hazard_concerns: tuple[str, ...]


@dataclass(frozen=True)
class RequirementContext:
    """Requirements facts accepted by the context-first decision function.

    These are approved requirement/consequence facts, never empirical model or
    sensor outcomes.  Incomplete evidence must remain unresolved.
    """

    robot_link_name: str
    environment_class: str
    operating_mode: str = "UNSPECIFIED"
    contact_type: str = "UNCLASSIFIED"
    duration_status: str = "UNRESOLVED"
    repetition_status: str = "UNRESOLVED"
    stability_consequence_status: str = "UNRESOLVED"
    task_consequence_status: str = "UNRESOLVED"
    uncertainty_status: str = "UNRESOLVED"
    ordinary_support_contact: bool = False
    self_contact: bool = False
    categorical_person_fragile_safety_critical_or_prohibited: bool = False
    approved_hard_separation_requirement: bool = False
    approved_conditional_recoverability_requirement: bool = False
    approved_monitor_and_recover_requirement: bool = False
    object_consequence_evidence_complete: bool = False
    physical_severity_evidence_complete: bool = False
    recovery_evidence_complete: bool = False
    approved_acceptance_criteria: bool = False


@dataclass(frozen=True)
class RequirementDecision:
    category: str
    rule_id: str
    rationale: str
    historical_h1_exclusion_preserved: bool
    scope_change_authorized: bool = False


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical encoding used by every Stage-A digest."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    if "content_digest" in value:
        raise ContractValidationError("content_digest must be attached exactly once")
    output = dict(value)
    output["content_digest"] = canonical_digest(output)
    return output


def validate_content_digest(value: Mapping[str, Any]) -> None:
    observed = value.get("content_digest")
    if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise ContractValidationError("missing or malformed content_digest")
    core = {key: item for key, item in value.items() if key != "content_digest"}
    if canonical_digest(core) != observed:
        raise ContractValidationError("content_digest mismatch")


def _component(
    geom_index: int,
    link_index: int,
    link_name: str,
    component_id: str,
    primitive: str,
    data: tuple[float, ...],
    urdf_primitive: str,
    urdf_lines: str,
    function: str,
    support_role: str,
    hazards: tuple[str, ...],
) -> CollisionComponent:
    return CollisionComponent(
        geom_index=geom_index,
        link_index=link_index,
        link_name=link_name,
        component_id=component_id,
        primitive=primitive,
        data=data,
        urdf_primitive=urdf_primitive,
        urdf_lines=urdf_lines,
        function=function,
        support_role=support_role,
        hazard_concerns=hazards,
    )


def protected_collision_components() -> tuple[CollisionComponent, ...]:
    """Return the exact invariant 13-link/27-shape component inventory."""

    no_support = "NO_INTENDED_SUPPORT_ROLE_SPECIFIED"
    calf_link_support = "HISTORICAL_CALF_LINK_PLANE_EXCLUSION_IS_LINK_GRANULAR"
    trunk_hazards = ("STRUCTURE", "PAYLOAD_OR_ELECTRONICS", "STABILITY", "ENTRAPMENT")
    head_hazards = ("SENSOR_OR_HEAD_ENVELOPE", "DAMAGE", "STABILITY", "ENTRAPMENT")
    hip_hazards = ("JOINT_OR_ACTUATOR", "STANCE_CONTROL", "STABILITY", "ENTRAPMENT")
    thigh_hazards = ("LOAD_TRANSFER", "SWING_OR_STANCE", "STABILITY", "ENTRAPMENT")
    calf_hazards = (
        "LOWER_LEG_OR_FOOT",
        "GROUND_SUPPORT",
        "STABILITY",
        "ENTANGLEMENT_OR_ENTRAPMENT",
        "RECOVERY",
    )
    rows: list[CollisionComponent] = [
        _component(0, 0, "base", "TRUNK_CHASSIS", "box", (0.3762, 0.0935, 0.114),
                   "box", "8-44", "CENTRAL_CHASSIS_PAYLOAD_AND_LEG_ROOT", no_support,
                   trunk_hazards),
        _component(1, 0, "base", "HEAD_UPPER", "capsule", (0.05, 0.09),
                   "cylinder", "45-82", "FIXED_HEAD_HOSTING_ENVELOPE", no_support,
                   head_hazards),
        _component(2, 0, "base", "HEAD_LOWER", "sphere", (0.047,),
                   "sphere", "83-120", "FIXED_HEAD_HOSTING_ENVELOPE", no_support,
                   head_hazards),
    ]
    hip_lines = ("122-175", "381-434", "640-693", "899-952")
    for offset, (name, lines) in enumerate(zip(PROTECTED_LINK_NAMES[1:5], hip_lines, strict=True)):
        rows.append(_component(
            3 + offset, 1 + offset, name, f"{name}_PROXIMAL_HIP", "capsule", (0.046, 0.04),
            "cylinder", lines, "PROXIMAL_ABDUCTION_ADDUCTION_AND_LOAD_TRANSFER",
            no_support, hip_hazards,
        ))
    thigh_lines = ("177-230", "436-489", "695-748", "954-1007")
    for offset, (name, lines) in enumerate(zip(PROTECTED_LINK_NAMES[5:9], thigh_lines, strict=True)):
        rows.append(_component(
            7 + offset, 5 + offset, name, f"{name}_UPPER_LEG", "box", (0.11, 0.0245, 0.034),
            "box", lines, "UPPER_LEG_PITCH_LOAD_TRANSFER_SWING_AND_STANCE",
            no_support, thigh_hazards,
        ))
    calf_lines = ("232-379", "491-638", "750-897", "1009-1156")
    for leg, (name, lines) in enumerate(zip(PROTECTED_LINK_NAMES[9:13], calf_lines, strict=True)):
        first = 11 + 4 * leg
        main_radius = 0.012 if leg == 0 else 0.013
        parts = (
            ("MAIN_CALF", "capsule", (main_radius, 0.12), "cylinder",
             "ARTICULATED_LOWER_LEG_LOAD_TRANSFER"),
            ("CALF_LOWER", "capsule", (0.011, 0.065), "cylinder",
             "FIXED_LOWER_LEG_EXTENSION"),
            ("CALF_LOWER_DISTAL", "capsule", (0.0155, 0.03), "cylinder",
             "FIXED_DISTAL_LOWER_LEG_EXTENSION"),
            ("FOOT", "sphere", (0.022,), "sphere", "TERMINAL_FOOT_SUPPORT"),
        )
        for part_offset, (part, primitive, data, urdf_primitive, function) in enumerate(parts):
            rows.append(_component(
                first + part_offset, 9 + leg, name, f"{name}_{part}", primitive, data,
                urdf_primitive, lines, function, calf_link_support, calf_hazards,
            ))
    result = tuple(rows)
    validate_component_inventory(result)
    return result


def validate_component_inventory(rows: Iterable[CollisionComponent | Mapping[str, Any]]) -> None:
    materialized = tuple(rows)
    if len(materialized) != 27:
        raise ContractValidationError("protected collision component count must be 27")
    normalized = [asdict(row) if isinstance(row, CollisionComponent) else dict(row) for row in materialized]
    if [int(row["geom_index"]) for row in normalized] != list(range(27)):
        raise ContractValidationError("geometry indices must be canonical 0..26")
    link_pairs = [(int(row["link_index"]), str(row["link_name"])) for row in normalized]
    observed_links: list[tuple[int, str]] = []
    for pair in link_pairs:
        if pair not in observed_links:
            observed_links.append(pair)
    if observed_links != list(enumerate(PROTECTED_LINK_NAMES)):
        raise ContractValidationError("protected link identity or order drift")
    counts = {index: link_pairs.count((index, name)) for index, name in enumerate(PROTECTED_LINK_NAMES)}
    if counts != {0: 3, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
                  8: 1, 9: 4, 10: 4, 11: 4, 12: 4}:
        raise ContractValidationError("per-link collision component cardinality drift")
    for row in normalized:
        data = tuple(float(item) for item in row["data"])
        if not data or not all(math.isfinite(item) and item > 0.0 for item in data):
            raise ContractValidationError("collision primitive data must be finite and positive")
        expected_size = {"box": 3, "capsule": 2, "sphere": 1}.get(str(row["primitive"]))
        if expected_size is None or len(data) != expected_size:
            raise ContractValidationError("collision primitive kind/data mismatch")


def component_inventory_receipt() -> dict[str, Any]:
    rows = []
    for component in protected_collision_components():
        row = asdict(component)
        row["data"] = list(component.data)
        row["hazard_concerns"] = list(component.hazard_concerns)
        rows.append(row)
    return with_content_digest({
        "schema": "protected_contact_scope_component_inventory_v1",
        "source_geometry_index_sha256":
            "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f",
        "source_geometry_contract_fields": [
            "geom_index", "link_index", "link_name", "kind", "data"
        ],
        "state_derived_local_transform_fields_are_not_requirement_authority": True,
        "protected_link_count": 13,
        "collision_component_count": 27,
        "protected_link_names": list(PROTECTED_LINK_NAMES),
        "components": rows,
    })


def normalize_environment_class(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"GROUND", "PLANE", "GROUNDPLANE", "GROUND_PLANE"}:
        return GROUND_CLASS
    return normalized


def is_preserved_historical_exclusion(
    *, robot_link_name: str, environment_class: str, self_contact: bool = False
) -> bool:
    """Reproduce only the historical pair exclusions, with no safety claim."""

    if self_contact:
        return True
    return (
        str(robot_link_name) in CALF_LINK_NAMES
        and normalize_environment_class(environment_class) == GROUND_CLASS
    )


def decide_requirement(context: RequirementContext) -> RequirementDecision:
    """Apply frozen context-first rules; missing authority fails unresolved."""

    if context.robot_link_name not in PROTECTED_LINK_NAMES:
        raise ContractValidationError(f"unknown protected link: {context.robot_link_name}")
    approval_flags = (
        context.approved_hard_separation_requirement,
        context.approved_conditional_recoverability_requirement,
        context.approved_monitor_and_recover_requirement,
    )
    if sum(bool(item) for item in approval_flags) > 1:
        raise ContractValidationError("conflicting approved requirement categories")
    context_axes = (
        context.operating_mode,
        context.contact_type,
        context.duration_status,
        context.repetition_status,
        context.stability_consequence_status,
        context.task_consequence_status,
        context.uncertainty_status,
    )
    if any(not isinstance(item, str) or not item.strip() for item in context_axes):
        raise ContractValidationError("every scope-unit context axis must be a non-empty string")
    environment_class = normalize_environment_class(context.environment_class)
    calf_ground = context.robot_link_name in CALF_LINK_NAMES and environment_class == GROUND_CLASS
    historical_exclusion = context.self_contact or calf_ground
    if context.ordinary_support_contact and not calf_ground:
        raise ContractValidationError("ordinary support contact is valid only for a calf-link/ground pair")
    if context.self_contact:
        return RequirementDecision(
            category=PERMITTED_SUPPORT_OR_SELF_CONTACT,
            rule_id="CTX_SELF_CONTACT_PRESERVED_EXCLUSION",
            rationale=(
                "unchanged historical self-contact exclusion; this is not evidence "
                "that a particular self-contact is harmless"
            ),
            historical_h1_exclusion_preserved=True,
        )
    valid_ordinary_support = (
        calf_ground
        and context.ordinary_support_contact
        and context.operating_mode == "ORDINARY_STANCE_OR_LOCOMOTION"
        and context.contact_type == "INTENDED_GROUND_SUPPORT"
        and context.duration_status == "ORDINARY_SUPPORT"
        and context.repetition_status == "ORDINARY_SUPPORT"
        and context.stability_consequence_status == "STABLE_SUPPORT"
        and context.task_consequence_status == "MISSION_SUPPORT"
        and context.uncertainty_status == "REQUIREMENTS_CONTEXT_RESOLVED"
    )
    if valid_ordinary_support:
        return RequirementDecision(
            category=PERMITTED_SUPPORT_OR_SELF_CONTACT,
            rule_id="CTX_EXPLICIT_ORDINARY_CALF_GROUND_SUPPORT",
            rationale=(
                "explicit ordinary support context is complete; the historical "
                "calf-link/Plane exclusion remains link-granular"
            ),
            historical_h1_exclusion_preserved=True,
        )
    if calf_ground:
        return RequirementDecision(
            category=SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            rule_id="CTX_CALF_GROUND_NOT_VALIDATED_AS_ORDINARY_SUPPORT",
            rationale=(
                "the historical label exclusion is preserved, but abnormal, unknown, "
                "or incompletely specified calf-ground contact is not prospectively "
                "classified as permitted support"
            ),
            historical_h1_exclusion_preserved=historical_exclusion,
        )
    if (
        context.categorical_person_fragile_safety_critical_or_prohibited
        or context.approved_hard_separation_requirement
    ):
        return RequirementDecision(
            category=HARD_PREACTION_SEPARATION,
            rule_id="CTX_APPROVED_OR_CATEGORICAL_HARD_HAZARD",
            rationale="application/object consequence requires separation before action",
            historical_h1_exclusion_preserved=historical_exclusion,
        )
    complete_physics = (
        context.object_consequence_evidence_complete
        and context.physical_severity_evidence_complete
        and context.recovery_evidence_complete
        and context.approved_acceptance_criteria
    )
    if context.approved_conditional_recoverability_requirement:
        if complete_physics:
            return RequirementDecision(
                category=CONDITIONAL_RECOVERABLE_CONTACT,
                rule_id="CTX_APPROVED_COMPLETE_CONDITIONAL_RECOVERABILITY",
                rationale="approved consequence, severity, recovery, and acceptance evidence is complete",
                historical_h1_exclusion_preserved=historical_exclusion,
            )
        return RequirementDecision(
            category=SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            rule_id="CTX_CONDITIONAL_RECOVERABILITY_EVIDENCE_INCOMPLETE",
            rationale="conditional recoverability may not be inferred from incomplete evidence",
            historical_h1_exclusion_preserved=historical_exclusion,
        )
    if context.approved_monitor_and_recover_requirement:
        if context.recovery_evidence_complete and context.approved_acceptance_criteria:
            return RequirementDecision(
                category=MONITOR_AND_RECOVER,
                rule_id="CTX_APPROVED_EFFECTIVE_MONITOR_AND_RECOVER",
                rationale="approved monitored-contact response and effective recovery criteria are complete",
                historical_h1_exclusion_preserved=historical_exclusion,
            )
        return RequirementDecision(
            category=SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            rule_id="CTX_MONITOR_AND_RECOVER_EVIDENCE_INCOMPLETE",
            rationale="a monitor without an approved effective response cannot narrow protection",
            historical_h1_exclusion_preserved=historical_exclusion,
        )
    return RequirementDecision(
        category=SEVERITY_OR_REQUIREMENT_UNRESOLVED,
        rule_id="CTX_FAIL_CLOSED_REQUIREMENTS_UNRESOLVED",
        rationale="no approved context-specific requirement and consequence evidence resolves this contact",
        historical_h1_exclusion_preserved=historical_exclusion,
    )


def _walk_mapping_keys(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            yield path, key
            yield from _walk_mapping_keys(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_mapping_keys(item, (*path, str(index)))


def assert_requirements_only_payload(value: Any) -> None:
    """Reject scientific outcome fields anywhere in a requirements input."""

    for path, raw_key in _walk_mapping_keys(value):
        key = raw_key.strip().lower()
        if key in _BARRIER_METADATA_KEYS:
            continue
        if key in FORBIDDEN_OUTCOME_FIELD_NAMES or any(token in key for token in FORBIDDEN_OUTCOME_KEY_TOKENS):
            location = ".".join((*path, raw_key))
            raise ContractValidationError(f"scientific outcome field forbidden at requirements barrier: {location}")


def build_source_inventory() -> dict[str, Any]:
    """Build immutable source citations without opening any source or result."""

    sources = (
        ("SRC-GEOMETRY-BINDING", "lewm/safety/body_centric_range_coverage_corpus_v1.py",
         "817ec5de371b855ac956b350c095e92a723970741b60a7a31e0a7ffc3e5f819a",
         "88-150,1010-1029,1365-1428", "FROZEN_GEOMETRY_IDENTITY"),
        ("SRC-GEOMETRY-INDEX", ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/geometry_index.json",
         "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f",
         "3409-4096 (first repeated invariant component contract)", "FROZEN_COMPONENT_INVENTORY_ONLY"),
        ("SRC-GENESIS-URDF", ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
         "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4",
         "8-1156", "SIMULATED_ROBOT_COLLISION_ASSET"),
        ("SRC-H1", "lewm/oracle/go2_branch_oracle_v1_2.py",
         "6d7a6b20bcfb5da112ff10e95a7d3573ebf07884e7b4e58315a733254d6f4fc2",
         "74-109,287-319", "HISTORICAL_CONTACT_PAIR_ONTOLOGY"),
        ("SRC-ONTOLOGY", "lewm/safety/contact_hazard_ontology_v1.py",
         "69550fe787e84331560013678abed3aab58f573719b7198c53bfef786fb6204b",
         "1-45,82-95,141-214,237-336", "PROSPECTIVE_REQUIREMENTS_ONTOLOGY"),
        ("SRC-INSTRUMENTATION", "docs/lewm_contact_hazard_instrumentation_contract_v1.md",
         "71e120fc9e8db1040b02d64363b53ce5ca0d96d0f680e46fce5d28e85c300747",
         "13-40", "EVIDENCE_AVAILABILITY_AND_LIMITATIONS"),
        ("SRC-HAZARD-REVIEW", "docs/lewm_contact_hazard_analysis_and_ontology_v1.md",
         "53edab3192dc3a0da79435f7472ae01c234d6787cebbd71cfad63e1696c3389f",
         "7-61", "HAZARD_AND_REQUIREMENT_BOUNDARY"),
        ("SRC-TASK", "docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md",
         "2ebe307cf7ea1bc6902f6214c875cd4abde1c3194c564d4abeba2dc1f08bb959",
         "3-30", "BOUNDED_LOCAL_TASK_REQUIREMENTS"),
        ("SRC-PLANNER", "docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md",
         "b035ce80bd1d639f8d683472166807eff4e648f542a7605cf2d54c95b3a31929",
         "3-5,28-52", "NONCOMPENSABLE_SAFETY_AND_TASK_REQUIREMENTS"),
        ("SRC-PLATFORM", "config/go2_platform_manifest.yaml",
         "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
         "15-33,35-72,93-133", "PLATFORM_AND_UNRESOLVED_PARITY_BOUNDARY"),
        ("SRC-PRIMITIVES", "config/go2_primitive_registry.yaml",
         "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8",
         "13-124", "AVAILABLE_AND_DISABLED_CONTROL_MODES"),
        ("SRC-WORLDS", "lewm_worlds/lewm_worlds/families.py",
         "87515b932773aa518d18f9777af7c1567ae76368ce31e5519da7019f72d30189",
         "317-453,1697-1777", "REPRESENTED_AND_NONREPRESENTED_SCENE_FAMILIES"),
        ("SRC-ROS-XACRO", "third_party/unitree_go2_ros2/unitree_go2_description/urdf/leg.xacro",
         "a1d993ac67399602a33b4cfa0edea874dfcc0394519cf3606402025b9892a3ec",
         "9-172", "NONPARITY_ROS_COLLISION_DESCRIPTION"),
    )
    rows = [
        {"source_id": source_id, "path": path, "sha256": sha256,
         "line_ranges": line_ranges, "authority": authority,
         "scientific_outcomes_accessed": False}
        for source_id, path, sha256, line_ranges, authority in sources
    ]
    return with_content_digest({
        "schema": "protected_contact_scope_source_inventory_v1",
        "start_commit": START_COMMIT,
        "sources": rows,
        "outcome_fields_used": [],
    })


def build_assumption_inventory() -> dict[str, Any]:
    assumptions = [
        {"assumption_id": "PCS-A-001", "statement":
         "The current use case is bounded indoor local-waypoint research, not an approved deployment mission.",
         "status": "BOUNDARY_ONLY", "scope_change_authority": False},
        {"assumption_id": "PCS-A-002", "statement":
         "The Genesis 13-link/27-shape envelope is historical simulated evidence, not a validated physical or ROS/Gazebo collision envelope.",
         "status": "SIM_TO_PLATFORM_PARITY_UNRESOLVED", "scope_change_authority": False},
        {"assumption_id": "PCS-A-003", "statement":
         "Current maze objects are fixed ground, wall boxes, and landmark boxes without consequence-complete material, mass, fragility, person, safety-critical, or damage metadata.",
         "status": "OBJECT_CONSEQUENCE_EVIDENCE_MISSING", "scope_change_authority": False},
        {"assumption_id": "PCS-A-004", "statement":
         "The historical calf-link/Plane exclusion applies to the complete collapsed calf link and is not reinterpreted as foot-sphere-only support.",
         "status": "PRESERVED_HISTORICAL_SEMANTICS", "scope_change_authority": False},
        {"assumption_id": "PCS-A-005", "statement":
         "Perfect sensing supplies occurrence and attribution but not severity, acceptable consequence, recovery, stopping, ODD, or mission requirements.",
         "status": "REQUIREMENTS_REMAIN_UNRESOLVED", "scope_change_authority": False},
        {"assumption_id": "PCS-A-006", "statement":
         "Available reverse, yaw, hold, or curriculum behavior does not establish effective recovery, self-righting, stopping parity, or operator handoff.",
         "status": "RECOVERY_REQUIREMENTS_UNRESOLVED", "scope_change_authority": False},
    ]
    unresolved_evidence = [
        {
            "evidence_id": "PCS-UE-ODD-001",
            "needed": "approved search or inspection application, ODD, human presence, asset classes, supervision, and mission consequences",
            "current_status": "MISSING",
            "decision_affected": "deployment contact scope and task-performance requirements",
        },
        {
            "evidence_id": "PCS-UE-OBJECT-001",
            "needed": "object material, mass, mobility, fragility, safety criticality, person/proxy, permission, prohibition, and damage metadata",
            "current_status": "MISSING_FROM_CURRENT_MAZE",
            "decision_affected": "hard separation versus unresolved object context",
        },
        {
            "evidence_id": "PCS-UE-CONSEQUENCE-001",
            "needed": "approved component/object force or validated substitute, impulse, speed, duration, repetition, penetration, stability, entrapment, and damage limits",
            "current_status": "NO_APPROVED_LIMITS",
            "decision_affected": "material hazard and conditional recoverability",
        },
        {
            "evidence_id": "PCS-UE-COLLISION-PARITY-001",
            "needed": "Genesis, ROS/Gazebo, payload, and physical Go2 collision-envelope reconciliation",
            "current_status": "NON_EQUIVALENT_DESCRIPTIONS",
            "decision_affected": "physical region and component scope",
        },
        {
            "evidence_id": "PCS-UE-INSTRUMENTATION-001",
            "needed": "physical contact instrumentation and Genesis force/impulse calibration or a validated alternative",
            "current_status": "MISSING",
            "decision_affected": "consequence threshold verification",
        },
        {
            "evidence_id": "PCS-UE-STOP-001",
            "needed": "platform-equivalent stopping mode, request/acknowledgement latency, stopping distributions, and uncertainty margin",
            "current_status": "PARITY_PENDING",
            "decision_affected": "preaction clearance and minimum-risk response",
        },
        {
            "evidence_id": "PCS-UE-RECOVERY-001",
            "needed": "stuck/entrapment definition, successful separation and restored-capability criteria, recovery budget, repeat limit, and operator escalation",
            "current_status": "MISSING",
            "decision_affected": "conditional recoverability and monitor/recover effectiveness",
        },
        {
            "evidence_id": "PCS-UE-TASK-001",
            "needed": "stakeholder-approved progress, coverage, revisit, inspection completeness, deadline, and missed-hazard consequence",
            "current_status": "MISSING",
            "decision_affected": "safety-related task performance",
        },
        {
            "evidence_id": "PCS-UE-ASSURANCE-001",
            "needed": "formal hazard log, requirement ownership/allocation, acceptance criteria, verification independence, and SACE/AMLAS argument review",
            "current_status": "INCOMPLETE",
            "decision_affected": "requirements sufficiency and deployment claim",
        },
    ]
    return with_content_digest({
        "schema": "protected_contact_scope_assumption_inventory_v1",
        "assumptions": assumptions,
        "unresolved_evidence": unresolved_evidence,
        "unresolved_evidence_count": len(unresolved_evidence),
        "outcome_fields_used": [],
    })


def build_traceability_matrix() -> dict[str, Any]:
    rows = []
    for component in protected_collision_components():
        historical_override = (
            "SELF_CONTACT_AND_CALF_LINK_PLANE_PAIR_ONLY"
            if component.link_name in CALF_LINK_NAMES else "SELF_CONTACT_ONLY"
        )
        rows.append({
            "trace_id": f"PCS-GEOM-{component.geom_index:03d}",
            "geom_index": component.geom_index,
            "link_index": component.link_index,
            "link_name": component.link_name,
            "component_id": component.component_id,
            "function": component.function,
            "function_authority": "URDF_FACT_WITH_EXPLICIT_ENGINEERING_HAZARD_HYPOTHESES",
            "hazard_concerns": list(component.hazard_concerns),
            "context_dimensions_required": [
                "environment_object_and_consequence_class",
                "support_or_swing_mode",
                "payload_and_operating_domain",
                "contact_severity_and_repetition",
                "stability_and_entrapment_consequence",
                "effective_stop_or_recovery_response",
                "mission_progress_or_completion_consequence",
            ],
            "context_decision_default": SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            "historical_exclusion_override": historical_override,
            "current_disallowed_context_disposition": "PRESERVE_HISTORICAL_PROTECTED_SCOPE",
            "scope_change_authorized": False,
            "stage_b_status": STAGE_B_STATUS,
        })
    hazard_rows = [
        {
            "hazard_id": "PCS-HZ-PERSON",
            "hazard": "person collision",
            "safety_requirement": "PCS-SR-CONTACT-001",
            "robot_region": "ALL_PROTECTED_COMPONENTS",
            "object_context": "PERSON_OR_PERSON_PROXY_NOT_REPRESENTED_IN_CURRENT_TEST_ENVIRONMENT",
            "required_treatment": HARD_PREACTION_SEPARATION,
            "evidence": "contact hazard analysis and prospective person object field",
            "assumption": "person consequence requires separation unless an approved hazard process states otherwise",
            "residual_limitation": "no person model, injury criterion, detector, or approved separation distance",
        },
        {
            "hazard_id": "PCS-HZ-FRAGILE-CRITICAL",
            "hazard": "fragile or safety-critical infrastructure contact",
            "safety_requirement": "PCS-SR-CONTACT-001",
            "robot_region": "ALL_PROTECTED_COMPONENTS",
            "object_context": "FRAGILE_CRITICAL_OR_PROHIBITED_ASSET_NOT_REPRESENTED_IN_CURRENT_TEST_ENVIRONMENT",
            "required_treatment": HARD_PREACTION_SEPARATION,
            "evidence": "object-consequence schema and hazard analysis",
            "assumption": "object class and prohibition are available before action",
            "residual_limitation": "no fragility, criticality, material, damage, or permission metadata",
        },
        {
            "hazard_id": "PCS-HZ-TRUNK-PAYLOAD",
            "hazard": "trunk or payload impact",
            "safety_requirement": "PCS-SR-CONTACT-001_AND_PCS-SR-STABILITY-001",
            "robot_region": "BASE_G0_HEAD_PAYLOAD_ENVELOPE_G1_G2",
            "object_context": "CURRENT_FIXED_WALL_OR_LANDMARK_OR_DEPLOYMENT_ASSET",
            "required_treatment": SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            "evidence": "Genesis component inventory and platform manifest",
            "assumption": "simulated geometry is evidence rather than a physical payload envelope",
            "residual_limitation": "no payload load case, damage limit, impact calibration, or collision-envelope parity",
        },
        {
            "hazard_id": "PCS-HZ-PROXIMAL-ENTRAPMENT",
            "hazard": "proximal limb entrapment",
            "safety_requirement": "PCS-SR-STABILITY-001_AND_PCS-SR-RECOVERY-001",
            "robot_region": "HIPS_G3_G6_AND_THIGHS_G7_G10",
            "object_context": "GAP_WALL_LANDMARK_OR_MOVING_STRUCTURE_ONLY_FIXED_BOXES_REPRESENTED",
            "required_treatment": SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            "evidence": "URDF function and stuck-or-entrapment ontology annotation",
            "assumption": "joint obstruction is plausible but not established by link identity",
            "residual_limitation": "no entrapment definition, saturation consequence, dynamic object, or recovery envelope",
        },
        {
            "hazard_id": "PCS-HZ-DISTAL-CALF",
            "hazard": "distal limb or calf contact",
            "safety_requirement": "PCS-SR-CONTACT-001_PCS-SR-STABILITY-001_PCS-SR-RECOVERY-001",
            "robot_region": "CALF_LOWER_LEG_AND_FOOT_G11_G26",
            "object_context": "NON_GROUND_OR_ABNORMAL_OR_UNKNOWN_GROUND",
            "required_treatment": SEVERITY_OR_REQUIREMENT_UNRESOLVED,
            "evidence": "historical calf-link Plane exclusion and component inventory",
            "assumption": "primitive and phase can distinguish intended support from abnormal contact",
            "residual_limitation": "link-level H1 exclusion supplies no support phase, severity, or entanglement criterion",
        },
        {
            "hazard_id": "PCS-HZ-STABILITY-FALL",
            "hazard": "destabilisation or fall",
            "safety_requirement": "PCS-SR-STABILITY-001",
            "robot_region": "ANY_REGION_AFFECTING_BODY_OR_SUPPORT_STATE",
            "object_context": "ANY_OBJECT_AND_OPERATING_MODE",
            "required_treatment": "HARD_IF_DEMONSTRATED_OR_PROSPECTIVELY_REQUIRED_OTHERWISE_UNRESOLVED",
            "evidence": "contact ontology stability and fall fields",
            "assumption": "stability consequence can be prospectively detected and bounded",
            "residual_limitation": "no approved fall limit, physical calibration, or minimum-risk response",
        },
        {
            "hazard_id": "PCS-HZ-REPEATED-CONTACT",
            "hazard": "repeated contact",
            "safety_requirement": "PCS-SR-RECOVERY-001",
            "robot_region": "ANY_CONTACTING_COMPONENT",
            "object_context": "REPEATED_EXTERNAL_CONTACT",
            "required_treatment": "PRIMARY_CONTACT_CATEGORY_PLUS_MANDATORY_RECOVERY_OVERLAY",
            "evidence": "ontology repetition and task annotations",
            "assumption": "a repetition budget and effective response can be specified",
            "residual_limitation": "no approved repeat, recovery-cycle, or escalation limit",
        },
        {
            "hazard_id": "PCS-HZ-CONTACT-STUCK",
            "hazard": "contact followed by stuck",
            "safety_requirement": "PCS-SR-RECOVERY-001",
            "robot_region": "ANY_CONTACTING_COMPONENT",
            "object_context": "CONTACT_PLUS_INEFFECTIVE_COMMANDED_MOTION",
            "required_treatment": "PRIMARY_CONTACT_CATEGORY_PLUS_MANDATORY_RECOVERY_OVERLAY",
            "evidence": "stuck detector and prospective contact ontology",
            "assumption": "stuck is a useful monitor signal",
            "residual_limitation": "stuck does not identify entrapment, controller failure, deliberate hold, or recoverability",
        },
        {
            "hazard_id": "PCS-HZ-PROGRESS",
            "hazard": "loss of task progress",
            "safety_requirement": "PCS-SR-TASK-001",
            "robot_region": "SYSTEM_TASK_LEVEL",
            "object_context": "NO_CONTACT_OR_CONTACT_ASSOCIATED_LOCAL_WAYPOINT_OPERATION",
            "required_treatment": MONITOR_AND_RECOVER,
            "evidence": "progression non-performance text and local-waypoint specification",
            "assumption": "the application can define minimum meaningful progress",
            "residual_limitation": "no approved progress, deadline, intervention, or mission-loss criterion",
        },
        {
            "hazard_id": "PCS-HZ-ABSTENTION",
            "hazard": "repeated abstention",
            "safety_requirement": "PCS-SR-TASK-001_AND_PCS-SR-FALLBACK-001",
            "robot_region": "INTEGRATED_SYSTEM",
            "object_context": "REPEATED_VETO_OR_STOP_WITHOUT_CONTACT",
            "required_treatment": MONITOR_AND_RECOVER,
            "evidence": "progression document and factorised planner design",
            "assumption": "abstention is observable and a bounded fallback can be implemented",
            "residual_limitation": "no approved abstention count, operator handoff, or mission consequence",
        },
        {
            "hazard_id": "PCS-HZ-INSPECTION-INCOMPLETE",
            "hazard": "failure to complete inspection or search",
            "safety_requirement": "PCS-SR-INSPECTION-001",
            "robot_region": "MISSION_SYSTEM",
            "object_context": "MISSED_ASSET_AREA_COVERAGE_OR_DEADLINE_NOT_REPRESENTED_IN_CURRENT_TEST_ENVIRONMENT",
            "required_treatment": MONITOR_AND_RECOVER,
            "evidence": "progression responsible-research and inspection discussion",
            "assumption": "deployment stakeholders can define inspection sufficiency",
            "residual_limitation": "no approved coverage, revisit, deadline, or missed-hazard consequence",
        },
    ]
    result = with_content_digest({
        "schema": "protected_contact_scope_traceability_matrix_v1",
        "decision_unit": "COLLISION_COMPONENT_X_CONTACT_CONTEXT",
        "link_rows_are_aggregates_only": True,
        "hazard_rows": hazard_rows,
        "hazard_row_count": len(hazard_rows),
        "rows": rows,
        "row_count": len(rows),
        "outcome_fields_used": [],
    })
    validate_traceability_matrix(result)
    return result


def validate_traceability_matrix(matrix: Mapping[str, Any]) -> None:
    validate_content_digest(matrix)
    rows = matrix.get("rows")
    hazards = matrix.get("hazard_rows")
    if not isinstance(rows, list) or len(rows) != 27 or matrix.get("row_count") != 27:
        raise ContractValidationError("component traceability cardinality drift")
    if not isinstance(hazards, list) or len(hazards) != 11 or matrix.get("hazard_row_count") != 11:
        raise ContractValidationError("hazard traceability cardinality drift")
    required = {
        "hazard_id", "hazard", "safety_requirement", "robot_region",
        "object_context", "required_treatment", "evidence", "assumption",
        "residual_limitation",
    }
    if any(set(row) != required for row in hazards):
        raise ContractValidationError("hazard traceability field set drift")
    if len({row["hazard_id"] for row in hazards}) != len(hazards):
        raise ContractValidationError("hazard traceability identities are not unique")


def _matrix_context_templates() -> tuple[tuple[str, bool, dict[str, Any]], ...]:
    common = {
        "operating_mode": "ANY_OPERATION",
        "contact_type": "EXTERNAL_ENVIRONMENT_CONTACT",
        "duration_status": "UNRESOLVED",
        "repetition_status": "UNRESOLVED",
        "stability_consequence_status": "UNRESOLVED",
        "task_consequence_status": "UNRESOLVED",
        "uncertainty_status": "UNRESOLVED",
    }
    hard = {**common, "categorical_person_fragile_safety_critical_or_prohibited": True}
    return (
        ("PERSON", False, {**hard, "environment_class": "PERSON"}),
        ("FRAGILE_OR_SAFETY_CRITICAL_ASSET", False,
         {**hard, "environment_class": "FRAGILE_OR_SAFETY_CRITICAL_ASSET"}),
        ("PROHIBITED_OBJECT_OR_REGION", False,
         {**hard, "environment_class": "PROHIBITED_OBJECT_OR_REGION"}),
        ("FIXED_WALL", True, {**common, "environment_class": "FIXED_WALL"}),
        ("LANDMARK_BOX", True, {**common, "environment_class": "LANDMARK_BOX"}),
        ("GROUND_ORDINARY_SUPPORT", True, {
            "environment_class": GROUND_CLASS,
            "operating_mode": "ORDINARY_STANCE_OR_LOCOMOTION",
            "contact_type": "INTENDED_GROUND_SUPPORT",
            "duration_status": "ORDINARY_SUPPORT",
            "repetition_status": "ORDINARY_SUPPORT",
            "stability_consequence_status": "STABLE_SUPPORT",
            "task_consequence_status": "MISSION_SUPPORT",
            "uncertainty_status": "REQUIREMENTS_CONTEXT_RESOLVED",
        }),
        ("GROUND_ABNORMAL_OR_UNKNOWN", True, {
            "environment_class": GROUND_CLASS,
            "operating_mode": "ABNORMAL_OR_UNKNOWN",
            "contact_type": "UNCLASSIFIED_GROUND_CONTACT",
            "duration_status": "UNRESOLVED",
            "repetition_status": "UNRESOLVED",
            "stability_consequence_status": "UNRESOLVED",
            "task_consequence_status": "UNRESOLVED",
            "uncertainty_status": "UNRESOLVED",
        }),
        ("ROBOT_SELF_CONTACT", True, {
            **common,
            "environment_class": "ROBOT_BODY",
            "contact_type": "SELF_CONTACT",
            "self_contact": True,
        }),
        ("UNCLASSIFIED_ENVIRONMENT_OBJECT", False,
         {**common, "environment_class": "UNCLASSIFIED_ENVIRONMENT_OBJECT"}),
    )


def build_link_object_context_matrix() -> dict[str, Any]:
    """Expand every frozen component over the prospective context classes.

    The matrix is a requirements counterfactual, not an empirical result.  Its
    rows are generated exclusively from the context-first decision function.
    """

    rows: list[dict[str, Any]] = []
    templates = _matrix_context_templates()
    for component in protected_collision_components():
        for context_id, represented, template in templates:
            values = dict(template)
            values["robot_link_name"] = component.link_name
            # Ordinary support is link/context specific.  The historic H1
            # exclusion alone never sets this prospective requirements fact.
            values["ordinary_support_contact"] = (
                context_id == "GROUND_ORDINARY_SUPPORT"
                and component.link_name in CALF_LINK_NAMES
            )
            context = RequirementContext(**values)
            decision = decide_requirement(context)
            rows.append({
                "matrix_row_id": f"PCS-CONTEXT-{component.geom_index:03d}-{context_id}",
                "geom_index": component.geom_index,
                "link_index": component.link_index,
                "link_name": component.link_name,
                "component_id": component.component_id,
                "context_id": context_id,
                "represented_in_current_simulator": represented,
                "context": asdict(context),
                "category": decision.category,
                "rule_id": decision.rule_id,
                "historical_h1_exclusion_preserved":
                    decision.historical_h1_exclusion_preserved,
                "scope_change_authorized": decision.scope_change_authorized,
                "perfect_sensing_counterfactual": True,
            })
    category_counts = {
        category: sum(row["category"] == category for row in rows)
        for category in REQUIREMENT_CATEGORIES
    }
    result = with_content_digest({
        "schema": "protected_contact_scope_link_object_context_matrix_v1",
        "decision_unit": "COLLISION_COMPONENT_X_OBJECT_CONTACT_CONTEXT",
        "component_count": 27,
        "context_count": len(templates),
        "expected_row_count": 27 * len(templates),
        "context_ids": [item[0] for item in templates],
        "every_row_has_exactly_one_category": True,
        "perfect_sensing_counterfactual": True,
        "category_counts": category_counts,
        "rows": rows,
        "outcome_fields_used": [],
    })
    validate_link_object_context_matrix(result)
    return result


def validate_link_object_context_matrix(matrix: Mapping[str, Any]) -> None:
    validate_content_digest(matrix)
    templates = _matrix_context_templates()
    context_ids = tuple(item[0] for item in templates)
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != 27 * len(templates):
        raise ContractValidationError("link/object/context matrix cardinality drift")
    if matrix.get("component_count") != 27 or matrix.get("context_count") != len(templates):
        raise ContractValidationError("matrix axis cardinality drift")
    if tuple(matrix.get("context_ids", ())) != context_ids:
        raise ContractValidationError("matrix context identity or order drift")
    expected_pairs = [
        (component.geom_index, context_id)
        for component in protected_collision_components()
        for context_id in context_ids
    ]
    observed_pairs = [(row.get("geom_index"), row.get("context_id")) for row in rows]
    if observed_pairs != expected_pairs:
        raise ContractValidationError("matrix component/context exhaustion or order drift")
    for row in rows:
        if row.get("category") not in REQUIREMENT_CATEGORIES:
            raise ContractValidationError("matrix row lacks exactly one authorized category")
        context_data = row.get("context")
        if not isinstance(context_data, Mapping):
            raise ContractValidationError("matrix row context missing")
        decision = decide_requirement(RequirementContext(**dict(context_data)))
        if row.get("category") != decision.category or row.get("rule_id") != decision.rule_id:
            raise ContractValidationError("matrix decision does not reproduce context-first rule")
        if row.get("historical_h1_exclusion_preserved") is not decision.historical_h1_exclusion_preserved:
            raise ContractValidationError("matrix historical exclusion receipt drift")
        if row.get("scope_change_authorized") is not False:
            raise ContractValidationError("Stage-A matrix authorized a scope change")
        if row.get("perfect_sensing_counterfactual") is not True:
            raise ContractValidationError("matrix row is not explicitly counterfactual")
    counts = {
        category: sum(row.get("category") == category for row in rows)
        for category in REQUIREMENT_CATEGORIES
    }
    if matrix.get("category_counts") != counts:
        raise ContractValidationError("matrix category-count receipt drift")
    if counts[CONDITIONAL_RECOVERABLE_CONTACT] != 0:
        raise ContractValidationError("Stage-A matrix inferred conditional recoverability")
    if matrix.get("outcome_fields_used") != []:
        raise ContractValidationError("matrix used scientific outcomes")
    assert_requirements_only_payload(matrix)


def build_perfect_sensing_counterfactual() -> dict[str, Any]:
    return with_content_digest({
        "schema": "protected_contact_scope_perfect_sensing_counterfactual_v1",
        "assumption": (
            "Every current and future full-body clearance/contact event is observed exactly, "
            "with zero latency and exact link, shape, object, time, and geometry attribution."
        ),
        "could_establish": [
            "simulated contact occurrence",
            "simulated geometric separation",
            "simulated link/shape/object attribution",
            "simulated event timing",
        ],
        "cannot_establish": [
            "material, human, property, fragile-asset, or safety-critical consequence",
            "approved force, impulse, speed, duration, penetration, repetition, or damage limit",
            "acceptable fall, stability, entrapment, or control-loss consequence",
            "effective platform stopping or recovery response",
            "approved operating domain and stakeholder risk tolerance",
            "acceptable mission progress, coverage, completion, or missed-inspection consequence",
        ],
        "primary_classification": PRIMARY_CLASSIFICATION,
        "protected_scope_change_authorized": False,
        "stage_b_status": STAGE_B_STATUS,
        "outcome_fields_used": [],
    })


def build_stage_b_gate() -> dict[str, Any]:
    blockers = [
        "approved application and operational design domain",
        "stakeholder-reviewed harm and mission non-performance consequences",
        "object material, mass, mobility, fragility, person, safety-critical, and damage metadata",
        "approved component/object severity, stability, entrapment, and damage limits",
        "validated physical instrumentation or permanent simulated-proxy restriction",
        "platform-equivalent stop request, acknowledgement, response time, and stopping envelope",
        "successful separation, restored capability, recovery budget, repetition, and handoff requirements",
        "system allocation, acceptance criteria, traceability, and independent verification plan",
    ]
    return with_content_digest({
        "schema": "protected_contact_scope_stage_b_gate_v1",
        "status": STAGE_B_STATUS,
        "authorized": False,
        "blockers": blockers,
        "prohibitions": [
            "remove a protected link, shape, body region, direction, or swept volume",
            "reinterpret or change a completed contact label",
            "select a narrower scope using sensor, model, calibration, or evaluation performance",
            "collect a new panel or train a model under a narrowed scope",
        ],
        "outcome_fields_used": [],
    })


def build_requirements_sufficiency_gate() -> dict[str, Any]:
    """Evaluate the prompt's scope-freeze sufficiency criteria explicitly."""

    criteria = [
        {
            "criterion": "every included hard-veto context has a stated hazard rationale",
            "pass": True,
            "finding": "person, fragile/critical/prohibited, damage, fall, stability, entrapment, and control-loss contexts are hazard-derived",
        },
        {
            "criterion": "every excluded hard-veto context has a defensible alternative treatment",
            "pass": False,
            "finding": "fixed-wall, landmark, abnormal-ground, and region-specific physical consequences lack approved limits and recovery authority",
        },
        {
            "criterion": "permitted contacts are explicitly defined",
            "pass": True,
            "finding": "only valid self-contact and explicitly complete ordinary calf/ground support context are prospectively permitted; H1 custody remains link-level",
        },
        {
            "criterion": "unresolved contacts remain unresolved",
            "pass": True,
            "finding": "no incomplete contact is relabelled harmless or recoverable",
        },
        {
            "criterion": "operational safety and task performance are represented separately",
            "pass": True,
            "finding": "physical-contact category and monitor/recovery/task overlays are separate",
        },
        {
            "criterion": "simulated proxy and deployment claim are distinguished",
            "pass": True,
            "finding": "H1 remains a rigid-simulation separation proxy only",
        },
        {
            "criterion": "scope is not derived from sensor outcomes",
            "pass": True,
            "finding": "requirements objective outcome field set is empty and perfect-sensing counterfactual is unchanged",
        },
        {
            "criterion": "decision is deterministic and machine readable",
            "pass": True,
            "finding": "context-first rules, exhaustive matrix, canonical receipts, and validators are frozen",
        },
        {
            "criterion": "all assumptions and evidence gaps are recorded",
            "pass": True,
            "finding": "assumption inventory and Stage-B blocker ledger are complete for the reviewed sources",
        },
    ]
    return with_content_digest({
        "schema": "protected_contact_scope_requirements_sufficiency_gate_v1",
        "criteria": criteria,
        "criteria_passed": sum(bool(row["pass"]) for row in criteria),
        "criteria_total": len(criteria),
        "pass": all(bool(row["pass"]) for row in criteria),
        "requirements_review_disposition_frozen": True,
        "deployment_hard_scope_frozen": False,
        "primary_classification": PRIMARY_CLASSIFICATION,
        "outcome_fields_used": [],
    })


def build_historical_h1_contract() -> dict[str, Any]:
    return with_content_digest({
        "schema": "protected_contact_scope_historical_h1_binding_v1",
        "target": H1_TARGET,
        "horizon_s": 0.1,
        "physics_dt_s": 0.002,
        "physics_steps": 50,
        "positive_scope": "any frozen disallowed robot-environment contact at any physics step",
        "preserved_exclusions": [
            "robot self-contact",
            "any of the four collapsed calf links paired with the simulator ground Plane",
        ],
        "calf_plane_exclusion_granularity": "LINK_LEVEL_NOT_FOOT_SHAPE_ONLY",
        "all_other_robot_environment_pairs": "DISALLOWED_UNDER_UNCHANGED_HISTORICAL_PROXY",
        "force_floor_n": 0.001,
        "force_floor_role": "HISTORICAL_NUMERICAL_LABEL_REPRODUCTION_NOT_SEVERITY_LIMIT",
        "reinterpretation_authorized": False,
        "label_change_authorized": False,
        "outcome_fields_used": [],
    })


def build_contract_receipt() -> dict[str, Any]:
    """Build the complete deterministic Stage-A executable contract receipt."""

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "start_commit": START_COMMIT,
        "stage": STAGE,
        "execution_class": "READ_ONLY_REQUIREMENTS_REVIEW_NO_SCIENTIFIC_MATERIALISATION",
        "preserved_predecessor_authority": {
            "result_commit": START_COMMIT,
            "classifications": list(PRESERVED_PREDECESSOR_CLASSIFICATIONS),
            "exact_geometry_finding": PRESERVED_EXACT_GEOMETRY_FINDING,
            "range_finding": PRESERVED_RANGE_FINDING,
            "role": "PRESERVED_AGGREGATE_CONTEXT_NOT_SCOPE_SELECTION_INPUT",
            "outcome_fields_used_by_scope_decision": [],
        },
        "requirements_only_barrier": {
            "status": "PASS_BY_CONSTRUCTION",
            "outcome_fields_used": [],
            "forbidden_outcome_fields": sorted(FORBIDDEN_OUTCOME_FIELD_NAMES),
            "forbidden_outcome_key_tokens": list(FORBIDDEN_OUTCOME_KEY_TOKENS),
            "sensor_or_model_performance_may_select_scope": False,
        },
        "requirement_categories": list(REQUIREMENT_CATEGORIES),
        "context_first_decision_order": [
            "preserved self-contact exclusion",
            "explicit context-complete ordinary calf-link/ground support",
            "unvalidated abnormal or unknown calf-link/ground contact remains unresolved",
            "categorical or approved hard preaction separation",
            "approved consequence-complete conditional recoverability",
            "approved effective monitor-and-recover requirement",
            "fail closed as severity or requirement unresolved",
        ],
        "historical_h1": build_historical_h1_contract(),
        "component_inventory": component_inventory_receipt(),
        "sources": build_source_inventory(),
        "assumptions": build_assumption_inventory(),
        "traceability": build_traceability_matrix(),
        "link_object_context_matrix": build_link_object_context_matrix(),
        "perfect_sensing_counterfactual": build_perfect_sensing_counterfactual(),
        "requirements_sufficiency_gate": build_requirements_sufficiency_gate(),
        "classifications": {
            "primary_exactly_one": PRIMARY_CLASSIFICATION,
            "secondary_exactly": list(ALLOWED_SECONDARY_CLASSIFICATIONS),
            "other_secondary_classifications_authorized": False,
        },
        "stage_b_gate": build_stage_b_gate(),
        "scope_decision": {
            "protected_links_removed": [],
            "protected_shapes_removed": [],
            "contact_labels_changed": False,
            "scope_narrowing_authorized": False,
            "primary_classification": PRIMARY_CLASSIFICATION,
        },
        "training_or_evaluation": {
            "model_training": False,
            "fresh_panel_collection": False,
            "sensor_evaluation": False,
            "jepa_opened": False,
            "memory_navigation_or_routing_executed": False,
        },
    }
    result = with_content_digest(receipt)
    validate_contract_receipt(result)
    return result


def validate_contract_receipt(receipt: Mapping[str, Any]) -> None:
    validate_content_digest(receipt)
    expected_top_level = {
        "schema_version", "experiment_id", "start_commit", "stage", "execution_class",
        "preserved_predecessor_authority",
        "requirements_only_barrier", "requirement_categories",
        "context_first_decision_order", "historical_h1", "component_inventory", "sources",
        "assumptions", "traceability", "link_object_context_matrix",
        "perfect_sensing_counterfactual", "requirements_sufficiency_gate",
        "classifications", "stage_b_gate",
        "scope_decision", "training_or_evaluation", "content_digest",
    }
    if set(receipt) != expected_top_level:
        raise ContractValidationError("contract receipt field set drift")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ContractValidationError("schema version drift")
    if receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ContractValidationError("experiment identity drift")
    if receipt.get("start_commit") != START_COMMIT:
        raise ContractValidationError("start commit drift")
    if receipt.get("stage") != STAGE:
        raise ContractValidationError("Stage-A identity drift")
    predecessor = receipt.get("preserved_predecessor_authority", {})
    if predecessor != {
        "result_commit": START_COMMIT,
        "classifications": list(PRESERVED_PREDECESSOR_CLASSIFICATIONS),
        "exact_geometry_finding": PRESERVED_EXACT_GEOMETRY_FINDING,
        "range_finding": PRESERVED_RANGE_FINDING,
        "role": "PRESERVED_AGGREGATE_CONTEXT_NOT_SCOPE_SELECTION_INPUT",
        "outcome_fields_used_by_scope_decision": [],
    }:
        raise ContractValidationError("preserved predecessor authority drift")
    if tuple(receipt.get("requirement_categories", ())) != REQUIREMENT_CATEGORIES:
        raise ContractValidationError("requirement category enum drift")
    if receipt.get("context_first_decision_order") != [
        "preserved self-contact exclusion",
        "explicit context-complete ordinary calf-link/ground support",
        "unvalidated abnormal or unknown calf-link/ground contact remains unresolved",
        "categorical or approved hard preaction separation",
        "approved consequence-complete conditional recoverability",
        "approved effective monitor-and-recover requirement",
        "fail closed as severity or requirement unresolved",
    ]:
        raise ContractValidationError("context-first decision order drift")
    classifications = receipt.get("classifications", {})
    if classifications.get("primary_exactly_one") != PRIMARY_CLASSIFICATION:
        raise ContractValidationError("primary classification drift")
    if tuple(classifications.get("secondary_exactly", ())) != ALLOWED_SECONDARY_CLASSIFICATIONS:
        raise ContractValidationError("secondary classifications are not the authorized exact set")
    if classifications.get("other_secondary_classifications_authorized") is not False:
        raise ContractValidationError("unauthorized secondary classifications enabled")
    barrier = receipt.get("requirements_only_barrier", {})
    if barrier.get("outcome_fields_used") != []:
        raise ContractValidationError("requirements-only barrier used scientific outcomes")
    if barrier.get("sensor_or_model_performance_may_select_scope") is not False:
        raise ContractValidationError("sensor/model performance may not select scope")
    if barrier.get("status") != "PASS_BY_CONSTRUCTION":
        raise ContractValidationError("requirements-only barrier status drift")
    if barrier.get("forbidden_outcome_fields") != sorted(FORBIDDEN_OUTCOME_FIELD_NAMES):
        raise ContractValidationError("forbidden outcome field barrier drift")
    if barrier.get("forbidden_outcome_key_tokens") != list(FORBIDDEN_OUTCOME_KEY_TOKENS):
        raise ContractValidationError("forbidden outcome token barrier drift")
    inventory = receipt.get("component_inventory", {})
    validate_content_digest(inventory)
    validate_component_inventory(inventory.get("components", ()))
    if inventory.get("protected_link_count") != 13 or inventory.get("collision_component_count") != 27:
        raise ContractValidationError("protected geometry cardinality drift")
    for child_key in (
        "historical_h1", "sources", "assumptions", "traceability",
        "link_object_context_matrix",
        "perfect_sensing_counterfactual", "requirements_sufficiency_gate", "stage_b_gate",
    ):
        child = receipt.get(child_key)
        if not isinstance(child, Mapping):
            raise ContractValidationError(f"missing receipt child: {child_key}")
        validate_content_digest(child)
        if child.get("outcome_fields_used") != []:
            raise ContractValidationError(f"{child_key} used scientific outcomes")
    expected_children = {
        "historical_h1": build_historical_h1_contract(),
        "component_inventory": component_inventory_receipt(),
        "sources": build_source_inventory(),
        "assumptions": build_assumption_inventory(),
        "traceability": build_traceability_matrix(),
        "link_object_context_matrix": build_link_object_context_matrix(),
        "perfect_sensing_counterfactual": build_perfect_sensing_counterfactual(),
        "requirements_sufficiency_gate": build_requirements_sufficiency_gate(),
        "stage_b_gate": build_stage_b_gate(),
    }
    for child_key, expected_child in expected_children.items():
        if receipt.get(child_key) != expected_child:
            raise ContractValidationError(f"{child_key} semantic content drift")
    h1 = receipt["historical_h1"]
    if h1.get("calf_plane_exclusion_granularity") != "LINK_LEVEL_NOT_FOOT_SHAPE_ONLY":
        raise ContractValidationError("historical calf-link/Plane exclusion was reinterpreted")
    if h1.get("reinterpretation_authorized") is not False or h1.get("label_change_authorized") is not False:
        raise ContractValidationError("historical H1 mutation was authorized")
    validate_link_object_context_matrix(receipt["link_object_context_matrix"])
    sufficiency = receipt["requirements_sufficiency_gate"]
    if sufficiency.get("pass") is not False:
        raise ContractValidationError("unresolved requirements sufficiency gate must fail")
    if sufficiency.get("deployment_hard_scope_frozen") is not False:
        raise ContractValidationError("deployment hard scope must remain unfrozen")
    if sufficiency.get("requirements_review_disposition_frozen") is not True:
        raise ContractValidationError("requirements review disposition must be frozen")
    stage_b = receipt["stage_b_gate"]
    if stage_b.get("status") != STAGE_B_STATUS or stage_b.get("authorized") is not False:
        raise ContractValidationError("Stage B must remain unauthorized")
    decision = receipt.get("scope_decision", {})
    if decision != {
        "protected_links_removed": [],
        "protected_shapes_removed": [],
        "contact_labels_changed": False,
        "scope_narrowing_authorized": False,
        "primary_classification": PRIMARY_CLASSIFICATION,
    }:
        raise ContractValidationError("Stage-A scope decision drift")
    execution = receipt.get("training_or_evaluation", {})
    if any(value is not False for value in execution.values()):
        raise ContractValidationError("Stage-A contract authorized execution")
    # Exclude the barrier's own declarative forbidden-field list from this
    # recursive check; every other receipt subtree remains outcome-free.
    requirements_view = {
        key: value for key, value in receipt.items()
        if key not in {"requirements_only_barrier", "content_digest"}
    }
    assert_requirements_only_payload(requirements_view)


__all__ = [
    "ALLOWED_SECONDARY_CLASSIFICATIONS",
    "CALF_LINK_NAMES",
    "CollisionComponent",
    "CONDITIONAL_RECOVERABLE_CONTACT",
    "ContractValidationError",
    "EXPERIMENT_ID",
    "GROUND_CLASS",
    "H1_TARGET",
    "HARD_PREACTION_SEPARATION",
    "MONITOR_AND_RECOVER",
    "PERMITTED_SUPPORT_OR_SELF_CONTACT",
    "PRIMARY_CLASSIFICATION",
    "PRESERVED_EXACT_GEOMETRY_FINDING",
    "PRESERVED_PREDECESSOR_CLASSIFICATIONS",
    "PRESERVED_RANGE_FINDING",
    "PROTECTED_LINK_NAMES",
    "REQUIREMENT_CATEGORIES",
    "RequirementContext",
    "RequirementDecision",
    "SCHEMA_VERSION",
    "SEVERITY_OR_REQUIREMENT_UNRESOLVED",
    "STAGE",
    "STAGE_B_STATUS",
    "START_COMMIT",
    "assert_requirements_only_payload",
    "build_assumption_inventory",
    "build_contract_receipt",
    "build_historical_h1_contract",
    "build_link_object_context_matrix",
    "build_perfect_sensing_counterfactual",
    "build_requirements_sufficiency_gate",
    "build_source_inventory",
    "build_stage_b_gate",
    "build_traceability_matrix",
    "canonical_digest",
    "canonical_json_bytes",
    "component_inventory_receipt",
    "decide_requirement",
    "is_preserved_historical_exclusion",
    "normalize_environment_class",
    "protected_collision_components",
    "validate_component_inventory",
    "validate_content_digest",
    "validate_contract_receipt",
    "validate_link_object_context_matrix",
    "validate_traceability_matrix",
    "with_content_digest",
]
