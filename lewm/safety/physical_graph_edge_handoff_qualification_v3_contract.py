"""Pure prospective authority for semantic physical handoff qualification V3.

V3 preserves the complete V2 scientific design.  It changes only versioned
identity/custody and replaces allocation-dependent snapshot transport identity
with an independently reproducible semantic identity plus a frozen behavioural
probe.  The V2 exact-persisted-byte correction and its two narrowly authorized
writer alignments remain unchanged.  No outcome, model, threshold, formula,
classification, precedence rule, or next decision is changed here.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
from pathlib import Path
import types
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as _V2


class PhysicalGraphEdgeHandoffV3ContractError(ValueError):
    """Raised when the V3 prospective authority drifts."""


# Re-export every inherited scientific/runtime constant.  Version-specific
# names are overwritten below.  This mirrors V2's exact V1-science wrapper.
for _name in tuple(_V2.__all__):
    if _name.isupper():
        _value = getattr(_V2, _name)
        try:
            _value = copy.deepcopy(_value)
        except (TypeError, ValueError):
            pass
        globals()[_name] = _value

# The inherited V1 contract exports an uppercase ``V2`` module constant.  Put
# our actual immediate predecessor alias back after the re-export loop.
V2 = _V2


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
V2_EXPERIMENT_ID = V2.EXPERIMENT_ID
V1_EXPERIMENT_ID = V2.V1_EXPERIMENT_ID
V2_SOURCE_FREEZE_COMMIT = "10117870fe00bbfd8709932cab7b6e9df3be9bfc"
V1_SOURCE_FREEZE_COMMIT = V2.V1_SOURCE_FREEZE_COMMIT
SOURCE_PARENT_COMMIT = V2_SOURCE_FREEZE_COMMIT
SOURCE_BASELINE_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
CONTRACT_FREEZE_COMMIT_SUBJECT = (
    "Freeze semantic physical graph edge handoff qualification V3"
)
RESULT_COMMIT_SUBJECT = (
    "Evaluate semantic physical graph edge handoff qualification V3"
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v3"
)
MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v3_material"
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v3_regeneration_receipt.json"
)
HISTORICAL_CUSTODY_RECEIPT_PATH = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v1_v2_custody_receipt.json"
)
V1_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1"
V1_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1_material"
V2_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v2"
V2_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v2_material"

NEW_RUNTIME_OUTPUT_PATHS = {
    "v1_v2_custody_and_nonreuse": "v1_v2_custody_and_nonreuse.json",
    "scientific_invariance_receipt": "scientific_invariance_receipt.json",
    "v1_v2_v3_first_eight_reproduction": "v1_v2_v3_first_eight_reproduction.json",
    "snapshot_equivalence_index": "snapshot_equivalence_index.json",
    "snapshot_behavioural_probes": "snapshot_behavioural_probes.npz",
}
RUNTIME_OUTPUT_PATHS = {
    **copy.deepcopy(V2.V1.RUNTIME_OUTPUT_PATHS),
    **NEW_RUNTIME_OUTPUT_PATHS,
}
OUTPUT_LEAF_COUNT = 28
SUCCESS_OUTPUT_LEAVES = tuple(RUNTIME_OUTPUT_PATHS.values())
REPRODUCTION_MISMATCH_LEAVES = (
    "contract.json",
    "v1_v2_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_v3_first_eight_reproduction.json",
)
REPRODUCTION_MISMATCH_DISPOSITION = (
    "V1_V2_V3_SEMANTIC_OR_BEHAVIOURAL_REPRODUCTION_MISMATCH"
)


# ---------------------------------------------------------------------------
# Historical custody and immutable terminal V2 evidence
# ---------------------------------------------------------------------------

V1_CUSTODY_RECEIPT_BINDING = copy.deepcopy(V2.V1_CUSTODY_RECEIPT_BINDING)
V2_TERMINAL_ROOT_BINDING = {
    "path": str(V2_OFFICIAL_ROOT),
    "source_freeze_commit": V2_SOURCE_FREEZE_COMMIT,
    "disposition": V2.REPRODUCTION_MISMATCH_DISPOSITION,
    "file_count": 4,
    "regular_file_apparent_bytes": 102527,
    "files": [
        {
            "path": "contract.json",
            "bytes": 91741,
            "sha256": "9f8146581f9d45bcc6e1ee226ac14db4344be3d4069a723abcf27538f8a43902",
        },
        {
            "path": "scientific_invariance_receipt.json",
            "bytes": 3077,
            "sha256": "710f5dd2a7743cd69bc6ac7e10cf8555c5e4345a6311054a851e961648cdd966",
        },
        {
            "path": "v1_custody_and_nonreuse.json",
            "bytes": 912,
            "sha256": "80966d0815058f2dbed7dfaed4604f8da68fab5fb1108b0c3b7b0c2cf61af5bc",
        },
        {
            "path": "v1_v2_first_eight_reproduction.json",
            "bytes": 6797,
            "sha256": "d3437d2560af6005b871b94bed6ef3f7f0f185497984725eae41de80224069c6",
        },
    ],
    "material_file_count": 34,
    "material_regular_file_apparent_bytes": 4382459,
    "material_manifest_sha256_bound_by_external_receipt": True,
    "external_regeneration_receipt_present": False,
}
# The complete material-manifest digest is supplied by the independently
# generated combined custody receipt and is exact-validated there.  The prefix
# above is descriptive only until that ordinary receipt is frozen; no runtime
# contract may use this partial value as a file binding.
HISTORICAL_CUSTODY_RECEIPT_BINDING = {
    "path": str(HISTORICAL_CUSTODY_RECEIPT_PATH),
    "bytes": 2572931,
    "sha256": (
        "38d900b29ddb6ed672d771bb19e5b9878fb011ccf4061f847a76f75fde9d7bbc"
    ),
}
HISTORICAL_CUSTODY_RECEIPT_PATH_AUTHORITY = {
    "path": str(HISTORICAL_CUSTODY_RECEIPT_PATH),
    "exact_binding": copy.deepcopy(HISTORICAL_CUSTODY_RECEIPT_BINDING),
    "schema": (
        "physical_graph_edge_handoff_qualification_v1_v2.custody_receipt.v1"
    ),
    "ordinary_canonical_json_without_self_digest": True,
    "must_be_bound_by_exact_bytes_and_sha256_before_simulator_creation": True,
}
HISTORICAL_CUSTODY_BINDING_FIELDS = frozenset({"path", "bytes", "sha256"})


# ---------------------------------------------------------------------------
# Exact inherited correction/design invariance
# ---------------------------------------------------------------------------

NPZ_ARCHIVE_COMMENT = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3:FRESH"
PERSISTED_ARRAY_HASH_AUTHORITY = copy.deepcopy(V2.PERSISTED_ARRAY_HASH_AUTHORITY)
PERSISTED_ARRAY_HASH_AUTHORITY.update(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "persisted_array_hash_authority.v1"
        ),
        "npz_archive_comment_utf8": NPZ_ARCHIVE_COMMENT,
        "npz_archive_comment_scope": (
            "every V3 material shard payload.npz governed by persisted_array_evidence"
        ),
    }
)
PERSISTED_ARRAY_HASH_AUTHORITY["npz_archive_comment_semantics"] = (
    "V3-only deterministic container provenance; it changes no NPZ member, dtype, "
    "shape, logical value, candidate identity, or scientific outcome"
)
PERSISTED_ARRAY_ROW_FIELDS = copy.deepcopy(V2.PERSISTED_ARRAY_ROW_FIELDS)
PERSISTED_ARRAY_EVIDENCE_FIELDS = copy.deepcopy(V2.PERSISTED_ARRAY_EVIDENCE_FIELDS)
PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS = copy.deepcopy(
    V2.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS
)
REGRESSION_REQUIREMENT_IDS = tuple(V2.REGRESSION_REQUIREMENT_IDS)
REGRESSION_FIXTURE = copy.deepcopy(V2.REGRESSION_FIXTURE)
REGRESSION_GATE_AUTHORITY = copy.deepcopy(V2.REGRESSION_GATE_AUTHORITY)
PORT_HEADING_ALIGNMENT_DISPOSITION = V2.PORT_HEADING_ALIGNMENT_DISPOSITION
PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY = copy.deepcopy(
    V2.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
)
CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION = (
    V2.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION
)
CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY = copy.deepcopy(
    V2.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
)

V2_SCIENTIFIC_CONTRACT = V2.build_contract()
V2_CONTRACT_CONTENT_DIGEST = V2_SCIENTIFIC_CONTRACT["content_digest"]
V2_SCIENTIFIC_PROJECTION = copy.deepcopy(V2.V1_SCIENTIFIC_PROJECTION)
V2_SCIENTIFIC_PROJECTION_SHA256 = V2.V1_SCIENTIFIC_PROJECTION_SHA256
V2_SCIENTIFIC_CONSTANT_PROJECTION = copy.deepcopy(
    V2.V2_SCIENTIFIC_CONSTANT_PROJECTION
)
V2_SCIENTIFIC_CONSTANTS_SHA256 = V2.V2_SCIENTIFIC_CONSTANTS_SHA256
V2_CANDIDATE_SPECS_SHA256 = V2.V1_CANDIDATE_SPECS_SHA256
V2_SOURCE_DEPENDENCY_PATHS_SHA256 = V2.V1_SOURCE_DEPENDENCY_PATHS_SHA256


def canonical_json_bytes(value: Any) -> bytes:
    return V2.canonical_json_bytes(value)


def _canonical_no_lf_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return V2.attach_content_digest(value)


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return V2.validate_content_digest(value)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3ContractError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Runtime-only external-artifact path custody
# ---------------------------------------------------------------------------

# V1/V2 bound the correct immutable Genesis asset bytes through a lexical
# workspace path whose ``.generated/venvs`` component is a symlink.  V3 keeps
# the artifact role, bytes, and SHA-256 exactly unchanged, but binds the same
# ordinary file through its fully resolved RecoveryStorage path so the strict
# no-symlink custody walker can open it without weakening that walker.
GENESIS_GO2_URDF_INHERITED_LEXICAL_PATH = (
    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
    "genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis/assets/"
    "urdf/go2/urdf/go2.urdf"
)
GENESIS_GO2_URDF_RESOLVED_ORDINARY_PATH = (
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
    "genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis/assets/"
    "urdf/go2/urdf/go2.urdf"
)
GENESIS_GO2_URDF_BYTES = 24170
GENESIS_GO2_URDF_SHA256 = (
    "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4"
)


def _v3_external_artifact_bindings() -> tuple[dict[str, Any], ...]:
    rows = [copy.deepcopy(row) for row in V2.V1.EXTERNAL_ARTIFACT_BINDINGS]
    matches = [row for row in rows if row.get("role") == "genesis_go2_urdf"]
    if len(matches) != 1:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "inherited Genesis Go2 URDF binding cardinality drift"
        )
    row = matches[0]
    if row != {
        "role": "genesis_go2_urdf",
        "path": GENESIS_GO2_URDF_INHERITED_LEXICAL_PATH,
        "bytes": GENESIS_GO2_URDF_BYTES,
        "sha256": GENESIS_GO2_URDF_SHA256,
        "kind": "frozen_physical_robot_asset",
    }:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "inherited Genesis Go2 URDF artifact identity drift"
        )
    row["path"] = GENESIS_GO2_URDF_RESOLVED_ORDINARY_PATH
    return tuple(rows)


EXTERNAL_ARTIFACT_BINDINGS = _v3_external_artifact_bindings()
EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "external_artifact_runtime_path_correction_authority.v1"
        ),
        "scope": "V3_RUNTIME_CUSTODY_PATH_ONLY",
        "role": "genesis_go2_urdf",
        "inherited_lexical_path": GENESIS_GO2_URDF_INHERITED_LEXICAL_PATH,
        "resolved_ordinary_path": GENESIS_GO2_URDF_RESOLVED_ORDINARY_PATH,
        "bytes": GENESIS_GO2_URDF_BYTES,
        "sha256": GENESIS_GO2_URDF_SHA256,
        "kind": "frozen_physical_robot_asset",
        "same_immutable_artifact_bytes_and_sha256": True,
        "inherited_path_must_resolve_to_resolved_ordinary_path": True,
        "resolved_path_and_every_existing_component_must_not_be_a_symlink": True,
        "strict_no_symlink_external_binding_validator_unchanged": True,
        "scientific_projection_model_controller_geometry_and_outcomes_unchanged": True,
    }
)


# ---------------------------------------------------------------------------
# Snapshot semantic identity
# ---------------------------------------------------------------------------

SNAPSHOT_IDENTITY_FIELDS = frozenset(
    {
        "artifact_file_sha256",
        "snapshot_semantic_digest_v1",
        "snapshot_behavioural_digest_v1",
    }
)
STRUCTURED_TYPE_FIELD_AUTHORITY = {
    "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot": [
        "solver_state",
        "step_index",
        "last_actions",
        "harness",
        "rng",
        "counters",
        "goal",
        "identity",
        "boundary",
        "digest",
    ],
    "lewm_genesis.lewm_contract.EpisodeState": [
        "scene_id",
        "episode_id",
        "reset_count",
        "episode_step",
        "scene_family",
        "split",
        "manifest_sha256",
    ],
}
NONFINITE_SENTINEL_AUTHORITY = {
    "default": "reject every nonfinite Python scalar, NumPy scalar/array, and torch tensor value",
    "allowed_numpy_arrays": [
        {
            "path": (
                "$root/solver_state/"
                "Scene._sim._coupler.rigid_solver.dofs_info.force_range"
            ),
            "dtype_str": "<f4",
            "shape": [18, 2],
            "strides_bytes": [8, 4],
            "positive_infinity_count": 6,
            "negative_infinity_count": 6,
            "nan_count": 0,
        },
        {
            "path": (
                "$root/solver_state/"
                "Scene._sim._coupler.rigid_solver.dofs_info.limit"
            ),
            "dtype_str": "<f4",
            "shape": [18, 2],
            "strides_bytes": [8, 4],
            "positive_infinity_count": 6,
            "negative_infinity_count": 6,
            "nan_count": 0,
        },
    ],
    "coverage": "independently inventoried across V1 and V2 pool indices 0..7",
    "encoding": (
        "preserve exact dtype, shape, original strides, signed zero, infinity signs, "
        "and IEEE C-contiguous logical bytes; no normalization"
    ),
    "all_nan_forbidden": True,
    "off_path_nonfinite_forbidden": True,
}
SEMANTIC_SERIALIZER_REGRESSION_IDS = (
    "TYPED_SCALAR_MAPPING_KEYS_SORTED_BOOL_DISTINCT_FROM_INT",
    "LIST_TUPLE_BYTES_ORDER_UNICODE_AND_TYPE_TAGS_EXACT",
    "SET_FROZENSET_SCALAR_SORT_AND_COMPOUND_REJECTION",
    "NUMPY_DTYPE_SHAPE_STRIDE_STORAGE_OFFSET_VALUE_MUTATION_EXACT",
    "REFERENCE_ALIAS_DEF_REF_AND_SAVE_RELOAD_REHASH_IDENTICAL",
    "REFERENCE_CYCLE_DEF_REF_DETERMINISTIC",
    "NUMPY_SHARED_STORAGE_GROUPS_ALLOCATION_INDEPENDENT",
    "TORCH_SHARED_STORAGE_STRIDE_CPU_RESTORED_DEVICE_RULE_EXACT",
    "STRUCTURED_ALLOWLIST_AND_DECLARED_FIELD_ORDER_EXACT",
    "PATH_SCOPED_INFINITY_SENTINELS_EXACT_IEEE_BITS",
    "OFF_PATH_NONFINITE_AND_NAN_REJECTED",
    "UNKNOWN_TYPE_AND_OBJECT_DTYPE_REJECTED",
    "TWO_PROCESS_TORCH_TRANSPORT_DRIFT_SEMANTIC_EQUALITY",
    "ALL_EIGHT_V1_V2_SEMANTIC_AND_EVIDENCE_BYTES_EQUAL",
)
SEMANTIC_SERIALIZER_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "snapshot_semantic_serializer_authority.v1"
        ),
        "serializer_schema": (
            "physical_graph_edge_handoff_snapshot_semantics_v1.canonical_binary.v1"
        ),
        "digest_algorithm": "sha256",
        "digest_key": "snapshot_semantic_digest_v1",
        "binary_magic_hex": b"PGEHQ-SNAPSHOT-SEMANTICS\x00\x01".hex(),
        "record_framing": "uint8 tag-byte-count || tag || uint64be payload-byte-count || payload",
        "scalar_types": ["none", "bool", "arbitrary_signed_int", "finite_binary64", "strict_utf8", "raw_bytes"],
        "mapping_key_types": ["none", "bool", "arbitrary_signed_int", "finite_binary64", "strict_utf8", "raw_bytes"],
        "mapping_key_rules": (
            "explicit type tags; bool distinct from int; sort by complete canonical "
            "encoded key bytes; reject tuple, frozenset, and every compound key"
        ),
        "ordered_containers": ["list", "tuple"],
        "unordered_value_containers": ["set", "frozenset"],
        "set_rule": (
            "elements must be scalar canonical values; sort complete typed encodings; "
            "reject referenceable/compound elements and duplicate canonical encodings"
        ),
        "mapping_types": ["builtins.dict", "collections.OrderedDict"],
        "structured_type_fields_in_order": copy.deepcopy(
            STRUCTURED_TYPE_FIELD_AUTHORITY
        ),
        "unknown_structured_type_or_field_inventory": "hard_fail",
        "repr_pickle_or_implicit_object_fallback": False,
        "reference_graph": (
            "referenceable objects receive monotonically increasing uint64 IDs on first "
            "canonical traversal; later occurrences and cycles emit REF(ID)"
        ),
        "reference_policy": (
            "CANONICAL_FIRST_TRAVERSAL_REFERENCE_AND_STORAGE_GRAPH_V1"
        ),
        "referenceable_types": [
            "mapping", "list", "tuple", "set", "frozenset", "numpy.ndarray",
            "torch.Tensor", "registered structured object",
        ],
        "numpy_rule": (
            "bind dtype.str, shape, original byte strides, deterministic shared-storage "
            "group ID, byte storage offset, and exact C-contiguous logical bytes"
        ),
        "torch_rule": (
            "bind torch dtype string, shape, original element stride, deterministic "
            "shared-storage group ID, byte storage offset (storage_offset multiplied by "
            "element_size), semantic device class "
            "cpu/accelerator, and detached CPU C-contiguous logical bytes"
        ),
        "torch_device_rule": (
            "encode cpu versus accelerator only; containing registered RNG field/list "
            "position preserves logical per-device restoration association"
        ),
        "storage_identity_rule": (
            "pointer or _cdata may be used transiently only to detect live aliasing; no "
            "address, pointer, _cdata, allocation key, or process identity enters bytes"
        ),
        "numpy_object_structured_subarray_dtype": "hard_fail",
        "torch_forbidden": [
            "sparse", "quantized", "meta", "lazy conjugate", "lazy negative",
            "requires_grad",
        ],
        "nonfinite_sentinel_authority": copy.deepcopy(
            NONFINITE_SENTINEL_AUTHORITY
        ),
        "regression_requirements": list(SEMANTIC_SERIALIZER_REGRESSION_IDS),
        "production_type_inventory_required": True,
        "production_pool_000_observed": {
            "numpy_array_count": 506,
            "torch_tensor_count": 3,
            "torch_tensor_device_class": "cpu",
            "torch_tensor_dtype": "torch.uint8",
            "torch_tensor_shapes": [[5056], [16], [16]],
            "set_count": 3,
            "all_sets_empty": True,
            "reference_alias_count": 0,
            "shared_storage_alias_count": 0,
        },
    }
)

HISTORICAL_DESERIALIZER_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "historical_snapshot_deserializer_authority.v1"
        ),
        "worker_interpreter": (
            "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
            "genesis_rocm_0_4_6_v1/bin/python"
        ),
        "worker_scope": (
            "isolated read-only restricted unpickle worker; no runner, model, encoder, "
            "ranker, simulator, reset, or physics construction"
        ),
        "outer_pickle_protocol": 4,
        "allowed_opcodes": [
            "APPEND", "APPENDS", "BINFLOAT", "BINGET", "BININT", "BININT1",
            "BININT2", "BINBYTES", "BUILD", "EMPTY_DICT", "EMPTY_LIST",
            "EMPTY_SET", "EMPTY_TUPLE", "FRAME", "LONG1", "LONG_BINGET",
            "MARK", "MEMOIZE", "NEWFALSE", "NEWOBJ", "NEWTRUE", "NONE",
            "PROTO", "REDUCE", "SETITEM", "SETITEMS", "SHORT_BINBYTES",
            "SHORT_BINUNICODE", "STACK_GLOBAL", "STOP", "TUPLE", "TUPLE1",
            "TUPLE2", "TUPLE3",
        ],
        "allowed_globals": [
            ["scripts.run_go2_oracle_branch_pilot_v1", "BranchSnapshot"],
            ["numpy._core.multiarray", "_reconstruct"],
            ["numpy", "ndarray"],
            ["numpy", "dtype"],
            ["lewm_genesis.lewm_contract", "EpisodeState"],
            ["torch._utils", "_rebuild_tensor_v2"],
            ["torch.storage", "_load_from_bytes"],
            ["collections", "OrderedDict"],
        ],
        "structured_global_handling": (
            "BranchSnapshot and EpisodeState resolve only to inert stubs carrying exact "
            "frozen FQCN, declared fields, and no methods or imports"
        ),
        "global_opcode_and_constructor_preflight_required": True,
        "unknown_global_opcode_reduce_target_or_persistent_id": "hard_fail",
        "worker_output": (
            "canonical semantic bytes plus semantic evidence/type/alias/storage/device "
            "manifests; main portable evaluator independently parses and hashes bytes"
        ),
    }
)

SNAPSHOT_SEMANTIC_EVIDENCE_FIELDS = frozenset(
    {
        "serializer_schema",
        "snapshot_semantic_digest_v1",
        "canonical_semantic_byte_count",
        "reference_policy",
        "referenceable_object_count",
        "reference_alias_edge_count",
        "reference_cycle_edge_count",
        "reference_manifest",
        "reference_edge_manifest",
        "storage_manifest",
        "type_inventory",
        "structured_type_inventory",
        "tensor_device_manifest",
        "nonfinite_sentinel_inventory",
    }
)


# ---------------------------------------------------------------------------
# Behavioural probe and first-eight gate
# ---------------------------------------------------------------------------

BEHAVIOURAL_PROBE_COMMAND = [0.2, 0.0, 0.0]
BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND = [
    0.20000000298023224,
    0.0,
    0.0,
]
BEHAVIOURAL_PROBE_COMMAND_TICKS = 15
BEHAVIOURAL_PROBE_PHYSICS_SAMPLES = 750
BEHAVIOURAL_PROBE_POOL_COUNT = 256
BEHAVIOURAL_PROBE_VERSION_ORDER = ("V1", "V2", "V3")
BEHAVIOURAL_PROBE_TRIALS_PER_VERSION = 2
HISTORICAL_BEHAVIOURAL_PROBE_POOL_COUNT = 8
HISTORICAL_BEHAVIOURAL_PROBE_TRACE_COUNT = 32
V3_BEHAVIOURAL_PROBE_TRACE_COUNT = 512
BEHAVIOURAL_PROBE_TRACE_COUNT = 544
BEHAVIOURAL_PROBE_TOTAL_SAMPLES = 408000
BEHAVIOURAL_TRACE_MEMBER_AUTHORITY = {
    "timestamp_s": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "base_pose_world": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 7], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "base_twist_world": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 6], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "joint_position": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "joint_velocity": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "requested_command": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "post_slew_applied_command": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "physics_contact": {"descr": "|u1", "digest_dtype": "uint8", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "source_region_member": {"descr": "|u1", "digest_dtype": "uint8", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "correct_edge_region_member": {"descr": "|u1", "digest_dtype": "uint8", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "wrong_edge_region_member": {"descr": "|u1", "digest_dtype": "uint8", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "target_region_member": {"descr": "|u1", "digest_dtype": "uint8", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "controller_observation": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 45], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    "policy_output": {"descr": "<f8", "digest_dtype": "float64", "shape": [BEHAVIOURAL_PROBE_TOTAL_SAMPLES, 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
}
BEHAVIOURAL_PROBE_NPZ_AUTHORITY = {
    "trace_offsets": {"descr": "<i8", "digest_dtype": "int64", "shape": [545], "hash_mode": "whole"},
    "version_code": {"descr": "<i8", "digest_dtype": "int64", "shape": [544], "hash_mode": "rows_axis0"},
    "pool_index": {"descr": "<i8", "digest_dtype": "int64", "shape": [544], "hash_mode": "rows_axis0"},
    "trial_index": {"descr": "<i8", "digest_dtype": "int64", "shape": [544], "hash_mode": "rows_axis0"},
    "stuck": {"descr": "|u1", "digest_dtype": "uint8", "shape": [544], "hash_mode": "rows_axis0"},
    "termination_code": {"descr": "<i8", "digest_dtype": "int64", "shape": [544], "hash_mode": "rows_axis0"},
    "final_snapshot_semantic_digest_bytes": {"descr": "|u1", "digest_dtype": "uint8", "shape": [544, 32], "hash_mode": "rows_axis0"},
    **copy.deepcopy(BEHAVIOURAL_TRACE_MEMBER_AUTHORITY),
}
FIRST_EIGHT_BEHAVIOURAL_PROBE_NPZ_AUTHORITY = {
    "trace_offsets": {"descr": "<i8", "digest_dtype": "int64", "shape": [49], "hash_mode": "whole"},
    "version_code": {"descr": "<i8", "digest_dtype": "int64", "shape": [48], "hash_mode": "rows_axis0"},
    "pool_index": {"descr": "<i8", "digest_dtype": "int64", "shape": [48], "hash_mode": "rows_axis0"},
    "trial_index": {"descr": "<i8", "digest_dtype": "int64", "shape": [48], "hash_mode": "rows_axis0"},
    "stuck": {"descr": "|u1", "digest_dtype": "uint8", "shape": [48], "hash_mode": "rows_axis0"},
    "termination_code": {"descr": "<i8", "digest_dtype": "int64", "shape": [48], "hash_mode": "rows_axis0"},
    "final_snapshot_semantic_digest_bytes": {"descr": "|u1", "digest_dtype": "uint8", "shape": [48, 32], "hash_mode": "rows_axis0"},
    **{
        member: {
            "descr": authority["descr"],
            "digest_dtype": authority["digest_dtype"],
            "shape": [36000, *authority["shape"][1:]],
            "hash_mode": authority["hash_mode"],
            "offsets_member": authority["offsets_member"],
        }
        for member, authority in BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items()
    },
}
BEHAVIOURAL_PROBE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "snapshot_behavioural_probe_authority.v1"
        ),
        "command": BEHAVIOURAL_PROBE_COMMAND,
        "requested_command_trace_value": (
            BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND
        ),
        "requested_command_trace_rule": (
            "the nominal command is cast once to the inherited float32 requested tape "
            "and each exact executed tape value is then promoted to float64 by trace "
            "sampling; no ideal-float64 normalization is permitted"
        ),
        "command_ticks": BEHAVIOURAL_PROBE_COMMAND_TICKS,
        "physics_dt_s": 0.002,
        "physics_samples_per_trial": BEHAVIOURAL_PROBE_PHYSICS_SAMPLES,
        "v3_pool_indices": list(range(256)),
        "historical_pool_indices": list(range(8)),
        "version_order": list(BEHAVIOURAL_PROBE_VERSION_ORDER),
        "version_codes": {"V1": 1, "V2": 2, "V3": 3},
        "trials_per_version": BEHAVIOURAL_PROBE_TRIALS_PER_VERSION,
        "historical_trace_count": HISTORICAL_BEHAVIOURAL_PROBE_TRACE_COUNT,
        "v3_trace_count": V3_BEHAVIOURAL_PROBE_TRACE_COUNT,
        "trace_count": BEHAVIOURAL_PROBE_TRACE_COUNT,
        "trace_order": (
            "pool_index ascending; for pools 0..7 V1,V2,V3 then trial 0/1; "
            "for pools 8..255 V3 only then trial 0/1"
        ),
        "trace_offsets": list(
            range(0, BEHAVIOURAL_PROBE_TOTAL_SAMPLES + 1, 750)
        ),
        "npz_members": copy.deepcopy(BEHAVIOURAL_PROBE_NPZ_AUTHORITY),
        "first_eight_material_npz_members": copy.deepcopy(
            FIRST_EIGHT_BEHAVIOURAL_PROBE_NPZ_AUTHORITY
        ),
        "pair_comparison": copy.deepcopy(V2.RESET_TRACE_PAIR_COMPARISON_AUTHORITY),
        "controller_policy_sampling": {
            "command_ticks": 15,
            "policy_acts_per_command_tick": 5,
            "physics_samples_per_policy_act": 10,
            "policy_acts_per_trial": 75,
            "physics_samples_per_trial": 750,
            "controller_observation": (
                "exact flattened float64[45] controller input passed into policy.act; "
                "captured immediately before that act and repeated unchanged across the "
                "ten consecutive 2 ms physics samples governed by the act"
            ),
            "policy_output": (
                "exact raw float64[12] policy._last_actions immediately after policy.act "
                "returns and before action scaling, latency, or application; repeat the "
                "immutable value unchanged across the ten consecutive 2 ms physics "
                "samples governed by the act"
            ),
            "excluded_policy_output_interpretations": [
                "policy action from the preceding latency slot",
                "scaled or returned joint target",
                "applied joint command",
                "pre-snapshot policy-last-action array",
            ],
        },
        "controller_policy_samplewise_tolerance": 1.0e-9,
        "per_trace_metadata": {
            "stuck": "regenerated from the full raw trace under PHYSICAL_OUTCOME_AUTHORITY",
            "termination_code": {"H3_COMPLETE": 1},
            "final_snapshot_semantic_digest_bytes": (
                "32 raw bytes decoded from exact final snapshot_semantic_digest_v1 hex"
            ),
        },
        "fresh_simulator_instance_per_restoration_trial": True,
        "probe_session_topology": {
            "rule": (
                "construct a new simulator/controller session for every single restore "
                "trial; never execute trial 1 in the session used by trial 0"
            ),
            "pool_000_to_007_sessions": (
                "one decision-state capture/teacher session plus six independent probe "
                "sessions (V1/V2/V3 times trials 0/1)"
            ),
            "pool_008_to_255_sessions": (
                "one decision-state capture/teacher session plus two independent V3 "
                "probe sessions (trials 0/1)"
            ),
            "production_fixture_sessions": 7,
            "same_session_second_restore_forbidden": True,
        },
        "historical_v1_v2_snapshots_read_only": True,
        "historical_probe_must_be_actual_restore_execution": True,
        "behavioural_digest_domain": (
            "SHA-256 of canonical no-LF JSON for one deterministic restore trial: "
            "member manifests (dtype.str, shape, exact C-byte SHA), termination, stuck, "
            "and final semantic snapshot digest; excludes version, pool, and trial IDs"
        ),
        "snapshot_behavioural_digest_v1_designated_trial": 0,
        "trial_1_behavioural_digest_persisted_separately": True,
        "digest_equality_is_descriptive": True,
        "gate": (
            "each within-version trial pair and, for pools 0..7, all V1/V2/V3 "
            "cross-version corresponding-trial pairs (trial 0 against trial 0 and trial 1 "
            "against trial 1 for each unordered version pair) "
            "must pass RESET_TRACE_PAIR_COMPARISON_AUTHORITY; its named exact members "
            "including requested/post-slew commands remain byte-exact, and controller_observation "
            "plus policy_output compare samplewise at controller_policy_samplewise_tolerance; "
            "missing historical probe evidence is a mismatch"
        ),
    }
)

SNAPSHOT_EQUIVALENCE_VERSION_FIELDS = SNAPSHOT_IDENTITY_FIELDS
TRACE_MEMBER_MANIFEST_FIELDS = frozenset(
    {"member", "dtype_str", "shape", "array_bytes_sha256"}
)
BEHAVIOURAL_PROBE_VERSION_EVIDENCE_FIELDS = frozenset(
    {
        "snapshot_identity",
        "trial_1_behavioural_digest_v1",
        "final_snapshot_semantic_digests",
        "trial_stuck",
        "trial_termination_reasons",
        "trial_pair_comparison",
        "trace_member_manifests",
    }
)
QUALIFICATION_SHARD_AUGMENTATION_FIELDS = frozenset(
    {"snapshot_semantic_evidence", "snapshot_identity", "behavioural_probes"}
)
QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "qualification_shard_augmentation_authority.v1"
        ),
        "root_fields": sorted(QUALIFICATION_SHARD_AUGMENTATION_FIELDS),
        "snapshot_identity_fields": sorted(SNAPSHOT_IDENTITY_FIELDS),
        "behavioural_probe_version_fields": sorted(
            BEHAVIOURAL_PROBE_VERSION_EVIDENCE_FIELDS
        ),
        "trace_member_manifest_fields": sorted(TRACE_MEMBER_MANIFEST_FIELDS),
        "version_inventory": (
            "V1,V2,V3 exactly for pool indices 0..7; V3 exactly for 8..255"
        ),
        "npz_snapshot_semantic_member": {
            "member": "snapshot_semantic_bytes",
            "descr": "|u1",
            "shape": ["S"],
            "digest_domain": "exact canonical semantic bytes",
        },
        "npz_probe_member_template": (
            "probe__{V1|V2|V3}__{0|1}__{trace_member}; exact per-trial shape "
            "[750,*]; only the version inventory authorized for that pool may exist"
        ),
        "npz_probe_final_digest_template": (
            "probe__{V1|V2|V3}__{0|1}__final_snapshot_semantic_digest_bytes; "
            "uint8[32]"
        ),
        "per_version_json_metadata": {
            "trial_stuck": "exact two booleans regenerated from the two raw traces",
            "trial_termination_reasons": ["H3_COMPLETE", "H3_COMPLETE"],
            "trace_member_manifests": (
                "two ordered manifests, each containing every frozen trace member with "
                "dtype.str, shape, and exact C-byte SHA-256"
            ),
            "trial_pair_comparison": (
                "exact output of compare_behavioural_probe_traces over trials 0 and 1"
            ),
        },
        "artifact_identity_cross_link": (
            "snapshot_identity.artifact_file_sha256 equals SHA-256 of the exact "
            "snapshot_payload_bytes uint8 member C bytes"
        ),
        "semantic_identity_cross_link": (
            "V3 snapshot_semantic_bytes exact length/SHA equals snapshot_semantic_evidence; "
            "snapshot_identity.snapshot_semantic_digest_v1 equals the same digest"
        ),
        "behavioural_identity_cross_link": (
            "snapshot_identity.snapshot_behavioural_digest_v1 is trial 0 only; the "
            "trial 1 digest is separately persisted and neither digest equality nor "
            "inequality substitutes for the frozen tolerance-based trace comparison"
        ),
        "save_reopen": (
            "all augmentation arrays are written in the same immutable qualification "
            "payload.npz and immediately reopened under persisted-array evidence"
        ),
        "snapshot_identity_behavioural_trial": 0,
        "trial_1_digest_is_separate": True,
        "historical_payload_copy_or_adoption": False,
    }
)
SNAPSHOT_EQUIVALENCE_RECORD_FIELDS = frozenset(
    {
        "pool_index", "candidate_spec_id", "state_id", "scene_id", "episode_id",
        "graph_id", "versions", "v1_v2_artifact_file_sha256_equal",
        "v1_v3_artifact_file_sha256_equal", "v2_v3_artifact_file_sha256_equal",
        "v1_v2_semantic_equal", "v1_v3_semantic_equal", "v2_v3_semantic_equal",
        "v1_v2_behavioural_equal", "v1_v3_behavioural_equal",
        "v2_v3_behavioural_equal", "v1_restore_trials_equal",
        "v2_restore_trials_equal", "v3_restore_trials_equal",
        "historical_behavioural_probe_evidence_present", "semantic_manifests_equal",
        "historical_comparison_applicable", "trial_1_behavioural_digests",
        "final_snapshot_semantic_digests",
        "behavioural_probe_trace_indices",
        "pass",
    }
)
SNAPSHOT_EQUIVALENCE_INDEX_FIELDS = frozenset(
    {
        "schema", "experiment_id", "serializer_authority_content_digest",
        "behavioural_probe_authority_content_digest", "row_count",
        "historical_row_count", "v3_only_row_count", "records", "pass",
        "behavioural_probe_npz_binding",
        "behavioural_probe_npz_projection_sha256",
        "first_eight_material_probe_binding",
        "first_eight_material_probe_projection_sha256",
        "first_eight_prefix_exact",
        "content_digest",
    }
)
V3_ADDITIONAL_RECOMPUTE_EVIDENCE_KEYS = frozenset(
    {
        "external_historical_custody_receipt",
        "v1_v2_custody_and_nonreuse",
        "scientific_invariance_receipt",
        "v1_v2_v3_first_eight_reproduction",
        "snapshot_equivalence_index",
        "snapshot_behavioural_probe_arrays",
        "first_eight_behavioural_probe_arrays",
    }
)
V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS = frozenset(
    {
        "serializer_authority_content_digest",
        "behavioural_probe_authority_content_digest",
        "qualification_shard_authority_content_digest",
        "snapshot_identity_fields", "snapshot_identity_contract",
        "semantic_regression_results", "qualification_state_count",
        "equivalence_row_count", "historical_row_count", "v3_only_row_count",
        "historical_semantic_pair_pass_count",
        "historical_behavioural_pair_pass_count", "v3_restore_pair_pass_count",
        "v1_v2_raw_artifact_equal_count", "v1_v3_raw_artifact_equal_count",
        "v2_v3_raw_artifact_equal_count", "raw_artifact_equality_is_descriptive",
        "behavioural_probe_trace_count", "behavioural_probe_total_samples",
        "controller_observation_persisted", "raw_policy_output_persisted",
        "first_eight_prefix_exact", "first_eight_gate_pass", "all_pass",
    }
)
V3_HISTORICAL_CUSTODY_METRIC_FIELDS = frozenset(
    {
        "external_receipt_binding", "external_receipt_projection_sha256",
        "v1_source_freeze_commit", "v2_source_freeze_commit",
        "v2_terminal_disposition", "v2_terminal_official_leaf_count",
        "v2_terminal_material_pair_count", "v1_v2_semantic_equal_count",
        "v1_v2_raw_artifact_equal_count", "raw_artifact_inequality_is_descriptive",
        "historical_deserializer_invocations", "model_initializations",
        "training_runs", "simulator_initializations", "runner_calls",
        "encoder_calls", "ranker_calls", "historical_roots_shared_inode_count",
        "historical_runtime_artifact_or_shard_reused", "all_roots_unchanged",
        "pass",
    }
)
V3_RESULT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "source_commit", "source_baseline_commit",
        "predecessor_result_commit", "development_only", "primary_classification",
        "secondary_classifications", "next_experiment", "selected_target_id",
        "evidence_counts", "panel_metrics", "development_metrics",
        "heldout_metrics", "repeatability", "command_tracking",
        "runtime_environments", "stratified_metrics", "gate",
        "component_failures", "v3_snapshot_qualification",
        "v3_historical_custody", "metrics_sha256",
        "independent_reducer_receipt_sha256", "runtime_seconds",
        "scientific_storage_bytes", "models_trained",
        "prohibited_components_trained_or_implemented", "content_digest",
    }
)
V3_SNAPSHOT_EQUIVALENCE_PUBLICATION_FIELDS = frozenset(
    {
        "terminal",
        "first_eight_row_count",
        "complete_row_count",
        "historical_row_count",
        "v3_only_row_count",
        "first_eight_receipt_exact_rebuild",
        "snapshot_equivalence_index_binding",
        "snapshot_behavioural_probe_binding",
        "snapshot_behavioural_probe_projection_sha256",
        "first_eight_material_binding",
        "first_eight_projection_sha256",
        "all_semantic_and_behavioural_rows_pass",
        "scientific_result_authorized",
    }
)
V3_RESULT_PUBLICATION_PROJECTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "result_document",
        "recomputed_metrics",
        "content_digest",
    }
)
V3_RESULT_REPORT_SECTION_ORDER = (
    "Disposition",
    "V1/V2 historical custody and V2 terminal diagnosis",
    "Snapshot identity contract and serializer fixtures",
    "V3 semantic and behavioural qualification",
    "Physical evidence and panel",
    "Development target selection",
    "Held-out conditions",
    "Reset, repeat, and controller qualification",
    "Runtime, storage, and training",
    "Claims boundary",
)
FIRST_EIGHT_REPRODUCTION_FIELDS = frozenset(
    {
        "schema", "experiment_id", "status", "historical_custody_receipt_binding",
        "comparison_rule", "row_count", "rows", "semantic_all_pass",
        "behavioural_all_pass", "pass", "technical_disposition",
        "full_collection_authorized", "compared_before_pool_index",
        "candidate_ranker_development_heldout_outcomes_opened",
        "first_eight_behavioural_probe_material_binding",
        "first_eight_behavioural_probe_projection_sha256",
    }
)
FIRST_EIGHT_REPRODUCTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "first_eight_reproduction_authority.v1"
        ),
        "pool_indices": list(range(8)),
        "row_fields": sorted(SNAPSHOT_EQUIVALENCE_RECORD_FIELDS),
        "root_fields": sorted(FIRST_EIGHT_REPRODUCTION_FIELDS),
        "comparison_rule": (
            "fresh V3 and immutable historical V1/V2 snapshots must have equal canonical "
            "semantic digests and pass pairwise frozen behavioural-probe comparisons; "
            "raw artifact-file SHA equality or inequality is descriptive only"
        ),
        "raw_artifact_sha_equality_required": False,
        "first_eight_material_probe": {
            "path": "reproduction/first_eight_behavioural_probes.npz",
            "trace_count": 48,
            "physics_samples": 36000,
            "immutable_binding_required_on_success_or_mismatch": True,
            "official_final_npz_not_written_before_pool_8": True,
        },
        "final_success_projection": (
            "snapshot_behavioural_probes.npz is written once with 544 traces after all "
            "256 V3 probes; its first 48 traces must be byte/logical-slice exact to the "
            "bound first-eight material subset"
        ),
        "final_snapshot_equivalence_index_row_count": 256,
        "final_snapshot_equivalence_index_historical_row_count": 8,
        "required_before_pool_index": 8,
        "success_status": "PASS",
        "mismatch_status": REPRODUCTION_MISMATCH_DISPOSITION,
        "mismatch_is_technical_terminal_outside_scientific_classes": True,
        "mismatch_output_leaves": list(REPRODUCTION_MISMATCH_LEAVES),
        "mismatch_forbids_official_equivalence_index_probe_npz_panel_roles_candidates_metrics_result_and_commit": True,
        "success_output_leaves": list(SUCCESS_OUTPUT_LEAVES),
    }
)


# ---------------------------------------------------------------------------
# Receipts, runtime policy, source closure
# ---------------------------------------------------------------------------

SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "v2_contract_content_digest",
        "v3_contract_content_digest", "v2_scientific_projection_sha256",
        "v3_scientific_projection_sha256", "scientific_projection_equal",
        "candidate_specs_equal", "source_dependency_paths_equal",
        "raw_previous_command_hash_authority_retained",
        "port_heading_alignment_authority_content_digest",
        "candidate_port_metric_alignment_authority_content_digest",
        "serializer_authority_content_digest",
        "historical_deserializer_authority_content_digest",
        "behavioural_probe_authority_content_digest",
        "qualification_shard_authority_content_digest",
        "result_publication_authority_content_digest",
        "external_artifact_runtime_path_correction_authority_content_digest",
        "semantic_regression_results", "all_semantic_regressions_passed",
        "official_documents_and_result_use_v3_identity_only", "pass",
    }
)
SEMANTIC_REGRESSION_RESULT_FIELDS = frozenset(
    {"requirement_id", "passed", "evidence"}
)
V1_V2_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "v3_source_freeze_commit",
        "external_historical_custody_receipt_binding",
        "external_historical_custody_projection_sha256",
        "v1_official_root_unchanged", "v1_material_root_unchanged",
        "v2_official_root_unchanged", "v2_material_root_unchanged",
        "historical_payloads_copied_into_v3", "historical_hardlinks_into_v3",
        "historical_shared_inodes_with_v3",
        "historical_runtime_artifact_or_shard_reused",
        "allowed_read_scope", "pass",
    }
)
EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "generated_before_v3_simulator_creation",
        "v1_external_custody_receipt_binding", "v1_official_root",
        "v1_material_root", "v2_official_root", "v2_material_root",
        "v2_terminal_evidence", "first_eight_pairs", "scientific_boundary",
        "repository", "immutability", "nonreuse",
    }
)
HISTORICAL_CUSTODY_ROOT_FIELDS = frozenset(
    {
        "path", "file_count", "directory_count",
        "regular_file_apparent_bytes", "regular_file_allocated_bytes",
        "directory_allocated_bytes", "allocated_bytes", "files", "directories",
        "manifest_sha256",
    }
)
HISTORICAL_CUSTODY_FILE_FIELDS = frozenset(
    {"path", "bytes", "allocated_bytes", "sha256", "device", "inode", "nlink"}
)
HISTORICAL_CUSTODY_DIRECTORY_FIELDS = frozenset(
    {"path", "allocated_bytes", "device", "inode", "nlink"}
)
HISTORICAL_CUSTODY_VERSION_FIELDS = frozenset(
    {
        "metadata_binding", "payload_binding", "artifact_file_sha256",
        "artifact_bytes", "snapshot_semantic_digest_v1",
        "semantic_payload_bytes", "semantic_payload_sha256", "semantic_evidence",
    }
)
HISTORICAL_CUSTODY_PAIR_FIELDS = frozenset(
    {
        "pool_index", "candidate_spec_id", "state_id", "scene_id", "episode_id",
        "graph_id", "v1", "v2", "payload_member_inventory_equal",
        "payload_member_dtypes_equal", "non_snapshot_member_shapes_equal",
        "non_snapshot_members_equal", "artifact_file_sha256_equal",
        "snapshot_semantic_digest_v1_equal", "semantic_evidence_equal",
        "qualified_equal", "rejection_reason_equal", "contact_sequence_equal",
        "stuck_equal", "pass",
    }
)
HISTORICAL_CUSTODY_V2_TERMINAL_FIELDS = frozenset(
    {
        "disposition", "official_leaf_count", "material_pair_count",
        "full_collection_authorized",
        "candidate_ranker_development_heldout_outcomes_opened",
        "external_regeneration_receipt_present", "first_eight_receipt_binding",
    }
)
HISTORICAL_CUSTODY_SCIENTIFIC_BOUNDARY_FIELDS = frozenset(
    {
        "v1_teacher_qualification_rows_opened", "v1_teacher_qualified",
        "v1_teacher_rejected", "v2_fresh_teacher_qualification_rows_opened",
        "v2_teacher_qualified", "v2_teacher_rejected",
        "v2_pool_002_physics_contact_reproduced", "selection_performed",
        "reset_fixture_executions", "candidate_fanout_executions",
        "encoder_initializations", "ranker_inference_calls",
        "heldout_outcomes_opened", "metrics_persisted", "result_persisted",
    }
)
HISTORICAL_CUSTODY_REPOSITORY_FIELDS = frozenset(
    {
        "v1_source_freeze_commit", "v2_source_freeze_commit",
        "v1_freeze_subject", "v2_freeze_subject",
    }
)
HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS = frozenset(
    {
        "audit_mode", "v1_official_root_unchanged_during_audit",
        "v1_material_root_unchanged_during_audit",
        "v2_official_root_unchanged_during_audit",
        "v2_material_root_unchanged_during_audit",
        "all_files_regular_single_link", "receipt_outside_all_experiment_roots",
    }
)
HISTORICAL_CUSTODY_NONREUSE_FIELDS = frozenset(
    {
        "historical_roots_shared_inode_count",
        "v1_v2_artifact_file_sha256_equal_count",
        "v1_v2_snapshot_semantic_digest_v1_equal_count",
        "raw_artifact_inequality_is_descriptive",
        "historical_payload_copy_into_v3_count",
        "historical_hardlink_into_v3_count",
        "historical_runtime_artifact_or_shard_reused",
        "historical_deserializer_invocations", "model_initializations",
        "training_runs", "simulator_initializations", "runner_calls",
        "encoder_calls", "ranker_calls",
    }
)

V1_V2_CUSTODY_AND_NONREUSE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "v1_v2_custody_and_nonreuse_authority.v1"
        ),
        "external_receipt_path_authority": copy.deepcopy(
            HISTORICAL_CUSTODY_RECEIPT_PATH_AUTHORITY
        ),
        "external_receipt_binding": copy.deepcopy(
            HISTORICAL_CUSTODY_RECEIPT_BINDING
        ),
        "v1_external_receipt_binding": copy.deepcopy(V1_CUSTODY_RECEIPT_BINDING),
        "v2_terminal_root_binding": copy.deepcopy(V2_TERMINAL_ROOT_BINDING),
        "v1_v2_roots_must_remain_byte_and_inode_stable": True,
        "external_receipt_schema": (
            "physical_graph_edge_handoff_qualification_v1_v2.custody_receipt.v1"
        ),
        "external_receipt_experiment_id": EXPERIMENT_ID,
        "external_receipt_fields": sorted(
            EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS
        ),
        "root_fields": sorted(HISTORICAL_CUSTODY_ROOT_FIELDS),
        "root_file_fields": sorted(HISTORICAL_CUSTODY_FILE_FIELDS),
        "root_directory_fields": sorted(HISTORICAL_CUSTODY_DIRECTORY_FIELDS),
        "historical_version_fields": sorted(HISTORICAL_CUSTODY_VERSION_FIELDS),
        "historical_pair_fields": sorted(HISTORICAL_CUSTODY_PAIR_FIELDS),
        "v2_terminal_fields": sorted(HISTORICAL_CUSTODY_V2_TERMINAL_FIELDS),
        "scientific_boundary_fields": sorted(
            HISTORICAL_CUSTODY_SCIENTIFIC_BOUNDARY_FIELDS
        ),
        "repository_fields": sorted(HISTORICAL_CUSTODY_REPOSITORY_FIELDS),
        "immutability_fields": sorted(HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS),
        "nonreuse_fields": sorted(HISTORICAL_CUSTODY_NONREUSE_FIELDS),
        "historical_deserializer_invocations": 16,
        "first_eight_pair_count": 8,
        "semantic_pair_equality_required": True,
        "semantic_evidence_pair_equality_required": True,
        "raw_artifact_pair_equality_required": False,
        "external_projection_digest_domain": (
            "SHA-256 of the complete strictly validated external receipt encoded as "
            "canonical compact JSON without the trailing LF"
        ),
        "historical_runtime_artifact_or_shard_reuse_authorized": False,
        "historical_copy_or_hardlink_into_v3_authorized": False,
        "permitted_reads": (
            "read-only custody verification, restricted semantic reconstruction, and "
            "behavioural probe restoration for pool indices 0..7 only"
        ),
        "v3_first_eight_must_be_fresh_physical_executions": True,
    }
)

RESULT_PUBLICATION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "result_publication_authority.v1"
        ),
        "result_fields": sorted(V3_RESULT_FIELDS),
        "publication_projection_fields": sorted(
            V3_RESULT_PUBLICATION_PROJECTION_FIELDS
        ),
        "snapshot_equivalence_publication_fields": sorted(
            V3_SNAPSHOT_EQUIVALENCE_PUBLICATION_FIELDS
        ),
        "scientific_input_binding_count": OUTPUT_LEAF_COUNT - 3,
        "scientific_input_binding_fields": ["path", "bytes", "sha256"],
        "scientific_storage_bytes_rule": (
            "exact sum of bytes over the 25 pre-publication official scientific "
            "leaf bindings"
        ),
        "metrics_sha256_rule": (
            "exact SHA-256 of canonical LF-terminated metrics.json bytes and exact "
            "metrics.json scientific binding"
        ),
        "contract_sha256_rule": (
            "exact SHA-256 of canonical LF-terminated runtime contract.json bytes and "
            "exact contract.json scientific binding"
        ),
        "historical_and_snapshot_cross_links_required": True,
        "build_api": "build_result_publication_projection",
        "validate_api": "validate_result_publication_projection",
        "report_bytes_api": "build_result_report_bytes",
        "snapshot_qualification_fields": sorted(
            V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS
        ),
        "historical_custody_fields": sorted(
            V3_HISTORICAL_CUSTODY_METRIC_FIELDS
        ),
        "report_section_order": list(V3_RESULT_REPORT_SECTION_ORDER),
        "result_schema": (
            "physical_graph_edge_handoff_qualification_v3.result.v1"
        ),
        "no_training": True,
        "scientific_classification_and_next_decision": (
            "exact inherited V2/V1 metrics projection; the reproduction mismatch is a "
            "technical terminal and never receives a scientific result"
        ),
        "result_and_report_rebuilt_by_independent_evaluator": True,
    }
)

V3_RUNTIME_POLICY = {
    "before_simulator_creation": [
        "validate exact V2 scientific invariance and both inherited alignments",
        "validate the immutable combined V1/V2 custody receipt and all historical roots",
        "pass the exact persisted-array and semantic-serializer regressions",
        "prove V3 official/material roots are fresh and share no inode with V1/V2",
    ],
    "first_eight": (
        "freshly execute V3 pool indices 0..7; historical artifacts may be opened only "
        "through the restricted read-only semantic/probe authorities; for every fresh V3 "
        "qualification shard, serialize semantic evidence and complete both behavioural "
        "restore trials before exposing teacher eligibility"
    ),
    "before_pool_index_8": (
        "persist and validate v1_v2_v3_first_eight_reproduction.json; continue iff all "
        "semantic and behavioural comparisons pass"
    ),
    "mismatch": (
        f"terminal {REPRODUCTION_MISMATCH_DISPOSITION} outside the six scientific "
        "classes; retain exactly four official leaves and no scientific result commit"
    ),
    "success": (
        "resume the exact inherited V2/V1 stage order beginning at pool index 8; each of "
        "the remaining 248 V3 qualification shards must likewise complete both behavioural "
        "restore probes before its teacher eligibility is exposed"
    ),
    "historical_runtime_artifact_reuse": False,
    "historical_copy_or_hardlink": False,
    "changed_scientific_gate_formula_threshold_or_tuning": False,
    "raw_previous_command_hash_correction_retained": True,
    "external_artifact_runtime_path_correction_authority_content_digest": (
        EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY["content_digest"]
    ),
    "authorized_inherited_implementation_alignments": [
        PORT_HEADING_ALIGNMENT_DISPOSITION,
        CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION,
    ],
}

DOC_PREFIX = "docs/lewm_go2_physical_graph_edge_handoff_qualification_v3"
TRACKED_SOURCE_PATHS = (
    f"{DOC_PREFIX}_contract_2026-09-02.json",
    f"{DOC_PREFIX}_fixture_2026-09-02.json",
    f"{DOC_PREFIX}_output_schema_2026-09-02.json",
    f"{DOC_PREFIX}_preregistration_2026-09-02.md",
    f"{DOC_PREFIX}_source_closure_2026-09-02.json",
    f"{DOC_PREFIX}_scientific_invariance_2026-09-02.json",
    f"{DOC_PREFIX}_v1_v2_custody_binding_2026-09-02.json",
    "lewm/safety/physical_graph_edge_handoff_snapshot_semantics_v1.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v3_contract.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v3_metrics.py",
    "lewm/tests/test_physical_graph_edge_handoff_snapshot_semantics_v1.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v3_contract.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v3_metrics.py",
    "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v3.py",
    "lewm/tests/test_evaluate_physical_graph_edge_handoff_qualification_v3.py",
    "scripts/run_physical_graph_edge_handoff_qualification_v3.py",
    "scripts/evaluate_physical_graph_edge_handoff_qualification_v3.py",
)
SOURCE_DEPENDENCY_PATHS = tuple(V2.SOURCE_DEPENDENCY_PATHS)
V3_WRAPPER_DEPENDENCY_PATHS = tuple(V2.SOURCE_CLOSURE_PATHS)
SOURCE_CLOSURE_PATHS = (
    TRACKED_SOURCE_PATHS[7:] + V3_WRAPPER_DEPENDENCY_PATHS
)


def build_candidate_specs() -> list[dict[str, Any]]:
    return V2.build_candidate_specs()


def build_prospective_pool_specs() -> list[dict[str, Any]]:
    return V2.build_prospective_pool_specs()


def scientific_invariance_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the inherited V1/V2 scientific design from a V3 contract."""

    return V2.scientific_invariance_projection(value)


def scientific_constant_projection() -> dict[str, Any]:
    """Return the exact inherited scientific constant projection."""

    return copy.deepcopy(V2_SCIENTIFIC_CONSTANT_PROJECTION)


SCIENTIFIC_INVARIANCE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "scientific_invariance_authority.v1"
        ),
        "v2_experiment_id": V2_EXPERIMENT_ID,
        "v3_experiment_id": EXPERIMENT_ID,
        "v2_source_freeze_commit": V2_SOURCE_FREEZE_COMMIT,
        "v2_contract_content_digest": V2_CONTRACT_CONTENT_DIGEST,
        "v2_scientific_projection_sha256": V2_SCIENTIFIC_PROJECTION_SHA256,
        "v3_scientific_projection_sha256": V2_SCIENTIFIC_PROJECTION_SHA256,
        "v2_scientific_constants_sha256": V2_SCIENTIFIC_CONSTANTS_SHA256,
        "v3_scientific_constants_sha256": V2_SCIENTIFIC_CONSTANTS_SHA256,
        "v2_candidate_specs_sha256": V2_CANDIDATE_SPECS_SHA256,
        "v3_candidate_specs_sha256": V2_CANDIDATE_SPECS_SHA256,
        "v2_source_dependency_paths_sha256": V2_SOURCE_DEPENDENCY_PATHS_SHA256,
        "v3_source_dependency_paths_sha256": V2_SOURCE_DEPENDENCY_PATHS_SHA256,
        "logical_scene_state_spec_episode_ids_and_seeds_unchanged": True,
        "all_metric_formulas_gates_classes_precedence_and_next_decisions_unchanged": True,
        "all_model_controller_geometry_candidate_and_runtime_science_unchanged": True,
        "raw_previous_command_hash_authority_retained": True,
        "port_heading_implementation_alignment_authority_content_digest": (
            PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
        ),
        "candidate_port_metric_implementation_alignment_authority_content_digest": (
            CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
        ),
        "external_artifact_runtime_path_correction_authority_content_digest": (
            EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY["content_digest"]
        ),
        "external_artifact_runtime_path_changes_bytes_or_scientific_identity": False,
        "v3_changes_only_identity_custody_and_snapshot_semantic_behavioural_qualification": True,
    }
)


def build_contract() -> dict[str, Any]:
    base = copy.deepcopy(V2.V1_SCIENTIFIC_CONTRACT)
    base.pop("content_digest", None)
    base.update(
        {
            "schema": "physical_graph_edge_handoff_qualification_v3.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "output": {
                "root": str(OUTPUT_ROOT),
                "material_root": str(MATERIAL_ROOT),
                "external_regeneration_receipt": str(
                    EXTERNAL_REGENERATION_RECEIPT
                ),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "leaf_count": OUTPUT_LEAF_COUNT,
                "successful_complete_inventory": list(SUCCESS_OUTPUT_LEAVES),
                "reproduction_mismatch_inventory": list(
                    REPRODUCTION_MISMATCH_LEAVES
                ),
                "receipt_self_digests": False,
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "persisted_array_hash_authority": copy.deepcopy(
                PERSISTED_ARRAY_HASH_AUTHORITY
            ),
            "regression_gate_authority": copy.deepcopy(REGRESSION_GATE_AUTHORITY),
            "scientific_invariance_authority": copy.deepcopy(
                SCIENTIFIC_INVARIANCE_AUTHORITY
            ),
            "port_heading_implementation_alignment_authority": copy.deepcopy(
                PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "candidate_port_metric_implementation_alignment_authority": copy.deepcopy(
                CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "snapshot_semantic_serializer_authority": copy.deepcopy(
                SEMANTIC_SERIALIZER_AUTHORITY
            ),
            "historical_snapshot_deserializer_authority": copy.deepcopy(
                HISTORICAL_DESERIALIZER_AUTHORITY
            ),
            "snapshot_behavioural_probe_authority": copy.deepcopy(
                BEHAVIOURAL_PROBE_AUTHORITY
            ),
            "qualification_shard_augmentation_authority": copy.deepcopy(
                QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY
            ),
            "v1_v2_custody_and_nonreuse_authority": copy.deepcopy(
                V1_V2_CUSTODY_AND_NONREUSE_AUTHORITY
            ),
            "first_eight_reproduction_authority": copy.deepcopy(
                FIRST_EIGHT_REPRODUCTION_AUTHORITY
            ),
            "result_publication_authority": copy.deepcopy(
                RESULT_PUBLICATION_AUTHORITY
            ),
            "external_artifact_runtime_path_correction_authority": copy.deepcopy(
                EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY
            ),
            "v3_runtime_policy": copy.deepcopy(V3_RUNTIME_POLICY),
            "v3_wrapper_dependency_paths": list(V3_WRAPPER_DEPENDENCY_PATHS),
        }
    )
    result = attach_content_digest(base)
    if scientific_invariance_projection(result) != V2_SCIENTIFIC_PROJECTION:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "V3 scientific projection differs from V2"
        )
    if build_candidate_specs() != V2.build_candidate_specs():
        raise PhysicalGraphEdgeHandoffV3ContractError("V2 candidate specs changed")
    if tuple(SOURCE_DEPENDENCY_PATHS) != tuple(V2.SOURCE_DEPENDENCY_PATHS):
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "V2 source dependency authority changed"
        )
    return result


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV3ContractError("contract value drift")
    return copy.deepcopy(dict(value))


RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "status", "source_freeze_commit",
        "source_parent_commit", "source_baseline_commit", "v1_source_freeze_commit",
        "v2_source_freeze_commit", "scientific_contract",
        "predecessor_result_binding", "external_artifact_bindings",
        "runtime_policy", "historical_custody_receipt_binding",
        "v3_runtime_policy", "content_digest",
    }
)


def _commit(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalGraphEdgeHandoffV3ContractError(f"invalid {label}")
    return value


def _historical_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != HISTORICAL_CUSTODY_BINDING_FIELDS:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "historical custody binding field drift"
        )
    row = copy.deepcopy(dict(value))
    if row["path"] != str(HISTORICAL_CUSTODY_RECEIPT_PATH):
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "historical custody path drift"
        )
    if (
        not isinstance(row["bytes"], int)
        or isinstance(row["bytes"], bool)
        or row["bytes"] <= 0
        or not isinstance(row["sha256"], str)
        or len(row["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in row["sha256"])
    ):
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "historical custody binding value drift"
        )
    if row != HISTORICAL_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "historical custody binding differs from the pre-simulator immutable receipt"
        )
    return row


def build_runtime_contract(
    source_freeze_commit: str,
    historical_custody_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    source_freeze_commit = _commit(source_freeze_commit, "source_freeze_commit")
    binding = _historical_binding(historical_custody_receipt_binding)
    return attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3.runtime_contract.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "source_freeze_commit": source_freeze_commit,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "v1_source_freeze_commit": V1_SOURCE_FREEZE_COMMIT,
            "v2_source_freeze_commit": V2_SOURCE_FREEZE_COMMIT,
            "scientific_contract": build_contract(),
            "predecessor_result_binding": copy.deepcopy(
                V2.V1.PREDECESSOR_RESULT_BINDING
            ),
            "external_artifact_bindings": [
                copy.deepcopy(row) for row in EXTERNAL_ARTIFACT_BINDINGS
            ],
            "runtime_policy": copy.deepcopy(V2.V1.DIRECT_RUNTIME_POLICY),
            "historical_custody_receipt_binding": binding,
            "v3_runtime_policy": copy.deepcopy(V3_RUNTIME_POLICY),
        }
    )


def validate_runtime_contract(
    value: Mapping[str, Any],
    *,
    source_freeze_commit: str | None = None,
    historical_custody_receipt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != RUNTIME_CONTRACT_FIELDS:
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "runtime contract field drift"
        )
    observed = _commit(str(row["source_freeze_commit"]), "source_freeze_commit")
    if source_freeze_commit is not None and observed != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffV3ContractError("source freeze commit drift")
    binding = _historical_binding(row["historical_custody_receipt_binding"])
    if historical_custody_receipt_binding is not None and binding != _historical_binding(
        historical_custody_receipt_binding
    ):
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "historical custody binding cross-link drift"
        )
    expected = build_runtime_contract(observed, binding)
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV3ContractError(
            "runtime contract value drift"
        )
    return copy.deepcopy(row)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "PhysicalGraphEdgeHandoffV3ContractError",
    "attach_content_digest",
    "build_candidate_specs",
    "build_contract",
    "build_prospective_pool_specs",
    "build_runtime_contract",
    "canonical_json_bytes",
    "scientific_constant_projection",
    "scientific_invariance_projection",
    "validate_content_digest",
    "validate_contract",
    "validate_runtime_contract",
]
