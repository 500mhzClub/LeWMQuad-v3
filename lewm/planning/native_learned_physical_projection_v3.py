"""Standalone V3 development boundary for native learned physical evidence.

V3 owns its complete admission, issuance, commit, and retraction lifecycle.  It
reuses only frozen V1 raw-data, calibration, receipt, and pure geometry helpers;
it does not subclass, compose, retain, or import an older adapter class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from lewm.planning.native_learned_physical_projection_v1 import (
    FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1,
    FrozenNativeLearnedProjectionCalibrationV1,
    NativeLearnedPhysicalProjectionReceiptV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionRejectedError,
    NativeLearnedProjectionReplayError,
    NativeV4SourceGeometryV1,
    RawGroundClearCellQueriesV1,
    RawOrderedRayHitDepthV1,
    SyntheticNativeV4RawOutcomeV1,
    SyntheticNativeV4RunnerV1,
    _apply_transform,
    _candidate_cells_for_polygon,
    _cells_json,
    _closed_point_supercover,
    _compose_transform,
    _destination_square,
    _inverse_transform,
    _polygon_covered_by_source_union,
    _polygons_intersect_closed,
    _require_sha256,
    _sha256,
    _sigmoid,
    _source_square,
    canonical_ground_query_xy_body_v1,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    TransactionReceipt,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    PHYSICAL_CELL_SIZE_M,
    PROFILE_SHA256,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


Cell = tuple[int, int]
XY = tuple[float, float]
Transform2 = tuple[float, float, float]

PRODUCTION_NATIVE_V4_RUNNER = None
PRODUCTION_V4_CHECKPOINT_FILE_SHA256 = None
PRODUCTION_G2_REPORT_FILE_SHA256 = None
PRODUCTION_V4_CALIBRATION_SHA256 = None
PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V3 = None


@dataclass(frozen=True)
class QualifiedLearnedPhysicalDevelopmentAdmissionV3:
    admission_kind: str
    admission_id_sha256: str
    adapter_contract_sha256: str
    source_outcome_sha256: str
    projection_receipt_sha256: str
    physical_transaction_sha256: str
    observation_id: str
    observation_payload_sha256: str
    observation_producer_sha256: str
    pose_sha256: str
    memory_config_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision_before: int
    configuration_revision: int
    runner_execution_identity_sha256: str
    inference_implementation_sha256: str
    projection_implementation_sha256: str
    access_ledger_source_sha256: str
    checkpoint_file_sha256: str
    g2_report_file_sha256: str
    calibration_sha256: str
    rgb_frame_sha256: str
    raw_outcome_file_sha256: str
    retracts_observation_id: str | None
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.admission_kind not in {"projection", "retraction"}:
            raise ValueError("unsupported V3 development admission kind")
        for name in (
            "admission_id_sha256",
            "adapter_contract_sha256",
            "source_outcome_sha256",
            "projection_receipt_sha256",
            "physical_transaction_sha256",
            "observation_payload_sha256",
            "observation_producer_sha256",
            "pose_sha256",
            "memory_config_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "runner_execution_identity_sha256",
            "inference_implementation_sha256",
            "projection_implementation_sha256",
            "access_ledger_source_sha256",
            "checkpoint_file_sha256",
            "g2_report_file_sha256",
            "calibration_sha256",
            "rgb_frame_sha256",
            "raw_outcome_file_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.observation_id) is not str or not self.observation_id:
            raise ValueError("observation_id must be nonempty")
        if (
            isinstance(self.physical_revision_before, bool)
            or not isinstance(self.physical_revision_before, int)
            or self.physical_revision_before < 0
        ):
            raise ValueError("physical_revision_before must be non-negative")
        if (
            isinstance(self.configuration_revision, bool)
            or not isinstance(self.configuration_revision, int)
            or self.configuration_revision <= 0
        ):
            raise ValueError("configuration_revision must be positive")
        if self.admission_kind == "projection":
            if self.retracts_observation_id is not None:
                raise ValueError("projection admission cannot retract an observation")
        elif (
            type(self.retracts_observation_id) is not str
            or not self.retracts_observation_id
        ):
            raise ValueError("retraction admission requires an observation ID")
        expected_id = _sha256(
            {
                "schema": "lewm_g3_qualified_learned_admission_id_v3",
                "kind": self.admission_kind,
                "adapter_contract_sha256": self.adapter_contract_sha256,
                "source_outcome_sha256": self.source_outcome_sha256,
                "projection_receipt_sha256": self.projection_receipt_sha256,
                "physical_transaction_sha256": self.physical_transaction_sha256,
                "observation_id": self.observation_id,
                "memory_revision_before": self.physical_revision_before,
                "retracts_observation_id": self.retracts_observation_id,
            }
        )
        if self.admission_id_sha256 != expected_id:
            raise NativeLearnedProjectionBindingError(
                "V3 development admission identity changed"
            )
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_qualified_learned_development_admission_v3",
            "admission_kind": self.admission_kind,
            "admission_id_sha256": self.admission_id_sha256,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "source_outcome_sha256": self.source_outcome_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "physical_transaction_sha256": self.physical_transaction_sha256,
            "observation_id": self.observation_id,
            "observation_payload_sha256": self.observation_payload_sha256,
            "observation_producer_sha256": self.observation_producer_sha256,
            "pose_sha256": self.pose_sha256,
            "memory_config_sha256": self.memory_config_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": (
                self.configuration_map_frame_sha256
            ),
            "physical_revision_before": self.physical_revision_before,
            "configuration_revision": self.configuration_revision,
            "runner_execution_identity_sha256": (
                self.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": (
                self.inference_implementation_sha256
            ),
            "projection_implementation_sha256": (
                self.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": self.access_ledger_source_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "g2_report_file_sha256": self.g2_report_file_sha256,
            "calibration_sha256": self.calibration_sha256,
            "rgb_frame_sha256": self.rgb_frame_sha256,
            "raw_outcome_file_sha256": self.raw_outcome_file_sha256,
            "retracts_observation_id": self.retracts_observation_id,
            "development_only": True,
            "hardware_execution_authorized": False,
            "production_promotion_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "V3 development admission was mutated"
            )


@dataclass(frozen=True)
class QualifiedLearnedPhysicalDevelopmentTransactionV3:
    admission: QualifiedLearnedPhysicalDevelopmentAdmissionV3
    pose: PoseProvenance
    projection_receipt: NativeLearnedPhysicalProjectionReceiptV1
    retracts_observation_id: str | None
    _issuance_capability_v3: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.admission) is not QualifiedLearnedPhysicalDevelopmentAdmissionV3:
            raise TypeError("V3 transaction admission has the wrong type")
        if type(self.pose) is not PoseProvenance:
            raise TypeError("V3 transaction pose has the wrong type")
        if type(self.projection_receipt) is not NativeLearnedPhysicalProjectionReceiptV1:
            raise TypeError("V3 projection receipt has the wrong type")
        self._assert_bindings()
        if self._issuance_capability_v3 is None:
            raise TypeError("V3 transaction requires an issuance capability")
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def _assert_bindings(self) -> None:
        self.admission.assert_integrity()
        self.projection_receipt.assert_integrity()
        if self.pose.content_sha256 != self.admission.pose_sha256:
            raise NativeLearnedProjectionBindingError(
                "V3 transaction pose binding changed"
            )
        if (
            self.projection_receipt.content_sha256
            != self.admission.projection_receipt_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "V3 transaction projection receipt binding changed"
            )
        if self.retracts_observation_id != self.admission.retracts_observation_id:
            raise NativeLearnedProjectionBindingError(
                "V3 transaction retraction binding changed"
            )

    @property
    def observation_id(self) -> str:
        return self.admission.observation_id

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g3_qualified_learned_development_transaction_v3",
            "admission": self.admission.to_dict(),
            "pose": self.pose.to_dict(),
            "projection_receipt": self.projection_receipt.to_dict(),
            "retracts_observation_id": self.retracts_observation_id,
            "development_only": True,
            "hardware_execution_authorized": False,
            "production_promotion_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        self._assert_bindings()
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise NativeLearnedProjectionBindingError(
                "V3 development transaction was mutated"
            )

    def __copy__(self) -> "QualifiedLearnedPhysicalDevelopmentTransactionV3":
        raise TypeError("V3 development transactions are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "QualifiedLearnedPhysicalDevelopmentTransactionV3":
        del memo
        raise TypeError("V3 development transactions are non-copyable")


class _RetractionStateV3(str, Enum):
    LIVE = "live"
    STALE = "stale"
    CONSUMED = "consumed"


@dataclass
class _RetractionReservationV3:
    target_key: tuple[str, int]
    target_package: QualifiedLearnedPhysicalDevelopmentTransactionV3
    target_issued_content_sha256: str
    retraction_package: QualifiedLearnedPhysicalDevelopmentTransactionV3
    retraction_issued_content_sha256: str
    snapshot: TwoResolutionConfigurationSnapshotV2
    snapshot_issued_content_sha256: str
    state: _RetractionStateV3 = _RetractionStateV3.LIVE


class NativeLearnedPhysicalProjectionAdapterV3:
    """One standalone issue/commit authority for synthetic native evidence."""

    __slots__ = (
        "__memory_v3",
        "__projection_v3",
        "__runner_v3",
        "__calibration_v3",
        "__expected_v3",
        "__contract_v3",
        "__capability_v3",
        "__issued_v3",
        "__snapshots_v3",
        "__consumed_v3",
        "__committed_v3",
        "__reservations_v3",
        "__live_by_target_v3",
        "__sequence_v3",
    )

    def __init__(
        self,
        *,
        memory: RevisionedPhysicalMemory,
        projection: TwoResolutionConfigurationProjectionV2,
        runner: SyntheticNativeV4RunnerV1,
        calibration: FrozenNativeLearnedProjectionCalibrationV1,
        runner_execution_identity_sha256: str,
        inference_implementation_sha256: str,
        projection_implementation_sha256: str,
        access_ledger_source_sha256: str,
        checkpoint_file_sha256: str,
        g2_report_file_sha256: str,
        camera_transform_sha256: str,
        _synthetic_test_fixture: bool = False,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError(
                "no production native learned physical projection V3 is configured"
            )
        if type(memory) is not RevisionedPhysicalMemory:
            raise TypeError("memory must be RevisionedPhysicalMemory")
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError(
                "projection must be TwoResolutionConfigurationProjectionV2"
            )
        if getattr(projection, "_memory", None) is not memory:
            raise NativeLearnedProjectionBindingError(
                "projection does not own the supplied physical memory"
            )
        if type(runner) is not SyntheticNativeV4RunnerV1:
            raise TypeError("runner must be SyntheticNativeV4RunnerV1")
        if calibration is not FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1:
            raise NativeLearnedProjectionBindingError(
                "V3 requires the exact frozen synthetic calibration object"
            )
        calibration.assert_integrity()
        expected = {
            "runner_execution_identity_sha256": _require_sha256(
                runner_execution_identity_sha256,
                "runner_execution_identity_sha256",
            ),
            "inference_implementation_sha256": _require_sha256(
                inference_implementation_sha256,
                "inference_implementation_sha256",
            ),
            "projection_implementation_sha256": _require_sha256(
                projection_implementation_sha256,
                "projection_implementation_sha256",
            ),
            "access_ledger_source_sha256": _require_sha256(
                access_ledger_source_sha256,
                "access_ledger_source_sha256",
            ),
            "checkpoint_file_sha256": _require_sha256(
                checkpoint_file_sha256,
                "checkpoint_file_sha256",
            ),
            "g2_report_file_sha256": _require_sha256(
                g2_report_file_sha256,
                "g2_report_file_sha256",
            ),
            "camera_transform_sha256": _require_sha256(
                camera_transform_sha256,
                "camera_transform_sha256",
            ),
        }
        if memory.config.promoted_runtime:
            raise PermissionError(
                "development learned projection V3 cannot target promoted memory"
            )
        if (
            not math.isclose(
                memory.map_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or memory.config.require_registered_lattice
            or memory.config.physical_projection_contract_sha256 != PROFILE_SHA256
            or memory.config.expected_camera_transform_sha256
            != expected["camera_transform_sha256"]
            or memory.config.pose_covariance_diagonal_limits
            != calibration.covariance_diagonal_max
        ):
            raise NativeLearnedProjectionBindingError(
                "development memory is not the native G3 V2 0.05 m contract"
            )
        runner_bindings = {
            name: getattr(runner, name)
            for name in (
                "runner_execution_identity_sha256",
                "inference_implementation_sha256",
                "projection_implementation_sha256",
                "access_ledger_source_sha256",
                "checkpoint_file_sha256",
                "g2_report_file_sha256",
            )
        }
        if runner_bindings != {
            name: expected[name] for name in runner_bindings
        } or runner.calibration_sha256 != calibration.content_sha256:
            raise NativeLearnedProjectionBindingError(
                "runner/checkpoint/G2/calibration identity differs from V3"
            )
        self.__memory_v3 = memory
        self.__projection_v3 = projection
        self.__runner_v3 = runner
        self.__calibration_v3 = calibration
        self.__expected_v3 = expected
        self.__contract_v3 = _sha256(
            {
                "schema": "lewm_g3_native_learned_projection_contract_v3",
                **expected,
                "calibration_sha256": calibration.content_sha256,
                "physical_cell_size_m": PHYSICAL_CELL_SIZE_M,
                "configuration_cell_size_m": CONFIGURATION_CELL_SIZE_M,
                "standalone_adapter_state": True,
                "older_adapter_composition": False,
                "v3_package_type_required": True,
                "development_only": True,
                "hardware_execution_authorized": False,
                "production_promotion_authorized": False,
            }
        )
        self.__capability_v3 = object()
        self.__issued_v3: dict[
            int,
            tuple[QualifiedLearnedPhysicalDevelopmentTransactionV3, str],
        ] = {}
        self.__snapshots_v3: dict[
            int,
            tuple[TwoResolutionConfigurationSnapshotV2, str],
        ] = {}
        self.__consumed_v3: set[int] = set()
        self.__committed_v3: dict[
            str,
            QualifiedLearnedPhysicalDevelopmentTransactionV3,
        ] = {}
        self.__reservations_v3: dict[
            int,
            _RetractionReservationV3,
        ] = {}
        self.__live_by_target_v3: dict[tuple[str, int], int] = {}
        self.__sequence_v3 = 0

    def __copy__(self) -> "NativeLearnedPhysicalProjectionAdapterV3":
        raise TypeError("native learned projection V3 adapters are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "NativeLearnedPhysicalProjectionAdapterV3":
        del memo
        raise TypeError("native learned projection V3 adapters are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError(
            "native learned projection V3 adapters are non-serializable"
        )

    @property
    def adapter_contract_sha256(self) -> str:
        return self.__contract_v3

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def _assert_outcome_bindings_v3(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> None:
        self.__calibration_v3.assert_integrity()
        expected = self.__expected_v3
        if (
            outcome.runner_execution_identity_sha256
            != expected["runner_execution_identity_sha256"]
            or outcome.inference_implementation_sha256
            != expected["inference_implementation_sha256"]
            or outcome.projection_implementation_sha256
            != expected["projection_implementation_sha256"]
            or outcome.access_ledger_source_sha256
            != expected["access_ledger_source_sha256"]
            or outcome.checkpoint_file_sha256
            != expected["checkpoint_file_sha256"]
            or outcome.g2_report_file_sha256
            != expected["g2_report_file_sha256"]
            or outcome.calibration_sha256
            != self.__calibration_v3.content_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "runner/checkpoint/G2/calibration/source identity changed"
            )
        if (
            outcome.physical_map_frame is not snapshot.physical_map_frame
            or outcome.configuration_map_frame
            is not snapshot.configuration_map_frame
            or outcome.physical_map_frame.content_sha256
            != snapshot.physical_map_frame_sha256
            or outcome.configuration_map_frame.content_sha256
            != snapshot.configuration_map_frame_sha256
            or outcome.physical_shape != snapshot.physical_shape
            or outcome.configuration_shape != snapshot.configuration_shape
            or outcome.physical_shape
            != (
                2 * outcome.configuration_shape[0],
                2 * outcome.configuration_shape[1],
            )
            or outcome.physical_map_frame.origin_xy_m
            != outcome.configuration_map_frame.origin_xy_m
            or not math.isclose(
                outcome.physical_map_frame.cell_size_m,
                PHYSICAL_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                outcome.configuration_map_frame.cell_size_m,
                CONFIGURATION_CELL_SIZE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome two-frame origin/shape identity changed"
            )
        if (
            outcome.physical_revision != snapshot.physical_revision
            or outcome.configuration_revision != snapshot.configuration_revision
            or outcome.physical_content_sha256
            != snapshot.physical_content_sha256
            or outcome.configuration_snapshot_sha256 != snapshot.content_sha256
            or outcome.projection_source_sha256
            != snapshot.projection_source_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome physical/configuration revision or source changed"
            )
        if (
            outcome.pose.source is not PoseSource.DEPLOYMENT_ODOMETRY
            or outcome.pose.frame_id != snapshot.physical_map_frame.frame_id
            or outcome.pose.camera_transform_sha256
            != expected["camera_transform_sha256"]
        ):
            raise NativeLearnedProjectionBindingError(
                "raw outcome pose/camera identity changed"
            )
        geometry = outcome.source_geometry
        if (
            outcome.source_derivation != "native_raw_v4_0p05"
            or geometry.shape != self.__calibration_v3.source_shape
            or geometry.origin_forward_left_m
            != self.__calibration_v3.source_origin_forward_left_m
            or not math.isclose(
                geometry.cell_size_m,
                self.__calibration_v3.source_cell_size_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise NativeLearnedProjectionRejectedError(
                "raw source is not native V4 0.05 m logits; "
                "0.10 m/upsampling rejects"
            )
        covariance = outcome.pose.covariance_xy_yaw
        if any(
            abs(covariance[row][column]) > 1e-12
            for row in range(3)
            for column in range(3)
            if row != column
        ):
            raise NativeLearnedProjectionRejectedError(
                "registered development uncertainty requires diagonal covariance"
            )
        diagonal = tuple(covariance[index][index] for index in range(3))
        if any(
            value > limit + 1e-12
            for value, limit in zip(
                diagonal,
                self.__calibration_v3.covariance_diagonal_max,
            )
        ):
            raise NativeLearnedProjectionRejectedError(
                "pose covariance exceeds frozen learned-projection envelope"
            )

    def _uncertainty_transforms_v3(
        self,
        pose: PoseProvenance,
    ) -> tuple[Transform2, ...]:
        diagonal = tuple(
            pose.covariance_xy_yaw[index][index] for index in range(3)
        )
        standard_deviation = tuple(math.sqrt(value) for value in diagonal)
        transforms: list[Transform2] = []
        for multiplier in self.__calibration_v3.pose_sigma_multipliers:
            pose_transform = (
                pose.mean_xy_yaw[0] + multiplier[0] * standard_deviation[0],
                pose.mean_xy_yaw[1] + multiplier[1] * standard_deviation[1],
                pose.mean_xy_yaw[2] + multiplier[2] * standard_deviation[2],
            )
            for camera_transform in (
                self.__calibration_v3.camera_local_uncertainty_transforms
            ):
                transforms.append(
                    _compose_transform(pose_transform, camera_transform)
                )
        unique: list[Transform2] = []
        for transform in transforms:
            if not any(
                all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
                    for left, right in zip(transform, existing)
                )
                for existing in unique
            ):
                unique.append(transform)
        return tuple(unique)

    def _free_source_cells_v3(
        self,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> frozenset[Cell]:
        free: set[Cell] = set()
        for row in outcome.ground_clear_query_tensor:
            if not (
                0 <= row.source_cell[0] < outcome.source_geometry.shape[0]
                and 0 <= row.source_cell[1] < outcome.source_geometry.shape[1]
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ground query cell is outside native source geometry"
                )
            expected = canonical_ground_query_xy_body_v1(
                outcome.source_geometry,
                row.source_cell,
            )
            if any(
                not (
                    math.isclose(
                        actual[0],
                        target[0],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        actual[1],
                        target[1],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
                for actual, target in zip(row.query_xy_body_m, expected)
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ground query geometry differs from canonical supports"
                )
            probabilities = tuple(
                _sigmoid(value) for value in row.clear_to_target_logits
            )
            known = tuple(
                max(probability, 1.0 - probability)
                >= self.__calibration_v3.known_probability_threshold
                for probability in probabilities
            )
            if all(
                in_frustum
                and is_known
                and probability
                >= self.__calibration_v3.free_given_known_probability_threshold
                for in_frustum, is_known, probability in zip(
                    row.query_in_frustum,
                    known,
                    probabilities,
                )
            ):
                free.add(row.source_cell)
        return frozenset(free)

    def _selected_local_hits_v3(
        self,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> tuple[XY, ...]:
        hits: list[XY] = []
        for row in outcome.ordered_ray_hit_depth_tensor:
            if any(
                depth < self.__calibration_v3.ray_depth_min_m - 1e-12
                or depth > self.__calibration_v3.ray_depth_max_m + 1e-12
                for depth in row.ordered_depth_m
            ):
                raise NativeLearnedProjectionRejectedError(
                    "raw ordered ray depth is outside frozen calibration"
                )
            survival = 1.0
            selected_depth: float | None = None
            for logit, depth in zip(row.ordered_hit_logits, row.ordered_depth_m):
                hazard = _sigmoid(logit)
                first_hit = survival * hazard
                if (
                    max(first_hit, 1.0 - first_hit)
                    >= self.__calibration_v3.known_probability_threshold
                    and first_hit
                    >= self.__calibration_v3.occupied_given_known_probability_threshold
                ):
                    selected_depth = depth
                    break
                survival *= 1.0 - hazard
            if selected_depth is not None:
                hits.append(
                    (
                        row.ray_origin_xy_body_m[0]
                        + row.ray_direction_xy_body[0] * selected_depth,
                        row.ray_origin_xy_body_m[1]
                        + row.ray_direction_xy_body[1] * selected_depth,
                    )
                )
        return tuple(hits)

    def _project_v3(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> NativeLearnedPhysicalProjectionReceiptV1:
        transforms = self._uncertainty_transforms_v3(outcome.pose)
        free_source = self._free_source_cells_v3(outcome)
        physical_frame = snapshot.physical_map_frame
        shape = snapshot.physical_shape
        free_by_transform: list[frozenset[Cell]] = []
        domain: set[Cell] = set()
        all_ground_cells = frozenset(
            row.source_cell for row in outcome.ground_clear_query_tensor
        )
        for transform in transforms:
            candidates: set[Cell] = set()
            for source_cell in free_source:
                minimum, maximum = _source_square(
                    outcome.source_geometry,
                    source_cell,
                )
                polygon = tuple(
                    _apply_transform(point, transform)
                    for point in (
                        minimum,
                        (maximum[0], minimum[1]),
                        maximum,
                        (minimum[0], maximum[1]),
                    )
                )
                candidates.update(
                    _candidate_cells_for_polygon(
                        physical_frame,
                        shape,
                        polygon,
                    )
                )
            transform_free: set[Cell] = set()
            for destination_cell in candidates:
                inverse_polygon = tuple(
                    _inverse_transform(point, transform)
                    for point in _destination_square(
                        physical_frame,
                        destination_cell,
                    )
                )
                if _polygon_covered_by_source_union(
                    inverse_polygon,
                    outcome.source_geometry,
                    free_source,
                ):
                    transform_free.add(destination_cell)
            free_by_transform.append(frozenset(transform_free))

            for source_cell in all_ground_cells:
                minimum, maximum = _source_square(
                    outcome.source_geometry,
                    source_cell,
                )
                polygon = tuple(
                    _apply_transform(point, transform)
                    for point in (
                        minimum,
                        (maximum[0], minimum[1]),
                        maximum,
                        (minimum[0], maximum[1]),
                    )
                )
                for destination_cell in _candidate_cells_for_polygon(
                    physical_frame,
                    shape,
                    polygon,
                ):
                    if _polygons_intersect_closed(
                        polygon,
                        _destination_square(physical_frame, destination_cell),
                    ):
                        domain.add(destination_cell)

        free = (
            set.intersection(*(set(cells) for cells in free_by_transform))
            if free_by_transform
            else set()
        )
        occupied: set[Cell] = set()
        for local_hit in self._selected_local_hits_v3(outcome):
            for transform in transforms:
                occupied.update(
                    _closed_point_supercover(
                        physical_frame,
                        shape,
                        _apply_transform(local_hit, transform),
                    )
                )
        domain.update(occupied)
        free.difference_update(occupied)
        unknown = domain - free - occupied
        transform_hash = _sha256(
            {
                "schema": "lewm_g3_native_projection_transform_set_v1",
                "transforms_xy_yaw": [list(value) for value in transforms],
            }
        )
        return NativeLearnedPhysicalProjectionReceiptV1(
            raw_outcome_sha256=outcome.content_sha256,
            raw_tensor_content_sha256=outcome.raw_tensor_content_sha256,
            calibration_sha256=self.__calibration_v3.content_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                snapshot.configuration_map_frame_sha256
            ),
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_shape=snapshot.physical_shape,
            configuration_shape=snapshot.configuration_shape,
            transform_uncertainty_set=transforms,
            transform_uncertainty_set_sha256=transform_hash,
            free_cells=frozenset(free),
            occupied_cells=frozenset(occupied),
            unknown_cells=frozenset(unknown),
        )

    @staticmethod
    def _projection_payload_sha256_v3(
        receipt: NativeLearnedPhysicalProjectionReceiptV1,
    ) -> str:
        return _sha256(
            {
                "schema": "lewm_g3_native_learned_projection_payload_v3",
                "projection_receipt_sha256": receipt.content_sha256,
                "free_cells": _cells_json(receipt.free_cells),
                "occupied_cells": _cells_json(receipt.occupied_cells),
                "unknown_cells": _cells_json(receipt.unknown_cells),
            }
        )

    def _build_physical_transaction_v3(
        self,
        *,
        admission_kind: str,
        observation_id: str,
        observation_payload_sha256: str,
        observation_producer_sha256: str,
        pose: PoseProvenance,
        projection_receipt: NativeLearnedPhysicalProjectionReceiptV1,
        retracts_observation_id: str | None,
    ) -> PhysicalEvidenceTransaction:
        if admission_kind == "projection":
            evidence = tuple(
                PhysicalCellEvidence(cell=cell, label=PhysicalLabel.FREE)
                for cell in sorted(projection_receipt.free_cells)
            ) + tuple(
                PhysicalCellEvidence(cell=cell, label=PhysicalLabel.OCCUPIED)
                for cell in sorted(projection_receipt.occupied_cells)
            )
            unknown = tuple(sorted(projection_receipt.unknown_cells))
            retractions: tuple[str, ...] = ()
        elif admission_kind == "retraction":
            evidence = ()
            unknown = ()
            if retracts_observation_id is None:
                raise NativeLearnedProjectionBindingError(
                    "V3 retraction transaction lost its target"
                )
            retractions = (retracts_observation_id,)
        else:
            raise NativeLearnedProjectionBindingError(
                "V3 development transaction kind changed"
            )
        return PhysicalEvidenceTransaction(
            observation=ObservationIdentity(
                observation_id=observation_id,
                payload_sha256=observation_payload_sha256,
                producer_sha256=observation_producer_sha256,
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
            ),
            map_frame=self.__memory_v3.map_frame,
            pose=pose,
            physical_evidence=evidence,
            observed_unknown_cells=unknown,
            retract_learned_observation_ids=retractions,
            projection_contract_sha256=PROFILE_SHA256,
        )

    def _make_admission_v3(
        self,
        *,
        admission_kind: str,
        source_outcome_sha256: str,
        projection_receipt: NativeLearnedPhysicalProjectionReceiptV1,
        physical_transaction: PhysicalEvidenceTransaction,
        configuration_revision: int,
        identities: dict[str, str],
        retracts_observation_id: str | None,
    ) -> QualifiedLearnedPhysicalDevelopmentAdmissionV3:
        observation = physical_transaction.observation
        core = {
            "schema": "lewm_g3_qualified_learned_admission_id_v3",
            "kind": admission_kind,
            "adapter_contract_sha256": self.__contract_v3,
            "source_outcome_sha256": source_outcome_sha256,
            "projection_receipt_sha256": projection_receipt.content_sha256,
            "physical_transaction_sha256": physical_transaction.content_sha256,
            "observation_id": observation.observation_id,
            "memory_revision_before": self.__memory_v3.revision,
            "retracts_observation_id": retracts_observation_id,
        }
        return QualifiedLearnedPhysicalDevelopmentAdmissionV3(
            admission_kind=admission_kind,
            admission_id_sha256=_sha256(core),
            adapter_contract_sha256=self.__contract_v3,
            source_outcome_sha256=source_outcome_sha256,
            projection_receipt_sha256=projection_receipt.content_sha256,
            physical_transaction_sha256=physical_transaction.content_sha256,
            observation_id=observation.observation_id,
            observation_payload_sha256=observation.payload_sha256,
            observation_producer_sha256=observation.producer_sha256,
            pose_sha256=physical_transaction.pose.content_sha256,
            memory_config_sha256=self.__memory_v3.config.content_sha256,
            physical_map_frame_sha256=(
                self.__memory_v3.map_frame.content_sha256
            ),
            configuration_map_frame_sha256=(
                self.__projection_v3.configuration_map_frame.content_sha256
            ),
            physical_revision_before=self.__memory_v3.revision,
            configuration_revision=configuration_revision,
            runner_execution_identity_sha256=identities[
                "runner_execution_identity_sha256"
            ],
            inference_implementation_sha256=identities[
                "inference_implementation_sha256"
            ],
            projection_implementation_sha256=identities[
                "projection_implementation_sha256"
            ],
            access_ledger_source_sha256=identities[
                "access_ledger_source_sha256"
            ],
            checkpoint_file_sha256=identities["checkpoint_file_sha256"],
            g2_report_file_sha256=identities["g2_report_file_sha256"],
            calibration_sha256=self.__calibration_v3.content_sha256,
            rgb_frame_sha256=identities["rgb_frame_sha256"],
            raw_outcome_file_sha256=identities["raw_outcome_file_sha256"],
            retracts_observation_id=retracts_observation_id,
        )

    def _register_v3(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
        snapshot: TwoResolutionConfigurationSnapshotV2,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV3:
        self.__issued_v3[id(package)] = (package, package.content_sha256)
        self.__snapshots_v3[id(package)] = (
            snapshot,
            snapshot.content_sha256,
        )
        return package

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV3:
        self.__projection_v3.assert_current_snapshot(snapshot)
        self.__runner_v3.assert_issued(outcome)
        self._assert_outcome_bindings_v3(snapshot, outcome)
        projection_receipt = self._project_v3(snapshot, outcome)
        physical_transaction = self._build_physical_transaction_v3(
            admission_kind="projection",
            observation_id=outcome.observation_id,
            observation_payload_sha256=(
                self._projection_payload_sha256_v3(projection_receipt)
            ),
            observation_producer_sha256=(
                outcome.runner_execution_identity_sha256
            ),
            pose=outcome.pose,
            projection_receipt=projection_receipt,
            retracts_observation_id=None,
        )
        identities = {
            "runner_execution_identity_sha256": (
                outcome.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": (
                outcome.inference_implementation_sha256
            ),
            "projection_implementation_sha256": (
                outcome.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": (
                outcome.access_ledger_source_sha256
            ),
            "checkpoint_file_sha256": outcome.checkpoint_file_sha256,
            "g2_report_file_sha256": outcome.g2_report_file_sha256,
            "rgb_frame_sha256": outcome.rgb_frame_sha256,
            "raw_outcome_file_sha256": outcome.raw_outcome_file_sha256,
        }
        admission = self._make_admission_v3(
            admission_kind="projection",
            source_outcome_sha256=outcome.content_sha256,
            projection_receipt=projection_receipt,
            physical_transaction=physical_transaction,
            configuration_revision=snapshot.configuration_revision,
            identities=identities,
            retracts_observation_id=None,
        )
        package = QualifiedLearnedPhysicalDevelopmentTransactionV3(
            admission=admission,
            pose=outcome.pose,
            projection_receipt=projection_receipt,
            retracts_observation_id=None,
            _issuance_capability_v3=self.__capability_v3,
        )
        self.__runner_v3.assert_issued(outcome, consume=True)
        return self._register_v3(package, snapshot)

    def _assert_issued_digest_v3(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> str:
        if type(package) is not QualifiedLearnedPhysicalDevelopmentTransactionV3:
            raise TypeError("package has the wrong V3 transaction type")
        issued = self.__issued_v3.get(id(package))
        if (
            package._issuance_capability_v3 is not self.__capability_v3
            or issued is None
            or issued[0] is not package
        ):
            raise NativeLearnedProjectionBindingError(
                "package is not the exact live object issued by this V3 adapter"
            )
        package.assert_integrity()
        if package.content_sha256 != issued[1]:
            raise NativeLearnedProjectionBindingError(
                "V3 package differs from its issued content"
            )
        return issued[1]

    def _assert_exact_package_v3(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> TwoResolutionConfigurationSnapshotV2:
        self._assert_issued_digest_v3(package)
        if id(package) in self.__consumed_v3:
            raise NativeLearnedProjectionReplayError(
                "V3 development transaction was already consumed"
            )
        snapshot_record = self.__snapshots_v3[id(package)]
        snapshot, issued_snapshot_sha256 = snapshot_record
        if snapshot.content_sha256 != issued_snapshot_sha256:
            raise NativeLearnedProjectionBindingError(
                "V3 package snapshot differs from its issued content"
            )
        self.__projection_v3.assert_current_snapshot(snapshot)
        admission = package.admission
        if (
            admission.adapter_contract_sha256 != self.__contract_v3
            or admission.memory_config_sha256
            != self.__memory_v3.config.content_sha256
            or admission.physical_map_frame_sha256
            != self.__memory_v3.map_frame.content_sha256
            or admission.configuration_map_frame_sha256
            != self.__projection_v3.configuration_map_frame.content_sha256
            or admission.physical_revision_before != self.__memory_v3.revision
            or admission.configuration_revision
            != snapshot.configuration_revision
            or package.development_only is not True
            or package.hardware_execution_authorized is not False
            or package.production_promotion_authorized is not False
        ):
            raise NativeLearnedProjectionBindingError(
                "V3 transaction authority/frame/revision binding changed"
            )
        return snapshot

    def _commit_core_v3(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> TransactionReceipt:
        self._assert_exact_package_v3(package)
        admission = package.admission
        physical_transaction = self._build_physical_transaction_v3(
            admission_kind=admission.admission_kind,
            observation_id=admission.observation_id,
            observation_payload_sha256=admission.observation_payload_sha256,
            observation_producer_sha256=admission.observation_producer_sha256,
            pose=package.pose,
            projection_receipt=package.projection_receipt,
            retracts_observation_id=package.retracts_observation_id,
        )
        if (
            physical_transaction.content_sha256
            != admission.physical_transaction_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "V3 reconstructed physical transaction changed"
            )
        receipt = self.__memory_v3.apply_transaction(physical_transaction)
        self.__consumed_v3.add(id(package))
        if admission.admission_kind == "projection":
            self.__committed_v3[admission.observation_id] = package
        else:
            target = admission.retracts_observation_id
            if target is not None:
                self.__committed_v3.pop(target, None)
        return receipt

    def _assert_exact_active_target_v3(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> tuple[str, int]:
        self._assert_issued_digest_v3(package)
        observation_id = package.admission.observation_id
        if (
            package.admission.admission_kind != "projection"
            or self.__committed_v3.get(observation_id) is not package
            or id(package) not in self.__consumed_v3
            or observation_id
            not in self.__memory_v3.learned_observation_ids
        ):
            raise NativeLearnedProjectionBindingError(
                "retraction target is not an exact active V3 projection"
            )
        return observation_id, id(package)

    def _assert_reservation_digests_v3(
        self,
        record: _RetractionReservationV3,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> None:
        target_digest = self._assert_issued_digest_v3(record.target_package)
        if target_digest != record.target_issued_content_sha256:
            raise NativeLearnedProjectionBindingError(
                "V3 committed projection differs from its issued content"
            )
        if record.retraction_package is not package:
            raise NativeLearnedProjectionBindingError(
                "V3 reservation does not own this exact retraction package"
            )
        retraction_digest = self._assert_issued_digest_v3(package)
        if retraction_digest != record.retraction_issued_content_sha256:
            raise NativeLearnedProjectionBindingError(
                "V3 retraction differs from its issued content"
            )

    def _assert_reservation_target_active_v3(
        self,
        record: _RetractionReservationV3,
    ) -> None:
        observation_id = record.target_key[0]
        if (
            self.__committed_v3.get(observation_id)
            is not record.target_package
            or observation_id
            not in self.__memory_v3.learned_observation_ids
        ):
            raise NativeLearnedProjectionBindingError(
                "V3 retraction target is no longer exact active evidence"
            )

    def _reservation_snapshot_is_current_v3(
        self,
        record: _RetractionReservationV3,
    ) -> bool:
        if record.snapshot.content_sha256 != record.snapshot_issued_content_sha256:
            raise NativeLearnedProjectionBindingError(
                "V3 live reservation snapshot differs from issued content"
            )
        try:
            self.__projection_v3.assert_current_snapshot(record.snapshot)
        except SnapshotBindingError:
            return False
        self._assert_reservation_digests_v3(
            record,
            record.retraction_package,
        )
        self._assert_reservation_target_active_v3(record)
        return True

    def _mark_reservation_terminal_v3(
        self,
        record: _RetractionReservationV3,
        state: _RetractionStateV3,
    ) -> None:
        if state not in (_RetractionStateV3.STALE, _RetractionStateV3.CONSUMED):
            raise ValueError("V3 terminal state must be stale or consumed")
        if record.state is not _RetractionStateV3.LIVE:
            return
        record.state = state
        live_package_id = self.__live_by_target_v3.get(record.target_key)
        if live_package_id == id(record.retraction_package):
            self.__live_by_target_v3.pop(record.target_key, None)

    def issue_retraction(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        committed_projection: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV3:
        self.__projection_v3.assert_current_snapshot(snapshot)
        target_key = self._assert_exact_active_target_v3(committed_projection)
        live_package_id = self.__live_by_target_v3.get(target_key)
        if live_package_id is not None:
            record = self.__reservations_v3[live_package_id]
            if record.state is not _RetractionStateV3.LIVE:
                raise NativeLearnedProjectionBindingError(
                    "V3 live index contains a terminal reservation"
                )
            if self._reservation_snapshot_is_current_v3(record):
                raise NativeLearnedProjectionReplayError(
                    "an exact live V3 retraction already exists for this target"
                )
            self._mark_reservation_terminal_v3(
                record,
                _RetractionStateV3.STALE,
            )

        self.__sequence_v3 += 1
        observation_id = committed_projection.observation_id
        retraction_observation_id = (
            f"qualified-native-v4-v3-retract:{self.__sequence_v3}:"
            f"{observation_id}"
        )
        payload_sha256 = _sha256(
            {
                "schema": "lewm_g3_qualified_learned_retraction_payload_v3",
                "retracts_observation_id": observation_id,
                "source_admission_sha256": (
                    committed_projection.admission.content_sha256
                ),
            }
        )
        physical_transaction = self._build_physical_transaction_v3(
            admission_kind="retraction",
            observation_id=retraction_observation_id,
            observation_payload_sha256=payload_sha256,
            observation_producer_sha256=self.__contract_v3,
            pose=committed_projection.pose,
            projection_receipt=committed_projection.projection_receipt,
            retracts_observation_id=observation_id,
        )
        original = committed_projection.admission
        identities = {
            "runner_execution_identity_sha256": (
                original.runner_execution_identity_sha256
            ),
            "inference_implementation_sha256": (
                original.inference_implementation_sha256
            ),
            "projection_implementation_sha256": (
                original.projection_implementation_sha256
            ),
            "access_ledger_source_sha256": (
                original.access_ledger_source_sha256
            ),
            "checkpoint_file_sha256": original.checkpoint_file_sha256,
            "g2_report_file_sha256": original.g2_report_file_sha256,
            "rgb_frame_sha256": original.rgb_frame_sha256,
            "raw_outcome_file_sha256": original.raw_outcome_file_sha256,
        }
        admission = self._make_admission_v3(
            admission_kind="retraction",
            source_outcome_sha256=original.source_outcome_sha256,
            projection_receipt=committed_projection.projection_receipt,
            physical_transaction=physical_transaction,
            configuration_revision=snapshot.configuration_revision,
            identities=identities,
            retracts_observation_id=observation_id,
        )
        package = QualifiedLearnedPhysicalDevelopmentTransactionV3(
            admission=admission,
            pose=committed_projection.pose,
            projection_receipt=committed_projection.projection_receipt,
            retracts_observation_id=observation_id,
            _issuance_capability_v3=self.__capability_v3,
        )
        self._register_v3(package, snapshot)
        record = _RetractionReservationV3(
            target_key=target_key,
            target_package=committed_projection,
            target_issued_content_sha256=(
                self.__issued_v3[id(committed_projection)][1]
            ),
            retraction_package=package,
            retraction_issued_content_sha256=(
                self.__issued_v3[id(package)][1]
            ),
            snapshot=snapshot,
            snapshot_issued_content_sha256=snapshot.content_sha256,
        )
        self.__reservations_v3[id(package)] = record
        self.__live_by_target_v3[target_key] = id(package)
        return package

    def commit(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV3,
    ) -> TransactionReceipt:
        record = self.__reservations_v3.get(id(package))
        if record is None:
            if (
                type(package)
                is QualifiedLearnedPhysicalDevelopmentTransactionV3
                and package.admission.admission_kind == "retraction"
            ):
                raise NativeLearnedProjectionBindingError(
                    "V3 retraction has no exact reservation"
                )
            return self._commit_core_v3(package)
        self._assert_reservation_digests_v3(record, package)
        if record.state is _RetractionStateV3.STALE:
            raise NativeLearnedProjectionReplayError(
                "V3 retraction package is terminally stale"
            )
        if record.state is _RetractionStateV3.CONSUMED:
            raise NativeLearnedProjectionReplayError(
                "V3 retraction package was already consumed"
            )
        if record.state is not _RetractionStateV3.LIVE:
            raise NativeLearnedProjectionBindingError(
                "V3 retraction reservation state is invalid"
            )
        self._assert_reservation_target_active_v3(record)
        try:
            receipt = self._commit_core_v3(package)
        except SnapshotBindingError:
            self._mark_reservation_terminal_v3(
                record,
                _RetractionStateV3.STALE,
            )
            raise
        self._mark_reservation_terminal_v3(
            record,
            _RetractionStateV3.CONSUMED,
        )
        return receipt


def require_production_native_learned_projection_adapter_v3() -> object:
    if (
        PRODUCTION_NATIVE_V4_RUNNER is None
        or PRODUCTION_V4_CHECKPOINT_FILE_SHA256 is None
        or PRODUCTION_G2_REPORT_FILE_SHA256 is None
        or PRODUCTION_V4_CALIBRATION_SHA256 is None
        or PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V3 is None
    ):
        raise PermissionError(
            "production native learned-projection V3 identities are unset"
        )
    return PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V3


__all__ = [
    "FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1",
    "FrozenNativeLearnedProjectionCalibrationV1",
    "NativeLearnedPhysicalProjectionAdapterV3",
    "NativeLearnedPhysicalProjectionReceiptV1",
    "NativeLearnedProjectionBindingError",
    "NativeLearnedProjectionRejectedError",
    "NativeLearnedProjectionReplayError",
    "NativeV4SourceGeometryV1",
    "QualifiedLearnedPhysicalDevelopmentAdmissionV3",
    "QualifiedLearnedPhysicalDevelopmentTransactionV3",
    "RawGroundClearCellQueriesV1",
    "RawOrderedRayHitDepthV1",
    "SyntheticNativeV4RawOutcomeV1",
    "SyntheticNativeV4RunnerV1",
    "canonical_ground_query_xy_body_v1",
    "require_production_native_learned_projection_adapter_v3",
]
