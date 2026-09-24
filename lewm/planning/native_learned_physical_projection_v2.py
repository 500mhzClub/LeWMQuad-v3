"""Additive V2 lifecycle closure for native learned physical projection.

V2 reuses the frozen V1 raw-outcome, conservative geometry, admission, and
transaction types.  It adds an exact-identity retraction reservation state
machine.  Production, hardware execution, and promotion remain unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from lewm.planning.native_learned_physical_projection_v1 import (
    FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1,
    FrozenNativeLearnedProjectionCalibrationV1,
    NativeLearnedPhysicalProjectionAdapterV1,
    NativeLearnedPhysicalProjectionReceiptV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionRejectedError,
    NativeLearnedProjectionReplayError,
    NativeV4SourceGeometryV1,
    QualifiedLearnedPhysicalDevelopmentAdmissionV1,
    QualifiedLearnedPhysicalDevelopmentTransactionV1,
    RawGroundClearCellQueriesV1,
    RawOrderedRayHitDepthV1,
    SyntheticNativeV4RawOutcomeV1,
    SyntheticNativeV4RunnerV1,
    canonical_ground_query_xy_body_v1,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    TransactionReceipt,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)


PRODUCTION_NATIVE_V4_RUNNER = None
PRODUCTION_V4_CHECKPOINT_FILE_SHA256 = None
PRODUCTION_G2_REPORT_FILE_SHA256 = None
PRODUCTION_V4_CALIBRATION_SHA256 = None
PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V2 = None


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class _RetractionStateV2(str, Enum):
    LIVE = "live"
    STALE = "stale"
    CONSUMED = "consumed"


@dataclass
class _RetractionReservationV2:
    target_key: tuple[str, int]
    target_package: QualifiedLearnedPhysicalDevelopmentTransactionV1
    target_issued_content_sha256: str
    retraction_package: QualifiedLearnedPhysicalDevelopmentTransactionV1
    retraction_issued_content_sha256: str
    snapshot: TwoResolutionConfigurationSnapshotV2
    snapshot_issued_content_sha256: str
    state: _RetractionStateV2 = _RetractionStateV2.LIVE


class NativeLearnedPhysicalProjectionAdapterV2:
    """V1 projection semantics with retryable exact retraction reservations."""

    __slots__ = (
        "__inner",
        "_v2_base_adapter_contract_sha256",
        "_v2_retractions_by_package",
        "_v2_live_retraction_by_target",
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
        self.__inner = NativeLearnedPhysicalProjectionAdapterV1(
            memory=memory,
            projection=projection,
            runner=runner,
            calibration=calibration,
            runner_execution_identity_sha256=(
                runner_execution_identity_sha256
            ),
            inference_implementation_sha256=(
                inference_implementation_sha256
            ),
            projection_implementation_sha256=(
                projection_implementation_sha256
            ),
            access_ledger_source_sha256=access_ledger_source_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            g2_report_file_sha256=g2_report_file_sha256,
            camera_transform_sha256=camera_transform_sha256,
            _synthetic_test_fixture=_synthetic_test_fixture,
        )
        self._v2_base_adapter_contract_sha256 = (
            self.__inner.adapter_contract_sha256
        )
        v2_contract_sha256 = _canonical_sha256(
            {
                "schema": (
                    "lewm_g3_native_learned_projection_adapter_contract_v2"
                ),
                "frozen_v1_adapter_contract_sha256": (
                    self._v2_base_adapter_contract_sha256
                ),
                "retraction_reservation": (
                    "exact_package_identity_live_stale_consumed_v2"
                ),
                "committed_target_original_digest_required": True,
                "development_only": True,
                "hardware_execution_authorized": False,
                "production_promotion_authorized": False,
            }
        )
        self.__inner._adapter_contract_sha256 = v2_contract_sha256
        self._v2_retractions_by_package: dict[
            int,
            _RetractionReservationV2,
        ] = {}
        self._v2_live_retraction_by_target: dict[
            tuple[str, int],
            int,
        ] = {}

    def __copy__(self) -> "NativeLearnedPhysicalProjectionAdapterV2":
        raise TypeError(
            "native learned physical projection V2 adapters are non-copyable"
        )

    def __deepcopy__(
        self,
        memo: object,
    ) -> "NativeLearnedPhysicalProjectionAdapterV2":
        del memo
        raise TypeError(
            "native learned physical projection V2 adapters are non-copyable"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError(
            "native learned physical projection V2 adapters are non-serializable"
        )

    @property
    def adapter_contract_sha256(self) -> str:
        return self.__inner.adapter_contract_sha256

    @property
    def development_only(self) -> bool:
        return True

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        outcome: SyntheticNativeV4RawOutcomeV1,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV1:
        return self.__inner.issue(snapshot, outcome)

    def _assert_exact_committed_projection_v2(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> tuple[str, int]:
        if type(package) is not QualifiedLearnedPhysicalDevelopmentTransactionV1:
            raise TypeError("committed_projection has the wrong type")
        issued = self.__inner._issued.get(id(package))
        if issued is None or issued[0] is not package:
            raise NativeLearnedProjectionBindingError(
                "retraction requires the exact package issued by this V2 adapter"
            )
        package.assert_integrity()
        if package.content_sha256 != issued[1]:
            raise NativeLearnedProjectionBindingError(
                "committed projection differs from its issued content"
            )
        observation_id = package.admission.observation_id
        if (
            package.admission.admission_kind != "projection"
            or self.__inner._committed_by_observation.get(observation_id)
            is not package
            or id(package) not in self.__inner._consumed
            or observation_id
            not in self.__inner._memory.learned_observation_ids
        ):
            raise NativeLearnedProjectionBindingError(
                "retraction target is not an exact active committed projection"
            )
        return observation_id, id(package)

    def _assert_reservation_package_v2(
        self,
        record: _RetractionReservationV2,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> None:
        target_issued = self.__inner._issued.get(id(record.target_package))
        if (
            target_issued is None
            or target_issued[0] is not record.target_package
        ):
            raise NativeLearnedProjectionBindingError(
                "retraction target is no longer the exact V2-issued package"
            )
        record.target_package.assert_integrity()
        if (
            record.target_package.content_sha256 != target_issued[1]
            or record.target_package.content_sha256
            != record.target_issued_content_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "committed projection differs from its issued content"
            )
        if record.retraction_package is not package:
            raise NativeLearnedProjectionBindingError(
                "retraction reservation does not own this exact package"
            )
        issued = self.__inner._issued.get(id(package))
        if issued is None or issued[0] is not package:
            raise NativeLearnedProjectionBindingError(
                "retraction package is not the exact V2-issued object"
            )
        package.assert_integrity()
        if (
            package.content_sha256 != issued[1]
            or package.content_sha256
            != record.retraction_issued_content_sha256
        ):
            raise NativeLearnedProjectionBindingError(
                "retraction package differs from its issued content"
            )

    def _reservation_snapshot_is_current_v2(
        self,
        record: _RetractionReservationV2,
    ) -> bool:
        if record.snapshot.content_sha256 != record.snapshot_issued_content_sha256:
            raise NativeLearnedProjectionBindingError(
                "live retraction snapshot differs from its issued content"
            )
        try:
            self.__inner._projection.assert_current_snapshot(record.snapshot)
        except SnapshotBindingError:
            return False
        self._assert_reservation_package_v2(
            record,
            record.retraction_package,
        )
        return True

    def _mark_retraction_terminal_v2(
        self,
        record: _RetractionReservationV2,
        state: _RetractionStateV2,
    ) -> None:
        if state not in (
            _RetractionStateV2.STALE,
            _RetractionStateV2.CONSUMED,
        ):
            raise ValueError("retraction terminal state must be stale or consumed")
        if record.state is not _RetractionStateV2.LIVE:
            return
        record.state = state
        live_package_id = self._v2_live_retraction_by_target.get(
            record.target_key
        )
        if live_package_id == id(record.retraction_package):
            self._v2_live_retraction_by_target.pop(record.target_key, None)
        self.__inner._retraction_issued.discard(record.target_key[0])

    def issue_retraction(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        committed_projection: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> QualifiedLearnedPhysicalDevelopmentTransactionV1:
        self.__inner._projection.assert_current_snapshot(snapshot)
        target_key = self._assert_exact_committed_projection_v2(
            committed_projection
        )
        live_package_id = self._v2_live_retraction_by_target.get(target_key)
        if live_package_id is not None:
            record = self._v2_retractions_by_package[live_package_id]
            if record.state is not _RetractionStateV2.LIVE:
                raise NativeLearnedProjectionBindingError(
                    "V2 live-retraction index contains a terminal reservation"
                )
            if self._reservation_snapshot_is_current_v2(record):
                raise NativeLearnedProjectionReplayError(
                    "an exact live retraction already exists for this target"
                )
            self._mark_retraction_terminal_v2(
                record,
                _RetractionStateV2.STALE,
            )

        self.__inner._retraction_issued.discard(target_key[0])
        package = self.__inner.issue_retraction(snapshot, committed_projection)
        issued = self.__inner._issued[id(package)]
        record = _RetractionReservationV2(
            target_key=target_key,
            target_package=committed_projection,
            target_issued_content_sha256=(
                self.__inner._issued[id(committed_projection)][1]
            ),
            retraction_package=package,
            retraction_issued_content_sha256=issued[1],
            snapshot=snapshot,
            snapshot_issued_content_sha256=snapshot.content_sha256,
        )
        self._v2_retractions_by_package[id(package)] = record
        self._v2_live_retraction_by_target[target_key] = id(package)
        return package

    def commit(
        self,
        package: QualifiedLearnedPhysicalDevelopmentTransactionV1,
    ) -> TransactionReceipt:
        record = self._v2_retractions_by_package.get(id(package))
        if record is None:
            if (
                type(package)
                is QualifiedLearnedPhysicalDevelopmentTransactionV1
                and package.admission.admission_kind == "retraction"
            ):
                raise NativeLearnedProjectionBindingError(
                    "retraction package has no exact V2 reservation"
                )
            return self.__inner.commit(package)
        self._assert_reservation_package_v2(record, package)
        if record.state is _RetractionStateV2.STALE:
            raise NativeLearnedProjectionReplayError(
                "retraction package is terminally stale"
            )
        if record.state is _RetractionStateV2.CONSUMED:
            raise NativeLearnedProjectionReplayError(
                "retraction package was already consumed"
            )
        if record.state is not _RetractionStateV2.LIVE:
            raise NativeLearnedProjectionBindingError(
                "retraction reservation has an invalid lifecycle state"
            )
        try:
            receipt = self.__inner.commit(package)
        except SnapshotBindingError:
            self._mark_retraction_terminal_v2(
                record,
                _RetractionStateV2.STALE,
            )
            raise
        self._mark_retraction_terminal_v2(
            record,
            _RetractionStateV2.CONSUMED,
        )
        return receipt


def require_production_native_learned_projection_adapter_v2() -> object:
    if (
        PRODUCTION_NATIVE_V4_RUNNER is None
        or PRODUCTION_V4_CHECKPOINT_FILE_SHA256 is None
        or PRODUCTION_G2_REPORT_FILE_SHA256 is None
        or PRODUCTION_V4_CALIBRATION_SHA256 is None
        or PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V2 is None
    ):
        raise PermissionError(
            "production native learned-projection V2 identities are unset"
        )
    return PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V2


__all__ = [
    "FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1",
    "FrozenNativeLearnedProjectionCalibrationV1",
    "NativeLearnedPhysicalProjectionAdapterV2",
    "NativeLearnedPhysicalProjectionReceiptV1",
    "NativeLearnedProjectionBindingError",
    "NativeLearnedProjectionRejectedError",
    "NativeLearnedProjectionReplayError",
    "NativeV4SourceGeometryV1",
    "QualifiedLearnedPhysicalDevelopmentAdmissionV1",
    "QualifiedLearnedPhysicalDevelopmentTransactionV1",
    "RawGroundClearCellQueriesV1",
    "RawOrderedRayHitDepthV1",
    "SyntheticNativeV4RawOutcomeV1",
    "SyntheticNativeV4RunnerV1",
    "canonical_ground_query_xy_body_v1",
    "require_production_native_learned_projection_adapter_v2",
]
