from __future__ import annotations

import hashlib

from lewm.planning.revisioned_physical_configuration_memory import (
    FusionMode,
    MapFrameIdentity,
    PhysicalMemoryConfig,
    RevisionedPhysicalMemory,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    FIXED_PROFILE_V2,
    FREE_SUPPORT_COUNT,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_COUNT,
    OCCUPIED_SUPPORT_SHA256,
    PROFILE_SHA256,
    TwoResolutionConfigurationProjectionV2,
    assert_fixed_profile_integrity,
    physical_index_for_configuration_offset,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_empty_revisioned_memory_roundtrips_with_identity_intact() -> None:
    frame = MapFrameIdentity(
        session_id="closure-memory",
        origin_xy_m=(-1.0, -2.0),
    )
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=frame,
            fusion_mode=FusionMode.PERSISTENT,
            expected_camera_transform_sha256=_hash("camera-transform"),
        )
    )

    restored = RevisionedPhysicalMemory.deserialize(memory.serialize())

    assert restored.to_dict() == memory.to_dict()
    assert restored.revision == 0
    assert restored.physical_content_sha256 == memory.physical_content_sha256
    assert restored.known_physical_cells == frozenset()
    assert restored.seen_observation_ids == frozenset()


def test_two_resolution_profile_and_empty_projection_are_deterministic() -> None:
    assert_fixed_profile_integrity()
    assert len(FIXED_PROFILE_V2.free_support_offsets) == FREE_SUPPORT_COUNT
    assert len(FIXED_PROFILE_V2.occupied_support_offsets) == OCCUPIED_SUPPORT_COUNT
    assert FIXED_PROFILE_V2.free_support_sha256 == FREE_SUPPORT_SHA256
    assert FIXED_PROFILE_V2.occupied_support_sha256 == OCCUPIED_SUPPORT_SHA256
    assert FIXED_PROFILE_V2.content_sha256 == PROFILE_SHA256
    assert physical_index_for_configuration_offset((2, 3), (1, -2)) == (5, 4)

    physical_frame = MapFrameIdentity(
        session_id="closure-projection",
        origin_xy_m=(0.0, 0.0),
        cell_size_m=0.05,
        frame_id="physical",
    )
    configuration_frame = MapFrameIdentity(
        session_id="closure-projection",
        origin_xy_m=(0.0, 0.0),
        cell_size_m=0.10,
        frame_id="configuration",
    )
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=physical_frame,
            require_registered_lattice=False,
            physical_projection_contract_sha256=PROFILE_SHA256,
        )
    )
    projection = TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=configuration_frame,
        physical_shape=(20, 24),
        configuration_shape=(10, 12),
    )

    first = projection.project()
    projection.assert_current_snapshot(first)
    assert first.configuration_revision == 1
    assert (
        first.free_cells | first.occupied_cells | first.unknown_cells
        == frozenset((x, y) for x in range(10) for y in range(12))
    )
    assert not (first.free_cells & first.occupied_cells)
    assert not (first.free_cells & first.unknown_cells)
    assert not (first.occupied_cells & first.unknown_cells)

    second = projection.project()
    projection.assert_current_snapshot(second)
    assert second.configuration_revision == 2
    assert second.projection_source_sha256 == first.projection_source_sha256
