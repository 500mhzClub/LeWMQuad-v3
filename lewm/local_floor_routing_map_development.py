"""Estimate floor coverage locally while retaining original obstacle observations."""
from lewm.eligible_floor_registration_development import bind
from lewm.multirate_routing_map_development import Geometry
from lewm.partial_height_round_trip_development import PartialHeightMap
from lewm.current_pair_routing_memory_development import CapturedCurrentPairMap
from lewm.local_inverse_depth_floor_development import local_depth


class LocalCoverageGeometry(Geometry):
    def floor_coverage(self, depth, valid, *args, **kwargs):
        estimated, supported = local_depth(depth, valid)
        return super().floor_coverage(estimated, supported, *args, **kwargs)


class LocalCoveragePartialMap(PartialHeightMap):
    # Initial floor-height fitting, packet identities and obstacle points retain
    # their original inputs. Only floor_coverage uses the local depth estimate.
    update = bind(PartialHeightMap.update, Geometry=LocalCoverageGeometry)


class LocalFloorRoutingMap(CapturedCurrentPairMap, LocalCoveragePartialMap):
    """Retain current-pair capture and accumulated 1-cm obstacle observations."""


def initialize_mapping():
    from lewm import process_mapped_runtime_development as process
    from lewm.two_cm_floor_extent_development import configure
    configure()
    process.initialize_mapping()
    process._mapper = LocalFloorRoutingMap()
