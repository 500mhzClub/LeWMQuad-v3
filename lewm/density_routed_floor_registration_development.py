"""Original measured registration with privately bound density-routed indexing."""
from lewm import floor_pose_registration_development as plane
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.eligible_floor_registration_development import bind
from lewm.density_routed_floor_cell_index_development import observed_floor_cell_index

measured_candidates = bind(plane.measured_candidates, observed_floor_cell_index=observed_floor_cell_index)


class DensityRoutedFloorRegistration(MeasuredFloorTransportRegistration):
    observe = bind(MeasuredFloorTransportRegistration.observe, measured_candidates=measured_candidates)
