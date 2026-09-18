"""Prefer a measured in-place view before routing to a coverage viewpoint."""
import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits
from lewm.camera_frontier_viewpoint_development import directed_rotation, floor_cell_projection
from lewm.observed_floor_waypoint_development import centre, segment_cells


class CurrentPositionCoverageVisits(CameraFrontierVisits):
    def __init__(self):
        super().__init__()
        self.failed_current_positions = {}

    def _choose(self, snapshot, route, p, R, unknown):
        p = np.asarray(p, float)
        attempted = self.failed_current_positions.get(unknown, [])
        if unknown not in snapshot.floor | snapshot.occupied and not any(
                np.linalg.norm(p[:2]-old)<=.10 for old in attempted):
            directed, heading = directed_rotation(R,p[:2],unknown)
            cameras = [r for r in floor_cell_projection(unknown,p,directed,snapshot.floor_height)
                if r['fully_projected'] and not (segment_cells(
                    r['camera_origin_map_xy_m'],centre(unknown)) & snapshot.occupied)]
            if cameras:
                return dict(viewpoint_cell=np.floor(p[:2]/.05).astype(int).tolist(),
                    viewpoint_map_xy_m=p[:2].tolist(),route_cells=[],unknown_cell=list(unknown),
                    frontier_target_map_xy_m=centre(unknown).tolist(),view_heading_rad=heading,
                    projected_cameras=cameras,current_roll_pitch_height_used=True,
                    current_measured_position_view=True,translation_to_viewpoint_required=False,
                    current_floor_not_declared_observed=True,future_visibility_requires_new_observation=True,
                    unknown_floor_admitted=False,motion_authorized=False)
        return super()._choose(snapshot,route,p,R,unknown)

    def _finish(self,snapshot,now,reason,observed):
        if reason=='FRESH_VIEW_PATCH_STILL_UNKNOWN' and self.visit['camera_viewpoint'].get(
                'current_measured_position_view'):
            target=tuple(self.visit['unknown_neighbour'])
            self.failed_current_positions.setdefault(target,[]).append(np.asarray(self.context[0][:2]).copy())
        super()._finish(snapshot,now,reason,observed)


class CurrentPositionCoverageMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.coverage_views=CurrentPositionCoverageVisits()
