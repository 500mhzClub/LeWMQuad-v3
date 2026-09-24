"""Prospective larger retained histories with unchanged observed geometry.

These components are not installed in a native controller. Full controller,
recording, resource and physical-prefix validation remains necessary.
"""
from copy import deepcopy
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.extended_return_budget_mission_development import MAX_OBSERVATIONS
from lewm.extended_return_budget_transport_development import current_measured_floor_pose
from lewm.measured_floor_transport_controller_development import (
    MeasuredFloorTransportMemory, MeasuredFloorTransportResidual)
from lewm.later_floor_evidence_development import LaterFloorEvidence
from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry, proper
from lewm.body_projected_tiled_controller_development import BodyProjectedRecordingFloorGeometry


class ExtendedReturnBudgetLaterFloorEvidence(LaterFloorEvidence):
    record_pair = bind(LaterFloorEvidence.record_pair, MAX_FRAMES=MAX_OBSERVATIONS)


class ExtendedReturnBudgetMemory(MeasuredFloorTransportMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.later_floor_evidence = ExtendedReturnBudgetLaterFloorEvidence()

    observe = bind(MeasuredFloorTransportMemory.observe, MAX_FRAMES=MAX_OBSERVATIONS,
        current_measured_floor_pose=current_measured_floor_pose)


class ExtendedReturnBudgetResidual(MeasuredFloorTransportResidual):
    observe = bind(MeasuredFloorTransportResidual.observe,
        current_measured_floor_pose=current_measured_floor_pose)


class ExtendedReturnBudgetFloorGeometry(BodyProjectedFloorGeometry):
    def append_patch(self,history,depth,valid,rotation_map_from_body,position_map,floor_height,witness):
        R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
        if (p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(floor_height)
                or type(witness['frame']) is not int or witness['frame']!=len(history.frames) or len(history.frames)>=MAX_OBSERVATIONS
                or (history.frames and (floor_height!=history.frames[0]['floor_height'] or
                    witness['measured_ns']-history.frames[-1]['witness']['measured_ns']!=100_000_000))):
            raise ValueError('bounded uninterrupted patch history and fixed measured floor required')
        index=self.index(depth,valid,R[2])
        heights=self.body_projection(depth)@R[2]+p[2]
        near=np.abs(heights-floor_height)<=.01
        good=index['ground_cells']&near[:-1,:-1]&near[:-1,1:]&near[1:,:-1]&near[1:,1:]
        # At most 479*639 invalid pixels: exact int32 sums cannot overflow.
        prefix=np.zeros((480,640),np.int32);prefix[1:,1:]=(~good).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
        prefix.flags.writeable=False
        history.frames.append(dict(R=R.copy(),p=p.copy(),floor_height=float(floor_height),prefix=prefix,witness=deepcopy(witness)))

class ExtendedReturnBudgetRecordingFloorGeometry(
        BodyProjectedRecordingFloorGeometry, ExtendedReturnBudgetFloorGeometry):
    """Keep tiled indexing and paired recording with the larger patch ceiling."""
