"""Current scalar floor height when the visible patch cannot measure a full normal."""
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm import measured_floor_transport_development as original
from lewm import extended_return_budget_transport_development as extended
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneFloorRegistration
from lewm.joint_measured_floor_plane_development import fit_joint_plane

SCHEMA='partial_height_visual_floor_transport_development.v1'
MISSING_EXTENT='insufficient_combined_two_axis_extent'


def height_correction(anchor,raw,plane,*,identity,now_ns):
    correction=extended.composition(anchor,raw,plane,identity=identity,now_ns=now_ns)
    if plane['reason']!=MISSING_EXTENT or plane['candidate_count']<100:
        raise ValueError('height update requires at least 100 current points with missing two-axis extent')
    mean=sum(s['count']*np.asarray(s['mean_body_m']) for s in plane['camera_statistics'] if s['count'])/plane['candidate_count']
    normal=np.asarray(correction['transported_reference_normal_body'])
    update=-float(normal@mean+correction['transported_reference_offset_body_m'])
    reference=anchor['floor_registration']['reference']['joint_plane']
    initial_normal=np.asarray(reference['normal_body'])
    p=np.asarray(correction['position_initial_body_m'])+update*initial_normal
    magnitude=float(np.linalg.norm(p-np.asarray(correction['raw_position_initial_body_m'])))
    if magnitude>.05 or abs(update)>.05:raise ValueError('partial height exceeds original 5 cm correction bound')
    return correction|dict(position_initial_body_m=p.tolist(),correction_magnitude_m=magnitude,
        transported_reference_offset_body_m=float(reference['offset_body_m']+initial_normal@p),
        height_update_m=update,current_scalar_height_measurement=True,current_normal_measured=False,
        normal_source='uninterrupted_visual_transport_from_full_floor_anchor')


_read_partial=bind(original.current_measured_floor_pose,SCHEMA=SCHEMA,composition=height_correction)


def read_pose(evidence,*,identity,now_ns):
    if evidence['schema']==SCHEMA:return _read_partial(evidence,identity=identity,now_ns=now_ns)
    return extended.current_measured_floor_pose(evidence,identity=identity,now_ns=now_ns)


_partial_transport=bind(original.transport_evidence,SCHEMA=SCHEMA,composition=height_correction,
    current_measured_floor_pose=read_pose)


def transport(anchor,raw,plane,clouds,**kwargs):
    if plane['reason']==MISSING_EXTENT and plane['candidate_count']>=100:
        return _partial_transport(anchor,raw,plane,clouds,**kwargs)
    return extended.transport_evidence(anchor,raw,plane,clouds,**kwargs)


class PartialHeightRegistration(SampledPlaneFloorRegistration):
    observe=bind(SampledPlaneFloorRegistration.observe,transport_evidence=transport)


def fit_gyro_height(primary,auxiliary,up):
    plane=fit_joint_plane(primary,auxiliary,up)
    if plane['available'] or plane['reason']!=MISSING_EXTENT:return plane
    points=np.concatenate((primary,auxiliary))
    if len(points)<100:return plane
    normal=np.asarray(up);offset=-float(points.mean(0)@normal)
    residuals=points@normal+offset;maximum=float(np.abs(residuals).max())
    if maximum>.003:return plane|dict(partial_height_maximum_residual_m=maximum)
    return plane|dict(available=True,reason='CURRENT_HEIGHT_CONDITIONED_ON_GYRO_NORMAL',
        normal_body=normal.tolist(),offset_body_m=offset,normal_measured_from_current_points=False,
        partial_height_maximum_residual_m=maximum,full_plane_qualification=False,
        observation_type='scalar_height_with_gyro_orientation_prior')
