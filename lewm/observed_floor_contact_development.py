"""Separate measured foot/ground contact from obstacle intersection evidence.

This explicitly changes the predecessor contact policy. Complete foot coverage
remains recorded; neither support nor unobserved free space is certified.
"""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45SurfaceMemory,AuxiliaryDownward45FloorMap
from lewm.auxiliary_downward45_goal_probe_development import AuxiliaryDownward45GoalProbe


def separate_ground_contact(original):
    primary={r['shape_id']:r for r in original['foot_floor_contacts']}
    auxiliary={r['shape_id']:r for r in original['auxiliary_foot_floor_contacts']}
    if not set(primary)==set(auxiliary)<=set(FOOT_IDS):
        raise SensorContractError('paired nominal foot classification required')
    shapes=[];auxiliary_shapes=[];contacts=[]
    for p,a in zip(original['shapes'],original['auxiliary_shapes'],strict=True):
        key=p['shape_id']
        if a['shape_id']!=key:raise SensorContractError('paired ordered shapes required')
        if key in primary:
            p=dict(shape_id=key,**deepcopy(primary[key]['other_or_unknown']))
            a=dict(shape_id=key,**deepcopy(auxiliary[key]['auxiliary_other_or_unknown']))
            contacts.append(dict(shape_id=key,
                primary_measured_floor=deepcopy(primary[key]['floor']),
                auxiliary_measured_floor=deepcopy(auxiliary[key]['auxiliary_floor']),
                complete_projection_observed=auxiliary[key]['measured_floor_contact_rule_eligible'],
                floor_contact_allowed=True,nonfloor_or_unknown_contact_allowed=False,
                support_status='UNKNOWN',ground_support_approved=False))
        shapes.append(p);auxiliary_shapes.append(a)
    if len(contacts)!=len(primary):raise SensorContractError('all classified feet require shape evidence')
    return original|dict(shapes=shapes,auxiliary_shapes=auxiliary_shapes,
        coverage_required_primary_shapes=original['shapes'],
        coverage_required_auxiliary_shapes=original['auxiliary_shapes'],
        coverage_required_possible_intersection=original['possible_intersection'],
        primary_possible_intersection=any(r['intersecting_voxels'] for r in shapes),
        auxiliary_possible_intersection=any(r['intersecting_voxels'] for r in auxiliary_shapes),
        possible_intersection=any(r['intersecting_voxels'] for r in shapes+auxiliary_shapes),
        observed_ground_contacts=contacts,
        ground_contact_scope='four_nominal_foot_spheres_classified_floor_only_support_unknown',
        collision_contact_policy='observed_floor_contact_v1',
        complete_projection_required_for_floor_exemption=False,
        non_foot_contacts_exempted=False,non_floor_or_unknown_contacts_exempted=False,
        ground_support_approved=False,unobserved_space_certified=False)


class ObservedFloorContactMemory(AuxiliaryDownward45SurfaceMemory):
    def footprint(self,geometry,displacement_body_xy,yaw_rad,*,now_ns,persistent=True):
        original=super().footprint(geometry,displacement_body_xy,yaw_rad,now_ns=now_ns,persistent=persistent)
        return separate_ground_contact(original)


class ObservedFloorContactMap(AuxiliaryDownward45FloorMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.surface=ObservedFloorContactMemory(identity=identity)


class ObservedFloorContactGoalProbe(AuxiliaryDownward45GoalProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mapper=ObservedFloorContactMap(identity=(0,0,0));self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(
            controller='observed_floor_contact_goal_probe_v1',
            collision_contact_policy='observed_floor_contact_v1',ground_support_approved=False)
