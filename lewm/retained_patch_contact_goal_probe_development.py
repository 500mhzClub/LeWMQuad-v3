"""Footprint-specific retained-patch witnesses extend measured-grid coverage."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.measured_floor_contact_development import MeasuredFloorContactMemory
from lewm.measured_floor_contact_goal_probe_development import MeasuredFloorContactMap,MeasuredFloorContactGoalProbe


class RetainedPatchContactMemory(MeasuredFloorContactMemory):
    def __init__(self,*,identity):
        super().__init__(identity=identity);self.patches=RetainedFloorPatches()

    def classify_current(self,policy,depth,B,floor_height,floor_cells,*,now_ns):
        super().classify_current(policy,depth,B,floor_height,floor_cells,now_ns=now_ns)
        witness={k:self.classification_receipt[k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        self.patches.append(depth['depth_m'],depth['valid'],B@self.rotation,B@self.position,floor_height,witness)

    def footprint(self,geometry,displacement_body_xy,yaw_rad,*,now_ns,persistent=True):
        if len(self.patches.frames)!=len(self.route):raise SensorContractError('every causal floor patch must be retained')
        original=super().footprint(geometry,displacement_body_xy,yaw_rad,now_ns=now_ns,persistent=persistent)
        rules=deepcopy(original['foot_floor_contacts']);hits={r['shape_id']:r for r in original['shapes']}
        pending=[r for r in rules if not r['measured_floor_contact_rule_eligible'] and
            hits[r['shape_id']]['intersecting_voxels'] and not r['other_or_unknown']['intersecting_voxels']]
        coverage={}
        if pending:
            c,s=np.cos(yaw_rad),np.sin(yaw_rad);R=self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            p=self.position+self.rotation@np.r_[displacement_body_xy,0.]
            shapes={s['shape_id']:s for s in geometry.supports(self.joints,R)['shapes']}
            centres=[(self.map_from_initial@(p+R@np.asarray(shapes[r['shape_id']]['center_body_m'])))[:2].tolist() for r in pending]
            coverage=dict(zip((r['shape_id'] for r in pending),self.patches.coverage(centres),strict=True))
        revised=deepcopy(hits)
        for rule in rules:
            key=rule['shape_id'];rule['original_grid_contact_rule_eligible']=rule['measured_floor_contact_rule_eligible']
            rule['retained_patch']=coverage.get(key)
            if key in coverage and coverage[key]['complete_nominal_foot_patch']:
                revised[key]=dict(shape_id=key,**deepcopy(rule['other_or_unknown']))
                rule['measured_floor_contact_rule_eligible']=True
        shapes=[revised[r['shape_id']] for r in original['shapes']]
        return original|dict(shapes=shapes,possible_intersection=any(r['intersecting_voxels'] for r in shapes),
            foot_floor_contacts=rules,grid_contact_shapes=original['shapes'],grid_contact_possible_intersection=original['possible_intersection'],
            retained_patch_frames=len(self.patches.frames),retained_prefix_bytes=sum(f['prefix'].nbytes for f in self.patches.frames),
            ground_contact_scope='four_foot_spheres_with_complete_measured_grid_or_single_retained_depth_patch_coverage',
            original_floor_grid_changed=False)


class RetainedPatchContactMap(MeasuredFloorContactMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.surface=RetainedPatchContactMemory(identity=identity)


class RetainedPatchContactGoalProbe(MeasuredFloorContactGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.mapper=RetainedPatchContactMap(identity=(0,0,0));self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='retained_patch_contact_goal_probe_v1')
