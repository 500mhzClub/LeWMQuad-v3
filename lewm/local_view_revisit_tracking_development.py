"""Retain eight local viewing directions for measured reference revisits."""
import math
import numpy as np

from lewm.coherent_reference_refresh_development import CoherentReferenceRefreshPose, CoherentReferenceRefreshMotion
from lewm.joint_rgbd_rigid_pose_development import angle


def view_bin(gyro):
    return int(math.floor((math.atan2(gyro[1,0],gyro[0,0])+math.pi/8)/(math.pi/4)))%8


def select_revisit(bank, references, position, gyro, now):
    active={r.frame for r in references}
    candidates=[entry for entry in bank.values() if entry[0].frame not in active
        and now-entry[0].measured_ns>=1_000_000_000
        and np.linalg.norm(entry[0].position-position)<=.20
        and angle(entry[0].gyro.T@gyro)<=.20]
    return min(candidates,key=lambda e:e[0].frame) if candidates else None


class LocalViewRevisitPose(CoherentReferenceRefreshPose):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.local_view_bank={};self.last_revisit_attempt=None

    def _remember(self,current,R,G,p,now):
        super()._remember(current,R,G,p,now)
        reference=next(r for r in self.references if r.frame==self.frame)
        key=view_bin(G);old=self.local_view_bank.get(key)
        if old is None or np.linalg.norm(old[0].position-p)>.30:
            # Keep the original accepted pose and its exact acquired plane.
            self.local_view_bank[key]=(reference,self._planes[self.frame])

    def _measure(self,current,G,now):
        entry=select_revisit(self.local_view_bank,self.references,self.last_p,G,now)
        if entry is None: return super()._measure(current,G,now)
        reference,plane=entry
        old_references=self.references;old_stable=self.stable_reference;old_planes=self._planes
        self.references=old_references[-7:]+[reference]
        self.stable_reference=reference
        self._planes=old_planes|{reference.frame:plane}
        self.last_revisit_attempt=dict(reference_frame=reference.frame,
            reference_age_ns=now-reference.measured_ns,
            estimated_prior_distance_m=float(np.linalg.norm(reference.position-self.last_p)),
            gyro_rotation_difference_rad=angle(reference.gyro.T@G),selected=False)
        try:
            result=super()._measure(current,G,now)
            self.last_revisit_attempt['selected']=result[0]['reference'].frame==reference.frame
            return result
        finally:
            self.references=old_references;self.stable_reference=old_stable;self._planes=old_planes

    def observe(self,*args,**kwargs):
        self.last_revisit_attempt=None
        result=super().observe(*args,**kwargs)
        return result|dict(local_view_reference_bank_frames=sorted(r.frame for r,_ in self.local_view_bank.values()),
            local_view_revisit_attempt=self.last_revisit_attempt,
            original_pair_and_continuity_acceptance_unchanged=True)


class LocalViewRevisitMotion(CoherentReferenceRefreshMotion):
    def __init__(self,*,identity=(0,0,0),activation_frame=0):
        super().__init__(identity=identity,activation_frame=activation_frame)
        self.model=LocalViewRevisitPose(activation_frame=activation_frame)
