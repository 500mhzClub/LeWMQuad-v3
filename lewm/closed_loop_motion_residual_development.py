"""Frozen development residual fit, driven only by available registered pose history."""
import hashlib
from threading import Lock
import numpy as np
from scripts.fit_closed_loop_motion_residual_development import features,pose_features,OUTPUT
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.clearance_lookahead_development import StandOffFrontierRuntime

FIT_SHA256='ef7511b29afd0117291f600d7afad6adccdcd90199d0ea1f45519d7b81b01638'


class FrozenMotionResidual:
    def __init__(self):
        path=OUTPUT/'residual_fit.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FIT_SHA256:
            raise ValueError('frozen motion residual fit changed')
        with np.load(path,allow_pickle=False) as arrays:
            self.fit={k:arrays[k].copy() for k in ('mean','scale','bias','coefficient')}

    def correct(self,prediction,poses,frame,prefix):
        past,_,_=pose_features(poses,frame)
        corrected=prediction.copy()
        for i,action in enumerate(ACTIONS):
            commands=np.asarray(list(prefix)+[candidate_commands(action)[0]]*4+[[0.,0.,0.]])
            x=features(prediction[i],past,commands)
            for h in range(8):
                correction=((x[h]-self.fit['mean'][h])/self.fit['scale'][h])@self.fit['coefficient'][h]+self.fit['bias'][h]
                corrected[i,h,:2]+=correction
        if not np.isfinite(corrected).all():raise ValueError('finite motion correction required')
        return corrected


class MotionResidualRuntime(StandOffFrontierRuntime):
    def __init__(self,*args,**kwargs):
        self.motion_residual=FrozenMotionResidual()
        self.correction_poses={};self.correction_pose_lock=Lock()
        original_sink=kwargs.get('evidence_sink')
        def sink(frame,raw,evidence):
            pose=evidence['current_pose']
            with self.correction_pose_lock:
                self.correction_poses[frame]=pose
                while len(self.correction_poses)>256:
                    del self.correction_poses[min(self.correction_poses)]
            if original_sink is not None:original_sink(frame,raw,evidence)
        kwargs['evidence_sink']=sink
        super().__init__(*args,**kwargs)

    def _correct_prediction(self,prediction,packet,evidence,prefix):
        with self.correction_pose_lock:
            poses={f:self.correction_poses[f] for f in range(packet.frame-3,packet.frame+1)}
        if poses[packet.frame]['measured_ns']!=packet.measured_ns:
            raise ValueError('same-observation causal pose history required')
        corrected=self.motion_residual.correct(prediction,poses,packet.frame,prefix)
        return corrected,dict(fit_sha256=FIT_SHA256,pose_history_frames=list(poses),
            raw_forecast_xy_m=prediction[:,:,:2].tolist(),corrected_forecast_xy_m=corrected[:,:,:2].tolist(),
            maximum_absolute_correction_m=float(np.max(np.abs(corrected[:,:,:2]-prediction[:,:,:2]))),
            future_pose_input=False,neural_weights_changed=False,yaw_and_contact_unchanged=True)


class PanoramicMotionResidualRuntime(MotionResidualRuntime):
    frontier_panorama=True
