"""Matched XY-source intervention; learned yaw/contact and downstream code remain.

Both arms compute both XY alternatives before selecting the assigned one. This
is an ablation of neural XY prediction, not a fully model-free controller.
"""
import hashlib
import numpy as np
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.fit_closed_loop_motion_residual_development import features, pose_features
from scripts.fit_pose_command_motion_control_development import OUTPUT, COLUMNS

FIT_SHA256='bc0c215b1a4ca171d63de7eb7be30f057dc4e6a97dc4ec0ea9f7ee4fb3bf00d6'


class FrozenPoseCommandXY:
    def __init__(self):
        path=OUTPUT/'motion_fit.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FIT_SHA256:
            raise ValueError('frozen pose-command control fit differs')
        with np.load(path,allow_pickle=False) as arrays:
            self.fit={k:arrays[k].copy() for k in ('mean','scale','bias','coefficient')}

    def predict(self, poses, frame, prefix, *, pulse):
        if set(poses)!=set(range(frame-3,frame+1)):
            raise ValueError('exact four causal visual poses required')
        past,_,_=pose_features(poses,frame)
        predictions=[]
        for commands in command_sequences(prefix,pulse=pulse):
            x=features(np.zeros((8,5)),past,commands)[:,COLUMNS]
            predictions.append(np.stack([((x[h]-self.fit['mean'][h])/self.fit['scale'][h])@
                self.fit['coefficient'][h]+self.fit['bias'][h] for h in range(8)]))
        result=np.asarray(predictions)
        if result.shape!=(6,8,2) or not np.isfinite(result).all():
            raise ValueError('complete finite causal pose-command XY prediction required')
        return result


def choose_xy(learned, pose_command_xy, source):
    learned=np.asarray(learned);xy=np.asarray(pose_command_xy)
    if (source not in ('learned','pose_command') or learned.shape!=(6,8,5) or
            xy.shape!=(6,8,2) or not np.isfinite(learned).all() or not np.isfinite(xy).all()):
        raise ValueError('explicit XY source and complete finite alternatives required')
    result=learned.copy()
    if source=='pose_command':result[:,:,:2]=xy
    return result


class PoseCommandXYSourceMixin:
    def __init__(self,*args,forecast_xy_source,**kwargs):
        if forecast_xy_source not in ('learned','pose_command'):
            raise ValueError('fixed learned or pose-command XY source required')
        self.forecast_xy_source=forecast_xy_source
        self.pose_command_xy=FrozenPoseCommandXY()
        super().__init__(*args,**kwargs)

    def _correct_prediction(self,prediction,packet,evidence,prefix):
        learned,original=super()._correct_prediction(prediction,packet,evidence,prefix)
        with self.correction_pose_lock:
            poses={f:self.correction_poses[f] for f in range(packet.frame-3,packet.frame+1)}
        if poses[packet.frame]['measured_ns']!=packet.measured_ns:
            raise ValueError('same current visual history as learned correction required')
        xy=self.pose_command_xy.predict(poses,packet.frame,prefix,
            pulse=bool(self.planning_translation_pulse))
        result=choose_xy(learned,xy,self.forecast_xy_source)
        control=self.forecast_xy_source=='pose_command'
        receipt=original | dict(
            fit_sha256=FIT_SHA256 if control else original['fit_sha256'],
            correction_root=OUTPUT.name if control else original['correction_root'],
            correction_base_model='pose_command_only' if control else original['correction_base_model'],
            corrected_forecast_xy_m=result[:,:,:2].tolist(),
            maximum_absolute_correction_m=float(np.max(np.abs(result[:,:,:2]-prediction[:,:,:2]))),
            forecast_xy_source=self.forecast_xy_source,
            pose_command_fit_sha256=FIT_SHA256,
            pose_command_forecast_xy_m=xy.tolist(),
            learned_corrected_forecast_xy_m=learned[:,:,:2].tolist(),
            learned_motion_correction=original,
            both_xy_alternatives_computed_in_both_arms=True,
            neural_xy_used_for_scoring=not control,
            learned_yaw_and_contact_retained=True,
            downstream_planner_and_recovery_unchanged=True,
            fully_model_free_controller=False)
        return result,receipt
