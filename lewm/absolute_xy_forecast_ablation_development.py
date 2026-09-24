"""Prospective XY-input ablation with shared neural yaw/contact and route control."""
import hashlib
import numpy as np
from scripts.fit_pose_action_xy_forecast_ablation_development import OUTPUT
from scripts.fit_closed_loop_motion_residual_development import features,pose_features
from lewm.clearance_preferred_route_development import ClearancePreferredTurnRecoveryRuntime
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands

FIT_SHA256={
    'all_forecast_features':'1bcdb395bf9a16b216368ecef61052349065464f56253e8b21652b53e15c45d6',
    'pose_history_and_commands':'b509184a7bb52a18701ba248e8dcd770c87daf72b346d4116061ac6a3b04074b'}


def predict_xy(fit,prediction,past,commands):
    x=features(prediction,past,commands)[:,fit['columns']]
    return np.einsum('hf,hfo->ho',(x-fit['mean'])/fit['scale'],fit['coefficient'])+fit['bias']


class AbsoluteXYForecastRuntime(ClearancePreferredTurnRecoveryRuntime):
    xy_variant='all_forecast_features'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=OUTPUT/(self.xy_variant+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FIT_SHA256[self.xy_variant]:
            raise ValueError('frozen absolute XY fit changed')
        with np.load(path,allow_pickle=False) as d:self.absolute_xy_fit={k:d[k].copy() for k in d.files}
        expected=np.arange(42) if self.xy_variant=='all_forecast_features' else np.r_[np.arange(12),np.arange(16,42)]
        if not np.array_equal(self.absolute_xy_fit['columns'],expected):
            raise ValueError('declared XY forecast feature population required')

    def _correct_prediction(self,prediction,packet,evidence,prefix):
        with self.correction_pose_lock:
            poses={f:self.correction_poses[f] for f in range(packet.frame-3,packet.frame+1)}
        if poses[packet.frame]['measured_ns']!=packet.measured_ns:
            raise ValueError('same-observation causal pose history required')
        past,_,_=pose_features(poses,packet.frame)
        corrected=prediction.copy()
        for i,action in enumerate(ACTIONS):
            commands=np.asarray(list(prefix)+[candidate_commands(action)[0]]*4+[[0.,0.,0.]])
            corrected[i,:,:2]=predict_xy(self.absolute_xy_fit,prediction[i],past,commands)
        if not np.isfinite(corrected).all():raise ValueError('finite absolute XY forecast required')
        return corrected,dict(fit_sha256=FIT_SHA256[self.xy_variant],xy_forecast_variant=self.xy_variant,
            pose_history_frames=list(poses),original_motion_residual_applied=False,
            raw_forecast_xy_m=prediction[:,:,:2].tolist(),corrected_forecast_xy_m=corrected[:,:,:2].tolist(),
            future_pose_input=False,neural_weights_changed=False,yaw_and_contact_unchanged=True,
            neural_forecast_inputs_to_xy=self.xy_variant=='all_forecast_features')


class PoseActionXYForecastRuntime(AbsoluteXYForecastRuntime):
    xy_variant='pose_history_and_commands'
