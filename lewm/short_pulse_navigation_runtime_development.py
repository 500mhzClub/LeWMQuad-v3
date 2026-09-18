"""Use frozen pulse-trained forecasts directly, without predecessor corrections."""
import hashlib
import json
import numpy as np
from lewm.auxiliary_only_turn_recovery_development import AuxiliaryTurnDispatch
from lewm.floor_reacquisition_development import FloorReacquisitionRuntimeMixin
from lewm.neural_input_treatment_development import NeuralInputTreatmentMixin
from lewm.signed_veto_view_recovery_development import SignedVetoViewMixin
from lewm.instantaneous_waypoint_score_development import InstantaneousWaypointScoreMixin
from lewm.terminal_translation_pulse_development import command_sequences
from lewm.commanded_planar_motion_development import forecast as commanded_forecast
from scripts.run_go2_auxiliary_turn_recovery_development import RecoveryViewMixin
from scripts.run_go2_planned_stopping_projection_development import StoppingAwareRuntime
from scripts.navigation_artifact_root_development import BASE
from scripts.fit_go2_local_motion_controls_development import nominal

COMMAND_FIT=BASE/'go2_short_pulse_command_control_v1_attempt_001'


def past_commands(packets,now_ns):
    if len(packets)!=4:raise ValueError('four past public observations required')
    rows=[]
    for i,p in enumerate(packets):
        now=p['sensor_state']['decision_ns']
        if now!=now_ns+(i-3)*100_000_000:raise ValueError('same causal camera history required')
        c=p['sensor_state']['control']['applied_command']
        measured=np.asarray(c['measured_ns']);available=np.asarray(c['available_ns'])
        if np.any(measured>now) or np.any(available>now):raise ValueError('future command observations forbidden')
        age=np.where(measured>=0,(now-measured)/1e9,1.5)[:,None]
        rows.append(np.concatenate((np.asarray(c['values'])/[.3,1.,.5],np.asarray(c['valid']).astype(float),age),axis=1).ravel())
    result=np.concatenate(rows)
    if result.shape!=(420,) or not np.isfinite(result).all():raise ValueError('finite command history required')
    return result


def command_predictions(model,history,commands):
    base=np.asarray([nominal(c) for c in commands]);result=base.copy()
    for h in range(8):
        known=np.zeros((6,8,3));known[:,:h+1]=commands[:,:h+1]
        x=np.concatenate((known.reshape(6,24),base[:,h],np.broadcast_to(history,(6,420))),axis=1)
        result[:,h]+=((x-model['mean'][h])/model['scale'][h])@model['coefficient'][h]+model['bias'][h]
    if not np.isfinite(result).all():raise ValueError('finite command forecasts required')
    return result


class PulsePredictiveRuntime(NeuralInputTreatmentMixin,SignedVetoViewMixin,RecoveryViewMixin,
        FloorReacquisitionRuntimeMixin,StoppingAwareRuntime,AuxiliaryTurnDispatch):
    def __init__(self,*args,prediction_source,**kwargs):
        if prediction_source not in ('neural','pose_command','command_history'):
            raise ValueError('fixed prospective prediction source required')
        self.pulse_prediction_source=prediction_source
        path=COMMAND_FIT/'command_only.npz'
        self.command_fit_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if json.loads((COMMAND_FIT/'result.json').read_text())['status']!='COMPLETE':
            raise ValueError('completed command-history fit required')
        with np.load(path,allow_pickle=False) as arrays:
            self.command_model={k:arrays[k].copy() for k in ('mean','scale','bias','coefficient')}
        # Parent construction retains the shared tracker, map, recovery and
        # guards. Its predecessor correction objects are never used below.
        super().__init__(*args,motion_prediction_source='learned',**kwargs)

    def _correct_prediction(self,prediction,packet,evidence,prefix):
        if prediction.shape!=(6,8,5) or not np.isfinite(prediction).all():
            raise ValueError('complete finite frozen neural predictions required')
        with self.correction_pose_lock:
            poses={f:self.correction_poses[f] for f in range(packet.frame-3,packet.frame+1)}
        if poses[packet.frame]['measured_ns']!=packet.measured_ns:raise ValueError('current causal pose history required')
        pulse=bool(self.planning_translation_pulse)
        pose_xy=self.pose_command_xy.predict(poses,packet.frame,prefix,pulse=pulse)
        commands=command_sequences(prefix,pulse=pulse)
        command=command_predictions(self.command_model,past_commands(packet.history,packet.measured_ns),commands)
        nominal_prediction=commanded_forecast(prefix,pulse=pulse)
        selected=prediction.copy()
        if self.pulse_prediction_source=='pose_command':
            selected[:,:,:2]=pose_xy;selected[:,:,2:4]=nominal_prediction[:,:,2:4]
        elif self.pulse_prediction_source=='command_history':
            selected[:,:,:2]=command[:,:,:2]
            selected[:,:,2]=np.sin(command[:,:,2]);selected[:,:,3]=np.cos(command[:,:,2])
        selected[:,:,4]=-1000.
        # Preserve familiar evaluator fields but explicitly record that the
        # neural XY is raw composed output, with no external correction fit.
        receipt=dict(prediction_source=self.pulse_prediction_source,
            forecast_xy_source=self.pulse_prediction_source,
            forecast_yaw_source={'neural':'learned','pose_command':'command','command_history':'command_history'}[self.pulse_prediction_source],
            contact_score_mode='disabled',external_neural_correction_applied=False,
            neural_outcomes_used_for_scoring=self.pulse_prediction_source=='neural',
            raw_forecast_xy_m=prediction[:,:,:2].tolist(),corrected_forecast_xy_m=selected[:,:,:2].tolist(),
            learned_corrected_forecast_xy_m=prediction[:,:,:2].tolist(),
            learned_corrected_field_means='raw nominal-composed neural forecast; no external correction',
            pose_command_forecast_xy_m=pose_xy.tolist(),command_history_forecast_xy_yaw=command.tolist(),
            command_history_fit_sha256=self.command_fit_sha256,
            upstream_prediction_for_yaw_ablation=prediction.tolist(),
            applied_prediction_after_yaw_ablation=selected.tolist(),
            neural_yaw_retained=self.pulse_prediction_source=='neural',
            all_prediction_alternatives_computed=True,future_pose_input=False,
            zero_contact_score_is_not_a_contact_free_prediction=True)
        return selected,receipt


class PulseInstantaneousRuntime(InstantaneousWaypointScoreMixin,PulsePredictiveRuntime):
    pass
