"""Measured heading stabilization through local operational holds.

Unlike a zero-command release assay this operator actively holds heading.
The full mission still uses actual zero release at its terminal and on faults.
"""
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import wrap
from lewm.measured_line_integral_navigation_development import BoundedIntegralAlignment, MeasuredLineIntegralNavigation
from lewm.observable_approach_development import ObservableApproachRegions, ObservableApproachTraversal


class HeadingServo:
    def __init__(self,direction):
        self.direction=np.asarray(direction,dtype=float).copy()
        if self.direction.shape!=(3,) or not np.isfinite(self.direction).all() or np.linalg.norm(self.direction[:2])<.2:
            raise SensorContractError('finite observed heading target required')
        self.last_ns=None; self.integral=0.

    def observe(self,packet,attitude,*,now_ns):
        if self.last_ns is not None and now_ns-self.last_ns!=100_000_000:
            raise SensorContractError('consecutive hold observations required')
        if attitude['decision_ns']!=now_ns or packet['sensor_state']['decision_ns']!=now_ns:
            raise SensorContractError('current hold observation required')
        r=np.asarray(attitude['rotation_initial_body_from_current_body'],dtype=float)
        g=packet['sensor_state']['sensed']['gyro']
        if (r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-7,rtol=0)
                or abs(np.linalg.det(r)-1)>1e-7 or not g['valid'][-1].all() or not np.isfinite(g['values'][-1]).all()):
            raise SensorContractError('proper attitude and valid gyro required')
        f=r[:,0]
        if np.linalg.norm(f[:2])<.2: raise SensorContractError('near vertical heading')
        error=wrap(math.atan2(self.direction[1],self.direction[0])-math.atan2(f[1],f[0]))
        derivative=np.cross(r@g['values'][-1],f)
        rate=float((f[0]*derivative[1]-f[1]*derivative[0])/(f[0]**2+f[1]**2))
        raw=1.5*error+self.integral
        if abs(raw)<.35 or raw*error<0:
            self.integral=float(np.clip(self.integral+.4*error*.1,-.12,.12))
        self.last_ns=now_ns
        return {'heading_error_rad':error,'projected_heading_rate_rad_s':rate,
                'requested_command':[0.,0.,float(np.clip(1.5*error+self.integral,-.35,.35))],
                'integral_command_rad_s':self.integral,'decision_ns':now_ns,
                'holding_contract':'active_measured_heading_not_zero_command_release'}


class HoldingAlignment(BoundedIntegralAlignment):
    def __init__(self,direction_initial_body):
        super().__init__(direction_initial_body); self.servo=HeadingServo(direction_initial_body)

    def observe(self,packet,attitude,*,now_ns):
        if self.terminal: raise SensorContractError('terminal alignment cannot restart')
        row=self.servo.observe(packet,attitude,now_ns=now_ns)
        if self.start_ns is None: self.start_ns=now_ns
        # Tighter handoff readiness avoids declaring alignment while an
        # integral-driven transient is still crossing the outer acceptance
        # region. The original outer tolerance/rate are not relaxed.
        quiet=abs(row['heading_error_rad'])<=.01 and abs(row['projected_heading_rate_rad_s'])<=.02
        if quiet:
            if self.quiet_since is None: self.quiet_since=now_ns
        else: self.quiet_since=None
        complete=self.quiet_since is not None and now_ns-self.quiet_since>=300_000_000
        status='COMPLETE' if complete else 'FAILED_TIMEOUT' if now_ns-self.start_ns>=12_000_000_000 else 'ALIGNING'
        self.last_ns=now_ns; self.terminal=status!='ALIGNING'
        if status=='FAILED_TIMEOUT': row['requested_command']=[0.,0.,0.]
        # At COMPLETE the same servo is handed off, with its integral and
        # observed target intact. Completion does not claim zero-command rest.
        return {**row,'status':status,'translation_compensated':False,'clearance_qualified':False,
                'controller':'continuous_holding_alignment_development_v1'}


class ObservableHoldNavigation(MeasuredLineIntegralNavigation):
    def __init__(self,method,geometry,template=None,*,memory_arm):
        if method!='observable_hold' or template is not None: raise ValueError('distinct observable-hold baseline required')
        super().__init__('measured_line_integral',geometry,memory_arm=memory_arm)
        self.regions=ObservableApproachRegions(geometry); self.hold_servo=None

    def observe_rgbd(self,packet,fast_packet,depth,relative,*,now_ns):
        try:
            row=super().observe_rgbd(packet,fast_packet,depth,relative,now_ns=now_ns)
            if self.stage=='TRAVERSE' and not isinstance(self.child,ObservableApproachTraversal):
                if self.child is None or self.child.tick!=-1: raise SensorContractError('only replace unstarted approach')
                self.child=ObservableApproachTraversal(self.geometry,self.regions); self.children[-1]=self.child
            if self.stage=='ALIGN' and not isinstance(self.alignment,HoldingAlignment):
                if self.alignment is None or self.alignment.start_ns is not None: raise SensorContractError('only replace unstarted alignment')
                self.alignment=HoldingAlignment(self.selected['direction_initial_body'])
            hold=None
            if not row['terminal']:
                handoff=row['turn'] is not None and row['turn']['status']=='COMPLETE'
                quiet_stage=(row['stage'] in ('HOLD_ALIGN','HOLD_TRAVERSE','HOLD_SCAN')
                    or row['next_stage'] in ('HOLD_ALIGN','HOLD_TRAVERSE','HOLD_SCAN')
                    or (row['stage']=='TRAVERSE' and row['child'] is not None and row['child']['status']=='WARMUP'))
                if handoff:
                    self.hold_servo=self.alignment.servo; hold=dict(row['turn'])
                elif quiet_stage and self.completed_legs>0 and row['requested_command']==[0.,0.,0.]:
                    if self.hold_servo is None:
                        self.hold_servo=HeadingServo(self.regions.memory.rotation[:,0])
                    hold=self.hold_servo.observe(packet,row['global_orientation'],now_ns=now_ns)
                    row['requested_command']=hold['requested_command']
                else: self.hold_servo=None
                if hold is not None and abs(row['requested_command'][2])>0:
                    clearance=self.regions.clearance(packet,now_ns=now_ns)
                    row['holding_turn_volume']=clearance
                    if not clearance['all_samples_supported']:
                        self._fail('FAILED_UNOBSERVED_HOLD_VOLUME',now_ns)
                        row.update(status=self.status,terminal=True,requested_command=[0.,0.,0.])
            else: self.hold_servo=None
            row['heading_hold']=hold
            row['local_controller']='observable_hold_navigation_development_v1'
            return row
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self._fail('FAILED_SENSOR',self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('observable-hold navigation failure; apply zero') from error
