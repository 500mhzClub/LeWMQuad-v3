"""Fixed development excitation, not a navigation or safety controller."""
from dataclasses import replace
import hashlib
import json
import math
import numpy as np

from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.longer_motion_collection_development import specification as old_spec, pack as old_pack

TRIALS=('nominal_a','lower_friction_a','nominal_b','lower_friction_b')
COMMANDS=(('forward_small',(.12,0.,0.)),('forward_bank',(.20,0.,0.)),
          ('left_small',(0.,0.,.03)),('right_small',(0.,0.,-.03)),
          ('left_intermediate',(0.,0.,.25)),('right_intermediate',(0.,0.,-.25)),
          ('left_bank',(0.,0.,.45)),('right_bank',(0.,0.,-.45)))
DURATIONS=(2,5)
BRAKE_TICKS=20
INITIAL_ZERO_TICKS=10
MAXIMUM_TICKS=386


def events(order):
    if order not in ('a','b'): raise ValueError('fixed order a or b required')
    items=[dict(command_name=name,pulse_ticks=n,requested_command=list(command))
           for name,command in COMMANDS for n in DURATIONS]
    if order=='b': items=list(reversed(items))
    return [dict(event_index=i,**item) for i,item in enumerate(items)]


def schedule(order):
    rows=[dict(stage='initial_zero',phase=1,event_index=None,command_name='hold',
               pulse_ticks=0,event_offset=i,requested_command=[0.,0.,0.]) for i in range(INITIAL_ZERO_TICKS)]
    for event in events(order):
        for i in range(event['pulse_ticks']+BRAKE_TICKS):
            drive=i<event['pulse_ticks']
            rows.append(event|dict(stage='pulse' if drive else 'brake',phase=2 if drive else 3,event_offset=i,
                                   requested_command=event['requested_command'] if drive else [0.,0.,0.]))
    assert len(rows)==MAXIMUM_TICKS
    return rows


def validate_command(command):
    c=np.asarray(command,float)
    if c.shape!=(3,) or not np.isfinite(c).all() or not 0<=c[0]<=.20 or c[1]!=0 or abs(c[2])>.45:
        raise ValueError('declared pulse command domain required')
    return c.tolist()


def specification(trial):
    if trial not in TRIALS: raise ValueError('fixed four-trial development design required')
    order=trial[-1]; condition=trial[:-2]; previous=old_spec('fit')
    spawn=[-.70,-.25,.02] if order=='a' else [-.45,.25,-.04]
    return previous|dict(trial=trial,condition=condition,order=order,
        scene_id='command-pulse-response-v1-'+trial,family='CONTROLLED_FLOOR_COMMAND_PULSE_RESPONSE',
        procedural_seed=2026090657+(order=='b'),appearance_seed=2026090659+(order=='b'),
        friction_mu=1. if condition=='nominal' else .15,
        geometry=previous['geometry']|dict(spawn_se2_world=spawn),
        controlled_continuous_level_floor=True,hidden_robot_ideal_camera=True)


def pack(spec):
    if spec!=specification(spec['trial']): raise ValueError('exact pulse experiment specification required')
    old=old_pack(old_spec('fit')); x,y,yaw=spec['geometry']['spawn_se2_world']
    return replace(old,scene_id=spec['scene_id'],family=spec['family'],physics_seed=spec['procedural_seed'],
        topology_seed=spec['procedural_seed'],visual_seed=spec['appearance_seed'],
        robot=replace(old.robot,spawn_xyz_m=(x,y,.375),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
        physics_randomization=replace(old.physics_randomization,floor_friction_mu=spec['friction_mu']),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


class PulseSequence:
    def __init__(self,order,*,identity=(0,0,0)):
        self.rows=schedule(order); self.identity=_identity(identity); self.tick=0; self.last_ns=None
        self.terminal=None; self.reason=None

    def step(self,evidence,*,now_ns):
        now=_ns(now_ns,'pulse decision')
        if self.terminal is None:
            try:
                if self.last_ns is not None and now-self.last_ns!=100_000_000:
                    raise SensorContractError('exact 10Hz required')
                if (evidence['schema']!='visual_led_motion_evidence_development.v1'
                        or tuple(evidence['identity'])!=self.identity or evidence['decision_ns']!=now
                        or evidence['status']!='CURRENT_VISUAL_POSE' or evidence['terminal_failure'] is not None):
                    raise SensorContractError('same-episode current visual evidence required')
                p=evidence['current_pose']; xyz=np.asarray(p['position_initial_body_m'],float)
                proper(p['rotation_initial_body_from_current_body'])
                if (p['mode']!='gyro' or p['measured_ns']!=now or p['available_ns']>now
                        or xyz.shape!=(3,) or not np.isfinite(xyz).all()):
                    raise SensorContractError('current finite gyro visual pose required')
                if np.linalg.norm(xyz)>1.: raise SensorContractError('one-metre observed excursion limit')
                self.last_ns=now
                if self.tick==len(self.rows):
                    self.terminal='PULSE_SCHEDULE_COMPLETE'; self.reason='fixed acquisition schedule exhausted'
            except (ValueError,TypeError,KeyError,IndexError) as error:
                self.terminal='PULSE_SCHEDULE_FAILED'; self.reason=str(error)
        if self.terminal is None:
            row=self.rows[self.tick]; self.tick+=1
        else:
            row=dict(stage='terminal',phase=9,event_index=None,command_name='hold',pulse_ticks=0,
                     event_offset=None,requested_command=[0.,0.,0.])
        return row|dict(decision_ns=now,terminal=self.terminal,reason=self.reason,
                        fixed_development_excitation=True,native_pose_used=False,navigation_qualified=False)
