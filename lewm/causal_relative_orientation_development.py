"""Relative attitude from current body gyros only; no translation or world heading."""
import copy
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import validate_policy_packet


class CausalRelativeOrientation:
    """Midpoint SO(3) integration on the ideal 50-Hz sensor/10-Hz packet contract.

    A fault latches failure; an old attitude cannot be reused as a fresh estimate.
    There is no turn-specific timeout, bias estimator or hardware accuracy claim.
    """
    def __init__(self):
        self.status='NEW'; self._rotation=np.eye(3); self._gyro=None
        self.identity=None; self.start_ns=None; self.last_ns=None; self.samples_integrated=0

    def _packet(self,packet,now_ns):
        validate_policy_packet(packet); now_ns=_ns(now_ns,'relative orientation clock')
        state=packet['sensor_state']; source=state['sensed']['gyro']
        gyro={**source,**{k:np.asarray(source[k]) for k in ('values','valid','measured_ns','available_ns')}}
        if now_ns%100_000_000 or state['decision_ns']!=now_ns or packet['image']['measured_ns']!=now_ns:
            raise SensorContractError('current command-clock packet required')
        if (gyro['measured_ns'][-1]!=now_ns or not gyro['valid'].all()
                or not np.all(np.diff(gyro['measured_ns'])==20_000_000)):
            raise SensorContractError('complete regular current gyro history required')
        return _identity(state['identity']),now_ns,gyro

    def begin(self,packet,*,now_ns):
        if self.status!='NEW': raise SensorContractError('fresh orientation tracker required')
        try:
            identity,ns,gyro=self._packet(packet,now_ns)
            self.identity=identity; self.start_ns=self.last_ns=ns; self._gyro=copy.deepcopy(gyro)
            self.status='ACTIVE'; return self.snapshot(now_ns=ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('orientation initialization failed') from error

    def step(self,packet,*,now_ns):
        if self.status!='ACTIVE': raise SensorContractError('active orientation tracker required')
        try:
            identity,ns,gyro=self._packet(packet,now_ns)
            if identity!=self.identity or ns-self.last_ns!=100_000_000:
                raise SensorContractError('orientation episode or packet-clock discontinuity')
            old={int(t):i for i,t in enumerate(self._gyro['measured_ns'])}
            for i,t in enumerate(gyro['measured_ns']):
                if int(t) not in old: continue
                for field in ('values','valid','available_ns'):
                    if not np.array_equal(np.asarray(gyro[field])[i],np.asarray(self._gyro[field])[old[int(t)]]):
                        raise SensorContractError('gyro history rewritten')
            if self.last_ns not in gyro['measured_ns']: raise SensorContractError('gyro boundary missing')
            rotation=self._rotation.copy(); previous=np.asarray(self._gyro['values'][-1]); last=self.last_ns; count=0
            for i in np.flatnonzero(gyro['measured_ns']>self.last_ns):
                if int(gyro['measured_ns'][i])-last!=20_000_000: raise SensorContractError('new gyro sample gap')
                current=np.asarray(gyro['values'][i]); rotation=rotation@rotation_increment((previous+current)*.01)
                previous=current; last=int(gyro['measured_ns'][i]); count+=1
            if last!=ns or count!=5: raise SensorContractError('incomplete new gyro interval')
            if not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-8) or abs(np.linalg.det(rotation)-1)>1e-8:
                raise SensorContractError('invalid integrated rotation')
            self._rotation=rotation; self._gyro=copy.deepcopy(gyro); self.last_ns=ns; self.samples_integrated+=count
            return self.snapshot(now_ns=ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('orientation update failed') from error

    def snapshot(self,*,now_ns):
        now_ns=_ns(now_ns,'relative orientation query clock')
        if self.status!='ACTIVE' or now_ns!=self.last_ns: raise SensorContractError('fresh active orientation required')
        forward=self._rotation[:,0]
        heading=math.atan2(forward[1],forward[0]) if math.hypot(forward[0],forward[1])>=.2 else None
        return {'status':self.status,'decision_ns':now_ns,'start_ns':self.start_ns,
            'rotation_initial_body_from_current_body':self._rotation.tolist(),'relative_heading_rad':heading,
            'samples_integrated':self.samples_integrated,'scope':'ideal relative attitude only; no translation, absolute heading or bias correction'}

    def transport_xy(self,vector_initial_body,*,now_ns):
        self.snapshot(now_ns=now_ns)
        vector=np.asarray(vector_initial_body,dtype=float)
        if vector.shape!=(2,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<=1e-12:
            raise SensorContractError('finite nonzero initial-body direction required')
        current=self._rotation.T@np.array([vector[0],vector[1],0.])
        if np.linalg.norm(current[:2])<.2*np.linalg.norm(vector):
            raise SensorContractError('direction projects near vertical in current body frame')
        return current[:2].copy()
