"""Causal four-frame live history for the ideal-sensor temporal model interface."""
import copy

import numpy as np

from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.simulated_body_observation_development import validate_policy_packet


class OnlineRGBHistory:
    """No model, planner, oracle state or persistent place memory is stored.

    Call push at every 100-ms observation boundary, even between replans. A gap
    clears image continuity; four new consecutive packets are needed afterward.
    Invalid input clears readiness and raises, so callers must not reuse old
    predictions. This module does not itself issue a physical stop command.
    """
    def __init__(self):
        self._seen=set(); self._episode=None; self._packets=[]; self._last_received=None

    def begin_episode(self,identity):
        identity=_identity(identity)
        if identity in self._seen: raise SensorContractError('episode/reset identity already used')
        self._seen.add(identity); self._episode=identity; self._packets=[]; self._last_received=None

    def push(self,packet,*,now_ns):
        try:
            now_ns=_ns(now_ns,'online history clock')
            if self._last_received is not None and now_ns<=self._last_received:
                raise SensorContractError('online history clock must advance')
            previous=self._last_received; self._last_received=now_ns
            validate_policy_packet(packet); state=packet['sensor_state']
            if self._episode is None or _identity(state['identity'])!=self._episode:
                raise SensorContractError('inactive or changed episode/reset')
            if now_ns%100_000_000 or state['decision_ns']!=now_ns or packet['image']['measured_ns']!=now_ns:
                raise SensorContractError('fresh command-clock image required')
            for role in ('sensed','control'):
                for row in state[role].values():
                    if not np.asarray(row['valid']).all() or row['measured_ns'][-1]!=now_ns:
                        raise SensorContractError('complete fresh ideal sensor history required')
            contiguous=previous is not None and now_ns-previous==100_000_000
            if contiguous and self._packets:
                old=self._packets[-1]['sensor_state']
                for role in ('sensed','control'):
                    for name,row in state[role].items():
                        prior=old[role][name]; lookup={int(t):i for i,t in enumerate(prior['measured_ns'])}
                        for i,t in enumerate(row['measured_ns']):
                            if int(t) not in lookup: continue
                            j=lookup[int(t)]
                            for field in ('values','valid','available_ns'):
                                if not np.array_equal(np.asarray(row[field])[i],np.asarray(prior[field])[j]):
                                    raise SensorContractError('previous sensor sample was rewritten')
            if not contiguous: self._packets=[]
            self._packets.append(copy.deepcopy(packet)); self._packets=self._packets[-4:]
            return {'ready':len(self._packets)==4,'frames':len(self._packets),
                'reset_for_gap':previous is not None and not contiguous,'decision_ns':now_ns}
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self._packets=[]
            if isinstance(error,SensorContractError): raise
            raise SensorContractError('invalid online history packet') from error

    def tensors(self,*,now_ns):
        now_ns=_ns(now_ns,'online history inference clock')
        if len(self._packets)!=4 or self._last_received!=now_ns:
            raise SensorContractError('four fresh consecutive packets required before inference')
        # Conversion allocates new tensors; caller mutations cannot rewrite memory.
        return causal_history_tensors(self._packets,now_ns)
