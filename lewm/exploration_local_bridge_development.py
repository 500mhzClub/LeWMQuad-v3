"""Join observed symbolic exits to the existing causal local JEPA controller.

No arrival/place/exit detector is supplied here. The executor must submit actual
arrival evidence; a selected five-tick plan never completes a graph traversal.
The .8-m cue is a local direction, not a metric exit position. Sensor/selection
errors latch and require an explicit executor stop, not reuse of the last plan.
"""
import copy
import math

from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.memory.observed_exploration_development import ObservedExploration
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.online_temporal_choice_development import OnlineTemporalChoice


class ExplorationLocalBridge:
    def __init__(self,memory,template,episode_identity):
        if not isinstance(memory,ObservedExploration) or not isinstance(template,OnlineTemporalChoice):
            raise ValueError('observed memory and fixed local controller template required')
        self.memory=memory; self.template=template; self.identity=_identity(episode_identity)
        self.history=OnlineRGBHistory(); self.history.begin_episode(self.identity)
        self._packets=[]; self._clock=None; self.active=None; self.fault=False

    def observe(self,packet,*,now_ns):
        if self.fault: raise SensorContractError('bridge input fault latched; executor must stop')
        try:
            status=self.history.push(packet,now_ns=now_ns)
            if self.active is not None: self.active.observe(packet,now_ns=now_ns)
            if status['reset_for_gap']: self._packets=[]
            self._packets=(self._packets+[copy.deepcopy(packet)])[-4:]; self._clock=now_ns
            return status
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.fault=True; raise SensorContractError('bridge observation failed; executor must stop') from error

    def start_next(self,action_id,*,now_ns,return_home=False):
        if self.fault: raise SensorContractError('fault-free bridge required')
        if self.active is not None or self.memory.pending is not None: raise ValueError('traversal already active')
        now_ns=_ns(now_ns,'exploration dispatch clock')
        decision=self.memory.decide(now_ns=now_ns,return_home=return_home)
        if decision['kind']!='TRAVERSE_EXIT': return {'memory_decision':decision,'local_selection':None}
        try:
            if self._clock!=now_ns or decision['observation_ns']!=now_ns:
                raise SensorContractError('exit bearing must refer to the actual current packet')
            self.history.tensors(now_ns=now_ns)
            adapter=OnlineTemporalChoice(self.template.method,self.template.models,self.template.bindings)
            adapter.begin_episode(self.identity)
            # Exact stored past packets, never history reconstructed from a
            # later buffer. Each traversal gets a fresh relative direction frame.
            for packet in self._packets: adapter.observe(packet,now_ns=packet['image']['measured_ns'])
            angle=decision['bearing_body_rad']; direction=[.8*math.cos(angle),.8*math.sin(angle)]
            adapter.begin_control(direction,now_ns=now_ns); selection=adapter.select(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.fault=True; raise SensorContractError('local dispatch failed; executor must stop') from error
        # Commit an attempt only after a valid causal plan exists. This remains
        # an outstanding attempt, not an executed or qualified graph edge.
        self.memory.begin(action_id,decision['exit_id'],now_ns=now_ns); self.active=adapter
        return {'memory_decision':decision,'local_selection':selection,'action_id':action_id}

    def select_active(self,*,now_ns):
        if self.fault or self.active is None or self.memory.pending is None:
            raise SensorContractError('active fault-free local traversal required')
        try: return self.active.select(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.fault=True; raise SensorContractError('active local selection failed; executor must stop') from error

    def finish(self,fix,*,reached,viable_arrival):
        if self.active is None or self.memory.pending is None: raise ValueError('no active local traversal')
        if self.fault and (reached or viable_arrival): raise ValueError('faulted local execution cannot qualify arrival')
        self.memory.finish(self.memory.pending['action_id'],fix,reached=reached,viable_arrival=viable_arrival)
        self.active=None
        # A fault is not cleared by ending a graph attempt. Recovery needs a
        # separately constructed, explicitly qualified new execution session.
