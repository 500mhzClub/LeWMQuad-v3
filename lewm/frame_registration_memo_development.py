"""Reuse exact pure registration results within one camera frame."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError


class FrameRegistrationMemo:
    def __init__(self, *, enabled=False, capacity=128):
        self.enabled=enabled;self.capacity=capacity;self.frame=None;self.entries={}
        self.hits=self.misses=self.failure_hits=0

    def call(self, function, a, b, ua, ub, *, gyro_rotation, mode, frame):
        arguments=(a,b,ua,ub)
        keywords=dict(gyro_rotation=gyro_rotation,mode=mode,frame=frame)
        if not self.enabled or type(frame) is not int or not isinstance(mode,str):
            return function(*arguments,**keywords)
        if frame!=self.frame:
            self.frame=frame;self.entries.clear()
        namespace=function.__globals__
        rules=tuple((name,tuple(sorted(namespace[name].items()))) for name in ('RULES','RIGID_RULES'))
        helpers=tuple(namespace[name] for name in ('proper','fit','inliers','cells','angle'))
        arrays=[]
        for value in (*arguments,gyro_rotation):
            array=np.asarray(value,float)
            arrays.append((array.shape,array.strides,array.tobytes()))
        key=(function.__code__,frame,mode,rules,helpers,tuple(arrays))
        found=self.entries.get(key)
        if found is not None:
            self.hits+=1
            success,value=found
            if not success:
                self.failure_hits+=1
                raise SensorContractError(*deepcopy(value))
            return deepcopy(value)
        self.misses+=1
        try:
            result=function(*arguments,**keywords)
        except SensorContractError as error:
            self._store(key,(False,deepcopy(error.args)))
            raise
        self._store(key,(True,deepcopy(result)))
        return result

    def _store(self,key,value):
        if len(self.entries)>=self.capacity:
            self.entries.pop(next(iter(self.entries)))
        self.entries[key]=value


REGISTRATION_MEMO=FrameRegistrationMemo()
