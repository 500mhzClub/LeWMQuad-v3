"""Typed foot-load observations. Neither load nor missingness is terrain truth."""
from dataclasses import dataclass

import numpy as np

FEET=('FL_foot','FR_foot','RL_foot','RR_foot')


def clocks(measured_ns,available_ns):
    if any(type(v) is not int or v<0 for v in (measured_ns,available_ns)) or available_ns<measured_ns:
        raise ValueError('ordered nonnegative integer sensor clocks required')


@dataclass(frozen=True)
class RawFootCounts:
    """Raw hardware transport only; ordering, units and calibration UNVERIFIED."""
    device_identity: str
    reported_channel_order: tuple
    measured_ns: int
    available_ns: int
    values: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray

    def __post_init__(self):
        clocks(self.measured_ns,self.available_ns)
        if not isinstance(self.device_identity,str) or not self.device_identity:
            raise ValueError('explicit device/recording identity required')
        order=self.reported_channel_order
        if type(order) is not tuple or len(order)!=4 or any(not isinstance(v,str) or not v for v in order) or len(set(order))!=4:
            raise ValueError('four distinct reported channel names, not inferred ordering')
        values=np.asarray(self.values); valid=np.asarray(self.valid); saturated=np.asarray(self.saturated)
        if values.shape!=(4,) or values.dtype.kind not in 'iu': raise ValueError('four raw integer counts required')
        if np.any(values< -32768) or np.any(values>32767): raise ValueError('signed16 raw count range required')
        if valid.shape!=(4,) or saturated.shape!=(4,) or valid.dtype!=bool or saturated.dtype!=bool:
            raise ValueError('explicit per-channel validity and saturation required')
        for name,v in [('values',values),('valid',valid),('saturated',saturated)]:
            copy=v.copy(); copy.flags.writeable=False; object.__setattr__(self,name,copy)

    def diagnostic(self):
        return dict(kind='RAW_HARDWARE_COUNTS_UNCALIBRATED',units='raw_integer_counts',
            device_identity=self.device_identity,reported_channel_order=list(self.reported_channel_order),
            measured_ns=self.measured_ns,available_ns=self.available_ns,values=self.values.tolist(),
            valid=self.valid.tolist(),saturated=self.saturated.tolist(),
            canonical_foot_order_verified=False,calibrated_force_n=None,contact_or_support_inferred=False)


@dataclass(frozen=True)
class IdealFootForceSample:
    """Hypothetical three-axis net contact transducers in each URDF foot frame.

    Not vendor foot_force/foot_force_est, nor a calibrated robot sensor.
    Opposing contacts may cancel; resultant force does not encode contact count,
    position, torque, terrain identity or slip.
    """
    acquisition_identity: str
    measured_ns: int
    available_ns: int
    force_foot_n: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray

    def __post_init__(self):
        clocks(self.measured_ns,self.available_ns)
        if not isinstance(self.acquisition_identity,str) or not self.acquisition_identity:
            raise ValueError('explicit acquisition identity required')
        f=np.asarray(self.force_foot_n,float); v=np.asarray(self.valid); s=np.asarray(self.saturated)
        if f.shape!=(4,3) or v.shape!=(4,) or s.shape!=(4,) or v.dtype!=bool or s.dtype!=bool:
            raise ValueError('four explicitly valid/saturated three-axis foot measurements required')
        if not np.isfinite(f[v]).all(): raise ValueError('valid force observations must be finite')
        if not np.isfinite(np.linalg.norm(f[v],axis=1)).all(): raise ValueError('representable force magnitude required')
        for name,value in [('force_foot_n',f),('valid',v),('saturated',s)]:
            copy=value.copy(); copy.flags.writeable=False; object.__setattr__(self,name,copy)


class LocalLoadHistory:
    """Causal dwell on conditional resultant load, NEVER a support predicate."""
    def __init__(self,*,acquisition_identity,threshold_n=5.,dwell_ns=20_000_000,max_age_ns=10_000_000):
        if not isinstance(acquisition_identity,str) or not acquisition_identity: raise ValueError('acquisition identity required')
        if not np.isfinite(threshold_n) or threshold_n<=0: raise ValueError('positive finite load threshold required')
        if any(type(v) is not int or v<=0 for v in (dwell_ns,max_age_ns)): raise ValueError('positive integer timing bounds required')
        self.identity=acquisition_identity; self.threshold=float(threshold_n); self.dwell=dwell_ns; self.max_age=max_age_ns
        self.last=None; self.since=[None]*4

    def observe(self,sample,*,now_ns,conditional_force_error_n=None):
        if not isinstance(sample,IdealFootForceSample): raise ValueError('hardware counts cannot enter an ideal-vector load model')
        clocks(sample.measured_ns,now_ns)
        if sample.acquisition_identity!=self.identity or now_ns<sample.available_ns: raise ValueError('same causal available sensor stream required')
        if self.last is not None and sample.measured_ns<=self.last: raise ValueError('strictly advancing sensor samples required')
        error=conditional_force_error_n
        if error is not None and (np.asarray(error).shape!=() or not np.isfinite(error) or error<0):
            raise ValueError('explicit nonnegative conditional vector error, or unknown')
        if self.last is not None and sample.measured_ns-self.last>self.max_age: self.since=[None]*4
        self.last=sample.measured_ns
        stale=now_ns-sample.measured_ns>self.max_age; rows=[]
        for i,foot in enumerate(FEET):
            lo=hi=None
            if stale: status='STALE'
            elif not sample.valid[i]: status='MISSING'
            elif sample.saturated[i]: status='SATURATED'
            elif error is None: status='UNKNOWN_FORCE_ERROR'
            else:
                norm=float(np.linalg.norm(sample.force_foot_n[i])); lo=max(0.,norm-float(error)); hi=norm+float(error)
                if not np.isfinite(hi): raise ValueError('representable force interval required')
                status='ABOVE_LOAD_THRESHOLD' if lo>self.threshold else ('BELOW_LOAD_THRESHOLD' if hi<=self.threshold else 'AMBIGUOUS_LOAD')
            if status=='ABOVE_LOAD_THRESHOLD':
                if self.since[i] is None: self.since[i]=sample.measured_ns
            else: self.since[i]=None
            rows.append(dict(foot=foot,status=status,resultant_lower_n=lo,resultant_upper_n=hi,
                dwell_observed=bool(self.since[i] is not None and sample.measured_ns-self.since[i]>=self.dwell),
                ground_support_established=False,slip_excluded=False))
        return dict(measured_ns=sample.measured_ns,available_ns=sample.available_ns,now_ns=now_ns,feet=rows,
            hypothesis='IDEAL_SIMULATED_RESULTANT_LOAD_NOT_VENDOR_FORCE',physical_error_calibrated=False,
            continuous_floor_established=False,future_footfall_validated=False,body_sweep_validated=False,
            navigation_qualified=False)
