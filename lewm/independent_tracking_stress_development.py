"""Fixed development interventions, not a sensor calibration or native launcher.

Only private sensor copies and explicit estimator-reference fault hooks change.
Neither native truth nor geometry enters this module. Production classes/gates
are unchanged. Reference faults test a mechanism, not physical occlusion.
"""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import MIN_DEPTH_M, MAX_DEPTH_M, rgb_digest
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose, MultiReferenceVisualLedMotion
from lewm.temporal_anchor_continuity_development import TemporalAnchorRGBDPose, TemporalAnchorVisualLedMotion

# Frame84 falls within the fixed challenge's first commanded turn. Actual
# rotation/coverage is evaluated later; an early stop may never reach this frame.
ONSET_FRAME = 84
INTERVAL_NS = 100_000_000
SCENARIOS = ('nominal', 'anchor_absence_1', 'anchor_absence_10', 'anchor_absence_11',
             'rgb_unavailable_1', 'depth_unavailable_1', 'gyro_unavailable_1',
             'repeated_rgb', 'depth_drift', 'shared_gyro_bias', 'anchor_increment_conflict')
ARMS = ('original', 'temporal_anchor')


def definition():
    """Return a private, explicit, outcome-independent intervention contract."""
    return dict(schema='independent_tracking_fixed_stress.v1', scenarios=list(SCENARIOS),
        arms=list(ARMS), onset_frame=ONSET_FRAME, interval_ns=INTERVAL_NS,
        anchor_absence_frames=[1, 10, 11], unavailable_current_frames=1,
        depth_drift_m_per_frame=.002, depth_drift_max_m=.04,
        shared_gyro_bias_body_rad_s=[0., 0., .02],
        repeated_rgb=dict(cell_pixels=20, shift_pixels_per_frame=3, levels=[0, 255]),
        incremental_conflict_initial_body_m=[0., .03, 0.],
        noise_distribution_calibrated=False, physical_occlusion_simulated=False,
        independent_observations=False, navigation_qualified=False)


def identity():
    return hashlib.sha256(json.dumps(definition(), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _valid(scenario, frame):
    if scenario not in SCENARIOS or type(frame) is not int or frame < 0:
        raise ValueError('fixed stress scenario and nonnegative integer frame required')


def packet_digest(packet):
    """Bind metadata and all array bytes without serializing pixels into rows."""
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('object arrays cannot bind sensor bytes')
            return dict(array_dtype=value.dtype.str, shape=list(value.shape),
                        sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        if isinstance(value, np.generic): return encode(value.item())
        if isinstance(value, dict):
            if not all(type(k) is str for k in value): raise ValueError('string sensor keys required')
            return {k:encode(v) for k,v in value.items()}
        if isinstance(value, (tuple, list)): return [encode(v) for v in value]
        if value is None or type(value) in (str, int, float, bool): return value
        raise ValueError('unsupported sensor binding value')
    return hashlib.sha256(json.dumps(encode(packet), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def transform(packet, scenario, frame, first_ns):
    """Deterministic private-copy intervention; unchanged truth stays external.

    Shared gyro samples are transformed by measured timestamp, not history-row
    index, so overlapping fast/slow/boundary samples are never rewritten.
    """
    _valid(scenario, frame)
    p,d,f,now = deepcopy(packet)
    if type(first_ns) is not int or now != first_ns + frame*INTERVAL_NS:
        raise ValueError('complete fixed observation clock required')
    offset = frame-ONSET_FRAME
    active = offset >= 0
    changes = []
    affected_gyro_history_entries = 0
    if scenario == 'rgb_unavailable_1' and offset == 0:
        p['image']['available_ns'] = now+1
        changes.append('current_rgb_not_yet_available')
    elif scenario == 'depth_unavailable_1' and offset == 0:
        d['depth_m'][:] = 0.; d['valid'][:] = False
        changes.append('all_current_depth_rays_unknown')
    elif scenario in ('gyro_unavailable_1', 'shared_gyro_bias'):
        onset_ns = first_ns+ONSET_FRAME*INTERVAL_NS
        for channel in (p['sensor_state']['sensed']['gyro'], f):
            mask = channel['measured_ns'] >= onset_ns
            if scenario == 'gyro_unavailable_1':
                mask &= channel['measured_ns'] < onset_ns+INTERVAL_NS
                channel['valid'][mask] = False
                channel['values'][mask] = 0.
            else:
                channel['values'][mask, 2] += .02
            affected_gyro_history_entries += int(np.count_nonzero(mask))
        if affected_gyro_history_entries:
            changes.append('timestamp_consistent_shared_gyro_intervention')
    elif scenario == 'depth_drift' and active:
        delta = np.float32(min(.002*(offset+1), .04))
        values = d['depth_m']; valid = d['valid']
        values[valid] += delta
        valid &= np.isfinite(values) & (values >= MIN_DEPTH_M) & (values <= MAX_DEPTH_M)
        values[~valid] = 0.
        changes.append('correlated_depth_drift_unknown_out_of_range')
    elif scenario == 'repeated_rgb' and active:
        # Synthetic replacement is not physically rendered scene geometry.
        rows,columns = np.indices(p['image']['rgb'].shape[:2])
        pixels = (((rows//20 + ((columns-3*offset) % 640)//20) % 2)*255).astype(np.uint8)
        p['image']['rgb'] = np.repeat(pixels[...,None], 3, axis=2)
        d['rgb_sha256'] = rgb_digest(p)
        changes.append('synthetic_repeated_rgb_with_unchanged_recorded_depth')
    return (p,d,f,now), dict(scenario=scenario, frame=frame,
        onset_reached=active, packet_changes=changes, synthetic_intervention=True,
        affected_gyro_history_entries=affected_gyro_history_entries,
        noise_distribution_calibrated=False, native_pose_input=False)


class _ReferenceInterventions:
    """Per-instance hook; never patch global classes or change acceptance gates."""
    def _candidate(self, ref, current, G):
        scenario = self.stress_scenario
        offset = self.frame-ONSET_FRAME
        is_previous = ref is getattr(self, 'previous', None)
        if scenario.startswith('anchor_absence_') and 0 <= offset < int(scenario.rsplit('_',1)[1]):
            if not is_previous:
                self.injection_events.append(dict(kind='retained_reference_denied', reference_frame=ref.frame))
                raise SensorContractError('explicit development retained-reference denial')
        candidate = super()._candidate(ref,current,G)
        if scenario == 'anchor_increment_conflict' and offset == 0 and is_previous:
            candidate = candidate | dict(p=candidate['p']+np.array([0., .03, 0.]))
            self.injection_events.append(dict(kind='qualified_increment_position_corrupted', reference_frame=ref.frame))
        return candidate


class _OriginalStress(_ReferenceInterventions, MultiReferenceRGBDPose):
    pass


class _TemporalStress(_ReferenceInterventions, TemporalAnchorRGBDPose):
    pass


class PairedStressObserver:
    """One scenario, complete sequential sensor stream, two private observers.

    Rows need a future cohort writer/phase authenticator before native evaluation.
    This callable neither writes files nor grants experiment-execution authority.
    """
    def __init__(self, scenario):
        _valid(scenario,0)
        self.scenario = scenario
        self.models = dict(original=MultiReferenceVisualLedMotion(), temporal_anchor=TemporalAnchorVisualLedMotion())
        for arm, cls in (('original',_OriginalStress), ('temporal_anchor',_TemporalStress)):
            model = cls(); model.stress_scenario = scenario; model.injection_events = []
            self.models[arm].model = model
        self.frame = 0
        self.first_ns = None
        self.failed = False

    def observe(self, packet):
        if self.failed: raise ValueError('failed stress stream cannot restart')
        try:
            if self.first_ns is None: self.first_ns = packet[3]
            before = packet_digest(packet)
            modified, intervention = transform(packet,self.scenario,self.frame,self.first_ns)
            modified_sha = packet_digest(modified)
            rows = {}
            for arm,model in self.models.items():
                p,d,f,now = deepcopy(modified)
                model.model.injection_events = []
                update_attempted = model.failure is None
                start = time.perf_counter_ns()
                row = model.observe(p,d,f,now_ns=now)
                rows[arm] = dict(pose=row['current_pose'], selection=row['reference_selection'],
                    failure=row['terminal_failure'], continuity=row.get('continuity_evidence'),
                    observer_wall_ms=(time.perf_counter_ns()-start)/1e6,
                    observer_update_attempted=update_attempted,
                    reference_injections=deepcopy(model.model.injection_events))
            if packet_digest(packet) != before: raise ValueError('source sensor bytes mutated')
            result = dict(frame=self.frame, measured_ns=packet[3], scenario=self.scenario,
                definition_sha256=identity(), source_packet_sha256=before,
                intervened_packet_sha256=modified_sha, intervention=intervention, arms=rows,
                native_pose_input=False, navigation_qualified=False,
                physical_occlusion_simulated=False, uncertainty_calibrated=False)
            self.frame += 1
            return result
        except BaseException:
            self.failed = True
            raise
