"""Read-only capture around ordinary source-controller decisions.

No alternative selector is run here. Raw depth stops at the existing perception
boundary; its resulting observed map, pose and obstacle evidence are retained.
"""
import copy
from dataclasses import fields, is_dataclass

import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS


NON_DECISION_FIELDS = frozenset((
    'model', 'model_input_hook', 'motion', 'registration', 'mapper', 'depth_observer',
    'registration_executor', 'mapping_executor', 'pose_executor', 'obstacle_executor',
    'lock', 'correction_pose_lock', 'stopped', 'queues', 'threads', 'clock_ns',
    'evidence_sink', 'events', 'planning', 'plan_profile_rows', 'profile_current',
    'mission_rows', 'visual_support_rows', 'visual_dispatch_events', 'audit_packets',
    '_audit_route', '_audit_route_target',
))


def check_retained_inputs(value, path='packet', seen=None):
    """Reject prohibited raw depth/dense features rather than silently dropping them."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    shape = getattr(value, 'shape', ())
    if len(shape) >= 2 and tuple(shape[-2:]) == (768, 1024):
        raise ValueError(f'dense feature retention prohibited: {path}')
    if isinstance(value, dict):
        members = value.items()
    elif isinstance(value, (tuple, list)):
        members = enumerate(value)
    elif is_dataclass(value):
        members = ((f.name, getattr(value, f.name)) for f in fields(value))
    elif hasattr(value, '__dict__'):
        members = vars(value).items()
    else:
        return
    for key, child in members:
        if str(key) in ('depth_m', 'native_optical_depth_m', 'depth', 'auxiliary_depth') and len(getattr(child, 'shape', ())) >= 2:
            raise ValueError(f'raw depth may be required at {path}.{key}; stop for scoped retention amendment')
        check_retained_inputs(child, f'{path}.{key}', seen)


class DecisionPacketCaptureMixin:
    """Audit-only instrumentation; source decision and return values are unchanged."""
    audit_snapshot_frames = (12, 52, 92, 132)

    def _route(self, *args, **kwargs):
        self._audit_route_target = None
        route = super()._route(*args, **kwargs)
        self._audit_route = copy.deepcopy(route)
        return route

    def _route_target(self, *args, **kwargs):
        target = super()._route_target(*args, **kwargs)
        self._audit_route_target = np.asarray(target).copy()
        return target

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        if not hasattr(self, 'audit_packets'):
            self.audit_packets = {}
        capture = packet.frame in self.audit_snapshot_frames
        if capture:
            if packet.frame in self.audit_packets:
                raise ValueError('duplicate source decision packet')
            stamps = [packet.measured_ns - d for d in (1_000_000_000, 500_000_000, 0)]
            state = {k:v for k,v in vars(self).items() if k not in NON_DECISION_FIELDS}
            record = dict(schema='decision_headroom_frozen_decision_input.v1',
                frame=packet.frame, measured_ns=packet.measured_ns,
                packet=dict(frame=packet.frame, measured_ns=packet.measured_ns,
                    policy=packet.policy, history=packet.history),
                native_context=[self.native_context_packets[t] for t in stamps],
                context_times_ns=stamps, evidence=evidence, committed_prefix=prefix,
                route_target_body_xy=goal_body, scan_error_rad=scan_error,
                route=copy.deepcopy(self._audit_route),
                route_target_observed_map_xy=copy.deepcopy(self._audit_route_target),
                observed_map=snapshot, observed_position=q, observed_rotation=Q,
                controller_state=state, excluded_runtime_fields=sorted(NON_DECISION_FIELDS),
                preprocessing='lewm.dense_native_observation_development.dense_native_context; native 640x480 RGB, PIL bicubic to 512x384, existing normalization',
                source_choice_only=True, comparative_audit_rows_computed=False,
                candidate_order=list(ACTIONS),
                privileged_physics_or_true_geometry=False)
            check_retained_inputs(record)
            record = copy.deepcopy(record)
        selected, correction = super()._select_action(packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q)
        if capture:
            receipt = self.model.receipts[-1]
            if receipt['observed_ns'] != packet.measured_ns:
                raise ValueError('source candidate receipt is not from this decision')
            tapes = np.asarray(receipt['requested_commands'])
            if tapes.shape != (6, 8, 3) or not np.allclose(tapes[:, :3], np.asarray(prefix)[None], rtol=0, atol=1e-7):
                raise ValueError('exact six tapes and unchanged committed prefix required')
            record.update(source_selection=copy.deepcopy(selected), source_correction=copy.deepcopy(correction),
                candidate_requested_commands=tapes.copy(),
                candidate_applied_commands=np.asarray(receipt['applied_commands']).copy(),
                source_model_receipt=copy.deepcopy(receipt))
            metric = selected.get('mission_coordinate_metric', {})
            record['active_target'] = (dict(kind='initial_frame_xy',
                xy=copy.deepcopy(metric['planning_goal_initial_xy_m'])) if metric.get('applied_to_terminal_position')
                else dict(kind='observed_map_xy', xy=copy.deepcopy(self._audit_route_target))
                if self._audit_route_target is not None else
                dict(kind='NO_XY_ROUTE_TARGET', scan_error_rad=scan_error,
                    reason='View-seeking source decision; do not invent a positional target for the reference.'))
            self.audit_packets[packet.frame] = record
        return selected, correction
