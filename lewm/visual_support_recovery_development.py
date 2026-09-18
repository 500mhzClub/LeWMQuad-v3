"""Request a previously measured local view before image support disappears.

Feature counts are development heuristics, not calibrated tracking confidence.
The existing forecast, clearance, dispatch and measured-pose gates still apply.
"""
import math
import numpy as np

from lewm.pipeline_age_dispatch_development import PipelineAgeRuntime

LOW_FEATURES = 48
STRONG_FEATURES = 96
MAX_VIEW_AGE_NS = 10_000_000_000
MAX_VIEW_DISTANCE_M = .20


def support(raw):
    pose = raw.get('current_pose')
    if pose is None:
        return None
    witnesses = [raw.get(k) for k in
        ('last_accepted_feature_witness', 'auxiliary_feature_witness')]
    if any(w is None for w in witnesses):
        return None
    return dict(frame=pose['frame'], measured_ns=pose['measured_ns'],
        selected_features=[w['selected_features'] for w in witnesses])


class SupportRegistration:
    def __init__(self, original):
        self.original = original

    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        result = self.original.observe(policy, primary, auxiliary, raw, now_ns=now_ns)
        return result | dict(visual_support=support(raw))


class LocalSupportedView:
    def __init__(self, *, maximum_view_age_ns=MAX_VIEW_AGE_NS):
        self.good = None
        self.active = None
        self.generation = None
        self.maximum_view_age_ns = maximum_view_age_ns

    def advance(self, counts, position, rotation, now_ns, generation):
        if self.generation != generation:
            self.good = self.active = None
            self.generation = generation
        strength = max(counts)
        if self.active is not None:
            target = self.active['rotation']
            heading_error = math.atan2(target[1, 0], target[0, 0])-math.atan2(rotation[1, 0], rotation[0, 0])
            error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            if abs(error) <= .1 and strength >= LOW_FEATURES:
                self.active = None
        if strength >= STRONG_FEATURES:
            self.good = dict(position=position.copy(), rotation=rotation.copy(),
                measured_ns=now_ns, selected_features=list(counts))
        if self.active is None and strength < LOW_FEATURES and self.good is not None:
            age = now_ns-self.good['measured_ns']
            if (age >= 0 and (self.maximum_view_age_ns is None or age <= self.maximum_view_age_ns)
                    and np.linalg.norm(position-self.good['position']) <= MAX_VIEW_DISTANCE_M):
                self.active = self.good | dict(trigger_ns=now_ns)
        return self.active


class VisualSupportRuntime(PipelineAgeRuntime):
    def __init__(self, *args, **kwargs):
        self.supported_view = LocalSupportedView()
        self.visual_support_rows = []
        super().__init__(*args, **kwargs)
        self.registration = SupportRegistration(self.registration)

    def _recovery_state(self, receipt, position, rotation, measured_ns):
        return self.supported_view.advance(receipt['selected_features'], position, rotation,
            measured_ns, self.mission_generation)

    def _route(self, snapshot, evidence, goal, *, measured_ns):
        result = super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        receipt = evidence.get('visual_support')
        if receipt is None:
            return result
        p, R, pose = self._pose(evidence, identity=(0, 0, 0), now_ns=measured_ns)
        if receipt['measured_ns'] != measured_ns or receipt['frame'] != pose['frame']:
            raise ValueError('visual support must belong to the planning pose')
        active = self._recovery_state(receipt, p, R, measured_ns)
        row = receipt | dict(recovery_active=active is not None,
            feature_count_is_calibrated_confidence=False)
        if active is not None:
            target_R = np.asarray(snapshot.map_from_initial)@active['rotation']
            target = math.atan2(target_R[1, 0], target_R[0, 0])
            row.update(target_heading_rad=target, reference_measured_ns=active['measured_ns'],
                trigger_ns=active['trigger_ns'])
            # Retain every forecast clearance check, but do not carry a turn
            # direction chosen for the previous route objective into this view.
            self.clearance_turn = None
            result = result | dict(route_cells=[], view_heading_rad=target,
                status='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW')
        self.visual_support_rows.append(row)
        return result
