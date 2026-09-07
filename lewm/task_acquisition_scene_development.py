"""Four-fixture fixed-control factorial plus matched learned joint arms."""
import copy

from lewm.persistent_alignment_scene_development import trials as predecessor_trials
from lewm.task_acquisition_continuation_development import POLICIES


def trials():
    previous = {(row['case_index'], row['method']): row for row in predecessor_trials()}
    rows = []
    arms = [('fixed_forward', p) for p in POLICIES] + [
        (m, 'both') for m in ('direct_direct', 'supervised_rollout', 'jepa_rollout')]
    for case in range(4):
        for method, policy in arms:
            spec = copy.deepcopy(previous[(case, method)])
            spec.update(scene_id=f'task-acquisition-development-v1-{case:02d}-{method}-{policy}',
                        family='TASK_ACQUISITION_CONTINUATION_DEVELOPMENT',
                        procedural_seed=2026100100+case, acquisition_policy=policy)
            rows.append(spec)
    return rows
