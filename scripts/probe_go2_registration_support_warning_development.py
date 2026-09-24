"""Causal warning-state diagnostic on fixed recorded paths, not navigation."""
import json
import math
from statistics import median

import numpy as np

from lewm.visual_support_recovery_development import LocalSupportedView
from scripts import run_go2_route_turn_memory_transfer_development as run


class CorrespondenceWarning:
    def __init__(self):
        self.good = self.active = None

    def advance(self, counts, inliers, p, R, now):
        strong = max(counts) >= 96 and inliers is not None and inliers >= 24
        weak = max(counts) < 48 or (inliers is not None and inliers < 18)
        if self.active is not None:
            target = self.active['rotation']
            delta = math.atan2(target[1, 0], target[0, 0])-math.atan2(R[1, 0], R[0, 0])
            if abs(math.atan2(math.sin(delta), math.cos(delta))) <= .1 and not weak:
                self.active = None
        if strong:
            self.good = dict(position=p.copy(), rotation=R.copy(), measured_ns=now)
        if self.active is None and weak and self.good is not None:
            if now >= self.good['measured_ns'] and np.linalg.norm(p-self.good['position']) <= .20:
                self.active = self.good | dict(trigger_ns=now)
        return self.active


def main():
    summaries = []
    for number in (1, 2):
        root = run.BASE / run.root_name(number)
        read = lambda name: json.loads((root/name).read_text())
        replay = read('registration_support_replay_v1/result.json')
        poses = {r['frame']: r['registered_pose'] for r in read('poses.json')}
        planning = {r['frame']: r for r in read('visual_support_recovery.json')}
        baseline = LocalSupportedView(maximum_view_age_ns=None)
        proposed = CorrespondenceWarning()
        onsets = {'baseline': [], 'proposed': []}
        seen = {'baseline': set(), 'proposed': set()}
        active_counts = dict(baseline=0, proposed=0)
        first_difference = None
        checked = 0
        for row in replay['rows']:
            frame = row['frame']; pose = poses[frame]; now = row['measured_ns']
            p = np.asarray(pose['position_initial_body_m'])
            R = np.asarray(pose['rotation_initial_body_from_current_body'])
            fit = row['selected_fit']; inliers = None if fit is None else fit['inliers']
            old = baseline.advance(row['selected_features'], p, R, now, 0)
            new = proposed.advance(row['selected_features'], inliers, p, R, now)
            if frame in planning:
                recorded = planning[frame]['recovery_state_at_observation']
                assert (recorded is None) == (old is None)
                if old is not None:
                    assert old['trigger_ns'] == recorded['trigger_ns']
                    assert old['measured_ns'] == recorded['measured_ns']
                    np.testing.assert_array_equal(old['rotation'], recorded['rotation'])
                    np.testing.assert_array_equal(old['position'], recorded['position'])
                checked += 1
            for name, state in (('baseline', old), ('proposed', new)):
                if state is not None:
                    active_counts[name] += 1
                    if state['trigger_ns'] not in seen[name]:
                        seen[name].add(state['trigger_ns'])
                        onsets[name].append(dict(frame=frame, trigger_ns=state['trigger_ns'],
                            reference_ns=state['measured_ns'], inliers=inliers,
                            selected_features=row['selected_features']))
            if first_difference is None and ((old is None) != (new is None) or
                    (old is not None and new is not None and (old['trigger_ns'], old['measured_ns']) !=
                     (new['trigger_ns'], new['measured_ns']))):
                first_difference = frame
        fits = [r for r in replay['rows'] if r['selected_fit'] is not None]
        result = dict(assignment=number, accepted_frames=len(replay['rows']),
            original_planning_recovery_states_reproduced=checked,
            median_selected_fit_inliers=median(r['selected_fit']['inliers'] for r in fits),
            frames_below_18_inliers=sum(r['selected_fit']['inliers'] < 18 for r in fits),
            first_hypothetical_state_difference_frame=first_difference,
            active_camera_frames=active_counts, onsets=onsets,
            thresholds=dict(weak_inliers=18, strong_inliers=24, weak_features=48, strong_features=96),
            recorded_trajectory_unchanged=True, navigation_outcome_tested=False,
            later_hypothetical_states_are_not_counterfactual_navigation=True)
        with (root/'registration_support_replay_v1/warning_probe_v1.json').open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
        summaries.append({k: v for k, v in result.items() if k != 'onsets'} |
            dict(onset_counts={k: len(v) for k, v in onsets.items()}))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
