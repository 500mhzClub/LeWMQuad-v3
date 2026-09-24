"""Compare tracking on the saved second-maze failure, without native inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from lewm.cached_moments_deferred_copy_tracking_development import CachedMomentsDeferredCopyMotion
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitMotion, CadencedViewRevisitPose
from lewm.causal_sensor_state import SensorContractError
from lewm.local_view_revisit_tracking_development import select_revisit
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.chained_anchor_dual_camera_pose_development import TRACE_FIELDS
from lewm.recent_local_view_bank_development import RecentLocalViewBankMotion
from scripts.run_go2_cadenced_view_maze00_development import study, ROOT
from scripts.live_depth_noise_session_development import NoisyPublicReplay


class FailureViewProbePose(CadencedViewRevisitPose):
    """Record availability on failure without retrying or changing tracking."""
    probe_stored_fits = False

    def _measure(self, current, gyro, now):
        self.failure_view_probe = None
        try:
            return super()._measure(current, gyro, now)
        except SensorContractError:
            eligible = select_revisit(self.local_view_bank, self.references,
                self.last_p, gyro, now)
            active = {r.frame for r in self.references}
            self.failure_view_probe = dict(frame=self.frame,
                eligible_reference_frame=None if eligible is None else eligible[0].frame,
                continuity_status=(self.last_continuity or {}).get('status'),
                current_gyro=gyro.tolist(), previous_accepted_position_m=self.last_p.tolist(),
                active_reference_frames=sorted(active),
                bank=[dict(frame=r.frame, age_ns=now-r.measured_ns,
                    distance_m=float(np.linalg.norm(r.position-self.last_p)),
                    rotation_difference_rad=angle(r.gyro.T@gyro), already_active=r.frame in active)
                    for r, plane in self.local_view_bank.values()],
                measurement_retry_attempted=False, tracking_behavior_changed=False)
            if self.probe_stored_fits:
                self.failure_view_probe['stored_view_fits'] = self._probe_fits(current, gyro)
            raise

    def _probe_fits(self, current, gyro):
        # Diagnostic only: these candidates never supply a pose or restart the
        # failed observer. Retain the original terminal evidence unchanged.
        saved = {k: deepcopy(getattr(self, k)) for k in (*TRACE_FIELDS, 'rejected_plane_pairs')}
        planes, conflict = self._planes, self._plane_conflict
        rows = []
        try:
            for reference, plane in self.local_view_bank.values():
                self._planes = planes | {reference.frame: plane}
                for camera in ('primary', 'auxiliary'):
                    self.camera = camera
                    self.rotation_measurements = []
                    self._plane_conflict = None
                    row = dict(reference_frame=reference.frame, camera=camera,
                        candidate_qualified=False, pose_accepted=False)
                    try:
                        candidate = self._candidate(reference, current, gyro)
                        reg = candidate['registration']
                        row.update(candidate_qualified=True,
                            position_initial_body_m=candidate['p'].tolist(),
                            rotation_initial_body_from_current_body=candidate['R'].tolist(),
                            inliers=reg['inliers'], inlier_fraction=reg['inlier_fraction'],
                            residual_rms_m=reg['residual_rms_m'])
                    except SensorContractError as error:
                        row['failure'] = str(error)
                    rows.append(row)
        finally:
            for key, value in saved.items():
                setattr(self, key, value)
            self._planes, self._plane_conflict = planes, conflict
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', choices=('cadenced', 'original', 'recent_bank'), required=True)
    parser.add_argument('--support-trace', action='store_true')
    parser.add_argument('--failure-view-probe', action='store_true')
    parser.add_argument('--probe-stored-view-fits', action='store_true')
    parser.add_argument('--source-root', default=ROOT)
    parser.add_argument('--output-root')
    args = parser.parse_args()
    if args.probe_stored_view_fits and not args.failure_view_probe:
        raise ValueError('stored-view fits require the diagnostic failure probe')
    if args.failure_view_probe and args.tracker != 'cadenced':
        raise ValueError('failure availability probe requires the unchanged cadenced tracker')
    suffix = '_support' if args.support_trace else ''
    for name in (args.source_root, args.output_root):
        if name is not None and (Path(name).name != name or name.startswith('sealed_') or name == 'sealed'):
            raise ValueError('ordinary development root name required')
    output = study.BASE/(args.output_root or f'go2_cadenced_maze00_failure_{args.tracker}{suffix}_replay_v1_attempt_001')
    output.mkdir(exist_ok=False)
    (output/'replay.py').write_text(Path(__file__).read_text())
    if args.tracker == 'recent_bank':
        (output/'recent_local_view_bank_development.py').write_text(
            Path('lewm/recent_local_view_bank_development.py').read_text())
    root = study.BASE/args.source_root
    if (root/'DEPTH_RETIRED').exists():
        raise ValueError('full original sensor replay unavailable after depth retirement')
    reader = NoisyPublicReplay(root/'native')
    recorded = {r['frame']: r['raw_pose'] for r in json.loads((root/'poses.json').read_text())}
    count = len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    study.previous.reference.initialize_pose()
    motion = dict(cadenced=CadencedViewRevisitMotion, original=CachedMomentsDeferredCopyMotion,
        recent_bank=RecentLocalViewBankMotion)[args.tracker]()
    if args.failure_view_probe:
        motion.model = FailureViewProbePose()
        motion.model.probe_stored_fits = args.probe_stored_view_fits
    rows = []
    started = time.monotonic()
    status = dict(status='COMPLETE')
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        begin = time.perf_counter_ns()
        result = motion.observe(policy, depth, fast, now_ns=now,
            auxiliary_rgb=rgb, auxiliary_depth=auxiliary)
        elapsed = (time.perf_counter_ns()-begin)/1e6
        current = result.get('current_pose')
        row = dict(frame=frame, wall_ms=elapsed, pose=current,
            revisit_attempt=motion.model.last_revisit_attempt,
            selection=motion.model.last_selection,
            reference_frames=[r.frame for r in motion.model.references])
        if args.tracker == 'recent_bank':
            row['local_view_bank_frames'] = {str(k): ref.frame
                for k, (ref, plane) in motion.model.local_view_bank.items()}
        rows.append(row)
        if args.failure_view_probe:
            row['failure_view_probe'] = getattr(motion.model, 'failure_view_probe', None)
        if args.support_trace:
            row['feature_support'] = {
                camera: result.get(key) for camera, key in (
                    ('primary', 'last_accepted_feature_witness'),
                    ('auxiliary', 'auxiliary_feature_witness'))}
            row['feature_support_frame'] = None if motion.model.previous is None else motion.model.previous.frame
            row['camera_selection'] = result.get('camera_selection')
        if current is None or result.get('failure') is not None:
            status = dict(status='FAILED', frame=frame, failure=result)
            break
        if args.tracker == 'cadenced' and frame in recorded:
            for key in ('position_initial_body_m', 'rotation_initial_body_from_current_body'):
                np.testing.assert_array_equal(current[key], recorded[frame][key])
            assert current['reference_frame'] == recorded[frame]['reference_frame']
            assert current['mode'] == recorded[frame]['mode']
            row['recorded_pose_reproduced'] = True
        if frame % 200 == 0:
            print('FAILURE_REPLAY', args.tracker, frame, round(time.monotonic()-started, 2), flush=True)
    result = status | dict(tracker=args.tracker, source_root=args.source_root, acquired_frames=count,
        rows=rows, wall_s=time.monotonic()-started, public_sensor_replay_only=True,
        native_truth_used=False, navigation_outcome_tested=False,
        native_concurrency_reproduced=False)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in status.items() if k != 'failure'}), flush=True)


if __name__ == '__main__':
    main()
