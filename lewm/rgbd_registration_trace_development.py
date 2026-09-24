"""Read-only witnesses from unchanged estimator calls; never a pose gate.

Scoped single-thread Python tracing observes only three exact frozen functions.
It neither replaces matching/registration nor feeds diagnostics to the observer.
Instrumented timings are not uninstrumented latency or realtime evidence.
"""
from copy import deepcopy
import sys
import time

import numpy as np

from lewm.joint_rgbd_rigid_pose_development import register, proper, RULES, RIGID_RULES
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose
from lewm.rgbd_correspondence_motion_development import cells


def conditioning(a, b, rotation, translation):
    """Conditional linear sensitivity, not calibrated uncertainty or permission.

    For fixed R, residuals are a_i - R b_i - t and J_t = -I. N point matches
    therefore give J_t.T J_t = N I regardless of image-grid occupancy. This
    assumes correct correspondences and does not turn correlated matches into
    independent observations. Common-mode point bias is invisible to residuals.
    """
    a, b, t = np.asarray(a, float), np.asarray(b, float), np.asarray(translation, float)
    R = proper(rotation)
    if a.ndim != 2 or a.shape[1:] != (3,) or b.shape != a.shape or len(a) < 3 or t.shape != (3,):
        raise ValueError('matched finite conditioned point population required')
    if not all(np.isfinite(v).all() for v in (a, b, t)):
        raise ValueError('finite conditioned point population required')
    residual = a - b @ R.T - t
    centered = b - b.mean(0)
    # Separate angular information units (m²), not a unit-mixed 6x6 condition
    # number presented as physical pose certainty. Rotation is not fitted here.
    angular = sum((np.dot(p, p) * np.eye(3) - np.outer(p, p) for p in centered), np.zeros((3, 3))) / len(b)
    return dict(points=len(a), fixed_rotation_translation_normal_eigenvalues=[float(len(a))] * 3,
        fixed_rotation_translation_rank=3, fixed_rotation_translation_condition_number=1.,
        reference_scatter_rms_m=(np.linalg.svd(a - a.mean(0), compute_uv=False) / np.sqrt(len(a))).tolist(),
        current_scatter_rms_m=(np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(b))).tolist(),
        centered_rotation_normal_eigenvalues_m2=np.linalg.eigvalsh(angular).tolist(),
        gyro_common_angle_translation_sensitivity_m_per_rad=float(np.linalg.norm(b.mean(0))),
        residual_rms_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        mean_residual_m=residual.mean(0).tolist(),
        covariance_calibrated=False, static_correct_correspondences_assumed=True,
        independent_point_noise_established=False, common_mode_bias_bounded=False,
        changes_acceptance=False, navigation_qualified=False)


def match_summary(values, result):
    """Count the actual surviving populations in the unchanged matcher."""
    kp0, kp1 = values['kp0'], values['kp1']
    forward, backward = values.get('forward', {}), values.get('backward', {})
    pairs = values.get('pairs', [])
    good = values.get('good', np.empty(0, bool))
    va, vb = values.get('va', np.empty(0, bool)), values.get('vb', np.empty(0, bool))
    a, b, ua, ub = result
    return dict(reference_keypoints=len(kp0), current_keypoints=len(kp1),
        forward_ratio_matches=len(forward), backward_ratio_matches=len(backward),
        mutual_ratio_before_dedup=sum(backward.get(j) == i for i, j in forward.items()),
        unique_mutual_pairs=len(pairs), bidirectional_flow_survivors=int(good.sum()),
        reference_depth_survivors=int(va.sum()), current_depth_survivors=int(vb.sum()),
        paired_depth_survivors=len(a), lifted_reference_grid_cells=cells(ua),
        lifted_current_grid_cells=cells(ub))


def registration_summary(values, result, error):
    accepted = result is not None
    final_gate_failure = error in (
        'rigid consensus fraction, grid support or displacement rejected',
        'image and gyro reference rotations disagree beyond diagnostic envelope')
    row = dict(original_registration_accepted=accepted, original_registration_error=None if accepted else error,
        lifted_matches=len(values['a']), valid_proposals=values.get('valid_candidates', 0),
        initial_consensus_points=values.get('initial_count'), pruning_rounds=values.get('rounds', 0),
        converged_candidate_available=accepted or final_gate_failure, candidate=None)
    if not row['converged_candidate_available']:
        return row  # Never label a stale intermediate fit as a converged estimate.
    R, t, mask = (result[:3] if accepted else (values['R'], values['t'], values['mask']))
    a, b, ua, ub = (values[k] for k in ('a', 'b', 'ua', 'ub'))
    rc, cc = cells(ua[mask]), cells(ub[mask])
    row.update(inliers=int(mask.sum()), inlier_fraction=float(mask.mean()),
        reference_grid_cells=rc, current_grid_cells=cc,
        original_gate_failures=dict(inlier_fraction=bool(mask.mean() < RULES['minimum_inlier_fraction']),
            reference_grid=rc < RULES['minimum_grid_cells'], current_grid=cc < RULES['minimum_grid_cells'],
            reference_translation=bool(np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m'])),
        conditioning=conditioning(a[mask], b[mask], R, t),
        candidate=dict(rotation=R.tolist(), translation=t.tolist(),
            diagnostic_only=True, passes_registration_only=accepted))
    return row


class TracePairs:
    """One observation call, bounded original primary/alternative candidates.

    Native state cannot be supplied to this interface. No extrapolation or
    diagnostic candidate is returned to the observer. Trace faults are retained
    and raised after tracing is restored, not converted into estimator success.
    """
    def __init__(self):
        self.pairs = []; self.error = None; self.current = None; self.errors = {}; self.started = {}

    def __enter__(self):
        if sys.gettrace() is not None:
            raise ValueError('untraced single-thread replay required; do not replace another tracer')
        self.codes = {MultiReferenceRGBDPose._candidate.__code__: 'candidate',
            matched_points.__code__: 'matching', register.__code__: 'registration'}
        sys.settrace(self._trace)
        return self

    def _trace(self, frame, event, arg):
        kind = self.codes.get(frame.f_code)
        if kind is None: return None
        frame.f_trace_lines = False
        try:
            values = frame.f_locals
            if event == 'call':
                self.errors[kind] = None
                self.started[kind] = time.perf_counter_ns()
                if kind == 'candidate':
                    if self.current is not None or len(self.pairs) >= 8:
                        raise ValueError('bounded nonrecursive original candidate calls required')
                    ref = values['ref']; model = values['self']
                    self.current = dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
                        current_frame=model.frame, reference_position=ref.position.tolist(),
                        reference_rotation=ref.rotation.tolist(), matching=None, registration=None,
                        original_candidate_accepted=False, original_candidate_error=None,
                        diagnostic_position_initial_body_m=None, diagnostic_rotation_initial_body=None)
            elif event == 'exception':
                self.errors[kind] = str(arg[1])
            elif event == 'return' and self.current is not None:
                self.current['instrumented_' + kind + '_wall_ms'] = (time.perf_counter_ns() - self.started[kind]) / 1e6
                if kind == 'matching':
                    self.current['matching'] = match_summary(values, arg) if arg is not None else dict(error=self.errors[kind])
                elif kind == 'registration':
                    self.current['registration'] = registration_summary(values, arg, self.errors[kind])
                else:
                    row = self.current
                    row['original_candidate_accepted'] = arg is not None
                    row['original_candidate_error'] = None if arg is not None else self.errors[kind]
                    reg = row['registration']
                    if reg and reg['candidate'] is not None:
                        c = reg['candidate']; ref = values['ref']
                        row['diagnostic_position_initial_body_m'] = (ref.position + ref.rotation @ c['translation']).tolist()
                        row['diagnostic_rotation_initial_body'] = (ref.rotation @ c['rotation']).tolist()
                    if arg is not None:
                        if not np.array_equal(arg['p'], row['diagnostic_position_initial_body_m']):
                            raise ValueError('diagnostic composition must exactly match original accepted candidate')
                    self.pairs.append(row); self.current = None
        except Exception as error:
            self.error = repr(error)
        return self._trace

    def __exit__(self, typ, value, traceback):
        sys.settrace(None)
        if self.error is not None or self.current is not None:
            raise RuntimeError('registration trace failed: ' + str(self.error))
        return False

    def record(self):
        return deepcopy(self.pairs)
