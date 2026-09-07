"""Conditional sensor-to-joint-pose sensitivities, not calibrated covariance.

Each named source is one shared, unit-variance latent variable over the entire
history. Explicit loadings may describe bias, shared frames, or independent
samples. Paired causal estimators propagate those sources through the actual
fast gyro, gravity initialization and moment-fusion code. Depth registration
rank/correspondences are CONDITIONAL inputs: this is not an ICP error model.
No result is connected to clearance or navigation approval.
"""
from copy import deepcopy

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator, MomentDepthInertialState
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.relative_pose_uncertainty_development import relative_point_moments


SHAPES = {'specific_force': (20, 3), 'gyro': (20, 3), 'fast_gyro': (51, 3),
          'depth_projection': (3,), 'depth_basis_rotation': (3,)}
RAW_SHAPES = {'specific_force': (20, 3), 'gyro': (20, 3), 'fast_gyro': (51, 3),
              'depth_m': (480, 640)}


class CorrelatedMomentSensitivity:
    """Finite-difference diagnostic with a fixed, explicit latent-source basis.

Loadings have physical units per dimensionless source standard deviation.
Different source IDs assert zero cross covariance; shared IDs preserve all
temporal/channel cross terms. That assertion must be validated externally.
The finite-difference step is NOT a sensor-error magnitude or confidence level.
"""

    def __init__(self, source_ids, *, difference_step=1e-3):
        names = tuple(source_ids)
        if (not names or len(names) > 64 or any(type(n) is not str or not n for n in names)
                or len(set(names)) != len(names) or not np.isfinite(difference_step)
                or not 0 < difference_step <= .01):
            raise SensorContractError('one to 64 unique error sources and finite small difference step required')
        self.source_ids, self.step = names, float(difference_step)
        self.models = [(FastRelativeOrientation(), MomentWeakSubspaceIntegrator())
                       for _ in range(1 + 2 * len(names))]
        self.previous = None
        self.current = None
        self.retained = {}
        self.failed = False

    def _loadings(self, policy, fast, loadings, *, shapes=SHAPES):
        if set(loadings) != set(shapes):
            raise SensorContractError('all explicit raw-sensor and conditional-depth loadings required')
        result = {name: np.asarray(loadings[name], dtype=float).copy() for name in shapes}
        for name, shape in shapes.items():
            if result[name].shape != (*shape, len(self.source_ids)) or not np.isfinite(result[name]).all():
                raise SensorContractError('finite source-bound loading shape required')
        sensed = policy['sensor_state']['sensed']
        histories = {name: (np.asarray(sensed[name]['measured_ns']), result[name])
                     for name in ('specific_force', 'gyro')}
        histories['fast_gyro'] = (np.asarray(fast['measured_ns']), result['fast_gyro'])
        # The same physical sample cannot acquire a new latent error when it
        # appears in an overlapping history or at the other sampling rate.
        for name, (times, values) in histories.items():
            if self.previous is not None:
                old_times, old_values = self.previous[name]
                lookup = {int(t): i for i, t in enumerate(old_times)}
                for i, t in enumerate(times):
                    if int(t) in lookup and not np.array_equal(values[i], old_values[lookup[int(t)]]):
                        raise SensorContractError('shared sensor error loading rewritten')
        lookup = {int(t): i for i, t in enumerate(histories['gyro'][0])}
        for i, t in enumerate(histories['fast_gyro'][0]):
            if int(t) in lookup and not np.array_equal(result['fast_gyro'][i], result['gyro'][lookup[int(t)]]):
                raise SensorContractError('fast/slow gyro error sources disagree')
        self.previous = deepcopy(histories)
        return result

    def observe(self, policy, depth_state, fast, loadings):
        if self.failed:
            raise SensorContractError('joint sensitivity fault latched')
        try:
            return self._observe(policy, depth_state, fast, loadings)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            self.current = None
            raise SensorContractError('invalid conditional sensor-error history') from error

    def _observe(self, policy, depth_state, fast, loadings):
        factors = self._loadings(policy, fast, loadings)
        motion = depth_state['motion']
        if motion is None:
            if np.any(factors['depth_projection']) or np.any(factors['depth_basis_rotation']):
                raise SensorContractError('initial anchor has no depth registration error input')
        else:
            weak = np.asarray(motion['weak_directions_previous_body'], dtype=float).reshape(-1, 3)
            if not np.allclose(weak @ factors['depth_projection'], 0., atol=1e-12, rtol=0):
                raise SensorContractError('depth-error loading must lie in the observed subspace')
        now = policy['sensor_state']['decision_ns']
        outputs = []
        for index, (orientation, integrator) in enumerate(self.models):
            if index == 0:
                p, f, observed = policy, fast, depth_state
            else:
                source, sign = (index - 1) // 2, (1. if index % 2 else -1.)
                scale = sign * self.step
                p, f, observed = deepcopy(policy), deepcopy(fast), deepcopy(depth_state)
                for name in ('specific_force', 'gyro'):
                    p['sensor_state']['sensed'][name]['values'] = (
                        p['sensor_state']['sensed'][name]['values'] + scale * factors[name][..., source])
                f['values'] = f['values'] + scale * factors['fast_gyro'][..., source]
                if motion is not None:
                    basis = rotation_increment(scale * factors['depth_basis_rotation'][:, source])
                    projection = basis @ (np.asarray(motion['observable_projection_previous_body_m'])
                                          + scale * factors['depth_projection'][:, source])
                    observed['motion']['observable_projection_previous_body_m'] = projection.tolist()
                    observed['motion']['weak_directions_previous_body'] = (weak @ basis.T).tolist()
                    if motion['rank'] == 3:
                        observed['motion']['translation_previous_body_m'] = projection.tolist()
            attitude = (orientation.begin(p, f, now_ns=now) if orientation.status == 'NEW'
                        else orientation.step(p, f, now_ns=now))
            if index == 0:
                if not np.allclose(attitude['rotation_initial_body_from_current_body'],
                                   observed['relative_orientation']['rotation_initial_body_from_current_body'],
                                   atol=1e-12, rtol=0):
                    raise SensorContractError('depth attitude must match the actual fast gyro history')
            else:
                observed['relative_orientation'] = attitude
            result = integrator.observe(p, observed)
            outputs.append((result, orientation.rotation.copy()))
        return self._summarize(outputs, now, conditioned=True)

    def _summarize(self, outputs, now, *, conditioned):
        nominal, rotation = outputs[0]
        position = np.asarray(nominal['position_initial_body_m'])
        influence = np.empty((6, len(self.source_ids)))
        curvature = []
        for source in range(len(self.source_ids)):
            (plus, rp), (minus, rm) = outputs[1 + 2 * source:3 + 2 * source]
            pp, pm = [np.asarray(s['position_initial_body_m']) for s in (plus, minus)]
            influence[:3, source] = (pp - pm) / (2 * self.step)
            tangent = ((rp - rm) / (2 * self.step)) @ rotation.T
            influence[3:, source] = np.array([tangent[2, 1] - tangent[1, 2],
                                             tangent[0, 2] - tangent[2, 0],
                                             tangent[1, 0] - tangent[0, 1]]) * .5
            curvature.append(float(np.linalg.norm((pp + pm) * .5 - position)))
        self.current = {'measured_ns': now, 'position': position.copy(), 'rotation': rotation,
                        'influence': influence, 'conditioned': conditioned}
        return {'nominal_fusion': deepcopy(nominal), 'pose_error_factor': influence.copy(),
                'conditional_pose_covariance': influence @ influence.T,
                'source_ids': self.source_ids, 'difference_step': self.step,
                'position_midpoint_remainder_m': curvature,
                'depth_rank_and_correspondences_conditioned_on': conditioned,
                'source_error_model_calibrated': False, 'first_order_only': True,
                'linearization_validated': False,
                'navigation_qualified': False, 'hardware_qualified': False}

    def retain(self, label):
        if self.failed or self.current is None or type(label) is not str or not label or label in self.retained:
            raise SensorContractError('active observation and fresh retained-pose label required')
        if len(self.retained) >= 64:
            raise SensorContractError('explicit 64-pose diagnostic retention limit')
        self.retained[label] = deepcopy(self.current)

    def relative_moments(self, label, points_body):
        if self.failed or self.current is None or label not in self.retained:
            raise SensorContractError('active current and retained conditional poses required')
        current, old = self.current, self.retained[label]
        joint_factor = np.concatenate((current['influence'], old['influence']))
        joint = joint_factor @ joint_factor.T
        result = relative_point_moments(points_body, current['position'], current['rotation'],
                                        old['position'], old['rotation'], joint)
        return result | {'joint_pose_covariance': joint, 'source_ids': self.source_ids,
                         'depth_rank_and_correspondences_conditioned_on': current['conditioned'],
                         'current_measured_ns': current['measured_ns'], 'stored_measured_ns': old['measured_ns']}


class RawRgbdMomentSensitivity(CorrelatedMomentSensitivity):
    """Actual paired RGBD/gyro registration and fusion; offline diagnostic only.

Unlike the conditional variant, this recomputes surface eligibility, matching,
registration and weak directions from each perturbed packet. All sensor-error
loadings remain explicit assumptions, not calibration. A changed accepted rank
invalidates this smooth local approximation and latches a fault. Unchanged rank
does not prove differentiability: correspondence switches and difference-step
dependence still require checking externally.
"""

    def __init__(self, source_ids, *, difference_step=1e-3):
        super().__init__(source_ids, difference_step=difference_step)
        self.models = [MomentDepthInertialState() for _ in range(1 + 2 * len(self.source_ids))]

    def _observe(self, policy, depth, fast, loadings):
        factors = self._loadings(policy, fast, loadings, shapes=RAW_SHAPES)
        if np.any(factors['depth_m'][~np.asarray(depth['valid'], dtype=bool)]):
            raise SensorContractError('missing depth has no measured range-error loading')
        now = policy['sensor_state']['decision_ns']
        outputs = []
        for index, model in enumerate(self.models):
            if index == 0:
                p, d, f = policy, depth, fast
            else:
                source, sign = (index - 1) // 2, (1. if index % 2 else -1.)
                scale = sign * self.step
                p, d, f = deepcopy(policy), deepcopy(depth), deepcopy(fast)
                for name in ('specific_force', 'gyro'):
                    p['sensor_state']['sensed'][name]['values'] = (
                        p['sensor_state']['sensed'][name]['values'] + scale * factors[name][..., source])
                f['values'] = f['values'] + scale * factors['fast_gyro'][..., source]
                # Preserve the declared range tensor dtype. Quantization can
                # make finite-difference results step-dependent; never silently
                # upgrade the sensor or claim this is a smooth depth model.
                d['depth_m'] = (d['depth_m'] + scale * factors['depth_m'][..., source]).astype(d['depth_m'].dtype)
            observed = model.observe(p, d, f, now_ns=now)
            if index and observed['fusion']['depth_rank'] != outputs[0][0]['depth_rank']:
                raise SensorContractError('raw sensor perturbation changed registration rank')
            outputs.append((observed['fusion'], np.asarray(
                observed['depth_state']['relative_orientation']['rotation_initial_body_from_current_body'])))
        result = self._summarize(outputs, now, conditioned=False)
        return result | {'raw_registration_recomputed': True, 'sensor_quantization_preserved': True}
