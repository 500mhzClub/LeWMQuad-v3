"""Shared raw-error diagnostic through the actual complementary RGB-D model.

Finite perturbation pairs are not calibrated covariance or bounds. The nominal
model is the deployed development RGBDInertialState, not the older depth-only
integrator. No inferred uncertainty is supplied to navigation.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_moment_sensitivity_development import CorrelatedMomentSensitivity, RAW_SHAPES
from lewm.rgbd_inertial_fusion_development import RGBDInertialState, PointFusionHypotheses
from lewm.setup_velocity_prior_development import SetupVelocityPrior

SHAPES = RAW_SHAPES | {'initial_velocity': (3,)}


def signature(observed):
    motion = observed['point_state']['motion']; fusion = observed['fusion']
    return dict(depth_rank=fusion['depth_rank'], fusion_kind=fusion['kind'], point_status=motion['status'],
        **{k: motion.get(k) for k in ('lifted_matches', 'inliers', 'previous_grid_cells', 'current_grid_cells')})


class RawComplementaryRGBDSensitivity(CorrelatedMomentSensitivity):
    def __init__(self, source_ids, *, prior, hypotheses, difference_step=1e-3):
        if not isinstance(prior, SetupVelocityPrior) or not isinstance(hypotheses, PointFusionHypotheses):
            raise SensorContractError('explicit original setup prior and RGBD hypotheses required')
        super().__init__(source_ids, difference_step=difference_step)
        self.prior, self.hypotheses = prior, hypotheses
        self.models = None; self.prior_loadings = None
        self.current_packets = None; self.current_observations = None
        self.branch_changes = []

    def _observe(self, policy, depth, fast, loadings):
        factors = self._loadings(policy, fast, loadings, shapes=SHAPES)
        if np.any(factors['depth_m'][~depth['valid']]):
            raise SensorContractError('unknown depth cannot acquire a range-error loading')
        for name in ('specific_force', 'gyro'):
            if np.any(factors[name][~policy['sensor_state']['sensed'][name]['valid']]):
                raise SensorContractError('unknown body sensors cannot acquire a loading')
        if np.any(factors['fast_gyro'][~fast['valid']]):
            raise SensorContractError('unknown gyro samples cannot acquire a loading')
        if self.prior_loadings is not None and not np.array_equal(factors['initial_velocity'], self.prior_loadings):
            raise SensorContractError('same initial prior error cannot be rewritten')
        if self.models is None:
            self.prior_loadings = factors['initial_velocity'].copy()
            priors = [self.prior]
            for source in range(len(self.source_ids)):
                for sign in (1., -1.):
                    mean = np.asarray(self.prior.mean_initial_body_m_s)+sign*self.step*self.prior_loadings[:, source]
                    priors.append(replace(self.prior, mean_initial_body_m_s=tuple(float(x) for x in mean)))
            self.models = [RGBDInertialState(prior=p, hypotheses=self.hypotheses) for p in priors]
        now = policy['sensor_state']['decision_ns']; outputs = []; packets = []; observations = []
        for index, model in enumerate(self.models):
            if index == 0:
                p, d, f = policy, depth, fast
            else:
                source, sign = (index-1)//2, (1. if index % 2 else -1.)
                p, d, f = deepcopy(policy), deepcopy(depth), deepcopy(fast)
                scale = sign*self.step
                for name in ('specific_force', 'gyro'):
                    p['sensor_state']['sensed'][name]['values'] += scale*factors[name][..., source]
                f['values'] += scale*factors['fast_gyro'][..., source]
                d['depth_m'] = (d['depth_m']+scale*factors['depth_m'][..., source]).astype(depth['depth_m'].dtype)
                # Preserve quantization and the actual validity contract.
                # A pair crossing a validity boundary is a retained fault,
                # not an invented smooth derivative through missing data.
            try:
                observed = model.observe(p, d, f, now_ns=now)
                if not observed['fusion']['usable_under_declared_proxy_budget']:
                    raise SensorContractError('paired RGBD model exceeded its original pose budget')
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
                raise SensorContractError(f'raw RGBD perturbation member {index} failed at {now}') from error
            R = np.asarray(observed['depth_state']['relative_orientation']['rotation_initial_body_from_current_body'])
            outputs.append((observed['fusion'], R)); observations.append(observed)
            packets.append(dict(depth=d['depth_m'].copy(), valid=d['valid'].copy(),
                position=model.integrator.position.copy(), rotation=R.copy(), up=R.T@(model.integrator.gravity/9.81)))
        signatures = [signature(o) for o in observations]
        changes = [dict(member=i, nominal=signatures[0], perturbed=s) for i, s in enumerate(signatures[1:], 1) if s != signatures[0]]
        if changes: self.branch_changes.append(dict(measured_ns=now, differences=changes))
        self.current_packets, self.current_observations = packets, observations
        result = self._summarize(outputs, now, conditioned=False)
        return result | dict(nominal_point_state=deepcopy(observations[0]['point_state']),
            nominal_raw_depth_state=deepcopy(observations[0]['depth_state']),
            raw_plane_and_point_registration_recomputed=True, original_rgbd_nominal_model=True,
            sensor_quantization_preserved=True, initial_velocity_prior_source_propagated=True,
            categorical_branch_changes=changes, categorical_change_seen=bool(self.branch_changes),
            exact_correspondence_identity_observable=False,
            source_covariance_assumption='unit-independent named sources; temporal/channel sharing by identical source ID',
            coverage_calibrated=False, motion_permission=False)
