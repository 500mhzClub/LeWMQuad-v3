"""Prelabelled analytical clearance boundaries; no native physics or audit rows."""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm.decision_headroom_reference_development import ReferenceGeometry, reference_cost


def main():
    source = Path('docs/go2_decision_headroom_clearance_boundary_cases_v2_2026-09-23.json')
    spec = json.loads(source.read_text())
    original = json.loads(Path('docs/go2_decision_headroom_reference_sanity_v1_2026-09-23.json').read_text())
    parameters = original['draft_cost_parameters']
    rows = []
    for case in spec['cases']:
        distance = case['centre_distance_um'] / 1e6
        xy = np.tile([-distance, 0.], (401, 1))
        geometry = ReferenceGeometry([dict(center=[.5, 0.], size=[1., 10.], yaw=0.)],
            [[-3., -6.], [3., 6.]], [-1., 0.], radius_m=.46, clearance_m=.005, resolution_m=.02)
        trace = dict(xy=xy, offset_ns=np.arange(401)*2_000_000, yaw=np.full(401,np.pi),
            velocity_xy=np.zeros((401,2)), yaw_rate=np.zeros(401),
            disallowed_contact=np.full(401,case['disallowed_contact']))
        result = reference_cost(geometry, trace, arrival_settling=False, parameters=parameters)
        # Exactly the frozen full-clear predicates. Recovery needs a path and
        # is outside these stationary boundary fixtures; no controller changes.
        centre = float(geometry.footprint_clearance(xy[:1])[0]) + .46
        actual = dict(legacy_reference_acceptable=result['acceptable'],
            controller_nominal_full_clear=centre > .45+1e-12,
            controller_translation_full_reserve_clear=centre > .48+1e-12,
            common_operating_margin=centre >= .48-spec['numerical_clearance_tolerance_m'])
        errors = {k:dict(expected=v, actual=actual[k]) for k,v in case['expected'].items()
                  if k in actual and actual[k] != v}
        rows.append(dict(id=case['id'], expected=case['expected'], actual=actual,
            measured_centre_distance_m=centre, reference=result, failures=errors,
            numerical_boundary=abs(centre-.48)<=spec['numerical_clearance_tolerance_m']))
    output = dict(status='PASS' if not any(r['failures'] for r in rows) else 'FAIL',
        cases=len(rows), failures=sum(bool(r['failures']) for r in rows), rows=rows,
        expectations_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_band_is_not_physical_uncertainty=True,
        native_physics_steps=0, model_calls=0, audit_regrets=0,
        original_failure_and_erratum_preserved=True)
    path=Path('docs/go2_decision_headroom_clearance_boundary_result_v2_2026-09-23.json')
    with path.open('x') as f:
        json.dump(output,f,indent=2);f.write('\n')
    print(output['status'],output['cases'],output['failures'])


if __name__ == '__main__':
    main()
