"""Recorded development wiring check for new factored configuration/trajectory APIs."""
import json

import numpy as np

from lewm.articulated_trajectory_evidence_development import command_baseline_trajectory, evaluate_trajectory
from lewm.factored_configuration_evidence_development import query_factored_configuration
from scripts import diagnose_go2_configuration_ground_veto_development as prior

p = prior.previous
SOURCES = prior.SOURCES + (
    'lewm/factored_configuration_evidence_development.py',
    'lewm/articulated_trajectory_evidence_development.py',
    'lewm/tests/test_factored_configuration_evidence_development.py',
    'lewm/tests/test_articulated_trajectory_evidence_development.py',
    'scripts/probe_go2_factored_trajectory_evidence_development.py',
    'docs/go2_factored_trajectory_recorded_diagnostic_protocol_2026-09-06.md',
)


def summary(result):
    return dict(conditionally_nonfloor_clear=[r['shape_id'] for r in result['primitives'] if r['conditional_nonfloor_clearance']],
        observed_separated=[r['shape_id'] for r in result['primitives'] if r['conditional_observed_separation']],
        nonfloor_conflicts={r['shape_id']:r['nonfloor_conflict_sources'] for r in result['primitives'] if r['nonfloor_conflict_sources']},
        required_residual_boxes=sum(len(r['residual_nonfloor_clearance_sources']) for r in result['primitives']),
        cleared_residual_boxes=sum(bool(s) for r in result['primitives'] for s in r['residual_nonfloor_clearance_sources']),
        ground={r['shape_id']:r['ground'] for r in result['primitives']},
        ground_support_permission=result['ground_support_permission'], navigation_action_permitted=result['navigation_action_permitted'])


def main():
    old_report = 'docs/go2_configuration_ground_veto_diagnostic_result_2026-09-06.json'
    bindings = {str((p.OUTPUT/name).relative_to(p.ROOT)):h for name,h in p.IDENTITIES.items()}
    bindings[old_report] = 'bd2d9b65b7517f8b12c0906f23346e5824c735941e44de7ee9d76412c8eae449'
    p.verify_bindings(bindings)
    predecessor = json.loads((p.ROOT/old_report).read_text())
    launch = json.loads((p.OUTPUT/'launch.json').read_text()); result = json.loads((p.OUTPUT/'result.json').read_text())
    bindings |= launch['source_sha256'] | launch['input_sha256'] | predecessor['source_sha256'] | {
        str((p.OUTPUT/name).relative_to(p.ROOT)):h for name,h in result['artifact_sha256'].items()}
    sources = {name:p.digest(p.ROOT/name) for name in SOURCES}
    if any(name in bindings and bindings[name] != h for name,h in sources.items()):
        raise ValueError('predecessor source identity changed')
    def verify():
        p.verify_bindings(bindings | sources); p.verify_native_bindings(launch['native_sha256'])
        p.verify_extensions(launch['native_geometry_sha256'])
    verify()
    velocity, region = p.make_priors(1_500_000_000,launch['source_sha256'][p.PROTOCOL])
    admission=json.loads((p.OUTPUT/'startup_admission.json').read_text()); admission['identity']=tuple(admission['identity'])
    owner=p.ContinuousStartupHandoff(p.ArticulatedCollisionGeometry(p.URDF),velocity_prior=velocity,region_prior=region,admission=admission)
    decisions=json.loads((p.OUTPUT/'startup_decisions.json').read_text())
    relatives=json.loads((p.OUTPUT/'relative_state_observations.json').read_text())
    for frame in range(result['rgbd_frames']):
        policy,depth=p.load_rgbd_observation(p.OUTPUT,frame); now=policy['sensor_state']['decision_ns']
        row=owner.observe(policy,depth,p.load_fast_packet(p.OUTPUT,frame),now_ns=now)
        if row['terminal']: raise ValueError('saved handoff failed')
        p.json_same(owner.relative_observation(now_ns=now),relatives[frame]['observer'])
        if frame<len(decisions): p.json_same(row['startup_decision'],decisions[frame]['decision'])
        elif row['startup_decision'] is not None: raise ValueError('terminal startup called again')
    tangent=p.gravity_basis(owner._memory._rays.latest_frame['evidence']['up'])[:,0]
    configs=[]
    for distance in (.75,1.):
        args=(owner,distance*tangent,np.eye(3),owner._memory._joints,0.)
        old=p.query_configuration(*args,now_ns=now,through_ns=3_300_000_000,reference='current_body')
        actual=query_factored_configuration(*args,now_ns=now,through_ns=3_300_000_000)
        reference=query_factored_configuration(*args,now_ns=now,through_ns=3_300_000_000,backend='reference')
        p.same(actual,reference)
        configs.append(dict(tangent_offset_m=distance,new_factored=summary(actual),
            original_conditional_clear_count=sum(s['conditional_nonfloor_clearance'] for s in old['primitives']),
            original_obstacle_veto=[s['shape_id'] for s in old['primitives'] if s['obstacle_veto_sources']],
            compiled_reference_all_fields_exact=True))
    # Explicit unvalidated assumptions for interface exercise, NOT fitted
    # limits and NOT a motion request. The recording ends at this anchor.
    predictions=[]
    for name, commands in (
        ('stop',[[0.,0.,0.]]*4),
        ('forward_then_zero',[[.08,0.,0.]]*2+[[0.,0.,0.]]*2),
        ('turn_then_zero',[[0.,0.,.35]]*2+[[0.,0.,0.]]*2)):
        trajectory=command_baseline_trajectory(owner,policy,commands,now_ns=now)
        kwargs=dict(point_errors_m=[0.,.05,.05,.05,.05],physical_point_speed_bounds_m_s=[2.]*4,now_ns=now)
        evidence=evaluate_trajectory(owner,trajectory,**kwargs)
        reference=evaluate_trajectory(owner,trajectory,backend='reference',**kwargs)
        p.same(evidence,reference)
        queries=evidence.pop('configuration_queries')
        predictions.append(dict(name=name,prediction=trajectory,evidence=evidence,
            nodes=[dict(offset_ns=t,**summary(q)) for t,q in zip(trajectory['offsets_ns'],queries,strict=True)],
            compiled_reference_all_fields_exact=True,actual_future_samples_available=False))
    verify()
    print(json.dumps(dict(status='FACTORED_TRAJECTORY_RECORDED_INTERFACE_CHECK_COMPLETE',source_sha256=sources,
        configurations=configs,trajectories=predictions,relative_observations_exact=len(relatives),
        startup_decisions_exact=len(decisions),scope='new development semantics and nominal future interface; no physics or error validation',
        navigation_qualified=False),allow_nan=False),flush=True)


if __name__=='__main__': main()
