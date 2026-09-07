from dataclasses import replace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.factored_configuration_evidence_development import ground_summary, query_factored_configuration
from lewm.tests.test_observed_setup_configuration_development import ready
from lewm.relative_gyro_turn_development import rotation_increment
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same


def witness(t, low, high, covered=True, vertex_low=None, vertex_high=None):
    return dict(measured_ns=t, floor_coverage=covered,
        gap=dict(minimum_gap_lower_m=low, minimum_gap_upper_m=high),
        plane_vertex_residual_lower_m=[low if vertex_low is None else vertex_low]*8,
        plane_vertex_residual_upper_m=[high if vertex_high is None else vertex_high]*8)


def test_wide_historical_intersection_is_not_a_contradiction_of_observed_separation():
    result = ground_summary('FL_calflower1:0', [witness(1,-.04,.09), witness(2,.026,.029)])
    assert result['physical_floor_separation_observed']
    assert result['possible_intersection_sources'] == [1]
    assert result['observed_separation_sources'] == [2]
    assert not result['incompatible_covered_plane_pairs'] and not result['contact_permitted']
    assert not result['common_surface_identity_established']


def test_covered_penetration_and_incompatible_planes_are_retained():
    result = ground_summary('FL_calflower1:0', [witness(1,-.04,-.02), witness(2,.026,.029)])
    assert result['observed_penetration_sources'] == [1]
    assert result['incompatible_covered_plane_pairs'] == [[1,2]]
    assert not result['physical_floor_separation_observed']
    # Equal minimum-gap intervals do not conceal a differently tilted plane.
    result = ground_summary('base:0', [witness(1,.02,.03,vertex_low=.05,vertex_high=.06),
                                      witness(2,.02,.03,vertex_low=.08,vertex_high=.09)])
    assert result['incompatible_covered_plane_pairs'] == [[1,2]]


def test_partial_floor_views_are_not_combined_into_coverage_or_penetration():
    result = ground_summary('base:0', [witness(1,.02,.03,False), witness(2,-.04,-.02,False)])
    assert result['status'] == 'UNKNOWN_FLOOR_COVERAGE'
    assert not result['physical_floor_separation_observed'] and not result['observed_penetration_sources']


def test_only_exact_foot_shapes_are_contact_candidates_never_permitted():
    for sid in ('FL_foot:0','FL_calflower:0','FL_calf:0'):
        result = ground_summary(sid,[witness(1,-.01,.01)])
        assert result['observed_foot_contact_candidate'] == (sid == 'FL_foot:0')
        assert not result['contact_permitted']


def query(owner, now, offset=(1.,0.,0.), error=0., backend='compiled', through=None):
    return query_factored_configuration(owner, offset, np.eye(3), owner._memory._joints, error,
        now_ns=now, through_ns=now if through is None else through, backend=backend)


@pytest.mark.parametrize('offset', [(0.,0.,0.), (1.,0.,0.), (2.2,0.,0.), (-1.3,0.,0.), (1.3,0.,-.10)])
def test_complete_factored_output_matches_reference(offset):
    owner, now = ready()
    same(query(owner,now,offset), query(owner,now,offset,backend='reference'))


def test_nonfloor_clearance_does_not_grant_own_body_ground_support():
    owner, now = ready(); result = query(owner,now,(0.,0.,0.))
    assert result['all_primitives_conditionally_nonfloor_clear']
    assert not result['all_primitives_observed_separated']
    assert not result['ground_support_permission'] and not result['navigation_action_permitted']


def test_measured_wall_overrides_false_large_setup_even_with_floor_mask():
    owner, now = ready(3.); result = query(owner,now,(2.2,0.,0.))
    base = next(r for r in result['primitives'] if r['shape_id']=='base:0')
    assert base['nonfloor_conflict_sources']
    assert not base['conditional_nonfloor_clearance'] and not base['supplied_nonfloor_clearance_used']


def test_ground_penetration_stays_rejected_even_when_nonfloor_channel_clears():
    owner, now = ready(); result = query(owner,now,(1.3,0.,-.1))
    rows = [r for r in result['primitives'] if r['ground']['observed_penetration_sources']]
    assert rows and any(r['conditional_nonfloor_clearance'] for r in rows)
    assert all(not r['conditional_observed_separation'] and not r['contact_candidate_with_nonfloor_clearance'] for r in rows)


def test_absent_plane_keeps_near_returns_and_never_invents_ground():
    owner, now = ready()
    for t,h in list(owner._memory._hypotheses.items()):
        owner._memory._hypotheses[t] = replace(h,cell_rc=None)
    result = query(owner,now,(1.3,0.,-.1))
    assert any(r['nonfloor_conflict_sources'] for r in result['primitives'])
    assert all(r['ground']['status']=='UNKNOWN_FLOOR_COVERAGE' for r in result['primitives'])


def test_missing_depth_and_expiry_leave_residual_unknown():
    owner, now = ready(missing_central_pixel=True)
    result = query(owner,now,through=owner._region.valid_until_ns+1)
    base = next(r for r in result['primitives'] if r['shape_id']=='base:0')
    assert not base['conditional_nonfloor_clearance'] and not base['supplied_nonfloor_clearance_used']
    assert not all(base['residual_nonfloor_clearance_sources'])


def test_virtual_corner_error_uses_coordinate_support_bound_and_own_transport():
    owner, now=ready(); rotation=rotation_increment([0.,0.,.4]); error=.01
    result=query_factored_configuration(owner,[1.,0.,0.],rotation,owner._memory._joints,error,
        now_ns=now,through_ns=now)
    for row in result['primitives']:
        w=next(w for w in row['ground_witnesses'] if w['measured_ns']==now)
        if w['gap'] is None: continue
        half_width=(np.asarray(w['plane_vertex_residual_upper_m'])-w['plane_vertex_residual_lower_m'])/2
        assert np.all(half_width >= .001+1.002*np.sqrt(3)*error)
    # Actual physical queries still use their original endpoint error, not
    # the virtual-vertex coordinate-support multiplier.
    assert all(next(w for w in r['ground_witnesses'] if w['measured_ns']==now)['gap']['point_error_m']==error
               for r in result['primitives'])


@pytest.mark.parametrize('fault', ['stale','raw_binding','plane_binding','nonfinite','negative_error'])
def test_unbound_or_invalid_queries_fail(fault):
    owner, now = ready(); offset=(1.,0.,0.); error=0.
    if fault=='stale': now-=1
    if fault=='raw_binding':
        stored = dict(owner._memory._rays.frames[0])
        stored['evidence'] = dict(stored['evidence']) | {'depth': np.zeros((480,640),np.float32)}
        owner._memory._rays.frames[0] = stored
    if fault=='plane_binding':
        t=next(iter(owner._memory._hypotheses)); h=owner._memory._hypotheses[t]
        owner._memory._hypotheses[t]=replace(h,depth_sha256='a'*64)
    if fault=='nonfinite': offset=(float('nan'),0.,0.)
    if fault=='negative_error': error=-.1
    with pytest.raises(SensorContractError): query(owner,now,offset,error)
