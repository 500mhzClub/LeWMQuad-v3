"""Read-only per-observation attribution; never changes a clearance decision."""
import json
from itertools import product

import numpy as np

from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.primitive_obstacle_memory_development import non_floor_box_evidence
from lewm.uncertain_ray_memory_development import transport_radius
from scripts import probe_go2_gravity_tangent_configuration_development as previous

SOURCES = previous.SOURCES + (
    'scripts/diagnose_go2_configuration_ground_veto_development.py',
    'lewm/tests/test_configuration_ground_veto_diagnostic_development.py',
    'docs/go2_configuration_ground_veto_diagnostic_protocol_2026-09-06.md',
)
_CORNERS = np.asarray(list(product((0, 1), repeat=3)), bool)


def trace_configuration(owner, offset, *, now_ns, through_ns):
    """Reconstruct ALL whole-query veto witnesses, retaining old query results.

    The unconditional plane-family branch is a diagnostic intervention only.
    It cannot grant support, contact, clearance or action permission. Missing
    coverage and uncertain physical intersection remain unresolved even when
    that branch removes a near-return conflict.
    """
    original = previous.query_configuration(owner, offset, np.eye(3), owner._memory._joints, 0.,
        now_ns=now_ns, through_ns=through_ns, reference='current_body', backend='compiled')
    reference = previous.query_configuration(owner, offset, np.eye(3), owner._memory._joints, 0.,
        now_ns=now_ns, through_ns=through_ns, reference='current_body', backend='reference')
    previous.same(original, reference)
    memory, rays = owner._memory, owner._memory._rays
    geometry, q = memory._geometry, memory._joints
    p = rays.position+rays.rotation@np.asarray(offset)
    R = rays.rotation
    local_shapes = geometry.supports(q, np.eye(3))['shapes']
    frames = list(rays.frames)
    if frames[-1]['measured_ns'] != now_ns:
        frames.append(rays.latest_frame)
    witnesses = []
    for stored in frames:
        stamp = stored['measured_ns']; frame = memory._prepared[stamp]
        cell = memory._hypotheses[stamp].cell_for(frame)
        rotation = np.asarray(stored['rotation']).T@R
        translation = (p-stored['position'])@stored['rotation']
        shapes = geometry.supports(q, rotation)['shapes']
        allowances = [float(transport_radius(np.where(_CORNERS, s['upper'], s['lower'])+offset,
            rays.latest_frame, stored).max()) for s in local_shapes]
        errors = {s['shape_id']: e for s,e in zip(shapes, allowances, strict=True)}
        low = np.asarray([s['lower']+translation-(e+.04) for s,e in zip(shapes, allowances, strict=True)])
        high = np.asarray([s['upper']+translation+(e+.04) for s,e in zip(shapes, allowances, strict=True)])
        floor = None if cell is None else frame.query(geometry, q, cell,
            rotation_observation_from_body=rotation, translation_observation_from_body=translation,
            point_error_by_shape=errors, floor_backend='cached', **memory._errors)
        plane = None if floor is None else (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
        branches = []
        for supplied_plane in (None, plane):
            kwargs = dict(plane=supplied_plane, normal_error=memory._errors['normal_error'],
                plane_offset_error=memory._errors['plane_offset_error'], range_error_m=memory._range_error)
            result = non_floor_box_evidence(frame, low, high, backend='compiled', **kwargs)
            previous.same(result, non_floor_box_evidence(frame, low, high, backend='reference', **kwargs))
            branches.append(result)
        unmasked, masked = branches
        for i, shape in enumerate(shapes):
            sid = shape['shape_id']
            gap = None if floor is None else floor['gap_bounds']['primitives'][i]
            if gap is not None and gap['shape_id'] != sid:
                raise ValueError('physical primitive identity mismatch')
            covered = False if floor is None else floor['floor_coverage'][sid]
            allowed = gap is not None and (gap['minimum_gap_lower_m'] > 0 or
                (sid in FOOT_SHAPES and gap['minimum_gap_lower_m'] <= 0 <= gap['minimum_gap_upper_m']))
            chosen = masked if allowed else unmasked
            conflict = bool(chosen['non_floor_conflict'][i])
            witnesses.append(dict(shape_id=sid, measured_ns=stamp, depth_sha256=frame.depth_sha256,
                plane_cell_rc=cell, plane_available=plane is not None, gap=gap,
                stored_position_scale_m=float(stored['position_scale_m']),
                current_position_scale_m=float(rays.latest_frame['position_scale_m']),
                relative_point_allowance_m=allowances[i], floor_coverage=bool(covered),
                original_plane_exemption=bool(allowed), original_obstacle_veto=conflict,
                original_penetration_veto=bool(covered and gap['minimum_gap_upper_m'] < 0) if gap else False,
                unmasked_conflict=bool(unmasked['non_floor_conflict'][i]),
                plane_masked_conflict=bool(masked['non_floor_conflict'][i]),
                plane_masked_complete_box_clearance=bool(masked['non_floor_clearance'][i]),
                plane_masked_floor_cells=int(masked['floor_family_pixels'][i]),
                conflict_removed_only_by_plane_branch=bool(plane is not None and conflict
                    and not allowed and not masked['non_floor_conflict'][i]),
                ground_support_permission=False, navigation_action_permitted=False))
    for row in original['primitives']:
        matching = [w for w in witnesses if w['shape_id'] == row['shape_id']]
        for original_key, witness_key in (('obstacle_veto_sources', 'original_obstacle_veto'),
                                          ('floor_penetration_sources', 'original_penetration_veto')):
            if row[original_key] != [w['measured_ns'] for w in matching if w[witness_key]]:
                raise ValueError('diagnostic does not reproduce original veto witnesses')
    # The diagnostic must not mutate the original consumer's result/state.
    after = previous.query_configuration(owner, offset, np.eye(3), q, 0., now_ns=now_ns,
        through_ns=through_ns, reference='current_body', backend='compiled')
    previous.same(original, after)
    return dict(offset_current_body_m=np.asarray(offset).tolist(), original_query=original,
        witnesses=witnesses, all_veto_witnesses_reproduced=True,
        compiled_reference_all_fields_exact=True, original_query_unchanged=True,
        ground_support_permission=False, navigation_action_permitted=False)


def main():
    bindings = {str((previous.OUTPUT/p).relative_to(previous.ROOT)):h for p,h in previous.IDENTITIES.items()}
    previous.verify_bindings(bindings)
    launch = json.loads((previous.OUTPUT/'launch.json').read_text())
    result = json.loads((previous.OUTPUT/'result.json').read_text())
    bindings |= launch['source_sha256'] | launch['input_sha256'] | {
        str((previous.OUTPUT/p).relative_to(previous.ROOT)):h for p,h in result['artifact_sha256'].items()}
    sources = {p:previous.digest(previous.ROOT/p) for p in SOURCES}
    def verify():
        previous.verify_bindings(bindings | sources)
        previous.verify_native_bindings(launch['native_sha256'])
        previous.verify_extensions(launch['native_geometry_sha256'])
    verify()
    velocity, region = previous.make_priors(1_500_000_000, launch['source_sha256'][previous.PROTOCOL])
    admission = json.loads((previous.OUTPUT/'startup_admission.json').read_text())
    admission['identity'] = tuple(admission['identity'])
    owner = previous.ContinuousStartupHandoff(previous.ArticulatedCollisionGeometry(previous.URDF),
        velocity_prior=velocity, region_prior=region, admission=admission)
    decisions = json.loads((previous.OUTPUT/'startup_decisions.json').read_text())
    relatives = json.loads((previous.OUTPUT/'relative_state_observations.json').read_text())
    for frame in range(result['rgbd_frames']):
        p,d = previous.load_rgbd_observation(previous.OUTPUT, frame); now = p['sensor_state']['decision_ns']
        row = owner.observe(p,d,previous.load_fast_packet(previous.OUTPUT,frame),now_ns=now)
        if row['terminal']: raise ValueError('saved handoff replay failed')
        previous.json_same(owner.relative_observation(now_ns=now), relatives[frame]['observer'])
        if frame < len(decisions): previous.json_same(row['startup_decision'], decisions[frame]['decision'])
        elif row['startup_decision'] is not None: raise ValueError('terminal startup controller restarted')
    tangent = previous.gravity_basis(owner._memory._rays.latest_frame['evidence']['up'])[:,0]
    rows = [trace_configuration(owner, distance*tangent, now_ns=now, through_ns=3_300_000_000)
            for distance in (.75, 1.)]
    # Keep stdout bounded; verification above still covers every primitive,
    # observation and original-query field. Report all vetoes and the four
    # investigated shapes in every view, including their non-veto witnesses.
    target_ids = {f'{leg}_{part}:0' for leg in ('FL', 'FR') for part in ('calflower', 'calflower1')}
    for row in rows:
        original = row.pop('original_query')
        row['original_summary'] = {key: [s['shape_id'] for s in original['primitives'] if s[key]]
            for key in ('conditional_nonfloor_clearance', 'obstacle_veto_sources', 'floor_penetration_sources')}
        row['total_witnesses_checked'] = len(row['witnesses'])
        row['witnesses'] = [w for w in row['witnesses'] if w['shape_id'] in target_ids
            or w['original_obstacle_veto'] or w['original_penetration_veto']]
    verify()
    print(json.dumps(dict(status='RECORDED_GROUND_VETO_ATTRIBUTION_COMPLETE', source_sha256=sources,
        configurations=rows, startup_decisions_exact=len(decisions), relative_observations_exact=len(relatives),
        scope='read-only exploratory attribution, no physics, no clearance-policy changes',
        navigation_qualified=False), allow_nan=False), flush=True)


if __name__ == '__main__': main()
