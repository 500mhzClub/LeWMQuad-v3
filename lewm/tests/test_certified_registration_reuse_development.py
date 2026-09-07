"""Exact-search comparisons, causal cache negatives and paired observer checks."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lewm.causal_sensor_state import SensorContractError
from lewm.certified_registration_reuse_development import (
    NearestQueryReference, surface_cloud_with_ids, register_translation_reuse, CertifiedRgbdMomentSensitivity,
    LeanCertifiedRgbdMomentSensitivity)
from lewm.correlated_moment_sensitivity_development import RawRgbdMomentSensitivity, RAW_SHAPES
from lewm.depth_relative_motion_development import register_translation, surface_cloud
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_depth_relative_motion_development import planes
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth


def compare_query(reference, target, queries, tids=None, qids=None):
    tids = np.arange(len(target)) if tids is None else tids
    qids = np.arange(len(queries)) if qids is None else qids
    tree = cKDTree(target)
    distance, index, stats = reference.query(target, queries, tids, qids, tree)
    expected_d, expected_i = tree.query(queries, distance_upper_bound=.2)
    np.testing.assert_array_equal(index, expected_i)
    np.testing.assert_allclose(distance, expected_d, atol=1e-14, rtol=0)
    assert stats['certified'] + stats['searched'] == len(queries)
    return stats


@pytest.mark.parametrize('scale', [0., 1e-6, 1e-3, .03, .3])
def test_nearest_reuse_matches_full_search_under_joint_point_perturbations(scale):
    rng = np.random.default_rng(38412)
    target = rng.uniform(-1, 1, (600, 3)); query = target[:400] + rng.normal(0, .001, (400, 3))
    reference = NearestQueryReference(target, query, np.arange(600), np.arange(400), cKDTree(target))
    changed_t = target + rng.uniform(-scale, scale, target.shape)
    changed_q = query + rng.uniform(-scale, scale, query.shape)
    stats = compare_query(reference, changed_t, changed_q)
    if scale <= 1e-6: assert stats['certified'] > 390
    if scale == .3: assert stats['searched'] > 390


def test_ties_and_changed_lineage_fall_back_to_exact_search():
    target = np.array([[0., 0, 0], [0., 0, 0], [1., 0, 0]])
    query = np.array([[0., 0, 0], [.5, 0, 0]])
    reference = NearestQueryReference(target, query, np.arange(3), np.arange(2), cKDTree(target))
    assert compare_query(reference, target, query)['searched'] == 2
    stats = compare_query(reference, target[::-1], query, np.arange(3)[::-1])
    assert stats['lineage_fallback'] and stats['searched'] == 2


@pytest.mark.parametrize('offset', [np.nextafter(.2, 0.), .2, np.nextafter(.2, 1.)])
def test_strict_search_radius_boundary_preserved(offset):
    target = np.array([[0., 0, 0], [10., 0, 0]])
    query = np.array([[.1, 0, 0]])
    reference = NearestQueryReference(target, query, np.arange(2), np.arange(1), cKDTree(target))
    assert compare_query(reference, target, np.array([[offset, 0, 0]]))['certified'] == 1


def test_cache_copies_arrays_and_rejects_stale_tree_and_duplicate_ids():
    target = np.array([[0., 0, 0], [1., 0, 0]])
    query = np.array([[.01, 0, 0]])
    reference = NearestQueryReference(target, query, np.arange(2), np.arange(1), cKDTree(target))
    target[0, 0] = .02; query[0, 0] = .03
    assert reference.target[0, 0] == 0. and reference.queries[0, 0] == .01
    with pytest.raises(ValueError): reference.target[0, 0] = 1.
    with pytest.raises(SensorContractError): reference.query(target, query, np.arange(2), np.arange(1), cKDTree(reference.target))
    with pytest.raises(SensorContractError): NearestQueryReference(target, query, [0, 0], [0], cKDTree(target))
    triple = np.vstack((target, [[2., 0, 0]]))
    with pytest.raises(SensorContractError):
        NearestQueryReference(triple, query, np.array([1, 0, 1], dtype=np.uint64), [0], cKDTree(triple))


@pytest.mark.parametrize('axes', [[0, 1, 2], [1, 2], [0]])
@pytest.mark.parametrize('angle', [0., .0001, .02])
def test_full_registration_matches_frozen_equations_in_observed_and_weak_cases(axes, angle):
    previous = planes(axes); delta = np.array([.023, -.011, .005])
    current = (previous[0] - delta, previous[1].copy())
    ids = np.arange(len(previous[0]))
    nominal, trace, _ = register_translation_reuse(previous, current, np.eye(3), ids, ids)
    assert nominal == register_translation(previous, current, np.eye(3))
    rotation = rotation_increment([.3 * angle, -.2 * angle, angle])
    result, unused, stats = register_translation_reuse(previous, current, rotation, ids, ids, references=trace)
    assert result == register_translation(previous, current, rotation)
    assert not unused
    assert stats['certified_queries'] + stats['searched_queries'] > 0


def test_new_surface_lineage_and_normal_changes_do_not_freeze_nominal_decision():
    previous = planes([0, 1, 2]); current = (previous[0] - [.02, 0, 0], previous[1].copy())
    ids = np.arange(len(previous[0]))
    _, trace, _ = register_translation_reuse(previous, current, np.eye(3), ids, ids)
    # Remove a surface, changing both point lineage and registration rank.
    keep = np.arange(len(ids)) < 1250
    p = (previous[0][keep], previous[1][keep]); c = (current[0][keep], current[1][keep])
    result, _, stats = register_translation_reuse(p, c, np.eye(3), ids[keep], ids[keep], references=trace)
    assert result == register_translation(p, c, np.eye(3))
    assert result['rank'] == 2 and stats['lineage_fallback_iterations'] > 0


def test_huber_reweighting_is_recomputed_for_changed_point_ranges():
    previous = planes([0, 1, 2]); current = (previous[0] - [.02, .005, 0], previous[1].copy())
    ids = np.arange(len(previous[0]))
    _, trace, _ = register_translation_reuse(previous, current, np.eye(3), ids, ids)
    changed = current[0].copy(); changed[::13] += current[1][::13] * .025
    perturbed = (changed, current[1])
    result, _, _ = register_translation_reuse(previous, perturbed, np.eye(3), ids, ids, references=trace)
    assert result == register_translation(previous, perturbed, np.eye(3))


@pytest.mark.parametrize('hole', [False, True])
def test_cloud_and_pixel_lineage_preserve_frozen_surface_selection(hole):
    _, p, _, _ = next(stream(1)); depth = room_depth(p)
    if hole:
        depth['valid'][250:300, 250:300] = False; depth['depth_m'][250:300, 250:300] = 0.
    now = p['sensor_state']['decision_ns']
    cloud, ids = surface_cloud_with_ids(depth, p, now_ns=now)
    original = surface_cloud(depth, p, now_ns=now)
    for actual, expected in zip(cloud, original, strict=True): np.testing.assert_array_equal(actual, expected)
    assert len(ids) == len(cloud[0]) and len(np.unique(ids)) == len(ids)


@pytest.mark.parametrize('factory', [CertifiedRgbdMomentSensitivity, LeanCertifiedRgbdMomentSensitivity])
def test_raw_paired_outputs_match_uncached_reference_with_shared_and_changed_depth(factory):
    names = ['gyro_bias', 'depth_scale']
    fast = factory(names, difference_step=.01)
    reference = RawRgbdMomentSensitivity(names, difference_step=.01)
    total_certified = 0
    for tick, p, f, _ in stream(5):
        depth = room_depth(p)
        loading = {name: np.zeros((*shape, 2)) for name, shape in RAW_SHAPES.items()}
        loading['gyro'][:, 2, 0] = .001; loading['fast_gyro'][:, 2, 0] = .001
        loading['depth_m'][..., 1] = .002 * depth['depth_m']
        a = fast.observe(p, depth, f, loading); b = reference.observe(p, depth, f, loading)
        for key, value in b.items():
            if isinstance(value, np.ndarray): np.testing.assert_array_equal(a[key], value)
            else: assert a[key] == value
        assert a['shared_surface_hits'] == (2 if factory is CertifiedRgbdMomentSensitivity else 0)
        assert a['shared_cloud_hits'] == 2
        assert 'surface_segments' in fast.models[0].depth.surfaces.snapshot()[-1]
        total_certified += sum(row.get('certified_queries', 0) for row in a['registration_query_accounting'])
    assert total_certified > 1000


@pytest.mark.parametrize('fault', ['duplicate', 'invalid_depth', 'future_fast', 'shared_gyro_rewrite'])
@pytest.mark.parametrize('factory', [CertifiedRgbdMomentSensitivity, LeanCertifiedRgbdMomentSensitivity])
def test_shared_geometry_cannot_bypass_causal_or_sensor_guards(fault, factory):
    model = factory(['zero'])
    items = stream(2); _, p, f, _ = next(items); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    model.observe(p, depth, f, loading)
    if fault != 'duplicate': _, p, f, _ = next(items); depth = room_depth(p)
    if fault == 'invalid_depth': depth['depth_m'][0, 0] = np.nan
    elif fault == 'future_fast': f['available_ns'][-1] += 1
    elif fault == 'shared_gyro_rewrite': loading['fast_gyro'][-1, 0, 0] = .001
    with pytest.raises(SensorContractError): model.observe(p, depth, f, loading)
    assert model.failed
