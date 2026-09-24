"""Correct native padding without relaxing physical or perception criteria."""
import ast
import inspect

import pytest

from lewm.marker_beacon_scene_development import trials
from lewm.tests.test_rgb_marker_beacon_development import static_rows
from scripts import audit_go2_marker_beacon_development_v1 as old
from scripts import audit_go2_marker_beacon_development_v2 as new


def padded_rows(spec):
    rows = static_rows(spec)
    for row in rows:
        row['native_box_size'] = [*row['native_box_size'], 0., 0., 0., 0.]
    return rows


def test_all_full_trial_checks_are_source_identical():
    assert ast.dump(ast.parse(inspect.getsource(old.audit_trial))) == ast.dump(ast.parse(inspect.getsource(new.audit_trial)))


def test_only_static_dimension_encoding_check_changes():
    source = inspect.getsource(new.check_static_objects)
    correction = """        native = np.asarray(row['native_box_size'], dtype=float)
        check(native.shape == (7,) and np.isfinite(native).all() and np.array_equal(native[3:], np.zeros(4)),
              'exact native box encoding: three extents plus four zero padding values')
        check(np.allclose(native[:3], box['size_xyz'], atol=1e-7, rtol=0), 'actual collision dimensions')"""
    original = "        check(np.allclose(row['native_box_size'], box['size_xyz'], atol=1e-7, rtol=0), 'actual collision dimensions')"
    assert source.count(correction) == 1
    assert source.replace(correction, original) == inspect.getsource(old.check_static_objects)


def test_native_seven_value_encoding_passes_all_declared_cases():
    for spec in trials():
        new.check_static_objects(spec, padded_rows(spec))


@pytest.mark.parametrize('fault', ['legacy_three', 'short', 'long', 'padding', 'nonfinite', 'extents'])
def test_malformed_or_changed_native_box_is_rejected(fault):
    spec = trials()[0]; rows = padded_rows(spec)
    value = rows[-1]['native_box_size']
    if fault == 'legacy_three': rows[-1]['native_box_size'] = value[:3]
    if fault == 'short': value.pop()
    if fault == 'long': value.append(0.)
    if fault == 'padding': value[-1] = .1
    if fault == 'nonfinite': value[-1] = float('nan')
    if fault == 'extents': value[0] = .001
    with pytest.raises(ValueError): new.check_static_objects(spec, rows)


def test_original_failure_remains_reproducible_and_unmodified():
    spec = trials()[0]
    with pytest.raises(ValueError): old.check_static_objects(spec, padded_rows(spec))
