import ast
import inspect
import textwrap

import numpy as np
import pytest

from scripts import audit_go2_bounded_floor_robot_interface_development_v1 as original
from scripts import audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 as corrected


def test_only_native_box_representation_assertion_changes():
    before = ast.parse(textwrap.dedent(inspect.getsource(original.audit)))
    after = ast.parse(textwrap.dedent(inspect.getsource(corrected.audit)))
    target = "np.testing.assert_allclose(row['native_box_size'], expected['size_xyz'], atol=2e-07, rtol=0)"
    changes = 0
    for node in ast.walk(before):
        if isinstance(node, ast.Expr) and ast.unparse(node) == target:
            node.value = ast.parse("check_native_box_size(row['native_box_size'], expected['size_xyz'])", mode='eval').body
            changes += 1
    assert changes == 1 and ast.dump(before) == ast.dump(after)


def test_seven_slot_native_box_is_not_three_slot_array():
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([.08, 5.16, .6, 0., 0., 0., 0.], [.08, 5.16, .6], atol=2e-7, rtol=0)
    corrected.check_native_box_size([.08, 5.16, .6, 0., 0., 0., 0.], [.08, 5.16, .6])


@pytest.mark.parametrize('fault', ['short', 'long', 'padding', 'size', 'nan', 'dimensions'])
def test_reader_does_not_relax_geometry_or_reserved_fields(fault):
    row = [.08, 5.16, .6, 0., 0., 0., 0.]; dimensions = [.08, 5.16, .6]
    if fault == 'short': row = row[:3]
    if fault == 'long': row += [0.]
    if fault == 'padding': row[3] = 1e-12
    if fault == 'size': row[0] += 1e-4
    if fault == 'nan': row[0] = np.nan
    if fault == 'dimensions': dimensions = [.08]
    with pytest.raises((ValueError, AssertionError)): corrected.check_native_box_size(row, dimensions)
