import ast
from pathlib import Path

import numpy as np
import pytest

from lewm.physical_semantics import world_from_optical
from lewm_genesis.optical_camera_readback_development import check_optical_pose


def test_readback_changes_axes_not_world_position():
    optical = world_from_optical([.2, -.3, 1], [.7, .2, -.3], [.1, 0, 1])
    native = optical @ np.diag([1., -1., -1., 1.])
    got = check_optical_pose(native.astype(np.float32), optical)
    np.testing.assert_allclose(got, optical, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(got[:3, 3], native.astype(np.float32)[:3, 3])
    with pytest.raises(AssertionError):
        check_optical_pose(optical, optical)
    native[0, 3] += .001
    with pytest.raises(AssertionError):
        check_optical_pose(native, optical)


@pytest.mark.parametrize('bad', [np.zeros((3,3)), np.full((4,4),np.nan)])
def test_invalid_pose_rejected(bad):
    with pytest.raises(ValueError):
        check_optical_pose(bad, np.eye(4))


def test_only_collect_change_is_explicit_readback_conversion():
    root = Path(__file__).resolve().parents[2]
    old = (root/'scripts/run_go2_appearance_information_meshset_development_v1.py').read_text()
    new = (root/'scripts/run_go2_appearance_information_meshset_development_v2.py').read_text()
    corrected = old.replace('np.testing.assert_allclose(camera.transform,transform,atol=1e-6,rtol=0)',
                            'check_optical_pose(camera.transform,transform)')
    funcs = lambda text: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    a, b = funcs(corrected), funcs(new)
    for name in ('boxes_from_recording','build','native_identity','collect','score','main'):
        assert a[name] == b[name], name
