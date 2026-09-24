from types import SimpleNamespace as NS
import sys
import pytest
from lewm_genesis.camera_renderer_identity_development import renderer_identity_readback


@pytest.mark.parametrize('fault', [None, 'wrong_target', 'empty', 'missing', 'non_ascii', 'query_error', 'bad_profile'])
def test_identity_uses_camera_context_and_releases_it_without_framebuffer_mutation(monkeypatch, fault):
    calls = []
    constants = ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION', 'GL_SHADING_LANGUAGE_VERSION',
        'GL_CONTEXT_PROFILE_MASK', 'GL_DRAW_FRAMEBUFFER_BINDING')
    gl = NS(**{key: key for key in constants})
    def get_integer(key):
        calls.append(('integer', key))
        if key == 'GL_DRAW_FRAMEBUFFER_BINDING': return 99 if fault == 'wrong_target' else 11
        assert key == 'GL_CONTEXT_PROFILE_MASK'
        return -1 if fault == 'bad_profile' else 1
    def get_string(key):
        calls.append(('string', key))
        if fault == 'query_error': raise RuntimeError('query failure')
        return {'empty': b'', 'missing': None, 'non_ascii': b'\xff'}.get(fault, b'synthetic context')
    gl.glGetIntegerv = get_integer; gl.glGetString = get_string
    monkeypatch.setitem(sys.modules, 'OpenGL.GL', gl)
    context = NS(make_current=lambda: calls.append('current'), make_uncurrent=lambda: calls.append('uncurrent'),
        _platform=NS(_egl_device=NS(name='/dev/dri/card0')), _is_software=False)
    camera = NS(uid=7, _rasterizer=NS(_renderer=context, _camera_targets={7:NS(_main_fb=11)}))
    if fault:
        with pytest.raises((ValueError, RuntimeError)): renderer_identity_readback(camera)
    else:
        value = renderer_identity_readback(camera)
        assert value['renderer'] == 'synthetic context' and value['egl_device_name'] == '/dev/dri/card0'
        assert value['camera_uid'] == '7' and value['framebuffer_matches_camera_depth_target']
        assert not value['source_implementation_equivalence_proven'] and not value['raster_error_bound_proven']
    assert calls[0] == 'current' and calls[-1] == 'uncurrent'
    # The fake GL interface deliberately provides no state-changing commands.
